// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Merge-on-read MERGE INTO execution for Iceberg V3 tables.
//!
//! Plan shape (the MoR sibling of the Copy-on-Write chain):
//!
//! ```text
//! IcebergMorMergeCommitExec        one RowDelta snapshot: appended data files
//!   └ CoalescePartitionsExec       + new deletion vectors + superseded DVs removed
//!      └ IcebergMorMergeWriteExec  DV construction + late-materialized matched-row
//!         └ IcebergMorMergeExec      fetch by (_file, _pos) + appends
//!            └ HashJoinExec          source (build) ⟕ narrow target (probe)
//!               ├ source plan
//!               └ IcebergMorTargetScanExec (join keys + expr columns + _file + _pos)
//! ```
//!
//! MATCHED rows become deletion-vector entries against the scanned
//! `(_file, _pos)`; UPDATE additionally appends the new row version, with the
//! unreferenced columns fetched late — only data files that actually contain
//! matched rows are read for full-row materialization, never the whole
//! current set. A file that already carries a deletion vector gets ONE
//! superseding DV (prior deleted positions unioned in, prior DV removed in
//! the same commit) — never a second live DV per data file.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use datafusion::arrow::compute::{cast, concat, filter_record_batch, take};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::{DataFusionError, NullEquality, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::JoinType;
use datafusion::logical_expr::dml::MergeIntoClauseKind;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr, PlanProperties,
    execute_input_stream,
};
use futures::{StreamExt, TryStreamExt};
use iceberg::Catalog;
use iceberg::arrow::variant_shred::{
    shred_record_batch, shred_types_from_file_schema, shredded_output_type, variant_column_count,
};
use iceberg::arrow::{
    ArrowFileReader, PROJECTED_PARTITION_VALUE_COLUMN, PartitionValueCalculator,
    schema_to_arrow_schema,
};
use iceberg::delete_vector::DeleteVector;
use iceberg::io::FileMetadata;
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::scan::FileScanTask;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, ManifestContentType, ManifestList, Struct,
    deserialize_data_file_from_json, serialize_data_file_to_json,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use roaring::RoaringTreemap;
use uuid::Uuid;

use crate::task_writer::TaskWriter;
use crate::to_datafusion_error;

/// Name of the clause-routing column appended by [`IcebergMorMergeExec`].
pub(crate) const MOR_CLAUSE_COL: &str = "__mor_clause";

/// Rows per processing chunk inside the write node. The hash join emits its
/// unmatched-build output — the entire NOT MATCHED (insert) set — as ONE
/// batch, bypassing the output coalescer's target size. Processing that
/// whole defeats file rolling (the rolling writer only checks size between
/// write calls) and multiplies wide-row memory through every filter /
/// evaluate / cast copy. Slicing is zero-copy; everything downstream then
/// works on bounded rows.
const MOR_WRITE_CHUNK_ROWS: usize = 8192;

/// How many existing data files to probe (parquet footers) when deriving the
/// variant shredding layout under `write.parquet.shred-variants`. First
/// non-canonical derivation wins per column; probing stops early once every
/// variant column has one.
const MERGE_SHRED_PROBE_FILES: usize = 4;

/// A WHEN clause with its expressions bound to the join output schema.
#[derive(Debug, Clone)]
pub(crate) struct MorClausePlan {
    pub kind: MergeIntoClauseKind,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub action: MorActionPlan,
}

/// The bound action of a WHEN clause.
#[derive(Debug, Clone)]
pub(crate) enum MorActionPlan {
    /// UPDATE SET assignments: (target column name, bound value expression).
    Update(Vec<(String, Arc<dyn PhysicalExpr>)>),
    /// INSERT: (target column name, bound value expression).
    Insert(Vec<(String, Arc<dyn PhysicalExpr>)>),
    /// DELETE the matched row.
    Delete,
}

// ---------------------------------------------------------------------------
// IcebergMorTargetScanExec — narrow target scan with plain-typed _file/_pos
// ---------------------------------------------------------------------------

/// Scans the merge target with a NARROW projection: only the target columns
/// the merge expressions reference, plus the `_file` and `_pos` metadata
/// columns MoR write needs to construct deletion vectors. Metadata columns
/// are cast to plain types (`_file` arrives run-end-encoded from the reader,
/// which join/take kernels don't handle).
#[derive(Debug)]
pub(crate) struct IcebergMorTargetScanExec {
    table: Table,
    snapshot_id: Option<i64>,
    /// Data column names to project (metadata columns appended internally).
    columns: Vec<String>,
    schema: ArrowSchemaRef,
    plan_properties: Arc<PlanProperties>,
}

impl IcebergMorTargetScanExec {
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        columns: Vec<String>,
        data_fields: Vec<Field>,
    ) -> Self {
        let mut fields = data_fields;
        fields.push(Field::new(RESERVED_COL_NAME_FILE, DataType::Utf8, false));
        fields.push(Field::new(RESERVED_COL_NAME_POS, DataType::Int64, false));
        let schema = Arc::new(ArrowSchema::new(fields));
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            table,
            snapshot_id,
            columns,
            schema,
            plan_properties,
        }
    }
}

impl DisplayAs for IcebergMorTargetScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IcebergMorTargetScanExec: columns=[{}]",
            self.columns.join(",")
        )
    }
}

impl ExecutionPlan for IcebergMorTargetScanExec {
    fn name(&self) -> &str {
        "IcebergMorTargetScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let table = self.table.clone();
        let snapshot_id = self.snapshot_id;
        let mut select: Vec<String> = self.columns.clone();
        select.push(RESERVED_COL_NAME_FILE.to_string());
        select.push(RESERVED_COL_NAME_POS.to_string());
        let out_schema = Arc::clone(&self.schema);

        let stream_schema = Arc::clone(&out_schema);
        let fut = async move {
            let mut builder = table.scan();
            if let Some(id) = snapshot_id {
                builder = builder.snapshot_id(id);
            }
            let scan = builder
                .select(select)
                .build()
                .map_err(to_datafusion_error)?;
            let stream = scan.to_arrow().await.map_err(to_datafusion_error)?;
            let out = stream.map(move |batch| {
                let batch = batch.map_err(to_datafusion_error)?;
                plain_cast_batch(&batch, &stream_schema)
            });
            Ok::<_, DataFusionError>(Box::pin(out))
        };
        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(out_schema, stream)))
    }
}

/// Rebuild `batch` against `schema`, casting any column whose type differs
/// (notably run-end-encoded metadata columns to their plain value type).
fn plain_cast_batch(batch: &RecordBatch, schema: &ArrowSchemaRef) -> DFResult<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(idx);
        if col.data_type() == field.data_type() {
            columns.push(Arc::clone(col));
        } else {
            columns.push(cast(col.as_ref(), field.data_type())?);
        }
    }
    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), Some("cast scan batch".to_string())))
}

// ---------------------------------------------------------------------------
// IcebergMorMergeExec — join + clause routing
// ---------------------------------------------------------------------------

/// Joins the source (build side) against the narrow target scan (probe side)
/// and routes every surviving row to the first WHEN clause it satisfies.
///
/// Output schema: `[source columns…, target columns…, _file, _pos, __mor_clause]`
/// — the join output with a nullable `UInt32` clause index appended. Rows
/// matching no clause are dropped.
#[derive(Debug)]
pub(crate) struct IcebergMorMergeExec {
    source: Arc<dyn ExecutionPlan>,
    target_scan: Arc<dyn ExecutionPlan>,
    join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>,
    clauses: Arc<Vec<MorClausePlan>>,
    /// Index of `_file` in the join output (matched-row marker).
    file_idx: usize,
    schema: ArrowSchemaRef,
    plan_properties: Arc<PlanProperties>,
}

impl IcebergMorMergeExec {
    #[allow(clippy::type_complexity)]
    pub(crate) fn new(
        source: Arc<dyn ExecutionPlan>,
        target_scan: Arc<dyn ExecutionPlan>,
        join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>,
        clauses: Arc<Vec<MorClausePlan>>,
    ) -> Self {
        let source_fields = source.schema().fields().to_vec();
        let target_fields: Vec<_> = target_scan
            .schema()
            .fields()
            .iter()
            .map(|f| Arc::new(f.as_ref().clone().with_nullable(true)))
            .collect();
        let file_idx = source_fields.len()
            + target_scan
                .schema()
                .index_of(RESERVED_COL_NAME_FILE)
                .expect("target scan carries _file");
        let mut fields = source_fields;
        fields.extend(target_fields);
        fields.push(Arc::new(Field::new(MOR_CLAUSE_COL, DataType::UInt32, true)));
        let schema = Arc::new(ArrowSchema::new(fields));
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            source,
            target_scan,
            join_on,
            clauses,
            file_idx,
            schema,
            plan_properties,
        }
    }
}

impl DisplayAs for IcebergMorMergeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IcebergMorMergeExec: clauses={}, join_on={}",
            self.clauses.len(),
            self.join_on.len()
        )
    }
}

impl ExecutionPlan for IcebergMorMergeExec {
    fn name(&self) -> &str {
        "IcebergMorMergeExec"
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.source, &self.target_scan]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Internal(
                "IcebergMorMergeExec requires exactly 2 children".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
            self.join_on.clone(),
            Arc::clone(&self.clauses),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Source is the BUILD side (bounded batch + demote set); the target
        // current-set streams through as the probe. LEFT join preserves every
        // source row; target-only rows are irrelevant without
        // NOT MATCHED BY SOURCE support.
        let source = if self
            .source
            .properties()
            .output_partitioning()
            .partition_count()
            > 1
        {
            Arc::new(
                datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec::new(
                    Arc::clone(&self.source),
                ),
            ) as Arc<dyn ExecutionPlan>
        } else {
            Arc::clone(&self.source)
        };
        let join = Arc::new(HashJoinExec::try_new(
            source,
            Arc::clone(&self.target_scan),
            self.join_on.clone(),
            None,
            &JoinType::Left,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?) as Arc<dyn ExecutionPlan>;

        if partition != 0 {
            return Err(DataFusionError::Internal(
                "IcebergMorMergeExec is single-partition".to_string(),
            ));
        }
        // The join's output partitioning follows the probe side; drain every
        // partition through a coalesce so no routed row is lost.
        let join = Arc::new(
            datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec::new(join),
        ) as Arc<dyn ExecutionPlan>;
        let join_stream =
            execute_input_stream(Arc::clone(&join), join.schema(), partition, context)?;

        let clauses = Arc::clone(&self.clauses);
        let file_idx = self.file_idx;
        let out_schema = Arc::clone(&self.schema);
        let stream_schema = Arc::clone(&out_schema);

        let stream = join_stream.and_then(move |batch| {
            let clauses = Arc::clone(&clauses);
            let out_schema = Arc::clone(&stream_schema);
            async move { route_clauses(&batch, &clauses, file_idx, &out_schema) }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(out_schema, stream)))
    }
}

/// Append the `__mor_clause` column: for each row, the index of the FIRST
/// clause whose kind matches the row's MATCHED state and whose predicate
/// (if any) evaluates to true. Rows claiming no clause carry NULL.
fn route_clauses(
    batch: &RecordBatch,
    clauses: &[MorClausePlan],
    file_idx: usize,
    out_schema: &ArrowSchemaRef,
) -> DFResult<RecordBatch> {
    let num_rows = batch.num_rows();
    // A row is MATCHED iff the target-side _file is non-null (the scan emits
    // it non-null for every target row; a LEFT-join miss nulls it).
    let file_col = batch.column(file_idx);
    let matched: Vec<bool> = (0..num_rows).map(|i| file_col.is_valid(i)).collect();

    let mut clause_idx: Vec<Option<u32>> = vec![None; num_rows];
    for (ci, clause) in clauses.iter().enumerate() {
        let want_matched = match clause.kind {
            MergeIntoClauseKind::Matched => true,
            MergeIntoClauseKind::NotMatched | MergeIntoClauseKind::NotMatchedByTarget => false,
            MergeIntoClauseKind::NotMatchedBySource => {
                return Err(DataFusionError::NotImplemented(
                    "MERGE WHEN NOT MATCHED BY SOURCE is not supported for merge-on-read yet"
                        .to_string(),
                ));
            }
        };
        let pred = match &clause.predicate {
            Some(p) => {
                let arr = p.evaluate(batch)?.into_array(num_rows)?;
                let bools = cast(arr.as_ref(), &DataType::Boolean)?;
                Some(bools)
            }
            None => None,
        };
        for row in 0..num_rows {
            if clause_idx[row].is_some() || matched[row] != want_matched {
                continue;
            }
            let pred_ok = match &pred {
                None => true,
                Some(b) => {
                    let b = b
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                        .expect("cast to boolean");
                    b.is_valid(row) && b.value(row)
                }
            };
            if pred_ok {
                clause_idx[row] = Some(ci as u32);
            }
        }
    }

    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(UInt32Array::from(clause_idx)));
    RecordBatch::try_new(Arc::clone(out_schema), columns).map_err(|e| {
        DataFusionError::ArrowError(Box::new(e), Some("route merge clauses".to_string()))
    })
}

// ---------------------------------------------------------------------------
// IcebergMorMergeWriteExec — DVs + late-materialized updates + appends
// ---------------------------------------------------------------------------

/// Result-lane column names handed to the commit node.
pub(crate) const DATA_FILES_LANE: &str = "data_files";
pub(crate) const DELETE_FILES_LANE: &str = "delete_files";
pub(crate) const REMOVED_DELETE_FILES_LANE: &str = "removed_delete_files";

/// Consumes the clause-routed join stream and produces, per partition:
/// - appended data files (INSERT rows + updated versions of MATCHED rows),
/// - one CONSOLIDATED deletion vector per touched data file (prior deleted
///   positions unioned with newly matched positions),
/// - the prior delete files those DVs supersede.
///
/// Matched-row UPDATE output is materialized LATE: only the data files that
/// contain matched rows are re-read (full projection + `_pos`), and the new
/// row version is the fetched row with the SET columns overridden.
#[derive(Debug)]
pub(crate) struct IcebergMorMergeWriteExec {
    table: Table,
    snapshot_id: Option<i64>,
    input: Arc<dyn ExecutionPlan>,
    clauses: Arc<Vec<MorClausePlan>>,
    result_schema: ArrowSchemaRef,
    plan_properties: Arc<PlanProperties>,
}

impl IcebergMorMergeWriteExec {
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        input: Arc<dyn ExecutionPlan>,
        clauses: Arc<Vec<MorClausePlan>>,
    ) -> Self {
        let result_schema = Self::make_result_schema();
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&result_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            table,
            snapshot_id,
            input,
            clauses,
            result_schema,
            plan_properties,
        }
    }

    fn make_result_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new(DATA_FILES_LANE, DataType::Utf8, false),
            Field::new(DELETE_FILES_LANE, DataType::Utf8, false),
            Field::new(REMOVED_DELETE_FILES_LANE, DataType::Utf8, false),
        ]))
    }
}

impl DisplayAs for IcebergMorMergeWriteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IcebergMorMergeWriteExec: table={}",
            self.table.identifier()
        )
    }
}

impl ExecutionPlan for IcebergMorMergeWriteExec {
    fn name(&self) -> &str {
        "IcebergMorMergeWriteExec"
    }

    fn required_input_distribution(&self) -> Vec<datafusion::physical_plan::Distribution> {
        vec![datafusion::physical_plan::Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "IcebergMorMergeWriteExec expects exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            self.table.clone(),
            self.snapshot_id,
            Arc::clone(&children[0]),
            Arc::clone(&self.clauses),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let table = self.table.clone();
        let snapshot_id = self.snapshot_id;
        let clauses = Arc::clone(&self.clauses);
        let result_schema = Arc::clone(&self.result_schema);
        let input = execute_input_stream(
            Arc::clone(&self.input),
            self.input.schema(),
            partition,
            context,
        )?;
        let input_schema = self.input.schema();

        let run_schema = Arc::clone(&result_schema);
        let stream = futures::stream::once(async move {
            run_mor_write(
                table,
                snapshot_id,
                input,
                input_schema,
                &clauses,
                &run_schema,
            )
            .await
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            result_schema,
            stream,
        )))
    }
}

/// One matched row's bookkeeping: which clause claimed it and (for UPDATE)
/// which row of the clause's evaluated-assignment store holds its SET values.
#[derive(Clone, Copy)]
struct MatchedRow {
    clause: u32,
    update_row: usize,
}

/// Derive per-variant-column PLAIN shredding types for the merge output by
/// probing existing data files' parquet footers (shred-preserving — the
/// layout is carried forward from what the table already stores, never
/// invented). An empty table, or one whose files are all canonical, derives
/// nothing and the output stays canonical.
async fn merge_shred_types(
    table: &Table,
    snapshot_id: Option<i64>,
) -> DFResult<HashMap<String, DataType>> {
    let table_schema = table.metadata().current_schema();
    let n_variant = variant_column_count(table_schema);
    let Some(snapshot_id) = snapshot_id else {
        return Ok(HashMap::new());
    };
    if n_variant == 0 {
        return Ok(HashMap::new());
    }
    let scan = table
        .scan()
        .snapshot_id(snapshot_id)
        .build()
        .map_err(to_datafusion_error)?;
    let mut tasks = scan.plan_files().await.map_err(to_datafusion_error)?;
    let mut out: HashMap<String, DataType> = HashMap::new();
    let mut probed = 0usize;
    while let Some(task) = tasks.try_next().await.map_err(to_datafusion_error)? {
        let input = table
            .file_io()
            .new_input(task.data_file_path())
            .map_err(to_datafusion_error)?;
        let reader = input.reader().await.map_err(to_datafusion_error)?;
        let mut reader = ArrowFileReader::new(
            FileMetadata {
                size: task.file_size_in_bytes,
            },
            reader,
        );
        let meta = ArrowReaderMetadata::load_async(&mut reader, Default::default())
            .await
            .map_err(|e| {
                DataFusionError::External(
                    format!(
                        "loading parquet footer of {} for shred derivation: {e}",
                        task.data_file_path()
                    )
                    .into(),
                )
            })?;
        for (name, plain) in shred_types_from_file_schema(meta.schema(), table_schema) {
            out.entry(name).or_insert(plain);
        }
        probed += 1;
        if out.len() == n_variant || probed >= MERGE_SHRED_PROBE_FILES {
            break;
        }
    }
    Ok(out)
}

async fn run_mor_write(
    table: Table,
    snapshot_id: Option<i64>,
    mut input: SendableRecordBatchStream,
    input_schema: ArrowSchemaRef,
    clauses: &[MorClausePlan],
    result_schema: &ArrowSchemaRef,
) -> DFResult<RecordBatch> {
    let table_schema = table.metadata().current_schema().clone();
    let table_arrow: ArrowSchemaRef =
        Arc::new(schema_to_arrow_schema(&table_schema).map_err(to_datafusion_error)?);
    let table_props = table
        .metadata()
        .table_properties()
        .map_err(to_datafusion_error)?;
    let file_format =
        <DataFileFormat as std::str::FromStr>::from_str(&table_props.write_format_default)
            .map_err(to_datafusion_error)?;
    if file_format != DataFileFormat::Parquet {
        return Err(to_datafusion_error(iceberg::Error::new(
            iceberg::ErrorKind::FeatureUnsupported,
            format!("File format {file_format} is not supported for MERGE"),
        )));
    }

    // Shred-preserving output under `write.parquet.shred-variants`: derive
    // each variant column's layout from the table's existing files, and give
    // the writer the exact shredded arrow types the (shredded) batches will
    // carry. Empty map = canonical output, the untouched default.
    let shred_plain = if table_props.parquet_shred_variants {
        merge_shred_types(&table, snapshot_id).await?
    } else {
        HashMap::new()
    };
    let shred_overrides: HashMap<String, DataType> = shred_plain
        .iter()
        .map(|(name, plain)| {
            let idx = table_arrow
                .index_of(name)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let out_type = shredded_output_type(table_arrow.field(idx).data_type(), plain)
                .map_err(to_datafusion_error)?;
            Ok((name.clone(), out_type))
        })
        .collect::<DFResult<_>>()?;

    // Derive writer properties from the table's write.parquet.* settings —
    // parquet-rs defaults are UNCOMPRESSED, which inflates real merge output
    // ~20x and was the bulk of an oversized single-file write at soak scale.
    let parquet_writer_builder =
        ParquetWriterBuilder::from_table_properties(&table_props, table_schema.clone())
            .with_variant_shred_types(shred_overrides);
    let location_generator =
        DefaultLocationGenerator::new(table.metadata()).map_err(to_datafusion_error)?;
    let file_name_generator = DefaultFileNameGenerator::new(
        format!("merge-{}", Uuid::now_v7()),
        None,
        DataFileFormat::Parquet,
    );
    let rolling_writer_builder = RollingFileWriterBuilder::new(
        parquet_writer_builder,
        table_props.write_target_file_size_bytes,
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );
    let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);
    let partition_spec = table.metadata().default_partition_spec().clone();
    // Partitioned writes require the computed `_partition` column on every
    // batch (the fanout splitter routes rows by it).
    let partition_calc = if partition_spec.is_unpartitioned() {
        None
    } else {
        Some(
            PartitionValueCalculator::try_new(&partition_spec, &table_schema)
                .map_err(to_datafusion_error)?,
        )
    };
    let mut writer = TaskWriter::try_new(
        data_file_writer_builder,
        table_props.write_datafusion_fanout_enabled,
        table_schema.clone(),
        partition_spec,
    )
    .map_err(to_datafusion_error)?;

    let clause_idx_col = input_schema
        .index_of(MOR_CLAUSE_COL)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
    // _file/_pos sit at the tail of the target side, right before __mor_clause.
    let pos_idx = clause_idx_col - 1;
    let file_idx = clause_idx_col - 2;

    // Per-UPDATE-clause store of evaluated SET values (one array per
    // assignment, concatenated across batches at the end).
    let n_clauses = clauses.len();
    let mut update_stores: Vec<Vec<Vec<ArrayRef>>> = vec![Vec::new(); n_clauses];
    let mut update_store_rows: Vec<usize> = vec![0; n_clauses];
    // (file, pos) -> matched bookkeeping. Duplicate claims are an error.
    let mut matched: HashMap<(String, u64), MatchedRow> = HashMap::new();

    while let Some(batch) = input.try_next().await? {
        let mut chunk_start = 0;
        while chunk_start < batch.num_rows() {
            let chunk_len = MOR_WRITE_CHUNK_ROWS.min(batch.num_rows() - chunk_start);
            let chunk = batch.slice(chunk_start, chunk_len);
            chunk_start += chunk_len;

            let clause_arr = chunk
                .column(clause_idx_col)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| DataFusionError::Internal("__mor_clause must be UInt32".into()))?
                .clone();

            for (ci, clause) in clauses.iter().enumerate() {
                let mask: Vec<bool> = (0..chunk.num_rows())
                    .map(|r| clause_arr.is_valid(r) && clause_arr.value(r) == ci as u32)
                    .collect();
                if !mask.iter().any(|&m| m) {
                    continue;
                }
                let mask_arr = datafusion::arrow::array::BooleanArray::from(mask);
                let subset = filter_record_batch(&chunk, &mask_arr)?;

                match &clause.action {
                    MorActionPlan::Insert(assignments) => {
                        let out = build_full_rows(&subset, assignments, &table_arrow, None)?;
                        let out =
                            shred_record_batch(&out, &shred_plain).map_err(to_datafusion_error)?;
                        let out = with_partition_column(out, partition_calc.as_ref())?;
                        writer.write(out).await.map_err(to_datafusion_error)?;
                    }
                    MorActionPlan::Update(assignments) => {
                        // Evaluate SET values now (they reference join columns);
                        // the full old row arrives with the late fetch.
                        let mut evaluated: Vec<ArrayRef> = Vec::with_capacity(assignments.len());
                        for (_, expr) in assignments {
                            evaluated.push(expr.evaluate(&subset)?.into_array(subset.num_rows())?);
                        }
                        record_matched(
                            &subset,
                            file_idx,
                            pos_idx,
                            ci as u32,
                            update_store_rows[ci],
                            &mut matched,
                        )?;
                        update_store_rows[ci] += subset.num_rows();
                        update_stores[ci].push(evaluated);
                    }
                    MorActionPlan::Delete => {
                        record_matched(&subset, file_idx, pos_idx, ci as u32, 0, &mut matched)?;
                    }
                }
            }
        }
    }

    // Concatenate each update clause's evaluated assignment arrays.
    let update_values: Vec<Vec<ArrayRef>> = update_stores
        .into_iter()
        .map(|chunks| -> DFResult<Vec<ArrayRef>> {
            if chunks.is_empty() {
                return Ok(Vec::new());
            }
            let n_cols = chunks[0].len();
            (0..n_cols)
                .map(|c| {
                    let parts: Vec<&dyn Array> =
                        chunks.iter().map(|chunk| chunk[c].as_ref()).collect();
                    concat(&parts).map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                })
                .collect()
        })
        .collect::<DFResult<_>>()?;

    let mut new_delete_files: Vec<DataFile> = Vec::new();
    let mut removed_delete_files: Vec<DataFile> = Vec::new();

    if !matched.is_empty() {
        let snapshot_id = snapshot_id.ok_or_else(|| {
            DataFusionError::Internal("matched rows against a table with no snapshot".into())
        })?;
        let affected: HashSet<String> = matched.keys().map(|(f, _)| f.clone()).collect();

        // Plan the pinned snapshot; keep only tasks for affected files.
        let mut select: Vec<String> = table_schema
            .as_struct()
            .fields()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        select.push(RESERVED_COL_NAME_FILE.to_string());
        select.push(RESERVED_COL_NAME_POS.to_string());
        let scan = table
            .scan()
            .snapshot_id(snapshot_id)
            .select(select)
            .build()
            .map_err(to_datafusion_error)?;
        let tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .map_err(to_datafusion_error)?
            .try_collect()
            .await
            .map_err(to_datafusion_error)?;
        let fetch_tasks: Vec<FileScanTask> = tasks
            .into_iter()
            .filter(|t| affected.contains(t.data_file_path()))
            .collect();
        {
            let planned: HashSet<&str> = fetch_tasks.iter().map(|t| t.data_file_path()).collect();
            if let Some((missing, _)) = matched.keys().find(|(f, _)| !planned.contains(f.as_str()))
            {
                return Err(DataFusionError::Internal(format!(
                    "matched data file {missing} not found in the scanned snapshot"
                )));
            }
        }

        // Guard: only deletion vectors / positional deletes are superseded.
        for task in &fetch_tasks {
            for del in task.deletes.iter() {
                if del.file_type == DataContentType::EqualityDeletes {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE into a table with equality deletes is not supported".to_string(),
                    ));
                }
            }
        }

        let per_file: HashMap<String, &FileScanTask> = fetch_tasks
            .iter()
            .map(|t| (t.data_file_path().to_string(), t))
            .collect();

        // Late materialization: read ONLY the affected files, full projection
        // + _file/_pos, prior deletes applied (alive rows only).
        let reader = table.reader_builder().build();
        let task_stream = Box::pin(futures::stream::iter(
            fetch_tasks.iter().cloned().map(Ok).collect::<Vec<_>>(),
        )) as iceberg::scan::FileScanTaskStream;
        let mut fetch_stream = reader
            .read(task_stream)
            .map_err(to_datafusion_error)?
            .stream();

        let mut alive: HashMap<String, RoaringTreemap> = HashMap::new();
        while let Some(fbatch) = fetch_stream.try_next().await.map_err(to_datafusion_error)? {
            if fbatch.num_rows() == 0 {
                continue;
            }
            let fb_schema = fbatch.schema();
            let f_file_idx = fb_schema
                .index_of(RESERVED_COL_NAME_FILE)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let f_pos_idx = fb_schema
                .index_of(RESERVED_COL_NAME_POS)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let file_arr = cast(fbatch.column(f_file_idx).as_ref(), &DataType::Utf8)?;
            let file_arr = file_arr
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("cast to Utf8");
            let pos_arr = fbatch
                .column(f_pos_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| DataFusionError::Internal("_pos must be Int64".into()))?;

            // Track alive positions + collect the rows this merge updates.
            let mut fetch_rows: Vec<u32> = Vec::new();
            let mut stored: Vec<MatchedRow> = Vec::new();
            for row in 0..fbatch.num_rows() {
                let file = file_arr.value(row);
                let pos = pos_arr.value(row) as u64;
                alive.entry(file.to_string()).or_default().insert(pos);
                if let Some(m) = matched.get(&(file.to_string(), pos))
                    && clauses
                        .get(m.clause as usize)
                        .is_some_and(|c| matches!(c.action, MorActionPlan::Update(_)))
                {
                    fetch_rows.push(row as u32);
                    stored.push(*m);
                }
            }
            if fetch_rows.is_empty() {
                continue;
            }

            // Build updated row versions: fetched row, SET columns overridden.
            // Group by clause (different clauses assign different columns).
            let mut by_clause: HashMap<u32, (Vec<u32>, Vec<u64>)> = HashMap::new();
            for (i, m) in stored.iter().enumerate() {
                let e = by_clause.entry(m.clause).or_default();
                e.0.push(fetch_rows[i]);
                e.1.push(m.update_row as u64);
            }
            for (clause, (rows, store_rows)) in by_clause {
                let MorActionPlan::Update(assignments) = &clauses[clause as usize].action else {
                    unreachable!("only update clauses collect fetch rows");
                };
                let take_idx = UInt32Array::from(rows);
                let store_idx = UInt64Array::from(store_rows);
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(table_arrow.fields().len());
                for field in table_arrow.fields() {
                    let assigned = assignments
                        .iter()
                        .position(|(name, _)| name == field.name());
                    let arr = match assigned {
                        Some(a) => {
                            take(update_values[clause as usize][a].as_ref(), &store_idx, None)?
                        }
                        None => {
                            let src_idx = fb_schema
                                .index_of(field.name())
                                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                            take(fbatch.column(src_idx).as_ref(), &take_idx, None)?
                        }
                    };
                    let arr = if arr.data_type() == field.data_type() {
                        arr
                    } else {
                        cast(arr.as_ref(), field.data_type())?
                    };
                    columns.push(arr);
                }
                let out = RecordBatch::try_new(Arc::clone(&table_arrow), columns)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                let out = shred_record_batch(&out, &shred_plain).map_err(to_datafusion_error)?;
                let out = with_partition_column(out, partition_calc.as_ref())?;
                writer.write(out).await.map_err(to_datafusion_error)?;
            }
        }

        // One consolidated DV per affected file: everything already deleted
        // (full range minus alive) plus this merge's matched positions.
        let location = table.metadata().location().to_string();
        for (file, task) in &per_file {
            let record_count = task.record_count.ok_or_else(|| {
                DataFusionError::Internal(format!("no record count for data file {file}"))
            })?;
            let alive_set = alive.remove(file).unwrap_or_default();
            let mut bitmap = RoaringTreemap::new();
            for pos in 0..record_count {
                if !alive_set.contains(pos) {
                    bitmap.insert(pos);
                }
            }
            for ((f, pos), _) in matched.iter().filter(|((f, _), _)| f == file) {
                debug_assert_eq!(f, file);
                bitmap.insert(*pos);
            }
            let partition = task.partition.clone().unwrap_or(Struct::empty());
            let spec_id = task
                .partition_spec
                .as_ref()
                .map(|s| s.spec_id())
                .unwrap_or_else(|| table.metadata().default_partition_spec_id());
            let dv_path = format!("{location}/data/{}-deletes.puffin", Uuid::now_v7());
            let dv_file = DeleteVector::new(bitmap)
                .write_to_puffin_file(table.file_io(), dv_path, file.clone(), partition, spec_id)
                .await
                .map_err(to_datafusion_error)?;
            new_delete_files.push(dv_file);
        }

        // Resolve the prior delete files being superseded to their manifest
        // DataFile entries (sourced from the SCAN's per-file binding).
        let prior_paths: HashSet<String> = fetch_tasks
            .iter()
            .flat_map(|t| t.deletes.iter().map(|d| d.file_path.clone()))
            .collect();
        if !prior_paths.is_empty() {
            removed_delete_files = resolve_delete_files(&table, snapshot_id, &prior_paths).await?;
        }
    }

    let data_files = writer.close().await.map_err(to_datafusion_error)?;

    // Serialize the three lanes.
    let partition_type = table.metadata().default_partition_type().clone();
    let format_version = table.metadata().format_version();
    let ser = |files: &[DataFile]| -> DFResult<Vec<String>> {
        files
            .iter()
            .map(|f| {
                serialize_data_file_to_json(f.clone(), &partition_type, format_version)
                    .map_err(to_datafusion_error)
            })
            .collect()
    };
    let data_lane = ser(&data_files)?;
    let dv_lane = ser(&new_delete_files)?;
    let removed_lane = ser(&removed_delete_files)?;

    let rows = data_lane.len().max(dv_lane.len()).max(removed_lane.len());
    let pad = |mut v: Vec<String>| -> ArrayRef {
        let n = v.len();
        v.extend(std::iter::repeat_n(String::new(), rows - n));
        Arc::new(StringArray::from(v))
    };
    RecordBatch::try_new(Arc::clone(result_schema), vec![
        pad(data_lane),
        pad(dv_lane),
        pad(removed_lane),
    ])
    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Append the computed `_partition` column for partitioned tables; pass
/// batches through unchanged for unpartitioned ones.
fn with_partition_column(
    batch: RecordBatch,
    calc: Option<&PartitionValueCalculator>,
) -> DFResult<RecordBatch> {
    let Some(calc) = calc else {
        return Ok(batch);
    };
    let partition_array = calc.calculate(&batch).map_err(to_datafusion_error)?;
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        PROJECTED_PARTITION_VALUE_COLUMN,
        partition_array.data_type().clone(),
        false,
    )));
    let mut columns = batch.columns().to_vec();
    columns.push(partition_array);
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Record each row of `subset` as matched (file, pos) -> (clause, store row).
/// A row already claimed by another source row is an error (SQL MERGE forbids
/// updating or deleting the same target row twice).
fn record_matched(
    subset: &RecordBatch,
    file_idx: usize,
    pos_idx: usize,
    clause: u32,
    store_base: usize,
    matched: &mut HashMap<(String, u64), MatchedRow>,
) -> DFResult<()> {
    let file_arr = subset
        .column(file_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Internal("_file must be Utf8".into()))?;
    let pos_arr = subset
        .column(pos_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Internal("_pos must be Int64".into()))?;
    for row in 0..subset.num_rows() {
        let key = (file_arr.value(row).to_string(), pos_arr.value(row) as u64);
        if matched
            .insert(key, MatchedRow {
                clause,
                update_row: store_base + row,
            })
            .is_some()
        {
            return Err(DataFusionError::Execution(
                "MERGE matched the same target row from more than one source row".to_string(),
            ));
        }
    }
    Ok(())
}

/// Build full-width rows for INSERT: assigned columns take their evaluated
/// expressions; unassigned columns are NULL.
fn build_full_rows(
    subset: &RecordBatch,
    assignments: &[(String, Arc<dyn PhysicalExpr>)],
    table_arrow: &ArrowSchemaRef,
    _unused: Option<()>,
) -> DFResult<RecordBatch> {
    let num_rows = subset.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(table_arrow.fields().len());
    for field in table_arrow.fields() {
        let assigned = assignments.iter().find(|(name, _)| name == field.name());
        let arr: ArrayRef = match assigned {
            Some((_, expr)) => {
                let arr = expr.evaluate(subset)?.into_array(num_rows)?;
                if arr.data_type() == field.data_type() {
                    arr
                } else {
                    cast(arr.as_ref(), field.data_type())?
                }
            }
            None => datafusion::arrow::array::new_null_array(field.data_type(), num_rows),
        };
        columns.push(arr);
    }
    RecordBatch::try_new(Arc::clone(table_arrow), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Resolve delete-file paths to their manifest `DataFile` entries in the
/// pinned snapshot (needed by `RowDelta::remove_delete_files`).
async fn resolve_delete_files(
    table: &Table,
    snapshot_id: i64,
    paths: &HashSet<String>,
) -> DFResult<Vec<DataFile>> {
    let metadata = table.metadata();
    let snapshot = metadata
        .snapshot_by_id(snapshot_id)
        .ok_or_else(|| DataFusionError::Internal(format!("snapshot {snapshot_id} not found")))?;
    let bytes = table
        .file_io()
        .new_input(snapshot.manifest_list())
        .map_err(to_datafusion_error)?
        .read()
        .await
        .map_err(to_datafusion_error)?;
    let manifest_list = ManifestList::parse_with_version(&bytes, metadata.format_version())
        .map_err(to_datafusion_error)?;
    let mut out = Vec::new();
    for mf in manifest_list.entries() {
        if mf.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = mf
            .load_manifest(table.file_io())
            .await
            .map_err(to_datafusion_error)?;
        for entry in manifest.entries() {
            if entry.is_alive() && paths.contains(entry.data_file().file_path()) {
                out.push(entry.data_file().clone());
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// IcebergMorMergeCommitExec — one RowDelta snapshot
// ---------------------------------------------------------------------------

/// Commits the three lanes as ONE atomic `RowDelta` snapshot: appended data
/// files + new deletion vectors added, superseded delete files removed.
pub(crate) struct IcebergMorMergeCommitExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    snapshot_id: Option<i64>,
    input: Arc<dyn ExecutionPlan>,
    count_schema: ArrowSchemaRef,
    plan_properties: Arc<PlanProperties>,
}

impl fmt::Debug for IcebergMorMergeCommitExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergMorMergeCommitExec")
            .field("table", &self.table.identifier())
            .finish()
    }
}

impl IcebergMorMergeCommitExec {
    pub(crate) fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        snapshot_id: Option<i64>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let count_schema: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![Field::new(
            "count",
            DataType::UInt64,
            false,
        )]));
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&count_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            table,
            catalog,
            snapshot_id,
            input,
            count_schema,
            plan_properties,
        }
    }
}

impl DisplayAs for IcebergMorMergeCommitExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IcebergMorMergeCommitExec: table={}",
            self.table.identifier()
        )
    }
}

impl ExecutionPlan for IcebergMorMergeCommitExec {
    fn name(&self) -> &str {
        "IcebergMorMergeCommitExec"
    }

    fn required_input_distribution(&self) -> Vec<datafusion::physical_plan::Distribution> {
        vec![datafusion::physical_plan::Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "IcebergMorMergeCommitExec expects exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            self.table.clone(),
            Arc::clone(&self.catalog),
            self.snapshot_id,
            Arc::clone(&children[0]),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(
                "IcebergMorMergeCommitExec is single-partition".to_string(),
            ));
        }
        let table = self.table.clone();
        let catalog = Arc::clone(&self.catalog);
        let snapshot_id = self.snapshot_id;
        let input = self.input.clone();
        let count_schema = Arc::clone(&self.count_schema);

        let stream = futures::stream::once(async move {
            let spec_id = table.metadata().default_partition_spec_id();
            let partition_type = table.metadata().default_partition_type().clone();
            let current_schema = table.metadata().current_schema().clone();

            let mut added: Vec<DataFile> = Vec::new();
            let mut dvs: Vec<DataFile> = Vec::new();
            let mut removed: Vec<DataFile> = Vec::new();

            let mut batches = input.execute(0, context)?;
            while let Some(batch) = batches.try_next().await? {
                let lane = |name: &str| -> DFResult<Vec<DataFile>> {
                    let arr = batch
                        .column_by_name(name)
                        .ok_or_else(|| DataFusionError::Internal(format!("missing '{name}' lane")))?
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(format!("'{name}' lane must be Utf8"))
                        })?;
                    arr.iter()
                        .flatten()
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            deserialize_data_file_from_json(
                                s,
                                spec_id,
                                &partition_type,
                                &current_schema,
                            )
                            .map_err(to_datafusion_error)
                        })
                        .collect()
                };
                added.extend(lane(DATA_FILES_LANE)?);
                dvs.extend(lane(DELETE_FILES_LANE)?);
                removed.extend(lane(REMOVED_DELETE_FILES_LANE)?);
            }

            if added.is_empty() && dvs.is_empty() && removed.is_empty() {
                return Self::make_count_batch(&count_schema, 0);
            }

            let count: u64 = added.iter().map(|f| f.record_count()).sum();

            let tx = Transaction::new(&table);
            let mut action = tx.row_delta();
            if !added.is_empty() {
                action = action.add_data_files(added);
            }
            if !dvs.is_empty() {
                action = action.add_delete_files(dvs);
            }
            if !removed.is_empty() {
                action = action.remove_delete_files(removed);
            }
            if let Some(snap) = snapshot_id {
                action = action.validate_from_snapshot(snap);
            }
            action
                .apply(tx)
                .map_err(to_datafusion_error)?
                .commit(catalog.as_ref())
                .await
                .map_err(to_datafusion_error)?;

            Self::make_count_batch(&count_schema, count)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.count_schema),
            stream,
        )))
    }
}

impl IcebergMorMergeCommitExec {
    fn make_count_batch(schema: &ArrowSchemaRef, count: u64) -> DFResult<RecordBatch> {
        RecordBatch::try_new(Arc::clone(schema), vec![
            Arc::new(UInt64Array::from(vec![count])) as ArrayRef,
        ])
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}
