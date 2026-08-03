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
//! current set. Under [`MorMergeOptions::late_materialization`] (default ON)
//! the fetch is clipped further: matched positions are grouped per file and
//! only the row groups that contain them are decoded (sub-file byte-range
//! tasks against the SAME pinned snapshot), with the prior deletion-vector
//! state loaded from the delete files instead of reconstructed from a
//! whole-file scan. A file that already carries a deletion vector gets ONE
//! superseding DV (prior deleted positions unioned in, prior DV removed in
//! the same commit) — never a second live DV per data file.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use datafusion::arrow::compute::{cast, filter_record_batch, take};
use datafusion::arrow::datatypes::{
    DataType, Field, FieldRef, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
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
    conform_batch_variants, shred_record_batch, shred_shape_compatible,
    shred_types_with_arrow_from_file_schema, shredded_output_type, variant_column_count,
};
use iceberg::arrow::{
    ArrowFileReader, ArrowReader as IcebergArrowReader, PROJECTED_PARTITION_VALUE_COLUMN,
    PartitionValueCalculator, schema_to_arrow_schema,
};
use iceberg::delete_vector::DeleteVector;
use iceberg::expr::Predicate as IcebergPredicate;
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

/// Row CEILING per processing chunk inside the write node. The hash join
/// emits its unmatched-build output — the entire NOT MATCHED (insert) set —
/// as ONE batch, bypassing the output coalescer's target size. Processing
/// that whole defeats file rolling (the rolling writer only checks size
/// between write calls) and multiplies wide-row memory through every filter /
/// evaluate / cast copy. Slicing is zero-copy; everything downstream then
/// works on bounded rows. The effective chunk is the SMALLER of this ceiling
/// and the byte-derived row count ([`byte_bounded_chunk_rows`]) — a fixed
/// row count alone lets wide-TEXT rows blow the byte budget (8192 rows at
/// 500 KB/row is 4 GB).
const MOR_WRITE_CHUNK_ROWS: usize = 8192;

/// Default byte target for materialized write-path batches (see
/// [`MorMergeOptions::chunk_target_bytes`]). Peak write-path memory scales
/// roughly as target x (workers x channel depth + in-flight), so 32 MiB
/// keeps the default pool under ~0.5 GiB while staying decode-efficient.
const MOR_CHUNK_TARGET_BYTES_DEFAULT: usize = 32 * 1024 * 1024;

/// Floor for byte-derived batch row counts — protects decode efficiency
/// against pathological width estimates.
const MOR_CHUNK_MIN_ROWS: usize = 32;

/// Rows per chunk for `batch` under `target_bytes`: the measured in-memory
/// row width divides the byte target, clamped to
/// [[`MOR_CHUNK_MIN_ROWS`], [`MOR_WRITE_CHUNK_ROWS`]].
fn byte_bounded_chunk_rows(batch: &RecordBatch, target_bytes: usize) -> usize {
    if batch.num_rows() == 0 {
        return MOR_WRITE_CHUNK_ROWS;
    }
    let row_bytes = (batch.get_array_memory_size() / batch.num_rows()).max(1);
    (target_bytes / row_bytes).clamp(MOR_CHUNK_MIN_ROWS, MOR_WRITE_CHUNK_ROWS)
}

/// How many existing data files to probe (parquet footers) when deriving the
/// variant shredding layout under `write.parquet.shred-variants`. First
/// non-canonical derivation wins per column; probing stops early once every
/// variant column has one. Sized to survive MIXED estates: a run of canonical
/// files (e.g. merges written before the property was set) at the head of the
/// plan order must not exhaust the probe budget before a shredded file is
/// seen — footer reads are cheap metadata GETs.
const MERGE_SHRED_PROBE_FILES: usize = 16;

/// Default writer-pool size when [`MorMergeOptions::write_workers`] is unset.
/// Bounded low: each writer buffers its own parquet row groups, so memory
/// scales with pool size on wide rows.
const MOR_DEFAULT_WRITE_WORKERS: usize = 4;

/// Concurrent deletion-vector (Puffin) uploads during DV construction.
const MOR_DV_WRITE_CONCURRENCY: usize = 8;

/// Concurrent parquet-footer reads while planning the row-group-ranged
/// late-materialization fetch (metadata GETs only).
const MOR_FETCH_FOOTER_CONCURRENCY: usize = 8;

/// Session-level execution options for the MoR MERGE, set by the caller via
/// `SessionConfig::with_extension(Arc<MorMergeOptions>)`.
#[derive(Debug)]
pub struct MorMergeOptions {
    /// Cooperative deadline for the scan/write phase. Enforced at every
    /// await point of the write node BEFORE files are handed to the commit
    /// node — a merge past its deadline aborts with an error and NO snapshot
    /// is written. The commit itself is deliberately never cancelled
    /// (aborting a REST commit in flight leaves the outcome unknown).
    pub deadline: Option<std::time::Instant>,
    /// Writer-pool size for the append paths. Appended batches (INSERT rows
    /// and late-materialized updated rows) fan out round-robin to this many
    /// writer tasks, each owning its own rolling file writer.
    pub write_workers: Option<usize>,
    /// Row-group-ranged late materialization of the matched-row fetch
    /// (default ON). Matched positions are grouped by data file and only the
    /// row groups containing them are decoded — sub-file byte-range tasks
    /// against the same pinned snapshot — while the prior deletion-vector
    /// state comes from loading the delete files directly (no data rows).
    /// OFF is the kill switch restoring the previous behavior: every
    /// affected file is decoded in full (all alive rows, full width) and the
    /// prior-delete state is reconstructed by inverting that scan. Both
    /// paths commit byte-identical results.
    pub late_materialization: bool,
    /// Byte target for materialized batches on the write path (consume-loop
    /// chunks and the late-fetch decode batch size). Row counts are derived
    /// from this against the measured/estimated row width, so wide-TEXT rows
    /// get proportionally smaller batches. `None` = the built-in default
    /// ([`MOR_CHUNK_TARGET_BYTES_DEFAULT`]).
    pub chunk_target_bytes: Option<usize>,
    /// Inline DV micro-reabsorb: after the RowDelta commit, rewrite THIS
    /// merge's DV-target files whose dead fraction (consolidated DV
    /// cardinality / file rows) exceeds this threshold, reabsorbing their
    /// DVs — hot tables self-maintain instead of waiting for the periodic
    /// maintenance pass. The rewrite is a second, OPTIONAL commit from the
    /// same run: skipped when less than [`REABSORB_MIN_REMAINING`] of the
    /// deadline remains (hygiene never spends freshness budget), capped at
    /// [`REABSORB_MAX_FILES`] deepest files per merge, and aborted — never
    /// retried — on a concurrent conflict (the Replace-side rebase
    /// validation); the threshold simply re-fires on the next merge.
    /// `None` = disabled.
    pub reabsorb_dead_frac: Option<f64>,
}

/// Inline reabsorb runs only when at least this much of the merge deadline
/// remains.
const REABSORB_MIN_REMAINING: std::time::Duration = std::time::Duration::from_secs(60);
/// Per-merge cap on inline-reabsorb rewrites (deepest dead-fraction first);
/// the remainder re-fires on later merges — the amortization backstop.
const REABSORB_MAX_FILES: usize = 32;

impl MorMergeOptions {
    fn chunk_target_bytes(&self) -> usize {
        self.chunk_target_bytes
            .unwrap_or(MOR_CHUNK_TARGET_BYTES_DEFAULT)
            .max(1)
    }
}

impl Default for MorMergeOptions {
    fn default() -> Self {
        Self {
            deadline: None,
            write_workers: None,
            late_materialization: true,
            chunk_target_bytes: None,
            reabsorb_dead_frac: None,
        }
    }
}

fn deadline_error(what: &str) -> DataFusionError {
    DataFusionError::Execution(format!(
        "MERGE deadline exceeded while {what}; the merge was aborted before \
         commit — no snapshot was written"
    ))
}

/// Await `fut`, aborting with a deadline error when the merge deadline
/// passes first.
async fn with_deadline<T>(
    deadline: Option<std::time::Instant>,
    what: &str,
    fut: impl Future<Output = T>,
) -> DFResult<T> {
    match deadline {
        None => Ok(fut.await),
        Some(d) => tokio::time::timeout_at(tokio::time::Instant::from_std(d), fut)
            .await
            .map_err(|_| deadline_error(what)),
    }
}

/// Shared, immutable context each writer task works from.
#[derive(Debug)]
struct WriterCtx {
    table: Table,
    table_schema: iceberg::spec::SchemaRef,
    table_arrow: ArrowSchemaRef,
    clauses: Arc<Vec<MorClausePlan>>,
    /// Plain shredding types per variant column (empty = canonical output).
    shred_plain: HashMap<String, DataType>,
    /// Writer schema overrides matching the shredded batches.
    shred_overrides: HashMap<String, DataType>,
    /// One id per merge — combined with the worker index for file names.
    run_id: Uuid,
    deadline: Option<std::time::Instant>,
}

/// One unit of append work, routed to a writer task.
enum WriteItem {
    /// A clause-filtered chunk of NOT MATCHED rows; the worker builds the
    /// full-width rows from the clause's INSERT assignments.
    Insert { subset: RecordBatch, clause: usize },
    /// A late-fetch batch with its matched rows; the worker builds the
    /// updated row versions (SET columns overridden) per clause.
    Fetched {
        batch: RecordBatch,
        rows: Vec<u32>,
        stored: Vec<MatchedRow>,
        /// Chunked evaluated SET values: [clause][chunk][assignment column].
        update_values: Arc<Vec<Vec<Vec<ArrayRef>>>>,
    },
}

/// Round-robin fan-out of append work to writer tasks. Each task owns its
/// own writer chain (UNIQUE file-name prefix per worker — a shared prefix
/// would collide file names across workers, every generator counting from
/// zero, and silently overwrite output). Bounded channels keep memory flat
/// under backpressure. Parallelizing here (not via input partitioning)
/// keeps the matched-row bookkeeping — the double-match guard and the
/// per-file DV consolidation — on the single consuming thread.
struct WriterPool {
    txs: Vec<tokio::sync::mpsc::Sender<WriteItem>>,
    handles: Vec<tokio::task::JoinHandle<DFResult<Vec<DataFile>>>>,
    next: usize,
    deadline: Option<std::time::Instant>,
}

impl WriterPool {
    fn spawn(ctx: Arc<WriterCtx>, workers: usize) -> Self {
        let deadline = ctx.deadline;
        let mut txs = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for idx in 0..workers {
            let (tx, rx) = tokio::sync::mpsc::channel::<WriteItem>(2);
            handles.push(tokio::spawn(writer_task(Arc::clone(&ctx), idx, rx)));
            txs.push(tx);
        }
        Self {
            txs,
            handles,
            next: 0,
            deadline,
        }
    }

    async fn dispatch(&mut self, item: WriteItem) -> DFResult<()> {
        let i = self.next % self.txs.len();
        self.next += 1;
        with_deadline(
            self.deadline,
            "dispatching merge output to a writer",
            self.txs[i].send(item),
        )
        .await?
        .map_err(|_| DataFusionError::Internal("merge writer task terminated early".to_string()))
    }

    /// Close every writer and collect the data files they produced.
    async fn finish(self) -> DFResult<Vec<DataFile>> {
        drop(self.txs);
        let mut files = Vec::new();
        for h in self.handles {
            let joined = with_deadline(self.deadline, "closing merge writers", h)
                .await?
                .map_err(|e| {
                    DataFusionError::Internal(format!("merge writer task panicked: {e}"))
                })?;
            files.extend(joined?);
        }
        Ok(files)
    }
}

/// One writer task: builds full-width rows for its items, shreds/partitions
/// them, and writes through its own rolling writer chain. The writer is
/// created lazily so an idle worker leaves no file behind.
async fn writer_task(
    ctx: Arc<WriterCtx>,
    idx: usize,
    mut rx: tokio::sync::mpsc::Receiver<WriteItem>,
) -> DFResult<Vec<DataFile>> {
    let table_props = ctx
        .table
        .metadata()
        .table_properties()
        .map_err(to_datafusion_error)?;
    let parquet_writer_builder =
        ParquetWriterBuilder::from_table_properties(&table_props, ctx.table_schema.clone())
            .with_variant_shred_types(ctx.shred_overrides.clone());
    let location_generator =
        DefaultLocationGenerator::new(ctx.table.metadata()).map_err(to_datafusion_error)?;
    let file_name_generator = DefaultFileNameGenerator::new(
        format!("merge-{}-w{idx}", ctx.run_id),
        None,
        DataFileFormat::Parquet,
    );
    let rolling_writer_builder = RollingFileWriterBuilder::new(
        parquet_writer_builder,
        table_props.write_target_file_size_bytes,
        ctx.table.file_io().clone(),
        location_generator,
        file_name_generator,
    );
    let partition_spec = ctx.table.metadata().default_partition_spec().clone();
    let partition_calc = if partition_spec.is_unpartitioned() {
        None
    } else {
        Some(
            PartitionValueCalculator::try_new(&partition_spec, &ctx.table_schema)
                .map_err(to_datafusion_error)?,
        )
    };
    let mut pending = Some((
        DataFileWriterBuilder::new(rolling_writer_builder),
        partition_spec,
    ));
    let mut writer = None;

    while let Some(item) = rx.recv().await {
        if let Some(d) = ctx.deadline
            && std::time::Instant::now() >= d
        {
            return Err(deadline_error("writing merge output"));
        }
        let outs: Vec<RecordBatch> = match item {
            WriteItem::Insert { subset, clause } => {
                let MorActionPlan::Insert(assignments) = &ctx.clauses[clause].action else {
                    return Err(DataFusionError::Internal(
                        "insert work routed to a non-insert clause".to_string(),
                    ));
                };
                vec![build_full_rows(
                    &subset,
                    assignments,
                    &ctx.table_arrow,
                    None,
                )?]
            }
            WriteItem::Fetched {
                batch,
                rows,
                stored,
                update_values,
            } => build_update_rows(
                &batch,
                &rows,
                &stored,
                &ctx.clauses,
                &update_values,
                &ctx.table_arrow,
                &ctx.shred_overrides,
            )?,
        };
        for out in outs {
            // Shredded passthrough: columns already carrying the writer's
            // shredded type (a passthrough late-fetch) skip the shred kernel;
            // canonical columns (INSERT rows, or folded fetches from
            // non-matching files) are shredded as usual.
            let to_shred: HashMap<String, DataType> = ctx
                .shred_plain
                .iter()
                .filter(|(name, _)| match out.schema().index_of(name.as_str()) {
                    Ok(i) => {
                        ctx.shred_overrides.get(name.as_str())
                            != Some(out.schema().field(i).data_type())
                    }
                    Err(_) => true,
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let out = if to_shred.is_empty() {
                out
            } else {
                // The kernel's output layout differs from the file-derived
                // writer layout (child order, BinaryView); conform it —
                // columnar metadata shuffling, not row work.
                let shredded = shred_record_batch(&out, &to_shred).map_err(to_datafusion_error)?;
                conform_batch_variants(&shredded, &ctx.shred_overrides)
                    .map_err(to_datafusion_error)?
            };
            let out = with_partition_column(out, partition_calc.as_ref())?;
            if writer.is_none() {
                let (builder, spec) = pending.take().expect("writer built once");
                writer = Some(
                    TaskWriter::try_new(
                        builder,
                        table_props.write_datafusion_fanout_enabled,
                        ctx.table_schema.clone(),
                        spec,
                    )
                    .map_err(to_datafusion_error)?,
                );
            }
            writer
                .as_mut()
                .expect("writer just created")
                .write(out)
                .await
                .map_err(to_datafusion_error)?;
        }
    }
    match writer {
        Some(w) => w.close().await.map_err(to_datafusion_error),
        None => Ok(Vec::new()),
    }
}

/// Build the updated row versions for one late-fetch batch: the fetched row
/// with the claiming clause's SET columns overridden, grouped by
/// (clause, store chunk) — different clauses assign different columns, and
/// the evaluated SET values live in per-chunk arrays that are indexed
/// directly (never concatenated into one whole-merge array).
fn build_update_rows(
    fbatch: &RecordBatch,
    fetch_rows: &[u32],
    stored: &[MatchedRow],
    clauses: &[MorClausePlan],
    update_values: &[Vec<Vec<ArrayRef>>],
    table_arrow: &ArrowSchemaRef,
    shred_overrides: &HashMap<String, DataType>,
) -> DFResult<Vec<RecordBatch>> {
    let fb_schema = fbatch.schema();
    let mut by_clause_chunk: HashMap<(u32, u32), (Vec<u32>, Vec<u32>)> = HashMap::new();
    for (i, m) in stored.iter().enumerate() {
        let e = by_clause_chunk
            .entry((m.clause, m.update_chunk))
            .or_default();
        e.0.push(fetch_rows[i]);
        e.1.push(m.update_row);
    }
    let mut out_batches = Vec::with_capacity(by_clause_chunk.len());
    for ((clause, chunk), (rows, store_rows)) in by_clause_chunk {
        let MorActionPlan::Update(assignments) = &clauses[clause as usize].action else {
            return Err(DataFusionError::Internal(
                "only update clauses collect fetch rows".to_string(),
            ));
        };
        let take_idx = UInt32Array::from(rows);
        let store_idx = UInt32Array::from(store_rows);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(table_arrow.fields().len());
        let mut out_fields: Vec<FieldRef> = Vec::with_capacity(table_arrow.fields().len());
        for field in table_arrow.fields() {
            let assigned = assignments
                .iter()
                .position(|(name, _)| name == field.name());
            let arr = match assigned {
                Some(a) => take(
                    update_values[clause as usize][chunk as usize][a].as_ref(),
                    &store_idx,
                    None,
                )?,
                None => {
                    let src_idx = fb_schema
                        .index_of(field.name())
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                    take(fbatch.column(src_idx).as_ref(), &take_idx, None)?
                }
            };
            // Shredded passthrough: a fetched variant column that already
            // carries the writer's shredded type is adopted verbatim — the
            // output field takes the shredded type instead of casting the
            // column back to canonical.
            let adopt_shredded = assigned.is_none()
                && shred_overrides.get(field.name().as_str()) == Some(arr.data_type());
            let (arr, out_field) = if adopt_shredded {
                let f = Arc::new(
                    field
                        .as_ref()
                        .clone()
                        .with_data_type(arr.data_type().clone()),
                );
                (arr, f)
            } else if arr.data_type() == field.data_type() {
                (arr, Arc::clone(field))
            } else {
                (cast(arr.as_ref(), field.data_type())?, Arc::clone(field))
            };
            columns.push(arr);
            out_fields.push(out_field);
        }
        let out_schema = Arc::new(ArrowSchema::new_with_metadata(
            out_fields,
            table_arrow.metadata().clone(),
        ));
        let out = RecordBatch::try_new(out_schema, columns)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        out_batches.push(out);
    }
    Ok(out_batches)
}

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
    /// Iceberg predicate from target-only ON residuals — partition/file
    /// PRUNING only (best-effort; may be None when unconvertible).
    predicate: Option<IcebergPredicate>,
    /// The same residuals bound row-exactly against the scan output schema;
    /// the semantics guarantee (applied to every batch).
    residual: Option<Arc<dyn PhysicalExpr>>,
}

impl IcebergMorTargetScanExec {
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        columns: Vec<String>,
        data_fields: Vec<Field>,
        predicate: Option<IcebergPredicate>,
        residual: Option<Arc<dyn PhysicalExpr>>,
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
            predicate,
            residual,
        }
    }
}

impl DisplayAs for IcebergMorTargetScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IcebergMorTargetScanExec: columns=[{}]",
            self.columns.join(",")
        )?;
        if let Some(p) = &self.predicate {
            write!(f, " prune:[{p}]")?;
        }
        if let Some(r) = &self.residual {
            write!(f, " residual:[{r}]")?;
        }
        Ok(())
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
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let table = self.table.clone();
        let snapshot_id = self.snapshot_id;
        // Decode-memory accounting (crate::memory_gate): the target scan's
        // reads register against the session pool like any provider scan.
        let scan_gate = crate::memory_gate::pool_scan_gate(&context.runtime_env().memory_pool);
        let mut select: Vec<String> = self.columns.clone();
        select.push(RESERVED_COL_NAME_FILE.to_string());
        select.push(RESERVED_COL_NAME_POS.to_string());
        let out_schema = Arc::clone(&self.schema);
        let predicate = self.predicate.clone();
        let residual = self.residual.clone();

        let stream_schema = Arc::clone(&out_schema);
        let fut = async move {
            let mut builder = table.scan();
            if let Some(id) = snapshot_id {
                builder = builder.snapshot_id(id);
            }
            if let Some(pred) = predicate {
                builder = builder.with_filter(pred);
            }
            if let Some(gate) = scan_gate {
                builder = builder.with_scan_memory_gate(gate);
            }
            let scan = builder
                .select(select)
                .build()
                .map_err(to_datafusion_error)?;
            let stream = scan.to_arrow().await.map_err(to_datafusion_error)?;
            let out = stream.map(move |batch| {
                let batch = batch.map_err(to_datafusion_error)?;
                let batch = plain_cast_batch(&batch, &stream_schema)?;
                match &residual {
                    None => Ok(batch),
                    Some(expr) => {
                        let mask = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
                        let mask = mask
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                            .ok_or_else(|| {
                                DataFusionError::Internal(
                                    "target residual must evaluate to boolean".to_string(),
                                )
                            })?
                            .clone();
                        filter_record_batch(&batch, &mask)
                            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                    }
                }
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
        let options = context
            .session_config()
            .get_extension::<MorMergeOptions>()
            .unwrap_or_default();
        let memory_pool = Arc::clone(&context.runtime_env().memory_pool);
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
                clauses,
                options,
                memory_pool,
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
    /// Which evaluated-SET-values chunk of the clause's store holds this
    /// row's values (chunks are never concatenated — a whole-merge concat
    /// would build one giant contiguous array per SET column).
    update_chunk: u32,
    /// Row within that chunk (chunks are at most `MOR_WRITE_CHUNK_ROWS`).
    update_row: u32,
}

/// Derive per-variant-column PLAIN shredding types for the merge output by
/// probing existing data files' parquet footers (shred-preserving — the
/// layout is carried forward from what the table already stores, never
/// invented). An empty table, or one whose files are all canonical, derives
/// nothing and the output stays canonical.
async fn merge_shred_types(
    table: &Table,
    snapshot_id: Option<i64>,
) -> DFResult<HashMap<String, (DataType, DataType)>> {
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
    let mut out: HashMap<String, (DataType, DataType)> = HashMap::new();
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
        for (name, pair) in shred_types_with_arrow_from_file_schema(meta.schema(), table_schema) {
            out.entry(name).or_insert(pair);
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
    clauses: Arc<Vec<MorClausePlan>>,
    options: Arc<MorMergeOptions>,
    memory_pool: Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
    result_schema: &ArrowSchemaRef,
) -> DFResult<RecordBatch> {
    // #18 read-RSS accounting: the late-materialization fetch registers
    // decode working sets, and the merge's own residents (evaluated-SET
    // store, matched bookkeeping, writer buffers) grow a pool consumer —
    // so `peak_mem_bytes` reflects true residency and a bounded pool fails
    // CLEANLY (checkpointable) instead of the kernel killing the process.
    let scan_gate = crate::memory_gate::pool_scan_gate(&memory_pool);
    let mut resident_reservation = crate::memory_gate::scan_gate_enabled().then(|| {
        datafusion::execution::memory_pool::MemoryConsumer::new("mor-merge-residents")
            .register(&memory_pool)
    });
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
    let deadline = options.deadline;

    // Shred-preserving output under `write.parquet.shred-variants`: derive
    // each variant column's layout from the table's existing files, and give
    // the writer the exact shredded arrow types the (shredded) batches will
    // carry. Empty map = canonical output, the untouched default.
    let shred_pairs = if table_props.parquet_shred_variants {
        with_deadline(deadline, "deriving the shredding layout", async {
            merge_shred_types(&table, snapshot_id).await
        })
        .await??
    } else {
        HashMap::new()
    };
    // The writer layout is the FILE's own arrow type (not the shred kernel's
    // probe output): late-fetched columns from same-layout files then compare
    // equal and PASS THROUGH; kernel-shredded batches are conformed to it
    // (child reorder + view->binary) in the writer task.
    let shred_plain: HashMap<String, DataType> = shred_pairs
        .iter()
        .map(|(name, (plain, _))| (name.clone(), plain.clone()))
        .collect();
    let shred_overrides: HashMap<String, DataType> = shred_pairs
        .into_iter()
        .map(|(name, (plain, file_type))| {
            let idx = table_arrow
                .index_of(&name)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let kernel_type = shredded_output_type(table_arrow.field(idx).data_type(), &plain)
                .map_err(to_datafusion_error)?;
            // Only adopt the file layout when the kernel's output can be
            // conformed onto it (same child sets, order/view aside); a
            // value-only shred node in the file would make the kernel output
            // unconformable -> keep the kernel layout for that column (no
            // passthrough, PQ-1 behavior).
            let target = if shred_shape_compatible(&kernel_type, &file_type) {
                file_type
            } else {
                kernel_type
            };
            Ok((name, target))
        })
        .collect::<DFResult<_>>()?;

    // Appended output fans out to a writer pool — the encode/compress/upload
    // path is the write node's wall-clock at demote-heavy scale, and one
    // writer serialized all of it. Matched-row bookkeeping stays here on the
    // single consuming thread.
    let write_workers = options
        .write_workers
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(2)
                .min(MOR_DEFAULT_WRITE_WORKERS)
        })
        .max(1);
    let ctx = Arc::new(WriterCtx {
        table: table.clone(),
        table_schema: table_schema.clone(),
        table_arrow: Arc::clone(&table_arrow),
        clauses: Arc::clone(&clauses),
        shred_plain,
        shred_overrides,
        run_id: Uuid::now_v7(),
        deadline,
    });
    // Parquet writers buffer pages between row-group flushes; account a
    // modest 16 MiB floor per worker. (True high-water tracking via the
    // writers' in-progress sizes is a follow-up — the observed kill classes
    // are decode + the SET store, both accounted exactly; a large upfront
    // writer reservation would instead make small bounded pools unusable.)
    if let Some(r) = resident_reservation.as_mut() {
        r.try_grow(write_workers * 16 * 1024 * 1024)?;
    }
    let mut pool = WriterPool::spawn(Arc::clone(&ctx), write_workers);

    let clause_idx_col = input_schema
        .index_of(MOR_CLAUSE_COL)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
    // _file/_pos sit at the tail of the target side, right before __mor_clause.
    let pos_idx = clause_idx_col - 1;
    let file_idx = clause_idx_col - 2;

    // Per-UPDATE-clause store of evaluated SET values, kept CHUNKED for the
    // merge's lifetime — rows are addressed as (chunk, row) and taken per
    // chunk at write time. Never concatenate the store: one whole-merge
    // concat per SET column materializes a single giant contiguous array
    // (2x peak memory, and the historical i32 offset-overflow site on
    // wide-TEXT columns).
    let n_clauses = clauses.len();
    let mut update_stores: Vec<Vec<Vec<ArrayRef>>> = vec![Vec::new(); n_clauses];
    // (file, pos) -> matched bookkeeping. Duplicate claims are an error.
    let mut matched: HashMap<(String, u64), MatchedRow> = HashMap::new();
    let chunk_target_bytes = options.chunk_target_bytes();

    while let Some(batch) =
        with_deadline(deadline, "reading the merge input", input.try_next()).await??
    {
        // Byte-budgeted chunking: MOR_WRITE_CHUNK_ROWS is only the row
        // CEILING — wide rows shrink the chunk so the materialized copies
        // downstream (filter, SET evaluation, full-width build) stay near
        // the byte target regardless of row width.
        let chunk_rows = byte_bounded_chunk_rows(&batch, chunk_target_bytes);
        let mut chunk_start = 0;
        while chunk_start < batch.num_rows() {
            let chunk_len = chunk_rows.min(batch.num_rows() - chunk_start);
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
                    MorActionPlan::Insert(_) => {
                        pool.dispatch(WriteItem::Insert { subset, clause: ci })
                            .await?;
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
                            update_stores[ci].len() as u32,
                            &mut matched,
                        )?;
                        if let Some(r) = resident_reservation.as_mut() {
                            // Evaluated arrays live until write time; matched
                            // bookkeeping adds ~96B/row (path key + entry).
                            let chunk_bytes: usize = evaluated
                                .iter()
                                .map(|a| a.get_array_memory_size())
                                .sum::<usize>()
                                + subset.num_rows() * 96;
                            r.try_grow(chunk_bytes)?;
                        }
                        update_stores[ci].push(evaluated);
                    }
                    MorActionPlan::Delete => {
                        record_matched(&subset, file_idx, pos_idx, ci as u32, 0, &mut matched)?;
                    }
                }
            }
        }
    }

    // Hand the CHUNKED store to the writers as-is — (chunk, row) addressing
    // replaces the former whole-merge per-column concat.
    let update_values: Arc<Vec<Vec<Vec<ArrayRef>>>> = Arc::new(update_stores);

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
        let tasks: Vec<FileScanTask> =
            with_deadline(deadline, "planning the late-materialization scan", async {
                scan.plan_files()
                    .await
                    .map_err(to_datafusion_error)?
                    .try_collect()
                    .await
                    .map_err(to_datafusion_error)
            })
            .await??;
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

        // Matched positions per file, ascending — drives the consolidated-DV
        // union and (late path) the row-group-ranged fetch planning.
        let mut positions_by_file: HashMap<String, Vec<u64>> = HashMap::new();
        for (file, pos) in matched.keys() {
            positions_by_file
                .entry(file.clone())
                .or_default()
                .push(*pos);
        }
        for positions in positions_by_file.values_mut() {
            positions.sort_unstable();
        }
        let late = options.late_materialization;

        // Late materialization: read ONLY the affected files, full projection
        // + _file/_pos — and with `late_materialization` (default ON) only
        // the ROW GROUPS containing matched positions, via sub-file
        // byte-range tasks over the same pinned snapshot.
        //
        // Shredded passthrough: when the writer outputs the SAME shredded
        // layout (`write.parquet.shred-variants`), fetched rows carry their
        // physical shredded variant columns straight through to the writer —
        // skipping BOTH the per-row unshred fold at read and the per-row
        // re-shred at write (the dominant merge cost on shredded estates).
        // Per file: a non-matching file (canonical, or a different layout)
        // still folds and re-shreds. Disabled when any SET assignment writes
        // a variant column (assignments evaluate against the canonical type).
        let passthrough_ok = !ctx.shred_overrides.is_empty()
            && !clauses.iter().any(|c| match &c.action {
                MorActionPlan::Update(assignments) => assignments
                    .iter()
                    .any(|(name, _)| ctx.shred_overrides.contains_key(name)),
                _ => false,
            });
        // The fetch task list: late -> sub-file byte-range tasks covering
        // only the row groups holding matched positions (whole-file fallback
        // per file when its layout cannot be ranged exactly), plus the
        // widest per-file uncompressed row width from the footers; legacy ->
        // the whole affected files.
        let (read_tasks, est_row_bytes): (Vec<FileScanTask>, Option<usize>) = if late {
            with_deadline(
                deadline,
                "planning the ranged late-materialization fetch",
                plan_ranged_fetch_tasks(&table, &fetch_tasks, &positions_by_file),
            )
            .await??
        } else {
            (fetch_tasks.clone(), None)
        };

        let mut reader_builder = table.reader_builder();
        if let Some(gate) = scan_gate.clone() {
            reader_builder = reader_builder.with_scan_memory_gate(gate);
        }
        if passthrough_ok {
            reader_builder = reader_builder.with_shredded_passthrough(ctx.shred_overrides.clone());
        }
        // Byte-aware decode batches: without a hint the parquet reader's
        // default row-count batches multiply per-row width unboundedly —
        // wide-TEXT rows get proportionally fewer rows per batch.
        if let Some(row_bytes) = est_row_bytes {
            let batch_rows = (chunk_target_bytes / row_bytes.max(1))
                .clamp(MOR_CHUNK_MIN_ROWS, MOR_WRITE_CHUNK_ROWS);
            reader_builder = reader_builder.with_batch_size(batch_rows);
        }
        let reader = reader_builder.build();

        // Prior positional-delete state per affected file:
        // - late path: loaded straight from the delete files (no data rows);
        //   the load shares the reader's delete cache, so the fetch below
        //   reuses it instead of re-fetching.
        // - legacy path: reconstructed from the whole-file alive scan below.
        let mut prior_deletes: HashMap<String, DeleteVector> = if late {
            with_deadline(
                deadline,
                "loading the prior delete state",
                reader.load_positional_deletes(&fetch_tasks),
            )
            .await?
            .map_err(to_datafusion_error)?
        } else {
            HashMap::new()
        };

        let task_stream = Box::pin(futures::stream::iter(
            read_tasks.into_iter().map(Ok).collect::<Vec<_>>(),
        )) as iceberg::scan::FileScanTaskStream;
        let mut fetch_stream = reader
            .read(task_stream)
            .map_err(to_datafusion_error)?
            .stream();

        let mut alive: HashMap<String, RoaringTreemap> = HashMap::new();
        while let Some(fbatch) = with_deadline(
            deadline,
            "reading matched rows for late materialization",
            fetch_stream.try_next(),
        )
        .await?
        .map_err(to_datafusion_error)?
        {
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
                if !late {
                    // Legacy whole-file fetch: the prior-deleted set is
                    // reconstructed by inverting the alive positions.
                    alive.entry(file.to_string()).or_default().insert(pos);
                }
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
            // The updated row versions (fetched row, SET columns overridden)
            // are built inside the writer task — the full-width copy is part
            // of the parallelized write path.
            pool.dispatch(WriteItem::Fetched {
                batch: fbatch,
                rows: fetch_rows,
                stored,
                update_values: Arc::clone(&update_values),
            })
            .await?;
        }

        // One consolidated DV per affected file: everything already deleted
        // plus this merge's matched positions. The prior-deleted set comes
        // from the loaded delete state (late path) or from inverting the
        // whole-file alive scan (legacy path) — identical by construction.
        // The bitmaps are built synchronously (they read the bookkeeping
        // maps); the Puffin uploads are small independent PUTs, written
        // concurrently from fully owned state.
        let location = table.metadata().location().to_string();
        let mut dv_jobs = Vec::with_capacity(per_file.len());
        for (file, task) in &per_file {
            let mut bitmap = if late {
                prior_deletes
                    .remove(file)
                    .map(DeleteVector::into_inner)
                    .unwrap_or_default()
            } else {
                let record_count = task.record_count.ok_or_else(|| {
                    DataFusionError::Internal(format!("no record count for data file {file}"))
                })?;
                let alive_set = alive.remove(file).unwrap_or_default();
                let mut inverted = RoaringTreemap::new();
                for pos in 0..record_count {
                    if !alive_set.contains(pos) {
                        inverted.insert(pos);
                    }
                }
                inverted
            };
            for pos in positions_by_file.get(file).into_iter().flatten() {
                bitmap.insert(*pos);
            }
            let partition = task.partition.clone().unwrap_or(Struct::empty());
            let spec_id = task
                .partition_spec
                .as_ref()
                .map(|s| s.spec_id())
                .unwrap_or_else(|| table.metadata().default_partition_spec_id());
            let dv_path = format!("{location}/data/{}-deletes.puffin", Uuid::now_v7());
            dv_jobs.push((file.clone(), bitmap, partition, spec_id, dv_path));
        }
        let file_io = table.file_io().clone();
        let dv_futs =
            dv_jobs
                .into_iter()
                .map(move |(file, bitmap, partition, spec_id, dv_path)| {
                    let file_io = file_io.clone();
                    async move {
                        DeleteVector::new(bitmap)
                            .write_to_puffin_file(&file_io, dv_path, file, partition, spec_id)
                            .await
                            .map_err(to_datafusion_error)
                    }
                });
        new_delete_files = with_deadline(
            deadline,
            "writing deletion vectors",
            futures::stream::iter(dv_futs)
                .buffer_unordered(MOR_DV_WRITE_CONCURRENCY)
                .try_collect::<Vec<DataFile>>(),
        )
        .await??;

        // Resolve the prior delete files being superseded to their manifest
        // DataFile entries (sourced from the SCAN's per-file binding).
        let prior_paths: HashSet<String> = fetch_tasks
            .iter()
            .flat_map(|t| t.deletes.iter().map(|d| d.file_path.clone()))
            .collect();
        if !prior_paths.is_empty() {
            removed_delete_files = with_deadline(
                deadline,
                "resolving superseded delete files",
                resolve_delete_files(&table, snapshot_id, &prior_paths),
            )
            .await??;
        }
    }

    let data_files = pool.finish().await?;

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

/// Plan the row-group-ranged late-materialization fetch: for each affected
/// file, read the parquet footer and clip its task to byte ranges covering
/// ONLY the row groups that contain matched positions — the reader's
/// byte-range row-group ownership then decodes just those groups, and the
/// `_pos` virtual column stays file-absolute, so matched rows resolve
/// exactly as under a whole-file read. A file whose layout cannot be ranged
/// exactly falls back to its whole-file task. Footer reads are cheap
/// metadata GETs, issued with bounded concurrency.
///
/// Also returns the WIDEST per-file mean uncompressed row width seen in the
/// footers (bytes/row from row-group `total_byte_size`), which sizes the
/// fetch decode batches by bytes — the footers are already in hand, so the
/// estimate is free.
async fn plan_ranged_fetch_tasks(
    table: &Table,
    fetch_tasks: &[FileScanTask],
    positions_by_file: &HashMap<String, Vec<u64>>,
) -> DFResult<(Vec<FileScanTask>, Option<usize>)> {
    let mut plans = Vec::with_capacity(fetch_tasks.len());
    for task in fetch_tasks {
        let positions = positions_by_file
            .get(task.data_file_path())
            .cloned()
            .unwrap_or_default();
        plans.push(plan_one_ranged_fetch(
            table.clone(),
            task.clone(),
            positions,
        ));
    }
    let nested: Vec<(Vec<FileScanTask>, Option<usize>)> = futures::stream::iter(plans)
        .buffered(MOR_FETCH_FOOTER_CONCURRENCY)
        .try_collect()
        .await?;
    let max_row_bytes = nested.iter().filter_map(|(_, w)| *w).max();
    Ok((
        nested.into_iter().flat_map(|(tasks, _)| tasks).collect(),
        max_row_bytes,
    ))
}

/// Clip ONE affected file's task to the byte ranges owning its matched
/// positions (whole-file fallback when the layout cannot be ranged exactly).
/// The second element is the file's mean UNCOMPRESSED bytes/row from its
/// row-group stats (None when the footer reports no rows).
async fn plan_one_ranged_fetch(
    table: Table,
    task: FileScanTask,
    positions: Vec<u64>,
) -> DFResult<(Vec<FileScanTask>, Option<usize>)> {
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
                    "loading parquet footer of {} for the ranged fetch: {e}",
                    task.data_file_path()
                )
                .into(),
            )
        })?;
    let (total_bytes, total_rows) = meta
        .metadata()
        .row_groups()
        .iter()
        .fold((0i64, 0i64), |(b, r), rg| {
            (b + rg.total_byte_size(), r + rg.num_rows())
        });
    let row_bytes = (total_rows > 0).then(|| ((total_bytes / total_rows).max(1)) as usize);
    let ranges = IcebergArrowReader::byte_ranges_for_row_positions(meta.metadata(), &positions)
        .map_err(to_datafusion_error)?;
    let tasks = match ranges {
        Some(ranges) => ranges
            .into_iter()
            .map(|(start, length)| {
                let mut clipped = task.clone();
                clipped.start = start;
                clipped.length = length;
                // Sub-file read: the manifest record count no longer
                // describes what this task reads.
                clipped.record_count = None;
                clipped
            })
            .collect(),
        None => vec![task],
    };
    Ok((tasks, row_bytes))
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

/// Record each row of `subset` as matched (file, pos) -> (clause, chunk, row).
/// A row already claimed by another source row is an error (SQL MERGE forbids
/// updating or deleting the same target row twice).
fn record_matched(
    subset: &RecordBatch,
    file_idx: usize,
    pos_idx: usize,
    clause: u32,
    store_chunk: u32,
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
                update_chunk: store_chunk,
                update_row: row as u32,
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
        let count_schema: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("count", DataType::UInt64, false),
            Field::new("reabsorbed_files", DataType::UInt64, false),
            Field::new("reabsorb_ms", DataType::UInt64, false),
            Field::new("reabsorb_skipped", DataType::Utf8, true),
        ]));
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
        let options = context
            .session_config()
            .get_extension::<MorMergeOptions>()
            .unwrap_or_default();

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
                return Self::make_count_batch(&count_schema, 0, 0, 0, None);
            }

            let count: u64 = added.iter().map(|f| f.record_count()).sum();
            // The reabsorb needs this merge's DV descriptors after the
            // RowDelta consumes `dvs` — a handful of metadata clones.
            let merge_dvs = dvs.clone();

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
            action = match snapshot_id {
                Some(snap) => action.validate_from_snapshot(snap),
                // Planned against an empty table — a concurrent first
                // writer must fail this commit, not be rebased over.
                None => action.validate_from_empty_table(),
            };
            let committed = action
                .apply(tx)
                .map_err(to_datafusion_error)?
                .commit(catalog.as_ref())
                .await
                .map_err(to_datafusion_error)?;

            // Inline DV micro-reabsorb — optional hygiene, never fails the
            // merge (the RowDelta above already committed).
            let (reab_files, reab_ms, reab_skipped) = match options.reabsorb_dead_frac {
                Some(frac) if !merge_dvs.is_empty() => {
                    inline_reabsorb(&catalog, &committed, &merge_dvs, frac, options.deadline).await
                }
                Some(_) => (0, 0, Some("threshold".to_string())),
                None => (0, 0, None),
            };

            Self::make_count_batch(&count_schema, count, reab_files, reab_ms, reab_skipped)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.count_schema),
            stream,
        )))
    }
}

impl IcebergMorMergeCommitExec {
    fn make_count_batch(
        schema: &ArrowSchemaRef,
        count: u64,
        reabsorbed_files: u64,
        reabsorb_ms: u64,
        reabsorb_skipped: Option<String>,
    ) -> DFResult<RecordBatch> {
        RecordBatch::try_new(Arc::clone(schema), vec![
            Arc::new(UInt64Array::from(vec![count])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![reabsorbed_files])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![reabsorb_ms])) as ArrayRef,
            Arc::new(StringArray::from(vec![reabsorb_skipped])) as ArrayRef,
        ])
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

/// Threshold-select this merge's DV-target files and rewrite the crossing
/// ones via the compaction engine's path-scoped entry
/// ([`iceberg_compaction::engine::compact_files`]) — DVs reabsorbed, sort +
/// shred preserved per the table contract. Returns
/// `(reabsorbed_files, elapsed_ms, skipped_reason)`; infallible by design —
/// every failure folds into the skip reason (`budget` | `threshold` |
/// `conflict` | `error:…`) and the threshold re-fires on a later merge.
async fn inline_reabsorb(
    catalog: &Arc<dyn Catalog>,
    committed: &Table,
    dvs: &[DataFile],
    dead_frac: f64,
    deadline: Option<std::time::Instant>,
) -> (u64, u64, Option<String>) {
    let t0 = std::time::Instant::now();
    if let Some(d) = deadline
        && d.checked_duration_since(std::time::Instant::now())
            .is_none_or(|rem| rem < REABSORB_MIN_REMAINING)
    {
        return (0, 0, Some("budget".to_string()));
    }
    let data = match iceberg_compaction::engine::current_data_files(committed).await {
        Ok(map) => map,
        Err(e) => return (0, 0, Some(format!("error:{e:#}"))),
    };
    // dv.record_count() is the CONSOLIDATED cardinality (prior DVs unioned
    // by the merge), i.e. the file's TOTAL dead rows.
    let mut crossing: Vec<(f64, String)> = dvs
        .iter()
        .filter_map(|dv| {
            let path = dv.referenced_data_file()?;
            let file = data.get(&path)?;
            let frac = dv.record_count() as f64 / file.record_count().max(1) as f64;
            (frac > dead_frac).then_some((frac, path))
        })
        .collect();
    if crossing.is_empty() {
        return (0, 0, Some("threshold".to_string()));
    }
    crossing.sort_by(|a, b| b.0.total_cmp(&a.0));
    crossing.truncate(REABSORB_MAX_FILES);
    let paths: HashSet<String> = crossing.into_iter().map(|(_, p)| p).collect();
    let cfg = iceberg_compaction::config::Config {
        shred_variants: committed
            .metadata()
            .table_properties()
            .map(|p| p.parquet_shred_variants)
            .unwrap_or(false),
        ..Default::default()
    };
    match iceberg_compaction::engine::compact_files(
        catalog.as_ref(),
        committed.identifier(),
        &paths,
        &cfg,
    )
    .await
    {
        Ok(out) => (out.rewritten as u64, t0.elapsed().as_millis() as u64, None),
        Err(e) => {
            let msg = format!("{e:#}");
            let reason = if msg.contains("conflicting concurrent commit") {
                "conflict".to_string()
            } else {
                let mut m = msg;
                m.truncate(160);
                format!("error:{m}")
            };
            (0, t0.elapsed().as_millis() as u64, Some(reason))
        }
    }
}

#[cfg(test)]
mod chunking_tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{Int64Array, LargeStringArray, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::{MOR_CHUNK_MIN_ROWS, MOR_WRITE_CHUNK_ROWS, byte_bounded_chunk_rows};

    fn batch(rows: usize, payload: &str) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::LargeUtf8, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(LargeStringArray::from_iter_values(std::iter::repeat_n(
                payload, rows,
            ))),
        ])
        .unwrap()
    }

    #[test]
    fn narrow_rows_use_the_row_ceiling() {
        let b = batch(1024, "x");
        assert_eq!(
            byte_bounded_chunk_rows(&b, 32 * 1024 * 1024),
            MOR_WRITE_CHUNK_ROWS
        );
    }

    #[test]
    fn wide_rows_shrink_the_chunk() {
        // ~64 KiB rows against a 1 MiB target -> low-double-digit rows,
        // never the 8192-row ceiling that would materialize 512 MiB copies.
        let b = batch(64, &"y".repeat(64 * 1024));
        let rows = byte_bounded_chunk_rows(&b, 1024 * 1024);
        assert!(rows < 64, "wide rows must shrink the chunk, got {rows}");
        assert!(rows >= MOR_CHUNK_MIN_ROWS);
    }

    #[test]
    fn floor_holds_for_pathological_widths() {
        let b = batch(4, &"z".repeat(8 * 1024 * 1024));
        assert_eq!(byte_bounded_chunk_rows(&b, 1), MOR_CHUNK_MIN_ROWS);
        // Empty batches fall back to the ceiling (nothing to measure).
        let empty = batch(0, "x");
        assert_eq!(byte_bounded_chunk_rows(&empty, 1), MOR_WRITE_CHUNK_ROWS);
    }
}
