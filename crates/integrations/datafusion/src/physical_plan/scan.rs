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

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::table::Table;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug)]
pub struct IcebergTableScan {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Stores certain, often expensive to compute,
    /// plan properties used in query optimization.
    plan_properties: Arc<PlanProperties>,
    /// Projection column names, None means all columns
    projection: Option<Vec<String>>,
    /// Filters to apply to the table scan
    predicates: Option<Predicate>,
    /// Optional limit on the number of rows to return
    limit: Option<usize>,
    /// When set, scan ONLY the data files whose path is in the set
    /// (externally planned file subset; deletes still apply).
    file_allowlist: Option<Arc<HashSet<String>>>,
}

impl IcebergTableScan {
    /// Creates a new [`IcebergTableScan`] object.
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        file_allowlist: Option<Arc<HashSet<String>>>,
    ) -> Self {
        let output_schema = match projection {
            None => schema.clone(),
            Some(projection) => Arc::new(schema.project(projection).unwrap()),
        };
        let plan_properties = Self::compute_properties(output_schema.clone());
        let projection = get_column_names(schema.clone(), projection);
        let predicates = convert_filters_to_predicate(&schema, filters);

        Self {
            table,
            snapshot_id,
            plan_properties,
            projection,
            predicates,
            limit,
            file_allowlist,
        }
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.projection.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Computes [`PlanProperties`] used in query optimization.
    fn compute_properties(schema: ArrowSchemaRef) -> Arc<PlanProperties> {
        // TODO:
        // This is more or less a placeholder, to be replaced
        // once we support output-partitioning
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn execute(
        &self,
        _partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Decode-memory accounting: reads register against the session pool
        // (peak honesty + bounded admission — crate::memory_gate).
        let gate = crate::memory_gate::pool_scan_gate(&context.runtime_env().memory_pool);
        let fut = get_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            self.projection.clone(),
            self.predicates.clone(),
            self.file_allowlist.clone(),
            gate,
        );
        let stream = futures::stream::once(fut).try_flatten();

        // Apply limit if specified
        let limited_stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> =
            if let Some(limit) = self.limit {
                let mut remaining = limit;
                Box::pin(stream.try_filter_map(move |batch| {
                    futures::future::ready(if remaining == 0 {
                        Ok(None)
                    } else if batch.num_rows() <= remaining {
                        remaining -= batch.num_rows();
                        Ok(Some(batch))
                    } else {
                        let limited_batch = batch.slice(0, remaining);
                        remaining = 0;
                        Ok(Some(limited_batch))
                    })
                }))
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            limited_stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(
            f,
            "IcebergTableScan projection:[{}] predicate:[{}]",
            self.projection
                .clone()
                .map_or(String::new(), |v| v.join(",")),
            self.predicates
                .clone()
                .map_or(String::from(""), |p| format!("{p}"))
        )?;
        if let Some(limit) = self.limit {
            write!(f, " limit:[{limit}]")?;
        }
        if let Some(files) = &self.file_allowlist {
            write!(f, " scan_files:[{}]", files.len())?;
        }
        Ok(())
    }
}

/// Asynchronously retrieves a stream of [`RecordBatch`] instances
/// from a given table.
///
/// This function initializes a [`TableScan`], builds it,
/// and then converts it into a stream of Arrow [`RecordBatch`]es.
async fn get_batch_stream(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicates: Option<Predicate>,
    file_allowlist: Option<Arc<HashSet<String>>>,
    scan_memory_gate: Option<Arc<dyn iceberg::arrow::ScanMemoryGate>>,
) -> DFResult<Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    let mut scan_builder = match column_names {
        Some(column_names) => scan_builder.select(column_names),
        None => scan_builder.select_all(),
    };
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    if let Some(files) = file_allowlist {
        // Entries are either a plain data-file path or `path@<start>+<length>`
        // — a byte-range clip (externally-planned sub-file split; the reader's
        // midpoint-ownership row-group filter makes disjoint ranges a disjoint,
        // complete cover). The suffix is unambiguous: object-store paths never
        // contain `@<digits>+<digits>` terminally.
        let mut plain: Vec<String> = Vec::new();
        let mut ranges: Vec<(String, (u64, u64))> = Vec::new();
        for f in files.iter() {
            if let Some((path, spec)) = f.rsplit_once('@') {
                if let Some((st, ln)) = spec.split_once('+') {
                    if let (Ok(st), Ok(ln)) = (st.parse::<u64>(), ln.parse::<u64>()) {
                        plain.push(path.to_string());
                        ranges.push((path.to_string(), (st, ln)));
                        continue;
                    }
                }
            }
            plain.push(f.clone());
        }
        scan_builder = scan_builder.with_data_file_path_filter(plain);
        if !ranges.is_empty() {
            scan_builder = scan_builder.with_data_file_path_ranges(ranges);
        }
    }
    // In-flight data-file read concurrency: the scan default couples this to
    // num_cpus, which under-drives many-tiny-file reads (network-RTT-bound,
    // not CPU-bound — 50 trickle files at concurrency 4 = ~13 sequential RTT
    // waves). Async I/O tasks, not threads; CPU/memory footprint is unchanged.
    if let Ok(v) = std::env::var("ICEBERG_SCAN_DATA_FILE_CONCURRENCY") {
        if let Ok(limit) = v.parse::<usize>() {
            if limit > 0 {
                scan_builder = scan_builder.with_data_file_concurrency_limit(limit);
            }
        }
    }
    // Page-index row selection (off by default, matching the scan default):
    // with a pushed-down predicate, build parquet RowSelections from the
    // column/offset indexes so only matching pages decode — the page-level
    // pruning a sorted layout unlocks. ICEBERG_SCAN_ROW_SELECTION=1|true.
    if matches!(
        std::env::var("ICEBERG_SCAN_ROW_SELECTION").as_deref(),
        Ok("1") | Ok("true")
    ) {
        scan_builder = scan_builder.with_row_selection_enabled(true);
    }
    if let Some(gate) = scan_memory_gate {
        scan_builder = scan_builder.with_scan_memory_gate(gate);
    }
    let table_scan = scan_builder.build().map_err(to_datafusion_error)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

fn get_column_names(
    schema: ArrowSchemaRef,
    projection: Option<&Vec<usize>>,
) -> Option<Vec<String>> {
    projection.map(|v| {
        v.iter()
            .map(|p| schema.field(*p).name().clone())
            .collect::<Vec<String>>()
    })
}
