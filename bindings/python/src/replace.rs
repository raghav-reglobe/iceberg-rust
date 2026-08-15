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

//! Atomic partition-scoped REPLACE doorway — the sync worker's bronze
//! "full-load chunk re-backfill" write: one `RowDelta` snapshot that
//! equality-deletes a key chunk's prior rows inside the full-load identity
//! partition and appends the replacements (see
//! `iceberg::atomic_replace::atomic_partition_replace` and the sync-engine
//! RFC). Input rows arrive as an Arrow IPC STREAM (`pyarrow.ipc.new_stream`
//! serialized bytes) — version-proof across pyarrow/pyo3 combinations.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use arrow::array::RecordBatch;
use iceberg::atomic_replace::{
    ReplaceInput, atomic_partition_append, atomic_partition_replace_key_range,
    atomic_partition_replace_key_set, atomic_partition_replace_prefix,
    atomic_partition_replace_with_props,
};
use iceberg::spec::Literal;
use iceberg::{NamespaceIdent, TableIdent};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

/// Resolve the input mode: `parquet_path` (streamed, bounded memory — the
/// giant-chunk mode) is mutually exclusive with a non-empty `batches_ipc`.
fn replace_input(batches_ipc: &[u8], parquet_path: Option<String>) -> PyResult<ReplaceInput> {
    match parquet_path {
        Some(path) => {
            if !batches_ipc.is_empty() {
                return Err(PyValueError::new_err(
                    "pass EITHER batches_ipc OR parquet_path, not both",
                ));
            }
            Ok(ReplaceInput::LocalParquet(path))
        }
        None => Ok(ReplaceInput::Batches(decode_ipc(batches_ipc)?)),
    }
}

fn split_fqn(fqn: &str) -> PyResult<(String, NamespaceIdent, String)> {
    let parts: Vec<&str> = fqn.split('.').collect();
    if parts.len() < 3 {
        return Err(PyValueError::new_err(format!(
            "table `{fqn}` must be `catalog.namespace.table`"
        )));
    }
    let namespace = NamespaceIdent::from_vec(
        parts[1..parts.len() - 1]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        parts[0].to_string(),
        namespace,
        parts[parts.len() - 1].to_string(),
    ))
}

fn decode_ipc(batches_ipc: &[u8]) -> PyResult<Vec<RecordBatch>> {
    let reader = arrow::ipc::reader::StreamReader::try_new(Cursor::new(batches_ipc), None)
        .map_err(|e| PyValueError::new_err(format!("reading Arrow IPC stream: {e}")))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(format!("reading Arrow IPC stream: {e}")))
}

/// Atomically replace one key-chunk of the `partition_column =
/// partition_value` identity partition: equality-delete the chunk's prior
/// rows (pk tuples = the input rows' keys) and append the input rows, as
/// ONE snapshot, with OCC retry-by-rerun on commit conflicts.
///
/// `batches_ipc` is an Arrow IPC stream (`pyarrow.ipc.new_stream` bytes);
/// columns are matched to the table BY NAME (order-free, no field-id
/// metadata needed), primitives are cast, and string columns targeting a
/// VARIANT column are parsed as JSON. `dry_run=True` validates + counts
/// without writing or committing. Returns
/// `{snapshot_id, rows_appended, delete_tuples_est, attempts}`.
#[pyfunction]
#[pyo3(signature = (catalogs, table, batches_ipc, pk_columns, partition_column="_is_backfill".to_string(), partition_value=true, max_retries=6, dry_run=false, parquet_path=None, summary_props=None))]
#[allow(clippy::too_many_arguments)]
fn bronze_replace(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    batches_ipc: Vec<u8>,
    pk_columns: Vec<String>,
    partition_column: String,
    partition_value: bool,
    max_retries: u32,
    dry_run: bool,
    parquet_path: Option<String>,
    summary_props: Option<HashMap<String, String>>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    if pk_columns.is_empty() {
        return Err(PyValueError::new_err("pk_columns must be non-empty"));
    }
    let input = replace_input(&batches_ipc, parquet_path)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let outcome = atomic_partition_replace_with_props(
                catalog.as_ref(),
                &ident,
                &partition_column,
                Literal::bool(partition_value),
                &pk_columns,
                input,
                summary_props.unwrap_or_default(),
                max_retries,
                dry_run,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("atomic replace: {e}")))?;
            Ok(HashMap::from([
                (
                    "snapshot_id".to_string(),
                    outcome
                        .snapshot_id
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "rows_appended".to_string(),
                    outcome.rows_appended.to_string(),
                ),
                (
                    "delete_tuples_est".to_string(),
                    outcome.delete_tuples.to_string(),
                ),
                ("attempts".to_string(), outcome.attempts.to_string()),
            ]))
        })
    })
}

/// Atomically replace one KEY-PREFIX chunk of the `partition_column =
/// partition_value` identity partition — the envelope-shaped (mongo) bronze
/// variant: prior rows matching `starts_with(key_column, key_prefix)` (and
/// `op_column IN op_values` when given) are deleted via scan + consolidated
/// V3 DELETION VECTORS (a prefix is inexpressible as an equality delete),
/// and the input rows appended, as ONE RowDelta snapshot at snapshot
/// isolation, with rescan-retry on commit conflicts.
///
/// `batches_ipc` as in `bronze_replace`. A ZERO-ROW input still deletes the
/// prefix's prior rows (duck `DELETE ... LIKE` parity — the docs vanished at
/// source). `dry_run=True` validates + counts the appends without scanning,
/// writing, or committing. Returns
/// `{snapshot_id, rows_appended, delete_tuples_est, attempts}` where
/// `delete_tuples_est` is the ACTUAL matched-row count.
#[pyfunction]
#[pyo3(signature = (catalogs, table, batches_ipc, key_prefix, key_column="_cdc.key".to_string(), op_column=Some("_cdc.op".to_string()), op_values=vec!["R".to_string(), "r".to_string()], partition_column="_is_backfill".to_string(), partition_value=true, max_retries=6, dry_run=false, parquet_path=None))]
#[allow(clippy::too_many_arguments)]
fn bronze_replace_prefix(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    batches_ipc: Vec<u8>,
    key_prefix: String,
    key_column: String,
    op_column: Option<String>,
    op_values: Vec<String>,
    partition_column: String,
    partition_value: bool,
    max_retries: u32,
    dry_run: bool,
    parquet_path: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    if key_prefix.is_empty() {
        return Err(PyValueError::new_err("key_prefix must be non-empty"));
    }
    let input = replace_input(&batches_ipc, parquet_path)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let outcome = atomic_partition_replace_prefix(
                catalog.as_ref(),
                &ident,
                &partition_column,
                partition_value,
                &key_column,
                &key_prefix,
                op_column.as_deref(),
                &op_values,
                input,
                max_retries,
                dry_run,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("atomic prefix replace: {e}")))?;
            Ok(HashMap::from([
                (
                    "snapshot_id".to_string(),
                    outcome
                        .snapshot_id
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "rows_appended".to_string(),
                    outcome.rows_appended.to_string(),
                ),
                (
                    "delete_tuples_est".to_string(),
                    outcome.delete_tuples.to_string(),
                ),
                ("attempts".to_string(), outcome.attempts.to_string()),
            ]))
        })
    })
}

/// Atomically replace one NUMERIC-KEY-RANGE chunk of the `partition_column
/// = partition_value` identity partition — the `bronze_replace_prefix`
/// sibling for document stores whose keys are Int64 stringified into the
/// key column (`_cdc.key = "28753650"`): hex-prefix chunks cannot express a
/// numeric range (lexicographic ≠ numeric across digit lengths), so prior
/// rows with `key_lo <= int(key) <= key_hi` (inclusive; unparseable keys
/// never match; `op_column IN op_values` when given) are deleted via scan +
/// consolidated V3 DELETION VECTORS and the input rows appended, as ONE
/// RowDelta snapshot at snapshot isolation with rescan-retry.
///
/// File-level planning bounds engage only when `key_lo`/`key_hi` have the
/// same digit count — chunk planners SHOULD emit same-length ranges. Same
/// zero-row-still-deletes / `dry_run` / return-shape contract as
/// `bronze_replace_prefix`.
#[pyfunction]
#[pyo3(signature = (catalogs, table, batches_ipc, key_lo, key_hi, key_column="_cdc.key".to_string(), op_column=Some("_cdc.op".to_string()), op_values=vec!["R".to_string(), "r".to_string()], partition_column="_is_backfill".to_string(), partition_value=true, max_retries=6, dry_run=false, parquet_path=None))]
#[allow(clippy::too_many_arguments)]
fn bronze_replace_range(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    batches_ipc: Vec<u8>,
    key_lo: i64,
    key_hi: i64,
    key_column: String,
    op_column: Option<String>,
    op_values: Vec<String>,
    partition_column: String,
    partition_value: bool,
    max_retries: u32,
    dry_run: bool,
    parquet_path: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    if key_lo > key_hi {
        return Err(PyValueError::new_err(format!(
            "key range is inverted: key_lo {key_lo} > key_hi {key_hi}"
        )));
    }
    let input = replace_input(&batches_ipc, parquet_path)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let outcome = atomic_partition_replace_key_range(
                catalog.as_ref(),
                &ident,
                &partition_column,
                partition_value,
                &key_column,
                key_lo,
                key_hi,
                op_column.as_deref(),
                &op_values,
                input,
                max_retries,
                dry_run,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("atomic range replace: {e}")))?;
            Ok(HashMap::from([
                (
                    "snapshot_id".to_string(),
                    outcome
                        .snapshot_id
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "rows_appended".to_string(),
                    outcome.rows_appended.to_string(),
                ),
                (
                    "delete_tuples_est".to_string(),
                    outcome.delete_tuples.to_string(),
                ),
                ("attempts".to_string(), outcome.attempts.to_string()),
            ]))
        })
    })
}

/// Atomically APPEND rows into the NULL slot of the `partition_column`
/// identity partition — the streaming-consumer (kafka → bronze) write: no
/// deletes, ONE RowDelta append snapshot at snapshot isolation carrying
/// `snapshot_properties` in its summary (the consumer's exactly-once offset
/// ledger, e.g. `pulse.kafka.offset.<topic>`). Input rows are conformed to
/// the table schema BY NAME (string columns targeting VARIANT are parsed as
/// JSON). A ZERO-ROW input is a NO-OP (no snapshot — the ledger only
/// advances with data). Returns `{snapshot_id, rows_appended, attempts}`.
#[pyfunction]
#[pyo3(signature = (catalogs, table, batches_ipc, snapshot_properties, partition_column="_is_backfill".to_string(), max_retries=6, parquet_path=None))]
fn bronze_append(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    batches_ipc: Vec<u8>,
    snapshot_properties: HashMap<String, String>,
    partition_column: String,
    max_retries: u32,
    parquet_path: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    let input = replace_input(&batches_ipc, parquet_path)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let outcome = atomic_partition_append(
                catalog.as_ref(),
                &ident,
                &partition_column,
                input,
                snapshot_properties,
                max_retries,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("atomic append: {e}")))?;
            Ok(HashMap::from([
                (
                    "snapshot_id".to_string(),
                    outcome
                        .snapshot_id
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "rows_appended".to_string(),
                    outcome.rows_appended.to_string(),
                ),
                ("attempts".to_string(), outcome.attempts.to_string()),
            ]))
        })
    })
}

/// Atomically UPSERT an exact `_cdc.key` SET — the repair-through-the-stream
/// doorway: prior rows with `key_column IN keys` (and `op_column IN
/// op_values`) inside the `partition_column = partition_value` identity
/// partition are deleted via scan + consolidated V3 DELETION VECTORS, and
/// the input rows appended, as ONE RowDelta snapshot at snapshot isolation
/// with rescan-retry. `summary_props` land on the committed snapshot — the
/// streaming lane's `pulse.kafka.offset.<topic>` ledger (offsets-in-
/// snapshot; the ledger must ride the SAME commit as the data it covers).
///
/// LAYERING (operator ruling 2026-08-12): the safety scope is BAKED —
/// `_is_backfill = true AND op IN ('R','r')` is a write INVARIANT ("a
/// repair commit physically cannot touch CDC rows"), enforced by
/// construction here rather than by caller convention. A generic
/// `replace_where(filters=…)` was considered and REJECTED.
///
/// Defensive no-ops (the lane never issues either): EMPTY `keys` or a
/// ZERO-ROW input returns the current snapshot with `attempts=0` and
/// commits NOTHING (unlike the prefix mode, where zero-row still deletes —
/// an upsert with nothing to upsert must not delete, and the caller's
/// ledger must never ride a phantom commit).
///
/// Returns `{snapshot_id, rows_appended, delete_tuples_est, attempts}`.
#[pyfunction]
#[pyo3(signature = (catalogs, table, batches_ipc, keys, summary_props, key_column="_cdc.key".to_string(), op_column=Some("_cdc.op".to_string()), op_values=vec!["R".to_string(), "r".to_string()], partition_column="_is_backfill".to_string(), partition_value=true, max_retries=6, parquet_path=None))]
#[allow(clippy::too_many_arguments)]
fn bronze_replace_keys(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    batches_ipc: Vec<u8>,
    keys: Vec<String>,
    summary_props: HashMap<String, String>,
    key_column: String,
    op_column: Option<String>,
    op_values: Vec<String>,
    partition_column: String,
    partition_value: bool,
    max_retries: u32,
    parquet_path: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    let input = replace_input(&batches_ipc, parquet_path)?;
    let zero_rows = match &input {
        ReplaceInput::Batches(b) => b.iter().map(|x| x.num_rows()).sum::<usize>() == 0,
        ReplaceInput::LocalParquet(_) => false,
    };
    let key_set: HashSet<String> = keys.into_iter().filter(|k| !k.is_empty()).collect();
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let outcome = if key_set.is_empty() || zero_rows {
                // Defensive no-op — read the current snapshot, commit nothing.
                let t = catalog
                    .load_table(&ident)
                    .await
                    .map_err(|e| PyValueError::new_err(format!("atomic key-set replace: {e}")))?;
                iceberg::atomic_replace::ReplaceOutcome {
                    snapshot_id: t.metadata().current_snapshot_id(),
                    rows_appended: 0,
                    delete_tuples: 0,
                    attempts: 0,
                }
            } else {
                atomic_partition_replace_key_set(
                    catalog.as_ref(),
                    &ident,
                    &partition_column,
                    partition_value,
                    &key_column,
                    key_set,
                    op_column.as_deref(),
                    &op_values,
                    input,
                    summary_props,
                    max_retries,
                    false,
                )
                .await
                .map_err(|e| PyValueError::new_err(format!("atomic key-set replace: {e}")))?
            };
            Ok(HashMap::from([
                (
                    "snapshot_id".to_string(),
                    outcome
                        .snapshot_id
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "rows_appended".to_string(),
                    outcome.rows_appended.to_string(),
                ),
                (
                    "delete_tuples_est".to_string(),
                    outcome.delete_tuples.to_string(),
                ),
                ("attempts".to_string(), outcome.attempts.to_string()),
            ]))
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "replace")?;
    this.add_function(wrap_pyfunction!(bronze_replace, &this)?)?;
    this.add_function(wrap_pyfunction!(bronze_replace_prefix, &this)?)?;
    this.add_function(wrap_pyfunction!(bronze_replace_range, &this)?)?;
    this.add_function(wrap_pyfunction!(bronze_replace_keys, &this)?)?;
    this.add_function(wrap_pyfunction!(bronze_append, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
