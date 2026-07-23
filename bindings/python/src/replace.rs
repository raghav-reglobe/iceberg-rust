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

use std::collections::HashMap;
use std::io::Cursor;

use arrow::array::RecordBatch;
use iceberg::atomic_replace::{atomic_partition_replace, atomic_partition_replace_prefix};
use iceberg::spec::Literal;
use iceberg::{NamespaceIdent, TableIdent};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

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
#[pyo3(signature = (catalogs, table, batches_ipc, pk_columns, partition_column="_is_backfill".to_string(), partition_value=true, max_retries=6, dry_run=false))]
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
    let batches = decode_ipc(&batches_ipc)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let outcome = atomic_partition_replace(
                catalog.as_ref(),
                &ident,
                &partition_column,
                Literal::bool(partition_value),
                &pk_columns,
                batches,
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
#[pyo3(signature = (catalogs, table, batches_ipc, key_prefix, key_column="_cdc.key".to_string(), op_column=Some("_cdc.op".to_string()), op_values=vec!["R".to_string(), "r".to_string()], partition_column="_is_backfill".to_string(), partition_value=true, max_retries=6, dry_run=false))]
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
    let batches = decode_ipc(&batches_ipc)?;
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
                batches,
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

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "replace")?;
    this.add_function(wrap_pyfunction!(bronze_replace, &this)?)?;
    this.add_function(wrap_pyfunction!(bronze_replace_prefix, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
