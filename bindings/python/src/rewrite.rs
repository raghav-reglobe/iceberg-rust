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

//! Copy-on-write rewrites: MySQL `TIME` string spellings into one signed `long`
//! of microseconds (see `iceberg::time_rewrite`), and caller-supplied rows for
//! named data files (see `iceberg::replace_file_rows`).
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::RecordBatch;
use iceberg::replace_file_rows::{
    BoundFloor, FileRowSource, FileRows, ReplaceFileRowsOutcome, RowBatches, RowSource,
    replace_file_rows_from_sources_with_floors as replace_rows_with_floors,
};
use iceberg::time_rewrite::{mysql_time_text_to_us as parse_time, rewrite_time_columns as rewrite};
use iceberg::{ErrorKind, NamespaceIdent, TableIdent};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

/// Parse one MySQL `TIME` spelling into signed microseconds since midnight
/// (`None` for an empty string; raises on an unrecognised spelling) — the
/// Python-side twin for verification scripts.
#[pyfunction]
fn mysql_time_text_to_us(s: &str) -> PyResult<Option<i64>> {
    parse_time(s)
        .map(|o| o.map(|(us, _)| us))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Fill each `(source, target)` pair of `table` (`catalog.namespace.table`) —
/// `source` a string column of spellings, `target` an existing `long` column —
/// with signed microseconds, copy-on-write, in ONE replace snapshot; the source
/// is kept. Add the targets first (`schema.update_schema(add_columns=...)`); swap
/// the names at the end (`drop_columns=[source], rename_columns=[(target, source)]`). With
/// `dry_run` nothing is written or committed — the counts say what a run would
/// meet, and an unknown spelling raises. `base_snapshot` pins the plan to the
/// current snapshot; `carry_summary_prefix` copies the base snapshot's summary
/// keys with that prefix onto the new snapshot. Returns a dict of strings.
#[pyfunction]
#[pyo3(signature = (catalogs, table, columns, base_snapshot=None, dry_run=false, snapshot_properties=None, carry_summary_prefix=None))]
fn rewrite_time_columns(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    columns: Vec<(String, String)>,
    base_snapshot: Option<i64>,
    dry_run: bool,
    snapshot_properties: Option<HashMap<String, String>>,
    carry_summary_prefix: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    if columns.is_empty() {
        return Err(PyValueError::new_err("columns must be non-empty"));
    }
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let out = rewrite(
                catalog.as_ref(),
                &ident,
                &columns,
                base_snapshot,
                dry_run,
                snapshot_properties.unwrap_or_default(),
                carry_summary_prefix.as_deref(),
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("time rewrite {table}: {e}")))?;
            Ok(HashMap::from([
                (
                    "snapshot_before".to_string(),
                    out.snapshot_before.to_string(),
                ),
                (
                    "snapshot_after".to_string(),
                    out.snapshot_after
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                ("rows".to_string(), out.rows.to_string()),
                (
                    "records_before".to_string(),
                    out.records_before
                        .map(|n| n.to_string())
                        .unwrap_or_default(),
                ),
                (
                    "partition_fields".to_string(),
                    out.partition_fields.to_string(),
                ),
                ("files_deleted".to_string(), out.files_deleted.to_string()),
                ("files_added".to_string(), out.files_added.to_string()),
                (
                    "delete_files_removed".to_string(),
                    out.delete_files_removed.to_string(),
                ),
                ("digits".to_string(), out.counts.digits.to_string()),
                ("dated".to_string(), out.counts.dated.to_string()),
                ("clock".to_string(), out.counts.clock.to_string()),
                ("days".to_string(), out.counts.days.to_string()),
                ("nulls".to_string(), out.counts.nulls.to_string()),
                ("dry_run".to_string(), dry_run.to_string()),
            ]))
        })
    })
}

/// `appended_lower_bounds_above`: a list of `{"fields": [dotted names],
/// "above": int}`, each a proof-5 floor (see `replace_file_rows`).
fn bound_floors(groups: Option<Vec<Bound<'_, PyDict>>>) -> PyResult<Vec<BoundFloor>> {
    let mut out = Vec::new();
    for g in groups.unwrap_or_default() {
        let fields: Vec<String> = g
            .get_item("fields")?
            .ok_or_else(|| PyValueError::new_err("a bound floor needs `fields`"))?
            .extract()?;
        let above: i64 = g
            .get_item("above")?
            .ok_or_else(|| PyValueError::new_err("a bound floor needs `above`"))?
            .extract()?;
        out.push(BoundFloor { fields, above });
    }
    Ok(out)
}

fn outcome_dict(out: &ReplaceFileRowsOutcome, dry_run: bool) -> HashMap<String, String> {
    HashMap::from([
        (
            "snapshot_before".to_string(),
            out.snapshot_before.to_string(),
        ),
        (
            "snapshot_after".to_string(),
            out.snapshot_after
                .map(|s| s.to_string())
                .unwrap_or_default(),
        ),
        ("files_replaced".to_string(), out.files_replaced.to_string()),
        ("files_added".to_string(), out.files_added.to_string()),
        ("rows_live".to_string(), out.rows_live.to_string()),
        ("rows_written".to_string(), out.rows_written.to_string()),
        (
            "delete_files_removed".to_string(),
            out.delete_files_removed.to_string(),
        ),
        ("dry_run".to_string(), dry_run.to_string()),
        (
            "rebase_rechecks".to_string(),
            serde_json::Value::Array(
                out.rebase_rechecks
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "attempt": r.attempt,
                            "expected": r.expected,
                            "current": r.current,
                            "snapshots": r.snapshots,
                        })
                    })
                    .collect(),
            )
            .to_string(),
        ),
    ])
}

/// Replacement rows kept in an Arrow IPC stream file, read one batch at a
/// time on every open.
#[derive(Debug)]
struct IpcStreamFile {
    data_file_path: String,
    path: PathBuf,
}

impl RowSource for IpcStreamFile {
    fn open(&self) -> iceberg::Result<RowBatches<'_>> {
        let about = |what: &str| {
            iceberg::Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "{what} the Arrow IPC stream `{}` for `{}`",
                    self.path.display(),
                    self.data_file_path
                ),
            )
        };
        let file = File::open(&self.path).map_err(|e| about("opening").with_source(e))?;
        let reader = arrow::ipc::reader::StreamReader::try_new(BufReader::new(file), None)
            .map_err(|e| about("opening").with_source(e))?;
        let read_error = about("reading");
        Ok(Box::new(reader.map(move |b| {
            b.map_err(|e| {
                iceberg::Error::new(read_error.kind(), read_error.message()).with_source(e)
            })
        })))
    }
}

fn decode_ipc(path: &str, ipc: &[u8]) -> PyResult<Vec<RecordBatch>> {
    let reader =
        arrow::ipc::reader::StreamReader::try_new(Cursor::new(ipc), None).map_err(|e| {
            PyValueError::new_err(format!("reading Arrow IPC stream for `{path}`: {e}"))
        })?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(format!("reading Arrow IPC stream for `{path}`: {e}")))
}

/// Replace the rows of named data files of `table` (`catalog.namespace.table`),
/// copy-on-write, in ONE replace snapshot. `files` is a list of `(data_file_path,
/// ipc_bytes)`: each path as the table metadata records it, each payload an Arrow
/// IPC stream (`pyarrow.ipc.new_stream` bytes) holding every row the file should
/// hold afterwards — as many rows as the file has live rows at `base_snapshot`,
/// columns matched to the table schema by name (a `variant` column may arrive
/// as JSON text). Delete files bound to a named file are removed with it. The
/// call refuses when the table is no longer at `base_snapshot`, when a row count
/// differs, or when a row's partition value differs from its file's; the commit
/// lands on top of snapshots committed after `base_snapshot` only when every
/// one is a plain append that leaves the replaced files alone (else it refuses
/// as a stale snapshot), at most 5 attempts. With `dry_run` everything is
/// validated and nothing is written. `carry_summary_prefix` copies the summary
/// keys with that prefix from the snapshot the commit lands on. Returns a dict
/// of strings; `rebase_rechecks` is a JSON list of `{attempt, expected,
/// current, snapshots}`, one per attempt that found the table moved.
/// `appended_lower_bounds_above` (proof 5): a list of `{"fields": [dotted
/// names], "above": int}`; every data file appended after `base_snapshot`
/// must have the minimum of its PRESENT lower bounds among each floor's
/// fields strictly above `above` (microseconds for timestamp fields); none
/// present refuses, as the stale-snapshot refusal with "proof 5" in it.
#[pyfunction]
#[pyo3(signature = (catalogs, table, base_snapshot, files, dry_run=false, snapshot_properties=None, carry_summary_prefix=None, appended_lower_bounds_above=None))]
#[allow(clippy::too_many_arguments)]
fn replace_file_rows(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    base_snapshot: i64,
    files: Vec<(String, Vec<u8>)>,
    dry_run: bool,
    snapshot_properties: Option<HashMap<String, String>>,
    carry_summary_prefix: Option<String>,
    appended_lower_bounds_above: Option<Vec<Bound<'_, PyDict>>>,
) -> PyResult<HashMap<String, String>> {
    let floors = bound_floors(appended_lower_bounds_above)?;
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    let files = files
        .iter()
        .map(|(path, ipc)| {
            Ok(FileRows {
                data_file_path: path.clone(),
                batches: decode_ipc(path, ipc)?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let files = files
                .into_iter()
                .map(|f| FileRowSource {
                    data_file_path: f.data_file_path,
                    source: Arc::new(f.batches) as Arc<dyn RowSource>,
                })
                .collect();
            let out = replace_rows_with_floors(
                catalog.as_ref(),
                &ident,
                base_snapshot,
                files,
                dry_run,
                snapshot_properties.unwrap_or_default(),
                carry_summary_prefix.as_deref(),
                &floors,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("replace file rows {table}: {e}")))?;
            Ok(outcome_dict(&out, dry_run))
        })
    })
}

/// `replace_file_rows` with each file's rows read from a local Arrow IPC
/// stream file instead of bytes: `files` is a list of `(data_file_path,
/// ipc_stream_path)`. Every file is read once to validate and once more to
/// write, one batch at a time, so memory stays at one batch plus the writer's
/// buffers; the write refuses when the second read yields a different row
/// count. Gates, return dict and `dry_run` are those of `replace_file_rows`.
#[pyfunction]
#[pyo3(signature = (catalogs, table, base_snapshot, files, dry_run=false, snapshot_properties=None, carry_summary_prefix=None, appended_lower_bounds_above=None))]
#[allow(clippy::too_many_arguments)]
fn replace_file_rows_from_paths(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    base_snapshot: i64,
    files: Vec<(String, String)>,
    dry_run: bool,
    snapshot_properties: Option<HashMap<String, String>>,
    carry_summary_prefix: Option<String>,
    appended_lower_bounds_above: Option<Vec<Bound<'_, PyDict>>>,
) -> PyResult<HashMap<String, String>> {
    let floors = bound_floors(appended_lower_bounds_above)?;
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    let files = files
        .into_iter()
        .map(|(data_file_path, path)| FileRowSource {
            source: Arc::new(IpcStreamFile {
                data_file_path: data_file_path.clone(),
                path: PathBuf::from(path),
            }) as Arc<dyn RowSource>,
            data_file_path,
        })
        .collect();
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let out = replace_rows_with_floors(
                catalog.as_ref(),
                &ident,
                base_snapshot,
                files,
                dry_run,
                snapshot_properties.unwrap_or_default(),
                carry_summary_prefix.as_deref(),
                &floors,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("replace file rows {table}: {e}")))?;
            Ok(outcome_dict(&out, dry_run))
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "rewrite")?;
    this.add_function(wrap_pyfunction!(rewrite_time_columns, &this)?)?;
    this.add_function(wrap_pyfunction!(mysql_time_text_to_us, &this)?)?;
    this.add_function(wrap_pyfunction!(replace_file_rows, &this)?)?;
    this.add_function(wrap_pyfunction!(replace_file_rows_from_paths, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
