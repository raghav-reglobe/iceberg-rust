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

//! Planner metadata doorways — read-only table-metadata/manifest reads for an
//! external (Python) merge planner.
//!
//! Three functions, all single-`loadTable` (no catalog mount, no listing):
//! - `head`            — current-snapshot pointer (id, sequence, timestamp).
//! - `append_window`   — the delta-planner read: live DATA files with
//!   data sequence_number > the caller's cursor, from the CURRENT snapshot's
//!   manifests (survives snapshot expiry). The caller keeps ALL cursor/plan
//!   semantics; this returns raw facts.
//! - `manifest_stats`  — manifest-LIST-level counts (no entry fetch): live
//!   data/delete file + row counts, manifest + snapshot counts.
//! - `location`        — the table's base location + current metadata-file
//!   location (the pointer an EXTERNAL metadata walker starts from).
//!
//! `append_window` intentionally reproduces the reference reader's exact
//! semantics (its consumer runs a byte-exact parity gate against a pure-Python
//! implementation): DATA-content manifests only; manifest-level prune on
//! `manifest.sequence_number <= after_seq`; deleted entries discarded; an
//! entry's missing data sequence number falls back UNCONDITIONALLY to its
//! manifest's sequence number (some writers leave ADDED-entry seqs null —
//! load-time inheritance covers ADDED, the explicit fallback covers the rest);
//! result sorted by (sequence_number, path) — a stable total order within a
//! sequence, load-bearing for the caller's file-level checkpoint.

use std::collections::HashMap;
use std::future::Future;

use futures::stream::{StreamExt, TryStreamExt};
use iceberg::spec::ManifestContentType;
use iceberg::table::Table;
use iceberg::{NamespaceIdent, TableIdent};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::runtime::runtime;

fn split_fqn(fqn: &str) -> PyResult<(String, Vec<String>, String)> {
    let parts: Vec<&str> = fqn.split('.').collect();
    if parts.len() < 3 {
        return Err(PyValueError::new_err(format!(
            "fqn must be catalog.namespace.table, got `{fqn}`"
        )));
    }
    Ok((
        parts[0].to_string(),
        parts[1..parts.len() - 1]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        parts[parts.len() - 1].to_string(),
    ))
}

/// Bound a read-only metadata doorway by a hard deadline — the compact
/// `timeout_s` twin. These calls are catalog/manifest READS (no commit to
/// protect), so a blanket cancel is safe; the REST client's own
/// connect/read timeouts bound individual requests, and this bounds the
/// whole call (many-manifest walks included) so a stalled catalog can never
/// hang the caller for minutes.
async fn metadata_deadline<T>(
    timeout_s: Option<u64>,
    what: &str,
    fut: impl Future<Output = PyResult<T>>,
) -> PyResult<T> {
    match timeout_s {
        None => fut.await,
        Some(s) => tokio::time::timeout(std::time::Duration::from_secs(s), fut)
            .await
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "metadata timeout exceeded while {what} (read-only — nothing was written)"
                ))
            })?,
    }
}

async fn load_table_only(
    catalog_props: HashMap<String, String>,
    catalog_name: String,
    ns: Vec<String>,
    table_name: String,
) -> PyResult<Table> {
    // Memoized catalog (merge.rs): one OAuth handle per (name, props) per
    // process + 401 re-auth + the foyer object cache — a checkpoint-cadence
    // caller must not re-auth per call.
    let catalog = crate::merge::get_or_build_catalog(&catalog_name, catalog_props).await?;
    let namespace =
        NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ident = TableIdent::new(namespace, table_name);
    catalog
        .load_table(&ident)
        .await
        .map_err(|e| PyValueError::new_err(format!("load table: {e}")))
}

struct HeadOut {
    snapshot_id: i64,
    sequence_number: i64,
    timestamp_ms: i64,
}

/// Current-snapshot pointer for one table: `{"snapshot_id", "sequence_number",
/// "timestamp_ms"}`, or `None` when the table has no current snapshot. One
/// metadata.json read — no manifest IO.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, timeout_s=None))]
fn head(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    timeout_s: Option<u64>,
) -> PyResult<Option<Py<PyAny>>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: Option<HeadOut> = py.detach(|| {
        runtime().block_on(metadata_deadline(
            timeout_s,
            "reading the table head",
            async move {
                let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
                Ok::<_, PyErr>(table.metadata().current_snapshot().map(|s| HeadOut {
                    snapshot_id: s.snapshot_id(),
                    sequence_number: s.sequence_number(),
                    timestamp_ms: s.timestamp_ms(),
                }))
            },
        ))
    })?;
    match out {
        None => Ok(None),
        Some(h) => {
            let d = PyDict::new(py);
            d.set_item("snapshot_id", h.snapshot_id)?;
            d.set_item("sequence_number", h.sequence_number)?;
            d.set_item("timestamp_ms", h.timestamp_ms)?;
            Ok(Some(d.into_any().unbind()))
        }
    }
}

/// Table pointers for an external metadata walker: `{"location",
/// "metadata_location"}`. `metadata_location` is `None` when the catalog
/// response carries no metadata file pointer (never the case for a REST
/// catalog table). One loadTable — no manifest IO.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, timeout_s=None))]
fn location(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    timeout_s: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: (String, Option<String>) = py.detach(|| {
        runtime().block_on(metadata_deadline(
            timeout_s,
            "reading table pointers",
            async move {
                let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
                Ok::<_, PyErr>((
                    table.metadata().location().to_string(),
                    table.metadata_location().map(|s| s.to_string()),
                ))
            },
        ))
    })?;
    let d = PyDict::new(py);
    d.set_item("location", out.0)?;
    d.set_item("metadata_location", out.1)?;
    Ok(d.into_any().unbind())
}

struct FileRow {
    path: String,
    rows: u64,
    seq: i64,
    bytes: u64,
}

struct WindowOut {
    head_snapshot_id: Option<i64>,
    head_seq: Option<i64>,
    resolved_cursor_seq: Option<i64>,
    from_seq: i64,
    walked: bool,
    files: Vec<FileRow>,
}

/// The delta-planner read. Returns a dict:
///
/// ```text
/// {"head_snapshot_id": int|None, "head_seq": int|None,
///  "resolved_cursor_seq": int|None,   # cursor_seq verbatim, else the
///                                     # snapshot log's seq for cursor_snap
///  "from_seq": int,                   # the after-seq the walk used
///  "walked": bool,                    # False when caught-up short-circuit
///  "files": [(path, record_count, sequence_number, file_size_in_bytes), ...]}
/// ```
///
/// `files` lists live DATA files of the CURRENT snapshot whose data
/// sequence_number > from_seq, sorted by (sequence_number, path). When the
/// resolved cursor is at/past the head sequence the manifest walk is skipped
/// (`walked=False`, `files=[]`) — that walk is provably empty, not a
/// different semantic. Cursor/plan-kind decisions stay with the caller.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, cursor_snap=None, cursor_seq=None, timeout_s=None))]
fn append_window(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    cursor_snap: Option<i64>,
    cursor_seq: Option<i64>,
    timeout_s: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: WindowOut = py.detach(|| {
        runtime().block_on(metadata_deadline(
            timeout_s,
            "walking the append window",
            async move {
                let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
                let meta = table.metadata_ref();
                let Some(current) = meta.current_snapshot() else {
                    return Ok::<_, PyErr>(WindowOut {
                        head_snapshot_id: None,
                        head_seq: None,
                        resolved_cursor_seq: None,
                        from_seq: 0,
                        walked: false,
                        files: vec![],
                    });
                };
                let head_id = current.snapshot_id();
                let head_seq = current.sequence_number();
                let resolved = match cursor_seq {
                    Some(s) => Some(s),
                    None => cursor_snap
                        .and_then(|cs| meta.snapshot_by_id(cs))
                        .map(|s| s.sequence_number()),
                };
                if let Some(r) = resolved {
                    if r >= head_seq {
                        return Ok(WindowOut {
                            head_snapshot_id: Some(head_id),
                            head_seq: Some(head_seq),
                            resolved_cursor_seq: resolved,
                            from_seq: r,
                            walked: false,
                            files: vec![],
                        });
                    }
                }
                let after_seq = resolved.unwrap_or(0);
                let mlist = table
                    .manifest_list_reader(current)
                    .load()
                    .await
                    .map_err(|e| PyValueError::new_err(format!("manifest list: {e}")))?;
                // Manifest-level prune: every entry's data seq <= its manifest's
                // seq (spec invariant), so an at-or-below-cursor manifest holds
                // only consumed files.
                let candidates: Vec<_> = mlist
                    .entries()
                    .iter()
                    .filter(|mf| {
                        mf.content == ManifestContentType::Data && mf.sequence_number > after_seq
                    })
                    .collect();
                // Concurrent manifest loads through the table's PARSED-manifest
                // cache (shared with the scan path; manifest files are immutable
                // so path-keyed reuse is always valid — re-walks after a new
                // commit re-fetch only the NEW manifests). The walk is I/O-bound:
                // sequential GETs were the dominant cost on trickle-manifest
                // tables. Completion order is irrelevant — the final (seq, path)
                // sort normalizes it.
                let concurrency = std::env::var("ICEBERG_METADATA_MANIFEST_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|v| *v > 0)
                    .unwrap_or(16);
                let loaded: Vec<_> = futures::stream::iter(candidates.into_iter().map(|mf| {
                    let table = &table;
                    async move {
                        table
                            .load_manifest_cached(mf)
                            .await
                            .map(|m| (mf, m))
                            .map_err(|e| {
                                PyValueError::new_err(format!("manifest {}: {e}", mf.manifest_path))
                            })
                    }
                }))
                .buffer_unordered(concurrency)
                .try_collect()
                .await?;
                let mut files: Vec<FileRow> = Vec::new();
                for (mf, manifest) in &loaded {
                    for entry in manifest.entries() {
                        if !entry.is_alive() {
                            continue;
                        }
                        let seq = entry.sequence_number().unwrap_or(mf.sequence_number);
                        if seq > after_seq {
                            let df = entry.data_file();
                            files.push(FileRow {
                                path: df.file_path().to_string(),
                                rows: df.record_count(),
                                seq,
                                bytes: df.file_size_in_bytes(),
                            });
                        }
                    }
                }
                files.sort_by(|a, b| (a.seq, a.path.as_str()).cmp(&(b.seq, b.path.as_str())));
                Ok(WindowOut {
                    head_snapshot_id: Some(head_id),
                    head_seq: Some(head_seq),
                    resolved_cursor_seq: resolved,
                    from_seq: after_seq,
                    walked: true,
                    files,
                })
            },
        ))
    })?;
    let d = PyDict::new(py);
    d.set_item("head_snapshot_id", out.head_snapshot_id)?;
    d.set_item("head_seq", out.head_seq)?;
    d.set_item("resolved_cursor_seq", out.resolved_cursor_seq)?;
    d.set_item("from_seq", out.from_seq)?;
    d.set_item("walked", out.walked)?;
    let rows = PyList::empty(py);
    for f in &out.files {
        rows.append(PyTuple::new(py, [
            f.path.clone().into_pyobject(py)?.into_any(),
            f.rows.into_pyobject(py)?.into_any(),
            f.seq.into_pyobject(py)?.into_any(),
            f.bytes.into_pyobject(py)?.into_any(),
        ])?)?;
    }
    d.set_item("files", rows)?;
    Ok(d.into_any().unbind())
}

struct DataFileRow {
    path: String,
    size: u64,
    records: Option<u64>,
    partition: Vec<Option<serde_json::Value>>,
    n_deletes: usize,
    bound_lower: Option<serde_json::Value>,
    bound_upper: Option<serde_json::Value>,
}

fn literal_to_json(lit: &iceberg::spec::Literal) -> Option<serde_json::Value> {
    use iceberg::spec::{Literal, PrimitiveLiteral};
    let Literal::Primitive(p) = lit else {
        return None; // non-primitive partition/bound values are not expected
    };
    Some(match p {
        PrimitiveLiteral::Boolean(v) => serde_json::Value::Bool(*v),
        PrimitiveLiteral::Int(v) => serde_json::Value::from(*v),
        PrimitiveLiteral::Long(v) => serde_json::Value::from(*v),
        PrimitiveLiteral::Float(v) => serde_json::Value::from(v.into_inner()),
        PrimitiveLiteral::Double(v) => serde_json::Value::from(v.into_inner()),
        PrimitiveLiteral::String(v) => serde_json::Value::from(v.clone()),
        _ => return None,
    })
}

/// Live DATA-file inventory of the CURRENT snapshot — the facts a
/// seeding/registration harness needs, WITHOUT a Python Iceberg client.
/// Returns a list of dicts, sorted by path:
///
/// ```text
/// {"path": str, "file_size_in_bytes": int, "record_count": int|None,
///  "partition": [json values in DEFAULT-spec field order]  # the same
///                shape catalog.add_data_files accepts,
///  "n_deletes": int,          # delete files bound by scan planning
///  "bound_lower"/"bound_upper": json|None}  # column stats for
///                `bound_field` (full dotted name), when requested
/// ```
///
/// Scan planning provides path/size/partition/delete bindings; when
/// `bound_field` is given, a manifest walk (through the parsed-manifest
/// cache — cache-hot after the plan) merges that field's lower/upper
/// column-stat bounds by path.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, bound_field=None, timeout_s=None))]
fn data_files(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    bound_field: Option<String>,
    timeout_s: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let rows: Vec<DataFileRow> = py.detach(|| {
        runtime().block_on(metadata_deadline(
            timeout_s,
            "listing data files",
            async move {
                let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
                let meta = table.metadata_ref();
                if meta.current_snapshot().is_none() {
                    return Ok::<_, PyErr>(vec![]);
                }
                let n_spec_fields = meta.default_partition_spec().fields().len();

                let scan = table
                    .scan()
                    .build()
                    .map_err(|e| PyValueError::new_err(format!("building scan: {e}")))?;
                let tasks: Vec<_> = scan
                    .plan_files()
                    .await
                    .map_err(|e| PyValueError::new_err(format!("planning: {e}")))?
                    .try_collect()
                    .await
                    .map_err(|e| PyValueError::new_err(format!("planning: {e}")))?;

                let mut rows: Vec<DataFileRow> = tasks
                    .iter()
                    .map(|t: &iceberg::scan::FileScanTask| {
                        let partition = match t.partition.as_ref() {
                            Some(p) => (0..n_spec_fields)
                                .map(|i| p.iter().nth(i).flatten().and_then(literal_to_json))
                                .collect(),
                            None => vec![None; n_spec_fields],
                        };
                        DataFileRow {
                            path: t.data_file_path.clone(),
                            size: t.file_size_in_bytes,
                            records: t.record_count,
                            partition,
                            n_deletes: t.deletes.len(),
                            bound_lower: None,
                            bound_upper: None,
                        }
                    })
                    .collect();

                if let Some(field_name) = bound_field {
                    let field = meta
                        .current_schema()
                        .field_by_name(&field_name)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!("bound_field `{field_name}` not found"))
                        })?;
                    let field_id = field.id;
                    let current = meta.current_snapshot().unwrap();
                    let mlist = table
                        .manifest_list_reader(current)
                        .load()
                        .await
                        .map_err(|e| PyValueError::new_err(format!("manifest list: {e}")))?;
                    let mut bounds: HashMap<String, (Option<serde_json::Value>, Option<serde_json::Value>)> =
                        HashMap::new();
                    for mf in mlist.entries() {
                        if mf.content != ManifestContentType::Data {
                            continue;
                        }
                        let manifest = table.load_manifest_cached(mf).await.map_err(|e| {
                            PyValueError::new_err(format!("manifest {}: {e}", mf.manifest_path))
                        })?;
                        for entry in manifest.entries() {
                            if !entry.is_alive() {
                                continue;
                            }
                            let df = entry.data_file();
                            let lo = df
                                .lower_bounds()
                                .get(&field_id)
                                .and_then(|d| literal_to_json(&d.clone().into()));
                            let hi = df
                                .upper_bounds()
                                .get(&field_id)
                                .and_then(|d| literal_to_json(&d.clone().into()));
                            bounds.insert(df.file_path().to_string(), (lo, hi));
                        }
                    }
                    for r in rows.iter_mut() {
                        if let Some((lo, hi)) = bounds.get(&r.path) {
                            r.bound_lower = lo.clone();
                            r.bound_upper = hi.clone();
                        }
                    }
                }
                rows.sort_by(|a, b| a.path.cmp(&b.path));
                Ok(rows)
            },
        ))
    })?;
    let out = PyList::empty(py);
    for r in &rows {
        let d = PyDict::new(py);
        d.set_item("path", &r.path)?;
        d.set_item("file_size_in_bytes", r.size)?;
        d.set_item("record_count", r.records)?;
        let part = PyList::empty(py);
        for v in &r.partition {
            match v {
                None => part.append(py.None())?,
                Some(j) => part.append(json_value_to_py(py, j)?)?,
            }
        }
        d.set_item("partition", part)?;
        d.set_item("n_deletes", r.n_deletes)?;
        d.set_item("bound_lower", opt_json_to_py(py, &r.bound_lower)?)?;
        d.set_item("bound_upper", opt_json_to_py(py, &r.bound_upper)?)?;
        out.append(d)?;
    }
    Ok(out.into_any().unbind())
}

fn json_value_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        serde_json::Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any().unbind()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        other => other.to_string().into_pyobject(py)?.into_any().unbind(),
    })
}

fn opt_json_to_py(py: Python<'_>, v: &Option<serde_json::Value>) -> PyResult<Py<PyAny>> {
    match v {
        None => Ok(py.None()),
        Some(j) => json_value_to_py(py, j),
    }
}

struct StatsOut {
    data_files: u64,
    data_records: u64,
    delete_files: u64,
    delete_records: u64,
    manifests: usize,
    snapshots: usize,
}

/// Manifest-LIST-level pressure counts for one table (no per-entry Avro
/// fetch): `{"data_files", "data_records", "delete_files", "delete_records",
/// "manifests", "snapshots"}`. Live counts = ADDED + EXISTING per manifest.
/// `delete_files` is the live DV/delete-file count (the dv-pressure signal);
/// counts come from the manifest list, never the snapshot summary (summary
/// totals are cosmetic carry-forwards on REPLACE).
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, timeout_s=None))]
fn manifest_stats(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    timeout_s: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: StatsOut = py.detach(|| {
        runtime().block_on(metadata_deadline(
            timeout_s,
            "reading manifest stats",
            async move {
                let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
                let meta = table.metadata_ref();
                let snapshots = meta.snapshots().len();
                let Some(current) = meta.current_snapshot() else {
                    return Ok::<_, PyErr>(StatsOut {
                        data_files: 0,
                        data_records: 0,
                        delete_files: 0,
                        delete_records: 0,
                        manifests: 0,
                        snapshots,
                    });
                };
                let mlist = table
                    .manifest_list_reader(current)
                    .load()
                    .await
                    .map_err(|e| PyValueError::new_err(format!("manifest list: {e}")))?;
                let mut s = StatsOut {
                    data_files: 0,
                    data_records: 0,
                    delete_files: 0,
                    delete_records: 0,
                    manifests: mlist.entries().len(),
                    snapshots,
                };
                for mf in mlist.entries() {
                    let live_files = u64::from(mf.added_files_count.unwrap_or(0))
                        + u64::from(mf.existing_files_count.unwrap_or(0));
                    let live_rows =
                        mf.added_rows_count.unwrap_or(0) + mf.existing_rows_count.unwrap_or(0);
                    match mf.content {
                        ManifestContentType::Data => {
                            s.data_files += live_files;
                            s.data_records += live_rows;
                        }
                        ManifestContentType::Deletes => {
                            s.delete_files += live_files;
                            s.delete_records += live_rows;
                        }
                    }
                }
                Ok(s)
            },
        ))
    })?;
    let d = PyDict::new(py);
    d.set_item("data_files", out.data_files)?;
    d.set_item("data_records", out.data_records)?;
    d.set_item("delete_files", out.delete_files)?;
    d.set_item("delete_records", out.delete_records)?;
    d.set_item("manifests", out.manifests)?;
    d.set_item("snapshots", out.snapshots)?;
    Ok(d.into_any().unbind())
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "metadata")?;
    this.add_function(wrap_pyfunction!(head, &this)?)?;
    this.add_function(wrap_pyfunction!(append_window, &this)?)?;
    this.add_function(wrap_pyfunction!(manifest_stats, &this)?)?;
    this.add_function(wrap_pyfunction!(location, &this)?)?;
    this.add_function(wrap_pyfunction!(data_files, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
