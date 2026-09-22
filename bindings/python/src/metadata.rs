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
    arrow_schema: Option<Vec<(String, String)>>,
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
///  "bound_lower"/"bound_upper": json|None,  # column stats for
///                `bound_field` (full dotted name), when requested
///  "arrow_schema": {field: type_string}|None}  # the parquet FOOTER's
///                top-level arrow fields, ENGINE-rendered, when
///                `include_arrow_schema` (physical-layout inspection —
///                e.g. variant shredding shapes — without a Python
///                parquet reader)
/// ```
///
/// Scan planning provides path/size/partition/delete bindings; when
/// `bound_field` is given, a manifest walk (through the parsed-manifest
/// cache — cache-hot after the plan) merges that field's lower/upper
/// column-stat bounds by path.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, bound_field=None, include_arrow_schema=false, timeout_s=None))]
fn data_files(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    bound_field: Option<String>,
    include_arrow_schema: bool,
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
                            arrow_schema: None,
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
                    let mut bounds: HashMap<
                        String,
                        (Option<serde_json::Value>, Option<serde_json::Value>),
                    > = HashMap::new();
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
                if include_arrow_schema {
                    use iceberg::arrow::ArrowFileReader;
                    use iceberg::io::FileMetadata;
                    use parquet::arrow::arrow_reader::ArrowReaderMetadata;
                    for r in rows.iter_mut() {
                        let input = table
                            .file_io()
                            .new_input(&r.path)
                            .map_err(|e| PyValueError::new_err(format!("{}: {e}", r.path)))?;
                        let reader = input
                            .reader()
                            .await
                            .map_err(|e| PyValueError::new_err(format!("{}: {e}", r.path)))?;
                        let mut pr = ArrowFileReader::new(FileMetadata { size: r.size }, reader);
                        let meta = ArrowReaderMetadata::load_async(&mut pr, Default::default())
                            .await
                            .map_err(|e| {
                                PyValueError::new_err(format!("footer {}: {e}", r.path))
                            })?;
                        r.arrow_schema = Some(
                            meta.schema()
                                .fields()
                                .iter()
                                .map(|f| (f.name().clone(), format!("{}", f.data_type())))
                                .collect(),
                        );
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
        match &r.arrow_schema {
            None => d.set_item("arrow_schema", py.None())?,
            Some(fields) => {
                let sd = PyDict::new(py);
                for (name, ty) in fields {
                    sd.set_item(name, ty)?;
                }
                d.set_item("arrow_schema", sd)?;
            }
        }
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
                n.as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
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
    /// Every snapshot's `timestamp-ms`, ascending.
    snapshot_timestamps_ms: Vec<i64>,
    /// Each manifest's on-disk length (bytes), manifest-list order.
    manifest_lengths: Vec<i64>,
    /// Each manifest's content, `"data"` / `"deletes"`, manifest-list order.
    manifest_contents: Vec<&'static str>,
    /// Each manifest's partition spec id, manifest-list order.
    manifest_spec_ids: Vec<i32>,
    /// Each manifest's live entries (ADDED + EXISTING), manifest-list order.
    manifest_alive_files: Vec<u64>,
    /// The table's default partition spec id.
    default_spec_id: i32,
    /// The table's `commit.manifest.target-size-bytes`, when set.
    manifest_target_size_bytes: Option<i64>,
}

/// Manifest-LIST-level pressure counts for one table (no per-entry Avro
/// fetch): `{"data_files", "data_records", "delete_files", "delete_records",
/// "manifests", "snapshots"}`. Live counts = ADDED + EXISTING per manifest.
/// `delete_files` is the live DV/delete-file count (the dv-pressure signal);
/// counts come from the manifest list, never the snapshot summary (summary
/// totals are cosmetic carry-forwards on REPLACE).
///
/// The same call also returns the raw facts a caller needs to decide whether
/// snapshot expiry or manifest consolidation would DO anything, taken from
/// data this call already loads: `"snapshot_timestamps_ms"` (every snapshot's
/// timestamp, ascending — the count older than an expiry floor is the number
/// expiry can remove); per manifest in manifest-list order,
/// `"manifest_lengths"` (bytes), `"manifest_contents"` (`"data"` /
/// `"deletes"`), `"manifest_spec_ids"` and `"manifest_alive_files"`
/// (ADDED + EXISTING entries) — a manifest rewrite consolidates only ALIVE
/// `data` manifests on the `"default_spec_id"`, and only their lengths decide
/// whether fewer target-sized manifests come out; and
/// `"manifest_target_size_bytes"` (the table's `commit.manifest.target-size-bytes`,
/// `None` when unset; the caller applies the format default). Policy — the
/// floor, the retain count, the default target — stays with the caller.
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
                let mut snapshot_timestamps_ms: Vec<i64> =
                    meta.snapshots().map(|s| s.timestamp_ms()).collect();
                snapshot_timestamps_ms.sort_unstable();
                let manifest_target_size_bytes = meta
                    .properties()
                    .get("commit.manifest.target-size-bytes")
                    .and_then(|v| v.trim().parse::<i64>().ok());
                let default_spec_id = meta.default_partition_spec_id();
                let Some(current) = meta.current_snapshot() else {
                    return Ok::<_, PyErr>(StatsOut {
                        data_files: 0,
                        data_records: 0,
                        delete_files: 0,
                        delete_records: 0,
                        manifests: 0,
                        snapshots,
                        snapshot_timestamps_ms,
                        manifest_lengths: Vec::new(),
                        manifest_contents: Vec::new(),
                        manifest_spec_ids: Vec::new(),
                        manifest_alive_files: Vec::new(),
                        default_spec_id,
                        manifest_target_size_bytes,
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
                    snapshot_timestamps_ms,
                    manifest_lengths: Vec::with_capacity(mlist.entries().len()),
                    manifest_contents: Vec::with_capacity(mlist.entries().len()),
                    manifest_spec_ids: Vec::with_capacity(mlist.entries().len()),
                    manifest_alive_files: Vec::with_capacity(mlist.entries().len()),
                    default_spec_id,
                    manifest_target_size_bytes,
                };
                for mf in mlist.entries() {
                    let live_files = u64::from(mf.added_files_count.unwrap_or(0))
                        + u64::from(mf.existing_files_count.unwrap_or(0));
                    s.manifest_lengths.push(mf.manifest_length);
                    s.manifest_contents.push(match mf.content {
                        ManifestContentType::Data => "data",
                        ManifestContentType::Deletes => "deletes",
                    });
                    s.manifest_spec_ids.push(mf.partition_spec_id);
                    s.manifest_alive_files.push(live_files);
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
    d.set_item("snapshot_timestamps_ms", out.snapshot_timestamps_ms)?;
    d.set_item("manifest_lengths", out.manifest_lengths)?;
    d.set_item("manifest_contents", out.manifest_contents)?;
    d.set_item("manifest_spec_ids", out.manifest_spec_ids)?;
    d.set_item("manifest_alive_files", out.manifest_alive_files)?;
    d.set_item("default_spec_id", out.default_spec_id)?;
    d.set_item("manifest_target_size_bytes", out.manifest_target_size_bytes)?;
    Ok(d.into_any().unbind())
}

// ── cdc_probe ────────────────────────────────────────────────────────────────
// The merge worker's per-slice bronze probe, on the SAME parquet stack that
// writes bronze (the ingest lane's compaction writes parquet-rs files). It
// replaces a pyarrow (Arrow C++) read of the whole `_cdc` struct: pyarrow
// 25.0.0 mis-decoded 1-bit RLE_DICTIONARY indices on Neoverse V1 and aborted
// the worker from a C++ thread (2026-09-19/20 incident). Raw facts only — the
// caller keeps every gate/fail-open rule:
//   * `rows`   — footer row count.
//   * `ops`    — distinct non-null `_cdc.op` values, read through a ONE-LEAF
//                projection (the old read decoded all seven `_cdc` leaves);
//                stops early once both an R-class and a non-R value are seen.
//   * `stats`  — per requested leaf path: footer min/max aggregated over row
//                groups, typed `ts` (epoch micros) or `int`, with `present`,
//                `missing` (some row group lacks min/max) and `other` (some
//                row group's stats are neither of the wanted types) flags.
// Fail-open PER FILE: a file that cannot be probed returns `{"path","error"}`.

const CDC_OP_LEAF: &str = "_cdc.op";

#[derive(Default, Debug, Clone, PartialEq)]
struct LeafStats {
    present: bool,
    missing: bool,
    other: bool,
    /// "ts" | "int" | "" (no typed stats seen)
    kind: &'static str,
    min: Option<i64>,
    max: Option<i64>,
}

#[derive(Default, Debug)]
struct FileProbe {
    path: String,
    /// The footer parsed: `rows`/`stats` are valid even when `error` is set
    /// (an error then means only the `_cdc.op` read failed).
    footer_ok: bool,
    rows: i64,
    ops: Option<Vec<String>>,
    stats: Vec<(String, LeafStats)>,
    error: Option<String>,
}

fn is_r_class(op: &str) -> bool {
    op.eq_ignore_ascii_case("r")
}

/// Footer statistics of one leaf, aggregated over every row group.
fn leaf_stats(md: &parquet::file::metadata::ParquetMetaData, leaf_path: &str) -> LeafStats {
    use parquet::basic::{ConvertedType, LogicalType, TimeUnit, Type as PhysicalType};
    use parquet::file::statistics::Statistics;
    let mut out = LeafStats::default();
    let descr = md.file_metadata().schema_descr();
    let Some(j) = (0..descr.num_columns()).find(|&j| descr.column(j).path().string() == leaf_path)
    else {
        return out;
    };
    out.present = true;
    let col = descr.column(j);
    // Timestamp unit -> micros multiplier/divisor. None = not a timestamp.
    let ts_to_micros: Option<(i64, i64)> = match col.logical_type_ref() {
        Some(LogicalType::Timestamp(t)) => Some(match t.unit {
            TimeUnit::MILLIS => (1_000, 1),
            TimeUnit::MICROS => (1, 1),
            TimeUnit::NANOS => (1, 1_000),
        }),
        Some(_) => None,
        None => match col.converted_type() {
            ConvertedType::TIMESTAMP_MILLIS => Some((1_000, 1)),
            ConvertedType::TIMESTAMP_MICROS => Some((1, 1)),
            _ => None,
        },
    };
    // Plain signed integers only (an unsigned/decimal/date/time annotation is
    // not an exact int bound for the caller) — fail safe to `other`.
    let plain_int = matches!(
        col.physical_type(),
        PhysicalType::INT32 | PhysicalType::INT64
    ) && ts_to_micros.is_none()
        && match col.logical_type_ref() {
            None => matches!(
                col.converted_type(),
                ConvertedType::NONE
                    | ConvertedType::INT_8
                    | ConvertedType::INT_16
                    | ConvertedType::INT_32
                    | ConvertedType::INT_64
            ),
            Some(LogicalType::Integer(int)) => int.is_signed,
            Some(_) => false,
        };
    for rg in md.row_groups() {
        let raw: Option<(i64, i64)> = match rg.column(j).statistics() {
            Some(Statistics::Int64(v)) => match (v.min_opt(), v.max_opt()) {
                (Some(lo), Some(hi)) => Some((*lo, *hi)),
                _ => None,
            },
            Some(Statistics::Int32(v)) => match (v.min_opt(), v.max_opt()) {
                (Some(lo), Some(hi)) => Some((*lo as i64, *hi as i64)),
                _ => None,
            },
            Some(_) => {
                out.other = true;
                continue;
            }
            None => None,
        };
        let Some((lo, hi)) = raw else {
            out.missing = true;
            continue;
        };
        let (lo, hi, kind) = if let Some((mul, div)) = ts_to_micros {
            (
                lo.saturating_mul(mul) / div,
                hi.saturating_mul(mul) / div,
                "ts",
            )
        } else if plain_int {
            (lo, hi, "int")
        } else {
            out.other = true;
            continue;
        };
        out.kind = kind;
        out.min = Some(out.min.map_or(lo, |m| m.min(lo)));
        out.max = Some(out.max.map_or(hi, |m| m.max(hi)));
    }
    out
}

/// Distinct non-null strings of a (possibly nested) arrow array's single
/// string leaf — walks struct wrappers down to the leaf the projection kept.
fn collect_ops(
    array: &arrow::array::ArrayRef,
    seen: &mut std::collections::BTreeSet<String>,
) -> Result<(), String> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::DataType;
    match array.data_type() {
        DataType::Struct(_) => {
            let st = array.as_struct();
            if st.num_columns() != 1 {
                return Err(format!(
                    "projection kept {} children of `_cdc`, expected 1",
                    st.num_columns()
                ));
            }
            collect_ops(st.column(0), seen)
        }
        _ => {
            let utf8 = arrow::compute::cast(array, &DataType::Utf8)
                .map_err(|e| format!("cast `_cdc.op` to utf8: {e}"))?;
            let strs = utf8.as_string::<i32>();
            for i in 0..strs.len() {
                if strs.is_valid(i) {
                    let v = strs.value(i);
                    if !seen.contains(v) {
                        seen.insert(v.to_string());
                    }
                }
            }
            Ok(())
        }
    }
}

/// Probe one parquet file through any async reader (S3 via the table's
/// FileIO in production; a local file in tests).
async fn probe_parquet<R>(
    path: String,
    mut reader: R,
    stat_leaves: &[String],
    want_ops: bool,
) -> FileProbe
where
    R: parquet::arrow::async_reader::AsyncFileReader + Unpin + Send + 'static,
{
    use parquet::arrow::ProjectionMask;
    use parquet::arrow::arrow_reader::ArrowReaderMetadata;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
    let mut out = FileProbe {
        path,
        ..Default::default()
    };
    let meta = match ArrowReaderMetadata::load_async(&mut reader, Default::default()).await {
        Ok(m) => m,
        Err(e) => {
            out.error = Some(format!("footer: {e}"));
            return out;
        }
    };
    let md = meta.metadata().clone();
    out.footer_ok = true;
    out.rows = md.file_metadata().num_rows();
    out.stats = stat_leaves
        .iter()
        .map(|p| (p.clone(), leaf_stats(&md, p)))
        .collect();
    if !want_ops || out.rows == 0 {
        return out;
    }
    let descr = md.file_metadata().schema_descr();
    let Some(leaf) =
        (0..descr.num_columns()).find(|&j| descr.column(j).path().string() == CDC_OP_LEAF)
    else {
        out.error = Some(format!("no `{CDC_OP_LEAF}` leaf"));
        return out;
    };
    let mask = ProjectionMask::leaves(descr, [leaf]);
    let stream = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, meta)
        .with_projection(mask)
        .with_batch_size(65_536)
        .build();
    let mut stream = match stream {
        Ok(s) => s,
        Err(e) => {
            out.error = Some(format!("open `{CDC_OP_LEAF}`: {e}"));
            return out;
        }
    };
    let mut seen = std::collections::BTreeSet::new();
    while let Some(batch) = stream.next().await {
        let batch = match batch {
            Ok(b) => b,
            Err(e) => {
                out.error = Some(format!("read `{CDC_OP_LEAF}`: {e}"));
                return out;
            }
        };
        if batch.num_columns() != 1 {
            out.error = Some(format!("projection kept {} columns", batch.num_columns()));
            return out;
        }
        if let Err(e) = collect_ops(batch.column(0), &mut seen) {
            out.error = Some(e);
            return out;
        }
        // Both classes present: the caller's gate is already decided.
        if seen.iter().any(|o| is_r_class(o)) && seen.iter().any(|o| !is_r_class(o)) {
            break;
        }
    }
    out.ops = Some(seen.into_iter().collect());
    out
}

/// Per-slice bronze probe: `[{path, rows, ops, stats: {leaf: {...}}} | {path,
/// error}]` for each of `paths` (order preserved). ONE `loadTable` for the
/// FileIO; footers + the single `_cdc.op` leaf are the only bytes read.
/// `stat_leaves` = leaf paths whose footer min/max the caller wants
/// (`_cdc.ts`, a business timestamp, a single-column integer PK).
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, paths, stat_leaves=None, want_ops=true, timeout_s=None))]
fn cdc_probe(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    paths: Vec<String>,
    stat_leaves: Option<Vec<String>>,
    want_ops: bool,
    timeout_s: Option<u64>,
) -> PyResult<Py<PyAny>> {
    use iceberg::arrow::ArrowFileReader;
    use iceberg::io::FileMetadata;
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let stat_leaves = stat_leaves.unwrap_or_default();
    let probes: Vec<FileProbe> = py.detach(|| {
        runtime().block_on(metadata_deadline(
            timeout_s,
            "probing bronze files",
            async move {
                let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
                let file_io = table.file_io().clone();
                let leaves = std::sync::Arc::new(stat_leaves);
                let out: Vec<FileProbe> = futures::stream::iter(paths.into_iter().map(|path| {
                    let file_io = file_io.clone();
                    let leaves = leaves.clone();
                    async move {
                        let fail = |path: String, e: String| FileProbe {
                            path,
                            error: Some(e),
                            ..Default::default()
                        };
                        let input = match file_io.new_input(&path) {
                            Ok(i) => i,
                            Err(e) => return fail(path, format!("open: {e}")),
                        };
                        let size = match input.metadata().await {
                            Ok(m) => m.size,
                            Err(e) => return fail(path, format!("stat: {e}")),
                        };
                        let reader = match input.reader().await {
                            Ok(r) => r,
                            Err(e) => return fail(path, format!("reader: {e}")),
                        };
                        let pr = ArrowFileReader::new(FileMetadata { size }, reader);
                        probe_parquet(path, pr, &leaves, want_ops).await
                    }
                }))
                .buffered(8)
                .collect()
                .await;
                Ok::<_, PyErr>(out)
            },
        ))
    })?;
    let out = PyList::empty(py);
    for p in &probes {
        let d = PyDict::new(py);
        d.set_item("path", &p.path)?;
        if let Some(e) = &p.error {
            d.set_item("error", e)?;
            // rows may still be known (footer read, op read failed)
        }
        d.set_item("footer_ok", p.footer_ok)?;
        d.set_item("rows", p.rows)?;
        match &p.ops {
            None => d.set_item("ops", py.None())?,
            Some(v) => d.set_item("ops", v.clone())?,
        }
        let sd = PyDict::new(py);
        for (leaf, st) in &p.stats {
            let one = PyDict::new(py);
            one.set_item("present", st.present)?;
            one.set_item("missing", st.missing)?;
            one.set_item("other", st.other)?;
            one.set_item("kind", st.kind)?;
            one.set_item("min", st.min)?;
            one.set_item("max", st.max)?;
            sd.set_item(leaf, one)?;
        }
        d.set_item("stats", sd)?;
        out.append(d)?;
    }
    Ok(out.into_any().unbind())
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "metadata")?;
    this.add_function(wrap_pyfunction!(head, &this)?)?;
    this.add_function(wrap_pyfunction!(append_window, &this)?)?;
    this.add_function(wrap_pyfunction!(manifest_stats, &this)?)?;
    this.add_function(wrap_pyfunction!(location, &this)?)?;
    this.add_function(wrap_pyfunction!(data_files, &this)?)?;
    this.add_function(wrap_pyfunction!(cdc_probe, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}

#[cfg(test)]
mod cdc_probe_tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int64Array, StringArray, StructArray, TimestampMicrosecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    /// A bronze-shaped file: `id` int64, `uzi_updated` ts, `_cdc{op, ts, offset}`.
    /// Dictionary encoding is on by default, so a 2-value `op` column gets the
    /// 1-bit RLE_DICTIONARY indices that the incident files carry.
    fn write_file(path: &std::path::Path, ops: &[Option<&str>], row_group_rows: usize) {
        let n = ops.len();
        let id: ArrayRef = Arc::new(Int64Array::from_iter_values((0..n as i64).map(|i| 100 + i)));
        let upd: ArrayRef = Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..n as i64).map(|i| 1_700_000_000_000_000 + i * 1_000_000),
        ));
        let op: ArrayRef = Arc::new(StringArray::from(ops.to_vec()));
        let cts: ArrayRef = Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..n as i64).map(|i| 1_700_000_500_000_000 + i * 1_000_000),
        ));
        let off: ArrayRef = Arc::new(Int64Array::from_iter_values(0..n as i64));
        let cdc_fields = vec![
            Arc::new(Field::new("op", DataType::Utf8, true)),
            Arc::new(Field::new(
                "ts",
                DataType::Timestamp(ArrowTimeUnit::Microsecond, None),
                false,
            )),
            Arc::new(Field::new("offset", DataType::Int64, false)),
        ];
        let cdc: ArrayRef = Arc::new(StructArray::new(
            cdc_fields.clone().into(),
            vec![op, cts, off],
            None,
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "uzi_updated",
                DataType::Timestamp(ArrowTimeUnit::Microsecond, None),
                false,
            ),
            Field::new("_cdc", DataType::Struct(cdc_fields.into()), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![id, upd, cdc]).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group_rows))
            .build();
        let mut w = ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, Some(props))
            .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn probe(path: &std::path::Path, leaves: &[&str], want_ops: bool) -> FileProbe {
        let leaves: Vec<String> = leaves.iter().map(|s| s.to_string()).collect();
        let p = path.to_str().unwrap().to_string();
        runtime().block_on(async {
            let f = tokio::fs::File::open(&p).await.unwrap();
            probe_parquet(p.clone(), f, &leaves, want_ops).await
        })
    }

    #[test]
    fn two_value_dictionary_op_column_reads_exactly() {
        // 200k rows of alternating U/I = the incident shape (1-bit indices),
        // across several row groups.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cdc.parquet");
        let ops: Vec<Option<&str>> = (0..200_000)
            .map(|i| Some(if i % 3 == 0 { "U" } else { "I" }))
            .collect();
        write_file(&path, &ops, 65_536);
        let out = probe(&path, &["_cdc.ts", "uzi_updated", "id"], true);
        assert_eq!(out.error, None);
        assert!(out.footer_ok);
        assert_eq!(out.rows, 200_000);
        assert_eq!(out.ops, Some(vec!["I".to_string(), "U".to_string()]));
        let st: HashMap<_, _> = out.stats.into_iter().collect();
        let ts = &st["_cdc.ts"];
        assert!(ts.present && !ts.missing && !ts.other && ts.kind == "ts");
        assert_eq!(ts.min, Some(1_700_000_500_000_000));
        assert_eq!(ts.max, Some(1_700_000_500_000_000 + 199_999 * 1_000_000));
        let id = &st["id"];
        assert!(id.present && id.kind == "int");
        assert_eq!((id.min, id.max), (Some(100), Some(100 + 199_999)));
    }

    #[test]
    fn r_class_mix_stops_early_and_nulls_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mix.parquet");
        let mut ops: Vec<Option<&str>> = vec![None, Some("r"), Some("U")];
        ops.extend(std::iter::repeat(Some("D")).take(500_000));
        write_file(&path, &ops, 100_000);
        let out = probe(&path, &[], true);
        assert_eq!(out.error, None);
        let got = out.ops.unwrap();
        // the first batch already holds an R-class and a non-R value -> early
        // stop: "D" may or may not be present, the gate inputs must be.
        assert!(got.iter().any(|o| is_r_class(o)) && got.iter().any(|o| !is_r_class(o)));
        assert!(!got.iter().any(|o| o.is_empty()));
    }

    #[test]
    fn only_r_and_absent_leaf_and_no_ops_wanted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.parquet");
        write_file(&path, &vec![Some("R"); 10], 1024);
        let out = probe(&path, &["nope", "_cdc.op"], true);
        assert_eq!(out.ops, Some(vec!["R".to_string()]));
        let st: HashMap<_, _> = out.stats.into_iter().collect();
        assert!(!st["nope"].present);
        // a string leaf has stats of neither wanted type
        assert!(st["_cdc.op"].present && st["_cdc.op"].other && st["_cdc.op"].min.is_none());
        let out = probe(&path, &[], false);
        assert_eq!((out.rows, out.ops), (10, None));
    }
}
