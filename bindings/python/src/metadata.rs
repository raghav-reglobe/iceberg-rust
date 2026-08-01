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
#[pyo3(signature = (catalog_props, fqn))]
fn head(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
) -> PyResult<Option<Py<PyAny>>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: Option<HeadOut> = py.detach(|| {
        runtime().block_on(async move {
            let table = load_table_only(catalog_props, catalog_name, ns, table_name).await?;
            Ok::<_, PyErr>(table.metadata().current_snapshot().map(|s| HeadOut {
                snapshot_id: s.snapshot_id(),
                sequence_number: s.sequence_number(),
                timestamp_ms: s.timestamp_ms(),
            }))
        })
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
#[pyo3(signature = (catalog_props, fqn, cursor_snap=None, cursor_seq=None))]
fn append_window(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    cursor_snap: Option<i64>,
    cursor_seq: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: WindowOut = py.detach(|| {
        runtime().block_on(async move {
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
        })
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
#[pyo3(signature = (catalog_props, fqn))]
fn manifest_stats(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
) -> PyResult<Py<PyAny>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let out: StatsOut = py.detach(|| {
        runtime().block_on(async move {
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
        })
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
    m.add_submodule(&this)?;
    Ok(())
}
