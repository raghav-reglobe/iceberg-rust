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

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::maintenance::{
    ExpireSnapshotsWithCleanupAction, PrefixMismatchMode, RemoveOrphanFilesAction,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableIdent};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_compaction::config::Config;
use iceberg_compaction::engine::compact_table;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

/// Compact one Iceberg table via `RewriteFiles` (Operation::Replace): rewrite
/// undersized / delete-bearing data files into compacted ones and reabsorb their
/// deletion vectors, in one snapshot.
///
/// `catalog_props` are standard Iceberg REST catalog properties (`uri`,
/// `warehouse`, `credential`, `oauth2-server-uri`, `scope`, ...); `fqn` is
/// `catalog.namespace.table`. The optional ints override the compaction config
/// (target file size + the candidate / delete-pressure thresholds). Blocks until
/// the rewrite commits; raises `ValueError` on failure.
/// Positive-integer MiB env knob (unset / unparsable / 0 = None).
fn env_mb(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, target_file_size_bytes=None, min_input_files=None, delete_file_threshold=None, shred_variants=None, sort_column=None, rewrite_all=None, timeout_s=None))]
fn compact(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    target_file_size_bytes: Option<u64>,
    min_input_files: Option<usize>,
    delete_file_threshold: Option<usize>,
    shred_variants: Option<bool>,
    sort_column: Option<String>,
    rewrite_all: Option<bool>,
    timeout_s: Option<u64>,
) -> PyResult<()> {
    // FQN = catalog . namespace[.namespace...] . table
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;

    let mut cfg = Config::default();
    // Cooperative deadline (the merge doorway's `timeout_s` twin): read/
    // sort/write phases abort once it elapses; the commit, once entered,
    // always runs to completion. A timed-out pass commits NOTHING.
    cfg.deadline = timeout_s.map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
    if let Some(v) = target_file_size_bytes {
        cfg.target_file_size_bytes = v;
    }
    if let Some(v) = min_input_files {
        cfg.min_input_files = v;
    }
    if let Some(v) = delete_file_threshold {
        cfg.delete_file_threshold = v;
    }
    // Preserve the input files' variant SHREDDING on rewrite (default false =
    // canonical output). See `Config::shred_variants`.
    if let Some(v) = shred_variants {
        cfg.shred_variants = v;
    }
    // Rewrite sort key override (default `_valid_from`). See `Config::sort_column`.
    if sort_column.is_some() {
        cfg.sort_column = sort_column;
    }
    // Plan every live data file (Spark `rewrite-all` parity). See `Config::rewrite_all`.
    if let Some(v) = rewrite_all {
        cfg.rewrite_all = v;
    }
    // Wide-row memory bounds, env-tunable per pod (see Config docs):
    // - ICEBERG_COMPACT_CHUNK_MB       — sort-chunk budget (default 1024).
    //   A group whose decoded arrow data exceeds it is rewritten as several
    //   sorted runs instead of buffered whole (the giant-TEXT OOM guard).
    // - ICEBERG_COMPACT_WRITE_BATCH_MB — per-slice writer batch (default 32).
    //   Also keeps every output array far below arrow's i32 string-offset
    //   range (the "Offset overflow" guard).
    if let Some(mb) = env_mb("ICEBERG_COMPACT_CHUNK_MB") {
        cfg.sort_chunk_bytes = mb.saturating_mul(1024 * 1024);
    }
    if let Some(mb) = env_mb("ICEBERG_COMPACT_WRITE_BATCH_MB") {
        cfg.write_batch_bytes = mb.saturating_mul(1024 * 1024);
    }
    cfg.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Release the GIL (pyo3 0.28 `detach`): the rewrite is a long-running I/O +
    // compute call, so other Python threads (e.g. a queue-drain pool) can run.
    py.detach(|| {
        runtime().block_on(async move {
            let builder = RestCatalogBuilder::default()
                // S3 (and other object stores) via opendal — without a storage
                // factory the RestCatalog can't issue any data-file IO.
                .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
                .with_object_bytes_cache(crate::runtime::global_object_cache().await);
            let builder = match crate::runtime::global_data_cache().await {
                Some(dc) => builder.with_data_bytes_cache(dc),
                None => builder,
            };
            let catalog = builder
                .load(catalog_name.clone(), catalog_props)
                .await
                .map_err(|e| {
                    PyValueError::new_err(format!("build catalog `{catalog_name}`: {e}"))
                })?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ident = TableIdent::new(namespace, table_name);
            compact_table(&catalog, &ident, &cfg)
                .await
                .map_err(|e| PyValueError::new_err(format!("compacting {fqn}: {e}")))
        })
    })
}

/// Consolidate one table's data manifests into fewer, target-sized manifests
/// without changing any data (`RewriteManifests`, Operation::Replace). Alive
/// entries keep their original lineage (snapshot id + sequence numbers);
/// already-DELETED entries are dropped, never resurrected.
///
/// Returns a dict with the rewrite counts (`manifests-replaced`,
/// `manifests-created`, `manifests-kept`, `entries-processed`,
/// `snapshot-id`) on commit, or `None` when there was nothing to rewrite
/// (fewer rewritable data manifests than `min_input_manifests`, or an empty
/// table) — so callers can treat that as a clean no-op. Raises `ValueError`
/// on any real failure.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, target_manifest_size_bytes=None, min_input_manifests=None))]
fn rewrite_manifests(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    target_manifest_size_bytes: Option<u64>,
    min_input_manifests: Option<usize>,
) -> PyResult<Option<HashMap<String, String>>> {
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;

    // Release the GIL: manifest reads/writes are long-running I/O.
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
                .with_object_bytes_cache(crate::runtime::global_object_cache().await)
                .load(catalog_name.clone(), catalog_props)
                .await
                .map_err(|e| {
                    PyValueError::new_err(format!("build catalog `{catalog_name}`: {e}"))
                })?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ident = TableIdent::new(namespace, table_name);
            let table = catalog
                .load_table(&ident)
                .await
                .map_err(|e| PyValueError::new_err(format!("load table {fqn}: {e}")))?;

            let tx = Transaction::new(&table);
            let mut action = tx.rewrite_manifests();
            if let Some(v) = target_manifest_size_bytes {
                action = action.target_manifest_size_bytes(v);
            }
            if let Some(v) = min_input_manifests {
                action = action.min_input_manifests(v);
            }
            let tx = action
                .apply(tx)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            match tx.commit(&catalog).await {
                Ok(new_table) => {
                    let mut out = HashMap::new();
                    if let Some(snapshot) = new_table.metadata().current_snapshot() {
                        out.insert(
                            "snapshot-id".to_string(),
                            snapshot.snapshot_id().to_string(),
                        );
                        for key in [
                            "manifests-replaced",
                            "manifests-created",
                            "manifests-kept",
                            "entries-processed",
                        ] {
                            if let Some(v) = snapshot.summary().additional_properties.get(key) {
                                out.insert(key.to_string(), v.clone());
                            }
                        }
                    }
                    Ok(Some(out))
                }
                // "Nothing to rewrite" / empty table → clean no-op for callers.
                Err(e) if e.kind() == ErrorKind::PreconditionFailed => Ok(None),
                Err(e) => Err(PyValueError::new_err(format!(
                    "rewriting manifests of {fqn}: {e}"
                ))),
            }
        })
    })
}

/// Identify (and, unless `dry_run`, delete) files under the table location
/// that no retained snapshot or table metadata references.
///
/// **`dry_run` defaults to `True`** — the report-only mode. Deleting a
/// referenced file is unrecoverable, so callers are expected to verify a
/// dry-run report (e.g. against an independent engine's `all_files` metadata)
/// before ever passing `dry_run=False`.
///
/// `older_than_ms` is the REQUIRED absolute cutoff (epoch ms): only files
/// last-modified before it are candidates — set it comfortably in the past
/// (e.g. now − 3 days) so an in-flight commit's freshly-written files are
/// never touched. Files without a modification time are always skipped.
///
/// Returns a dict: `orphan_files`, `deleted_files`, `failed_deletes`
/// (list of `[path, error]`), `listed_count`, `referenced_count`,
/// `skipped_recent`, `skipped_missing_mtime`.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, older_than_ms, dry_run=true, delete_concurrency=None))]
fn remove_orphan_files(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    older_than_ms: i64,
    dry_run: bool,
    delete_concurrency: Option<usize>,
) -> PyResult<Py<PyAny>> {
    if older_than_ms <= 0 {
        return Err(PyValueError::new_err(
            "older_than_ms must be a positive epoch-milliseconds cutoff",
        ));
    }
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;

    let result = py.detach(|| {
        runtime().block_on(async move {
            let catalog = RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
                .with_object_bytes_cache(crate::runtime::global_object_cache().await)
                .load(catalog_name.clone(), catalog_props)
                .await
                .map_err(|e| {
                    PyValueError::new_err(format!("build catalog `{catalog_name}`: {e}"))
                })?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ident = TableIdent::new(namespace, table_name);
            let table = catalog
                .load_table(&ident)
                .await
                .map_err(|e| PyValueError::new_err(format!("load table {fqn}: {e}")))?;

            let mut action = RemoveOrphanFilesAction::new(table)
                .older_than_ms(older_than_ms)
                .dry_run(dry_run)
                // Everything on this platform writes plain `s3://` URIs; a
                // mismatched scheme/authority is unexpected → fail loud.
                .prefix_mismatch_mode(PrefixMismatchMode::Error);
            if let Some(c) = delete_concurrency {
                action = action.delete_concurrency(c);
            }
            action
                .execute()
                .await
                .map_err(|e| PyValueError::new_err(format!("orphan scan of {fqn}: {e}")))
        })
    })?;

    let out = pyo3::types::PyDict::new(py);
    out.set_item("orphan_files", result.orphan_files)?;
    out.set_item("deleted_files", result.deleted_files)?;
    out.set_item("failed_deletes", result.failed_deletes)?;
    out.set_item("listed_count", result.listed_count)?;
    out.set_item("referenced_count", result.referenced_count)?;
    out.set_item("skipped_recent", result.skipped_recent)?;
    out.set_item("skipped_missing_mtime", result.skipped_missing_mtime)?;
    Ok(out.into_any().unbind())
}

/// Expire snapshots strictly older than `older_than_ms` (always keeping the
/// current snapshot, each branch's `retain_last` most recent snapshots, and
/// any snapshot referenced by a branch or tag), then delete the files only
/// the expired snapshots referenced.
///
/// Ordering is the crash-safety mechanism: the `remove-snapshots` metadata
/// commit lands FIRST (with rebase-per-attempt retry on commit conflicts —
/// every attempt refetches metadata and recomputes the selection); files are
/// deleted only after it succeeds, best-effort. A crash mid-cleanup leaves
/// plain orphans for `remove_orphan_files` — never a referenced-but-deleted
/// file. A file shared between an expired and a retained snapshot always
/// survives (exclusive-file diff against the post-commit metadata).
///
/// `dry_run=True` computes the selection + diff and returns the report
/// without committing or deleting; `cleanup=False` commits the metadata
/// change but skips the delete phase (candidates still reported).
///
/// Returns a dict — `removed_snapshots` (`{"count", "ids"}`), `removed_refs`,
/// `dry_run`, `deleted_files` / `failed_deletes` (counts), and per-category
/// candidate counts (`manifest_lists`, `manifests`, `data_files`,
/// `delete_files`, `stats_files`) — or `None` when there is nothing to
/// expire (same no-op convention as `rewrite_manifests`). Raises `ValueError`
/// on any real failure.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, older_than_ms, retain_last, cleanup=true, dry_run=false, delete_concurrency=None))]
fn expire_snapshots(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    older_than_ms: i64,
    retain_last: usize,
    cleanup: bool,
    dry_run: bool,
    delete_concurrency: Option<usize>,
) -> PyResult<Option<Py<PyAny>>> {
    if older_than_ms <= 0 {
        return Err(PyValueError::new_err(
            "older_than_ms must be a positive epoch-milliseconds cutoff",
        ));
    }
    if retain_last == 0 {
        return Err(PyValueError::new_err("retain_last must be at least 1"));
    }
    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;

    let result = py.detach(|| {
        runtime().block_on(async move {
            let catalog = RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
                .with_object_bytes_cache(crate::runtime::global_object_cache().await)
                .load(catalog_name.clone(), catalog_props)
                .await
                .map_err(|e| {
                    PyValueError::new_err(format!("build catalog `{catalog_name}`: {e}"))
                })?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ident = TableIdent::new(namespace, table_name);
            let table = catalog
                .load_table(&ident)
                .await
                .map_err(|e| PyValueError::new_err(format!("load table {fqn}: {e}")))?;

            let mut action = ExpireSnapshotsWithCleanupAction::new(table)
                .older_than_ms(older_than_ms)
                .retain_last(retain_last)
                .cleanup(cleanup)
                .dry_run(dry_run);
            if let Some(c) = delete_concurrency {
                action = action.delete_concurrency(c);
            }
            action
                .execute(&catalog)
                .await
                .map_err(|e| PyValueError::new_err(format!("expiring snapshots of {fqn}: {e}")))
        })
    })?;

    // Nothing to expire -> clean no-op for callers (rewrite_manifests parity).
    if result.is_noop() {
        return Ok(None);
    }

    let removed = pyo3::types::PyDict::new(py);
    removed.set_item("count", result.removed_snapshot_ids.len())?;
    removed.set_item("ids", result.removed_snapshot_ids)?;

    let out = pyo3::types::PyDict::new(py);
    out.set_item("removed_snapshots", removed)?;
    out.set_item("removed_refs", result.removed_ref_names)?;
    out.set_item("dry_run", result.dry_run)?;
    out.set_item("deleted_files", result.deleted_files.len())?;
    out.set_item("failed_deletes", result.failed_deletes.len())?;
    out.set_item("manifest_lists", result.candidate_manifest_lists.len())?;
    out.set_item("manifests", result.candidate_manifests.len())?;
    out.set_item("data_files", result.candidate_data_files.len())?;
    out.set_item("delete_files", result.candidate_delete_files.len())?;
    out.set_item("stats_files", result.candidate_stats_files.len())?;
    Ok(Some(out.into_any().unbind()))
}

/// READ-ONLY compaction-plan inspection — the `dry_run_inspect` doctrine for
/// the rewrite path: load the table, run the SAME planner `compact` uses
/// (same Config knobs), and return the plan as data. Zero writes, zero
/// commits. Built for balloon-group diagnosis: per group, the exact file
/// list (path/size/rows) plus every scan-bound delete file (path/size/type/
/// equality ids) so a "group N/M balloons" report maps to concrete files.
///
/// Returns:
/// ```text
/// {"groups": [{"partition": str, "total_size_bytes": int,
///              "delete_file_count": int,
///              "files": [{"path", "size_bytes", "rows", "n_deletes"}...],
///              "deletes": [{"path", "size_bytes", "type", "n_eq_ids",
///                           "bound_files"}...]}],   # deduped per group
///  "skipped_files": int, "total_input_files": int,
///  "total_input_bytes": int, "est_output_files": int,
///  "delete_applicability_len": int}
/// ```
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, target_file_size_bytes=None, min_input_files=None, delete_file_threshold=None, rewrite_all=None))]
fn plan_inspect(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    target_file_size_bytes: Option<u64>,
    min_input_files: Option<usize>,
    delete_file_threshold: Option<usize>,
    rewrite_all: Option<bool>,
) -> PyResult<Py<PyAny>> {
    use pyo3::types::{PyDict, PyList};

    let (catalog_name, ns, table_name) = split_fqn(&fqn)?;
    let mut cfg = Config::default();
    if let Some(v) = target_file_size_bytes {
        cfg.target_file_size_bytes = v;
    }
    if let Some(v) = min_input_files {
        cfg.min_input_files = v;
    }
    if let Some(v) = delete_file_threshold {
        cfg.delete_file_threshold = v;
    }
    if let Some(v) = rewrite_all {
        cfg.rewrite_all = v;
    }
    cfg.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let plan = py.detach(|| {
        runtime().block_on(async move {
            let builder = RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
                .with_object_bytes_cache(crate::runtime::global_object_cache().await);
            let builder = match crate::runtime::global_data_cache().await {
                Some(dc) => builder.with_data_bytes_cache(dc),
                None => builder,
            };
            let catalog = builder
                .load(catalog_name.clone(), catalog_props)
                .await
                .map_err(|e| {
                    PyValueError::new_err(format!("build catalog `{catalog_name}`: {e}"))
                })?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ident = TableIdent::new(namespace, table_name);
            let table = catalog
                .load_table(&ident)
                .await
                .map_err(|e| PyValueError::new_err(format!("load table: {e}")))?;
            iceberg_compaction::engine::plan_table(&table, &cfg)
                .await
                .map_err(|e| PyValueError::new_err(format!("plan: {e}")))
        })
    })?;

    let out = PyDict::new(py);
    let groups = PyList::empty(py);
    for g in &plan.groups {
        let gd = PyDict::new(py);
        gd.set_item("partition", &g.partition_key)?;
        gd.set_item("total_size_bytes", g.total_size_bytes)?;
        gd.set_item("delete_file_count", g.delete_file_count)?;
        let files = PyList::empty(py);
        let mut deletes: HashMap<String, (u64, String, usize, usize)> = HashMap::new();
        for t in &g.tasks {
            let fd = PyDict::new(py);
            fd.set_item("path", t.data_file_path())?;
            fd.set_item("size_bytes", t.file_size_in_bytes)?;
            fd.set_item("rows", t.record_count)?;
            fd.set_item("n_deletes", t.deletes.len())?;
            files.append(fd)?;
            for d in &t.deletes {
                let e = deletes.entry(d.file_path.clone()).or_insert((
                    d.file_size_in_bytes,
                    format!("{:?}", d.file_type),
                    d.equality_ids.as_ref().map(|v| v.len()).unwrap_or(0),
                    0,
                ));
                e.3 += 1;
            }
        }
        gd.set_item("files", files)?;
        let dl = PyList::empty(py);
        let mut sorted: Vec<_> = deletes.into_iter().collect();
        sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        for (p, (sz, ty, neq, bound)) in sorted {
            let dd = PyDict::new(py);
            dd.set_item("path", p)?;
            dd.set_item("size_bytes", sz)?;
            dd.set_item("type", ty)?;
            dd.set_item("n_eq_ids", neq)?;
            dd.set_item("bound_files", bound)?;
            dl.append(dd)?;
        }
        gd.set_item("deletes", dl)?;
        groups.append(gd)?;
    }
    out.set_item("groups", groups)?;
    out.set_item("skipped_files", plan.skipped_files)?;
    out.set_item("total_input_files", plan.total_input_files)?;
    out.set_item("total_input_bytes", plan.total_input_bytes)?;
    out.set_item("est_output_files", plan.est_output_files)?;
    out.set_item("delete_applicability_len", plan.delete_applicability.len())?;
    Ok(out.into_any().unbind())
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "compaction")?;
    this.add_function(wrap_pyfunction!(compact, &this)?)?;
    this.add_function(wrap_pyfunction!(plan_inspect, &this)?)?;
    this.add_function(wrap_pyfunction!(rewrite_manifests, &this)?)?;
    this.add_function(wrap_pyfunction!(remove_orphan_files, &this)?)?;
    this.add_function(wrap_pyfunction!(expire_snapshots, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
