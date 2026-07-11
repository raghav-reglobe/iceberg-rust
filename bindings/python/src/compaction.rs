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

use iceberg::maintenance::{PrefixMismatchMode, RemoveOrphanFilesAction};
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
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, target_file_size_bytes=None, min_input_files=None, delete_file_threshold=None, shred_variants=None))]
fn compact(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    target_file_size_bytes: Option<u64>,
    min_input_files: Option<usize>,
    delete_file_threshold: Option<usize>,
    shred_variants: Option<bool>,
) -> PyResult<()> {
    // FQN = catalog . namespace[.namespace...] . table
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
    // Preserve the input files' variant SHREDDING on rewrite (default false =
    // canonical output). See `Config::shred_variants`.
    if let Some(v) = shred_variants {
        cfg.shred_variants = v;
    }
    cfg.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Release the GIL (pyo3 0.28 `detach`): the rewrite is a long-running I/O +
    // compute call, so other Python threads (e.g. a queue-drain pool) can run.
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = RestCatalogBuilder::default()
                // S3 (and other object stores) via opendal — without a storage
                // factory the RestCatalog can't issue any data-file IO.
                .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
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

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "compaction")?;
    this.add_function(wrap_pyfunction!(compact, &this)?)?;
    this.add_function(wrap_pyfunction!(rewrite_manifests, &this)?)?;
    this.add_function(wrap_pyfunction!(remove_orphan_files, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
