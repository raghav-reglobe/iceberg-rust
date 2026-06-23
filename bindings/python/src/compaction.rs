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

use iceberg::{CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_compaction::config::Config;
use iceberg_compaction::engine::compact_table;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

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
#[pyo3(signature = (catalog_props, fqn, target_file_size_bytes=None, min_input_files=None, delete_file_threshold=None))]
fn compact(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    target_file_size_bytes: Option<u64>,
    min_input_files: Option<usize>,
    delete_file_threshold: Option<usize>,
) -> PyResult<()> {
    // FQN = catalog . namespace[.namespace...] . table
    let parts: Vec<&str> = fqn.split('.').collect();
    if parts.len() < 3 {
        return Err(PyValueError::new_err(format!(
            "fqn must be catalog.namespace.table, got `{fqn}`"
        )));
    }
    let catalog_name = parts[0].to_string();
    let table_name = parts[parts.len() - 1].to_string();
    let ns: Vec<String> = parts[1..parts.len() - 1]
        .iter()
        .map(|s| s.to_string())
        .collect();

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

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "compaction")?;
    this.add_function(wrap_pyfunction!(compact, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
