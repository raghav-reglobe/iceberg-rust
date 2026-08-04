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

use iceberg::spec::{FormatVersion, Schema, SortOrder, UnboundPartitionSpec};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

/// Create a new Iceberg table via the REST catalog.
///
/// `catalog_props` are standard Iceberg REST catalog properties (`uri`,
/// `warehouse`, `credential`, `oauth2-server-uri`, `scope`, ...); `fqn` is
/// `catalog.namespace.table`.
///
/// `schema_json` is a standard Iceberg schema JSON document
/// (`{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"x",...}]}`).
/// VARIANT columns (a field with `"type":"variant"`) are supported, but a
/// variant schema requires `format_version=3`.
///
/// `partition_spec_json` / `sort_order_json`, when given, are standard Iceberg
/// unbound-partition-spec / sort-order JSON documents.
///
/// Blocks until the catalog commits the new table; raises `ValueError` on
/// failure (including table-already-exists).
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, schema_json, format_version=2, properties=None, partition_spec_json=None, sort_order_json=None, location=None))]
#[allow(clippy::too_many_arguments)]
fn create_table(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    schema_json: String,
    format_version: u8,
    properties: Option<HashMap<String, String>>,
    partition_spec_json: Option<String>,
    sort_order_json: Option<String>,
    location: Option<String>,
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

    let format_version = match format_version {
        1 => FormatVersion::V1,
        2 => FormatVersion::V2,
        3 => FormatVersion::V3,
        v => {
            return Err(PyValueError::new_err(format!(
                "format_version must be 1, 2, or 3, got {v}"
            )));
        }
    };

    let schema: Schema = serde_json::from_str(&schema_json)
        .map_err(|e| PyValueError::new_err(format!("parsing schema_json: {e}")))?;

    let partition_spec: Option<UnboundPartitionSpec> = match partition_spec_json {
        Some(s) => Some(
            serde_json::from_str(&s)
                .map_err(|e| PyValueError::new_err(format!("parsing partition_spec_json: {e}")))?,
        ),
        None => None,
    };
    let sort_order: Option<SortOrder> = match sort_order_json {
        Some(s) => Some(
            serde_json::from_str(&s)
                .map_err(|e| PyValueError::new_err(format!("parsing sort_order_json: {e}")))?,
        ),
        None => None,
    };

    // All fields are `pub`; build the struct directly to avoid the typed-builder
    // type-state dance for the conditional optional fields.
    let creation = TableCreation {
        name: table_name,
        location,
        schema,
        partition_spec,
        sort_order,
        properties: properties.unwrap_or_default(),
        format_version,
    };

    // Release the GIL (pyo3 0.28 `detach`): the create round-trips to the REST
    // catalog + object store, so other Python threads can run meanwhile.
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
            catalog
                .create_table(&namespace, creation)
                .await
                .map_err(|e| PyValueError::new_err(format!("creating {fqn}: {e}")))?;
            Ok(())
        })
    })
}

/// Shared: split `catalog.namespace[.namespace...].table` and build the REST
/// catalog (storage factory + caches wired, same as `create_table`).
async fn build_catalog(
    catalog_name: String,
    catalog_props: HashMap<String, String>,
) -> PyResult<Arc<dyn Catalog>> {
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
        .map_err(|e| PyValueError::new_err(format!("build catalog `{catalog_name}`: {e}")))?;
    Ok(Arc::new(catalog))
}

fn split_table_fqn(fqn: &str) -> PyResult<(String, Vec<String>, String)> {
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

/// Create a namespace if it does not already exist (idempotent).
/// `namespace_fqn` is `catalog.namespace[.namespace...]`.
#[pyfunction]
fn create_namespace(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    namespace_fqn: String,
) -> PyResult<()> {
    let parts: Vec<&str> = namespace_fqn.split('.').collect();
    if parts.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "namespace_fqn must be catalog.namespace, got `{namespace_fqn}`"
        )));
    }
    let catalog_name = parts[0].to_string();
    let ns: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let exists = catalog
                .namespace_exists(&namespace)
                .await
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if !exists {
                catalog
                    .create_namespace(&namespace, HashMap::new())
                    .await
                    .map_err(|e| PyValueError::new_err(format!("creating {namespace_fqn}: {e}")))?;
            }
            Ok(())
        })
    })
}

/// Whether `catalog.namespace.table` exists.
#[pyfunction]
fn table_exists(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
) -> PyResult<bool> {
    let (catalog_name, ns, table) = split_table_fqn(&fqn)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            catalog
                .table_exists(&iceberg::TableIdent::new(namespace, table))
                .await
                .map_err(|e| PyValueError::new_err(format!("probing {fqn}: {e}")))
        })
    })
}

/// Set (and optionally remove) table properties in one commit.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, updates, removals=None))]
fn set_properties(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    updates: HashMap<String, String>,
    removals: Option<Vec<String>>,
) -> PyResult<()> {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    let (catalog_name, ns, table) = split_table_fqn(&fqn)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let t = catalog
                .load_table(&iceberg::TableIdent::new(namespace, table))
                .await
                .map_err(|e| PyValueError::new_err(format!("loading {fqn}: {e}")))?;
            let tx = Transaction::new(&t);
            let mut action = tx.update_table_properties();
            for (k, v) in updates {
                action = action.set(k, v);
            }
            for k in removals.unwrap_or_default() {
                action = action.remove(k);
            }
            action
                .apply(tx)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .commit(catalog.as_ref())
                .await
                .map_err(|e| PyValueError::new_err(format!("committing {fqn}: {e}")))?;
            Ok(())
        })
    })
}

/// Expire snapshots by age with a retain-last floor (Java `RemoveSnapshots`
/// parity — see `ExpireSnapshotsAction`). A no-op resolution commits nothing.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, older_than_ms, retain_last=None))]
fn expire_snapshots(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    older_than_ms: i64,
    retain_last: Option<usize>,
) -> PyResult<()> {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    let (catalog_name, ns, table) = split_table_fqn(&fqn)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let t = catalog
                .load_table(&iceberg::TableIdent::new(namespace, table))
                .await
                .map_err(|e| PyValueError::new_err(format!("loading {fqn}: {e}")))?;
            let tx = Transaction::new(&t);
            let mut action = tx.expire_snapshots().expire_older_than_ms(older_than_ms);
            if let Some(n) = retain_last {
                action = action.retain_last(n);
            }
            action
                .apply(tx)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .commit(catalog.as_ref())
                .await
                .map_err(|e| PyValueError::new_err(format!("expiring {fqn}: {e}")))?;
            Ok(())
        })
    })
}

/// The table's CURRENT schema as an Iceberg schema-JSON document. The
/// pyiceberg-free source for schema projection (VARIANT columns come back
/// as real `"variant"` — no UnknownType detour).
#[pyfunction]
fn table_schema_json(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
) -> PyResult<String> {
    let (catalog_name, ns, table) = split_table_fqn(&fqn)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let t = catalog
                .load_table(&iceberg::TableIdent::new(namespace, table))
                .await
                .map_err(|e| PyValueError::new_err(format!("loading {fqn}: {e}")))?;
            serde_json::to_string(t.metadata().current_schema().as_ref())
                .map_err(|e| PyValueError::new_err(format!("serializing schema: {e}")))
        })
    })
}

/// Drop a table (metadata-only — never a purge; orphaned files are the
/// maintenance sweep's job).
#[pyfunction]
fn drop_table(py: Python<'_>, catalog_props: HashMap<String, String>, fqn: String) -> PyResult<()> {
    let (catalog_name, ns, table) = split_table_fqn(&fqn)?;
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            catalog
                .drop_table(&iceberg::TableIdent::new(namespace, table))
                .await
                .map_err(|e| PyValueError::new_err(format!("dropping {fqn}: {e}")))?;
            Ok(())
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "catalog")?;
    this.add_function(wrap_pyfunction!(create_table, &this)?)?;
    this.add_function(wrap_pyfunction!(create_namespace, &this)?)?;
    this.add_function(wrap_pyfunction!(table_exists, &this)?)?;
    this.add_function(wrap_pyfunction!(set_properties, &this)?)?;
    this.add_function(wrap_pyfunction!(expire_snapshots, &this)?)?;
    this.add_function(wrap_pyfunction!(table_schema_json, &this)?)?;
    this.add_function(wrap_pyfunction!(drop_table, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
