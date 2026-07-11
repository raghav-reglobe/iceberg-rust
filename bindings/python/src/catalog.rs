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
            catalog
                .create_table(&namespace, creation)
                .await
                .map_err(|e| PyValueError::new_err(format!("creating {fqn}: {e}")))?;
            Ok(())
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "catalog")?;
    this.add_function(wrap_pyfunction!(create_table, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
