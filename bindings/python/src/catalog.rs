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

/// The table's schema as an Iceberg schema-JSON document. The pyiceberg-free
/// source for schema projection (VARIANT columns come back as real
/// `"variant"` — no UnknownType detour).
///
/// Without `snapshot_id`: the table's CURRENT schema.
///
/// With `snapshot_id`: the schema THAT SNAPSHOT was written with — the schema
/// a time-travel read of the snapshot binds, which is not the current schema
/// when the table's schema was changed after the snapshot was committed. A
/// snapshot that records no schema id resolves to the current schema (the
/// document's own `schema-id` says which schema came back). A snapshot id
/// that is not in the table's metadata (unknown, or expired) is an error,
/// never a silent fall-back to the current schema.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, snapshot_id=None))]
fn table_schema_json(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    snapshot_id: Option<i64>,
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
            let schema = schema_of(t.metadata(), snapshot_id)
                .map_err(|e| PyValueError::new_err(format!("{fqn}: {e}")))?;
            serde_json::to_string(schema.as_ref())
                .map_err(|e| PyValueError::new_err(format!("serializing schema: {e}")))
        })
    })
}

/// The current schema, or the schema `snapshot_id` was written with (see
/// [`table_schema_json`]).
fn schema_of(
    metadata: &iceberg::spec::TableMetadata,
    snapshot_id: Option<i64>,
) -> Result<iceberg::spec::SchemaRef, String> {
    let Some(snapshot_id) = snapshot_id else {
        return Ok(metadata.current_schema().clone());
    };
    let snapshot = metadata.snapshot_by_id(snapshot_id).ok_or_else(|| {
        format!("snapshot {snapshot_id} is not in the table's metadata (unknown or expired)")
    })?;
    snapshot
        .schema(metadata)
        .map_err(|e| format!("schema of snapshot {snapshot_id}: {e}"))
}

/// One snapshot's record as JSON — the current snapshot, or `snapshot_id`
/// when given: `{"snapshot-id", "sequence-number", "timestamp-ms",
/// "schema-id", "summary": {"operation", ...the summary's properties}}`.
/// `None` when the table has no snapshot; a `snapshot_id` the metadata does
/// not hold (unknown or expired) is an error. One loadTable, no manifest IO —
/// the reader for a tool that gates on `total-records` or on a producer's
/// summary keys without a Python metadata library.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, snapshot_id=None))]
fn snapshot_summary_json(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    snapshot_id: Option<i64>,
) -> PyResult<Option<String>> {
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
            let metadata = t.metadata();
            let snapshot = match snapshot_id {
                None => metadata.current_snapshot().cloned(),
                Some(id) => Some(metadata.snapshot_by_id(id).cloned().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "{fqn}: snapshot {id} is not in the table's metadata (unknown or expired)"
                    ))
                })?),
            };
            let Some(s) = snapshot else {
                return Ok(None);
            };
            let summary = serde_json::to_value(s.summary())
                .map_err(|e| PyValueError::new_err(format!("serializing summary: {e}")))?;
            let doc = serde_json::json!({
                "snapshot-id": s.snapshot_id(),
                "sequence-number": s.sequence_number(),
                "timestamp-ms": s.timestamp_ms(),
                "schema-id": s.schema_id(),
                "summary": summary,
            });
            serde_json::to_string(&doc)
                .map(Some)
                .map_err(|e| PyValueError::new_err(format!("serializing snapshot: {e}")))
        })
    })
}

/// The table's DEFAULT partition spec as JSON (`{"spec-id", "fields": [{"source-id",
/// "field-id", "name", "transform"}]}`) — the document `create_table(partition_spec_json=)`
/// accepts, so a twin of a table can be created with the same partitioning without a
/// Python metadata library.
#[pyfunction]
fn table_partition_spec_json(
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
            serde_json::to_string(t.metadata().default_partition_spec().as_ref())
                .map_err(|e| PyValueError::new_err(format!("serializing partition spec: {e}")))
        })
    })
}

/// Move the table's main branch back to `snapshot_id` (an id the metadata
/// still holds): one metadata commit, nothing written, nothing expired — the
/// later snapshots stay reachable by id until an expiry removes them. Refused
/// when the id is unknown or already current, and when `main` moved meanwhile.
/// Returns the new current snapshot id.
#[pyfunction]
fn rollback_to_snapshot(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    snapshot_id: i64,
) -> PyResult<i64> {
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
            let updated = tx
                .rollback_to_snapshot(snapshot_id)
                .apply(tx)
                .map_err(|e| PyValueError::new_err(format!("rollback {fqn}: {e}")))?
                .commit(catalog.as_ref())
                .await
                .map_err(|e| PyValueError::new_err(format!("rollback {fqn}: {e}")))?;
            updated
                .metadata()
                .current_snapshot_id()
                .ok_or_else(|| PyValueError::new_err("rollback left no current snapshot".to_string()))
        })
    })
}

/// Commit an EMPTY append: a new snapshot with the same files that records
/// the table's CURRENT schema id. A schema change creates no snapshot, so a
/// reader that binds the schema a snapshot recorded (a pinned read) keeps
/// seeing the pre-change column set until a snapshot made after the change
/// exists; this is that snapshot. `snapshot_properties` land on its summary.
/// Returns the new snapshot id.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, snapshot_properties=None))]
fn touch_snapshot(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    snapshot_properties: Option<HashMap<String, String>>,
) -> PyResult<i64> {
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
            let updated = tx
                .fast_append()
                .set_snapshot_properties(snapshot_properties.unwrap_or_default())
                .apply(tx)
                .map_err(|e| PyValueError::new_err(format!("touch {fqn}: {e}")))?
                .commit(catalog.as_ref())
                .await
                .map_err(|e| PyValueError::new_err(format!("touch {fqn}: {e}")))?;
            updated
                .metadata()
                .current_snapshot_id()
                .ok_or_else(|| PyValueError::new_err("touch produced no snapshot".to_string()))
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

/// The table's CURRENT properties as a dict (metadata-only read).
#[pyfunction]
fn table_properties(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
) -> PyResult<HashMap<String, String>> {
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
            Ok(t.metadata().properties().clone())
        })
    })
}

/// Register EXISTING parquet files as data files of a table (metadata-only
/// adoption — the files are NOT rewritten, moved, or validated beyond their
/// footers). One `fast_append` commit; returns the new snapshot id.
///
/// `files_json` is a JSON array of `{"path": "s3://...", "partition": [...]}`
/// where `partition` is the file's partition tuple as JSON values in the
/// table's DEFAULT partition-spec field order (`null` allowed per field;
/// `[]`/omitted for unpartitioned tables). The caller owns partition
/// correctness — values are typed against the spec's result types
/// (boolean/int/long/date/string/timestamp-micros supported) but never
/// derived from data.
///
/// Intended for test twins / adoption tooling: registering copies of REAL
/// files (e.g. another table's duck-written estate) into a scratch table
/// preserves their physical layout byte-for-byte, which no engine rewrite
/// can do.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, files_json, snapshot_properties=None))]
fn add_data_files(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    files_json: String,
    snapshot_properties: Option<HashMap<String, String>>,
) -> PyResult<i64> {
    use iceberg::spec::{Literal, PrimitiveLiteral, PrimitiveType, Struct, Type};
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::writer::file_writer::ParquetWriter;

    struct FileEntry {
        path: String,
        partition: Vec<serde_json::Value>,
    }
    let parsed: serde_json::Value = serde_json::from_str(&files_json)
        .map_err(|e| PyValueError::new_err(format!("files_json: {e}")))?;
    let raw = parsed
        .as_array()
        .ok_or_else(|| PyValueError::new_err("files_json must be a JSON array"))?;
    let mut entries: Vec<FileEntry> = Vec::with_capacity(raw.len());
    for item in raw {
        let path = item
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or_else(|| PyValueError::new_err("files_json entry missing `path`"))?
            .to_string();
        let partition = match item.get("partition") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::Array(a)) => a.clone(),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "`partition` must be an array, got {other}"
                )));
            }
        };
        entries.push(FileEntry { path, partition });
    }
    if entries.is_empty() {
        return Err(PyValueError::new_err("files_json is empty"));
    }
    let (catalog_name, ns, table_name) = split_table_fqn(&fqn)?;

    fn json_to_literal(v: &serde_json::Value, ty: &Type) -> PyResult<Option<Literal>> {
        if v.is_null() {
            return Ok(None);
        }
        let Type::Primitive(p) = ty else {
            return Err(PyValueError::new_err(format!(
                "unsupported partition field type {ty}"
            )));
        };
        let lit = match p {
            PrimitiveType::Boolean => Literal::Primitive(PrimitiveLiteral::Boolean(
                v.as_bool()
                    .ok_or_else(|| PyValueError::new_err(format!("expected bool, got {v}")))?,
            )),
            PrimitiveType::Int | PrimitiveType::Date => Literal::Primitive(PrimitiveLiteral::Int(
                v.as_i64()
                    .ok_or_else(|| PyValueError::new_err(format!("expected int, got {v}")))?
                    as i32,
            )),
            PrimitiveType::Long | PrimitiveType::Timestamp | PrimitiveType::Timestamptz => {
                Literal::Primitive(PrimitiveLiteral::Long(
                    v.as_i64()
                        .ok_or_else(|| PyValueError::new_err(format!("expected long, got {v}")))?,
                ))
            }
            PrimitiveType::String => Literal::Primitive(PrimitiveLiteral::String(
                v.as_str()
                    .ok_or_else(|| PyValueError::new_err(format!("expected string, got {v}")))?
                    .to_string(),
            )),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported partition field type {other}"
                )));
            }
        };
        Ok(Some(lit))
    }

    py.detach(|| {
        runtime().block_on(async move {
            let catalog = build_catalog(catalog_name, catalog_props).await?;
            let namespace =
                NamespaceIdent::from_vec(ns).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let table = catalog
                .load_table(&iceberg::TableIdent::new(namespace, table_name))
                .await
                .map_err(|e| PyValueError::new_err(format!("loading {fqn}: {e}")))?;
            let metadata = table.metadata();
            let spec = metadata.default_partition_spec();
            let partition_type = spec
                .partition_type(metadata.current_schema())
                .map_err(|e| PyValueError::new_err(format!("partition type: {e}")))?;
            let fields = partition_type.fields();

            let mut typed: Vec<(String, Struct)> = Vec::with_capacity(entries.len());
            for e in &entries {
                if e.partition.len() != fields.len() {
                    return Err(PyValueError::new_err(format!(
                        "{}: partition tuple has {} values, spec has {} fields",
                        e.path,
                        e.partition.len(),
                        fields.len()
                    )));
                }
                let mut lits: Vec<Option<Literal>> = Vec::with_capacity(fields.len());
                for (v, f) in e.partition.iter().zip(fields.iter()) {
                    lits.push(json_to_literal(v, &f.field_type)?);
                }
                typed.push((e.path.clone(), Struct::from_iter(lits)));
            }

            let data_files = ParquetWriter::parquet_files_to_data_files_with_partition(
                table.file_io(),
                typed,
                metadata,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("reading parquet footers: {e}")))?;

            let tx = Transaction::new(&table);
            let updated = tx
                .fast_append()
                .add_data_files(data_files)
                .set_snapshot_properties(snapshot_properties.unwrap_or_default())
                .apply(tx)
                .map_err(|e| PyValueError::new_err(format!("fast_append: {e}")))?
                .commit(catalog.as_ref())
                .await
                .map_err(|e| PyValueError::new_err(format!("committing {fqn}: {e}")))?;
            updated
                .metadata()
                .current_snapshot()
                .map(|s| s.snapshot_id())
                .ok_or_else(|| PyValueError::new_err("commit produced no snapshot".to_string()))
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "catalog")?;
    this.add_function(wrap_pyfunction!(create_table, &this)?)?;
    this.add_function(wrap_pyfunction!(add_data_files, &this)?)?;
    this.add_function(wrap_pyfunction!(create_namespace, &this)?)?;
    this.add_function(wrap_pyfunction!(table_exists, &this)?)?;
    this.add_function(wrap_pyfunction!(set_properties, &this)?)?;
    this.add_function(wrap_pyfunction!(expire_snapshots, &this)?)?;
    this.add_function(wrap_pyfunction!(table_schema_json, &this)?)?;
    this.add_function(wrap_pyfunction!(snapshot_summary_json, &this)?)?;
    this.add_function(wrap_pyfunction!(table_partition_spec_json, &this)?)?;
    this.add_function(wrap_pyfunction!(rollback_to_snapshot, &this)?)?;
    this.add_function(wrap_pyfunction!(touch_snapshot, &this)?)?;
    this.add_function(wrap_pyfunction!(drop_table, &this)?)?;
    this.add_function(wrap_pyfunction!(table_properties, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}

#[cfg(test)]
mod schema_of_tests {
    use iceberg::spec::TableMetadata;

    use super::schema_of;

    /// Two schemas (0 = `x`; 1 = `x, y, z`, the current one) and two
    /// snapshots: the older one recording `older_schema_id`, the head
    /// recording schema 1.
    const OLDER: i64 = 3051729675574597004;
    const HEAD: i64 = 3055729675574597004;

    fn metadata(older_schema_id: Option<i32>) -> TableMetadata {
        let mut doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../crates/iceberg/testdata/table_metadata/TableMetadataV2Valid.json"
        ))
        .unwrap();
        let older = doc["snapshots"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|s| s["snapshot-id"] == OLDER)
            .unwrap();
        match older_schema_id {
            Some(id) => older["schema-id"] = id.into(),
            None => {
                older.as_object_mut().unwrap().remove("schema-id");
            }
        }
        serde_json::from_value(doc).unwrap()
    }

    fn names(metadata: &TableMetadata, snapshot_id: Option<i64>) -> (i32, Vec<String>) {
        let schema = schema_of(metadata, snapshot_id).unwrap();
        let names = schema
            .as_struct()
            .fields()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        (schema.schema_id(), names)
    }

    #[test]
    fn no_pin_is_the_current_schema() {
        let m = metadata(Some(0));
        assert_eq!(
            names(&m, None),
            (1, vec!["x".into(), "y".into(), "z".into()])
        );
    }

    #[test]
    fn a_pin_resolves_to_the_schema_the_snapshot_recorded() {
        let m = metadata(Some(0));
        // written before `y` and `z` were added: the pin does not see them
        assert_eq!(names(&m, Some(OLDER)), (0, vec!["x".into()]));
        assert_eq!(
            names(&m, Some(HEAD)),
            (1, vec!["x".into(), "y".into(), "z".into()])
        );
    }

    #[test]
    fn a_snapshot_without_a_recorded_schema_resolves_to_the_current_one() {
        let m = metadata(None);
        assert_eq!(
            names(&m, Some(OLDER)),
            (1, vec!["x".into(), "y".into(), "z".into()])
        );
    }

    #[test]
    fn an_unknown_snapshot_is_an_error_not_the_current_schema() {
        let m = metadata(Some(0));
        let err = schema_of(&m, Some(42)).unwrap_err();
        assert!(err.contains("snapshot 42"), "{err}");
        assert!(err.contains("unknown or expired"), "{err}");
    }
}
