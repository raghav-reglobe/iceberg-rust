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

use arrow_schema::{DataType, Fields};
use futures::TryStreamExt;
use iceberg::arrow::{merge_variant_schemas, schema_of_variant};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use parquet::variant::VariantArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

/// Infer the top-level field schema of a V3 Variant column by sampling rows of
/// an Iceberg table.
///
/// Reads up to `sample_limit` non-null values of `column` (a Variant column),
/// runs Spark's `SchemaOfVariant` inference + merge on each, and returns the
/// merged top-level struct's fields as `(field_name, type_str)` pairs where
/// `type_str` is a DuckDB-ish DDL type the merge worker can use to flatten the
/// document into typed columns. Nested `Struct`/`List` fields map to `"JSON"`
/// (the worker keeps them in-doc rather than flattening).
///
/// `catalog_props` are standard Iceberg REST catalog properties (`uri`,
/// `warehouse`, `credential`, `oauth2-server-uri`, `scope`, ...); `fqn` is
/// `catalog.namespace.table`. Blocks until done; raises `ValueError` on
/// failure.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, column, sample_limit=10000))]
fn infer_variant_schema(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    column: String,
    sample_limit: usize,
) -> PyResult<Vec<(String, String)>> {
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

    // Release the GIL: the catalog build + scan are long-running I/O.
    let merged: DataType = py.detach(|| {
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
            let table = catalog
                .load_table(&ident)
                .await
                .map_err(|e| PyValueError::new_err(format!("load table {fqn}: {e}")))?;

            // Scan only the variant column. The read pipeline unshreds
            // shredded variants, so the column arrives as the canonical
            // `Struct{metadata: Binary, value: Binary}`.
            let mut stream = table
                .scan()
                .select([column.clone()])
                .build()
                .map_err(|e| PyValueError::new_err(format!("build scan {fqn}: {e}")))?
                .to_arrow()
                .await
                .map_err(|e| PyValueError::new_err(format!("scan {fqn}: {e}")))?;

            // Seed with the null-sentinel and fold every sampled value in.
            let mut merged = DataType::Null;
            let mut seen = 0usize;
            'outer: while let Some(batch) = stream
                .try_next()
                .await
                .map_err(|e| PyValueError::new_err(format!("read {fqn}: {e}")))?
            {
                let Some(col) = batch.column_by_name(&column) else {
                    return Err(PyValueError::new_err(format!(
                        "column `{column}` not found in {fqn}"
                    )));
                };
                let variant_array = VariantArray::try_new(col.as_ref()).map_err(|e| {
                    PyValueError::new_err(format!("column `{column}` is not a Variant column: {e}"))
                })?;
                for i in 0..variant_array.len() {
                    if variant_array.is_null(i) {
                        continue;
                    }
                    let variant = variant_array.value(i);
                    let schema = schema_of_variant(&variant);
                    merged = merge_variant_schemas(&merged, &schema);
                    seen += 1;
                    if seen >= sample_limit {
                        break 'outer;
                    }
                }
            }
            Ok::<DataType, PyErr>(merged)
        })
    })?;

    // Top-level must be a struct (a MongoDB document). Anything else (empty
    // sample -> Null, or a scalar/array variant) yields no flattenable fields.
    let DataType::Struct(fields) = merged else {
        return Ok(Vec::new());
    };
    Ok(fields
        .iter()
        .map(|f| (f.name().to_string(), ddl_type(f.data_type())))
        .collect())
}

/// Map an Arrow [`DataType`] to a DuckDB-ish DDL type string. Nested
/// `Struct`/`List` -> `"JSON"` (the merge worker keeps them in-doc).
fn ddl_type(dt: &DataType) -> String {
    match dt {
        DataType::Int64 => "BIGINT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Float32 => "FLOAT".to_string(),
        DataType::Boolean => "BOOLEAN".to_string(),
        DataType::Utf8 => "VARCHAR".to_string(),
        DataType::Timestamp(_, _) => "TIMESTAMP".to_string(),
        DataType::Date32 => "DATE".to_string(),
        DataType::Time64(_) => "TIME".to_string(),
        DataType::Decimal128(p, s) => format!("DECIMAL({p},{s})"),
        DataType::Binary | DataType::FixedSizeBinary(_) => "BLOB".to_string(),
        // A single-field struct may be a MongoDB ExtendedJSON wrapper
        // ({"$oid": ...}, {"$date": ...}, ...) — a SCALAR, not a nested
        // document — in which case unwrap it to the scalar DDL.
        DataType::Struct(fields) => {
            extended_json_scalar(fields).unwrap_or_else(|| "JSON".to_string())
        }
        // Nested types + the Null sentinel + anything else: keep in-doc.
        _ => "JSON".to_string(),
    }
}

/// MongoDB ExtendedJSON represents scalars as single-field wrapper objects
/// (`{"$oid": "..."}`, `{"$date": ...}`, `{"$numberLong": "..."}`, ...). These
/// are SCALARS, not nested documents, so map them to the unwrapped scalar DDL
/// (mirrors the merge worker's `mongo_parse._infer_itype`) — otherwise a date /
/// oid / number field is misclassified as nested `JSON` and never flattened to a
/// typed column. Returns `None` for a genuine multi-field nested struct.
fn extended_json_scalar(fields: &Fields) -> Option<String> {
    if fields.len() != 1 {
        return None;
    }
    match fields[0].name().as_str() {
        "$oid" => Some("VARCHAR".to_string()),
        "$date" => Some("TIMESTAMP".to_string()),
        "$numberLong" | "$numberInt" => Some("BIGINT".to_string()),
        "$numberDecimal" | "$numberDouble" => Some("DOUBLE".to_string()),
        _ => None,
    }
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "variant_schema")?;
    this.add_function(wrap_pyfunction!(infer_variant_schema, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
