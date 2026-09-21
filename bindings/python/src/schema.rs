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

use iceberg::spec::{PrimitiveType, Type, VariantType};
use iceberg::transaction::{AddColumn, ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

/// Map a simple type keyword to an Iceberg `Type`: scalars + `variant`.
/// `variant` is the point of evolving schemas through this path — engines whose
/// `ALTER ADD COLUMN` cannot add a VARIANT column exist, and a Python-side
/// metadata round trip can turn VARIANT into `unknown`.
fn parse_type(s: &str) -> PyResult<Type> {
    let norm = s.trim().to_ascii_lowercase();
    // `decimal(p,s)` / `decimal(p, s)` — variant-schema discovery
    // emits `DECIMAL(p,s)` for unwrapped ExtendedJSON `$numberDecimal` fields.
    if let Some(args) = norm
        .strip_prefix("decimal(")
        .and_then(|r| r.strip_suffix(')'))
    {
        let mut it = args.splitn(2, ',');
        let (p, sc) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        let precision: u32 = p
            .trim()
            .parse()
            .map_err(|_| PyValueError::new_err(format!("bad decimal precision in `{s}`")))?;
        let scale: u32 = sc
            .trim()
            .parse()
            .map_err(|_| PyValueError::new_err(format!("bad decimal scale in `{s}`")))?;
        return Ok(Type::Primitive(PrimitiveType::Decimal { precision, scale }));
    }
    Ok(match norm.as_str() {
        "variant" => Type::Variant(VariantType),
        "string" | "varchar" | "text" => Type::Primitive(PrimitiveType::String),
        "long" | "bigint" => Type::Primitive(PrimitiveType::Long),
        "int" | "integer" => Type::Primitive(PrimitiveType::Int),
        "double" | "float" => Type::Primitive(PrimitiveType::Double),
        "boolean" | "bool" => Type::Primitive(PrimitiveType::Boolean),
        "timestamp" => Type::Primitive(PrimitiveType::Timestamp),
        // A source column the upstream writer typed WITH a zone must evolve
        // onto the target with the same type a fresh create would give it;
        // refusing the keyword failed the whole evolution commit (every
        // other added column with it), and did so again on every merge.
        "timestamptz" | "timestamp_tz" => Type::Primitive(PrimitiveType::Timestamptz),
        "date" => Type::Primitive(PrimitiveType::Date),
        "time" => Type::Primitive(PrimitiveType::Time),
        "binary" | "blob" => Type::Primitive(PrimitiveType::Binary),
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported add_column type `{other}` (expected one of \
                 variant/string/long/int/double/boolean/timestamp/timestamptz/\
                 date/time/binary/decimal(p,s))"
            )));
        }
    })
}

/// Evolve an existing Iceberg table's schema in ONE metadata-only commit:
/// optionally add nullable columns (`add_columns` = list of `(name, type_keyword)`)
/// and/or drop columns (`drop_columns`). Both happen in a single
/// `UpdateSchemaAction` -> `Transaction::commit`.
///
/// VARIANT-SAFE: goes through the same iceberg-rust path that creates VARIANT
/// tables, so (a) it never turns VARIANT into `unknown` on load/write, and (b)
/// it CAN add a VARIANT column (a nested document field). Added columns are always
/// optional/nullable (a new column is undefined for existing rows).
///
/// `catalog_props` are standard Iceberg REST catalog properties (`uri`,
/// `warehouse`, `credential`, `oauth2-server-uri`, `scope`, ...); `fqn` is
/// `catalog.namespace.table`. Blocks until the commit lands; raises
/// `ValueError` on failure. A no-op (both lists empty) returns immediately.
#[pyfunction]
#[pyo3(signature = (catalog_props, fqn, add_columns=Vec::new(), drop_columns=Vec::new(), update_columns=Vec::new()))]
fn update_schema(
    py: Python<'_>,
    catalog_props: HashMap<String, String>,
    fqn: String,
    add_columns: Vec<(String, String)>,
    drop_columns: Vec<String>,
    update_columns: Vec<(String, String)>,
) -> PyResult<()> {
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

    if add_columns.is_empty() && drop_columns.is_empty() && update_columns.is_empty() {
        return Ok(());
    }
    // Parse types up-front so a bad keyword errors before we touch the catalog.
    let adds: Vec<(String, Type)> = add_columns
        .into_iter()
        .map(|(n, t)| parse_type(&t).map(|ty| (n, ty)))
        .collect::<PyResult<_>>()?;
    // SAFE promotions on existing root-level columns (int->long, float->double,
    // decimal precision widen — validated at commit by the action).
    let updates: Vec<(String, Type)> = update_columns
        .into_iter()
        .map(|(n, t)| parse_type(&t).map(|ty| (n, ty)))
        .collect::<PyResult<_>>()?;

    // Release the GIL (pyo3 0.28 `detach`): catalog IO + commit is a blocking
    // call, so other Python threads (e.g. a queue-drain pool) can run.
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
                .map_err(|e| PyValueError::new_err(format!("load {fqn}: {e}")))?;

            let tx = Transaction::new(&table);
            let mut action = tx.update_schema();
            for (name, ty) in adds {
                // Dotted add name = NESTED add: "address.zip" -> leaf "zip"
                // under parent path "address" (AddColumn::with_parent). A
                // column name with a literal dot cannot be expressed here;
                // the split takes the LAST dot.
                match name.rsplit_once('.') {
                    Some((parent, leaf)) => {
                        action =
                            action.add_column(AddColumn::optional(leaf, ty).with_parent(parent));
                    }
                    None => {
                        action = action.add_column(AddColumn::optional(name, ty));
                    }
                }
            }
            for name in drop_columns {
                action = action.delete_column(name);
            }
            for (name, ty) in updates {
                action = action.update_column_type(name, ty);
            }
            let tx = action
                .apply(tx)
                .map_err(|e| PyValueError::new_err(format!("apply update_schema {fqn}: {e}")))?;
            tx.commit(&catalog)
                .await
                .map_err(|e| PyValueError::new_err(format!("commit update_schema {fqn}: {e}")))?;
            Ok(())
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "schema")?;
    this.add_function(wrap_pyfunction!(update_schema, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}

#[cfg(test)]
mod parse_type_tests {
    use super::*;

    #[test]
    fn zoned_timestamp_is_its_own_type() {
        for kw in ["timestamptz", "TIMESTAMPTZ", " timestamp_tz "] {
            assert_eq!(
                parse_type(kw).unwrap(),
                Type::Primitive(PrimitiveType::Timestamptz),
                "{kw}"
            );
        }
        // the zoneless keyword keeps its meaning
        assert_eq!(
            parse_type("timestamp").unwrap(),
            Type::Primitive(PrimitiveType::Timestamp)
        );
    }

    #[test]
    fn the_rest_of_the_vocabulary_is_unchanged() {
        assert_eq!(
            parse_type("bigint").unwrap(),
            Type::Primitive(PrimitiveType::Long)
        );
        assert_eq!(
            parse_type("float").unwrap(),
            Type::Primitive(PrimitiveType::Double)
        );
        assert_eq!(
            parse_type("DECIMAL(18, 4)").unwrap(),
            Type::Primitive(PrimitiveType::Decimal {
                precision: 18,
                scale: 4
            })
        );
        assert_eq!(parse_type("variant").unwrap(), Type::Variant(VariantType));
    }

    #[test]
    fn an_unknown_keyword_is_still_refused_and_names_the_vocabulary() {
        // pyo3 errors need the interpreter only to be RENDERED; is_err does not.
        assert!(parse_type("timestamptz_ns").is_err());
        assert!(parse_type("geometry").is_err());
    }
}
