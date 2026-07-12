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

//! Shred-preserving variant WRITE support, shared by every writer chain
//! (compaction rewrite and MoR merge).
//!
//! The read path folds SHREDDED variant columns back to the canonical
//! `{metadata, value}` layout, so a plain write of what was read silently
//! UN-shreds the output — correct, but it discards the columnar
//! `typed_value` subtree query engines prune into. The pieces here let a
//! writer re-shred instead:
//!
//! 1. [`shred_types_from_file_schema`] derives each variant column's PLAIN
//!    shredding type from an existing file's parquet (arrow) schema —
//!    shred-preserving: the output keeps the input's layout, a column that
//!    isn't shredded in the input stays canonical, and the layout is never
//!    invented from data;
//! 2. [`shred_record_batch`] runs a canonical batch through the
//!    `shred_variant` kernel;
//! 3. [`shredded_output_type`] computes the kernel's output type for a plain
//!    type WITHOUT data (a zero-row probe) — the writer schema override must
//!    be known before the first batch streams in.
//!
//! Values that don't match the shredding type stay in the binary `value`
//! (partial shredding) — semantics are unchanged either way, only the
//! physical layout differs.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StructArray, new_empty_array};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema as ArrowSchema};
use parquet::variant::{VariantArray, shred_variant};

use crate::spec::{Schema, Type};
use crate::{Error, ErrorKind, Result};

/// Is this field a `{value?, typed_value?}` shred node — the wrapper
/// `shred_variant` puts around each shredded object field / array element?
fn is_shred_node(t: &DataType) -> bool {
    matches!(t, DataType::Struct(node)
        if !node.is_empty()
            && node.iter().all(|f| f.name() == "value" || f.name() == "typed_value"))
}

/// Is this fields-set an OBJECT-shredding node set — every child a shred node?
fn is_object_node_set(children: &Fields) -> bool {
    !children.is_empty() && children.iter().all(|c| is_shred_node(c.data_type()))
}

/// Recover the PLAIN requested type from a file's `typed_value` tree — the
/// inverse of the `{value, typed_value}` wrapping `shred_variant` applies.
/// Fields shredded value-only (no `typed_value`) are skipped: re-shredding
/// simply leaves their values in the binary `value`, which is still correct.
/// Array-shredded nodes (`typed_value: List<{value?, typed_value?}>`) derive
/// to a plain `List<element>`; a value-only element derives nothing.
/// Returns `None` when nothing typed survives (column stays canonical).
fn unwrap_typed_value(t: &DataType) -> Option<DataType> {
    match t {
        DataType::Struct(children) if is_object_node_set(children) => {
            let fields: Vec<Field> = children
                .iter()
                .filter_map(|c| {
                    let DataType::Struct(node) = c.data_type() else {
                        return None;
                    };
                    let tv = node.iter().find(|f| f.name() == "typed_value")?;
                    let plain = unwrap_typed_value(tv.data_type())?;
                    Some(Field::new(c.name(), plain, true))
                })
                .collect();
            (!fields.is_empty()).then(|| DataType::Struct(fields.into()))
        }
        DataType::List(el) | DataType::LargeList(el) if is_shred_node(el.data_type()) => {
            let DataType::Struct(node) = el.data_type() else {
                return None;
            };
            let tv = node.iter().find(|f| f.name() == "typed_value")?;
            let plain = unwrap_typed_value(tv.data_type())?;
            Some(DataType::List(Arc::new(Field::new(el.name(), plain, true))))
        }
        DataType::List(_) | DataType::LargeList(_) => None,
        other => Some(other.clone()),
    }
}

/// Derive per-variant-column PLAIN shredding types from an existing file's
/// arrow schema (as loaded from its parquet footer). Columns not shredded in
/// the file (or whose shredding can't be derived) are absent from the map —
/// they stay canonical on write.
pub fn shred_types_from_file_schema(
    file_schema: &ArrowSchema,
    table_schema: &Schema,
) -> HashMap<String, DataType> {
    let mut out = HashMap::new();
    for field in file_schema.fields() {
        let Some(iceberg_field) = table_schema.field_by_name(field.name()) else {
            continue;
        };
        if !matches!(iceberg_field.field_type.as_ref(), Type::Variant(_)) {
            continue;
        }
        let DataType::Struct(children) = field.data_type() else {
            continue;
        };
        let Some(tv) = children.iter().find(|c| c.name() == "typed_value") else {
            continue; // canonical in the input -> canonical in the output
        };
        if let Some(plain) = unwrap_typed_value(tv.data_type()) {
            out.insert(field.name().to_string(), plain);
        }
    }
    out
}

/// The number of variant columns in an iceberg schema — the derivation's
/// natural early-stop bound when probing multiple files.
pub fn variant_column_count(table_schema: &Schema) -> usize {
    table_schema
        .as_struct()
        .fields()
        .iter()
        .filter(|f| matches!(f.field_type.as_ref(), Type::Variant(_)))
        .count()
}

/// The shredded output type `shred_variant` produces for `plain`, given the
/// CANONICAL variant column type the batches carry — computed from a zero-row
/// probe so it is available before any data streams through. Deterministic:
/// the kernel builds the output layout purely from the requested type (plus
/// the input's `metadata` child, which the canonical type fixes).
pub fn shredded_output_type(canonical: &DataType, plain: &DataType) -> Result<DataType> {
    let DataType::Struct(children) = canonical else {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("canonical variant type is not a struct: {canonical:?}"),
        ));
    };
    let empty: Vec<ArrayRef> = children
        .iter()
        .map(|f| new_empty_array(f.data_type()))
        .collect();
    let probe: ArrayRef = Arc::new(StructArray::new(children.clone(), empty, None));
    let variant = VariantArray::try_new(&probe).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "canonical variant type is not a variant layout",
        )
        .with_source(e)
    })?;
    let shredded = shred_variant(&variant, plain).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("deriving shredded output type for {plain:?}"),
        )
        .with_source(e)
    })?;
    Ok(ArrayRef::from(shredded).data_type().clone())
}

/// Shred the named variant columns of one CANONICAL batch. The transformed
/// batch's variant columns carry the types [`shredded_output_type`] predicts
/// for the same canonical/plain pair.
pub fn shred_record_batch(
    batch: &RecordBatch,
    plain_types: &HashMap<String, DataType>,
) -> Result<RecordBatch> {
    if plain_types.is_empty() {
        return Ok(batch.clone());
    }
    let schema = batch.schema();
    let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    for (name, plain) in plain_types {
        let idx = schema.index_of(name).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("variant column {name} missing from batch"),
            )
            .with_source(e)
        })?;
        let variant = VariantArray::try_new(&columns[idx]).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("column {name} is not a variant array"),
            )
            .with_source(e)
        })?;
        let shredded = shred_variant(&variant, plain).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, format!("shredding column {name}")).with_source(e)
        })?;
        let arr = ArrayRef::from(shredded);
        fields[idx] = Arc::new(
            fields[idx]
                .as_ref()
                .clone()
                .with_data_type(arr.data_type().clone()),
        );
        columns[idx] = arr;
    }
    let new_schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ));
    RecordBatch::try_new(new_schema, columns)
        .map_err(|e| Error::new(ErrorKind::DataInvalid, "rebuilding shredded batch").with_source(e))
}

#[cfg(test)]
mod tests {
    use arrow_array::BinaryArray;
    use arrow_buffer::NullBuffer;
    use parquet::variant::{Variant, VariantBuilder};

    use super::*;
    use crate::spec::{NestedField, PrimitiveType, VariantType};

    fn canonical_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::Binary, false),
        ]))
    }

    fn table_schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "doc", Type::Variant(VariantType)).into(),
            ])
            .build()
            .unwrap()
    }

    fn doc_bytes(a: i64, tags: &[i64]) -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        let mut obj = builder.new_object();
        obj.insert("a", a);
        let mut list = obj.new_list("tags");
        for t in tags {
            list.append_value(*t);
        }
        list.finish();
        obj.finish();
        builder.finish()
    }

    fn canonical_doc_array(rows: &[Option<(Vec<u8>, Vec<u8>)>]) -> ArrayRef {
        let metas = BinaryArray::from_iter_values(
            rows.iter()
                .map(|v| v.as_ref().map(|(m, _)| m.clone()).unwrap_or_default()),
        );
        let vals = BinaryArray::from_iter_values(
            rows.iter()
                .map(|v| v.as_ref().map(|(_, x)| x.clone()).unwrap_or_default()),
        );
        let validity = NullBuffer::from(rows.iter().map(|v| v.is_some()).collect::<Vec<_>>());
        let DataType::Struct(fields) = canonical_type() else {
            unreachable!()
        };
        Arc::new(StructArray::new(
            fields,
            vec![Arc::new(metas) as ArrayRef, Arc::new(vals) as ArrayRef],
            Some(validity),
        ))
    }

    /// Shred a real batch on an object-with-array plain type; derive back
    /// from the OUTPUT layout — the round trip must reproduce the plain type
    /// (object fields AND the array element).
    #[test]
    fn derive_round_trips_object_and_array_shredding() {
        let plain = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
                true,
            ),
        ]));
        let rows = vec![Some(doc_bytes(1, &[10, 11])), None];
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("doc", canonical_type(), true),
            ])),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1, 2])),
                canonical_doc_array(&rows),
            ],
        )
        .unwrap();

        let plain_types = HashMap::from([("doc".to_string(), plain.clone())]);
        let shredded = shred_record_batch(&batch, &plain_types).unwrap();
        let out_type = shredded.column(1).data_type().clone();

        // The writer-override probe predicts exactly the batch's output type.
        assert_eq!(
            shredded_output_type(&canonical_type(), &plain).unwrap(),
            out_type
        );

        // Footer derivation over the shredded layout recovers the plain type.
        let file_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("doc", out_type, true),
        ]);
        let derived = shred_types_from_file_schema(&file_schema, &table_schema());
        assert_eq!(derived.get("doc"), Some(&plain));
    }

    /// Canonical input derives nothing — the column stays canonical.
    #[test]
    fn canonical_file_derives_nothing() {
        let file_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("doc", canonical_type(), true),
        ]);
        assert!(shred_types_from_file_schema(&file_schema, &table_schema()).is_empty());
    }

    /// A value-only array element (no typed_value) derives nothing.
    #[test]
    fn value_only_array_element_falls_back_to_canonical() {
        let element = DataType::Struct(Fields::from(vec![Field::new(
            "value",
            DataType::BinaryView,
            true,
        )]));
        let shredded_doc = DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::BinaryView, true),
            Field::new(
                "typed_value",
                DataType::List(Arc::new(Field::new("element", element, true))),
                true,
            ),
        ]));
        let file_schema = ArrowSchema::new(vec![Field::new("doc", shredded_doc, true)]);
        assert!(shred_types_from_file_schema(&file_schema, &table_schema()).is_empty());
    }

    /// Shredding moves matching values into the typed columns: the output's
    /// `typed_value.a.typed_value` Int64 column carries the field values.
    #[test]
    fn shredded_values_land_in_typed_columns() {
        let plain = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        let rows = vec![Some(doc_bytes(7, &[1])), Some(doc_bytes(9, &[]))];
        let arr = canonical_doc_array(&rows);
        let variant = VariantArray::try_new(&arr).unwrap();
        let shredded = ArrayRef::from(shred_variant(&variant, &plain).unwrap());
        let root = shredded
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("shredded root is a struct");
        let tv = root
            .column_by_name("typed_value")
            .expect("typed_value subtree present")
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let a_node = tv
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let a_typed = a_node
            .column_by_name("typed_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(a_typed.value(0), 7);
        assert_eq!(a_typed.value(1), 9);
    }

    /// `variant_column_count` counts only variant fields.
    #[test]
    fn variant_count() {
        assert_eq!(variant_column_count(&table_schema()), 1);
    }
}
