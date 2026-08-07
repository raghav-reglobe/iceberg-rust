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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, new_empty_array};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema as ArrowSchema};
use parquet::variant::{VariantArray, shred_variant, unshred_variant};

use crate::spec::{Schema, Type};
use crate::{Error, ErrorKind, Result};

/// Is this arrow type a SHREDDED variant column layout — a struct carrying a
/// `typed_value` subtree beside `metadata`? A canonical variant is
/// `{metadata, value}` with no `typed_value`; anything non-struct is not a
/// variant layout at all. Callers deciding passthrough MUST pair this with an
/// iceberg-schema check that the column really is `Type::Variant` — a plain
/// struct column could carry these child names by coincidence.
pub fn is_shredded_variant_type(t: &DataType) -> bool {
    matches!(t, DataType::Struct(sub)
        if sub.iter().any(|f| f.name() == "metadata")
            && sub.iter().any(|f| f.name() == "typed_value"))
}

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
    shred_types_with_arrow_from_file_schema(file_schema, table_schema)
        .into_iter()
        .map(|(name, (plain, _))| (name, plain))
        .collect()
}

/// Like [`shred_types_from_file_schema`], but ALSO returns the file's OWN
/// full arrow type for each shredded variant column (verbatim, as the scan
/// will produce it raw). The file type — not the shred kernel's probe output
/// — is the writer layout that makes shredded PASSTHROUGH possible: a
/// late-fetched column from a same-layout file compares equal byte-for-byte
/// (the kernel's output differs in child ORDER and uses BinaryView where
/// files carry Binary, so exact equality against a kernel-derived layout
/// never holds).
pub fn shred_types_with_arrow_from_file_schema(
    file_schema: &ArrowSchema,
    table_schema: &Schema,
) -> HashMap<String, (DataType, DataType)> {
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
            out.insert(field.name().to_string(), (plain, field.data_type().clone()));
        }
    }
    out
}

/// Fold one SHREDDED variant column back to the canonical
/// `Struct([metadata: Binary, value: Binary])` layout via arrow-rs's
/// [`unshred_variant`] kernel. Per-row blob reconstruction — the exact cost
/// the shredded-passthrough paths exist to avoid; this is the shared,
/// bounded fallback for columns that cannot ride a passthrough writer.
/// The returned field keeps `field`'s name, metadata and nullability.
pub fn fold_shredded_column(field: &Field, col: &ArrayRef) -> Result<(Field, ArrayRef)> {
    let variant = VariantArray::try_new(col.as_ref()).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("column {} is not a variant array", field.name()),
        )
        .with_source(e)
    })?;
    let unshredded = unshred_variant(&variant).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("unshredding column {}", field.name()),
        )
        .with_source(e)
    })?;
    let inner: StructArray = unshredded.into_inner();
    let metadata = inner.column_by_name("metadata").ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            "unshredded variant is missing its 'metadata' field",
        )
    })?;
    let value = inner.column_by_name("value").ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            "unshredded variant is missing its 'value' field",
        )
    })?;
    // `unshred_variant` emits BinaryView; canonical iceberg variant carries
    // plain Binary.
    let metadata = arrow_cast::cast(metadata.as_ref(), &DataType::Binary)?;
    let value = arrow_cast::cast(value.as_ref(), &DataType::Binary)?;
    let sub_fields = Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ]);
    let folded = StructArray::new(
        sub_fields.clone(),
        vec![metadata, value],
        col.nulls().cloned(),
    );
    let new_field = Field::new(
        field.name(),
        DataType::Struct(sub_fields),
        field.is_nullable(),
    )
    .with_metadata(field.metadata().clone());
    Ok((new_field, Arc::new(folded) as ArrayRef))
}

/// Fold exactly the NAMED shredded variant columns of a batch back to the
/// canonical layout (see [`fold_shredded_column`]). Named columns that are
/// not shredded (already canonical) pass through untouched.
pub fn unshred_batch_columns(batch: &RecordBatch, columns: &HashSet<String>) -> Result<RecordBatch> {
    if columns.is_empty() {
        return Ok(batch.clone());
    }
    let schema = batch.schema();
    let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    let mut changed = false;
    for name in columns {
        let Ok(idx) = schema.index_of(name) else {
            continue;
        };
        if !is_shredded_variant_type(fields[idx].data_type()) {
            continue;
        }
        let (new_field, folded) = fold_shredded_column(&fields[idx], &cols[idx])?;
        fields[idx] = Arc::new(new_field);
        cols[idx] = folded;
        changed = true;
    }
    if !changed {
        return Ok(batch.clone());
    }
    let new_schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ));
    RecordBatch::try_new(new_schema, cols)
        .map_err(|e| Error::new(ErrorKind::DataInvalid, "rebuilding folded batch").with_source(e))
}

/// Are two shredded layouts the SAME SHAPE modulo child order and
/// Binary/BinaryView leaves — i.e. can [`conform_variant_to_type`] map one
/// onto the other losslessly? False when either side has a child the other
/// lacks (e.g. a value-only shred node the plain derivation skips).
pub fn shred_shape_compatible(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Struct(ac), DataType::Struct(bc)) => {
            if ac.len() != bc.len() {
                return false;
            }
            ac.iter().all(|af| {
                bc.iter()
                    .find(|bf| bf.name() == af.name())
                    .is_some_and(|bf| shred_shape_compatible(af.data_type(), bf.data_type()))
            })
        }
        (DataType::List(ae), DataType::List(be))
        | (DataType::LargeList(ae), DataType::LargeList(be)) => {
            shred_shape_compatible(ae.data_type(), be.data_type())
        }
        (DataType::Binary, DataType::BinaryView)
        | (DataType::BinaryView, DataType::Binary)
        | (DataType::Utf8, DataType::Utf8View)
        | (DataType::Utf8View, DataType::Utf8) => true,
        (x, y) => x == y,
    }
}

/// Structurally conform a (shredded) variant array to `target`: match struct
/// children BY NAME (the shred kernel orders object fields differently than
/// files do), recurse through lists, and cast BinaryView <-> Binary leaves
/// (the kernel emits views; files carry plain binary). This is COLUMNAR
/// metadata shuffling — child arrays are moved, not row-decoded; the only
/// buffer work is the view->binary cast. Errors loudly on a child the target
/// has that the source lacks (a genuine layout mismatch — the caller should
/// have folded + re-shredded that column instead).
pub fn conform_variant_to_type(arr: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    if arr.data_type() == target {
        return Ok(Arc::clone(arr));
    }
    match (arr.data_type(), target) {
        (DataType::Struct(schildren), DataType::Struct(tchildren)) => {
            // A source child ABSENT from the target would be silently
            // dropped — for a `typed_value` subtree that is data loss (those
            // values are NOT duplicated in the binary `value` residual).
            // Callers must fold + re-shred such columns instead.
            if let Some(extra) = schildren
                .iter()
                .find(|sf| !tchildren.iter().any(|tf| tf.name() == sf.name()))
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "conforming variant layout: target lacks source child `{}` — \
                         dropping it would lose shredded values",
                        extra.name()
                    ),
                ));
            }
            let src = arr
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("struct type is a StructArray");
            let mut cols: Vec<ArrayRef> = Vec::with_capacity(tchildren.len());
            for tf in tchildren.iter() {
                let child = src.column_by_name(tf.name()).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "conforming variant layout: source struct lacks child `{}`",
                            tf.name()
                        ),
                    )
                })?;
                cols.push(conform_variant_to_type(child, tf.data_type())?);
            }
            Ok(Arc::new(StructArray::new(
                tchildren.clone(),
                cols,
                src.nulls().cloned(),
            )) as ArrayRef)
        }
        (DataType::List(_), DataType::List(tel)) => {
            let src = arr
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .expect("list type is a ListArray");
            let values = conform_variant_to_type(src.values(), tel.data_type())?;
            Ok(Arc::new(arrow_array::ListArray::new(
                Arc::clone(tel),
                src.offsets().clone(),
                values,
                src.nulls().cloned(),
            )) as ArrayRef)
        }
        // Leaf conversions the kernel-vs-file split produces (BinaryView vs
        // Binary); any other leaf difference is a real mismatch -> cast()
        // errors loudly.
        _ => arrow_cast::cast(arr.as_ref(), target).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("conforming variant leaf {} -> {target}", arr.data_type()),
            )
            .with_source(e)
        }),
    }
}

/// Conform the named (already shredded) variant columns of a batch to the
/// writer's target layouts. Columns already matching pass through untouched.
pub fn conform_batch_variants(
    batch: &RecordBatch,
    targets: &HashMap<String, DataType>,
) -> Result<RecordBatch> {
    if targets.is_empty() {
        return Ok(batch.clone());
    }
    let schema = batch.schema();
    let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    let mut changed = false;
    for (name, target) in targets {
        let Ok(idx) = schema.index_of(name) else {
            continue;
        };
        if columns[idx].data_type() == target {
            continue;
        }
        let conformed = conform_variant_to_type(&columns[idx], target)?;
        fields[idx] = Arc::new(fields[idx].as_ref().clone().with_data_type(target.clone()));
        columns[idx] = conformed;
        changed = true;
    }
    if !changed {
        return Ok(batch.clone());
    }
    let new_schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ));
    RecordBatch::try_new(new_schema, columns).map_err(|e| {
        Error::new(ErrorKind::DataInvalid, "rebuilding conformed batch").with_source(e)
    })
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
            Field::new("value", DataType::Binary, true),
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

    #[test]
    fn shredded_variant_type_detection() {
        assert!(!is_shredded_variant_type(&canonical_type()));
        assert!(!is_shredded_variant_type(&DataType::Int64));
        let plain = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        let arr = canonical_doc_array(&[Some(doc_bytes(1, &[]))]);
        let variant = VariantArray::try_new(&arr).unwrap();
        let shredded = ArrayRef::from(shred_variant(&variant, &plain).unwrap());
        assert!(is_shredded_variant_type(shredded.data_type()));
    }

    /// Conforming a shredded layout to a target that LACKS one of its
    /// `typed_value` children must refuse — silently dropping the child
    /// would lose the shredded values (they are not in the `value` residual).
    #[test]
    fn conform_refuses_dropping_source_children() {
        let wide = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]));
        let rows = vec![Some(doc_bytes(1, &[10]))];
        let arr = canonical_doc_array(&rows);
        let variant = VariantArray::try_new(&arr).unwrap();
        let shredded = ArrayRef::from(shred_variant(&variant, &wide).unwrap());

        // Target: the same layout with the `b` shred node removed from the
        // typed_value subtree.
        let DataType::Struct(root) = shredded.data_type() else {
            panic!("shredded root is a struct");
        };
        let narrowed: Vec<Field> = root
            .iter()
            .map(|f| {
                if f.name() != "typed_value" {
                    return f.as_ref().clone();
                }
                let DataType::Struct(obj) = f.data_type() else {
                    panic!("typed_value is an object node");
                };
                let kept: Vec<Field> = obj
                    .iter()
                    .filter(|c| c.name() != "b")
                    .map(|c| c.as_ref().clone())
                    .collect();
                f.as_ref()
                    .clone()
                    .with_data_type(DataType::Struct(kept.into()))
            })
            .collect();
        let target = DataType::Struct(narrowed.into());

        let err = conform_variant_to_type(&shredded, &target).unwrap_err();
        assert!(
            err.to_string().contains("lose shredded values"),
            "refusal names the hazard: {err}"
        );
    }

    /// `unshred_batch_columns` folds exactly the NAMED shredded columns back
    /// to canonical — values reconstructed, other columns untouched.
    #[test]
    fn unshred_batch_columns_folds_named_only() {
        let plain = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        let rows = vec![Some(doc_bytes(7, &[70])), None];
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("doc", canonical_type(), true),
                Field::new("doc2", canonical_type(), true),
            ])),
            vec![canonical_doc_array(&rows), canonical_doc_array(&rows)],
        )
        .unwrap();
        let both = HashMap::from([
            ("doc".to_string(), plain.clone()),
            ("doc2".to_string(), plain.clone()),
        ]);
        let shredded = shred_record_batch(&batch, &both).unwrap();
        assert!(is_shredded_variant_type(shredded.column(0).data_type()));
        assert!(is_shredded_variant_type(shredded.column(1).data_type()));

        let folded =
            unshred_batch_columns(&shredded, &HashSet::from(["doc".to_string()])).unwrap();
        // Named column back at canonical shape (metadata + value, no
        // typed_value); the other keeps its shredded shape.
        assert!(!is_shredded_variant_type(folded.column(0).data_type()));
        let DataType::Struct(ch) = folded.column(0).data_type() else {
            panic!("folded column is a struct");
        };
        assert_eq!(
            ch.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            vec!["metadata", "value"]
        );
        assert!(is_shredded_variant_type(folded.column(1).data_type()));
        // Null slot preserved; values reconstruct.
        assert!(folded.column(0).is_null(1));
        let variant = VariantArray::try_new(folded.column(0)).unwrap();
        let Variant::Object(obj) = variant.value(0) else {
            panic!("row 0 folds back to an object");
        };
        assert_eq!(obj.get("a"), Some(Variant::Int64(7)));
    }
}
