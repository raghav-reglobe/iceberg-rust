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

//! 64-bit-offset widening of variable-length Parquet columns.

use std::sync::Arc;

use arrow_schema::{
    DataType, Field, FieldRef, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::schema::{ArrowSchemaVisitor, DEFAULT_MAP_FIELD_NAME, visit_schema};
use crate::error::Result;
use crate::spec::{PrimitiveType, Schema, Type};
use crate::{Error, ErrorKind};

/// Widen 32-bit-offset variable-length Arrow types (`Utf8`/`Binary`) to their
/// 64-bit forms (`LargeUtf8`/`LargeBinary`) for columns whose Iceberg type is
/// `string`/`binary`.
///
/// arrow-rs infers `Utf8`/`Binary` for Parquet `BYTE_ARRAY` columns unless the
/// file's embedded Arrow metadata says otherwise. A 32-bit-offset array caps
/// at 2 GiB of values, which wide-TEXT tables exceed within a single decoded
/// batch or downstream concat ("Offset overflow" at i32::MAX). Supplying the
/// widened schema via arrow-rs's schema hint (`ArrowReaderOptions::with_schema`,
/// the same mechanism as INT96 coercion) makes the reader materialize 64-bit
/// offsets directly — no post-decode cast that would first build the
/// overflowing 32-bit array.
///
/// Matching is by Parquet field-id against the Iceberg schema (never by name):
/// variant sub-fields (`metadata`/`value`/`typed_value`) carry no field ids
/// and keep their exact physical types, which the variant kernels require.
pub(crate) fn widen_variable_length_types(
    arrow_schema: &ArrowSchemaRef,
    iceberg_schema: &Schema,
) -> Option<Arc<ArrowSchema>> {
    let mut visitor = LargeOffsetVisitor::new(iceberg_schema);
    let widened = visit_schema(arrow_schema, &mut visitor).ok()?;
    if visitor.changed {
        Some(Arc::new(widened))
    } else {
        None
    }
}

/// Visitor that widens `Utf8`→`LargeUtf8` and `Binary`→`LargeBinary` where the
/// Iceberg schema declares `string`/`binary` for the field id.
struct LargeOffsetVisitor<'a> {
    iceberg_schema: &'a Schema,
    field_stack: Vec<FieldRef>,
    changed: bool,
}

impl<'a> LargeOffsetVisitor<'a> {
    fn new(iceberg_schema: &'a Schema) -> Self {
        Self {
            iceberg_schema,
            field_stack: Vec::new(),
            changed: false,
        }
    }

    /// The widened type for this field, or None to keep it as-is. Only fields
    /// that carry a field id resolving to an Iceberg `string`/`binary` widen —
    /// id-less fields (variant sub-fields, foreign extras) are untouched.
    fn widened_type(&self, field: &FieldRef) -> Option<DataType> {
        let target = match field.data_type() {
            DataType::Utf8 => DataType::LargeUtf8,
            DataType::Binary => DataType::LargeBinary,
            _ => return None,
        };
        let iceberg_type = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id_str| id_str.parse::<i32>().ok())
            .and_then(|field_id| self.iceberg_schema.field_by_id(field_id))
            .map(|f| &*f.field_type)?;
        match (iceberg_type, &target) {
            (Type::Primitive(PrimitiveType::String), DataType::LargeUtf8)
            | (Type::Primitive(PrimitiveType::Binary), DataType::LargeBinary) => Some(target),
            _ => None,
        }
    }
}

impl ArrowSchemaVisitor for LargeOffsetVisitor<'_> {
    type T = Field;
    type U = ArrowSchema;

    fn before_field(&mut self, field: &FieldRef) -> Result<()> {
        self.field_stack.push(field.clone());
        Ok(())
    }

    fn after_field(&mut self, _field: &FieldRef) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn before_list_element(&mut self, field: &FieldRef) -> Result<()> {
        self.field_stack.push(field.clone());
        Ok(())
    }

    fn after_list_element(&mut self, _field: &FieldRef) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn before_map_key(&mut self, field: &FieldRef) -> Result<()> {
        self.field_stack.push(field.clone());
        Ok(())
    }

    fn after_map_key(&mut self, _field: &FieldRef) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn before_map_value(&mut self, field: &FieldRef) -> Result<()> {
        self.field_stack.push(field.clone());
        Ok(())
    }

    fn after_map_value(&mut self, _field: &FieldRef) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn schema(&mut self, schema: &ArrowSchema, values: Vec<Field>) -> Result<ArrowSchema> {
        Ok(ArrowSchema::new_with_metadata(
            values,
            schema.metadata().clone(),
        ))
    }

    fn r#struct(&mut self, _fields: &Fields, results: Vec<Field>) -> Result<Field> {
        let field_info = self
            .field_stack
            .last()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in struct"))?;
        Ok(Field::new(
            field_info.name(),
            DataType::Struct(Fields::from(results)),
            field_info.is_nullable(),
        )
        .with_metadata(field_info.metadata().clone()))
    }

    fn list(&mut self, list: &DataType, value: Field) -> Result<Field> {
        let field_info = self
            .field_stack
            .last()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in list"))?;
        let list_type = match list {
            DataType::List(_) => DataType::List(Arc::new(value)),
            DataType::LargeList(_) => DataType::LargeList(Arc::new(value)),
            DataType::FixedSizeList(_, size) => DataType::FixedSizeList(Arc::new(value), *size),
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Expected list type, got {list}"),
                ));
            }
        };
        Ok(
            Field::new(field_info.name(), list_type, field_info.is_nullable())
                .with_metadata(field_info.metadata().clone()),
        )
    }

    fn map(&mut self, map: &DataType, key_value: Field, value: Field) -> Result<Field> {
        let field_info = self
            .field_stack
            .last()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in map"))?;
        let sorted = match map {
            DataType::Map(_, sorted) => *sorted,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Expected map type, got {map}"),
                ));
            }
        };
        let struct_field = Field::new(
            DEFAULT_MAP_FIELD_NAME,
            DataType::Struct(Fields::from(vec![key_value, value])),
            false,
        );
        Ok(Field::new(
            field_info.name(),
            DataType::Map(Arc::new(struct_field), sorted),
            field_info.is_nullable(),
        )
        .with_metadata(field_info.metadata().clone()))
    }

    fn primitive(&mut self, _p: &DataType) -> Result<Field> {
        let field_info = self.field_stack.last().ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, "Field stack underflow in primitive")
        })?;

        if let Some(widened) = self.widened_type(field_info) {
            self.changed = true;
            Ok(
                Field::new(field_info.name(), widened, field_info.is_nullable())
                    .with_metadata(field_info.metadata().clone()),
            )
        } else {
            Ok(field_info.as_ref().clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::spec::{NestedField, Schema as IcebergSchema, StructType, Type};

    fn field_id_meta(id: i32) -> HashMap<String, String> {
        HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
    }

    fn iceberg_schema() -> IcebergSchema {
        IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Binary)).into(),
                NestedField::optional(3, "i", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    4,
                    "nested",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(5, "inner_s", Type::Primitive(PrimitiveType::String))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap()
    }

    #[test]
    fn widens_top_level_and_nested_string_binary() {
        let arrow = Arc::new(ArrowSchema::new(vec![
            Field::new("s", DataType::Utf8, true).with_metadata(field_id_meta(1)),
            Field::new("b", DataType::Binary, true).with_metadata(field_id_meta(2)),
            Field::new("i", DataType::Int32, true).with_metadata(field_id_meta(3)),
            Field::new(
                "nested",
                DataType::Struct(Fields::from(vec![
                    Field::new("inner_s", DataType::Utf8, true).with_metadata(field_id_meta(5)),
                ])),
                true,
            )
            .with_metadata(field_id_meta(4)),
        ]));
        let widened = widen_variable_length_types(&arrow, &iceberg_schema()).unwrap();
        assert_eq!(widened.field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(widened.field(1).data_type(), &DataType::LargeBinary);
        assert_eq!(widened.field(2).data_type(), &DataType::Int32);
        let DataType::Struct(inner) = widened.field(3).data_type() else {
            panic!("nested must stay a struct");
        };
        assert_eq!(inner[0].data_type(), &DataType::LargeUtf8);
        // Field ids and nullability survive the rewrite.
        assert_eq!(
            widened.field(0).metadata().get(PARQUET_FIELD_ID_META_KEY),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn no_change_returns_none() {
        let arrow = Arc::new(ArrowSchema::new(vec![
            Field::new("s", DataType::LargeUtf8, true).with_metadata(field_id_meta(1)),
            Field::new("i", DataType::Int32, true).with_metadata(field_id_meta(3)),
        ]));
        assert!(widen_variable_length_types(&arrow, &iceberg_schema()).is_none());
    }

    #[test]
    fn id_less_fields_untouched() {
        // Variant-shaped struct: Binary sub-fields with NO field ids must keep
        // their exact physical types (the variant kernels depend on them).
        let arrow = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "doc",
                DataType::Struct(Fields::from(vec![
                    Field::new("metadata", DataType::Binary, false),
                    Field::new("value", DataType::Binary, false),
                ])),
                true,
            )
            .with_metadata(field_id_meta(6)),
        ]));
        assert!(widen_variable_length_types(&arrow, &iceberg_schema()).is_none());
    }
}
