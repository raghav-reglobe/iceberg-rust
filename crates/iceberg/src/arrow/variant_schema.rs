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

//! Infer an Arrow [`DataType`] schema from a [`Variant`] value, and merge two
//! such schemas to their tightest common type.
//!
//! This is a direct port of Apache Spark's `SchemaOfVariant.schemaOf` +
//! `SchemaOfVariant.mergeSchema` (which delegates to
//! `JsonInferSchema.compatibleType(t1, t2, fallback = VariantType)`):
//! - <https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/variant/variantExpressions.scala>
//! - <https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/json/JsonInferSchema.scala>
//!
//! Differences from Spark, all intentional:
//! - Spark's variant `Type` is coarse (one `LONG`, one `DOUBLE`, one
//!   `DECIMAL`, ...). The Rust [`Variant`] enum is finer-grained (`Int8`..
//!   `Int64`, `Float`/`Double`, `Decimal4`/`8`/`16`, ...). We normalize the
//!   finer cases to Spark's coarse mapping below (all integers -> `Int64`,
//!   etc.).
//! - Spark uses a dedicated `VariantType` as the `compatibleType` fallback
//!   ("incompatible -> keep as variant/JSON"). Arrow has no such type, so we
//!   use [`DataType::Utf8`] as the fallback — the merge worker reads `Utf8`
//!   as "leave this field in the document as JSON, don't flatten it".

use arrow_schema::{DataType, Field, Fields, TimeUnit};
use parquet::variant::Variant;

/// Infer the Arrow [`DataType`] of a single [`Variant`] value.
///
/// Port of Spark's `SchemaOfVariant.schemaOf`:
/// - Object -> [`DataType::Struct`], one nullable field per entry, fields
///   sorted alphabetically by key (the variant spec already orders object
///   fields, but we sort defensively).
/// - Array -> [`DataType::List`] of `element` = the fold of
///   [`merge_variant_schemas`] over every element (seeded with the
///   null-sentinel [`DataType::Null`], exactly like Spark seeds with
///   `NullType`).
/// - Scalars -> Spark's scalar mapping, normalized for the finer Rust enum
///   (all integer widths -> `Int64`, etc.).
pub fn schema_of_variant(v: &Variant) -> DataType {
    match v {
        Variant::Object(obj) => {
            // (key, schema) for every entry. `VariantObject::iter` yields
            // entries already ordered by key per the spec; sort to be safe so
            // the StructType is deterministic + matches the "fields sorted by
            // name" invariant that `merge_variant_schemas` relies on.
            let mut entries: Vec<(String, DataType)> = obj
                .iter()
                .map(|(key, value)| (key.to_string(), schema_of_variant(&value)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let fields: Fields = entries
                .into_iter()
                .map(|(name, dt)| Field::new(name, dt, true))
                .collect();
            DataType::Struct(fields)
        }
        Variant::List(list) => {
            // Spark seeds the element type with NullType and folds the
            // per-element schema in; we use DataType::Null as the sentinel.
            let mut element = DataType::Null;
            for item in list.iter() {
                element = merge_variant_schemas(&element, &schema_of_variant(&item));
            }
            DataType::List(std::sync::Arc::new(Field::new("element", element, true)))
        }

        // Null sentinel (X + Null -> X is handled in merge_variant_schemas).
        Variant::Null => DataType::Null,

        // Boolean.
        Variant::BooleanTrue | Variant::BooleanFalse => DataType::Boolean,

        // Spark normalizes ALL variant integers to LONG.
        Variant::Int8(_) | Variant::Int16(_) | Variant::Int32(_) | Variant::Int64(_) => {
            DataType::Int64
        }

        // Floating point.
        Variant::Float(_) => DataType::Float32,
        Variant::Double(_) => DataType::Float64,

        // Decimal: precision is the variant's storage-width max
        // (Decimal4 -> 9, Decimal8 -> 18, Decimal16 -> 38), matching the
        // declared precision Spark reads off `v.getDecimal.precision`. scale
        // is the value's own scale.
        Variant::Decimal4(d) => {
            decimal_type(parquet::variant::VariantDecimal4::MAX_PRECISION, d.scale())
        }
        Variant::Decimal8(d) => {
            decimal_type(parquet::variant::VariantDecimal8::MAX_PRECISION, d.scale())
        }
        Variant::Decimal16(d) => {
            decimal_type(parquet::variant::VariantDecimal16::MAX_PRECISION, d.scale())
        }

        // Strings (the long-form `String` + the inline `ShortString` both map
        // to Utf8).
        Variant::String(_) | Variant::ShortString(_) => DataType::Utf8,
        Variant::Binary(_) => DataType::Binary,

        // Date / time / timestamp.
        Variant::Date(_) => DataType::Date32,
        Variant::Time(_) => DataType::Time64(TimeUnit::Microsecond),
        Variant::TimestampMicros(_) => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        Variant::TimestampNtzMicros(_) => DataType::Timestamp(TimeUnit::Microsecond, None),
        Variant::TimestampNanos(_) => DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        Variant::TimestampNtzNanos(_) => DataType::Timestamp(TimeUnit::Nanosecond, None),

        Variant::Uuid(_) => DataType::FixedSizeBinary(16),
    }
}

/// Build a `Decimal128(precision, scale)`, clamping precision to the Arrow
/// maximum (38).
fn decimal_type(precision: u8, scale: u8) -> DataType {
    let p = precision.min(arrow_schema::DECIMAL128_MAX_PRECISION);
    DataType::Decimal128(p, scale as i8)
}

/// Merge two inferred [`DataType`]s to their tightest common type.
///
/// Port of Spark's `JsonInferSchema.compatibleType(t1, t2, VariantType)`
/// (the same function `SchemaOfVariant.mergeSchema` calls). The first tier
/// mirrors `TypeCoercion.findTightestCommonType`; the structural / decimal /
/// fallback tier mirrors `compatibleType`'s explicit `match`.
pub fn merge_variant_schemas(a: &DataType, b: &DataType) -> DataType {
    use DataType::*;

    match (a, b) {
        // findTightestCommonType: equal, and NullType identity.
        (x, y) if x == y => x.clone(),
        (Null, y) => y.clone(),
        (x, Null) => x.clone(),

        // Integral that a Decimal can fully hold -> the Decimal.
        // (Our only integral type is Int64; Decimal128(p,s) can hold an
        // Int64 iff its integer part `p - s` has at least 19 digits.)
        (Int64, Decimal128(p, s)) if (*p as i16 - *s as i16) >= 19 => b.clone(),
        (Decimal128(p, s), Int64) if (*p as i16 - *s as i16) >= 19 => a.clone(),

        // Promote two non-decimal numerics to the higher of the two by Spark's
        // numericPrecedence = [Byte, Short, Int, Long, Float, Double]. We only
        // ever produce Int64 / Float32 / Float64, so: Long < Float < Double.
        (x, y) if is_plain_numeric(x) && is_plain_numeric(y) => numeric_max(x, y),

        // Datetime widening: TimestampNtz + Timestamp(tz) -> Timestamp(tz).
        // (Spark's findWiderDateTimeType; only the NTZ<->LTZ pair widens.)
        (Timestamp(u1, None), Timestamp(u2, Some(_)))
        | (Timestamp(u1, Some(_)), Timestamp(u2, None))
            if u1 == u2 =>
        {
            Timestamp(u1.clone(), Some("UTC".into()))
        }

        // --- compatibleType explicit match below ---

        // Double has a larger range than any fixed decimal.
        (Float64, Decimal128(_, _)) | (Decimal128(_, _), Float64) => Float64,
        // SchemaOfVariant-only branch: Float + Decimal -> Double.
        (Float32, Decimal128(_, _)) | (Decimal128(_, _), Float32) => Float64,

        // Decimal + Decimal: scale = max(s1,s2); range = max(p1-s1, p2-s2);
        // if range+scale > 38 -> Double, else Decimal(range+scale, scale).
        (Decimal128(p1, s1), Decimal128(p2, s2)) => {
            let scale = (*s1).max(*s2);
            let range = (*p1 as i16 - *s1 as i16).max(*p2 as i16 - *s2 as i16);
            if range + scale as i16 > 38 {
                Float64
            } else {
                Decimal128((range + scale as i16) as u8, scale)
            }
        }

        // Struct + Struct: union of field names; same-named fields merged
        // recursively; result fields sorted by name (both inputs are already
        // sorted, so a sorted-merge produces a sorted result).
        (Struct(f1), Struct(f2)) => Struct(merge_struct_fields(f1, f2)),

        // List + List: merge element types; containsNull is always true here.
        (List(e1), List(e2)) => List(std::sync::Arc::new(Field::new(
            "element",
            merge_variant_schemas(e1.data_type(), e2.data_type()),
            true,
        ))),

        // Integral not fully held by the Decimal -> widen the integral to its
        // own DecimalType and re-merge. Int64 -> Decimal128(19, 0).
        (Int64, Decimal128(_, _)) => merge_variant_schemas(&Decimal128(19, 0), b),
        (Decimal128(_, _), Int64) => merge_variant_schemas(a, &Decimal128(19, 0)),

        // Spark uses VariantType as the fallback; we use Utf8 (keep as
        // JSON/string — the merge worker treats anything non-scalar/ambiguous
        // as "stay in the doc").
        _ => Utf8,
    }
}

/// A non-decimal numeric type that participates in Spark's `numericPrecedence`
/// promotion. We only ever produce Int64 / Float32 / Float64.
fn is_plain_numeric(t: &DataType) -> bool {
    matches!(t, DataType::Int64 | DataType::Float32 | DataType::Float64)
}

/// The higher of two plain numeric types by Spark's `numericPrecedence`
/// ([Byte, Short, Int, Long, Float, Double]): Int64 < Float32 < Float64.
fn numeric_max(a: &DataType, b: &DataType) -> DataType {
    fn rank(t: &DataType) -> u8 {
        match t {
            DataType::Int64 => 3,   // Long
            DataType::Float32 => 4, // Float
            DataType::Float64 => 5, // Double
            _ => 0,
        }
    }
    if rank(a) >= rank(b) {
        a.clone()
    } else {
        b.clone()
    }
}

/// Sorted-merge two name-sorted field lists, merging same-named fields
/// recursively. Mirrors `compatibleType`'s `(StructType, StructType)` branch.
fn merge_struct_fields(f1: &Fields, f2: &Fields) -> Fields {
    let mut out: Vec<Field> = Vec::with_capacity(f1.len() + f2.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < f1.len() && j < f2.len() {
        let a = &f1[i];
        let b = &f2[j];
        match a.name().cmp(b.name()) {
            std::cmp::Ordering::Equal => {
                let dt = merge_variant_schemas(a.data_type(), b.data_type());
                out.push(Field::new(a.name(), dt, true));
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                out.push(Field::new(a.name(), a.data_type().clone(), true));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(Field::new(b.name(), b.data_type().clone(), true));
                j += 1;
            }
        }
    }
    while i < f1.len() {
        let a = &f1[i];
        out.push(Field::new(a.name(), a.data_type().clone(), true));
        i += 1;
    }
    while j < f2.len() {
        let b = &f2[j];
        out.push(Field::new(b.name(), b.data_type().clone(), true));
        j += 1;
    }
    Fields::from(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Fields, TimeUnit};
    use parquet::variant::{Variant, VariantBuilder};

    use super::{merge_variant_schemas, schema_of_variant};

    /// Build a [`Variant`] from a closure that populates a [`VariantBuilder`],
    /// returning owned (metadata, value) buffers + reparsing into a Variant.
    fn build(f: impl FnOnce(&mut VariantBuilder)) -> (Vec<u8>, Vec<u8>) {
        let mut b = VariantBuilder::new();
        f(&mut b);
        b.finish()
    }

    #[test]
    fn scalar_long() {
        let (m, v) = build(|b| b.append_value(7i64));
        let var = Variant::try_new(&m, &v).unwrap();
        assert_eq!(schema_of_variant(&var), DataType::Int64);
    }

    #[test]
    fn scalar_int_widths_all_long() {
        for (m, v) in [
            build(|b| b.append_value(7i8)),
            build(|b| b.append_value(7i16)),
            build(|b| b.append_value(7i32)),
            build(|b| b.append_value(7i64)),
        ] {
            let var = Variant::try_new(&m, &v).unwrap();
            assert_eq!(schema_of_variant(&var), DataType::Int64);
        }
    }

    #[test]
    fn scalar_string_and_double() {
        let (m, v) = build(|b| b.append_value("hello"));
        assert_eq!(
            schema_of_variant(&Variant::try_new(&m, &v).unwrap()),
            DataType::Utf8
        );
        let (m, v) = build(|b| b.append_value(2.75f64));
        assert_eq!(
            schema_of_variant(&Variant::try_new(&m, &v).unwrap()),
            DataType::Float64
        );
    }

    #[test]
    fn object_sorted_fields() {
        // {"b":"x", "a":1} -> Struct[a:Int64, b:Utf8] (sorted by key).
        let (m, v) = build(|b| {
            let mut obj = b.new_object();
            obj.insert("b", "x");
            obj.insert("a", 1i64);
            obj.finish();
        });
        let var = Variant::try_new(&m, &v).unwrap();
        let expected = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        assert_eq!(schema_of_variant(&var), expected);
    }

    #[test]
    fn nested_object() {
        // {"outer": {"n": 1, "s": "x"}}
        let (m, v) = build(|b| {
            let mut obj = b.new_object();
            {
                let mut inner = obj.new_object("outer");
                inner.insert("s", "x");
                inner.insert("n", 1i64);
                inner.finish();
            }
            obj.finish();
        });
        let var = Variant::try_new(&m, &v).unwrap();
        let inner = DataType::Struct(Fields::from(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let expected = DataType::Struct(Fields::from(vec![Field::new("outer", inner, true)]));
        assert_eq!(schema_of_variant(&var), expected);
    }

    #[test]
    fn array_mixed_int_double() {
        // [1, 2.5] -> List(Float64): Long folded with Double -> Double.
        let (m, v) = build(|b| {
            let mut list = b.new_list();
            list.append_value(1i64);
            list.append_value(2.5f64);
            list.finish();
        });
        let var = Variant::try_new(&m, &v).unwrap();
        let expected = DataType::List(Arc::new(Field::new("element", DataType::Float64, true)));
        assert_eq!(schema_of_variant(&var), expected);
    }

    #[test]
    fn merge_struct_union() {
        // Struct[a:Int64] + Struct[b:Utf8] -> Struct[a:Int64, b:Utf8].
        let a = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        let b = DataType::Struct(Fields::from(vec![Field::new("b", DataType::Utf8, true)]));
        let merged = merge_variant_schemas(&a, &b);
        let expected = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        assert_eq!(merged, expected);
    }

    #[test]
    fn merge_struct_same_field_widens() {
        // Struct[a:Int64] + Struct[a:Float64] -> Struct[a:Float64].
        let a = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        let b = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Float64, true)]));
        let merged = merge_variant_schemas(&a, &b);
        let expected =
            DataType::Struct(Fields::from(vec![Field::new("a", DataType::Float64, true)]));
        assert_eq!(merged, expected);
    }

    #[test]
    fn merge_numeric_and_null() {
        assert_eq!(
            merge_variant_schemas(&DataType::Int64, &DataType::Float64),
            DataType::Float64
        );
        // Int64 + Float32 -> Float32 (Spark numericPrecedence: Float > Long).
        assert_eq!(
            merge_variant_schemas(&DataType::Int64, &DataType::Float32),
            DataType::Float32
        );
        // X + Null -> X (both directions).
        assert_eq!(
            merge_variant_schemas(&DataType::Int64, &DataType::Null),
            DataType::Int64
        );
        assert_eq!(
            merge_variant_schemas(&DataType::Null, &DataType::Utf8),
            DataType::Utf8
        );
    }

    #[test]
    fn merge_incompatible_falls_back_to_utf8() {
        // Struct + Utf8 -> Utf8 (Spark's VariantType fallback == our Utf8).
        let s = DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        assert_eq!(merge_variant_schemas(&s, &DataType::Utf8), DataType::Utf8);
        // Timestamp NTZ + Timestamp UTC (same unit) -> Timestamp UTC.
        let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
        let utc = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        assert_eq!(merge_variant_schemas(&ntz, &utc), utc);
    }
}
