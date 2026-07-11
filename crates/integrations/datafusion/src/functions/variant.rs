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

//! Scalar UDFs over Iceberg V3 / Parquet [`Variant`] columns, wrapping the
//! `parquet-variant-compute` kernels (re-exported as [`parquet::variant`]).
//!
//! DataFusion has no built-in variant functions yet, so tables with variant
//! columns can be scanned but their values cannot be addressed from SQL.
//! These UDFs close that gap for the common flatten/extract patterns:
//!
//! | SQL | Returns |
//! |---|---|
//! | `variant_get(v, path)` | variant (canonical `{metadata, value}` struct) |
//! | `variant_get_string(v, path)` | `Utf8` |
//! | `variant_get_bigint(v, path)` | `Int64` |
//! | `variant_get_double(v, path)` | `Float64` |
//! | `variant_get_boolean(v, path)` | `Boolean` |
//! | `variant_get_timestamp(v, path)` | `Timestamp(µs, UTC)` |
//! | `variant_to_json(v)` | `Utf8` (JSON text) |
//! | `json_to_variant(s)` | variant |
//!
//! Semantics:
//! - `path` must be a non-null string literal using the `parquet-variant`
//!   path syntax (`a.b`, `a[0].b`, `['a.b']`), optionally prefixed with the
//!   conventional `$.` / `$` root marker used by other engines' JSON/variant
//!   accessors.
//! - Extraction is *safe*: a missing path or a value that cannot be cast to
//!   the requested type yields `NULL`, never an error — the natural fit for
//!   schemaless documents where field types drift row to row.
//! - `variant_get` output is normalized to the canonical unshredded layout
//!   (`Struct<metadata: BinaryView, value: BinaryView>`) so its type is
//!   stable regardless of the input's shredding state, and calls chain.
//!
//! [`Variant`]: parquet::variant::Variant

use std::sync::{Arc, OnceLock};

use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::compute::{CastOptions, cast};
use datafusion::arrow::datatypes::{DataType, Field, Fields, TimeUnit};
use datafusion::common::{Result as DFResult, ScalarValue, exec_datafusion_err, exec_err};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;
use parquet::variant::{
    GetOptions, VariantArray, VariantArrayBuilder, VariantPath, json_to_variant, variant_get,
    variant_to_json,
};

/// All variant scalar UDFs provided by this module.
pub fn all_variant_functions() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::from(VariantGetUdf::variant()),
        ScalarUDF::from(VariantGetUdf::typed("variant_get_string", DataType::Utf8)),
        ScalarUDF::from(VariantGetUdf::typed("variant_get_bigint", DataType::Int64)),
        ScalarUDF::from(VariantGetUdf::typed(
            "variant_get_double",
            DataType::Float64,
        )),
        ScalarUDF::from(VariantGetUdf::typed(
            "variant_get_boolean",
            DataType::Boolean,
        )),
        ScalarUDF::from(VariantGetUdf::typed(
            "variant_get_timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        )),
        ScalarUDF::from(VariantToJsonUdf::new()),
        ScalarUDF::from(JsonToVariantUdf::new()),
    ]
}

/// Register all variant scalar UDFs on the given [`SessionContext`].
pub fn register_variant_functions(ctx: &SessionContext) {
    for udf in all_variant_functions() {
        ctx.register_udf(udf);
    }
}

/// The canonical (unshredded) arrow type of a variant column, as produced by
/// [`VariantArrayBuilder`]: `Struct<metadata: BinaryView, value: BinaryView>`.
fn canonical_variant_type() -> &'static DataType {
    static TYPE: OnceLock<DataType> = OnceLock::new();
    TYPE.get_or_init(|| {
        ArrayRef::from(VariantArrayBuilder::new(0).build())
            .data_type()
            .clone()
    })
}

/// Strip the conventional `$` / `$.` root marker; the remainder is
/// `parquet-variant` path syntax.
fn strip_root(raw: &str) -> &str {
    if raw == "$" {
        ""
    } else {
        raw.strip_prefix("$.").unwrap_or(raw)
    }
}

/// The `path` argument must be a non-null string literal — it selects the
/// output shape, so a per-row path is not meaningful.
fn path_literal(cv: &ColumnarValue, fn_name: &str) -> DFResult<String> {
    match cv {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) => Ok(s.clone()),
        _ => exec_err!("{fn_name}: path must be a non-null string literal"),
    }
}

fn input_array(cv: &ColumnarValue, number_rows: usize) -> DFResult<ArrayRef> {
    match cv {
        ColumnarValue::Array(a) => Ok(Arc::clone(a)),
        ColumnarValue::Scalar(s) => Ok(s.to_array_of_size(number_rows)?),
    }
}

/// Re-emit through [`VariantArrayBuilder`] so the output type is exactly
/// [`canonical_variant_type`] regardless of the kernel's output layout
/// (e.g. a shredded pass-through or a `Binary` metadata column).
fn to_canonical(arr: ArrayRef) -> DFResult<ArrayRef> {
    if arr.data_type() == canonical_variant_type() {
        return Ok(arr);
    }
    let variant = VariantArray::try_new(&arr).map_err(DataFusionError::from)?;
    let mut builder = VariantArrayBuilder::new(variant.len());
    for i in 0..variant.len() {
        if variant.is_null(i) {
            builder.append_null();
        } else {
            builder.append_variant(variant.value(i));
        }
    }
    Ok(ArrayRef::from(builder.build()))
}

/// `variant_get*` family: extract `path` from a variant column, either as a
/// variant (`as_type: None`) or cast to a fixed arrow type.
#[derive(Debug, PartialEq, Eq, Hash)]
struct VariantGetUdf {
    name: &'static str,
    /// `None` → return a (canonical) variant; `Some` → cast to this type.
    as_type: Option<DataType>,
    signature: Signature,
}

impl VariantGetUdf {
    fn variant() -> Self {
        Self {
            name: "variant_get",
            as_type: None,
            signature: Signature::any(2, Volatility::Immutable),
        }
    }

    fn typed(name: &'static str, as_type: DataType) -> Self {
        Self {
            name,
            as_type: Some(as_type),
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantGetUdf {
    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(self
            .as_type
            .clone()
            .unwrap_or_else(|| canonical_variant_type().clone()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let raw_path = path_literal(&args.args[1], self.name)?;
        let stripped = strip_root(&raw_path);
        let path = VariantPath::try_from(stripped)
            .map_err(|e| exec_datafusion_err!("{}: invalid path {raw_path:?}: {e}", self.name))?;

        let input = input_array(&args.args[0], args.number_rows)?;
        let options = GetOptions {
            path,
            as_type: self
                .as_type
                .as_ref()
                .map(|t| Arc::new(Field::new(self.name, t.clone(), true))),
            // safe: true — cast failures (schemaless type drift) become NULL.
            cast_options: CastOptions::default(),
        };
        let out = variant_get(&input, options).map_err(DataFusionError::from)?;
        let out = match self.as_type {
            Some(_) => out,
            None => to_canonical(out)?,
        };
        Ok(ColumnarValue::Array(out))
    }
}

/// `variant_to_json(v)` → JSON text.
#[derive(Debug, PartialEq, Eq, Hash)]
struct VariantToJsonUdf {
    signature: Signature,
}

impl VariantToJsonUdf {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantToJsonUdf {
    fn name(&self) -> &str {
        "variant_to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        // The 58.x `variant_to_json` kernel still expects the pre-BinaryView
        // layout `Struct<metadata: Binary, value: Binary>`, while the rest of
        // the kernel family (and this module) produce `BinaryView` children —
        // normalize, then cast the children down to `Binary` for the kernel.
        let input = to_canonical(input_array(&args.args[0], args.number_rows)?)?;
        let DataType::Struct(fields) = input.data_type() else {
            return exec_err!("variant_to_json: expected a variant struct input");
        };
        let binary_fields: Fields = fields
            .iter()
            .map(|f| Arc::new(f.as_ref().clone().with_data_type(DataType::Binary)))
            .collect();
        let casted = cast(&input, &DataType::Struct(binary_fields))?;
        let json = variant_to_json(&casted).map_err(DataFusionError::from)?;
        Ok(ColumnarValue::Array(Arc::new(json)))
    }
}

/// `json_to_variant(s)` → variant.
#[derive(Debug, PartialEq, Eq, Hash)]
struct JsonToVariantUdf {
    signature: Signature,
}

impl JsonToVariantUdf {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for JsonToVariantUdf {
    fn name(&self) -> &str {
        "json_to_variant"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(canonical_variant_type().clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let input = input_array(&args.args[0], args.number_rows)?;
        let variant = json_to_variant(&input).map_err(DataFusionError::from)?;
        to_canonical(ArrayRef::from(variant)).map(ColumnarValue::Array)
    }
}
