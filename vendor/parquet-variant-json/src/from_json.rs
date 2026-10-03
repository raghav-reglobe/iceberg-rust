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

//! Module for parsing JSON strings as Variant

use std::collections::BTreeMap;

use arrow_schema::ArrowError;
use parquet_variant::{
    ObjectFieldBuilder, Variant, VariantBuilderExt, VariantDecimal4, VariantDecimal8,
    VariantDecimal16,
};
use serde_json::value::RawValue;
use serde_json::{Number, Value};

/// Converts a JSON string to Variant using a [`VariantBuilderExt`], such as
/// [`VariantBuilder`].
///
/// The resulting `value` and `metadata` buffers can be
/// extracted using `builder.finish()`
///
/// # Arguments
/// * `json` - The JSON string to parse as Variant.
///
/// # Returns
///
/// * `Ok(())` if successful
/// * `Err` with error details if the conversion fails
///
/// [`VariantBuilder`]: parquet_variant::VariantBuilder
///
/// ```rust
/// # use parquet_variant::VariantBuilder;
/// # use parquet_variant_json::{JsonToVariant, VariantToJson};
///
/// let mut variant_builder = VariantBuilder::new();
/// let person_string = "{\"name\":\"Alice\", \"age\":30, ".to_string()
/// + "\"email\":\"alice@example.com\", \"is_active\": true, \"score\": 95.7,"
/// + "\"additional_info\": null}";
/// variant_builder.append_json(&person_string)?;
///
/// let (metadata, value) = variant_builder.finish();
///
/// let variant = parquet_variant::Variant::try_new(&metadata, &value)?;
///
/// let json_result = variant.to_json_string()?;
/// let json_value = variant.to_json_value()?;
///
/// let mut buffer = Vec::new();
/// variant.to_json(&mut buffer)?;
/// let buffer_result = String::from_utf8(buffer)?;
/// assert_eq!(json_result, "{\"additional_info\":null,\"age\":30,".to_string() +
/// "\"email\":\"alice@example.com\",\"is_active\":true,\"name\":\"Alice\",\"score\":95.7}");
/// assert_eq!(json_result, buffer_result);
/// assert_eq!(json_result, serde_json::to_string(&json_value)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait JsonToVariant {
    /// Create a Variant from a JSON string
    fn append_json(&mut self, json: &str) -> Result<(), ArrowError>;
}

impl<T: VariantBuilderExt> JsonToVariant for T {
    fn append_json(&mut self, json: &str) -> Result<(), ArrowError> {
        // `RawValue` validates the whole document and keeps every value's
        // TEXT, so a number is typed from the digits it was written with
        // (`12.30` is a decimal of scale 2, not the double 12.3). A
        // `serde_json::Value` has already folded a non-integer into an f64
        // by the time it is visible; see [`append_json`].
        let raw: &RawValue = serde_json::from_str(json).map_err(json_format_error)?;
        append_raw(raw, self)
    }
}

fn json_format_error(e: serde_json::Error) -> ArrowError {
    ArrowError::InvalidArgumentError(format!("JSON format error: {e}"))
}

/// Append one JSON value, given as its raw text, to `builder`. Objects and
/// arrays are split one level at a time (each level re-validates its own
/// subtree, so the cost is the document size times its depth); the scalar
/// kinds are told apart by their first byte, which JSON's grammar fixes.
fn append_raw(raw: &RawValue, builder: &mut impl VariantBuilderExt) -> Result<(), ArrowError> {
    let text = raw.get();
    match text.as_bytes().first() {
        Some(b'{') => {
            // Keys sorted; a repeated key keeps its LAST value — the same
            // object the `serde_json::Map`-based path produced.
            let fields: BTreeMap<String, &RawValue> =
                serde_json::from_str(text).map_err(json_format_error)?;
            let mut obj_builder = builder.try_new_object()?;
            for (key, value) in &fields {
                let mut field_builder = ObjectFieldBuilder::new(key, &mut obj_builder);
                append_raw(value, &mut field_builder)?;
            }
            obj_builder.finish();
        }
        Some(b'[') => {
            let items: Vec<&RawValue> = serde_json::from_str(text).map_err(json_format_error)?;
            let mut list_builder = builder.try_new_list()?;
            for item in items {
                append_raw(item, &mut list_builder)?;
            }
            list_builder.finish();
        }
        Some(b'"') => {
            let s: String = serde_json::from_str(text).map_err(json_format_error)?;
            builder.append_value(s.as_str());
        }
        Some(b't') => builder.append_value(true),
        Some(b'f') => builder.append_value(false),
        Some(b'n') => builder.append_value(Variant::Null),
        Some(_) => builder.append_value(variant_from_number_text(text)?),
        None => {
            return Err(ArrowError::InvalidArgumentError(
                "JSON format error: empty value".to_string(),
            ));
        }
    }
    Ok(())
}

/// Type a JSON number from its TEXT, as Spark's `VariantBuilder` does:
///
/// - an integer that fits `i64` → the narrowest of int8 / int16 / int32 / int64;
/// - otherwise, when the text is plain decimal notation (an optional `-`,
///   digits, at most one `.`; no exponent) with at most 38 significant
///   digits and a scale of at most 38 → decimal4 / decimal8 / decimal16, the
///   narrowest width whose precision AND scale limits both hold, keeping the
///   text's scale (`12.30` → decimal4 unscaled 1230 scale 2;
///   `9223372036854775808` → decimal16 scale 0);
/// - otherwise → double (`1e3`, `1.5E-7`, more than 38 digits).
///
/// `-0.0` has no decimal form distinct from `0.0` and becomes decimal4(0, 1):
/// value-equal, sign dropped — as in Spark.
fn variant_from_number_text(text: &str) -> Result<Variant<'static, 'static>, ArrowError> {
    let integral = !text.bytes().any(|b| matches!(b, b'.' | b'e' | b'E'));
    if integral {
        if let Ok(i) = text.parse::<i64>() {
            return Ok(narrowest_int(i));
        }
    }
    if let Some(decimal) = variant_decimal_from_text(text)? {
        return Ok(decimal);
    }
    text.parse::<f64>()
        .map(Variant::from)
        .map_err(|_| ArrowError::InvalidArgumentError(format!("Failed to parse {text} as number")))
}

fn narrowest_int(i: i64) -> Variant<'static, 'static> {
    if let Ok(v) = i8::try_from(i) {
        v.into()
    } else if let Ok(v) = i16::try_from(i) {
        v.into()
    } else if let Ok(v) = i32::try_from(i) {
        v.into()
    } else {
        i.into()
    }
}

/// `Ok(Some(_))` when `text` is plain decimal notation within the 38-digit
/// limits; `Ok(None)` when it is not a decimal candidate (an exponent, or
/// too many digits) and the caller falls back to double.
fn variant_decimal_from_text(text: &str) -> Result<Option<Variant<'static, 'static>>, ArrowError> {
    if !text.bytes().all(|b| b == b'-' || b == b'.' || b.is_ascii_digit()) {
        return Ok(None);
    }
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let (int_digits, frac_digits) = body.split_once('.').unwrap_or((body, ""));
    // JSON's grammar already guarantees this shape; the check keeps the
    // function total for any caller.
    if int_digits.is_empty()
        || body.contains('-')
        || frac_digits.contains('.')
        || (body.contains('.') && frac_digits.is_empty())
    {
        return Ok(None);
    }
    let scale = frac_digits.len();
    let digits = [int_digits, frac_digits].concat();
    let significant = digits.trim_start_matches('0');
    // `BigDecimal::precision()`: digits of the unscaled value, at least 1.
    let precision = significant.len().max(1);
    if scale > VariantDecimal16::MAX_PRECISION as usize
        || precision > VariantDecimal16::MAX_PRECISION as usize
    {
        return Ok(None);
    }
    let mut unscaled: i128 = if significant.is_empty() {
        0
    } else {
        significant.parse().map_err(|_| {
            ArrowError::InvalidArgumentError(format!("Failed to parse {text} as number"))
        })?
    };
    if negative {
        unscaled = -unscaled;
    }
    let scale = scale as u8;
    let d4 = VariantDecimal4::MAX_PRECISION as usize;
    let d8 = VariantDecimal8::MAX_PRECISION as usize;
    let variant = if scale as usize <= d4 && precision <= d4 {
        VariantDecimal4::try_new(unscaled as i32, scale)?.into()
    } else if scale as usize <= d8 && precision <= d8 {
        VariantDecimal8::try_new(unscaled as i64, scale)?.into()
    } else {
        VariantDecimal16::try_new(unscaled, scale)?.into()
    };
    Ok(Some(variant))
}

/// Type a JSON number that has already been parsed into a
/// [`serde_json::Number`]. The number's text is gone at this point, so a
/// non-integer can only become a double; prefer [`JsonToVariant::append_json`]
/// on the JSON text, which types decimals.
fn variant_from_number<'m, 'v>(n: &Number) -> Result<Variant<'m, 'v>, ArrowError> {
    if let Some(i) = n.as_i64() {
        Ok(narrowest_int(i))
    } else {
        match n.as_f64() {
            Some(f) => Ok(f.into()),
            None => Err(ArrowError::InvalidArgumentError(format!(
                "Failed to parse {n} as number",
            ))),
        }
    }
}

/// Append an already-parsed [`serde_json::Value`] to `builder`. Numbers are
/// typed by [`variant_from_number`] — see its note on decimals.
pub fn append_json(json: &Value, builder: &mut impl VariantBuilderExt) -> Result<(), ArrowError> {
    match json {
        Value::Null => builder.append_value(Variant::Null),
        Value::Bool(b) => builder.append_value(*b),
        Value::Number(n) => {
            builder.append_value(variant_from_number(n)?);
        }
        Value::String(s) => builder.append_value(s.as_str()),
        Value::Array(arr) => {
            let mut list_builder = builder.try_new_list()?;
            for val in arr {
                append_json(val, &mut list_builder)?;
            }
            list_builder.finish();
        }
        Value::Object(obj) => {
            let mut obj_builder = builder.try_new_object()?;
            for (key, value) in obj.iter() {
                let mut field_builder = ObjectFieldBuilder::new(key, &mut obj_builder);
                append_json(value, &mut field_builder)?;
            }
            obj_builder.finish();
        }
    };
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::VariantToJson;
    use arrow_schema::ArrowError;
    use parquet_variant::{
        ShortString, Variant, VariantBuilder, VariantDecimal4, VariantDecimal8, VariantDecimal16,
    };

    struct JsonToVariantTest<'a> {
        json: &'a str,
        expected: Variant<'a, 'a>,
    }

    impl JsonToVariantTest<'_> {
        fn run(self) -> Result<(), ArrowError> {
            let mut variant_builder = VariantBuilder::new();
            variant_builder.append_json(self.json)?;
            let (metadata, value) = variant_builder.finish();
            let variant = Variant::try_new(&metadata, &value)?;
            assert_eq!(variant, self.expected);
            Ok(())
        }
    }

    #[test]
    fn test_json_to_variant_null() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "null",
            expected: Variant::Null,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_boolean_true() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "true",
            expected: Variant::BooleanTrue,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_boolean_false() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "false",
            expected: Variant::BooleanFalse,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_int8_positive() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "  127 ",
            expected: Variant::Int8(127),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_int8_negative() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "  -128 ",
            expected: Variant::Int8(-128),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_int16() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "  27134  ",
            expected: Variant::Int16(27134),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_int32() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: " -32767431  ",
            expected: Variant::Int32(-32767431),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_int64() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "92842754201389",
            expected: Variant::Int64(92842754201389),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal4_basic() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "1.23",
            expected: Variant::from(VariantDecimal4::try_new(123, 2)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal4_large_positive() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "99999999.9",
            expected: Variant::from(VariantDecimal4::try_new(999999999, 1)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal4_large_negative() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "-99999999.9",
            expected: Variant::from(VariantDecimal4::try_new(-999999999, 1)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal4_small_positive() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "0.999999999",
            expected: Variant::from(VariantDecimal4::try_new(999999999, 9)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal4_tiny_positive() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "0.000000001",
            expected: Variant::from(VariantDecimal4::try_new(1, 9)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal4_small_negative() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "-0.999999999",
            expected: Variant::from(VariantDecimal4::try_new(-999999999, 9)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal8_positive() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "999999999.0",
            expected: Variant::from(VariantDecimal8::try_new(9999999990, 1)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal8_negative() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "-999999999.0",
            expected: Variant::from(VariantDecimal8::try_new(-9999999990, 1)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal8_high_precision() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "0.999999999999999999",
            expected: Variant::from(VariantDecimal8::try_new(999999999999999999, 18)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal8_large_with_scale() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "9999999999999999.99",
            expected: Variant::from(VariantDecimal8::try_new(999999999999999999, 2)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal8_large_negative_with_scale() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "-9999999999999999.99",
            expected: Variant::from(VariantDecimal8::try_new(-999999999999999999, 2)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal16_large_integer() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "9999999999999999999", // integer larger than i64
            expected: Variant::from(VariantDecimal16::try_new(9999999999999999999, 0)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal16_high_precision() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "0.9999999999999999999",
            expected: Variant::from(VariantDecimal16::try_new(9999999999999999999, 19)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal16_max_value() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "79228162514264337593543950335", // 2 ^ 96 - 1
            expected: Variant::from(VariantDecimal16::try_new(79228162514264337593543950335, 0)?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_decimal16_max_scale() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "7.9228162514264337593543950335",
            expected: Variant::from(VariantDecimal16::try_new(
                79228162514264337593543950335,
                28,
            )?),
        }
        .run()
    }

    /// 29 significant digits at scale 29: within decimal16's 38-digit
    /// limits, so a decimal (the released kernel, which never typed
    /// decimals, expected a double here).
    #[test]
    fn test_json_to_variant_decimal16_scale_29() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "0.79228162514264337593543950335",
            expected: Variant::from(VariantDecimal16::try_new(
                79228162514264337593543950335,
                29,
            )?),
        }
        .run()
    }

    /// The text's scale survives: trailing zeros are digits of the decimal.
    #[test]
    fn test_json_to_variant_decimal_keeps_text_scale() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "12.30",
            expected: Variant::from(VariantDecimal4::try_new(1230, 2)?),
        }
        .run()?;
        JsonToVariantTest {
            json: "-12.30",
            expected: Variant::from(VariantDecimal4::try_new(-1230, 2)?),
        }
        .run()?;
        JsonToVariantTest {
            json: "100.00",
            expected: Variant::from(VariantDecimal4::try_new(10000, 2)?),
        }
        .run()?;
        JsonToVariantTest {
            json: "0.68",
            expected: Variant::from(VariantDecimal4::try_new(68, 2)?),
        }
        .run()?;
        JsonToVariantTest {
            json: "0.000000000000000000001",
            expected: Variant::from(VariantDecimal16::try_new(1, 21)?),
        }
        .run()
    }

    /// Exponent notation is never a decimal candidate.
    #[test]
    fn test_json_to_variant_exponent_is_double() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "1e3",
            expected: Variant::Double(1000.0),
        }
        .run()?;
        JsonToVariantTest {
            json: "1.5E-7",
            expected: Variant::Double(1.5e-7),
        }
        .run()?;
        JsonToVariantTest {
            json: "1.7976931348623157e308",
            expected: Variant::Double(f64::MAX),
        }
        .run()
    }

    /// Beyond 38 significant digits only a double can hold the value.
    #[test]
    fn test_json_to_variant_beyond_38_digits_is_double() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "1234567890123456789012345678901234567890",
            expected: Variant::Double(1.2345678901234568e39),
        }
        .run()?;
        JsonToVariantTest {
            json: "1.123456789012345678901234567890123456789",
            expected: Variant::Double(1.1234567890123457),
        }
        .run()?;
        // 38 nines: the widest decimal16.
        JsonToVariantTest {
            json: "99999999999999999999999999999999999999",
            expected: Variant::from(VariantDecimal16::try_new(
                99999999999999999999999999999999999999,
                0,
            )?),
        }
        .run()
    }

    /// An integer past int64 is a decimal16 of scale 0, not a double.
    #[test]
    fn test_json_to_variant_integer_past_int64_is_decimal16() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "9223372036854775808",
            expected: Variant::from(VariantDecimal16::try_new(9223372036854775808, 0)?),
        }
        .run()?;
        JsonToVariantTest {
            json: "-9223372036854775809",
            expected: Variant::from(VariantDecimal16::try_new(-9223372036854775809, 0)?),
        }
        .run()?;
        // int64's own extremes stay int64.
        JsonToVariantTest {
            json: "9223372036854775807",
            expected: Variant::Int64(i64::MAX),
        }
        .run()?;
        JsonToVariantTest {
            json: "-9223372036854775808",
            expected: Variant::Int64(i64::MIN),
        }
        .run()
    }

    /// `-0` is the integer 0; `-0.0` is decimal4(0, 1) — no sign, value-equal.
    #[test]
    fn test_json_to_variant_negative_zero() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "-0",
            expected: Variant::Int8(0),
        }
        .run()?;
        JsonToVariantTest {
            json: "-0.0",
            expected: Variant::from(VariantDecimal4::try_new(0, 1)?),
        }
        .run()
    }

    /// Numbers nested in objects and arrays are typed from their text too;
    /// surrounding whitespace is not part of any value.
    #[test]
    fn test_json_to_variant_nested_numbers_keep_scale() -> Result<(), ArrowError> {
        let mut builder = VariantBuilder::new();
        builder.append_json(" { \"a\" : [ 1.50 , 2 , -0.0 ] , \"b\" : { \"c\" : 12.30 } } ")?;
        let (metadata, value) = builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;
        let obj = variant.as_object().expect("object");
        let a = obj.get("a").expect("a");
        let list = a.as_list().expect("list");
        assert_eq!(list.get(0), Some(Variant::from(VariantDecimal4::try_new(150, 2)?)));
        assert_eq!(list.get(1), Some(Variant::Int8(2)));
        assert_eq!(list.get(2), Some(Variant::from(VariantDecimal4::try_new(0, 1)?)));
        let b = obj.get("b").expect("b");
        let c = b.as_object().expect("object").get("c");
        assert_eq!(c, Some(Variant::from(VariantDecimal4::try_new(1230, 2)?)));
        Ok(())
    }

    /// A repeated key keeps its last value; keys come out sorted.
    #[test]
    fn test_json_to_variant_duplicate_keys_keep_last() -> Result<(), ArrowError> {
        let mut builder = VariantBuilder::new();
        builder.append_json("{\"z\": 1, \"a\": 1, \"a\": 2}")?;
        let (metadata, value) = builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;
        assert_eq!(variant.to_json_string()?, "{\"a\":2,\"z\":1}");
        Ok(())
    }

    /// Strings pass through with their escapes decoded, whatever they contain.
    #[test]
    fn test_json_to_variant_string_escapes_decoded() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "\"a\\\"b\\\\c\\n\\t\\u00e9 12.30\"",
            expected: Variant::from("a\"b\\c\n\t\u{e9} 12.30"),
        }
        .run()
    }

    /// Malformed documents are refused before anything is appended.
    #[test]
    fn test_json_to_variant_rejects_malformed() {
        for bad in ["", " ", "{", "[1,", "{\"a\":}", "+1", "01", "1.", ".5", "tru", "nul"] {
            let mut builder = VariantBuilder::new();
            assert!(builder.append_json(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn test_json_to_variant_double_scientific_positive() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "15e-1",
            expected: Variant::Double(15e-1f64),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_double_scientific_negative() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "-15e-1",
            expected: Variant::Double(-15e-1f64),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_short_string() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: "\"harsh\"",
            expected: Variant::ShortString(ShortString::try_new("harsh")?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_short_string_max_length() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: &format!("\"{}\"", "a".repeat(63)),
            expected: Variant::ShortString(ShortString::try_new(&"a".repeat(63))?),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_long_string() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: &format!("\"{}\"", "a".repeat(64)),
            expected: Variant::String(&"a".repeat(64)),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_very_long_string() -> Result<(), ArrowError> {
        JsonToVariantTest {
            json: &format!("\"{}\"", "b".repeat(100000)),
            expected: Variant::String(&"b".repeat(100000)),
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_array_simple() -> Result<(), ArrowError> {
        let mut variant_builder = VariantBuilder::new();
        let mut list_builder = variant_builder.new_list();
        list_builder.append_value(Variant::Int8(127));
        list_builder.append_value(Variant::Int16(128));
        list_builder.append_value(Variant::Int32(-32767431));
        list_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;

        JsonToVariantTest {
            json: "[127, 128, -32767431]",
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_array_with_object() -> Result<(), ArrowError> {
        let mut variant_builder = VariantBuilder::new();
        let mut list_builder = variant_builder.new_list();
        let mut object_builder_inner = list_builder.new_object();
        object_builder_inner.insert("age", Variant::Int8(32));
        object_builder_inner.finish();
        list_builder.append_value(Variant::Int16(128));
        list_builder.append_value(Variant::BooleanFalse);
        list_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;

        JsonToVariantTest {
            json: "[{\"age\": 32}, 128, false]",
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_array_large_u16_offset() -> Result<(), ArrowError> {
        // u16 offset - 128 i8's + 1 "true" = 257 bytes
        let mut variant_builder = VariantBuilder::new();
        let mut list_builder = variant_builder.new_list();
        for _ in 0..128 {
            list_builder.append_value(Variant::Int8(1));
        }
        list_builder.append_value(Variant::BooleanTrue);
        list_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;

        JsonToVariantTest {
            json: &format!("[{} true]", "1, ".repeat(128)),
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_array_nested_large() -> Result<(), ArrowError> {
        // verify u24, and large_size
        let mut variant_builder = VariantBuilder::new();
        let mut list_builder = variant_builder.new_list();
        for _ in 0..256 {
            let mut list_builder_inner = list_builder.new_list();
            for _ in 0..255 {
                list_builder_inner.append_value(Variant::Null);
            }
            list_builder_inner.finish();
        }
        list_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;
        let intermediate = format!("[{}]", vec!["null"; 255].join(", "));
        let json = format!("[{}]", vec![intermediate; 256].join(", "));
        JsonToVariantTest {
            json: json.as_str(),
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_object_simple() -> Result<(), ArrowError> {
        let mut variant_builder = VariantBuilder::new();
        let mut object_builder = variant_builder.new_object();
        object_builder.insert("a", Variant::Int8(3));
        object_builder.insert("b", Variant::Int8(2));
        object_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;
        JsonToVariantTest {
            json: "{\"b\": 2, \"a\": 1, \"a\": 3}",
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_object_complex() -> Result<(), ArrowError> {
        let mut variant_builder = VariantBuilder::new();
        let mut object_builder = variant_builder.new_object();
        let mut inner_list_builder = object_builder.new_list("booleans");
        inner_list_builder.append_value(Variant::BooleanTrue);
        inner_list_builder.append_value(Variant::BooleanFalse);
        inner_list_builder.finish();
        object_builder.insert("null", Variant::Null);
        let mut inner_list_builder = object_builder.new_list("numbers");
        inner_list_builder.append_value(Variant::Int8(4));
        inner_list_builder.append_value(Variant::Double(-3e0));
        inner_list_builder.append_value(Variant::Double(1001e-3));
        inner_list_builder.finish();
        object_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;
        JsonToVariantTest {
            json: "{\"numbers\": [4, -3e0, 1001e-3], \"null\": null, \"booleans\": [true, false]}",
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_object_very_large() -> Result<(), ArrowError> {
        // 256 elements (keys: 000-255) - each element is an object of 256 elements (240-495) - each
        // element a list of numbers from 0-127
        let keys: Vec<String> = (0..=255).map(|n| format!("{n:03}")).collect();
        let innermost_list: String = format!(
            "[{}]",
            (0..=127)
                .map(|n| format!("{n}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let inner_keys: Vec<String> = (240..=495).map(|n| format!("{n}")).collect();
        let inner_object = format!(
            "{{{}:{}}}",
            inner_keys
                .iter()
                .map(|k| format!("\"{k}\""))
                .collect::<Vec<String>>()
                .join(format!(":{innermost_list},").as_str()),
            innermost_list
        );
        let json = format!(
            "{{{}:{}}}",
            keys.iter()
                .map(|k| format!("\"{k}\""))
                .collect::<Vec<String>>()
                .join(format!(":{inner_object},").as_str()),
            inner_object
        );
        // Manually verify raw JSON value size
        let mut variant_builder = VariantBuilder::new();
        variant_builder.append_json(&json)?;
        let (metadata, value) = variant_builder.finish();
        let v = Variant::try_new(&metadata, &value)?;
        let output_string = v.to_json_string()?;
        assert_eq!(output_string, json);
        // Verify metadata size = 1 + 2 + 2 * 497 + 3 * 496
        assert_eq!(metadata.len(), 2485);
        // Verify value size.
        // Size of innermost_list: 1 + 1 + 2*(128 + 1) + 2*128 = 516
        // Size of inner object: 1 + 4 + 2*256 + 3*(256 + 1) + 256 * 516 = 133384
        // Size of json: 1 + 4 + 2*256 + 4*(256 + 1) + 256 * 133384 = 34147849
        assert_eq!(value.len(), 34147849);

        let mut variant_builder = VariantBuilder::new();
        let mut object_builder = variant_builder.new_object();
        keys.iter().for_each(|key| {
            let mut inner_object_builder = object_builder.new_object(key);
            inner_keys.iter().for_each(|inner_key| {
                let mut list_builder = inner_object_builder.new_list(inner_key);
                for i in 0..=127 {
                    list_builder.append_value(Variant::Int8(i));
                }
                list_builder.finish();
            });
            inner_object_builder.finish();
        });
        object_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;

        JsonToVariantTest {
            json: &json,
            expected: variant,
        }
        .run()
    }

    #[test]
    fn test_json_to_variant_unicode() -> Result<(), ArrowError> {
        let json = "{\"爱\":\"अ\",\"a\":1}";
        let mut variant_builder = VariantBuilder::new();
        variant_builder.append_json(json)?;
        let (metadata, value) = variant_builder.finish();
        let v = Variant::try_new(&metadata, &value)?;
        let output_string = v.to_json_string()?;
        assert_eq!(output_string, "{\"a\":1,\"爱\":\"अ\"}");
        let mut variant_builder = VariantBuilder::new();
        let mut object_builder = variant_builder.new_object();
        object_builder.insert("a", Variant::Int8(1));
        object_builder.insert("爱", Variant::ShortString(ShortString::try_new("अ")?));
        object_builder.finish();
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;

        assert_eq!(
            value,
            &[
                2u8, 2u8, 0u8, 1u8, 0u8, 2u8, 6u8, 12u8, 1u8, 13u8, 0xe0u8, 0xa4u8, 0x85u8
            ]
        );
        assert_eq!(
            metadata,
            &[17u8, 2u8, 0u8, 1u8, 4u8, 97u8, 0xe7u8, 0x88u8, 0xb1u8]
        );
        JsonToVariantTest {
            json,
            expected: variant,
        }
        .run()
    }
}
