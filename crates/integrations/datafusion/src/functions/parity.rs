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

//! Cross-engine PARITY helpers — tiny UDFs that make row-hash checksum SQL
//! expressible identically here and on MySQL-compatible engines.
//!
//! `conv16(s)` is the missing piece of the md5-slice scheme
//! `SUM(CONV(SUBSTRING(MD5(CONCAT_WS(...)), 1, 15), 16, 10))`: it converts
//! a hexadecimal string (up to 16 digits — the scheme uses 15, i.e. 60
//! bits) to its unsigned decimal value, with MySQL `CONV(_, 16, 10)`
//! semantics: the longest valid leading hex prefix is parsed,
//! case-insensitively; an empty/invalid input yields 0; NULL stays NULL.
//! Sum the result with exact integer semantics on BOTH sides
//! (`SUM(CAST(CONV(...) AS UNSIGNED))` on MySQL) — never as a double.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, StringArray, UInt64Array};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;

/// `conv16(hex_string)` → UInt64 (MySQL `CONV(hex_string, 16, 10)`).
#[derive(Debug, PartialEq, Eq, Hash)]
struct Conv16Udf {
    signature: Signature,
}

impl Conv16Udf {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

/// MySQL CONV semantics for base 16: parse the longest valid leading hex
/// prefix (case-insensitive); empty prefix ⇒ 0. Capped at 16 digits (u64).
fn conv16_str(s: &str) -> u64 {
    let mut out: u64 = 0;
    for (i, c) in s.chars().enumerate() {
        let Some(d) = c.to_digit(16) else { break };
        if i >= 16 {
            break;
        }
        out = (out << 4) | u64::from(d);
    }
    out
}

impl ScalarUDFImpl for Conv16Udf {
    fn name(&self) -> &str {
        "conv16"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::UInt64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let input = match &args.args[0] {
            ColumnarValue::Array(a) => Arc::clone(a),
            ColumnarValue::Scalar(s) => s.to_array_of_size(args.number_rows)?,
        };
        let strings = cast(input.as_ref(), &DataType::Utf8)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        let strings = strings
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to Utf8");
        let out: UInt64Array = strings.iter().map(|v| v.map(conv16_str)).collect();
        Ok(ColumnarValue::Array(Arc::new(out) as ArrayRef))
    }
}

/// Register the parity UDFs on the given [`SessionContext`].
pub fn register_parity_functions(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(Conv16Udf::new()));
}

#[cfg(test)]
mod tests {
    use super::conv16_str;

    #[test]
    fn mysql_conv_semantics() {
        // Goldens precomputed with MySQL CONV(_, 16, 10) semantics.
        assert_eq!(conv16_str("900150983cd24fb"), 648541476951500027);
        assert_eq!(conv16_str("d41d8cd98f00b20"), 955282973525019424);
        assert_eq!(conv16_str("FF"), 255); // case-insensitive
        assert_eq!(conv16_str("7fzz9"), 0x7f); // longest valid leading prefix
        assert_eq!(conv16_str(""), 0);
        assert_eq!(conv16_str("zzz"), 0);
        assert_eq!(conv16_str("ffffffffffffffff"), u64::MAX); // 16 digits fit
    }
}
