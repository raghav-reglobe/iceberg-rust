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

//! PARITY-SQL probes — locks the DataFusion behaviors a MySQL-vs-lakehouse
//! row-hash parity scheme depends on (the MySQL side computes
//! `SUM(CONV(SUBSTRING(MD5(CONCAT_WS(...)),1,15),16,10))` over canonicalized
//! column renderings). Each probe asserts the CURRENT engine behavior so any
//! upgrade that would silently break hash parity fails HERE first, and the
//! divergences that need explicit canonicalization SQL are documented next
//! to the probe that proves them.

use datafusion::arrow::array::{Array, StringArray};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;

async fn one_string(ctx: &SessionContext, sql: &str) -> Option<String> {
    let batches: Vec<RecordBatch> = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    // Results may arrive as Utf8View (the engine default) — normalize.
    let col = datafusion::arrow::compute::cast(
        batches[0].column(0).as_ref(),
        &datafusion::arrow::datatypes::DataType::Utf8,
    )
    .unwrap();
    let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
    arr.is_valid(0).then(|| arr.value(0).to_string())
}

/// md5() exists and is byte-identical to MySQL's MD5() (both emit lowercase
/// hex of the UTF-8 bytes).
#[tokio::test]
async fn md5_matches_mysql() {
    let ctx = SessionContext::new();
    assert_eq!(
        one_string(&ctx, "SELECT md5('abc')").await.as_deref(),
        Some("900150983cd24fb0d6963f7d28e17f72")
    );
    assert_eq!(
        one_string(&ctx, "SELECT md5('')").await.as_deref(),
        Some("d41d8cd98f00b204e9800998ecf8427e")
    );
}

/// concat_ws() SKIPS NULL arguments exactly like MySQL's CONCAT_WS (no
/// separator emitted for a null) — same skip semantics, so a NULL column
/// and an ABSENT column hash identically on BOTH engines. That ambiguity is
/// why the parity scheme must canonicalize NULLs to an explicit sentinel
/// (`COALESCE(col, '\\N')` on both sides) BEFORE concat.
#[tokio::test]
async fn concat_ws_skips_nulls_like_mysql() {
    let ctx = SessionContext::new();
    assert_eq!(
        one_string(&ctx, "SELECT concat_ws('|', 'a', NULL, 'b')")
            .await
            .as_deref(),
        Some("a|b")
    );
    // NULL vs empty string are DIFFERENT (empty keeps its separator).
    assert_eq!(
        one_string(&ctx, "SELECT concat_ws('|', 'a', '', 'b')")
            .await
            .as_deref(),
        Some("a||b")
    );
}

/// Type renderings that FEED the hash. Divergences from MySQL and their
/// canonicalizations:
///
/// | type | MySQL renders | DataFusion CAST AS VARCHAR | canonicalization |
/// |---|---|---|---|
/// | DECIMAL(38,6) | full scale `1.500000` | full scale `1.500000` | none needed (CAST both sides to DECIMAL(38,6) first) |
/// | DOUBLE | shortest round-trip-ish (`1.5`, `0.1`) | Rust ryu-style (`1.5`, `0.1`); scientific forms may diverge | CAST to DECIMAL(38,6) on BOTH sides — never hash raw floats |
/// | BOOLEAN | `1` / `0` (tinyint) | `true` / `false` | CAST(bool AS INT) before rendering |
/// | TIMESTAMP | `2026-01-02 03:04:05` (space) | `2026-01-02T03:04:05` (T) | `date_format(ts, '%Y-%m-%d %H:%M:%S')` / `to_char` on the lakehouse side |
/// | NULL | skipped by CONCAT_WS | skipped by concat_ws | `COALESCE(x, '\\N')` sentinel on both sides |
#[tokio::test]
async fn renderings_that_feed_the_hash() {
    let ctx = SessionContext::new();
    // Decimal: full-scale rendering, matches MySQL DECIMAL rendering.
    assert_eq!(
        one_string(&ctx, "SELECT CAST(CAST(1.5 AS DECIMAL(38,6)) AS VARCHAR)")
            .await
            .as_deref(),
        Some("1.500000")
    );
    // Boolean: 'true'/'false' — DIVERGES from MySQL's 1/0; cast to int.
    assert_eq!(
        one_string(&ctx, "SELECT CAST(true AS VARCHAR)")
            .await
            .as_deref(),
        Some("true")
    );
    assert_eq!(
        one_string(&ctx, "SELECT CAST(CAST(true AS INT) AS VARCHAR)")
            .await
            .as_deref(),
        Some("1")
    );
    // Timestamp: RFC3339 'T' separator — DIVERGES from MySQL's space;
    // canonicalize via to_char.
    assert_eq!(
        one_string(
            &ctx,
            "SELECT CAST(TIMESTAMP '2026-01-02 03:04:05' AS VARCHAR)"
        )
        .await
        .as_deref(),
        Some("2026-01-02T03:04:05")
    );
    assert_eq!(
        one_string(
            &ctx,
            "SELECT to_char(TIMESTAMP '2026-01-02 03:04:05', '%Y-%m-%d %H:%M:%S')"
        )
        .await
        .as_deref(),
        Some("2026-01-02 03:04:05")
    );
}

/// MySQL's CONV(hex, 16, 10) — the 60-bit md5-prefix-to-decimal step — has
/// NO DataFusion builtin: this probe pins the GAP (a ~20-line UDF at the
/// session seam, like the variant functions, closes it: parse 15 hex chars
/// as u64, render as decimal string / UInt64).
#[tokio::test]
async fn conv_16_10_is_a_udf_gap() {
    let ctx = SessionContext::new();
    let err = ctx.sql("SELECT conv('7f', 16, 10)").await.err();
    assert!(
        err.is_some(),
        "if conv() ever appears natively, revisit the parity UDF"
    );
    // The intended UDF semantics, computed by hand for the golden:
    // SUBSTRING(MD5('abc'),1,15) = '900150983cd24fb' -> CONV(_,16,10)
    // = 648518346341351han... asserted platform-side; here we only pin the
    // md5 prefix input.
    assert_eq!(
        one_string(&ctx, "SELECT substr(md5('abc'), 1, 15)")
            .await
            .as_deref(),
        Some("900150983cd24fb")
    );
}

/// The FULL md5-slice expression with the registered `conv16` UDF, against
/// PRECOMPUTED MySQL goldens (hashlib/int(_,16) — exact CONV(_,16,10)
/// integer semantics). The canonicalized row strings follow the contract:
/// pipe separator, `\N` NULL sentinel (note: a MySQL string literal must
/// spell it '\\N' — MySQL drops the backslash from unrecognized escapes),
/// DECIMAL(38,6) full-scale rendering, booleans as 0/1, timestamps
/// space-separated via to_char. MySQL side: SUM(CAST(CONV(...) AS
/// UNSIGNED)) for the same exact-integer sum.
#[tokio::test]
async fn md5slice_full_expression_matches_mysql_goldens() {
    use datafusion::arrow::array::UInt64Array;

    let ctx = SessionContext::new();
    iceberg_datafusion::functions::register_parity_functions(&ctx);

    // Canonicalization happens IN SQL, from typed values — the same shapes
    // the parity reader would emit.
    let sql = "WITH src(id, name, amount, flag, ts) AS (VALUES \
                   (1, 'alpha', 1.5,   true,  TIMESTAMP '2026-01-02 03:04:05'), \
                   (2, NULL,    2.0,   false, TIMESTAMP '2026-01-02 03:04:06'), \
                   (3, 'gamma', -7.25, true,  TIMESTAMP '2026-12-31 23:59:59')), \
               canon AS (SELECT id, concat_ws('|', \
                   CAST(id AS VARCHAR), \
                   coalesce(name, '\\N'), \
                   CAST(CAST(amount AS DECIMAL(38,6)) AS VARCHAR), \
                   CAST(CAST(flag AS INT) AS VARCHAR), \
                   to_char(ts, '%Y-%m-%d %H:%M:%S')) AS row_str \
               FROM src) \
               SELECT id, conv16(substr(md5(row_str), 1, 15)) AS h FROM canon ORDER BY id";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut hashes = Vec::new();
    for b in &batches {
        let h = b.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        hashes.extend(h.values().iter().copied());
    }
    // MySQL goldens: CONV(SUBSTRING(MD5('<canonical row>'),1,15),16,10).
    assert_eq!(hashes, vec![
        520256900126346162,  // '1|alpha|1.500000|1|2026-01-02 03:04:05'
        1147288043085901310, // '2|\N|2.000000|0|2026-01-02 03:04:06'
        33466909318705873,   // '3|gamma|-7.250000|1|2026-12-31 23:59:59'
    ]);

    // And the slice SUM — exact integer aggregation on both sides.
    let sum_sql = "WITH src(id, name, amount, flag, ts) AS (VALUES \
                   (1, 'alpha', 1.5,   true,  TIMESTAMP '2026-01-02 03:04:05'), \
                   (2, NULL,    2.0,   false, TIMESTAMP '2026-01-02 03:04:06'), \
                   (3, 'gamma', -7.25, true,  TIMESTAMP '2026-12-31 23:59:59')) \
               SELECT CAST(SUM(conv16(substr(md5(concat_ws('|', \
                   CAST(id AS VARCHAR), \
                   coalesce(name, '\\N'), \
                   CAST(CAST(amount AS DECIMAL(38,6)) AS VARCHAR), \
                   CAST(CAST(flag AS INT) AS VARCHAR), \
                   to_char(ts, '%Y-%m-%d %H:%M:%S'))), 1, 15))) AS VARCHAR) FROM src";
    let batches = ctx.sql(sum_sql).await.unwrap().collect().await.unwrap();
    let col = datafusion::arrow::compute::cast(
        batches[0].column(0).as_ref(),
        &datafusion::arrow::datatypes::DataType::Utf8,
    )
    .unwrap();
    let sum = col
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0)
        .to_string();
    assert_eq!(sum, "1701011852530953345", "slice checksum golden");
}
