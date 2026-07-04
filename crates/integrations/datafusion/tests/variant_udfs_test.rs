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

//! Integration tests for the variant scalar UDFs.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use iceberg_datafusion::functions::register_variant_functions;
use parquet::variant::{Variant, VariantArrayBuilder, json_to_variant};

/// Three rows of schemaless documents:
/// - row 0: fully populated (nested object, array, mixed scalars)
/// - row 1: sparse, with a type-drifted `qty` (string instead of number)
/// - row 2: NULL document
fn docs_batch() -> RecordBatch {
    let jsons: ArrayRef = Arc::new(StringArray::from(vec![
        Some(
            r#"{"_id": {"$oid": "665f0abc"}, "qty": 3, "price": 19.5, "active": true,
                "name": "alice", "tags": [7, 8], "meta": {"city": "BLR"}}"#,
        ),
        Some(r#"{"_id": {"$oid": "665f0def"}, "qty": "not-a-number", "name": "bob"}"#),
        None,
    ]));
    let docs = ArrayRef::from(json_to_variant(&jsons).unwrap());

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("doc", docs.data_type().clone(), true),
    ]);
    let ids: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
    RecordBatch::try_new(Arc::new(schema), vec![ids, docs]).unwrap()
}

async fn ctx_with_docs() -> SessionContext {
    let ctx = SessionContext::new();
    register_variant_functions(&ctx);
    ctx.register_batch("t", docs_batch()).unwrap();
    ctx
}

async fn one_column(ctx: &SessionContext, sql: &str) -> ArrayRef {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(batches.len(), 1, "expected one batch for {sql}");
    assert_eq!(batches[0].num_columns(), 1);
    batches[0].column(0).clone()
}

#[tokio::test]
async fn typed_extraction_with_nulls_on_missing_and_drift() {
    let ctx = ctx_with_docs().await;

    // Nested path with a `$`-containing field name and the `$.` root marker.
    let oid = one_column(
        &ctx,
        "SELECT variant_get_string(doc, '$._id.$oid') FROM t ORDER BY id",
    )
    .await;
    let oid = oid.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(oid.value(0), "665f0abc");
    assert_eq!(oid.value(1), "665f0def");
    assert!(oid.is_null(2));

    // Number on row 0; type-drifted string on row 1 -> NULL (safe cast);
    // NULL document on row 2 -> NULL.
    let qty = one_column(
        &ctx,
        "SELECT variant_get_bigint(doc, 'qty') FROM t ORDER BY id",
    )
    .await;
    let qty = qty.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(qty.value(0), 3);
    assert!(qty.is_null(1));
    assert!(qty.is_null(2));

    // Missing field -> NULL everywhere it is absent.
    let price = one_column(
        &ctx,
        "SELECT variant_get_double(doc, 'price') FROM t ORDER BY id",
    )
    .await;
    let price = price.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(price.value(0), 19.5);
    assert!(price.is_null(1));
    assert!(price.is_null(2));

    let active = one_column(
        &ctx,
        "SELECT variant_get_boolean(doc, 'active') FROM t ORDER BY id",
    )
    .await;
    let active = active.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(active.value(0));
    assert!(active.is_null(1));
    assert!(active.is_null(2));

    // Array indexing.
    let tag0 = one_column(
        &ctx,
        "SELECT variant_get_bigint(doc, 'tags[0]') FROM t ORDER BY id",
    )
    .await;
    let tag0 = tag0.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(tag0.value(0), 7);
    assert!(tag0.is_null(1));
}

#[tokio::test]
async fn variant_out_chains_and_serializes() {
    let ctx = ctx_with_docs().await;

    // variant_get -> variant, then a chained getter on the sub-document.
    let city = one_column(
        &ctx,
        "SELECT variant_get_string(variant_get(doc, 'meta'), 'city') FROM t ORDER BY id",
    )
    .await;
    let city = city.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(city.value(0), "BLR");
    assert!(city.is_null(1));
    assert!(city.is_null(2));

    // Sub-document rendered back to JSON.
    let meta = one_column(
        &ctx,
        "SELECT variant_to_json(variant_get(doc, 'meta')) FROM t ORDER BY id",
    )
    .await;
    let meta = meta.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(meta.value(0), r#"{"city":"BLR"}"#);
    assert!(meta.is_null(1));
}

#[tokio::test]
async fn json_to_variant_roundtrip_in_sql() {
    let ctx = SessionContext::new();
    register_variant_functions(&ctx);

    let a = one_column(
        &ctx,
        r#"SELECT variant_get_bigint(json_to_variant('{"a": 41}'), '$.a') + 1"#,
    )
    .await;
    let a = a.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(a.value(0), 42);
}

#[tokio::test]
async fn timestamp_extraction_from_variant_timestamp() {
    // Build a variant carrying a real timestamp value (micros, UTC).
    let ts = chrono::DateTime::from_timestamp_micros(1_714_557_600_000_000).unwrap(); // 2024-05-01T10:00:00Z
    let mut builder = VariantArrayBuilder::new(1);
    builder.append_variant(Variant::from(ts));
    let docs = ArrayRef::from(builder.build());

    let schema = Schema::new(vec![Field::new("doc", docs.data_type().clone(), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![docs]).unwrap();

    let ctx = SessionContext::new();
    register_variant_functions(&ctx);
    ctx.register_batch("t", batch).unwrap();

    let out = one_column(&ctx, "SELECT variant_get_timestamp(doc, '$') FROM t").await;
    let out = out
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(out.value(0), 1_714_557_600_000_000);
}

#[tokio::test]
async fn non_literal_path_is_rejected() {
    let ctx = ctx_with_docs().await;
    // NB: a plain aliased literal ('qty' AS name) gets constant-folded by the
    // optimizer and reaches the UDF as a scalar — use a data-dependent CASE.
    let err = ctx
        .sql("SELECT variant_get_string(doc, CASE WHEN id >= 0 THEN 'qty' ELSE 'name' END) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("path must be a non-null string literal"),
        "unexpected error: {err}"
    );
}
