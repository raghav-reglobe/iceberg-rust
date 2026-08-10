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

//! ATOMIC PARTITION-NULL APPEND with snapshot-summary ledger properties —
//! the streaming-consumer (kafka → bronze) write: rows land in the NULL
//! slot of the identity partition (the CDC slot), the consumer's offset
//! ledger rides the snapshot summary, and a zero-row input never commits.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{BooleanArray, LargeStringArray, RecordBatch, StructArray};
use arrow_schema::DataType;
use futures::TryStreamExt;
use iceberg::atomic_replace::atomic_partition_append;
use iceberg::spec::{
    DataFileFormat, FormatVersion, ManifestContentType, ManifestList, NestedField, PrimitiveType,
    Schema, StructType, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder, NamespaceIdent,
    TableCreation, TableIdent,
};
use tempfile::TempDir;

const FIELD_ID_AFTER: i32 = 1;
const FIELD_ID_CDC: i32 = 2;
const FIELD_ID_CDC_OP: i32 = 3;
const FIELD_ID_CDC_KEY: i32 = 4;
const FIELD_ID_IS_BACKFILL: i32 = 5;

fn table_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(
                FIELD_ID_AFTER,
                "after",
                Type::Primitive(PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                FIELD_ID_CDC,
                "_cdc",
                Type::Struct(StructType::new(vec![
                    NestedField::required(
                        FIELD_ID_CDC_OP,
                        "op",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    NestedField::required(
                        FIELD_ID_CDC_KEY,
                        "key",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                ])),
            )
            .into(),
            NestedField::optional(
                FIELD_ID_IS_BACKFILL,
                "_is_backfill",
                Type::Primitive(PrimitiveType::Boolean),
            )
            .into(),
        ])
        .build()
        .unwrap()
}

/// Rows: (after, op, key) — `_is_backfill` always NULL (the CDC shape).
fn batch(table: &Table, rows: &[(&str, &str, &str)]) -> RecordBatch {
    let schema = Arc::new(
        iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema()).unwrap(),
    );
    let (op_field, key_field) = match schema.field_with_name("_cdc").unwrap().data_type() {
        DataType::Struct(fields) => (fields[0].clone(), fields[1].clone()),
        other => panic!("_cdc must be a struct, got {other:?}"),
    };
    let ops: Vec<&str> = rows.iter().map(|r| r.1).collect();
    let keys: Vec<&str> = rows.iter().map(|r| r.2).collect();
    let cdc = StructArray::from(vec![
        (
            op_field,
            Arc::new(LargeStringArray::from(ops)) as arrow_array::ArrayRef,
        ),
        (
            key_field,
            Arc::new(LargeStringArray::from(keys)) as arrow_array::ArrayRef,
        ),
    ]);
    RecordBatch::try_new(schema, vec![
        Arc::new(LargeStringArray::from(
            rows.iter().map(|r| r.0.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(cdc),
        Arc::new(BooleanArray::from(vec![None::<bool>; rows.len()])),
    ])
    .unwrap()
}

async fn setup(warehouse: &TempDir) -> (impl Catalog, TableIdent) {
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap();
    let ns = NamespaceIdent::new("db".to_string());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(FIELD_ID_IS_BACKFILL, "_is_backfill", Transform::Identity)
        .unwrap()
        .build();
    catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("bronze_mongo_t".to_string())
                .schema(table_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    (catalog, TableIdent::new(ns, "bronze_mongo_t".to_string()))
}

async fn read_keys(table: &Table) -> Vec<String> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let cdc = b
            .column(b.schema().index_of("_cdc").unwrap())
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .clone();
        let keys = cdc
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap()
            .clone();
        for i in 0..b.num_rows() {
            out.push(keys.value(i).to_string());
        }
    }
    out.sort();
    out
}

/// Data-manifest entries of the current snapshot: (file_path, partition).
async fn live_data_partitions(table: &Table) -> Vec<iceberg::spec::Struct> {
    let metadata = table.metadata();
    let snapshot = metadata.current_snapshot().unwrap();
    let bytes = table
        .file_io()
        .new_input(snapshot.manifest_list())
        .unwrap()
        .read()
        .await
        .unwrap();
    let manifest_list =
        ManifestList::parse_with_version(&bytes, metadata.format_version()).unwrap();
    let mut out = Vec::new();
    for mf in manifest_list.entries() {
        if mf.content != ManifestContentType::Data {
            continue;
        }
        let manifest = mf.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                assert_eq!(entry.data_file().file_format(), DataFileFormat::Parquet);
                out.push(entry.data_file().partition().clone());
            }
        }
    }
    out
}

#[test]
fn json_float_parse_is_correctly_rounded() {
    // serde_json's default fast float path can land 1 ULP off the
    // correctly-rounded f64; the workspace `float_roundtrip` feature pins
    // the exact parse. Without it every JSON→VARIANT double write
    // (parquet-variant append_json → as_f64) diverges from Java writers
    // (Jackson is correctly-rounded) — caught live by the kafka→bronze
    // golden gate as 76/1000 product_quote rows differing by 1 ULP.
    let v: serde_json::Value = serde_json::from_str("61101.263999999996").unwrap();
    assert_eq!(
        v.as_f64().unwrap().to_bits(),
        0x40edd5a872b020c4u64,
        "serde_json float parse is not correctly rounded — is the \
         workspace float_roundtrip feature still on?"
    );
}

#[tokio::test]
async fn append_lands_in_null_partition_with_ledger_properties() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    let table = catalog.load_table(&ident).await.unwrap();

    let props = HashMap::from([(
        "pulse.kafka.offset.bronze_mongo.db.t".to_string(),
        "41".to_string(),
    )]);
    let out = atomic_partition_append(
        &catalog,
        &ident,
        "_is_backfill",
        vec![batch(&table, &[
            ("doc-1", "I", "{\"$oid\": \"aa01\"}"),
            ("doc-2", "U", "{\"$oid\": \"aa02\"}"),
        ])],
        props,
        3,
    )
    .await
    .unwrap();
    assert_eq!(out.rows_appended, 2);
    assert_eq!(out.attempts, 1);

    let table = catalog.load_table(&ident).await.unwrap();
    let snapshot = table.metadata().current_snapshot().unwrap();
    assert_eq!(Some(snapshot.snapshot_id()), out.snapshot_id);
    assert_eq!(
        snapshot
            .summary()
            .additional_properties
            .get("pulse.kafka.offset.bronze_mongo.db.t")
            .map(String::as_str),
        Some("41"),
        "the offset ledger must ride the snapshot summary"
    );
    assert_eq!(read_keys(&table).await, vec![
        "{\"$oid\": \"aa01\"}".to_string(),
        "{\"$oid\": \"aa02\"}".to_string(),
    ]);
    // Every appended file sits in the NULL partition slot (the CDC slot).
    let partitions = live_data_partitions(&table).await;
    assert!(!partitions.is_empty());
    for p in partitions {
        assert_eq!(
            p.iter().collect::<Vec<_>>(),
            vec![None],
            "appended rows must land in the NULL `_is_backfill` partition"
        );
    }
}

#[tokio::test]
async fn zero_row_input_is_a_noop() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    let table = catalog.load_table(&ident).await.unwrap();
    let before = table.metadata().current_snapshot_id();

    let out = atomic_partition_append(
        &catalog,
        &ident,
        "_is_backfill",
        Vec::<RecordBatch>::new(),
        HashMap::from([("pulse.kafka.offset.t".to_string(), "7".to_string())]),
        3,
    )
    .await
    .unwrap();
    assert_eq!(out.rows_appended, 0);
    assert_eq!(out.attempts, 0);

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        table.metadata().current_snapshot_id(),
        before,
        "a zero-row append must never commit (the ledger only advances with data)"
    );
}

#[tokio::test]
async fn successive_appends_keep_their_own_ledgers() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    let table = catalog.load_table(&ident).await.unwrap();

    let ledger = |v: &str| {
        HashMap::from([(
            "pulse.kafka.offset.bronze_mongo.db.t".to_string(),
            v.to_string(),
        )])
    };
    atomic_partition_append(
        &catalog,
        &ident,
        "_is_backfill",
        vec![batch(&table, &[("doc-1", "I", "{\"$oid\": \"aa01\"}")])],
        ledger("10"),
        3,
    )
    .await
    .unwrap();
    atomic_partition_append(
        &catalog,
        &ident,
        "_is_backfill",
        vec![batch(&table, &[("doc-2", "U", "{\"$oid\": \"aa01\"}")])],
        ledger("25"),
        3,
    )
    .await
    .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let current = table.metadata().current_snapshot().unwrap();
    assert_eq!(
        current
            .summary()
            .additional_properties
            .get("pulse.kafka.offset.bronze_mongo.db.t")
            .map(String::as_str),
        Some("25"),
        "the CURRENT snapshot carries the latest consumed offset"
    );
    // The prior ledger stays in history (resume walks newest-first).
    let mut ledgers: Vec<Option<String>> = table
        .metadata()
        .snapshots()
        .map(|s| {
            s.summary()
                .additional_properties
                .get("pulse.kafka.offset.bronze_mongo.db.t")
                .cloned()
        })
        .collect();
    ledgers.sort();
    assert_eq!(ledgers, vec![
        Some("10".to_string()),
        Some("25".to_string())
    ]);
}
