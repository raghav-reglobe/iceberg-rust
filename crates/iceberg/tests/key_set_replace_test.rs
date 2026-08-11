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

//! ATOMIC KEY-SET UPSERT via scan + consolidated deletion vectors — the
//! repair-through-the-stream primitive: prior rows whose `_cdc.key` is in
//! an EXACT KEY SET (∧ op ∈ {R,r} ∧ `_is_backfill=true`) are deleted by
//! scan + ONE consolidated V3 deletion vector per affected data file, the
//! replacement rows appended, and the caller's offset-ledger summary
//! properties stamped — all in ONE `RowDelta` snapshot. The idempotency
//! golden lives here: producing the same key set twice converges (row set
//! unchanged), the at-least-once contract the streaming lane rides on.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{BooleanArray, LargeStringArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::atomic_replace::atomic_partition_replace_key_set;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList,
    NestedField, PartitionKey, PrimitiveType, Schema, Struct, StructType, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{
    Catalog, CatalogBuilder, MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder, NamespaceIdent,
    TableCreation, TableIdent,
};
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;
use uuid::Uuid;

const FIELD_ID_AFTER: i32 = 1;
const FIELD_ID_CDC: i32 = 2;
const FIELD_ID_CDC_OP: i32 = 3;
const FIELD_ID_CDC_KEY: i32 = 4;
const FIELD_ID_IS_BACKFILL: i32 = 5;

/// The mongo-bronze envelope shape (reduced): `after` payload + `_cdc`
/// struct carrying the op marker and the stringified document key.
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

fn table_arrow_schema(table: &Table) -> Arc<ArrowSchema> {
    Arc::new(iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema()).unwrap())
}

/// Rows: (after, op, key, is_backfill), built against the table's schema.
fn batch(table: &Table, rows: &[(&str, &str, &str, Option<bool>)]) -> RecordBatch {
    let schema = table_arrow_schema(table);
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
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

fn partition_key(table: &Table, is_backfill: Option<bool>) -> PartitionKey {
    PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter(vec![is_backfill.map(Literal::bool)]),
    )
}

async fn write_data(
    table: &Table,
    prefix: &str,
    is_backfill: Option<bool>,
    b: RecordBatch,
) -> Vec<DataFile> {
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            table.metadata().current_schema().clone(),
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(
            format!("{prefix}-{}", Uuid::now_v7()),
            None,
            DataFileFormat::Parquet,
        ),
    );
    let mut w = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key(table, is_backfill)))
        .await
        .unwrap();
    w.write(b).await.unwrap();
    w.close().await.unwrap()
}

/// Read every live row back as (after, op, key), sorted.
async fn read_rows(table: &Table) -> Vec<(String, String, String)> {
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
        let schema = b.schema();
        let afters = b
            .column(schema.index_of("after").unwrap())
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        let cdc = b
            .column(schema.index_of("_cdc").unwrap())
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let ops = cdc
            .column_by_name("op")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        let keys = cdc
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            out.push((
                afters.value(i).to_string(),
                ops.value(i).to_string(),
                keys.value(i).to_string(),
            ));
        }
    }
    out.sort();
    out
}

/// Live delete-file entries of the current snapshot: (file_path, format).
async fn live_delete_entries(table: &Table) -> Vec<(String, DataFileFormat)> {
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
        if mf.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = mf.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                out.push((
                    entry.data_file().file_path().to_string(),
                    entry.data_file().file_format(),
                ));
            }
        }
    }
    out
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

/// Seed the two partitions the way the pipeline does: full-load rows (op=r)
/// in `_is_backfill=true` (one file spanning MULTIPLE prefixes — the legacy
/// whole-collection load shape); CDC rows (op=c) in the null partition, one
/// sharing a key with a full-load row. Returns the seeded backfill files.
async fn seed(catalog: &impl Catalog, ident: &TableIdent) -> Vec<DataFile> {
    let table = catalog.load_table(ident).await.unwrap();
    let backfill_files = write_data(
        &table,
        "seed-backfill",
        Some(true),
        batch(&table, &[
            ("r-old-aa01", "r", "{\"$oid\": \"aa01\"}", Some(true)),
            ("r-old-aa02", "R", "{\"$oid\": \"aa02\"}", Some(true)),
            ("r-old-ab99", "r", "{\"$oid\": \"ab99\"}", Some(true)),
            ("r-old-bb01", "r", "{\"$oid\": \"bb01\"}", Some(true)),
        ]),
    )
    .await;
    let mut all = backfill_files.clone();
    all.extend(
        write_data(
            &table,
            "seed-cdc",
            None,
            batch(&table, &[
                ("c-aa01", "c", "{\"$oid\": \"aa01\"}", None),
                ("c-zz01", "c", "{\"$oid\": \"zz01\"}", None),
            ]),
        )
        .await,
    );
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(all)
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap();
    backfill_files
}

async fn run_upsert(
    catalog: &impl Catalog,
    ident: &TableIdent,
    keys: &[&str],
    rows: &[(&str, &str, &str, Option<bool>)],
    ledger_offset: &str,
) -> iceberg::atomic_replace::ReplaceOutcome {
    let table = catalog.load_table(ident).await.unwrap();
    let batches = if rows.is_empty() {
        vec![]
    } else {
        vec![batch(&table, rows)]
    };
    atomic_partition_replace_key_set(
        catalog,
        ident,
        "_is_backfill",
        true,
        "_cdc.key",
        keys.iter().map(|k| k.to_string()).collect(),
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        batches,
        HashMap::from([(
            "pulse.kafka.offset.bronze_mongo.db.t".to_string(),
            ledger_offset.to_string(),
        )]),
        3,
        false,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn key_set_upsert_replaces_exact_keys_and_stamps_ledger() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let before_snapshots = {
        let t = catalog.load_table(&ident).await.unwrap();
        t.metadata().snapshots().count()
    };

    // Upsert EXACTLY {aa01, zz09}: aa01's prior op=R baseline replaced,
    // zz09 newly appears. aa02/ab99/bb01 (op=R, NOT in the set) survive —
    // set-exact scope, unlike the prefix mode. The CDC row sharing key
    // aa01 (op=c, null partition) survives — the baked op+partition scope.
    let outcome = run_upsert(
        &catalog,
        &ident,
        &["{\"$oid\": \"aa01\"}", "{\"$oid\": \"zz09\"}"],
        &[
            ("r-new-aa01", "R", "{\"$oid\": \"aa01\"}", Some(true)),
            ("r-new-zz09", "R", "{\"$oid\": \"zz09\"}", Some(true)),
        ],
        "42",
    )
    .await;
    assert_eq!(outcome.rows_appended, 2);
    assert_eq!(outcome.delete_tuples, 1); // only aa01 had a prior baseline
    assert_eq!(outcome.attempts, 1);

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(table.metadata().snapshots().count(), before_snapshots + 1);
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties
            .get("pulse.kafka.offset.bronze_mongo.db.t")
            .map(String::as_str),
        Some("42"),
        "the offset ledger must ride the SAME snapshot as the upsert"
    );
    assert_eq!(read_rows(&table).await, vec![
        ("c-aa01".into(), "c".into(), "{\"$oid\": \"aa01\"}".into()),
        ("c-zz01".into(), "c".into(), "{\"$oid\": \"zz01\"}".into()),
        (
            "r-new-aa01".into(),
            "R".into(),
            "{\"$oid\": \"aa01\"}".into()
        ),
        (
            "r-new-zz09".into(),
            "R".into(),
            "{\"$oid\": \"zz09\"}".into()
        ),
        (
            "r-old-aa02".into(),
            "R".into(),
            "{\"$oid\": \"aa02\"}".into()
        ),
        (
            "r-old-ab99".into(),
            "r".into(),
            "{\"$oid\": \"ab99\"}".into()
        ),
        (
            "r-old-bb01".into(),
            "r".into(),
            "{\"$oid\": \"bb01\"}".into()
        ),
    ]);
}

/// THE IDEMPOTENCY GOLDEN (the streaming lane's at-least-once contract):
/// producing the same key set + rows twice leaves the row set IDENTICAL —
/// the second pass deletes the first pass's appends and re-appends equal
/// rows. One snapshot per pass, each carrying its ledger stamp.
#[tokio::test]
async fn key_set_upsert_is_replay_idempotent() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let keys = ["{\"$oid\": \"aa01\"}", "{\"$oid\": \"zz09\"}"];
    let rows: &[(&str, &str, &str, Option<bool>)] = &[
        ("r-new-aa01", "R", "{\"$oid\": \"aa01\"}", Some(true)),
        ("r-new-zz09", "R", "{\"$oid\": \"zz09\"}", Some(true)),
    ];

    run_upsert(&catalog, &ident, &keys, rows, "42").await;
    let after_first = {
        let t = catalog.load_table(&ident).await.unwrap();
        read_rows(&t).await
    };

    let outcome = run_upsert(&catalog, &ident, &keys, rows, "77").await;
    assert_eq!(outcome.rows_appended, 2);
    assert_eq!(
        outcome.delete_tuples, 2,
        "the replay deletes its own prior pass"
    );

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        read_rows(&table).await,
        after_first,
        "replaying the same key set must converge — row set unchanged"
    );
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties
            .get("pulse.kafka.offset.bronze_mongo.db.t")
            .map(String::as_str),
        Some("77"),
        "the replay's ledger stamp advances with its commit"
    );
}

#[tokio::test]
async fn key_set_empty_keys_is_noop() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let before = {
        let t = catalog.load_table(&ident).await.unwrap();
        (t.metadata().snapshots().count(), read_rows(&t).await)
    };
    let outcome = atomic_partition_replace_key_set(
        &catalog,
        &ident,
        "_is_backfill",
        true,
        "_cdc.key",
        HashSet::new(),
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        Vec::<RecordBatch>::new(),
        HashMap::new(),
        3,
        false,
    )
    .await
    .unwrap();
    assert_eq!(outcome.rows_appended, 0);
    assert_eq!(outcome.attempts, 0, "a no-op must not commit");

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        (
            table.metadata().snapshots().count(),
            read_rows(&table).await
        ),
        before,
        "empty key set = structural no-op (no snapshot, no changes)"
    );
}
