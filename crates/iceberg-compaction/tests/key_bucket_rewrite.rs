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

//! The KEY-BUCKET rewrite of one identity partition (`Config::key_bucket` +
//! `Config::partition_filter`): a bronze-shaped V3 table (`_is_backfill`
//! identity partition, `_cdc{op, key}` struct) whose backfill files hold
//! ObjectId-style keys of three 3-hex prefixes interleaved, plus a CDC-partition
//! file and a deletion vector on one backfill file. After the rewrite:
//!   - the CDC-partition file is untouched (partition filter);
//!   - the backfill partition holds one file per bucket (`69c`, `69e`, `6a1`,
//!     and `other` for the NULL key and the non-hex `$numberLong` key), every
//!     prefix file carrying key min/max stats that share its prefix — the
//!     bracket a key-range reader prunes on;
//!   - the DV'd row is gone and the DV reabsorbed; every other row survives.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Int32Array, LargeStringArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey,
    PrimitiveLiteral, PrimitiveType, Schema, Struct, StructType, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{
    Catalog, CatalogBuilder, MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder, NamespaceIdent,
    TableCommit, TableCreation, TableIdent, TableUpdate,
};
use iceberg_compaction::config::{Config, KeyBucket};
use iceberg_compaction::engine::{compact_table, current_data_files, current_delete_files};
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;
use uuid::Uuid;

fn table_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(
                3,
                "_cdc",
                Type::Struct(StructType::new(vec![
                    NestedField::required(4, "op", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(5, "key", Type::Primitive(PrimitiveType::String)).into(),
                ])),
            )
            .into(),
            NestedField::optional(6, "_is_backfill", Type::Primitive(PrimitiveType::Boolean))
                .into(),
        ])
        .build()
        .unwrap()
}

fn table_arrow_schema(table: &Table) -> Arc<ArrowSchema> {
    Arc::new(iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema()).unwrap())
}

/// `{"$oid": "<prefix><pad>"}` — the mongo bronze key text (the hex run starts at byte 10).
fn oid_key(prefix: &str, n: u32) -> String {
    format!("{{\"$oid\": \"{prefix}{:021x}\"}}", n)
}

/// Rows: (id, payload, op, key, is_backfill), built against the table's schema.
fn batch(table: &Table, rows: &[(i32, &str, &str, Option<String>, Option<bool>)]) -> RecordBatch {
    let schema = table_arrow_schema(table);
    let cdc_fields = match schema.field_with_name("_cdc").unwrap().data_type() {
        DataType::Struct(fields) => fields.clone(),
        other => panic!("_cdc must be a struct, got {other:?}"),
    };
    let ops: Vec<&str> = rows.iter().map(|r| r.2).collect();
    let keys: Vec<Option<&str>> = rows.iter().map(|r| r.3.as_deref()).collect();
    let cdc = StructArray::from(vec![
        (
            cdc_fields[0].clone(),
            Arc::new(LargeStringArray::from(ops)) as arrow_array::ArrayRef,
        ),
        (
            cdc_fields[1].clone(),
            Arc::new(LargeStringArray::from(keys)) as arrow_array::ArrayRef,
        ),
    ]);
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>())),
        Arc::new(LargeStringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(cdc),
        Arc::new(BooleanArray::from(rows.iter().map(|r| r.4).collect::<Vec<_>>())),
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

/// (id, key) of every live row, sorted by id.
async fn read_rows(table: &Table) -> Vec<(i32, Option<String>)> {
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
        let ids = b
            .column(schema.index_of("id").unwrap())
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let cdc = b
            .column(schema.index_of("_cdc").unwrap())
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let keys = cdc
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            out.push((
                ids.value(i),
                keys.is_valid(i).then(|| keys.value(i).to_string()),
            ));
        }
    }
    out.sort();
    out
}

fn key_bound(file: &DataFile, field_id: i32, lower: bool) -> Option<String> {
    let bounds = if lower { file.lower_bounds() } else { file.upper_bounds() };
    bounds.get(&field_id).map(|d| match d.literal() {
        PrimitiveLiteral::String(s) => s.clone(),
        other => panic!("key bound is not a string: {other:?}"),
    })
}

#[tokio::test]
async fn key_bucket_rewrite_partitions_the_key_space_of_one_partition() {
    let warehouse = TempDir::new().unwrap();
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
        .add_partition_field(6, "_is_backfill", Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("bronze_coll".to_string())
                .schema(table_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    let ident = TableIdent::new(ns, "bronze_coll".to_string());

    // Two key-MIXED backfill files (three 3-hex prefixes interleaved, one NULL
    // key) + one CDC-partition file.
    let mk = |id: i32, prefix: &str| (id, "p", "r", Some(oid_key(prefix, id as u32)), Some(true));
    let file_a = write_data(
        &table,
        "bf-a",
        Some(true),
        batch(&table, &[
            mk(1, "69c"),
            mk(2, "69e"),
            mk(3, "6a1"),
            mk(4, "69c"),
            mk(5, "69e"),
            mk(6, "6a1"),
        ]),
    )
    .await;
    let file_b = write_data(
        &table,
        "bf-b",
        Some(true),
        batch(&table, &[
            mk(7, "6a1"),
            mk(8, "69c"),
            mk(9, "69e"),
            (10, "p", "r", None, Some(true)), // NULL key -> the `other` bucket
            mk(11, "69c"),
            (14, "p", "r", Some("{\"$numberLong\": \"5\"}".to_string()), Some(true)), // non-hex at byte 10 -> `other`
        ]),
    )
    .await;
    let file_c = write_data(
        &table,
        "cdc",
        None,
        batch(&table, &[
            (12, "p", "c", Some(oid_key("69c", 12)), None),
            (13, "p", "u", Some(oid_key("6a1", 13)), None),
        ]),
    )
    .await;
    let a_path = file_a[0].file_path().to_string();
    let c_path = file_c[0].file_path().to_string();
    let mut seed = file_a;
    seed.extend(file_b);
    seed.extend(file_c);
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(seed)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // A deletion vector dropping id=1 (position 0 of file A), the repair
    // doorway's delete shape.
    let mut positions = RoaringTreemap::new();
    positions.insert(0);
    let dv_path = format!("{}/dv-a.puffin", warehouse.path().to_str().unwrap());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            a_path.clone(),
            Struct::from_iter(vec![Some(Literal::bool(true))]),
            0,
        )
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(read_rows(&table).await.len(), 13, "12 backfill (one DV'd) + 2 CDC rows live");
    assert_eq!(current_delete_files(&table).await.unwrap().len(), 1);

    let cfg = Config {
        min_input_files: 1,
        delete_file_threshold: 0,
        rewrite_all: true,
        partition_filter: Some(("_is_backfill".to_string(), true)),
        key_bucket: Some(KeyBucket {
            leaf: "_cdc.key".to_string(),
            skip: 10,
            hex_len: 3,
            row_group_rows: 1024,
        }),
        ..Config::default()
    };
    let out = compact_table(&catalog, &ident, &cfg).await.unwrap();
    assert_eq!(out.groups_planned, 1, "one group = every file of the partition");
    assert_eq!(out.groups_rewritten, 1);
    assert_eq!(out.rewritten, 2, "both backfill files rewritten");
    assert_eq!(out.added, 4, "one output file per bucket: 69c, 69e, 6a1, other");

    let table = catalog.load_table(&ident).await.unwrap();
    let files = current_data_files(&table).await.unwrap();
    assert!(files.contains_key(&c_path), "the CDC-partition file is untouched");
    assert!(!files.contains_key(&a_path), "the rewritten input is gone");
    assert_eq!(current_delete_files(&table).await.unwrap().len(), 0, "the DV is reabsorbed");

    // Every PREFIX output file carries key min/max stats (the doorway brackets on them) that share ONE
    // 3-hex prefix at byte 10; the `other` file (NULL + non-hex keys) is the one whose bounds are absent
    // or not hex at byte 10.
    let key_field = table
        .metadata()
        .current_schema()
        .field_id_by_name("_cdc.key")
        .expect("the key leaf has a field id");
    let backfill_partition = Struct::from_iter(vec![Some(Literal::bool(true))]);
    let hex3 = |s: &str| s.len() >= 13 && s.as_bytes()[10..13].iter().all(u8::is_ascii_hexdigit);
    let mut prefixes: Vec<(String, u64)> = Vec::new();
    let mut other: Vec<u64> = Vec::new();
    for f in files.values() {
        if *f.partition() != backfill_partition {
            continue;
        }
        match (key_bound(f, key_field, true), key_bound(f, key_field, false)) {
            (Some(lo), Some(hi)) if hex3(&lo) && hex3(&hi) => {
                assert_eq!(&lo[10..13], &hi[10..13], "file {} spans prefixes {lo}..{hi}", f.file_path());
                prefixes.push((lo[10..13].to_string(), f.record_count()));
            }
            _ => other.push(f.record_count()),
        }
    }
    prefixes.sort();
    assert_eq!(
        prefixes,
        vec![("69c".to_string(), 3), ("69e".to_string(), 3), ("6a1".to_string(), 3)],
        "one output file per prefix, with key stats, 3 live rows each (id=1 was DV'd out of 69c)"
    );
    assert_eq!(other, vec![2], "ONE `other` file holding the NULL-key and the non-hex-key rows");

    // Rows: the DV'd id=1 stays gone, everything else survives, keys intact.
    let rows = read_rows(&table).await;
    let ids: Vec<i32> = rows.iter().map(|r| r.0).collect();
    assert_eq!(ids, (2..=14).collect::<Vec<i32>>());
    assert_eq!(rows[8].1, None, "id=10 keeps its NULL key");
    assert_eq!(rows[0].1.as_deref(), Some(oid_key("69e", 2).as_str()));
    assert_eq!(rows[12].1.as_deref(), Some("{\"$numberLong\": \"5\"}"), "id=14 keeps its non-hex key");
}


/// (id, `_is_backfill` column) of every live row, sorted by id — the re-home must move rows between
/// PARTITIONS without touching the column they carry.
async fn read_flags(table: &Table) -> Vec<(i32, Option<bool>)> {
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
        let ids = b
            .column(schema.index_of("id").unwrap())
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let flags = b
            .column(schema.index_of("_is_backfill").unwrap())
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), flags.is_valid(i).then(|| flags.value(i))));
        }
    }
    out.sort();
    out
}

/// A file written under an UNPARTITIONED spec (the loader era): no partition key at all.
async fn write_unpartitioned(table: &Table, prefix: &str, b: RecordBatch) -> Vec<DataFile> {
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
    let mut w = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
    w.write(b).await.unwrap();
    w.close().await.unwrap()
}

/// 2026-10-18 (device): a bronze table that was UNPARTITIONED first (the loader era wrote its op=r rows there) and
/// evolved the `_is_backfill` identity partition later. The repair doorway predicates on `_is_backfill = true`,
/// which can never exclude the unpartitioned files, so every repair commit read them all (≈ 6 GiB) — and its DVs
/// landed in them (167/167), so re-homing those rows anywhere ELSE would strand the versions the repair supersedes.
/// The pass: plan ONLY the unpartitioned files (`partition_filter_unpartitioned`), bucket-sort them and write every
/// output row under `_is_backfill = true` (`partition_override`) although the rows' own column says false; the
/// partitioned files (backfill + CDC) stay untouched; rows, keys and the column value survive unchanged.
#[tokio::test]
async fn key_bucket_rewrite_rehomes_unpartitioned_loader_era_files_into_the_override_partition() {
    let warehouse = TempDir::new().unwrap();
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
    // Spec 0: unpartitioned, as the loader era left it.
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("bronze_device".to_string())
                .schema(table_schema())
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    let ident = TableIdent::new(ns, "bronze_device".to_string());
    assert!(table.metadata().default_partition_spec().is_unpartitioned());

    // Two loader-era files: key-MIXED across three prefixes, the `_is_backfill` COLUMN false (the loader never set
    // it), one NULL key — exactly device's spec-0 shape.
    let mk = |id: i32, prefix: &str| (id, "p", "r", Some(oid_key(prefix, id as u32)), Some(false));
    let s0_a = write_unpartitioned(
        &table,
        "s0-a",
        batch(&table, &[mk(1, "69c"), mk(2, "69e"), mk(3, "6a1"), mk(4, "69c"), mk(5, "69e")]),
    )
    .await;
    let s0_b = write_unpartitioned(
        &table,
        "s0-b",
        batch(&table, &[mk(6, "6a1"), mk(7, "69c"), (8, "p", "r", None, Some(false)), mk(9, "69e")]),
    )
    .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(s0_a.iter().chain(s0_b.iter()).cloned().collect::<Vec<_>>())
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let s0_paths: Vec<String> =
        s0_a.iter().chain(s0_b.iter()).map(|f| f.file_path().to_string()).collect();

    // Evolve: the `_is_backfill` identity partition becomes the default spec (spec 1).
    let evolve = TableCommit::builder()
        .ident(ident.clone())
        .updates(vec![
            TableUpdate::AddSpec {
                spec: UnboundPartitionSpec::builder()
                    .add_partition_field(6, "_is_backfill", Transform::Identity)
                    .unwrap()
                    .build(),
            },
            TableUpdate::SetDefaultSpec { spec_id: -1 },
        ])
        .requirements(vec![])
        .build();
    let table = catalog.update_table(evolve).await.unwrap();
    assert!(!table.metadata().default_partition_spec().is_unpartitioned());

    // Under spec 1: one backfill-partition file (a later repair baseline) and one CDC file (partition NULL).
    let bf = write_data(
        &table,
        "bf",
        Some(true),
        batch(&table, &[(10, "p", "R", Some(oid_key("69c", 10)), Some(true))]),
    )
    .await;
    let cdc = write_data(
        &table,
        "cdc",
        None,
        batch(&table, &[(11, "p", "u", Some("{id={\"$oid\": \"69e00000000000000000000b\"}}".to_string()), Some(false))]),
    )
    .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(bf.iter().chain(cdc.iter()).cloned().collect::<Vec<_>>())
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let bf_path = bf[0].file_path().to_string();
    let cdc_path = cdc[0].file_path().to_string();
    let before_rows = read_rows(&table).await;
    let before_flags = read_flags(&table).await;
    assert_eq!(before_rows.len(), 11);

    let cfg = Config {
        min_input_files: 1,
        delete_file_threshold: 0,
        rewrite_all: true,
        partition_filter: None,
        partition_filter_unpartitioned: true,
        partition_override: Some(("_is_backfill".to_string(), true)),
        key_bucket: Some(KeyBucket {
            leaf: "_cdc.key".to_string(),
            skip: 10,
            hex_len: 3,
            row_group_rows: 1024,
        }),
        ..Config::default()
    };
    let out = compact_table(&catalog, &ident, &cfg).await.unwrap();
    assert_eq!(out.groups_planned, 1, "one group = the unpartitioned files only");
    assert_eq!(out.groups_rewritten, 1);
    assert_eq!(out.rewritten, 2, "both loader-era files rewritten");
    assert_eq!(out.added, 4, "one output file per bucket: 69c, 69e, 6a1, other");

    let table = catalog.load_table(&ident).await.unwrap();
    let files = current_data_files(&table).await.unwrap();
    assert!(files.contains_key(&bf_path), "the backfill-partition file is untouched");
    assert!(files.contains_key(&cdc_path), "the CDC-partition file is untouched");
    for p in &s0_paths {
        assert!(!files.contains_key(p), "the loader-era input {p} is gone");
    }
    // Every output file sits in the OVERRIDE partition (`_is_backfill = true`), one 3-hex prefix per file, with
    // key bounds the doorway's bracket can prune on; the `other` file holds the NULL-key row.
    let key_field = table
        .metadata()
        .current_schema()
        .field_id_by_name("_cdc.key")
        .expect("the key leaf has a field id");
    let true_partition = Struct::from_iter(vec![Some(Literal::bool(true))]);
    let mut prefixes: Vec<(String, u64)> = Vec::new();
    let mut other: Vec<u64> = Vec::new();
    let hex3 = |s: &str| s.len() >= 13 && s.as_bytes()[10..13].iter().all(u8::is_ascii_hexdigit);
    for (path, f) in files.iter() {
        if *path == bf_path || *path == cdc_path {
            continue;
        }
        assert_eq!(*f.partition(), true_partition, "output {path} is not in the override partition");
        match (key_bound(f, key_field, true), key_bound(f, key_field, false)) {
            (Some(lo), Some(hi)) if hex3(&lo) && hex3(&hi) => {
                assert_eq!(&lo[10..13], &hi[10..13], "file {path} spans prefixes {lo}..{hi}");
                prefixes.push((lo[10..13].to_string(), f.record_count()));
            }
            _ => other.push(f.record_count()),
        }
    }
    prefixes.sort();
    assert_eq!(
        prefixes,
        vec![("69c".to_string(), 3), ("69e".to_string(), 3), ("6a1".to_string(), 2)],
        "one re-homed output file per prefix, with key stats"
    );
    assert_eq!(other, vec![1], "ONE `other` file holding the NULL-key row");

    // Rows re-homed, never edited: the same ids and keys, and the `_is_backfill` COLUMN still false on them.
    assert_eq!(read_rows(&table).await, before_rows, "every row and key survives the re-home");
    assert_eq!(read_flags(&table).await, before_flags, "the column value is untouched — only the partition moved");
}
