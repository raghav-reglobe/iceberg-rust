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

//! EQUALITY-delete reabsorption through `compact_table`, DV-parity semantics:
//!
//! 1. APPLY — rows matched by an equality delete (delete sequence number >
//!    the data file's data sequence number) must be ABSENT from the rewritten
//!    output (the scan-based read path applies them, same as DVs).
//! 2. REMOVE — an equality-delete file is reabsorbed (removed in the same
//!    `Replace` commit) only when EVERY data file it applies to is rewritten
//!    in that commit; otherwise it must be RETAINED (removing it would
//!    resurrect deleted rows in the unrewritten files).
//!
//! Assertions go through MANIFEST ENTRIES (live files + record counts), not
//! snapshot-summary totals — summaries have been cosmetically wrong before.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList,
    NestedField, PartitionKey, PrimitiveType, Schema, Struct, Transform, Type,
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
use iceberg_compaction::config::Config;
use iceberg_compaction::engine::compact_table;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use uuid::Uuid;

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

fn arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "payload", DataType::Utf8, true),
    ]))
}

/// Semi-incompressible payload so parquet size tracks raw size (the test
/// writer uses UNCOMPRESSED defaults) — file size drives candidacy.
fn payload(seed: u64, len: usize) -> String {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(97);
    let mut out = String::with_capacity(len);
    while out.len() < len {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push_str(&format!("{x:016x}"));
    }
    out.truncate(len);
    out
}

fn batch(rows: &[(i32, String)]) -> RecordBatch {
    RecordBatch::try_new(arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

async fn write_data(table: &Table, prefix: &str, b: RecordBatch) -> Vec<DataFile> {
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
        .build(None)
        .await
        .unwrap();
    w.write(b).await.unwrap();
    w.close().await.unwrap()
}

/// pk-only equality-delete file whose tuples are `ids` (the crate's own
/// equality-delete writer — the same idiom RowDelta tests use).
async fn write_eq_delete(table: &Table, ids: &[i32]) -> Vec<DataFile> {
    let pk_id = table
        .metadata()
        .current_schema()
        .field_id_by_name("id")
        .unwrap();
    let config =
        EqualityDeleteWriterConfig::new(vec![pk_id], table.metadata().current_schema().clone())
            .unwrap();
    let delete_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), delete_schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(
            format!("eqdel-{}", Uuid::now_v7()),
            None,
            DataFileFormat::Parquet,
        ),
    );
    let mut w = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(None)
        .await
        .unwrap();
    let rows: Vec<(i32, String)> = ids.iter().map(|i| (*i, String::new())).collect();
    w.write(batch(&rows)).await.unwrap();
    w.close().await.unwrap()
}

async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap();
}

async fn commit_eq_delete(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) {
    let tx = Transaction::new(table);
    tx.row_delta()
        .add_delete_files(files)
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap();
}

async fn read_ids(table: &Table) -> Vec<i32> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id"])
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
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        out.extend(ids.values().iter().copied());
    }
    out.sort_unstable();
    out
}

/// Live manifest entries of the given content type: (file_path, record_count).
async fn live_entries(table: &Table, content: ManifestContentType) -> Vec<(String, u64)> {
    let Some(snap) = table.metadata().current_snapshot() else {
        return Vec::new();
    };
    let bytes = table
        .file_io()
        .new_input(snap.manifest_list())
        .unwrap()
        .read()
        .await
        .unwrap();
    let ml = ManifestList::parse_with_version(&bytes, table.metadata().format_version()).unwrap();
    let mut out = Vec::new();
    for mf in ml.entries() {
        if mf.content != content {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                out.push((
                    e.data_file().file_path().to_string(),
                    e.data_file().record_count(),
                ));
            }
        }
    }
    out.sort();
    out
}

async fn fresh_catalog_table(warehouse: &TempDir) -> (impl Catalog, Table, TableIdent) {
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
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("t".to_string())
                .schema(schema)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    let ident = TableIdent::new(ns, "t".to_string());
    (catalog, table, ident)
}

/// Full pass over an equality-delete-bearing table (delete_file_threshold=1,
/// min_input_files=1): the rewritten output must have the deletes APPLIED
/// (keys 2 and 4 absent) and the fully-superseded equality-delete file must
/// be REMOVED from the commit (zero live eq-delete entries afterwards).
#[tokio::test]
async fn full_rewrite_applies_and_reabsorbs_equality_deletes() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, table, ident) = fresh_catalog_table(&warehouse).await;

    // Two data files carrying keys {1..10}.
    let a = write_data(
        &table,
        "a",
        batch(
            &(1..=5)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
        ),
    )
    .await;
    append(&catalog, &table, a).await;
    let table = catalog.load_table(&ident).await.unwrap();
    let b = write_data(
        &table,
        "b",
        batch(
            &(6..=10)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
        ),
    )
    .await;
    append(&catalog, &table, b).await;

    // Equality delete (key column `id`, higher sequence number) deleting {2, 4}.
    let table = catalog.load_table(&ident).await.unwrap();
    let eq = write_eq_delete(&table, &[2, 4]).await;
    commit_eq_delete(&catalog, &table, eq).await;

    // Pre-compaction: the deletes mask both keys at read.
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(read_ids(&table).await, vec![1, 3, 5, 6, 7, 8, 9, 10]);

    // Full pass: every file with >=1 bound delete is a candidate.
    let cfg = Config {
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();

    // APPLY: the rewritten output contains no rows for the deleted keys, and
    // the live manifest record counts total exactly the 8 survivors.
    let data = live_entries(&table, ManifestContentType::Data).await;
    let total_records: u64 = data.iter().map(|(_, rc)| rc).sum();
    assert_eq!(
        total_records, 8,
        "rewritten output must have the equality deletes applied (manifest record counts): {data:?}"
    );

    // REMOVE: every data file the delete applied to was rewritten in this
    // commit, so the equality-delete file must be gone from the manifests.
    let deletes = live_entries(&table, ManifestContentType::Deletes).await;
    assert!(
        deletes.is_empty(),
        "fully-superseded equality-delete file must be removed: {deletes:?}"
    );

    // A fresh scan returns exactly the 8 surviving keys.
    assert_eq!(read_ids(&table).await, vec![1, 3, 5, 6, 7, 8, 9, 10]);
}

/// Partial-rewrite safety: when the plan does NOT rewrite one of the data
/// files an equality delete applies to (here: a file sized into the optimal
/// window with the delete threshold set high), the delete file must be
/// RETAINED — removing it would resurrect its deleted rows in the
/// unrewritten file — and a fresh scan must still return the correct rows.
#[tokio::test]
async fn partial_rewrite_retains_equality_delete() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, table, ident) = fresh_catalog_table(&warehouse).await;

    // File A: tiny (5 rows / 64 B payloads) -> undersized candidate.
    let a = write_data(
        &table,
        "a",
        batch(
            &(1..=5)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
        ),
    )
    .await;
    append(&catalog, &table, a).await;
    // File B: sized INTO the [min, max) window -> NOT a candidate.
    // 60 rows x ~600 B ~= 36 KB uncompressed.
    let table = catalog.load_table(&ident).await.unwrap();
    let b_rows: Vec<(i32, String)> = (100..160).map(|i| (i, payload(i as u64, 600))).collect();
    let b = write_data(&table, "b", batch(&b_rows)).await;
    append(&catalog, &table, b).await;

    // One equality-delete file hitting BOTH files: key 2 (in A), key 101 (in B).
    let table = catalog.load_table(&ident).await.unwrap();
    let eq = write_eq_delete(&table, &[2, 101]).await;
    commit_eq_delete(&catalog, &table, eq).await;

    let table = catalog.load_table(&ident).await.unwrap();
    let pre = read_ids(&table).await;
    assert!(!pre.contains(&2) && !pre.contains(&101), "{pre:?}");
    assert_eq!(pre.len(), 63, "5 + 60 rows minus the two deletes");

    // Size window keeps B out; high delete threshold keeps B's bound delete
    // from making it a candidate. Only A is rewritten.
    let target: u64 = 32 * 1024;
    let cfg = Config {
        target_file_size_bytes: target,
        min_file_size_bytes: target * 3 / 4, // 24 KB
        max_file_size_bytes: target * 9 / 5, // 57.6 KB
        min_input_files: 1,
        delete_file_threshold: 10,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let data = live_entries(&table, ManifestContentType::Data).await;
    assert!(
        data.iter().any(|(p, _)| p.contains("/b-")),
        "precondition: B must NOT have been rewritten: {data:?}"
    );
    assert!(
        !data.iter().any(|(p, _)| p.contains("/a-")),
        "precondition: A must have been rewritten: {data:?}"
    );

    // The delete still applies to unrewritten B -> it must be RETAINED.
    assert_eq!(
        live_entries(&table, ManifestContentType::Deletes)
            .await
            .len(),
        1,
        "equality delete must be retained while an unrewritten file still needs it"
    );

    // Reads stay correct: 2 stayed applied in A's rewrite, 101 still masked in B.
    let post = read_ids(&table).await;
    assert!(!post.contains(&2) && !post.contains(&101), "{post:?}");
    assert_eq!(post.len(), 63);
}

/// The FULLY-SUPERSEDED shape — the reabsorption gap this suite exists for.
///
/// A DELETE-then-INSERT replacement leaves the OLD data file 100% masked by
/// an equality delete, with the replacement rows appended at a higher
/// sequence number. When the replacement file is not itself a rewrite
/// candidate, the old file forms a group of its own, and that group reads
/// ZERO live rows through the delete filter.
///
/// PRE-FIX FAILURE MODE (proven against the unfixed engine): `compact_table`
/// treated an empty group read as "nothing to do" (`added.is_empty()` ->
/// skip the group), so the fully-dead data file was never removed and the
/// equality-delete file masking it was RETAINED — on every pass, forever.
/// Reads stayed correct (the retained file keeps its low sequence number, so
/// the delete keeps applying), but the delete pile never shrank: zero
/// deletes applied to any rewritten output, zero delete files removed, while
/// undersized live files still binpacked by size. The fix removes the
/// fully-dead files (every input carried bound deletes and read empty
/// through the delete filter) and reabsorbs their now-superseded deletes in
/// the same `Replace` commit — which therefore may remove files while adding
/// NONE.
#[tokio::test]
async fn fully_superseded_file_and_its_equality_delete_are_reabsorbed() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, table, ident) = fresh_catalog_table(&warehouse).await;

    // Round-1 baseline: file A, keys 1..5 (tiny -> candidate under any window).
    let a = write_data(
        &table,
        "a",
        batch(
            &(1..=5)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
        ),
    )
    .await;
    append(&catalog, &table, a).await;

    // Round-2 replacement: DELETE the whole chunk...
    let table = catalog.load_table(&ident).await.unwrap();
    let eq = write_eq_delete(&table, &[1, 2, 3, 4, 5]).await;
    commit_eq_delete(&catalog, &table, eq).await;

    // ...then INSERT the fresh baseline: file A2, same keys, HIGHER sequence
    // number, sized INTO the optimal window so it is NOT a candidate — the
    // dead file A must stand alone in its group.
    let table = catalog.load_table(&ident).await.unwrap();
    let a2_rows: Vec<(i32, String)> = (1..=5)
        .map(|i| (i, payload(1000 + i as u64, 6 * 1024)))
        .collect();
    let a2 = write_data(&table, "a2", batch(&a2_rows)).await;
    append(&catalog, &table, a2).await;

    // Pre-compaction: A is 100% masked; only A2's rows serve.
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(read_ids(&table).await, vec![1, 2, 3, 4, 5]);

    // A2 (~30 KB) sits inside [min, max); A (tiny, delete-bearing) is a
    // candidate and groups alone.
    let target: u64 = 32 * 1024;
    let cfg = Config {
        target_file_size_bytes: target,
        min_file_size_bytes: target * 3 / 4, // 24 KB
        max_file_size_bytes: target * 9 / 5, // 57.6 KB
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let data = live_entries(&table, ManifestContentType::Data).await;
    assert!(
        data.iter().any(|(p, _)| p.contains("/a2-")),
        "precondition: the optimally-sized replacement file must not be rewritten: {data:?}"
    );

    // REMOVE (the gap): the fully-masked file must be gone...
    assert!(
        !data.iter().any(|(p, _)| p.contains("/a-")),
        "fully-superseded data file must be removed, not retained forever: {data:?}"
    );
    // ...and with every file it applied to removed, so must the delete file.
    let deletes = live_entries(&table, ManifestContentType::Deletes).await;
    assert!(
        deletes.is_empty(),
        "equality delete masking only removed files must be reabsorbed: {deletes:?}"
    );

    // Live manifest record counts: exactly the 5 replacement rows.
    let total_records: u64 = data.iter().map(|(_, rc)| rc).sum();
    assert_eq!(total_records, 5, "{data:?}");

    // Reads unchanged.
    assert_eq!(read_ids(&table).await, vec![1, 2, 3, 4, 5]);
}

// ---------------------------------------------------------------------------
// Partition-scoped equality deletes: an identity partition on a NULLABLE
// boolean column whose live values are {true, NULL} — the append-only CDC
// layout where backfill rows land in [true] and streamed rows in [null] —
// with the delete file scoped to the [true] partition, plus a data file
// appended AFTER the delete (higher sequence number, so the delete must not
// bind to it or block its own removal).
// ---------------------------------------------------------------------------

fn part_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "payload", DataType::Utf8, true),
        field(3, "bf", DataType::Boolean, true),
    ]))
}

fn part_batch(rows: &[(i32, String)], bf: Option<bool>) -> RecordBatch {
    RecordBatch::try_new(part_arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(arrow_array::BooleanArray::from(vec![bf; rows.len()])),
    ])
    .unwrap()
}

async fn write_part_data(
    table: &Table,
    prefix: &str,
    b: RecordBatch,
    key: PartitionKey,
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
        .build(Some(key))
        .await
        .unwrap();
    w.write(b).await.unwrap();
    w.close().await.unwrap()
}

/// pk-only equality-delete file scoped to `key`'s partition.
async fn write_part_eq_delete(table: &Table, ids: &[i32], key: PartitionKey) -> Vec<DataFile> {
    let pk_id = table
        .metadata()
        .current_schema()
        .field_id_by_name("id")
        .unwrap();
    let config =
        EqualityDeleteWriterConfig::new(vec![pk_id], table.metadata().current_schema().clone())
            .unwrap();
    let delete_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), delete_schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(
            format!("eqdel-{}", Uuid::now_v7()),
            None,
            DataFileFormat::Parquet,
        ),
    );
    let mut w = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(key))
        .await
        .unwrap();
    let rows: Vec<(i32, String)> = ids.iter().map(|i| (*i, String::new())).collect();
    w.write(part_batch(&rows, Some(true))).await.unwrap();
    w.close().await.unwrap()
}

/// Full pass over an identity-partitioned (nullable boolean) table with a
/// partition-scoped equality delete: the delete applies to the lower-sequence
/// files of its partition only, the rewrite must apply it there, and it must
/// be removed once every file it applies to is rewritten — including when a
/// HIGHER-sequence data file (appended after the delete) is rewritten in the
/// same pass.
#[tokio::test]
async fn partitioned_rewrite_applies_and_reabsorbs_partition_scoped_equality_delete() {
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
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "bf", Type::Primitive(PrimitiveType::Boolean)).into(),
        ])
        .build()
        .unwrap();
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(3, "bf".to_string(), Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("t".to_string())
                .schema(schema)
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    let ident = TableIdent::new(ns, "t".to_string());

    let key_true = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::bool(true))]),
    );
    let key_null = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([None]),
    );

    // A: keys 1..5 in [true] (backfill rows); B: keys 6..10 in [null] (CDC rows).
    let a = write_part_data(
        &table,
        "a",
        part_batch(
            &(1..=5)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
            Some(true),
        ),
        key_true.clone(),
    )
    .await;
    append(&catalog, &table, a).await;
    let table = catalog.load_table(&ident).await.unwrap();
    let b = write_part_data(
        &table,
        "b",
        part_batch(
            &(6..=10)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
            None,
        ),
        key_null.clone(),
    )
    .await;
    append(&catalog, &table, b).await;

    // Equality delete scoped to the [true] partition, deleting keys {2, 4}.
    let table = catalog.load_table(&ident).await.unwrap();
    let eq = write_part_eq_delete(&table, &[2, 4], key_true.clone()).await;
    commit_eq_delete(&catalog, &table, eq).await;

    // C: keys 11..15 in [true], appended AFTER the delete — higher sequence
    // number, so the delete does not apply to it.
    let table = catalog.load_table(&ident).await.unwrap();
    let c = write_part_data(
        &table,
        "c",
        part_batch(
            &(11..=15)
                .map(|i| (i, payload(i as u64, 64)))
                .collect::<Vec<_>>(),
            Some(true),
        ),
        key_true,
    )
    .await;
    append(&catalog, &table, c).await;

    // Pre-compaction: 15 rows minus the two deletes.
    let table = catalog.load_table(&ident).await.unwrap();
    let pre = read_ids(&table).await;
    assert!(!pre.contains(&2) && !pre.contains(&4), "{pre:?}");
    assert_eq!(pre.len(), 13);

    let cfg = Config {
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();

    // APPLY: manifest record counts total exactly the 13 survivors.
    let data = live_entries(&table, ManifestContentType::Data).await;
    let total_records: u64 = data.iter().map(|(_, rc)| rc).sum();
    assert_eq!(
        total_records, 13,
        "partition-scoped equality delete must be applied on rewrite: {data:?}"
    );

    // REMOVE: the only file the delete applied to (A) was rewritten, so the
    // delete file must be gone.
    let deletes = live_entries(&table, ManifestContentType::Deletes).await;
    assert!(
        deletes.is_empty(),
        "fully-superseded partition-scoped equality delete must be removed: {deletes:?}"
    );

    let post = read_ids(&table).await;
    assert!(!post.contains(&2) && !post.contains(&4), "{post:?}");
    assert_eq!(post.len(), 13);
}
