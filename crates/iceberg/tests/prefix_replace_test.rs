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

//! ATOMIC KEY-PREFIX REPLACE via scan + consolidated deletion vectors — the
//! envelope-shaped (document-store) bronze variant of the atomic replace:
//! no top-level pk column exists, and the chunk scope is a PREFIX of a
//! nested string key (`_cdc.key`), which no equality-delete file can
//! express. Prior rows are deleted by scanning the pinned snapshot for
//! `partition ∧ starts_with(key, prefix) ∧ op ∈ {R,r}` and writing ONE
//! consolidated V3 deletion vector per affected data file (prior live DV
//! positions unioned in, prior delete files superseded) — committed with
//! the replacement appends as ONE `RowDelta` snapshot.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{BooleanArray, LargeStringArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::atomic_replace::atomic_partition_replace_prefix;
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

fn oid_prefix() -> String {
    "{\"$oid\": \"aa".to_string()
}

async fn run_replace(
    catalog: &impl Catalog,
    ident: &TableIdent,
    rows: &[(&str, &str, &str, Option<bool>)],
) -> iceberg::atomic_replace::ReplaceOutcome {
    let table = catalog.load_table(ident).await.unwrap();
    let batches = if rows.is_empty() {
        vec![]
    } else {
        vec![batch(&table, rows)]
    };
    atomic_partition_replace_prefix(
        catalog,
        ident,
        "_is_backfill",
        true,
        "_cdc.key",
        &oid_prefix(),
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        batches,
        3,
        false,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn prefix_replace_deletes_prefix_scoped_and_appends() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let before_snapshots = {
        let t = catalog.load_table(&ident).await.unwrap();
        t.metadata().snapshots().count()
    };

    // Replace the `aa` prefix chunk: aa01 upgraded, aa02 VANISHED at source
    // (not re-inserted), aa03 newly appeared.
    let outcome = run_replace(&catalog, &ident, &[
        ("r-new-aa01", "R", "{\"$oid\": \"aa01\"}", Some(true)),
        ("r-new-aa03", "R", "{\"$oid\": \"aa03\"}", Some(true)),
    ])
    .await;
    assert_eq!(outcome.rows_appended, 2);
    assert_eq!(outcome.delete_tuples, 2); // aa01 + aa02 (prefix-scoped)
    assert_eq!(outcome.attempts, 1);

    let table = catalog.load_table(&ident).await.unwrap();
    // Exactly ONE new snapshot (delete + append atomic).
    assert_eq!(table.metadata().snapshots().count(), before_snapshots + 1);
    assert_eq!(read_rows(&table).await, vec![
        ("c-aa01".into(), "c".into(), "{\"$oid\": \"aa01\"}".into()), // CDC survives
        ("c-zz01".into(), "c".into(), "{\"$oid\": \"zz01\"}".into()),
        (
            "r-new-aa01".into(),
            "R".into(),
            "{\"$oid\": \"aa01\"}".into()
        ),
        (
            "r-new-aa03".into(),
            "R".into(),
            "{\"$oid\": \"aa03\"}".into()
        ),
        (
            "r-old-ab99".into(),
            "r".into(),
            "{\"$oid\": \"ab99\"}".into()
        ), // other prefix
        (
            "r-old-bb01".into(),
            "r".into(),
            "{\"$oid\": \"bb01\"}".into()
        ),
    ]);
}

#[tokio::test]
async fn prefix_replace_is_replay_idempotent() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let rows: &[(&str, &str, &str, Option<bool>)] = &[
        ("r-new-aa01", "R", "{\"$oid\": \"aa01\"}", Some(true)),
        ("r-new-aa03", "R", "{\"$oid\": \"aa03\"}", Some(true)),
    ];
    run_replace(&catalog, &ident, rows).await;
    let table = catalog.load_table(&ident).await.unwrap();
    let first = read_rows(&table).await;

    // Replay: the second run deletes the FIRST run's appends (they match
    // the prefix) and re-appends identical content.
    let outcome = run_replace(&catalog, &ident, rows).await;
    assert_eq!(outcome.delete_tuples, 2);
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(read_rows(&table).await, first);

    // V3 invariant: at most ONE live DV per referenced data file.
    let live = live_delete_entries(&table).await;
    let mut seen = std::collections::HashSet::new();
    for (path, format) in &live {
        assert_eq!(*format, DataFileFormat::Puffin, "only DVs expected: {path}");
        assert!(seen.insert(path.clone()), "duplicate DV file {path}");
    }
}

#[tokio::test]
async fn prefix_replace_zero_rows_still_deletes() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    // The prefix's docs all vanished at source: delete-only commit.
    let outcome = run_replace(&catalog, &ident, &[]).await;
    assert_eq!(outcome.rows_appended, 0);
    assert_eq!(outcome.delete_tuples, 2);

    let table = catalog.load_table(&ident).await.unwrap();
    let rows = read_rows(&table).await;
    assert!(
        !rows
            .iter()
            .any(|(_, op, key)| { (op == "r" || op == "R") && key.starts_with(&oid_prefix()) })
    );
    assert_eq!(rows.len(), 4); // 2 CDC + ab99 + bb01
}

#[tokio::test]
async fn prefix_replace_consolidates_prior_dvs() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    let backfill_files = seed(&catalog, &ident).await;
    assert_eq!(backfill_files.len(), 1);
    let seed_file = backfill_files[0].file_path().to_string();

    // A PRIOR deletion vector on the seed file: position 3 (bb01) deleted —
    // e.g. an earlier engine's delete. The prefix replace must UNION it
    // into its consolidated DV, or bb01 resurrects.
    let table = catalog.load_table(&ident).await.unwrap();
    let mut bitmap = RoaringTreemap::new();
    bitmap.insert(3);
    let dv_path = format!(
        "{}/data/{}-deletes.puffin",
        table.metadata().location(),
        Uuid::now_v7()
    );
    let dv_file = DeleteVector::new(bitmap)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            seed_file.clone(),
            Struct::from_iter(vec![Some(Literal::bool(true))]),
            table.metadata().default_partition_spec_id(),
        )
        .await
        .unwrap();
    let prior_dv_path = dv_file.file_path().to_string();
    let tx = Transaction::new(&table);
    tx.row_delta()
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let outcome = run_replace(&catalog, &ident, &[(
        "r-new-aa01",
        "R",
        "{\"$oid\": \"aa01\"}",
        Some(true),
    )])
    .await;
    assert_eq!(outcome.delete_tuples, 2); // aa01 + aa02

    let table = catalog.load_table(&ident).await.unwrap();
    // bb01 stays deleted (prior DV consolidated), aa old rows gone.
    assert_eq!(read_rows(&table).await, vec![
        ("c-aa01".into(), "c".into(), "{\"$oid\": \"aa01\"}".into()),
        ("c-zz01".into(), "c".into(), "{\"$oid\": \"zz01\"}".into()),
        (
            "r-new-aa01".into(),
            "R".into(),
            "{\"$oid\": \"aa01\"}".into()
        ),
        (
            "r-old-ab99".into(),
            "r".into(),
            "{\"$oid\": \"ab99\"}".into()
        ),
    ]);
    // Exactly ONE live DV references the seed file; the prior DV file is
    // superseded (not a live entry anymore).
    let live = live_delete_entries(&table).await;
    let seed_dvs: Vec<_> = live.iter().filter(|(p, _)| *p != prior_dv_path).collect();
    assert_eq!(live.len(), 1, "one consolidated DV expected: {live:?}");
    assert_eq!(seed_dvs.len(), 1, "prior DV must be superseded: {live:?}");
}

#[tokio::test]
async fn prefix_replace_local_parquet_input() {
    // The giant-chunk mode: rows via a LOCAL parquet path (streamed) —
    // same semantics as the batches mode; a ZERO-ROW file still deletes
    // (the prefix's docs vanished at source — delete-only commit).
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;
    let table = catalog.load_table(&ident).await.unwrap();

    let b = batch(&table, &[(
        "r-file-aa01",
        "R",
        "{\"$oid\": \"aa01\"}",
        Some(true),
    )]);
    let path = warehouse.path().join("chunk-input.parquet");
    let mut w = parquet::arrow::arrow_writer::ArrowWriter::try_new(
        std::fs::File::create(&path).unwrap(),
        b.schema(),
        None,
    )
    .unwrap();
    w.write(&b).unwrap();
    w.close().unwrap();

    let out = atomic_partition_replace_prefix(
        &catalog,
        &ident,
        "_is_backfill",
        true,
        "_cdc.key",
        &oid_prefix(),
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        iceberg::atomic_replace::ReplaceInput::LocalParquet(path.to_str().unwrap().to_string()),
        3,
        false,
    )
    .await
    .unwrap();
    assert_eq!((out.rows_appended, out.delete_tuples), (1, 2)); // aa01+aa02 deleted
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(read_rows(&table).await, vec![
        ("c-aa01".into(), "c".into(), "{\"$oid\": \"aa01\"}".into()),
        ("c-zz01".into(), "c".into(), "{\"$oid\": \"zz01\"}".into()),
        (
            "r-file-aa01".into(),
            "R".into(),
            "{\"$oid\": \"aa01\"}".into()
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

    // Zero-row file: delete-only (removes the row appended above).
    let empty = warehouse.path().join("empty-input.parquet");
    let mut w = parquet::arrow::arrow_writer::ArrowWriter::try_new(
        std::fs::File::create(&empty).unwrap(),
        b.schema(),
        None,
    )
    .unwrap();
    w.close().unwrap();
    let out = atomic_partition_replace_prefix(
        &catalog,
        &ident,
        "_is_backfill",
        true,
        "_cdc.key",
        &oid_prefix(),
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        iceberg::atomic_replace::ReplaceInput::LocalParquet(empty.to_str().unwrap().to_string()),
        3,
        false,
    )
    .await
    .unwrap();
    assert_eq!((out.rows_appended, out.delete_tuples), (0, 1));
    let table = catalog.load_table(&ident).await.unwrap();
    let rows = read_rows(&table).await;
    assert!(
        !rows
            .iter()
            .any(|(_, op, key)| { (op == "r" || op == "R") && key.starts_with(&oid_prefix()) })
    );
}

#[tokio::test]
async fn prefix_replace_rejects_equality_deletes() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    // Attach an equality delete to the table (the MySQL-mode delete class).
    let table = catalog.load_table(&ident).await.unwrap();
    let after_id = table
        .metadata()
        .current_schema()
        .field_id_by_name("after")
        .unwrap();
    let config =
        EqualityDeleteWriterConfig::new(vec![after_id], table.metadata().current_schema().clone())
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
        .build(Some(partition_key(&table, Some(true))))
        .await
        .unwrap();
    let eq_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            arrow_schema::Field::new("after", DataType::LargeUtf8, true).with_metadata(
                HashMap::from([(
                    parquet::arrow::PARQUET_FIELD_ID_META_KEY.to_string(),
                    after_id.to_string(),
                )]),
            ),
        ])),
        vec![Arc::new(LargeStringArray::from(vec!["r-old-aa01"])) as arrow_array::ArrayRef],
    )
    .unwrap();
    w.write(eq_batch).await.unwrap();
    let eq_files = w.close().await.unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let tx = Transaction::new(&table);
    tx.row_delta()
        .add_delete_files(eq_files)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let err = atomic_partition_replace_prefix(
        &catalog,
        &ident,
        "_is_backfill",
        true,
        "_cdc.key",
        &oid_prefix(),
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        vec![batch(&table, &[(
            "r-new-aa01",
            "R",
            "{\"$oid\": \"aa01\"}",
            Some(true),
        )])],
        3,
        false,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("EQUALITY delete"),
        "unexpected error: {err}"
    );
}
