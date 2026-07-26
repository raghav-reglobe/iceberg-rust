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

//! ATOMIC NUMERIC-KEY-RANGE REPLACE — the [`atomic_partition_replace_prefix`]
//! sibling for document stores whose keys are Int64 stringified into the
//! nested key column (`_cdc.key = "28753650"`). A numeric range is
//! inexpressible as a string prefix (lexicographic ≠ numeric across digit
//! lengths), so the row-wise match parses each key; planning bounds engage
//! only for same-digit-length ranges (a lex superset of the numeric range).
//! Mixed-length seeds below deliberately stress both paths.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::atomic_replace::atomic_partition_replace_key_range;
use iceberg::spec::{
    DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList, NestedField,
    PartitionKey, PrimitiveType, Schema, Struct, StructType, Transform, Type, UnboundPartitionSpec,
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
    TableCreation, TableIdent,
};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use uuid::Uuid;

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

fn table_arrow_schema(table: &Table) -> Arc<ArrowSchema> {
    Arc::new(iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema()).unwrap())
}

/// Rows: (after, op, key, is_backfill).
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
            Arc::new(StringArray::from(ops)) as arrow_array::ArrayRef,
        ),
        (
            key_field,
            Arc::new(StringArray::from(keys)) as arrow_array::ArrayRef,
        ),
    ]);
    RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from(
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
) -> Vec<iceberg::spec::DataFile> {
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
            .downcast_ref::<StringArray>()
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
            .downcast_ref::<StringArray>()
            .unwrap();
        let keys = cdc
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
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
                .name("bronze_numkey_t".to_string())
                .schema(table_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    (catalog, TableIdent::new(ns, "bronze_numkey_t".to_string()))
}

/// Seed with MIXED-digit-length numeric keys spanning the range boundary
/// plus CDC rows in the null partition. Keys: 5, 42, 999 (in [1,999]) and
/// 1000007 (outside); "1500" is inside the LEXICOGRAPHIC window of
/// same-length bounds like ["1000000","9999999"] but outside many numeric
/// ranges — the row filter must decide, never the planning bounds.
async fn seed(catalog: &impl Catalog, ident: &TableIdent) {
    let table = catalog.load_table(ident).await.unwrap();
    let mut all = write_data(
        &table,
        "seed-backfill",
        Some(true),
        batch(&table, &[
            ("r-old-5", "r", "5", Some(true)),
            ("r-old-42", "R", "42", Some(true)),
            ("r-old-999", "r", "999", Some(true)),
            ("r-old-1500", "R", "1500", Some(true)),
            ("r-old-1000007", "r", "1000007", Some(true)),
        ]),
    )
    .await;
    all.extend(
        write_data(
            &table,
            "seed-cdc",
            None,
            batch(&table, &[
                ("c-42", "c", "42", None),
                ("c-777", "c", "777", None),
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
}

async fn run_replace(
    catalog: &impl Catalog,
    ident: &TableIdent,
    lo: i64,
    hi: i64,
    rows: &[(&str, &str, &str, Option<bool>)],
) -> iceberg::Result<iceberg::atomic_replace::ReplaceOutcome> {
    let table = catalog.load_table(ident).await.unwrap();
    let batches = if rows.is_empty() {
        vec![]
    } else {
        vec![batch(&table, rows)]
    };
    atomic_partition_replace_key_range(
        catalog,
        ident,
        "_is_backfill",
        true,
        "_cdc.key",
        lo,
        hi,
        Some("_cdc.op"),
        &["R".to_string(), "r".to_string()],
        batches,
        3,
        false,
    )
    .await
}

#[tokio::test]
async fn range_replace_mixed_length_deletes_numeric_scope_only() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let before_snapshots = {
        let t = catalog.load_table(&ident).await.unwrap();
        t.metadata().snapshots().count()
    };

    // Mixed-length range [1, 999]: planning falls back to partition+op
    // pruning; the row filter must catch 5, 42, 999 and NOTHING else.
    let outcome = run_replace(&catalog, &ident, 1, 999, &[
        ("r-new-5", "R", "5", Some(true)),
        ("r-new-7", "R", "7", Some(true)),
    ])
    .await
    .unwrap();
    assert_eq!(outcome.rows_appended, 2);
    assert_eq!(outcome.delete_tuples, 3); // 5, 42, 999
    assert_eq!(outcome.attempts, 1);

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(table.metadata().snapshots().count(), before_snapshots + 1);
    assert_eq!(read_rows(&table).await, vec![
        ("c-42".into(), "c".into(), "42".into()), // CDC untouched
        ("c-777".into(), "c".into(), "777".into()),
        ("r-new-5".into(), "R".into(), "5".into()),
        ("r-new-7".into(), "R".into(), "7".into()),
        ("r-old-1000007".into(), "r".into(), "1000007".into()), // outside range
        ("r-old-1500".into(), "R".into(), "1500".into()),       // outside range
    ]);
}

#[tokio::test]
async fn range_replace_same_length_bounds_stay_exact() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    // Same-digit-length range [1000, 9999]: the lexicographic planning
    // bounds engage. Numerically only "1500" is in range — "5"/"42"/"999"
    // lex-sort BELOW "1000" at shorter length? No: "5" > "1000" lex (char
    // '5' > '1') — a lex FALSE POSITIVE the row filter must drop; and
    // "1000007" lex-falls inside ["1000","9999"] too (prefix "1000" then
    // longer) — also a false positive. Exactly ONE numeric match.
    let outcome = run_replace(&catalog, &ident, 1000, 9999, &[(
        "r-new-2000",
        "R",
        "2000",
        Some(true),
    )])
    .await
    .unwrap();
    assert_eq!(outcome.rows_appended, 1);
    assert_eq!(outcome.delete_tuples, 1); // "1500" only

    let table = catalog.load_table(&ident).await.unwrap();
    let rows = read_rows(&table).await;
    assert!(rows.iter().any(|(a, _, _)| a == "r-old-5"));
    assert!(rows.iter().any(|(a, _, _)| a == "r-old-42"));
    assert!(rows.iter().any(|(a, _, _)| a == "r-old-999"));
    assert!(rows.iter().any(|(a, _, _)| a == "r-old-1000007"));
    assert!(rows.iter().any(|(a, _, _)| a == "r-new-2000"));
    assert!(!rows.iter().any(|(a, _, _)| a == "r-old-1500"));
}

#[tokio::test]
async fn range_replace_is_replay_idempotent() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let rows: &[(&str, &str, &str, Option<bool>)] = &[
        ("r-new-5", "R", "5", Some(true)),
        ("r-new-7", "R", "7", Some(true)),
    ];
    run_replace(&catalog, &ident, 1, 999, rows).await.unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let first = read_rows(&table).await;

    let outcome = run_replace(&catalog, &ident, 1, 999, rows).await.unwrap();
    assert_eq!(outcome.delete_tuples, 2); // the first run's appends
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
async fn range_replace_zero_rows_still_deletes_and_rejects_inverted() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = setup(&warehouse).await;
    seed(&catalog, &ident).await;

    let outcome = run_replace(&catalog, &ident, 1, 999, &[]).await.unwrap();
    assert_eq!(outcome.rows_appended, 0);
    assert_eq!(outcome.delete_tuples, 3);
    let table = catalog.load_table(&ident).await.unwrap();
    let rows = read_rows(&table).await;
    assert!(!rows.iter().any(|(a, _, _)| a.starts_with("r-old-5")
        || a.starts_with("r-old-42")
        || a.starts_with("r-old-999")));
    assert_eq!(rows.len(), 4); // 2 CDC + 1500 + 1000007

    let err = run_replace(&catalog, &ident, 10, 1, &[]).await.unwrap_err();
    assert!(
        err.to_string().contains("inverted"),
        "unexpected error: {err}"
    );
}
