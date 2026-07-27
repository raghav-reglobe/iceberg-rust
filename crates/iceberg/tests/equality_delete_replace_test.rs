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

//! ATOMIC REPLACE via PARTITION-SCOPED equality deletes — the append-only
//! CDC "full-load chunk re-backfill" shape: one `RowDelta` snapshot that
//! (i) equality-deletes a key chunk's PRIOR full-load rows and (ii) appends
//! their replacements.
//!
//! Design note proven here: the full-load rows live in their own IDENTITY
//! PARTITION (`_is_backfill=true`; CDC event rows carry null), so the
//! delete files use **top-level, pk-only equality ids** and are **scoped to
//! that partition** — CDC rows sharing the same keys are structurally out
//! of scope. This avoids NESTED equality ids (e.g. an op marker inside a
//! struct), which the read path's schema evolution does not support and
//! which are a cross-engine compatibility hazard.
//!
//! Also proven: rows appended AFTER the replace (higher data sequence
//! number) are unaffected, and the equality deletes coexist with
//! POSITIONAL deletes (V3 deletion vectors) on the same table — the
//! transitional cross-engine state.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    BooleanArray, Int32Array, LargeStringArray, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey, PrimitiveType,
    Schema, Struct, StructType, Transform, Type, UnboundPartitionSpec,
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
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;
use uuid::Uuid;

const FIELD_ID_PK: i32 = 1;
const FIELD_ID_PAYLOAD: i32 = 2;
const FIELD_ID_CDC: i32 = 3;
const FIELD_ID_CDC_OP: i32 = 4;
const FIELD_ID_IS_BACKFILL: i32 = 5;

fn table_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(FIELD_ID_PK, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(
                FIELD_ID_PAYLOAD,
                "payload",
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

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

/// The write-side arrow schema is derived from the TABLE's schema —
/// `create_table` reassigns field ids (top-level fields first, then nested),
/// so hardcoded ids diverge from the committed schema.
fn table_arrow_schema(table: &Table) -> Arc<ArrowSchema> {
    Arc::new(iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema()).unwrap())
}

/// Rows: (id, payload, op, is_backfill), built against the table's schema.
fn batch(table: &Table, rows: &[(i32, &str, &str, Option<bool>)]) -> RecordBatch {
    let schema = table_arrow_schema(table);
    let cdc_field = match schema.field_with_name("_cdc").unwrap().data_type() {
        DataType::Struct(fields) => fields[0].clone(),
        other => panic!("_cdc must be a struct, got {other:?}"),
    };
    let ops: Vec<&str> = rows.iter().map(|r| r.2).collect();
    let cdc = StructArray::from(vec![(
        cdc_field,
        Arc::new(LargeStringArray::from(ops)) as arrow_array::ArrayRef,
    )]);
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
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

/// Write a PARTITION-SCOPED (`_is_backfill=true`) equality-delete file with
/// pk-only equality ids: the delete tuples are the ids of `rows` — for a
/// replace, the NEW rows' own keys are exactly the tuples to delete. The
/// partition scope keeps same-key rows in OTHER partitions (CDC events)
/// untouched.
async fn write_eq_delete(table: &Table, rows: RecordBatch) -> Vec<DataFile> {
    let pk_id = table
        .metadata()
        .current_schema()
        .field_id_by_name("id")
        .unwrap();
    let config =
        EqualityDeleteWriterConfig::new(vec![pk_id], table.metadata().current_schema().clone())
            .unwrap();
    // The delete file's parquet schema is the PROJECTED equality-id schema,
    // not the table schema (upstream contract).
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
        .build(Some(partition_key(table, Some(true))))
        .await
        .unwrap();
    w.write(rows).await.unwrap();
    w.close().await.unwrap()
}

async fn read_rows(table: &Table) -> Vec<(i32, String, String)> {
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
        let id_idx = schema.index_of("id").unwrap();
        let payload_idx = schema.index_of("payload").unwrap();
        let cdc_idx = schema.index_of("_cdc").unwrap();
        let ids = b
            .column(id_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let payloads = b
            .column(payload_idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        let cdc = b
            .column(cdc_idx)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let ops = cdc
            .column_by_name("op")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            out.push((
                ids.value(i),
                payloads.value(i).to_string(),
                ops.value(i).to_string(),
            ));
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn atomic_replace_via_partition_scoped_equality_deletes() {
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
        .add_partition_field(FIELD_ID_IS_BACKFILL, "_is_backfill", Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("bronze_t".to_string())
                .schema(table_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    let ident = TableIdent::new(ns, "bronze_t".to_string());

    // Seed the two partitions the way the pipeline does: full-load rows
    // (op=r) in `_is_backfill=true`; CDC rows (op=c) in the null partition —
    // id 2 exists in BOTH.
    let mut seed = write_data(
        &table,
        "seed-backfill",
        Some(true),
        batch(&table, &[
            (1, "r-old-1", "r", Some(true)),
            (2, "r-old-2", "r", Some(true)),
            (3, "r-old-3", "r", Some(true)),
            (6, "r-old-6", "r", Some(true)),
        ]),
    )
    .await;
    seed.extend(
        write_data(
            &table,
            "seed-cdc",
            None,
            batch(&table, &[(2, "c-2", "c", None), (5, "c-5", "c", None)]),
        )
        .await,
    );
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(seed)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // ATOMIC REPLACE of chunk ids 1..=3 in the backfill partition: the new
    // rows are both the appended data AND (their ids) the delete tuples.
    let table = catalog.load_table(&ident).await.unwrap();
    let base_snapshot = table.metadata().current_snapshot_id().unwrap();
    let replacement = batch(&table, &[
        (1, "r-new-1", "r", Some(true)),
        (2, "r-new-2", "r", Some(true)),
        (3, "r-new-3", "r", Some(true)),
    ]);
    let new_data = write_data(&table, "replace", Some(true), replacement.clone()).await;
    let eq_deletes = write_eq_delete(&table, replacement).await;
    assert_eq!(eq_deletes.len(), 1);
    assert_eq!(
        eq_deletes[0].content_type(),
        iceberg::spec::DataContentType::EqualityDeletes
    );
    assert_eq!(
        eq_deletes[0].equality_ids(),
        Some(vec![
            table
                .metadata()
                .current_schema()
                .field_id_by_name("id")
                .unwrap()
        ])
    );

    let snaps_before = table.metadata().snapshots().count();
    let tx = Transaction::new(&table);
    tx.row_delta()
        .add_data_files(new_data)
        .add_delete_files(eq_deletes)
        .validate_from_snapshot(base_snapshot)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        table.metadata().snapshots().count(),
        snaps_before + 1,
        "delete + append commit as ONE snapshot"
    );

    // Old backfill rows of the chunk are GONE; CDC rows with the SAME keys
    // (other partition) survive; backfill rows outside the chunk survive.
    assert_eq!(read_rows(&table).await, vec![
        (1, "r-new-1".to_string(), "r".to_string()),
        (2, "c-2".to_string(), "c".to_string()),
        (2, "r-new-2".to_string(), "r".to_string()),
        (3, "r-new-3".to_string(), "r".to_string()),
        (5, "c-5".to_string(), "c".to_string()),
        (6, "r-old-6".to_string(), "r".to_string()),
    ]);

    // SEQUENCE semantics: a backfill row for id 2 appended AFTER the replace
    // (higher data sequence number) is NOT touched by the earlier delete.
    let later = write_data(
        &table,
        "later",
        Some(true),
        batch(&table, &[(2, "r-later-2", "r", Some(true))]),
    )
    .await;
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(later)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let rows = read_rows(&table).await;
    assert!(
        rows.contains(&(2, "r-later-2".to_string(), "r".to_string())),
        "rows appended after the delete's sequence number must survive: {rows:?}"
    );

    // COEXISTENCE with positional deletes (V3 DV): delete one row of the
    // replacement file by POSITION in a second RowDelta — the read applies
    // the equality deletes AND the DV together.
    let replace_file = {
        let tasks: Vec<_> = table
            .scan()
            .select_all()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        tasks
            .iter()
            .map(|t| t.data_file_path().to_string())
            .find(|p| p.contains("replace-"))
            .expect("replacement file planned")
    };
    let mut bitmap = RoaringTreemap::new();
    bitmap.insert(0); // first row of the replacement file = id 1
    let dv_path = format!(
        "{}/data/{}-deletes.puffin",
        table.metadata().location(),
        Uuid::now_v7()
    );
    let dv = DeleteVector::new(bitmap)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            replace_file,
            Struct::from_iter(vec![Some(Literal::bool(true))]),
            table.metadata().default_partition_spec_id(),
        )
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    tx.row_delta()
        .add_delete_files(vec![dv])
        .validate_from_snapshot(table.metadata().current_snapshot_id().unwrap())
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let rows = read_rows(&table).await;
    assert!(
        !rows.contains(&(1, "r-new-1".to_string(), "r".to_string())),
        "the DV must remove the positionally-deleted row: {rows:?}"
    );
    assert!(
        !rows.contains(&(2, "r-old-2".to_string(), "r".to_string())),
        "equality deletes still apply alongside the DV: {rows:?}"
    );
    assert_eq!(rows.len(), 6, "{rows:?}");
}

// ---------------------------------------------------------------------------
// The atomic_partition_replace API (the doorway's core)
// ---------------------------------------------------------------------------

mod replace_api {
    use arrow_array::{Array, BinaryArray};
    use iceberg::atomic_replace::{ReplaceOutcome, atomic_partition_replace};
    use iceberg::spec::VariantType;

    use super::*;

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
        let table = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("bronze_api".to_string())
                    .schema(table_schema())
                    .partition_spec(spec)
                    .format_version(FormatVersion::V3)
                    .build(),
            )
            .await
            .unwrap();
        let ident = TableIdent::new(ns, "bronze_api".to_string());

        // Seed both partitions: full-load rows + CDC rows (id 2 in both).
        let mut seed = write_data(
            &table,
            "seed-backfill",
            Some(true),
            batch(&table, &[
                (1, "r-old-1", "r", Some(true)),
                (2, "r-old-2", "r", Some(true)),
                (6, "r-old-6", "r", Some(true)),
            ]),
        )
        .await;
        seed.extend(
            write_data(
                &table,
                "seed-cdc",
                None,
                batch(&table, &[(2, "c-2", "c", None)]),
            )
            .await,
        );
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(seed)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        (catalog, ident)
    }

    /// A batch WITHOUT field-id metadata and with columns out of order — the
    /// doorway input shape (external Arrow producers carry no parquet ids).
    fn external_batch(table: &Table, rows: &[(i32, &str, &str, Option<bool>)]) -> RecordBatch {
        let plain = |name: &str, dt: DataType, nullable: bool| Field::new(name, dt, nullable);
        let schema = Arc::new(ArrowSchema::new(vec![
            plain("payload", DataType::Utf8, true),
            plain("_is_backfill", DataType::Boolean, true),
            plain(
                "_cdc",
                DataType::Struct(Fields::from(vec![plain("op", DataType::Utf8, false)])),
                false,
            ),
            plain("id", DataType::Int32, false),
        ]));
        let _ = table;
        let ops: Vec<&str> = rows.iter().map(|r| r.2).collect();
        let cdc = StructArray::from(vec![(
            Arc::new(plain("op", DataType::Utf8, false)),
            Arc::new(StringArray::from(ops)) as arrow_array::ArrayRef,
        )]);
        RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )),
            Arc::new(cdc),
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn replace_conforms_commits_once_and_is_rerunnable() {
        let warehouse = TempDir::new().unwrap();
        let (catalog, ident) = setup(&warehouse).await;
        let table = catalog.load_table(&ident).await.unwrap();
        let snaps_before = table.metadata().snapshots().count();

        // Dry run: counts only, no snapshot.
        let out = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string()],
            vec![external_batch(&table, &[
                (1, "r-new-1", "r", Some(true)),
                (2, "r-new-2", "r", Some(true)),
            ])],
            6,
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            (out.rows_appended, out.delete_tuples, out.attempts),
            (2, 2, 0)
        );
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .snapshots()
                .count(),
            snaps_before,
            "dry run must not commit"
        );

        // Real replace: out-of-order, id-less input conforms; ONE snapshot.
        let out: ReplaceOutcome = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string()],
            vec![external_batch(&table, &[
                (1, "r-new-1", "r", Some(true)),
                (2, "r-new-2", "r", Some(true)),
            ])],
            6,
            false,
        )
        .await
        .unwrap();
        assert_eq!(out.attempts, 1);
        assert_eq!(out.rows_appended, 2);
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().snapshots().count(), snaps_before + 1);
        assert_eq!(table.metadata().current_snapshot_id(), out.snapshot_id);
        assert_eq!(read_rows(&table).await, vec![
            (1, "r-new-1".to_string(), "r".to_string()),
            (2, "c-2".to_string(), "c".to_string()),
            (2, "r-new-2".to_string(), "r".to_string()),
            (6, "r-old-6".to_string(), "r".to_string()),
        ]);

        // RERUN of the same chunk with fresher payloads: idempotent replace
        // semantics — the latest run's rows win, no duplicates.
        atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string()],
            vec![external_batch(&table, &[
                (1, "r-newer-1", "r", Some(true)),
                (2, "r-newer-2", "r", Some(true)),
            ])],
            6,
            false,
        )
        .await
        .unwrap();
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_rows(&table).await, vec![
            (1, "r-newer-1".to_string(), "r".to_string()),
            (2, "c-2".to_string(), "c".to_string()),
            (2, "r-newer-2".to_string(), "r".to_string()),
            (6, "r-old-6".to_string(), "r".to_string()),
        ]);
    }

    #[tokio::test]
    async fn replace_composite_equality_columns_delete_by_tuple() {
        // Multi-column equality ids (the composite-PK seam): delete tuples
        // are (id, payload) PAIRS — a row sharing only the id with a
        // replacement row survives; an exact tuple match is replaced.
        let warehouse = TempDir::new().unwrap();
        let (catalog, ident) = setup(&warehouse).await;
        let table = catalog.load_table(&ident).await.unwrap();

        let out = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string(), "payload".to_string()],
            vec![external_batch(&table, &[
                // exact tuple match with seeded (1, "r-old-1") -> replaced
                // (the op marker flips r -> R, proving delete + re-insert)
                (1, "r-old-1", "R", Some(true)),
                // same id as seeded row 2 but DIFFERENT payload -> the old
                // (2, "r-old-2") row must SURVIVE alongside the new row
                (2, "zz-new-2", "R", Some(true)),
            ])],
            6,
            false,
        )
        .await
        .unwrap();
        assert_eq!((out.rows_appended, out.delete_tuples), (2, 2));

        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_rows(&table).await, vec![
            (1, "r-old-1".to_string(), "R".to_string()), // replaced (op R)
            (2, "c-2".to_string(), "c".to_string()),
            (2, "r-old-2".to_string(), "r".to_string()), // tuple mismatch: survives
            (2, "zz-new-2".to_string(), "R".to_string()),
            (6, "r-old-6".to_string(), "r".to_string()),
        ]);
    }

    #[tokio::test]
    async fn replace_local_parquet_input_streams_and_matches_batches() {
        // The giant-chunk mode: the SAME rows via a LOCAL parquet file
        // (streamed batch-wise, bounded memory) must produce the same table
        // state the in-memory batches mode does — incl. the two-pass read
        // (data files + equality-delete tuples re-stream the file).
        let warehouse = TempDir::new().unwrap();
        let (catalog, ident) = setup(&warehouse).await;
        let table = catalog.load_table(&ident).await.unwrap();

        let batch = external_batch(&table, &[
            (1, "r-file-1", "r", Some(true)),
            (2, "r-file-2", "r", Some(true)),
        ]);
        let path = warehouse.path().join("chunk-input.parquet");
        let mut w = parquet::arrow::arrow_writer::ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            batch.schema(),
            None,
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let out = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string()],
            iceberg::atomic_replace::ReplaceInput::LocalParquet(path.to_str().unwrap().to_string()),
            6,
            false,
        )
        .await
        .unwrap();
        assert_eq!((out.rows_appended, out.delete_tuples), (2, 2));
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_rows(&table).await, vec![
            (1, "r-file-1".to_string(), "r".to_string()),
            (2, "c-2".to_string(), "c".to_string()),
            (2, "r-file-2".to_string(), "r".to_string()),
            (6, "r-old-6".to_string(), "r".to_string()),
        ]);

        // Zero-row parquet input -> no-op (no snapshot, no writers).
        let empty = warehouse.path().join("empty-input.parquet");
        let mut w = parquet::arrow::arrow_writer::ArrowWriter::try_new(
            std::fs::File::create(&empty).unwrap(),
            batch.schema(),
            None,
        )
        .unwrap();
        w.close().unwrap();
        let snaps_before = table.metadata().snapshots().count();
        let out = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string()],
            iceberg::atomic_replace::ReplaceInput::LocalParquet(
                empty.to_str().unwrap().to_string(),
            ),
            6,
            false,
        )
        .await
        .unwrap();
        assert_eq!((out.rows_appended, out.attempts), (0, 0));
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().snapshots().count(), snaps_before);
    }

    #[tokio::test]
    async fn replace_rejects_nested_equality_and_wrong_spec() {
        let warehouse = TempDir::new().unwrap();
        let (catalog, ident) = setup(&warehouse).await;
        let table = catalog.load_table(&ident).await.unwrap();
        let rows = vec![external_batch(&table, &[(1, "x", "r", Some(true))])];

        // Nested equality column -> loud error.
        let err = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["_cdc.op".to_string()],
            rows.clone(),
            6,
            false,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("TOP-LEVEL"), "{err}");

        // Wrong partition column -> loud error.
        let err = atomic_partition_replace(
            &catalog,
            &ident,
            "payload",
            Literal::string("x"),
            &["id".to_string()],
            rows,
            6,
            false,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("IDENTITY partition"), "{err}");
    }

    /// The mongo-shaped path: a VARIANT column fed as JSON strings.
    #[tokio::test]
    async fn replace_converts_json_strings_to_variant() {
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
                NestedField::optional(2, "after", Type::Variant(VariantType)).into(),
                NestedField::optional(3, "_is_backfill", Type::Primitive(PrimitiveType::Boolean))
                    .into(),
            ])
            .build()
            .unwrap();
        let spec = UnboundPartitionSpec::builder()
            .add_partition_field(3, "_is_backfill", Transform::Identity)
            .unwrap()
            .build();
        catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("mongo_bronze".to_string())
                    .schema(schema)
                    .partition_spec(spec)
                    .format_version(FormatVersion::V3)
                    .build(),
            )
            .await
            .unwrap();
        let ident = TableIdent::new(ns, "mongo_bronze".to_string());

        // Input: `after` as plain JSON strings (no field ids, no variant).
        let plain_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("after", DataType::Utf8, true),
            Field::new("_is_backfill", DataType::Boolean, true),
        ]));
        let input = RecordBatch::try_new(plain_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some(r#"{"name":"a","n":1}"#), None])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(true)])),
        ])
        .unwrap();

        let out = atomic_partition_replace(
            &catalog,
            &ident,
            "_is_backfill",
            Literal::bool(true),
            &["id".to_string()],
            vec![input],
            6,
            false,
        )
        .await
        .unwrap();
        assert_eq!(out.rows_appended, 2);

        // Read back: canonical variant struct; row 1 non-null, row 2 null.
        let table = catalog.load_table(&ident).await.unwrap();
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
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2);
        let b = &batches[0];
        let schema = b.schema();
        let after = b
            .column(schema.index_of("after").unwrap())
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let id_col = b
            .column(schema.index_of("id").unwrap())
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            if id_col.value(i) == 1 {
                assert!(after.is_valid(i), "row id=1 carries a variant document");
                let metadata = after.column_by_name("metadata").unwrap();
                assert!(
                    metadata
                        .as_any()
                        .downcast_ref::<BinaryArray>()
                        .unwrap()
                        .value(i)
                        .len()
                        > 0
                );
            } else {
                assert!(after.is_null(i), "row id=2 is a NULL variant");
            }
        }
    }
}
