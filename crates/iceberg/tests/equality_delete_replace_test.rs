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

use arrow_array::{BooleanArray, Int32Array, RecordBatch, StringArray, StructArray};
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
        Arc::new(StringArray::from(ops)) as arrow_array::ArrayRef,
    )]);
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
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
            .downcast_ref::<StringArray>()
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
            .downcast_ref::<StringArray>()
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
