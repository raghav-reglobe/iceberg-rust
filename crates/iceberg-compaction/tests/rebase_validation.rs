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

//! Integration tests: input-protected rebase on the compaction commit
//! (`RewriteFilesAction::validate_rebase_from`, wired by `commit_rewrite`).
//!
//! The corruption class this guards: a rewrite planned at snapshot S rebases
//! over a concurrent merge that committed a deletion vector against one of
//! the files being rewritten. The rewrite's output was read BEFORE that DV
//! existed and carries a fresh data sequence, so blind-rebasing resurrects
//! the concurrently deleted rows (observed live as baked duplicate
//! `_is_current` pairs). Concurrent APPENDS must keep rebasing fine — a
//! long-running compaction on a hot table depends on it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PrimitiveType, Schema, Struct,
    Type,
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
use iceberg_compaction::rewrite::commit_rewrite;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;

fn id_batch(ids: Vec<i32>) -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]));
    RecordBatch::try_new(arrow_schema, vec![Arc::new(Int32Array::from(ids))]).unwrap()
}

async fn write_one_data_file(table: &Table, batch: RecordBatch, prefix: &str) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

/// Seed a V3 table with one data file (ids 1-4). Returns the catalog, ident,
/// the seeded table handle (the rewrite's STALE planning handle), and the
/// input data file being rewritten.
async fn seeded_table(warehouse: &TempDir) -> (impl Catalog, TableIdent, Table, DataFile) {
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
        ])
        .build()
        .unwrap();
    let ident = TableIdent::new(ns.clone(), "t".to_string());
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

    let data_files = write_one_data_file(&table, id_batch(vec![1, 2, 3, 4]), "seed").await;
    let input = data_files[0].clone();
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    (catalog, ident, table, input)
}

/// A concurrent DV against a file being rewritten aborts the rebase — the
/// live merge×maintenance corruption class (blind rebase resurrected the
/// concurrently demoted rows).
#[tokio::test]
async fn rewrite_aborts_on_concurrent_dv_against_inputs() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, planned, input) = seeded_table(&warehouse).await;

    // The rewrite's output, written from the planned state.
    let output = write_one_data_file(&planned, id_batch(vec![1, 2, 3, 4]), "compact").await;

    // Concurrent merge: DV deleting position 0 of the INPUT file, committed
    // after the rewrite planned (fresh handle; the planned handle is stale).
    let mut positions = RoaringTreemap::new();
    positions.insert(0);
    let dv_path = format!("{}/dv-race.puffin", warehouse.path().to_str().unwrap());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            planned.file_io(),
            dv_path,
            input.file_path().to_string(),
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap();
    let fresh = catalog.load_table(&ident).await.unwrap();
    let tx = Transaction::new(&fresh);
    tx.row_delta()
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let err = commit_rewrite(&planned, &catalog, vec![input], vec![], output)
        .await
        .expect_err("rebasing over a DV against an input must conflict");
    let msg = format!("{err}");
    assert!(
        msg.contains("conflicting concurrent commit"),
        "unexpected error: {msg}"
    );
}

/// A concurrent APPEND rebases fine — the normal case a long compaction on a
/// hot table depends on (streaming sink commits, merge inserts elsewhere).
#[tokio::test]
async fn rewrite_rebases_over_concurrent_append() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, planned, input) = seeded_table(&warehouse).await;

    let output = write_one_data_file(&planned, id_batch(vec![1, 2, 3, 4]), "compact").await;

    // Concurrent append of an unrelated data file.
    let fresh = catalog.load_table(&ident).await.unwrap();
    let appended = write_one_data_file(&fresh, id_batch(vec![5, 6]), "append").await;
    let tx = Transaction::new(&fresh);
    tx.fast_append()
        .add_data_files(appended)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let table = commit_rewrite(&planned, &catalog, vec![input], vec![], output)
        .await
        .expect("append-only skipped range must rebase cleanly");
    let snap = table.metadata().current_snapshot().unwrap();
    assert_eq!(
        snap.summary().operation,
        iceberg::spec::Operation::Replace,
        "rewrite committed as Replace on top of the append"
    );
}

/// A concurrent removal of an input (second rewrite / expiry) aborts.
#[tokio::test]
async fn rewrite_aborts_on_concurrent_removal_of_input() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, planned, input) = seeded_table(&warehouse).await;

    let output = write_one_data_file(&planned, id_batch(vec![1, 2, 3, 4]), "compact").await;

    // Concurrent competing rewrite replaces the same input first.
    let fresh = catalog.load_table(&ident).await.unwrap();
    let other_output = write_one_data_file(&fresh, id_batch(vec![1, 2, 3, 4]), "other").await;
    commit_rewrite(&fresh, &catalog, vec![input.clone()], vec![], other_output)
        .await
        .expect("first rewrite commits cleanly");

    let err = commit_rewrite(&planned, &catalog, vec![input], vec![], output)
        .await
        .expect_err("second rewrite of the same input must conflict");
    let msg = format!("{err}");
    assert!(
        msg.contains("conflicting concurrent commit"),
        "unexpected error: {msg}"
    );
}
