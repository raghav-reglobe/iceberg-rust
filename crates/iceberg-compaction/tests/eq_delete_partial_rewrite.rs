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

//! Regression: a PARTIAL rewrite of an equality-delete-bearing table must
//! not remove the equality-delete file while data files it still applies to
//! remain — doing so RESURRECTS deleted rows in the unrewritten files. An
//! equality delete binds to every lower-sequence data file in its scope
//! (unlike a positional DV, which binds to exactly one file), so the
//! removal rule is: remove a delete file only when EVERY data file the scan
//! bound it to was rewritten in the same pass.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestList, NestedField,
    PrimitiveType, Schema, Type,
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
/// writer uses UNCOMPRESSED defaults) — the file-size window drives which
/// files are compaction candidates.
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

/// pk-only equality-delete file whose tuples are `ids`.
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
    // The writer projects the equality-id columns from full-schema rows.
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

async fn live_files(table: &Table, content: ManifestContentType) -> Vec<String> {
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
                out.push(e.data_file().file_path().to_string());
            }
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn partial_rewrite_must_not_resurrect_equality_deleted_rows() {
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

    // File A: tiny (2 rows) -> compaction candidate under the size window.
    let a = write_data(
        &table,
        "a",
        batch(&[(1, payload(1, 64)), (2, payload(2, 64))]),
    )
    .await;
    append(&catalog, &table, a).await;
    // File B: sized INTO the [min, max) window -> NOT a candidate.
    // 60 rows x ~600 B ~= 36 KB uncompressed.
    let table = catalog.load_table(&ident).await.unwrap();
    let b_rows: Vec<(i32, String)> = (100..160).map(|i| (i, payload(i as u64, 600))).collect();
    let b = write_data(&table, "b", batch(&b_rows)).await;
    append(&catalog, &table, b).await;

    // ONE equality-delete file whose tuples hit BOTH files: id 1 (in A) and
    // id 101 (in B). Unpartitioned table -> it binds to every lower-sequence
    // data file.
    let table = catalog.load_table(&ident).await.unwrap();
    let eq = write_eq_delete(&table, &[1, 101]).await;
    let tx = Transaction::new(&table);
    tx.row_delta()
        .add_delete_files(eq)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // Pre-compaction: both deleted rows are gone.
    let table = catalog.load_table(&ident).await.unwrap();
    let ids = read_ids(&table).await;
    assert!(!ids.contains(&1) && !ids.contains(&101), "{ids:?}");
    assert_eq!(ids.len(), 60, "2 + 60 rows minus the two deletes");

    // PARTIAL rewrite: the size window makes A a candidate and leaves B
    // alone; the delete threshold is high so B's bound delete does not make
    // it a candidate either.
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
    let data = live_files(&table, ManifestContentType::Data).await;
    assert!(
        data.iter().any(|p| p.contains("/b-")),
        "precondition: B must NOT have been rewritten: {data:?}"
    );
    assert!(
        !data.iter().any(|p| p.contains("/a-")),
        "precondition: A must have been rewritten: {data:?}"
    );

    // THE REGRESSION: id 101 lives in unrewritten B — the equality delete
    // still applies to B and must NOT have been removed by A's rewrite.
    let ids = read_ids(&table).await;
    assert!(
        !ids.contains(&101),
        "equality-deleted row in the UNREWRITTEN file resurrected: {ids:?}"
    );
    assert!(!ids.contains(&1), "rewritten file re-applied its delete");
    assert_eq!(ids.len(), 60);
    assert_eq!(
        live_files(&table, ManifestContentType::Deletes).await.len(),
        1,
        "the equality delete must be RETAINED while B still needs it"
    );

    // FULL coverage pass: with delete-pressure enabled, B (1 bound delete)
    // becomes a candidate; once every file the delete applies to is
    // rewritten, the delete file IS removed — no permanent leak.
    let cfg_full = Config {
        target_file_size_bytes: target,
        min_file_size_bytes: target * 3 / 4,
        max_file_size_bytes: target * 9 / 5,
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg_full).await.unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let ids = read_ids(&table).await;
    assert!(!ids.contains(&101) && !ids.contains(&1), "{ids:?}");
    assert_eq!(ids.len(), 60);
    assert_eq!(
        live_files(&table, ManifestContentType::Deletes).await.len(),
        0,
        "fully-covered rewrite reabsorbs the equality delete"
    );
}
