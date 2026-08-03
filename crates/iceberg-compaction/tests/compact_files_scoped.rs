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

//! Path-scoped rewrite (`compact_files`) — the merge path's inline DV
//! micro-reabsorb entry.
//!
//! Three pins:
//!   - `scoped_rewrite_reabsorbs_only_scoped_dv` — exactly the scoped file is
//!     rewritten and ONLY its DV reabsorbed; an out-of-scope file keeps both
//!     its data file and its DV.
//!   - `scoped_rewrite_retains_equality_deletes` — an equality delete bound
//!     to the scoped file is RETAINED (a path-scoped scan cannot see its
//!     applicability to files outside the scope; removing it would resurrect
//!     their deleted rows — the pin asserts the out-of-scope row stays gone).
//!   - `empty_paths_noop` — no paths, no snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList,
    NestedField, PrimitiveType, Schema, Struct, Type,
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
use iceberg_compaction::engine::{compact_files, current_data_files};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
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

fn ids_batch(ids: &[i32]) -> RecordBatch {
    let rows: Vec<(i32, String)> = ids.iter().map(|i| (*i, format!("p{i}"))).collect();
    batch(&rows)
}

async fn make_table(warehouse: &TempDir) -> (impl Catalog, TableIdent, Table) {
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
    (catalog, ident, table)
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

async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap()
}

async fn commit_dv(catalog: &impl Catalog, table: &Table, dv: DataFile) -> Table {
    let tx = Transaction::new(table);
    tx.row_delta()
        .add_delete_files(vec![dv])
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap()
}

async fn write_dv(warehouse: &TempDir, table: &Table, data_path: &str, pos: u64) -> DataFile {
    let mut positions = RoaringTreemap::new();
    positions.insert(pos);
    let dv_path = format!(
        "{}/dv-{}.puffin",
        warehouse.path().to_str().unwrap(),
        Uuid::now_v7()
    );
    DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            data_path.to_string(),
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap()
}

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

async fn live_ids(table: &Table) -> Vec<i32> {
    let mut stream = table
        .scan()
        .select(["id"])
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let mut out = Vec::new();
    while let Some(b) = stream.try_next().await.unwrap() {
        let arr = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        out.extend(arr.iter().flatten());
    }
    out.sort_unstable();
    out
}

/// Live delete-file entries in the current snapshot: (path, is_positional).
async fn live_delete_files(table: &Table) -> Vec<String> {
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
        if mf.content != ManifestContentType::Deletes {
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

fn scoped_cfg() -> Config {
    Config::default()
}

#[tokio::test]
async fn scoped_rewrite_reabsorbs_only_scoped_dv() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = make_table(&warehouse).await;

    let a = write_data(&table, "a", ids_batch(&[1, 2, 3, 4])).await;
    let a_path = a[0].file_path().to_string();
    let table = append(&catalog, &table, a).await;
    let b = write_data(&table, "b", ids_batch(&[5, 6, 7, 8])).await;
    let b_path = b[0].file_path().to_string();
    let table = append(&catalog, &table, b).await;

    // DV on each file: drop id=1 (A pos 0) and id=5 (B pos 0).
    let dv_a = write_dv(&warehouse, &table, &a_path, 0).await;
    let table = commit_dv(&catalog, &table, dv_a).await;
    let dv_b = write_dv(&warehouse, &table, &b_path, 0).await;
    let table = commit_dv(&catalog, &table, dv_b).await;
    assert_eq!(live_ids(&table).await, vec![2, 3, 4, 6, 7, 8]);
    assert_eq!(live_delete_files(&table).await.len(), 2);

    let outcome = compact_files(
        &catalog,
        &ident,
        &HashSet::from([a_path.clone()]),
        &scoped_cfg(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.rewritten, 1);
    assert_eq!(outcome.added, 1);
    assert_eq!(outcome.reabsorbed_deletes, 1);

    let table = catalog.load_table(&ident).await.unwrap();
    // Row identity preserved; only B's DV remains; A's file replaced.
    assert_eq!(live_ids(&table).await, vec![2, 3, 4, 6, 7, 8]);
    assert_eq!(live_delete_files(&table).await.len(), 1, "B's DV retained");
    let data = current_data_files(&table).await.unwrap();
    assert!(!data.contains_key(&a_path), "A was rewritten");
    assert!(data.contains_key(&b_path), "B untouched");
}

#[tokio::test]
async fn scoped_rewrite_retains_equality_deletes() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = make_table(&warehouse).await;

    let a = write_data(&table, "a", ids_batch(&[1, 2, 3, 4])).await;
    let a_path = a[0].file_path().to_string();
    let table = append(&catalog, &table, a).await;
    let b = write_data(&table, "b", ids_batch(&[5, 6, 7, 8])).await;
    let table = append(&catalog, &table, b).await;

    // One equality delete masking id=2 (in A) AND id=6 (in B).
    let eq = write_eq_delete(&table, &[2, 6]).await;
    let table = {
        let tx = Transaction::new(&table);
        tx.row_delta()
            .add_delete_files(eq)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap()
    };
    assert_eq!(live_ids(&table).await, vec![1, 3, 4, 5, 7, 8]);

    let outcome = compact_files(
        &catalog,
        &ident,
        &HashSet::from([a_path.clone()]),
        &scoped_cfg(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.rewritten, 1);
    assert_eq!(
        outcome.reabsorbed_deletes, 0,
        "equality deletes must be RETAINED by a path-scoped rewrite"
    );

    let table = catalog.load_table(&ident).await.unwrap();
    // id=2 physically dropped from A's rewrite; id=6 still masked by the
    // RETAINED equality delete — removing it would have resurrected id=6.
    assert_eq!(live_ids(&table).await, vec![1, 3, 4, 5, 7, 8]);
    assert_eq!(
        live_delete_files(&table).await.len(),
        1,
        "the equality delete file is still live"
    );
}

#[tokio::test]
async fn empty_paths_noop() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = make_table(&warehouse).await;
    let a = write_data(&table, "a", ids_batch(&[1, 2])).await;
    let table = append(&catalog, &table, a).await;
    let snaps_before = table.metadata().snapshots().count();

    let outcome = compact_files(&catalog, &ident, &HashSet::new(), &scoped_cfg())
        .await
        .unwrap();
    assert_eq!(outcome.rewritten, 0);
    assert_eq!(outcome.added, 0);

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(table.metadata().snapshots().count(), snaps_before);
}
