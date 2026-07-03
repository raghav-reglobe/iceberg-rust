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

//! ATOMICITY SPIKE for the SCD2 merge engine (MoR MERGE INTO, requirement 1):
//! ONE `RowDelta` transaction must commit a deletion-vector DEMOTE against an
//! existing data file AND a new data-file APPEND as a SINGLE snapshot.
//!
//! Why it matters: a merge that demotes in one snapshot and appends in another
//! has a crash window between the two that orphans the append (dup-current) —
//! the exact bug class the DuckDB worker hit before wrapping its merge in a
//! transaction. One snapshot is atomic by Iceberg's metadata-swap model, so a
//! single-commit RowDelta makes the merge crash-atomic BY CONSTRUCTION.
//!
//! The scenario models an SCD2 micro-merge on a (id, ver) table:
//!   seed file A: (1,1) (2,1) (3,1) (4,1)      -- the "current" rows
//!   merge, in ONE commit:
//!     - DV on file A dropping position 1       -- demote old current (2,1)
//!     - new file B: (2,2) (5,1)                -- new current for 2 + insert 5
//!   end state: ids {1,2,3,4,5}, id=2 exactly once with ver=2.
//!
//! Also pinned: the same-sequence DV binds ONLY to its referenced data file —
//! file B commits at the SAME sequence number as the DV, and B's position 0
//! holds the NEW (2,2) row; if the DV leaked onto B, the new current would
//! vanish. Upstream-destined home: the core transaction tests (RowDelta PR).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestList,
    NestedField, Operation, PrimitiveType, Schema, Struct, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalogBuilder, NamespaceIdent, TableCreation, TableIdent,
    MEMORY_CATALOG_WAREHOUSE,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;

fn arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("ver", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]))
}

fn batch(ids: Vec<i32>, vers: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        arrow_schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int32Array::from(vers)),
        ],
    )
    .unwrap()
}

/// Write `batch` to a single new data file. `prefix` MUST be unique per write:
/// `DefaultFileNameGenerator`'s counter resets per instance, so a shared
/// prefix would collide two writes onto one path.
async fn write_data_file(table: &Table, prefix: &str, batch: RecordBatch) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

/// All live (id, ver) rows via a scan — deletion vectors applied by the reader.
async fn live_rows(table: &Table) -> Vec<(i32, i32)> {
    let mut stream = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let mut rows = Vec::new();
    while let Some(b) = stream.try_next().await.unwrap() {
        let ids = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let vers = b
            .column_by_name("ver")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            rows.push((ids.value(i), vers.value(i)));
        }
    }
    rows.sort_unstable();
    rows
}

/// (file_path, content_type, data_sequence_number) for every ALIVE manifest
/// entry in the current snapshot.
async fn live_entries(table: &Table) -> Vec<(String, DataContentType, i64)> {
    let snap = table.metadata().current_snapshot().unwrap();
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
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                out.push((
                    e.data_file().file_path().to_string(),
                    e.data_file().content_type(),
                    e.sequence_number().expect("sequence number inherited"),
                ));
            }
        }
    }
    out
}

/// Seed: V3 (id, ver) table with file A = (1..4, v1). Returns (catalog, ident,
/// file A's path).
async fn seed_table(warehouse: &TempDir) -> (impl Catalog, TableIdent, String) {
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
            NestedField::required(2, "ver", Type::Primitive(PrimitiveType::Int)).into(),
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

    let data_files = write_data_file(&table, "seed", batch(vec![1, 2, 3, 4], vec![1, 1, 1, 1])).await;
    let file_a = data_files[0].file_path().to_string();
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_rows(&table).await.len(), 4);

    (catalog, ident, file_a)
}

/// THE SPIKE: one RowDelta commit carries the DV demote + the data append.
#[tokio::test]
async fn scd2_demote_and_append_commit_as_single_snapshot() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, file_a) = seed_table(&warehouse).await;
    let table = catalog.load_table(&ident).await.unwrap();
    let snapshots_before = table.metadata().snapshots().count();

    // Demote: DV on file A dropping position 1 — the old current (2, v1).
    let mut positions = RoaringTreemap::new();
    positions.insert(1);
    let dv_path = format!("{}/dv-merge-0.puffin", warehouse.path().to_str().unwrap());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            file_a.clone(),
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap();

    // Append: file B with the new current (2, v2) at POSITION 0 + insert (5, v1).
    let new_files = write_data_file(&table, "merged", batch(vec![2, 5], vec![2, 1])).await;

    // ONE commit for both — the merge is atomic by construction.
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_data_files(new_files)
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // Exactly ONE new snapshot — no demote/append crash window exists.
    assert_eq!(
        table.metadata().snapshots().count(),
        snapshots_before + 1,
        "demote + append must land in a single snapshot"
    );
    assert_eq!(
        table.metadata().current_snapshot().unwrap().summary().operation,
        Operation::Overwrite,
        "RowDelta with data + deletes commits an Overwrite"
    );

    // End state: ids {1,2,3,4,5}; id=2 EXACTLY once, at ver=2 (one current per
    // key). This also pins that the same-sequence DV did NOT leak onto file B:
    // B's position 0 holds (2, v2) — if the DV applied to B, it would be gone.
    assert_eq!(
        live_rows(&table).await,
        vec![(1, 1), (2, 2), (3, 1), (4, 1), (5, 1)],
    );
}

/// Sequence-number semantics of the combined commit: file A keeps its seed
/// sequence; the DV and file B share the NEW snapshot's sequence; the DV
/// references file A only.
#[tokio::test]
async fn scd2_single_commit_sequence_numbers() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, file_a) = seed_table(&warehouse).await;
    let table = catalog.load_table(&ident).await.unwrap();

    let mut positions = RoaringTreemap::new();
    positions.insert(1);
    let dv_path = format!("{}/dv-merge-1.puffin", warehouse.path().to_str().unwrap());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            file_a.clone(),
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap();
    let new_files = write_data_file(&table, "merged", batch(vec![2, 5], vec![2, 1])).await;
    let file_b = new_files[0].file_path().to_string();

    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_data_files(new_files)
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let entries = live_entries(&table).await;
    let seq_of = |path: &str, content: DataContentType| -> i64 {
        entries
            .iter()
            .find(|(p, c, _)| p == path && *c == content)
            .unwrap_or_else(|| panic!("no alive entry for {path} ({content:?})"))
            .2
    };

    let seq_a = seq_of(&file_a, DataContentType::Data);
    let seq_b = seq_of(&file_b, DataContentType::Data);
    let seq_dv = entries
        .iter()
        .find(|(_, c, _)| *c == DataContentType::PositionDeletes)
        .expect("DV entry alive")
        .2;

    assert_eq!(seq_a, 1, "seed file keeps its original sequence");
    assert_eq!(seq_b, 2, "appended file gets the merge snapshot's sequence");
    assert_eq!(seq_dv, 2, "DV gets the same (merge snapshot) sequence");
    assert_eq!(entries.len(), 3, "A + B + DV, nothing else");
}
