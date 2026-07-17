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

//! End-to-end tests for `ExpireSnapshotsWithCleanupAction`: real tables
//! (memory catalog + parquet writer chain) exercising the exclusive-file diff
//! across full snapshot lifecycles.
//!
//! The safety-critical surface is the exclusive diff: a file reachable from
//! ANY retained snapshot must survive, even when an expired snapshot also
//! references it (the shared-file guard — see apache/iceberg#13568 for the
//! upstream cautionary bug). Deleting a referenced file is unrecoverable.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::maintenance::ExpireSnapshotsWithCleanupAction;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PrimitiveType, Schema,
    StatisticsFile, Struct, Type,
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
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;

fn future_cutoff_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
        + 60_000
}

fn arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]))
}

async fn write_one_data_file(table: &Table, prefix: &str, ids: Vec<i32>) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        // Unique per-write prefix: the generator's counter resets per instance,
        // so a shared prefix collides consecutive writes onto one path.
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    let batch =
        RecordBatch::try_new(arrow_schema(), vec![Arc::new(Int32Array::from(ids))]).unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

async fn live_row_count(table: &Table) -> usize {
    let mut n = 0;
    let mut stream = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    while let Some(batch) = stream.try_next().await.unwrap() {
        n += batch.num_rows();
    }
    n
}

async fn create_table(warehouse: &TempDir) -> (impl Catalog, Table) {
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
    (catalog, table)
}

async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap()
}

async fn reload(catalog: &impl Catalog, ident: &TableIdent) -> Table {
    catalog.load_table(ident).await.unwrap()
}

/// All manifest paths referenced by a snapshot's manifest list.
async fn manifest_paths_of_current(table: &Table) -> Vec<String> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    table
        .manifest_list_reader(snapshot)
        .load()
        .await
        .unwrap()
        .entries()
        .iter()
        .map(|m| m.manifest_path.clone())
        .collect()
}

/// Fast-append chain: only the expired snapshots' manifest LISTS are exclusive
/// (appends share manifests and data files with the retained head), so cleanup
/// must delete exactly those — the table keeps scanning every row.
#[tokio::test]
async fn expire_deletes_exclusive_lists_and_keeps_shared_manifests_and_data() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, mut table) = create_table(&warehouse).await;
    for (i, ids) in [vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]]
        .into_iter()
        .enumerate()
    {
        let files = write_one_data_file(&table, &format!("data-{i}"), ids).await;
        table = fast_append(&catalog, &table, files).await;
    }
    assert_eq!(live_row_count(&table).await, 12);

    let current_id = table.metadata().current_snapshot_id().unwrap();
    let expired_lists: Vec<String> = table
        .metadata()
        .snapshots()
        .filter(|s| s.snapshot_id() != current_id)
        .map(|s| s.manifest_list().to_string())
        .collect();
    let retained_list = table
        .metadata()
        .current_snapshot()
        .unwrap()
        .manifest_list()
        .to_string();
    let shared_manifests = manifest_paths_of_current(&table).await;

    let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
        .older_than_ms(future_cutoff_ms())
        .retain_last(1)
        .execute(&catalog)
        .await
        .unwrap();

    assert_eq!(result.removed_snapshot_ids.len(), 2);
    let mut expected = expired_lists.clone();
    expected.sort();
    assert_eq!(result.candidate_manifest_lists, expected);
    assert!(
        result.candidate_manifests.is_empty(),
        "manifests are shared"
    );
    assert!(result.candidate_data_files.is_empty(), "data is shared");
    assert!(result.candidate_delete_files.is_empty());
    assert!(result.candidate_stats_files.is_empty());
    assert_eq!(result.deleted_files.len(), 2);
    assert!(result.failed_deletes.is_empty());

    // Physically: expired lists gone; retained list, shared manifests intact.
    for list in &expired_lists {
        assert!(!table.file_io().exists(list).await.unwrap());
    }
    assert!(table.file_io().exists(&retained_list).await.unwrap());
    for manifest in &shared_manifests {
        assert!(table.file_io().exists(manifest).await.unwrap());
    }

    // Metadata: only the current snapshot remains; the table scans in full.
    let table = reload(&catalog, table.identifier()).await;
    assert_eq!(table.metadata().snapshots().len(), 1);
    assert_eq!(live_row_count(&table).await, 12);

    // Idempotency: a second call is a clean no-op that deletes nothing.
    let again = ExpireSnapshotsWithCleanupAction::new(table.clone())
        .older_than_ms(future_cutoff_ms())
        .retain_last(1)
        .execute(&catalog)
        .await
        .unwrap();
    assert!(again.is_noop());
    assert!(again.deleted_files.is_empty());
    assert_eq!(live_row_count(&table).await, 12);
}

/// Full compaction lifecycle: appends -> rewrite (compaction) -> manifest
/// rewrite -> expire. The replaced data files become exclusive and are
/// physically deleted; the compacted file — although also listed by an
/// expired snapshot's (deleted) manifest — survives because a retained
/// manifest still references it (the shared-file guard, apache/iceberg#13568).
#[tokio::test]
async fn expire_frees_replaced_data_files_but_keeps_shared_compacted_file() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, table) = create_table(&warehouse).await;

    // S1: f1 = [1..4]; S2: f2 = [5..8].
    let f1 = write_one_data_file(&table, "data-0", vec![1, 2, 3, 4]).await;
    let table = fast_append(&catalog, &table, f1.clone()).await;
    let f2 = write_one_data_file(&table, "data-1", vec![5, 6, 7, 8]).await;
    let table = fast_append(&catalog, &table, f2.clone()).await;

    // S3: compaction — f3 = [1..8] replaces f1 + f2.
    let f3 = write_one_data_file(&table, "compacted", vec![1, 2, 3, 4, 5, 6, 7, 8]).await;
    let f3_path = f3[0].file_path().to_string();
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .add_data_files(f3)
        .delete_data_files(f1.clone().into_iter().chain(f2.clone()))
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_row_count(&table).await, 8);

    // S4: manifest rewrite — consolidates the data manifests, dropping the
    // DELETED entries for f1/f2 so they stop being reachable from the head.
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_manifests()
        .min_input_manifests(1)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    // S3's manifests (incl. the one that lists f3 as ADDED) are now referenced
    // by no retained snapshot once S1-S3 expire.
    let retained_manifests: HashSet<String> = manifest_paths_of_current(&table)
        .await
        .into_iter()
        .collect();

    let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
        .older_than_ms(future_cutoff_ms())
        .retain_last(1)
        .execute(&catalog)
        .await
        .unwrap();

    assert_eq!(result.removed_snapshot_ids.len(), 3, "S1, S2, S3 expired");

    // The replaced data files are exclusive -> deleted.
    let mut expected_data: Vec<String> = f1
        .iter()
        .chain(f2.iter())
        .map(|f| f.file_path().to_string())
        .collect();
    expected_data.sort();
    assert_eq!(result.candidate_data_files, expected_data);
    for path in &expected_data {
        assert!(!table.file_io().exists(path).await.unwrap());
    }

    // Shared-file guard: f3 is listed by an expired-exclusive manifest that IS
    // deleted, but the retained consolidated manifest also lists it -> the
    // data file itself must survive.
    assert!(!result.candidate_data_files.contains(&f3_path));
    assert!(table.file_io().exists(&f3_path).await.unwrap());
    assert!(!result.candidate_manifests.is_empty());
    for manifest in &result.candidate_manifests {
        assert!(
            !retained_manifests.contains(manifest),
            "retained manifest in diff"
        );
        assert!(!table.file_io().exists(manifest).await.unwrap());
    }
    for manifest in &retained_manifests {
        assert!(table.file_io().exists(manifest).await.unwrap());
    }

    assert!(result.failed_deletes.is_empty());

    // The table still scans the full compacted content.
    let table = reload(&catalog, table.identifier()).await;
    assert_eq!(table.metadata().snapshots().len(), 1);
    assert_eq!(live_row_count(&table).await, 8);
}

/// Deletion-vector lifecycle: a reabsorbed DV's Puffin file is freed once no
/// retained snapshot's manifests mention it, and lands in the delete-file
/// category of the report.
#[tokio::test]
async fn expire_frees_reabsorbed_deletion_vector() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, table) = create_table(&warehouse).await;

    // S1: f1 = [1..4].
    let f1 = write_one_data_file(&table, "data-0", vec![1, 2, 3, 4]).await;
    let f1_path = f1[0].file_path().to_string();
    let table = fast_append(&catalog, &table, f1.clone()).await;

    // S2: deletion vector dropping position 0 (id=1) -> row_delta.
    let mut positions = RoaringTreemap::new();
    positions.insert(0);
    let dv_path = format!("{}/data/dv-0.puffin", table.metadata().location());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path.clone(),
            f1_path.clone(),
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_delete_files(vec![dv_file.clone()])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_row_count(&table).await, 3, "DV drops id=1");

    // S3: compaction — f2 = [2,3,4] replaces f1 and reabsorbs the DV.
    let f2 = write_one_data_file(&table, "compacted-0", vec![2, 3, 4]).await;
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .add_data_files(f2.clone())
        .delete_data_files(f1)
        .delete_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_row_count(&table).await, 3);

    // S4: a second rewrite drops the now-dead manifests (all-DELETED entries
    // are not carried forward) — f3 = [2,3,4] replaces f2.
    let f3 = write_one_data_file(&table, "compacted-1", vec![2, 3, 4]).await;
    let f3_path = f3[0].file_path().to_string();
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .add_data_files(f3)
        .delete_data_files(f2)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // S5: manifest rewrite so the head references only live entries.
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_manifests()
        .min_input_manifests(1)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
        .older_than_ms(future_cutoff_ms())
        .retain_last(1)
        .execute(&catalog)
        .await
        .unwrap();

    assert_eq!(result.removed_snapshot_ids.len(), 4, "S1-S4 expired");

    // The DV's Puffin file is exclusive now -> categorized + deleted.
    assert_eq!(result.candidate_delete_files, vec![dv_path.clone()]);
    assert!(!table.file_io().exists(&dv_path).await.unwrap());
    // The replaced data files went with it; the live file survives.
    assert_eq!(result.candidate_data_files.len(), 2, "f1 + f2");
    assert!(!result.candidate_data_files.contains(&f3_path));
    assert!(table.file_io().exists(&f3_path).await.unwrap());
    assert!(result.failed_deletes.is_empty());

    let table = reload(&catalog, table.identifier()).await;
    assert_eq!(live_row_count(&table).await, 3);
}

/// Statistics files of expired snapshots are removed from metadata AND from
/// storage; a retained snapshot's statistics survive both.
#[tokio::test]
async fn expire_deletes_statistics_of_expired_snapshots_only() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, mut table) = create_table(&warehouse).await;

    let mut stats_paths: Vec<(i64, String)> = vec![];
    for (i, ids) in [vec![1, 2, 3, 4], vec![5, 6, 7, 8]].into_iter().enumerate() {
        let files = write_one_data_file(&table, &format!("data-{i}"), ids).await;
        table = fast_append(&catalog, &table, files).await;
        let snapshot_id = table.metadata().current_snapshot_id().unwrap();

        // Plant a real (placeholder-content) stats file and register it.
        let stats_path = format!("{}/metadata/stats-{i}.puffin", table.metadata().location());
        table
            .file_io()
            .new_output(&stats_path)
            .unwrap()
            .write(bytes::Bytes::from_static(b"stats"))
            .await
            .unwrap();
        let tx = Transaction::new(&table);
        table = tx
            .update_statistics()
            .set_statistics(StatisticsFile {
                snapshot_id,
                statistics_path: stats_path.clone(),
                file_size_in_bytes: 5,
                file_footer_size_in_bytes: 1,
                key_metadata: None,
                blob_metadata: vec![],
            })
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        stats_paths.push((snapshot_id, stats_path));
    }

    let (expired_snapshot, expired_stats) = stats_paths[0].clone();
    let (_, retained_stats) = stats_paths[1].clone();

    let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
        .older_than_ms(future_cutoff_ms())
        .retain_last(1)
        .execute(&catalog)
        .await
        .unwrap();

    assert_eq!(result.removed_snapshot_ids, vec![expired_snapshot]);
    assert_eq!(result.candidate_stats_files, vec![expired_stats.clone()]);
    assert!(!table.file_io().exists(&expired_stats).await.unwrap());
    assert!(table.file_io().exists(&retained_stats).await.unwrap());

    // The expired snapshot's statistics entry is gone from metadata; the
    // retained snapshot keeps its entry.
    let table = reload(&catalog, table.identifier()).await;
    let remaining: Vec<i64> = table
        .metadata()
        .statistics_iter()
        .map(|s| s.snapshot_id)
        .collect();
    assert_eq!(remaining, vec![stats_paths[1].0]);
    assert_eq!(live_row_count(&table).await, 8);
}

/// Dry run computes the full report but commits nothing and deletes nothing.
#[tokio::test]
async fn expire_dry_run_changes_nothing() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, mut table) = create_table(&warehouse).await;
    for (i, ids) in [vec![1, 2, 3, 4], vec![5, 6, 7, 8]].into_iter().enumerate() {
        let files = write_one_data_file(&table, &format!("data-{i}"), ids).await;
        table = fast_append(&catalog, &table, files).await;
    }
    let all_lists: Vec<String> = table
        .metadata()
        .snapshots()
        .map(|s| s.manifest_list().to_string())
        .collect();

    let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
        .older_than_ms(future_cutoff_ms())
        .retain_last(1)
        .dry_run(true)
        .execute(&catalog)
        .await
        .unwrap();

    assert!(result.dry_run);
    assert_eq!(result.removed_snapshot_ids.len(), 1);
    assert_eq!(result.candidate_manifest_lists.len(), 1);
    assert!(result.deleted_files.is_empty());
    assert!(result.failed_deletes.is_empty());

    // Nothing changed: both snapshots still in metadata, every file on disk.
    let table = reload(&catalog, table.identifier()).await;
    assert_eq!(table.metadata().snapshots().len(), 2);
    for list in &all_lists {
        assert!(table.file_io().exists(list).await.unwrap());
    }
    assert_eq!(live_row_count(&table).await, 8);
}
