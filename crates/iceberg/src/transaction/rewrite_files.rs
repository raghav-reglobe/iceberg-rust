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

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// Transaction action for compaction-style rewrites: replace existing data files
/// with new (compacted) ones and reabsorb their delete files — position/equality
/// deletes and V3 deletion vectors — in a single `Replace` snapshot.
///
/// Corresponds to `org.apache.iceberg.RewriteFiles` in the Java implementation.
/// Unlike [`RowDeltaAction`](crate::transaction::row_delta) (row-level
/// merge-on-read deltas, which *add* delete files), a rewrite **preserves table
/// data** — it only reorganizes files (`Operation::Replace`): compaction,
/// changing the data file format, or relocating data files.
pub struct RewriteFilesAction {
    added_data_files: Vec<DataFile>,
    removed_data_files: Vec<DataFile>,
    /// Delete files (incl. V3 deletion vectors) to reabsorb — removed because the
    /// rewritten data files have the deletes already applied.
    removed_delete_files: Vec<DataFile>,
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    starting_snapshot_id: Option<i64>,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            removed_data_files: vec![],
            removed_delete_files: vec![],
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            starting_snapshot_id: None,
        }
    }

    /// Add the rewritten (compacted) data files.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Mark the data files being replaced as removed (DELETED in the new
    /// snapshot). Corresponds to `RewriteFiles.deleteFile(DataFile)` in Java.
    pub fn delete_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_data_files.extend(data_files);
        self
    }

    /// Mark delete files (position/equality deletes, incl. V3 deletion vectors)
    /// as removed — reabsorbed into the rewritten data files, so no delete file
    /// references the replaced data afterward.
    pub fn delete_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_delete_files.extend(delete_files);
        self
    }

    /// Set the commit UUID used for manifest file naming.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Attach custom key/value metadata to the snapshot summary.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Reject the commit if the table has advanced past `snapshot_id` — the
    /// rewrite was planned against this snapshot (optimistic concurrency).
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.starting_snapshot_id = Some(snapshot_id);
        self
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if let Some(expected_snapshot_id) = self.starting_snapshot_id
            && table.metadata().current_snapshot_id() != Some(expected_snapshot_id)
        {
            return Err(crate::Error::new(
                crate::ErrorKind::DataInvalid,
                format!(
                    "Cannot commit RewriteFiles based on stale snapshot. Expected: {}, Current: {:?}",
                    expected_snapshot_id,
                    table.metadata().current_snapshot_id()
                ),
            ));
        }

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
        );

        // Validate the newly added (rewritten) data files. removed_data_files and
        // removed_delete_files are existing table files — already validated when
        // first committed — and a rewrite adds no new delete files.
        snapshot_producer.validate_added_data_files()?;

        let operation = RewriteFilesOperation {
            removed_data_files: self.removed_data_files.clone(),
            removed_delete_files: self.removed_delete_files.clone(),
        };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

struct RewriteFilesOperation {
    removed_data_files: Vec<DataFile>,
    removed_delete_files: Vec<DataFile>,
}

impl SnapshotProduceOperation for RewriteFilesOperation {
    /// A compaction rewrite preserves table data, so the operation is
    /// `Replace` ("compaction, changing the data file format, or relocating
    /// data files"), never `Overwrite`/`Delete`.
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    /// Delete entries are handled inside `existing_manifest` by rewriting the
    /// manifests in place.
    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    /// Rewrite the previous snapshot's manifests: replaced data files (Data
    /// manifests) and reabsorbed delete files / DVs (Deletes manifests) become
    /// DELETED entries, survivors EXISTING. Shared with RowDelta via the
    /// `SnapshotProducer` helper.
    async fn existing_manifest(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        snapshot_produce
            .rewrite_existing_manifests_removing(
                &self.removed_data_files,
                &self.removed_delete_files,
            )
            .await
    }

    fn removed_data_files(&self) -> &[DataFile] {
        &self.removed_data_files
    }

    fn removed_delete_files(&self) -> &[DataFile] {
        &self.removed_delete_files
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use arrow_array::{BooleanArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use crate::catalog::TableCommit;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{
        DataFile, DataFileFormat, FormatVersion, Literal, ManifestStatus, NestedField,
        PartitionKey, PrimitiveType, Schema, Struct, Transform, Type, UnboundPartitionSpec,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent, TableUpdate};

    async fn write_data_file(
        table: &Table,
        name: &str,
        ids: Vec<i32>,
        flag: bool,
        partition_key: Option<PartitionKey>,
    ) -> DataFile {
        let schema = table.metadata().current_schema().clone();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            // Unique prefix per file — DefaultFileNameGenerator's counter resets
            // per instance, so a shared prefix would collide paths.
            DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("flag", DataType::Boolean, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let n = ids.len();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(BooleanArray::from(vec![flag; n])),
        ])
        .unwrap();
        let mut writer = DataFileWriterBuilder::new(rolling)
            .build(partition_key)
            .await
            .unwrap();
        writer.write(batch).await.unwrap();
        let files = writer.close().await.unwrap();
        assert_eq!(files.len(), 1);
        files.into_iter().next().unwrap()
    }

    async fn scan_ids(table: &Table) -> Vec<i32> {
        let mut stream = table
            .scan()
            .select_all()
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        let mut ids = Vec::new();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let idx = batch.schema().index_of("id").unwrap();
            let col = batch
                .column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend(col.iter().flatten());
        }
        ids.sort_unstable();
        ids
    }

    /// Regression: a `RewriteFiles` commit on a table whose current snapshot
    /// still holds live manifests written under an OLDER partition spec must
    /// bind each rewritten (carried-forward) manifest to its SOURCE manifest's
    /// spec — a manifest binds exactly ONE partition spec (Java parity:
    /// `SnapshotProducer`/`ManifestFilterManager` keep manifests grouped by
    /// spec id).
    ///
    /// Before the fix, `rewrite_existing_manifests_removing` created every
    /// output writer from the table's current DEFAULT spec, so rewriting a
    /// manifest from an older spec wrote entries whose partition tuple arity
    /// differed from the writer's spec and panicked in
    /// `construct_partition_summaries`: "itertools: .zip_eq() reached end of
    /// one iterator before the other" (verified failing with exactly that
    /// panic before the fix: spec-0 unpartitioned entries carried through a
    /// writer bound to a 1-field identity spec).
    #[tokio::test]
    async fn rewrite_binds_carried_manifests_to_their_source_spec() {
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
                NestedField::optional(2, "flag", Type::Primitive(PrimitiveType::Boolean)).into(),
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
        let spec0_id = table.metadata().default_partition_spec_id();

        // S1: append A + B under the UNPARTITIONED spec-0 → one data manifest
        // whose entries carry ZERO partition literals.
        let file_a = write_data_file(&table, "file-a", vec![1, 2, 3], false, None).await;
        let file_b = write_data_file(&table, "file-b", vec![4, 5, 6], false, None).await;
        let a_path = file_a.file_path().to_string();
        let b_path = file_b.file_path().to_string();
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files(vec![file_a.clone(), file_b])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        let s1_id = table.metadata().current_snapshot_id().unwrap();

        // Evolve the partition spec: add an identity field on `flag` and make
        // it the default (spec-1, 1 partition field).
        let evolve = TableCommit::builder()
            .ident(ident.clone())
            .updates(vec![
                TableUpdate::AddSpec {
                    spec: UnboundPartitionSpec::builder()
                        .add_partition_field(2, "flag".to_string(), Transform::Identity)
                        .unwrap()
                        .build(),
                },
                TableUpdate::SetDefaultSpec { spec_id: -1 },
            ])
            .requirements(vec![])
            .build();
        let table = catalog.update_table(evolve).await.unwrap();
        let spec1_id = table.metadata().default_partition_spec_id();
        assert_ne!(spec0_id, spec1_id, "spec evolution must mint a new spec id");
        let spec1_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::bool(true))]),
        );

        // S2: append C under spec-1 (entries carry ONE partition literal).
        let file_c = write_data_file(
            &table,
            "file-c",
            vec![7, 8, 9],
            true,
            Some(spec1_key.clone()),
        )
        .await;
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files(vec![file_c])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        // S3: compaction-style rewrite replacing A (a spec-0 file) with A2
        // (written under the current spec-1). This forces the carry-forward to
        // REWRITE the spec-0 manifest (A → DELETED, B → EXISTING); the pre-fix
        // code panicked here.
        let file_a2 = write_data_file(
            &table,
            "file-a2",
            vec![101, 102, 103],
            true,
            Some(spec1_key),
        )
        .await;
        let tx = Transaction::new(&table);
        let table = tx
            .rewrite_files()
            .delete_data_files(vec![file_a])
            .add_data_files(vec![file_a2])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        // The new snapshot's manifest list must cover BOTH specs, and every
        // manifest must be bound to the spec its entries were written under.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let mut spec_ids = HashSet::new();
        let mut saw_rewritten_spec0_manifest = false;
        for manifest_file in manifest_list.entries() {
            spec_ids.insert(manifest_file.partition_spec_id);
            let expected_arity = table
                .metadata()
                .partition_spec_by_id(manifest_file.partition_spec_id)
                .expect("manifest bound to a spec present in table metadata")
                .fields()
                .len();
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                assert_eq!(
                    entry.data_file().partition().fields().len(),
                    expected_arity,
                    "entry partition arity must match the manifest's bound spec"
                );
            }
            if manifest_file.partition_spec_id == spec0_id {
                // The rewritten spec-0 manifest: A DELETED, B EXISTING with its
                // original (S1) lineage preserved.
                saw_rewritten_spec0_manifest = true;
                assert_eq!(manifest.entries().len(), 2);
                for entry in manifest.entries() {
                    match entry.file_path() {
                        p if p == a_path => assert_eq!(entry.status(), ManifestStatus::Deleted),
                        p if p == b_path => {
                            assert_eq!(entry.status(), ManifestStatus::Existing);
                            assert_eq!(entry.snapshot_id(), Some(s1_id), "B keeps S1 lineage");
                        }
                        other => panic!("unexpected file in the spec-0 manifest: {other}"),
                    }
                }
            }
        }
        assert!(saw_rewritten_spec0_manifest);
        assert_eq!(
            spec_ids,
            HashSet::from([spec0_id, spec1_id]),
            "manifest list must keep per-spec manifest binding across both specs"
        );

        // Full scan: A's rows replaced by A2's; B and C intact.
        assert_eq!(scan_ids(&table).await, vec![
            4, 5, 6, 7, 8, 9, 101, 102, 103
        ]);
    }
}
