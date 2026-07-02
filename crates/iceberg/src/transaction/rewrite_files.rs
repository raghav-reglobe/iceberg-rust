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
}
