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
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{ManifestContentType, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// Table property controlling the target size of rewritten manifests.
const COMMIT_MANIFEST_TARGET_SIZE_BYTES: &str = "commit.manifest.target-size-bytes";
const COMMIT_MANIFEST_TARGET_SIZE_BYTES_DEFAULT: u64 = 8 * 1024 * 1024;

/// Snapshot summary properties, matching Java's `BaseRewriteManifests`.
const KEPT_MANIFESTS_COUNT: &str = "manifests-kept";
const CREATED_MANIFESTS_COUNT: &str = "manifests-created";
const REPLACED_MANIFESTS_COUNT: &str = "manifests-replaced";
const PROCESSED_ENTRY_COUNT: &str = "entries-processed";

/// Transaction action that consolidates the current snapshot's **data**
/// manifests into fewer, target-sized manifests without changing any data
/// (`Operation::Replace`). Corresponds to `org.apache.iceberg.RewriteManifests`
/// (Java `BaseRewriteManifests`).
///
/// Semantics:
/// - Only `content=Data` manifests written under the table's **default
///   partition spec** are rewritten; delete manifests and older-spec data
///   manifests are carried forward unchanged (Java parity — rewriting across
///   partition specs requires per-spec grouping, not implemented).
/// - ALIVE entries (`Added`/`Existing`) are carried into the new manifests as
///   `Existing`, preserving their original `snapshot_id`, data sequence number
///   and file sequence number (resolved via manifest-list inheritance at load).
/// - Already-DELETED entries are **dropped**, never resurrected — carrying a
///   superseded entry forward as `Existing` would restore a removed file (for
///   V3 deletion vectors this yields multiple "live" DVs per data file and
///   readers fail with "Can't index multiple DVs"). The deletion stays recorded
///   in its originating snapshot's manifests.
/// - Manifests with no alive entries at all are dropped from the new snapshot.
///
/// Unlike [`RewriteFilesAction`](crate::transaction::rewrite_files), which
/// commits an externally-computed file plan, this action plans **inside**
/// `commit()` from the table state it is handed — so a transaction commit-retry
/// against a refreshed table re-plans against the new current snapshot instead
/// of re-applying a stale plan. Rewriting the refreshed snapshot's manifests is
/// always safe: the operation reorganizes metadata only.
pub struct RewriteManifestsAction {
    target_manifest_size_bytes: Option<u64>,
    min_input_manifests: usize,
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    starting_snapshot_id: Option<i64>,
}

impl RewriteManifestsAction {
    pub(crate) fn new() -> Self {
        Self {
            target_manifest_size_bytes: None,
            min_input_manifests: 2,
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            starting_snapshot_id: None,
        }
    }

    /// Target size of rewritten manifests in bytes. Defaults to the table's
    /// `commit.manifest.target-size-bytes` property, falling back to 8 MiB.
    pub fn target_manifest_size_bytes(mut self, size_bytes: u64) -> Self {
        self.target_manifest_size_bytes = Some(size_bytes);
        self
    }

    /// Minimum number of rewritable data manifests for the action to proceed;
    /// with fewer candidates `commit` fails with `PreconditionFailed` so
    /// callers can treat "nothing to consolidate" as a no-op. Defaults to 2;
    /// set 1 to force a rewrite (e.g. to purge DELETED entries from a single
    /// manifest).
    pub fn min_input_manifests(mut self, count: usize) -> Self {
        self.min_input_manifests = count.max(1);
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

    /// Reject the commit if the table has advanced past `snapshot_id` —
    /// optimistic concurrency for callers that planned externally. Usually
    /// unnecessary: the action re-plans from the current snapshot on retry.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.starting_snapshot_id = Some(snapshot_id);
        self
    }

    fn resolve_target_size(&self, table: &Table) -> u64 {
        if let Some(size) = self.target_manifest_size_bytes {
            return size.max(1);
        }
        table
            .metadata()
            .properties()
            .get(COMMIT_MANIFEST_TARGET_SIZE_BYTES)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(COMMIT_MANIFEST_TARGET_SIZE_BYTES_DEFAULT)
            .max(1)
    }
}

#[async_trait]
impl TransactionAction for RewriteManifestsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if let Some(expected_snapshot_id) = self.starting_snapshot_id
            && table.metadata().current_snapshot_id() != Some(expected_snapshot_id)
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot commit RewriteManifests based on stale snapshot. Expected: {}, Current: {:?}",
                    expected_snapshot_id,
                    table.metadata().current_snapshot_id()
                ),
            ));
        }

        let Some(snapshot) = table.metadata().current_snapshot() else {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "Cannot rewrite manifests: table has no current snapshot",
            ));
        };

        let manifest_list = table.manifest_list_reader(snapshot).load().await?;
        let default_spec_id = table.metadata().default_partition_spec_id();

        // Partition the current manifest list:
        // - dead manifests (no alive entries) are dropped from the new snapshot;
        // - Deletes manifests and data manifests written under a non-default
        //   partition spec are kept unchanged (a rewrite writer uses the default
        //   spec, so mixing specs would corrupt partition summaries);
        // - remaining data manifests are the rewrite candidates.
        let mut kept: Vec<ManifestFile> = Vec::new();
        let mut candidates: Vec<ManifestFile> = Vec::new();
        for manifest_file in manifest_list.entries() {
            if !manifest_file.has_added_files() && !manifest_file.has_existing_files() {
                continue;
            }
            if manifest_file.content == ManifestContentType::Data
                && manifest_file.partition_spec_id == default_spec_id
            {
                candidates.push(manifest_file.clone());
            } else {
                kept.push(manifest_file.clone());
            }
        }

        if candidates.len() < self.min_input_manifests {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                format!(
                    "Nothing to rewrite: {} rewritable data manifest(s), need at least {}",
                    candidates.len(),
                    self.min_input_manifests
                ),
            ));
        }

        // Collect alive entries from the candidate manifests. Entries have their
        // snapshot_id / sequence numbers resolved by manifest-list inheritance at
        // load time (`ManifestEntry::inherit_data`), so writing them as EXISTING
        // preserves the original commit lineage.
        let mut alive_entries: Vec<ManifestEntry> = Vec::new();
        let mut total_entries: u64 = 0;
        let mut total_bytes: u64 = 0;
        for manifest_file in &candidates {
            let manifest = manifest_file.load_manifest(table.file_io()).await?;
            total_entries += manifest.entries().len() as u64;
            total_bytes += manifest_file.manifest_length.max(0) as u64;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    // Already-DELETED entry — DROP. Never resurrect (see doc).
                    continue;
                }
                // An Existing/Deleted entry must carry explicit sequence numbers
                // (spec) and an Added entry resolves them via inheritance at load.
                // Anything still unresolved would silently adopt the NEW
                // snapshot's sequence number when rewritten — files would appear
                // newer than they are and deletes would stop applying. Fail loud.
                if entry.sequence_number.is_none() || entry.file_sequence_number.is_none() {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot rewrite manifest entry with unresolved sequence numbers: {}",
                            entry.file_path()
                        ),
                    ));
                }
                alive_entries.push((**entry).clone());
            }
        }

        // Bin-pack alive entries into target-sized manifests using the average
        // per-entry byte size of the source manifests as the estimator (the
        // writer exposes no running length).
        let target_size = self.resolve_target_size(table);
        let avg_entry_bytes = if total_entries > 0 {
            (total_bytes / total_entries).max(1)
        } else {
            1
        };
        let entries_per_manifest = ((target_size / avg_entry_bytes).max(1)) as usize;

        let chunks: Vec<Vec<ManifestEntry>> = alive_entries
            .chunks(entries_per_manifest)
            .map(|c| c.to_vec())
            .collect();

        let mut snapshot_properties = self.snapshot_properties.clone();
        snapshot_properties.insert(
            REPLACED_MANIFESTS_COUNT.to_string(),
            candidates.len().to_string(),
        );
        snapshot_properties.insert(KEPT_MANIFESTS_COUNT.to_string(), kept.len().to_string());
        snapshot_properties.insert(
            CREATED_MANIFESTS_COUNT.to_string(),
            chunks.len().to_string(),
        );
        snapshot_properties.insert(PROCESSED_ENTRY_COUNT.to_string(), total_entries.to_string());

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            snapshot_properties,
            vec![],
        );

        let operation = RewriteManifestsOperation {
            expected_entries: alive_entries.len(),
            chunks: Mutex::new(Some(chunks)),
            kept_manifests: kept,
        };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

struct RewriteManifestsOperation {
    /// Number of alive entries collected at plan time — every one of them must
    /// be written into the new manifests (live-file preservation invariant).
    expected_entries: usize,
    chunks: Mutex<Option<Vec<Vec<ManifestEntry>>>>,
    kept_manifests: Vec<ManifestFile>,
}

impl SnapshotProduceOperation for RewriteManifestsOperation {
    /// Manifest consolidation preserves table data — `Replace`, like compaction.
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    /// Write the consolidated data manifests (all entries EXISTING with
    /// original lineage) and carry the kept manifests forward unchanged.
    async fn existing_manifest(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let chunks = self
            .chunks
            .lock()
            .map_err(|_| Error::new(ErrorKind::Unexpected, "rewrite plan lock poisoned"))?
            .take()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "RewriteManifests plan already consumed",
                )
            })?;

        let mut result = self.kept_manifests.clone();
        let mut written = 0usize;
        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }
            let mut writer = snapshot_produce.new_manifest_writer(ManifestContentType::Data)?;
            for entry in chunk {
                writer.add_existing_entry(entry)?;
                written += 1;
            }
            result.push(writer.write_manifest_file().await?);
        }

        // Live-file preservation invariant: every alive entry collected at plan
        // time must land in the new snapshot.
        if written != self.expected_entries {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "RewriteManifests lost entries: planned {}, wrote {}",
                    self.expected_entries, written
                ),
            ));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH,
        ManifestContentType, ManifestStatus, Operation, Struct, TableMetadataBuilder,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::transaction::{Transaction, TransactionAction};
    use crate::{ErrorKind, TableIdent, TableUpdate};

    fn make_data_file(table: &Table, path: &str, size: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap()
    }

    /// Build a table that has `snapshot` as its current snapshot, backed by the same FileIO.
    async fn table_with_snapshot(base: &Table, snapshot: crate::spec::Snapshot) -> Table {
        let updated_metadata =
            TableMetadataBuilder::new_from_metadata(base.metadata_ref().as_ref().clone(), None)
                .set_branch_snapshot(snapshot, MAIN_BRANCH)
                .unwrap()
                .build()
                .unwrap()
                .metadata;

        Table::builder()
            .metadata(updated_metadata)
            .metadata_location("s3://bucket/test/location/metadata/v2.json".to_string())
            .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
            .file_io(base.file_io().clone())
            .runtime(crate::test_utils::test_runtime())
            .build()
            .unwrap()
    }

    fn take_snapshot(update: TableUpdate) -> crate::spec::Snapshot {
        if let TableUpdate::AddSnapshot { snapshot } = update {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        }
    }

    /// Two appends → two data manifests → rewrite consolidates to ONE manifest
    /// with both files EXISTING, original snapshot ids + sequence numbers
    /// preserved.
    #[tokio::test]
    async fn test_rewrite_manifests_consolidates() {
        let base = make_v2_minimal_table();

        // S1: append file-a.
        let file_a = make_data_file(&base, "test/a.parquet", 100);
        let mut c1 = Arc::new(
            Transaction::new(&base)
                .fast_append()
                .add_data_files(vec![file_a]),
        )
        .commit(&base)
        .await
        .unwrap();
        let snap1 = take_snapshot(c1.take_updates().into_iter().next().unwrap());
        let snap1_id = snap1.snapshot_id();
        let snap1_seq = snap1.sequence_number();
        let table_s1 = table_with_snapshot(&base, snap1).await;

        // S2: append file-b.
        let file_b = make_data_file(&table_s1, "test/b.parquet", 200);
        let mut c2 = Arc::new(
            Transaction::new(&table_s1)
                .fast_append()
                .add_data_files(vec![file_b]),
        )
        .commit(&table_s1)
        .await
        .unwrap();
        let snap2 = take_snapshot(c2.take_updates().into_iter().next().unwrap());
        let snap2_id = snap2.snapshot_id();
        let snap2_seq = snap2.sequence_number();
        let table_s2 = table_with_snapshot(&table_s1, snap2).await;

        // Sanity: S2 has two data manifests.
        let list_s2 = table_s2
            .manifest_list_reader(table_s2.metadata().current_snapshot().unwrap())
            .load()
            .await
            .unwrap();
        assert_eq!(
            list_s2.entries().len(),
            2,
            "expected 2 manifests before rewrite"
        );

        // S3: rewrite manifests.
        let mut c3 = Arc::new(Transaction::new(&table_s2).rewrite_manifests())
            .commit(&table_s2)
            .await
            .unwrap();
        let updates3 = c3.take_updates();
        let snap3 = if let TableUpdate::AddSnapshot { ref snapshot } = updates3[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };

        assert_eq!(snap3.summary().operation, Operation::Replace);
        let props = &snap3.summary().additional_properties;
        assert_eq!(
            props.get("manifests-replaced").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            props.get("manifests-created").map(String::as_str),
            Some("1")
        );
        assert_eq!(props.get("manifests-kept").map(String::as_str), Some("0"));

        let list_s3 = table_s2
            .manifest_list_reader(&std::sync::Arc::new(snap3.clone()))
            .load()
            .await
            .unwrap();
        assert_eq!(
            list_s3.entries().len(),
            1,
            "expected 1 consolidated manifest"
        );
        let manifest = list_s3.entries()[0]
            .load_manifest(table_s2.file_io())
            .await
            .unwrap();
        assert_eq!(manifest.entries().len(), 2);
        for entry in manifest.entries() {
            assert_eq!(entry.status(), ManifestStatus::Existing);
            match entry.file_path() {
                "test/a.parquet" => {
                    assert_eq!(
                        entry.snapshot_id(),
                        Some(snap1_id),
                        "file-a keeps S1 lineage"
                    );
                    assert_eq!(entry.sequence_number(), Some(snap1_seq));
                }
                "test/b.parquet" => {
                    assert_eq!(
                        entry.snapshot_id(),
                        Some(snap2_id),
                        "file-b keeps S2 lineage"
                    );
                    assert_eq!(entry.sequence_number(), Some(snap2_seq));
                }
                other => panic!("unexpected file {other}"),
            }
        }
    }

    /// The resurrection guard: a DELETED entry in a rewritten manifest is
    /// dropped, not carried forward — the removed file must NOT reappear.
    #[tokio::test]
    async fn test_rewrite_manifests_drops_deleted_entries() {
        let base = make_v2_minimal_table();

        // S1: append a + b.
        let file_a = make_data_file(&base, "test/a.parquet", 100);
        let file_b = make_data_file(&base, "test/b.parquet", 200);
        let mut c1 = Arc::new(
            Transaction::new(&base)
                .fast_append()
                .add_data_files(vec![file_a.clone(), file_b]),
        )
        .commit(&base)
        .await
        .unwrap();
        let snap1 = take_snapshot(c1.take_updates().into_iter().next().unwrap());
        let table_s1 = table_with_snapshot(&base, snap1).await;

        // S2: compaction-style rewrite a -> a2 (leaves a DELETED entry in the
        // rewritten manifest + a2 in an added manifest).
        let file_a2 = make_data_file(&table_s1, "test/a2.parquet", 100);
        let mut c2 = Arc::new(
            Transaction::new(&table_s1)
                .rewrite_files()
                .delete_data_files(vec![file_a])
                .add_data_files(vec![file_a2]),
        )
        .commit(&table_s1)
        .await
        .unwrap();
        let snap2 = take_snapshot(c2.take_updates().into_iter().next().unwrap());
        let table_s2 = table_with_snapshot(&table_s1, snap2).await;

        // S3: rewrite manifests.
        let mut c3 = Arc::new(Transaction::new(&table_s2).rewrite_manifests())
            .commit(&table_s2)
            .await
            .unwrap();
        let updates3 = c3.take_updates();
        let snap3 = if let TableUpdate::AddSnapshot { ref snapshot } = updates3[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };

        let list_s3 = table_s2
            .manifest_list_reader(&std::sync::Arc::new(snap3.clone()))
            .load()
            .await
            .unwrap();
        let mut seen = std::collections::HashSet::new();
        for manifest_file in list_s3.entries() {
            let manifest = manifest_file
                .load_manifest(table_s2.file_io())
                .await
                .unwrap();
            for entry in manifest.entries() {
                assert_eq!(entry.status(), ManifestStatus::Existing);
                assert!(entry.is_alive());
                seen.insert(entry.file_path().to_string());
            }
        }
        assert!(
            !seen.contains("test/a.parquet"),
            "removed file must NOT be resurrected by a manifest rewrite"
        );
        assert_eq!(
            seen,
            ["test/b.parquet", "test/a2.parquet"]
                .into_iter()
                .map(String::from)
                .collect::<std::collections::HashSet<_>>()
        );
    }

    /// Delete manifests are kept unchanged — only data manifests are rewritten.
    #[tokio::test]
    async fn test_rewrite_manifests_keeps_delete_manifests() {
        let base = make_v2_minimal_table();

        // S1: append a + b (via two appends → 2 data manifests so the rewrite
        // has candidates).
        let file_a = make_data_file(&base, "test/a.parquet", 100);
        let mut c1 = Arc::new(
            Transaction::new(&base)
                .fast_append()
                .add_data_files(vec![file_a]),
        )
        .commit(&base)
        .await
        .unwrap();
        let snap1 = take_snapshot(c1.take_updates().into_iter().next().unwrap());
        let table_s1 = table_with_snapshot(&base, snap1).await;

        let file_b = make_data_file(&table_s1, "test/b.parquet", 200);
        let mut c2 = Arc::new(
            Transaction::new(&table_s1)
                .fast_append()
                .add_data_files(vec![file_b]),
        )
        .commit(&table_s1)
        .await
        .unwrap();
        let snap2 = take_snapshot(c2.take_updates().into_iter().next().unwrap());
        let table_s2 = table_with_snapshot(&table_s1, snap2).await;

        // S3: MoR position delete on file-a → a content=Deletes manifest.
        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/pos-delete.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table_s2.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .referenced_data_file(Some("test/a.parquet".to_string()))
            .build()
            .unwrap();
        let mut c3 = Arc::new(
            Transaction::new(&table_s2)
                .row_delta()
                .add_delete_files(vec![delete_file]),
        )
        .commit(&table_s2)
        .await
        .unwrap();
        let snap3 = take_snapshot(c3.take_updates().into_iter().next().unwrap());
        let table_s3 = table_with_snapshot(&table_s2, snap3).await;

        // S4: rewrite manifests — the two data manifests consolidate, the
        // delete manifest is carried forward unchanged.
        let mut c4 = Arc::new(Transaction::new(&table_s3).rewrite_manifests())
            .commit(&table_s3)
            .await
            .unwrap();
        let updates4 = c4.take_updates();
        let snap4 = if let TableUpdate::AddSnapshot { ref snapshot } = updates4[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };
        let props = &snap4.summary().additional_properties;
        assert_eq!(
            props.get("manifests-replaced").map(String::as_str),
            Some("2")
        );
        assert_eq!(props.get("manifests-kept").map(String::as_str), Some("1"));

        let list_s4 = table_s3
            .manifest_list_reader(&std::sync::Arc::new(snap4.clone()))
            .load()
            .await
            .unwrap();
        let data_manifests: Vec<_> = list_s4
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Data)
            .collect();
        let delete_manifests: Vec<_> = list_s4
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Deletes)
            .collect();
        assert_eq!(data_manifests.len(), 1, "data manifests consolidated");
        assert_eq!(delete_manifests.len(), 1, "delete manifest carried forward");
        let dv_manifest = delete_manifests[0]
            .load_manifest(table_s3.file_io())
            .await
            .unwrap();
        assert_eq!(dv_manifest.entries().len(), 1);
        assert_eq!(
            dv_manifest.entries()[0].data_file().content_type(),
            DataContentType::PositionDeletes
        );
    }

    /// Fewer candidates than `min_input_manifests` → PreconditionFailed (no-op
    /// for callers).
    #[tokio::test]
    async fn test_rewrite_manifests_nothing_to_do() {
        let base = make_v2_minimal_table();
        let file_a = make_data_file(&base, "test/a.parquet", 100);
        let mut c1 = Arc::new(
            Transaction::new(&base)
                .fast_append()
                .add_data_files(vec![file_a]),
        )
        .commit(&base)
        .await
        .unwrap();
        let snap1 = take_snapshot(c1.take_updates().into_iter().next().unwrap());
        let table_s1 = table_with_snapshot(&base, snap1).await;

        let result = Arc::new(Transaction::new(&table_s1).rewrite_manifests())
            .commit(&table_s1)
            .await;
        match result {
            Ok(_) => panic!("expected PreconditionFailed for a single manifest"),
            Err(e) => assert_eq!(e.kind(), ErrorKind::PreconditionFailed),
        }
    }

    /// Stale-snapshot validation mirrors RewriteFiles.
    #[tokio::test]
    async fn test_rewrite_manifests_validate_from_snapshot() {
        let base = make_v2_minimal_table();
        let result = Arc::new(
            Transaction::new(&base)
                .rewrite_manifests()
                .validate_from_snapshot(99999),
        )
        .commit(&base)
        .await;
        match result {
            Ok(_) => panic!("expected DataInvalid error for stale snapshot"),
            Err(e) => assert_eq!(e.kind(), ErrorKind::DataInvalid),
        }
    }
}
