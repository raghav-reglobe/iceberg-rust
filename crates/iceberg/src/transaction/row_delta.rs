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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFile, ManifestContentType, ManifestEntry, ManifestFile, ManifestStatus, Operation,
};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// Commit-time conflict-validation mode, resolved from the table's
/// `write.merge.isolation-level` property (Java `IsolationLevel` parity).
/// Any value other than `snapshot` — including absent or unrecognized —
/// resolves to `Serializable` (fail closed; serializable is also the Java
/// default for row-level operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IsolationLevel {
    Serializable,
    SnapshotIsolation,
}

fn isolation_level(props: &HashMap<String, String>) -> IsolationLevel {
    match props.get("write.merge.isolation-level") {
        Some(v) if v.trim().eq_ignore_ascii_case("snapshot") => IsolationLevel::SnapshotIsolation,
        _ => IsolationLevel::Serializable,
    }
}

/// A conflicting concurrent commit. Deliberately NON-retryable: retrying
/// would rebase-and-commit a semantically wrong result (e.g. a second
/// deletion vector on a data file another writer already covered — the V3
/// multi-DV corruption class). The caller must fail its run and re-plan
/// from its cursor against the new table state.
fn conflict(msg: String) -> crate::Error {
    crate::Error::new(
        crate::ErrorKind::DataInvalid,
        format!("Found conflicting concurrent commit: {msg}"),
    )
}

/// Transaction action for Copy-on-Write row-level modifications (UPDATE, DELETE, MERGE INTO).
///
/// Corresponds to `org.apache.iceberg.RowDelta` in the Java implementation.
pub struct RowDeltaAction {
    added_data_files: Vec<DataFile>,
    removed_data_files: Vec<DataFile>,
    /// MoR delete files (position/equality deletes, incl. V3 deletion vectors) to add.
    added_delete_files: Vec<DataFile>,
    /// MoR delete files to mark removed — e.g. a compaction reabsorbing a
    /// rewritten data file's deletion vectors.
    removed_delete_files: Vec<DataFile>,
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    starting_snapshot_id: Option<i64>,
    /// Set when the action was PLANNED against an empty table (no current
    /// snapshot): any snapshot present at commit time appeared concurrently
    /// and must be validated. Distinct from "no validation requested"
    /// (neither this nor `starting_snapshot_id` set — Java parity: a
    /// RowDelta without `validateFromSnapshot` performs no base validation).
    validate_empty_base: bool,
    /// Per-action isolation override: validate the skipped range at SNAPSHOT
    /// isolation regardless of the table's `write.merge.isolation-level`
    /// property (Java parity — the RowDelta CALLER chooses validation
    /// strictness via which `validate*` methods it invokes). Used by writers
    /// whose delete files carry exact file references (V3 DVs): concurrent
    /// pure appends are semantically irrelevant to them, while double-DV /
    /// referenced-file-removal conflicts still fail closed.
    snapshot_isolation_override: bool,
}

impl RowDeltaAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            removed_data_files: vec![],
            added_delete_files: vec![],
            removed_delete_files: vec![],
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            starting_snapshot_id: None,
            validate_empty_base: false,
            snapshot_isolation_override: false,
        }
    }

    /// Add new data files (INSERT rows or Copy-on-Write rewritten files).
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Mark existing data files as deleted (Copy-on-Write mode).
    ///
    /// Corresponds to `removeRows(DataFile)` in the Java implementation.
    pub fn remove_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_data_files.extend(data_files);
        self
    }

    /// Add Merge-on-Read delete files (position/equality deletes, incl. V3 deletion
    /// vectors). Written into a content=Deletes manifest at commit time.
    pub fn add_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_delete_files.extend(delete_files);
        self
    }

    /// Mark existing Merge-on-Read delete files (position/equality deletes, incl.
    /// V3 deletion vectors) as removed — e.g. a compaction that reabsorbs a
    /// rewritten data file's deletes. Written as DELETED entries in the
    /// content=Deletes manifest at commit (mirrors `remove_data_files` for data).
    pub fn remove_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
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

    /// Reject the commit if the table has advanced past `snapshot_id` (optimistic concurrency).
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.starting_snapshot_id = Some(snapshot_id);
        self
    }

    /// Validate the skipped range at SNAPSHOT isolation for THIS action,
    /// regardless of the table's `write.merge.isolation-level` property
    /// (Java parity — the RowDelta caller chooses validation strictness).
    /// Concurrent pure data APPENDS then pass; a concurrent delete file
    /// colliding with this commit's DV targets (double-DV), a wildcard
    /// equality delete, or a removal of a referenced file still conflicts.
    pub fn with_snapshot_isolation(mut self) -> Self {
        self.snapshot_isolation_override = true;
        self
    }

    /// Declare that this action was planned against an EMPTY table (no
    /// current snapshot). Any snapshot present at commit time then appeared
    /// concurrently and is validated like a skipped range. Use this when the
    /// plan-time snapshot id is None; without it, an action carrying no
    /// `validate_from_snapshot` performs no concurrency validation at all.
    pub fn validate_from_empty_table(mut self) -> Self {
        self.validate_empty_base = true;
        self
    }

    /// Validate the snapshots committed AFTER this action's base snapshot —
    /// the range an optimistic-concurrency rebase would silently skip. Mirrors
    /// Java's `MergingSnapshotProducer` validations driven by
    /// `write.merge.isolation-level`:
    ///
    /// - `serializable` (default, and the fail-closed fallback): ANY skipped
    ///   snapshot that added or removed files is a conflict. Manifest-
    ///   reorganization-only snapshots (`rewrite_manifests`: EXISTING entries
    ///   only) are allowed — they cannot change merge semantics.
    /// - `snapshot`: concurrent pure data APPENDS are allowed. Conflicts:
    ///   a new delete file (incl. V3 DV) whose `referenced_data_file` is a
    ///   file this commit also targets with a DV (double-DV) or that carries
    ///   no reference (equality delete — undecidable, fail closed); and any
    ///   concurrent REMOVAL of a file this commit references (DV target,
    ///   removed data file, or reabsorbed delete file).
    ///
    /// `base = None` means the action was planned against an empty table —
    /// every current snapshot is "skipped". Errors are NON-retryable (see
    /// `conflict`): the engine's backoff retry must not rebase across a
    /// semantic conflict.
    async fn validate_skipped_range(&self, table: &Table, base: Option<i64>) -> Result<()> {
        let meta = table.metadata();

        // 1. Collect the skipped snapshot ids, current -> base (base exclusive).
        let mut skipped: Vec<i64> = Vec::new();
        let mut cursor = meta.current_snapshot_id();
        loop {
            match cursor {
                None => {
                    if let Some(b) = base {
                        return Err(conflict(format!(
                            "base snapshot {b} is not an ancestor of the current snapshot \
                             (concurrent rollback, branch reset, or expired lineage)"
                        )));
                    }
                    break;
                }
                Some(id) if Some(id) == base => break,
                Some(id) => {
                    let snap = meta.snapshot_by_id(id).ok_or_else(|| {
                        conflict(format!(
                            "ancestor snapshot {id} is missing from table metadata — \
                             cannot validate the skipped range"
                        ))
                    })?;
                    skipped.push(id);
                    cursor = snap.parent_snapshot_id();
                }
            }
        }
        if skipped.is_empty() {
            return Ok(());
        }

        let isolation = if self.snapshot_isolation_override {
            IsolationLevel::SnapshotIsolation
        } else {
            isolation_level(meta.properties())
        };

        // 2. This commit's conflict sets.
        let our_dv_targets: HashSet<String> = self
            .added_delete_files
            .iter()
            .filter_map(|f| f.referenced_data_file())
            .collect();
        // An added delete file WITHOUT a referenced data file (equality
        // delete) can apply to any file — treat as a wildcard.
        let our_wildcard_delete = self
            .added_delete_files
            .iter()
            .any(|f| f.referenced_data_file().is_none());
        let our_removed_data: HashSet<&str> = self
            .removed_data_files
            .iter()
            .map(|f| f.file_path())
            .collect();
        let our_removed_deletes: HashSet<&str> = self
            .removed_delete_files
            .iter()
            .map(|f| f.file_path())
            .collect();

        // 3. Walk each skipped snapshot's DELTA manifests (added_snapshot_id
        //    == that snapshot). Removals are visible here too: a removal
        //    rewrites the affected manifest under the removing snapshot's id
        //    with DELETED entries.
        for id in skipped {
            let snap = meta
                .snapshot_by_id(id)
                .expect("skipped snapshot resolved above");
            let manifest_list = table.manifest_list_reader(snap).load().await?;
            for mf in manifest_list
                .entries()
                .iter()
                .filter(|m| m.added_snapshot_id == id)
            {
                // Unknown counts read as "has files" (fail closed).
                let mutating = mf.has_added_files() || mf.has_deleted_files();
                if !mutating {
                    continue; // manifest reorganization only — semantics unchanged
                }

                if isolation == IsolationLevel::Serializable {
                    return Err(conflict(format!(
                        "serializable isolation violation — concurrent snapshot {id} \
                         mutated files (manifest {}, content {:?}, added={:?} deleted={:?}) \
                         in the range skipped since base {base:?}",
                        mf.manifest_path, mf.content, mf.added_files_count, mf.deleted_files_count,
                    )));
                }

                // Snapshot isolation: pure data appends are allowed; anything
                // touching files this commit references needs the file-level check.
                let added_deletes =
                    mf.content == ManifestContentType::Deletes && mf.has_added_files();
                if !added_deletes && !mf.has_deleted_files() {
                    continue;
                }
                let manifest = mf.load_manifest(table.file_io()).await?;
                for entry in manifest.entries() {
                    match entry.status() {
                        ManifestStatus::Added if mf.content == ManifestContentType::Deletes => {
                            let refd = entry.data_file().referenced_data_file();
                            let collides = match refd.as_deref() {
                                Some(p) => our_wildcard_delete || our_dv_targets.contains(p),
                                // Concurrent equality delete — could apply to
                                // any of our rows. Fail closed.
                                None => true,
                            };
                            if collides {
                                return Err(conflict(format!(
                                    "concurrent snapshot {id} added delete file {} \
                                     (referenced_data_file={refd:?}) conflicting with this \
                                     commit's delete targets",
                                    entry.data_file().file_path(),
                                )));
                            }
                        }
                        ManifestStatus::Deleted => {
                            let p = entry.data_file().file_path();
                            let hit = match mf.content {
                                ManifestContentType::Data => {
                                    our_dv_targets.contains(p) || our_removed_data.contains(p)
                                }
                                ManifestContentType::Deletes => our_removed_deletes.contains(p),
                            };
                            if hit {
                                return Err(conflict(format!(
                                    "concurrent snapshot {id} removed {} which this commit \
                                     references (DV target, removed file, or reabsorbed delete)",
                                    p,
                                )));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl TransactionAction for RowDeltaAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Optimistic-concurrency validation (Java SnapshotValidator parity).
        // Fast path: table still at the action's base — nothing was skipped.
        // Otherwise validate the skipped range instead of the previous
        // unconditional stale-snapshot abort: a benign rebase (manifest
        // reorganization; concurrent appends under snapshot isolation) is
        // allowed, a semantic conflict aborts NON-retryably.
        let current = table.metadata().current_snapshot_id();
        match self.starting_snapshot_id {
            Some(base) if current == Some(base) => {}
            Some(base) => self.validate_skipped_range(table, Some(base)).await?,
            // Planned against an empty table (explicit opt-in): any current
            // snapshot appeared concurrently — the whole history is skipped.
            None if self.validate_empty_base && current.is_some() => {
                self.validate_skipped_range(table, None).await?
            }
            // No validation requested (Java parity).
            None => {}
        }

        let mut snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
        );

        // Validate newly added data files (partition value type-checks, etc.).
        // removed_data_files are not re-validated: they are existing table files that were
        // already validated when originally committed. This matches Java's MergingSnapshotProducer.
        snapshot_producer.validate_added_data_files()?;

        // MoR delete files (position/equality deletes, incl. V3 deletion vectors) are
        // written into a separate content=Deletes manifest by the snapshot producer.
        snapshot_producer.set_added_delete_files(self.added_delete_files.clone());

        let operation = RowDeltaOperation {
            removed_data_files: self.removed_data_files.clone(),
            removed_delete_files: self.removed_delete_files.clone(),
            has_added_data_files: !self.added_data_files.is_empty(),
            has_added_delete_files: !self.added_delete_files.is_empty(),
        };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

struct RowDeltaOperation {
    removed_data_files: Vec<DataFile>,
    removed_delete_files: Vec<DataFile>,
    has_added_data_files: bool,
    has_added_delete_files: bool,
}

impl SnapshotProduceOperation for RowDeltaOperation {
    /// Operation type (mirrors Java `BaseRowDelta.operation()`):
    /// - Any data files removed → `Overwrite`
    /// - MoR delete files added → `Overwrite` if data files also added, else `Delete`
    /// - Only data files added (or nothing) → `Append`
    fn operation(&self) -> Operation {
        if !self.removed_data_files.is_empty() || !self.removed_delete_files.is_empty() {
            Operation::Overwrite
        } else if self.has_added_delete_files {
            if self.has_added_data_files {
                Operation::Overwrite
            } else {
                Operation::Delete
            }
        } else {
            Operation::Append
        }
    }

    /// Delete entries are handled inside `existing_manifest` by rewriting the manifest.
    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    /// Returns manifest files for the new snapshot, rewriting any that contain
    /// removed data files (and reabsorbing removed delete files / DVs). Delegates
    /// to the shared `SnapshotProducer::rewrite_existing_manifests_removing`.
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
    use std::sync::Arc;

    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH,
        ManifestStatus, Struct, TableMetadataBuilder,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::transaction::{Transaction, TransactionAction};
    use crate::{TableIdent, TableUpdate};

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

    #[tokio::test]
    async fn test_row_delta_add_only() {
        let table = make_v2_minimal_table();
        let data_file = make_data_file(&table, "test/1.parquet", 100);
        let action = Transaction::new(&table)
            .row_delta()
            .add_data_files(vec![data_file]);

        let mut commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = commit.take_updates();

        if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            assert_eq!(snapshot.summary().operation, crate::spec::Operation::Append);
        } else {
            panic!("expected AddSnapshot");
        }
    }

    #[tokio::test]
    async fn test_row_delta_with_snapshot_properties() {
        let table = make_v2_minimal_table();
        let data_file = make_data_file(&table, "test/1.parquet", 100);
        let mut props = std::collections::HashMap::new();
        props.insert("key".to_string(), "value".to_string());
        let action = Transaction::new(&table)
            .row_delta()
            .set_snapshot_properties(props)
            .add_data_files(vec![data_file]);

        let mut commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = commit.take_updates();

        if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            assert_eq!(
                snapshot.summary().additional_properties.get("key").unwrap(),
                "value"
            );
        } else {
            panic!("expected AddSnapshot");
        }
    }

    #[tokio::test]
    async fn test_row_delta_validate_from_snapshot() {
        let table = make_v2_minimal_table();
        let data_file = make_data_file(&table, "test/1.parquet", 100);
        let action = Transaction::new(&table)
            .row_delta()
            .validate_from_snapshot(99999)
            .add_data_files(vec![data_file]);

        let result = Arc::new(action).commit(&table).await;
        match result {
            Ok(_) => panic!("expected DataInvalid error for stale snapshot"),
            Err(e) => assert_eq!(e.kind(), crate::ErrorKind::DataInvalid),
        }
    }

    // ── SnapshotValidator (skipped-range validation) tests ──────────────

    /// Rebuild `base`'s table with the given properties merged in.
    fn table_with_properties(
        base: &Table,
        props: std::collections::HashMap<String, String>,
    ) -> Table {
        let updated_metadata =
            TableMetadataBuilder::new_from_metadata(base.metadata_ref().as_ref().clone(), None)
                .set_properties(props)
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

    /// Commit `action_table` → one snapshot via fast_append of `paths`,
    /// returning the table advanced to that snapshot.
    async fn append_snapshot(table: &Table, paths: &[&str]) -> Table {
        let files: Vec<DataFile> = paths
            .iter()
            .map(|p| make_data_file(table, p, 100))
            .collect();
        let mut c = Arc::new(Transaction::new(table).fast_append().add_data_files(files))
            .commit(table)
            .await
            .unwrap();
        let snap = if let TableUpdate::AddSnapshot { snapshot } =
            c.take_updates().into_iter().next().unwrap()
        {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };
        table_with_snapshot(table, snap).await
    }

    fn make_dv(table: &Table, path: &str, referenced: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .referenced_data_file(Some(referenced.to_string()))
            .build()
            .unwrap()
    }

    /// Commit a RowDelta adding `dv` and return the table advanced to it.
    async fn dv_snapshot(table: &Table, dv: DataFile) -> Table {
        let mut c = Arc::new(
            Transaction::new(table)
                .row_delta()
                .add_delete_files(vec![dv]),
        )
        .commit(table)
        .await
        .unwrap();
        let snap = if let TableUpdate::AddSnapshot { snapshot } =
            c.take_updates().into_iter().next().unwrap()
        {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };
        table_with_snapshot(table, snap).await
    }

    #[tokio::test]
    async fn test_serializable_conflicts_on_concurrent_append_from_empty_base() {
        // Planned against an empty table; a concurrent append landed first.
        let base = make_v2_minimal_table(); // no isolation prop -> serializable
        let table_s1 = append_snapshot(&base, &["test/concurrent.parquet"]).await;

        let action = Transaction::new(&table_s1)
            .row_delta()
            .add_data_files(vec![make_data_file(&table_s1, "test/mine.parquet", 100)])
            .validate_from_empty_table();
        let err = match Arc::new(action).commit(&table_s1).await {
            Ok(_) => panic!("expected conflict"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(err.to_string().contains("serializable isolation violation"));
    }

    #[tokio::test]
    async fn test_validate_fast_path_base_equals_current() {
        let base = make_v2_minimal_table();
        let table_s1 = append_snapshot(&base, &["test/data.parquet"]).await;
        let s1 = table_s1.metadata().current_snapshot_id().unwrap();

        let action = Transaction::new(&table_s1)
            .row_delta()
            .add_data_files(vec![make_data_file(&table_s1, "test/mine.parquet", 100)])
            .validate_from_snapshot(s1);
        assert!(Arc::new(action).commit(&table_s1).await.is_ok());
    }

    #[tokio::test]
    async fn test_snapshot_isolation_allows_concurrent_data_append() {
        let base = table_with_properties(
            &make_v2_minimal_table(),
            std::collections::HashMap::from([(
                "write.merge.isolation-level".to_string(),
                "snapshot".to_string(),
            )]),
        );
        let table_s1 = append_snapshot(&base, &["test/data.parquet"]).await;
        let s1 = table_s1.metadata().current_snapshot_id().unwrap();
        // Concurrent append lands after our plan-time snapshot S1.
        let table_s2 = append_snapshot(&table_s1, &["test/concurrent.parquet"]).await;

        let action = Transaction::new(&table_s2)
            .row_delta()
            .add_data_files(vec![make_data_file(&table_s2, "test/mine.parquet", 100)])
            .validate_from_snapshot(s1);
        assert!(
            Arc::new(action).commit(&table_s2).await.is_ok(),
            "snapshot isolation must allow a rebase over a pure data append"
        );
    }

    #[tokio::test]
    async fn test_serializable_conflicts_on_concurrent_data_append() {
        // Same shape as above but WITHOUT the snapshot-isolation prop.
        let base = make_v2_minimal_table();
        let table_s1 = append_snapshot(&base, &["test/data.parquet"]).await;
        let s1 = table_s1.metadata().current_snapshot_id().unwrap();
        let table_s2 = append_snapshot(&table_s1, &["test/concurrent.parquet"]).await;

        let action = Transaction::new(&table_s2)
            .row_delta()
            .add_data_files(vec![make_data_file(&table_s2, "test/mine.parquet", 100)])
            .validate_from_snapshot(s1);
        let err = match Arc::new(action).commit(&table_s2).await {
            Ok(_) => panic!("expected conflict"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(err.to_string().contains("serializable isolation violation"));
    }

    #[tokio::test]
    async fn test_per_action_snapshot_isolation_override() {
        // Same shape as the serializable-conflict test — NO table property —
        // but the ACTION opts into snapshot isolation (Java parity: the
        // RowDelta caller chooses validation strictness). The concurrent
        // pure data append must then pass.
        let base = make_v2_minimal_table();
        let table_s1 = append_snapshot(&base, &["test/data.parquet"]).await;
        let s1 = table_s1.metadata().current_snapshot_id().unwrap();
        let table_s2 = append_snapshot(&table_s1, &["test/concurrent.parquet"]).await;

        let action = Transaction::new(&table_s2)
            .row_delta()
            .add_data_files(vec![make_data_file(&table_s2, "test/mine.parquet", 100)])
            .validate_from_snapshot(s1)
            .with_snapshot_isolation();
        assert!(
            Arc::new(action).commit(&table_s2).await.is_ok(),
            "the per-action override must allow a rebase over a pure data append"
        );
    }

    #[tokio::test]
    async fn test_snapshot_isolation_conflicts_on_concurrent_dv_same_file() {
        // The V3 multi-DV hazard: a concurrent commit added a DV for the SAME
        // data file our commit targets — must abort even at snapshot isolation.
        let base = table_with_properties(
            &make_v2_minimal_table(),
            std::collections::HashMap::from([(
                "write.merge.isolation-level".to_string(),
                "snapshot".to_string(),
            )]),
        );
        let table_s1 = append_snapshot(&base, &["test/data.parquet"]).await;
        let s1 = table_s1.metadata().current_snapshot_id().unwrap();
        let table_s2 = dv_snapshot(
            &table_s1,
            make_dv(&table_s1, "test/concurrent-dv.parquet", "test/data.parquet"),
        )
        .await;

        let action = Transaction::new(&table_s2)
            .row_delta()
            .add_delete_files(vec![make_dv(
                &table_s2,
                "test/my-dv.parquet",
                "test/data.parquet",
            )])
            .validate_from_snapshot(s1);
        let err = match Arc::new(action).commit(&table_s2).await {
            Ok(_) => panic!("expected conflict"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(err.to_string().contains("conflicting"), "got: {err}");
    }

    #[tokio::test]
    async fn test_snapshot_isolation_allows_concurrent_dv_other_file() {
        let base = table_with_properties(
            &make_v2_minimal_table(),
            std::collections::HashMap::from([(
                "write.merge.isolation-level".to_string(),
                "snapshot".to_string(),
            )]),
        );
        let table_s1 = append_snapshot(&base, &["test/data.parquet", "test/other.parquet"]).await;
        let s1 = table_s1.metadata().current_snapshot_id().unwrap();
        let table_s2 = dv_snapshot(
            &table_s1,
            make_dv(
                &table_s1,
                "test/concurrent-dv.parquet",
                "test/other.parquet",
            ),
        )
        .await;

        let action = Transaction::new(&table_s2)
            .row_delta()
            .add_delete_files(vec![make_dv(
                &table_s2,
                "test/my-dv.parquet",
                "test/data.parquet",
            )])
            .validate_from_snapshot(s1);
        assert!(
            Arc::new(action).commit(&table_s2).await.is_ok(),
            "snapshot isolation must allow DVs on disjoint data files"
        );
    }

    #[tokio::test]
    async fn test_row_delta_empty_action() {
        let table = make_v2_minimal_table();
        assert!(
            Arc::new(Transaction::new(&table).row_delta())
                .commit(&table)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_row_delta_incompatible_partition_value() {
        let table = make_v2_minimal_table();
        let bad_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/bad.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::string("wrong"))]))
            .build()
            .unwrap();
        let action = Transaction::new(&table)
            .row_delta()
            .add_data_files(vec![bad_file]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    /// MoR: adding a position-delete file via RowDelta commits a content=Deletes
    /// manifest and an `Operation::Delete` snapshot (replaces the old "errors" test
    /// now that `add_delete_files` is implemented).
    #[tokio::test]
    async fn test_row_delta_add_delete_files_mor() {
        let base = make_v2_minimal_table();

        // S1: append a data file.
        let data_file = make_data_file(&base, "test/data.parquet", 100);
        let mut c1 = Arc::new(
            Transaction::new(&base)
                .fast_append()
                .add_data_files(vec![data_file]),
        )
        .commit(&base)
        .await
        .unwrap();
        let snap_s1 = if let TableUpdate::AddSnapshot { snapshot } =
            c1.take_updates().into_iter().next().unwrap()
        {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };
        let table_s1 = table_with_snapshot(&base, snap_s1).await;

        // S2: add a MoR position-delete file referencing the data file.
        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/pos-delete.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table_s1.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .referenced_data_file(Some("test/data.parquet".to_string()))
            .build()
            .unwrap();
        let mut c2 = Arc::new(
            Transaction::new(&table_s1)
                .row_delta()
                .add_delete_files(vec![delete_file]),
        )
        .commit(&table_s1)
        .await
        .unwrap();
        let updates2 = c2.take_updates();
        let snap_s2 = if let TableUpdate::AddSnapshot { ref snapshot } = updates2[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };

        // Only delete files added (no data adds/removes) → Operation::Delete.
        assert_eq!(snap_s2.summary().operation, crate::spec::Operation::Delete);

        // A PositionDeletes entry must exist in the new snapshot's manifests.
        let manifest_list = table_s1
            .manifest_list_reader(&Arc::new(snap_s2.clone()))
            .load()
            .await
            .unwrap();
        let mut found_position_delete = false;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(table_s1.file_io())
                .await
                .unwrap();
            for entry in manifest.entries() {
                if entry.data_file().content_type() == DataContentType::PositionDeletes {
                    found_position_delete = true;
                }
            }
        }
        assert!(
            found_position_delete,
            "expected a PositionDeletes entry in the RowDelta snapshot's manifests"
        );
    }

    /// End-to-end CoW test: append two files, then remove one via RowDelta.
    ///
    /// Verifies:
    /// - The removed file appears as DELETED with correct sequence numbers.
    /// - The surviving file appears as EXISTING with correct sequence numbers.
    /// - The new file appears as ADDED.
    /// - The snapshot summary counts `deleted-data-files = 1`.
    #[tokio::test]
    async fn test_row_delta_cow_manifest_rewrite() {
        let base_table = make_v2_minimal_table();

        // --- S1: append file-A and file-B ---
        let file_a = make_data_file(&base_table, "test/a.parquet", 100);
        let file_b = make_data_file(&base_table, "test/b.parquet", 200);

        let action1 = Transaction::new(&base_table)
            .fast_append()
            .add_data_files(vec![file_a.clone(), file_b.clone()]);
        let mut commit1 = Arc::new(action1).commit(&base_table).await.unwrap();
        let updates1 = commit1.take_updates();

        let snapshot_s1 =
            if let TableUpdate::AddSnapshot { snapshot } = updates1.into_iter().next().unwrap() {
                snapshot
            } else {
                panic!("expected AddSnapshot");
            };

        let table_s1 = table_with_snapshot(&base_table, snapshot_s1).await;

        // --- S2: remove file-A (CoW), add file-C ---
        let file_c = make_data_file(&table_s1, "test/c.parquet", 300);
        let action2 = Transaction::new(&table_s1)
            .row_delta()
            .remove_data_files(vec![file_a.clone()])
            .add_data_files(vec![file_c.clone()]);
        let mut commit2 = Arc::new(action2).commit(&table_s1).await.unwrap();
        let updates2 = commit2.take_updates();

        let snapshot_s2 = if let TableUpdate::AddSnapshot { ref snapshot } = updates2[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };

        assert_eq!(
            snapshot_s2.summary().operation,
            crate::spec::Operation::Overwrite
        );

        // Verify snapshot summary metrics
        let props = &snapshot_s2.summary().additional_properties;
        assert_eq!(
            props.get("deleted-data-files").map(String::as_str),
            Some("1"),
            "summary should count 1 deleted file"
        );

        // Scan all manifest entries in S2
        let manifest_list = table_s1
            .manifest_list_reader(&Arc::new(snapshot_s2.clone()))
            .load()
            .await
            .unwrap();

        let mut found_deleted_a = false;
        let mut found_existing_b = false;
        let mut found_added_c = false;

        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(table_s1.file_io())
                .await
                .unwrap();
            for entry in manifest.entries() {
                match entry.data_file().file_path() {
                    "test/a.parquet" => {
                        assert_eq!(
                            entry.status(),
                            ManifestStatus::Deleted,
                            "file-A must be DELETED"
                        );
                        assert!(
                            entry.sequence_number().is_some(),
                            "DELETED entry must have sequence number"
                        );
                        assert!(
                            entry.file_sequence_number.is_some(),
                            "DELETED entry must have file sequence number"
                        );
                        found_deleted_a = true;
                    }
                    "test/b.parquet" => {
                        assert_eq!(
                            entry.status(),
                            ManifestStatus::Existing,
                            "file-B must be EXISTING"
                        );
                        assert!(
                            entry.sequence_number().is_some(),
                            "EXISTING entry must have sequence number"
                        );
                        found_existing_b = true;
                    }
                    "test/c.parquet" => {
                        found_added_c = true;
                    }
                    other => panic!("unexpected file in S2 manifests: {other}"),
                }
            }
        }

        assert!(found_deleted_a, "file-A should have a DELETED entry in S2");
        assert!(
            found_existing_b,
            "file-B should have an EXISTING entry in S2"
        );
        assert!(found_added_c, "file-C should have an ADDED entry in S2");
    }
}
