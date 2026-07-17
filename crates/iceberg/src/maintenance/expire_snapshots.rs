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

//! Snapshot expiry WITH inline physical file cleanup.
//!
//! The metadata-only [`ExpireSnapshotsAction`](crate::transaction::Transaction::expire_snapshots)
//! removes snapshot entries but leaves the now-unreferenced files behind,
//! manufacturing orphans. [`ExpireSnapshotsWithCleanupAction`] layers Java's
//! `ExpireSnapshots`-with-cleanup semantics on top of it: commit the
//! `remove-snapshots` metadata change first, then delete exactly the files
//! that only the expired snapshots referenced.

use std::collections::HashSet;
use std::sync::Arc;

use futures::{StreamExt, stream};

use crate::spec::DataContentType;
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};
use crate::{Catalog, Result, TableUpdate};

/// Expires snapshots and deletes the files that only they referenced.
///
/// Two phases, crash-safe BY ORDERING (not a transaction):
///
/// 1. **Metadata first** — a `remove-snapshots` commit through the catalog
///    commit path via [`Transaction`], inheriting its rebase-per-attempt retry
///    on commit conflicts (`commit.retry.*` table properties): every attempt
///    refetches the table metadata and recomputes the snapshot selection from
///    scratch. Selection follows Java `RemoveSnapshots`: snapshots strictly
///    older than [`older_than_ms`](Self::older_than_ms) are removed while the
///    current snapshot, each branch's [`retain_last`](Self::retain_last) most
///    recent snapshots, and every snapshot referenced by a branch or tag are
///    always kept.
/// 2. **Files after** — only once the commit has succeeded, delete the files
///    reachable from the expired snapshots but NOT from any retained snapshot
///    or current metadata: manifest lists, manifests, data files, delete files
///    (incl. Puffin deletion vectors) and statistics files. Deletion is
///    best-effort with per-file failure reporting; a crash mid-cleanup leaves
///    plain orphans for [`RemoveOrphanFilesAction`](super::RemoveOrphanFilesAction)
///    to reclaim — never a referenced-but-deleted file.
///
/// A file referenced by BOTH an expired and a retained snapshot always
/// survives: candidates are an exclusive diff against everything reachable
/// from the post-commit metadata (mirroring Java's reachable-file cleanup),
/// and manifests shared with retained snapshots are not even descended into.
///
/// ```ignore
/// let result = ExpireSnapshotsWithCleanupAction::new(table)
///     .older_than_ms(cutoff_ms)
///     .retain_last(3)
///     .execute(&catalog)
///     .await?;
/// ```
pub struct ExpireSnapshotsWithCleanupAction {
    table: Table,
    older_than_ms: Option<i64>,
    retain_last: Option<usize>,
    cleanup: bool,
    dry_run: bool,
    delete_concurrency: usize,
}

/// Result of an [`ExpireSnapshotsWithCleanupAction`] execution.
///
/// The `candidate_*` vectors are the exclusive-file diff per category — files
/// reachable from the expired snapshots and from nothing retained. In dry-run
/// (or `cleanup(false)`) mode nothing is deleted and the candidates ARE the
/// report; otherwise `deleted_files` / `failed_deletes` record the outcome of
/// the best-effort delete phase.
#[derive(Debug, Default)]
pub struct ExpireSnapshotsResult {
    /// Snapshot ids removed from table metadata (would-be-removed in dry-run).
    pub removed_snapshot_ids: Vec<i64>,
    /// Branch/tag refs removed because they aged out (`max_ref_age_ms`).
    pub removed_ref_names: Vec<String>,
    /// Whether this was a dry run (no commit, no deletes).
    pub dry_run: bool,
    /// Manifest-list files only expired snapshots referenced.
    pub candidate_manifest_lists: Vec<String>,
    /// Manifest files only expired snapshots referenced.
    pub candidate_manifests: Vec<String>,
    /// Data files only expired snapshots referenced.
    pub candidate_data_files: Vec<String>,
    /// Delete files (position/equality deletes incl. Puffin deletion vectors)
    /// only expired snapshots referenced.
    pub candidate_delete_files: Vec<String>,
    /// Statistics + partition-statistics files of expired snapshots that no
    /// retained metadata entry references.
    pub candidate_stats_files: Vec<String>,
    /// Files actually deleted (empty in dry-run / no-cleanup mode).
    pub deleted_files: Vec<String>,
    /// Per-file delete failures as `(path, error)` — deletion is best-effort.
    pub failed_deletes: Vec<(String, String)>,
}

impl ExpireSnapshotsResult {
    /// Whether the action found nothing to expire (no snapshots, no refs).
    pub fn is_noop(&self) -> bool {
        self.removed_snapshot_ids.is_empty() && self.removed_ref_names.is_empty()
    }

    /// All candidate files across categories, leaf files first (data, deletes,
    /// stats, then manifests, then manifest lists).
    pub fn candidate_files(&self) -> impl Iterator<Item = &String> {
        self.candidate_data_files
            .iter()
            .chain(self.candidate_delete_files.iter())
            .chain(self.candidate_stats_files.iter())
            .chain(self.candidate_manifests.iter())
            .chain(self.candidate_manifest_lists.iter())
    }
}

impl ExpireSnapshotsWithCleanupAction {
    /// Creates the action for `table` with cleanup enabled, no dry-run, and
    /// the table's `history.expire.*` defaults for selection.
    pub fn new(table: Table) -> Self {
        Self {
            table,
            older_than_ms: None,
            retain_last: None,
            cleanup: true,
            dry_run: false,
            delete_concurrency: super::DEFAULT_DELETE_CONCURRENCY,
        }
    }

    /// Expire snapshots strictly older than this timestamp (epoch ms).
    /// Defaults to `now - history.expire.max-snapshot-age-ms`.
    pub fn older_than_ms(mut self, timestamp_ms: i64) -> Self {
        self.older_than_ms = Some(timestamp_ms);
        self
    }

    /// Always keep at least this many most-recent snapshots per branch
    /// (must be at least 1). Defaults to `history.expire.min-snapshots-to-keep`.
    pub fn retain_last(mut self, retain_last: usize) -> Self {
        self.retain_last = Some(retain_last);
        self
    }

    /// Whether to delete the exclusively-referenced files after the metadata
    /// commit (default `true`). With `false` this is a metadata-only expire;
    /// the candidates are still reported.
    pub fn cleanup(mut self, cleanup: bool) -> Self {
        self.cleanup = cleanup;
        self
    }

    /// Compute the selection + exclusive-file diff and return the report
    /// WITHOUT committing or deleting anything.
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Concurrency for the delete phase.
    pub fn delete_concurrency(mut self, concurrency: usize) -> Self {
        self.delete_concurrency = concurrency.max(1);
        self
    }

    /// Executes the expiry (and, unless dry-run / `cleanup(false)`, the file
    /// cleanup) against `catalog`.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<ExpireSnapshotsResult> {
        // Fresh base: the selection and the exclusive diff must start from
        // current metadata, not from a possibly-stale table handle.
        let base = catalog.load_table(self.table.identifier()).await?;

        // Plan preview (no catalog write): what WOULD the commit remove.
        // Surfaces selection errors (gc.enabled=false, retain_last=0) early
        // and lets a nothing-to-expire call return without committing.
        let (planned_ids, planned_refs) =
            Self::plan(&base, self.older_than_ms, self.retain_last).await?;
        if planned_ids.is_empty() && planned_refs.is_empty() {
            return Ok(ExpireSnapshotsResult {
                dry_run: self.dry_run,
                ..Default::default()
            });
        }

        if self.dry_run {
            let expired: HashSet<i64> = planned_ids.iter().copied().collect();
            let retained = super::reachable_files(&base, Some(&expired)).await?;
            let mut result = collect_candidates(&base, &expired, &retained).await?;
            result.removed_snapshot_ids = planned_ids;
            result.removed_ref_names = planned_refs;
            result.dry_run = true;
            return Ok(result);
        }

        // Phase 1 — metadata FIRST: the `remove-snapshots` commit through the
        // catalog commit path. `Transaction::commit` retries conflicts with
        // rebase-per-attempt: each attempt refetches metadata and re-plans the
        // selection from scratch. NEVER delete before this succeeds.
        let tx = Transaction::new(&base);
        let mut action = tx.expire_snapshots();
        if let Some(v) = self.older_than_ms {
            action = action.expire_older_than_ms(v);
        }
        if let Some(v) = self.retain_last {
            action = action.retain_last(v);
        }
        let tx = action.apply(tx)?;
        let committed = tx.commit(catalog).await?;

        // What ACTUALLY got removed (a conflict retry can rebase and reselect):
        // diff the base metadata against the post-commit metadata. A snapshot
        // committed concurrently after `base` and expired in the same retry is
        // not knowable from `base`; skipping it merely leaves plain orphans for
        // the orphan sweep — never an over-delete.
        let committed_ids: HashSet<i64> = committed
            .metadata()
            .snapshots()
            .map(|s| s.snapshot_id())
            .collect();
        let expired: HashSet<i64> = base
            .metadata()
            .snapshots()
            .map(|s| s.snapshot_id())
            .filter(|id| !committed_ids.contains(id))
            .collect();
        let mut removed_snapshot_ids: Vec<i64> = expired.iter().copied().collect();
        removed_snapshot_ids.sort_unstable();
        let mut removed_ref_names: Vec<String> = base
            .metadata()
            .refs
            .keys()
            .filter(|name| !committed.metadata().refs.contains_key(*name))
            .cloned()
            .collect();
        removed_ref_names.sort();

        // Exclusive diff: reachable from the expired snapshots (via the base
        // metadata — nothing has been deleted yet, so every file is readable)
        // MINUS everything reachable from the post-commit metadata (which is
        // authoritative: it includes any concurrently-committed snapshot).
        let retained = super::reachable_files(&committed, None).await?;
        let mut result = collect_candidates(&base, &expired, &retained).await?;
        result.removed_snapshot_ids = removed_snapshot_ids;
        result.removed_ref_names = removed_ref_names;

        if !self.cleanup {
            return Ok(result);
        }

        // Phase 2 — files AFTER the commit succeeded: bounded concurrency,
        // best-effort per file (a failed delete is an orphan, not an error).
        let file_io = base.file_io().clone();
        let candidates: Vec<String> = result.candidate_files().cloned().collect();
        let outcomes: Vec<(String, std::result::Result<(), String>)> = stream::iter(candidates)
            .map(|path| {
                let file_io = file_io.clone();
                async move {
                    let outcome = file_io.delete(&path).await.map_err(|e| e.to_string());
                    (path, outcome)
                }
            })
            .buffer_unordered(self.delete_concurrency)
            .collect()
            .await;
        for (path, outcome) in outcomes {
            match outcome {
                Ok(()) => result.deleted_files.push(path),
                Err(e) => result.failed_deletes.push((path, e)),
            }
        }

        Ok(result)
    }

    /// Resolves which snapshot ids + ref names the expire commit would remove,
    /// without touching the catalog: the transaction action's own planning
    /// applied to this table handle's metadata. (`Transaction::expire_snapshots`
    /// is the only constructor for the action, hence the throwaway transaction.)
    async fn plan(
        table: &Table,
        older_than_ms: Option<i64>,
        retain_last: Option<usize>,
    ) -> Result<(Vec<i64>, Vec<String>)> {
        let tx = Transaction::new(table);
        let mut action = tx.expire_snapshots();
        if let Some(v) = older_than_ms {
            action = action.expire_older_than_ms(v);
        }
        if let Some(v) = retain_last {
            action = action.retain_last(v);
        }
        let mut action_commit = Arc::new(action).commit(table).await?;
        let mut ids: Vec<i64> = vec![];
        let mut refs: Vec<String> = vec![];
        for update in action_commit.take_updates() {
            match update {
                TableUpdate::RemoveSnapshots { snapshot_ids } => ids.extend(snapshot_ids),
                TableUpdate::RemoveSnapshotRef { ref_name } => refs.push(ref_name),
                _ => {}
            }
        }
        ids.sort_unstable();
        refs.sort();
        Ok((ids, refs))
    }
}

/// Files reachable from the `expired` snapshots (as recorded in `view`'s
/// metadata) that are NOT in the `retained` reachable set, categorized.
///
/// Manifests shared with the retained side are not descended into: every file
/// they mention is reachable from a retained snapshot by definition. Files in
/// expired-exclusive manifests are still checked individually against the
/// retained set — a data file listed by both an expired-exclusive manifest and
/// a retained manifest (e.g. after a manifest rewrite) MUST survive.
async fn collect_candidates(
    view: &Table,
    expired: &HashSet<i64>,
    retained: &HashSet<String>,
) -> Result<ExpireSnapshotsResult> {
    let metadata = view.metadata();
    let file_io = view.file_io();
    let mut manifest_lists: HashSet<String> = HashSet::new();
    let mut manifests: HashSet<String> = HashSet::new();
    let mut data_files: HashSet<String> = HashSet::new();
    let mut delete_files: HashSet<String> = HashSet::new();
    let mut stats_files: HashSet<String> = HashSet::new();

    let mut manifests_to_load = Vec::new();
    for snapshot in metadata.snapshots() {
        if !expired.contains(&snapshot.snapshot_id()) {
            continue;
        }
        let manifest_list_path = snapshot.manifest_list();
        if manifest_list_path.is_empty() {
            continue;
        }
        if !retained.contains(manifest_list_path) {
            manifest_lists.insert(manifest_list_path.to_string());
        }
        let manifest_list = view.manifest_list_reader(snapshot).load().await?;
        for manifest_file in manifest_list.entries() {
            if retained.contains(&manifest_file.manifest_path) {
                continue;
            }
            // insert() returning true = first sighting -> load once.
            if manifests.insert(manifest_file.manifest_path.clone()) {
                manifests_to_load.push(manifest_file.clone());
            }
        }
    }
    for manifest_file in manifests_to_load {
        let manifest = manifest_file.load_manifest(file_io).await?;
        for entry in manifest.entries() {
            let path = entry.data_file().file_path();
            if retained.contains(path) {
                continue;
            }
            match entry.data_file().content_type() {
                DataContentType::Data => {
                    data_files.insert(path.to_string());
                }
                DataContentType::PositionDeletes | DataContentType::EqualityDeletes => {
                    delete_files.insert(path.to_string());
                }
            }
        }
    }

    for stats in metadata.statistics_iter() {
        if expired.contains(&stats.snapshot_id) && !retained.contains(&stats.statistics_path) {
            stats_files.insert(stats.statistics_path.clone());
        }
    }
    for stats in metadata.partition_statistics_iter() {
        if expired.contains(&stats.snapshot_id) && !retained.contains(&stats.statistics_path) {
            stats_files.insert(stats.statistics_path.clone());
        }
    }

    let sorted = |set: HashSet<String>| {
        let mut paths: Vec<String> = set.into_iter().collect();
        paths.sort();
        paths
    };
    Ok(ExpireSnapshotsResult {
        candidate_manifest_lists: sorted(manifest_lists),
        candidate_manifests: sorted(manifests),
        candidate_data_files: sorted(data_files),
        candidate_delete_files: sorted(delete_files),
        candidate_stats_files: sorted(stats_files),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::ExpireSnapshotsWithCleanupAction;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
        SnapshotReference, SnapshotRetention, Struct, Type,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, NamespaceIdent, TableCommit, TableCreation, TableUpdate};

    fn future_cutoff_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            + 60_000
    }

    async fn create_table(catalog: &impl Catalog) -> Table {
        let ns = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("t".to_string())
                    .schema(schema)
                    .build(),
            )
            .await
            .unwrap()
    }

    /// Metadata-only fast append (the data file bytes never exist; manifests
    /// and manifest lists DO get written through the catalog's FileIO, which
    /// is all the diff computation reads).
    async fn append_synthetic_file(catalog: &impl Catalog, table: &Table, name: &str) -> Table {
        let file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!(
                "{}/data/{name}.parquet",
                table.metadata().location()
            ))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(4)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        let tx = Transaction::new(table);
        tx.fast_append()
            .add_data_files(vec![file])
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    async fn seed_table(catalog: &impl Catalog, appends: usize) -> Table {
        let mut table = create_table(catalog).await;
        for i in 0..appends {
            table = append_synthetic_file(catalog, &table, &format!("f{i}")).await;
        }
        table
    }

    #[tokio::test]
    async fn test_noop_when_nothing_to_expire() {
        let catalog = new_memory_catalog().await;
        let table = seed_table(&catalog, 1).await;

        // One snapshot: the current snapshot is always retained.
        let result = ExpireSnapshotsWithCleanupAction::new(table)
            .older_than_ms(future_cutoff_ms())
            .retain_last(1)
            .execute(&catalog)
            .await
            .unwrap();
        assert!(result.is_noop());
        assert!(result.deleted_files.is_empty());
        assert!(result.candidate_files().next().is_none());
    }

    #[tokio::test]
    async fn test_dry_run_reports_without_committing_or_deleting() {
        let catalog = new_memory_catalog().await;
        let table = seed_table(&catalog, 3).await;
        let snapshot_count = table.metadata().snapshots().len();
        let manifest_lists: Vec<String> = table
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
        assert_eq!(result.removed_snapshot_ids.len(), 2, "2 of 3 would expire");
        // Fast appends share manifests + data files with the retained head:
        // only the two old manifest lists are exclusive.
        assert_eq!(result.candidate_manifest_lists.len(), 2);
        assert!(result.candidate_manifests.is_empty());
        assert!(result.candidate_data_files.is_empty());
        assert!(result.deleted_files.is_empty(), "dry run must not delete");

        // Nothing committed: the catalog still has every snapshot and every
        // manifest list is still on storage.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(reloaded.metadata().snapshots().len(), snapshot_count);
        for manifest_list in &manifest_lists {
            assert!(table.file_io().exists(manifest_list).await.unwrap());
        }
    }

    #[tokio::test]
    async fn test_expire_deletes_exclusive_manifest_lists_and_keeps_shared_files() {
        let catalog = new_memory_catalog().await;
        let table = seed_table(&catalog, 3).await;
        let mut snapshots: Vec<(i64, String)> = table
            .metadata()
            .snapshots()
            .map(|s| (s.snapshot_id(), s.manifest_list().to_string()))
            .collect();
        // Oldest first (timestamps are monotonic within this test).
        snapshots
            .sort_by_key(|(id, _)| table.metadata().snapshot_by_id(*id).unwrap().timestamp_ms());
        let current_id = table.metadata().current_snapshot_id().unwrap();

        let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
            .older_than_ms(future_cutoff_ms())
            .retain_last(1)
            .execute(&catalog)
            .await
            .unwrap();

        let expected_removed: Vec<i64> = {
            let mut ids: Vec<i64> = snapshots
                .iter()
                .map(|(id, _)| *id)
                .filter(|id| *id != current_id)
                .collect();
            ids.sort_unstable();
            ids
        };
        assert_eq!(result.removed_snapshot_ids, expected_removed);
        assert!(result.removed_ref_names.is_empty());

        // Exclusive diff: fast appends share manifests + data files with the
        // retained head, so ONLY the two old manifest lists are deleted.
        assert_eq!(result.candidate_manifest_lists.len(), 2);
        assert!(result.candidate_manifests.is_empty());
        assert!(result.candidate_data_files.is_empty());
        assert!(result.candidate_delete_files.is_empty());
        assert_eq!(result.deleted_files.len(), 2);
        assert!(result.failed_deletes.is_empty());

        // Physically: expired manifest lists gone, retained one intact.
        for (id, manifest_list) in &snapshots {
            let should_exist = *id == current_id;
            assert_eq!(
                table.file_io().exists(manifest_list).await.unwrap(),
                should_exist,
                "manifest list of snapshot {id}"
            );
        }

        // Metadata: only the current snapshot remains.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let remaining: Vec<i64> = reloaded
            .metadata()
            .snapshots()
            .map(|s| s.snapshot_id())
            .collect();
        assert_eq!(remaining, vec![current_id]);

        // Idempotency: a second call is a clean no-op.
        let again = ExpireSnapshotsWithCleanupAction::new(table)
            .older_than_ms(future_cutoff_ms())
            .retain_last(1)
            .execute(&catalog)
            .await
            .unwrap();
        assert!(again.is_noop());
        assert!(again.deleted_files.is_empty());
    }

    #[tokio::test]
    async fn test_tag_protects_snapshot_and_its_files() {
        let catalog = new_memory_catalog().await;
        let table = seed_table(&catalog, 3).await;
        let mut ids: Vec<i64> = table
            .metadata()
            .snapshots()
            .map(|s| s.snapshot_id())
            .collect();
        ids.sort_by_key(|id| table.metadata().snapshot_by_id(*id).unwrap().timestamp_ms());
        let (oldest, middle) = (ids[0], ids[1]);
        let oldest_manifest_list = table
            .metadata()
            .snapshot_by_id(oldest)
            .unwrap()
            .manifest_list()
            .to_string();

        // Tag the oldest snapshot through the catalog commit path.
        let commit = TableCommit::builder()
            .ident(table.identifier().clone())
            .updates(vec![TableUpdate::SetSnapshotRef {
                ref_name: "keep".to_string(),
                reference: SnapshotReference {
                    snapshot_id: oldest,
                    retention: SnapshotRetention::Tag {
                        max_ref_age_ms: None,
                    },
                },
            }])
            .requirements(vec![])
            .build();
        let table = catalog.update_table(commit).await.unwrap();

        let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
            .older_than_ms(future_cutoff_ms())
            .retain_last(1)
            .execute(&catalog)
            .await
            .unwrap();

        // Only the untagged, non-current middle snapshot expires.
        assert_eq!(result.removed_snapshot_ids, vec![middle]);
        assert!(result.removed_ref_names.is_empty());

        // The tagged snapshot and its files survive.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(reloaded.metadata().snapshot_by_id(oldest).is_some());
        assert!(reloaded.metadata().refs.contains_key("keep"));
        assert!(table.file_io().exists(&oldest_manifest_list).await.unwrap());
        assert!(
            !result
                .candidate_manifest_lists
                .contains(&oldest_manifest_list)
        );
    }

    #[tokio::test]
    async fn test_cleanup_false_commits_but_keeps_files() {
        let catalog = new_memory_catalog().await;
        let table = seed_table(&catalog, 2).await;

        let result = ExpireSnapshotsWithCleanupAction::new(table.clone())
            .older_than_ms(future_cutoff_ms())
            .retain_last(1)
            .cleanup(false)
            .execute(&catalog)
            .await
            .unwrap();

        assert_eq!(result.removed_snapshot_ids.len(), 1);
        assert_eq!(result.candidate_manifest_lists.len(), 1);
        assert!(
            result.deleted_files.is_empty(),
            "cleanup(false) never deletes"
        );

        // Metadata committed...
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(reloaded.metadata().snapshots().len(), 1);
        // ...but the exclusive manifest list is still on storage (an orphan now).
        assert!(
            table
                .file_io()
                .exists(&result.candidate_manifest_lists[0])
                .await
                .unwrap()
        );
    }

    /// A brand-new table with no snapshots at all: nothing to plan, no error.
    #[tokio::test]
    async fn test_empty_table_is_noop() {
        let catalog = new_memory_catalog().await;
        let table = create_table(&catalog).await;

        let result = ExpireSnapshotsWithCleanupAction::new(table)
            .older_than_ms(future_cutoff_ms())
            .retain_last(1)
            .execute(&catalog)
            .await
            .unwrap();
        assert!(result.is_noop());
    }
}
