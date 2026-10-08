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
use std::sync::{Arc, Mutex, RwLock};

use tokio::sync::Notify;
use tokio::sync::futures::OwnedNotified;
use tokio::sync::oneshot::Receiver;

use super::caching_delete_file_loader::{EqDeleteKey, EqDeleteKeys, EqDeleteSet, datum_as_i64};
use crate::delete_vector::DeleteVector;
use crate::runtime::Runtime;
use crate::scan::{FileScanTask, FileScanTaskDeleteFile};
use crate::spec::{DataContentType, Struct};
use crate::{Error, ErrorKind, Result};

#[derive(Debug)]
enum EqDelState {
    Loading(Arc<Notify>),
    Loaded(Arc<EqDeleteSet>),
}

/// A group of equality delete sets sharing one field layout. Rows are
/// removed when their key is in ANY of the sets — identical semantics to the
/// union of the sets. When `indexed` is set the row is decided by the
/// filter's equality-delete INDEX instead (one probe per row: key -> the
/// highest delete sequence number, compared with the task's data sequence
/// number); `sets` stays the exact per-file form, used as the fallback and
/// by the referee. Probing N sets per row made a task bound to ~1,700
/// equality-delete files cost N hash probes per row.
#[derive(Debug, Clone)]
pub(crate) struct EqDeleteGroup {
    /// Ordered `(field_name, field_id)` — identical across `sets`.
    pub(crate) fields: Vec<(String, i32)>,
    /// The per-delete-file sets (shared cache `Arc`s).
    pub(crate) sets: Vec<Arc<EqDeleteSet>>,
    /// The indexed probe for this task, when every bound delete file and the
    /// task itself carry a data sequence number.
    pub(crate) indexed: Option<IndexedEqProbe>,
}

/// One probe per row against the equality-delete indexes a task's bound
/// delete files were folded into (its own partition's and the global one).
#[derive(Debug, Clone)]
pub(crate) struct IndexedEqProbe {
    pub(crate) indexes: Vec<Arc<EqDeleteIndex>>,
    /// The task's data sequence number: a key is deleted iff some folded
    /// delete file holding it has a strictly greater sequence number.
    pub(crate) data_seq: i64,
}

impl IndexedEqProbe {
    pub(crate) fn deleted(&self, key: &EqDeleteKey) -> bool {
        self.indexes
            .iter()
            .any(|ix| ix.max_seq(key).is_some_and(|seq| seq > self.data_seq))
    }
}

/// Where a loaded equality-delete file is folded: its data sequence number
/// and the partition it applies to (`None` = unpartitioned, applies to every
/// data file). Mirrors `DeleteFileIndex`'s binding: a partitioned delete is
/// scoped to data files of the same spec id and partition tuple.
#[derive(Debug, Clone)]
pub(crate) struct EqFoldTarget {
    pub(crate) sequence_number: i64,
    pub(crate) partition_spec_id: i32,
    pub(crate) partition: Option<Struct>,
}

/// Identity of one equality-delete index: the delete's partition scope plus
/// the field layout of its keys.
type EqIndexKey = (i32, Option<Struct>, Vec<(String, i32)>);

const EQ_INDEX_SHARDS: usize = 64;

#[derive(Debug, Default)]
struct EqIndexShard {
    single_int: HashMap<i64, i64>,
    generic: HashMap<EqDeleteKey, i64>,
    /// Highest sequence number among delete files holding a NULL key
    /// (single-column integer sets); kept on shard 0.
    null_max_seq: Option<i64>,
}

/// key -> the highest data sequence number of any equality-delete file
/// holding it, for one partition scope and field layout. Built once per
/// filter (per reader) by folding each delete file as it loads; sharded by
/// key hash so a fold of one file never blocks readers of the others for
/// its whole length.
#[derive(Debug)]
pub(crate) struct EqDeleteIndex {
    shards: Vec<RwLock<EqIndexShard>>,
}

impl EqDeleteIndex {
    fn new() -> Self {
        Self {
            shards: (0..EQ_INDEX_SHARDS)
                .map(|_| RwLock::new(EqIndexShard::default()))
                .collect(),
        }
    }

    fn shard_of_i64(k: i64) -> usize {
        let x = (k as u64) ^ ((k as u64) >> 32);
        (x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize % EQ_INDEX_SHARDS
    }

    fn shard_of_key(key: &EqDeleteKey) -> usize {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        (h.finish() % EQ_INDEX_SHARDS as u64) as usize
    }

    /// Fold one loaded delete file: every key takes the max of its recorded
    /// sequence number and `seq`.
    fn fold(&self, set: &EqDeleteSet, seq: i64) {
        match &set.keys {
            EqDeleteKeys::SingleInt {
                keys,
                contains_null,
            } => {
                let mut buckets: Vec<Vec<i64>> = vec![Vec::new(); EQ_INDEX_SHARDS];
                for k in keys {
                    buckets[Self::shard_of_i64(*k)].push(*k);
                }
                for (i, bucket) in buckets.into_iter().enumerate() {
                    if bucket.is_empty() {
                        continue;
                    }
                    let mut shard = self.shards[i].write().unwrap();
                    for k in bucket {
                        let e = shard.single_int.entry(k).or_insert(seq);
                        if *e < seq {
                            *e = seq;
                        }
                    }
                }
                if *contains_null {
                    let mut s0 = self.shards[0].write().unwrap();
                    s0.null_max_seq = Some(s0.null_max_seq.map_or(seq, |m| m.max(seq)));
                }
            }
            EqDeleteKeys::Generic(keys) => {
                let mut buckets: Vec<Vec<&EqDeleteKey>> = vec![Vec::new(); EQ_INDEX_SHARDS];
                for k in keys {
                    buckets[Self::shard_of_key(k)].push(k);
                }
                for (i, bucket) in buckets.into_iter().enumerate() {
                    if bucket.is_empty() {
                        continue;
                    }
                    let mut shard = self.shards[i].write().unwrap();
                    for k in bucket {
                        let e = shard.generic.entry(k.clone()).or_insert(seq);
                        if *e < seq {
                            *e = seq;
                        }
                    }
                }
            }
        }
    }

    /// The highest delete sequence number holding `key`, if any file does.
    pub(crate) fn max_seq(&self, key: &EqDeleteKey) -> Option<i64> {
        let mut best: Option<i64> = None;
        if key.0.len() <= 1 {
            match key.0.first() {
                Some(Some(datum)) => {
                    if let Ok(v) = datum_as_i64(datum) {
                        let shard = self.shards[Self::shard_of_i64(v)].read().unwrap();
                        best = shard.single_int.get(&v).copied();
                    }
                }
                Some(None) | None => {
                    best = self.shards[0].read().unwrap().null_max_seq;
                }
            }
        }
        let shard = self.shards[Self::shard_of_key(key)].read().unwrap();
        if let Some(s) = shard.generic.get(key) {
            best = Some(best.map_or(*s, |b| b.max(*s)));
        }
        best
    }
}

/// State tracking for positional delete files.
/// Unlike equality deletes, positional deletes must be fully loaded before
/// the ArrowReader proceeds because retrieval is synchronous and non-blocking.
#[derive(Debug)]
enum PosDelState {
    /// The file is currently being loaded by a task.
    /// The notifier allows other tasks to wait for completion.
    Loading(Arc<Notify>),
    /// The file has been fully loaded and merged into the delete vector map.
    Loaded,
}

/// Identity of one positional-delete LOAD UNIT: the file path plus, for V3
/// deletion vectors, the blob's content offset within its Puffin container.
///
/// A container may pack MULTIPLE DV blobs (Doris and iceberg-java's
/// DVFileWriter both do this): N delete-file entries share one `file_path`,
/// distinguished only by `content_offset`, each referencing a DIFFERENT data
/// file. Keying load state by path alone loads only the FIRST blob — the
/// rest report AlreadyLoaded and their deletes are silently unapplied
/// (deleted rows resurface). `None` = a whole-file positional delete
/// (Parquet pos-del stream), which keeps path-level identity.
type PosDelKey = (String, Option<u64>);

#[derive(Debug, Default)]
struct DeleteFileFilterState {
    delete_vectors: HashMap<String, Arc<Mutex<DeleteVector>>>,
    equality_deletes: HashMap<String, EqDelState>,
    positional_deletes: HashMap<PosDelKey, PosDelState>,
    /// The equality-delete indexes, one per partition scope + field layout.
    eq_indexes: HashMap<EqIndexKey, Arc<EqDeleteIndex>>,
}

#[derive(Clone, Debug)]
pub(crate) struct DeleteFilter {
    state: Arc<RwLock<DeleteFileFilterState>>,
    runtime: Runtime,
}

/// Action to take when trying to start loading a positional delete file
pub(crate) enum PosDelLoadAction {
    /// The file is not loaded, the caller should load it.
    Load,
    /// The file is already loaded, nothing to do.
    AlreadyLoaded,
    /// The file is currently being loaded by another task.
    /// The caller *must* await this future to ensure data availability before
    /// returning, as subsequent access (get_delete_vector) is synchronous. The
    /// future is created under the state lock so it cannot miss the loader's
    /// `notify_waiters()` (which stores no permit).
    WaitFor(OwnedNotified),
}

impl DeleteFilter {
    /// Create a new DeleteFilter with the given runtime.
    pub(crate) fn new(runtime: Runtime) -> Self {
        Self {
            state: Arc::new(RwLock::new(DeleteFileFilterState::default())),
            runtime,
        }
    }

    /// Retrieve a delete vector for the data file associated with a given file scan task
    pub(crate) fn get_delete_vector(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Option<Arc<Mutex<DeleteVector>>> {
        self.get_delete_vector_for_path(file_scan_task.data_file_path())
    }

    /// Retrieve a delete vector for a data file
    pub(crate) fn get_delete_vector_for_path(
        &self,
        data_file_path: &str,
    ) -> Option<Arc<Mutex<DeleteVector>>> {
        self.state
            .read()
            .ok()
            .and_then(|st| st.delete_vectors.get(data_file_path).cloned())
    }

    pub(crate) fn try_start_eq_del_load(&self, file_path: &str) -> Option<Arc<Notify>> {
        let mut state = self.state.write().unwrap();

        // Skip if already loaded/loading - another task owns it
        if state.equality_deletes.contains_key(file_path) {
            return None;
        }

        // Mark as loading to prevent duplicate work
        let notifier = Arc::new(Notify::new());
        state
            .equality_deletes
            .insert(file_path.to_string(), EqDelState::Loading(notifier.clone()));

        Some(notifier)
    }

    /// Attempts to mark a positional delete file as "loading".
    ///
    /// Returns an action dictating whether the caller should load the file,
    /// wait for another task to load it, or do nothing.
    pub(crate) fn try_start_pos_del_load(
        &self,
        file_path: &str,
        content_offset: Option<u64>,
    ) -> PosDelLoadAction {
        let mut state = self.state.write().unwrap();

        let key: PosDelKey = (file_path.to_string(), content_offset);
        if let Some(state) = state.positional_deletes.get(&key) {
            match state {
                PosDelState::Loaded => return PosDelLoadAction::AlreadyLoaded,
                PosDelState::Loading(notify) => {
                    return PosDelLoadAction::WaitFor(notify.clone().notified_owned());
                }
            }
        }

        let notifier = Arc::new(Notify::new());
        state
            .positional_deletes
            .insert(key, PosDelState::Loading(notifier));

        PosDelLoadAction::Load
    }

    /// Marks a positional delete load unit (file, or DV blob within a Puffin
    /// container) as successfully loaded and notifies any waiting tasks.
    pub(crate) fn finish_pos_del_load(&self, file_path: &str, content_offset: Option<u64>) {
        let notify = {
            let mut state = self.state.write().unwrap();
            if let Some(PosDelState::Loading(notify)) = state
                .positional_deletes
                .insert((file_path.to_string(), content_offset), PosDelState::Loaded)
            {
                Some(notify)
            } else {
                None
            }
        };

        if let Some(notify) = notify {
            notify.notify_waiters();
        }
    }

    /// Retrieve the equality delete set for a given eq delete file path.
    /// Waits asynchronously if the set is still being loaded.
    pub(crate) async fn get_equality_delete_set_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<Arc<EqDeleteSet>> {
        // The `Notified` is created UNDER the state lock: the loader's
        // `notify_waiters()` stores no permit, so a waiter that registered
        // after the signal would wait forever. Creating it while the entry is
        // observably `Loading` closes that window (the positional path has the
        // same guard in `try_start_pos_del_load`). This was the
        // equality-delete lost-wakeup hang: a scan task sharing a delete file
        // with another in-flight task parked forever once that task's load
        // finished between the state read and the await.
        let notified = {
            let state = self.state.read().unwrap();
            match state.equality_deletes.get(file_path) {
                None => return None,
                Some(EqDelState::Loaded(eq_delete_set)) => {
                    return Some(eq_delete_set.clone());
                }
                Some(EqDelState::Loading(notifier)) => notifier.clone().notified_owned(),
            }
        };

        notified.await;

        match self.state.read().unwrap().equality_deletes.get(file_path) {
            Some(EqDelState::Loaded(eq_delete_set)) => Some(eq_delete_set.clone()),
            // The loader failed and cleared its entry (`insert_equality_delete`):
            // report the set as missing so the caller fails the task loudly
            // instead of waiting on a load that will never finish.
            _ => None,
        }
    }

    /// Builds equality delete groups for the provided task.
    ///
    /// Returns one group per distinct `equality_ids` field layout, each
    /// holding the cached `Arc`s of every applicable delete file's set and,
    /// when the task and every bound delete carry a data sequence number,
    /// the indexed probe over the indexes those files were folded into.
    pub(crate) async fn build_equality_delete_groups(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Result<Vec<EqDeleteGroup>> {
        // (sets, index keys, unindexable) per field layout
        let mut groups: HashMap<Vec<(String, i32)>, (Vec<Arc<EqDeleteSet>>, Vec<EqIndexKey>, bool)> =
            HashMap::new();
        for delete in &file_scan_task.deletes {
            if !is_equality_delete(delete) {
                continue;
            }
            let Some(eq_set) = self
                .get_equality_delete_set_for_delete_file_path(&delete.file_path)
                .await
            else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Missing equality delete set for file '{}'",
                        delete.file_path
                    ),
                ));
            };
            if eq_set.is_empty() {
                continue;
            }
            let entry = groups.entry(eq_set.fields.clone()).or_default();
            match delete.sequence_number {
                Some(_) => {
                    let key: EqIndexKey = (
                        delete.partition_spec_id,
                        delete.partition.clone(),
                        eq_set.fields.clone(),
                    );
                    if !entry.1.contains(&key) {
                        entry.1.push(key);
                    }
                }
                // A delete without a sequence number is bound only when the
                // data file has none either (every delete applies then); the
                // index cannot express that — the per-set path can.
                None => entry.2 = true,
            }
            entry.0.push(eq_set);
        }
        let mut out = Vec::with_capacity(groups.len());
        for (fields, (sets, keys, unindexable)) in groups {
            let indexed = match file_scan_task.sequence_number {
                Some(data_seq) if !unindexable => {
                    let state = self.state.read().unwrap();
                    let indexes: Option<Vec<Arc<EqDeleteIndex>>> =
                        keys.iter().map(|k| state.eq_indexes.get(k).cloned()).collect();
                    indexes.map(|indexes| IndexedEqProbe { indexes, data_seq })
                }
                _ => None,
            };
            out.push(EqDeleteGroup {
                fields,
                sets,
                indexed,
            });
        }
        Ok(out)
    }

    pub(crate) fn upsert_delete_vector(
        &mut self,
        data_file_path: String,
        delete_vector: DeleteVector,
    ) {
        let mut state = self.state.write().unwrap();

        let Some(entry) = state.delete_vectors.get_mut(&data_file_path) else {
            state
                .delete_vectors
                .insert(data_file_path, Arc::new(Mutex::new(delete_vector)));
            return;
        };

        *entry.lock().unwrap() |= delete_vector;
    }

    pub(crate) fn insert_equality_delete(
        &self,
        delete_file_path: &str,
        eq_del: Receiver<Arc<EqDeleteSet>>,
        fold: Option<EqFoldTarget>,
    ) {
        let notify = Arc::new(Notify::new());
        {
            let mut state = self.state.write().unwrap();
            state.equality_deletes.insert(
                delete_file_path.to_string(),
                EqDelState::Loading(notify.clone()),
            );
        }

        let state = self.state.clone();
        let delete_file_path = delete_file_path.to_string();
        self.runtime.cpu().spawn(async move {
            // A dropped sender means the loader failed before producing the
            // set. The old `unwrap()` panicked here, which left the entry
            // `Loading` forever and parked every later waiter (the stranded
            // Loading state): clear the entry instead, so waiters observe
            // "missing" and a later task can retry the load.
            let loaded = eq_del.await.ok();
            // Fold BEFORE publishing Loaded: a task awaits its own bound files,
            // so once it sees Loaded every key of that file is in the index.
            // The fold takes the state lock only to find the index; the keys
            // go in under the index's own shard locks.
            if let (Some(set), Some(target)) = (&loaded, fold) {
                let index = {
                    let mut st = state.write().unwrap();
                    st.eq_indexes
                        .entry((
                            target.partition_spec_id,
                            target.partition,
                            set.fields.clone(),
                        ))
                        .or_insert_with(|| Arc::new(EqDeleteIndex::new()))
                        .clone()
                };
                index.fold(set, target.sequence_number);
            }
            {
                let mut state = state.write().unwrap();
                match loaded {
                    Some(eq_del) => {
                        state
                            .equality_deletes
                            .insert(delete_file_path, EqDelState::Loaded(eq_del));
                    }
                    None => {
                        state.equality_deletes.remove(&delete_file_path);
                    }
                }
            }
            notify.notify_waiters();
        });
    }
}

pub(crate) fn is_equality_delete(f: &FileScanTaskDeleteFile) -> bool {
    matches!(f.file_type, DataContentType::EqualityDeletes)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::Schema as ArrowSchema;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::caching_delete_file_loader::{
        CachingDeleteFileLoader, EqDeleteKey, EqDeleteKeys, EqDeleteSet,
    };
    use crate::io::FileIO;
    use crate::spec::{DataFileFormat, Datum, Literal, NestedField, PrimitiveType, Schema, Struct, Type};

    type ArrowSchemaRef = Arc<ArrowSchema>;

    const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
    const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

    // Regression test for the positional-delete lost-wakeup hang.
    //
    // Drives the real API through the losing interleaving: the loader fires
    // `notify_waiters()` (via `finish_pos_del_load`) *before* the waiter awaits the notifier
    // handed back by `WaitFor`. `notify_waiters()` stores no permit, so this only completes if
    // the waiter's `Notified` was created before the signal. Because `WaitFor` now carries an
    // `OwnedNotified` created under the lock in `try_start_pos_del_load`, it is; on the old
    // `WaitFor(Arc<Notify>)` contract the waiter created its `Notified` too late and hung.
    #[tokio::test]
    async fn test_wait_for_completes_when_load_finishes_before_await() {
        let filter = DeleteFilter::new(Runtime::current());
        let path = "s3://bucket/pos-delete.parquet";

        assert!(matches!(
            filter.try_start_pos_del_load(path, None),
            PosDelLoadAction::Load
        ));

        let PosDelLoadAction::WaitFor(notified) = filter.try_start_pos_del_load(path, None) else {
            panic!("expected WaitFor for an in-progress load");
        };

        // Loader completes and signals before the waiter awaits.
        filter.finish_pos_del_load(path, None);

        // Distinct content offsets in the SAME container are independent load
        // units: a second blob at another offset must be a fresh Load, not
        // AlreadyLoaded (the multi-blob puffin container class).
        assert!(matches!(
            filter.try_start_pos_del_load(path, Some(42)),
            PosDelLoadAction::Load
        ));

        let waited = tokio::time::timeout(std::time::Duration::from_secs(5), notified).await;
        assert!(
            waited.is_ok(),
            "WaitFor future must resolve after finish_pos_del_load"
        );
    }

    // Regression test for the EQUALITY-delete lost-wakeup hang: the loader
    // completes and fires `notify_waiters()` while a waiter is between its
    // state read and its await. The waiter's `Notified` is now created under
    // the state lock, so the signal cannot be missed; the old getter created
    // it after releasing the lock and could park forever.
    #[tokio::test]
    async fn test_eq_del_waiter_completes_when_load_finishes_concurrently() {
        let filter = DeleteFilter::new(Runtime::current());
        let path = "s3://bucket/eq-delete.parquet";
        assert!(filter.try_start_eq_del_load(path).is_some());
        let (sender, receiver) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete(path, receiver, None);
        let waiter = {
            let filter = filter.clone();
            tokio::spawn(async move {
                filter
                    .get_equality_delete_set_for_delete_file_path(path)
                    .await
            })
        };
        // Complete the load while the waiter is (or is about to be) parked.
        tokio::task::yield_now().await;
        sender
            .send(Arc::new(EqDeleteSet {
                keys: EqDeleteKeys::Generic(std::collections::HashSet::new()),
                fields: vec![("id".to_string(), 1)],
            }))
            .unwrap();
        let got = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("eq-delete waiter must wake once the load completes")
            .unwrap();
        assert!(got.is_some(), "the loaded set is handed to the waiter");
    }

    // Regression test for the stranded `Loading` state: the loader drops its
    // sender (its load failed). Waiters must observe "missing" instead of
    // parking forever, and the entry must be clear so a later task can retry.
    #[tokio::test]
    async fn test_eq_del_failed_load_releases_waiters_and_clears_entry() {
        let filter = DeleteFilter::new(Runtime::current());
        let path = "s3://bucket/eq-delete-failed.parquet";
        assert!(filter.try_start_eq_del_load(path).is_some());
        let (sender, receiver) = tokio::sync::oneshot::channel::<Arc<EqDeleteSet>>();
        filter.insert_equality_delete(path, receiver, None);
        let waiter = {
            let filter = filter.clone();
            tokio::spawn(async move {
                filter
                    .get_equality_delete_set_for_delete_file_path(path)
                    .await
            })
        };
        tokio::task::yield_now().await;
        drop(sender); // the loader failed
        let got = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("a failed load must release its waiters")
            .unwrap();
        assert!(got.is_none(), "a failed load reports the set as missing");
        assert!(
            filter.try_start_eq_del_load(path).is_some(),
            "the failed entry is cleared so the load can be retried"
        );
    }

    /// Load one synthetic equality-delete file into `filter` (keys on field
    /// `id`), folded at `seq` into the scope `(spec_id, partition)`.
    async fn load_eq_delete(
        filter: &DeleteFilter,
        path: &str,
        keys: &[i64],
        seq: i64,
        spec_id: i32,
        partition: Option<Struct>,
    ) -> FileScanTaskDeleteFile {
        assert!(filter.try_start_eq_del_load(path).is_some());
        let (sender, receiver) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete(
            path,
            receiver,
            Some(EqFoldTarget {
                sequence_number: seq,
                partition_spec_id: spec_id,
                partition: partition.clone(),
            }),
        );
        sender
            .send(Arc::new(EqDeleteSet {
                keys: EqDeleteKeys::SingleInt {
                    keys: keys.iter().copied().collect(),
                    contains_null: false,
                },
                fields: vec![("id".to_string(), 1)],
            }))
            .unwrap();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                filter.get_equality_delete_set_for_delete_file_path(path)
            )
            .await
            .unwrap()
            .is_some()
        );
        FileScanTaskDeleteFile {
                sequence_number: None,
                partition: None,
            file_path: path.to_string(),
            file_size_in_bytes: 1,
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: spec_id,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            key_metadata: None,
            sequence_number: Some(seq),
            partition,
        }
    }

    fn data_task(
        tmp: &Path,
        seq: Option<i64>,
        partition: Option<Struct>,
        deletes: Vec<FileScanTaskDeleteFile>,
    ) -> FileScanTask {
        let mut task = setup(tmp).into_iter().next().expect("a fixture task");
        task.sequence_number = seq;
        task.partition = partition;
        task.deletes = deletes;
        task
    }

    fn key(v: i64) -> EqDeleteKey {
        EqDeleteKey(vec![Some(Datum::long(v))])
    }

    // (i) Partition scoping on a spec partitioned by a boolean: one pk in
    // both partitions, a repair-style equality delete in the `true`
    // partition at a sequence number above both files'. The `true` row
    // dies; the `false` row survives (its task binds nothing — the index
    // never binds a partitioned delete across partitions — so no group is
    // built at all). The sequence rule: a file at or above the delete's
    // sequence number keeps its row.
    #[tokio::test]
    async fn test_indexed_probe_scopes_partitioned_deletes_and_applies_the_seq_rule() {
        let tmp = TempDir::new().unwrap();
        let filter = DeleteFilter::new(Runtime::current());
        let p_true = Some(Struct::from_iter([Some(Literal::bool(true))]));
        let p_false = Some(Struct::from_iter([Some(Literal::bool(false))]));
        let d = load_eq_delete(&filter, "s3://b/del-true.parquet", &[42], 10, 1, p_true.clone()).await;

        let task_true = data_task(tmp.path(), Some(5), p_true.clone(), vec![d.clone()]);
        let groups = filter.build_equality_delete_groups(&task_true).await.unwrap();
        assert_eq!(groups.len(), 1);
        let probe = groups[0].indexed.as_ref().expect("indexed probe");
        assert!(probe.deleted(&key(42)), "the true-partition row dies");
        assert!(!probe.deleted(&key(43)), "an unrelated key survives");

        let task_false = data_task(tmp.path(), Some(5), p_false, vec![]);
        assert!(
            filter.build_equality_delete_groups(&task_false).await.unwrap().is_empty(),
            "the false-partition file binds nothing and keeps its row"
        );

        let task_later = data_task(tmp.path(), Some(10), p_true, vec![d]);
        let groups = filter.build_equality_delete_groups(&task_later).await.unwrap();
        assert!(
            !groups[0].indexed.as_ref().unwrap().deleted(&key(42)),
            "a file at the delete's own sequence number keeps the row (strict >)"
        );
    }

    // (ii) Two specs: a spec-0 unpartitioned data file plus spec-1
    // partitioned files; one global delete and one partitioned delete. The
    // spec-0 file honours only the global delete even though the
    // partitioned delete's key sits in the filter's other index.
    #[tokio::test]
    async fn test_indexed_probe_consults_only_the_global_index_for_an_unpartitioned_file() {
        let tmp = TempDir::new().unwrap();
        let filter = DeleteFilter::new(Runtime::current());
        let p_true = Some(Struct::from_iter([Some(Literal::bool(true))]));
        let g = load_eq_delete(&filter, "s3://b/del-global.parquet", &[7], 10, 0, None).await;
        let p = load_eq_delete(&filter, "s3://b/del-part.parquet", &[8], 10, 1, p_true.clone()).await;

        let spec0 = data_task(tmp.path(), Some(5), None, vec![g.clone()]);
        let probe0 = filter.build_equality_delete_groups(&spec0).await.unwrap().remove(0).indexed.unwrap();
        assert!(probe0.deleted(&key(7)), "the global delete applies");
        assert!(!probe0.deleted(&key(8)), "the partitioned delete does not reach a spec-0 file");

        let spec1 = data_task(tmp.path(), Some(5), p_true, vec![g, p]);
        let probe1 = filter.build_equality_delete_groups(&spec1).await.unwrap().remove(0).indexed.unwrap();
        assert!(probe1.deleted(&key(7)) && probe1.deleted(&key(8)), "both apply in the partition");
    }

    // A task without a data sequence number falls back to the per-set path
    // (every bound delete applies, which the index cannot express).
    #[tokio::test]
    async fn test_task_without_sequence_number_keeps_per_set_probing() {
        let tmp = TempDir::new().unwrap();
        let filter = DeleteFilter::new(Runtime::current());
        let d = load_eq_delete(&filter, "s3://b/del-noseq.parquet", &[1], 10, 0, None).await;
        let task = data_task(tmp.path(), None, None, vec![d]);
        let groups = filter.build_equality_delete_groups(&task).await.unwrap();
        assert!(groups[0].indexed.is_none());
        assert_eq!(groups[0].sets.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_file_filter_load_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader =
            CachingDeleteFileLoader::new(file_io.clone(), 10, Runtime::current());

        let file_scan_tasks = setup(table_location);

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();
        assert_eq!(result.lock().unwrap().len(), 12); // pos dels from pos del file 1 and 2

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[1].deletes, file_scan_tasks[1].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[1])
            .unwrap();
        assert_eq!(result.lock().unwrap().len(), 8); // no pos dels for file 3
    }

    pub(crate) fn setup(table_location: &Path) -> Vec<FileScanTask> {
        let data_file_schema = Arc::new(Schema::builder().build().unwrap());
        let positional_delete_schema = create_pos_del_schema();

        let file_path_values = [
            vec![format!("{}/1.parquet", table_location.to_str().unwrap()); 8],
            vec![format!("{}/1.parquet", table_location.to_str().unwrap()); 8],
            vec![format!("{}/2.parquet", table_location.to_str().unwrap()); 8],
        ];
        let pos_values = [
            vec![0i64, 1, 3, 5, 6, 8, 1022, 1023],
            vec![0i64, 1, 3, 5, 20, 21, 22, 23],
            vec![0i64, 1, 3, 5, 6, 8, 1022, 1023],
        ];

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        for n in 1..=3 {
            let file_path_vals = file_path_values.get(n - 1).unwrap();
            let file_path_col = Arc::new(StringArray::from_iter_values(file_path_vals));

            let pos_vals = pos_values.get(n - 1).unwrap();
            let pos_col = Arc::new(Int64Array::from_iter_values(pos_vals.clone()));

            let positional_deletes_to_write =
                RecordBatch::try_new(positional_delete_schema.clone(), vec![
                    file_path_col.clone(),
                    pos_col.clone(),
                ])
                .unwrap();

            let file = File::create(format!(
                "{}/pos-del-{}.parquet",
                table_location.to_str().unwrap(),
                n
            ))
            .unwrap();
            let mut writer = ArrowWriter::try_new(
                file,
                positional_deletes_to_write.schema(),
                Some(props.clone()),
            )
            .unwrap();

            writer
                .write(&positional_deletes_to_write)
                .expect("Writing batch");

            // writer must be closed to write footer
            writer.close().unwrap();
        }

        let pos_del_1 = FileScanTaskDeleteFile::builder()
            .with_file_path(format!(
                "{}/pos-del-1.parquet",
                table_location.to_str().unwrap()
            ))
            .with_file_size_in_bytes(
                std::fs::metadata(format!(
                    "{}/pos-del-1.parquet",
                    table_location.to_str().unwrap()
                ))
                .unwrap()
                .len(),
            )
            .with_file_type(DataContentType::PositionDeletes)
            .with_partition_spec_id(0)
            .build();

        let pos_del_2 = FileScanTaskDeleteFile::builder()
            .with_file_path(format!(
                "{}/pos-del-2.parquet",
                table_location.to_str().unwrap()
            ))
            .with_file_size_in_bytes(
                std::fs::metadata(format!(
                    "{}/pos-del-2.parquet",
                    table_location.to_str().unwrap()
                ))
                .unwrap()
                .len(),
            )
            .with_file_type(DataContentType::PositionDeletes)
            .with_partition_spec_id(0)
            .build();

        let pos_del_3 = FileScanTaskDeleteFile::builder()
            .with_file_path(format!(
                "{}/pos-del-3.parquet",
                table_location.to_str().unwrap()
            ))
            .with_file_size_in_bytes(
                std::fs::metadata(format!(
                    "{}/pos-del-3.parquet",
                    table_location.to_str().unwrap()
                ))
                .unwrap()
                .len(),
            )
            .with_file_type(DataContentType::PositionDeletes)
            .with_partition_spec_id(0)
            .build();

        let file_scan_tasks = vec![
            FileScanTask::builder()
                .with_file_size_in_bytes(0)
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{}/1.parquet", table_location.to_str().unwrap()))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(data_file_schema.clone())
                .with_project_field_ids(vec![])
                .with_deletes(vec![pos_del_1, pos_del_2.clone()])
                .with_case_sensitive(false)
                .build(),
            FileScanTask::builder()
                .with_file_size_in_bytes(0)
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{}/2.parquet", table_location.to_str().unwrap()))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(data_file_schema.clone())
                .with_project_field_ids(vec![])
                .with_deletes(vec![pos_del_3])
                .with_case_sensitive(false)
                .build(),
        ];

        file_scan_tasks
    }

    pub(crate) fn create_pos_del_schema() -> ArrowSchemaRef {
        let fields = vec![
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false)
                .with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    FIELD_ID_POSITIONAL_DELETE_FILE_PATH.to_string(),
                )])),
            arrow_schema::Field::new("pos", arrow_schema::DataType::Int64, false).with_metadata(
                HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    FIELD_ID_POSITIONAL_DELETE_POS.to_string(),
                )]),
            ),
        ];
        Arc::new(arrow_schema::Schema::new(fields))
    }

    #[tokio::test]
    async fn test_build_equality_delete_set_unions_multiple_files() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let task = FileScanTask::builder()
            .with_file_size_in_bytes(0)
            .with_start(0)
            .with_length(0)
            .with_data_file_path("data.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(vec![1])
            .with_deletes(vec![
                FileScanTaskDeleteFile::builder()
                    .with_file_path("eq-del-1.parquet".to_string())
                    .with_file_size_in_bytes(1)
                    .with_file_type(DataContentType::EqualityDeletes)
                    .with_partition_spec_id(0)
                    .with_equality_ids(Some(vec![1]))
                    .build(),
                FileScanTaskDeleteFile::builder()
                    .with_file_path("eq-del-2.parquet".to_string())
                    .with_file_size_in_bytes(1)
                    .with_file_type(DataContentType::EqualityDeletes)
                    .with_partition_spec_id(0)
                    .with_equality_ids(Some(vec![1]))
                    .build(),
            ])
            .with_case_sensitive(true)
            .build();

        let filter = DeleteFilter::new(Runtime::current());

        // Insert two equality delete sets with different keys
        let mut set1 = EqDeleteSet {
            keys: EqDeleteKeys::Generic(std::collections::HashSet::new()),
            fields: vec![("id".to_string(), 1)],
        };
        set1.keys.insert_tuple(vec![Some(Datum::long(10))]).unwrap();
        set1.keys.insert_tuple(vec![Some(Datum::long(20))]).unwrap();

        let mut set2 = EqDeleteSet {
            keys: EqDeleteKeys::Generic(std::collections::HashSet::new()),
            fields: vec![("id".to_string(), 1)],
        };
        set2.keys.insert_tuple(vec![Some(Datum::long(30))]).unwrap();

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-1.parquet", rx1, None);
        tx1.send(Arc::new(set1)).unwrap();

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-2.parquet", rx2, None);
        tx2.send(Arc::new(set2)).unwrap();

        // Small delay to allow the spawned tasks to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let result = filter.build_equality_delete_groups(&task).await;
        assert!(result.is_ok());

        let eq_groups = result.unwrap();
        // Same equality_ids → ONE group holding both sets (never unioned).
        assert_eq!(eq_groups.len(), 1);
        let group = &eq_groups[0];
        assert_eq!(group.sets.len(), 2);
        assert_eq!(
            group.sets.iter().map(|s| s.keys.len()).sum::<usize>(),
            3,
            "no key is copied or lost across the group"
        );
        // Every key is reachable through the group (the union semantics).
        for key in [10, 20, 30] {
            assert!(
                group.sets.iter().any(|s| s
                    .keys
                    .contains_tuple(&EqDeleteKey(vec![Some(Datum::long(key))]))),
                "key {key} must be probeable through the group"
            );
        }
    }

    /// Delete files with different equality_ids must NOT be unioned — they
    /// produce separate sets, each applied independently.
    #[tokio::test]
    async fn test_build_equality_delete_sets_different_equality_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let task = FileScanTask::builder()
            .with_file_size_in_bytes(0)
            .with_start(0)
            .with_length(0)
            .with_data_file_path("data.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(vec![1, 2])
            .with_deletes(vec![
                FileScanTaskDeleteFile::builder()
                    .with_file_path("eq-del-by-id.parquet".to_string())
                    .with_file_size_in_bytes(1)
                    .with_file_type(DataContentType::EqualityDeletes)
                    .with_partition_spec_id(0)
                    .with_equality_ids(Some(vec![1]))
                    .build(),
                FileScanTaskDeleteFile::builder()
                    .with_file_path("eq-del-by-name.parquet".to_string())
                    .with_file_size_in_bytes(1)
                    .with_file_type(DataContentType::EqualityDeletes)
                    .with_partition_spec_id(0)
                    .with_equality_ids(Some(vec![2]))
                    .build(),
            ])
            .with_case_sensitive(true)
            .build();

        let filter = DeleteFilter::new(Runtime::current());

        // Delete file 1: delete by id
        let mut set_by_id = EqDeleteSet {
            keys: EqDeleteKeys::Generic(std::collections::HashSet::new()),
            fields: vec![("id".to_string(), 1)],
        };
        set_by_id
            .keys
            .insert_tuple(vec![Some(Datum::long(10))])
            .unwrap();

        // Delete file 2: delete by name
        let mut set_by_name = EqDeleteSet {
            keys: EqDeleteKeys::Generic(std::collections::HashSet::new()),
            fields: vec![("name".to_string(), 2)],
        };
        set_by_name
            .keys
            .insert_tuple(vec![Some(Datum::string("alice"))])
            .unwrap();

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-by-id.parquet", rx1, None);
        tx1.send(Arc::new(set_by_id)).unwrap();

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-by-name.parquet", rx2, None);
        tx2.send(Arc::new(set_by_name)).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let eq_groups = filter
            .build_equality_delete_groups(&task)
            .await
            .expect("should succeed");

        // Different equality_ids → two separate groups
        assert_eq!(
            eq_groups.len(),
            2,
            "Delete files with different equality_ids must produce separate groups"
        );

        // Each group holds exactly one set with exactly one key
        for group in &eq_groups {
            assert_eq!(group.sets.len(), 1);
            assert_eq!(group.sets[0].keys.len(), 1);
        }
    }
}
