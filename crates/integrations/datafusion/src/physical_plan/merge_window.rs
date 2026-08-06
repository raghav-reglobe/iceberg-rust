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

//! Windowed MoR MERGE execution — N slice statements over ONE session with
//! a HELD target current-set.
//!
//! The per-slice call shape re-runs the SCD2 demote-union TARGET read on
//! every slice (the count=0 W1 datum: ~185s of pure target scan+join per
//! slice on giants). This module holds that read across the window:
//!
//! - Slice 1 executes normally; the target scan TEES its output (narrow
//!   projection + `_file`/`_pos`, residual-filtered) into a
//!   [`HeldCurrentSet`] installed on the shared [`MorWindowState`].
//! - Between slices the held set is MAINTAINED, never re-scanned:
//!   `held' = held − victims(slice) + current-rows(slice's appended files)`.
//!   A frozen snapshot would dup-current any PK touched by two slices — the
//!   maintenance step is what makes the held scan correct.
//! - Slices 2..N serve the target probe from the held batches, and the
//!   USING-side self-read (the demote-union `t2`) from the same set via the
//!   table provider, when its projection is covered and its pushed filters
//!   imply current-only rows.
//! - Every consumer VALIDATES the held set against the freshly-loaded head
//!   snapshot (equality only — snapshot ids are random longs) and falls
//!   back to a direct scan on any mismatch: an out-of-band commit discards
//!   the hold instead of serving stale rows.
//!
//! Commit cadence: one RowDelta per slice by default (the checkpoint
//! contract — completed slices stay committed when a later slice poisons).
//! `commit_every > 1` defers lanes and folds N slices into one RowDelta,
//! flushing early whenever a slice's deletion vectors touch a file already
//! carried by the pending fold (the same-PK / same-file signature — the
//! deferred view cannot see pending state, so the slice re-executes against
//! the flushed head). Fold mode disables the held scan: it targets
//! commit-dominated small-slice tables where the target scan is cheap.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::compute::filter_record_batch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use iceberg::Catalog;
use iceberg::expr::Predicate as IcebergPredicate;
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::DataFile;
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};

use super::merge_mor::plain_cast_batch;
use crate::{IcebergCatalogProvider, to_datafusion_error};

/// Default byte cap for the held current-set (overridable per call).
pub const HELD_MAX_BYTES_DEFAULT: usize = 2 * 1024 * 1024 * 1024;

/// The held target current-set: the narrow (+ extras) projection of every
/// current row, with `_file`/`_pos`, exactly as the target scan emits it
/// (plain-cast, residual-filtered). Maintained across slices.
pub struct HeldCurrentSet {
    /// Fully-qualified target table identity (`ns.table` form of
    /// [`iceberg::TableIdent`]).
    pub target_ident: String,
    /// The teeing scan's output schema (narrow + extras + `_file` + `_pos`).
    pub schema: ArrowSchemaRef,
    pub batches: Vec<RecordBatch>,
    pub bytes: usize,
    /// Head snapshot the held rows represent. Consumers serve ONLY on
    /// equality with their freshly-pinned snapshot.
    pub valid_for_snapshot: Option<i64>,
    /// The target scan's prune predicate + row-exact residual — re-applied
    /// to the appended-file delta scans so maintenance filters exactly the
    /// way the tee capture did.
    pub predicate: Option<IcebergPredicate>,
    pub residual: Option<Arc<dyn PhysicalExpr>>,
    /// Pool accounting for the held bytes (resized on maintenance).
    reservation: Option<MemoryReservation>,
}

impl std::fmt::Debug for HeldCurrentSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldCurrentSet")
            .field("target_ident", &self.target_ident)
            .field("bytes", &self.bytes)
            .field("batches", &self.batches.len())
            .field("valid_for_snapshot", &self.valid_for_snapshot)
            .finish()
    }
}

/// A committed slice's facts the maintenance step needs.
#[derive(Debug, Clone)]
pub struct CommittedSlice {
    /// Post-commit table handle (its current snapshot holds the appends).
    pub table: Table,
    pub snapshot_id: Option<i64>,
    pub added_paths: Vec<String>,
}

/// One deferred slice's lanes (commit_every > 1).
pub struct SliceLanes {
    pub table: Table,
    pub catalog: Arc<dyn Catalog>,
    /// The snapshot this slice's plan was pinned to.
    pub base_snapshot: Option<i64>,
    pub added: Vec<DataFile>,
    pub dvs: Vec<DataFile>,
    pub removed: Vec<DataFile>,
}

/// Accumulated deferred lanes awaiting one folded RowDelta.
pub struct PendingCommit {
    table: Table,
    catalog: Arc<dyn Catalog>,
    base_snapshot: Option<i64>,
    added: Vec<DataFile>,
    dvs: Vec<DataFile>,
    removed: Vec<DataFile>,
    added_paths: HashSet<String>,
    dv_referenced: HashSet<String>,
    slices_folded: usize,
}

/// Shared, interior-mutable window state — rides on
/// [`super::merge_mor::MorMergeOptions::window`] so the exec nodes, the
/// table provider and the driver loop all see one instance.
pub struct MorWindowState {
    hold_requested: AtomicBool,
    hold_overflow: AtomicBool,
    held_extra_columns: Vec<String>,
    held_max_bytes: usize,
    commit_every: usize,
    slice_deadline: Mutex<Option<Instant>>,
    held: Mutex<Option<HeldCurrentSet>>,
    /// Out-channel from the write node: last slice's newly-matched victims
    /// (file → sorted positions), i.e. demoted + deleted rows.
    last_victims: Mutex<HashMap<String, Vec<u64>>>,
    /// Out-channel from the commit node (commit mode).
    last_commit: Mutex<Option<CommittedSlice>>,
    /// Out-channel from the commit node (defer mode).
    last_lanes: Mutex<Option<SliceLanes>>,
    pending: Mutex<Option<PendingCommit>>,
    /// Result telemetry.
    pub target_scans_direct: AtomicU64,
    pub target_scans_served: AtomicU64,
    pub provider_serves: AtomicU64,
}

impl std::fmt::Debug for MorWindowState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorWindowState")
            .field("hold_requested", &self.hold_requested)
            .field("commit_every", &self.commit_every)
            .finish()
    }
}

impl MorWindowState {
    pub fn new(
        hold_requested: bool,
        held_extra_columns: Vec<String>,
        held_max_bytes: usize,
        commit_every: usize,
    ) -> Self {
        let commit_every = commit_every.max(1);
        Self {
            // Fold mode disables the held scan (deferred appends are not
            // scannable via the table until committed).
            hold_requested: AtomicBool::new(hold_requested && commit_every == 1),
            hold_overflow: AtomicBool::new(false),
            held_extra_columns,
            held_max_bytes,
            commit_every,
            slice_deadline: Mutex::new(None),
            held: Mutex::new(None),
            last_victims: Mutex::new(HashMap::new()),
            last_commit: Mutex::new(None),
            last_lanes: Mutex::new(None),
            pending: Mutex::new(None),
            target_scans_direct: AtomicU64::new(0),
            target_scans_served: AtomicU64::new(0),
            provider_serves: AtomicU64::new(0),
        }
    }

    pub fn commit_every(&self) -> usize {
        self.commit_every
    }

    pub fn defer_commits(&self) -> bool {
        self.commit_every > 1
    }

    pub fn hold_requested(&self) -> bool {
        self.hold_requested.load(Ordering::Acquire) && !self.hold_overflow.load(Ordering::Acquire)
    }

    pub fn mark_hold_overflow(&self) {
        self.hold_overflow.store(true, Ordering::Release);
        self.held.lock().expect("held lock").take();
    }

    pub fn held_extra_columns(&self) -> &[String] {
        &self.held_extra_columns
    }

    pub fn held_max_bytes(&self) -> usize {
        self.held_max_bytes
    }

    pub fn set_slice_deadline(&self, deadline: Option<Instant>) {
        *self.slice_deadline.lock().expect("deadline lock") = deadline;
    }

    pub fn slice_deadline(&self) -> Option<Instant> {
        *self.slice_deadline.lock().expect("deadline lock")
    }

    /// Install the teed capture (once per window, or again after an
    /// out-of-band-commit discard).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn install_held(
        &self,
        target_ident: String,
        schema: ArrowSchemaRef,
        batches: Vec<RecordBatch>,
        bytes: usize,
        valid_for_snapshot: Option<i64>,
        predicate: Option<IcebergPredicate>,
        residual: Option<Arc<dyn PhysicalExpr>>,
        reservation: Option<MemoryReservation>,
    ) {
        let mut held = self.held.lock().expect("held lock");
        // One held set per window: a second table's tee never displaces the
        // first (slices of a window target one table by contract).
        if held.is_none() {
            *held = Some(HeldCurrentSet {
                target_ident,
                schema,
                batches,
                bytes,
                valid_for_snapshot,
                predicate,
                residual,
                reservation,
            });
        }
    }

    /// Serve check: batches (zero-copy clones) iff the held set matches the
    /// consumer's table identity, pinned snapshot, and (for the target
    /// scan) exact schema. A snapshot mismatch means an out-of-band commit
    /// happened — the hold is DISCARDED so the next tee reinstalls.
    pub(crate) fn held_batches_for_target_scan(
        &self,
        target_ident: &str,
        snapshot_id: Option<i64>,
        schema: &ArrowSchemaRef,
    ) -> Option<Vec<RecordBatch>> {
        let mut guard = self.held.lock().expect("held lock");
        let held = guard.as_ref()?;
        if held.target_ident != target_ident {
            return None;
        }
        if held.valid_for_snapshot != snapshot_id {
            *guard = None;
            return None;
        }
        if &held.schema != schema {
            return None;
        }
        Some(held.batches.clone())
    }

    /// Provider-side serve check (the demote-union `t2` self-read):
    /// projection must be covered by the held columns; snapshot must match
    /// the provider's freshly-loaded head. Returns the projected schema +
    /// batches (the schema also covers the zero-batch empty-table case).
    pub(crate) fn held_batches_for_provider(
        &self,
        target_ident: &str,
        head_snapshot: Option<i64>,
        projected_names: &[String],
    ) -> Option<(ArrowSchemaRef, Vec<RecordBatch>)> {
        let mut guard = self.held.lock().expect("held lock");
        let held = guard.as_ref()?;
        if held.target_ident != target_ident {
            return None;
        }
        if held.valid_for_snapshot != head_snapshot {
            *guard = None;
            return None;
        }
        let mut indices = Vec::with_capacity(projected_names.len());
        for name in projected_names {
            indices.push(held.schema.index_of(name).ok()?);
        }
        let projected_schema: ArrowSchemaRef = Arc::new(
            held.schema
                .project(&indices)
                .expect("indices verified above"),
        );
        let out = held
            .batches
            .iter()
            .map(|b| b.project(&indices).expect("indices verified above"))
            .collect();
        Some((projected_schema, out))
    }

    pub(crate) fn record_victims(&self, victims: HashMap<String, Vec<u64>>) {
        *self.last_victims.lock().expect("victims lock") = victims;
    }

    pub(crate) fn record_commit(&self, committed: CommittedSlice) {
        *self.last_commit.lock().expect("commit lock") = Some(committed);
    }

    pub(crate) fn record_lanes(&self, lanes: SliceLanes) {
        *self.last_lanes.lock().expect("lanes lock") = Some(lanes);
    }

    fn take_victims(&self) -> HashMap<String, Vec<u64>> {
        std::mem::take(&mut *self.last_victims.lock().expect("victims lock"))
    }

    fn take_commit(&self) -> Option<CommittedSlice> {
        self.last_commit.lock().expect("commit lock").take()
    }

    fn take_lanes(&self) -> Option<SliceLanes> {
        self.last_lanes.lock().expect("lanes lock").take()
    }

    pub fn held_stats(&self) -> (bool, usize, usize) {
        let guard = self.held.lock().expect("held lock");
        match guard.as_ref() {
            Some(h) => (true, h.batches.iter().map(|b| b.num_rows()).sum(), h.bytes),
            None => (false, 0, 0),
        }
    }

    /// Post-commit maintenance: `held' = held − victims + appended-currents`.
    /// Infallible by contract — ANY failure drops the hold (subsequent
    /// slices direct-scan, correct via fresh loads) and reports the reason;
    /// it must never fail the slice, whose commit already landed.
    pub(crate) async fn apply_slice_delta(&self) -> Option<String> {
        let victims = self.take_victims();
        let commit = self.take_commit();
        let Some(commit) = commit else {
            // Empty slice: nothing committed, head unchanged, held stays
            // valid as-is.
            return None;
        };
        // Take the held set OUT of the mutex — the delta scan awaits.
        let Some(mut held) = self.held.lock().expect("held lock").take() else {
            return None;
        };

        if !victims.is_empty() {
            match remove_victims(&held.batches, &victims) {
                Ok((batches, bytes)) => {
                    held.batches = batches;
                    held.bytes = bytes;
                }
                Err(e) => return Some(format!("held victim removal failed: {e}")),
            }
        }

        if !commit.added_paths.is_empty() {
            match scan_added_currents(&held, &commit).await {
                Ok(new_batches) => {
                    for b in new_batches {
                        held.bytes += b.get_array_memory_size();
                        held.batches.push(b);
                    }
                }
                Err(e) => return Some(format!("held delta scan failed: {e}")),
            }
        }

        if held.bytes > self.held_max_bytes {
            self.mark_hold_overflow();
            return Some(format!(
                "held set exceeded cap after maintenance ({} bytes)",
                held.bytes
            ));
        }
        if let Some(r) = held.reservation.as_mut()
            && r.try_resize(held.bytes).is_err()
        {
            self.mark_hold_overflow();
            return Some("held set reservation exceeded the memory pool".to_string());
        }
        held.valid_for_snapshot = commit.snapshot_id;
        *self.held.lock().expect("held lock") = Some(held);
        None
    }

    /// Defer-mode integration of the last slice's lanes. Returns
    /// `NeedsFlushAndRetry` when the slice's DVs touch files already carried
    /// by the pending fold — the slice was planned blind to pending state,
    /// so its lanes are WRONG for those files and it must re-execute after
    /// the flush.
    pub(crate) fn integrate_deferred(&self) -> DeferOutcome {
        let Some(lanes) = self.take_lanes() else {
            // Empty slice — nothing written, nothing to fold.
            return DeferOutcome::Deferred { folded: false };
        };
        let new_dv_refs: HashSet<String> = lanes
            .dvs
            .iter()
            .filter_map(|d| d.referenced_data_file())
            .collect();
        // A DV without a recorded referenced file cannot be conflict-checked
        // — treat as conflicting (conservative; our writer always records
        // field 143, so this is a foreign-lane guard).
        let unattributed_dv = lanes.dvs.iter().any(|d| d.referenced_data_file().is_none());

        let mut pending_guard = self.pending.lock().expect("pending lock");
        match pending_guard.as_mut() {
            None => {
                let added_paths = lanes
                    .added
                    .iter()
                    .map(|f| f.file_path().to_string())
                    .collect();
                *pending_guard = Some(PendingCommit {
                    table: lanes.table,
                    catalog: lanes.catalog,
                    base_snapshot: lanes.base_snapshot,
                    added: lanes.added,
                    dvs: lanes.dvs,
                    removed: lanes.removed,
                    added_paths,
                    dv_referenced: new_dv_refs,
                    slices_folded: 1,
                });
                DeferOutcome::Deferred { folded: true }
            }
            Some(pending) => {
                let conflict = unattributed_dv
                    || new_dv_refs.iter().any(|f| {
                        pending.dv_referenced.contains(f) || pending.added_paths.contains(f)
                    });
                if conflict {
                    // The slice's already-written files orphan (never
                    // committed); the sweep reclaims them.
                    return DeferOutcome::NeedsFlushAndRetry;
                }
                for f in &lanes.added {
                    pending.added_paths.insert(f.file_path().to_string());
                }
                pending.dv_referenced.extend(new_dv_refs);
                pending.added.extend(lanes.added);
                pending.dvs.extend(lanes.dvs);
                pending.removed.extend(lanes.removed);
                pending.slices_folded += 1;
                DeferOutcome::Deferred { folded: true }
            }
        }
    }

    pub(crate) fn pending_folded(&self) -> usize {
        self.pending
            .lock()
            .expect("pending lock")
            .as_ref()
            .map(|p| p.slices_folded)
            .unwrap_or(0)
    }

    /// Commit the pending fold as ONE RowDelta. Returns the committed
    /// snapshot id (None = nothing pending).
    pub(crate) async fn flush_pending(&self) -> DFResult<Option<i64>> {
        let pending = self.pending.lock().expect("pending lock").take();
        let Some(p) = pending else { return Ok(None) };
        if p.added.is_empty() && p.dvs.is_empty() && p.removed.is_empty() {
            return Ok(None);
        }
        let tx = Transaction::new(&p.table);
        let mut action = tx.row_delta();
        if !p.added.is_empty() {
            action = action.add_data_files(p.added);
        }
        if !p.dvs.is_empty() {
            action = action.add_delete_files(p.dvs);
        }
        if !p.removed.is_empty() {
            action = action.remove_delete_files(p.removed);
        }
        action = match p.base_snapshot {
            Some(snap) => action.validate_from_snapshot(snap),
            None => action.validate_from_empty_table(),
        };
        let committed = action
            .apply(tx)
            .map_err(to_datafusion_error)?
            .commit(p.catalog.as_ref())
            .await
            .map_err(to_datafusion_error)?;
        Ok(committed.metadata().current_snapshot_id())
    }
}

/// Outcome of [`MorWindowState::integrate_deferred`].
#[derive(Debug, PartialEq, Eq)]
pub enum DeferOutcome {
    Deferred { folded: bool },
    NeedsFlushAndRetry,
}

/// Drop the victim rows from the held batches by `(file, pos)`.
fn remove_victims(
    batches: &[RecordBatch],
    victims: &HashMap<String, Vec<u64>>,
) -> DFResult<(Vec<RecordBatch>, usize)> {
    let mut out = Vec::with_capacity(batches.len());
    let mut bytes = 0usize;
    for batch in batches {
        let file_idx = batch
            .schema()
            .index_of(RESERVED_COL_NAME_FILE)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        let pos_idx = batch
            .schema()
            .index_of(RESERVED_COL_NAME_POS)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        let files = batch
            .column(file_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataFusionError::Internal("held _file must be Utf8".to_string()))?;
        let positions = batch
            .column(pos_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataFusionError::Internal("held _pos must be Int64".to_string()))?;
        let mut any_removed = false;
        let mask: BooleanArray = (0..batch.num_rows())
            .map(|i| {
                let hit = victims
                    .get(files.value(i))
                    .map(|v| v.binary_search(&(positions.value(i) as u64)).is_ok())
                    .unwrap_or(false);
                any_removed |= hit;
                Some(!hit)
            })
            .collect();
        let kept = if any_removed {
            filter_record_batch(batch, &mask)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
        } else {
            batch.clone()
        };
        if kept.num_rows() > 0 {
            bytes += kept.get_array_memory_size();
            out.push(kept);
        }
    }
    Ok((out, bytes))
}

/// Scan the slice's appended data files (at the committed snapshot) for
/// rows passing the held predicate/residual — the new current rows the held
/// set gains. Closed copies and tombstones the same files carry are
/// filtered out by exactly the tee's own predicate + residual.
async fn scan_added_currents(
    held: &HeldCurrentSet,
    commit: &CommittedSlice,
) -> DFResult<Vec<RecordBatch>> {
    let select: Vec<String> = held
        .schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let mut builder = commit
        .table
        .scan()
        .with_data_file_path_filter(commit.added_paths.iter().cloned());
    if let Some(snap) = commit.snapshot_id {
        builder = builder.snapshot_id(snap);
    }
    if let Some(pred) = held.predicate.clone() {
        builder = builder.with_filter(pred);
    }
    let scan = builder
        .select(select)
        .build()
        .map_err(to_datafusion_error)?;
    let stream = scan.to_arrow().await.map_err(to_datafusion_error)?;
    let raw: Vec<RecordBatch> = stream.try_collect().await.map_err(to_datafusion_error)?;
    let mut out = Vec::with_capacity(raw.len());
    for batch in raw {
        let batch = plain_cast_batch(&batch, &held.schema)?;
        let batch = match &held.residual {
            None => batch,
            Some(expr) => {
                let mask = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
                let mask = mask
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "held residual must evaluate to boolean".to_string(),
                        )
                    })?
                    .clone();
                filter_record_batch(&batch, &mask)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
            }
        };
        if batch.num_rows() > 0 {
            out.push(batch);
        }
    }
    Ok(out)
}

/// Build a pool reservation for the held bytes; `None` when the pool
/// refuses (the caller treats that as hold overflow).
pub(crate) fn held_reservation(
    pool: &Arc<dyn MemoryPool>,
    bytes: usize,
) -> Option<MemoryReservation> {
    let r = MemoryConsumer::new("mor-window-held").register(pool);
    r.try_grow(bytes).ok().map(|_| r)
}

// ---------------------------------------------------------------------------
// The window driver loop
// ---------------------------------------------------------------------------

/// One slice's parameterization of the window's MERGE template.
#[derive(Debug, Clone, Default)]
pub struct WindowSliceSpec {
    /// Full MERGE statement override; `None` = the window template.
    pub sql: Option<String>,
    /// Per-slice scan-file allowlist, `{catalog.ns.table: [paths]}` —
    /// applied to the mounted providers before planning the slice.
    pub scan_files: Option<HashMap<String, Vec<String>>>,
}

/// One slice's outcome, in slice order.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SliceOutcome {
    pub count: u64,
    pub exec_ms: u64,
    pub plan_ms: u64,
    pub write_ms: u64,
    pub commit_ms: u64,
    pub scan_ms: u64,
    pub join_ms: u64,
    pub write_node_ms: u64,
    /// True once this slice's lanes are durably committed (immediately in
    /// commit mode; at the fold flush in defer mode).
    pub committed: bool,
    /// Defer mode: the slice re-executed after a fold-conflict flush.
    pub reexecuted: bool,
}

/// The window run's aggregate result.
#[derive(Debug, Default)]
pub struct WindowRunResult {
    pub outcomes: Vec<SliceOutcome>,
    /// 0-based index of the slice that failed; completed slices' commits
    /// (and, in defer mode, their flushed fold) are preserved.
    pub failed_slice: Option<usize>,
    pub failed_error: Option<String>,
    pub budget_stopped: bool,
    /// Held-maintenance drop reason, if the hold was abandoned mid-window
    /// (subsequent slices direct-scanned; correctness unaffected).
    pub held_dropped: Option<String>,
}

impl WindowRunResult {
    pub fn total_count(&self) -> u64 {
        self.outcomes.iter().map(|o| o.count).sum()
    }

    pub fn slices_committed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.committed).count()
    }
}

/// Apply a slice's scan-file allowlist to the mounted catalogs.
fn apply_scan_files(
    ctx: &SessionContext,
    scan_files: &HashMap<String, Vec<String>>,
) -> DFResult<()> {
    for (fqn, files) in scan_files {
        let parts: Vec<&str> = fqn.splitn(3, '.').collect();
        if parts.len() != 3 {
            return Err(DataFusionError::Plan(format!(
                "scan_files key `{fqn}` must be `catalog.namespace.table`"
            )));
        }
        let catalog = ctx.catalog(parts[0]).ok_or_else(|| {
            DataFusionError::Plan(format!("scan_files catalog `{}` not mounted", parts[0]))
        })?;
        let iceberg = (catalog.as_ref() as &dyn std::any::Any)
            .downcast_ref::<IcebergCatalogProvider>()
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "scan_files catalog `{}` is not Iceberg-backed",
                    parts[0]
                ))
            })?;
        iceberg
            .with_table_scan_file_allowlist(parts[1], parts[2], files.iter().cloned())
            .map_err(to_datafusion_error)?;
    }
    Ok(())
}

/// Scrape the commit exec's result columns out of the collected batches.
fn scrape_slice(batches: &[RecordBatch], outcome: &mut SliceOutcome) {
    let read = |name: &str| -> u64 {
        batches
            .iter()
            .filter_map(|b| b.column_by_name(name))
            .filter_map(|c| {
                c.as_any()
                    .downcast_ref::<datafusion::arrow::array::UInt64Array>()
            })
            .map(|a| a.iter().flatten().sum::<u64>())
            .sum()
    };
    outcome.count = read("count");
    outcome.write_ms = read("write_ms");
    outcome.commit_ms = read("commit_ms");
    outcome.scan_ms = read("scan_ms");
    outcome.join_ms = read("join_ms");
    outcome.write_node_ms = read("write_node_ms");
}

/// Execute the window: N slice statements over one mounted session.
///
/// The checkpoint contract of the per-slice call shape is preserved
/// byte-for-byte: one RowDelta commit per slice (commit_every=1), a poison
/// slice stops the loop with completed slices' commits intact, and the
/// per-slice outcomes let the caller's cursor logic run unchanged.
pub async fn run_window(
    ctx: &SessionContext,
    window: &Arc<MorWindowState>,
    template_sql: &str,
    slices: &[WindowSliceSpec],
    slice_timeout: Option<Duration>,
    budget_deadline: Option<Instant>,
) -> WindowRunResult {
    let mut result = WindowRunResult::default();
    let commit_every = window.commit_every();
    // Defer mode: outcome indices folded since the last flush (marked
    // committed when their fold lands).
    let mut unflushed: Vec<usize> = Vec::new();

    'slices: for (i, spec) in slices.iter().enumerate() {
        if let Some(bd) = budget_deadline
            && Instant::now() >= bd
            && i > 0
        {
            result.budget_stopped = true;
            break;
        }
        window.set_slice_deadline(slice_timeout.map(|t| Instant::now() + t));
        if let Some(sf) = &spec.scan_files
            && let Err(e) = apply_scan_files(ctx, sf)
        {
            result.failed_slice = Some(i);
            result.failed_error = Some(e.to_string());
            break;
        }
        let sql = spec.sql.as_deref().unwrap_or(template_sql);

        let mut reexecuted = false;
        loop {
            let mut outcome = SliceOutcome {
                reexecuted,
                ..Default::default()
            };
            let plan_t0 = Instant::now();
            let df = match ctx.sql(sql).await {
                Ok(df) => df,
                Err(e) => {
                    result.failed_slice = Some(i);
                    result.failed_error = Some(format!("planning MERGE: {e}"));
                    break 'slices;
                }
            };
            outcome.plan_ms = plan_t0.elapsed().as_millis() as u64;
            let exec_t0 = Instant::now();
            let batches = match df.collect().await {
                Ok(b) => b,
                Err(e) => {
                    result.failed_slice = Some(i);
                    result.failed_error = Some(format!("executing MERGE: {e}"));
                    break 'slices;
                }
            };
            outcome.exec_ms = exec_t0.elapsed().as_millis() as u64;
            scrape_slice(&batches, &mut outcome);

            if commit_every > 1 {
                match window.integrate_deferred() {
                    DeferOutcome::Deferred { .. } => {
                        result.outcomes.push(outcome);
                        unflushed.push(result.outcomes.len() - 1);
                    }
                    DeferOutcome::NeedsFlushAndRetry => {
                        if reexecuted {
                            result.failed_slice = Some(i);
                            result.failed_error = Some(
                                "window fold conflict persisted after re-execution".to_string(),
                            );
                            break 'slices;
                        }
                        match window.flush_pending().await {
                            Ok(_) => {
                                for idx in unflushed.drain(..) {
                                    result.outcomes[idx].committed = true;
                                }
                                reexecuted = true;
                                continue; // re-execute against the flushed head
                            }
                            Err(e) => {
                                result.failed_slice = Some(i);
                                result.failed_error = Some(format!("flushing fold: {e}"));
                                break 'slices;
                            }
                        }
                    }
                }
                if window.pending_folded() >= commit_every {
                    match window.flush_pending().await {
                        Ok(_) => {
                            for idx in unflushed.drain(..) {
                                result.outcomes[idx].committed = true;
                            }
                        }
                        Err(e) => {
                            result.failed_slice = Some(i);
                            result.failed_error = Some(format!("flushing fold: {e}"));
                            break 'slices;
                        }
                    }
                }
            } else {
                outcome.committed = true;
                result.outcomes.push(outcome);
                // Held maintenance — never fails the slice; a failure drops
                // the hold and later slices direct-scan.
                if window.hold_requested()
                    && let Some(reason) = window.apply_slice_delta().await
                {
                    window.mark_hold_overflow();
                    result.held_dropped = Some(reason);
                }
                // Commit mode consumes the channels even when unused so a
                // later slice never sees stale facts.
                let _ = window.take_commit();
            }
            break;
        }
    }

    // Final flush (defer mode): completed folded slices commit even when a
    // later slice poisoned or the budget stopped the loop — maximal durable
    // progress, mirroring the poison-checkpoint contract.
    if commit_every > 1 {
        match window.flush_pending().await {
            Ok(_) => {
                for idx in unflushed.drain(..) {
                    result.outcomes[idx].committed = true;
                }
            }
            Err(e) => {
                let msg = format!("final fold flush failed: {e}");
                result.failed_error = Some(match result.failed_error.take() {
                    Some(prev) => format!("{prev}; {msg}"),
                    None => msg,
                });
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Int64Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

    fn held_batch(files: &[&str], positions: &[i64]) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new(RESERVED_COL_NAME_FILE, DataType::Utf8, false),
            Field::new(RESERVED_COL_NAME_POS, DataType::Int64, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(
                (0..files.len() as i64).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(files.to_vec())),
            Arc::new(Int64Array::from(positions.to_vec())),
        ])
        .unwrap()
    }

    #[test]
    fn remove_victims_filters_by_file_and_pos() {
        let batch = held_batch(&["a", "a", "b"], &[0, 1, 0]);
        let victims = HashMap::from([("a".to_string(), vec![1u64])]);
        let (out, _) = remove_victims(&[batch], &victims).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num_rows(), 2);
        let files = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let positions = out[0]
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            (
                files.value(0),
                positions.value(0),
                files.value(1),
                positions.value(1)
            ),
            ("a", 0, "b", 0)
        );
    }

    #[test]
    fn remove_victims_drops_empty_batches() {
        let batch = held_batch(&["a"], &[0]);
        let victims = HashMap::from([("a".to_string(), vec![0u64])]);
        let (out, bytes) = remove_victims(&[batch], &victims).unwrap();
        assert!(out.is_empty());
        assert_eq!(bytes, 0);
    }

    #[test]
    fn held_snapshot_mismatch_discards_the_hold() {
        let state = MorWindowState::new(true, vec![], HELD_MAX_BYTES_DEFAULT, 1);
        let batch = held_batch(&["a"], &[0]);
        let schema = batch.schema();
        state.install_held(
            "ns.t".to_string(),
            Arc::clone(&schema),
            vec![batch],
            64,
            Some(7),
            None,
            None,
            None,
        );
        // Wrong snapshot: serve refused AND the hold is discarded (an
        // out-of-band commit invalidates it permanently).
        assert!(
            state
                .held_batches_for_target_scan("ns.t", Some(8), &schema)
                .is_none()
        );
        assert!(
            state
                .held_batches_for_target_scan("ns.t", Some(7), &schema)
                .is_none(),
            "the mismatch must have cleared the hold"
        );
    }

    #[test]
    fn held_serves_on_matching_snapshot_and_schema() {
        let state = MorWindowState::new(true, vec![], HELD_MAX_BYTES_DEFAULT, 1);
        let batch = held_batch(&["a"], &[0]);
        let schema = batch.schema();
        state.install_held(
            "ns.t".to_string(),
            Arc::clone(&schema),
            vec![batch],
            64,
            Some(7),
            None,
            None,
            None,
        );
        let served = state
            .held_batches_for_target_scan("ns.t", Some(7), &schema)
            .expect("must serve");
        assert_eq!(served.len(), 1);
        // Provider-side projection subset works and preserves rows.
        let (projected_schema, projected) = state
            .held_batches_for_provider("ns.t", Some(7), &["id".to_string()])
            .expect("projection covered");
        assert_eq!(projected_schema.fields().len(), 1);
        assert_eq!(projected[0].num_columns(), 1);
        assert_eq!(projected[0].num_rows(), 1);
        // Uncovered projection declines without discarding the hold.
        assert!(
            state
                .held_batches_for_provider("ns.t", Some(7), &["nope".to_string()])
                .is_none()
        );
        assert!(
            state
                .held_batches_for_target_scan("ns.t", Some(7), &schema)
                .is_some()
        );
    }

    #[test]
    fn fold_mode_disables_the_hold() {
        let state = MorWindowState::new(true, vec![], HELD_MAX_BYTES_DEFAULT, 4);
        assert!(!state.hold_requested());
        assert_eq!(state.commit_every(), 4);
    }
}
