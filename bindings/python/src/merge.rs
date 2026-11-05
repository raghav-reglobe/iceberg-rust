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

//! Read-only SQL doorway: run `SELECT` statements against Iceberg tables
//! mounted as DataFusion catalogs.
//!
//! `catalogs` maps each SQL catalog name used in the statement to standard
//! Iceberg REST catalog properties, so every `catalog.namespace.table`
//! reference resolves with no register steps. A call may restrict a table's
//! scan to a planned set of data files, pin it to a snapshot, or mount only a
//! named set of tables. Results come back as JSON text or as an Arrow IPC
//! stream. Catalog handles are memoized per process, so repeated calls reuse
//! the same authenticated clients instead of re-fetching config and tokens.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use datafusion::common::resources_datafusion_err;
use datafusion::execution::context::SessionContext;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation, TrackConsumersPool, UnboundedMemoryPool, human_readable_size};
use iceberg::Catalog;
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_datafusion::functions::{register_parity_functions, register_variant_functions};
use iceberg_datafusion::IcebergCatalogProvider;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

/// Per-process memoized catalog handles, keyed by catalog name + properties.
/// A REST catalog re-auths (`/config` + OAuth token) on construction; reusing
/// the handle across calls avoids hammering the catalog server from a worker
/// that merges every few seconds.
static CATALOGS: OnceLock<Mutex<HashMap<String, Arc<dyn Catalog>>>> = OnceLock::new();

fn catalog_cache_key(name: &str, props: &HashMap<String, String>) -> String {
    let mut entries: Vec<_> = props.iter().collect();
    entries.sort();
    let mut key = String::from(name);
    for (k, v) in entries {
        key.push('\u{1f}');
        key.push_str(k);
        key.push('\u{1f}');
        key.push_str(v);
    }
    key
}

pub(crate) async fn get_or_build_catalog(
    name: &str,
    props: HashMap<String, String>,
) -> PyResult<Arc<dyn Catalog>> {
    let key = catalog_cache_key(name, &props);
    if let Some(cat) = CATALOGS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .get(&key)
    {
        return Ok(Arc::clone(cat));
    }
    let builder = RestCatalogBuilder::default()
        .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
        .with_object_bytes_cache(crate::runtime::global_object_cache().await);
    let builder = match crate::runtime::global_data_cache().await {
        Some(dc) => builder.with_data_bytes_cache(dc),
        None => builder,
    };
    let catalog = builder
        .load(name.to_string(), props)
        .await
        .map_err(|e| PyValueError::new_err(format!("build catalog `{name}`: {e}")))?;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    CATALOGS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(key, Arc::clone(&catalog));
    Ok(catalog)
}

/// `scan_files` keys are `catalog.namespace.table` (namespace may be
/// multi-level: first segment = catalog, last = table, middle = namespace).
/// Grouped per catalog as `(namespace, table) -> files`.
type ScanFiles = HashMap<String, HashMap<(String, String), Vec<String>>>;

fn parse_scan_files(scan_files: Option<HashMap<String, Vec<String>>>) -> PyResult<ScanFiles> {
    let mut out: ScanFiles = HashMap::new();
    for (fqn, files) in scan_files.unwrap_or_default() {
        let parts: Vec<&str> = fqn.split('.').collect();
        if parts.len() < 3 {
            return Err(PyValueError::new_err(format!(
                "scan_files key `{fqn}` must be `catalog.namespace.table`"
            )));
        }
        let catalog = parts[0].to_string();
        let table = parts[parts.len() - 1].to_string();
        let namespace = parts[1..parts.len() - 1].join(".");
        out.entry(catalog)
            .or_default()
            .insert((namespace, table), files);
    }
    Ok(out)
}

/// `snapshot_ids` keys are `catalog.namespace.table` like `scan_files`; the
/// value is the Iceberg snapshot id every scan of that table reads (time
/// travel for externally pinned reads: a twin build or a referee load names
/// the exact snapshot it was planned against and the read stays exact for as
/// long as the snapshot is retained — an expired pin FAILS the scan, it never
/// reads 0 rows). Grouped per catalog as `(namespace, table) -> snapshot id`.
/// Read doorways only; the merge doorways always read current.
type SnapshotIds = HashMap<String, HashMap<(String, String), i64>>;

fn parse_snapshot_ids(snapshot_ids: Option<HashMap<String, i64>>) -> PyResult<SnapshotIds> {
    let mut out: SnapshotIds = HashMap::new();
    for (fqn, snapshot_id) in snapshot_ids.unwrap_or_default() {
        let parts: Vec<&str> = fqn.split('.').collect();
        if parts.len() < 3 {
            return Err(PyValueError::new_err(format!(
                "snapshot_ids key `{fqn}` must be `catalog.namespace.table`"
            )));
        }
        let catalog = parts[0].to_string();
        let table = parts[parts.len() - 1].to_string();
        let namespace = parts[1..parts.len() - 1].join(".");
        out.entry(catalog)
            .or_default()
            .insert((namespace, table), snapshot_id);
    }
    Ok(out)
}

type ScopedTables = HashMap<String, HashMap<String, Vec<String>>>;

/// `{catalog: ["namespace.table", ...]}` -> catalog -> namespace -> tables.
/// A catalog WITH an entry mounts scoped (zero list calls, one load_table
/// per named table — the per-call latency floor + Polaris-stampede fix);
/// a catalog WITHOUT one keeps the full eager mount.
fn parse_scoped_tables(scoped: Option<HashMap<String, Vec<String>>>) -> PyResult<ScopedTables> {
    let mut out: ScopedTables = HashMap::new();
    for (catalog, idents) in scoped.unwrap_or_default() {
        for ident in idents {
            let (ns, tbl) = ident.rsplit_once('.').ok_or_else(|| {
                PyValueError::new_err(format!(
                    "scoped_tables entry `{ident}` must be `namespace.table`"
                ))
            })?;
            out.entry(catalog.clone())
                .or_default()
                .entry(ns.to_string())
                .or_default()
                .push(tbl.to_string());
        }
    }
    Ok(out)
}

/// A [`MemoryPool`] wrapper recording the pool-wide high-water mark of
/// reserved bytes. `TrackConsumersPool` keeps per-consumer peaks, but
/// `unregister` DROPS a consumer's entry the moment its operator finishes —
/// an end-of-query read misses everything already deregistered. A global
/// `fetch_max` after every grow survives all deregistrations and reflects
/// the true concurrent peak, not a sum of non-coincident per-consumer peaks.
///
/// Scope caveat: this observes only DataFusion-REGISTERED reservations
/// (join builds, sorts, aggregates) — arrow allocations made directly by the
/// merge exec nodes (fetch batches, evaluated SET stores, writer buffers)
/// are byte-bounded separately and do not pass through the pool.
/// Keeps the LAST `reserve` bytes of a bounded pool for consumers that
/// cannot spill.
///
/// DataFusion's `ExternalSorter` grows its (spillable) in-memory reservation
/// until the pool refuses and only THEN spills. Under a greedy pool, N sort
/// partitions therefore fill 100% of the pool with spillable batches, and
/// whichever UNSPILLABLE consumer asks next — the hash-join build side, a
/// partition's own pre-spill merge reservation — fails the whole statement.
/// Pool SIZE only scales the fill: the same hot-key bucket class failed at
/// 18G, 30G and 36.9G pools with one signature (2026-09-07). This wrapper
/// changes the RULE instead: a spillable `try_grow` is refused once the
/// pool's total reservation would exceed `limit - reserve` (the sorter then
/// spills and retries — its normal path), while unspillable requests pass
/// straight to the inner pool and always find at least `reserve` bytes free.
/// A spillable consumer holding nothing yet always gets its first grant
/// (a sorter with no in-memory batches cannot spill, so refusing it would
/// be a hard failure); the cap governs growth.
/// Unspillable consumers still fail cleanly at the true limit (the inner
/// pool's top-consumers report is preserved on that path).
///
/// Installed only when `MERGE_DF_UNSPILLABLE_RESERVE_PCT` (1..=90, percent of
/// the bounded limit) is set on a bounded pool; unset = the pre-existing
/// greedy behaviour, byte-identical.
#[derive(Debug)]
struct UnspillableReservePool {
    inner: Arc<dyn MemoryPool>,
    /// The bounded limit of the inner pool.
    limit: usize,
    /// `limit - reserve`: the most the pool may hold in TOTAL at the moment a
    /// spillable consumer is granted more.
    spillable_cap: usize,
}

impl UnspillableReservePool {
    fn new(inner: Arc<dyn MemoryPool>, limit: usize, reserve_pct: usize) -> Self {
        let reserve = limit / 100 * reserve_pct.min(90);
        Self {
            inner,
            limit,
            spillable_cap: limit.saturating_sub(reserve),
        }
    }

    /// `MERGE_DF_UNSPILLABLE_RESERVE_PCT` parsed and bounded to 1..=90;
    /// `None` = no reserve (unset, unparsable, or out of range).
    fn reserve_pct_from_env() -> Option<usize> {
        std::env::var("MERGE_DF_UNSPILLABLE_RESERVE_PCT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|p| (1..=90).contains(p))
    }
}

impl std::fmt::Display for UnspillableReservePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UnspillableReservePool(spillable_cap: {}, limit: {}, inner: {})",
            human_readable_size(self.spillable_cap),
            human_readable_size(self.limit),
            self.inner
        )
    }
}

impl MemoryPool for UnspillableReservePool {
    fn name(&self) -> &str {
        "unspillable_reserve"
    }

    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer);
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        // The cap applies to GROWTH: a spillable consumer holding nothing yet
        // always gets its first grant (bounded by one batch). Refusing it
        // would be a hard failure — a sorter with no in-memory batches has
        // nothing to spill — while a late-starting sorter meeting a pool
        // already at the cap is exactly what a skewed partition produces.
        if reservation.consumer().can_spill() && reservation.size() > 0 {
            let reserved = self.inner.reserved();
            if reserved.saturating_add(additional) > self.spillable_cap {
                return Err(resources_datafusion_err!(
                    "Failed to allocate additional {} for {} (spillable) with {} already allocated for this reservation - spillable consumers are capped at {} of the {} pool ({} kept for unspillable consumers); {} reserved in total",
                    human_readable_size(additional),
                    reservation.consumer().name(),
                    human_readable_size(reservation.size()),
                    human_readable_size(self.spillable_cap),
                    human_readable_size(self.limit),
                    human_readable_size(self.limit - self.spillable_cap),
                    human_readable_size(reserved)
                ));
            }
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

/// Bytes reserved per consumer NAME right now, and that table as it stood at
/// the pool's high-water mark.
#[derive(Debug, Default)]
struct ConsumerLedger {
    current: HashMap<String, usize>,
    at_peak: Vec<(String, usize)>,
}

#[derive(Debug)]
struct PeakTrackingPool {
    inner: Arc<dyn MemoryPool>,
    peak: AtomicUsize,
    /// WHO held the peak, not only how big it was. The pool's total is the
    /// operators' own ACCOUNTING, not resident memory: a reservation sized
    /// from `get_array_memory_size` counts a buffer once per batch that
    /// shares it (view arrays, slices), so a total can exceed the machine by
    /// orders of magnitude on an unbounded pool — and on a bounded one the
    /// same inflation spills or refuses early. Naming the holders is what
    /// tells accounting from memory.
    ///
    /// `None` unless the caller asked for it: the peak alone costs one atomic
    /// per reservation change; the ledger adds a lock and a map lookup, so a
    /// run nobody is investigating pays nothing for it.
    ledger: Option<Mutex<ConsumerLedger>>,
}

/// How many holders `peak_mem_consumers` names.
const PEAK_CONSUMERS: usize = 5;

/// Environment switch for the ledger: `MERGE_DF_PEAK_CONSUMERS` = `1` /
/// `true` / `on` / `yes` (any case). Unset or anything else = off.
const PEAK_CONSUMERS_ENV: &str = "MERGE_DF_PEAK_CONSUMERS";

fn peak_consumers_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// `ExternalSorter[3]` and `ExternalSorter[7]` are one holder: partitions of
/// one operator.
fn consumer_family(name: &str) -> &str {
    match name.rfind('[') {
        // a non-empty name, then `[digits]` with at least one digit
        Some(i)
            if i > 0
                && name.ends_with(']')
                && i + 2 < name.len()
                && name[i + 1..name.len() - 1]
                    .bytes()
                    .all(|b| b.is_ascii_digit()) =>
        {
            &name[..i]
        }
        _ => name,
    }
}

impl PeakTrackingPool {
    /// The high-water mark only (one atomic per reservation change).
    fn new(inner: Arc<dyn MemoryPool>) -> Self {
        Self {
            inner,
            peak: AtomicUsize::new(0),
            ledger: None,
        }
    }

    /// The high-water mark AND who held it.
    fn with_consumers(inner: Arc<dyn MemoryPool>) -> Self {
        Self {
            ledger: Some(Mutex::new(ConsumerLedger::default())),
            ..Self::new(inner)
        }
    }

    /// `with_consumers` when [`PEAK_CONSUMERS_ENV`] asks for it, else `new`.
    fn from_env(inner: Arc<dyn MemoryPool>) -> Self {
        if peak_consumers_enabled(std::env::var(PEAK_CONSUMERS_ENV).ok().as_deref()) {
            Self::with_consumers(inner)
        } else {
            Self::new(inner)
        }
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// The largest holders at the pool's high-water mark, as a JSON object
    /// `{consumer: bytes}` (at most [`PEAK_CONSUMERS`], largest first).
    /// `None` when the ledger was not asked for.
    fn peak_consumers_json(&self) -> Option<String> {
        let ledger = self
            .ledger
            .as_ref()?
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let body: Vec<String> = ledger
            .at_peak
            .iter()
            .map(|(name, bytes)| {
                format!(
                    "\"{}\":{bytes}",
                    name.replace('\\', "\\\\").replace('"', "\\\"")
                )
            })
            .collect();
        Some(format!("{{{}}}", body.join(",")))
    }

    fn grew(&self, reservation: &MemoryReservation, additional: usize) {
        let Some(ledger) = self.ledger.as_ref() else {
            self.peak
                .fetch_max(self.inner.reserved(), Ordering::Relaxed);
            return;
        };
        let mut ledger = ledger.lock().unwrap_or_else(|e| e.into_inner());
        // No allocation on the hot path: a name is copied once, on first sight.
        let family = consumer_family(reservation.consumer().name());
        match ledger.current.get_mut(family) {
            Some(bytes) => *bytes += additional,
            None => {
                ledger.current.insert(family.to_string(), additional);
            }
        }
        let reserved = self.inner.reserved();
        if reserved > self.peak.fetch_max(reserved, Ordering::Relaxed) {
            let mut holders: Vec<(String, usize)> = ledger
                .current
                .iter()
                .filter(|(_, bytes)| **bytes > 0)
                .map(|(name, bytes)| (name.clone(), *bytes))
                .collect();
            holders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            holders.truncate(PEAK_CONSUMERS);
            ledger.at_peak = holders;
        }
    }

    fn shrank(&self, reservation: &MemoryReservation, shrink: usize) {
        let Some(ledger) = self.ledger.as_ref() else {
            return;
        };
        let mut ledger = ledger.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(bytes) = ledger
            .current
            .get_mut(consumer_family(reservation.consumer().name()))
        {
            *bytes = bytes.saturating_sub(shrink);
        }
    }
}

impl std::fmt::Display for PeakTrackingPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PeakTrackingPool(inner: {})", self.inner)
    }
}

impl MemoryPool for PeakTrackingPool {
    fn name(&self) -> &str {
        "peak_tracking"
    }

    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer);
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.grew(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
        self.shrank(reservation, shrink);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.grew(reservation, additional);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

async fn session_with_catalogs(
    catalogs: HashMap<String, HashMap<String, String>>,
    mut scan_files: ScanFiles,
    mut snapshot_ids: SnapshotIds,
    mut scoped_tables: ScopedTables,
    local_tables: HashMap<String, String>,
) -> PyResult<(SessionContext, Arc<PeakTrackingPool>)> {
    // Preserve identifier case: column names coming from document stores
    // are mixed-case, and callers quote them as written.
    let mut config = datafusion::execution::context::SessionConfig::new()
        .set_bool("datafusion.sql_parser.enable_ident_normalization", false);
    // Plan-level parallelism: DataFusion defaults target_partitions to the
    // HOST core count, which on an over-subscribed container multiplies every
    // per-partition buffer (sorts, repartitions, join builds) past the pod's
    // memory for zero extra throughput. Env-tunable alongside the pool.
    if let Some(n) = std::env::var("MERGE_DF_TARGET_PARTITIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        config = config.with_target_partitions(n);
    }
    // Memory safety: an explicit memory pool makes DataFusion's sort/window/
    // aggregate operators SPILL instead of OOM-killing the process (the
    // merge's union sort is the big resident; a row cap alone cannot bound
    // memory — width varies per table). Env-configured so the worker sizes
    // it to the pod: MERGE_DF_MEMORY_LIMIT_MB (unset = unbounded, the
    // previous behavior) + MERGE_DF_SPILL_DIR (unset = OS temp dir).
    //
    // A pool is installed EVEN when unbounded, wrapped in the peak tracker,
    // so every call reports its pool-visible high-water mark
    // (`peak_mem_bytes` in the result; the holders behind it only on
    // request — MERGE_DF_PEAK_CONSUMERS=1). The bounded shape mirrors what
    // `RuntimeEnvBuilder::with_memory_limit` builds (Greedy inside
    // TrackConsumersPool), keeping the top-consumers OOM report.
    let inner: Arc<dyn MemoryPool> = match std::env::var("MERGE_DF_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        None => Arc::new(UnboundedMemoryPool::default()),
        Some(limit_mb) => {
            let limit = limit_mb * 1024 * 1024;
            let tracked: Arc<dyn MemoryPool> = Arc::new(TrackConsumersPool::new(
                GreedyMemoryPool::new(limit),
                std::num::NonZeroUsize::new(5).expect("non-zero"),
            ));
            // MERGE_DF_UNSPILLABLE_RESERVE_PCT: keep the last N% of the bounded
            // pool for consumers that cannot spill (see UnspillableReservePool).
            // Unset = the greedy pool alone, byte-identical to before.
            match UnspillableReservePool::reserve_pct_from_env() {
                Some(pct) => Arc::new(UnspillableReservePool::new(tracked, limit, pct))
                    as Arc<dyn MemoryPool>,
                None => tracked,
            }
        }
    };
    let pool = Arc::new(PeakTrackingPool::from_env(inner));
    let mut rt = datafusion::execution::runtime_env::RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>);
    if let Ok(dir) = std::env::var("MERGE_DF_SPILL_DIR") {
        rt = rt.with_disk_manager_builder(
            datafusion::execution::disk_manager::DiskManagerBuilder::default().with_mode(
                datafusion::execution::disk_manager::DiskManagerMode::Directories(vec![dir.into()]),
            ),
        );
    }
    let rt = rt
        .build_arc()
        .map_err(|e| PyValueError::new_err(format!("building runtime env: {e}")))?;
    let ctx = SessionContext::new_with_config_rt(config, rt);
    register_variant_functions(&ctx);
    register_parity_functions(&ctx);
    for (name, props) in catalogs {
        let catalog = get_or_build_catalog(&name, props).await?;
        let provider = match scoped_tables.remove(&name) {
            Some(scope) => IcebergCatalogProvider::try_new_scoped(catalog, scope).await,
            None => IcebergCatalogProvider::try_new(catalog).await,
        }
        .map_err(|e| PyValueError::new_err(format!("mount catalog `{name}`: {e}")))?;
        // Scan-file allowlists (externally planned file subsets) are applied
        // per call — providers are rebuilt per session; only the catalog
        // handle above is memoized.
        for ((namespace, table), files) in scan_files.remove(&name).unwrap_or_default() {
            provider
                .with_table_scan_file_allowlist(&namespace, &table, files)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "scan_files for `{name}.{namespace}.{table}`: {e}"
                    ))
                })?;
        }
        // Snapshot pins (time travel for externally planned reads) — applied
        // per call like the allowlists and composing with them: the
        // allowlist filters the PINNED snapshot's plan.
        for ((namespace, table), snapshot_id) in snapshot_ids.remove(&name).unwrap_or_default() {
            provider
                .with_table_snapshot_id(&namespace, &table, snapshot_id)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "snapshot_ids for `{name}.{namespace}.{table}`: {e}"
                    ))
                })?;
        }
        ctx.register_catalog(&name, Arc::new(provider));
    }
    if let Some(unmatched) = scan_files.keys().next() {
        return Err(PyValueError::new_err(format!(
            "scan_files references catalog `{unmatched}` which is not in `catalogs`"
        )));
    }
    if let Some(unmatched) = snapshot_ids.keys().next() {
        return Err(PyValueError::new_err(format!(
            "snapshot_ids references catalog `{unmatched}` which is not in `catalogs`"
        )));
    }
    if let Some(unmatched) = scoped_tables.keys().next() {
        return Err(PyValueError::new_err(format!(
            "scoped_tables references catalog `{unmatched}` which is not in `catalogs`"
        )));
    }
    // Local parquet tables — `{name: path}` registered in the session's
    // default catalog, referenced by BARE name in the SQL. The sanctioned
    // way to join side data (e.g. a value-remap table) WITHOUT inlining the
    // values as SQL literals: statement text surfaces in planner errors and
    // logs, so sensitive values must ride in files, never literals.
    for (name, path) in local_tables {
        ctx.register_parquet(&name, &path, Default::default())
            .await
            .map_err(|e| {
                PyValueError::new_err(format!("register local table `{name}` from `{path}`: {e}"))
            })?;
    }
    Ok((ctx, pool))
}

#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, max_rows=100_000, scoped_tables=None, local_tables=None, snapshot_ids=None))]
fn sql_collect(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    max_rows: usize,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    local_tables: Option<HashMap<String, String>>,
    snapshot_ids: Option<HashMap<String, i64>>,
) -> PyResult<String> {
    use datafusion::logical_expr::LogicalPlan;

    let scan_files = parse_scan_files(scan_files)?;
    let snapshot_ids = parse_snapshot_ids(snapshot_ids)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    py.detach(|| {
        runtime().block_on(async move {
            let (ctx, _pool) = session_with_catalogs(
                catalogs,
                scan_files,
                snapshot_ids,
                scoped_tables,
                local_tables,
            )
            .await?;
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| PyValueError::new_err(format!("planning query: {e}")))?;
            if matches!(
                df.logical_plan(),
                LogicalPlan::Dml(_) | LogicalPlan::Ddl(_) | LogicalPlan::Copy(_)
            ) {
                return Err(PyValueError::new_err(
                    "sql_collect is read-only",
                ));
            }
            let batches = df
                .collect()
                .await
                .map_err(|e| PyValueError::new_err(format!("executing query: {e}")))?;
            let total: usize = batches.iter().map(|b| b.num_rows()).sum();
            if total > max_rows {
                return Err(PyValueError::new_err(format!(
                    "result has {total} rows > max_rows={max_rows} — narrow the query or raise the cap"
                )));
            }
            let mut buf = Vec::new();
            {
                let mut writer = arrow::json::ArrayWriter::new(&mut buf);
                for batch in &batches {
                    writer
                        .write(batch)
                        .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))?;
                }
                writer
                    .finish()
                    .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))?;
            }
            String::from_utf8(buf)
                .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))
        })
    })
}

/// Run one READ-ONLY SQL statement and return its result as ARROW IPC
/// STREAM bytes (`pyarrow.ipc.open_stream(...).read_all()`) — the
/// byte-fidelity twin of [`sql_collect`]. JSON round-trips lose exactness
/// (float rendering, binary, timestamp precision); IPC preserves the arrow
/// values verbatim, which value-keyed consumers (a remap's `original`
/// column must re-match the stored value EXACTLY) require. Same read-only
/// guard and loud `max_rows` cap; the stream always carries the schema, so
/// a zero-row result is a valid empty table, not an error.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, max_rows=100_000, scoped_tables=None, local_tables=None, snapshot_ids=None))]
fn sql_collect_ipc(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    max_rows: usize,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    local_tables: Option<HashMap<String, String>>,
    snapshot_ids: Option<HashMap<String, i64>>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    use datafusion::logical_expr::LogicalPlan;

    let scan_files = parse_scan_files(scan_files)?;
    let snapshot_ids = parse_snapshot_ids(snapshot_ids)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    let buf: Vec<u8> = py.detach(|| {
        runtime().block_on(async move {
            let (ctx, _pool) = session_with_catalogs(
                catalogs,
                scan_files,
                snapshot_ids,
                scoped_tables,
                local_tables,
            )
            .await?;
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| PyValueError::new_err(format!("planning query: {e}")))?;
            if matches!(
                df.logical_plan(),
                LogicalPlan::Dml(_) | LogicalPlan::Ddl(_) | LogicalPlan::Copy(_)
            ) {
                return Err(PyValueError::new_err(
                    "sql_collect_ipc is read-only",
                ));
            }
            let schema = Arc::new(df.schema().as_arrow().clone());
            let batches = df
                .collect()
                .await
                .map_err(|e| PyValueError::new_err(format!("executing query: {e}")))?;
            let total: usize = batches.iter().map(|b| b.num_rows()).sum();
            if total > max_rows {
                return Err(PyValueError::new_err(format!(
                    "result has {total} rows > max_rows={max_rows} — narrow the query or raise the cap"
                )));
            }
            let mut buf = Vec::new();
            {
                let mut writer =
                    arrow::ipc::writer::StreamWriter::try_new(&mut buf, &schema)
                        .map_err(|e| {
                            PyValueError::new_err(format!("serializing rows: {e}"))
                        })?;
                for batch in &batches {
                    writer
                        .write(batch)
                        .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))?;
                }
                writer
                    .finish()
                    .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))?;
            }
            Ok(buf)
        })
    })?;
    Ok(pyo3::types::PyBytes::new(py, &buf).unbind())
}


pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "merge")?;
    this.add_function(wrap_pyfunction!(sql_collect, &this)?)?;
    this.add_function(wrap_pyfunction!(sql_collect_ipc, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}

#[cfg(test)]
mod peak_tracking_tests {
    use super::*;

    #[test]
    fn names_the_holders_at_the_high_water_mark() {
        let inner: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let tracker = Arc::new(PeakTrackingPool::with_consumers(inner));
        let pool: Arc<dyn MemoryPool> = Arc::clone(&tracker) as Arc<dyn MemoryPool>;
        // two partitions of one operator + one other consumer
        let s0 = MemoryConsumer::new("ExternalSorter[0]").register(&pool);
        let s1 = MemoryConsumer::new("ExternalSorter[1]").register(&pool);
        let scan = MemoryConsumer::new("iceberg-scan-decode").register(&pool);
        s0.try_grow(600).unwrap();
        scan.try_grow(100).unwrap();
        s1.try_grow(300).unwrap(); // high-water mark: 1000
        assert_eq!(tracker.peak(), 1000);
        assert_eq!(
            tracker.peak_consumers_json().as_deref(),
            Some(r#"{"ExternalSorter":900,"iceberg-scan-decode":100}"#)
        );
        // Below the mark nothing moves: the table is a snapshot OF the peak.
        s0.shrink(600);
        scan.try_grow(50).unwrap();
        assert_eq!(tracker.peak(), 1000);
        assert_eq!(
            tracker.peak_consumers_json().as_deref(),
            Some(r#"{"ExternalSorter":900,"iceberg-scan-decode":100}"#)
        );
        // A new mark re-takes it — a released holder is gone from it.
        scan.try_grow(900).unwrap(); // 300 + 1050 = 1350
        assert_eq!(tracker.peak(), 1350);
        assert_eq!(
            tracker.peak_consumers_json().as_deref(),
            Some(r#"{"iceberg-scan-decode":1050,"ExternalSorter":300}"#)
        );
    }

    #[test]
    fn a_dropped_reservation_leaves_the_ledger() {
        let inner: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let tracker = Arc::new(PeakTrackingPool::with_consumers(inner));
        let pool: Arc<dyn MemoryPool> = Arc::clone(&tracker) as Arc<dyn MemoryPool>;
        {
            let gone = MemoryConsumer::new("HashJoinInput[0]").register(&pool);
            gone.try_grow(500).unwrap();
        } // dropped: frees through shrink
        let stays = MemoryConsumer::new("mor-window-held").register(&pool);
        stays.try_grow(700).unwrap();
        assert_eq!(tracker.peak(), 700);
        assert_eq!(
            tracker.peak_consumers_json().as_deref(),
            Some(r#"{"mor-window-held":700}"#)
        );
    }

    #[test]
    fn without_the_ledger_the_peak_is_still_tracked_and_no_holders_are_named() {
        let inner: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let tracker = Arc::new(PeakTrackingPool::new(inner));
        let pool: Arc<dyn MemoryPool> = Arc::clone(&tracker) as Arc<dyn MemoryPool>;
        let a = MemoryConsumer::new("ExternalSorter[0]").register(&pool);
        let b = MemoryConsumer::new("iceberg-scan-decode").register(&pool);
        a.try_grow(600).unwrap();
        b.try_grow(100).unwrap();
        a.shrink(600);
        assert_eq!(tracker.peak(), 700);
        assert_eq!(tracker.peak_consumers_json(), None);
    }

    #[test]
    fn the_switch_reads_only_an_explicit_yes() {
        for yes in ["1", "true", "TRUE", " on ", "Yes"] {
            assert!(peak_consumers_enabled(Some(yes)), "{yes:?}");
        }
        for no in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("off"),
            Some("2"),
        ] {
            assert!(!peak_consumers_enabled(no), "{no:?}");
        }
    }

    #[test]
    fn only_a_trailing_partition_index_is_folded() {
        assert_eq!(consumer_family("ExternalSorter[12]"), "ExternalSorter");
        assert_eq!(consumer_family("mor-window-held"), "mor-window-held");
        assert_eq!(consumer_family("odd[name]"), "odd[name]");
        assert_eq!(consumer_family("x[]"), "x[]");
        assert_eq!(consumer_family("[3]"), "[3]");
    }
}

#[cfg(test)]
mod unspillable_reserve_tests {
    use super::*;

    fn pool(limit: usize, pct: usize) -> Arc<dyn MemoryPool> {
        let greedy: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        Arc::new(UnspillableReservePool::new(greedy, limit, pct))
    }

    #[test]
    fn spillable_is_capped_below_the_reserve_and_unspillable_is_not() {
        let p = pool(1000, 40); // spillable_cap = 600
        let sorter = MemoryConsumer::new("sorter")
            .with_can_spill(true)
            .register(&p);
        let join = MemoryConsumer::new("join")
            .with_can_spill(false)
            .register(&p);
        sorter.try_grow(600).expect("spillable fills up to the cap");
        assert!(
            sorter.try_grow(1).is_err(),
            "spillable may not enter the reserve"
        );
        join.try_grow(300)
            .expect("unspillable grows into the reserve");
        assert_eq!(p.reserved(), 900);
        assert!(
            join.try_grow(200).is_err(),
            "unspillable still bounded by the limit"
        );
        // once the sorter spills (shrinks), the join can proceed
        sorter.shrink(300);
        join.try_grow(100)
            .expect("freed spillable memory is usable by unspillable");
        assert_eq!(p.reserved(), 700);
    }

    #[test]
    fn empty_spillable_always_gets_its_first_grant() {
        let p = pool(1000, 40); // spillable_cap = 600
        let fat = MemoryConsumer::new("fat").with_can_spill(true).register(&p);
        let thin = MemoryConsumer::new("thin")
            .with_can_spill(true)
            .register(&p);
        fat.try_grow(600).unwrap();
        thin.try_grow(100)
            .expect("first grant passes even at the cap");
        assert!(
            thin.try_grow(1).is_err(),
            "growth beyond the cap is refused"
        );
        assert!(fat.try_grow(1).is_err());
        assert_eq!(p.reserved(), 700);
    }

    #[test]
    fn spillable_cap_accounts_for_unspillable_holdings() {
        let p = pool(1000, 20); // spillable_cap = 800
        let join = MemoryConsumer::new("join")
            .with_can_spill(false)
            .register(&p);
        let sorter = MemoryConsumer::new("sorter")
            .with_can_spill(true)
            .register(&p);
        join.try_grow(500).unwrap();
        // total may not exceed 800 when a spillable asks: 500 + 300 ok, +1 refused
        sorter.try_grow(300).unwrap();
        assert!(sorter.try_grow(1).is_err());
    }

    #[test]
    fn env_parse_is_bounded() {
        for (raw, want) in [
            ("40", Some(40)),
            (" 5 ", Some(5)),
            ("90", Some(90)),
            ("0", None),
            ("91", None),
            ("abc", None),
            ("", None),
        ] {
            unsafe { std::env::set_var("MERGE_DF_UNSPILLABLE_RESERVE_PCT", raw) };
            assert_eq!(
                UnspillableReservePool::reserve_pct_from_env(),
                want,
                "raw={raw:?}"
            );
        }
        unsafe { std::env::remove_var("MERGE_DF_UNSPILLABLE_RESERVE_PCT") };
        assert_eq!(UnspillableReservePool::reserve_pct_from_env(), None);
    }
}

#[cfg(test)]
mod snapshot_pin_tests {
    use super::*;

    #[test]
    fn snapshot_ids_group_per_catalog_and_split_multilevel_namespaces() {
        let ids = parse_snapshot_ids(Some(HashMap::from([
            (
                "bronze_dbnew.cashify_mart.orders".to_string(),
                1_833_894_294_735_745_083_i64,
            ),
            (
                "bronze_dbadhoc.rawtwin.kapture.ticket_report_data_t6".to_string(),
                8_076_658_723_558_311_002_i64,
            ),
        ])))
        .unwrap();
        assert_eq!(
            ids["bronze_dbnew"][&("cashify_mart".to_string(), "orders".to_string())],
            1_833_894_294_735_745_083
        );
        assert_eq!(
            ids["bronze_dbadhoc"][&(
                "rawtwin.kapture".to_string(),
                "ticket_report_data_t6".to_string()
            )],
            8_076_658_723_558_311_002
        );
    }

    #[test]
    fn snapshot_ids_reject_a_two_part_key() {
        assert!(
            parse_snapshot_ids(Some(HashMap::from([("bronze.orders".to_string(), 1_i64)])))
                .is_err()
        );
    }

    #[test]
    fn no_snapshot_ids_is_empty() {
        assert!(parse_snapshot_ids(None).unwrap().is_empty());
    }
}
