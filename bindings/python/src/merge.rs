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

//! One-shot `MERGE INTO` doorway: run genuine MERGE SQL text against Iceberg
//! tables mounted as DataFusion catalogs.
//!
//! The statement is planned by DataFusion's MERGE planner and executed by the
//! merge-on-read chain behind `TableProvider::merge_into` (deletion vectors
//! on the scanned `(_file, _pos)`, late-materialized matched-row appends, one
//! `RowDelta` snapshot). `catalogs` maps each SQL catalog name used in the
//! statement to standard Iceberg REST catalog properties, so every
//! `catalog.namespace.table` reference resolves with no register steps.
//! Catalog handles are memoized per process — repeated calls reuse the same
//! authenticated clients instead of re-fetching config/tokens per call.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use datafusion::arrow::array::{Array, StringArray, UInt64Array};
use datafusion::execution::context::SessionContext;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
    TrackConsumersPool, UnboundedMemoryPool,
};
use datafusion::physical_plan::displayable;
use iceberg::{Catalog, CatalogBuilder};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_datafusion::functions::{register_parity_functions, register_variant_functions};
use iceberg_datafusion::{
    HELD_MAX_BYTES_DEFAULT, IcebergCatalogProvider, MorMergeOptions, MorWindowState,
    WindowSliceSpec, run_window,
};
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
#[derive(Debug)]
struct PeakTrackingPool {
    inner: Arc<dyn MemoryPool>,
    peak: AtomicUsize,
}

impl PeakTrackingPool {
    fn new(inner: Arc<dyn MemoryPool>) -> Self {
        Self {
            inner,
            peak: AtomicUsize::new(0),
        }
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    fn bump(&self) {
        self.peak
            .fetch_max(self.inner.reserved(), Ordering::Relaxed);
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
        self.bump();
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.bump();
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
    mut scoped_tables: ScopedTables,
    local_tables: HashMap<String, String>,
    options: Option<Arc<MorMergeOptions>>,
) -> PyResult<(SessionContext, Arc<PeakTrackingPool>)> {
    // Preserve identifier case (duckdb/Spark semantics): mongo-derived columns
    // are mixed-case, and the in-flight MERGE planner drops the quote flag on
    // INSERT/SET column names, so normalization would lowercase them anyway.
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
    if let Some(options) = options {
        config = config.with_extension(options);
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
    // (`peak_mem_bytes` in the result). The bounded shape mirrors what
    // `RuntimeEnvBuilder::with_memory_limit` builds (Greedy inside
    // TrackConsumersPool), keeping the top-consumers OOM report.
    let inner: Arc<dyn MemoryPool> = match std::env::var("MERGE_DF_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        None => Arc::new(UnboundedMemoryPool::default()),
        Some(limit_mb) => Arc::new(TrackConsumersPool::new(
            GreedyMemoryPool::new(limit_mb * 1024 * 1024),
            std::num::NonZeroUsize::new(5).expect("non-zero"),
        )),
    };
    let pool = Arc::new(PeakTrackingPool::new(inner));
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
        ctx.register_catalog(&name, Arc::new(provider));
    }
    if let Some(unmatched) = scan_files.keys().next() {
        return Err(PyValueError::new_err(format!(
            "scan_files references catalog `{unmatched}` which is not in `catalogs`"
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

/// The env-derived execution options both merge doorways share. `deadline`
/// is the one-shot call's static bound (merge_into); the window doorway
/// leaves it None and carries per-slice deadlines on the window state.
fn mor_options_from_env(
    deadline: Option<std::time::Instant>,
    write_workers: Option<usize>,
    late_materialization: Option<bool>,
    window: Option<Arc<MorWindowState>>,
) -> MorMergeOptions {
    MorMergeOptions {
        deadline,
        write_workers,
        late_materialization: late_materialization.unwrap_or(true),
        // Byte target for write-path batches (consume chunks + the
        // late-fetch decode batch size). Env-tunable alongside the
        // pool knobs; unset = the engine default.
        chunk_target_bytes: std::env::var("MERGE_MOR_CHUNK_TARGET_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|mb| *mb > 0)
            .map(|mb| mb * 1024 * 1024),
        // Inline DV micro-reabsorb — env-gated, ships OFF. When
        // enabled, DV-target files whose dead fraction crossed the
        // threshold are rewritten as a second, optional commit after
        // the merge (skipped near the deadline / on conflict).
        reabsorb_dead_frac: if std::env::var("MERGE_INLINE_REABSORB_ENABLED")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
        {
            Some(
                std::env::var("MERGE_REABSORB_DEAD_FRAC")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|f| *f > 0.0 && *f < 1.0)
                    .unwrap_or(0.3),
            )
        } else {
            None
        },
        window,
    }
}

/// Await `fut`, raising when the merge deadline passes first. Only used for
/// phases that perform no writes (mounting, planning) — the execution phase
/// enforces the same deadline cooperatively inside the write node so the
/// commit is never cancelled mid-flight.
async fn doorway_deadline<T>(
    deadline: Option<std::time::Instant>,
    what: &str,
    fut: impl Future<Output = T>,
) -> PyResult<T> {
    match deadline {
        None => Ok(fut.await),
        Some(d) => tokio::time::timeout_at(tokio::time::Instant::from_std(d), fut)
            .await
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "merge timeout exceeded while {what} (no writes were performed)"
                ))
            }),
    }
}

/// Execute one `MERGE INTO` statement and block until its snapshot commits.
///
/// `catalogs` maps each SQL catalog name to Iceberg REST catalog properties
/// (`uri`, `warehouse`, `credential`, `oauth2-server-uri`, `scope`, ...);
/// `sql` is the full MERGE statement (DataFusion dialect). `scan_files`
/// optionally restricts named tables' scans to an externally planned set of
/// data-file paths: `{"catalog.namespace.table": ["s3://.../f.parquet", ...]}`
/// (deletes still apply to the retained files; unknown identifiers raise).
/// `timeout_s` bounds the merge: mounting, planning and the scan/write phase
/// abort once it elapses (cooperatively inside the write node — the snapshot
/// commit itself is never cancelled; a timed-out merge writes NO snapshot).
/// `write_workers` sizes the writer pool for appended output (default:
/// min(4, cores)). `late_materialization` (default True) keeps the
/// matched-row fetch row-group-ranged — only the row groups containing
/// matched positions are decoded, and prior deletion-vector state is loaded
/// from the delete files; pass False as the kill switch restoring the
/// whole-file fetch (both produce identical commits). Returns a dict with
/// `count` — the number of rows appended by the merge (inserts plus updated
/// row versions) — and `peak_mem_bytes`, the DataFusion memory pool's
/// high-water mark for this call (pool-registered operators: joins, sorts,
/// aggregates). Raises `ValueError` on planning or execution failure, and
/// on deadline expiry.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, timeout_s=None, write_workers=None, scoped_tables=None, late_materialization=None, local_tables=None))]
fn merge_into(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    timeout_s: Option<u64>,
    write_workers: Option<usize>,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    late_materialization: Option<bool>,
    local_tables: Option<HashMap<String, String>>,
) -> PyResult<HashMap<String, String>> {
    let scan_files = parse_scan_files(scan_files)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    // The deadline covers catalog mounting, planning and the scan/write
    // phase (enforced cooperatively inside the merge write node). The
    // snapshot COMMIT is deliberately outside it — cancelling a REST commit
    // in flight leaves the outcome unknown; once the write phase finishes
    // under deadline, the commit runs to completion.
    let deadline = timeout_s.map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
    py.detach(|| {
        runtime().block_on(async move {
            let options = Arc::new(mor_options_from_env(
                deadline,
                write_workers,
                late_materialization,
                None,
            ));
            // Phase timing (merge_into_window W1): mount / plan / exec at the
            // doorway, write (scan+join+write drain) + commit from the commit
            // exec's result columns. exec_ms − write_ms − commit_ms −
            // reabsorb_ms ≈ stream/dispatch overhead. The worker's per-slice
            // "-> {out}" log line surfaces all of it with zero plumbing.
            let mount_t0 = std::time::Instant::now();
            let (ctx, pool) = doorway_deadline(
                deadline,
                "mounting catalogs",
                session_with_catalogs(
                    catalogs,
                    scan_files,
                    scoped_tables,
                    local_tables,
                    Some(options),
                ),
            )
            .await??;
            let mount_ms = mount_t0.elapsed().as_millis() as u64;
            let plan_t0 = std::time::Instant::now();
            let df = doorway_deadline(deadline, "planning MERGE", ctx.sql(&sql))
                .await?
                .map_err(|e| PyValueError::new_err(format!("planning MERGE: {e}")))?;
            let plan_ms = plan_t0.elapsed().as_millis() as u64;
            let exec_t0 = std::time::Instant::now();
            let batches = df
                .collect()
                .await
                .map_err(|e| PyValueError::new_err(format!("executing MERGE: {e}")))?;
            let exec_ms = exec_t0.elapsed().as_millis() as u64;
            let mut count: u64 = 0;
            let mut reabsorbed_files: u64 = 0;
            let mut reabsorb_ms: u64 = 0;
            let mut reabsorb_skipped: Option<String> = None;
            let mut write_ms: u64 = 0;
            let mut commit_ms: u64 = 0;
            let mut scan_ms: u64 = 0;
            let mut join_ms: u64 = 0;
            let mut write_node_ms: u64 = 0;
            for batch in &batches {
                if let Some(col) = batch.column_by_name("count")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    count += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("reabsorbed_files")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    reabsorbed_files += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("reabsorb_ms")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    reabsorb_ms += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("reabsorb_skipped")
                    && let Some(arr) = col.as_any().downcast_ref::<StringArray>()
                    && arr.len() == 1
                    && arr.is_valid(0)
                {
                    reabsorb_skipped = Some(arr.value(0).to_string());
                }
                if let Some(col) = batch.column_by_name("write_ms")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    write_ms += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("commit_ms")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    commit_ms += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("scan_ms")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    scan_ms += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("join_ms")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    join_ms += arr.iter().flatten().sum::<u64>();
                }
                if let Some(col) = batch.column_by_name("write_node_ms")
                    && let Some(arr) = col.as_any().downcast_ref::<UInt64Array>()
                {
                    write_node_ms += arr.iter().flatten().sum::<u64>();
                }
            }
            let mut out = HashMap::from([("count".to_string(), count.to_string())]);
            out.insert("mount_ms".to_string(), mount_ms.to_string());
            out.insert("plan_ms".to_string(), plan_ms.to_string());
            out.insert("exec_ms".to_string(), exec_ms.to_string());
            out.insert("write_ms".to_string(), write_ms.to_string());
            out.insert("commit_ms".to_string(), commit_ms.to_string());
            // write_ms sub-split (post-drain metrics walk over the plan) —
            // BUSY time per operator (elapsed_compute), NOT wall: pipelined
            // operators overlap so scan+join+write_node ≠ write_ms; join_ms
            // includes the build-side (source) collect per DataFusion's
            // accounting; write_node_ms excludes upstream-input wait but
            // includes late-fetch I/O + writer-pool joins.
            out.insert("scan_ms".to_string(), scan_ms.to_string());
            out.insert("join_ms".to_string(), join_ms.to_string());
            out.insert("write_node_ms".to_string(), write_node_ms.to_string());
            // Inline-reabsorb telemetry (constraint 5 of the handoff): the
            // worker's per-slice "-> {out}" log line surfaces these with
            // zero plumbing; absent keys = feature disabled.
            if reabsorbed_files > 0 || reabsorb_ms > 0 || reabsorb_skipped.is_some() {
                out.insert("reabsorbed_files".to_string(), reabsorbed_files.to_string());
                out.insert("reabsorb_ms".to_string(), reabsorb_ms.to_string());
            }
            if let Some(reason) = reabsorb_skipped {
                out.insert("reabsorb_skipped".to_string(), reason);
            }
            // Pool-visible peak memory for this merge — the governor's
            // per-slice pressure signal. The worker's existing "-> {out}"
            // slice log prints it with zero new plumbing.
            out.insert("peak_mem_bytes".to_string(), pool.peak().to_string());
            // Cumulative per-process cache stats (manifest + data tiers) —
            // the worker logs this result line per slice, so cache
            // effectiveness lands in run-pod logs with zero new plumbing.
            if let Some(stats) = crate::runtime::cache_stats_json() {
                out.insert("cache_stats".to_string(), stats);
            }
            Ok(out)
        })
    })
}

/// Parse the window's slice list: `[{"sql": ..., "scan_files": {...}}, ...]`
/// — both keys optional (`sql` defaults to the window template), unknown
/// keys refuse loudly.
fn parse_window_slices(
    py: Python<'_>,
    slices: Vec<HashMap<String, Py<PyAny>>>,
) -> PyResult<Vec<WindowSliceSpec>> {
    slices
        .into_iter()
        .enumerate()
        .map(|(i, m)| {
            let mut spec = WindowSliceSpec::default();
            for (k, v) in m {
                if v.is_none(py) {
                    continue;
                }
                match k.as_str() {
                    "sql" => spec.sql = Some(v.extract::<String>(py)?),
                    "scan_files" => {
                        let sf = v.extract::<HashMap<String, Vec<String>>>(py)?;
                        for fqn in sf.keys() {
                            if fqn.splitn(3, '.').count() != 3 {
                                return Err(PyValueError::new_err(format!(
                                    "slice {i}: scan_files key `{fqn}` must be \
                                     `catalog.namespace.table`"
                                )));
                            }
                        }
                        spec.scan_files = Some(sf);
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "slice {i}: unknown key `{other}` (expected sql / scan_files)"
                        )));
                    }
                }
            }
            Ok(spec)
        })
        .collect()
}

/// Execute a WINDOW of `MERGE INTO` slices over ONE mounted session with a
/// HELD target current-set — the one-shot windowed twin of [`merge_into`].
///
/// `sql` is the window's MERGE template; each slice in `slices` may carry a
/// full-statement `sql` override (per-slice bounds literals / clause
/// toggles) and a per-slice `scan_files` allowlist, applied before the
/// slice plans. All slices must target the same table. The engine mounts
/// once, executes the slices in order, and holds the target current-set
/// across them (slice 1 tees its target scan; later slices serve the probe
/// AND the demote-union self-read from the held set, maintained per commit
/// as `held − victims + appends`). A 1-slice window routes through the same
/// path with the hold OFF — it costs exactly a `merge_into` call.
///
/// Commit cadence: one RowDelta per slice (`commit_every=1`, the default —
/// the poison-checkpoint contract: a failing slice stops the loop, returns
/// `failed_slice` + `failed_error`, and completed slices' commits stand).
/// `commit_every=N` folds N slices into one RowDelta (defer mode, holds
/// OFF), flushing early + re-executing the slice when its deletion vectors
/// touch a file the pending fold already carries. `timeout_s` bounds EACH
/// slice; `budget_s` bounds the window wall — when it trips, the loop stops
/// cleanly after the current slice (`budget_stopped=true`, partial results).
/// `held_extra_columns` widens the held projection so the USING-side
/// self-read is coverable (pass the demote-union's t2 columns);
/// `held_max_mb` caps the held bytes (env `MERGE_WINDOW_HELD_MAX_MB`,
/// default 2048; overflow falls back to per-slice direct scans;
/// `MERGE_WINDOW_HELD_DISABLED=1` is the kill switch).
///
/// Returns a dict: `count` (total appended rows), `slices_total` /
/// `slices_done` (executed) / `slices_committed`, `mount_ms`,
/// `peak_mem_bytes`, held telemetry (`held_installed`, `held_rows`,
/// `held_bytes`, `target_scans_direct`, `target_scans_served`,
/// `provider_serves`, optional `held_dropped`), optional `failed_slice` /
/// `failed_error` / `budget_stopped`, and `slices` — a JSON list of
/// per-slice outcomes (count, phase timings incl. the busy-time sub-split,
/// committed, reexecuted). Slice failures are returned in-band (the
/// per-slice results ARE the checkpoint evidence), never raised; only
/// argument/mount failures raise `ValueError`.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, slices, timeout_s=None, budget_s=None, write_workers=None, scoped_tables=None, late_materialization=None, local_tables=None, commit_every=None, held_extra_columns=None, held_max_mb=None))]
#[allow(clippy::too_many_arguments)]
fn merge_into_window(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    slices: Vec<HashMap<String, Py<PyAny>>>,
    timeout_s: Option<u64>,
    budget_s: Option<u64>,
    write_workers: Option<usize>,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    late_materialization: Option<bool>,
    local_tables: Option<HashMap<String, String>>,
    commit_every: Option<usize>,
    held_extra_columns: Option<Vec<String>>,
    held_max_mb: Option<usize>,
) -> PyResult<HashMap<String, String>> {
    let specs = parse_window_slices(py, slices)?;
    if specs.is_empty() {
        return Err(PyValueError::new_err(
            "merge_into_window requires at least one slice",
        ));
    }
    let scan_files = parse_scan_files(None)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    let commit_every = commit_every.unwrap_or(1).max(1);
    let held_max_bytes = held_max_mb
        .or_else(|| {
            std::env::var("MERGE_WINDOW_HELD_MAX_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        })
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(HELD_MAX_BYTES_DEFAULT);
    let hold_disabled = std::env::var("MERGE_WINDOW_HELD_DISABLED")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    // The 1-slice hot path pays ZERO hold machinery — it must benchmark at
    // parity with merge_into.
    let hold = specs.len() > 1 && !hold_disabled;
    let window = Arc::new(MorWindowState::new(
        hold,
        held_extra_columns.unwrap_or_default(),
        held_max_bytes,
        commit_every,
    ));
    let options = Arc::new(mor_options_from_env(
        None,
        write_workers,
        late_materialization,
        Some(Arc::clone(&window)),
    ));
    py.detach(|| {
        runtime().block_on(async move {
            let mount_deadline =
                timeout_s.map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
            let mount_t0 = std::time::Instant::now();
            let (ctx, pool) = doorway_deadline(
                mount_deadline,
                "mounting catalogs",
                session_with_catalogs(
                    catalogs,
                    scan_files,
                    scoped_tables,
                    local_tables,
                    Some(options),
                ),
            )
            .await??;
            let mount_ms = mount_t0.elapsed().as_millis() as u64;
            let slice_timeout = timeout_s.map(std::time::Duration::from_secs);
            let budget_deadline =
                budget_s.map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
            let run = run_window(&ctx, &window, &sql, &specs, slice_timeout, budget_deadline).await;

            let mut out = HashMap::from([("count".to_string(), run.total_count().to_string())]);
            out.insert("mount_ms".to_string(), mount_ms.to_string());
            out.insert("slices_total".to_string(), specs.len().to_string());
            out.insert("slices_done".to_string(), run.outcomes.len().to_string());
            out.insert(
                "slices_committed".to_string(),
                run.slices_committed().to_string(),
            );
            if let Some(i) = run.failed_slice {
                out.insert("failed_slice".to_string(), i.to_string());
            }
            if let Some(e) = &run.failed_error {
                out.insert("failed_error".to_string(), e.clone());
            }
            if run.budget_stopped {
                out.insert("budget_stopped".to_string(), "true".to_string());
            }
            if let Some(r) = &run.held_dropped {
                out.insert("held_dropped".to_string(), r.clone());
            }
            let (held_installed, held_rows, held_bytes) = window.held_stats();
            out.insert("held_installed".to_string(), held_installed.to_string());
            out.insert("held_rows".to_string(), held_rows.to_string());
            out.insert("held_bytes".to_string(), held_bytes.to_string());
            let relaxed = Ordering::Relaxed;
            out.insert(
                "target_scans_direct".to_string(),
                window.target_scans_direct.load(relaxed).to_string(),
            );
            out.insert(
                "target_scans_served".to_string(),
                window.target_scans_served.load(relaxed).to_string(),
            );
            out.insert(
                "provider_serves".to_string(),
                window.provider_serves.load(relaxed).to_string(),
            );
            out.insert(
                "slices".to_string(),
                serde_json::to_string(&run.outcomes).map_err(|e| {
                    PyValueError::new_err(format!("serializing slice outcomes: {e}"))
                })?,
            );
            out.insert("peak_mem_bytes".to_string(), pool.peak().to_string());
            if let Some(stats) = crate::runtime::cache_stats_json() {
                out.insert("cache_stats".to_string(), stats);
            }
            Ok(out)
        })
    })
}

/// Plan one `MERGE INTO` statement WITHOUT executing it and return the
/// physical plan as text — the read-only twin of [`merge_into`]. Planning
/// resolves every table reference against the live catalogs (metadata reads
/// only, no data IO, no commit), so a failing or suspicious merge can be
/// inspected with zero writes.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, scoped_tables=None, local_tables=None))]
fn dry_run_inspect(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    local_tables: Option<HashMap<String, String>>,
) -> PyResult<HashMap<String, String>> {
    let scan_files = parse_scan_files(scan_files)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    py.detach(|| {
        runtime().block_on(async move {
            let (ctx, _pool) =
                session_with_catalogs(catalogs, scan_files, scoped_tables, local_tables, None)
                    .await?;
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| PyValueError::new_err(format!("planning MERGE: {e}")))?;
            let logical = df.logical_plan().display_indent().to_string();
            let physical = df
                .create_physical_plan()
                .await
                .map_err(|e| PyValueError::new_err(format!("physical planning MERGE: {e}")))?;
            let physical = displayable(physical.as_ref()).indent(true).to_string();
            Ok(HashMap::from([
                ("logical_plan".to_string(), logical),
                ("physical_plan".to_string(), physical),
            ]))
        })
    })
}

/// Run one READ-ONLY SQL statement and return its rows as a JSON array
/// string — the query twin of [`merge_into`], for harnesses, referees and
/// parity checks. DML/DDL/COPY plans are rejected before execution; the
/// result is capped at `max_rows` (error, not truncation — a capped result
/// silently read as complete is worse than a loud failure). `scan_files`
/// bounds named tables' scans exactly as in [`merge_into`]. Columns must be
/// JSON-serializable by the arrow JSON writer — wrap variant/binary columns
/// in `variant_to_json(...)` in the statement.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, max_rows=100_000, scoped_tables=None, local_tables=None))]
fn sql_collect(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    max_rows: usize,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    local_tables: Option<HashMap<String, String>>,
) -> PyResult<String> {
    use datafusion::logical_expr::LogicalPlan;

    let scan_files = parse_scan_files(scan_files)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    py.detach(|| {
        runtime().block_on(async move {
            let (ctx, _pool) =
                session_with_catalogs(catalogs, scan_files, scoped_tables, local_tables, None)
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
                    "sql_collect is read-only — use merge_into for writes",
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
#[pyo3(signature = (catalogs, sql, scan_files=None, max_rows=100_000, scoped_tables=None, local_tables=None))]
fn sql_collect_ipc(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    max_rows: usize,
    scoped_tables: Option<HashMap<String, Vec<String>>>,
    local_tables: Option<HashMap<String, String>>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    use datafusion::logical_expr::LogicalPlan;

    let scan_files = parse_scan_files(scan_files)?;
    let scoped_tables = parse_scoped_tables(scoped_tables)?;
    let local_tables = local_tables.unwrap_or_default();
    let buf: Vec<u8> = py.detach(|| {
        runtime().block_on(async move {
            let (ctx, _pool) =
                session_with_catalogs(catalogs, scan_files, scoped_tables, local_tables, None)
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
                    "sql_collect_ipc is read-only — use merge_into for writes",
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
    this.add_function(wrap_pyfunction!(merge_into, &this)?)?;
    this.add_function(wrap_pyfunction!(merge_into_window, &this)?)?;
    this.add_function(wrap_pyfunction!(dry_run_inspect, &this)?)?;
    this.add_function(wrap_pyfunction!(sql_collect, &this)?)?;
    this.add_function(wrap_pyfunction!(sql_collect_ipc, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
