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

use std::sync::{Arc, OnceLock};

use iceberg::cache::{DataBytesCache, ObjectBytesCacheRef};
use iceberg_cache_foyer::{EvictionPolicy, FoyerObjectBytesCacheBuilder};
use tokio::runtime::{Handle, Runtime};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Process-global manifest / manifest-list bytes cache (foyer), shared by
/// every catalog this binding builds. The store is PATH-KEYED over
/// immutable objects, so sharing across calls (and across catalogs) is
/// correct by construction: a fresh Table still resolves its CURRENT
/// metadata per call (freshness), while re-reads of unchanged manifests hit
/// warm cache instead of re-fetching per call.
///
/// Knobs (read once, at first use):
/// - `ICEBERG_OBJECT_CACHE_MB` — in-memory capacity in MiB (default 128).
/// - `ICEBERG_OBJECT_CACHE_DIR` — when set, enables foyer's DISK tier at
///   this directory (e.g. `/var/cache/pulse-iceberg`). Unset = memory-only
///   (the default; disk enablement is deliberately opt-in).
/// - `ICEBERG_OBJECT_CACHE_DISK_MB` — disk-tier capacity in MiB
///   (default 1024; only meaningful with the DIR set).
static OBJECT_CACHE: tokio::sync::OnceCell<ObjectBytesCacheRef> =
    tokio::sync::OnceCell::const_new();

fn env_mb(name: &str, default_mb: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default_mb)
}

/// The disk-cache root, when configured. NOTE: foyer's disk engine takes NO
/// cross-process lock — the directory must be PRIVATE to this process (in
/// K8s: a per-pod path, e.g. hostPath + subPathExpr on the pod name), never
/// shared between pods.
fn cache_dir() -> Option<std::path::PathBuf> {
    std::env::var("ICEBERG_OBJECT_CACHE_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(std::path::PathBuf::from)
}

pub async fn global_object_cache() -> ObjectBytesCacheRef {
    OBJECT_CACHE
        .get_or_init(|| async {
            let memory_bytes = env_mb("ICEBERG_OBJECT_CACHE_MB", 128) * 1024 * 1024;
            let mut builder = FoyerObjectBytesCacheBuilder::new(memory_bytes as usize);
            if let Some(dir) = cache_dir() {
                let disk_bytes = env_mb("ICEBERG_OBJECT_CACHE_DISK_MB", 1024) * 1024 * 1024;
                // Own subdirectory: the DATA cache (below) shares the same
                // root, and two foyer devices must never share one dir.
                builder = builder.with_disk(dir.join("manifest"), disk_bytes as usize);
            }
            match builder.build().await {
                Ok(cache) => Arc::new(cache) as ObjectBytesCacheRef,
                Err(e) => {
                    // The cache must never block the platform: fall back to
                    // a memory-only store (e.g. when the disk dir is bad).
                    eprintln!("iceberg object cache: falling back to memory-only ({e})");
                    let fallback = FoyerObjectBytesCacheBuilder::new(
                        (env_mb("ICEBERG_OBJECT_CACHE_MB", 128) * 1024 * 1024) as usize,
                    )
                    .build()
                    .await
                    .expect("memory-only foyer cache must build");
                    Arc::new(fallback) as ObjectBytesCacheRef
                }
            }
        })
        .await
        .clone()
}

pub fn runtime() -> Handle {
    match Handle::try_current() {
        Ok(h) => h.clone(),
        _ => {
            let rt = RUNTIME.get_or_init(|| Runtime::new().unwrap());
            rt.handle().clone()
        }
    }
}

/// Process-global WHOLE-FILE data cache (foyer disk tier), shared by every
/// catalog this binding builds. DISABLED by default — turns on only when
/// BOTH knobs are set:
/// - `ICEBERG_OBJECT_CACHE_DIR` — the (process-private) disk-cache root;
///   the data tier lives under `<dir>/data`.
/// - `ICEBERG_DATA_CACHE_MB` — disk budget in MiB (default 0 = OFF).
///
/// With it on, the first read of a data file fetches the WHOLE object once
/// (one GET replaces the scan's N ranged GETs) and every ranged read —
/// parquet footer, column chunks, row-group byte ranges — is served from
/// the node-local copy; files are immutable so there is no invalidation.
/// `ICEBERG_CACHE_MAX_FILE_MB` (default 256) caps the per-file size: larger
/// files bypass the cache (and set the disk engine's block size, which an
/// entry must fit in). When OFF, the read path is byte-identical to today.
static DATA_CACHE: tokio::sync::OnceCell<Option<DataBytesCache>> =
    tokio::sync::OnceCell::const_new();

pub async fn global_data_cache() -> Option<DataBytesCache> {
    DATA_CACHE
        .get_or_init(|| async {
            let dir = cache_dir()?;
            let disk_mb = env_mb("ICEBERG_DATA_CACHE_MB", 0);
            if disk_mb == 0 {
                return None;
            }
            let max_file_bytes = env_mb("ICEBERG_CACHE_MAX_FILE_MB", 256) * 1024 * 1024;
            // Small memory tier (the disk tier is the store); block size
            // sized above the per-file cap so capped files are admitted.
            // Eviction: the merge access pattern is a CYCLIC whole-set scan
            // per slice — plain LRU (foyer's default) collapses to ~0% hits
            // once the working set exceeds capacity, so the data tier
            // defaults to scan-resistant S3-FIFO
            // (`ICEBERG_DATA_CACHE_POLICY`: s3fifo | lfu | lru | fifo).
            let policy = match std::env::var("ICEBERG_DATA_CACHE_POLICY")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str()
            {
                "lru" => EvictionPolicy::Lru,
                "lfu" => EvictionPolicy::Lfu,
                "fifo" => EvictionPolicy::Fifo,
                _ => EvictionPolicy::S3Fifo,
            };
            // Memory-tier budget: ICEBERG_DATA_CACHE_MEM_MB (default 64).
            // Fat dedicated pods (whole-file VARIANT slices) size it up so
            // the working set stops thrashing the disk tier (~61k misses +
            // 1.4TB disk churn per slice measured on the queries catch-up at
            // the 64MB default); the fleet default stays 64.
            let mem_bytes = env_mb("ICEBERG_DATA_CACHE_MEM_MB", 64) * 1024 * 1024;
            let builder = FoyerObjectBytesCacheBuilder::new(mem_bytes as usize)
                .with_disk(dir.join("data"), (disk_mb * 1024 * 1024) as usize)
                .with_disk_block_bytes((max_file_bytes + 16 * 1024 * 1024) as usize)
                .with_eviction_policy(policy);
            match builder.build().await {
                Ok(cache) => Some(DataBytesCache {
                    cache: Arc::new(cache) as ObjectBytesCacheRef,
                    max_file_bytes,
                }),
                Err(e) => {
                    // The cache must never block the platform.
                    eprintln!("iceberg data cache: disabled ({e})");
                    None
                }
            }
        })
        .await
        .clone()
}

/// Compact JSON of both cache tiers' cumulative stats (counters grow for
/// the process lifetime; diff across observations). Only initialized tiers
/// appear; returns None when no cache has been touched yet.
///
/// Shape (one line, log-friendly):
/// `{"manifest":{"hits":..,"misses":..,"inserts":..,"evictions":..,
///   "mem_bytes":..,"mem_cap_bytes":..,"disk_write_bytes":..,
///   "disk_read_bytes":..},"data":{...}}`
pub fn cache_stats_json() -> Option<String> {
    fn tier(stats: iceberg::cache::ObjectCacheStats) -> String {
        format!(
            "{{\"hits\":{},\"misses\":{},\"inserts\":{},\"evictions\":{},\"mem_bytes\":{},\"mem_cap_bytes\":{},\"disk_write_bytes\":{},\"disk_read_bytes\":{}}}",
            stats.hits,
            stats.misses,
            stats.inserts,
            stats.evictions,
            stats.memory_usage_bytes,
            stats.memory_capacity_bytes,
            stats.disk_write_bytes,
            stats.disk_read_bytes,
        )
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(cache) = OBJECT_CACHE.get()
        && let Some(stats) = cache.stats()
    {
        parts.push(format!("\"manifest\":{}", tier(stats)));
    }
    if let Some(Some(dc)) = DATA_CACHE.get()
        && let Some(stats) = dc.cache.stats()
    {
        parts.push(format!("\"data\":{}", tier(stats)));
    }
    if parts.is_empty() {
        None
    } else {
        Some(format!("{{{}}}", parts.join(",")))
    }
}

/// One-line per-process totals, registered with Python's `atexit` at module
/// init so every run pod logs its final cache effectiveness.
pub fn log_cache_stats_at_exit() {
    if let Some(stats) = cache_stats_json() {
        eprintln!("INFO iceberg object-cache stats (process totals): {stats}");
    }
}
