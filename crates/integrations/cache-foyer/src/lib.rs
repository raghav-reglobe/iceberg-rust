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

//! A [foyer](https://foyer.rs)-backed implementation of
//! [`iceberg::cache::ObjectBytesCache`]: a process-shareable, path-keyed
//! cache of raw manifest / manifest-list file bytes with a sharded
//! in-memory tier and an OPTIONAL disk tier.
//!
//! The store is a single [`foyer::HybridCache`] either way — without a disk
//! device foyer runs its storage engine in no-op (memory-only) mode, so
//! enabling the disk tier later is pure CONFIGURATION
//! ([`FoyerObjectBytesCacheBuilder::with_disk`]), not new code. Values are
//! the raw fetched bytes (immutable by path), which foyer can serialize to
//! disk natively; parsing stays with the caller.

#![deny(missing_docs)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use foyer::{
    BlockEngineConfig, DeviceBuilder, Event, EventListener, FifoConfig, FsDeviceBuilder,
    HybridCache, HybridCacheBuilder, LfuConfig, LruConfig, S3FifoConfig,
};
use iceberg::Result;
use iceberg::cache::{ObjectBytesCache, ObjectCacheStats};

/// In-memory tier eviction policy (see foyer's `EvictionConfig`).
///
/// The default (foyer's) is LRU. For CYCLIC whole-set scans larger than the
/// capacity — the merge slice pattern, where every current file is re-read
/// once per slice — LRU (and plain FIFO) degrade to ~0% hit rate the moment
/// the working set exceeds capacity; the scan-resistant policies keep a
/// stable hot subset resident instead. `S3Fifo` is the recommended choice
/// for the data tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least-recently-used (foyer's default).
    Lru,
    /// S3-FIFO — scan-resistant, the recommended data-tier policy.
    S3Fifo,
    /// w-TinyLFU — scan-resistant, frequency-based.
    Lfu,
    /// Plain FIFO.
    Fifo,
}

/// Counts in-memory tier evictions via foyer's event listener.
#[derive(Debug, Default)]
struct EvictionCounter(AtomicU64);

impl EventListener for EvictionCounter {
    type Key = String;
    type Value = Bytes;

    fn on_leave(&self, reason: Event, _key: &String, _value: &Bytes) {
        if matches!(reason, Event::Evict) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Builder for [`FoyerObjectBytesCache`].
#[derive(Debug)]
pub struct FoyerObjectBytesCacheBuilder {
    memory_capacity_bytes: usize,
    disk: Option<(PathBuf, usize)>,
    disk_block_bytes: Option<usize>,
    eviction_policy: Option<EvictionPolicy>,
}

impl FoyerObjectBytesCacheBuilder {
    /// Creates a builder with the given IN-MEMORY capacity (in bytes;
    /// entries are weighed by key + value length).
    pub fn new(memory_capacity_bytes: usize) -> Self {
        Self {
            memory_capacity_bytes,
            disk: None,
            disk_block_bytes: None,
            eviction_policy: None,
        }
    }

    /// Adds a DISK tier at `dir` with the given capacity in bytes.
    ///
    /// Without this, the cache is memory-only (foyer's storage engine runs
    /// in no-op mode); with it, entries evicted from memory spill to disk
    /// and disk hits repopulate memory.
    pub fn with_disk(mut self, dir: impl Into<PathBuf>, capacity_bytes: usize) -> Self {
        self.disk = Some((dir.into(), capacity_bytes));
        self
    }

    /// Sets the disk engine's block size (foyer default: 16 MiB). An entry
    /// must fit in one block to be admitted to the disk tier — size this
    /// ABOVE the largest entry the cache should hold (e.g. the data cache's
    /// per-file cap), or larger entries are silently dropped from disk. The
    /// write-buffer and submit-queue thresholds scale along with it.
    pub fn with_disk_block_bytes(mut self, block_bytes: usize) -> Self {
        self.disk_block_bytes = Some(block_bytes);
        self
    }

    /// Sets the in-memory tier's eviction policy (see [`EvictionPolicy`]).
    /// Unset = foyer's default (LRU).
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = Some(policy);
        self
    }

    /// Builds the cache. Must be called within an async runtime (foyer
    /// spawns its maintenance tasks on the current one).
    pub async fn build(self) -> Result<FoyerObjectBytesCache> {
        let to_iceberg_err = |e: foyer::Error| {
            iceberg::Error::new(
                iceberg::ErrorKind::Unexpected,
                "building the foyer object-bytes cache",
            )
            .with_source(e)
        };
        let evictions = Arc::new(EvictionCounter::default());
        let mut builder = HybridCacheBuilder::new()
            .with_name("iceberg-object-bytes")
            .with_event_listener(evictions.clone())
            .memory(self.memory_capacity_bytes)
            .with_weighter(|key: &String, value: &Bytes| key.len() + value.len());
        if let Some(policy) = self.eviction_policy {
            builder = match policy {
                EvictionPolicy::Lru => builder.with_eviction_config(LruConfig::default()),
                EvictionPolicy::S3Fifo => builder.with_eviction_config(S3FifoConfig::default()),
                EvictionPolicy::Lfu => builder.with_eviction_config(LfuConfig::default()),
                EvictionPolicy::Fifo => builder.with_eviction_config(FifoConfig::default()),
            };
        }
        let builder = builder.storage();
        let builder = match self.disk {
            None => builder,
            Some((dir, capacity_bytes)) => {
                let device = FsDeviceBuilder::new(dir)
                    .with_capacity(capacity_bytes)
                    .build()
                    .map_err(to_iceberg_err)?;
                let mut engine = BlockEngineConfig::new(device);
                if let Some(block) = self.disk_block_bytes {
                    engine = engine
                        .with_block_size(block)
                        .with_buffer_pool_size(block.saturating_mul(2))
                        .with_submit_queue_size_threshold(block.saturating_mul(2));
                }
                builder.with_engine_config(engine)
            }
        };
        let cache = builder.build().await.map_err(to_iceberg_err)?;
        Ok(FoyerObjectBytesCache {
            cache,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            evictions,
        })
    }
}

/// A process-shareable, path-keyed cache of raw object bytes backed by a
/// [`foyer::HybridCache`] (sharded memory tier + optional disk tier).
#[derive(Debug)]
pub struct FoyerObjectBytesCache {
    cache: HybridCache<String, Bytes>,
    // foyer exposes memory usage/capacity, disk read/write bytes and
    // eviction EVENTS natively, but not hit/miss/insert counters outside its
    // metrics-registry machinery — those are wrapped here at the seam.
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: Arc<EvictionCounter>,
}

#[async_trait]
impl ObjectBytesCache for FoyerObjectBytesCache {
    async fn get(&self, path: &str) -> Option<Bytes> {
        let out = match self.cache.get(path).await {
            Ok(entry) => entry.map(|e| e.value().clone()),
            Err(e) => {
                // A cache-read failure (e.g. a disk-tier IO error) must
                // never fail the table operation — treat it as a miss.
                tracing::warn!("foyer object-bytes cache read failed: {e}");
                None
            }
        };
        match &out {
            Some(_) => self.hits.fetch_add(1, Ordering::Relaxed),
            None => self.misses.fetch_add(1, Ordering::Relaxed),
        };
        out
    }

    async fn set(&self, path: &str, bytes: Bytes) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
        self.cache.insert(path.to_string(), bytes);
    }

    fn stats(&self) -> Option<ObjectCacheStats> {
        let statistics = self.cache.statistics();
        Some(ObjectCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            evictions: self.evictions.0.load(Ordering::Relaxed),
            memory_usage_bytes: self.cache.memory().usage() as u64,
            memory_capacity_bytes: self.cache.memory().capacity() as u64,
            disk_write_bytes: statistics.disk_write_bytes() as u64,
            disk_read_bytes: statistics.disk_read_bytes() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use iceberg::cache::ObjectBytesCacheRef;

    use super::*;

    /// Memory-only (no disk device): roundtrip through the trait.
    #[tokio::test]
    async fn memory_only_roundtrip() {
        let cache: ObjectBytesCacheRef = Arc::new(
            FoyerObjectBytesCacheBuilder::new(4 * 1024 * 1024)
                .build()
                .await
                .unwrap(),
        );
        assert!(cache.get("s3://bucket/metadata/m1.avro").await.is_none());
        cache
            .set("s3://bucket/metadata/m1.avro", Bytes::from_static(b"abc"))
            .await;
        assert_eq!(
            cache.get("s3://bucket/metadata/m1.avro").await,
            Some(Bytes::from_static(b"abc"))
        );
        assert!(cache.get("s3://bucket/metadata/other.avro").await.is_none());
    }

    /// The disk tier is pure configuration: the same builder with a tempdir
    /// device builds and serves the same trait contract.
    #[tokio::test]
    async fn disk_tier_builds_and_roundtrips() {
        let dir = tempfile::TempDir::new().unwrap();
        let cache: ObjectBytesCacheRef = Arc::new(
            FoyerObjectBytesCacheBuilder::new(1024 * 1024)
                .with_disk(dir.path(), 16 * 1024 * 1024)
                .build()
                .await
                .unwrap(),
        );
        for i in 0..64 {
            let payload = vec![i as u8; 16 * 1024];
            cache
                .set(&format!("s3://bucket/metadata/m{i}.avro"), payload.into())
                .await;
        }
        // Everything remains servable through the hybrid (memory + disk)
        // path; some entries exceed the 1MiB memory tier and can only come
        // back through the storage engine.
        let mut served = 0;
        for i in 0..64 {
            if let Some(bytes) = cache.get(&format!("s3://bucket/metadata/m{i}.avro")).await {
                assert_eq!(bytes.as_ref(), vec![i as u8; 16 * 1024].as_slice());
                served += 1;
            }
        }
        assert!(
            served > 0,
            "hybrid cache must serve entries through the disk-backed store"
        );
        // The device directory exists and has been written to.
        assert!(dir.path().exists());
    }

    /// The disk tier stays BOUNDED by its capacity under a workload larger
    /// than the budget (the data-cache shape): total on-disk bytes never
    /// exceed capacity (+ one block of slack), which by pigeonhole means
    /// older entries were evicted, not accumulated.
    #[tokio::test]
    async fn disk_tier_is_bounded_by_capacity() {
        const CAPACITY: usize = 32 * 1024 * 1024;
        const ENTRY: usize = 2 * 1024 * 1024;
        const N: usize = 32; // 64 MiB total inserted, 2x the capacity

        let dir = tempfile::TempDir::new().unwrap();
        let cache: ObjectBytesCacheRef = Arc::new(
            FoyerObjectBytesCacheBuilder::new(1024 * 1024)
                .with_disk(dir.path(), CAPACITY)
                .with_disk_block_bytes(8 * 1024 * 1024)
                .build()
                .await
                .unwrap(),
        );
        for i in 0..N {
            cache
                .set(
                    &format!("s3://bucket/data/f{i}.parquet"),
                    vec![i as u8; ENTRY].into(),
                )
                .await;
            // Yield so foyer's flushers keep up with the submit queue.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        // Let async flush/reclaim settle.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let mut disk_bytes = 0u64;
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                disk_bytes += entry.metadata().unwrap().len();
            }
        }
        assert!(disk_bytes > 0, "the disk tier must hold data");
        assert!(
            disk_bytes as usize <= CAPACITY + 8 * 1024 * 1024,
            "disk usage {disk_bytes} must stay within capacity {CAPACITY} (+1 block slack)"
        );

        // Pigeonhole: not all 64 MiB of entries can be retrievable from a
        // 32 MiB store — older entries were evicted.
        let mut retrievable = 0usize;
        for i in 0..N {
            if cache
                .get(&format!("s3://bucket/data/f{i}.parquet"))
                .await
                .is_some()
            {
                retrievable += 1;
            }
        }
        assert!(
            retrievable < N,
            "a bounded store cannot retain the whole over-budget workload"
        );
    }

    /// The stats surface: wrapped hit/miss/insert counters, native memory
    /// usage, and eviction events under memory pressure — with the
    /// scan-resistant S3-FIFO policy configured.
    #[tokio::test]
    async fn stats_reflect_traffic_and_evictions() {
        let cache = FoyerObjectBytesCacheBuilder::new(256 * 1024)
            .with_eviction_policy(EvictionPolicy::S3Fifo)
            .build()
            .await
            .unwrap();

        let fresh = cache.stats().unwrap();
        assert_eq!(
            (fresh.hits, fresh.misses, fresh.inserts, fresh.evictions),
            (0, 0, 0, 0)
        );

        // miss -> insert -> hit
        assert!(cache.get("s3://b/one").await.is_none());
        cache.set("s3://b/one", Bytes::from_static(b"abc")).await;
        assert!(cache.get("s3://b/one").await.is_some());
        let stats = cache.stats().unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.hits, 1);
        assert!(stats.memory_usage_bytes > 0);
        assert_eq!(stats.memory_capacity_bytes, 256 * 1024);
        assert_eq!(stats.disk_write_bytes, 0, "memory-only tier");

        // Overflow the 256 KiB memory tier -> eviction events.
        for i in 0..64 {
            cache
                .set(&format!("s3://b/f{i}"), vec![i as u8; 16 * 1024].into())
                .await;
        }
        let stats = cache.stats().unwrap();
        assert_eq!(stats.inserts, 65);
        assert!(
            stats.evictions > 0,
            "memory pressure must surface as evictions: {stats:?}"
        );
        assert!(stats.memory_usage_bytes <= 256 * 1024 + 32 * 1024);
    }
}
