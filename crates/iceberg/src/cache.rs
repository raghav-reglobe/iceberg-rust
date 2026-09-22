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

//! Cache management for Iceberg.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use crate::spec::{Manifest, ManifestList};

/// A trait for caching in-memory objects of given type.
///
/// # Notes
///
/// ObjectCache will store deeply nested objects, such as `Manifest`,
/// which contains `Schema`. Please ensure that the cache stores the
/// object in memory as-is, without attempting to serialize it, as
/// serialization could be extremely expensive.
pub trait ObjectCache<K, V>: Send + Sync {
    /// Gets an object from the cache by its key.
    fn get(&self, key: &K) -> Option<V>;
    /// Sets an object in the cache with the given key and value.
    fn set(&self, key: K, value: V);
}

/// A trait for caching different in-memory objects used by iceberg.
///
/// # Notes
///
/// ObjectCache will store deeply nested objects, such as `Manifest`,
/// which contains `Schema`. Please ensure that the cache stores the
/// object in memory as-is, without attempting to serialize it, as
/// serialization could be extremely expensive.
pub trait ObjectCacheProvide: Send + Sync {
    /// Gets a cache for manifests.
    fn manifest_cache(&self) -> &dyn ObjectCache<String, Arc<Manifest>>;
    /// Gets a cache for manifest lists.
    fn manifest_list_cache(&self) -> &dyn ObjectCache<String, Arc<ManifestList>>;
}

/// CacheProvider is a type alias for a thread-safe reference-counted pointer to a CacheProvide trait object.
pub type ObjectCacheProvider = Arc<dyn ObjectCacheProvide>;

/// A process-shareable, path-keyed cache of RAW manifest / manifest-list
/// file bytes.
///
/// Manifest and manifest-list files are IMMUTABLE once written, so one
/// store may be shared across every [`crate::table::Table`] a process
/// builds (and across catalogs): repeat loads of the same table hit warm
/// cache instead of re-fetching from object storage, while table METADATA
/// stays fresh per load — a new snapshot carries a new manifest-list path,
/// which is simply a new key. Fetches on miss always go through the
/// *calling* table's `FileIO` (different tables/catalogs may carry
/// different credentials); parsing happens per call, so cached bytes are
/// context-free and the store may be backed by memory, disk, or both.
///
/// Async so implementations may serve from a disk tier.
#[async_trait]
pub trait ObjectBytesCache: Send + Sync + std::fmt::Debug {
    /// Gets the raw file bytes cached for `path`, if present.
    async fn get(&self, path: &str) -> Option<Bytes>;
    /// [`Self::get`] with the object's known length when the caller has it
    /// (manifest-recorded data-file / manifest sizes). Local stores don't
    /// need it (default: ignored); a remote tier can use it to reserve an
    /// exact fetch budget instead of a worst-case one. Pass the TRUE size —
    /// never clamp — and `None` when unknown.
    async fn get_with_size_hint(&self, path: &str, _size_hint: Option<u64>) -> Option<Bytes> {
        self.get(path).await
    }
    /// Caches the raw file bytes fetched from `path`.
    async fn set(&self, path: &str, bytes: Bytes);
    /// Cumulative statistics for this store, when the implementation tracks
    /// them (see [`ObjectCacheStats`]). Default: none.
    fn stats(&self) -> Option<ObjectCacheStats> {
        None
    }
}

/// Cumulative, process-lifetime statistics of an [`ObjectBytesCache`].
/// Counters only ever grow (consumers diff across observations); usage
/// gauges are point-in-time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectCacheStats {
    /// `get` calls served from the store.
    pub hits: u64,
    /// `get` calls that missed (the caller then fetched from storage).
    pub misses: u64,
    /// `set` calls (fetched objects entering the store).
    pub inserts: u64,
    /// Entries evicted from the in-memory tier (capacity pressure).
    pub evictions: u64,
    /// Current in-memory tier usage in (weighted) bytes.
    pub memory_usage_bytes: u64,
    /// In-memory tier capacity in (weighted) bytes.
    pub memory_capacity_bytes: u64,
    /// Cumulative bytes written to the disk tier (0 when memory-only).
    pub disk_write_bytes: u64,
    /// Cumulative bytes read from the disk tier (0 when memory-only).
    pub disk_read_bytes: u64,
}

/// A thread-safe, reference-counted handle to an [`ObjectBytesCache`].
pub type ObjectBytesCacheRef = Arc<dyn ObjectBytesCache>;

/// A WHOLE-FILE read-through cache for immutable DATA files (parquet),
/// path-keyed like [`ObjectBytesCache`] but sized for data: on the first
/// read of a file at most `max_file_bytes` large, the ENTIRE object is
/// fetched once and cached; every ranged read (parquet footer, column
/// chunks, row-group byte ranges) is then served from the local copy — one
/// object-store GET replaces the N ranged GETs of a parquet scan, first
/// touch included. Files larger than `max_file_bytes` bypass the cache
/// entirely (ranged reads go straight to storage), protecting the store
/// from giant-file eviction storms. Data files are immutable by path — no
/// invalidation.
///
/// Admission is per SCAN, not only per file: a scan that projects a small
/// share of a file's bytes (the manifest's `column_sizes` say how much)
/// gains little from the local copy but pays for the whole object over the
/// wire and in cache churn, so with `min_projected_fraction` set such a
/// scan reads its byte ranges directly instead — see [`Self::admits`].
#[derive(Clone, Debug)]
pub struct DataBytesCache {
    /// The shared, path-keyed bytes store (typically disk-backed).
    pub cache: ObjectBytesCacheRef,
    /// Files larger than this are never cached (read directly).
    pub max_file_bytes: u64,
    /// A scan projecting less than this share of a file's bytes reads its
    /// ranges directly instead of admitting the whole object. `0.0` (the
    /// default) admits every file within `max_file_bytes`.
    pub min_projected_fraction: f64,
    /// Files at most this large are admitted whatever the projected share:
    /// one GET is cheaper than several ranged reads.
    pub small_file_bytes: u64,
}

impl DataBytesCache {
    /// A cache admitting every file within `max_file_bytes`; the projected
    /// share is not consulted.
    pub fn new(cache: ObjectBytesCacheRef, max_file_bytes: u64) -> Self {
        Self {
            cache,
            max_file_bytes,
            min_projected_fraction: 0.0,
            small_file_bytes: 0,
        }
    }

    /// Admit a file only when the scan projects at least
    /// `min_projected_fraction` of its bytes (clamped to `0.0..=1.0`; `0.0`
    /// disables the check), files of at most `small_file_bytes` excepted.
    pub fn with_admission(mut self, min_projected_fraction: f64, small_file_bytes: u64) -> Self {
        self.min_projected_fraction = if min_projected_fraction.is_finite() {
            min_projected_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.small_file_bytes = small_file_bytes;
        self
    }

    /// Whether a scan projecting `projected_share` of a `file_size`-byte
    /// file (`None` = unknown) should read the WHOLE object through the
    /// cache — or read its byte ranges straight from storage.
    ///
    /// Whole-file iff the file is within `max_file_bytes` and any of: the
    /// share check is disabled, the file is at most `small_file_bytes`, the
    /// share is unknown (the manifest recorded no column sizes), or the
    /// share reaches `min_projected_fraction`.
    pub fn admits(&self, file_size: u64, projected_share: Option<f64>) -> bool {
        if file_size == 0 || file_size > self.max_file_bytes {
            return false;
        }
        if self.min_projected_fraction <= 0.0 || file_size <= self.small_file_bytes {
            return true;
        }
        projected_share.is_none_or(|share| share >= self.min_projected_fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct _TestDynCompatibleForObjectCache(Arc<dyn ObjectCache<String, Arc<Manifest>>>);
    struct _TestDynCompatibleForObjectCacheProvider(ObjectCacheProvider);
    struct _TestDynCompatibleForObjectBytesCache(ObjectBytesCacheRef);

    /// A store that holds nothing — admission is decided before it is touched.
    #[derive(Debug)]
    struct Empty;

    #[async_trait]
    impl ObjectBytesCache for Empty {
        async fn get(&self, _path: &str) -> Option<Bytes> {
            None
        }
        async fn set(&self, _path: &str, _bytes: Bytes) {}
    }

    fn cache(min_projected_fraction: f64, small_file_bytes: u64) -> DataBytesCache {
        DataBytesCache::new(Arc::new(Empty), 1000).with_admission(min_projected_fraction, small_file_bytes)
    }

    #[test]
    fn size_cap_applies_before_anything_else() {
        let dc = cache(0.0, 0);
        assert!(!dc.admits(0, None), "an unknown size is never admitted");
        assert!(dc.admits(1000, None));
        assert!(!dc.admits(1001, None), "above the cap: read directly");
        assert!(!dc.admits(1001, Some(1.0)), "even a full projection");
    }

    #[test]
    fn disabled_check_admits_every_file_within_the_cap() {
        let dc = cache(0.0, 0);
        assert!(dc.admits(500, Some(0.0)));
        assert!(dc.admits(500, Some(0.01)));
        assert!(dc.admits(500, None));
    }

    #[test]
    fn share_below_the_fraction_reads_directly() {
        let dc = cache(0.5, 0);
        assert!(!dc.admits(500, Some(0.03)), "a narrow projection");
        assert!(!dc.admits(500, Some(0.49)));
        assert!(dc.admits(500, Some(0.5)), "the fraction itself admits");
        assert!(dc.admits(500, Some(1.0)));
    }

    #[test]
    fn unknown_share_keeps_the_whole_file_path() {
        assert!(cache(0.5, 0).admits(500, None));
    }

    #[test]
    fn small_files_are_admitted_regardless_of_share() {
        let dc = cache(0.5, 64);
        assert!(dc.admits(64, Some(0.01)));
        assert!(!dc.admits(65, Some(0.01)));
    }

    #[test]
    fn fraction_is_clamped_and_nan_disables() {
        assert_eq!(cache(7.0, 0).min_projected_fraction, 1.0);
        assert_eq!(cache(-1.0, 0).min_projected_fraction, 0.0);
        assert_eq!(cache(f64::NAN, 0).min_projected_fraction, 0.0);
        assert!(cache(f64::NAN, 0).admits(500, Some(0.01)));
    }
}
