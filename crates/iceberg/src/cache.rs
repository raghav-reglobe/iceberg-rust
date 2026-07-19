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
    /// Caches the raw file bytes fetched from `path`.
    async fn set(&self, path: &str, bytes: Bytes);
}

/// A thread-safe, reference-counted handle to an [`ObjectBytesCache`].
pub type ObjectBytesCacheRef = Arc<dyn ObjectBytesCache>;

#[cfg(test)]
mod tests {
    use super::*;

    struct _TestDynCompatibleForObjectCache(Arc<dyn ObjectCache<String, Arc<Manifest>>>);
    struct _TestDynCompatibleForObjectCacheProvider(ObjectCacheProvider);
    struct _TestDynCompatibleForObjectBytesCache(ObjectBytesCacheRef);
}
