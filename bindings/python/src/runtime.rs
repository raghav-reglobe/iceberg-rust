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

use iceberg::cache::ObjectBytesCacheRef;
use iceberg_cache_foyer::FoyerObjectBytesCacheBuilder;
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

pub async fn global_object_cache() -> ObjectBytesCacheRef {
    OBJECT_CACHE
        .get_or_init(|| async {
            let memory_bytes = env_mb("ICEBERG_OBJECT_CACHE_MB", 128) * 1024 * 1024;
            let mut builder = FoyerObjectBytesCacheBuilder::new(memory_bytes as usize);
            if let Ok(dir) = std::env::var("ICEBERG_OBJECT_CACHE_DIR") {
                if !dir.is_empty() {
                    let disk_bytes = env_mb("ICEBERG_OBJECT_CACHE_DISK_MB", 1024) * 1024 * 1024;
                    builder = builder.with_disk(dir, disk_bytes as usize);
                }
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
