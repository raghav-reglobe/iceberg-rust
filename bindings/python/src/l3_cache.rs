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

//! L3 shared-cache client — the FAIL-OPEN tier under the local foyer
//! caches, speaking plain HTTP to the pulse-cached daemon (`GET /o`,
//! path in the `x-path` header; the daemon is read-through so a client
//! never inserts into L3).
//!
//! Contract (dark by default): [`TieredBytesCache`] wraps a local
//! [`ObjectBytesCache`] and consults L3 only on a local miss. ANY L3
//! error/timeout is a miss — the caller falls through to direct S3, the
//! byte-identical path that runs with the tier disabled. A circuit
//! breaker bounds a down daemon's cost to one timeout per trip window
//! per process.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use iceberg::cache::{ObjectBytesCache, ObjectBytesCacheRef, ObjectCacheStats};

/// Consecutive-failure circuit breaker. Pure state machine (unit-tested);
/// `now_ms` is injected so tests never sleep.
#[derive(Debug, Default)]
pub struct Breaker {
    consecutive_failures: AtomicU32,
    open_until_ms: AtomicU64,
    threshold: u32,
    cooldown_ms: u64,
}

impl Breaker {
    pub fn new(threshold: u32, cooldown_ms: u64) -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            open_until_ms: AtomicU64::new(0),
            threshold: threshold.max(1),
            cooldown_ms,
        }
    }

    /// May a call proceed at `now_ms`?
    pub fn allows(&self, now_ms: u64) -> bool {
        now_ms >= self.open_until_ms.load(Ordering::Relaxed)
    }

    pub fn on_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    pub fn on_failure(&self, now_ms: u64) {
        let n = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= self.threshold {
            self.open_until_ms
                .store(now_ms + self.cooldown_ms, Ordering::Relaxed);
            // A fresh window after the cooldown: one probe failure
            // re-opens immediately rather than needing a full run-up.
            self.consecutive_failures
                .store(self.threshold.saturating_sub(1), Ordering::Relaxed);
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// HTTP client for the pulse-cached daemon's `/o` lane.
#[derive(Debug)]
pub struct L3Http {
    endpoint: String,
    client: reqwest::Client,
    breaker: Breaker,
}

impl L3Http {
    /// `endpoint` e.g. `http://pulse-cached.pulse-compute.svc:8090`.
    pub fn new(endpoint: &str, connect_timeout_ms: u64, timeout_ms: u64) -> Option<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(connect_timeout_ms))
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .ok()?;
        Some(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client,
            breaker: Breaker::new(5, 30_000),
        })
    }

    /// Fetch `path` through the daemon. `None` on ANY failure (fail-open).
    ///
    /// `size_hint` = the object's known length (manifest-recorded), sent as
    /// `x-size-hint` so the daemon reserves an EXACT fetch budget on a miss
    /// instead of a worst-case `max_file` slot (or a probe HEAD). Always the
    /// TRUE size, never clamped — an over-`max_file` hint is the daemon's
    /// bypass signal. Absent/0 = unknown.
    pub async fn get(&self, path: &str, size_hint: Option<u64>) -> Option<Bytes> {
        let now = now_ms();
        if !self.breaker.allows(now) {
            return None;
        }
        let url = format!("{}/o", self.endpoint);
        let mut req = self.client.get(&url).header("x-path", path);
        if let Some(size) = size_hint.filter(|s| *s > 0) {
            req = req.header("x-size-hint", size.to_string());
        }
        let resp = req.send().await;
        match resp {
            Ok(r) if r.status().is_success() => match r.bytes().await {
                Ok(b) => {
                    self.breaker.on_success();
                    Some(b)
                }
                Err(_) => {
                    self.breaker.on_failure(now_ms());
                    None
                }
            },
            // 404 = the object genuinely doesn't exist upstream; that is a
            // HEALTHY answer (don't trip the breaker) but still a miss —
            // the caller's own S3 read will surface the real error.
            Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => {
                self.breaker.on_success();
                None
            }
            _ => {
                self.breaker.on_failure(now_ms());
                None
            }
        }
    }
}

/// Local-first, L3-second, fail-open composite. `local: None` = L3-only
/// (used when the pod-local data cache is disabled but the shared tier is
/// configured).
#[derive(Debug)]
pub struct TieredBytesCache {
    local: Option<ObjectBytesCacheRef>,
    l3: Arc<L3Http>,
}

impl TieredBytesCache {
    pub fn new(local: Option<ObjectBytesCacheRef>, l3: Arc<L3Http>) -> Self {
        Self { local, l3 }
    }
}

#[async_trait]
impl ObjectBytesCache for TieredBytesCache {
    async fn get(&self, path: &str) -> Option<Bytes> {
        self.get_with_size_hint(path, None).await
    }

    async fn get_with_size_hint(&self, path: &str, size_hint: Option<u64>) -> Option<Bytes> {
        if let Some(local) = &self.local {
            if let Some(bytes) = local.get(path).await {
                return Some(bytes);
            }
        }
        let bytes = self.l3.get(path, size_hint).await?;
        // Populate the pod-local tier so subsequent reads skip the hop.
        if let Some(local) = &self.local {
            local.set(path, bytes.clone()).await;
        }
        Some(bytes)
    }

    async fn set(&self, path: &str, bytes: Bytes) {
        // Local only: the daemon self-populates via read-through, so a
        // client set to L3 would be redundant write traffic.
        if let Some(local) = &self.local {
            local.set(path, bytes).await;
        }
    }

    fn stats(&self) -> Option<ObjectCacheStats> {
        // The local tier's stats stay the pod-lifecycle story; the L3
        // view (hits/misses/errors fleet-wide) lives on the daemon's
        // /metrics endpoint.
        self.local.as_ref().and_then(|l| l.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaker_opens_after_threshold_and_reopens_after_cooldown() {
        let b = Breaker::new(3, 1_000);
        assert!(b.allows(0));
        b.on_failure(0);
        b.on_failure(0);
        assert!(b.allows(0), "below threshold stays closed");
        b.on_failure(0); // 3rd -> trips
        assert!(!b.allows(500), "open during cooldown");
        assert!(b.allows(1_000), "allows at cooldown expiry");
        // Post-cooldown, a single probe failure re-trips immediately.
        b.on_failure(1_000);
        assert!(!b.allows(1_500));
        // A success resets the run-up entirely.
        assert!(b.allows(2_000));
        b.on_success();
        b.on_failure(2_000);
        b.on_failure(2_000);
        assert!(b.allows(2_000), "reset run-up is below threshold again");
    }

    /// One-shot HTTP responder: accepts a single connection, captures the
    /// request head, answers 200 with `body`. Returns (endpoint, captured).
    async fn one_shot_server(body: &'static [u8]) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = sock.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.write_all(body).await.unwrap();
            head
        });
        (endpoint, handle)
    }

    /// The known object length rides `x-size-hint` (exact-budget reserve on
    /// the daemon side); an unknown length sends NO hint header.
    #[tokio::test]
    async fn size_hint_header_rides_the_request() {
        let (endpoint, captured) = one_shot_server(b"abc").await;
        let l3 = L3Http::new(&endpoint, 1_000, 2_000).unwrap();
        assert_eq!(
            l3.get("s3://b/k.parquet", Some(12_345)).await,
            Some(Bytes::from_static(b"abc"))
        );
        let head = captured.await.unwrap().to_ascii_lowercase();
        assert!(head.contains("x-path: s3://b/k.parquet"), "{head}");
        assert!(head.contains("x-size-hint: 12345"), "{head}");

        let (endpoint, captured) = one_shot_server(b"abc").await;
        let l3 = L3Http::new(&endpoint, 1_000, 2_000).unwrap();
        l3.get("s3://b/k.parquet", None).await.unwrap();
        let head = captured.await.unwrap().to_ascii_lowercase();
        assert!(!head.contains("x-size-hint"), "{head}");
    }

    /// A dead endpoint must be a MISS, never an error (fail-open), and the
    /// local tier must still serve.
    #[tokio::test]
    async fn tiered_fails_open_on_dead_l3() {
        #[derive(Debug)]
        struct StubLocal(std::sync::Mutex<std::collections::HashMap<String, Bytes>>);
        #[async_trait]
        impl ObjectBytesCache for StubLocal {
            async fn get(&self, path: &str) -> Option<Bytes> {
                self.0.lock().unwrap().get(path).cloned()
            }
            async fn set(&self, path: &str, bytes: Bytes) {
                self.0.lock().unwrap().insert(path.to_string(), bytes);
            }
        }

        let local: ObjectBytesCacheRef =
            Arc::new(StubLocal(std::sync::Mutex::new(Default::default())));
        // Port 1 refuses immediately on loopback.
        let l3 = Arc::new(L3Http::new("http://127.0.0.1:1", 200, 500).unwrap());
        let tiered = TieredBytesCache::new(Some(local.clone()), l3);

        assert!(tiered.get("s3://b/miss.parquet").await.is_none());
        tiered
            .set("s3://b/k.parquet", Bytes::from_static(b"abc"))
            .await;
        assert_eq!(
            tiered.get("s3://b/k.parquet").await,
            Some(Bytes::from_static(b"abc")),
            "local tier serves regardless of L3 health"
        );
    }
}
