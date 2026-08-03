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

//! [`ScanMemoryGate`] backed by a DataFusion [`MemoryPool`].
//!
//! Registers scan decode working sets as a pool consumer so
//! - the pool's peak metric includes read memory (honest accounting even on
//!   an unbounded pool),
//! - on a BOUNDED pool, decode admission waits (bounded) for concurrent
//!   releases — operators under memory pressure spill/finish and return
//!   budget — and then fails the scan CLEANLY instead of letting resident
//!   memory grow past the process budget into an external kill.
//!
//! Deadlock note: a pool-holding operator that is idle (e.g. a sort holding
//! its build without growing) never releases on our behalf — the bounded
//! wait converts that case into a clean resources-exhausted error rather
//! than an unbounded stall.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use futures::future::BoxFuture;
use iceberg::arrow::ScanMemoryGate;
use iceberg::{Error, ErrorKind, Result};

/// Disable knob: `ICEBERG_SCAN_MEMORY_GATE=0|false` turns decode accounting
/// off (reads become invisible to the pool — the pre-gate behavior).
pub fn scan_gate_enabled() -> bool {
    !matches!(
        std::env::var("ICEBERG_SCAN_MEMORY_GATE").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Max total wait for pool headroom before failing the scan cleanly.
/// `ICEBERG_SCAN_GATE_WAIT_S` (default 60).
fn gate_wait() -> Duration {
    Duration::from_secs(
        std::env::var("ICEBERG_SCAN_GATE_WAIT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60),
    )
}

/// Build a pool-backed gate, or `None` when disabled by env.
pub fn pool_scan_gate(pool: &Arc<dyn MemoryPool>) -> Option<Arc<dyn ScanMemoryGate>> {
    if !scan_gate_enabled() {
        return None;
    }
    Some(Arc::new(PoolScanGate {
        reservation: Mutex::new(MemoryConsumer::new("iceberg-scan-decode").register(pool)),
        pool: Arc::clone(pool),
    }))
}

pub(crate) struct PoolScanGate {
    reservation: Mutex<MemoryReservation>,
    pool: Arc<dyn MemoryPool>,
}

impl std::fmt::Debug for PoolScanGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PoolScanGate")
    }
}

impl ScanMemoryGate for PoolScanGate {
    fn acquire(&self, bytes: u64) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            // Clamp a single file's reservation to HALF the pool: the
            // estimate models the full decode working set, but a streaming
            // read holds the compressed pages + a few in-flight batches —
            // hard-refusing any file whose estimate exceeds the pool would
            // fail reads that stream fine today. The clamp keeps big files
            // admissible (partial, honest-enough accounting) while pool
            // CONTENTION still gates admission and fails cleanly.
            let bytes = match self.pool.memory_limit() {
                datafusion::execution::memory_pool::MemoryLimit::Finite(limit) => {
                    bytes.min((limit as u64) / 2)
                }
                _ => bytes,
            };
            let wait = gate_wait();
            let start = std::time::Instant::now();
            loop {
                let res = self
                    .reservation
                    .lock()
                    .expect("gate reservation poisoned")
                    .try_grow(bytes as usize);
                match res {
                    Ok(()) => return Ok(()),
                    Err(e) if start.elapsed() >= wait => {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            format!(
                                "scan memory gate: pool cannot fit a {bytes}-byte decode \
                                 working set after {}s: {e}",
                                wait.as_secs()
                            ),
                        ));
                    }
                    Err(_) => tokio::time::sleep(Duration::from_millis(200)).await,
                }
            }
        })
    }

    fn release(&self, bytes: u64) {
        // Mirror acquire's clamp exactly (the pool limit is fixed for the
        // pool's lifetime, so the mapping is deterministic) — an asymmetric
        // shrink would underflow the reservation.
        let bytes = match self.pool.memory_limit() {
            datafusion::execution::memory_pool::MemoryLimit::Finite(limit) => {
                bytes.min((limit as u64) / 2)
            }
            _ => bytes,
        };
        self.reservation
            .lock()
            .expect("gate reservation poisoned")
            .shrink(bytes as usize);
    }
}
