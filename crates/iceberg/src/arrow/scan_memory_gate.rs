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

//! Scan decode-memory accounting.
//!
//! Parquet decode holds a row group's compressed pages plus their decoded
//! arrow arrays per in-flight file — a working set the engine's query memory
//! manager never sees when reads are unaccounted. A [`ScanMemoryGate`] lets an
//! embedding engine (e.g. a DataFusion memory pool) RESERVE each file's
//! estimated decode working set before decode begins:
//!
//! - accounting: the engine's peak-memory metric includes read memory;
//! - backpressure: a contended pool delays new file decodes instead of
//!   letting resident memory grow past the process budget;
//! - clean failure: when the budget genuinely cannot fit, the scan fails
//!   with an error (checkpointable, retryable) instead of the process dying
//!   on an external kill with no failure path.
//!
//! The gate is engine-agnostic by design — this crate never depends on any
//! particular memory-pool implementation.

use std::fmt::Debug;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::Result;

/// Reserves decode working-set memory for scan reads.
///
/// Implementations may wait (bounded) for concurrent releases before giving
/// up; an `Err` from [`ScanMemoryGate::acquire`] fails the scan cleanly.
pub trait ScanMemoryGate: Debug + Send + Sync {
    /// Reserve `bytes` before a file's decode begins.
    fn acquire(&self, bytes: u64) -> BoxFuture<'_, Result<()>>;
    /// Release a prior reservation of `bytes`.
    fn release(&self, bytes: u64);
}

/// RAII reservation: releases the acquired bytes on drop, so a scan stream
/// that is dropped early (limit, abort, error) always returns its budget.
#[derive(Debug)]
pub struct GateGuard {
    gate: Arc<dyn ScanMemoryGate>,
    bytes: u64,
}

impl GateGuard {
    /// Acquire `bytes` from `gate`, returning a guard that releases on drop.
    pub async fn acquire(gate: Arc<dyn ScanMemoryGate>, bytes: u64) -> Result<Self> {
        gate.acquire(bytes).await?;
        Ok(Self { gate, bytes })
    }
}

impl Drop for GateGuard {
    fn drop(&mut self) {
        self.gate.release(self.bytes);
    }
}
