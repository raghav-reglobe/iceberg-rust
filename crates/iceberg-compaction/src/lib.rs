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

//! JVM-free Iceberg table compaction for iceberg-rust.
//!
//! Planner + execution on top of the core `iceberg` crate. Parity spec is
//! iceberg-go's `table/compaction`; the commit primitive is the core crate's
//! `iceberg::transaction::RewriteFiles` (`Operation::Replace`).
//!
//! Module map:
//!   - config  : compaction tunables (iceberg-go `Config`, 128 MB target)
//!   - planner : partition-grouped candidate selection + bin-packing
//!   - sort    : sort group batches by `_valid_from` (arrow kernels)
//!   - rewrite : read (DVs applied) -> sort -> write (+partition +bloom) ->
//!               commit (RewriteFiles: swap data files + reabsorb DVs)
//!   - engine  : load -> enumerate data+delete files -> plan -> rewrite-loop

pub mod config;
pub mod engine;
pub mod planner;
pub mod rewrite;
mod sort;
