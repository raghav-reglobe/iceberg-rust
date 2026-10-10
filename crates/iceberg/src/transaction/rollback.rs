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

use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::{MAIN_BRANCH, SnapshotReference, SnapshotRetention};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableRequirement, TableUpdate};

/// A transaction action that moves the main branch back to an existing
/// snapshot (the `ManageSnapshots.rollbackTo` of the reference implementation).
///
/// Nothing is written and nothing is expired: the current snapshot stays in
/// the metadata (reachable by id, removable by a later expiry), only the
/// `main` reference moves. The commit requires that `main` still points at
/// the snapshot the table carried when the action was committed, so a
/// concurrent commit makes the rollback fail instead of clobbering it.
pub struct RollbackToSnapshotAction {
    snapshot_id: i64,
}

impl RollbackToSnapshotAction {
    /// Roll `main` back to `snapshot_id`, which must be a snapshot the table's
    /// metadata still holds.
    pub fn new(snapshot_id: i64) -> Self {
        RollbackToSnapshotAction { snapshot_id }
    }
}

#[async_trait]
impl TransactionAction for RollbackToSnapshotAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let metadata = table.metadata();
        if metadata.snapshot_by_id(self.snapshot_id).is_none() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "snapshot {} is not in the table's metadata (unknown or expired) — nothing to roll back to",
                    self.snapshot_id
                ),
            ));
        }
        let current = metadata.current_snapshot_id();
        if current == Some(self.snapshot_id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("snapshot {} is already the current snapshot", self.snapshot_id),
            ));
        }
        let updates = vec![TableUpdate::SetSnapshotRef {
            ref_name: MAIN_BRANCH.to_string(),
            reference: SnapshotReference::new(
                self.snapshot_id,
                SnapshotRetention::branch(None, None, None),
            ),
        }];
        let requirements = vec![TableRequirement::RefSnapshotIdMatch {
            r#ref: MAIN_BRANCH.to_string(),
            snapshot_id: current,
        }];
        Ok(ActionCommit::new(updates, requirements))
    }
}
