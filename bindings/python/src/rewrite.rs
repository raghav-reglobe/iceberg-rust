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

//! Copy-on-write column rewrites — today: MySQL `TIME` string spellings into one
//! signed `long` of microseconds (see `iceberg::time_rewrite`).
use std::collections::HashMap;

use iceberg::time_rewrite::{mysql_time_text_to_us as parse_time, rewrite_time_columns as rewrite};
use iceberg::{NamespaceIdent, TableIdent};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

fn split_fqn(fqn: &str) -> PyResult<(String, NamespaceIdent, String)> {
    let parts: Vec<&str> = fqn.split('.').collect();
    if parts.len() < 3 {
        return Err(PyValueError::new_err(format!(
            "table `{fqn}` must be `catalog.namespace.table`"
        )));
    }
    let namespace = NamespaceIdent::from_vec(
        parts[1..parts.len() - 1]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        parts[0].to_string(),
        namespace,
        parts[parts.len() - 1].to_string(),
    ))
}

/// Parse one MySQL `TIME` spelling into signed microseconds since midnight
/// (`None` for an empty string; raises on an unrecognised spelling) — the
/// Python-side twin for verification scripts.
#[pyfunction]
fn mysql_time_text_to_us(s: &str) -> PyResult<Option<i64>> {
    parse_time(s)
        .map(|o| o.map(|(us, _)| us))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Fill each `(source, target)` pair of `table` (`catalog.namespace.table`) —
/// `source` a string column of spellings, `target` an existing `long` column —
/// with signed microseconds, copy-on-write, in ONE replace snapshot; the source
/// is kept. Add the targets first (`schema.update_schema(add_columns=...)`); swap
/// the names at the end (`drop_columns=[source], rename_columns=[(target, source)]`). With
/// `dry_run` nothing is written or committed — the counts say what a run would
/// meet, and an unknown spelling raises. `base_snapshot` pins the plan to the
/// current snapshot; `carry_summary_prefix` copies the base snapshot's summary
/// keys with that prefix onto the new snapshot. Returns a dict of strings.
#[pyfunction]
#[pyo3(signature = (catalogs, table, columns, base_snapshot=None, dry_run=false, snapshot_properties=None, carry_summary_prefix=None))]
fn rewrite_time_columns(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    table: String,
    columns: Vec<(String, String)>,
    base_snapshot: Option<i64>,
    dry_run: bool,
    snapshot_properties: Option<HashMap<String, String>>,
    carry_summary_prefix: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let (catalog_name, namespace, table_name) = split_fqn(&table)?;
    let Some(props) = catalogs.get(&catalog_name).cloned() else {
        return Err(PyValueError::new_err(format!(
            "catalog `{catalog_name}` not in `catalogs`"
        )));
    };
    if columns.is_empty() {
        return Err(PyValueError::new_err("columns must be non-empty"));
    }
    py.detach(|| {
        runtime().block_on(async move {
            let catalog = crate::merge::get_or_build_catalog(&catalog_name, props).await?;
            let ident = TableIdent::new(namespace, table_name);
            let out = rewrite(
                catalog.as_ref(),
                &ident,
                &columns,
                base_snapshot,
                dry_run,
                snapshot_properties.unwrap_or_default(),
                carry_summary_prefix.as_deref(),
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("time rewrite {table}: {e}")))?;
            Ok(HashMap::from([
                ("snapshot_before".to_string(), out.snapshot_before.to_string()),
                (
                    "snapshot_after".to_string(),
                    out.snapshot_after.map(|s| s.to_string()).unwrap_or_default(),
                ),
                ("rows".to_string(), out.rows.to_string()),
                ("files_deleted".to_string(), out.files_deleted.to_string()),
                ("files_added".to_string(), out.files_added.to_string()),
                ("digits".to_string(), out.counts.digits.to_string()),
                ("dated".to_string(), out.counts.dated.to_string()),
                ("clock".to_string(), out.counts.clock.to_string()),
                ("days".to_string(), out.counts.days.to_string()),
                ("nulls".to_string(), out.counts.nulls.to_string()),
                ("dry_run".to_string(), dry_run.to_string()),
            ]))
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "rewrite")?;
    this.add_function(wrap_pyfunction!(rewrite_time_columns, &this)?)?;
    this.add_function(wrap_pyfunction!(mysql_time_text_to_us, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
