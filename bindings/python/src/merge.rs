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

//! One-shot `MERGE INTO` doorway: run genuine MERGE SQL text against Iceberg
//! tables mounted as DataFusion catalogs.
//!
//! The statement is planned by DataFusion's MERGE planner and executed by the
//! merge-on-read chain behind `TableProvider::merge_into` (deletion vectors
//! on the scanned `(_file, _pos)`, late-materialized matched-row appends, one
//! `RowDelta` snapshot). `catalogs` maps each SQL catalog name used in the
//! statement to standard Iceberg REST catalog properties, so every
//! `catalog.namespace.table` reference resolves with no register steps.
//! Catalog handles are memoized per process — repeated calls reuse the same
//! authenticated clients instead of re-fetching config/tokens per call.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
use iceberg::{Catalog, CatalogBuilder};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_datafusion::functions::register_variant_functions;
use iceberg_datafusion::{IcebergCatalogProvider, MorMergeOptions};
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::runtime;

/// Per-process memoized catalog handles, keyed by catalog name + properties.
/// A REST catalog re-auths (`/config` + OAuth token) on construction; reusing
/// the handle across calls avoids hammering the catalog server from a worker
/// that merges every few seconds.
static CATALOGS: OnceLock<Mutex<HashMap<String, Arc<dyn Catalog>>>> = OnceLock::new();

fn catalog_cache_key(name: &str, props: &HashMap<String, String>) -> String {
    let mut entries: Vec<_> = props.iter().collect();
    entries.sort();
    let mut key = String::from(name);
    for (k, v) in entries {
        key.push('\u{1f}');
        key.push_str(k);
        key.push('\u{1f}');
        key.push_str(v);
    }
    key
}

async fn get_or_build_catalog(
    name: &str,
    props: HashMap<String, String>,
) -> PyResult<Arc<dyn Catalog>> {
    let key = catalog_cache_key(name, &props);
    if let Some(cat) = CATALOGS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .get(&key)
    {
        return Ok(Arc::clone(cat));
    }
    let catalog = RestCatalogBuilder::default()
        .with_storage_factory(Arc::new(OpenDalResolvingStorageFactory::new()))
        .load(name.to_string(), props)
        .await
        .map_err(|e| PyValueError::new_err(format!("build catalog `{name}`: {e}")))?;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    CATALOGS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(key, Arc::clone(&catalog));
    Ok(catalog)
}

/// `scan_files` keys are `catalog.namespace.table` (namespace may be
/// multi-level: first segment = catalog, last = table, middle = namespace).
/// Grouped per catalog as `(namespace, table) -> files`.
type ScanFiles = HashMap<String, HashMap<(String, String), Vec<String>>>;

fn parse_scan_files(scan_files: Option<HashMap<String, Vec<String>>>) -> PyResult<ScanFiles> {
    let mut out: ScanFiles = HashMap::new();
    for (fqn, files) in scan_files.unwrap_or_default() {
        let parts: Vec<&str> = fqn.split('.').collect();
        if parts.len() < 3 {
            return Err(PyValueError::new_err(format!(
                "scan_files key `{fqn}` must be `catalog.namespace.table`"
            )));
        }
        let catalog = parts[0].to_string();
        let table = parts[parts.len() - 1].to_string();
        let namespace = parts[1..parts.len() - 1].join(".");
        out.entry(catalog)
            .or_default()
            .insert((namespace, table), files);
    }
    Ok(out)
}

async fn session_with_catalogs(
    catalogs: HashMap<String, HashMap<String, String>>,
    mut scan_files: ScanFiles,
    options: Option<Arc<MorMergeOptions>>,
) -> PyResult<SessionContext> {
    // Preserve identifier case (duckdb/Spark semantics): mongo-derived columns
    // are mixed-case, and the in-flight MERGE planner drops the quote flag on
    // INSERT/SET column names, so normalization would lowercase them anyway.
    let mut config = datafusion::execution::context::SessionConfig::new()
        .set_bool("datafusion.sql_parser.enable_ident_normalization", false);
    if let Some(options) = options {
        config = config.with_extension(options);
    }
    // Memory safety: an explicit memory pool makes DataFusion's sort/window/
    // aggregate operators SPILL instead of OOM-killing the process (the
    // merge's union sort is the big resident; a row cap alone cannot bound
    // memory — width varies per table). Env-configured so the worker sizes
    // it to the pod: MERGE_DF_MEMORY_LIMIT_MB (unset = unbounded, the
    // previous behavior) + MERGE_DF_SPILL_DIR (unset = OS temp dir).
    let ctx = match std::env::var("MERGE_DF_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        None => SessionContext::new_with_config(config),
        Some(limit_mb) => {
            let mut rt = datafusion::execution::runtime_env::RuntimeEnvBuilder::new()
                .with_memory_limit(limit_mb * 1024 * 1024, 1.0);
            if let Ok(dir) = std::env::var("MERGE_DF_SPILL_DIR") {
                rt = rt.with_disk_manager_builder(
                    datafusion::execution::disk_manager::DiskManagerBuilder::default()
                        .with_mode(
                            datafusion::execution::disk_manager::DiskManagerMode::Directories(
                                vec![dir.into()],
                            ),
                        ),
                );
            }
            let rt = rt
                .build_arc()
                .map_err(|e| PyValueError::new_err(format!("building runtime env: {e}")))?;
            SessionContext::new_with_config_rt(config, rt)
        }
    };
    register_variant_functions(&ctx);
    for (name, props) in catalogs {
        let catalog = get_or_build_catalog(&name, props).await?;
        let provider = IcebergCatalogProvider::try_new(catalog)
            .await
            .map_err(|e| PyValueError::new_err(format!("mount catalog `{name}`: {e}")))?;
        // Scan-file allowlists (externally planned file subsets) are applied
        // per call — providers are rebuilt per session; only the catalog
        // handle above is memoized.
        for ((namespace, table), files) in scan_files.remove(&name).unwrap_or_default() {
            provider
                .with_table_scan_file_allowlist(&namespace, &table, files)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "scan_files for `{name}.{namespace}.{table}`: {e}"
                    ))
                })?;
        }
        ctx.register_catalog(&name, Arc::new(provider));
    }
    if let Some(unmatched) = scan_files.keys().next() {
        return Err(PyValueError::new_err(format!(
            "scan_files references catalog `{unmatched}` which is not in `catalogs`"
        )));
    }
    Ok(ctx)
}

/// Await `fut`, raising when the merge deadline passes first. Only used for
/// phases that perform no writes (mounting, planning) — the execution phase
/// enforces the same deadline cooperatively inside the write node so the
/// commit is never cancelled mid-flight.
async fn doorway_deadline<T>(
    deadline: Option<std::time::Instant>,
    what: &str,
    fut: impl std::future::Future<Output = T>,
) -> PyResult<T> {
    match deadline {
        None => Ok(fut.await),
        Some(d) => tokio::time::timeout_at(tokio::time::Instant::from_std(d), fut)
            .await
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "merge timeout exceeded while {what} (no writes were performed)"
                ))
            }),
    }
}

/// Execute one `MERGE INTO` statement and block until its snapshot commits.
///
/// `catalogs` maps each SQL catalog name to Iceberg REST catalog properties
/// (`uri`, `warehouse`, `credential`, `oauth2-server-uri`, `scope`, ...);
/// `sql` is the full MERGE statement (DataFusion dialect). `scan_files`
/// optionally restricts named tables' scans to an externally planned set of
/// data-file paths: `{"catalog.namespace.table": ["s3://.../f.parquet", ...]}`
/// (deletes still apply to the retained files; unknown identifiers raise).
/// `timeout_s` bounds the merge: mounting, planning and the scan/write phase
/// abort once it elapses (cooperatively inside the write node — the snapshot
/// commit itself is never cancelled; a timed-out merge writes NO snapshot).
/// `write_workers` sizes the writer pool for appended output (default:
/// min(4, cores)). Returns a dict with `count` — the number of rows appended
/// by the merge (inserts plus updated row versions). Raises `ValueError` on
/// planning or execution failure, and on deadline expiry.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, timeout_s=None, write_workers=None))]
fn merge_into(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    timeout_s: Option<u64>,
    write_workers: Option<usize>,
) -> PyResult<HashMap<String, String>> {
    let scan_files = parse_scan_files(scan_files)?;
    // The deadline covers catalog mounting, planning and the scan/write
    // phase (enforced cooperatively inside the merge write node). The
    // snapshot COMMIT is deliberately outside it — cancelling a REST commit
    // in flight leaves the outcome unknown; once the write phase finishes
    // under deadline, the commit runs to completion.
    let deadline = timeout_s.map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
    py.detach(|| {
        runtime().block_on(async move {
            let options = Arc::new(MorMergeOptions {
                deadline,
                write_workers,
            });
            let ctx = doorway_deadline(
                deadline,
                "mounting catalogs",
                session_with_catalogs(catalogs, scan_files, Some(options)),
            )
            .await??;
            let df = doorway_deadline(deadline, "planning MERGE", ctx.sql(&sql))
                .await?
                .map_err(|e| PyValueError::new_err(format!("planning MERGE: {e}")))?;
            let batches = df
                .collect()
                .await
                .map_err(|e| PyValueError::new_err(format!("executing MERGE: {e}")))?;
            let mut count: u64 = 0;
            for batch in &batches {
                if let Some(col) = batch.column_by_name("count")
                    && let Some(arr) = col
                        .as_any()
                        .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                {
                    count += arr.iter().flatten().sum::<u64>();
                }
            }
            Ok(HashMap::from([("count".to_string(), count.to_string())]))
        })
    })
}

/// Plan one `MERGE INTO` statement WITHOUT executing it and return the
/// physical plan as text — the read-only twin of [`merge_into`]. Planning
/// resolves every table reference against the live catalogs (metadata reads
/// only, no data IO, no commit), so a failing or suspicious merge can be
/// inspected with zero writes.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None))]
fn dry_run_inspect(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
) -> PyResult<HashMap<String, String>> {
    let scan_files = parse_scan_files(scan_files)?;
    py.detach(|| {
        runtime().block_on(async move {
            let ctx = session_with_catalogs(catalogs, scan_files, None).await?;
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| PyValueError::new_err(format!("planning MERGE: {e}")))?;
            let logical = df.logical_plan().display_indent().to_string();
            let physical = df
                .create_physical_plan()
                .await
                .map_err(|e| PyValueError::new_err(format!("physical planning MERGE: {e}")))?;
            let physical = displayable(physical.as_ref()).indent(true).to_string();
            Ok(HashMap::from([
                ("logical_plan".to_string(), logical),
                ("physical_plan".to_string(), physical),
            ]))
        })
    })
}

/// Run one READ-ONLY SQL statement and return its rows as a JSON array
/// string — the query twin of [`merge_into`], for harnesses, referees and
/// parity checks. DML/DDL/COPY plans are rejected before execution; the
/// result is capped at `max_rows` (error, not truncation — a capped result
/// silently read as complete is worse than a loud failure). `scan_files`
/// bounds named tables' scans exactly as in [`merge_into`]. Columns must be
/// JSON-serializable by the arrow JSON writer — wrap variant/binary columns
/// in `variant_to_json(...)` in the statement.
#[pyfunction]
#[pyo3(signature = (catalogs, sql, scan_files=None, max_rows=100_000))]
fn sql_collect(
    py: Python<'_>,
    catalogs: HashMap<String, HashMap<String, String>>,
    sql: String,
    scan_files: Option<HashMap<String, Vec<String>>>,
    max_rows: usize,
) -> PyResult<String> {
    use datafusion::logical_expr::LogicalPlan;

    let scan_files = parse_scan_files(scan_files)?;
    py.detach(|| {
        runtime().block_on(async move {
            let ctx = session_with_catalogs(catalogs, scan_files, None).await?;
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| PyValueError::new_err(format!("planning query: {e}")))?;
            if matches!(
                df.logical_plan(),
                LogicalPlan::Dml(_) | LogicalPlan::Ddl(_) | LogicalPlan::Copy(_)
            ) {
                return Err(PyValueError::new_err(
                    "sql_collect is read-only — use merge_into for writes",
                ));
            }
            let batches = df
                .collect()
                .await
                .map_err(|e| PyValueError::new_err(format!("executing query: {e}")))?;
            let total: usize = batches.iter().map(|b| b.num_rows()).sum();
            if total > max_rows {
                return Err(PyValueError::new_err(format!(
                    "result has {total} rows > max_rows={max_rows} — narrow the query or raise the cap"
                )));
            }
            let mut buf = Vec::new();
            {
                let mut writer = datafusion::arrow::json::ArrayWriter::new(&mut buf);
                for batch in &batches {
                    writer
                        .write(batch)
                        .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))?;
                }
                writer
                    .finish()
                    .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))?;
            }
            String::from_utf8(buf)
                .map_err(|e| PyValueError::new_err(format!("serializing rows: {e}")))
        })
    })
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "merge")?;
    this.add_function(wrap_pyfunction!(merge_into, &this)?)?;
    this.add_function(wrap_pyfunction!(dry_run_inspect, &this)?)?;
    this.add_function(wrap_pyfunction!(sql_collect, &this)?)?;
    m.add_submodule(&this)?;
    Ok(())
}
