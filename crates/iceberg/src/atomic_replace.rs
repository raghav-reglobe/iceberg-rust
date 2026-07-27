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

//! ATOMIC PARTITION-SCOPED REPLACE — the append-only CDC "full-load chunk
//! re-backfill" primitive: one `RowDelta` snapshot that (i) equality-deletes
//! the PRIOR rows of a key chunk inside one identity partition (top-level,
//! pk-only equality ids, partition-scoped — so same-key rows in other
//! partitions are structurally out of scope) and (ii) appends the
//! replacement rows. The delete tuples ARE the replacement rows' keys.
//!
//! Concurrency: `validate_from_snapshot` OCC; concurrent PURE APPENDS (a
//! streaming sink) pass under snapshot isolation, conflicting delete
//! commits fail closed, and a conflict is retried BY RERUN — the
//! already-written data + delete files are re-committed from a fresh base
//! (equality-delete sequence numbers come from the commit, so a re-commit
//! stays correct).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use futures::TryStreamExt;
use parquet::variant::{VariantArrayBuilder, json_to_variant};
use roaring::RoaringTreemap;
use uuid::Uuid;

use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use crate::delete_vector::DeleteVector;
use crate::expr::{Predicate, Reference};
use crate::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use crate::scan::FileScanTask;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, Literal, ManifestContentType, ManifestList,
    PartitionKey, Struct, Transform,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

/// Input rows for an atomic replace: in-memory record batches (the IPC
/// doorway mode) or a LOCAL parquet file streamed batch-wise — bounded
/// memory for arbitrarily large chunks. The equality mode reads the input
/// TWICE (data pass + delete-tuple pass); each pass re-opens the file.
/// In-memory batches are conformed per pass too (deterministic — a small
/// CPU cost for one shared code path).
pub enum ReplaceInput {
    /// Decoded record batches held in memory.
    Batches(Vec<RecordBatch>),
    /// Path to a local parquet file (the caller's staged chunk).
    LocalParquet(String),
}

impl From<Vec<RecordBatch>> for ReplaceInput {
    fn from(batches: Vec<RecordBatch>) -> Self {
        ReplaceInput::Batches(batches)
    }
}

enum InputPass<'a> {
    Mem(std::slice::Iter<'a, RecordBatch>),
    File(parquet::arrow::arrow_reader::ParquetRecordBatchReader),
}

impl ReplaceInput {
    fn pass(&self) -> Result<InputPass<'_>> {
        match self {
            ReplaceInput::Batches(v) => Ok(InputPass::Mem(v.iter())),
            ReplaceInput::LocalParquet(path) => {
                let file = std::fs::File::open(path).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("opening local parquet input `{path}`: {e}"),
                    )
                })?;
                let reader =
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                        .map_err(|e| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("reading local parquet input `{path}`: {e}"),
                            )
                        })?
                        .with_batch_size(8192)
                        .build()
                        .map_err(|e| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("reading local parquet input `{path}`: {e}"),
                            )
                        })?;
                Ok(InputPass::File(reader))
            }
        }
    }
}

impl Iterator for InputPass<'_> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            InputPass::Mem(it) => it.next().map(|b| Ok(b.clone())),
            InputPass::File(r) => r.next().map(|res| {
                res.map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("decoding local parquet input batch: {e}"),
                    )
                })
            }),
        }
    }
}

/// The result of an [`atomic_partition_replace`].
#[derive(Debug, Clone)]
pub struct ReplaceOutcome {
    /// The committed snapshot id (the pre-existing current snapshot for a
    /// no-op or dry run).
    pub snapshot_id: Option<i64>,
    /// Rows appended (= rows in the input batches).
    pub rows_appended: u64,
    /// Equality-delete tuples written (= rows; an ESTIMATE of rows deleted —
    /// tuples with no prior match delete nothing).
    pub delete_tuples: u64,
    /// Commit attempts used (0 for dry runs and no-ops).
    pub attempts: u32,
}

/// Atomically replace one key-chunk of an identity partition: equality-delete
/// the chunk's PRIOR rows (`partition_column = partition_value`, pk in the
/// input rows' keys) and append the input rows, as ONE snapshot.
///
/// Contracts:
/// - `equality_columns` must be TOP-LEVEL, non-float table columns (the
///   partition scope replaces any nested discriminator column — see the
///   sync-engine RFC).
/// - the table's default partition spec must be a single IDENTITY partition
///   on `partition_column`; the replacement rows must all belong to
///   `partition_value`'s partition.
/// - input batches are conformed to the table schema BY NAME (field-id
///   metadata attached, primitives cast, structs realigned recursively);
///   string columns targeting a VARIANT column are parsed as JSON via the
///   arrow variant kernel. [`ReplaceInput::LocalParquet`] streams the file
///   batch-wise — memory stays O(batch) regardless of chunk size.
/// - `dry_run` stops after validation + conformance: nothing is written,
///   nothing committed; the outcome carries the would-be counts.
#[allow(clippy::too_many_arguments)]
pub async fn atomic_partition_replace(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    partition_column: &str,
    partition_value: Literal,
    equality_columns: &[String],
    input: impl Into<ReplaceInput>,
    max_attempts: u32,
    dry_run: bool,
) -> Result<ReplaceOutcome> {
    let input = input.into();
    let table = catalog.load_table(ident).await?;
    let schema = table.metadata().current_schema().clone();

    // Equality ids: top-level, by name, against the LOADED schema (create
    // paths reassign field ids — never trust creation-time constants).
    let mut equality_ids = Vec::with_capacity(equality_columns.len());
    for col in equality_columns {
        let field_id = schema.field_id_by_name(col).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("equality column `{col}` not found in table schema"),
            )
        })?;
        if schema.as_struct().field_by_id(field_id).is_none() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "equality column `{col}` is not a TOP-LEVEL column — nested equality ids \
                     are not supported (scope by partition instead)"
                ),
            ));
        }
        equality_ids.push(field_id);
    }

    // The identity partition this replace is scoped to.
    let spec = table.metadata().default_partition_spec().clone();
    let ok_spec = spec.fields().len() == 1
        && spec.fields()[0].transform == Transform::Identity
        && schema
            .field_id_by_name(partition_column)
            .is_some_and(|id| id == spec.fields()[0].source_id);
    if !ok_spec {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "atomic replace requires a single-field IDENTITY partition on \
                 `{partition_column}`; the table's default spec is {spec:?}"
            ),
        ));
    }
    let partition_key = PartitionKey::new(
        spec.as_ref().clone(),
        schema.clone(),
        Struct::from_iter(vec![Some(partition_value)]),
    );

    // Conform the input to the table's arrow schema (names -> field ids,
    // casts, struct realignment, JSON -> VARIANT).
    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema)?);

    if dry_run {
        // Conform-only validation pass — nothing written, nothing committed.
        let mut rows = 0u64;
        for batch in input.pass()? {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            rows += conform_batch(&batch, &target_arrow)?.num_rows() as u64;
        }
        return Ok(ReplaceOutcome {
            snapshot_id: table.metadata().current_snapshot_id(),
            rows_appended: rows,
            delete_tuples: rows,
            attempts: 0,
        });
    }

    // Pass 1 — the replacement data files (into the scoped partition),
    // streamed. The writer opens LAZILY on the first non-empty batch so a
    // zero-row input never creates writers.
    let run = Uuid::now_v7();
    let mut data_writer = None;
    let mut rows_appended = 0u64;
    for batch in input.pass()? {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let conformed = conform_batch(&batch, &target_arrow)?;
        rows_appended += conformed.num_rows() as u64;
        if data_writer.is_none() {
            let data_rolling = RollingFileWriterBuilder::new_with_default_file_size(
                ParquetWriterBuilder::new(writer_properties(&table), schema.clone()),
                table.file_io().clone(),
                DefaultLocationGenerator::new(table.metadata())?,
                DefaultFileNameGenerator::new(
                    format!("replace-{run}"),
                    None,
                    DataFileFormat::Parquet,
                ),
            );
            data_writer = Some(
                DataFileWriterBuilder::new(data_rolling)
                    .build(Some(partition_key.clone()))
                    .await?,
            );
        }
        data_writer
            .as_mut()
            .expect("writer opened above")
            .write(conformed)
            .await?;
    }
    if rows_appended == 0 {
        return Ok(ReplaceOutcome {
            snapshot_id: table.metadata().current_snapshot_id(),
            rows_appended: 0,
            delete_tuples: 0,
            attempts: 0,
        });
    }
    let data_files = data_writer.expect("rows_appended > 0").close().await?;

    // Pass 2 — the partition-scoped equality-delete file: the delete tuples
    // are the replacement rows' own keys (the writer projects the
    // equality-id columns; its parquet schema is the PROJECTED equality-id
    // schema). Re-streams the input.
    let eq_config = EqualityDeleteWriterConfig::new(equality_ids, schema.clone())?;
    let delete_schema = Arc::new(arrow_schema_to_schema(
        eq_config.projected_arrow_schema_ref(),
    )?);
    let eq_rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            delete_schema,
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata())?,
        DefaultFileNameGenerator::new(format!("replace-del-{run}"), None, DataFileFormat::Parquet),
    );
    let mut eq_writer = EqualityDeleteFileWriterBuilder::new(eq_rolling, eq_config)
        .build(Some(partition_key))
        .await?;
    for batch in input.pass()? {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        eq_writer
            .write(conform_batch(&batch, &target_arrow)?)
            .await?;
    }
    let delete_files = eq_writer.close().await?;

    // Commit as ONE RowDelta snapshot; on commit conflict, RERUN with the
    // SAME files from a fresh base.
    let max_attempts = max_attempts.max(1);
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let table = catalog.load_table(ident).await?;
        let tx = Transaction::new(&table);
        let mut action = tx
            .row_delta()
            .add_data_files(data_files.clone())
            .add_delete_files(delete_files.clone());
        action = match table.metadata().current_snapshot_id() {
            Some(base) => action.validate_from_snapshot(base),
            None => action.validate_from_empty_table(),
        };
        match async { action.apply(tx)?.commit(catalog).await }.await {
            Ok(committed) => {
                return Ok(ReplaceOutcome {
                    snapshot_id: committed.metadata().current_snapshot_id(),
                    rows_appended,
                    delete_tuples: rows_appended,
                    attempts,
                });
            }
            Err(e) if attempts < max_attempts && is_commit_conflict(&e) => {
                continue; // rerun from a fresh base with the same files
            }
            Err(e) => return Err(e),
        }
    }
}

/// A retryable optimistic-concurrency conflict: the catalog's commit
/// conflict (HTTP 409 class) or the RowDelta validator's concurrent-commit
/// rejection.
fn is_commit_conflict(e: &Error) -> bool {
    e.kind() == ErrorKind::CatalogCommitConflicts
        || e.to_string()
            .contains("Found conflicting concurrent commit")
}

/// How a scan-based replace matches a chunk's prior rows against the
/// (stringified) key column. Two shapes, one scan/DV/commit path:
///
/// - [`KeyMatcher::Prefix`] — `starts_with(key, prefix)`; the ObjectId-hex
///   chunk shape (`{"$oid": "<hex>…`).
/// - [`KeyMatcher::I64Range`] — inclusive numeric range on a stringified
///   integer key (`str(_id)` for document stores with Int64 keys). A
///   numeric range is inexpressible as a string prefix, and lexicographic
///   string ordering diverges from numeric ordering across digit lengths —
///   so the row-wise match parses each key as i64 (unparseable keys never
///   match). For PLANNING, when `lo` and `hi` have the same digit count
///   (and are non-negative) a lexicographic string-range predicate is
///   emitted: over equal-length decimal strings lex == numeric, and every
///   numeric match has that length, so the bounds are a SUPERSET of the
///   numeric range (shorter/longer keys that lex-fall inside are false
///   positives the row filter drops). Mixed-length bounds plan on the
///   partition + op predicates only (correct, less pruning).
#[derive(Debug, Clone)]
pub enum KeyMatcher {
    /// `starts_with(key_column, .0)`.
    Prefix(String),
    /// `.0 <= parse::<i64>(key_column) <= .1`, inclusive.
    I64Range(i64, i64),
}

impl KeyMatcher {
    fn planning_predicate(&self, key_column: &str) -> Option<Predicate> {
        match self {
            KeyMatcher::Prefix(p) => {
                Some(Reference::new(key_column).starts_with(Datum::string(p.clone())))
            }
            KeyMatcher::I64Range(lo, hi) => {
                let (lo_s, hi_s) = (lo.to_string(), hi.to_string());
                if *lo >= 0 && lo_s.len() == hi_s.len() {
                    Some(
                        Reference::new(key_column)
                            .greater_than_or_equal_to(Datum::string(lo_s))
                            .and(
                                Reference::new(key_column)
                                    .less_than_or_equal_to(Datum::string(hi_s)),
                            ),
                    )
                } else {
                    None
                }
            }
        }
    }

    fn matches(&self, key: &str) -> bool {
        match self {
            KeyMatcher::Prefix(p) => key.starts_with(p.as_str()),
            KeyMatcher::I64Range(lo, hi) => key
                .trim()
                .parse::<i64>()
                .is_ok_and(|k| *lo <= k && k <= *hi),
        }
    }
}

/// Atomically replace one KEY-PREFIX chunk of an identity partition — the
/// envelope-shaped (mongo) bronze variant of [`atomic_partition_replace`]:
/// there is no top-level pk column to equality-delete on, and the chunk's
/// scope is a PREFIX of a nested string key (`_cdc.key LIKE '{"$oid":
/// "<hex>%'`), which no equality-delete file can express. Instead the prior
/// rows are deleted by SCAN + DELETION VECTOR — the same delete class the
/// DuckDB path writes today, so every existing bronze reader applies them
/// by construction:
///
/// 1. scan the pinned snapshot for `partition_column = partition_value AND
///    starts_with(key_column, key_prefix) [AND op_column IN op_values]`,
///    projecting only `_file`/`_pos` (the reader applies prior DVs, so
///    matched positions are disjoint from already-deleted ones);
/// 2. per affected file: consolidated DV = prior live DV positions ∪
///    matched positions (ONE DV per data file — the V3 invariant), prior
///    delete files superseded via `remove_delete_files`;
/// 3. append the replacement rows;
/// 4. commit 1–3 as ONE `RowDelta` snapshot at SNAPSHOT isolation
///    (concurrent sink appends are irrelevant to file-referenced DVs;
///    double-DV and referenced-file-removal conflicts still fail closed).
///
/// Concurrency: a conflict retries by RE-SCAN — unlike the equality mode,
/// the DVs reference specific files/positions of the pinned snapshot, so a
/// rerun must re-derive them against the fresh base (the replacement DATA
/// files are reused). Failed attempts orphan their DV puffins —
/// `remove_orphan_files`' job.
///
/// Semantics parity with the duck `DELETE ... LIKE` path: a source doc that
/// VANISHED since the last backfill loses its baseline row (prefix-scoped
/// delete, nothing re-inserts it) — stronger than the equality mode's
/// keep-until-CDC-tombstone. A ZERO-ROW input still deletes (the prefix's
/// docs are gone at source); `dry_run` stops after validation + conformance
/// (no scan, no writes; `delete_tuples` reports 0).
///
/// Fail-closed guards: any attached EQUALITY delete file, or a positional
/// delete file WITHOUT a `referenced_data_file` binding (could span data
/// files — superseding it would resurrect other files' deletes), errors
/// `FeatureUnsupported` — compact the table first.
#[allow(clippy::too_many_arguments)]
pub async fn atomic_partition_replace_prefix(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    partition_column: &str,
    partition_value: bool,
    key_column: &str,
    key_prefix: &str,
    op_column: Option<&str>,
    op_values: &[String],
    input: impl Into<ReplaceInput>,
    max_attempts: u32,
    dry_run: bool,
) -> Result<ReplaceOutcome> {
    atomic_partition_replace_scan(
        catalog,
        ident,
        partition_column,
        partition_value,
        key_column,
        &KeyMatcher::Prefix(key_prefix.to_string()),
        op_column,
        op_values,
        input,
        max_attempts,
        dry_run,
    )
    .await
}

/// Atomically replace one NUMERIC-RANGE chunk of an identity partition —
/// the [`atomic_partition_replace_prefix`] sibling for document stores
/// whose keys are Int64 (stringified into the key column, e.g.
/// `_cdc.key = "28753650"`). Hex-prefix chunks are inexpressible for
/// numeric keys (lexicographic ≠ numeric across digit lengths), so the
/// chunk scope is `key_lo <= key <= key_hi` inclusive, matched by parsing
/// each key row-wise (see [`KeyMatcher::I64Range`] for the planning-bounds
/// contract — chunk planners SHOULD emit same-digit-length ranges so file
/// pruning engages). Same scan + consolidated-DV + one-RowDelta semantics,
/// guards, retry, and zero-row-still-deletes behavior as the prefix mode.
#[allow(clippy::too_many_arguments)]
pub async fn atomic_partition_replace_key_range(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    partition_column: &str,
    partition_value: bool,
    key_column: &str,
    key_lo: i64,
    key_hi: i64,
    op_column: Option<&str>,
    op_values: &[String],
    input: impl Into<ReplaceInput>,
    max_attempts: u32,
    dry_run: bool,
) -> Result<ReplaceOutcome> {
    if key_lo > key_hi {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("key range is inverted: lo {key_lo} > hi {key_hi}"),
        ));
    }
    atomic_partition_replace_scan(
        catalog,
        ident,
        partition_column,
        partition_value,
        key_column,
        &KeyMatcher::I64Range(key_lo, key_hi),
        op_column,
        op_values,
        input,
        max_attempts,
        dry_run,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn atomic_partition_replace_scan(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    partition_column: &str,
    partition_value: bool,
    key_column: &str,
    matcher: &KeyMatcher,
    op_column: Option<&str>,
    op_values: &[String],
    input: impl Into<ReplaceInput>,
    max_attempts: u32,
    dry_run: bool,
) -> Result<ReplaceOutcome> {
    let input = input.into();
    let table = catalog.load_table(ident).await?;
    let schema = table.metadata().current_schema().clone();

    // The scan filter references the key column by (possibly nested) name —
    // resolve it against the LOADED schema to fail fast on typos.
    if schema.field_id_by_name(key_column).is_none() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("key column `{key_column}` not found in table schema"),
        ));
    }
    if let Some(op_col) = op_column
        && schema.field_id_by_name(op_col).is_none()
    {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("op column `{op_col}` not found in table schema"),
        ));
    }

    // Same single-field IDENTITY-partition contract as the equality mode.
    let spec = table.metadata().default_partition_spec().clone();
    let ok_spec = spec.fields().len() == 1
        && spec.fields()[0].transform == Transform::Identity
        && schema
            .field_id_by_name(partition_column)
            .is_some_and(|id| id == spec.fields()[0].source_id);
    if !ok_spec {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "atomic replace requires a single-field IDENTITY partition on \
                 `{partition_column}`; the table's default spec is {spec:?}"
            ),
        ));
    }
    let partition_key = PartitionKey::new(
        spec.as_ref().clone(),
        schema.clone(),
        Struct::from_iter(vec![Some(Literal::bool(partition_value))]),
    );

    // Conform + write the replacement data files ONCE (reused across commit
    // attempts), streamed — the writer opens LAZILY on the first non-empty
    // batch (zero rows -> no data files, the commit may be delete-only).
    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema)?);

    if dry_run {
        let mut rows = 0u64;
        for batch in input.pass()? {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            rows += conform_batch(&batch, &target_arrow)?.num_rows() as u64;
        }
        return Ok(ReplaceOutcome {
            snapshot_id: table.metadata().current_snapshot_id(),
            rows_appended: rows,
            delete_tuples: 0,
            attempts: 0,
        });
    }

    let run = Uuid::now_v7();
    let mut data_writer = None;
    let mut rows_appended = 0u64;
    for batch in input.pass()? {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let conformed = conform_batch(&batch, &target_arrow)?;
        rows_appended += conformed.num_rows() as u64;
        if data_writer.is_none() {
            let data_rolling = RollingFileWriterBuilder::new_with_default_file_size(
                ParquetWriterBuilder::new(writer_properties(&table), schema.clone()),
                table.file_io().clone(),
                DefaultLocationGenerator::new(table.metadata())?,
                DefaultFileNameGenerator::new(
                    format!("replace-{run}"),
                    None,
                    DataFileFormat::Parquet,
                ),
            );
            data_writer = Some(
                DataFileWriterBuilder::new(data_rolling)
                    .build(Some(partition_key.clone()))
                    .await?,
            );
        }
        data_writer
            .as_mut()
            .expect("writer opened above")
            .write(conformed)
            .await?;
    }
    let data_files = match data_writer {
        Some(mut w) => w.close().await?,
        None => Vec::new(),
    };

    let max_attempts = max_attempts.max(1);
    let mut attempts = 0u32;
    let mut table = table;
    loop {
        attempts += 1;

        let Some(base_snapshot) = table.metadata().current_snapshot_id() else {
            // Empty table: nothing to delete — commit the appends (if any).
            if data_files.is_empty() {
                return Ok(ReplaceOutcome {
                    snapshot_id: None,
                    rows_appended,
                    delete_tuples: 0,
                    attempts: attempts - 1,
                });
            }
            let tx = Transaction::new(&table);
            let action = tx
                .row_delta()
                .add_data_files(data_files.clone())
                .validate_from_empty_table();
            match async { action.apply(tx)?.commit(catalog).await }.await {
                Ok(committed) => {
                    return Ok(ReplaceOutcome {
                        snapshot_id: committed.metadata().current_snapshot_id(),
                        rows_appended,
                        delete_tuples: 0,
                        attempts,
                    });
                }
                Err(e) if attempts < max_attempts && is_commit_conflict(&e) => {
                    table = catalog.load_table(ident).await?;
                    continue;
                }
                Err(e) => return Err(e),
            }
        };

        // 1. Scan the pinned snapshot for the chunk's prior rows. The full
        //    filter drives PLANNING (partition pruning + manifest/file
        //    bounds work on nested field ids); it is STRIPPED from the read
        //    tasks below — the arrow row filter rejects nested (non-root)
        //    predicate columns — and re-applied row-wise in the collect
        //    loop over the projected key/op ancestor column.
        let mut filter = Reference::new(partition_column).equal_to(Datum::bool(partition_value));
        if let Some(key_pred) = matcher.planning_predicate(key_column) {
            filter = filter.and(key_pred);
        }
        if let Some(op_col) = op_column {
            filter = filter.and(
                Reference::new(op_col).is_in(op_values.iter().map(|v| Datum::string(v.clone()))),
            );
        }
        let key_root = key_column.split('.').next().expect("non-empty key column");
        let mut select_cols = vec![key_root, RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS];
        if let Some(op_col) = op_column {
            let op_root = op_col.split('.').next().expect("non-empty op column");
            if op_root != key_root {
                select_cols.insert(1, op_root);
            }
        }
        let scan = table
            .scan()
            .snapshot_id(base_snapshot)
            .with_filter(filter)
            .select(select_cols)
            .build()?;
        let tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await?
            .try_collect::<Vec<FileScanTask>>()
            .await?
            .into_iter()
            .map(|mut t| {
                t.predicate = None; // row-wise filtering happens below
                t
            })
            .collect();

        // Fail-closed delete-file guards (see docstring).
        for task in &tasks {
            for del in &task.deletes {
                if del.file_type == DataContentType::EqualityDeletes {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "prefix replace cannot supersede EQUALITY delete file {} — \
                             equality deletes are undecidable per-file; compact first",
                            del.file_path
                        ),
                    ));
                }
                if del.file_type == DataContentType::PositionDeletes
                    && del.referenced_data_file.is_none()
                {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "prefix replace cannot supersede positional delete file {} \
                             carrying no referenced_data_file (it may span data files); \
                             compact first",
                            del.file_path
                        ),
                    ));
                }
            }
        }

        // 2. Read matched rows' (file, pos) — prior DVs applied by the reader.
        let reader = table.reader_builder().build();
        let task_stream = Box::pin(futures::stream::iter(
            tasks.iter().cloned().map(Ok).collect::<Vec<_>>(),
        )) as crate::scan::FileScanTaskStream;
        // The clone shares the reader's delete cache — prior DVs loaded for
        // this read are reused by `load_positional_deletes` below.
        let mut stream = reader.clone().read(task_stream)?.stream();
        let mut positions_by_file: HashMap<String, Vec<u64>> = HashMap::new();
        let mut matched: u64 = 0;
        while let Some(batch) = stream.try_next().await? {
            if batch.num_rows() == 0 {
                continue;
            }
            let schema = batch.schema();
            let file_idx = schema.index_of(RESERVED_COL_NAME_FILE).map_err(|e| {
                Error::new(ErrorKind::Unexpected, format!("_file column missing: {e}"))
            })?;
            let pos_idx = schema.index_of(RESERVED_COL_NAME_POS).map_err(|e| {
                Error::new(ErrorKind::Unexpected, format!("_pos column missing: {e}"))
            })?;
            let files = arrow_cast::cast(batch.column(file_idx).as_ref(), &DataType::Utf8)
                .map_err(|e| {
                    Error::new(ErrorKind::Unexpected, format!("_file cast failed: {e}"))
                })?;
            let files = files
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("cast to Utf8");
            let positions = batch
                .column(pos_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::new(ErrorKind::Unexpected, "_pos must be Int64"))?;
            let keys = string_leaf(&batch, key_column)?;
            let ops = op_column.map(|c| string_leaf(&batch, c)).transpose()?;
            for row in 0..batch.num_rows() {
                if keys.is_null(row) || !matcher.matches(keys.value(row)) {
                    continue;
                }
                if let Some(ops) = &ops
                    && !op_values.iter().any(|v| v == ops.value(row))
                {
                    continue;
                }
                positions_by_file
                    .entry(files.value(row).to_string())
                    .or_default()
                    .push(positions.value(row) as u64);
                matched += 1;
            }
        }

        if matched == 0 && data_files.is_empty() {
            return Ok(ReplaceOutcome {
                snapshot_id: Some(base_snapshot),
                rows_appended,
                delete_tuples: 0,
                attempts: attempts - 1,
            });
        }

        // 3. One consolidated DV per affected file (prior live positions ∪
        //    matched), superseding that file's prior delete files.
        let affected_tasks: Vec<FileScanTask> = tasks
            .iter()
            .filter(|t| positions_by_file.contains_key(t.data_file_path()))
            .cloned()
            .collect();
        let mut prior_deletes: HashMap<String, DeleteVector> =
            reader.load_positional_deletes(&affected_tasks).await?;
        let location = table.metadata().location().to_string();
        let mut new_delete_files: Vec<DataFile> = Vec::with_capacity(affected_tasks.len());
        for task in &affected_tasks {
            let file = task.data_file_path();
            let mut bitmap: RoaringTreemap = prior_deletes
                .remove(file)
                .map(DeleteVector::into_inner)
                .unwrap_or_default();
            for pos in positions_by_file.get(file).into_iter().flatten() {
                bitmap.insert(*pos);
            }
            let partition = task.partition.clone().unwrap_or(Struct::empty());
            let spec_id = task
                .partition_spec
                .as_ref()
                .map(|s| s.spec_id())
                .unwrap_or_else(|| table.metadata().default_partition_spec_id());
            let dv_path = format!("{location}/data/{}-deletes.puffin", Uuid::now_v7());
            new_delete_files.push(
                DeleteVector::new(bitmap)
                    .write_to_puffin_file(
                        table.file_io(),
                        dv_path,
                        file.to_string(),
                        partition,
                        spec_id,
                    )
                    .await?,
            );
        }
        let prior_paths: HashSet<String> = affected_tasks
            .iter()
            .flat_map(|t| t.deletes.iter().map(|d| d.file_path.clone()))
            .collect();
        let removed_delete_files = if prior_paths.is_empty() {
            Vec::new()
        } else {
            resolve_delete_files(&table, base_snapshot, &prior_paths).await?
        };

        // 4. ONE RowDelta snapshot, snapshot-isolation validated.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(data_files.clone())
            .add_delete_files(new_delete_files)
            .remove_delete_files(removed_delete_files)
            .validate_from_snapshot(base_snapshot)
            .with_snapshot_isolation();
        match async { action.apply(tx)?.commit(catalog).await }.await {
            Ok(committed) => {
                return Ok(ReplaceOutcome {
                    snapshot_id: committed.metadata().current_snapshot_id(),
                    rows_appended,
                    delete_tuples: matched,
                    attempts,
                });
            }
            Err(e) if attempts < max_attempts && is_commit_conflict(&e) => {
                // RE-SCAN against the fresh base (positions/files/prior DVs
                // may have changed); this attempt's DV puffins are orphans.
                table = catalog.load_table(ident).await?;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Extract a (possibly nested) string leaf from a record batch by dotted
/// path — `_cdc.key` walks the `_cdc` struct column to its `key` child.
/// The leaf may decode as Utf8 or LargeUtf8 (reader offset widening); the
/// wide form is folded back to Utf8 — key/op values are small, so the
/// per-batch cast is bounded. Non-string leaves still refuse: the replace
/// contract requires STRING key and op columns.
fn string_leaf(batch: &RecordBatch, dotted: &str) -> Result<StringArray> {
    let mut parts = dotted.split('.');
    let root = parts.next().expect("non-empty dotted path");
    let mut current: &ArrayRef = batch.column(
        batch
            .schema()
            .index_of(root)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("column `{root}`: {e}")))?,
    );
    for part in parts {
        let s = current
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("`{dotted}`: `{part}`'s parent is not a struct"),
                )
            })?;
        current = s.column_by_name(part).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("`{dotted}`: struct field `{part}` not found"),
            )
        })?;
    }
    if !matches!(
        current.data_type(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) {
        return Err(Error::new(
            ErrorKind::Unexpected,
            format!("`{dotted}` must be a string column"),
        ));
    }
    let utf8 = arrow_cast::cast(current.as_ref(), &DataType::Utf8)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("`{dotted}` cast: {e}")))?;
    utf8.as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("`{dotted}` must be a string column"),
            )
        })
}

/// Resolve live delete-file paths of `snapshot_id` to their manifest
/// `DataFile` entries (needed for `remove_delete_files` — a scan task's
/// `FileScanTaskDeleteFile` is not a full manifest entry).
async fn resolve_delete_files(
    table: &Table,
    snapshot_id: i64,
    paths: &HashSet<String>,
) -> Result<Vec<DataFile>> {
    let metadata = table.metadata();
    let snapshot = metadata.snapshot_by_id(snapshot_id).ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            format!("snapshot {snapshot_id} not found"),
        )
    })?;
    let bytes = table
        .file_io()
        .new_input(snapshot.manifest_list())?
        .read()
        .await?;
    let manifest_list = ManifestList::parse_with_version(&bytes, metadata.format_version())?;
    let mut out = Vec::new();
    for mf in manifest_list.entries() {
        if mf.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = mf.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() && paths.contains(entry.data_file().file_path()) {
                out.push(entry.data_file().clone());
            }
        }
    }
    Ok(out)
}

/// Conform `batch` to the table's arrow schema: resolve columns BY NAME,
/// attach the target fields (field-id metadata), realign struct children
/// recursively, cast primitives, and parse string columns targeting VARIANT
/// as JSON via the arrow variant kernel.
pub fn conform_batch(batch: &RecordBatch, target: &ArrowSchemaRef) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(target.fields().len());
    for field in target.fields() {
        let idx = batch.schema().index_of(field.name()).map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("input batch is missing column `{}`", field.name()),
            )
        })?;
        columns.push(conform_array(batch.column(idx), field)?);
    }
    RecordBatch::try_new(Arc::new(ArrowSchema::new(target.fields().clone())), columns).map_err(
        |e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("conforming input batch to the table schema: {e}"),
            )
        },
    )
}

fn conform_array(array: &ArrayRef, target: &Field) -> Result<ArrayRef> {
    if array.data_type() == target.data_type() {
        return Ok(Arc::clone(array));
    }
    // JSON string -> VARIANT.
    if is_variant_type(target.data_type())
        && matches!(
            array.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    {
        return json_strings_to_variant(array, target);
    }
    // Struct: realign children by name recursively.
    if let (DataType::Struct(target_fields), DataType::Struct(_)) =
        (target.data_type(), array.data_type())
    {
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("checked struct");
        let mut children = Vec::with_capacity(target_fields.len());
        for child_field in target_fields {
            let child = struct_array
                .column_by_name(child_field.name())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("input struct is missing field `{}`", child_field.name()),
                    )
                })?;
            children.push(conform_array(child, child_field)?);
        }
        let out = StructArray::try_new(
            target_fields.clone(),
            children,
            struct_array.nulls().cloned(),
        )
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("conforming struct column: {e}"),
            )
        })?;
        return Ok(Arc::new(out));
    }
    // Primitives and everything else: arrow cast.
    arrow_cast::cast(array.as_ref(), target.data_type()).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "cannot convert input column `{}` from {:?} to {:?}: {e}",
                target.name(),
                array.data_type(),
                target.data_type()
            ),
        )
    })
}

/// The iceberg-arrow canonical VARIANT rendering: a struct of exactly
/// `metadata: Binary` + `value: Binary`.
fn is_variant_type(dt: &DataType) -> bool {
    match dt {
        DataType::Struct(fields) => {
            fields.len() == 2
                && fields.iter().any(|f| f.name() == "metadata")
                && fields.iter().any(|f| f.name() == "value")
                && fields.iter().all(|f| f.data_type() == &DataType::Binary)
        }
        _ => false,
    }
}

/// Parse a string column of JSON documents into the canonical VARIANT
/// struct layout (`metadata`/`value` Binary), preserving nulls, via the
/// arrow variant kernel — the same kernel the `json_to_variant` SQL UDF
/// wraps.
fn json_strings_to_variant(array: &ArrayRef, target: &Field) -> Result<ArrayRef> {
    let strings = arrow_cast::cast(array.as_ref(), &DataType::Utf8)
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("casting JSON input: {e}")))?;
    let variant = json_to_variant(&strings).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("parsing JSON as variant: {e}"),
        )
    })?;
    // Re-emit through the builder for a canonical layout, then coerce the
    // metadata/value children to plain Binary under the TARGET fields.
    let variant = {
        let mut builder = VariantArrayBuilder::new(variant.len());
        for i in 0..variant.len() {
            if variant.is_null(i) {
                builder.append_null();
            } else {
                builder.append_variant(variant.value(i));
            }
        }
        builder.build()
    };
    let inner: StructArray = variant.into_inner();
    let metadata = inner
        .column_by_name("metadata")
        .ok_or_else(|| Error::new(ErrorKind::Unexpected, "variant missing `metadata`"))?;
    let value = inner
        .column_by_name("value")
        .ok_or_else(|| Error::new(ErrorKind::Unexpected, "variant missing `value`"))?;
    let metadata = arrow_cast::cast(metadata.as_ref(), &DataType::Binary)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("variant metadata cast: {e}")))?;
    let value = arrow_cast::cast(value.as_ref(), &DataType::Binary)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("variant value cast: {e}")))?;
    let DataType::Struct(target_fields) = target.data_type() else {
        unreachable!("checked variant type");
    };
    let out = StructArray::try_new(
        target_fields.clone(),
        vec![metadata, value],
        inner.nulls().cloned(),
    )
    .map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("building variant column: {e}"),
        )
    })?;
    Ok(Arc::new(out))
}

/// Writer properties honoring the table's `write.parquet.compression-codec`
/// (defaulting to ZSTD, the iceberg convention — parquet-rs's own default is
/// UNCOMPRESSED, which would silently inflate output ~20x).
fn writer_properties(table: &Table) -> parquet::file::properties::WriterProperties {
    use parquet::basic::{Compression, GzipLevel, ZstdLevel};
    let codec = table
        .metadata()
        .properties()
        .get("write.parquet.compression-codec")
        .map(|s| s.to_ascii_lowercase());
    let compression = match codec.as_deref() {
        Some("uncompressed") => Compression::UNCOMPRESSED,
        Some("snappy") => Compression::SNAPPY,
        Some("gzip") => Compression::GZIP(GzipLevel::default()),
        Some("lz4") => Compression::LZ4,
        _ => Compression::ZSTD(ZstdLevel::default()),
    };
    parquet::file::properties::WriterProperties::builder()
        .set_compression(compression)
        .build()
}
