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

//! Copy-on-write replacement of the rows of NAMED data files with rows the
//! caller supplies — for a caller that computes new column values outside the
//! engine (a value only the caller knows how to render) and must put them back
//! without changing the set of rows.
//!
//! For each named live data file of the planning snapshot the caller hands the
//! file's complete replacement rows. The rows are conformed to the table schema
//! BY NAME (string columns targeting a VARIANT column are parsed as JSON), the
//! row count must equal the file's LIVE rows (its record count minus the rows
//! its delete files remove), identity-partition columns must carry the file's
//! own partition values, and one `Replace` snapshot removes the named files and
//! the delete files bound to them and adds the new files. Files not named are
//! untouched.
//!
//! Refusals (nothing is written or committed):
//!   * the table is not at `base_snapshot` (planning snapshot) when the call
//!     starts;
//!   * at commit, a snapshot committed after `base_snapshot` is anything but a
//!     plain append that leaves the replaced files alone (see below);
//!   * a named path is not a live data file, or was written under an older
//!     partition spec;
//!   * a delete file bound to a named file is an equality delete, or is also
//!     bound to a file that is not named (removing it would resurrect rows there);
//!   * a replacement's row count differs from the file's live rows;
//!   * a replacement row's identity-partition value differs from the file's.
//!
//! Appends that land while the replacement is written do not refuse it: the
//! commit is re-applied on the current snapshot when every snapshot committed
//! after `base_snapshot` is an `append` that adds no delete file and removes no
//! file, the table keeps its schema, default partition spec and default sort
//! order, and every replaced file (and delete file removed with it) still has
//! its base manifest entry (size, record count, data sequence number). The
//! transaction's own retry re-applies on a commit conflict, at most
//! [`REPLACE_COMMIT_MAX_ATTEMPTS`] attempts. Each passing proof is listed in
//! [`ReplaceFileRowsOutcome::rebase_rechecks`]; a failing one refuses with the
//! stale-snapshot error and commits nothing.
//!
//! With [`BoundFloor`]s (proof 5), every data file those appends added must
//! also rank after the replaced rows: its data sequence number above the
//! planning snapshot's, and for each floor the minimum of the PRESENT column
//! lower bounds among the floor's fields strictly above the floor (none
//! present refuses). Floors on timestamp fields are given in microseconds and
//! scaled for a nanosecond field.
//!
//! The summary properties starting with `carry_summary_prefix` are copied onto
//! the new snapshot from the snapshot it lands on, read at commit time, so the
//! carried values are the table's latest even after appends landed.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, SchemaRef as ArrowSchemaRef};
use futures::TryStreamExt;
use uuid::Uuid;

use crate::arrow::{ArrowReaderBuilder, schema_to_arrow_schema};
use crate::atomic_replace::{conform_batch, writer_properties};
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Literal, ManifestStatus, PartitionKey,
    PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::transaction::{
    AppendRebaseGuard, ApplyTransactionAction, LowerBoundFloor, RebaseLog, RebaseRecheck,
    Transaction,
};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

/// The complete replacement rows of one live data file.
#[derive(Debug, Clone)]
pub struct FileRows {
    /// The data file's path exactly as the table metadata records it.
    pub data_file_path: String,
    /// Every row the file should hold after the replace.
    pub batches: Vec<RecordBatch>,
}

/// A re-openable supply of replacement rows, read one batch at a time.
///
/// [`replace_file_rows_from_sources`] opens a source once to validate its rows
/// and, unless it is a dry run, once more to write them, so rows too many to
/// hold in memory can be streamed from disk instead of given as
/// [`FileRows::batches`]. Every open must yield the same rows in the same order.
pub trait RowSource: Send + Sync + fmt::Debug {
    /// Start a fresh read of every row.
    fn open(&self) -> Result<RowBatches<'_>>;
}

/// The batches one [`RowSource::open`] yields.
pub type RowBatches<'a> = Box<dyn Iterator<Item = Result<RecordBatch>> + Send + 'a>;

impl RowSource for Vec<RecordBatch> {
    fn open(&self) -> Result<RowBatches<'_>> {
        Ok(Box::new(self.iter().cloned().map(Ok)))
    }
}

/// The complete replacement rows of one live data file, read on demand.
#[derive(Debug, Clone)]
pub struct FileRowSource {
    /// The data file's path exactly as the table metadata records it.
    pub data_file_path: String,
    /// Every row the file should hold after the replace.
    pub source: Arc<dyn RowSource>,
}

/// What a replace did (or, on a dry run, would do).
#[derive(Debug, Clone, Default)]
pub struct ReplaceFileRowsOutcome {
    /// The planning snapshot.
    pub snapshot_before: i64,
    /// `None` on a dry run.
    pub snapshot_after: Option<i64>,
    /// Named data files removed.
    pub files_replaced: usize,
    /// Data files written (0 on a dry run).
    pub files_added: usize,
    /// Live rows of the named files at the planning snapshot.
    pub rows_live: u64,
    /// Rows written (or, on a dry run, validated).
    pub rows_written: u64,
    /// Delete files bound to the named files, removed with them.
    pub delete_files_removed: usize,
    /// The appends committed after the planning snapshot that the commit was
    /// proven safe to land on, one per commit attempt that found the table
    /// moved (empty when it landed on the planning snapshot).
    pub rebase_rechecks: Vec<RebaseRecheck>,
}

/// Commit attempts a replace takes, the first included, before it refuses.
pub const REPLACE_COMMIT_MAX_ATTEMPTS: usize = 5;

/// A floor every data file appended after the planning snapshot must clear
/// before the replace lands on it: the minimum of the PRESENT column lower
/// bounds among `fields` (dotted names) must be strictly above `above`.
/// `above` is the field's integer value — microseconds for a timestamp field
/// (scaled for a nanosecond one). Fields of one floor share a unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundFloor {
    /// Dotted field names whose present lower bounds are pooled.
    pub fields: Vec<String>,
    /// The exclusive floor.
    pub above: i64,
}

/// Resolve floors against the planning schema: field ids, and the floor in
/// each field's stored unit.
fn resolve_floors(
    ident: &TableIdent,
    schema: &Schema,
    floors: &[BoundFloor],
) -> Result<Vec<LowerBoundFloor>> {
    let mut out = Vec::with_capacity(floors.len());
    for f in floors {
        if f.fields.is_empty() {
            return Err(invalid(format!("{ident}: a bound floor names no field")));
        }
        let mut ids = Vec::with_capacity(f.fields.len());
        let mut scale: Option<i64> = None;
        for name in &f.fields {
            let field = schema.field_by_name(name).ok_or_else(|| {
                invalid(format!(
                    "{ident}: bound floor field `{name}` is not in the schema"
                ))
            })?;
            let s = match field.field_type.as_ref() {
                Type::Primitive(
                    PrimitiveType::Int
                    | PrimitiveType::Long
                    | PrimitiveType::Timestamp
                    | PrimitiveType::Timestamptz,
                ) => 1,
                Type::Primitive(PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs) => 1000,
                other => {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "{ident}: bound floor field `{name}` is {other}, not an integer or timestamp"
                        ),
                    ));
                }
            };
            if scale.is_some_and(|x| x != s) {
                return Err(invalid(format!(
                    "{ident}: bound floor fields {:?} mix microsecond and nanosecond units",
                    f.fields
                )));
            }
            scale = Some(s);
            ids.push(field.id);
        }
        let above = f.above.checked_mul(scale.unwrap_or(1)).ok_or_else(|| {
            invalid(format!(
                "{ident}: bound floor {} overflows the field unit",
                f.above
            ))
        })?;
        out.push(LowerBoundFloor {
            field_ids: ids,
            above,
        });
    }
    Ok(out)
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::DataInvalid, msg)
}

fn literal_at(array: &ArrayRef, row: usize) -> Result<Option<Literal>> {
    if array.is_null(row) {
        return Ok(None);
    }
    Ok(Some(match array.data_type() {
        DataType::Boolean => Literal::bool(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("bool")
                .value(row),
        ),
        DataType::Int32 => Literal::int(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("int")
                .value(row),
        ),
        DataType::Int64 => Literal::long(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("long")
                .value(row),
        ),
        DataType::Utf8 => Literal::string(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("utf8")
                .value(row),
        ),
        DataType::LargeUtf8 => Literal::string(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("large utf8")
                .value(row),
        ),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "an identity partition column of type {other} is not supported by replace_file_rows"
                ),
            ));
        }
    }))
}

/// Every row of `batch` must carry its file's identity-partition values.
fn check_partition(
    batch: &RecordBatch,
    identity: &[(usize, String)],
    old: &DataFile,
    ident: &TableIdent,
    path: &str,
) -> Result<()> {
    for (pos, col) in identity {
        let want = old.partition().fields().get(*pos).cloned().flatten();
        let idx = batch.schema().index_of(col).map_err(|_| {
            invalid(format!(
                "{ident}: replacement rows lack the partition column `{col}`"
            ))
        })?;
        let arr = batch.column(idx).clone();
        for r in 0..arr.len() {
            if literal_at(&arr, r)? != want {
                return Err(invalid(format!(
                    "{ident}: a replacement row for {path} carries `{col}` = {:?}, the file's partition is {want:?}",
                    literal_at(&arr, r)?
                )));
            }
        }
    }
    Ok(())
}

/// Replace the rows of the named live data files of `ident` with the caller's
/// rows, copy-on-write, in one `Replace` snapshot (see the module docs). With
/// `dry_run` every check runs and nothing is written or committed.
pub async fn replace_file_rows(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    base_snapshot: i64,
    files: Vec<FileRows>,
    dry_run: bool,
    snapshot_properties: HashMap<String, String>,
    carry_summary_prefix: Option<&str>,
) -> Result<ReplaceFileRowsOutcome> {
    let files = files
        .into_iter()
        .map(|f| FileRowSource {
            data_file_path: f.data_file_path,
            source: Arc::new(f.batches) as Arc<dyn RowSource>,
        })
        .collect();
    replace_file_rows_from_sources(
        catalog,
        ident,
        base_snapshot,
        files,
        dry_run,
        snapshot_properties,
        carry_summary_prefix,
    )
    .await
}

/// [`replace_file_rows`] over sources read one batch at a time. Each file's
/// rows are read once to count, conform and check them, retaining nothing;
/// only when every file passed (and it is not a dry run) is each source read
/// again and written. The write refuses when the second read yields a
/// different row count than the first.
pub async fn replace_file_rows_from_sources(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    base_snapshot: i64,
    files: Vec<FileRowSource>,
    dry_run: bool,
    snapshot_properties: HashMap<String, String>,
    carry_summary_prefix: Option<&str>,
) -> Result<ReplaceFileRowsOutcome> {
    replace_file_rows_from_sources_with_floors(
        catalog,
        ident,
        base_snapshot,
        files,
        dry_run,
        snapshot_properties,
        carry_summary_prefix,
        &[],
    )
    .await
}

/// [`replace_file_rows_from_sources`] whose commit also holds every appended
/// data file to `floors` (proof 5; see the module docs). Floors are resolved
/// against the planning schema before anything is read or written.
#[allow(clippy::too_many_arguments)]
pub async fn replace_file_rows_from_sources_with_floors(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    base_snapshot: i64,
    files: Vec<FileRowSource>,
    dry_run: bool,
    snapshot_properties: HashMap<String, String>,
    carry_summary_prefix: Option<&str>,
    floors: &[BoundFloor],
) -> Result<ReplaceFileRowsOutcome> {
    if files.is_empty() {
        return Err(invalid("no files to replace".to_string()));
    }
    let mut named: HashSet<String> = HashSet::new();
    for f in &files {
        if !named.insert(f.data_file_path.clone()) {
            return Err(invalid(format!("{} is named twice", f.data_file_path)));
        }
    }
    let table = catalog.load_table(ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .cloned()
        .ok_or_else(|| invalid(format!("{ident}: no current snapshot")))?;
    let s0 = snapshot.snapshot_id();
    if s0 != base_snapshot {
        return Err(invalid(format!(
            "{ident}: planned against snapshot {base_snapshot} but the table is at {s0}"
        )));
    }
    let schema = table.metadata().current_schema().clone();
    let lower_bound_floors = resolve_floors(ident, &schema, floors)?;
    let spec = table.metadata().default_partition_spec().clone();
    let default_spec_id = table.metadata().default_partition_spec_id();
    // the identity partition columns: (position in the partition struct, column name)
    let mut identity: Vec<(usize, String)> = Vec::new();
    for (i, f) in spec.fields().iter().enumerate() {
        if f.transform != Transform::Identity {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: replace_file_rows supports identity partitions only, found {:?}",
                    f.transform
                ),
            ));
        }
        let name = schema
            .field_by_id(f.source_id)
            .map(|x| x.name.clone())
            .ok_or_else(|| {
                invalid(format!(
                    "{ident}: partition source id {} not in the schema",
                    f.source_id
                ))
            })?;
        identity.push((i, name));
    }

    // live entries of the planning snapshot
    let mut live_data: HashMap<String, DataFile> = HashMap::new();
    let mut live_deletes: HashMap<String, DataFile> = HashMap::new();
    let mut base_entries: HashMap<String, (u64, u64, Option<i64>)> = HashMap::new();
    let manifest_list = table.manifest_list_reader(&snapshot).load().await?;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            let df = entry.data_file().clone();
            base_entries.insert(
                df.file_path().to_string(),
                (
                    df.file_size_in_bytes(),
                    df.record_count(),
                    entry.sequence_number(),
                ),
            );
            match entry.content_type() {
                DataContentType::Data => {
                    live_data.insert(df.file_path().to_string(), df);
                }
                _ => {
                    live_deletes.insert(df.file_path().to_string(), df);
                }
            }
        }
    }
    let mut old_files: Vec<DataFile> = Vec::new();
    for f in &files {
        let df = live_data.get(&f.data_file_path).ok_or_else(|| {
            invalid(format!(
                "{ident}: {} is not a live data file of snapshot {s0}",
                f.data_file_path
            ))
        })?;
        if df.partition_spec_id != default_spec_id {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: {} was written under partition spec {} (default {default_spec_id})",
                    f.data_file_path, df.partition_spec_id
                ),
            ));
        }
        old_files.push(df.clone());
    }

    // the scan's own binding of delete files to data files
    let scan = table.scan().snapshot_id(s0).select_all().build()?;
    let mut tasks_by_file: HashMap<String, Vec<FileScanTask>> = HashMap::new();
    let mut bound_to: HashMap<String, HashSet<String>> = HashMap::new();
    let mut planned = scan.plan_files().await?;
    while let Some(task) = planned.try_next().await? {
        for d in &task.deletes {
            bound_to
                .entry(d.file_path.clone())
                .or_default()
                .insert(task.data_file_path.clone());
        }
        if named.contains(&task.data_file_path) {
            tasks_by_file
                .entry(task.data_file_path.clone())
                .or_default()
                .push(task);
        }
    }
    let mut removed_deletes: Vec<DataFile> = Vec::new();
    for (del, data_paths) in &bound_to {
        if !data_paths.iter().any(|p| named.contains(p)) {
            continue;
        }
        let df = live_deletes.get(del).ok_or_else(|| {
            invalid(format!(
                "{ident}: the scan binds delete file {del}, which is not live"
            ))
        })?;
        if df.content_type() == DataContentType::EqualityDeletes {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: equality delete file {del} applies to a named file — reabsorb it first"
                ),
            ));
        }
        if let Some(other) = data_paths.iter().find(|p| !named.contains(*p)) {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("{ident}: delete file {del} also applies to {other}, which is not named"),
            ));
        }
        removed_deletes.push(df.clone());
    }

    // each named file's live rows, read through the scan's own tasks (deletes applied)
    let mut outcome = ReplaceFileRowsOutcome {
        snapshot_before: s0,
        files_replaced: old_files.len(),
        delete_files_removed: removed_deletes.len(),
        ..Default::default()
    };
    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema)?);
    // first pass: count, conform and check every replacement row, retaining none
    let mut validated: Vec<(DataFile, Arc<dyn RowSource>, u64)> = Vec::new();
    for (f, old) in files.into_iter().zip(old_files) {
        let tasks = tasks_by_file.remove(&f.data_file_path).unwrap_or_default();
        let live = if tasks.is_empty() {
            0
        } else {
            let stream =
                Box::pin(futures::stream::iter(tasks.into_iter().map(Ok))) as FileScanTaskStream;
            let mut batches =
                ArrowReaderBuilder::new(table.file_io().clone(), table.runtime().clone())
                    .build()
                    .read(stream)?
                    .stream();
            let mut n = 0u64;
            while let Some(b) = batches.try_next().await? {
                n += b.num_rows() as u64;
            }
            n
        };
        let mut given = 0u64;
        for b in f.source.open()? {
            let b = b?;
            if b.num_rows() == 0 {
                continue;
            }
            let c = conform_batch(&b, &target_arrow)?;
            check_partition(&c, &identity, &old, ident, &f.data_file_path)?;
            given += c.num_rows() as u64;
        }
        if given != live {
            return Err(invalid(format!(
                "{ident}: {} has {live} live rows but {given} replacement rows were given",
                f.data_file_path
            )));
        }
        outcome.rows_live += live;
        outcome.rows_written += given;
        validated.push((old, f.source, given));
    }
    if dry_run {
        return Ok(outcome);
    }

    // second pass: read each source again and write it
    let run = Uuid::now_v7();
    let mut new_files: Vec<DataFile> = Vec::new();
    let mut old_files: Vec<DataFile> = Vec::new();
    for (i, (old, source, expected)) in validated.into_iter().enumerate() {
        if expected > 0 {
            let rolling = RollingFileWriterBuilder::new_with_default_file_size(
                ParquetWriterBuilder::new(writer_properties(&table), schema.clone()),
                table.file_io().clone(),
                DefaultLocationGenerator::new(table.metadata())?,
                DefaultFileNameGenerator::new(
                    format!("replace-{run}-f{i}"),
                    None,
                    DataFileFormat::Parquet,
                ),
            );
            let pkey = if identity.is_empty() {
                None
            } else {
                Some(PartitionKey::new(
                    spec.as_ref().clone(),
                    schema.clone(),
                    Struct::from_iter(old.partition().fields().iter().cloned()),
                ))
            };
            let mut writer = DataFileWriterBuilder::new(rolling).build(pkey).await?;
            let mut written = 0u64;
            for b in source.open()? {
                let b = b?;
                if b.num_rows() == 0 {
                    continue;
                }
                let c = conform_batch(&b, &target_arrow)?;
                written += c.num_rows() as u64;
                writer.write(c).await?;
            }
            if written != expected {
                return Err(invalid(format!(
                    "{ident}: {} yielded {expected} replacement rows when validated but {written} when written",
                    old.file_path()
                )));
            }
            new_files.extend(writer.close().await?);
        }
        old_files.push(old);
    }
    outcome.files_added = new_files.len();

    let guard = AppendRebaseGuard {
        base_snapshot_id: s0,
        schema_id: schema.schema_id(),
        default_spec_id,
        default_sort_order_id: table.metadata().default_sort_order_id(),
        removed_entries: old_files
            .iter()
            .chain(removed_deletes.iter())
            .map(|f| {
                let e = base_entries.get(f.file_path()).cloned().ok_or_else(|| {
                    invalid(format!(
                        "{ident}: {} has no manifest entry at {s0}",
                        f.file_path()
                    ))
                })?;
                Ok((f.file_path().to_string(), e))
            })
            .collect::<Result<_>>()?,
        max_attempts: REPLACE_COMMIT_MAX_ATTEMPTS,
        lower_bound_floors,
    };
    let log: RebaseLog = RebaseLog::default();
    let tx = Transaction::new(&table);
    let mut action = tx
        .rewrite_files()
        .delete_data_files(old_files)
        .delete_delete_files(removed_deletes)
        .add_data_files(new_files)
        .rebase_across_appends(guard, log.clone())
        .set_snapshot_properties(snapshot_properties);
    if let Some(prefix) = carry_summary_prefix {
        action = action.carry_parent_summary(prefix);
    }
    let committed = action.apply(tx)?.commit(catalog).await;
    outcome.rebase_rechecks = log.lock().expect("rebase log poisoned").clone();
    let table = committed?;
    outcome.snapshot_after = table.metadata().current_snapshot_id();
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{
        Array, ArrayRef, BooleanArray, Int64Array, LargeStringArray, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::variant::VariantArray;
    use roaring::RoaringTreemap;
    use tempfile::TempDir;

    use super::*;
    use crate::delete_vector::DeleteVector;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{
        FormatVersion, NestedField, PrimitiveType, Schema, Type, UnboundPartitionSpec, VariantType,
    };
    use crate::table::Table;
    use crate::{CatalogBuilder, NamespaceIdent, TableCreation};

    async fn setup(
        version: FormatVersion,
        partitioned: bool,
    ) -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
        let warehouse = TempDir::new().unwrap();
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.path().to_str().unwrap().to_string(),
                    )]),
                )
                .await
                .unwrap(),
        );
        let ns = NamespaceIdent::new("db".to_string());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let mut fields = vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "v", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "flag", Type::Primitive(PrimitiveType::Boolean)).into(),
        ];
        if version == FormatVersion::V3 {
            fields.push(NestedField::optional(4, "doc", Type::Variant(VariantType)).into());
        }
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(fields)
            .build()
            .unwrap();
        let mut b = TableCreation::builder()
            .name("t".to_string())
            .schema(schema)
            .format_version(version)
            .build();
        if partitioned {
            b.partition_spec = Some(
                UnboundPartitionSpec::builder()
                    .add_partition_field(3, "flag", Transform::Identity)
                    .unwrap()
                    .build(),
            );
        }
        let table = catalog.create_table(&ns, b).await.unwrap();
        (
            warehouse,
            catalog,
            TableIdent::new(ns, "t".to_string()),
            table,
        )
    }

    /// Rows (id, v, flag[, doc as JSON text]) in caller form — strings, conformed by the replace.
    fn rows(
        ids: &[i64],
        v: &[Option<&str>],
        flag: Option<bool>,
        doc: Option<&[Option<&str>]>,
    ) -> RecordBatch {
        let mut fields = vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
        ];
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(v.to_vec())),
            Arc::new(BooleanArray::from(vec![flag; ids.len()])),
        ];
        if let Some(d) = doc {
            fields.push(Field::new("doc", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from(d.to_vec())));
        }
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols).unwrap()
    }

    async fn write(table: &Table, name: &str, batch: RecordBatch, flag: Option<bool>) -> DataFile {
        let schema = table.metadata().current_schema().clone();
        let target = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let batch = conform_batch(&batch, &target).unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(table), schema.clone()),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
        );
        let pkey = if table
            .metadata()
            .default_partition_spec()
            .fields()
            .is_empty()
        {
            None
        } else {
            Some(PartitionKey::new(
                table.metadata().default_partition_spec().as_ref().clone(),
                schema,
                Struct::from_iter(vec![flag.map(Literal::bool)]),
            ))
        };
        let mut w = DataFileWriterBuilder::new(rolling)
            .build(pkey)
            .await
            .unwrap();
        w.write(batch).await.unwrap();
        w.close().await.unwrap().into_iter().next().unwrap()
    }

    async fn append(
        catalog: &dyn Catalog,
        table: &Table,
        files: Vec<DataFile>,
        props: HashMap<String, String>,
    ) -> Table {
        let tx = Transaction::new(table);
        tx.fast_append()
            .add_data_files(files)
            .set_snapshot_properties(props)
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    async fn read(table: &Table) -> Vec<(i64, Option<String>)> {
        let mut out = Vec::new();
        let mut s = table
            .scan()
            .select_all()
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        while let Some(b) = s.try_next().await.unwrap() {
            let id = b
                .column(b.schema().index_of("id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let v = b
                .column(b.schema().index_of("v").unwrap())
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap()
                .clone();
            for i in 0..b.num_rows() {
                out.push((id.value(i), v.is_valid(i).then(|| v.value(i).to_string())));
            }
        }
        out.sort();
        out
    }

    #[tokio::test]
    async fn replaces_only_the_named_file_and_carries_the_prefixed_summary() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), None),
            None,
        )
        .await;
        let f2 = write(
            &table,
            "f2",
            rows(&[3], &[Some("keep")], Some(false), None),
            None,
        )
        .await;
        let table = append(
            catalog.as_ref(),
            &table,
            vec![f1.clone(), f2],
            HashMap::from([
                ("ledger.offset.a".to_string(), "41".to_string()),
                ("other".to_string(), "x".to_string()),
            ]),
        )
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1, 2], &[Some("a"), Some("b")], Some(false), None)],
        }];
        let dry = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl.clone(),
            true,
            HashMap::new(),
            Some("ledger."),
        )
        .await
        .unwrap();
        assert_eq!(
            (
                dry.snapshot_after,
                dry.rows_live,
                dry.rows_written,
                dry.files_replaced
            ),
            (None, 2, 2, 1)
        );
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s0),
            "a dry run commits nothing"
        );

        let out = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::from([("mark".to_string(), "1".to_string())]),
            Some("ledger."),
        )
        .await
        .unwrap();
        assert_eq!(
            (out.files_replaced, out.files_added, out.rows_written),
            (1, 1, 2)
        );
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(out.snapshot_after, table.metadata().current_snapshot_id());
        assert_eq!(read(&table).await, vec![
            (1, Some("a".to_string())),
            (2, Some("b".to_string())),
            (3, Some("keep".to_string()))
        ]);
        let summary = &table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties;
        assert_eq!(
            summary.get("ledger.offset.a").map(String::as_str),
            Some("41"),
            "the prefixed key is carried"
        );
        assert_eq!(summary.get("other"), None, "only the prefix is carried");
        assert_eq!(summary.get("mark").map(String::as_str), Some("1"));
        assert_eq!(summary.get("total-records").map(String::as_str), Some("3"));
    }

    #[tokio::test]
    async fn a_stale_base_snapshot_refuses_even_after_a_plain_append() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let f1 = write(&table, "f1", rows(&[1], &[None], Some(false), None), None).await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let f2 = write(&table, "f2", rows(&[2], &[None], Some(false), None), None).await;
        let table = append(catalog.as_ref(), &table, vec![f2], HashMap::new()).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], Some(false), None)],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("planned against snapshot"),
            "{err}"
        );
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s1)
        );
    }

    #[tokio::test]
    async fn a_row_count_mismatch_and_an_unknown_file_refuse() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), None),
            None,
        )
        .await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let short = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], Some(false), None)],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            short,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("2 live rows but 1 replacement rows"),
            "{err}"
        );
        let unknown = vec![FileRows {
            data_file_path: "s3://nowhere/x.parquet".to_string(),
            batches: vec![],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            unknown,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("is not a live data file"), "{err}");
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s0),
            "nothing committed"
        );
    }

    #[tokio::test]
    async fn a_partition_value_that_moves_a_row_refuses() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, true).await;
        let f1 = write(&table, "f1", rows(&[1], &[None], None, None), None).await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let moved = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], Some(true), None)],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            moved,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("the file's partition is"), "{err}");
        // the same row kept in its (null) partition is accepted and lands there
        let kept = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], None, None)],
        }];
        let out = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            kept,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.files_added, 1);
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read(&table).await, vec![(1, Some("a".to_string()))]);
    }

    #[tokio::test]
    async fn json_text_lands_as_a_variant_document() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V3, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), Some(&[None, None])),
            None,
        )
        .await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 2],
                &[Some("a"), None],
                Some(false),
                Some(&[Some(r#"{"k": "x", "n": 12.50}"#), None]),
            )],
        }];
        replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        let table = catalog.load_table(&ident).await.unwrap();
        let mut s = table
            .scan()
            .select_all()
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        let mut docs = Vec::new();
        while let Some(b) = s.try_next().await.unwrap() {
            let ids = b
                .column(b.schema().index_of("id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let va = VariantArray::try_new(b.column(b.schema().index_of("doc").unwrap())).unwrap();
            for i in 0..b.num_rows() {
                let k = if va.is_null(i) {
                    None
                } else {
                    let v = va.value(i);
                    let obj = v.as_object().expect("object");
                    Some(
                        obj.get("k")
                            .expect("k")
                            .as_string()
                            .expect("string")
                            .to_string(),
                    )
                };
                docs.push((ids.value(i), k));
            }
        }
        docs.sort();
        assert_eq!(docs, vec![(1, Some("x".to_string())), (2, None)]);
    }

    #[tokio::test]
    async fn a_deletion_vector_on_the_replaced_file_counts_and_is_removed_with_it() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V3, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(
                &[1, 2, 3],
                &[None, None, None],
                Some(false),
                Some(&[None, None, None]),
            ),
            None,
        )
        .await;
        let f2 = write(
            &table,
            "f2",
            rows(&[4], &[Some("keep")], Some(false), Some(&[None])),
            None,
        )
        .await;
        let table = append(
            catalog.as_ref(),
            &table,
            vec![f1.clone(), f2],
            HashMap::new(),
        )
        .await;
        // delete row position 1 (id 2) of f1 through a deletion vector
        let dv = DeleteVector::new(RoaringTreemap::from_iter([1u64]));
        let loc = format!("{}/data/dv-f1.puffin", table.metadata().location());
        let del = dv
            .write_to_puffin_file(
                table.file_io(),
                loc,
                f1.file_path().to_string(),
                Struct::empty(),
                table.metadata().default_partition_spec_id(),
            )
            .await
            .unwrap();
        let tx = Transaction::new(&table);
        let table = tx
            .row_delta()
            .add_delete_files(vec![del])
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
        let s0 = table.metadata().current_snapshot_id().unwrap();
        assert_eq!(read(&table).await.len(), 3, "the DV hides id 2");
        // three replacement rows are one too many: the file has 2 live rows
        let three = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 2, 3],
                &[Some("a"), Some("b"), Some("c")],
                Some(false),
                Some(&[None, None, None]),
            )],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            three,
            true,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("2 live rows but 3"), "{err}");
        let two = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 3],
                &[Some("a"), Some("c")],
                Some(false),
                Some(&[None, None]),
            )],
        }];
        let out = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            two,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!((out.rows_live, out.delete_files_removed), (2, 1));
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read(&table).await, vec![
            (1, Some("a".to_string())),
            (3, Some("c".to_string())),
            (4, Some("keep".to_string()))
        ]);
        // no delete file survives: the DV left with its data file
        let snap = table.metadata().current_snapshot().unwrap();
        let ml = table.manifest_list_reader(snap).load().await.unwrap();
        let mut live_deletes = 0;
        for mf in ml.entries() {
            for e in mf.load_manifest(table.file_io()).await.unwrap().entries() {
                if e.status() != ManifestStatus::Deleted
                    && e.content_type() != DataContentType::Data
                {
                    live_deletes += 1;
                }
            }
        }
        assert_eq!(live_deletes, 0);
    }

    /// A source that counts its opens and, when `grows`, yields one extra row
    /// from the second open on, like a stream rewritten between the passes.
    #[derive(Debug)]
    struct Counting {
        batches: Vec<RecordBatch>,
        opens: AtomicUsize,
        grows: bool,
    }

    impl Counting {
        fn new(batches: Vec<RecordBatch>, grows: bool) -> Arc<Self> {
            Arc::new(Self {
                batches,
                opens: AtomicUsize::new(0),
                grows,
            })
        }

        fn opens(&self) -> usize {
            self.opens.load(Ordering::SeqCst)
        }
    }

    impl RowSource for Counting {
        fn open(&self) -> Result<RowBatches<'_>> {
            let n = self.opens.fetch_add(1, Ordering::SeqCst);
            let mut out = self.batches.clone();
            if self.grows && n >= 1 {
                out.push(rows(&[99], &[Some("late")], Some(false), None));
            }
            Ok(Box::new(out.into_iter().map(Ok)))
        }
    }

    /// Appends f1 (ids 1 and 2, `v` null) and f2 (id 3) and returns f1 with the table.
    async fn two_files(catalog: &dyn Catalog, table: &Table) -> (DataFile, Table) {
        let f1 = write(
            table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), None),
            Some(false),
        )
        .await;
        let f2 = write(
            table,
            "f2",
            rows(&[3], &[Some("keep")], Some(false), None),
            Some(false),
        )
        .await;
        let table = append(catalog, table, vec![f1.clone(), f2], HashMap::new()).await;
        (f1, table)
    }

    fn two_batches() -> Vec<RecordBatch> {
        vec![
            rows(&[1], &[Some("a")], Some(false), None),
            rows(&[2], &[Some("b")], Some(false), None),
        ]
    }

    #[tokio::test]
    async fn a_source_is_read_twice_and_lands_what_the_batches_entry_lands() {
        let (_wa, cat_a, ident_a, table_a) = setup(FormatVersion::V2, false).await;
        let (f1_a, table_a) = two_files(cat_a.as_ref(), &table_a).await;
        let by_batches = replace_file_rows(
            cat_a.as_ref(),
            &ident_a,
            table_a.metadata().current_snapshot_id().unwrap(),
            vec![FileRows {
                data_file_path: f1_a.file_path().to_string(),
                batches: two_batches(),
            }],
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();

        let (_wb, cat_b, ident_b, table_b) = setup(FormatVersion::V2, false).await;
        let (f1_b, table_b) = two_files(cat_b.as_ref(), &table_b).await;
        let source = Counting::new(two_batches(), false);
        let by_source = replace_file_rows_from_sources(
            cat_b.as_ref(),
            &ident_b,
            table_b.metadata().current_snapshot_id().unwrap(),
            vec![FileRowSource {
                data_file_path: f1_b.file_path().to_string(),
                source: source.clone(),
            }],
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(source.opens(), 2, "one read to validate, one to write");
        let shape = |o: &ReplaceFileRowsOutcome| {
            (
                o.files_replaced,
                o.files_added,
                o.rows_live,
                o.rows_written,
                o.delete_files_removed,
            )
        };
        assert_eq!(shape(&by_source), shape(&by_batches));
        assert_eq!(shape(&by_source), (1, 1, 2, 2, 0));
        assert!(by_source.snapshot_after.is_some());
        let rows_a = read(&cat_a.load_table(&ident_a).await.unwrap()).await;
        let rows_b = read(&cat_b.load_table(&ident_b).await.unwrap()).await;
        assert_eq!(rows_a, rows_b);
        assert_eq!(rows_b, vec![
            (1, Some("a".to_string())),
            (2, Some("b".to_string())),
            (3, Some("keep".to_string()))
        ]);
    }

    #[tokio::test]
    async fn a_dry_run_over_a_source_reads_it_once_and_writes_nothing() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let (f1, table) = two_files(catalog.as_ref(), &table).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let source = Counting::new(two_batches(), false);
        let out = replace_file_rows_from_sources(
            catalog.as_ref(),
            &ident,
            s0,
            vec![FileRowSource {
                data_file_path: f1.file_path().to_string(),
                source: source.clone(),
            }],
            true,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(source.opens(), 1);
        assert_eq!(
            (out.rows_live, out.rows_written, out.files_added),
            (2, 2, 0)
        );
        assert_eq!(out.snapshot_after, None);
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0));
        assert_eq!(read(&table).await, vec![
            (1, None),
            (2, None),
            (3, Some("keep".to_string()))
        ]);
    }

    #[tokio::test]
    async fn a_source_that_grows_between_the_passes_refuses() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let (f1, table) = two_files(catalog.as_ref(), &table).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let source = Counting::new(two_batches(), true);
        let err = replace_file_rows_from_sources(
            catalog.as_ref(),
            &ident,
            s0,
            vec![FileRowSource {
                data_file_path: f1.file_path().to_string(),
                source: source.clone(),
            }],
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("2 replacement rows when validated but 3 when written"),
            "{err}"
        );
        assert_eq!(source.opens(), 2);
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(
            table.metadata().current_snapshot_id(),
            Some(s0),
            "nothing committed"
        );
        assert_eq!(read(&table).await, vec![
            (1, None),
            (2, None),
            (3, Some("keep".to_string()))
        ]);
    }

    #[tokio::test]
    async fn a_short_source_refuses_before_any_write() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let (f1, table) = two_files(catalog.as_ref(), &table).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let source = Counting::new(vec![rows(&[1], &[Some("a")], Some(false), None)], false);
        let err = replace_file_rows_from_sources(
            catalog.as_ref(),
            &ident,
            s0,
            vec![FileRowSource {
                data_file_path: f1.file_path().to_string(),
                source: source.clone(),
            }],
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("2 live rows but 1 replacement rows"),
            "{err}"
        );
        assert_eq!(source.opens(), 1, "refused after the first pass");
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s0)
        );
    }

    #[tokio::test]
    async fn a_source_row_that_moves_partition_refuses() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, true).await;
        let (f1, table) = two_files(catalog.as_ref(), &table).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let source = Counting::new(
            vec![rows(&[1, 2], &[Some("a"), Some("b")], Some(true), None)],
            false,
        );
        let err = replace_file_rows_from_sources(
            catalog.as_ref(),
            &ident,
            s0,
            vec![FileRowSource {
                data_file_path: f1.file_path().to_string(),
                source: source.clone(),
            }],
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("the file's partition is"), "{err}");
        assert_eq!(source.opens(), 1);
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s0)
        );
    }

    #[tokio::test]
    async fn a_source_removes_the_bound_delete_and_a_stale_base_refuses() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V3, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(
                &[1, 2, 3],
                &[None, None, None],
                Some(false),
                Some(&[None, None, None]),
            ),
            None,
        )
        .await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let dv = DeleteVector::new(RoaringTreemap::from_iter([1u64]));
        let loc = format!("{}/data/dv-f1.puffin", table.metadata().location());
        let del = dv
            .write_to_puffin_file(
                table.file_io(),
                loc,
                f1.file_path().to_string(),
                Struct::empty(),
                table.metadata().default_partition_spec_id(),
            )
            .await
            .unwrap();
        let tx = Transaction::new(&table);
        let table = tx
            .row_delta()
            .add_delete_files(vec![del])
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let source = Counting::new(
            vec![rows(
                &[1, 3],
                &[Some("a"), Some("c")],
                Some(false),
                Some(&[None, None]),
            )],
            false,
        );
        let files = vec![FileRowSource {
            data_file_path: f1.file_path().to_string(),
            source: source.clone(),
        }];
        let out = replace_file_rows_from_sources(
            catalog.as_ref(),
            &ident,
            s0,
            files.clone(),
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!((out.rows_live, out.delete_files_removed), (2, 1));
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read(&table).await, vec![
            (1, Some("a".to_string())),
            (3, Some("c".to_string()))
        ]);
        // the same plan again: the head moved, the source is never opened
        let before = source.opens();
        let err = replace_file_rows_from_sources(
            catalog.as_ref(),
            &ident,
            s0,
            files,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("planned against snapshot"),
            "{err}"
        );
        assert_eq!(source.opens(), before);
    }

    // ---- commits landing between the write and the commit ----

    /// A commit another writer lands just before one `update_table` of ours.
    #[allow(clippy::large_enum_variant)]
    enum Interjection {
        Append(DataFile, HashMap<String, String>),
        DeleteFile(DataFile),
        Rewrite { remove: DataFile, add: DataFile },
    }

    /// A catalog that lands the queued interjections, one per `update_table`,
    /// right before forwarding ours — the lane committing while the replace
    /// writes.
    #[derive(Debug)]
    struct Interloper {
        inner: Arc<dyn Catalog>,
        queue: std::sync::Mutex<std::collections::VecDeque<Interjection>>,
        landed: std::sync::Mutex<Vec<i64>>,
    }

    impl fmt::Debug for Interjection {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("Interjection")
        }
    }

    impl Interloper {
        fn new(inner: Arc<dyn Catalog>, queue: Vec<Interjection>) -> Self {
            Self {
                inner,
                queue: std::sync::Mutex::new(queue.into()),
                landed: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn landed(&self) -> Vec<i64> {
            self.landed.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Catalog for Interloper {
        async fn list_namespaces(
            &self,
            parent: Option<&NamespaceIdent>,
        ) -> Result<Vec<NamespaceIdent>> {
            self.inner.list_namespaces(parent).await
        }
        async fn create_namespace(
            &self,
            namespace: &NamespaceIdent,
            properties: HashMap<String, String>,
        ) -> Result<crate::Namespace> {
            self.inner.create_namespace(namespace, properties).await
        }
        async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<crate::Namespace> {
            self.inner.get_namespace(namespace).await
        }
        async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
            self.inner.namespace_exists(namespace).await
        }
        async fn update_namespace(
            &self,
            namespace: &NamespaceIdent,
            properties: HashMap<String, String>,
        ) -> Result<()> {
            self.inner.update_namespace(namespace, properties).await
        }
        async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
            self.inner.drop_namespace(namespace).await
        }
        async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
            self.inner.list_tables(namespace).await
        }
        async fn create_table(
            &self,
            namespace: &NamespaceIdent,
            creation: TableCreation,
        ) -> Result<Table> {
            self.inner.create_table(namespace, creation).await
        }
        async fn load_table(&self, table: &TableIdent) -> Result<Table> {
            self.inner.load_table(table).await
        }
        async fn drop_table(&self, table: &TableIdent) -> Result<()> {
            self.inner.drop_table(table).await
        }
        async fn purge_table(&self, table: &TableIdent) -> Result<()> {
            self.inner.purge_table(table).await
        }
        async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
            self.inner.table_exists(table).await
        }
        async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
            self.inner.rename_table(src, dest).await
        }
        async fn register_table(
            &self,
            table: &TableIdent,
            metadata_location: String,
        ) -> Result<Table> {
            self.inner.register_table(table, metadata_location).await
        }
        async fn update_table(&self, commit: crate::TableCommit) -> Result<Table> {
            let next = self.queue.lock().unwrap().pop_front();
            if let Some(i) = next {
                let t = self.inner.load_table(commit.identifier()).await?;
                let tx = Transaction::new(&t);
                let t = match i {
                    Interjection::Append(f, props) => tx
                        .fast_append()
                        .add_data_files(vec![f])
                        .set_snapshot_properties(props)
                        .apply(tx)?,
                    Interjection::DeleteFile(d) => {
                        tx.row_delta().add_delete_files(vec![d]).apply(tx)?
                    }
                    Interjection::Rewrite { remove, add } => tx
                        .rewrite_files()
                        .delete_data_files(vec![remove])
                        .add_data_files(vec![add])
                        .apply(tx)?,
                }
                .commit(self.inner.as_ref())
                .await?;
                self.landed
                    .lock()
                    .unwrap()
                    .push(t.metadata().current_snapshot_id().unwrap());
            }
            self.inner.update_table(commit).await
        }
    }

    /// f1 (ids 1, 2) committed with `ledger.offset.a` = 41; the replacement of f1.
    async fn ledgered_f1(catalog: &dyn Catalog, table: &Table) -> (DataFile, Table, Vec<FileRows>) {
        let f1 = write(
            table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), None),
            None,
        )
        .await;
        let table = append(
            catalog,
            table,
            vec![f1.clone()],
            HashMap::from([("ledger.offset.a".to_string(), "41".to_string())]),
        )
        .await;
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1, 2], &[Some("a"), Some("b")], Some(false), None)],
        }];
        (f1, table, repl)
    }

    fn head_of(t: &Table) -> i64 {
        t.metadata().current_snapshot_id().unwrap()
    }

    #[tokio::test]
    async fn an_append_landing_during_the_write_is_rebased_with_its_recheck_listed() {
        let (_wh, inner, ident, table) = setup(FormatVersion::V2, false).await;
        let (_f1, table, repl) = ledgered_f1(inner.as_ref(), &table).await;
        let s0 = head_of(&table);
        let f2 = write(
            &table,
            "f2",
            rows(&[3], &[Some("lane")], Some(false), None),
            None,
        )
        .await;
        let cat = Interloper::new(inner.clone(), vec![Interjection::Append(
            f2,
            HashMap::from([("ledger.offset.a".to_string(), "42".to_string())]),
        )]);
        let out = replace_file_rows(
            &cat,
            &ident,
            s0,
            repl,
            false,
            HashMap::from([("mark".to_string(), "1".to_string())]),
            Some("ledger."),
        )
        .await
        .unwrap();
        let s1 = cat.landed()[0];
        assert_eq!(out.rebase_rechecks, vec![RebaseRecheck {
            attempt: 2,
            expected: s0,
            current: s1,
            snapshots: vec![s1],
        }]);
        let table = inner.load_table(&ident).await.unwrap();
        let snap = table.metadata().current_snapshot().unwrap();
        assert_eq!(Some(snap.snapshot_id()), out.snapshot_after);
        assert_eq!(snap.parent_snapshot_id(), Some(s1), "landed on the append");
        assert_eq!(read(&table).await, vec![
            (1, Some("a".to_string())),
            (2, Some("b".to_string())),
            (3, Some("lane".to_string()))
        ]);
        let summary = &snap.summary().additional_properties;
        assert_eq!(
            summary.get("ledger.offset.a").map(String::as_str),
            Some("42"),
            "the carried ledger is the append's, never the base's"
        );
        assert_eq!(summary.get("mark").map(String::as_str), Some("1"));
        assert_eq!(summary.get("total-records").map(String::as_str), Some("3"));
    }

    #[tokio::test]
    async fn a_replace_that_meets_no_commit_lists_no_recheck() {
        let (_wh, inner, ident, table) = setup(FormatVersion::V2, false).await;
        let (_f1, table, repl) = ledgered_f1(inner.as_ref(), &table).await;
        let s0 = head_of(&table);
        let out = replace_file_rows(
            inner.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            Some("ledger."),
        )
        .await
        .unwrap();
        assert!(out.rebase_rechecks.is_empty());
        let table = inner.load_table(&ident).await.unwrap();
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .parent_snapshot_id(),
            Some(s0)
        );
    }

    #[tokio::test]
    async fn a_delete_file_landing_during_the_write_refuses() {
        let (_wh, inner, ident, table) = setup(FormatVersion::V3, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), Some(&[None, None])),
            None,
        )
        .await;
        let f2 = write(
            &table,
            "f2",
            rows(&[3, 4], &[None, None], Some(false), Some(&[None, None])),
            None,
        )
        .await;
        let table = append(
            inner.as_ref(),
            &table,
            vec![f1.clone(), f2.clone()],
            HashMap::new(),
        )
        .await;
        let s0 = head_of(&table);
        // a deletion vector on f2 — a file the replace does not even name
        let dv = DeleteVector::new(RoaringTreemap::from_iter([0u64]));
        let del = dv
            .write_to_puffin_file(
                table.file_io(),
                format!("{}/data/dv-f2.puffin", table.metadata().location()),
                f2.file_path().to_string(),
                Struct::empty(),
                table.metadata().default_partition_spec_id(),
            )
            .await
            .unwrap();
        let cat = Interloper::new(inner.clone(), vec![Interjection::DeleteFile(del)]);
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 2],
                &[Some("a"), Some("b")],
                Some(false),
                Some(&[None, None]),
            )],
        }];
        let err = replace_file_rows(&cat, &ident, s0, repl, false, HashMap::new(), None)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Cannot commit RewriteFiles based on stale snapshot"),
            "{msg}"
        );
        assert!(msg.contains("not an append"), "{msg}");
        let table = inner.load_table(&ident).await.unwrap();
        assert_eq!(
            head_of(&table),
            cat.landed()[0],
            "nothing committed after the delete"
        );
        assert_eq!(read(&table).await, vec![(1, None), (2, None), (4, None)]);
    }

    #[tokio::test]
    async fn a_rewrite_of_the_replaced_file_landing_during_the_write_refuses() {
        let (_wh, inner, ident, table) = setup(FormatVersion::V2, false).await;
        let (f1, table, repl) = ledgered_f1(inner.as_ref(), &table).await;
        let s0 = head_of(&table);
        let f1b = write(
            &table,
            "f1b",
            rows(&[1, 2], &[Some("x"), Some("y")], Some(false), None),
            None,
        )
        .await;
        let cat = Interloper::new(inner.clone(), vec![Interjection::Rewrite {
            remove: f1,
            add: f1b,
        }]);
        let err = replace_file_rows(
            &cat,
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            Some("ledger."),
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Cannot commit RewriteFiles based on stale snapshot"),
            "{msg}"
        );
        let table = inner.load_table(&ident).await.unwrap();
        assert_eq!(head_of(&table), cat.landed()[0]);
        assert_eq!(read(&table).await, vec![
            (1, Some("x".to_string())),
            (2, Some("y".to_string()))
        ]);
    }

    #[tokio::test]
    async fn a_changed_entry_of_a_removed_file_refuses_even_behind_a_plain_append() {
        // The proof reads each removed file's entry at commit: a guard that saw
        // another record count than the table holds refuses.
        let (_wh, inner, ident, table) = setup(FormatVersion::V2, false).await;
        let (f1, table, _repl) = ledgered_f1(inner.as_ref(), &table).await;
        let s0 = head_of(&table);
        let f2 = write(&table, "f2", rows(&[3], &[None], Some(false), None), None).await;
        let table = append(inner.as_ref(), &table, vec![f2], HashMap::new()).await;
        let f1b = write(
            &table,
            "f1b",
            rows(&[1, 2], &[Some("a"), Some("b")], Some(false), None),
            None,
        )
        .await;
        let entry = |records: u64| AppendRebaseGuard {
            base_snapshot_id: s0,
            schema_id: table.metadata().current_schema_id(),
            default_spec_id: table.metadata().default_partition_spec_id(),
            default_sort_order_id: table.metadata().default_sort_order_id(),
            removed_entries: HashMap::from([(
                f1.file_path().to_string(),
                (f1.file_size_in_bytes(), records, Some(1)),
            )]),
            max_attempts: REPLACE_COMMIT_MAX_ATTEMPTS,
            lower_bound_floors: vec![],
        };
        let log = RebaseLog::default();
        let tx = Transaction::new(&table);
        let err = tx
            .rewrite_files()
            .delete_data_files(vec![f1.clone()])
            .add_data_files(vec![f1b.clone()])
            .rebase_across_appends(entry(3), log.clone())
            .apply(tx)
            .unwrap()
            .commit(inner.as_ref())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("the entry of"), "{err}");
        assert!(log.lock().unwrap().is_empty());
        // the true entry passes the same proof
        let tx = Transaction::new(&table);
        let t = tx
            .rewrite_files()
            .delete_data_files(vec![f1.clone()])
            .add_data_files(vec![f1b])
            .rebase_across_appends(entry(2), log.clone())
            .apply(tx)
            .unwrap()
            .commit(inner.as_ref())
            .await
            .unwrap();
        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(
            inner
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            t.metadata().current_snapshot_id()
        );
    }

    #[tokio::test]
    async fn a_table_that_keeps_moving_refuses_after_the_bound() {
        let (_wh, inner, ident, table) = setup(FormatVersion::V2, false).await;
        // let the transaction retry past the replace's own bound
        let tx = Transaction::new(&table);
        let table = tx
            .update_table_properties()
            .set("commit.retry.num-retries".to_string(), "20".to_string())
            .set("commit.retry.min-wait-ms".to_string(), "1".to_string())
            .set("commit.retry.max-wait-ms".to_string(), "2".to_string())
            .apply(tx)
            .unwrap()
            .commit(inner.as_ref())
            .await
            .unwrap();
        let (_f1, table, repl) = ledgered_f1(inner.as_ref(), &table).await;
        let s0 = head_of(&table);
        let mut queue = Vec::new();
        for i in 0..REPLACE_COMMIT_MAX_ATTEMPTS + 2 {
            let f = write(
                &table,
                &format!("lane{i}"),
                rows(&[100 + i as i64], &[None], Some(false), None),
                None,
            )
            .await;
            queue.push(Interjection::Append(f, HashMap::new()));
        }
        let cat = Interloper::new(inner.clone(), queue);
        let err = replace_file_rows(
            &cat,
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            Some("ledger."),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains(&format!(
                "{REPLACE_COMMIT_MAX_ATTEMPTS} commit attempts used"
            )),
            "{err}"
        );
        let landed = cat.landed();
        assert_eq!(
            landed.len(),
            REPLACE_COMMIT_MAX_ATTEMPTS,
            "one append per attempt"
        );
        let table = inner.load_table(&ident).await.unwrap();
        assert_eq!(
            head_of(&table),
            *landed.last().unwrap(),
            "the replace committed nothing"
        );
    }

    // ---- proof 5: appended files must rank after the replaced rows ----

    /// A table (id, off long, ts timestamp no-tz, sts timestamptz, k string).
    async fn setup_ordered() -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
        let warehouse = TempDir::new().unwrap();
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.path().to_str().unwrap().to_string(),
                    )]),
                )
                .await
                .unwrap(),
        );
        let ns = NamespaceIdent::new("db".to_string());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "off", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(3, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
                NestedField::optional(4, "sts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
                NestedField::optional(5, "k", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let creation = TableCreation::builder()
            .name("o".to_string())
            .schema(schema)
            .format_version(FormatVersion::V2)
            .build();
        let table = catalog.create_table(&ns, creation).await.unwrap();
        (
            warehouse,
            catalog,
            TableIdent::new(ns, "o".to_string()),
            table,
        )
    }

    /// Rows with `off` = ids, the given ts / sts (microseconds) per row.
    fn ordered_rows(
        table: &Table,
        ids: &[i64],
        ts: &[Option<i64>],
        sts: &[Option<i64>],
    ) -> RecordBatch {
        use arrow_array::TimestampMicrosecondArray;
        let target = Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).unwrap());
        let ts_type = target.field_with_name("sts").unwrap().data_type().clone();
        let tz = match ts_type {
            DataType::Timestamp(_, tz) => tz,
            _ => None,
        };
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(ts.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(sts.to_vec()).with_timezone_opt(tz)),
            Arc::new(StringArray::from(vec![Some("x"); ids.len()])),
        ];
        let b = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(
                target
                    .fields()
                    .iter()
                    .zip(&cols)
                    .map(|(f, c)| Field::new(f.name(), c.data_type().clone(), true))
                    .collect::<Vec<_>>(),
            )),
            cols,
        )
        .unwrap();
        conform_batch(&b, &target).unwrap()
    }

    fn floors() -> Vec<BoundFloor> {
        vec![
            BoundFloor {
                fields: vec!["off".to_string()],
                above: 100,
            },
            BoundFloor {
                fields: vec!["ts".to_string(), "sts".to_string()],
                above: 5_000,
            },
        ]
    }

    /// f1 (off 1..=100, ts = sts = 1000..=5000 step ~40) is replaced while a lane
    /// file `lane` lands; the replace holds appended files to `floors()`.
    async fn floor_case(
        lane_ids: &[i64],
        lane_ts: &[Option<i64>],
        lane_sts: &[Option<i64>],
    ) -> (
        Result<ReplaceFileRowsOutcome>,
        Arc<dyn Catalog>,
        TableIdent,
        i64,
        TempDir,
    ) {
        let (wh, inner, ident, table) = setup_ordered().await;
        let ids: Vec<i64> = (1..=100).collect();
        let t: Vec<Option<i64>> = ids.iter().map(|i| Some(1000 + (i - 1) * 40 + 40)).collect();
        let f1 = write(&table, "f1", ordered_rows(&table, &ids, &t, &t), None).await;
        let table = append(inner.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = head_of(&table);
        let lane = write(
            &table,
            "lane",
            ordered_rows(&table, lane_ids, lane_ts, lane_sts),
            None,
        )
        .await;
        let cat = Interloper::new(inner.clone(), vec![Interjection::Append(
            lane,
            HashMap::new(),
        )]);
        let files = vec![FileRowSource {
            data_file_path: f1.file_path().to_string(),
            source: Arc::new(vec![ordered_rows(&table, &ids, &t, &t)]) as Arc<dyn RowSource>,
        }];
        let out = replace_file_rows_from_sources_with_floors(
            &cat,
            &ident,
            s0,
            files,
            false,
            HashMap::new(),
            None,
            &floors(),
        )
        .await;
        let landed = cat.landed()[0];
        (out, inner, ident, landed, wh)
    }

    async fn assert_refused_at(
        out: Result<ReplaceFileRowsOutcome>,
        inner: &Arc<dyn Catalog>,
        ident: &TableIdent,
        landed: i64,
        why: &str,
    ) {
        let msg = out.unwrap_err().to_string();
        assert!(
            msg.contains("Cannot commit RewriteFiles based on stale snapshot"),
            "{msg}"
        );
        assert!(msg.contains("proof 5"), "{msg}");
        assert!(msg.contains(why), "{msg}");
        assert_eq!(
            head_of(&inner.load_table(ident).await.unwrap()),
            landed,
            "nothing committed"
        );
    }

    #[tokio::test]
    async fn proof5_an_appended_file_above_both_floors_is_rebased() {
        let (out, inner, ident, landed, _wh) =
            floor_case(&[101, 102], &[Some(6_000), Some(6_100)], &[
                Some(6_050),
                Some(6_200),
            ])
            .await;
        let out = out.unwrap();
        assert_eq!(out.rebase_rechecks.len(), 1);
        assert_eq!(out.rebase_rechecks[0].snapshots, vec![landed]);
        let t = inner.load_table(&ident).await.unwrap();
        assert_eq!(
            t.metadata()
                .current_snapshot()
                .unwrap()
                .parent_snapshot_id(),
            Some(landed)
        );
    }

    #[tokio::test]
    async fn proof5_an_appended_file_below_the_offset_floor_refuses() {
        let (out, inner, ident, landed, _wh) =
            floor_case(&[50, 150], &[Some(6_000), Some(6_100)], &[
                Some(6_000),
                Some(6_100),
            ])
            .await;
        assert_refused_at(
            out,
            &inner,
            &ident,
            landed,
            "lower bound 50 over fields [2], not above 100",
        )
        .await;
    }

    #[tokio::test]
    async fn proof5_an_appended_file_above_the_offset_but_earlier_in_time_refuses() {
        // sts is late, ts is early: the pooled minimum (ts) is at or below the floor
        let (out, inner, ident, landed, _wh) =
            floor_case(&[200], &[Some(5_000)], &[Some(9_000)]).await;
        assert_refused_at(
            out,
            &inner,
            &ident,
            landed,
            "lower bound 5000 over fields [3, 4], not above 5000",
        )
        .await;
    }

    #[tokio::test]
    async fn proof5_a_null_only_source_time_passes_on_the_event_time_alone() {
        // the no-tz `ts` alone carries the time (sts null-only: pre-contract rows, mongo tails)
        let (out, inner, ident, landed, _wh) =
            floor_case(&[300, 301], &[Some(7_000), Some(7_001)], &[None, None]).await;
        let out = out.unwrap();
        assert_eq!(out.rebase_rechecks[0].snapshots, vec![landed]);
        let t = inner.load_table(&ident).await.unwrap();
        assert_eq!(
            t.metadata()
                .current_snapshot()
                .unwrap()
                .parent_snapshot_id(),
            Some(landed)
        );
    }

    #[tokio::test]
    async fn proof5_an_appended_file_with_no_time_bound_refuses() {
        let (out, inner, ident, landed, _wh) = floor_case(&[400], &[None], &[None]).await;
        assert_refused_at(
            out,
            &inner,
            &ident,
            landed,
            "has no lower bound for any of fields [3, 4]",
        )
        .await;
    }

    #[tokio::test]
    async fn proof5_a_floor_on_a_non_integer_field_refuses_before_any_write() {
        let (_wh, inner, ident, table) = setup_ordered().await;
        let ids: Vec<i64> = (1..=3).collect();
        let t = vec![Some(1_000); 3];
        let f1 = write(&table, "f1", ordered_rows(&table, &ids, &t, &t), None).await;
        let table = append(inner.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = head_of(&table);
        let source = Counting::new(vec![ordered_rows(&table, &ids, &t, &t)], false);
        let err = replace_file_rows_from_sources_with_floors(
            inner.as_ref(),
            &ident,
            s0,
            vec![FileRowSource {
                data_file_path: f1.file_path().to_string(),
                source: source.clone(),
            }],
            false,
            HashMap::new(),
            None,
            &[BoundFloor {
                fields: vec!["k".to_string()],
                above: 0,
            }],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("bound floor field `k`"), "{err}");
        assert_eq!(source.opens(), 0, "refused before reading the rows");
        assert_eq!(head_of(&inner.load_table(&ident).await.unwrap()), s0);
    }
}
