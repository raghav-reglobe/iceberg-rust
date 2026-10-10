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

//! Copy-on-write rewrite of STRING columns that hold a MySQL `TIME` value in
//! several historical spellings into ONE signed `long` of microseconds since
//! midnight — the convergence a CDC pipeline needs after different producers
//! (a connector in milliseconds mode, a sink that rendered a dated text, a
//! backfill that wrote Python's `timedelta` text) wrote the same column their
//! own way.
//!
//! The caller first ADDS a `long` target column beside each string source (one
//! metadata commit); this module then reads the current snapshot, fills every
//! target from its source, writes new data files that carry BOTH columns and
//! commits a single `Replace` snapshot that removes every old data file. The
//! string source stays readable throughout — the caller swaps the names at the
//! end (delete the source, rename the target onto its name, one metadata
//! commit through `UpdateSchemaAction::rename_column`), so a rollback at any
//! point before that is a snapshot rollback and nothing is ever unreadable. No
//! delete files are written. A delete file the scan binds to a live data file refuses
//! the table (rewriting its data files would leave it dangling or lose its rows); a
//! delete file bound to NO live data file — every file it applied to was already
//! rewritten under a higher sequence number — is removed in the same commit. The base snapshot's summary properties that start
//! with `carry_summary_prefix` are copied onto the new snapshot (a producer's
//! offsets-in-snapshot ledger keeps its home when older snapshots expire).
//!
//! Spellings — the only transformation, pinned by tests:
//!   * `^-?[0-9]+$`                          milliseconds since midnight → × 1000
//!   * `^1970-01-01[ T]H:MM:SS(.f+)?$`        a dated clock (time-of-day) → µs
//!   * `^-?H{1,3}:MM:SS(.f+)?$`               a clock, hours never wrapped → signed µs
//!   * `^-?N days?, H:MM:SS(.f+)?$`           Python `timedelta` text → signed µs
//!   * NULL or empty                         NULL
//!   * anything else                         the rewrite REFUSES the table before any commit.
//!
//! A source may also be a `timestamp`/`timestamptz` column that holds the dated clock as a real
//! timestamp (`1970-01-01 HH:MM:SS`): its microseconds since the epoch ARE the microseconds since
//! midnight when, and only when, the value lies within the epoch day — any other date is refused.
//!
//! MySQL `TIME` spans -838:59:59 .. 838:59:59; a parsed value outside that range is refused too.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, TimeUnit};
use arrow_select::filter::filter_record_batch;
use futures::TryStreamExt;
use uuid::Uuid;

use crate::arrow::schema_to_arrow_schema;
use crate::atomic_replace::{conform_batch, writer_properties};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Literal, ManifestStatus, PartitionKey, PrimitiveType,
    Struct, Transform, Type,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

const US_PER_SEC: i64 = 1_000_000;
const US_PER_DAY: i64 = 86_400 * US_PER_SEC;
/// MySQL `TIME` range: ±838:59:59.999999.
const MAX_ABS_US: i64 = (838 * 3600 + 59 * 60 + 59) * US_PER_SEC + 999_999;

/// Which historical spelling a value carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeShape {
    /// Milliseconds since midnight as digits, optionally signed.
    Digits,
    /// A clock on the epoch date, `1970-01-01 H:MM:SS[.f]`.
    Dated,
    /// A clock, `[-]H:MM:SS[.f]`, hours never wrapped.
    Clock,
    /// Python's `timedelta` text, `[-]N day[s], H:MM:SS[.f]`.
    Days,
}

/// How many values of each spelling a rewrite met.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeCounts {
    /// Values spelled as milliseconds since midnight.
    pub digits: u64,
    /// Values spelled as a clock on the epoch date.
    pub dated: u64,
    /// Values spelled as a plain clock.
    pub clock: u64,
    /// Values spelled as `timedelta` text.
    pub days: u64,
    /// NULL or empty values.
    pub nulls: u64,
}

impl ShapeCounts {
    fn add(&mut self, other: ShapeCounts) {
        self.digits += other.digits;
        self.dated += other.dated;
        self.clock += other.clock;
        self.days += other.days;
        self.nulls += other.nulls;
    }
    fn note(&mut self, shape: TimeShape) {
        match shape {
            TimeShape::Digits => self.digits += 1,
            TimeShape::Dated => self.dated += 1,
            TimeShape::Clock => self.clock += 1,
            TimeShape::Days => self.days += 1,
        }
    }
}

fn frac_us(frac: Option<&str>) -> Option<i64> {
    match frac {
        None => Some(0),
        Some(f) if !f.is_empty() && f.len() <= 6 && f.bytes().all(|b| b.is_ascii_digit()) => {
            let mut digits = f.to_string();
            while digits.len() < 6 {
                digits.push('0');
            }
            digits.parse::<i64>().ok()
        }
        Some(_) => None,
    }
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::DataInvalid, msg)
}

fn checked(us: i64, shape: TimeShape, raw: &str) -> Result<Option<(i64, TimeShape)>> {
    if us.abs() > MAX_ABS_US {
        return Err(invalid(format!(
            "MySQL TIME out of range (±838:59:59): `{raw}`"
        )));
    }
    Ok(Some((us, shape)))
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `H:MM:SS[.f]` with 1–3 hour digits and exactly two-digit minutes/seconds →
/// (hours, minutes, seconds, fraction µs). `None` when the shape does not fit.
fn parse_clock(s: &str) -> Option<(i64, i64, i64, i64)> {
    let (hms, frac) = match s.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let mut parts = hms.split(':');
    let (h, m, sec) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    if !(all_digits(h) && h.len() <= 3 && m.len() == 2 && all_digits(m) && sec.len() == 2 && all_digits(sec)) {
        return None;
    }
    let (m, sec): (i64, i64) = (m.parse().ok()?, sec.parse().ok()?);
    if m > 59 || sec > 59 {
        return None;
    }
    Some((h.parse().ok()?, m, sec, frac_us(frac)?))
}

/// Parse one spelling into signed microseconds since midnight. `Ok(None)` for an
/// empty string; an unrecognised spelling or an out-of-range value is an error.
pub fn mysql_time_text_to_us(s: &str) -> Result<Option<(i64, TimeShape)>> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(None);
    }
    // milliseconds since midnight, optionally signed
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t),
    };
    if all_digits(body) {
        let ms: i64 = body
            .parse()
            .map_err(|_| invalid(format!("TIME milliseconds out of range: `{t}`")))?;
        let us = ms
            .checked_mul(1000)
            .ok_or_else(|| invalid(format!("TIME milliseconds out of range: `{t}`")))?;
        return checked(if neg { -us } else { us }, TimeShape::Digits, t);
    }
    // the dated clock: `1970-01-01 H:MM:SS[.f]` (or `T`) — a time of day only
    if let Some(rest) = t.strip_prefix("1970-01-01") {
        let clock = rest.strip_prefix(' ').or_else(|| rest.strip_prefix('T'));
        if let Some((h, m, sec, frac)) = clock.and_then(parse_clock) {
            if h > 23 {
                return Err(invalid(format!("dated TIME has a field out of range: `{t}`")));
            }
            return checked((h * 3600 + m * 60 + sec) * US_PER_SEC + frac, TimeShape::Dated, t);
        }
        return Err(invalid(format!("unrecognised MySQL TIME spelling: `{t}`")));
    }
    // Python's timedelta text: `[-]N day[s], H:MM:SS[.f]` — the day count carries the sign:
    // str(timedelta(days=-35, seconds=3601)) == "-35 days, 1:00:01" == -(35 × 86400) + 3601 s
    if let Some(pos) = t.find(" day") {
        let days_txt = &t[..pos];
        let after = &t[pos..];
        let after = after
            .strip_prefix(" days, ")
            .or_else(|| after.strip_prefix(" day, "))
            .ok_or_else(|| invalid(format!("unrecognised MySQL TIME spelling: `{t}`")))?;
        let (neg_days, dtxt) = match days_txt.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, days_txt),
        };
        if !all_digits(dtxt) {
            return Err(invalid(format!("unrecognised MySQL TIME spelling: `{t}`")));
        }
        let days: i64 = dtxt
            .parse()
            .map_err(|_| invalid(format!("timedelta days out of range: `{t}`")))?;
        let (h, m, sec, frac) =
            parse_clock(after).ok_or_else(|| invalid(format!("unrecognised MySQL TIME spelling: `{t}`")))?;
        if h > 23 {
            return Err(invalid(format!("timedelta TIME has a field out of range: `{t}`")));
        }
        let day_us = days
            .checked_mul(US_PER_DAY)
            .ok_or_else(|| invalid(format!("timedelta days out of range: `{t}`")))?;
        let tod = (h * 3600 + m * 60 + sec) * US_PER_SEC + frac;
        let us = if neg_days { -day_us + tod } else { day_us + tod };
        return checked(us, TimeShape::Days, t);
    }
    // the clock: `[-]H:MM:SS[.f]`, hours never wrapped
    if let Some((h, m, sec, frac)) = parse_clock(body) {
        let us = (h * 3600 + m * 60 + sec) * US_PER_SEC + frac;
        return checked(if neg { -us } else { us }, TimeShape::Clock, t);
    }
    Err(invalid(format!("unrecognised MySQL TIME spelling: `{t}`")))
}

/// Convert a Utf8 / LargeUtf8 array of spellings into an `Int64Array` of signed
/// microseconds (nulls preserved), counting the spellings met.
pub fn time_text_array_to_us(array: &ArrayRef) -> Result<(Int64Array, ShapeCounts)> {
    let mut counts = ShapeCounts::default();
    let mut out: Vec<Option<i64>> = Vec::with_capacity(array.len());
    let mut push = |v: Option<&str>| -> Result<()> {
        match v {
            None => {
                counts.nulls += 1;
                out.push(None);
            }
            Some(s) => match mysql_time_text_to_us(s)? {
                None => {
                    counts.nulls += 1;
                    out.push(None);
                }
                Some((us, shape)) => {
                    counts.note(shape);
                    out.push(Some(us));
                }
            },
        }
        Ok(())
    };
    match array.data_type() {
        DataType::Utf8 => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| invalid("Utf8 downcast".to_string()))?;
            for i in 0..a.len() {
                push(if a.is_null(i) { None } else { Some(a.value(i)) })?;
            }
        }
        DataType::LargeUtf8 => {
            let a = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| invalid("LargeUtf8 downcast".to_string()))?;
            for i in 0..a.len() {
                push(if a.is_null(i) { None } else { Some(a.value(i)) })?;
            }
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let a = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| invalid("Timestamp downcast".to_string()))?;
            for i in 0..a.len() {
                if a.is_null(i) {
                    counts.nulls += 1;
                    out.push(None);
                    continue;
                }
                let v = a.value(i);
                if !(0..US_PER_DAY).contains(&v) {
                    return Err(invalid(format!(
                        "timestamp TIME value {v} µs is not a time of day on the epoch date (1970-01-01) — refused"
                    )));
                }
                counts.note(TimeShape::Dated);
                out.push(Some(v));
            }
        }
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("TIME rewrite expects a string or microsecond-timestamp column, found {other}"),
            ));
        }
    }
    Ok((Int64Array::from(out), counts))
}

/// What a rewrite did.
#[derive(Debug, Clone, Default)]
pub struct RewriteOutcome {
    /// The snapshot the rewrite read (and replaced).
    pub snapshot_before: i64,
    /// `None` on a dry run (nothing written, nothing committed).
    pub snapshot_after: Option<i64>,
    /// Rows scanned (and, unless a dry run, rewritten).
    pub rows: u64,
    /// The base snapshot's `total-records`, when its summary carries it — the
    /// scan must meet it exactly before anything is committed.
    pub records_before: Option<u64>,
    /// Live data files of the base snapshot, all replaced.
    pub files_deleted: usize,
    /// Data files the rewrite wrote.
    pub files_added: usize,
    /// 0 (unpartitioned) or 1 (one identity partition field).
    pub partition_fields: usize,
    /// Delete files bound to no live data file, removed with the rewrite.
    pub delete_files_removed: usize,
    /// How many values of each spelling the rewrite met.
    pub counts: ShapeCounts,
}

fn partition_literal(array: &ArrayRef, row: usize) -> Result<Option<Literal>> {
    if array.is_null(row) {
        return Ok(None);
    }
    Ok(Some(match array.data_type() {
        DataType::Boolean => Literal::bool(array.as_any().downcast_ref::<BooleanArray>().expect("bool").value(row)),
        DataType::Int32 => Literal::int(array.as_any().downcast_ref::<Int32Array>().expect("int").value(row)),
        DataType::Int64 => Literal::long(array.as_any().downcast_ref::<Int64Array>().expect("long").value(row)),
        DataType::Utf8 => Literal::string(array.as_any().downcast_ref::<StringArray>().expect("utf8").value(row)),
        DataType::LargeUtf8 => {
            Literal::string(array.as_any().downcast_ref::<LargeStringArray>().expect("large utf8").value(row))
        }
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("partition column of type {other} is not supported by the TIME rewrite"),
            ));
        }
    }))
}

/// Fill each `(source, target)` pair of `ident` — `source` a string column of
/// spellings (or a microsecond timestamp column holding the dated clock), `target`
/// an existing `long` column — with signed microseconds,
/// copy-on-write, in one `Replace` snapshot; the source is kept as it was.
/// Unpartitioned tables and tables with ONE identity partition field are
/// supported (every row is written under its own partition). `base_snapshot`,
/// when given, must be the current snapshot (optimistic concurrency, enforced
/// again at commit). The scan must meet the base snapshot's `total-records`
/// exactly or nothing is committed. With `dry_run` the data is parsed and
/// counted but nothing is written or committed.
pub async fn rewrite_time_columns(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    columns: &[(String, String)],
    base_snapshot: Option<i64>,
    dry_run: bool,
    snapshot_properties: HashMap<String, String>,
    carry_summary_prefix: Option<&str>,
) -> Result<RewriteOutcome> {
    if columns.is_empty() {
        return Err(invalid("no columns to rewrite".to_string()));
    }
    let table = catalog.load_table(ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .cloned()
        .ok_or_else(|| invalid(format!("{ident}: no current snapshot — nothing to rewrite")))?;
    let s0 = snapshot.snapshot_id();
    if let Some(b) = base_snapshot {
        if b != s0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("{ident}: planned against snapshot {b} but the table is at {s0}"),
            ));
        }
    }
    let cur_schema = table.metadata().current_schema().clone();
    for (src, tgt) in columns {
        if src == tgt {
            return Err(invalid(format!("{ident}: `{src}` cannot be its own target — add a long column first")));
        }
        match cur_schema.field_by_name(src).map(|f| f.field_type.as_ref()) {
            Some(Type::Primitive(PrimitiveType::String))
            | Some(Type::Primitive(PrimitiveType::Timestamp))
            | Some(Type::Primitive(PrimitiveType::Timestamptz)) => {}
            other => {
                return Err(invalid(format!(
                    "{ident}: source `{src}` must be a string or timestamp column (found {other:?})"
                )));
            }
        }
        match cur_schema.field_by_name(tgt).map(|f| f.field_type.as_ref()) {
            Some(Type::Primitive(PrimitiveType::Long)) => {}
            other => {
                return Err(invalid(format!(
                    "{ident}: target `{tgt}` must be a long column (found {other:?}) — evolve the schema first"
                )));
            }
        }
    }
    let spec = table.metadata().default_partition_spec().clone();
    let partition_col: Option<String> = match spec.fields() {
        [] => None,
        [f] if f.transform == Transform::Identity => Some(
            cur_schema
                .field_by_id(f.source_id)
                .map(|x| x.name.clone())
                .ok_or_else(|| invalid(format!("{ident}: partition source id {} not in the schema", f.source_id)))?,
        ),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("{ident}: the TIME rewrite supports unpartitioned tables or ONE identity partition field, found {other:?}"),
            ));
        }
    };
    let records_before: Option<u64> = snapshot
        .summary()
        .additional_properties
        .get("total-records")
        .and_then(|v| v.parse::<u64>().ok());

    // Every live data file of the base snapshot is replaced. A delete file the scan
    // binds to a live data file refuses the table; one bound to nothing is removed.
    let manifest_list = table.manifest_list_reader(&snapshot).load().await?;
    let mut old_files: Vec<DataFile> = Vec::new();
    let mut delete_files: Vec<DataFile> = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            match entry.content_type() {
                DataContentType::Data => old_files.push(entry.data_file().clone()),
                _ => delete_files.push(entry.data_file().clone()),
            }
        }
    }
    let scan = table.scan().snapshot_id(s0).select_all().build()?;
    if !delete_files.is_empty() {
        let mut bound: HashSet<String> = HashSet::new();
        let mut tasks = scan.plan_files().await?;
        while let Some(task) = tasks.try_next().await? {
            for d in &task.deletes {
                bound.insert(d.file_path.clone());
            }
        }
        if let Some(d) = delete_files.iter().find(|d| bound.contains(d.file_path())) {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: snapshot {s0} carries a {:?} file bound to live data ({}) — reabsorb it first",
                    d.content_type(),
                    d.file_path()
                ),
            ));
        }
    }
    let mut outcome = RewriteOutcome {
        snapshot_before: s0,
        records_before,
        files_deleted: old_files.len(),
        partition_fields: spec.fields().len(),
        delete_files_removed: delete_files.len(),
        ..Default::default()
    };
    if old_files.is_empty() {
        return Ok(outcome);
    }

    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&cur_schema)?);
    let run = Uuid::now_v7();
    // one writer per partition key (a Vec: ≤ a handful of keys on an identity partition)
    let mut writers: Vec<(String, _)> = Vec::new();
    let mut stream = scan.to_arrow().await?;
    while let Some(batch) = stream.try_next().await? {
        if batch.num_rows() == 0 {
            continue;
        }
        let mut fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        for (src, tgt) in columns {
            let si = batch.schema().index_of(src).map_err(|_| {
                invalid(format!("{ident}: scanned batch lacks column `{src}`"))
            })?;
            let ti = batch.schema().index_of(tgt).map_err(|_| {
                invalid(format!("{ident}: scanned batch lacks column `{tgt}`"))
            })?;
            let (converted, counts) = time_text_array_to_us(&cols[si])?;
            outcome.counts.add(counts);
            fields[ti] = Field::new(tgt, DataType::Int64, true);
            cols[ti] = Arc::new(converted);
        }
        outcome.rows += batch.num_rows() as u64;
        if dry_run {
            continue;
        }
        let rebuilt = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
            .map_err(|e| invalid(format!("rebuilding the converted batch: {e}")))?;
        let conformed = conform_batch(&rebuilt, &target_arrow)?;
        // split by partition key (or one group when unpartitioned)
        let groups: Vec<(String, Option<Literal>, Option<BooleanArray>)> = match &partition_col {
            None => vec![("".to_string(), None, None)],
            Some(pc) => {
                let pi = conformed.schema().index_of(pc).map_err(|_| invalid(format!("{ident}: batch lacks the partition column `{pc}`")))?;
                let parr = conformed.column(pi).clone();
                let mut keys: Vec<(String, Option<Literal>)> = Vec::new();
                let mut row_key: Vec<usize> = Vec::with_capacity(parr.len());
                for r in 0..parr.len() {
                    let lit = partition_literal(&parr, r)?;
                    let k = format!("{lit:?}");
                    let idx = match keys.iter().position(|(kk, _)| *kk == k) {
                        Some(i) => i,
                        None => {
                            keys.push((k, lit));
                            keys.len() - 1
                        }
                    };
                    row_key.push(idx);
                }
                keys.into_iter()
                    .enumerate()
                    .map(|(i, (k, lit))| {
                        let mask = BooleanArray::from(row_key.iter().map(|x| *x == i).collect::<Vec<bool>>());
                        (k, lit, Some(mask))
                    })
                    .collect()
            }
        };
        for (key, lit, mask) in groups {
            let part = match mask {
                Some(m) => filter_record_batch(&conformed, &m)
                    .map_err(|e| invalid(format!("splitting the batch by partition: {e}")))?,
                None => conformed.clone(),
            };
            if part.num_rows() == 0 {
                continue;
            }
            let pos = match writers.iter().position(|(k, _)| *k == key) {
                Some(p) => p,
                None => {
                    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
                        ParquetWriterBuilder::new(writer_properties(&table), cur_schema.clone()),
                        table.file_io().clone(),
                        DefaultLocationGenerator::new(table.metadata())?,
                        DefaultFileNameGenerator::new(
                            format!("time-rewrite-{run}-p{}", writers.len()),
                            None,
                            DataFileFormat::Parquet,
                        ),
                    );
                    let pkey = if partition_col.is_some() {
                        Some(PartitionKey::new(spec.as_ref().clone(), cur_schema.clone(), Struct::from_iter(vec![lit])))
                    } else {
                        None
                    };
                    writers.push((key, DataFileWriterBuilder::new(rolling).build(pkey).await?));
                    writers.len() - 1
                }
            };
            writers[pos].1.write(part).await?;
        }
    }
    if let Some(n) = records_before {
        if n != outcome.rows {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("{ident}: scanned {} rows but snapshot {s0} records total-records {n} — nothing committed", outcome.rows),
            ));
        }
    }
    if dry_run {
        return Ok(outcome);
    }
    let mut new_files: Vec<DataFile> = Vec::new();
    for (_k, mut w) in writers {
        new_files.extend(w.close().await?);
    }
    outcome.files_added = new_files.len();

    let mut props = snapshot_properties;
    if let Some(prefix) = carry_summary_prefix {
        for (k, v) in &snapshot.summary().additional_properties {
            if k.starts_with(prefix) && !props.contains_key(k) {
                props.insert(k.clone(), v.clone());
            }
        }
    }
    props.insert(
        "time-rewrite.columns".to_string(),
        columns.iter().map(|(a, b)| format!("{a}->{b}")).collect::<Vec<_>>().join(","),
    );
    props.insert("time-rewrite.base-snapshot".to_string(), s0.to_string());

    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .delete_data_files(old_files)
        .delete_delete_files(delete_files)
        .add_data_files(new_files)
        .validate_from_snapshot(s0)
        .set_snapshot_properties(props)
        .apply(tx)?
        .commit(catalog)
        .await?;
    outcome.snapshot_after = table.metadata().current_snapshot_id();
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, LargeStringArray, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use tempfile::TempDir;

    use super::*;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{FormatVersion, NestedField, Schema, UnboundPartitionSpec};
    use crate::table::Table;
    use crate::transaction::AddColumn;
    use crate::{CatalogBuilder, NamespaceIdent, TableCreation};

    fn us(h: i64, m: i64, s: i64) -> i64 {
        (h * 3600 + m * 60 + s) * US_PER_SEC
    }

    #[test]
    fn every_spelling_parses_to_the_same_microseconds() {
        let cases: Vec<(&str, i64, TimeShape)> = vec![
            ("36000000", us(10, 0, 0), TimeShape::Digits),
            ("0", 0, TimeShape::Digits),
            ("-5000", -5_000_000, TimeShape::Digits),
            ("1970-01-01 10:00:00", us(10, 0, 0), TimeShape::Dated),
            ("1970-01-01T10:00:00.250", us(10, 0, 0) + 250_000, TimeShape::Dated),
            ("10:00:00", us(10, 0, 0), TimeShape::Clock),
            ("9:00:00", us(9, 0, 0), TimeShape::Clock),
            ("10:00:00.123456", us(10, 0, 0) + 123_456, TimeShape::Clock),
            ("-00:00:01", -US_PER_SEC, TimeShape::Clock),
            ("838:59:59", us(838, 59, 59), TimeShape::Clock),
            ("34 days, 22:59:59", 34 * US_PER_DAY + us(22, 59, 59), TimeShape::Days),
            ("1 day, 0:00:00", US_PER_DAY, TimeShape::Days),
            ("-35 days, 1:00:01", -35 * US_PER_DAY + us(1, 0, 1), TimeShape::Days),
            ("-1 day, 23:59:59", -US_PER_SEC, TimeShape::Days),
        ];
        for (raw, want, shape) in cases {
            let got = mysql_time_text_to_us(raw).unwrap().unwrap();
            assert_eq!(got, (want, shape), "{raw}");
        }
        assert_eq!(mysql_time_text_to_us("").unwrap(), None);
        assert_eq!(mysql_time_text_to_us("   ").unwrap(), None);
        for bad in ["abc", "10:60:00", "1970-01-02 00:00:00", "839:00:00", "12:00", "1e3", "10:00:00 PM"] {
            assert!(mysql_time_text_to_us(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn array_conversion_keeps_nulls_and_counts_shapes() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("36000000"),
            None,
            Some(""),
            Some("10:00:00"),
            Some("1970-01-01 01:00:00"),
            Some("2 days, 1:00:00"),
        ]));
        let (out, counts) = time_text_array_to_us(&arr).unwrap();
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            vec![
                Some(us(10, 0, 0)),
                None,
                None,
                Some(us(10, 0, 0)),
                Some(us(1, 0, 0)),
                Some(2 * US_PER_DAY + us(1, 0, 0)),
            ]
        );
        assert_eq!(
            counts,
            ShapeCounts { digits: 1, dated: 1, clock: 1, days: 1, nulls: 2 }
        );
        let bad: ArrayRef = Arc::new(StringArray::from(vec![Some("10:00:00"), Some("nope")]));
        assert!(time_text_array_to_us(&bad).is_err());
        // a microsecond timestamp on the epoch date is the time of day itself
        let ts: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(us(10, 0, 0)), None, Some(0)]).with_timezone("UTC"));
        let (out, counts) = time_text_array_to_us(&ts).unwrap();
        assert_eq!(out.iter().collect::<Vec<_>>(), vec![Some(us(10, 0, 0)), None, Some(0)]);
        assert_eq!(counts, ShapeCounts { digits: 0, dated: 2, clock: 0, days: 0, nulls: 1 });
        let next_day: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(US_PER_DAY)]));
        assert!(time_text_array_to_us(&next_day).is_err(), "a value past the epoch day is refused");
        let negative: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(-1)]));
        assert!(time_text_array_to_us(&negative).is_err());
    }

    #[tokio::test]
    async fn a_timestamptz_source_holding_the_dated_clock_converges_too() {
        let (_wh, catalog, ident, _table) = setup().await;
        // add a timestamptz column, write a file with it through the table's current schema
        let table = {
            let t = catalog.load_table(&ident).await.unwrap();
            let tx = Transaction::new(&t);
            tx.update_schema()
                .add_column(AddColumn::optional("shift", Type::Primitive(PrimitiveType::Timestamptz)))
                .apply(tx)
                .unwrap()
                .commit(catalog.as_ref())
                .await
                .unwrap()
        };
        let schema = table.metadata().current_schema().clone();
        let arrow = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let batch = RecordBatch::try_new(arrow, vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec![Some("10:00:00"), None, Some("1:00:00")])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec![None::<&str>, None, None])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![false, false, false])) as ArrayRef,
            // timestamptz maps to a microsecond timestamp in the "+00:00" zone
            Arc::new(TimestampMicrosecondArray::from(vec![Some(us(13, 0, 0)), None, Some(us(21, 30, 0))]).with_timezone("+00:00")) as ArrayRef,
        ])
        .unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(&table), schema),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new("ts".to_string(), None, DataFileFormat::Parquet),
        );
        let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
        writer.write(batch).await.unwrap();
        let file = writer.close().await.unwrap().into_iter().next().unwrap();
        let table = append(catalog.as_ref(), &table, file, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        add_long_targets(catalog.as_ref(), &table, &[("shift", "shift__us")]).await;
        let out = rewrite_time_columns(catalog.as_ref(), &ident, &[("shift".to_string(), "shift__us".to_string())], Some(s0), false, HashMap::new(), None)
            .await
            .unwrap();
        assert_eq!(out.counts, ShapeCounts { digits: 0, dated: 2, clock: 0, days: 0, nulls: 1 });
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_longs(&table, "shift__us").await, vec![Some(us(13, 0, 0)), None, Some(us(21, 30, 0))]);
        let table = swap_names(catalog.as_ref(), &table, &[("shift", "shift__us")]).await;
        assert_eq!(read_longs(&table, "shift").await, vec![Some(us(13, 0, 0)), None, Some(us(21, 30, 0))]);
    }

    async fn write_strings(table: &Table, name: &str, t: Vec<Option<&str>>, k: Vec<Option<&str>>) -> DataFile {
        write_strings_in(table, name, t, k, false, None).await
    }

    async fn write_strings_in(
        table: &Table,
        name: &str,
        t: Vec<Option<&str>>,
        k: Vec<Option<&str>>,
        backfill: bool,
        partition_key: Option<PartitionKey>,
    ) -> DataFile {
        let schema = table.metadata().current_schema().clone();
        let arrow = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let n = t.len();
        let ids: Vec<i64> = (1..=n as i64).collect();
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            // the crate maps `string` to 64-bit-offset LargeUtf8
            Arc::new(LargeStringArray::from(t)) as ArrayRef,
            Arc::new(LargeStringArray::from(k)) as ArrayRef,
            Arc::new(BooleanArray::from(vec![backfill; n])) as ArrayRef,
        ];
        // fields added to the schema after the base four (long targets) are written NULL
        for f in arrow.fields().iter().skip(4) {
            columns.push(arrow_array::new_null_array(f.data_type(), n));
        }
        let batch = RecordBatch::try_new(arrow, columns).unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(table), schema),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
        );
        let mut writer = DataFileWriterBuilder::new(rolling).build(partition_key).await.unwrap();
        writer.write(batch).await.unwrap();
        let files = writer.close().await.unwrap();
        assert_eq!(files.len(), 1);
        files.into_iter().next().unwrap()
    }

    async fn setup() -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
        setup_with(false).await
    }

    async fn setup_with(partitioned: bool) -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
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
                NestedField::optional(2, "t", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "k", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(4, "_is_backfill", Type::Primitive(PrimitiveType::Boolean)).into(),
            ])
            .build()
            .unwrap();
        let ident = TableIdent::new(ns.clone(), "times".to_string());
        let creation = if partitioned {
            TableCreation::builder()
                .name("times".to_string())
                .schema(schema)
                .format_version(FormatVersion::V2)
                .partition_spec(
                    UnboundPartitionSpec::builder()
                        .add_partition_field(4, "_is_backfill", Transform::Identity)
                        .unwrap()
                        .build(),
                )
                .build()
        } else {
            TableCreation::builder()
                .name("times".to_string())
                .schema(schema)
                .format_version(FormatVersion::V2)
                .build()
        };
        let table = catalog.create_table(&ns, creation).await.unwrap();
        (warehouse, catalog, ident, table)
    }

    async fn append(catalog: &dyn Catalog, table: &Table, file: DataFile, props: HashMap<String, String>) -> Table {
        let tx = Transaction::new(table);
        tx.fast_append()
            .add_data_files(vec![file])
            .set_snapshot_properties(props)
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    async fn add_long_targets(catalog: &dyn Catalog, table: &Table, pairs: &[(&str, &str)]) -> Table {
        let tx = Transaction::new(table);
        let mut action = tx.update_schema();
        for (_src, tgt) in pairs {
            action = action.add_column(AddColumn::optional(*tgt, Type::Primitive(PrimitiveType::Long)));
        }
        action.apply(tx).unwrap().commit(catalog).await.unwrap()
    }

    /// The final swap: drop the string source and move the long target onto its
    /// name — ONE metadata commit, ids kept.
    async fn swap_names(catalog: &dyn Catalog, table: &Table, pairs: &[(&str, &str)]) -> Table {
        let tx = Transaction::new(table);
        let mut action = tx.update_schema();
        for (src, tgt) in pairs {
            action = action.delete_column(*src).rename_column(*tgt, *src);
        }
        action.apply(tx).unwrap().commit(catalog).await.unwrap()
    }

    async fn read_longs(table: &Table, col: &str) -> Vec<Option<i64>> {
        let scan = table.scan().select_all().build().unwrap();
        let mut stream = scan.to_arrow().await.unwrap();
        let mut out = Vec::new();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let idx = batch.schema().index_of(col).unwrap();
            let a = batch.column(idx).as_any().downcast_ref::<Int64Array>().unwrap();
            out.extend(a.iter());
        }
        out
    }

    async fn read_strings(table: &Table, col: &str) -> Vec<Option<String>> {
        let scan = table.scan().select_all().build().unwrap();
        let mut stream = scan.to_arrow().await.unwrap();
        let mut out = Vec::new();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let idx = batch.schema().index_of(col).unwrap();
            let a = batch.column(idx).as_any().downcast_ref::<LargeStringArray>().unwrap();
            out.extend(a.iter().map(|v| v.map(|x| x.to_string())));
        }
        out
    }

    fn pairs() -> Vec<(String, String)> {
        vec![("t".to_string(), "t__us".to_string()), ("k".to_string(), "k__us".to_string())]
    }

    #[tokio::test]
    async fn rewrite_fills_the_targets_keeps_the_sources_and_the_final_swap_keeps_the_ids() {
        let (_wh, catalog, ident, table) = setup().await;
        let file = write_strings(
            &table,
            "f1",
            vec![Some("36000000"), Some("1970-01-01 10:00:00.250"), Some("34 days, 22:59:59"), Some("-35 days, 1:00:01"), None],
            vec![Some("10:00:00"), Some("-00:00:01"), Some("838:59:59"), Some(""), Some("9:00:00")],
        )
        .await;
        let table = append(
            catalog.as_ref(),
            &table,
            file,
            HashMap::from([("producer.offset.topic-a".to_string(), "77".to_string()), ("other".to_string(), "x".to_string())]),
        )
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = add_long_targets(catalog.as_ref(), &table, &[("t", "t__us"), ("k", "k__us")]).await;
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0), "schema evolution adds no snapshot");

        // dry run: counts, no commit
        let dry = rewrite_time_columns(catalog.as_ref(), &ident, &pairs(), Some(s0), true, HashMap::new(), Some("producer."))
            .await
            .unwrap();
        assert_eq!(dry.snapshot_after, None);
        assert_eq!(dry.rows, 5);
        assert_eq!(dry.records_before, Some(5));
        assert_eq!(dry.partition_fields, 0);
        assert_eq!(dry.counts, ShapeCounts { digits: 1, dated: 1, clock: 4, days: 2, nulls: 2 });
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0));

        let out = rewrite_time_columns(catalog.as_ref(), &ident, &pairs(), Some(s0), false, HashMap::from([("note".to_string(), "test".to_string())]), Some("producer."))
            .await
            .unwrap();
        assert_eq!(out.files_deleted, 1);
        assert_eq!(out.files_added, 1);
        assert_eq!(out.rows, 5);
        let table = catalog.load_table(&ident).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();
        assert_eq!(out.snapshot_after, Some(s1));
        assert_ne!(s1, s0);
        let t_want = vec![Some(us(10, 0, 0)), Some(us(10, 0, 0) + 250_000), Some(34 * US_PER_DAY + us(22, 59, 59)), Some(-35 * US_PER_DAY + us(1, 0, 1)), None];
        let k_want = vec![Some(us(10, 0, 0)), Some(-US_PER_SEC), Some(us(838, 59, 59)), None, Some(us(9, 0, 0))];
        assert_eq!(read_longs(&table, "t__us").await, t_want);
        assert_eq!(read_longs(&table, "k__us").await, k_want);
        // the sources are untouched — readable until the swap
        assert_eq!(read_strings(&table, "t").await[0].as_deref(), Some("36000000"));
        assert_eq!(read_strings(&table, "k").await[2].as_deref(), Some("838:59:59"));
        let summary = &table.metadata().current_snapshot().unwrap().summary().additional_properties;
        assert_eq!(summary.get("producer.offset.topic-a").map(String::as_str), Some("77"), "the ledger key is carried forward");
        assert_eq!(summary.get("other"), None, "only the prefix is carried");
        assert_eq!(summary.get("note").map(String::as_str), Some("test"));
        assert_eq!(summary.get("time-rewrite.columns").map(String::as_str), Some("t->t__us,k->k__us"));

        // the swap: drop the strings, move the longs onto their names — one metadata commit, no snapshot
        let before_ids: Vec<i32> = ["t__us", "k__us"].iter().map(|n| table.metadata().current_schema().field_by_name(n).unwrap().id).collect();
        let table = swap_names(catalog.as_ref(), &table, &[("t", "t__us"), ("k", "k__us")]).await;
        assert_eq!(table.metadata().current_snapshot_id(), Some(s1));
        let schema = table.metadata().current_schema();
        assert!(schema.field_by_name("t__us").is_none() && schema.field_by_name("k__us").is_none());
        let after_ids: Vec<i32> = ["t", "k"].iter().map(|n| schema.field_by_name(n).unwrap().id).collect();
        assert_eq!(after_ids, before_ids, "the rename keeps the field ids");
        assert_eq!(read_longs(&table, "t").await, t_want);
        assert_eq!(read_longs(&table, "k").await, k_want);
        // the swap created no snapshot: the head still RECORDS the pre-swap schema — a reader that
        // binds the snapshot's schema would see the string. An empty append ("touch") records the current one.
        assert_ne!(table.metadata().current_snapshot().unwrap().schema_id(), Some(table.metadata().current_schema_id()));
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .set_snapshot_properties(HashMap::from([("note".to_string(), "touch".to_string())]))
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
        let head = table.metadata().current_snapshot().unwrap().clone();
        assert_ne!(head.snapshot_id(), s1);
        assert_eq!(head.schema_id(), Some(table.metadata().current_schema_id()));
        assert_eq!(head.summary().additional_properties.get("total-records").map(String::as_str), Some("5"));
        assert_eq!(read_longs(&table, "t").await, t_want);

        // a second run finds no string source
        let again = rewrite_time_columns(catalog.as_ref(), &ident, &pairs(), None, true, HashMap::new(), None).await;
        assert!(again.is_err());
    }

    #[tokio::test]
    async fn a_rewrite_rolls_back_to_its_base_snapshot_in_one_metadata_commit() {
        let (_wh, catalog, ident, table) = setup().await;
        let file = write_strings(&table, "f1", vec![Some("10:00:00"), Some("36000000")], vec![Some("1:00:00"), None]).await;
        let table = append(catalog.as_ref(), &table, file, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = add_long_targets(catalog.as_ref(), &table, &[("t", "t__us")]).await;
        let out = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s0), false, HashMap::new(), None)
            .await
            .unwrap();
        let s1 = out.snapshot_after.unwrap();
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_longs(&table, "t__us").await, vec![Some(us(10, 0, 0)), Some(us(10, 0, 0))]);
        // roll back: main moves to s0, s1 stays in the metadata, the long column reads NULL (the old files)
        let tx = Transaction::new(&table);
        let table = tx.rollback_to_snapshot(s0).apply(tx).unwrap().commit(catalog.as_ref()).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0));
        assert!(table.metadata().snapshot_by_id(s1).is_some(), "the rewrite snapshot is kept, not expired");
        assert_eq!(read_longs(&table, "t__us").await, vec![None, None]);
        assert_eq!(read_strings(&table, "t").await, vec![Some("10:00:00".to_string()), Some("36000000".to_string())]);
        // rolling back to the current snapshot or to an unknown id is refused
        let tx = Transaction::new(&table);
        assert!(tx.rollback_to_snapshot(s0).apply(tx).unwrap().commit(catalog.as_ref()).await.is_err());
        let tx = Transaction::new(&table);
        assert!(tx.rollback_to_snapshot(s0 + 7).apply(tx).unwrap().commit(catalog.as_ref()).await.is_err());
        // the rewrite can run again from s0 (the targets are still there)
        let again = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s0), false, HashMap::new(), None)
            .await
            .unwrap();
        assert_ne!(again.snapshot_after, Some(s1));
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_longs(&table, "t__us").await, vec![Some(us(10, 0, 0)), Some(us(10, 0, 0))]);
    }

    /// A pk-only equality-delete file for `ids` through the crate's own writer.
    async fn write_eq_delete(table: &Table, ids: &[i64]) -> DataFile {
        use crate::writer::base_writer::equality_delete_writer::{EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig};
        let pk_id = table.metadata().current_schema().field_id_by_name("id").unwrap();
        let config = EqualityDeleteWriterConfig::new(vec![pk_id], table.metadata().current_schema().clone()).unwrap();
        let delete_schema = Arc::new(crate::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap());
        let batch = RecordBatch::try_new(config.projected_arrow_schema_ref().clone(), vec![Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef]).unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(table), delete_schema),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new("eqdel".to_string(), None, DataFileFormat::Parquet),
        );
        let mut w = EqualityDeleteFileWriterBuilder::new(rolling, config).build(None).await.unwrap();
        w.write(batch).await.unwrap();
        w.close().await.unwrap().into_iter().next().unwrap()
    }

    #[tokio::test]
    async fn a_bound_delete_file_refuses_and_a_dangling_one_is_removed_with_the_rewrite() {
        let (_wh, catalog, ident, table) = setup().await;
        let f1 = write_strings(&table, "f1", vec![Some("10:00:00"), Some("36000000")], vec![Some("1:00:00"), None]).await;
        let table = append(catalog.as_ref(), &table, f1.clone(), HashMap::new()).await;
        // an equality delete on id=2, committed after the data file: the scan binds it to f1
        let del = write_eq_delete(&table, &[2]).await;
        let tx = Transaction::new(&table);
        let table = tx.row_delta().add_delete_files(vec![del]).apply(tx).unwrap().commit(catalog.as_ref()).await.unwrap();
        let s_bound = table.metadata().current_snapshot_id().unwrap();
        add_long_targets(catalog.as_ref(), &table, &[("t", "t__us")]).await;
        let res = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s_bound), false, HashMap::new(), None).await;
        assert!(res.is_err(), "a delete file bound to live data refuses the table");
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s_bound), "nothing committed");
        // replace f1 by an identical file under a higher sequence number: the delete now binds to nothing
        let f1b = write_strings(&table, "f1b", vec![Some("10:00:00"), Some("36000000")], vec![Some("1:00:00"), None]).await;
        let tx = Transaction::new(&table);
        let table = tx.rewrite_files().delete_data_files(vec![f1]).add_data_files(vec![f1b]).apply(tx).unwrap().commit(catalog.as_ref()).await.unwrap();
        let s_dangling = table.metadata().current_snapshot_id().unwrap();
        let out = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s_dangling), false, HashMap::new(), None)
            .await
            .unwrap();
        assert_eq!(out.delete_files_removed, 1);
        assert_eq!(out.rows, 2, "the dangling delete removed nothing");
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read_longs(&table, "t__us").await, vec![Some(us(10, 0, 0)), Some(us(10, 0, 0))]);
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let ml = table.manifest_list_reader(&snapshot).load().await.unwrap();
        let mut live_deletes = 0;
        for mf in ml.entries() {
            let m = mf.load_manifest(table.file_io()).await.unwrap();
            for e in m.entries() {
                if e.status() != ManifestStatus::Deleted && e.content_type() != DataContentType::Data {
                    live_deletes += 1;
                }
            }
        }
        assert_eq!(live_deletes, 0, "the dangling delete file is gone from the live manifests");
    }

    #[tokio::test]
    async fn an_unknown_spelling_refuses_the_table_before_any_commit() {
        let (_wh, catalog, ident, table) = setup().await;
        let file = write_strings(&table, "f1", vec![Some("10:00:00"), Some("noon")], vec![Some("1"), Some("2")]).await;
        let table = append(catalog.as_ref(), &table, file, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        add_long_targets(catalog.as_ref(), &table, &[("t", "t__us")]).await;
        let res = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s0), false, HashMap::new(), None).await;
        assert!(res.is_err());
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0), "nothing committed");
    }

    #[tokio::test]
    async fn a_missing_target_or_a_stale_base_snapshot_is_refused() {
        let (_wh, catalog, ident, table) = setup().await;
        let file = write_strings(&table, "f1", vec![Some("10:00:00")], vec![Some("1")]).await;
        let table = append(catalog.as_ref(), &table, file, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        // no long target yet
        let res = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s0), false, HashMap::new(), None).await;
        assert!(res.is_err());
        add_long_targets(catalog.as_ref(), &table, &[("t", "t__us")]).await;
        let res = rewrite_time_columns(catalog.as_ref(), &ident, &[("t".to_string(), "t__us".to_string())], Some(s0 + 1), false, HashMap::new(), None).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn an_identity_partitioned_table_is_rewritten_per_partition() {
        let (_wh, catalog, ident, table) = setup_with(true).await;
        let spec = table.metadata().default_partition_spec().clone();
        let schema = table.metadata().current_schema().clone();
        let key = |v: bool| PartitionKey::new(spec.as_ref().clone(), schema.clone(), Struct::from_iter(vec![Some(Literal::bool(v))]));
        let f_live = write_strings_in(&table, "live", vec![Some("36000000"), Some("10:00:00")], vec![Some("1:00:00"), None], false, Some(key(false))).await;
        let f_bf = write_strings_in(&table, "backfill", vec![Some("2 days, 1:00:00")], vec![Some("-00:00:01")], true, Some(key(true))).await;
        let table = append(catalog.as_ref(), &table, f_live, HashMap::new()).await;
        let table = append(catalog.as_ref(), &table, f_bf, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        add_long_targets(catalog.as_ref(), &table, &[("t", "t__us"), ("k", "k__us")]).await;
        let out = rewrite_time_columns(catalog.as_ref(), &ident, &pairs(), Some(s0), false, HashMap::new(), None).await.unwrap();
        assert_eq!(out.partition_fields, 1);
        assert_eq!(out.files_deleted, 2);
        assert_eq!(out.files_added, 2, "one rewritten file per partition");
        assert_eq!(out.rows, 3);
        assert_eq!(out.records_before, Some(3));
        let table = catalog.load_table(&ident).await.unwrap();
        let mut got: Vec<(Option<i64>, Option<i64>)> = {
            let scan = table.scan().select_all().build().unwrap();
            let mut stream = scan.to_arrow().await.unwrap();
            let mut v = Vec::new();
            while let Some(b) = stream.try_next().await.unwrap() {
                let t = b.column(b.schema().index_of("t__us").unwrap()).as_any().downcast_ref::<Int64Array>().unwrap().clone();
                let k = b.column(b.schema().index_of("k__us").unwrap()).as_any().downcast_ref::<Int64Array>().unwrap().clone();
                for i in 0..b.num_rows() {
                    v.push((if t.is_null(i) { None } else { Some(t.value(i)) }, if k.is_null(i) { None } else { Some(k.value(i)) }));
                }
            }
            v
        };
        got.sort();
        let mut want = vec![
            (Some(us(10, 0, 0)), Some(us(1, 0, 0))),
            (Some(us(10, 0, 0)), None),
            (Some(2 * US_PER_DAY + us(1, 0, 0)), Some(-US_PER_SEC)),
        ];
        want.sort();
        assert_eq!(got, want);
        // every new file carries its partition
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let ml = table.manifest_list_reader(&snapshot).load().await.unwrap();
        let mut parts = Vec::new();
        for mf in ml.entries() {
            let m = mf.load_manifest(table.file_io()).await.unwrap();
            for e in m.entries() {
                if e.status() != ManifestStatus::Deleted {
                    parts.push(format!("{:?}", e.data_file().partition()));
                }
            }
        }
        parts.sort();
        assert_eq!(parts.len(), 2);
        assert_ne!(parts[0], parts[1]);
    }
}
