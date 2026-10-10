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
//! The caller first evolves the table's schema (delete the string column, add a
//! `long` column under the same name); this module then reads the base snapshot
//! under ITS schema (the string is still physically there), converts, writes new
//! data files under the current schema and commits a single `Replace` snapshot
//! that removes every old data file. No delete files are written. The base
//! snapshot's summary properties that start with `carry_summary_prefix` are
//! copied onto the new snapshot (a producer's offsets-in-snapshot ledger keeps
//! its home when older snapshots expire).
//!
//! Spellings — the only transformation, pinned by tests:
//!   * `^-?[0-9]+$`                          milliseconds since midnight → × 1000
//!   * `^1970-01-01[ T]H:MM:SS(.f+)?$`        a dated clock (time-of-day) → µs
//!   * `^-?H{1,3}:MM:SS(.f+)?$`               a clock, hours never wrapped → signed µs
//!   * `^-?N days?, H:MM:SS(.f+)?$`           Python `timedelta` text → signed µs
//!   * NULL or empty                         NULL
//!   * anything else                         the rewrite REFUSES the table before any commit.
//!
//! MySQL `TIME` spans -838:59:59 .. 838:59:59; a parsed value outside that range is refused too.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use futures::TryStreamExt;
use once_cell::sync::Lazy;
use regex::Regex;
use uuid::Uuid;

use crate::arrow::schema_to_arrow_schema;
use crate::atomic_replace::{conform_batch, writer_properties};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, ManifestStatus, PrimitiveType, Type,
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

static RE_DIGITS: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-?[0-9]+$").expect("static regex"));
static RE_DATED: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^1970-01-01[ T]([0-9]{1,2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,6}))?$")
        .expect("static regex")
});
static RE_CLOCK: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(-)?([0-9]{1,3}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,6}))?$").expect("static regex")
});
static RE_DAYS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(-?)([0-9]+) days?, ([0-9]{1,2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,6}))?$")
        .expect("static regex")
});

/// Which historical spelling a value carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeShape {
    Digits,
    Dated,
    Clock,
    Days,
}

/// How many values of each spelling a rewrite met.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeCounts {
    pub digits: u64,
    pub dated: u64,
    pub clock: u64,
    pub days: u64,
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

fn frac_us(frac: Option<&str>) -> i64 {
    match frac {
        None => 0,
        Some(f) => {
            let mut digits = f.to_string();
            while digits.len() < 6 {
                digits.push('0');
            }
            digits.parse::<i64>().unwrap_or(0)
        }
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

/// Parse one spelling into signed microseconds since midnight. `Ok(None)` for an
/// empty string; an unrecognised spelling or an out-of-range value is an error.
pub fn mysql_time_text_to_us(s: &str) -> Result<Option<(i64, TimeShape)>> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(None);
    }
    if RE_DIGITS.is_match(t) {
        let ms: i64 = t
            .parse()
            .map_err(|_| invalid(format!("TIME milliseconds out of range: `{t}`")))?;
        let us = ms
            .checked_mul(1000)
            .ok_or_else(|| invalid(format!("TIME milliseconds out of range: `{t}`")))?;
        return checked(us, TimeShape::Digits, t);
    }
    if let Some(c) = RE_DATED.captures(t) {
        let h: i64 = c[1].parse().expect("regex digits");
        let m: i64 = c[2].parse().expect("regex digits");
        let sec: i64 = c[3].parse().expect("regex digits");
        if h > 23 || m > 59 || sec > 59 {
            return Err(invalid(format!("dated TIME has a field out of range: `{t}`")));
        }
        let us = (h * 3600 + m * 60 + sec) * US_PER_SEC + frac_us(c.get(4).map(|x| x.as_str()));
        return checked(us, TimeShape::Dated, t);
    }
    if let Some(c) = RE_CLOCK.captures(t) {
        let neg = c.get(1).is_some();
        let h: i64 = c[2].parse().expect("regex digits");
        let m: i64 = c[3].parse().expect("regex digits");
        let sec: i64 = c[4].parse().expect("regex digits");
        if m > 59 || sec > 59 {
            return Err(invalid(format!("clock TIME has a field out of range: `{t}`")));
        }
        let us = (h * 3600 + m * 60 + sec) * US_PER_SEC + frac_us(c.get(5).map(|x| x.as_str()));
        return checked(if neg { -us } else { us }, TimeShape::Clock, t);
    }
    if let Some(c) = RE_DAYS.captures(t) {
        // Python: str(timedelta(days=-35, seconds=3601)) == "-35 days, 1:00:01"
        // == -(35 × 86400) + 3601 seconds — the day count carries the sign, the
        // clock part is always added.
        let neg_days = &c[1] == "-";
        let days: i64 = c[2]
            .parse()
            .map_err(|_| invalid(format!("timedelta days out of range: `{t}`")))?;
        let h: i64 = c[3].parse().expect("regex digits");
        let m: i64 = c[4].parse().expect("regex digits");
        let sec: i64 = c[5].parse().expect("regex digits");
        if h > 23 || m > 59 || sec > 59 {
            return Err(invalid(format!("timedelta TIME has a field out of range: `{t}`")));
        }
        let day_us = days
            .checked_mul(US_PER_DAY)
            .ok_or_else(|| invalid(format!("timedelta days out of range: `{t}`")))?;
        let tod = (h * 3600 + m * 60 + sec) * US_PER_SEC + frac_us(c.get(6).map(|x| x.as_str()));
        let us = if neg_days { -day_us + tod } else { day_us + tod };
        return checked(us, TimeShape::Days, t);
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
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("TIME rewrite expects a string column, found {other}"),
            ));
        }
    }
    Ok((Int64Array::from(out), counts))
}

/// What a rewrite did.
#[derive(Debug, Clone, Default)]
pub struct RewriteOutcome {
    pub snapshot_before: i64,
    /// `None` on a dry run (nothing written, nothing committed).
    pub snapshot_after: Option<i64>,
    pub rows: u64,
    pub files_deleted: usize,
    pub files_added: usize,
    pub counts: ShapeCounts,
}

/// Rewrite `columns` of `ident` from their string spellings into signed
/// microseconds, copy-on-write, in one `Replace` snapshot. The current schema
/// must already carry each column as `long` and the current snapshot's schema
/// must still carry it as a string; `base_snapshot`, when given, must be the
/// current snapshot (optimistic concurrency, enforced again at commit). With
/// `dry_run` the data is parsed and counted but nothing is written or committed.
pub async fn rewrite_time_columns(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    columns: &[String],
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
    let old_schema = snapshot.schema(table.metadata())?;
    let cur_schema = table.metadata().current_schema().clone();
    for col in columns {
        match cur_schema.field_by_name(col).map(|f| f.field_type.as_ref()) {
            Some(Type::Primitive(PrimitiveType::Long)) => {}
            other => {
                return Err(invalid(format!(
                    "{ident}: the current schema must carry `{col}` as long before the rewrite (found {other:?}) — evolve the schema first"
                )));
            }
        }
        match old_schema.field_by_name(col).map(|f| f.field_type.as_ref()) {
            Some(Type::Primitive(PrimitiveType::String)) => {}
            other => {
                return Err(invalid(format!(
                    "{ident}: snapshot {s0}'s schema does not carry `{col}` as a string (found {other:?}) — nothing to convert"
                )));
            }
        }
    }
    if !table.metadata().default_partition_spec().fields().is_empty() {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!("{ident}: the TIME rewrite supports unpartitioned tables only"),
        ));
    }

    // Every live data file of the base snapshot is replaced; delete files are not expected.
    let manifest_list = table.manifest_list_reader(&snapshot).load().await?;
    let mut old_files: Vec<DataFile> = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            match entry.content_type() {
                DataContentType::Data => old_files.push(entry.data_file().clone()),
                other => {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!("{ident}: snapshot {s0} carries a {other:?} file — the rewrite expects a table without delete files"),
                    ));
                }
            }
        }
    }
    let mut outcome = RewriteOutcome {
        snapshot_before: s0,
        files_deleted: old_files.len(),
        ..Default::default()
    };
    if old_files.is_empty() {
        return Ok(outcome);
    }

    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&cur_schema)?);
    let run = Uuid::now_v7();
    let mut writer = if dry_run {
        None
    } else {
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(&table), cur_schema.clone()),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata())?,
            DefaultFileNameGenerator::new(
                format!("time-rewrite-{run}"),
                None,
                DataFileFormat::Parquet,
            ),
        );
        Some(DataFileWriterBuilder::new(rolling).build(None).await?)
    };

    let scan = table.scan().snapshot_id(s0).select_all().build()?;
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
        for col in columns {
            let idx = batch.schema().index_of(col).map_err(|_| {
                invalid(format!("{ident}: scanned batch lacks column `{col}`"))
            })?;
            let (converted, counts) = time_text_array_to_us(&cols[idx])?;
            outcome.counts.add(counts);
            fields[idx] = Field::new(col, DataType::Int64, true);
            cols[idx] = Arc::new(converted);
        }
        outcome.rows += batch.num_rows() as u64;
        if let Some(w) = writer.as_mut() {
            let rebuilt = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
                .map_err(|e| invalid(format!("rebuilding the converted batch: {e}")))?;
            let conformed = conform_batch(&rebuilt, &target_arrow)?;
            w.write(conformed).await?;
        }
    }
    let Some(writer) = writer else {
        return Ok(outcome);
    };
    let new_files = writer.close().await?;
    outcome.files_added = new_files.len();

    let mut props = snapshot_properties;
    if let Some(prefix) = carry_summary_prefix {
        for (k, v) in &snapshot.summary().additional_properties {
            if k.starts_with(prefix) && !props.contains_key(k) {
                props.insert(k.clone(), v.clone());
            }
        }
    }
    props.insert("time-rewrite.columns".to_string(), columns.join(","));
    props.insert("time-rewrite.base-snapshot".to_string(), s0.to_string());

    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .delete_data_files(old_files)
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

    use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use tempfile::TempDir;

    use super::*;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{FormatVersion, NestedField, Schema};
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
    }

    async fn write_strings(table: &Table, name: &str, t: Vec<Option<&str>>, k: Vec<Option<&str>>) -> DataFile {
        let schema = table.metadata().current_schema().clone();
        let arrow = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let n = t.len();
        let ids: Vec<i64> = (1..=n as i64).collect();
        let batch = RecordBatch::try_new(arrow, vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(t)) as ArrayRef,
            Arc::new(StringArray::from(k)) as ArrayRef,
        ])
        .unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(table), schema),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
        );
        let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
        writer.write(batch).await.unwrap();
        let files = writer.close().await.unwrap();
        assert_eq!(files.len(), 1);
        files.into_iter().next().unwrap()
    }

    async fn setup() -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
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
            ])
            .build()
            .unwrap();
        let ident = TableIdent::new(ns.clone(), "times".to_string());
        let table = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("times".to_string())
                    .schema(schema)
                    .format_version(FormatVersion::V2)
                    .build(),
            )
            .await
            .unwrap();
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

    /// delete + re-add under the same name, as TWO commits (the schema action is
    /// not relied on to order a delete before an add of the same name).
    async fn retype_to_long(catalog: &dyn Catalog, table: &Table, cols: &[&str]) -> Table {
        let tx = Transaction::new(table);
        let mut action = tx.update_schema();
        for c in cols {
            action = action.delete_column(*c);
        }
        let table = action.apply(tx).unwrap().commit(catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let mut action = tx.update_schema();
        for c in cols {
            action = action.add_column(AddColumn::optional(*c, Type::Primitive(PrimitiveType::Long)));
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

    #[tokio::test]
    async fn rewrite_converges_four_spellings_in_one_replace_snapshot() {
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
            HashMap::from([("pulse.kafka.offset.topic-a".to_string(), "77".to_string()), ("other".to_string(), "x".to_string())]),
        )
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = retype_to_long(catalog.as_ref(), &table, &["t", "k"]).await;
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0), "schema evolution adds no snapshot");

        // dry run: counts, no commit
        let dry = rewrite_time_columns(catalog.as_ref(), &ident, &["t".to_string(), "k".to_string()], Some(s0), true, HashMap::new(), Some("pulse."))
            .await
            .unwrap();
        assert_eq!(dry.snapshot_after, None);
        assert_eq!(dry.rows, 5);
        assert_eq!(dry.counts, ShapeCounts { digits: 1, dated: 1, clock: 4, days: 2, nulls: 2 });
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0));

        let out = rewrite_time_columns(catalog.as_ref(), &ident, &["t".to_string(), "k".to_string()], Some(s0), false, HashMap::from([("note".to_string(), "test".to_string())]), Some("pulse."))
            .await
            .unwrap();
        assert_eq!(out.files_deleted, 1);
        assert_eq!(out.files_added, 1);
        assert_eq!(out.rows, 5);
        let table = catalog.load_table(&ident).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();
        assert_eq!(out.snapshot_after, Some(s1));
        assert_ne!(s1, s0);
        assert_eq!(
            read_longs(&table, "t").await,
            vec![Some(us(10, 0, 0)), Some(us(10, 0, 0) + 250_000), Some(34 * US_PER_DAY + us(22, 59, 59)), Some(-35 * US_PER_DAY + us(1, 0, 1)), None]
        );
        assert_eq!(
            read_longs(&table, "k").await,
            vec![Some(us(10, 0, 0)), Some(-US_PER_SEC), Some(us(838, 59, 59)), None, Some(us(9, 0, 0))]
        );
        let summary = &table.metadata().current_snapshot().unwrap().summary().additional_properties;
        assert_eq!(summary.get("pulse.kafka.offset.topic-a").map(String::as_str), Some("77"), "the ledger key is carried forward");
        assert_eq!(summary.get("other"), None, "only the prefix is carried");
        assert_eq!(summary.get("note").map(String::as_str), Some("test"));
        assert_eq!(summary.get("time-rewrite.columns").map(String::as_str), Some("t,k"));

        // a second run finds no string column to convert
        let again = rewrite_time_columns(catalog.as_ref(), &ident, &["t".to_string()], None, true, HashMap::new(), None).await;
        assert!(again.is_err());
    }

    #[tokio::test]
    async fn an_unknown_spelling_refuses_the_table_before_any_commit() {
        let (_wh, catalog, ident, table) = setup().await;
        let file = write_strings(&table, "f1", vec![Some("10:00:00"), Some("noon")], vec![Some("1"), Some("2")]).await;
        let table = append(catalog.as_ref(), &table, file, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        retype_to_long(catalog.as_ref(), &table, &["t"]).await;
        let res = rewrite_time_columns(catalog.as_ref(), &ident, &["t".to_string()], Some(s0), false, HashMap::new(), None).await;
        assert!(res.is_err());
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), Some(s0), "nothing committed");
    }

    #[tokio::test]
    async fn a_stale_base_snapshot_is_refused() {
        let (_wh, catalog, ident, table) = setup().await;
        let file = write_strings(&table, "f1", vec![Some("10:00:00")], vec![Some("1")]).await;
        let table = append(catalog.as_ref(), &table, file, HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        retype_to_long(catalog.as_ref(), &table, &["t"]).await;
        let res = rewrite_time_columns(catalog.as_ref(), &ident, &["t".to_string()], Some(s0 + 1), false, HashMap::new(), None).await;
        assert!(res.is_err());
    }
}
