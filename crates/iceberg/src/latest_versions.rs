//! LATEST LIVE VERSION PER KEY — a read-only lookup over an append-only,
//! versioned table: for each requested key, the newest LIVE row (the table's
//! deletes applied by the reader) ranked by `COALESCE(source instant, capture
//! instant)`, then a sequence column, then an offset column, returning that
//! row's value in a caller-named hash column and its source instant in
//! epoch-µs. A key with no live row is absent from the result; nothing here
//! writes.
//!
//! Keys. A STRING key column matches the requested text exactly. A STRUCT key
//! column (a primary-key tuple) matches a requested text that is a flat JSON
//! object `{"<field>": <value>, …}` field by field on CANONICAL text — integers
//! as decimal digits, strings verbatim, booleans `true`/`false`, JSON null = an
//! Arrow null — so `{"id": 7}` meets a row whose `key.id` is `7_i64`. Planning
//! prunes files with a lexicographic bracket on a string key (`[min, max]` of
//! the requested texts — never an IN list, whose evaluation stops pruning past
//! a few hundred values) or an integer / string bracket on a single-field
//! struct key; any other shape plans on the whole table (correct, less pruning).
//! Row-wise matching is exact in every case.
//!
//! Instants. A `Timestamp` column of any unit (zoned or naive — the stored
//! value is the instant) normalises to epoch-µs; an INTEGER instant column is
//! read in `VersionColumns::integer_instant_unit` (a document store's epoch-ms
//! `long`, for one). The sequence and offset columns are read as integers; a
//! string offset column parses its digits (anything else is a null component,
//! which sorts lowest).

use std::collections::{BTreeMap, HashMap, HashSet};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, TimeUnit};
use futures::TryStreamExt;

use crate::atomic_replace::KeyMatcher;
use crate::expr::{Predicate, Reference};
use crate::scan::FileScanTask;
use crate::spec::{Datum, PrimitiveType, Type};
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

/// The dotted paths of the columns that order a key's versions.
#[derive(Debug, Clone)]
pub struct VersionColumns {
    /// The instant the row became true at its source (leads the rank; may be
    /// absent from the schema or null on a row).
    pub source_instant: String,
    /// The instant the row was captured (the rank's fallback when the source
    /// instant is null).
    pub capture_instant: String,
    /// A per-stream sequence (second rank component), `None` = unused.
    pub sequence: Option<String>,
    /// A per-stream offset (third rank component), `None` = unused.
    pub offset: Option<String>,
    /// The unit an INTEGER instant column is stored in.
    pub integer_instant_unit: TimeUnit,
}

/// One key's latest live version.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LatestVersion {
    /// The hash column's value on that row, `None` when null.
    pub hash: Option<String>,
    /// The row's source instant as epoch-µs, `None` when null.
    pub source_instant_us: Option<i64>,
}

/// The version RANK `(COALESCE(source, capture), sequence, offset)` — larger
/// wins; a missing component sorts lowest.
pub type Rank = (i64, i64, i64);

/// Build the rank of one observed version (instants in epoch-µs).
pub fn rank(
    source_instant_us: Option<i64>,
    capture_instant_us: Option<i64>,
    sequence: Option<i64>,
    offset: Option<i64>,
) -> Rank {
    (
        source_instant_us.or(capture_instant_us).unwrap_or(i64::MIN),
        sequence.unwrap_or(i64::MIN),
        offset.unwrap_or(i64::MIN),
    )
}

/// Fold one observed version into the per-key map; a strictly greater rank
/// replaces, an equal rank keeps the first seen (the same stream coordinate).
pub fn fold(
    latest: &mut HashMap<String, (Rank, LatestVersion)>,
    key: &str,
    r: Rank,
    v: LatestVersion,
) {
    match latest.get_mut(key) {
        Some((cur, cv)) => {
            if r > *cur {
                *cur = r;
                *cv = v;
            }
        }
        None => {
            latest.insert(key.to_string(), (r, v));
        }
    }
}

/// Epoch-µs from an instant column's RAW integer: a `Timestamp` of any unit
/// (zoned or naive), or an integer column in `integer_unit`.
pub fn instant_us(dt: &DataType, raw: i64, integer_unit: TimeUnit) -> i64 {
    let unit = match dt {
        DataType::Timestamp(unit, _) => *unit,
        _ => integer_unit,
    };
    match unit {
        TimeUnit::Second => raw.saturating_mul(1_000_000),
        TimeUnit::Millisecond => raw.saturating_mul(1_000),
        TimeUnit::Microsecond => raw,
        TimeUnit::Nanosecond => raw.div_euclid(1_000),
    }
}

/// Canonical text of one JSON scalar — the struct-key comparison currency.
fn json_canon(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Null => Some(String::new()),
        _ => None,
    }
}

/// Parse a struct-key text (`{"id": 7}` / `{"a": 1, "b": "x"}`) into its
/// canonical field map; `None` for anything that is not a flat JSON object.
pub fn parse_struct_key(text: &str) -> Option<BTreeMap<String, String>> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = v.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let mut out = BTreeMap::new();
    for (k, v) in obj {
        out.insert(k.clone(), json_canon(v)?);
    }
    Some(out)
}

/// The canonical join of a field map — `a=1\u{1}b=x` in field-name order —
/// the one form both the requested texts and the scanned rows reduce to.
pub fn canonical_join(fields: &BTreeMap<String, String>) -> String {
    let mut s = String::new();
    for (i, (k, v)) in fields.iter().enumerate() {
        if i > 0 {
            s.push('\u{1}');
        }
        s.push_str(k);
        s.push('=');
        s.push_str(v);
    }
    s
}

/// How the requested keys are matched against the key column.
#[derive(Debug)]
enum KeyShape {
    /// The key column is a string: exact text membership.
    Text(HashSet<String>),
    /// The key column is a struct: canonical-join membership over `fields`,
    /// each join mapped back to the requested text.
    Struct {
        fields: Vec<String>,
        by_join: HashMap<String, String>,
    },
}

impl KeyShape {
    fn from_keys(key_is_struct: bool, keys: Vec<String>) -> Self {
        if !key_is_struct {
            return KeyShape::Text(keys.into_iter().collect());
        }
        let mut field_set: HashSet<String> = HashSet::new();
        let mut by_join = HashMap::with_capacity(keys.len());
        for k in keys {
            // a text that is not a flat JSON object can never match a struct
            // key — it stays absent from the result rather than failing the call
            let Some(parsed) = parse_struct_key(&k) else {
                continue;
            };
            field_set.extend(parsed.keys().cloned());
            by_join.insert(canonical_join(&parsed), k);
        }
        let mut fields: Vec<String> = field_set.into_iter().collect();
        fields.sort();
        KeyShape::Struct { fields, by_join }
    }

    fn is_empty(&self) -> bool {
        match self {
            KeyShape::Text(s) => s.is_empty(),
            KeyShape::Struct { by_join, .. } => by_join.is_empty(),
        }
    }
}

/// Walk a dotted path down struct columns to the leaf array.
fn leaf_array(batch: &RecordBatch, dotted: &str) -> Result<ArrayRef> {
    let mut parts = dotted.split('.');
    let root = parts.next().expect("non-empty dotted path");
    let mut current: ArrayRef = batch
        .column(batch.schema().index_of(root).map_err(|e| {
            Error::new(ErrorKind::Unexpected, format!("column `{root}`: {e}"))
        })?)
        .clone();
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
        current = s
            .column_by_name(part)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("`{dotted}`: struct field `{part}` not found"),
                )
            })?
            .clone();
    }
    Ok(current)
}

/// A column cast to Utf8 (`None` when the cast is impossible).
fn as_text(col: &ArrayRef) -> Option<StringArray> {
    let utf8 = arrow_cast::cast(col.as_ref(), &DataType::Utf8).ok()?;
    utf8.as_any().downcast_ref::<StringArray>().cloned()
}

/// A column cast to Int64 (`None` when the cast is impossible); a string
/// column's digits parse and anything else becomes null.
fn as_i64(col: &ArrayRef) -> Option<Int64Array> {
    let i = arrow_cast::cast(col.as_ref(), &DataType::Int64).ok()?;
    i.as_any().downcast_ref::<Int64Array>().cloned()
}

/// The planning predicate for the requested keys (a SUPERSET bracket; the
/// row-wise match is exact), or `None` to plan on the whole table.
fn planning_predicate(
    schema: &crate::spec::Schema,
    key_column: &str,
    shape: &KeyShape,
) -> Option<Predicate> {
    match shape {
        KeyShape::Text(keys) => KeyMatcher::KeySet(keys.clone()).planning_predicate(key_column),
        KeyShape::Struct { fields, by_join } => {
            if fields.len() != 1 {
                return None;
            }
            let path = format!("{key_column}.{}", fields[0]);
            let fid = schema.field_id_by_name(&path)?;
            let ftype = schema.field_by_id(fid)?.field_type.clone();
            // the values are the canonical texts after the `name=` prefix
            let values: Vec<&str> = by_join
                .keys()
                .map(|j| j.split_once('=').map(|(_, v)| v).unwrap_or(""))
                .collect();
            match *ftype {
                Type::Primitive(PrimitiveType::Long) => {
                    let nums: Vec<i64> = values
                        .iter()
                        .map(|v| v.parse::<i64>().ok())
                        .collect::<Option<Vec<_>>>()?;
                    let lo = *nums.iter().min()?;
                    let hi = *nums.iter().max()?;
                    Some(
                        Reference::new(path.clone())
                            .greater_than_or_equal_to(Datum::long(lo))
                            .and(Reference::new(path).less_than_or_equal_to(Datum::long(hi))),
                    )
                }
                Type::Primitive(PrimitiveType::Int) => {
                    let nums: Vec<i32> = values
                        .iter()
                        .map(|v| v.parse::<i32>().ok())
                        .collect::<Option<Vec<_>>>()?;
                    let lo = *nums.iter().min()?;
                    let hi = *nums.iter().max()?;
                    Some(
                        Reference::new(path.clone())
                            .greater_than_or_equal_to(Datum::int(lo))
                            .and(Reference::new(path).less_than_or_equal_to(Datum::int(hi))),
                    )
                }
                Type::Primitive(PrimitiveType::String) => {
                    let lo = values.iter().min()?.to_string();
                    let hi = values.iter().max()?.to_string();
                    Some(
                        Reference::new(path.clone())
                            .greater_than_or_equal_to(Datum::string(lo))
                            .and(Reference::new(path).less_than_or_equal_to(Datum::string(hi))),
                    )
                }
                _ => None,
            }
        }
    }
}

/// The latest live version of every requested key — see the module doc.
/// `key_column` is the dotted key path, `hash_column` a top-level column
/// (absent from the schema = every hash `None`). An empty `keys` or a table
/// without a snapshot returns an empty map.
pub async fn latest_versions_by_key(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    key_column: &str,
    hash_column: &str,
    version_columns: &VersionColumns,
    keys: Vec<String>,
) -> Result<HashMap<String, LatestVersion>> {
    let keys: Vec<String> = keys.into_iter().filter(|k| !k.is_empty()).collect();
    if keys.is_empty() {
        return Ok(HashMap::new());
    }
    let table = catalog.load_table(ident).await?;
    let Some(snapshot_id) = table.metadata().current_snapshot_id() else {
        return Ok(HashMap::new());
    };
    let schema = table.metadata().current_schema().clone();
    let key_fid = schema.field_id_by_name(key_column).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("key column `{key_column}` not found in table schema"),
        )
    })?;
    let key_is_struct = matches!(
        *schema
            .field_by_id(key_fid)
            .expect("field id resolved above")
            .field_type,
        Type::Struct(_)
    );
    let shape = KeyShape::from_keys(key_is_struct, keys);
    if shape.is_empty() {
        return Ok(HashMap::new());
    }
    // project the roots of every column read: the key's, the version columns'
    // and the hash column (a struct root brings all of its fields)
    let mut roots: Vec<&str> = Vec::new();
    let mut push_root = |dotted: &str| {
        if schema.field_id_by_name(dotted).is_some() {
            let root = dotted.split('.').next().expect("non-empty dotted path");
            if !roots.iter().any(|r| *r == root) {
                roots.push(root);
            }
        }
    };
    push_root(key_column);
    push_root(&version_columns.source_instant);
    push_root(&version_columns.capture_instant);
    if let Some(c) = &version_columns.sequence {
        push_root(c);
    }
    if let Some(c) = &version_columns.offset {
        push_root(c);
    }
    let has_hash = schema.field_id_by_name(hash_column).is_some();
    push_root(hash_column);
    let select_cols: Vec<String> = roots.iter().map(|r| r.to_string()).collect();
    let mut scan = table.scan().snapshot_id(snapshot_id).select(select_cols);
    if let Some(pred) = planning_predicate(&schema, key_column, &shape) {
        scan = scan.with_filter(pred);
    }
    let scan = scan.build()?;
    let tasks: Vec<FileScanTask> = scan
        .plan_files()
        .await?
        .try_collect::<Vec<FileScanTask>>()
        .await?
        .into_iter()
        .map(|mut t| {
            t.predicate = None; // row-wise matching happens below
            t
        })
        .collect();
    if tasks.is_empty() {
        return Ok(HashMap::new());
    }
    let reader = table.reader_builder().build();
    let task_stream = Box::pin(futures::stream::iter(
        tasks.into_iter().map(Ok).collect::<Vec<_>>(),
    )) as crate::scan::FileScanTaskStream;
    let mut stream = reader.read(task_stream)?.stream();

    let present = |dotted: &str| schema.field_id_by_name(dotted).is_some();
    let unit = version_columns.integer_instant_unit;
    let mut latest: HashMap<String, (Rank, LatestVersion)> = HashMap::new();
    while let Some(batch) = stream.try_next().await? {
        if batch.num_rows() == 0 {
            continue;
        }
        let key_arr = leaf_array(&batch, key_column)?;
        // every rank component: the column's raw integers + its data type
        let int_col = |dotted: &str| -> Option<(DataType, Int64Array)> {
            if !present(dotted) {
                return None;
            }
            let col = leaf_array(&batch, dotted).ok()?;
            let dt = col.data_type().clone();
            as_i64(&col).map(|a| (dt, a))
        };
        let source = int_col(&version_columns.source_instant);
        let capture = int_col(&version_columns.capture_instant);
        let sequence = version_columns
            .sequence
            .as_deref()
            .and_then(int_col)
            .map(|(_, a)| a);
        let offset = version_columns
            .offset
            .as_deref()
            .and_then(int_col)
            .map(|(_, a)| a);
        let hash = if has_hash {
            batch
                .schema()
                .index_of(hash_column)
                .ok()
                .and_then(|i| as_text(batch.column(i)))
        } else {
            None
        };
        let get_i64 = |a: &Option<Int64Array>, row: usize| -> Option<i64> {
            a.as_ref()
                .filter(|a| !a.is_null(row))
                .map(|a| a.value(row))
        };
        let get_instant = |a: &Option<(DataType, Int64Array)>, row: usize| -> Option<i64> {
            a.as_ref()
                .filter(|(_, a)| !a.is_null(row))
                .map(|(dt, a)| instant_us(dt, a.value(row), unit))
        };
        let version_at = |row: usize| -> (Rank, LatestVersion) {
            let s = get_instant(&source, row);
            let r = rank(
                s,
                get_instant(&capture, row),
                get_i64(&sequence, row),
                get_i64(&offset, row),
            );
            let v = LatestVersion {
                hash: hash
                    .as_ref()
                    .filter(|h| !h.is_null(row))
                    .map(|h| h.value(row).to_string()),
                source_instant_us: s,
            };
            (r, v)
        };

        match &shape {
            KeyShape::Text(keys) => {
                let Some(texts) = as_text(&key_arr) else {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("`{key_column}` must be a string column"),
                    ));
                };
                for row in 0..batch.num_rows() {
                    if texts.is_null(row) {
                        continue;
                    }
                    let k = texts.value(row);
                    if !keys.contains(k) {
                        continue;
                    }
                    let (r, v) = version_at(row);
                    fold(&mut latest, k, r, v);
                }
            }
            KeyShape::Struct { fields, by_join } => {
                let Some(st_arr) = key_arr.as_any().downcast_ref::<StructArray>() else {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("`{key_column}` must be a struct column"),
                    ));
                };
                // every requested field as text; a field the struct lacks = no match
                let cols: Option<Vec<StringArray>> = fields
                    .iter()
                    .map(|f| st_arr.column_by_name(f).and_then(as_text))
                    .collect();
                let Some(cols) = cols else {
                    continue;
                };
                for row in 0..batch.num_rows() {
                    if st_arr.is_null(row) {
                        continue;
                    }
                    let mut m = BTreeMap::new();
                    for (f, c) in fields.iter().zip(cols.iter()) {
                        m.insert(
                            f.clone(),
                            if c.is_null(row) {
                                String::new()
                            } else {
                                c.value(row).to_string()
                            },
                        );
                    }
                    let Some(k) = by_join.get(&canonical_join(&m)) else {
                        continue;
                    };
                    let (r, v) = version_at(row);
                    fold(&mut latest, k, r, v);
                }
            }
        }
    }
    Ok(latest.into_iter().map(|(k, (_, v))| (k, v)).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        BooleanArray, Int64Array, LargeStringArray, RecordBatch, StringArray, StructArray,
        TimestampMicrosecondArray,
    };
    use arrow_schema::{DataType, Fields};
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{
        DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey, Schema, Struct,
        StructType, Transform, UnboundPartitionSpec,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{CatalogBuilder, NamespaceIdent, TableCreation};

    // ── pure helpers ─────────────────────────────────────────────────────────

    #[test]
    fn rank_leads_with_the_source_instant_and_falls_back_to_capture() {
        assert!(rank(Some(100), Some(999), None, None) < rank(Some(101), Some(1), None, None));
        assert_eq!(rank(None, Some(5), None, None).0, 5);
        assert!(rank(Some(7), None, Some(1), Some(9)) < rank(Some(7), None, Some(2), Some(0)));
        assert!(rank(Some(7), None, Some(2), Some(0)) < rank(Some(7), None, Some(2), Some(1)));
        assert!(rank(Some(7), None, None, Some(5)) < rank(Some(7), None, Some(0), Some(0)));
    }

    #[test]
    fn fold_keeps_the_greater_rank_and_the_first_of_equals() {
        let mut m = HashMap::new();
        let v = |h: &str, s: Option<i64>| LatestVersion {
            hash: Some(h.into()),
            source_instant_us: s,
        };
        fold(&mut m, "k", (10, 0, 0), v("a", Some(10)));
        fold(&mut m, "k", (5, 0, 0), v("b", Some(5)));
        assert_eq!(m["k"].1.hash.as_deref(), Some("a"));
        fold(&mut m, "k", (10, 0, 0), v("c", Some(10)));
        assert_eq!(m["k"].1.hash.as_deref(), Some("a"), "an equal rank keeps the first");
        fold(&mut m, "k", (10, 0, 1), v("d", None));
        assert_eq!(m["k"].1, v("d", None));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn instant_us_normalises_every_timestamp_unit_and_integer_columns() {
        let tz = Some(Arc::from("UTC"));
        let ms = TimeUnit::Millisecond;
        assert_eq!(
            instant_us(&DataType::Timestamp(TimeUnit::Microsecond, tz.clone()), 1_700_000_000_123_456, ms),
            1_700_000_000_123_456
        );
        assert_eq!(
            instant_us(&DataType::Timestamp(TimeUnit::Millisecond, None), 1_700_000_000_123, ms),
            1_700_000_000_123_000
        );
        assert_eq!(
            instant_us(&DataType::Timestamp(TimeUnit::Second, None), 1_700_000_000, ms),
            1_700_000_000_000_000
        );
        assert_eq!(
            instant_us(&DataType::Timestamp(TimeUnit::Nanosecond, tz), 1_700_000_000_123_456_789, ms),
            1_700_000_000_123_456
        );
        // an integer instant column reads in the caller's unit
        assert_eq!(instant_us(&DataType::Int64, 1_700_000_000_123, ms), 1_700_000_000_123_000);
        assert_eq!(
            instant_us(&DataType::Int64, 1_700_000_000_123_456, TimeUnit::Microsecond),
            1_700_000_000_123_456
        );
    }

    #[test]
    fn struct_key_texts_parse_to_canonical_fields() {
        let p = parse_struct_key(r#"{"id": 7}"#).unwrap();
        assert_eq!(canonical_join(&p), "id=7");
        let p = parse_struct_key(r#"{"b":"x","a":12}"#).unwrap();
        assert_eq!(canonical_join(&p), "a=12\u{1}b=x", "field-name order, not the text's");
        let p = parse_struct_key(r#"{"flag": true, "n": null}"#).unwrap();
        assert_eq!(canonical_join(&p), "flag=true\u{1}n=");
        assert!(parse_struct_key("not json").is_none());
        assert!(parse_struct_key("{}").is_none());
        assert!(parse_struct_key(r#"{"id": [1]}"#).is_none(), "a nested value is not a key scalar");
        assert!(parse_struct_key(r#"{"$oid": "abc"}"#).is_some());
    }

    #[test]
    fn key_shape_struct_indexes_by_join_and_skips_unparseable_texts() {
        let shape = KeyShape::from_keys(
            true,
            vec![r#"{"id": 1}"#.to_string(), "garbage".to_string(), r#"{"id": 2}"#.to_string()],
        );
        match shape {
            KeyShape::Struct { fields, by_join } => {
                assert_eq!(fields, vec!["id".to_string()]);
                assert_eq!(by_join.len(), 2);
                assert_eq!(by_join["id=2"], r#"{"id": 2}"#);
            }
            other => panic!("expected a struct shape, got {other:?}"),
        }
        assert!(!KeyShape::from_keys(false, vec!["a".into(), "b".into()]).is_empty());
        assert!(KeyShape::from_keys(true, vec!["garbage".into()]).is_empty());
    }

    // ── the scan, on an in-memory catalog ────────────────────────────────────

    fn versions() -> VersionColumns {
        VersionColumns {
            source_instant: "meta.source_at".into(),
            capture_instant: "meta.captured_at".into(),
            sequence: Some("meta.seq".into()),
            offset: Some("meta.offset".into()),
            integer_instant_unit: TimeUnit::Millisecond,
        }
    }

    fn test_schema(key_type: Type) -> Schema {
        // id(1) hash(2) meta(3){op(4) captured_at(5) offset(6) key(7..) seq(10) source_at(11)} part(12)
        let meta = StructType::new(vec![
            NestedField::optional(4, "op", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(5, "captured_at", Type::Primitive(PrimitiveType::Timestamptz)).into(),
            NestedField::optional(6, "offset", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(7, "key", key_type).into(),
            NestedField::optional(10, "seq", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(11, "source_at", Type::Primitive(PrimitiveType::Timestamptz)).into(),
        ]);
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "hash", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "meta", Type::Struct(meta)).into(),
                NestedField::optional(12, "part", Type::Primitive(PrimitiveType::Boolean)).into(),
            ])
            .build()
            .unwrap()
    }

    async fn setup(key_type: Type) -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
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
        let ident = TableIdent::new(ns.clone(), "t".to_string());
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(test_schema(key_type))
            .format_version(FormatVersion::V2)
            .partition_spec(
                UnboundPartitionSpec::builder()
                    .add_partition_field(12, "part", Transform::Identity)
                    .unwrap()
                    .build(),
            )
            .build();
        let table = catalog.create_table(&ns, creation).await.unwrap();
        (warehouse, catalog, ident, table)
    }

    struct Row<'a> {
        id: i64,
        hash: Option<&'a str>,
        op: &'a str,
        captured_us: i64,
        offset: i64,
        seq: Option<i64>,
        source_us: Option<i64>,
    }

    fn string_array(vals: Vec<Option<&str>>, dt: &DataType) -> ArrayRef {
        match dt {
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vals)),
            _ => Arc::new(StringArray::from(vals)),
        }
    }

    async fn write_rows(
        catalog: &dyn Catalog,
        table: &Table,
        rows: &[Row<'_>],
        key_arr: ArrayRef,
        part: Option<bool>,
    ) -> Table {
        let schema = table.metadata().current_schema().clone();
        let arrow = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let n = rows.len();
        let DataType::Struct(meta_fields) = arrow.field_with_name("meta").unwrap().data_type().clone()
        else {
            panic!("meta must be a struct")
        };
        let field_type = |name: &str| -> DataType {
            meta_fields.iter().find(|f| f.name() == name).unwrap().data_type().clone()
        };
        let ts_array = |vals: Vec<Option<i64>>, name: &str| -> ArrayRef {
            let a = TimestampMicrosecondArray::from(vals);
            match field_type(name) {
                DataType::Timestamp(_, Some(tz)) => Arc::new(a.with_timezone(tz)),
                _ => Arc::new(a),
            }
        };
        let meta_cols: Vec<ArrayRef> = meta_fields
            .iter()
            .map(|f| -> ArrayRef {
                match f.name().as_str() {
                    "op" => string_array(rows.iter().map(|r| Some(r.op)).collect(), f.data_type()),
                    "captured_at" => ts_array(rows.iter().map(|r| Some(r.captured_us)).collect(), "captured_at"),
                    "offset" => Arc::new(Int64Array::from(rows.iter().map(|r| r.offset).collect::<Vec<_>>())),
                    "key" => key_arr.clone(),
                    "seq" => Arc::new(Int64Array::from(rows.iter().map(|r| r.seq).collect::<Vec<_>>())),
                    "source_at" => ts_array(rows.iter().map(|r| r.source_us).collect(), "source_at"),
                    other => panic!("unexpected meta field {other}"),
                }
            })
            .collect();
        let meta = StructArray::try_new(meta_fields.clone(), meta_cols, None).unwrap();
        let hash_arr = string_array(
            rows.iter().map(|r| r.hash).collect(),
            arrow.field_with_name("hash").unwrap().data_type(),
        );
        let batch = RecordBatch::try_new(
            arrow.clone(),
            vec![
                Arc::new(Int64Array::from(rows.iter().map(|r| r.id).collect::<Vec<_>>())) as ArrayRef,
                hash_arr,
                Arc::new(meta) as ArrayRef,
                Arc::new(BooleanArray::from(vec![part; n])) as ArrayRef,
            ],
        )
        .unwrap();
        let spec = table.metadata().default_partition_spec().clone();
        let partition_key = PartitionKey::new(
            spec.as_ref().clone(),
            schema.clone(),
            Struct::from_iter(vec![part.map(Literal::bool)]),
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(crate::atomic_replace::writer_properties(table), schema),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new("lv".to_string(), None, DataFileFormat::Parquet),
        );
        let mut writer = DataFileWriterBuilder::new(rolling)
            .build(Some(partition_key))
            .await
            .unwrap();
        writer.write(batch).await.unwrap();
        let files = writer.close().await.unwrap();
        let tx = Transaction::new(table);
        tx.fast_append()
            .add_data_files(files)
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    fn key_field_type(table: &Table) -> DataType {
        let arrow = schema_to_arrow_schema(table.metadata().current_schema()).unwrap();
        let DataType::Struct(meta_fields) = arrow.field_with_name("meta").unwrap().data_type().clone()
        else {
            panic!()
        };
        meta_fields.iter().find(|f| f.name() == "key").unwrap().data_type().clone()
    }

    fn struct_key(ids: Vec<Option<i64>>, key_fields: &Fields) -> ArrayRef {
        let id_arr: ArrayRef = Arc::new(Int64Array::from(ids));
        Arc::new(StructArray::try_new(key_fields.clone(), vec![id_arr], None).unwrap())
    }

    #[tokio::test]
    async fn struct_keys_rank_the_later_source_instant_across_partitions() {
        let key_type = Type::Struct(StructType::new(vec![
            NestedField::optional(8, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ]));
        let (_wh, catalog, ident, table) = setup(key_type).await;
        let DataType::Struct(key_fields) = key_field_type(&table) else {
            panic!("key must be a struct")
        };
        // the baseline partition: keys 1, 2, 3
        let table = write_rows(
            catalog.as_ref(),
            &table,
            &[
                Row { id: 1, hash: Some("h1:aaa"), op: "R", captured_us: 100, offset: 10, seq: None, source_us: Some(100) },
                Row { id: 2, hash: Some("h1:bbb"), op: "R", captured_us: 100, offset: 11, seq: None, source_us: Some(50) },
                Row { id: 3, hash: None, op: "R", captured_us: 100, offset: 12, seq: None, source_us: None },
            ],
            struct_key(vec![Some(1), Some(2), Some(3)], &key_fields),
            Some(true),
        )
        .await;
        // the stream partition (null): key 1 later by the source instant, key 3 later by capture
        // only, and an OLDER version of key 2 that must not win over its newer baseline
        let table = write_rows(
            catalog.as_ref(),
            &table,
            &[
                Row { id: 1, hash: Some("h1:ccc"), op: "u", captured_us: 150, offset: 500, seq: Some(7), source_us: Some(200) },
                Row { id: 3, hash: Some("h1:ddd"), op: "u", captured_us: 300, offset: 501, seq: Some(8), source_us: None },
                Row { id: 2, hash: Some("h1:old"), op: "c", captured_us: 1, offset: 502, seq: Some(9), source_us: Some(40) },
            ],
            struct_key(vec![Some(1), Some(3), Some(2)], &key_fields),
            None,
        )
        .await;
        let out = latest_versions_by_key(
            catalog.as_ref(),
            &ident,
            "meta.key",
            "hash",
            &versions(),
            vec![
                r#"{"id": 1}"#.into(),
                r#"{"id":2}"#.into(),
                r#"{"id": 3}"#.into(),
                r#"{"id": 9}"#.into(),
                "junk".into(),
            ],
        )
        .await
        .unwrap();
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(
            out[r#"{"id": 1}"#],
            LatestVersion { hash: Some("h1:ccc".into()), source_instant_us: Some(200) }
        );
        assert_eq!(
            out[r#"{"id":2}"#],
            LatestVersion { hash: Some("h1:bbb".into()), source_instant_us: Some(50) }
        );
        assert_eq!(
            out[r#"{"id": 3}"#],
            LatestVersion { hash: Some("h1:ddd".into()), source_instant_us: None }
        );
        assert!(!out.contains_key(r#"{"id": 9}"#), "a key with no version is absent");
        let _ = table;
    }

    #[tokio::test]
    async fn string_keys_match_the_text_exactly() {
        let (_wh, catalog, ident, table) = setup(Type::Primitive(PrimitiveType::String)).await;
        let key_dt = key_field_type(&table);
        let table = write_rows(
            catalog.as_ref(),
            &table,
            &[
                Row { id: 1, hash: Some("h1:r1"), op: "R", captured_us: 10, offset: 1, seq: None, source_us: Some(10) },
                Row { id: 2, hash: None, op: "R", captured_us: 10, offset: 2, seq: None, source_us: Some(10) },
            ],
            string_array(vec![Some("{$oid=aaa}"), Some("{$oid=bbb}")], &key_dt),
            Some(true),
        )
        .await;
        let _table = write_rows(
            catalog.as_ref(),
            &table,
            &[Row { id: 1, hash: Some("h1:u1"), op: "u", captured_us: 20, offset: 9, seq: Some(1), source_us: Some(20) }],
            string_array(vec![Some("{$oid=aaa}")], &key_dt),
            None,
        )
        .await;
        let out = latest_versions_by_key(
            catalog.as_ref(),
            &ident,
            "meta.key",
            "hash",
            &versions(),
            vec!["{$oid=aaa}".into(), "{$oid=bbb}".into(), "{$oid=zzz}".into()],
        )
        .await
        .unwrap();
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(
            out["{$oid=aaa}"],
            LatestVersion { hash: Some("h1:u1".into()), source_instant_us: Some(20) }
        );
        assert_eq!(out["{$oid=bbb}"], LatestVersion { hash: None, source_instant_us: Some(10) });
        // an empty key list reads nothing
        assert!(
            latest_versions_by_key(catalog.as_ref(), &ident, "meta.key", "hash", &versions(), vec![])
                .await
                .unwrap()
                .is_empty()
        );
    }
}
