//! Hot-column derivation at ENCODE time (§14.41 f, 2026-10-17). The typed shadows of document paths that the
//! Stream Load `columns:` header derived on the BE — `doris_scd4_native.mongo_hot_sql(text=True)`: 16
//! `json_extract_string(CAST(after AS STRING), '$.…')` expressions = 16 VARIANT→text casts + 16 JSON parses of a
//! 4.8-KB document PER ROW, measured 37× a plain load (M1, 2026-10-17 03:59Z: plain 0.17 ms/row, hot 6.2–6.5) —
//! are derived HERE from the document parsed once per row, and ride the JSON line as plain loaded fields. The
//! value rules are the header's, stated per kind below; a value the header would turn into NULL is NULL here
//! (never a strict-mode row failure). Lives in the `iceberg` crate because the wheel gate runs `cargo test -p iceberg`
//! (the pyo3 binding's own tests cannot link in the manylinux builder); the binding calls `append_fields` per row.
use serde_json::Value;

/// The lower bound of the DATETIME(6) range the header bounds every temporal derivation to (doris_scd4_v2.NS_MIN).
pub const NS_MIN: &str = "1677-09-21 00:12:43.145225";
/// The upper bound (doris_scd4_v2.NS_MAX).
pub const NS_MAX: &str = "2262-04-11 23:47:16.854775";

/// The header's four type arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// DATETIME / TIMESTAMP / DATE: the `$date` ladder, bounded to the DATETIME(6) range.
    Temporal,
    /// STRING / VARCHAR / CHAR / TEXT: `json_extract_string` verbatim.
    Text,
    /// BOOLEAN from true/false/1/0.
    Boolean,
    /// Every other type: the scalar's digits, or null when it is not a number.
    Numeric,
}

impl Kind {
    /// The header's type arms: HOT_TEMPORAL = DATETIME/TIMESTAMPTZ/TIMESTAMP/DATE, HOT_TEXT = STRING/VARCHAR/CHAR/TEXT,
    /// BOOLEAN, else a numeric CAST.
    pub fn of(doris_type: &str) -> Kind {
        let t = doris_type.trim().to_ascii_uppercase();
        if t.starts_with("DATETIME") || t.starts_with("TIMESTAMP") || t.starts_with("DATE") {
            Kind::Temporal
        } else if t.starts_with("STRING") || t.starts_with("VARCHAR") || t.starts_with("CHAR") || t.starts_with("TEXT") {
            Kind::Text
        } else if t == "BOOLEAN" || t == "BOOL" {
            Kind::Boolean
        } else {
            Kind::Numeric
        }
    }
}

/// One derived column: `col` on the line, from `src` (the document column of the projection), at `path`.
#[derive(Clone, Debug)]
pub struct Spec {
    /// The derived column's name on the JSON line (a pair column).
    pub col: String,
    /// The projected document column the value is read from.
    pub src: String,
    /// The dotted path, split.
    pub path: Vec<String>,
    /// The type arm.
    pub kind: Kind,
}

impl Spec {
    /// A spec from the Python tuple `(column, source_column, dotted_path, doris_type)`.
    pub fn new(col: &str, src: &str, dotted_path: &str, doris_type: &str) -> Spec {
        Spec {
            col: col.to_string(),
            src: src.to_string(),
            path: dotted_path.split('.').filter(|p| !p.is_empty()).map(str::to_string).collect(),
            kind: Kind::of(doris_type),
        }
    }
}

/// A derived value as it goes onto the JSON line: absent/unconvertible = null; a JSON string; a bare literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Derived {
    /// Absent, unconvertible or out of range: the header's NULL.
    Null,
    /// A JSON string (text, a datetime text, digits the BE casts).
    Str(String),
    /// A bare JSON literal (`true` / `false`).
    Raw(String),
}

/// The document element at a dotted path (objects only; a missing step = None).
pub fn walk<'a>(doc: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cur = doc;
    for p in path {
        cur = cur.as_object()?.get(p)?;
    }
    Some(cur)
}

/// `json_extract_string` over the ladder `$oid` → `$numberLong` → `$numberDecimal` → the value itself: a string
/// verbatim, a number/bool as its text, an object/array as its JSON text, null as None.
pub fn scalar_text(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(if *b { "true".into() } else { "false".into() }),
        Value::Number(n) => Some(n.to_string()),
        Value::Object(o) => {
            for k in ["$oid", "$numberLong", "$numberDecimal"] {
                if let Some(inner) = o.get(k) {
                    if let Some(t) = scalar_text(inner) {
                        return Some(t);
                    }
                }
            }
            serde_json::to_string(v).ok()
        }
        Value::Array(_) => serde_json::to_string(v).ok(),
    }
}

/// Epoch milliseconds → the header's `from_millisecond` text under a UTC session (the warehouse runs Etc/UTC,
/// read 2026-10-17 04:04Z: the stored create_date equals from_millisecond of the document's $date).
pub fn ms_text(ms: i64) -> Option<String> {
    let dt = chrono::DateTime::from_timestamp_millis(ms)?.naive_utc();
    Some(format!("{}.{:06}", dt.format("%Y-%m-%d %H:%M:%S"), dt.and_utc().timestamp_subsec_micros()))
}

/// A datetime text the header's `CAST(… AS DATETIME(6))` accepts: RFC 3339 (converted to UTC), or a naive
/// `YYYY-MM-DD[ T]HH:MM:SS[.frac]`, or a bare date. Anything else = None (the CAST's NULL).
pub fn text_to_datetime(s: &str) -> Option<String> {
    let s = s.trim();
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        let u = dt.naive_utc();
        return Some(format!("{}.{:06}", u.format("%Y-%m-%d %H:%M:%S"), u.and_utc().timestamp_subsec_micros()));
    }
    for f in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(n) = chrono::NaiveDateTime::parse_from_str(s, f) {
            return Some(format!("{}.{:06}", n.format("%Y-%m-%d %H:%M:%S"), n.and_utc().timestamp_subsec_micros()));
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(format!("{} 00:00:00.000000", d.format("%Y-%m-%d")));
    }
    None
}

fn in_ns_range(t: &str) -> bool {
    t >= NS_MIN && t <= NS_MAX
}

fn i64_of(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// The header's temporal ladder: `$date` as epoch ms → `$date.$numberLong` → `$date` as text → the value as
/// text; bounded to NS_MIN..NS_MAX.
pub fn temporal(v: &Value) -> Option<String> {
    let t = match v {
        Value::Object(o) => match o.get("$date") {
            Some(d @ Value::Number(_)) => i64_of(d).and_then(ms_text),
            Some(Value::Object(inner)) => inner.get("$numberLong").and_then(i64_of).and_then(ms_text),
            Some(Value::String(s)) => s.trim().parse::<i64>().ok().and_then(ms_text).or_else(|| text_to_datetime(s)),
            _ => None,
        },
        Value::String(s) => text_to_datetime(s),
        _ => None,
    }?;
    if in_ns_range(&t) {
        Some(t)
    } else {
        None
    }
}

/// The derived value of one spec over a parsed document.
pub fn derive(doc: &Value, spec: &Spec) -> Derived {
    let Some(v) = walk(doc, &spec.path) else { return Derived::Null };
    match spec.kind {
        Kind::Temporal => temporal(v).map(Derived::Str).unwrap_or(Derived::Null),
        Kind::Text => scalar_text(v).map(Derived::Str).unwrap_or(Derived::Null),
        Kind::Boolean => match scalar_text(v).map(|t| t.trim().to_ascii_lowercase()).as_deref() {
            Some("true") | Some("1") => Derived::Raw("true".into()),
            Some("false") | Some("0") => Derived::Raw("false".into()),
            _ => Derived::Null,
        },
        Kind::Numeric => match scalar_text(v) {
            Some(t) if t.trim().parse::<i64>().is_ok() || t.trim().parse::<f64>().map(|f| f.is_finite()).unwrap_or(false) => {
                Derived::Str(t.trim().to_string())
            }
            _ => Derived::Null,
        },
    }
}

/// Append `,"col":value` for every spec of one source document (already parsed) to a JSON-line buffer.
pub fn append_fields(out: &mut Vec<u8>, doc: Option<&Value>, specs: &[Spec], json_str: fn(&mut Vec<u8>, &str)) {
    for s in specs {
        out.push(b',');
        json_str(out, &s.col);
        out.push(b':');
        match doc.map(|d| derive(d, s)).unwrap_or(Derived::Null) {
            Derived::Null => out.extend_from_slice(b"null"),
            Derived::Str(t) => json_str(out, &t),
            Derived::Raw(r) => out.extend_from_slice(r.as_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(col: &str, path: &str, t: &str) -> Spec {
        Spec::new(col, "after", path, t)
    }

    #[test]
    fn kinds_follow_the_header_arms() {
        assert_eq!(Kind::of("DATETIME(6)"), Kind::Temporal);
        assert_eq!(Kind::of("timestamp"), Kind::Temporal);
        assert_eq!(Kind::of("STRING"), Kind::Text);
        assert_eq!(Kind::of("VARCHAR(64)"), Kind::Text);
        assert_eq!(Kind::of("BOOLEAN"), Kind::Boolean);
        assert_eq!(Kind::of("BIGINT"), Kind::Numeric);
        assert_eq!(Kind::of("DECIMAL(18,4)"), Kind::Numeric);
    }

    #[test]
    fn temporal_ladder() {
        let d = json!({"create_date": {"$date": 1771760391000i64}, "s": {"$date": "2026-02-22T11:39:51Z"},
                       "l": {"$date": {"$numberLong": "1771760391123"}}, "bare": "2026-02-22 11:39:51",
                       "far": {"$date": 32503680000000i64}, "junk": "not a date", "n": 5});
        let s = spec("c", "create_date", "DATETIME(6)");
        assert_eq!(derive(&d, &s), Derived::Str("2026-02-22 11:39:51.000000".into()));
        assert_eq!(derive(&d, &spec("c", "s", "DATETIME(6)")), Derived::Str("2026-02-22 11:39:51.000000".into()));
        assert_eq!(derive(&d, &spec("c", "l", "DATETIME(6)")), Derived::Str("2026-02-22 11:39:51.123000".into()));
        assert_eq!(derive(&d, &spec("c", "bare", "DATETIME(6)")), Derived::Str("2026-02-22 11:39:51.000000".into()));
        assert_eq!(derive(&d, &spec("c", "far", "DATETIME(6)")), Derived::Null, "outside NS_MAX → null");
        assert_eq!(derive(&d, &spec("c", "junk", "DATETIME(6)")), Derived::Null, "unparseable → null");
        assert_eq!(derive(&d, &spec("c", "n", "DATETIME(6)")), Derived::Null, "a bare number is not a date text");
        assert_eq!(derive(&d, &spec("c", "missing", "DATETIME(6)")), Derived::Null);
    }

    #[test]
    fn text_boolean_numeric_ladders() {
        let d = json!({"id": {"$oid": "62e33519f84b010b49f74315"}, "n": {"$numberLong": "9007199254740993"},
                       "dec": {"$numberDecimal": "123.4500"}, "obj": {"a": 1}, "arr": [1, 2], "t": true, "f": 0,
                       "price": 12.5, "txt": "x", "bad": "abc", "nul": null, "nested": {"k": {"v": "deep"}}});
        assert_eq!(derive(&d, &spec("id", "id", "STRING")), Derived::Str("62e33519f84b010b49f74315".into()));
        assert_eq!(derive(&d, &spec("n", "n", "BIGINT")), Derived::Str("9007199254740993".into()), "$numberLong digits exact");
        assert_eq!(derive(&d, &spec("dec", "dec", "DECIMAL(18,4)")), Derived::Str("123.4500".into()));
        assert_eq!(derive(&d, &spec("o", "obj", "STRING")), Derived::Str("{\"a\":1}".into()), "a non-scalar under STRING = its JSON text");
        assert_eq!(derive(&d, &spec("a", "arr", "STRING")), Derived::Str("[1,2]".into()));
        assert_eq!(derive(&d, &spec("t", "t", "BOOLEAN")), Derived::Raw("true".into()));
        assert_eq!(derive(&d, &spec("f", "f", "BOOLEAN")), Derived::Raw("false".into()));
        assert_eq!(derive(&d, &spec("x", "txt", "BOOLEAN")), Derived::Null);
        assert_eq!(derive(&d, &spec("p", "price", "DOUBLE")), Derived::Str("12.5".into()));
        assert_eq!(derive(&d, &spec("b", "bad", "DOUBLE")), Derived::Null, "a non-number under a numeric type = null");
        assert_eq!(derive(&d, &spec("z", "nul", "STRING")), Derived::Null);
        assert_eq!(derive(&d, &spec("v", "nested.k.v", "STRING")), Derived::Str("deep".into()));
        assert_eq!(derive(&d, &spec("m", "nested.k.missing", "STRING")), Derived::Null);
    }

    #[test]
    fn append_fields_renders_a_json_fragment() {
        fn js(out: &mut Vec<u8>, s: &str) {
            out.push(b'"');
            out.extend_from_slice(s.as_bytes());
            out.push(b'"');
        }
        let d = json!({"id": {"$oid": "abc"}, "t": true});
        let mut out = Vec::new();
        append_fields(&mut out, Some(&d), &[spec("id", "id", "STRING"), spec("t", "t", "BOOLEAN"), spec("m", "m", "STRING")], js);
        assert_eq!(String::from_utf8(out).unwrap(), ",\"id\":\"abc\",\"t\":true,\"m\":null");
        let mut out2 = Vec::new();
        append_fields(&mut out2, None, &[spec("id", "id", "STRING")], js);
        assert_eq!(String::from_utf8(out2).unwrap(), ",\"id\":null", "no document = every derived field null");
    }
}
