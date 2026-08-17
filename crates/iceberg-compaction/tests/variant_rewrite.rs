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

//! VARIANT rewrite spike: the compaction engine must round-trip a canonical
//! (unshredded) VARIANT column byte-for-byte — read the `{metadata, value}`
//! pair, binpack, and write it back unchanged. This is the gate for running
//! the maintenance tiers' rewrite (Tier 2/3) on mongo silver tables, whose
//! documents live in canonical VARIANT columns.
//!
//! Covers:
//! - canonical round-trip: two undersized data files with mixed variant
//!   payloads (object, string, long) plus a NULL variant slot → one compacted
//!   file; every surviving row's variant bytes equal the seeded bytes, and
//!   the null stays null.
//! - SHREDDED input: a data file whose variant column is (partially) shredded
//!   (`{metadata, value, typed_value}` — the layout engines like DuckDB and
//!   Spark produce) scans back correctly through the unshred fold, and a
//!   mixed shredded+canonical table compacts into one file whose values are
//!   semantically intact. Today's rewrite emits CANONICAL output (un-shreds);
//!   that behavior is pinned here so a future shred-preserving writer changes
//!   it deliberately.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, Int32Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use bytes::Bytes;
use futures::TryStreamExt;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
    ManifestList, NestedField, Operation, PrimitiveType, Schema, Struct as IcebergStruct, Type,
    VariantType,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{
    Catalog, CatalogBuilder, MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder, NamespaceIdent,
    TableCreation, TableIdent,
};
use iceberg_compaction::config::Config;
use iceberg_compaction::engine::compact_table;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::variant::{
    ShreddedSchemaBuilder, Variant, VariantArrayBuilder, VariantBuilder, shred_variant,
    variant_to_json,
};
use tempfile::TempDir;

fn aggressive_cfg() -> Config {
    Config {
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    }
}

/// (metadata, value) canonical variant buffers.
type VariantBytes = (Vec<u8>, Vec<u8>);

fn variant_object() -> VariantBytes {
    let mut b = VariantBuilder::new();
    let mut obj = b.new_object();
    obj.insert("a", 1i64);
    obj.insert("b", "x");
    obj.finish();
    b.finish()
}

fn variant_string(s: &str) -> VariantBytes {
    let mut b = VariantBuilder::new();
    b.append_value(s);
    b.finish()
}

fn variant_long(v: i64) -> VariantBytes {
    let mut b = VariantBuilder::new();
    b.append_value(v);
    b.finish()
}

/// Arrow schema matching the iceberg→arrow mapping for
/// [id: int (fid 1), doc: variant (fid 2, optional)]: the variant is a struct
/// of two required Binary fields with the field id on the OUTER field only.
fn arrow_schema() -> Arc<ArrowSchema> {
    let variant_fields = Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ]);
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("doc", DataType::Struct(variant_fields), true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]))
}

/// Build a batch of (id, variant) rows; `None` = NULL variant slot.
fn batch(rows: Vec<(i32, Option<VariantBytes>)>) -> RecordBatch {
    let ids = Int32Array::from(rows.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    let metas = BinaryArray::from_iter_values(
        rows.iter()
            .map(|(_, v)| v.as_ref().map(|(m, _)| m.clone()).unwrap_or_default()),
    );
    let vals = BinaryArray::from_iter_values(
        rows.iter()
            .map(|(_, v)| v.as_ref().map(|(_, x)| x.clone()).unwrap_or_default()),
    );
    let validity = NullBuffer::from(rows.iter().map(|(_, v)| v.is_some()).collect::<Vec<_>>());
    let variant_fields = Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ]);
    let doc = StructArray::new(
        variant_fields,
        vec![Arc::new(metas) as ArrayRef, Arc::new(vals) as ArrayRef],
        Some(validity),
    );
    RecordBatch::try_new(arrow_schema(), vec![Arc::new(ids), Arc::new(doc)]).unwrap()
}

async fn write_data_file(table: &Table, prefix: &str, batch: RecordBatch) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

/// All live rows as (id, Option<(metadata, value)>), scan-read, sorted by id.
async fn live_variant_rows(table: &Table) -> Vec<(i32, Option<VariantBytes>)> {
    let mut stream = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let mut rows = Vec::new();
    while let Some(b) = stream.try_next().await.unwrap() {
        let ids = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let docs = b
            .column_by_name("doc")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let metas = docs
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let vals = docs
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            let doc = if docs.is_null(i) {
                None
            } else {
                Some((metas.value(i).to_vec(), vals.value(i).to_vec()))
            };
            rows.push((ids.value(i), doc));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows
}

async fn data_file_count(table: &Table) -> usize {
    let snap = table.metadata().current_snapshot().unwrap();
    let bytes = table
        .file_io()
        .new_input(snap.manifest_list())
        .unwrap()
        .read()
        .await
        .unwrap();
    let ml = ManifestList::parse_with_version(&bytes, table.metadata().format_version()).unwrap();
    let mut n = 0;
    for mf in ml.entries() {
        if mf.content != ManifestContentType::Data {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        n += m.entries().iter().filter(|e| e.is_alive()).count();
    }
    n
}

/// Fresh V3 table [id: int, doc: variant] on a memory catalog.
async fn setup_table(warehouse: &TempDir) -> (impl Catalog, TableIdent, Table) {
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap();
    let ns = NamespaceIdent::new("db".to_string());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "doc", Type::Variant(VariantType)).into(),
        ])
        .build()
        .unwrap();
    let ident = TableIdent::new(ns.clone(), "t".to_string());
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("t".to_string())
                .schema(schema)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    (catalog, ident, table)
}

/// The spike: binpack-compact a canonical-VARIANT table; every variant byte
/// survives unchanged and the NULL slot stays NULL.
#[tokio::test]
async fn compaction_round_trips_canonical_variant() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = setup_table(&warehouse).await;

    // Two undersized files with mixed payloads + one NULL variant slot.
    let seeded: Vec<(i32, Option<VariantBytes>)> = vec![
        (1, Some(variant_object())),
        (2, Some(variant_string("hello mongo"))),
        (3, None),
        (4, Some(variant_long(42))),
        (5, Some(variant_string("second file"))),
        (6, Some(variant_object())),
    ];
    let file1 = write_data_file(&table, "v1", batch(seeded[..3].to_vec())).await;
    let file2 = write_data_file(&table, "v2", batch(seeded[3..].to_vec())).await;
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(file1.into_iter().chain(file2).collect::<Vec<_>>())
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // Pre-compaction sanity: scan already round-trips the seeded bytes.
    assert_eq!(live_variant_rows(&table).await, seeded);
    assert_eq!(data_file_count(&table).await, 2);

    compact_table(&catalog, &ident, &aggressive_cfg())
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        data_file_count(&table).await,
        1,
        "two undersized files binpack into one"
    );
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Replace,
        "compaction commits a Replace snapshot"
    );
    // THE assertion: canonical variant bytes are preserved verbatim, and the
    // NULL slot is still NULL.
    assert_eq!(
        live_variant_rows(&table).await,
        seeded,
        "variant column must round-trip byte-for-byte through the rewrite"
    );
}

// ---------------------------------------------------------------------------
// Shredded-input coverage
// ---------------------------------------------------------------------------

/// Build a (partially) SHREDDED variant column from canonical byte pairs:
/// shred on `a: Int64`, so object rows with an integer `a` get a typed_value
/// while other payload shapes (string, long, objects without `a`) stay in the
/// binary `value` — the mixed reality a schemaless-document writer produces.
fn shredded_doc_array(rows: &[(i32, Option<VariantBytes>)]) -> ArrayRef {
    let mut b = VariantArrayBuilder::new(rows.len());
    for (_, v) in rows {
        match v {
            None => b.append_null(),
            Some((m, val)) => b.append_variant(Variant::try_new(m, val).unwrap()),
        }
    }
    let canonical = b.build();
    let as_type = ShreddedSchemaBuilder::new()
        .with_path("a", DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&canonical, &as_type).unwrap();
    assert!(
        matches!(ArrayRef::from(shredded.clone()).data_type(),
                 DataType::Struct(f) if f.iter().any(|x| x.name() == "typed_value")),
        "seed must actually be shredded"
    );
    ArrayRef::from(shredded)
}

/// Batch of (id, shredded doc) with iceberg field ids on the OUTER fields —
/// the shape a shredding engine writes into a data file.
fn shredded_batch(rows: &[(i32, Option<VariantBytes>)]) -> RecordBatch {
    let ids = Int32Array::from(rows.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    let doc = shredded_doc_array(rows);
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("doc", doc.data_type().clone(), true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(ids), doc]).unwrap()
}

/// Write a batch as a raw parquet data file (bypassing the iceberg writer
/// chain, which only emits the canonical variant layout) and hand-build its
/// DataFile — how an external engine's shredded file enters the table.
async fn write_raw_data_file(table: &Table, name: &str, batch: RecordBatch) -> DataFile {
    let mut buf = Vec::new();
    let mut w = ArrowWriter::try_new(&mut buf, batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let path = format!("{}/data/{name}.parquet", table.metadata().location());
    table
        .file_io()
        .new_output(&path)
        .unwrap()
        .write(Bytes::from(buf.clone()))
        .await
        .unwrap();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(buf.len() as u64)
        .record_count(batch.num_rows() as u64)
        .partition_spec_id(table.metadata().default_partition_spec_id())
        .partition(IcebergStruct::empty())
        .build()
        .unwrap()
}

/// JSON view of canonical byte pairs (order-preserving with the ids).
fn jsons_of(rows: &[(i32, Option<VariantBytes>)]) -> Vec<(i32, Option<String>)> {
    let arr = {
        let metas = BinaryArray::from_iter_values(
            rows.iter()
                .map(|(_, v)| v.as_ref().map(|(m, _)| m.clone()).unwrap_or_default()),
        );
        let vals = BinaryArray::from_iter_values(
            rows.iter()
                .map(|(_, v)| v.as_ref().map(|(_, x)| x.clone()).unwrap_or_default()),
        );
        let validity = NullBuffer::from(rows.iter().map(|(_, v)| v.is_some()).collect::<Vec<_>>());
        let fields = Fields::from(vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::Binary, true),
        ]);
        Arc::new(StructArray::new(
            fields,
            vec![Arc::new(metas) as ArrayRef, Arc::new(vals) as ArrayRef],
            Some(validity),
        )) as ArrayRef
    };
    let json = variant_to_json(&arr).unwrap();
    rows.iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, (!json.is_null(i)).then(|| json.value(i).to_string())))
        .collect()
}

/// Scan the table and render every doc as JSON (via the same kernel), sorted
/// by id — semantic comparison for rows that crossed a shred/unshred boundary
/// (their re-canonicalized bytes need not be identical to the seed bytes).
async fn live_docs_as_json(table: &Table) -> Vec<(i32, Option<String>)> {
    jsons_of(&live_variant_rows(table).await)
}

/// Live data file paths from the current snapshot's manifests.
async fn live_data_file_paths(table: &Table) -> Vec<String> {
    let snap = table.metadata().current_snapshot().unwrap();
    let bytes = table
        .file_io()
        .new_input(snap.manifest_list())
        .unwrap()
        .read()
        .await
        .unwrap();
    let ml = ManifestList::parse_with_version(&bytes, table.metadata().format_version()).unwrap();
    let mut paths = Vec::new();
    for mf in ml.entries() {
        if mf.content != ManifestContentType::Data {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                paths.push(e.data_file().file_path().to_string());
            }
        }
    }
    paths
}

/// The variant struct children of column `doc` in a parquet file's arrow schema.
async fn doc_children_of(table: &Table, path: &str) -> Vec<String> {
    let bytes = table
        .file_io()
        .new_input(path)
        .unwrap()
        .read()
        .await
        .unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
    // The rewrite must emit COMPRESSED output (iceberg default zstd) — the
    // parquet-rs WriterProperties default is UNCOMPRESSED, which inflated
    // real rewrites ~20x before the codec property was translated.
    assert_eq!(
        reader.metadata().row_group(0).column(0).compression(),
        parquet::basic::Compression::ZSTD(Default::default()),
        "rewrite output must honor the table's compression codec"
    );
    // Rewritten variant columns — canonical AND shredded — must carry the
    // parquet VARIANT logical-type annotation, or readers see a raw struct.
    let parquet_doc = reader
        .metadata()
        .file_metadata()
        .schema()
        .get_fields()
        .iter()
        .find(|f| f.name() == "doc")
        .expect("doc column in parquet schema");
    assert!(
        matches!(
            parquet_doc.get_basic_info().logical_type_ref(),
            Some(parquet::basic::LogicalType::Variant { .. })
        ),
        "rewrite output must annotate variant columns"
    );
    let field = reader
        .schema()
        .field_with_name("doc")
        .expect("doc column in output file");
    match field.data_type() {
        DataType::Struct(children) => children.iter().map(|f| f.name().to_string()).collect(),
        other => panic!("doc is not a struct: {other:?}"),
    }
}

/// SHREDDED input: a shredded data file scans back correctly (the unshred
/// fold), and a mixed shredded+canonical table compacts into ONE file with
/// all values semantically intact. Pins today's rewrite output as CANONICAL
/// (un-shredded) — a shred-preserving writer must change this test on purpose.
#[tokio::test]
async fn compaction_reads_shredded_input_and_writes_canonical() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = setup_table(&warehouse).await;

    // File 1 (canonical, via the iceberg writer chain): mixed payloads.
    let canonical_rows: Vec<(i32, Option<VariantBytes>)> = vec![
        (1, Some(variant_object())),
        (2, Some(variant_string("canonical file"))),
        (3, None),
    ];
    let file1 = write_data_file(&table, "c1", batch(canonical_rows.clone())).await;

    // File 2 (SHREDDED, raw parquet): objects with a shreddable `a`, an
    // unshreddable string, and a NULL slot.
    let shredded_rows: Vec<(i32, Option<VariantBytes>)> = vec![
        (4, Some(variant_object())),
        (5, Some(variant_string("stays in value"))),
        (6, Some(variant_long(9000))),
        (7, None),
    ];
    let file2 = write_raw_data_file(&table, "s1", shredded_batch(&shredded_rows)).await;

    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(file1.into_iter().chain([file2]).collect::<Vec<_>>())
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let mut seeded = canonical_rows.clone();
    seeded.extend(shredded_rows.clone());
    let expected_json = jsons_of(&seeded);

    // Pre-compaction: the SCAN already unshreds — every row (incl. the ones
    // living in typed_value) reads back semantically equal.
    assert_eq!(
        live_docs_as_json(&table).await,
        expected_json,
        "shredded file must scan back through the unshred fold"
    );
    assert_eq!(data_file_count(&table).await, 2);

    compact_table(&catalog, &ident, &aggressive_cfg())
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        data_file_count(&table).await,
        1,
        "mixed input binpacks to one"
    );
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Replace,
    );
    assert_eq!(
        live_docs_as_json(&table).await,
        expected_json,
        "values must survive the shred -> unshred -> rewrite trip"
    );

    // Pin today's behavior: the rewrite emits the CANONICAL layout (no
    // typed_value) — i.e. compaction currently UN-shreds. A future
    // shred-preserving writer flips this assertion deliberately.
    let paths = live_data_file_paths(&table).await;
    assert_eq!(paths.len(), 1);
    assert_eq!(
        doc_children_of(&table, &paths[0]).await,
        vec!["metadata".to_string(), "value".to_string()],
        "current rewrite output is canonical (un-shredded)"
    );
}

/// The full doc field (struct children + the typed_value subtree, if any) of
/// a parquet file's arrow schema.
async fn doc_type_of(table: &Table, path: &str) -> DataType {
    let bytes = table
        .file_io()
        .new_input(path)
        .unwrap()
        .read()
        .await
        .unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
    reader
        .schema()
        .field_with_name("doc")
        .expect("doc column in output file")
        .data_type()
        .clone()
}

/// SHRED-WRITE: with `Config::shred_variants`, the rewrite re-shreds its
/// output to the input files' layout — the compacted file carries the
/// `typed_value` subtree (so engines keep pruning into it) and every value
/// still reads back semantically intact through the scan.
#[tokio::test]
async fn compaction_shred_write_preserves_input_shredding() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = setup_table(&warehouse).await;

    // Two SHREDDED files (shredded on `a: Int64`), with payloads that both
    // do and don't match the shredding schema, plus NULL slots.
    let rows1: Vec<(i32, Option<VariantBytes>)> = vec![
        (1, Some(variant_object())),
        (2, Some(variant_string("resides in value"))),
        (3, None),
    ];
    let rows2: Vec<(i32, Option<VariantBytes>)> =
        vec![(4, Some(variant_object())), (5, Some(variant_long(77)))];
    let file1 = write_raw_data_file(&table, "s1", shredded_batch(&rows1)).await;
    let file2 = write_raw_data_file(&table, "s2", shredded_batch(&rows2)).await;
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let mut seeded = rows1.clone();
    seeded.extend(rows2.clone());
    let expected_json = jsons_of(&seeded);
    assert_eq!(live_docs_as_json(&table).await, expected_json);
    assert_eq!(data_file_count(&table).await, 2);

    let cfg = Config {
        shred_variants: true,
        ..aggressive_cfg()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(data_file_count(&table).await, 1);

    // The output file is SHREDDED: {metadata, value, typed_value} with the
    // input's `a` field typed inside typed_value.
    let paths = live_data_file_paths(&table).await;
    assert_eq!(paths.len(), 1);
    let doc_type = doc_type_of(&table, &paths[0]).await;
    let DataType::Struct(children) = &doc_type else {
        panic!("doc is not a struct: {doc_type:?}");
    };
    let names: Vec<_> = children.iter().map(|f| f.name().to_string()).collect();
    assert!(
        names.contains(&"typed_value".to_string()),
        "shred-write output must carry typed_value, got {names:?}"
    );
    let tv = children.iter().find(|f| f.name() == "typed_value").unwrap();
    let DataType::Struct(tv_children) = tv.data_type() else {
        panic!("typed_value is not a struct: {:?}", tv.data_type());
    };
    assert_eq!(
        tv_children
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        vec!["a"],
        "the input's shredded field must survive the rewrite"
    );

    // And the shredded output still scans back semantically equal (the fold
    // reads our own shred-write output correctly).
    assert_eq!(
        live_docs_as_json(&table).await,
        expected_json,
        "values must survive the shred-preserving rewrite"
    );
}

/// Canonical input + `shred_variants: true` stays CANONICAL — the flag
/// preserves the input's layout, it does not invent shredding.
#[tokio::test]
async fn shred_write_flag_is_a_noop_on_canonical_input() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = setup_table(&warehouse).await;

    let rows: Vec<(i32, Option<VariantBytes>)> = vec![
        (1, Some(variant_object())),
        (2, Some(variant_string("plain"))),
        (3, None),
    ];
    let file1 = write_data_file(&table, "c1", batch(rows.clone())).await;
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(file1)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let cfg = Config {
        shred_variants: true,
        ..aggressive_cfg()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let paths = live_data_file_paths(&table).await;
    assert_eq!(paths.len(), 1);
    assert_eq!(
        doc_children_of(&table, &paths[0]).await,
        vec!["metadata".to_string(), "value".to_string()],
        "canonical input stays canonical under the shred-write flag"
    );
    assert_eq!(live_docs_as_json(&table).await, jsons_of(&rows));
}

/// ARRAY-shredded input: with `Config::shred_variants`, an input file whose
/// `typed_value` carries an array node (`tags: List<{value?, typed_value}>`)
/// keeps its array shredding through the rewrite — the derivation covers
/// arrays, not just objects and primitives.
#[tokio::test]
async fn compaction_shred_write_preserves_array_shredding() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident, table) = setup_table(&warehouse).await;

    let doc_with_tags = |a: i64, tags: &[i64]| -> VariantBytes {
        let mut b = VariantBuilder::new();
        let mut obj = b.new_object();
        obj.insert("a", a);
        let mut list = obj.new_list("tags");
        for t in tags {
            list.append_value(*t);
        }
        list.finish();
        obj.finish();
        b.finish()
    };
    let rows: Vec<(i32, Option<VariantBytes>)> = vec![
        (1, Some(doc_with_tags(1, &[10, 11]))),
        (2, Some(doc_with_tags(2, &[]))),
        (3, None),
    ];

    // Shred on {a: Int64, tags: List<Int64>} and land it as a raw file — the
    // layout an array-shredding engine writes.
    let mut b = VariantArrayBuilder::new(rows.len());
    for (_, v) in &rows {
        match v {
            None => b.append_null(),
            Some((m, val)) => b.append_variant(Variant::try_new(m, val).unwrap()),
        }
    }
    let plain = DataType::Struct(Fields::from(vec![
        Field::new("a", DataType::Int64, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
            true,
        ),
    ]));
    let shredded = shred_variant(&b.build(), &plain).unwrap();
    let doc = ArrayRef::from(shredded);
    let ids = Int32Array::from(rows.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("doc", doc.data_type().clone(), true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));
    let seed = RecordBatch::try_new(schema, vec![Arc::new(ids), doc]).unwrap();
    let file1 = write_raw_data_file(&table, "arr1", seed).await;
    // A second undersized file so the plan has something to binpack.
    let file2 = write_raw_data_file(
        &table,
        "arr2",
        shredded_batch(&[(4, Some(variant_object()))]),
    )
    .await;
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let mut seeded = rows.clone();
    seeded.push((4, Some(variant_object())));
    let expected_json = jsons_of(&seeded);
    assert_eq!(live_docs_as_json(&table).await, expected_json);

    let cfg = Config {
        shred_variants: true,
        ..aggressive_cfg()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let paths = live_data_file_paths(&table).await;
    assert_eq!(paths.len(), 1);
    let doc_type = doc_type_of(&table, &paths[0]).await;
    let DataType::Struct(children) = &doc_type else {
        panic!("doc is not a struct: {doc_type:?}");
    };
    let tv = children
        .iter()
        .find(|f| f.name() == "typed_value")
        .expect("output is shredded");
    let DataType::Struct(tv_children) = tv.data_type() else {
        panic!("typed_value is not a struct: {:?}", tv.data_type());
    };
    // The FIRST input file's layout wins (derivation reads one footer):
    // both `a` and the ARRAY `tags` survive.
    let tags_node = tv_children
        .iter()
        .find(|f| f.name() == "tags")
        .expect("array field survives the rewrite");
    let DataType::Struct(tags_children) = tags_node.data_type() else {
        panic!("tags is not a shred node: {:?}", tags_node.data_type());
    };
    let tags_tv = tags_children
        .iter()
        .find(|f| f.name() == "typed_value")
        .expect("tags carries a typed_value");
    assert!(
        matches!(tags_tv.data_type(), DataType::List(_)),
        "tags typed_value stays a list: {:?}",
        tags_tv.data_type()
    );

    // Values still read back semantically equal through the fold.
    assert_eq!(live_docs_as_json(&table).await, expected_json);
}
