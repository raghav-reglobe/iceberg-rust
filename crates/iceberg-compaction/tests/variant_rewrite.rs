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
//! Covers: two undersized data files with mixed variant payloads (object,
//! string, long) plus a NULL variant slot → one compacted file; every
//! surviving row's variant bytes equal the seeded bytes, and the null stays
//! null.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, Int32Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestList, NestedField,
    Operation, PrimitiveType, Schema, Type, VariantType,
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
    Catalog, CatalogBuilder, MemoryCatalogBuilder, NamespaceIdent, TableCreation, TableIdent,
    MEMORY_CATALOG_WAREHOUSE,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use parquet::variant::VariantBuilder;
use tempfile::TempDir;

use iceberg_compaction::config::Config;
use iceberg_compaction::engine::compact_table;

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
        Field::new("doc", DataType::Struct(variant_fields), true).with_metadata(HashMap::from([
            (PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string()),
        ])),
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
    let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
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

/// The spike: binpack-compact a canonical-VARIANT table; every variant byte
/// survives unchanged and the NULL slot stays NULL.
#[tokio::test]
async fn compaction_round_trips_canonical_variant() {
    let warehouse = TempDir::new().unwrap();
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
