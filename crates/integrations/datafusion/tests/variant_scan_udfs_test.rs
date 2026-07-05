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

//! End-to-end: variant scalar UDFs over a REAL Iceberg scan.
//!
//! An Iceberg V3 table with a variant column is fed one CANONICAL data file
//! (written through the iceberg writer chain) and one SHREDDED data file
//! (raw parquet with a `typed_value` subtree — the layout other engines
//! write). The table is registered through [`IcebergCatalogProvider`] and
//! flattened with plain SQL — proving the UDFs work on what the scan
//! actually delivers (post unshred-fold), not just on builder-made arrays.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
    StructArray,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
    PrimitiveType, Schema, Struct as IcebergStruct, Type, VariantType,
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
    TableCreation,
};
use iceberg_datafusion::IcebergCatalogProvider;
use iceberg_datafusion::functions::register_variant_functions;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::variant::{
    ShreddedSchemaBuilder, Variant, VariantArrayBuilder, VariantBuilder, shred_variant,
};
use tempfile::TempDir;

/// (metadata, value) canonical variant buffers.
type VariantBytes = (Vec<u8>, Vec<u8>);

fn doc(a: i64, b: &str) -> VariantBytes {
    let mut builder = VariantBuilder::new();
    let mut obj = builder.new_object();
    obj.insert("a", a);
    obj.insert("b", b);
    obj.finish();
    builder.finish()
}

fn doc_string(s: &str) -> VariantBytes {
    let mut builder = VariantBuilder::new();
    builder.append_value(s);
    builder.finish()
}

fn canonical_fields() -> Fields {
    Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ])
}

fn outer_schema(doc_type: DataType) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("doc", doc_type, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]))
}

/// Canonical batch of (id, doc) rows; `None` = NULL variant slot.
fn canonical_batch(rows: &[(i32, Option<VariantBytes>)]) -> RecordBatch {
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
    let docs = StructArray::new(
        canonical_fields(),
        vec![Arc::new(metas) as ArrayRef, Arc::new(vals) as ArrayRef],
        Some(validity),
    );
    RecordBatch::try_new(outer_schema(DataType::Struct(canonical_fields())), vec![
        Arc::new(ids),
        Arc::new(docs),
    ])
    .unwrap()
}

/// SHREDDED batch (shred on `a: Int64`) — the layout other engines write.
fn shredded_batch(rows: &[(i32, Option<VariantBytes>)]) -> RecordBatch {
    let mut builder = VariantArrayBuilder::new(rows.len());
    for (_, v) in rows {
        match v {
            None => builder.append_null(),
            Some((m, val)) => builder.append_variant(Variant::try_new(m, val).unwrap()),
        }
    }
    let as_type = ShreddedSchemaBuilder::new()
        .with_path("a", DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&builder.build(), &as_type).unwrap();
    let docs = ArrayRef::from(shredded);
    let ids = Int32Array::from(rows.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    RecordBatch::try_new(outer_schema(docs.data_type().clone()), vec![
        Arc::new(ids),
        docs,
    ])
    .unwrap()
}

async fn write_via_chain(table: &Table, prefix: &str, batch: RecordBatch) -> Vec<DataFile> {
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

async fn write_raw(table: &Table, name: &str, batch: RecordBatch) -> DataFile {
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

#[tokio::test]
async fn flatten_iceberg_variant_scan_with_sql() {
    let warehouse = TempDir::new().unwrap();
    let catalog: Arc<dyn Catalog> = Arc::new(
        MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
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
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "doc", Type::Variant(VariantType)).into(),
        ])
        .build()
        .unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("docs".to_string())
                .schema(schema)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    // One canonical file + one SHREDDED file.
    let canonical_rows: Vec<(i32, Option<VariantBytes>)> = vec![
        (1, Some(doc(10, "alpha"))),
        (2, Some(doc_string("not an object"))),
        (3, None),
    ];
    let shredded_rows: Vec<(i32, Option<VariantBytes>)> = vec![
        (4, Some(doc(40, "delta"))),
        (5, Some(doc(50, "echo"))),
        (6, None),
    ];
    let f1 = write_via_chain(&table, "c", canonical_batch(&canonical_rows)).await;
    let f2 = write_raw(&table, "s", shredded_batch(&shredded_rows)).await;
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(f1.into_iter().chain([f2]).collect::<Vec<_>>())
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();

    // Register through the catalog provider + the variant UDFs.
    let provider = Arc::new(IcebergCatalogProvider::try_new(catalog).await.unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);
    register_variant_functions(&ctx);

    // The flatten: typed getters + a boolean derived from presence.
    let batches = ctx
        .sql(
            "SELECT id, \
                    variant_get_bigint(doc, '$.a') AS a, \
                    variant_get_string(doc, '$.b') AS b, \
                    variant_get_string(doc, '$') AS whole_string, \
                    variant_get_bigint(doc, '$.a') IS NOT NULL AS has_a \
               FROM catalog.db.docs ORDER BY id",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = datafusion::arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), 6);

    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let a = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let b = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let whole = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let has_a = batch
        .column(4)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    let expect_a = [Some(10), None, None, Some(40), Some(50), None];
    let expect_b = [Some("alpha"), None, None, Some("delta"), Some("echo"), None];
    for i in 0..6 {
        assert_eq!(ids.value(i), (i + 1) as i32);
        assert_eq!(
            (!a.is_null(i)).then(|| a.value(i)),
            expect_a[i],
            "row {i} a"
        );
        assert_eq!(
            (!b.is_null(i)).then(|| b.value(i)),
            expect_b[i],
            "row {i} b"
        );
        assert_eq!(has_a.value(i), expect_a[i].is_some(), "row {i} has_a");
    }
    // Row 2's whole document IS a string — root extraction as string works;
    // object rows don't cast to string (NULL).
    assert_eq!(whole.value(1), "not an object");
    assert!(whole.is_null(0));
}
