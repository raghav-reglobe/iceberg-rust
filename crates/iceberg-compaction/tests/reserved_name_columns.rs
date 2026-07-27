//! Regression: compaction (and any scan) of a table whose DATA columns carry
//! reserved metadata NAMES — e.g. a media table with a `file_path` column,
//! colliding with the position-delete `file_path` (reserved field id
//! 2147483546) — must resolve those names to the TABLE SCHEMA's field ids.
//! Resolving reserved names first projected the reserved id, which no data
//! file produces, failing every select-all scan with "metadata column with
//! field id 2147483546 was projected but not produced by the reader".

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, Type,
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
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

async fn write_one_file(table: &Table, prefix: &str, batch: RecordBatch) -> Vec<DataFile> {
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

/// A media-shaped table: `id INT, file_path STRING, pos BIGINT` — both
/// `file_path` and `pos` are reserved position-delete column names.
#[tokio::test]
async fn compaction_handles_data_columns_with_reserved_metadata_names() {
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
            NestedField::optional(2, "file_path", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "pos", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("media".to_string())
                .schema(schema)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "file_path", DataType::Utf8, true),
        field(3, "pos", DataType::Int64, true),
    ]));
    for (prefix, ids) in [("a", [1, 2]), ("b", [3, 4])] {
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(
                ids.iter()
                    .map(|i| format!("s3://media/{i}.jpg"))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                ids.iter().map(|i| *i as i64).collect::<Vec<_>>(),
            )),
        ])
        .unwrap();
        let files = write_one_file(&table, prefix, batch).await;
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(files)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
    }
    let ident = TableIdent::new(ns, "media".to_string());

    // A plain select-all scan resolves `file_path`/`pos` to the DATA columns.
    let table = catalog.load_table(&ident).await.unwrap();
    let batches: Vec<RecordBatch> = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 4);

    // An explicit select of the colliding name returns the DATA column too.
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["file_path"])
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut paths: Vec<String> = Vec::new();
    for b in &batches {
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        paths.extend((0..b.num_rows()).map(|i| col.value(i).to_string()));
    }
    paths.sort();
    assert_eq!(paths, vec![
        "s3://media/1.jpg",
        "s3://media/2.jpg",
        "s3://media/3.jpg",
        "s3://media/4.jpg"
    ]);

    // And COMPACTION (select-all scan -> rewrite) works end to end.
    let cfg = Config {
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let batches: Vec<RecordBatch> = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut ids: Vec<i32> = Vec::new();
    for b in &batches {
        let schema = b.schema();
        let idx = schema.index_of("id").unwrap();
        let col = b.column(idx).as_any().downcast_ref::<Int32Array>().unwrap();
        ids.extend(col.values());
    }
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3, 4], "compacted data intact");
}
