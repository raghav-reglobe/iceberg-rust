//! Integration test: compaction over a V3 deletion vector.
//!
//! Mirrors iceberg-go's `dv_rewrite_test.go`. Builds a V3 table (MemoryCatalog
//! over a TempDir) with a data file + a deletion vector deleting one row, then
//! compacts. Two properties, mirrored from iceberg-go:
//!   - `compaction_applies_dv_and_preserves_data` — the deleted row stays gone
//!     (the DV is applied during the rewrite). PASSES.
//!   - `compaction_reabsorbs_dv` — the DV is expunged (no delete file references
//!     the rewritten data), via the fork's `RowDelta::remove_delete_files`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList,
    NestedField, Operation, PrimitiveType, Schema, Struct, Type,
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
use roaring::RoaringTreemap;
use tempfile::TempDir;

/// min_input_files=1 + delete_file_threshold=1 so a single delete-bearing file
/// is a candidate and forms a group.
fn aggressive_cfg() -> Config {
    Config {
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    }
}

/// Write `batch` to a single new data file in `table`; return its DataFile(s).
async fn write_one_data_file(table: &Table, batch: RecordBatch) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new("data".to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

/// Count live rows via a scan (deletion vectors applied by the reader).
async fn live_row_count(table: &Table) -> usize {
    let mut stream = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let mut n = 0;
    while let Some(b) = stream.try_next().await.unwrap() {
        n += b.num_rows();
    }
    n
}

/// Count live delete-file entries (deletion vectors) in the current snapshot.
async fn delete_file_count(table: &Table) -> usize {
    let Some(snap) = table.metadata().current_snapshot() else {
        return 0;
    };
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
        if mf.content == ManifestContentType::Data {
            continue; // delete manifests only
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                n += 1;
            }
        }
    }
    n
}

/// Build a V3 table (single int `id`) with rows [1,2,3,4] and a deletion vector
/// dropping position 0 (id=1). Returns the catalog + ident; the caller keeps the
/// `warehouse` alive. Asserts the pre-compaction state (3 live rows, 1 DV).
async fn table_with_dv(warehouse: &TempDir) -> (impl Catalog, TableIdent) {
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

    // data file id=[1,2,3,4] -> fast_append
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]));
    let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(Int32Array::from(vec![
        1, 2, 3, 4,
    ]))])
    .unwrap();
    let data_files = write_one_data_file(&table, batch).await;
    let data_file_path = data_files[0].file_path().to_string();
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_row_count(&table).await, 4);

    // deletion vector dropping position 0 (id=1) -> row_delta
    let mut positions = RoaringTreemap::new();
    positions.insert(0);
    let dv_path = format!("{}/dv-0.puffin", warehouse.path().to_str().unwrap());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            data_file_path,
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_row_count(&table).await, 3, "DV drops id=1");
    assert_eq!(
        delete_file_count(&table).await,
        1,
        "DV present pre-compaction"
    );

    (catalog, ident)
}

/// The DV is applied during the rewrite: the deleted row stays gone and the
/// output row count is correct. (Proves end-to-end the engine reads DVs right.)
#[tokio::test]
async fn compaction_applies_dv_and_preserves_data() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = table_with_dv(&warehouse).await;

    compact_table(&catalog, &ident, &aggressive_cfg())
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        live_row_count(&table).await,
        3,
        "id=1 stays gone after compaction (DV applied during rewrite)"
    );
}

/// The DV is reabsorbed: no delete file references the rewritten data.
/// `commit_rewrite` removes the rewritten data file's DV via the fork's
/// `RowDelta::remove_delete_files` (content-aware `existing_manifest`), so the
/// otherwise-orphaned DV is expunged — matching iceberg-go.
#[tokio::test]
async fn compaction_reabsorbs_dv() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = table_with_dv(&warehouse).await;

    compact_table(&catalog, &ident, &aggressive_cfg())
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        delete_file_count(&table).await,
        0,
        "deletion vector must be reabsorbed (no delete files reference the rewritten data)"
    );
    // The commit is a RewriteFiles (Operation::Replace) — table data is unchanged,
    // only reorganized — not a RowDelta Overwrite.
    let snap = table.metadata().current_snapshot().unwrap();
    assert_eq!(
        snap.summary().operation,
        Operation::Replace,
        "compaction commits a Replace snapshot via RewriteFiles"
    );
}
