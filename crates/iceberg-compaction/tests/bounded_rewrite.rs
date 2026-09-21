//! Integration test: a BUDGET-bounded pass (`Config::budget`) commits the groups
//! it finished instead of all-or-nothing, and successive bounded passes converge.
//!
//! With `target_file_size_bytes=1` every candidate file is its own group, so 4
//! files give a 4-group plan. A budget that is already exhausted when the pass
//! starts still runs exactly ONE group (the progress guarantee) — and because
//! groups execute most-read-cost-removed first, that group is a DELETE-BEARING
//! one. Asserts per pass: one `Replace` snapshot, exact live rows (no
//! resurrection, no loss), never more than one DV per data file, and the
//! delete-file count falling by exactly the group that ran.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

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

fn per_file_groups_cfg() -> Config {
    Config {
        target_file_size_bytes: 1,
        min_input_files: 1,
        delete_file_threshold: 1,
        ..Config::default()
    }
}

async fn write_one_data_file(table: &Table, name: &str, ids: Vec<i32>) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
    );
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]));
    let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    let files = writer.close().await.unwrap();
    assert_eq!(files.len(), 1);
    files.into_iter().next().unwrap()
}

async fn live_ids(table: &Table) -> Vec<i32> {
    let mut stream = table
        .scan()
        .select_all()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let mut ids = Vec::new();
    while let Some(b) = stream.try_next().await.unwrap() {
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .clone();
        ids.extend(col.iter().flatten());
    }
    ids.sort_unstable();
    ids
}

/// (alive delete files, max live DVs referencing one data file)
async fn delete_state(table: &Table) -> (usize, usize) {
    let Some(snap) = table.metadata().current_snapshot() else {
        return (0, 0);
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
    let mut counts: HashMap<String, usize> = HashMap::new();
    for mf in ml.entries() {
        if mf.content == ManifestContentType::Data {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                n += 1;
                if let Some(r) = e.data_file().referenced_data_file() {
                    *counts.entry(r).or_default() += 1;
                }
            }
        }
    }
    (n, counts.values().copied().max().unwrap_or(0))
}

async fn add_dv(
    table: Table,
    catalog: &dyn Catalog,
    warehouse: &TempDir,
    data_file_path: String,
    pos: u64,
    dv_name: &str,
) -> Table {
    let mut positions = RoaringTreemap::new();
    positions.insert(pos);
    let dv_path = format!("{}/{}.puffin", warehouse.path().to_str().unwrap(), dv_name);
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
    tx.row_delta()
        .add_delete_files(vec![dv_file])
        .apply(tx)
        .unwrap()
        .commit(catalog)
        .await
        .unwrap()
}

/// 4 data files [1..12]; DVs on the LAST-planned files C (drop id=7) and D
/// (drop id=10) — plan order alone would reach them last.
async fn table_with_trailing_dvs(warehouse: &TempDir) -> (impl Catalog, TableIdent) {
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

    let a = write_one_data_file(&table, "file-a", vec![1, 2, 3]).await;
    let b = write_one_data_file(&table, "file-b", vec![4, 5, 6]).await;
    let c = write_one_data_file(&table, "file-c", vec![7, 8, 9]).await;
    let d = write_one_data_file(&table, "file-d", vec![10, 11, 12]).await;
    let (c_path, d_path) = (c.file_path().to_string(), d.file_path().to_string());
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(vec![a, b, c, d])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    let table = add_dv(table, &catalog, warehouse, c_path, 0, "dv-c").await;
    let table = add_dv(table, &catalog, warehouse, d_path, 0, "dv-d").await;
    assert_eq!(delete_state(&table).await, (2, 1));
    (catalog, ident)
}

const LIVE: [i32; 10] = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12];

#[tokio::test]
async fn bounded_pass_commits_finished_groups_and_converges() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = table_with_trailing_dvs(&warehouse).await;
    assert_eq!(
        live_ids(&catalog.load_table(&ident).await.unwrap()).await,
        LIVE
    );

    // Pass 1 — budget already spent: exactly ONE group runs, and it is a
    // delete-bearing one (value order), not plan-first file A.
    let bounded = Config {
        budget: Some(Instant::now()),
        ..per_file_groups_cfg()
    };
    let out = compact_table(&catalog, &ident, &bounded).await.unwrap();
    assert_eq!(out.groups_planned, 4);
    assert_eq!(out.groups_rewritten, 1);
    assert_eq!(
        (out.rewritten, out.added, out.reabsorbed_deletes),
        (1, 1, 1)
    );
    assert!(!out.complete, "planned work remains");
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Replace
    );
    assert_eq!(live_ids(&table).await, LIVE, "no loss, no resurrection");
    assert_eq!(delete_state(&table).await, (1, 1), "one DV reabsorbed");
    let snaps_after_1 = table.metadata().snapshots().count();

    // Pass 2 — bounded again: the OTHER delete-bearing group goes first.
    let bounded = Config {
        budget: Some(Instant::now()),
        ..per_file_groups_cfg()
    };
    let out = compact_table(&catalog, &ident, &bounded).await.unwrap();
    assert_eq!(out.groups_rewritten, 1);
    assert_eq!(out.reabsorbed_deletes, 1);
    assert!(!out.complete);
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(table.metadata().snapshots().count(), snaps_after_1 + 1);
    assert_eq!(live_ids(&table).await, LIVE);
    assert_eq!(delete_state(&table).await, (0, 0), "every DV reabsorbed");

    // Pass 3 — unbounded: finishes the plan and says so.
    let out = compact_table(&catalog, &ident, &per_file_groups_cfg())
        .await
        .unwrap();
    assert!(out.complete);
    assert_eq!(out.groups_rewritten, out.groups_planned);
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(live_ids(&table).await, LIVE);
    assert_eq!(delete_state(&table).await, (0, 0));
}

#[tokio::test]
async fn generous_budget_is_an_unbounded_pass() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = table_with_trailing_dvs(&warehouse).await;
    let cfg = Config {
        budget: Some(Instant::now() + std::time::Duration::from_secs(3600)),
        ..per_file_groups_cfg()
    };
    let out = compact_table(&catalog, &ident, &cfg).await.unwrap();
    assert!(out.complete);
    assert_eq!((out.groups_planned, out.groups_rewritten), (4, 4));
    assert_eq!(out.reabsorbed_deletes, 2);
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(live_ids(&table).await, LIVE);
    assert_eq!(delete_state(&table).await, (0, 0));
}
