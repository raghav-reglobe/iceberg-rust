//! Integration test: compaction over MULTIPLE groups with pre-existing deletion
//! vectors — the multi-group path that `dv_reabsorption.rs` (single file, single
//! DV, single group) does not cover.
//!
//! With `target_file_size_bytes=1` every candidate file bin-packs into its own
//! group, so N files exercise an N-group plan. Regression test for two defects
//! that only surface with more than one group:
//!   - **output name collision** — a fresh `DefaultFileNameGenerator` per group
//!     with a constant prefix named every group's output the same path, so groups
//!     overwrote each other's files (corrupt/duplicated data);
//!   - **row duplication** — committing once per group re-ran the manifest
//!     carry-forward over each prior snapshot.
//! Fixed by a unique per-write name + a single accumulated `RewriteFiles` commit.
//! Asserts every DV is reabsorbed, no data file is referenced by more than one DV
//! ("Can't index multiple DVs"), and the row count is exact (DVs applied).

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

/// target=1 => each candidate file is its own bin => its own group => its own
/// commit. min_input_files=1 + delete_file_threshold=1 so every file (undersized
/// and/or delete-bearing) is a candidate.
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
        // unique prefix per file — DefaultFileNameGenerator's counter resets per
        // instance, so a shared prefix would collide all files to one path.
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
            continue;
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

/// Max number of live DVs referencing any single data file. >1 is the
/// "Can't index multiple DVs" corruption.
async fn max_dvs_per_file(table: &Table) -> usize {
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
    let mut counts: HashMap<String, usize> = HashMap::new();
    for mf in ml.entries() {
        if mf.content == ManifestContentType::Data {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                if let Some(r) = e.data_file().referenced_data_file() {
                    *counts.entry(r).or_default() += 1;
                }
            }
        }
    }
    counts.values().copied().max().unwrap_or(0)
}

/// Write a deletion vector dropping `pos` from `data_file_path`; commit via row_delta.
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

/// 4 data files [1..12], a DV on file A (drop id=1) and file C (drop id=7).
/// 10 live rows, 2 DVs pre-compaction.
async fn table_with_multiple_dvs(warehouse: &TempDir) -> (impl Catalog, TableIdent) {
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
    let (a_path, c_path) = (a.file_path().to_string(), c.file_path().to_string());
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(vec![a, b, c, d])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(live_row_count(&table).await, 12);

    let table = add_dv(table, &catalog, warehouse, a_path, 0, "dv-a").await;
    let table = add_dv(table, &catalog, warehouse, c_path, 0, "dv-c").await;
    assert_eq!(live_row_count(&table).await, 10, "DVs drop id=1 and id=7");
    assert_eq!(delete_file_count(&table).await, 2, "2 DVs pre-compaction");
    assert_eq!(
        max_dvs_per_file(&table).await,
        1,
        "clean: 1 DV per file pre-compaction"
    );

    (catalog, ident)
}

/// Across the 4 per-group commits, every DV is reabsorbed, no file ends with
/// >1 DV, and the row count is exact.
#[tokio::test]
async fn multi_group_reabsorbs_all_dvs_no_multi_dv() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ident) = table_with_multiple_dvs(&warehouse).await;

    compact_table(&catalog, &ident, &per_file_groups_cfg())
        .await
        .unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        live_row_count(&table).await,
        10,
        "rows preserved (DVs applied during rewrite, no duplication across groups)"
    );
    assert_eq!(
        max_dvs_per_file(&table).await,
        0,
        "no data file referenced by >1 DV after compaction (multi-DV corruption)"
    );
    assert_eq!(
        delete_file_count(&table).await,
        0,
        "all DVs reabsorbed in the single multi-group commit"
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
}

/// A SURVIVOR file (large/optimal → not a compaction candidate) that carries a DV
/// must pass through compaction UNTOUCHED — its DV neither dropped nor joined by
/// others — while smaller files are compacted and their DVs reabsorbed. This is the
/// shape the prod incident hit (a large survivor whose DV ended up clustered with
/// others) that `multi_group_reabsorbs_all_dvs_no_multi_dv` (all files compacted,
/// no survivor) never exercised.
///
/// NB on coverage: this reproduces the *structure* but not the prod *trigger*.
/// Prod's DVs were duckdb-written, where iceberg-rust's `referenced_data_file()`
/// accessor returns None — so the OLD engine (which keyed delete files by
/// referenced_data_file) silently missed them and left them dangling. iceberg-rust-
/// written DVs (here) always populate referenced_data_file, so both the old and the
/// fixed engine reabsorb them; the duckdb-specific path is only validatable on a real
/// cluster. The fix (sourcing removed deletes from the scan's `FileScanTask.deletes`,
/// matching iceberg-go/iceberg-java) is robust regardless. This test guards the
/// survivor invariant — untouched, no clustering, no orphans — against regressions.
#[tokio::test]
async fn survivor_with_dv_untouched_while_others_compacted() {
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

    // A large survivor + two small files (distinct id ranges).
    let survivor = write_one_data_file(&table, "survivor", (0..5000).collect()).await;
    let small1 = write_one_data_file(&table, "small-1", vec![900001, 900002]).await;
    let small2 = write_one_data_file(&table, "small-2", vec![900003, 900004]).await;
    let surv_size = survivor.file_size_in_bytes();
    let small_size = small1.file_size_in_bytes();
    assert!(
        surv_size > small_size,
        "test setup: survivor must be larger than the small files ({surv_size} vs {small_size})"
    );
    let surv_path = survivor.file_path().to_string();
    let small1_path = small1.file_path().to_string();
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(vec![survivor, small1, small2])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    let table = add_dv(table, &catalog, &warehouse, surv_path, 0, "dv-surv").await;
    let table = add_dv(table, &catalog, &warehouse, small1_path, 0, "dv-small1").await;
    assert_eq!(
        live_row_count(&table).await,
        5002,
        "5000 + 2 + 2, minus 2 DV'd rows"
    );
    assert_eq!(delete_file_count(&table).await, 2);
    assert_eq!(max_dvs_per_file(&table).await, 1);

    // Size-based candidacy: only undersized files (the small ones) are candidates;
    // the survivor is optimal (>= min) and its lone DV is below delete_file_threshold,
    // so it is NOT a candidate -> skipped.
    let cfg = Config {
        min_file_size_bytes: small_size + 1,
        max_file_size_bytes: surv_size * 4,
        target_file_size_bytes: surv_size,
        min_input_files: 1,
        delete_file_threshold: 1000,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(live_row_count(&table).await, 5002, "rows preserved");
    assert_eq!(
        max_dvs_per_file(&table).await,
        1,
        "the survivor's DV must stay alone — no clustering of other DVs onto it"
    );
    assert_eq!(
        delete_file_count(&table).await,
        1,
        "small file's DV reabsorbed; the survivor's untouched DV remains"
    );
}

/// Build a deletion vector (drop `pos` from `data_file_path`) WITHOUT committing —
/// so the caller can add several in one `row_delta` (same manifest) or later remove one.
async fn build_dv(
    table: &Table,
    warehouse: &TempDir,
    data_file_path: &str,
    pos: u64,
    dv_name: &str,
) -> DataFile {
    let mut positions = RoaringTreemap::new();
    positions.insert(pos);
    let dv_path = format!("{}/{}.puffin", warehouse.path().to_str().unwrap(), dv_name);
    DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            data_file_path.to_string(),
            Struct::from_iter(Vec::<Option<Literal>>::new()),
            0,
        )
        .await
        .unwrap()
}

/// The real-table corruption: a data file accumulates many **DELETED** (superseded)
/// DV entries across merges. When a later compaction rewrites a manifest that holds
/// such DELETED entries (because it also holds a live file being removed), the
/// rewrite must NOT resurrect them as EXISTING — doing so yields phantom live DVs
/// ("Can't index multiple DVs"). The all-compacted/survivor tests above never had a
/// DELETED entry, so they couldn't catch this.
///
/// Setup: `dv_s`(→S) and `dv_a`(→A) are added in ONE row_delta (same manifest M),
/// then `dv_s` is removed — so M becomes `[dv_s DELETED, dv_a EXISTING]`. Compacting
/// A rewrites M (dv_a is removed); `dv_s` must stay dead.
#[tokio::test]
async fn superseded_deleted_dv_not_resurrected_on_rewrite() {
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

    // S large (oversized, no DV → survivor); A small (undersized → compacted).
    let s = write_one_data_file(&table, "file-s", (0..5000).collect()).await;
    let a = write_one_data_file(&table, "file-a", vec![900001, 900002]).await;
    let s_path = s.file_path().to_string();
    let a_path = a.file_path().to_string();
    let (s_size, a_size) = (s.file_size_in_bytes(), a.file_size_in_bytes());
    assert!(
        s_size > a_size,
        "test setup: S must be larger than A ({s_size} vs {a_size})"
    );
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files(vec![s, a])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    // dv_s + dv_a in ONE row_delta → one delete manifest.
    let dv_s = build_dv(&table, &warehouse, &s_path, 0, "dv-s").await;
    let dv_a = build_dv(&table, &warehouse, &a_path, 0, "dv-a").await;
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_delete_files(vec![dv_s.clone(), dv_a])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(
        live_row_count(&table).await,
        5000,
        "dv_s drops 1 of S, dv_a drops 1 of A"
    );

    // Replace S's DV: remove dv_s + add dv_s2 (a DIFFERENT row) in one row_delta.
    // dv_s becomes a DELETED entry, co-located with dv_a (EXISTING) and dv_s2 (ADDED).
    let dv_s2 = build_dv(&table, &warehouse, &s_path, 1, "dv-s2").await;
    let tx = Transaction::new(&table);
    let table = tx
        .row_delta()
        .add_delete_files(vec![dv_s2])
        .remove_delete_files(vec![dv_s])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(
        delete_file_count(&table).await,
        2,
        "dv_s2 + dv_a live; dv_s removed"
    );
    assert_eq!(
        max_dvs_per_file(&table).await,
        1,
        "clean: one live DV per file"
    );
    assert_eq!(
        live_row_count(&table).await,
        5000,
        "dv_s2 drops 1 of S, dv_a drops 1 of A"
    );

    // Compact A (undersized → candidate). S is optimal with a sub-threshold DV count,
    // so it's a SURVIVOR. The compaction rewrites the delete manifest holding
    // [dv_s DELETED, dv_a EXISTING]; the DELETED dv_s must not be resurrected.
    let cfg = Config {
        min_file_size_bytes: a_size + 1,
        max_file_size_bytes: s_size * 4,
        target_file_size_bytes: a_size + 1,
        min_input_files: 1,
        delete_file_threshold: 1000,
        ..Config::default()
    };
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(
        delete_file_count(&table).await,
        1,
        "dv_a reabsorbed, dv_s2 survives; dv_s must NOT be resurrected"
    );
    assert_eq!(
        max_dvs_per_file(&table).await,
        1,
        "S has only dv_s2 — a resurrected dv_s would make 2 live DVs on S (multi-DV)"
    );
    assert_eq!(
        live_row_count(&table).await,
        5000,
        "S keeps 4999 (dv_s2) + A's surviving row; a resurrected dv_s would drop another (→4999)"
    );
}
