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

//! End-to-end merge-on-read MERGE INTO: the SCD2 upsert shape as plain SQL
//! against a V3 Iceberg table, through the DataFusion MERGE planner, the
//! `TableProvider::merge_into` hook and the MoR execution chain.
//!
//! The USING source is the application-level SCD2 "chained" shape: a UNION of
//! the incoming batch rows (future current versions) and the demote rows
//! (computed by joining the batch against the target's current set). MATCHED
//! rows are closed via deletion vectors on the scanned `(_file, _pos)` plus
//! an appended demoted version; NOT MATCHED rows append. One `RowDelta`
//! snapshot carries it all.
//!
//! The second test re-merges rows in a file that ALREADY carries a deletion
//! vector: the engine must produce ONE consolidated DV per data file and
//! remove the superseded DV in the same commit (the V3 invariant — never two
//! live DVs per file).

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::spec::{
    DataContentType, DataFileFormat, FormatVersion, ManifestContentType, ManifestList,
    NestedField, PrimitiveType, Schema, Type,
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
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const CATALOG: &str = "catalog";
const NS: &str = "db";
const TABLE: &str = "t";

fn scd2_iceberg_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "val", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "_valid_from", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(4, "_valid_to", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(5, "_is_current", Type::Primitive(PrimitiveType::Boolean))
                .into(),
            NestedField::required(6, "_cdc_offset", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap()
}

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

fn scd2_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "val", DataType::Utf8, true),
        field(3, "_valid_from", DataType::Int64, false),
        field(4, "_valid_to", DataType::Int64, true),
        field(5, "_is_current", DataType::Boolean, false),
        field(6, "_cdc_offset", DataType::Int64, false),
    ]))
}

#[allow(clippy::type_complexity)]
fn scd2_batch(rows: &[(i32, &str, i64, Option<i64>, bool, i64)]) -> RecordBatch {
    RecordBatch::try_new(scd2_arrow_schema(), vec![
        Arc::new(Int32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>())),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.3).collect::<Vec<_>>())),
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.4).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.5).collect::<Vec<_>>())),
    ])
    .unwrap()
}

/// CDC batch rows fed through a MemTable: (id, val, _valid_from, _cdc_offset).
fn cdc_batch(rows: &[(i32, &str, i64, i64)]) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
        Field::new("_valid_from", DataType::Int64, false),
        Field::new("_cdc_offset", DataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>())),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.3).collect::<Vec<_>>())),
    ])
    .unwrap()
}

async fn write_one_data_file(table: &Table, batch: RecordBatch) -> Vec<iceberg::spec::DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new("seed".to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

/// Build the V3 SCD2 table with a seeded current set and a registered
/// DataFusion session (`catalog.db.t` + the `batch` MemTable).
async fn setup(
    warehouse: &TempDir,
    seed: &[(i32, &str, i64, Option<i64>, bool, i64)],
    batch: &[(i32, &str, i64, i64)],
) -> (Arc<dyn Catalog>, SessionContext) {
    let catalog: Arc<dyn Catalog> = Arc::new(MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap());
    let ns = NamespaceIdent::new(NS.to_string());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(TABLE.to_string())
                .schema(scd2_iceberg_schema())
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    let data_files = write_one_data_file(&table, scd2_batch(seed)).await;
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();

    let ctx = SessionContext::new();
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(Arc::clone(&catalog))
            .await
            .unwrap(),
    );
    ctx.register_catalog(CATALOG, provider);

    let cdc = cdc_batch(batch);
    let mem = MemTable::try_new(cdc.schema(), vec![vec![cdc]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();

    (catalog, ctx)
}

/// The SCD2 merge statement — the application-SQL "chained" shape: incoming
/// batch rows insert as new current versions; demote rows (computed against
/// the target's current set) close the superseded versions.
fn scd2_merge_sql() -> String {
    format!(
        "MERGE INTO {CATALOG}.{NS}.{TABLE} AS t USING ( \
             SELECT id, val, _valid_from, CAST(NULL AS BIGINT) AS _valid_to, \
                    true AS _is_current, _cdc_offset \
             FROM batch \
             UNION ALL \
             SELECT t2.id, t2.val, t2._valid_from, b.new_vf AS _valid_to, \
                    false AS _is_current, t2._cdc_offset \
             FROM {CATALOG}.{NS}.{TABLE} t2 \
             JOIN (SELECT id, MIN(_valid_from) AS new_vf FROM batch GROUP BY id) b \
               ON t2.id = b.id AND t2._is_current \
         ) AS s \
         ON t.id = s.id AND t._valid_from = s._valid_from AND t._cdc_offset = s._cdc_offset \
         WHEN MATCHED THEN UPDATE SET _valid_to = s._valid_to, _is_current = s._is_current \
         WHEN NOT MATCHED THEN INSERT (id, val, _valid_from, _valid_to, _is_current, _cdc_offset) \
             VALUES (s.id, s.val, s._valid_from, s._valid_to, s._is_current, s._cdc_offset)"
    )
}

/// Sorted (id, val, _valid_from, _valid_to, _is_current) projection of the
/// full table, read back through DataFusion.
async fn read_state(ctx: &SessionContext) -> Vec<(i32, String, i64, Option<i64>, bool)> {
    let batches = ctx
        .sql(&format!(
            "SELECT id, val, _valid_from, _valid_to, _is_current \
             FROM {CATALOG}.{NS}.{TABLE} ORDER BY id, _valid_from"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let val = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let vf = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let vt = b.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let cur = b.column(4).as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((
                id.value(i),
                val.value(i).to_string(),
                vf.value(i),
                vt.is_valid(i).then(|| vt.value(i)),
                cur.value(i),
            ));
        }
    }
    out
}

/// Live delete-file entries: (referenced data file, DV cardinality).
async fn live_dvs(table: &Table) -> Vec<(String, u64)> {
    let snap = table.metadata().current_snapshot().unwrap();
    let bytes = table
        .file_io()
        .new_input(snap.manifest_list())
        .unwrap()
        .read()
        .await
        .unwrap();
    let ml = ManifestList::parse_with_version(&bytes, table.metadata().format_version()).unwrap();
    let mut out = Vec::new();
    for mf in ml.entries() {
        if mf.content != ManifestContentType::Deletes {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                assert_eq!(e.data_file().content_type(), DataContentType::PositionDeletes);
                out.push((
                    e.data_file()
                        .referenced_data_file()
                        .expect("DV carries its referenced data file"),
                    e.data_file().record_count(),
                ));
            }
        }
    }
    out.sort();
    out
}

async fn load_table(catalog: &Arc<dyn Catalog>) -> Table {
    catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new(NS.to_string()),
            TABLE.to_string(),
        ))
        .await
        .unwrap()
}

#[tokio::test]
async fn scd2_merge_demotes_via_dv_and_inserts_in_one_snapshot() {
    let warehouse = TempDir::new().unwrap();
    // Current set: ids 1..=3, one current row each.
    let (catalog, ctx) = setup(
        &warehouse,
        &[
            (1, "a", 10, None, true, 100),
            (2, "b", 10, None, true, 101),
            (3, "c", 10, None, true, 102),
        ],
        // CDC batch: a new version of id=1 and a brand-new id=4.
        &[(1, "a2", 20, 200), (4, "d", 20, 203)],
    )
    .await;

    let before = load_table(&catalog).await;
    let snaps_before = before.metadata().snapshots().count();

    ctx.sql(&scd2_merge_sql()).await.unwrap().collect().await.unwrap();

    let table = load_table(&catalog).await;
    // Exactly ONE new snapshot carries the demote + both appends.
    assert_eq!(table.metadata().snapshots().count(), snaps_before + 1);

    let state = read_state(&ctx).await;
    assert_eq!(state, vec![
        (1, "a".to_string(), 10, Some(20), false),
        (1, "a2".to_string(), 20, None, true),
        (2, "b".to_string(), 10, None, true),
        (3, "c".to_string(), 10, None, true),
        (4, "d".to_string(), 20, None, true),
    ]);

    // The demote is a deletion vector on the seed file (position 0 = id 1),
    // and it is the only live DV.
    let dvs = live_dvs(&table).await;
    assert_eq!(dvs.len(), 1, "one DV total: {dvs:?}");
    assert_eq!(dvs[0].1, 1, "seed DV covers exactly the demoted row");
}

#[tokio::test]
async fn second_merge_consolidates_dvs_per_file() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ctx) = setup(
        &warehouse,
        &[
            (1, "a", 10, None, true, 100),
            (2, "b", 10, None, true, 101),
            (3, "c", 10, None, true, 102),
        ],
        &[(1, "a2", 20, 200), (4, "d", 20, 203)],
    )
    .await;

    // Merge #1: demote id=1 (seed file gets its first DV).
    ctx.sql(&scd2_merge_sql()).await.unwrap().collect().await.unwrap();

    // Merge #2: new versions for id=2 (current row in the SEED file, which
    // already carries a DV) and id=1 (current row in the file appended by
    // merge #1). Replace the batch table content.
    ctx.deregister_table("batch").unwrap();
    let cdc = cdc_batch(&[(2, "b2", 30, 300), (1, "a3", 30, 301)]);
    let mem = MemTable::try_new(cdc.schema(), vec![vec![cdc]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();

    ctx.sql(&scd2_merge_sql()).await.unwrap().collect().await.unwrap();

    let state = read_state(&ctx).await;
    assert_eq!(state, vec![
        (1, "a".to_string(), 10, Some(20), false),
        (1, "a2".to_string(), 20, Some(30), false),
        (1, "a3".to_string(), 30, None, true),
        (2, "b".to_string(), 10, Some(30), false),
        (2, "b2".to_string(), 30, None, true),
        (3, "c".to_string(), 10, None, true),
        (4, "d".to_string(), 20, None, true),
    ]);

    // THE V3 invariant: at most one live DV per data file. The seed file's
    // DV was superseded by a consolidated one covering both demoted rows
    // (id=1 at position 0, id=2 at position 1).
    let table = load_table(&catalog).await;
    let dvs = live_dvs(&table).await;
    let mut per_file: HashMap<&str, usize> = HashMap::new();
    for (file, _) in &dvs {
        *per_file.entry(file.as_str()).or_default() += 1;
    }
    assert!(
        per_file.values().all(|&n| n == 1),
        "a data file must carry at most ONE live DV: {dvs:?}"
    );
    let seed_dv = dvs
        .iter()
        .find(|(file, _)| file.contains("seed"))
        .expect("seed file carries a DV");
    assert_eq!(
        seed_dv.1, 2,
        "seed DV consolidates both demoted rows: {dvs:?}"
    );
    assert_eq!(dvs.len(), 2, "seed file + merge-1 output file: {dvs:?}");
}

