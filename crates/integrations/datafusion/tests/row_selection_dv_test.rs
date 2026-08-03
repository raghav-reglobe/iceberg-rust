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

//! Row-identity gate for `ICEBERG_SCAN_ROW_SELECTION` on a DV-bearing table.
//!
//! With the gate ON and a pushed-down predicate, the reader builds TWO
//! `RowSelection`s — one from the parquet page index (predicate) and one
//! from the deletion vectors — and intersects them, both expressed in the
//! coordinate space of the row groups that survived row-group pruning.
//! A coordinate-space disagreement between the two (the classic
//! off-by-row-group class) silently returns wrong rows, so this test pins
//! exact row identity between gate-OFF (the production default, trusted
//! baseline) and gate-ON scans over a table whose DVs land inside pruned
//! groups, kept groups, and across a row-group boundary.
//!
//! Layout under test: one 2,048-row seed file (4×512-row groups, 64-row
//! pages, page statistics on) merged via the SCD2 MERGE shape so a DV
//! covers two scattered clusters — one inside row group 0, one SPANNING
//! the group 2/3 boundary — plus appended replacement rows in a second
//! file. Predicates then prune to: only group 0; only groups 2+3; a mixed
//! `_is_current` conjunction; and an empty page selection.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanArray, Int32Array, Int64Array, LargeStringArray, RecordBatch,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use iceberg::spec::{
    DataContentType, DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList,
    NestedField, PartitionKey, PrimitiveType, Schema, Struct as IcebergStruct, Transform, Type,
    UnboundPartitionSpec,
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

const SEED_ROWS: i32 = 2048;
const ROW_GROUP_ROWS: usize = 512;
const PAGE_ROWS: usize = 64;
/// Two updated clusters: inside row group 0, and spanning groups 2/3.
const CLUSTER_A: std::ops::RangeInclusive<i32> = 100..=140;
const CLUSTER_B: std::ops::RangeInclusive<i32> = 1500..=1540;
/// Seed `_valid_from` = VF_BASE + id (sorted — tight page statistics);
/// replacement versions get NEW_VF_BASE + id (far outside the seed range).
const VF_BASE: i64 = 1000;
const NEW_VF_BASE: i64 = 5_000_000;

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

fn iceberg_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "val", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "_valid_from", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(4, "_valid_to", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(5, "_is_current", Type::Primitive(PrimitiveType::Boolean)).into(),
        ])
        .build()
        .unwrap()
}

fn seed_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "val", DataType::LargeUtf8, true),
        field(3, "_valid_from", DataType::Int64, false),
        field(4, "_valid_to", DataType::Int64, true),
        field(5, "_is_current", DataType::Boolean, false),
    ]));
    let ids: Vec<i32> = (0..SEED_ROWS).collect();
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(ids.clone())),
        Arc::new(LargeStringArray::from(
            ids.iter().map(|i| format!("v{i}")).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            ids.iter().map(|i| VF_BASE + *i as i64).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(vec![None::<i64>; ids.len()])),
        Arc::new(BooleanArray::from(vec![true; ids.len()])),
    ])
    .unwrap()
}

fn cdc_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
        Field::new("_valid_from", DataType::Int64, false),
    ]));
    let ids: Vec<i32> = CLUSTER_A.chain(CLUSTER_B).collect();
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(ids.clone())),
        Arc::new(datafusion::arrow::array::StringArray::from(
            ids.iter().map(|i| format!("u{i}")).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            ids.iter()
                .map(|i| NEW_VF_BASE + *i as i64)
                .collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

async fn setup(warehouse: &TempDir) -> (Arc<dyn Catalog>, SessionContext) {
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
    let ns = NamespaceIdent::new(NS.to_string());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(5, "_is_current", Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(TABLE.to_string())
                .schema(iceberg_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    // Multi-row-group, multi-page seed with page statistics so the page
    // index actually drives sub-group selection.
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .build();
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(props, schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new("seed".to_string(), None, DataFileFormat::Parquet),
    );
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        IcebergStruct::from_iter(vec![Some(Literal::bool(true))]),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key))
        .await
        .unwrap();
    writer.write(seed_batch()).await.unwrap();
    let data_files = writer.close().await.unwrap();

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

    let cdc = cdc_batch();
    let mem = MemTable::try_new(cdc.schema(), vec![vec![cdc]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();

    (catalog, ctx)
}

/// The SCD2 chained MERGE shape: matched rows demote via DV + re-append as
/// closed versions; batch rows insert as new currents.
fn scd2_merge_sql() -> String {
    format!(
        "MERGE INTO {CATALOG}.{NS}.{TABLE} AS t USING ( \
             SELECT id, val, _valid_from, CAST(NULL AS BIGINT) AS _valid_to, \
                    true AS _is_current \
             FROM batch \
             UNION ALL \
             SELECT t2.id, t2.val, t2._valid_from, b.new_vf AS _valid_to, \
                    false AS _is_current \
             FROM {CATALOG}.{NS}.{TABLE} t2 \
             JOIN (SELECT id, MIN(_valid_from) AS new_vf FROM batch GROUP BY id) b \
               ON t2.id = b.id AND t2._is_current \
         ) AS s \
         ON t.id = s.id AND t._valid_from = s._valid_from \
         WHEN MATCHED THEN UPDATE SET _valid_to = s._valid_to, _is_current = s._is_current \
         WHEN NOT MATCHED THEN INSERT (id, val, _valid_from, _valid_to, _is_current) \
             VALUES (s.id, s.val, s._valid_from, s._valid_to, s._is_current)"
    )
}

/// Deterministic full-row projection for identity comparison.
async fn collect_rows(
    ctx: &SessionContext,
    predicate: &str,
) -> Vec<(i32, String, i64, Option<i64>, bool)> {
    let batches = ctx
        .sql(&format!(
            "SELECT id, val, _valid_from, _valid_to, _is_current \
             FROM {CATALOG}.{NS}.{TABLE} WHERE {predicate} \
             ORDER BY id, _valid_from"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let val = b
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
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

/// Live DV entries: (referenced data file, cardinality).
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
                assert_eq!(
                    e.data_file().content_type(),
                    DataContentType::PositionDeletes
                );
                out.push((
                    e.data_file().referenced_data_file().unwrap(),
                    e.data_file().record_count(),
                ));
            }
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn row_selection_composes_with_deletion_vectors() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ctx) = setup(&warehouse).await;

    // Ensure the trusted-baseline phase really runs gate-OFF.
    unsafe { std::env::remove_var("ICEBERG_SCAN_ROW_SELECTION") };

    ctx.sql(&scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // The merge produced ONE DV on the seed file covering both clusters.
    let table = catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new(NS.to_string()),
            TABLE.to_string(),
        ))
        .await
        .unwrap();
    let updated = (CLUSTER_A.count() + CLUSTER_B.count()) as u64;
    let dvs = live_dvs(&table).await;
    assert_eq!(dvs.len(), 1, "one consolidated DV on the seed file");
    assert_eq!(dvs[0].1, updated, "DV covers both updated clusters");

    // Predicates chosen against the seed layout (id == VF - VF_BASE,
    // 512-row groups): prune to group 0 only (cluster A's DV inside);
    // groups 2+3 (cluster B's DV spans their boundary); a mixed
    // `_is_current` conjunction; the demoted-version band (appended file
    // rows only + DV'd seed positions); and an empty page selection.
    let predicates = [
        "true",
        "_valid_from BETWEEN 1100 AND 1300",
        "_valid_from BETWEEN 2400 AND 2600",
        "_is_current = true AND _valid_from BETWEEN 2400 AND 2600",
        "_valid_to IS NOT NULL AND _valid_from BETWEEN 2490 AND 2560",
        "_valid_from > 100000000",
    ];

    let mut baselines = Vec::new();
    for p in &predicates {
        baselines.push(collect_rows(&ctx, p).await);
    }
    // Baseline sanity: full scan = seed − DV'd + re-appended versions;
    // exactly one current row per id.
    assert_eq!(baselines[0].len(), SEED_ROWS as usize + updated as usize);
    assert_eq!(
        baselines[0].iter().filter(|r| r.4).count(),
        SEED_ROWS as usize
    );
    assert!(!baselines[1].is_empty());
    assert!(!baselines[2].is_empty());
    assert!(baselines[5].is_empty());

    // Gate ON: identical row identity on every predicate. The env var is
    // read per scan execution; this test binary runs it single-threaded.
    unsafe { std::env::set_var("ICEBERG_SCAN_ROW_SELECTION", "1") };
    for (p, baseline) in predicates.iter().zip(&baselines) {
        let gated = collect_rows(&ctx, p).await;
        assert_eq!(
            &gated, baseline,
            "row-selection scan diverged from baseline for predicate: {p}"
        );
    }
    unsafe { std::env::remove_var("ICEBERG_SCAN_ROW_SELECTION") };
}
