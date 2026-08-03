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

//! Inline DV micro-reabsorb — the merge-path hygiene commit
//! (`MorMergeOptions::reabsorb_dead_frac`).
//!
//! Pins, per the design contract:
//!   - a merge whose consolidated DV crosses the dead-fraction threshold is
//!     followed by ONE Replace snapshot rewriting exactly the crossing
//!     file(s): the DV is reabsorbed (zero live DVs), row identity is
//!     byte-equal to a reabsorb-OFF baseline, and the result batch reports
//!     `reabsorbed_files`;
//!   - a merge below the threshold commits ONLY the RowDelta (no second
//!     snapshot), keeps its DV live, and reports `reabsorb_skipped =
//!     threshold` — a crossing merge later still self-heals;
//!   - an empty merge (no lanes) adds no snapshot and never reabsorbs;
//!   - an exhausted deadline skips with `reabsorb_skipped = budget` and the
//!     merge itself still succeeds (hygiene never spends freshness).

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanArray, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
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
use iceberg_datafusion::{IcebergCatalogProvider, MorMergeOptions};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const CATALOG: &str = "catalog";
const NS: &str = "db";
const TABLE: &str = "t";
const SEED_ROWS: i32 = 100;
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

fn cdc_batch(update_ids: &[i32]) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
        Field::new("_valid_from", DataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(update_ids.to_vec())),
        Arc::new(StringArray::from(
            update_ids
                .iter()
                .map(|i| format!("u{i}"))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            update_ids
                .iter()
                .map(|i| NEW_VF_BASE + *i as i64)
                .collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

async fn setup(
    warehouse: &TempDir,
    update_ids: &[i32],
    options: MorMergeOptions,
) -> (Arc<dyn Catalog>, SessionContext) {
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

    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
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

    let config =
        datafusion::execution::context::SessionConfig::new().with_extension(Arc::new(options));
    let ctx = SessionContext::new_with_config(config);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(Arc::clone(&catalog))
            .await
            .unwrap(),
    );
    ctx.register_catalog(CATALOG, provider);

    let cdc = cdc_batch(update_ids);
    let mem = MemTable::try_new(cdc.schema(), vec![vec![cdc]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();

    (catalog, ctx)
}

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

/// Run the merge; return (reabsorbed_files, reabsorb_skipped).
async fn run_merge(ctx: &SessionContext) -> (u64, Option<String>) {
    let batches = ctx
        .sql(&scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut files = 0u64;
    let mut skipped = None;
    for b in &batches {
        if let Some(col) = b.column_by_name("reabsorbed_files") {
            files += col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .iter()
                .flatten()
                .sum::<u64>();
        }
        if let Some(col) = b.column_by_name("reabsorb_skipped") {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            if arr.len() == 1 && arr.is_valid(0) {
                skipped = Some(arr.value(0).to_string());
            }
        }
    }
    (files, skipped)
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

async fn snapshot_count(catalog: &Arc<dyn Catalog>) -> usize {
    load_table(catalog).await.metadata().snapshots().count()
}

async fn live_dv_count(catalog: &Arc<dyn Catalog>) -> usize {
    let table = load_table(catalog).await;
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
                n += 1;
            }
        }
    }
    n
}

/// Sorted full-row projection for identity comparison across tables.
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

fn opts(reabsorb: Option<f64>) -> MorMergeOptions {
    MorMergeOptions {
        reabsorb_dead_frac: reabsorb,
        ..Default::default()
    }
}

#[tokio::test]
async fn threshold_crossing_merge_reabsorbs_its_dv() {
    // 40/100 rows updated = 40% dead on the seed file, over the 0.3 threshold.
    let update_ids: Vec<i32> = (0..40).collect();

    // Baseline: identical merge with reabsorb OFF.
    let wh_base = TempDir::new().unwrap();
    let (cat_base, ctx_base) = setup(&wh_base, &update_ids, opts(None)).await;
    let (files_base, skipped_base) = run_merge(&ctx_base).await;
    assert_eq!((files_base, skipped_base), (0, None), "OFF reports nothing");
    assert_eq!(snapshot_count(&cat_base).await, 2, "seed + RowDelta");
    assert_eq!(
        live_dv_count(&cat_base).await,
        1,
        "DV stays without reabsorb"
    );
    let baseline = read_state(&ctx_base).await;

    // Reabsorb ON.
    let wh = TempDir::new().unwrap();
    let (catalog, ctx) = setup(&wh, &update_ids, opts(Some(0.3))).await;
    let (files, skipped) = run_merge(&ctx).await;
    assert_eq!(skipped, None, "crossing merge must not skip: {skipped:?}");
    assert_eq!(files, 1, "the seed file crossed the threshold");
    assert_eq!(
        snapshot_count(&catalog).await,
        3,
        "seed + RowDelta + reabsorb Replace"
    );
    assert_eq!(live_dv_count(&catalog).await, 0, "DV reabsorbed");
    assert_eq!(
        read_state(&ctx).await,
        baseline,
        "row identity must match the reabsorb-OFF baseline"
    );
}

#[tokio::test]
async fn below_threshold_merge_keeps_its_dv() {
    // 10/100 = 10% dead, under the 0.3 threshold.
    let update_ids: Vec<i32> = (0..10).collect();
    let wh = TempDir::new().unwrap();
    let (catalog, ctx) = setup(&wh, &update_ids, opts(Some(0.3))).await;
    let (files, skipped) = run_merge(&ctx).await;
    assert_eq!(files, 0);
    assert_eq!(skipped.as_deref(), Some("threshold"));
    assert_eq!(snapshot_count(&catalog).await, 2, "no second snapshot");
    assert_eq!(
        live_dv_count(&catalog).await,
        1,
        "DV retained until it grows"
    );
}

#[tokio::test]
async fn empty_merge_never_reabsorbs() {
    let wh = TempDir::new().unwrap();
    let (catalog, ctx) = setup(&wh, &[], opts(Some(0.3))).await;
    let before = snapshot_count(&catalog).await;
    let (files, skipped) = run_merge(&ctx).await;
    assert_eq!((files, skipped), (0, None));
    assert_eq!(snapshot_count(&catalog).await, before, "+0 snapshots");
}

#[tokio::test]
async fn exhausted_deadline_skips_with_budget_reason() {
    let update_ids: Vec<i32> = (0..40).collect();
    let wh = TempDir::new().unwrap();
    // A deadline far enough away for the merge itself, but with less than
    // the reabsorb floor remaining once it commits.
    let options = MorMergeOptions {
        deadline: Some(std::time::Instant::now() + std::time::Duration::from_secs(45)),
        reabsorb_dead_frac: Some(0.3),
        ..Default::default()
    };
    let (catalog, ctx) = setup(&wh, &update_ids, options).await;
    let (files, skipped) = run_merge(&ctx).await;
    assert_eq!(files, 0);
    assert_eq!(
        skipped.as_deref(),
        Some("budget"),
        "hygiene never spends freshness"
    );
    assert_eq!(
        snapshot_count(&catalog).await,
        2,
        "merge committed, no reabsorb"
    );
    assert_eq!(live_dv_count(&catalog).await, 1);
}
