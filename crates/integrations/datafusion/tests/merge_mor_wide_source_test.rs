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

//! Memory shape of the SCD2 MERGE **statement** on wide rows.
//!
//! The MoR engine's own decide phase is narrow (target scan = referenced
//! columns + `_file`/`_pos`; matched rows fetched late). But the common
//! application statement threads the FULL-WIDTH batch through its USING
//! subquery: dedup window -> phantom-R / idempotency probes -> chain union
//! -> SCD2 windows. The probes decorrelate into a STACK of hash joins —
//! `LeftMark` (same-batch phantom-R), `LeftMark` (target phantom-R),
//! `LeftAnti` (idempotency) — whose LEFT input is the deduped batch at
//! FULL width. DataFusion's HashJoinExec always builds its LEFT child and,
//! with no statistics on Iceberg scans, never swaps sides and plans
//! `PartitionMode::Partitioned`: every one of those joins BUILDS the wide
//! batch (payload columns included) in a non-spillable per-partition hash
//! table, stacked builds overlap while draining, and the union's window
//! sorts buffer the same wide rows again. On multi-KB rows that is
//! gigabytes — `Resources exhausted: Additional allocation failed for
//! HashJoinInput[..] ... (can spill: false)` — regardless of how narrow
//! the engine's own scan is.
//!
//! The fix is STATEMENT SHAPE (the split-write insight applied to SQL):
//! thread ONLY keys + pointers (pk, `_valid_from`, `_cdc_offset`, cdc meta)
//! through dedup/filters/union/windows, and join the wide payload back at
//! the FINAL projection of the USING source — one LEFT JOIN whose BUILD
//! side is the narrow decided set and whose wide side only streams as the
//! probe. Wide data is then never held by any hash-join build; the only
//! wide buffering left is the payload dedup window, which sorts (and sorts
//! SPILL). `wide_scd2_merge_sql` / `narrow_scd2_merge_sql` below are the
//! reference texts for both shapes; the tests prove the wide shape
//! exhausts a bounded pool at a HashJoin build while the narrow shape
//! completes under the same pool with byte-identical results.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::ExecutionPlan;
use iceberg::spec::{
    DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList, NestedField,
    PartitionKey, PrimitiveType, Schema, Struct as IcebergStruct, Transform, Type,
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
const SILVER: &str = "silver_t";
const BRONZE: &str = "bronze_t";

fn silver_fqn() -> String {
    format!("{CATALOG}.{NS}.{SILVER}")
}
fn bronze_fqn() -> String {
    format!("{CATALOG}.{NS}.{BRONZE}")
}

// ---------------------------------------------------------------------------
// Schemas + writers
// ---------------------------------------------------------------------------

fn silver_iceberg_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "_cdc_op", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(4, "_cdc_ts", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(5, "_cdc_offset", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(6, "_valid_from", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(7, "_valid_to", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(8, "_is_current", Type::Primitive(PrimitiveType::Boolean)).into(),
        ])
        .build()
        .unwrap()
}

fn bronze_iceberg_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "op", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(4, "ts", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(5, "off", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(6, "src", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(7, "uzi", Type::Primitive(PrimitiveType::Long)).into(),
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

fn silver_arrow_schema() -> SchemaRef {
    Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "payload", DataType::Utf8, true),
        field(3, "_cdc_op", DataType::Utf8, false),
        field(4, "_cdc_ts", DataType::Int64, false),
        field(5, "_cdc_offset", DataType::Int64, false),
        field(6, "_valid_from", DataType::Int64, false),
        field(7, "_valid_to", DataType::Int64, true),
        field(8, "_is_current", DataType::Boolean, false),
    ]))
}

fn bronze_arrow_schema() -> SchemaRef {
    Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "payload", DataType::Utf8, true),
        field(3, "op", DataType::Utf8, false),
        field(4, "ts", DataType::Int64, false),
        field(5, "off", DataType::Int64, false),
        field(6, "src", DataType::Utf8, false),
        field(7, "uzi", DataType::Int64, true),
    ]))
}

/// A distinct multi-KB (or normal-width) payload per (id, off).
fn payload(id: i32, off: i64, len: usize) -> String {
    let unit = format!("{id:06}:{off:06}|");
    unit.repeat(len / unit.len() + 1)[..len].to_string()
}

/// Silver seed rows: (id, payload, _valid_from, _valid_to, _is_current, off).
/// `_cdc_op` is 'I', `_cdc_ts` = `_valid_from`.
#[allow(clippy::type_complexity)]
fn silver_batch(rows: &[(i32, String, i64, Option<i64>, bool, i64)]) -> RecordBatch {
    RecordBatch::try_new(silver_arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(vec!["I"; rows.len()])),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.5).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.4).collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

/// Bronze CDC events: (id, payload, op, ts, off, uzi). `src` is constant.
#[allow(clippy::type_complexity)]
fn bronze_batch(rows: &[(i32, String, &str, i64, i64, Option<i64>)]) -> RecordBatch {
    RecordBatch::try_new(bronze_arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.2.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.4).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(vec!["s1"; rows.len()])),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.5).collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

async fn append_file(
    catalog: &Arc<dyn Catalog>,
    table: &Table,
    batch: RecordBatch,
    prefix: &str,
    partition: Option<IcebergStruct>,
) {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let partition_key = partition.map(|p| {
        PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            p,
        )
    });
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(partition_key)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    let files = writer.close().await.unwrap();
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();
}

/// Build silver (partitioned by `_is_current`, seeded current rows) + bronze
/// (unpartitioned, seeded CDC events) in one memory catalog.
#[allow(clippy::type_complexity)]
async fn setup_tables(
    warehouse: &TempDir,
    silver_rows: &[(i32, String, i64, Option<i64>, bool, i64)],
    bronze_rows: &[(i32, String, &str, i64, i64, Option<i64>)],
) -> Arc<dyn Catalog> {
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
        .add_partition_field(8, "_is_current", Transform::Identity)
        .unwrap()
        .build();
    let silver = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(SILVER.to_string())
                .schema(silver_iceberg_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    append_file(
        &catalog,
        &silver,
        silver_batch(silver_rows),
        "seed",
        Some(IcebergStruct::from_iter(vec![Some(Literal::bool(true))])),
    )
    .await;

    let bronze = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(BRONZE.to_string())
                .schema(bronze_iceberg_schema())
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    append_file(&catalog, &bronze, bronze_batch(bronze_rows), "cdc", None).await;

    catalog
}

/// A session over the catalog, optionally memory-bounded (spill to OS temp
/// stays enabled — only NON-spillable reservations can then exhaust it).
async fn session(
    catalog: &Arc<dyn Catalog>,
    pool_mb: Option<usize>,
    target_partitions: usize,
) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_extension(Arc::new(MorMergeOptions::default()));
    let ctx = match pool_mb {
        None => SessionContext::new_with_config(config),
        Some(mb) => {
            // Keep sort spill floors SMALL relative to the pool (production
            // proportions: a multi-GB pool dwarfs the sorts' merge
            // reservations) so the non-spillable hash-join builds — not a
            // sort mid-merge — are what exhausts the pool.
            // ... and keep BATCH granularity proportional to the pool the
            // way production batches are to its multi-GB pool (a 8192-row
            // coalesced batch of multi-KB rows is a double-digit-MB single
            // allocation — against a sub-GB test pool every allocator's
            // failure point becomes batch-size noise).
            let config = config
                .with_sort_spill_reservation_bytes(2 * 1024 * 1024)
                .with_batch_size(512);
            let rt = RuntimeEnvBuilder::new()
                .with_memory_limit(mb * 1024 * 1024, 1.0)
                .build_arc()
                .unwrap();
            SessionContext::new_with_config_rt(config, rt)
        }
    };
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(Arc::clone(catalog))
            .await
            .unwrap(),
    );
    ctx.register_catalog(CATALOG, provider);
    ctx
}

// ---------------------------------------------------------------------------
// The two statement shapes
// ---------------------------------------------------------------------------

/// The CURRENT production-shaped SCD2 merge statement: the full-width batch
/// (payload included) is threaded through dedup, the phantom-R / idempotency
/// probes and the chain union. Every anti/mark join in `b_filtered` puts the
/// WIDE batch on its build side.
fn wide_scd2_merge_sql() -> String {
    let s = silver_fqn();
    let b = bronze_fqn();
    format!(
        "MERGE INTO {s} AS t\n\
         USING (\n\
           WITH b_flat AS (\n\
             SELECT id, payload,\n\
                    op AS _cdc_op_raw, ts AS _cdc_ts, off AS _cdc_offset, src AS _cdc_source,\n\
                    COALESCE(uzi, ts) AS _valid_from\n\
             FROM {b}\n\
           ),\n\
           b_dedup AS (\n\
             SELECT id, payload, _valid_from,\n\
                    CASE UPPER(_cdc_op_raw) WHEN 'C' THEN 'I'\n\
                         ELSE UPPER(_cdc_op_raw) END AS _cdc_op,\n\
                    _cdc_ts, _cdc_offset\n\
             FROM (\n\
               SELECT *, ROW_NUMBER() OVER (\n\
                   PARTITION BY id, _cdc_source, _cdc_offset\n\
                   ORDER BY _cdc_ts DESC) AS _rn\n\
               FROM b_flat\n\
             ) WHERE _rn = 1\n\
           ),\n\
           b_bounds AS (\n\
             SELECT MIN(_valid_from) AS lo, MAX(_valid_from) AS hi FROM b_dedup\n\
           ),\n\
           b_filtered AS (\n\
             SELECT v.* FROM b_dedup v\n\
             WHERE NOT (\n\
                 v._cdc_op = 'R'\n\
                 AND (\n\
                   EXISTS (SELECT 1 FROM b_dedup x\n\
                            WHERE x.id = v.id\n\
                              AND x._valid_from = v._valid_from\n\
                              AND x._cdc_op != 'R')\n\
                   OR EXISTS (SELECT 1 FROM {s} sp\n\
                               WHERE sp.id = v.id\n\
                                 AND sp._valid_from = v._valid_from\n\
                                 AND sp._valid_from BETWEEN (SELECT lo FROM b_bounds)\n\
                                                        AND (SELECT hi FROM b_bounds)\n\
                                 AND sp._is_current AND sp._cdc_op != 'R')))\n\
               AND NOT EXISTS (\n\
                 SELECT 1 FROM {s} si\n\
                 WHERE si.id = v.id\n\
                   AND si._valid_from = v._valid_from\n\
                   AND si._valid_from BETWEEN (SELECT lo FROM b_bounds)\n\
                                          AND (SELECT hi FROM b_bounds)\n\
                   AND si._cdc_offset = v._cdc_offset)\n\
           ),\n\
           unioned AS (\n\
             SELECT id, payload, _cdc_op, _cdc_ts, _cdc_offset, _valid_from\n\
             FROM b_filtered\n\
             UNION ALL\n\
             SELECT t2.id, CAST(NULL AS VARCHAR) AS payload,\n\
                    t2._cdc_op, t2._cdc_ts, t2._cdc_offset, t2._valid_from\n\
             FROM (SELECT DISTINCT id FROM b_filtered) k\n\
             JOIN {s} t2 ON t2.id = k.id\n\
             WHERE t2._is_current\n\
           )\n\
           SELECT *,\n\
                  LEAD(_valid_from) OVER (PARTITION BY id\n\
                      ORDER BY _valid_from ASC, _cdc_offset ASC)  AS _new_valid_to,\n\
                  (ROW_NUMBER() OVER (PARTITION BY id\n\
                      ORDER BY _valid_from DESC, _cdc_offset DESC) = 1) AS _new_is_current\n\
           FROM unioned\n\
         ) AS s\n\
         ON t.id = s.id AND t._valid_from = s._valid_from\n\
            AND t._cdc_offset = s._cdc_offset\n\
            AND t._is_current\n\
         WHEN MATCHED THEN UPDATE SET\n\
             _valid_to = s._new_valid_to, _is_current = s._new_is_current\n\
         WHEN NOT MATCHED THEN INSERT\n\
             (id, payload, _cdc_op, _cdc_ts, _cdc_offset, _valid_from, _valid_to, _is_current)\n\
         VALUES (s.id, s.payload, s._cdc_op, s._cdc_ts, s._cdc_offset, s._valid_from,\n\
                 s._new_valid_to, s._new_is_current)"
    )
}

/// The NARROW-shape rewrite (the split-write insight applied to the
/// statement): keys + pointers through dedup/filters/union/windows; the wide
/// payload is deduped independently (window over a sort — SPILLS under
/// memory pressure, unlike a hash-join build) and joined back ONLY at the
/// final projection, with the narrow decided set as the join's BUILD side.
///
/// Semantics notes (why this is byte-identical):
/// - The join-back key `(id, _valid_from, _cdc_offset)` is the statement's
///   own row identity (the MERGE ON key). `b_wide` dedups on the same
///   `(id, _cdc_source, _cdc_offset)` window as `b_dedup`, so redelivered
///   duplicates (identical events) collapse to one payload row.
/// - Demote rows may or may not find a payload row; their payload columns
///   are never read (they always MATCH, and the UPDATE SET only assigns
///   `_valid_to`/`_is_current` — the engine late-fetches the full old row).
/// - Batch rows always find exactly one payload row (same source row).
fn narrow_scd2_merge_sql() -> String {
    let s = silver_fqn();
    let b = bronze_fqn();
    format!(
        "MERGE INTO {s} AS t\n\
         USING (\n\
           WITH b_keys AS (\n\
             SELECT id,\n\
                    op AS _cdc_op_raw, ts AS _cdc_ts, off AS _cdc_offset, src AS _cdc_source,\n\
                    COALESCE(uzi, ts) AS _valid_from\n\
             FROM {b}\n\
           ),\n\
           b_dedup AS (\n\
             SELECT id, _valid_from,\n\
                    CASE UPPER(_cdc_op_raw) WHEN 'C' THEN 'I'\n\
                         ELSE UPPER(_cdc_op_raw) END AS _cdc_op,\n\
                    _cdc_ts, _cdc_offset\n\
             FROM (\n\
               SELECT *, ROW_NUMBER() OVER (\n\
                   PARTITION BY id, _cdc_source, _cdc_offset\n\
                   ORDER BY _cdc_ts DESC) AS _rn\n\
               FROM b_keys\n\
             ) WHERE _rn = 1\n\
           ),\n\
           b_bounds AS (\n\
             SELECT MIN(_valid_from) AS lo, MAX(_valid_from) AS hi FROM b_dedup\n\
           ),\n\
           b_filtered AS (\n\
             SELECT v.* FROM b_dedup v\n\
             WHERE NOT (\n\
                 v._cdc_op = 'R'\n\
                 AND (\n\
                   EXISTS (SELECT 1 FROM b_dedup x\n\
                            WHERE x.id = v.id\n\
                              AND x._valid_from = v._valid_from\n\
                              AND x._cdc_op != 'R')\n\
                   OR EXISTS (SELECT 1 FROM {s} sp\n\
                               WHERE sp.id = v.id\n\
                                 AND sp._valid_from = v._valid_from\n\
                                 AND sp._valid_from BETWEEN (SELECT lo FROM b_bounds)\n\
                                                        AND (SELECT hi FROM b_bounds)\n\
                                 AND sp._is_current AND sp._cdc_op != 'R')))\n\
               AND NOT EXISTS (\n\
                 SELECT 1 FROM {s} si\n\
                 WHERE si.id = v.id\n\
                   AND si._valid_from = v._valid_from\n\
                   AND si._valid_from BETWEEN (SELECT lo FROM b_bounds)\n\
                                          AND (SELECT hi FROM b_bounds)\n\
                   AND si._cdc_offset = v._cdc_offset)\n\
           ),\n\
           unioned AS (\n\
             SELECT id, _cdc_op, _cdc_ts, _cdc_offset, _valid_from\n\
             FROM b_filtered\n\
             UNION ALL\n\
             SELECT t2.id, t2._cdc_op, t2._cdc_ts, t2._cdc_offset, t2._valid_from\n\
             FROM (SELECT DISTINCT id FROM b_filtered) k\n\
             JOIN {s} t2 ON t2.id = k.id\n\
             WHERE t2._is_current\n\
           ),\n\
           chained AS (\n\
             SELECT *,\n\
                    LEAD(_valid_from) OVER (PARTITION BY id\n\
                        ORDER BY _valid_from ASC, _cdc_offset ASC)  AS _new_valid_to,\n\
                    (ROW_NUMBER() OVER (PARTITION BY id\n\
                        ORDER BY _valid_from DESC, _cdc_offset DESC) = 1) AS _new_is_current\n\
             FROM unioned\n\
           ),\n\
           b_wide AS (\n\
             SELECT id, payload, _cdc_offset, _valid_from\n\
             FROM (\n\
               SELECT id, payload, off AS _cdc_offset,\n\
                      COALESCE(uzi, ts) AS _valid_from,\n\
                      ROW_NUMBER() OVER (\n\
                          PARTITION BY id, src, off\n\
                          ORDER BY ts DESC) AS _rn\n\
               FROM {b}\n\
             ) WHERE _rn = 1\n\
           )\n\
           SELECT c.id, w.payload, c._cdc_op, c._cdc_ts, c._cdc_offset, c._valid_from,\n\
                  c._new_valid_to, c._new_is_current\n\
           FROM chained c\n\
           LEFT JOIN b_wide w\n\
             ON w.id = c.id AND w._valid_from = c._valid_from\n\
                AND w._cdc_offset = c._cdc_offset\n\
         ) AS s\n\
         ON t.id = s.id AND t._valid_from = s._valid_from\n\
            AND t._cdc_offset = s._cdc_offset\n\
            AND t._is_current\n\
         WHEN MATCHED THEN UPDATE SET\n\
             _valid_to = s._new_valid_to, _is_current = s._new_is_current\n\
         WHEN NOT MATCHED THEN INSERT\n\
             (id, payload, _cdc_op, _cdc_ts, _cdc_offset, _valid_from, _valid_to, _is_current)\n\
         VALUES (s.id, s.payload, s._cdc_op, s._cdc_ts, s._cdc_offset, s._valid_from,\n\
                 s._new_valid_to, s._new_is_current)"
    )
}

// ---------------------------------------------------------------------------
// Introspection helpers
// ---------------------------------------------------------------------------

/// Collect every HashJoinExec in the plan whose BUILD side (left child)
/// schema contains `col` — the joins that hold that column in a
/// non-spillable hash table. Returns "join_type/on_len" descriptors.
fn hash_joins_building_column(plan: &Arc<dyn ExecutionPlan>, col: &str, out: &mut Vec<String>) {
    if plan.name() == "HashJoinExec" {
        let build = plan.children()[0];
        if build.schema().fields().iter().any(|f| f.name() == col) {
            out.push(format!(
                "HashJoinExec(build_cols={})",
                build
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }
    for child in plan.children() {
        hash_joins_building_column(child, col, out);
    }
}

async fn read_state(ctx: &SessionContext) -> Vec<(i32, Option<String>, i64, Option<i64>, bool)> {
    let batches = ctx
        .sql(&format!(
            "SELECT id, payload, _valid_from, _valid_to, _is_current \
             FROM {} ORDER BY id, _valid_from, _cdc_offset",
            silver_fqn()
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let payload = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let vf = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let vt = b.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let cur = b.column(4).as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((
                id.value(i),
                payload.is_valid(i).then(|| payload.value(i).to_string()),
                vf.value(i),
                vt.is_valid(i).then(|| vt.value(i)),
                cur.value(i),
            ));
        }
    }
    out
}

async fn load_silver(catalog: &Arc<dyn Catalog>) -> Table {
    catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new(NS.to_string()),
            SILVER.to_string(),
        ))
        .await
        .unwrap()
}

/// Sorted live-DV cardinalities on the silver table.
async fn dv_cardinalities(table: &Table) -> Vec<u64> {
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
                out.push(e.data_file().record_count());
            }
        }
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// The scenario data
// ---------------------------------------------------------------------------

/// Silver: `n_seed` current rows. Bronze: one 'U' per seeded id in
/// `1..=n_updates` (demote + re-insert) and one 'I' per new id — every event
/// with a `width`-byte payload.
#[allow(clippy::type_complexity)]
fn scenario(
    n_seed: i32,
    n_updates: i32,
    n_inserts: i32,
    width: usize,
) -> (
    Vec<(i32, String, i64, Option<i64>, bool, i64)>,
    Vec<(i32, String, &'static str, i64, i64, Option<i64>)>,
) {
    let silver: Vec<_> = (1..=n_seed)
        .map(|id| {
            (
                id,
                payload(id, id as i64, width),
                1_000 + id as i64,
                None,
                true,
                id as i64,
            )
        })
        .collect();
    let mut bronze = Vec::new();
    let mut off = 100_000i64;
    for id in 1..=n_updates {
        off += 1;
        bronze.push((
            id,
            payload(id, off, width),
            "U",
            off,
            off,
            Some(10_000 + id as i64),
        ));
    }
    for id in (n_seed + 1)..=(n_seed + n_inserts) {
        off += 1;
        bronze.push((
            id,
            payload(id, off, width),
            "I",
            off,
            off,
            Some(10_000 + id as i64),
        ));
    }
    (silver, bronze)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The memory regression: on a wide-row table under a bounded pool, the
/// CURRENT statement shape exhausts the pool at a HashJoin build (the wide
/// batch on the build side of the phantom-R / idempotency joins — the
/// production `Resources exhausted ... HashJoinInput[..]` signature), while
/// the NARROW shape completes the SAME merge under the SAME pool.
/// Plan-shape assertions pin the cause: the wide plan holds the payload
/// column in hash-join build sides; the narrow plan holds it in NONE.
#[tokio::test]
async fn wide_statement_exhausts_pool_narrow_statement_survives() {
    const WIDTH: usize = 8_192;
    const POOL_MB: usize = 320;

    let warehouse = TempDir::new().unwrap();
    let (silver_rows, bronze_rows) = scenario(2_000, 2_000, 22_000, WIDTH);
    let catalog = setup_tables(&warehouse, &silver_rows, &bronze_rows).await;

    // Plan shape: the wide statement builds the payload in hash joins.
    let ctx = session(&catalog, None, 6).await;
    let wide_plan = ctx
        .sql(&wide_scd2_merge_sql())
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut wide_builds = Vec::new();
    hash_joins_building_column(&wide_plan, "payload", &mut wide_builds);
    assert!(
        !wide_builds.is_empty(),
        "the current statement shape must expose >=1 HashJoin building the wide payload"
    );
    let narrow_plan = ctx
        .sql(&narrow_scd2_merge_sql())
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut narrow_builds = Vec::new();
    hash_joins_building_column(&narrow_plan, "payload", &mut narrow_builds);
    assert!(
        narrow_builds.is_empty(),
        "the narrow shape must build NO hash join over the payload: {narrow_builds:?}"
    );

    // Execution under a bounded pool: wide shape dies at a hash-join build.
    let ctx = session(&catalog, Some(POOL_MB), 6).await;
    let err = match ctx
        .sql(&wide_scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
    {
        Ok(_) => panic!("wide statement must exhaust the {POOL_MB}MB pool"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("Resources exhausted"),
        "expected pool exhaustion, got: {err}"
    );
    assert!(
        err.contains("HashJoinInput"),
        "exhaustion must come from a hash-join build (the non-spillable \
         reservation), got: {err}"
    );

    // A failed merge commits nothing.
    let table = load_silver(&catalog).await;
    assert_eq!(table.metadata().snapshots().count(), 1);

    // Same tables, same pool, narrow shape: completes.
    let ctx = session(&catalog, Some(POOL_MB), 6).await;
    ctx.sql(&narrow_scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let table = load_silver(&catalog).await;
    assert_eq!(
        table.metadata().snapshots().count(),
        2,
        "narrow merge commits exactly one snapshot"
    );

    // Sanity: every seeded id was demoted + re-inserted, every new id
    // inserted (2000 demoted history rows + 24000 current rows).
    let state = read_state(&ctx).await;
    assert_eq!(state.len(), 2_000 + 24_000);
    assert_eq!(state.iter().filter(|r| r.4).count(), 24_000);
    let dvs = dv_cardinalities(&table).await;
    assert_eq!(dvs, vec![2_000], "one DV covering the demoted seed rows");
}

/// Byte-equality of the two statement shapes on a normal-width matrix:
/// updates (demote + re-insert), brand-new inserts, redelivered duplicates
/// (same id/src/off twice), a phantom-R colliding with a same-batch insert,
/// a phantom-R colliding with the target's current row, a surviving op=R,
/// and an idempotent replay (+0 snapshots) — run through BOTH shapes on
/// twin warehouses, asserting identical state, DV cardinalities and
/// snapshot counts at every step.
#[tokio::test]
async fn narrow_and_wide_statements_commit_identical_results() {
    const WIDTH: usize = 64;

    #[allow(clippy::type_complexity)]
    fn matrix() -> (
        Vec<(i32, String, i64, Option<i64>, bool, i64)>,
        Vec<(i32, String, &'static str, i64, i64, Option<i64>)>,
    ) {
        let (silver, mut bronze) = scenario(20, 6, 4, WIDTH);
        // Redelivered duplicate of the id=1 update (same id/src/off — must
        // collapse to ONE inserted row in both shapes).
        let dup = bronze[0].clone();
        bronze.push(dup);
        // Same-batch phantom-R: an 'R' sharing (id, _valid_from) with a
        // non-R event -> dropped by the EXISTS x probe.
        bronze.push((
            31,
            payload(31, 900, WIDTH),
            "I",
            200_001,
            200_001,
            Some(50_000),
        ));
        bronze.push((
            31,
            payload(31, 901, WIDTH),
            "R",
            200_002,
            200_002,
            Some(50_000),
        ));
        // Target phantom-R: an 'R' matching the CURRENT silver row of id=7
        // (same _valid_from = 1007, inside the batch bounds) -> dropped by
        // the EXISTS sp probe.
        bronze.push((
            7,
            payload(7, 902, WIDTH),
            "R",
            200_003,
            200_003,
            Some(1_007),
        ));
        // Surviving op=R (no collision): inserts as an R row.
        bronze.push((
            32,
            payload(32, 903, WIDTH),
            "R",
            200_004,
            200_004,
            Some(60_000),
        ));
        // Keep the batch bounds wide enough to cover the seed vf 1007.
        bronze.push((
            33,
            payload(33, 904, WIDTH),
            "I",
            200_005,
            200_005,
            Some(1_001),
        ));
        (silver, bronze)
    }

    async fn run(
        sql: &str,
    ) -> (
        Vec<(i32, Option<String>, i64, Option<i64>, bool)>,
        Vec<u64>,
        usize,
    ) {
        let warehouse = TempDir::new().unwrap();
        let (silver_rows, bronze_rows) = matrix();
        let catalog = setup_tables(&warehouse, &silver_rows, &bronze_rows).await;
        let ctx = session(&catalog, None, 6).await;
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let snaps_after_merge = load_silver(&catalog).await.metadata().snapshots().count();

        // Idempotent replay: every surviving batch row now exists in the
        // target with the same (id, _valid_from, _cdc_offset) -> the
        // idempotency probe empties the source -> +0 snapshots.
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let table = load_silver(&catalog).await;
        assert_eq!(
            table.metadata().snapshots().count(),
            snaps_after_merge,
            "replay must commit nothing"
        );

        let state = read_state(&ctx).await;
        let dvs = dv_cardinalities(&table).await;
        (state, dvs, snaps_after_merge)
    }

    let (state_wide, dvs_wide, snaps_wide) = run(&wide_scd2_merge_sql()).await;
    let (state_narrow, dvs_narrow, snaps_narrow) = run(&narrow_scd2_merge_sql()).await;

    assert_eq!(
        state_narrow, state_wide,
        "statement shapes must be byte-identical"
    );
    assert_eq!(dvs_narrow, dvs_wide, "DV cardinalities must match");
    assert_eq!(snaps_narrow, snaps_wide, "snapshot shape must match");

    // And against expected truth, not merely each other:
    // 6 updates demote 6 seed rows -> DV of 6; every op event inserts once.
    assert_eq!(dvs_wide, vec![6]);
    assert_eq!(snaps_wide, 2);
    // 20 seed rows (6 now historical) + 6 update versions + 4 inserts
    // + id31 'I' + id32 'R' + id33 'I' = 33 rows; dup + phantom-Rs add none.
    let state = &state_wide;
    assert_eq!(state.len(), 33);
    assert_eq!(
        state.iter().filter(|r| r.4).count(),
        27,
        "one current per live id"
    );
    // The duplicate id=1 update produced exactly ONE current row for id=1.
    assert_eq!(
        state.iter().filter(|r| r.0 == 1 && r.4).count(),
        1,
        "redelivered duplicate must collapse"
    );
    // id=7's phantom-R was dropped: its seed row is untouched and current.
    assert_eq!(
        state.iter().filter(|r| r.0 == 7).count(),
        1,
        "target phantom-R must not touch id=7"
    );
    assert!(state.iter().any(|r| r.0 == 7 && r.4 && r.3.is_none()));
    // id=31 kept the 'I', dropped the same-batch 'R'.
    assert_eq!(state.iter().filter(|r| r.0 == 31).count(), 1);
    // id=32's surviving 'R' inserted.
    assert_eq!(state.iter().filter(|r| r.0 == 32).count(), 1);
}
