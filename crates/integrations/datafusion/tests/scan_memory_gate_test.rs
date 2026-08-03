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

//! #18 read-RSS bounding — scan decode-memory accounting pins.
//!
//! - a scan on a CONTENDED pool fails CLEANLY with a catchable engine
//!   error naming the gate (the class that previously died as an
//!   unaccounted kernel OOM with no failure path);
//! - a big file on a small-but-EMPTY pool stays admissible (the clamp:
//!   estimates cap at half the pool, so streaming reads that fit today
//!   keep working — contention, not raw file size, gates admission);
//! - `ICEBERG_SCAN_MEMORY_GATE=0` disables accounting (the kill switch).

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, LargeStringArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use iceberg::spec::{DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, Type};
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
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const CATALOG: &str = "catalog";
const NS: &str = "db";
const TABLE: &str = "t";

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

async fn seed_table(warehouse: &TempDir) -> Arc<dyn Catalog> {
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
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(TABLE.to_string())
                .schema(schema)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "payload", DataType::LargeUtf8, true),
    ]));
    // ~8 MiB of payload so the projected decode estimate comfortably exceeds
    // a 6 MiB pool (and the 4 MiB floor).
    let unit = "x".repeat(8 * 1024);
    let rows = 1024;
    let payloads: Vec<String> = (0..rows).map(|i| format!("{i:08}{unit}")).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from((0..rows).collect::<Vec<_>>())),
        Arc::new(LargeStringArray::from(
            payloads.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        )),
    ])
    .unwrap();

    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            WriterProperties::builder()
                .set_compression(parquet::basic::Compression::UNCOMPRESSED)
                .set_dictionary_enabled(false)
                .build(),
            table.metadata().current_schema().clone(),
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new("seed".to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    let files = writer.close().await.unwrap();
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();
    catalog
}

async fn ctx_with_pool(catalog: Arc<dyn Catalog>, pool_bytes: usize) -> SessionContext {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(pool_bytes));
    let rt = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .build_arc()
        .unwrap();
    let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), rt);
    let provider = Arc::new(IcebergCatalogProvider::try_new(catalog).await.unwrap());
    ctx.register_catalog(CATALOG, provider);
    ctx
}

async fn run_scan(ctx: &SessionContext) -> Result<usize, datafusion::error::DataFusionError> {
    let batches = ctx
        .sql(&format!("SELECT id, payload FROM {CATALOG}.{NS}.{TABLE}"))
        .await?
        .collect()
        .await?;
    Ok(batches.iter().map(|b| b.num_rows()).sum())
}

#[tokio::test]
async fn scan_memory_gate_pins() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    // ONE sequential test: the cases mutate process env (gate kill switch,
    // wait budget), which would race across parallel test threads.
    let wh = TempDir::new().unwrap();
    let catalog = seed_table(&wh).await;

    // 1. Big file, small EMPTY pool: the clamp (limit/2) keeps the ~380MB
    //    estimate admissible on a 6MB pool — streaming reads that fit
    //    today keep working under the gate.
    unsafe { std::env::set_var("ICEBERG_SCAN_GATE_WAIT_S", "2") };
    let ctx = ctx_with_pool(Arc::clone(&catalog), 6 * 1024 * 1024).await;
    assert_eq!(run_scan(&ctx).await.unwrap(), 1024);

    // 2. CONTENDED pool: pre-occupy most of it, the clamped 3MB acquire
    //    cannot fit -> bounded wait -> CLEAN failure naming the gate.
    let pool = ctx.runtime_env().memory_pool.clone();
    let mut squatter = MemoryConsumer::new("squatter").register(&pool);
    squatter.try_grow(5 * 1024 * 1024).unwrap();
    let err = run_scan(&ctx).await.expect_err("scan must fail, not abort");
    let msg = format!("{err}");
    assert!(
        msg.contains("scan memory gate"),
        "error must name the gate: {msg}"
    );

    // 3. Kill switch: reads invisible to the pool -> the same contended
    //    pool passes.
    unsafe { std::env::set_var("ICEBERG_SCAN_MEMORY_GATE", "0") };
    let out = run_scan(&ctx).await;
    unsafe { std::env::remove_var("ICEBERG_SCAN_MEMORY_GATE") };
    unsafe { std::env::remove_var("ICEBERG_SCAN_GATE_WAIT_S") };
    assert_eq!(out.unwrap(), 1024);
    drop(squatter);

    // 4. Adequate pool, gate on, full width: succeeds.
    let ctx = ctx_with_pool(catalog, 256 * 1024 * 1024).await;
    assert_eq!(run_scan(&ctx).await.unwrap(), 1024);
}
