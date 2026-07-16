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

//! Provider-level scan-file allowlists: a table scan reads ONLY an
//! externally planned subset of data files. Consumers that plan the file set
//! themselves (e.g. incremental processing of known appended files) restrict
//! the mounted table via
//! [`IcebergCatalogProvider::with_table_scan_file_allowlist`]; deletes still
//! apply to the retained files; a MERGE whose USING source reads the
//! restricted table only sees the allowlisted files; unknown identifiers
//! fail loudly.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
use iceberg::spec::{DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, Type};
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

fn simple_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "val", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

fn simple_arrow_schema() -> Arc<ArrowSchema> {
    let field = |id: i32, name: &str, dt: DataType, nullable: bool| {
        Field::new(name, dt, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    };
    Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "val", DataType::Utf8, true),
    ]))
}

fn simple_batch(rows: &[(i32, &str)]) -> RecordBatch {
    RecordBatch::try_new(simple_arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

async fn memory_catalog(warehouse: &TempDir) -> Arc<dyn Catalog> {
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
    catalog
        .create_namespace(&NamespaceIdent::new(NS.to_string()), HashMap::new())
        .await
        .unwrap();
    catalog
}

async fn create_table(catalog: &Arc<dyn Catalog>, name: &str) -> Table {
    catalog
        .create_table(
            &NamespaceIdent::new(NS.to_string()),
            TableCreation::builder()
                .name(name.to_string())
                .schema(simple_schema())
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap()
}

/// Append `rows` as ONE data file (distinct `prefix` per call — the default
/// name generator's counter resets per instance) and return its path.
async fn append_one_file(
    catalog: &Arc<dyn Catalog>,
    table_name: &str,
    prefix: &str,
    rows: &[(i32, &str)],
) -> String {
    let ident = TableIdent::new(NamespaceIdent::new(NS.to_string()), table_name.to_string());
    let table = catalog.load_table(&ident).await.unwrap();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            table.metadata().current_schema().clone(),
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(simple_batch(rows)).await.unwrap();
    let data_files = writer.close().await.unwrap();
    assert_eq!(data_files.len(), 1);
    let path = data_files[0].file_path().to_string();
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();
    path
}

async fn ctx_with_provider(
    catalog: &Arc<dyn Catalog>,
    allowlist: Option<(&str, Vec<String>)>,
) -> SessionContext {
    let provider = IcebergCatalogProvider::try_new(Arc::clone(catalog))
        .await
        .unwrap();
    if let Some((table, files)) = allowlist {
        provider
            .with_table_scan_file_allowlist(NS, table, files)
            .unwrap();
    }
    let ctx = SessionContext::new();
    ctx.register_catalog(CATALOG, Arc::new(provider));
    ctx
}

async fn read_ids(ctx: &SessionContext, table: &str) -> Vec<i32> {
    let batches = ctx
        .sql(&format!(
            "SELECT id FROM {CATALOG}.{NS}.{table} ORDER BY id"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        out.extend(ids.iter().flatten());
    }
    out
}

#[tokio::test]
async fn allowlist_restricts_scan_to_named_files() {
    let warehouse = TempDir::new().unwrap();
    let catalog = memory_catalog(&warehouse).await;
    create_table(&catalog, "t").await;
    let file_a = append_one_file(&catalog, "t", "a", &[(1, "one"), (2, "two")]).await;
    let _file_b = append_one_file(&catalog, "t", "b", &[(3, "three"), (4, "four")]).await;

    // Control: unrestricted provider sees both files.
    let ctx = ctx_with_provider(&catalog, None).await;
    assert_eq!(read_ids(&ctx, "t").await, vec![1, 2, 3, 4]);

    // Allowlisted provider sees ONLY file A's rows.
    let ctx = ctx_with_provider(&catalog, Some(("t", vec![file_a]))).await;
    assert_eq!(read_ids(&ctx, "t").await, vec![1, 2]);

    // The restriction is visible in the physical plan display.
    let df = ctx
        .sql(&format!("SELECT id FROM {CATALOG}.{NS}.t"))
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        display.contains("scan_files:[1]"),
        "physical plan should show the allowlist: {display}"
    );
}

#[tokio::test]
async fn merge_using_source_respects_allowlist() {
    let warehouse = TempDir::new().unwrap();
    let catalog = memory_catalog(&warehouse).await;
    create_table(&catalog, "tgt").await;
    create_table(&catalog, "src").await;
    let src_a = append_one_file(&catalog, "src", "a", &[(10, "ten"), (11, "eleven")]).await;
    let _src_b = append_one_file(&catalog, "src", "b", &[(12, "twelve"), (13, "thirteen")]).await;

    let ctx = ctx_with_provider(&catalog, Some(("src", vec![src_a]))).await;
    ctx.sql(&format!(
        "MERGE INTO {CATALOG}.{NS}.tgt AS t \
         USING (SELECT id, val FROM {CATALOG}.{NS}.src) AS s \
         ON t.id = s.id \
         WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // Only the allowlisted source file's rows reached the target.
    assert_eq!(read_ids(&ctx, "tgt").await, vec![10, 11]);
}

#[tokio::test]
async fn deletes_still_apply_to_allowlisted_files() {
    let warehouse = TempDir::new().unwrap();
    let catalog = memory_catalog(&warehouse).await;
    create_table(&catalog, "t").await;
    let file_a = append_one_file(&catalog, "t", "a", &[(1, "one"), (2, "two")]).await;

    // MoR MATCHED-update: writes a deletion vector on file A's row for id=1
    // plus the updated version in a NEW data file.
    let ctx = ctx_with_provider(&catalog, None).await;
    let upd = simple_batch(&[(1, "one-v2")]);
    let mem = MemTable::try_new(upd.schema(), vec![vec![upd]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();
    ctx.sql(&format!(
        "MERGE INTO {CATALOG}.{NS}.t AS t USING (SELECT id, val FROM batch) AS s \
         ON t.id = s.id \
         WHEN MATCHED THEN UPDATE SET val = s.val"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // Allowlist = file A only: id=1's OLD version is DV-deleted (the DV must
    // still bind under the allowlist) and its NEW version lives in an
    // excluded file — so only id=2 remains visible.
    let ctx = ctx_with_provider(&catalog, Some(("t", vec![file_a]))).await;
    assert_eq!(read_ids(&ctx, "t").await, vec![2]);
}

#[tokio::test]
async fn unknown_identifiers_fail_loudly() {
    let warehouse = TempDir::new().unwrap();
    let catalog = memory_catalog(&warehouse).await;
    create_table(&catalog, "t").await;

    let provider = IcebergCatalogProvider::try_new(Arc::clone(&catalog))
        .await
        .unwrap();
    let err = provider
        .with_table_scan_file_allowlist(NS, "nope", vec!["x".to_string()])
        .unwrap_err();
    assert!(err.to_string().contains("nope"), "unexpected: {err}");
    let err = provider
        .with_table_scan_file_allowlist("no_ns", "t", vec!["x".to_string()])
        .unwrap_err();
    assert!(err.to_string().contains("no_ns"), "unexpected: {err}");
}

/// `path@<start>+<length>` allowlist entries clip the scan to a byte range.
/// Two complementary ranges over a multi-row-group file are a DISJOINT,
/// COMPLETE cover (midpoint ownership) — externally-planned sub-file
/// splitting for giant-file bucketing.
#[tokio::test]
async fn byte_range_entries_split_one_file_disjoint_and_complete() {
    let warehouse = TempDir::new().unwrap();
    let catalog = memory_catalog(&warehouse).await;
    let table = create_table(&catalog, "t").await;

    // ONE file, many tiny row groups (max_row_group_size=2 over 10 rows -> 5 RGs)
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            WriterProperties::builder().set_max_row_group_size(2).build(),
            table.metadata().current_schema().clone(),
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new("rg".to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
    // FAT rows (incompressible-ish payload) so data bytes >> footer bytes —
    // else every row-group midpoint lands in the first byte-range half.
    let payloads: Vec<String> = (0..10)
        .map(|i| format!("{i}-").repeat(2000))
        .collect();
    let rows: Vec<(i32, &str)> = payloads
        .iter()
        .enumerate()
        .map(|(i, p)| (i as i32, p.as_str()))
        .collect();
    writer.write(simple_batch(&rows)).await.unwrap();
    let data_files = writer.close().await.unwrap();
    assert_eq!(data_files.len(), 1);
    let path = data_files[0].file_path().to_string();
    let file_len = data_files[0].file_size_in_bytes();
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();

    let mid = file_len / 2;
    let lo = ctx_with_provider(&catalog, Some(("t", vec![format!("{path}@0+{mid}")]))).await;
    let hi = ctx_with_provider(
        &catalog,
        Some(("t", vec![format!("{path}@{mid}+{}", file_len - mid)])),
    )
    .await;
    let all = ctx_with_provider(&catalog, Some(("t", vec![path.clone()]))).await;

    let lo_ids = read_ids(&lo, "t").await;
    let hi_ids = read_ids(&hi, "t").await;
    let all_ids = read_ids(&all, "t").await;

    assert_eq!(all_ids, (0..10).collect::<Vec<_>>());
    assert!(!lo_ids.is_empty() && !hi_ids.is_empty(), "both ranges own rows");
    let mut union = [lo_ids.clone(), hi_ids.clone()].concat();
    union.sort();
    assert_eq!(union, all_ids, "disjoint + complete cover (no dup, no gap)");
}
