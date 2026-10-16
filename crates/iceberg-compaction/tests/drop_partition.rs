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

//! `drop_partition`: a bronze-shaped V3 table (`_is_backfill` identity partition,
//! `_cdc{op, key}` struct) with two backfill files, a CDC-partition file and a
//! deletion vector on one backfill file. Dropping `_is_backfill = true`:
//!   - removes both backfill data files AND the DV scoped to the partition in
//!     ONE metadata-only snapshot (nothing added);
//!   - leaves the CDC-partition file and its rows untouched;
//!   - a second drop of the now-empty partition commits nothing.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Int32Array, LargeStringArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::delete_vector::DeleteVector;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey,
    PrimitiveType, Schema, Struct, StructType, Transform, Type, UnboundPartitionSpec,
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
use iceberg_compaction::engine::{current_data_files, current_delete_files, drop_partition};
use parquet::file::properties::WriterProperties;
use roaring::RoaringTreemap;
use tempfile::TempDir;
use uuid::Uuid;

fn table_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(
                3,
                "_cdc",
                Type::Struct(StructType::new(vec![
                    NestedField::required(4, "op", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(5, "key", Type::Primitive(PrimitiveType::String)).into(),
                ])),
            )
            .into(),
            NestedField::optional(6, "_is_backfill", Type::Primitive(PrimitiveType::Boolean))
                .into(),
        ])
        .build()
        .unwrap()
}

fn table_arrow_schema(table: &Table) -> Arc<ArrowSchema> {
    Arc::new(iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema()).unwrap())
}

/// `{"$oid": "<prefix><pad>"}` — the mongo bronze key text (the hex run starts at byte 10).
fn oid_key(prefix: &str, n: u32) -> String {
    format!("{{\"$oid\": \"{prefix}{:021x}\"}}", n)
}

/// Rows: (id, payload, op, key, is_backfill), built against the table's schema.
fn batch(table: &Table, rows: &[(i32, &str, &str, Option<String>, Option<bool>)]) -> RecordBatch {
    let schema = table_arrow_schema(table);
    let cdc_fields = match schema.field_with_name("_cdc").unwrap().data_type() {
        DataType::Struct(fields) => fields.clone(),
        other => panic!("_cdc must be a struct, got {other:?}"),
    };
    let ops: Vec<&str> = rows.iter().map(|r| r.2).collect();
    let keys: Vec<Option<&str>> = rows.iter().map(|r| r.3.as_deref()).collect();
    let cdc = StructArray::from(vec![
        (
            cdc_fields[0].clone(),
            Arc::new(LargeStringArray::from(ops)) as arrow_array::ArrayRef,
        ),
        (
            cdc_fields[1].clone(),
            Arc::new(LargeStringArray::from(keys)) as arrow_array::ArrayRef,
        ),
    ]);
    RecordBatch::try_new(schema, vec![
        Arc::new(Int32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>())),
        Arc::new(LargeStringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(cdc),
        Arc::new(BooleanArray::from(rows.iter().map(|r| r.4).collect::<Vec<_>>())),
    ])
    .unwrap()
}

fn partition_key(table: &Table, is_backfill: Option<bool>) -> PartitionKey {
    PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter(vec![is_backfill.map(Literal::bool)]),
    )
}

async fn write_data(
    table: &Table,
    prefix: &str,
    is_backfill: Option<bool>,
    b: RecordBatch,
) -> Vec<DataFile> {
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            table.metadata().current_schema().clone(),
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(
            format!("{prefix}-{}", Uuid::now_v7()),
            None,
            DataFileFormat::Parquet,
        ),
    );
    let mut w = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key(table, is_backfill)))
        .await
        .unwrap();
    w.write(b).await.unwrap();
    w.close().await.unwrap()
}

/// (id, key) of every live row, sorted by id.
async fn read_rows(table: &Table) -> Vec<(i32, Option<String>)> {
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
    let mut out = Vec::new();
    for b in &batches {
        let schema = b.schema();
        let ids = b
            .column(schema.index_of("id").unwrap())
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let cdc = b
            .column(schema.index_of("_cdc").unwrap())
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let keys = cdc
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            out.push((
                ids.value(i),
                keys.is_valid(i).then(|| keys.value(i).to_string()),
            ));
        }
    }
    out.sort();
    out
}


#[tokio::test]
async fn drop_partition_removes_the_partitions_files_and_deletes_only() {
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
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(6, "_is_backfill", Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("bronze_coll".to_string())
                .schema(table_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();
    let ident = TableIdent::new(ns, "bronze_coll".to_string());
    let mk = |id: i32, prefix: &str| (id, "p", "r", Some(oid_key(prefix, id as u32)), Some(true));
    let file_a = write_data(&table, "bf-a", Some(true), batch(&table, &[mk(1, "69c"), mk(2, "69e"), mk(3, "6a1")])).await;
    let file_b = write_data(&table, "bf-b", Some(true), batch(&table, &[mk(4, "69c"), mk(5, "69e")])).await;
    let file_c = write_data(
        &table,
        "cdc",
        None,
        batch(&table, &[
            (12, "p", "c", Some(oid_key("69c", 12)), None),
            (13, "p", "u", Some(oid_key("6a1", 13)), None),
        ]),
    )
    .await;
    let a_path = file_a[0].file_path().to_string();
    let c_path = file_c[0].file_path().to_string();
    let mut seed = file_a;
    seed.extend(file_b);
    seed.extend(file_c);
    let tx = Transaction::new(&table);
    let table = tx.fast_append().add_data_files(seed).apply(tx).unwrap().commit(&catalog).await.unwrap();
    let mut positions = RoaringTreemap::new();
    positions.insert(0);
    let dv_path = format!("{}/dv-a.puffin", warehouse.path().to_str().unwrap());
    let dv_file = DeleteVector::new(positions)
        .write_to_puffin_file(
            table.file_io(),
            dv_path,
            a_path.clone(),
            Struct::from_iter(vec![Some(Literal::bool(true))]),
            0,
        )
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let table = tx.row_delta().add_delete_files(vec![dv_file]).apply(tx).unwrap().commit(&catalog).await.unwrap();
    assert_eq!(read_rows(&table).await.len(), 6, "4 live backfill rows (one DV'd) + 2 CDC rows");
    assert_eq!(current_delete_files(&table).await.unwrap().len(), 1);
    let before = table.metadata().current_snapshot_id();

    let out = drop_partition(&catalog, &ident, "_is_backfill", true).await.unwrap();
    assert_eq!((out.data_files, out.delete_files, out.rows), (2, 1, 5), "both backfill files, their DV, 5 manifest rows");
    assert!(out.bytes > 0);
    assert!(out.snapshot_id.is_some() && out.snapshot_id != before, "one new snapshot");

    let table = catalog.load_table(&ident).await.unwrap();
    let files = current_data_files(&table).await.unwrap();
    assert_eq!(files.len(), 1, "only the CDC-partition file remains");
    assert!(files.contains_key(&c_path));
    assert_eq!(current_delete_files(&table).await.unwrap().len(), 0, "the partition's DV went with its data");
    let rows = read_rows(&table).await;
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<i32>>(), vec![12, 13], "the CDC rows survive untouched");

    // a second drop of the empty partition is a no-op: nothing committed
    let again = drop_partition(&catalog, &ident, "_is_backfill", true).await.unwrap();
    assert_eq!((again.data_files, again.snapshot_id), (0, None));
    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(table.metadata().current_snapshot_id(), out.snapshot_id, "no new snapshot");

    // the wrong partition column is refused
    assert!(drop_partition(&catalog, &ident, "payload", true).await.is_err());
}
