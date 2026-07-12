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
    Array, ArrayRef, BinaryArray, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
    StructArray,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use iceberg::spec::{
    DataContentType, DataFileFormat, FormatVersion, Literal, ManifestContentType, ManifestList,
    NestedField, PartitionKey, PrimitiveType, Schema, Struct as IcebergStruct, Transform, Type,
    UnboundPartitionSpec, VariantType,
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
use iceberg_datafusion::functions::register_variant_functions;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, LogicalType};
use parquet::file::properties::WriterProperties;
use parquet::variant::VariantBuilder;
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
            NestedField::required(5, "_is_current", Type::Primitive(PrimitiveType::Boolean)).into(),
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
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
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
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.5).collect::<Vec<_>>(),
        )),
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
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.to_string()).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
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
    // Every seed row is `_is_current = true` — one identity partition.
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        IcebergStruct::from_iter(vec![Some(Literal::bool(true))]),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key))
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
    setup_with_props(warehouse, seed, batch, HashMap::new()).await
}

async fn setup_with_props(
    warehouse: &TempDir,
    seed: &[(i32, &str, i64, Option<i64>, bool, i64)],
    batch: &[(i32, &str, i64, i64)],
    props: HashMap<String, String>,
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
    // Partitioned like real silver: current/history physical separation.
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(5, "_is_current", Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(TABLE.to_string())
                .schema(scd2_iceberg_schema())
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .properties(props)
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
                assert_eq!(
                    e.data_file().content_type(),
                    DataContentType::PositionDeletes
                );
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

    ctx.sql(&scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

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
    ctx.sql(&scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Merge #2: new versions for id=2 (current row in the SEED file, which
    // already carries a DV) and id=1 (current row in the file appended by
    // merge #1). Replace the batch table content.
    ctx.deregister_table("batch").unwrap();
    let cdc = cdc_batch(&[(2, "b2", 30, 300), (1, "a3", 30, 301)]);
    let mem = MemTable::try_new(cdc.schema(), vec![vec![cdc]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();

    ctx.sql(&scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

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

/// Live data-file paths in the current snapshot.
async fn live_data_paths(table: &Table) -> Vec<String> {
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
        if mf.content != ManifestContentType::Data {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                out.push(e.data_file().file_path().to_string());
            }
        }
    }
    out.sort();
    out
}

/// A merge over a VARIANT column: the written files must carry the parquet
/// VARIANT logical-type annotation and the table's compression codec
/// (defaulting to zstd), and null semantics must be byte-faithful — an SQL
/// NULL stays a null group, a variant-null value keeps its
/// `{metadata: 0x11 0x00 0x00, value: 0x00}` encoding, and untouched variant
/// bytes round-trip verbatim through the late-materialized demote path.
#[tokio::test]
async fn merge_variant_output_is_annotated_compressed_and_null_faithful() {
    let warehouse = TempDir::new().unwrap();
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
            NestedField::optional(2, "doc", Type::Variant(VariantType)).into(),
            NestedField::required(3, "_valid_from", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(4, "_valid_to", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(5, "_is_current", Type::Primitive(PrimitiveType::Boolean)).into(),
            NestedField::required(6, "_cdc_offset", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(5, "_is_current", Transform::Identity)
        .unwrap()
        .build();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(TABLE.to_string())
                .schema(schema)
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    // Canonical variant docs.
    let doc = |a: i64| -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        let mut obj = builder.new_object();
        obj.insert("a", a);
        obj.finish();
        builder.finish()
    };
    let (m1, v1) = doc(1);
    let (m2, v2) = doc(2);
    // The spec encoding of a VARIANT NULL VALUE (not an SQL null): empty
    // metadata dictionary + the null primitive.
    let variant_null: (Vec<u8>, Vec<u8>) = (vec![0x11, 0x00, 0x00], vec![0x00]);

    let canonical_fields = Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ]);
    let doc_array = |rows: &[Option<(Vec<u8>, Vec<u8>)>]| -> ArrayRef {
        let metas = BinaryArray::from_iter_values(
            rows.iter()
                .map(|v| v.as_ref().map(|(m, _)| m.clone()).unwrap_or_default()),
        );
        let vals = BinaryArray::from_iter_values(
            rows.iter()
                .map(|v| v.as_ref().map(|(_, x)| x.clone()).unwrap_or_default()),
        );
        let validity = NullBuffer::from(rows.iter().map(|v| v.is_some()).collect::<Vec<_>>());
        Arc::new(StructArray::new(
            canonical_fields.clone(),
            vec![Arc::new(metas) as ArrayRef, Arc::new(vals) as ArrayRef],
            Some(validity),
        ))
    };

    // Seed: ids 1..=2, one current row each, both with real docs.
    let seed_schema = Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "doc", DataType::Struct(canonical_fields.clone()), true),
        field(3, "_valid_from", DataType::Int64, false),
        field(4, "_valid_to", DataType::Int64, true),
        field(5, "_is_current", DataType::Boolean, false),
        field(6, "_cdc_offset", DataType::Int64, false),
    ]));
    let seed = RecordBatch::try_new(seed_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])),
        doc_array(&[Some((m1.clone(), v1.clone())), Some(doc(9))]),
        Arc::new(Int64Array::from(vec![10, 10])),
        Arc::new(Int64Array::from(vec![None::<i64>, None])),
        Arc::new(BooleanArray::from(vec![true, true])),
        Arc::new(Int64Array::from(vec![100, 101])),
    ])
    .unwrap();
    let data_files = write_one_data_file(&table, seed).await;
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

    // CDC batch: a new version of id=1 (real doc), a new id=5 whose doc is a
    // VARIANT NULL value, and a new id=6 whose doc is SQL NULL.
    let batch_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("doc", DataType::Struct(canonical_fields.clone()), true),
        Field::new("_valid_from", DataType::Int64, false),
        Field::new("_cdc_offset", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(batch_schema, vec![
        Arc::new(Int32Array::from(vec![1, 5, 6])),
        doc_array(&[
            Some((m2.clone(), v2.clone())),
            Some(variant_null.clone()),
            None,
        ]),
        Arc::new(Int64Array::from(vec![20, 20, 20])),
        Arc::new(Int64Array::from(vec![200, 201, 202])),
    ])
    .unwrap();
    let mem = MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();

    let sql = format!(
        "MERGE INTO {CATALOG}.{NS}.{TABLE} AS t USING ( \
             SELECT id, doc, _valid_from, CAST(NULL AS BIGINT) AS _valid_to, \
                    true AS _is_current, _cdc_offset \
             FROM batch \
             UNION ALL \
             SELECT t2.id, t2.doc, t2._valid_from, b.new_vf AS _valid_to, \
                    false AS _is_current, t2._cdc_offset \
             FROM {CATALOG}.{NS}.{TABLE} t2 \
             JOIN (SELECT id, MIN(_valid_from) AS new_vf FROM batch GROUP BY id) b \
               ON t2.id = b.id AND t2._is_current \
         ) AS s \
         ON t.id = s.id AND t._valid_from = s._valid_from AND t._cdc_offset = s._cdc_offset \
         WHEN MATCHED THEN UPDATE SET _valid_to = s._valid_to, _is_current = s._is_current \
         WHEN NOT MATCHED THEN INSERT (id, doc, _valid_from, _valid_to, _is_current, _cdc_offset) \
             VALUES (s.id, s.doc, s._valid_from, s._valid_to, s._is_current, s._cdc_offset)"
    );
    ctx.sql(&sql).await.unwrap().collect().await.unwrap();

    // The read path must treat the variant-null VALUE as NOT NULL and the
    // SQL-null slot as NULL.
    let batches = ctx
        .sql(&format!(
            "SELECT id FROM {CATALOG}.{NS}.{TABLE} WHERE doc IS NULL"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let null_ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..b.num_rows()).map(|i| ids.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(null_ids, vec![6], "only the SQL-null doc row is NULL");

    // Byte-level checks on every merge-written file: VARIANT annotation,
    // compression, and per-row doc encoding.
    let table = load_table(&catalog).await;
    let merge_files: Vec<String> = live_data_paths(&table)
        .await
        .into_iter()
        .filter(|p| p.contains("merge-"))
        .collect();
    assert!(!merge_files.is_empty(), "merge appended data files");

    let mut seen: HashMap<i32, (bool, Vec<u8>, Vec<u8>)> = HashMap::new();
    for path in &merge_files {
        let bytes = table
            .file_io()
            .new_input(path)
            .unwrap()
            .read()
            .await
            .unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();

        // Finding-class regressions: the annotation + the codec.
        let doc_field = reader
            .metadata()
            .file_metadata()
            .schema()
            .get_fields()
            .iter()
            .find(|f| f.name() == "doc")
            .expect("doc column in file schema")
            .clone();
        assert!(
            matches!(
                doc_field.get_basic_info().logical_type_ref(),
                Some(LogicalType::Variant { .. })
            ),
            "merge output must annotate variant columns: {path}"
        );
        assert_eq!(
            reader.metadata().row_group(0).column(0).compression(),
            Compression::ZSTD(Default::default()),
            "merge output must honor the table's compression codec: {path}"
        );

        for batch in reader.build().unwrap() {
            let batch = batch.unwrap();
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone();
            let docs = batch
                .column_by_name("doc")
                .unwrap()
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone();
            let metas = docs
                .column_by_name("metadata")
                .unwrap()
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .clone();
            let vals = docs
                .column_by_name("value")
                .unwrap()
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .clone();
            for i in 0..batch.num_rows() {
                seen.insert(
                    ids.value(i),
                    (
                        docs.is_null(i),
                        metas.value(i).to_vec(),
                        vals.value(i).to_vec(),
                    ),
                );
            }
        }
    }

    // id=1's new version: the batch doc verbatim. id=1's demoted version came
    // through the late fetch — same id key, the LAST write wins in `seen`, so
    // assert via the distinct doc bytes instead: both versions carry m1/v1 or
    // m2/v2, never a corrupted mix. id=5: the variant-null value, byte-exact,
    // NOT an SQL null. id=6: SQL null.
    let (null5, meta5, val5) = seen.get(&5).expect("id=5 written");
    assert!(!null5, "variant-null is a VALUE, not an SQL null");
    assert_eq!(
        (meta5.as_slice(), val5.as_slice()),
        (variant_null.0.as_slice(), variant_null.1.as_slice())
    );
    let (null6, _, _) = seen.get(&6).expect("id=6 written");
    assert!(null6, "SQL-null doc stays a null group");
    let (null1, meta1, val1) = seen.get(&1).expect("id=1 written");
    assert!(!null1);
    assert!(
        (meta1 == &m2 && val1 == &v2) || (meta1 == &m1 && val1 == &v1),
        "id=1 doc bytes round-trip verbatim"
    );
}

/// An insert-heavy merge (the whole NOT MATCHED set arrives as ONE join
/// batch) against a tiny file-size target must ROLL the output into multiple
/// data files — and still commit exactly one snapshot.
#[tokio::test]
async fn insert_heavy_merge_rolls_output_files() {
    let warehouse = TempDir::new().unwrap();
    let rows: Vec<(i32, String, i64, i64)> = (10..20_010)
        .map(|i| (i, format!("v{i}"), 20i64, 1_000 + i as i64))
        .collect();
    let batch: Vec<(i32, &str, i64, i64)> = rows
        .iter()
        .map(|(i, v, vf, off)| (*i, v.as_str(), *vf, *off))
        .collect();
    let (catalog, ctx) = setup_with_props(
        &warehouse,
        &[(1, "a", 10, None, true, 100)],
        &batch,
        HashMap::from([(
            "write.target-file-size-bytes".to_string(),
            "8192".to_string(),
        )]),
    )
    .await;

    let before = load_table(&catalog).await;
    let snaps_before = before.metadata().snapshots().count();

    ctx.sql(&scd2_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let table = load_table(&catalog).await;
    assert_eq!(
        table.metadata().snapshots().count(),
        snaps_before + 1,
        "chunked writes still commit ONE atomic snapshot"
    );

    let count = ctx
        .sql(&format!("SELECT count(*) FROM {CATALOG}.{NS}.{TABLE}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let n = count[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 20_001, "seed row + 20k inserts");

    let merge_files: Vec<String> = live_data_paths(&table)
        .await
        .into_iter()
        .filter(|p| p.contains("merge-"))
        .collect();
    assert!(
        merge_files.len() > 1,
        "a giant insert batch must roll at the file-size target, got {} file(s)",
        merge_files.len()
    );
}

/// The stale-R shape (rust-lane phase 3): a `WHEN MATCHED AND <cond> THEN
/// DELETE` clause AHEAD of the demote-UPDATE removes superseded backfill
/// rows in the SAME atomic RowDelta as the demotes + inserts. Routing under
/// test: the stale row matches BOTH the conditional DELETE and the
/// unconditional UPDATE — first-match must pick DELETE; deleted rows join
/// the consolidated DV but are NOT re-appended (no demoted version).
#[tokio::test]
async fn matched_delete_clause_removes_stale_rows_via_dv() {
    let warehouse = TempDir::new().unwrap();
    let (catalog, ctx) = setup(
        &warehouse,
        &[
            // id=1 is a stale backfill row: _cdc_offset = -1 (the op=R
            // sentinel) — the source flags it for DELETE.
            (1, "r-stale", 10, None, true, -1),
            (2, "b", 10, None, true, 101),
            (3, "c", 10, None, true, 102),
        ],
        // Batch: replacements for ids 1 and 2, plus a brand-new id=4.
        &[(1, "a2", 20, 200), (2, "b2", 20, 201), (4, "d", 20, 203)],
    )
    .await;

    let before = load_table(&catalog).await;
    let snaps_before = before.metadata().snapshots().count();

    let sql = format!(
        "MERGE INTO {CATALOG}.{NS}.{TABLE} AS t USING ( \
             SELECT id, val, _valid_from, CAST(NULL AS BIGINT) AS _valid_to, \
                    true AS _is_current, _cdc_offset, false AS _stale_r \
             FROM batch \
             UNION ALL \
             SELECT t2.id, t2.val, t2._valid_from, b.new_vf AS _valid_to, \
                    false AS _is_current, t2._cdc_offset, \
                    (t2._cdc_offset = -1) AS _stale_r \
             FROM {CATALOG}.{NS}.{TABLE} t2 \
             JOIN (SELECT id, MIN(_valid_from) AS new_vf FROM batch GROUP BY id) b \
               ON t2.id = b.id AND t2._is_current \
         ) AS s \
         ON t.id = s.id AND t._valid_from = s._valid_from AND t._cdc_offset = s._cdc_offset \
         WHEN MATCHED AND s._stale_r THEN DELETE \
         WHEN MATCHED THEN UPDATE SET _valid_to = s._valid_to, _is_current = s._is_current \
         WHEN NOT MATCHED THEN INSERT (id, val, _valid_from, _valid_to, _is_current, _cdc_offset) \
             VALUES (s.id, s.val, s._valid_from, s._valid_to, s._is_current, s._cdc_offset)"
    );
    ctx.sql(&sql).await.unwrap().collect().await.unwrap();

    let table = load_table(&catalog).await;
    // ONE snapshot carries the delete + the demote + all three appends.
    assert_eq!(table.metadata().snapshots().count(), snaps_before + 1);

    let state = read_state(&ctx).await;
    assert_eq!(state, vec![
        // id=1: the stale backfill row is GONE (deleted, not demoted) —
        // only its replacement remains.
        (1, "a2".to_string(), 20, None, true),
        // id=2: normal SCD2 demote + new current.
        (2, "b".to_string(), 10, Some(20), false),
        (2, "b2".to_string(), 20, None, true),
        (3, "c".to_string(), 10, None, true),
        (4, "d".to_string(), 20, None, true),
    ]);

    // One consolidated DV on the seed file covering BOTH the deleted stale
    // row (id=1, pos 0) and the demoted row (id=2, pos 1).
    let dvs = live_dvs(&table).await;
    assert_eq!(dvs.len(), 1, "one DV total: {dvs:?}");
    assert_eq!(dvs[0].1, 2, "DV covers the deleted + the demoted row");
}

/// Shared variant-table scaffold for the shred-write tests: V3 table with a
/// `doc` variant column, partitioned like real silver, created with the
/// given properties.
async fn variant_table(warehouse: &TempDir, props: HashMap<String, String>) -> Arc<dyn Catalog> {
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
            NestedField::optional(2, "doc", Type::Variant(VariantType)).into(),
            NestedField::required(3, "_valid_from", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(4, "_valid_to", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(5, "_is_current", Type::Primitive(PrimitiveType::Boolean)).into(),
            NestedField::required(6, "_cdc_offset", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(5, "_is_current", Transform::Identity)
        .unwrap()
        .build();
    catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name(TABLE.to_string())
                .schema(schema)
                .partition_spec(spec)
                .format_version(FormatVersion::V3)
                .properties(props)
                .build(),
        )
        .await
        .unwrap();
    catalog
}

/// Canonical `{a, tags: [..]}` variant doc bytes.
fn doc_with_tags(a: i64, tags: &[i64]) -> (Vec<u8>, Vec<u8>) {
    let mut builder = VariantBuilder::new();
    let mut obj = builder.new_object();
    obj.insert("a", a);
    let mut list = obj.new_list("tags");
    for t in tags {
        list.append_value(*t);
    }
    list.finish();
    obj.finish();
    builder.finish()
}

fn variant_canonical_fields() -> Fields {
    Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ])
}

fn variant_doc_array(rows: &[Option<(Vec<u8>, Vec<u8>)>]) -> ArrayRef {
    let metas = BinaryArray::from_iter_values(
        rows.iter()
            .map(|v| v.as_ref().map(|(m, _)| m.clone()).unwrap_or_default()),
    );
    let vals = BinaryArray::from_iter_values(
        rows.iter()
            .map(|v| v.as_ref().map(|(_, x)| x.clone()).unwrap_or_default()),
    );
    let validity = NullBuffer::from(rows.iter().map(|v| v.is_some()).collect::<Vec<_>>());
    Arc::new(StructArray::new(
        variant_canonical_fields(),
        vec![Arc::new(metas) as ArrayRef, Arc::new(vals) as ArrayRef],
        Some(validity),
    ))
}

/// Seed batch of (id, doc) current rows for the variant table.
fn variant_seed_batch(rows: &[(i32, Option<(Vec<u8>, Vec<u8>)>)]) -> RecordBatch {
    let seed_schema = Arc::new(ArrowSchema::new(vec![
        field(1, "id", DataType::Int32, false),
        field(2, "doc", DataType::Struct(variant_canonical_fields()), true),
        field(3, "_valid_from", DataType::Int64, false),
        field(4, "_valid_to", DataType::Int64, true),
        field(5, "_is_current", DataType::Boolean, false),
        field(6, "_cdc_offset", DataType::Int64, false),
    ]));
    let n = rows.len();
    RecordBatch::try_new(seed_schema, vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        variant_doc_array(&rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>()),
        Arc::new(Int64Array::from(vec![10i64; n])),
        Arc::new(Int64Array::from(vec![None::<i64>; n])),
        Arc::new(BooleanArray::from(vec![true; n])),
        Arc::new(Int64Array::from(
            (0..n).map(|i| 100 + i as i64).collect::<Vec<_>>(),
        )),
    ])
    .unwrap()
}

/// Write one SHREDDED seed data file (the layout a shredding engine leaves
/// behind) and commit it.
async fn seed_shredded(catalog: &Arc<dyn Catalog>, batch: RecordBatch, plain: &DataType) {
    use iceberg::arrow::variant_shred::shred_record_batch as core_shred;
    let table = load_table(catalog).await;
    let plain_map = HashMap::from([("doc".to_string(), plain.clone())]);
    let shredded = core_shred(&batch, &plain_map).unwrap();
    let overrides = HashMap::from([(
        "doc".to_string(),
        shredded
            .schema()
            .field_with_name("doc")
            .unwrap()
            .data_type()
            .clone(),
    )]);
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
            .with_variant_shred_types(overrides),
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
    writer.write(shredded).await.unwrap();
    let data_files = writer.close().await.unwrap();
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();
}

/// Register the DataFusion session (catalog + `batch` MemTable + variant
/// UDFs) over an already-seeded variant table.
async fn variant_session(
    catalog: &Arc<dyn Catalog>,
    batch_rows: &[(i32, Option<(Vec<u8>, Vec<u8>)>, i64, i64)],
) -> SessionContext {
    let ctx = SessionContext::new();
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(Arc::clone(catalog))
            .await
            .unwrap(),
    );
    ctx.register_catalog(CATALOG, provider);
    register_variant_functions(&ctx);
    let batch_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("doc", DataType::Struct(variant_canonical_fields()), true),
        Field::new("_valid_from", DataType::Int64, false),
        Field::new("_cdc_offset", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(batch_schema, vec![
        Arc::new(Int32Array::from(
            batch_rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        variant_doc_array(&batch_rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>()),
        Arc::new(Int64Array::from(
            batch_rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            batch_rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
    ])
    .unwrap();
    let mem = MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();
    ctx
}

fn variant_merge_sql() -> String {
    format!(
        "MERGE INTO {CATALOG}.{NS}.{TABLE} AS t USING ( \
             SELECT id, doc, _valid_from, CAST(NULL AS BIGINT) AS _valid_to, \
                    true AS _is_current, _cdc_offset \
             FROM batch \
             UNION ALL \
             SELECT t2.id, t2.doc, t2._valid_from, b.new_vf AS _valid_to, \
                    false AS _is_current, t2._cdc_offset \
             FROM {CATALOG}.{NS}.{TABLE} t2 \
             JOIN (SELECT id, MIN(_valid_from) AS new_vf FROM batch GROUP BY id) b \
               ON t2.id = b.id AND t2._is_current \
         ) AS s \
         ON t.id = s.id AND t._valid_from = s._valid_from AND t._cdc_offset = s._cdc_offset \
         WHEN MATCHED THEN UPDATE SET _valid_to = s._valid_to, _is_current = s._is_current \
         WHEN NOT MATCHED THEN INSERT (id, doc, _valid_from, _valid_to, _is_current, _cdc_offset) \
             VALUES (s.id, s.doc, s._valid_from, s._valid_to, s._is_current, s._cdc_offset)"
    )
}

/// The `doc` arrow type of a written parquet file.
async fn doc_type_of(table: &Table, path: &str) -> DataType {
    let bytes = table
        .file_io()
        .new_input(path)
        .unwrap()
        .read()
        .await
        .unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
    reader
        .schema()
        .field_with_name("doc")
        .unwrap()
        .data_type()
        .clone()
}

/// Under `write.parquet.shred-variants`, a merge into a table whose existing
/// files are SHREDDED writes shredded output — object fields AND array
/// elements carried forward — on both append paths (inserts and the
/// late-materialized demote re-append), with the VARIANT annotation intact
/// and values/null semantics unchanged through the fold.
#[tokio::test]
async fn merge_shred_write_preserves_shredded_layout() {
    let warehouse = TempDir::new().unwrap();
    let catalog = variant_table(
        &warehouse,
        HashMap::from([(
            "write.parquet.shred-variants".to_string(),
            "true".to_string(),
        )]),
    )
    .await;

    let plain = DataType::Struct(Fields::from(vec![
        Field::new("a", DataType::Int64, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
            true,
        ),
    ]));
    // Seed: ids 1..=2 current, SHREDDED on {a, tags}.
    seed_shredded(
        &catalog,
        variant_seed_batch(&[
            (1, Some(doc_with_tags(1, &[10, 11]))),
            (2, Some(doc_with_tags(9, &[90]))),
        ]),
        &plain,
    )
    .await;

    // CDC batch: new version of id=1, a variant-null id=5, an SQL-null id=6.
    let variant_null: (Vec<u8>, Vec<u8>) = (vec![0x11, 0x00, 0x00], vec![0x00]);
    let ctx = variant_session(&catalog, &[
        (1, Some(doc_with_tags(2, &[20, 21])), 20, 200),
        (5, Some(variant_null), 20, 201),
        (6, None, 20, 202),
    ])
    .await;

    let before = load_table(&catalog).await;
    let snaps_before = before.metadata().snapshots().count();
    ctx.sql(&variant_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let table = load_table(&catalog).await;
    assert_eq!(table.metadata().snapshots().count(), snaps_before + 1);

    // Every merge-written file is SHREDDED: typed_value subtree with the
    // object field `a` (typed Int64) and the array `tags` (typed elements).
    let merge_files: Vec<String> = live_data_paths(&table)
        .await
        .into_iter()
        .filter(|p| p.contains("merge-"))
        .collect();
    assert!(!merge_files.is_empty(), "merge appended data files");
    for path in &merge_files {
        let doc_type = doc_type_of(&table, path).await;
        let DataType::Struct(children) = &doc_type else {
            panic!("doc is not a struct in {path}: {doc_type:?}");
        };
        let tv = children
            .iter()
            .find(|c| c.name() == "typed_value")
            .unwrap_or_else(|| panic!("merge output not shredded in {path}: {doc_type:?}"));
        let DataType::Struct(obj) = tv.data_type() else {
            panic!("typed_value is not an object node in {path}");
        };
        let a_node = obj.iter().find(|c| c.name() == "a").expect("a shredded");
        let DataType::Struct(a_children) = a_node.data_type() else {
            panic!("a is not a shred node");
        };
        assert!(
            a_children
                .iter()
                .any(|c| c.name() == "typed_value" && c.data_type() == &DataType::Int64),
            "a typed as Int64 in {path}"
        );
        let tags_node = obj
            .iter()
            .find(|c| c.name() == "tags")
            .expect("tags shredded");
        let DataType::Struct(tags_children) = tags_node.data_type() else {
            panic!("tags is not a shred node");
        };
        let tags_tv = tags_children
            .iter()
            .find(|c| c.name() == "typed_value")
            .expect("tags carries a typed_value");
        assert!(
            matches!(tags_tv.data_type(), DataType::List(_)),
            "tags typed_value is a list in {path}: {:?}",
            tags_tv.data_type()
        );
        // The annotation survives the shred override.
        let bytes = table
            .file_io()
            .new_input(path)
            .unwrap()
            .read()
            .await
            .unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        let doc_field = reader
            .metadata()
            .file_metadata()
            .schema()
            .get_fields()
            .iter()
            .find(|f| f.name() == "doc")
            .unwrap()
            .clone();
        assert!(
            matches!(
                doc_field.get_basic_info().logical_type_ref(),
                Some(LogicalType::Variant { .. })
            ),
            "shredded merge output keeps the VARIANT annotation: {path}"
        );
    }

    // Values + null semantics through the fold (the read path reconstructs
    // canonical from the shredded layout).
    let batches = ctx
        .sql(&format!(
            "SELECT id, variant_get_bigint(doc, '$.a') AS a, \
                    variant_get_bigint(doc, '$.tags[0]') AS t0, \
                    doc IS NULL AS doc_null \
             FROM {CATALOG}.{NS}.{TABLE} WHERE _is_current ORDER BY id"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows: Vec<(i32, Option<i64>, Option<i64>, bool)> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let a = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let t0 = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let dn = b.column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((
                ids.value(i),
                a.is_valid(i).then(|| a.value(i)),
                t0.is_valid(i).then(|| t0.value(i)),
                dn.value(i),
            ));
        }
    }
    assert_eq!(rows, vec![
        (1, Some(2), Some(20), false),
        (2, Some(9), Some(90), false),
        (5, None, None, false), // variant-null VALUE, not SQL null
        (6, None, None, true),  // SQL null
    ]);
}

/// Under `write.parquet.shred-variants`, a table whose existing files are
/// CANONICAL keeps writing canonical — the layout is preserved, never
/// invented.
#[tokio::test]
async fn merge_shred_write_stays_canonical_on_canonical_estate() {
    let warehouse = TempDir::new().unwrap();
    let catalog = variant_table(
        &warehouse,
        HashMap::from([(
            "write.parquet.shred-variants".to_string(),
            "true".to_string(),
        )]),
    )
    .await;

    // Canonical seed (no typed_value anywhere).
    let table = load_table(&catalog).await;
    let data_files = write_one_data_file(
        &table,
        variant_seed_batch(&[(1, Some(doc_with_tags(1, &[10])))]),
    )
    .await;
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(data_files)
        .apply(tx)
        .unwrap()
        .commit(catalog.as_ref())
        .await
        .unwrap();

    let ctx = variant_session(&catalog, &[(1, Some(doc_with_tags(2, &[20])), 20, 200)]).await;
    ctx.sql(&variant_merge_sql())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let table = load_table(&catalog).await;
    let merge_files: Vec<String> = live_data_paths(&table)
        .await
        .into_iter()
        .filter(|p| p.contains("merge-"))
        .collect();
    assert!(!merge_files.is_empty(), "merge appended data files");
    for path in &merge_files {
        let doc_type = doc_type_of(&table, path).await;
        let DataType::Struct(children) = &doc_type else {
            panic!("doc is not a struct in {path}: {doc_type:?}");
        };
        assert!(
            children.iter().all(|c| c.name() != "typed_value"),
            "canonical estate stays canonical (never invented) in {path}: {doc_type:?}"
        );
    }
    let state = read_state_ids(&ctx).await;
    assert_eq!(state, vec![(1, false), (1, true)]);
}

/// Sorted (id, _is_current) projection — a minimal state read for the
/// variant-table tests (val-less schema).
async fn read_state_ids(ctx: &SessionContext) -> Vec<(i32, bool)> {
    let batches = ctx
        .sql(&format!(
            "SELECT id, _is_current FROM {CATALOG}.{NS}.{TABLE} ORDER BY id, _valid_from"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let cur = b.column(1).as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((ids.value(i), cur.value(i)));
        }
    }
    out
}
