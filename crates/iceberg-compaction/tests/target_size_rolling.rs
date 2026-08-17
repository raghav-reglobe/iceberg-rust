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

//! Regression: compaction OUTPUT files must roll at the configured
//! `target_file_size_bytes` — regardless of how the planner accounted the
//! INPUT group. The sink used to build its rolling writer with the built-in
//! default target (512 MiB), so any group whose output exceeded the
//! requested target (delete-pressure giants, variant-heavy rewrites) emitted
//! uniform ~512 MiB files against a 128 MB request — 4x the target, and far
//! past what downstream readers budget for. Narrow tables masked the bug:
//! their ~target-sized input bins produce sub-default output that never
//! reaches the roll threshold.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, Int32Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestList, NestedField,
    PrimitiveType, Schema, Type, VariantType,
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
use parquet::variant::VariantBuilder;
use tempfile::TempDir;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// ~1 KB pseudo-random hex payload — incompressible enough that zstd can't
/// collapse the fixture below the roll threshold.
fn payload(seed: u64) -> String {
    let mut state = seed;
    (0..64)
        .map(|_| format!("{:016x}", splitmix64(&mut state)))
        .collect()
}

fn arrow_schema() -> Arc<ArrowSchema> {
    let variant_fields = Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ]);
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("doc", DataType::Struct(variant_fields), true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]))
}

fn variant_batch(ids: std::ops::Range<i32>) -> RecordBatch {
    let mut metas: Vec<Vec<u8>> = Vec::new();
    let mut vals: Vec<Vec<u8>> = Vec::new();
    for id in ids.clone() {
        let mut b = VariantBuilder::new();
        b.append_value(payload(id as u64).as_str());
        let (m, v) = b.finish();
        metas.push(m);
        vals.push(v);
    }
    let variant_fields = Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ]);
    let doc = StructArray::new(
        variant_fields,
        vec![
            Arc::new(BinaryArray::from_iter_values(metas)) as ArrayRef,
            Arc::new(BinaryArray::from_iter_values(vals)) as ArrayRef,
        ],
        None,
    );
    RecordBatch::try_new(arrow_schema(), vec![
        Arc::new(Int32Array::from(ids.collect::<Vec<_>>())),
        Arc::new(doc),
    ])
    .unwrap()
}

async fn write_data_file(table: &Table, prefix: &str, batch: RecordBatch) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap()
}

/// Sizes (bytes) of every alive data file in the current snapshot.
async fn alive_data_file_sizes(table: &Table) -> Vec<u64> {
    let snap = table.metadata().current_snapshot().unwrap();
    let bytes = table
        .file_io()
        .new_input(snap.manifest_list())
        .unwrap()
        .read()
        .await
        .unwrap();
    let ml = ManifestList::parse_with_version(&bytes, table.metadata().format_version()).unwrap();
    let mut sizes = Vec::new();
    for mf in ml.entries() {
        if mf.content != ManifestContentType::Data {
            continue;
        }
        let m = mf.load_manifest(table.file_io()).await.unwrap();
        for e in m.entries() {
            if e.is_alive() {
                sizes.push(e.data_file().file_size_in_bytes());
            }
        }
    }
    sizes
}

#[tokio::test]
async fn output_rolls_at_configured_target_not_writer_default() {
    const ROWS_PER_FILE: i32 = 8_192; // ~8 MB of ~1 KB payloads per input file
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
            NestedField::optional(2, "doc", Type::Variant(VariantType)).into(),
        ])
        .build()
        .unwrap();
    let mut table = catalog
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
    for f in 0..2i32 {
        let files = write_data_file(
            &table,
            &format!("in-{f}"),
            variant_batch(f * ROWS_PER_FILE..(f + 1) * ROWS_PER_FILE),
        )
        .await;
        let tx = Transaction::new(&table);
        table = tx
            .fast_append()
            .add_data_files(files)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
    }
    let ident = TableIdent::new(ns, "t".to_string());

    // The bronze_ladder shape: explicit small target, shred-preserving,
    // every file a candidate. Each input file's output alone is many times
    // the target — with the old default-threshold writer the whole group
    // came back as ONE file.
    let target: u64 = 512 * 1024;
    let cfg = Config {
        target_file_size_bytes: target,
        min_file_size_bytes: target * 3 / 4,
        max_file_size_bytes: target * 9 / 5,
        min_input_files: 1,
        delete_file_threshold: 1,
        write_batch_bytes: 128 * 1024,
        shred_variants: true,
        rewrite_all: true,
        ..Config::default()
    };
    cfg.validate().unwrap();
    compact_table(&catalog, &ident, &cfg).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    let sizes = alive_data_file_sizes(&table).await;
    let total: u64 = sizes.iter().sum();
    assert!(
        total > 4 * target,
        "fixture must be several targets of output ({total} B written, target {target} B)"
    );
    assert!(
        sizes.len() >= 4,
        "output must roll into ~total/target files, got {} of sizes {sizes:?}",
        sizes.len()
    );
    // Roll granularity: the writer checks between bounded slices, so each
    // file may overshoot by at most ~one write batch + footer. 2x target is
    // a generous ceiling that the old 512 MiB default still fails by 8x.
    for s in &sizes {
        assert!(
            *s <= 2 * target,
            "an output file exceeded the target beyond roll granularity: \
             {s} B vs target {target} B (all: {sizes:?})"
        );
    }

    // Every row survived the rewrite.
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
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 2 * ROWS_PER_FILE as usize);
}
