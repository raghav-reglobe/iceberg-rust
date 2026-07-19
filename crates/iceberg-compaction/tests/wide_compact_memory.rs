//! Regression: compacting a wide-row (multi-KB TEXT) group must run in
//! BOUNDED memory. The old rewrite materialized the whole group and then
//! concatenated + took it (three decoded copies) — a 128 MB zstd group of
//! ~18 KB rows decodes to GBs, and a delete-pressure giant file to tens of
//! GBs, OOM-killing the pod. The streaming rewrite buffers at most
//! ~`sort_chunk_bytes` at a time and writes bounded slices, so peak memory
//! is a small multiple of the chunk budget regardless of group size.
//!
//! Proven with a process-wide peak-tracking allocator (this test file is its
//! own test binary): compacting ~300 MB of decoded wide rows under a 32 MB
//! chunk budget must peak far below the group's decoded size. On the old
//! whole-group path the same fixture peaked at ~1 GB (input + concat + take).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, Type,
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
use tempfile::TempDir;

/// System allocator wrapper tracking current and PEAK allocated bytes.
struct PeakTracking;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakTracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let cur = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(cur, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            let old = layout.size();
            if new_size >= old {
                let cur = CURRENT.fetch_add(new_size - old, Ordering::Relaxed) + (new_size - old);
                PEAK.fetch_max(cur, Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(old - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: PeakTracking = PeakTracking;

fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn field(id: i32, name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

async fn write_one_file(table: &Table, prefix: &str, batch: RecordBatch) -> Vec<DataFile> {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wide_row_compaction_memory_is_bounded_by_chunk_budget() {
    const N_FILES: usize = 6;
    const ROWS_PER_FILE: usize = 8_192;
    const PAYLOAD_BYTES: usize = 6_000;
    // Decoded group ~= 6 * 8192 * 6KB ~= 295 MB of string data.

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
            NestedField::required(1, "_valid_from", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let table = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("wide".to_string())
                .schema(schema)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .unwrap();

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field(1, "_valid_from", DataType::Int64, false),
        field(2, "payload", DataType::Utf8, true),
    ]));
    for f in 0..N_FILES {
        let keys: Vec<i64> = (0..ROWS_PER_FILE)
            .map(|r| ((r * N_FILES + f) % 100_003) as i64)
            .collect();
        let payloads: Vec<String> = keys
            .iter()
            .map(|k| {
                format!("{k:012}-").repeat(PAYLOAD_BYTES / 13 + 1)[..PAYLOAD_BYTES].to_string()
            })
            .collect();
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(StringArray::from(
                payloads.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ])
        .unwrap();
        let files = write_one_file(&table, &format!("wide-{f}"), batch).await;
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(files)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
    }
    let ident = TableIdent::new(ns, "wide".to_string());

    // Compact under a 32 MB chunk budget with 4 MB write slices.
    let cfg = Config {
        min_input_files: 1,
        delete_file_threshold: 1,
        sort_chunk_bytes: 32 * 1024 * 1024,
        write_batch_bytes: 4 * 1024 * 1024,
        ..Config::default()
    };

    reset_peak();
    let base = CURRENT.load(Ordering::Relaxed);
    compact_table(&catalog, &ident, &cfg).await.unwrap();
    let peak_delta = PEAK.load(Ordering::Relaxed).saturating_sub(base);

    // ~295 MB of decoded strings flow through, but the working set must stay
    // a small multiple of the 32 MB chunk budget. The old whole-group path
    // (collect + concat + take) peaked at ~3x the decoded size (~900 MB).
    eprintln!("PEAK_DELTA_MB={}", peak_delta / (1024 * 1024));
    const PEAK_LIMIT: usize = 160 * 1024 * 1024;
    assert!(
        peak_delta < PEAK_LIMIT,
        "compaction working set must be bounded by the chunk budget: \
         peak {} MB >= limit {} MB",
        peak_delta / (1024 * 1024),
        PEAK_LIMIT / (1024 * 1024)
    );

    // Correctness: every row survived the rewrite.
    let table = catalog.load_table(&ident).await.unwrap();
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
    assert_eq!(rows, N_FILES * ROWS_PER_FILE);
}
