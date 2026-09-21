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

//! A rewrite pass reports progress as `tracing` events on
//! `engine::PROGRESS_TARGET` and prints nothing itself: a subscriber sees one
//! event per phase, per group started and per group finished (with that
//! group's own `wall_s`), and the budget stop.
//!
//! ONE test in its own binary on purpose: `tracing` caches each callsite's
//! interest per process, so a thread-scoped subscriber can miss an event
//! whose callsite another test thread registered first.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
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
use iceberg_compaction::engine::{PROGRESS_TARGET, compact_table};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

async fn write_one_data_file(table: &Table, name: &str, ids: Vec<i32>) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]));
    let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();
    writer.write(batch).await.unwrap();
    let mut files = writer.close().await.unwrap();
    assert_eq!(files.len(), 1);
    files.remove(0)
}

#[derive(Default)]
struct Fields(Vec<(String, String)>);

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .push((field.name().to_string(), format!("{value:?}")));
    }
}

type Events = Arc<Mutex<Vec<Vec<(String, String)>>>>;

struct Collect(Events);

impl<S: tracing::Subscriber> Layer<S> for Collect {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() == PROGRESS_TARGET {
            let mut fields = Fields::default();
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }
}

fn field(event: &[(String, String)], name: &str) -> Option<String> {
    event
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.trim_matches('"').to_string())
}

#[tokio::test]
async fn progress_is_reported_as_tracing_events() {
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
        ])
        .build()
        .unwrap();
    let ident = TableIdent::new(ns.clone(), "t".to_string());
    let table = catalog
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
    let a = write_one_data_file(&table, "file-a", vec![1, 2, 3]).await;
    let b = write_one_data_file(&table, "file-b", vec![4, 5, 6]).await;
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(vec![a, b])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();

    let events: Events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(Collect(Arc::clone(&events)));
    // Every progress event comes from the coordinating task, so a
    // thread-scoped subscriber on this current-thread runtime sees them all.
    let guard = tracing::subscriber::set_default(subscriber);
    // One group per file (`rewrite_all` keeps lone delete-free files in the
    // plan); a budget that is already spent stops the pass after the first.
    let cfg = Config {
        target_file_size_bytes: 1,
        min_input_files: 1,
        rewrite_all: true,
        budget: Some(Instant::now()),
        ..Config::default()
    };
    let out = compact_table(&catalog, &ident, &cfg).await.unwrap();
    drop(guard);
    assert_eq!((out.groups_planned, out.groups_rewritten), (2, 1));

    let events = events.lock().unwrap();
    let phases: Vec<String> = events.iter().filter_map(|e| field(e, "phase")).collect();
    assert_eq!(phases, [
        "load",
        "plan",
        "rewrite",
        "group-done",
        "budget-stop",
        "commit",
        "done"
    ]);
    for e in events.iter() {
        assert_eq!(field(e, "table").as_deref(), Some("db.t"), "{e:?}");
        assert!(
            field(e, "elapsed_s").is_some() || field(e, "reason").is_some(),
            "{e:?}"
        );
    }
    let done = events
        .iter()
        .find(|e| field(e, "phase").as_deref() == Some("group-done"))
        .unwrap();
    assert_eq!(field(done, "group").as_deref(), Some("1/2"));
    assert_eq!(field(done, "files").as_deref(), Some("1"));
    assert!(field(done, "wall_s").is_some() && field(done, "in_flight").is_some());
    let stop = events
        .iter()
        .find(|e| field(e, "phase").as_deref() == Some("budget-stop"))
        .unwrap();
    assert_eq!(field(stop, "groups_started").as_deref(), Some("1/2"));
}
