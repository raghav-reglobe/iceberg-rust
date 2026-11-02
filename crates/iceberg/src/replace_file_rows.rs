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

//! Copy-on-write replacement of the rows of NAMED data files with rows the
//! caller supplies — for a caller that computes new column values outside the
//! engine (a value only the caller knows how to render) and must put them back
//! without changing the set of rows.
//!
//! For each named live data file of the planning snapshot the caller hands the
//! file's complete replacement rows. The rows are conformed to the table schema
//! BY NAME (string columns targeting a VARIANT column are parsed as JSON), the
//! row count must equal the file's LIVE rows (its record count minus the rows
//! its delete files remove), identity-partition columns must carry the file's
//! own partition values, and one `Replace` snapshot removes the named files and
//! the delete files bound to them and adds the new files. Files not named are
//! untouched.
//!
//! Refusals (nothing is written or committed):
//!   * the table is not at `base_snapshot` (planning snapshot), here and again at
//!     commit — any snapshot committed in between, an append included, refuses;
//!   * a named path is not a live data file, or was written under an older
//!     partition spec;
//!   * a delete file bound to a named file is an equality delete, or is also
//!     bound to a file that is not named (removing it would resurrect rows there);
//!   * a replacement's row count differs from the file's live rows;
//!   * a replacement row's identity-partition value differs from the file's.
//!
//! The base snapshot's summary properties starting with `carry_summary_prefix`
//! are copied onto the new snapshot. Because any concurrent commit refuses the
//! replace, the carried values are the table's latest.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, SchemaRef as ArrowSchemaRef};
use futures::TryStreamExt;
use uuid::Uuid;

use crate::arrow::{ArrowReaderBuilder, schema_to_arrow_schema};
use crate::atomic_replace::{conform_batch, writer_properties};
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Literal, ManifestStatus, PartitionKey, Struct,
    Transform,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

/// The complete replacement rows of one live data file.
#[derive(Debug, Clone)]
pub struct FileRows {
    /// The data file's path exactly as the table metadata records it.
    pub data_file_path: String,
    /// Every row the file should hold after the replace.
    pub batches: Vec<RecordBatch>,
}

/// What a replace did (or, on a dry run, would do).
#[derive(Debug, Clone, Default)]
pub struct ReplaceFileRowsOutcome {
    /// The planning snapshot.
    pub snapshot_before: i64,
    /// `None` on a dry run.
    pub snapshot_after: Option<i64>,
    /// Named data files removed.
    pub files_replaced: usize,
    /// Data files written (0 on a dry run).
    pub files_added: usize,
    /// Live rows of the named files at the planning snapshot.
    pub rows_live: u64,
    /// Rows written (or, on a dry run, validated).
    pub rows_written: u64,
    /// Delete files bound to the named files, removed with them.
    pub delete_files_removed: usize,
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::DataInvalid, msg)
}

fn literal_at(array: &ArrayRef, row: usize) -> Result<Option<Literal>> {
    if array.is_null(row) {
        return Ok(None);
    }
    Ok(Some(match array.data_type() {
        DataType::Boolean => Literal::bool(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("bool")
                .value(row),
        ),
        DataType::Int32 => Literal::int(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("int")
                .value(row),
        ),
        DataType::Int64 => Literal::long(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("long")
                .value(row),
        ),
        DataType::Utf8 => Literal::string(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("utf8")
                .value(row),
        ),
        DataType::LargeUtf8 => Literal::string(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("large utf8")
                .value(row),
        ),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "an identity partition column of type {other} is not supported by replace_file_rows"
                ),
            ));
        }
    }))
}

/// Replace the rows of the named live data files of `ident` with the caller's
/// rows, copy-on-write, in one `Replace` snapshot (see the module docs). With
/// `dry_run` every check runs and nothing is written or committed.
pub async fn replace_file_rows(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    base_snapshot: i64,
    files: Vec<FileRows>,
    dry_run: bool,
    snapshot_properties: HashMap<String, String>,
    carry_summary_prefix: Option<&str>,
) -> Result<ReplaceFileRowsOutcome> {
    if files.is_empty() {
        return Err(invalid("no files to replace".to_string()));
    }
    let mut named: HashSet<String> = HashSet::new();
    for f in &files {
        if !named.insert(f.data_file_path.clone()) {
            return Err(invalid(format!("{} is named twice", f.data_file_path)));
        }
    }
    let table = catalog.load_table(ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .cloned()
        .ok_or_else(|| invalid(format!("{ident}: no current snapshot")))?;
    let s0 = snapshot.snapshot_id();
    if s0 != base_snapshot {
        return Err(invalid(format!(
            "{ident}: planned against snapshot {base_snapshot} but the table is at {s0}"
        )));
    }
    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().clone();
    let default_spec_id = table.metadata().default_partition_spec_id();
    // the identity partition columns: (position in the partition struct, column name)
    let mut identity: Vec<(usize, String)> = Vec::new();
    for (i, f) in spec.fields().iter().enumerate() {
        if f.transform != Transform::Identity {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: replace_file_rows supports identity partitions only, found {:?}",
                    f.transform
                ),
            ));
        }
        let name = schema
            .field_by_id(f.source_id)
            .map(|x| x.name.clone())
            .ok_or_else(|| {
                invalid(format!(
                    "{ident}: partition source id {} not in the schema",
                    f.source_id
                ))
            })?;
        identity.push((i, name));
    }

    // live entries of the planning snapshot
    let mut live_data: HashMap<String, DataFile> = HashMap::new();
    let mut live_deletes: HashMap<String, DataFile> = HashMap::new();
    let manifest_list = table.manifest_list_reader(&snapshot).load().await?;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            let df = entry.data_file().clone();
            match entry.content_type() {
                DataContentType::Data => {
                    live_data.insert(df.file_path().to_string(), df);
                }
                _ => {
                    live_deletes.insert(df.file_path().to_string(), df);
                }
            }
        }
    }
    let mut old_files: Vec<DataFile> = Vec::new();
    for f in &files {
        let df = live_data.get(&f.data_file_path).ok_or_else(|| {
            invalid(format!(
                "{ident}: {} is not a live data file of snapshot {s0}",
                f.data_file_path
            ))
        })?;
        if df.partition_spec_id != default_spec_id {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: {} was written under partition spec {} (default {default_spec_id})",
                    f.data_file_path, df.partition_spec_id
                ),
            ));
        }
        old_files.push(df.clone());
    }

    // the scan's own binding of delete files to data files
    let scan = table.scan().snapshot_id(s0).select_all().build()?;
    let mut tasks_by_file: HashMap<String, Vec<FileScanTask>> = HashMap::new();
    let mut bound_to: HashMap<String, HashSet<String>> = HashMap::new();
    let mut planned = scan.plan_files().await?;
    while let Some(task) = planned.try_next().await? {
        for d in &task.deletes {
            bound_to
                .entry(d.file_path.clone())
                .or_default()
                .insert(task.data_file_path.clone());
        }
        if named.contains(&task.data_file_path) {
            tasks_by_file
                .entry(task.data_file_path.clone())
                .or_default()
                .push(task);
        }
    }
    let mut removed_deletes: Vec<DataFile> = Vec::new();
    for (del, data_paths) in &bound_to {
        if !data_paths.iter().any(|p| named.contains(p)) {
            continue;
        }
        let df = live_deletes.get(del).ok_or_else(|| {
            invalid(format!(
                "{ident}: the scan binds delete file {del}, which is not live"
            ))
        })?;
        if df.content_type() == DataContentType::EqualityDeletes {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "{ident}: equality delete file {del} applies to a named file — reabsorb it first"
                ),
            ));
        }
        if let Some(other) = data_paths.iter().find(|p| !named.contains(*p)) {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("{ident}: delete file {del} also applies to {other}, which is not named"),
            ));
        }
        removed_deletes.push(df.clone());
    }

    // each named file's live rows, read through the scan's own tasks (deletes applied)
    let mut outcome = ReplaceFileRowsOutcome {
        snapshot_before: s0,
        files_replaced: old_files.len(),
        delete_files_removed: removed_deletes.len(),
        ..Default::default()
    };
    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema)?);
    let mut conformed: Vec<(DataFile, Vec<RecordBatch>)> = Vec::new();
    for (f, old) in files.into_iter().zip(old_files) {
        let tasks = tasks_by_file.remove(&f.data_file_path).unwrap_or_default();
        let live = if tasks.is_empty() {
            0
        } else {
            let stream =
                Box::pin(futures::stream::iter(tasks.into_iter().map(Ok))) as FileScanTaskStream;
            let mut batches =
                ArrowReaderBuilder::new(table.file_io().clone(), table.runtime().clone())
                    .build()
                    .read(stream)?
                    .stream();
            let mut n = 0u64;
            while let Some(b) = batches.try_next().await? {
                n += b.num_rows() as u64;
            }
            n
        };
        let given: u64 = f.batches.iter().map(|b| b.num_rows() as u64).sum();
        if given != live {
            return Err(invalid(format!(
                "{ident}: {} has {live} live rows but {given} replacement rows were given",
                f.data_file_path
            )));
        }
        outcome.rows_live += live;
        let mut out = Vec::with_capacity(f.batches.len());
        for b in &f.batches {
            if b.num_rows() == 0 {
                continue;
            }
            let c = conform_batch(b, &target_arrow)?;
            for (pos, col) in &identity {
                let want = old.partition().fields().get(*pos).cloned().flatten();
                let idx = c.schema().index_of(col).map_err(|_| {
                    invalid(format!(
                        "{ident}: replacement rows lack the partition column `{col}`"
                    ))
                })?;
                let arr = c.column(idx).clone();
                for r in 0..arr.len() {
                    if literal_at(&arr, r)? != want {
                        return Err(invalid(format!(
                            "{ident}: a replacement row for {} carries `{col}` = {:?}, the file's partition is {want:?}",
                            f.data_file_path,
                            literal_at(&arr, r)?
                        )));
                    }
                }
            }
            outcome.rows_written += c.num_rows() as u64;
            out.push(c);
        }
        conformed.push((old, out));
    }
    if dry_run {
        return Ok(outcome);
    }

    let run = Uuid::now_v7();
    let mut new_files: Vec<DataFile> = Vec::new();
    let mut old_files: Vec<DataFile> = Vec::new();
    for (i, (old, batches)) in conformed.into_iter().enumerate() {
        if !batches.is_empty() {
            let rolling = RollingFileWriterBuilder::new_with_default_file_size(
                ParquetWriterBuilder::new(writer_properties(&table), schema.clone()),
                table.file_io().clone(),
                DefaultLocationGenerator::new(table.metadata())?,
                DefaultFileNameGenerator::new(
                    format!("replace-{run}-f{i}"),
                    None,
                    DataFileFormat::Parquet,
                ),
            );
            let pkey = if identity.is_empty() {
                None
            } else {
                Some(PartitionKey::new(
                    spec.as_ref().clone(),
                    schema.clone(),
                    Struct::from_iter(old.partition().fields().iter().cloned()),
                ))
            };
            let mut writer = DataFileWriterBuilder::new(rolling).build(pkey).await?;
            for b in batches {
                writer.write(b).await?;
            }
            new_files.extend(writer.close().await?);
        }
        old_files.push(old);
    }
    outcome.files_added = new_files.len();

    let mut props = snapshot_properties;
    if let Some(prefix) = carry_summary_prefix {
        for (k, v) in &snapshot.summary().additional_properties {
            if k.starts_with(prefix) && !props.contains_key(k) {
                props.insert(k.clone(), v.clone());
            }
        }
    }
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .delete_data_files(old_files)
        .delete_delete_files(removed_deletes)
        .add_data_files(new_files)
        .validate_from_snapshot(s0)
        .set_snapshot_properties(props)
        .apply(tx)?
        .commit(catalog)
        .await?;
    outcome.snapshot_after = table.metadata().current_snapshot_id();
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, BooleanArray, Int64Array, LargeStringArray, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::variant::VariantArray;
    use roaring::RoaringTreemap;
    use tempfile::TempDir;

    use super::*;
    use crate::delete_vector::DeleteVector;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{
        FormatVersion, NestedField, PrimitiveType, Schema, Type, UnboundPartitionSpec, VariantType,
    };
    use crate::table::Table;
    use crate::{CatalogBuilder, NamespaceIdent, TableCreation};

    async fn setup(
        version: FormatVersion,
        partitioned: bool,
    ) -> (TempDir, Arc<dyn Catalog>, TableIdent, Table) {
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
        let ns = NamespaceIdent::new("db".to_string());
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
        let mut fields = vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "v", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "flag", Type::Primitive(PrimitiveType::Boolean)).into(),
        ];
        if version == FormatVersion::V3 {
            fields.push(NestedField::optional(4, "doc", Type::Variant(VariantType)).into());
        }
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(fields)
            .build()
            .unwrap();
        let mut b = TableCreation::builder()
            .name("t".to_string())
            .schema(schema)
            .format_version(version)
            .build();
        if partitioned {
            b.partition_spec = Some(
                UnboundPartitionSpec::builder()
                    .add_partition_field(3, "flag", Transform::Identity)
                    .unwrap()
                    .build(),
            );
        }
        let table = catalog.create_table(&ns, b).await.unwrap();
        (
            warehouse,
            catalog,
            TableIdent::new(ns, "t".to_string()),
            table,
        )
    }

    /// Rows (id, v, flag[, doc as JSON text]) in caller form — strings, conformed by the replace.
    fn rows(
        ids: &[i64],
        v: &[Option<&str>],
        flag: Option<bool>,
        doc: Option<&[Option<&str>]>,
    ) -> RecordBatch {
        let mut fields = vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
        ];
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(v.to_vec())),
            Arc::new(BooleanArray::from(vec![flag; ids.len()])),
        ];
        if let Some(d) = doc {
            fields.push(Field::new("doc", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from(d.to_vec())));
        }
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols).unwrap()
    }

    async fn write(table: &Table, name: &str, batch: RecordBatch, flag: Option<bool>) -> DataFile {
        let schema = table.metadata().current_schema().clone();
        let target = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let batch = conform_batch(&batch, &target).unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(writer_properties(table), schema.clone()),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata()).unwrap(),
            DefaultFileNameGenerator::new(name.to_string(), None, DataFileFormat::Parquet),
        );
        let pkey = if table
            .metadata()
            .default_partition_spec()
            .fields()
            .is_empty()
        {
            None
        } else {
            Some(PartitionKey::new(
                table.metadata().default_partition_spec().as_ref().clone(),
                schema,
                Struct::from_iter(vec![flag.map(Literal::bool)]),
            ))
        };
        let mut w = DataFileWriterBuilder::new(rolling)
            .build(pkey)
            .await
            .unwrap();
        w.write(batch).await.unwrap();
        w.close().await.unwrap().into_iter().next().unwrap()
    }

    async fn append(
        catalog: &dyn Catalog,
        table: &Table,
        files: Vec<DataFile>,
        props: HashMap<String, String>,
    ) -> Table {
        let tx = Transaction::new(table);
        tx.fast_append()
            .add_data_files(files)
            .set_snapshot_properties(props)
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    async fn read(table: &Table) -> Vec<(i64, Option<String>)> {
        let mut out = Vec::new();
        let mut s = table
            .scan()
            .select_all()
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        while let Some(b) = s.try_next().await.unwrap() {
            let id = b
                .column(b.schema().index_of("id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let v = b
                .column(b.schema().index_of("v").unwrap())
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap()
                .clone();
            for i in 0..b.num_rows() {
                out.push((id.value(i), v.is_valid(i).then(|| v.value(i).to_string())));
            }
        }
        out.sort();
        out
    }

    #[tokio::test]
    async fn replaces_only_the_named_file_and_carries_the_prefixed_summary() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), None),
            None,
        )
        .await;
        let f2 = write(
            &table,
            "f2",
            rows(&[3], &[Some("keep")], Some(false), None),
            None,
        )
        .await;
        let table = append(
            catalog.as_ref(),
            &table,
            vec![f1.clone(), f2],
            HashMap::from([
                ("ledger.offset.a".to_string(), "41".to_string()),
                ("other".to_string(), "x".to_string()),
            ]),
        )
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1, 2], &[Some("a"), Some("b")], Some(false), None)],
        }];
        let dry = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl.clone(),
            true,
            HashMap::new(),
            Some("ledger."),
        )
        .await
        .unwrap();
        assert_eq!(
            (
                dry.snapshot_after,
                dry.rows_live,
                dry.rows_written,
                dry.files_replaced
            ),
            (None, 2, 2, 1)
        );
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s0),
            "a dry run commits nothing"
        );

        let out = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::from([("mark".to_string(), "1".to_string())]),
            Some("ledger."),
        )
        .await
        .unwrap();
        assert_eq!(
            (out.files_replaced, out.files_added, out.rows_written),
            (1, 1, 2)
        );
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(out.snapshot_after, table.metadata().current_snapshot_id());
        assert_eq!(read(&table).await, vec![
            (1, Some("a".to_string())),
            (2, Some("b".to_string())),
            (3, Some("keep".to_string()))
        ]);
        let summary = &table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties;
        assert_eq!(
            summary.get("ledger.offset.a").map(String::as_str),
            Some("41"),
            "the prefixed key is carried"
        );
        assert_eq!(summary.get("other"), None, "only the prefix is carried");
        assert_eq!(summary.get("mark").map(String::as_str), Some("1"));
        assert_eq!(summary.get("total-records").map(String::as_str), Some("3"));
    }

    #[tokio::test]
    async fn a_stale_base_snapshot_refuses_even_after_a_plain_append() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let f1 = write(&table, "f1", rows(&[1], &[None], Some(false), None), None).await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let f2 = write(&table, "f2", rows(&[2], &[None], Some(false), None), None).await;
        let table = append(catalog.as_ref(), &table, vec![f2], HashMap::new()).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], Some(false), None)],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("planned against snapshot"),
            "{err}"
        );
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s1)
        );
    }

    #[tokio::test]
    async fn a_row_count_mismatch_and_an_unknown_file_refuse() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), None),
            None,
        )
        .await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let short = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], Some(false), None)],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            short,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("2 live rows but 1 replacement rows"),
            "{err}"
        );
        let unknown = vec![FileRows {
            data_file_path: "s3://nowhere/x.parquet".to_string(),
            batches: vec![],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            unknown,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("is not a live data file"), "{err}");
        assert_eq!(
            catalog
                .load_table(&ident)
                .await
                .unwrap()
                .metadata()
                .current_snapshot_id(),
            Some(s0),
            "nothing committed"
        );
    }

    #[tokio::test]
    async fn a_partition_value_that_moves_a_row_refuses() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V2, true).await;
        let f1 = write(&table, "f1", rows(&[1], &[None], None, None), None).await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let moved = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], Some(true), None)],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            moved,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("the file's partition is"), "{err}");
        // the same row kept in its (null) partition is accepted and lands there
        let kept = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(&[1], &[Some("a")], None, None)],
        }];
        let out = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            kept,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.files_added, 1);
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read(&table).await, vec![(1, Some("a".to_string()))]);
    }

    #[tokio::test]
    async fn json_text_lands_as_a_variant_document() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V3, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(&[1, 2], &[None, None], Some(false), Some(&[None, None])),
            None,
        )
        .await;
        let table = append(catalog.as_ref(), &table, vec![f1.clone()], HashMap::new()).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let repl = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 2],
                &[Some("a"), None],
                Some(false),
                Some(&[Some(r#"{"k": "x", "n": 12.50}"#), None]),
            )],
        }];
        replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            repl,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        let table = catalog.load_table(&ident).await.unwrap();
        let mut s = table
            .scan()
            .select_all()
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        let mut docs = Vec::new();
        while let Some(b) = s.try_next().await.unwrap() {
            let ids = b
                .column(b.schema().index_of("id").unwrap())
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let va = VariantArray::try_new(b.column(b.schema().index_of("doc").unwrap())).unwrap();
            for i in 0..b.num_rows() {
                let k = if va.is_null(i) {
                    None
                } else {
                    let v = va.value(i);
                    let obj = v.as_object().expect("object");
                    Some(
                        obj.get("k")
                            .expect("k")
                            .as_string()
                            .expect("string")
                            .to_string(),
                    )
                };
                docs.push((ids.value(i), k));
            }
        }
        docs.sort();
        assert_eq!(docs, vec![(1, Some("x".to_string())), (2, None)]);
    }

    #[tokio::test]
    async fn a_deletion_vector_on_the_replaced_file_counts_and_is_removed_with_it() {
        let (_wh, catalog, ident, table) = setup(FormatVersion::V3, false).await;
        let f1 = write(
            &table,
            "f1",
            rows(
                &[1, 2, 3],
                &[None, None, None],
                Some(false),
                Some(&[None, None, None]),
            ),
            None,
        )
        .await;
        let f2 = write(
            &table,
            "f2",
            rows(&[4], &[Some("keep")], Some(false), Some(&[None])),
            None,
        )
        .await;
        let table = append(
            catalog.as_ref(),
            &table,
            vec![f1.clone(), f2],
            HashMap::new(),
        )
        .await;
        // delete row position 1 (id 2) of f1 through a deletion vector
        let dv = DeleteVector::new(RoaringTreemap::from_iter([1u64]));
        let loc = format!("{}/data/dv-f1.puffin", table.metadata().location());
        let del = dv
            .write_to_puffin_file(
                table.file_io(),
                loc,
                f1.file_path().to_string(),
                Struct::empty(),
                table.metadata().default_partition_spec_id(),
            )
            .await
            .unwrap();
        let tx = Transaction::new(&table);
        let table = tx
            .row_delta()
            .add_delete_files(vec![del])
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
        let s0 = table.metadata().current_snapshot_id().unwrap();
        assert_eq!(read(&table).await.len(), 3, "the DV hides id 2");
        // three replacement rows are one too many: the file has 2 live rows
        let three = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 2, 3],
                &[Some("a"), Some("b"), Some("c")],
                Some(false),
                Some(&[None, None, None]),
            )],
        }];
        let err = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            three,
            true,
            HashMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("2 live rows but 3"), "{err}");
        let two = vec![FileRows {
            data_file_path: f1.file_path().to_string(),
            batches: vec![rows(
                &[1, 3],
                &[Some("a"), Some("c")],
                Some(false),
                Some(&[None, None]),
            )],
        }];
        let out = replace_file_rows(
            catalog.as_ref(),
            &ident,
            s0,
            two,
            false,
            HashMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!((out.rows_live, out.delete_files_removed), (2, 1));
        let table = catalog.load_table(&ident).await.unwrap();
        assert_eq!(read(&table).await, vec![
            (1, Some("a".to_string())),
            (3, Some("c".to_string())),
            (4, Some("keep".to_string()))
        ]);
        // no delete file survives: the DV left with its data file
        let snap = table.metadata().current_snapshot().unwrap();
        let ml = table.manifest_list_reader(snap).load().await.unwrap();
        let mut live_deletes = 0;
        for mf in ml.entries() {
            for e in mf.load_manifest(table.file_io()).await.unwrap().entries() {
                if e.status() != ManifestStatus::Deleted
                    && e.content_type() != DataContentType::Data
                {
                    live_deletes += 1;
                }
            }
        }
        assert_eq!(live_deletes, 0);
    }
}
