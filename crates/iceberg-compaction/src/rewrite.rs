//! Compaction rewrite — for each planned group, read its data files
//! (DV-applied), sort, write new data files, then atomically swap them in
//! (remove old + add new) in one snapshot. Mirrors iceberg-go's `RewriteFiles`.
//!
//! The commit (`commit_rewrite`) is complete and uses the primitive Phase 0
//! confirmed present: iceberg-rust's RowDelta (#2678). The read→sort→write that
//! produces the new files are the two net-new pieces (sort + bloom).

use anyhow::Result;
use arrow_array::RecordBatch;
use futures::{StreamExt, TryStreamExt};
use iceberg::arrow::{ArrowReaderBuilder, RecordBatchPartitionSplitter};
use iceberg::scan::{ArrowRecordBatchStream, FileScanTask, FileScanTaskStream};
use iceberg::spec::{DataFile, DataFileFormat};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::partitioning::fanout_writer::FanoutWriter;
use iceberg::writer::partitioning::PartitioningWriter;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::Catalog;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::planner::Group;

/// Atomically replace `removed` data files with `added` AND reabsorb
/// `removed_deletes` (the rewritten files' deletion vectors) in a single
/// snapshot — the compaction commit, via iceberg-rust's `RewriteFiles`
/// (`Operation::Replace`, mirroring iceberg-go/iceberg-java): `delete_data_files`
/// + `delete_delete_files` + `add_data_files`.
///
/// The rewritten output already has the deletes applied (read path, #2681), and
/// `delete_delete_files` expunges the old DVs by referenced data file (fork:
/// content-aware manifest rewrite shared with RowDelta), so no delete file
/// references the removed data afterward — true DV reabsorption. `Replace`
/// records that table data is unchanged (only reorganized), unlike `Overwrite`.
pub async fn commit_rewrite(
    table: &Table,
    catalog: &dyn Catalog,
    removed: Vec<DataFile>,
    removed_deletes: Vec<DataFile>,
    added: Vec<DataFile>,
) -> Result<Table> {
    let tx = Transaction::new(table);
    let action = tx
        .rewrite_files()
        .delete_data_files(removed)
        .delete_delete_files(removed_deletes)
        .add_data_files(added);
    let tx = action.apply(tx)?;
    let new_table = tx.commit(catalog).await?;
    Ok(new_table)
}

/// Rewrite one planned group end-to-end.
///
/// `removed` = the current-snapshot data files backing `group.tasks`;
/// `removed_deletes` = their deletion vectors (reabsorbed). `read_sort_write`
/// produces `added`. Returns the table unchanged if there's nothing to write
/// (e.g. a fully-deleted group) — **never removes without replacement.**
pub async fn rewrite_group(
    table: Table,
    catalog: &dyn Catalog,
    group: Group,
    removed: Vec<DataFile>,
    removed_deletes: Vec<DataFile>,
) -> Result<Table> {
    let added = read_sort_write(&table, &group).await?;
    if added.is_empty() {
        return Ok(table); // no live rows to write (e.g. fully-deleted group)
    }
    commit_rewrite(&table, catalog, removed, removed_deletes, added).await
}

/// Read the group's files (DVs applied, #2681), sort by `_valid_from`, write new
/// parquet -> the data files to add. Handles both unpartitioned and partitioned
/// tables (bloom-filter props are the remaining fork edit).
async fn read_sort_write(table: &Table, group: &Group) -> Result<Vec<DataFile>> {
    let batches = read_group(table, group.tasks.clone()).await?.try_collect().await?;
    let sorted = crate::sort::sort_by_valid_from(batches)?;
    write_data_files(table, sorted).await
}

/// Write sorted record batches to new parquet data files via iceberg-rust's
/// writer chain (ParquetWriter -> RollingFileWriter -> DataFileWriter). For a
/// partitioned table the batches are routed to per-partition files by a
/// `FanoutWriter` (rows split via `RecordBatchPartitionSplitter`). Returns the
/// `added` DataFiles. Per-column bloom filters are enabled from the table's
/// `write.parquet.bloom-filter-*` properties (see `bloom_writer_properties`).
async fn write_data_files(table: &Table, batches: Vec<RecordBatch>) -> Result<Vec<DataFile>> {
    if batches.iter().all(|b| b.num_rows() == 0) {
        return Ok(Vec::new()); // nothing to write
    }
    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(bloom_writer_properties(table), schema.clone()),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata())?,
        DefaultFileNameGenerator::new("compact".to_string(), None, DataFileFormat::Parquet),
    );
    let data_file_builder = DataFileWriterBuilder::new(rolling);
    let spec = table.metadata().default_partition_spec();

    if spec.is_unpartitioned() {
        let mut writer = data_file_builder.build(None).await?;
        for batch in batches {
            writer.write(batch).await?;
        }
        Ok(writer.close().await?)
    } else {
        // Route each row to its partition's file. The splitter computes partition
        // values from the rows; FanoutWriter keeps one open writer per partition.
        let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(schema, spec.clone())?;
        let mut writer = FanoutWriter::new(data_file_builder);
        for batch in batches {
            for (partition_key, partition_batch) in splitter.split(&batch)? {
                writer.write(partition_key, partition_batch).await?;
            }
        }
        Ok(writer.close().await?)
    }
}

/// Build `WriterProperties` honoring the table's `write.parquet.bloom-filter-*`
/// TBLPROPERTIES — enable a per-column bloom (with fpp) on each marked column,
/// matching what the Spark compaction-app writes. parquet-rs sizes blooms by
/// NDV/fpp; the parquet-mr adaptive-sizing flag
/// (`write.parquet.bloom-filter-adaptive-enabled`) has no rust equivalent and is
/// ignored.
fn bloom_writer_properties(table: &Table) -> WriterProperties {
    const ENABLED: &str = "write.parquet.bloom-filter-enabled.column.";
    const FPP: &str = "write.parquet.bloom-filter-fpp.column.";
    let props = table.metadata().properties();
    let mut builder = WriterProperties::builder();
    for (key, val) in props {
        let Some(col) = key.strip_prefix(ENABLED) else {
            continue;
        };
        if val.as_str() != "true" {
            continue;
        }
        let path = ColumnPath::new(vec![col.to_string()]);
        builder = builder.set_column_bloom_filter_enabled(path.clone(), true);
        if let Some(fpp) = props.get(&format!("{FPP}{col}")).and_then(|v| v.parse::<f64>().ok()) {
            builder = builder.set_column_bloom_filter_fpp(path, fpp);
        }
    }
    builder.build()
}

/// Scan a group's data files into Arrow record batches (deletes/DVs applied,
/// #2681) via iceberg-rust's `ArrowReader`.
async fn read_group(table: &Table, tasks: Vec<FileScanTask>) -> Result<ArrowRecordBatchStream> {
    let reader =
        ArrowReaderBuilder::new(table.file_io().clone(), table.runtime().clone()).build();
    let task_stream: FileScanTaskStream = futures::stream::iter(tasks.into_iter().map(Ok)).boxed();
    Ok(reader.read(task_stream)?.stream())
}
