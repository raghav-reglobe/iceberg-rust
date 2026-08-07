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
use iceberg::Catalog;
use iceberg::arrow::{ArrowReader, RecordBatchPartitionSplitter};
use iceberg::scan::{ArrowRecordBatchStream, FileScanTask, FileScanTaskStream};
use iceberg::spec::{DataFile, DataFileFormat};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::partitioning::PartitioningWriter;
use iceberg::writer::partitioning::fanout_writer::FanoutWriter;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::config::Config;
use crate::planner::Group;
use crate::variant_shred;

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
    let mut action = tx
        .rewrite_files()
        .delete_data_files(removed)
        .delete_delete_files(removed_deletes)
        .add_data_files(added);
    // Input-protected rebase: `table` is the handle the whole compaction
    // planned and read from, so its current snapshot is the planning base.
    // A concurrent commit that only APPENDS (streaming sink, merge inserts
    // on other files) rebases fine; one that touches the inputs — e.g. a
    // merge writing a DV against a file being rewritten — aborts
    // non-retryably instead of silently resurrecting its deleted rows (the
    // 2026-08-01 merge×maintenance dup-current class).
    if let Some(snap) = table.metadata().current_snapshot_id() {
        action = action.validate_rebase_from(snap);
    }
    let tx = action.apply(tx)?;
    let new_table = tx.commit(catalog).await?;
    Ok(new_table)
}

/// Read one group's files (DVs applied, #2681), sort by `_valid_from`, write new
/// parquet -> the data files to add. Handles both unpartitioned and partitioned
/// tables. The caller (engine `compact_table`) accumulates every group's output
/// and removed files, then commits them all in ONE `RewriteFiles` via
/// `commit_rewrite` — a single atomic `Replace` snapshot, NOT one commit per group
/// (per-group commits re-run the manifest carry-forward over each prior snapshot,
/// which duplicated data files).
///
/// MEMORY CONTRACT (giant-TEXT safety): the group is processed as a stream of
/// CHUNKS of at most ~`cfg.sort_chunk_bytes` estimated arrow bytes. Each chunk
/// is sorted independently (bounded `interleave` slices of
/// ~`cfg.write_batch_bytes`, never a whole-group concat) and written before
/// the next chunk is buffered, so peak memory is one chunk + one slice +
/// writer buffers — REGARDLESS of group size. A 128 MB zstd group of multi-KB
/// TEXT rows decodes to GBs (and a delete-pressure giant file to tens of
/// GBs); whole-group materialization OOM-killed the pod and its whole-group
/// concat overflowed arrow's i32 string offsets. Groups under the chunk
/// budget (the common case) still produce ONE fully-sorted run; oversized
/// groups degrade gracefully to several sorted runs (slightly looser
/// per-file `_valid_from` bounds, full correctness).
/// Cooperative deadline check — the compaction twin of the merge doorway's
/// `timeout_s`. Returns a plain error naming the phase; never called on the
/// commit path (a commit, once entered, runs to completion).
pub(crate) fn check_deadline(cfg: &Config, what: &str) -> Result<()> {
    if let Some(d) = cfg.deadline
        && std::time::Instant::now() >= d
    {
        anyhow::bail!("compaction deadline exceeded while {what} (no commit was performed)");
    }
    Ok(())
}

pub(crate) async fn read_sort_write(
    table: &Table,
    reader: &ArrowReader,
    group: &Group,
    cfg: &Config,
) -> Result<Vec<DataFile>> {
    // Shred-preserving layout is derived once per group (first input file's
    // footer); the writer itself is built lazily on the first non-empty
    // slice (so an all-deleted group writes nothing) with the ACTUAL
    // shredded arrow types of that slice.
    let shred_plain = if cfg.shred_variants {
        variant_shred::input_shred_types(table, group).await?
    } else {
        std::collections::HashMap::new()
    };

    let mut stream = read_group(reader, group.tasks.clone()).await?;
    let mut sink: Option<CompactSink> = None;
    let mut chunk: Vec<RecordBatch> = Vec::new();
    let mut chunk_bytes = 0usize;
    while let Some(batch) = stream.try_next().await? {
        check_deadline(cfg, "reading input")?;
        if batch.num_rows() == 0 {
            continue;
        }
        chunk_bytes += batch.get_array_memory_size();
        chunk.push(batch);
        if chunk_bytes >= cfg.sort_chunk_bytes {
            check_deadline(cfg, "sorting/writing a chunk")?;
            flush_chunk(
                table,
                cfg,
                &shred_plain,
                std::mem::take(&mut chunk),
                &mut sink,
            )
            .await?;
            chunk_bytes = 0;
        }
    }
    flush_chunk(table, cfg, &shred_plain, chunk, &mut sink).await?;

    match sink {
        Some(s) => s.close().await,
        None => Ok(Vec::new()), // no live rows in the group — nothing to write
    }
}

/// Sort one buffered chunk and stream its bounded slices into the (lazily
/// created) group sink.
async fn flush_chunk(
    table: &Table,
    cfg: &Config,
    shred_plain: &std::collections::HashMap<String, arrow_schema::DataType>,
    batches: Vec<RecordBatch>,
    sink: &mut Option<CompactSink>,
) -> Result<()> {
    if batches.is_empty() {
        return Ok(());
    }
    let sort_key = cfg.sort_column.as_deref().unwrap_or("_valid_from");
    let mut sorted = crate::sort::sort_chunk(batches, cfg.write_batch_bytes, sort_key)?;
    while let Some(slice) = sorted.next_batch()? {
        if slice.num_rows() == 0 {
            continue;
        }
        // Shred-preserving mode: re-shred each (canonical, post-fold) slice
        // to the input files' layout; the first slice's actual shredded
        // types become the writer's schema overrides.
        let (slice, overrides) = if shred_plain.is_empty() {
            (slice, std::collections::HashMap::new())
        } else {
            let (mut shredded, overrides) = variant_shred::shred_batches(vec![slice], shred_plain)?;
            (shredded.pop().expect("one batch in, one out"), overrides)
        };
        if sink.is_none() {
            *sink = Some(CompactSink::build(table, overrides).await?);
        }
        sink.as_mut().expect("just built").write(slice).await?;
    }
    Ok(())
}

type CompactWriterBuilder =
    DataFileWriterBuilder<ParquetWriterBuilder, DefaultLocationGenerator, DefaultFileNameGenerator>;

/// The per-group output writer: iceberg-rust's writer chain (ParquetWriter ->
/// RollingFileWriter -> DataFileWriter), fanned out per partition for
/// partitioned tables. Built lazily on the first non-empty slice; fed
/// bounded slices; closed once per group. Per-column bloom filters are
/// enabled from the table's `write.parquet.bloom-filter-*` properties (see
/// `bloom_writer_properties`).
enum CompactSink {
    Plain(
        Box<
            iceberg::writer::base_writer::data_file_writer::DataFileWriter<
                ParquetWriterBuilder,
                DefaultLocationGenerator,
                DefaultFileNameGenerator,
            >,
        >,
    ),
    Fanout(
        Box<FanoutWriter<CompactWriterBuilder>>,
        Box<RecordBatchPartitionSplitter>,
    ),
}

impl CompactSink {
    async fn build(
        table: &Table,
        variant_shred_types: std::collections::HashMap<String, arrow_schema::DataType>,
    ) -> Result<Self> {
        let schema = table.metadata().current_schema().clone();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(bloom_writer_properties(table), schema.clone())
                .with_variant_shred_types(variant_shred_types),
            table.file_io().clone(),
            DefaultLocationGenerator::new(table.metadata())?,
            // Unique per group: DefaultFileNameGenerator's counter resets with
            // each new instance and one sink is built per group. A constant
            // prefix would name every group's output `compact-00000.parquet` —
            // multi-group compaction overwriting its own files.
            DefaultFileNameGenerator::new(
                format!("compact-{}", uuid::Uuid::now_v7()),
                None,
                DataFileFormat::Parquet,
            ),
        );
        let data_file_builder = DataFileWriterBuilder::new(rolling);
        let spec = table.metadata().default_partition_spec();
        if spec.is_unpartitioned() {
            Ok(Self::Plain(Box::new(data_file_builder.build(None).await?)))
        } else {
            let splitter =
                RecordBatchPartitionSplitter::try_new_with_computed_values(schema, spec.clone())?;
            Ok(Self::Fanout(
                Box::new(FanoutWriter::new(data_file_builder)),
                Box::new(splitter),
            ))
        }
    }

    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        match self {
            Self::Plain(writer) => writer.write(batch).await?,
            Self::Fanout(writer, splitter) => {
                for (partition_key, partition_batch) in splitter.split(&batch)? {
                    writer.write(partition_key, partition_batch).await?;
                }
            }
        }
        Ok(())
    }

    async fn close(self) -> Result<Vec<DataFile>> {
        Ok(match self {
            Self::Plain(mut writer) => writer.close().await?,
            Self::Fanout(writer, _) => writer.close().await?,
        })
    }
}

/// Build `WriterProperties` honoring the table's `write.parquet.bloom-filter-*`
/// TBLPROPERTIES — enable a per-column bloom (with fpp) on each marked column,
/// matching the per-column blooms parquet-mr writes. parquet-rs sizes blooms by
/// NDV/fpp; the parquet-mr adaptive-sizing flag
/// (`write.parquet.bloom-filter-adaptive-enabled`) has no rust equivalent and is
/// ignored.
fn bloom_writer_properties(table: &Table) -> WriterProperties {
    use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
    const ENABLED: &str = "write.parquet.bloom-filter-enabled.column.";
    const FPP: &str = "write.parquet.bloom-filter-fpp.column.";
    let props = table.metadata().properties();
    // Compression from `write.parquet.compression-codec` (+ optional
    // `write.parquet.compression-level`), defaulting to ZSTD — the iceberg
    // default. parquet-rs's own default is UNCOMPRESSED, which silently
    // inflated rewritten files ~20x before this translated the property.
    let codec = props
        .get("write.parquet.compression-codec")
        .map(|s| s.as_str())
        .unwrap_or("zstd");
    let level = props
        .get("write.parquet.compression-level")
        .and_then(|s| s.parse::<i32>().ok());
    let compression = match codec.to_ascii_lowercase().as_str() {
        "uncompressed" => Compression::UNCOMPRESSED,
        "snappy" => Compression::SNAPPY,
        "gzip" => Compression::GZIP(
            level
                .and_then(|l| u32::try_from(l).ok())
                .and_then(|l| GzipLevel::try_new(l).ok())
                .unwrap_or_default(),
        ),
        "lz4" => Compression::LZ4,
        "brotli" => Compression::BROTLI(
            level
                .and_then(|l| u32::try_from(l).ok())
                .and_then(|l| BrotliLevel::try_new(l).ok())
                .unwrap_or_default(),
        ),
        _ => Compression::ZSTD(
            level
                .and_then(|l| ZstdLevel::try_new(l).ok())
                .unwrap_or_default(),
        ),
    };
    let mut builder = WriterProperties::builder().set_compression(compression);
    for (key, val) in props {
        let Some(col) = key.strip_prefix(ENABLED) else {
            continue;
        };
        if val.as_str() != "true" {
            continue;
        }
        let path = ColumnPath::new(vec![col.to_string()]);
        builder = builder.set_column_bloom_filter_enabled(path.clone(), true);
        if let Some(fpp) = props
            .get(&format!("{FPP}{col}"))
            .and_then(|v| v.parse::<f64>().ok())
        {
            builder = builder.set_column_bloom_filter_fpp(path, fpp);
        }
    }
    builder.build()
}

/// Scan a group's data files into Arrow record batches (deletes/DVs applied,
/// #2681) via iceberg-rust's `ArrowReader`.
///
/// The reader is built ONCE per compaction run and shared across groups
/// (`ArrowReader` is `Clone` over `Arc`-shared state): its delete-file
/// cache persists, so a delete pile bound to many groups is downloaded and
/// decoded ONCE per run instead of once per group — with N groups over an
/// op=R replace pile the per-group loader re-decoded overlapping subsets of
/// the same pile up to N times (the ~26 min/group grind).
pub(crate) async fn read_group(
    reader: &ArrowReader,
    tasks: Vec<FileScanTask>,
) -> Result<ArrowRecordBatchStream> {
    let task_stream: FileScanTaskStream = futures::stream::iter(tasks.into_iter().map(Ok)).boxed();
    Ok(reader.clone().read(task_stream)?.stream())
}

/// Build the shared per-run reader. `reader_builder()` inherits the table's
/// whole-file data cache (when configured), so compaction reads share the
/// node-local copy.
pub(crate) fn build_run_reader(table: &Table) -> ArrowReader {
    table.reader_builder().build()
}

#[cfg(test)]
mod deadline_tests {
    use super::*;

    #[test]
    fn expired_deadline_aborts_with_phase_name() {
        let mut cfg = Config::default();
        cfg.deadline = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        let err = check_deadline(&cfg, "reading input").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("deadline exceeded"), "{msg}");
        assert!(msg.contains("reading input"), "{msg}");
        assert!(msg.contains("no commit was performed"), "{msg}");
    }

    #[test]
    fn unset_and_future_deadlines_pass() {
        let cfg = Config::default();
        check_deadline(&cfg, "x").unwrap();
        let mut cfg2 = Config::default();
        cfg2.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
        check_deadline(&cfg2, "x").unwrap();
    }
}
