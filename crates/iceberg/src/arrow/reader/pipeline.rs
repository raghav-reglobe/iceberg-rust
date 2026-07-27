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

//! The main `ArrowReader` pipeline: reading a stream of `FileScanTask`s,
//! opening Parquet files and resolving schemas, then wiring projection,
//! predicates, row-group / row selection, and delete handling into a stream
//! of transformed Arrow `RecordBatch`es.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ParquetRecordBatchStreamBuilder, RowNumber};
use parquet::variant::{VariantArray, unshred_variant};

use super::{
    ArrowFileReader, ArrowReader, ParquetReadOptions, add_fallback_field_ids_to_arrow_schema,
    apply_name_mapping_to_arrow_schema,
};
use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
use crate::arrow::int96::coerce_int96_timestamps;
use crate::arrow::large_offsets::widen_variable_length_types;
use crate::arrow::record_batch_transformer::RecordBatchTransformerBuilder;
use crate::arrow::scan_metrics::{CountingFileRead, ScanMetrics, ScanResult};
use crate::cache::DataBytesCache;
use crate::error::Result;
use crate::io::{FileIO, FileMetadata, FileRead};
use crate::metadata_columns::{
    RESERVED_COL_NAME_POS, RESERVED_FIELD_ID_FILE, RESERVED_FIELD_ID_POS, is_metadata_field,
};
use crate::scan::{ArrowRecordBatchStream, FileScanTask, FileScanTaskStream};
use crate::spec::Datum;
use crate::{Error, ErrorKind};

impl ArrowReader {
    /// Take a stream of FileScanTasks and reads all the files.
    /// Returns a [`ScanResult`] containing the record batch stream and scan metrics.
    pub fn read(self, tasks: FileScanTaskStream) -> Result<ScanResult> {
        let concurrency_limit_data_files = self.concurrency_limit_data_files;
        let scan_metrics = ScanMetrics::new();
        let runtime = self.runtime.clone();

        let task_reader = FileScanTaskReader {
            batch_size: self.batch_size,
            file_io: self.file_io,
            delete_file_loader: self
                .delete_file_loader
                .with_scan_metrics(scan_metrics.clone()),
            row_group_filtering_enabled: self.row_group_filtering_enabled,
            row_selection_enabled: self.row_selection_enabled,
            parquet_read_options: self.parquet_read_options,
            scan_metrics: scan_metrics.clone(),
            shredded_passthrough: self.shredded_passthrough.clone(),
            data_bytes_cache: self.data_bytes_cache.clone(),
        };

        // Fast-path for single concurrency to avoid overhead of try_flatten_unordered
        let stream: ArrowRecordBatchStream = if concurrency_limit_data_files == 1 {
            Box::pin(
                tasks
                    .and_then(move |task| task_reader.clone().process(task))
                    .map_err(|err| {
                        Error::new(ErrorKind::Unexpected, "file scan task generate failed")
                            .with_source(err)
                    })
                    .try_flatten(),
            )
        } else {
            // Each file's processing (parquet page decode, variant unshred
            // fold, batch transform) is CPU-bound. `try_buffer_unordered` /
            // `try_flatten_unordered` alone only OVERLAP the futures on the
            // single consuming task — decode never uses more than one core.
            // Spawn each file's producer onto the CPU runtime so files decode
            // in PARALLEL; a small bounded channel per file provides
            // backpressure (dropping the receiver aborts the producer).
            Box::pin(
                tasks
                    .map_ok(move |task| {
                        let reader = task_reader.clone();
                        let cpu = runtime.cpu().clone();
                        async move {
                            let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(4);
                            cpu.spawn(async move {
                                match reader.process(task).await {
                                    Ok(mut s) => {
                                        while let Some(item) = s.next().await {
                                            if tx.send(item).await.is_err() {
                                                break; // consumer gone (limit/abort)
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(e)).await;
                                    }
                                }
                            });
                            let batches = futures::stream::unfold(rx, |mut rx| async move {
                                rx.recv().await.map(|item| (item, rx))
                            });
                            Ok(Box::pin(batches) as ArrowRecordBatchStream)
                        }
                    })
                    .map_err(|err| {
                        Error::new(ErrorKind::Unexpected, "file scan task generate failed")
                            .with_source(err)
                    })
                    .try_buffer_unordered(concurrency_limit_data_files)
                    .try_flatten_unordered(concurrency_limit_data_files),
            )
        };

        Ok(ScanResult::new(stream, scan_metrics))
    }

    /// Load the aggregated POSITIONAL delete state (V3 deletion vectors and
    /// V2 position-delete files) applying to each task's data file, WITHOUT
    /// reading any data rows.
    ///
    /// Returns the per-data-file delete vector for every task whose delete
    /// entries yield one; data files with no positional deletes are absent
    /// from the map. Equality deletes are loaded as predicates by the read
    /// path and never surface here — callers that must reason about them
    /// should inspect `task.deletes` themselves.
    ///
    /// The loads share this reader's delete-file cache: a subsequent
    /// [`ArrowReader::read`] over the same tasks reuses the already-loaded
    /// state instead of re-fetching the delete files.
    pub async fn load_positional_deletes(
        &self,
        tasks: &[FileScanTask],
    ) -> Result<HashMap<String, crate::delete_vector::DeleteVector>> {
        let mut out = HashMap::new();
        for task in tasks {
            if task.deletes.is_empty() {
                continue;
            }
            let delete_filter = self
                .delete_file_loader
                .load_deletes(&task.deletes, Arc::clone(&task.schema))
                .await
                .map_err(|_| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "delete-file loading task was cancelled",
                    )
                })??;
            if let Some(dv) = delete_filter.get_delete_vector(task) {
                let dv = dv.lock().map_err(|_| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "delete-vector lock poisoned during load",
                    )
                })?;
                out.insert(task.data_file_path().to_string(), dv.clone());
            }
        }
        Ok(out)
    }
}

/// Per-scan state for processing [`FileScanTask`]s. Created once per
/// [`ArrowReader::read`] call and cloned per task.
#[derive(Clone)]
struct FileScanTaskReader {
    batch_size: Option<usize>,
    file_io: FileIO,
    delete_file_loader: CachingDeleteFileLoader,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    parquet_read_options: ParquetReadOptions,
    scan_metrics: ScanMetrics,
    /// Shredded variant passthrough: column name -> the EXACT shredded Arrow
    /// type the consumer accepts verbatim. Per FILE, a variant column whose
    /// physical type equals the expected type skips the unshred fold (and the
    /// transformer targets the shredded type); any other layout — canonical,
    /// or a different shredding — folds to canonical as usual.
    shredded_passthrough: Option<Arc<HashMap<String, DataType>>>,
    /// See [`ArrowReaderBuilder::with_data_bytes_cache`].
    data_bytes_cache: Option<DataBytesCache>,
}

impl FileScanTaskReader {
    async fn process(self, task: FileScanTask) -> Result<ArrowRecordBatchStream> {
        let should_load_page_index =
            (self.row_selection_enabled && task.predicate.is_some()) || !task.deletes.is_empty();
        let mut parquet_read_options = self.parquet_read_options;
        parquet_read_options.preload_page_index = should_load_page_index;

        let delete_filter_rx = self
            .delete_file_loader
            .load_deletes(&task.deletes, Arc::clone(&task.schema));

        // Open the Parquet file once, loading its metadata
        let (parquet_file_reader, arrow_metadata) = ArrowReader::open_parquet_file(
            &task.data_file_path,
            &self.file_io,
            task.file_size_in_bytes,
            parquet_read_options,
            self.scan_metrics.bytes_read_counter(),
            self.data_bytes_cache.as_ref(),
        )
        .await?;

        // Check if Parquet file has embedded field IDs
        // Corresponds to Java's ParquetSchemaUtil.hasIds()
        // Reference: parquet/src/main/java/org/apache/iceberg/parquet/ParquetSchemaUtil.java:118
        let missing_field_ids = arrow_metadata
            .schema()
            .fields()
            .iter()
            .next()
            .is_some_and(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none());

        // Position-based fallback applies only when the file has no embedded field IDs
        // AND no name mapping is available. With a name mapping, field IDs are assigned
        // to the Arrow schema below, and projection/predicate planning must use them
        // (see #2403).
        let use_position_fallback = missing_field_ids && task.name_mapping.is_none();

        // Three-branch schema resolution strategy matching Java's ReadConf constructor
        //
        // Per Iceberg spec Column Projection rules:
        // "Columns in Iceberg data files are selected by field id. The table schema's column
        //  names and order may change after a data file is written, and projection must be done
        //  using field ids."
        // https://iceberg.apache.org/spec/#column-projection
        //
        // When Parquet files lack field IDs (e.g., Hive/Spark migrations via add_files),
        // we must assign field IDs BEFORE reading data to enable correct projection.
        //
        // Java's ReadConf determines field ID strategy:
        // - Branch 1: hasIds(fileSchema) → trust embedded field IDs, use pruneColumns()
        // - Branch 2: nameMapping present → applyNameMapping(), then pruneColumns()
        // - Branch 3: fallback → addFallbackIds(), then pruneColumnsFallback()
        let arrow_metadata = if missing_field_ids {
            // Parquet file lacks field IDs - must assign them before reading
            let arrow_schema = if let Some(name_mapping) = &task.name_mapping {
                // Branch 2: Apply name mapping to assign correct Iceberg field IDs
                // Per spec rule #2: "Use schema.name-mapping.default metadata to map field id
                // to columns without field id"
                // Corresponds to Java's ParquetSchemaUtil.applyNameMapping()
                apply_name_mapping_to_arrow_schema(
                    Arc::clone(arrow_metadata.schema()),
                    name_mapping,
                )?
            } else {
                // Branch 3: No name mapping - use position-based fallback IDs
                // Corresponds to Java's ParquetSchemaUtil.addFallbackIds()
                add_fallback_field_ids_to_arrow_schema(arrow_metadata.schema())
            };

            let options = ArrowReaderOptions::new().with_schema(arrow_schema);
            ArrowReaderMetadata::try_new(Arc::clone(arrow_metadata.metadata()), options).map_err(
                |e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to create ArrowReaderMetadata with field ID schema",
                    )
                    .with_source(e)
                },
            )?
        } else {
            // Branch 1: File has embedded field IDs - trust them
            arrow_metadata
        };

        // Coerce INT96 timestamp columns to the resolution specified by the Iceberg schema.
        // This must happen before building the stream reader to avoid i64 overflow in arrow-rs.
        let arrow_metadata = if let Some(coerced_schema) =
            coerce_int96_timestamps(arrow_metadata.schema(), &task.schema)
        {
            let options = ArrowReaderOptions::new().with_schema(Arc::clone(&coerced_schema));
            ArrowReaderMetadata::try_new(Arc::clone(arrow_metadata.metadata()), options).map_err(
                |e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Failed to create ArrowReaderMetadata with INT96-coerced schema: {coerced_schema}"
                        ),
                    )
                    .with_source(e)
                },
            )?
        } else {
            arrow_metadata
        };

        // Widen 32-bit-offset string/binary columns to their 64-bit forms so
        // decode materializes LargeUtf8/LargeBinary directly — a wide-TEXT
        // batch would overflow a 32-bit Utf8 array's 2GiB value cap before any
        // post-decode cast could run. Same schema-hint mechanism as INT96.
        let arrow_metadata = if let Some(widened_schema) =
            widen_variable_length_types(arrow_metadata.schema(), &task.schema)
        {
            let options = ArrowReaderOptions::new().with_schema(Arc::clone(&widened_schema));
            ArrowReaderMetadata::try_new(Arc::clone(arrow_metadata.metadata()), options).map_err(
                |e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Failed to create ArrowReaderMetadata with the large-offset schema: {widened_schema}"
                        ),
                    )
                    .with_source(e)
                },
            )?
        } else {
            arrow_metadata
        };

        // When `_pos` is projected, re-derive the reader metadata with a Parquet
        // virtual RowNumber column: the reader then emits each row's ORIGINAL
        // position within the data file (correct under row-group pruning, row
        // selection and row filters), which MoR writers need to construct
        // deletion vectors. The virtual field carries the reserved field id so
        // the RecordBatchTransformer can pass it through by id. Appended LAST,
        // after all file fields, so projection-mask leaf numbering is unaffected.
        let arrow_metadata = if task.project_field_ids().contains(&RESERVED_FIELD_ID_POS) {
            let pos_field = Arc::new(
                Field::new(RESERVED_COL_NAME_POS, DataType::Int64, false)
                    .with_metadata(std::collections::HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        RESERVED_FIELD_ID_POS.to_string(),
                    )]))
                    .with_extension_type(RowNumber),
            );
            let options = ArrowReaderOptions::new()
                .with_schema(Arc::clone(arrow_metadata.schema()))
                .with_virtual_columns(vec![pos_field])
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to register the _pos virtual column",
                    )
                    .with_source(e)
                })?;
            ArrowReaderMetadata::try_new(Arc::clone(arrow_metadata.metadata()), options).map_err(
                |e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to create ArrowReaderMetadata with the _pos virtual column",
                    )
                    .with_source(e)
                },
            )?
        } else {
            arrow_metadata
        };

        // Build the stream reader, reusing the already-opened file reader
        let mut record_batch_stream_builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(parquet_file_reader, arrow_metadata);

        // Filter out metadata fields for Parquet projection (they don't exist in files)
        let project_field_ids_without_metadata: Vec<i32> = task
            .project_field_ids
            .iter()
            .filter(|&&id| !is_metadata_field(id))
            .copied()
            .collect();

        // Create projection mask based on field IDs
        // - If file has embedded IDs: field-ID-based projection
        // - If name mapping applied: field-ID-based projection using the IDs the name
        //   mapping assigned to the Arrow schema
        // - Otherwise: position-based fallback projection
        let projection_mask = ArrowReader::get_arrow_projection_mask(
            &project_field_ids_without_metadata,
            &task.schema,
            record_batch_stream_builder.parquet_schema(),
            record_batch_stream_builder.schema(),
            use_position_fallback, // Whether to use position-based (true) or field-ID-based (false) projection
        )?;

        record_batch_stream_builder =
            record_batch_stream_builder.with_projection(projection_mask.clone());

        // RecordBatchTransformer performs any transformations required on the RecordBatches
        // that come back from the file, such as type promotion, default column insertion,
        // column re-ordering, partition constants, and virtual field addition (like _file)
        let mut record_batch_transformer_builder =
            RecordBatchTransformerBuilder::new(task.schema_ref(), task.project_field_ids());

        // Add the _file metadata column if it's in the projected fields
        if task.project_field_ids().contains(&RESERVED_FIELD_ID_FILE) {
            let file_datum = Datum::string(task.data_file_path.clone());
            record_batch_transformer_builder =
                record_batch_transformer_builder.with_constant(RESERVED_FIELD_ID_FILE, file_datum);
        }

        if let (Some(partition_spec), Some(partition_data)) =
            (task.partition_spec.clone(), task.partition.clone())
        {
            record_batch_transformer_builder =
                record_batch_transformer_builder.with_partition(partition_spec, partition_data)?;
        }

        // Shredded variant passthrough — decided PER FILE from its physical
        // schema: only a column whose on-disk type is byte-for-byte the
        // expected shredded type skips the fold; mixed estates (canonical
        // files alongside shredded ones) keep folding file-by-file.
        let mut fold_skip: HashSet<String> = HashSet::new();
        if let Some(pass) = &self.shredded_passthrough {
            let file_schema = record_batch_stream_builder.schema();
            for (name, expected) in pass.iter() {
                if let Ok(field) = file_schema.field_with_name(name)
                    && field.data_type() == expected
                    && let Some(iceberg_field) = task.schema.field_by_name(name)
                {
                    fold_skip.insert(name.clone());
                    record_batch_transformer_builder = record_batch_transformer_builder
                        .with_type_override(iceberg_field.id, expected.clone());
                }
            }
        }
        let fold_skip = Arc::new(fold_skip);

        let mut record_batch_transformer = record_batch_transformer_builder.build();

        if let Some(batch_size) = self.batch_size {
            record_batch_stream_builder = record_batch_stream_builder.with_batch_size(batch_size);
        }

        let delete_filter = delete_filter_rx.await.unwrap()?;
        let delete_predicate = delete_filter.build_equality_delete_predicate(&task).await?;

        // In addition to the optional predicate supplied in the `FileScanTask`,
        // we also have an optional predicate resulting from equality delete files.
        // If both are present, we logical-AND them together to form a single filter
        // predicate that we can pass to the `RecordBatchStreamBuilder`.
        let final_predicate = match (&task.predicate, delete_predicate) {
            (None, None) => None,
            (Some(predicate), None) => Some(predicate.clone()),
            (None, Some(ref predicate)) => Some(predicate.clone()),
            (Some(filter_predicate), Some(delete_predicate)) => {
                Some(filter_predicate.clone().and(delete_predicate))
            }
        };

        // There are three possible sources for potential lists of selected RowGroup indices,
        // and two for `RowSelection`s.
        // Selected RowGroup index lists can come from three sources:
        //   * When task.start and task.length specify a byte range (file splitting);
        //   * When there are equality delete files that are applicable;
        //   * When there is a scan predicate and row_group_filtering_enabled = true.
        // `RowSelection`s can be created in either or both of the following cases:
        //   * When there are positional delete files that are applicable;
        //   * When there is a scan predicate and row_selection_enabled = true
        // Note that row group filtering from predicates only happens when
        // there is a scan predicate AND row_group_filtering_enabled = true,
        // but we perform row selection filtering if there are applicable
        // equality delete files OR (there is a scan predicate AND row_selection_enabled),
        // since the only implemented method of applying positional deletes is
        // by using a `RowSelection`.
        let mut selected_row_group_indices = None;
        let mut row_selection = None;

        // Filter row groups based on byte range from task.start and task.length.
        // If both start and length are 0, read the entire file (backwards compatibility).
        if task.start != 0 || task.length != 0 {
            let byte_range_filtered_row_groups = ArrowReader::filter_row_groups_by_byte_range(
                record_batch_stream_builder.metadata(),
                task.start,
                task.length,
            )?;
            selected_row_group_indices = Some(byte_range_filtered_row_groups);
        }

        if let Some(predicate) = final_predicate {
            let (iceberg_field_ids, field_id_map) = ArrowReader::build_field_id_set_and_map(
                record_batch_stream_builder.parquet_schema(),
                record_batch_stream_builder.schema(),
                &predicate,
                use_position_fallback,
            )?;

            let row_filter = ArrowReader::get_row_filter(
                &predicate,
                record_batch_stream_builder.parquet_schema(),
                &iceberg_field_ids,
                &field_id_map,
            )?;
            record_batch_stream_builder = record_batch_stream_builder.with_row_filter(row_filter);

            if self.row_group_filtering_enabled {
                let predicate_filtered_row_groups = ArrowReader::get_selected_row_group_indices(
                    &predicate,
                    record_batch_stream_builder.metadata(),
                    &field_id_map,
                    &task.schema,
                )?;

                // Merge predicate-based filtering with byte range filtering (if present)
                // by taking the intersection of both filters
                selected_row_group_indices = match selected_row_group_indices {
                    Some(byte_range_filtered) => {
                        // Keep only row groups that are in both filters
                        let intersection: Vec<usize> = byte_range_filtered
                            .into_iter()
                            .filter(|idx| predicate_filtered_row_groups.contains(idx))
                            .collect();
                        Some(intersection)
                    }
                    None => Some(predicate_filtered_row_groups),
                };
            }

            if self.row_selection_enabled {
                row_selection = ArrowReader::get_row_selection_for_filter_predicate(
                    &predicate,
                    record_batch_stream_builder.metadata(),
                    &selected_row_group_indices,
                    &field_id_map,
                    &task.schema,
                )?;
            }
        }

        let positional_delete_indexes = delete_filter.get_delete_vector(&task);

        if let Some(positional_delete_indexes) = positional_delete_indexes {
            let delete_row_selection = {
                let positional_delete_indexes = positional_delete_indexes.lock().unwrap();

                ArrowReader::build_deletes_row_selection(
                    record_batch_stream_builder.metadata().row_groups(),
                    &selected_row_group_indices,
                    &positional_delete_indexes,
                )
            }?;

            // merge the row selection from the delete files with the row selection
            // from the filter predicate, if there is one from the filter predicate
            row_selection = match row_selection {
                None => Some(delete_row_selection),
                Some(filter_row_selection) => {
                    Some(filter_row_selection.intersection(&delete_row_selection))
                }
            };
        }

        if let Some(row_selection) = row_selection {
            record_batch_stream_builder =
                record_batch_stream_builder.with_row_selection(row_selection);
        }

        if let Some(selected_row_group_indices) = selected_row_group_indices {
            record_batch_stream_builder =
                record_batch_stream_builder.with_row_groups(selected_row_group_indices);
        }

        // Build the batch stream and send all the RecordBatches that it generates
        // to the requester.
        let record_batch_stream =
            record_batch_stream_builder
                .build()?
                .map(move |batch| match batch {
                    Ok(batch) => {
                        // Fold any shredded variant columns (`typed_value`) back into a
                        // plain `{metadata, value}` variant before the transformer maps
                        // columns by field id (which expects the canonical variant type).
                        // Columns in `fold_skip` (shredded passthrough) stay in their
                        // physical shredded shape; the transformer targets that type.
                        let batch = unshred_variant_columns(batch, &fold_skip)?;
                        // Process the record batch (type promotion, column reordering, virtual fields, etc.)
                        record_batch_transformer.process_record_batch(batch)
                    }
                    Err(err) => Err(err.into()),
                });

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }
}

impl ArrowReader {
    /// Opens a Parquet file and loads its metadata, wrapping the reader with
    /// [`CountingFileRead`] so all I/O is accumulated into `bytes_read`.
    ///
    /// With a [`DataBytesCache`] configured and the file within its size
    /// cap, the WHOLE object is read through the cache once (one storage GET
    /// on first touch) and every ranged read — footer, column chunks,
    /// row-group byte ranges — is served from the local copy. Larger files
    /// (or no cache) read directly from storage, byte-identical to the
    /// uncached path.
    pub(crate) async fn open_parquet_file(
        data_file_path: &str,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
        bytes_read: &Arc<AtomicU64>,
        data_bytes_cache: Option<&DataBytesCache>,
    ) -> Result<(ArrowFileReader, ArrowReaderMetadata)> {
        if let Some(dc) = data_bytes_cache
            && file_size_in_bytes > 0
            && file_size_in_bytes <= dc.max_file_bytes
        {
            let bytes = match dc.cache.get(data_file_path).await {
                Some(bytes) => bytes,
                None => {
                    // ONE whole-object fetch replaces the scan's N ranged
                    // reads even on first touch; files are immutable by
                    // path, so the copy never needs invalidation.
                    let bytes = file_io.new_input(data_file_path)?.read().await?;
                    dc.cache.set(data_file_path, bytes.clone()).await;
                    bytes
                }
            };
            // Serve from the ACTUAL byte length — authoritative over the
            // manifest-recorded size for footer location.
            let actual_size = bytes.len() as u64;
            let counting_reader = CountingFileRead::new(
                Box::new(CachedWholeFileRead { bytes }) as Box<dyn FileRead>,
                Arc::clone(bytes_read),
            );
            return Self::build_parquet_reader(
                Box::new(counting_reader),
                actual_size,
                parquet_read_options,
            )
            .await;
        }

        let parquet_file = file_io.new_input(data_file_path)?;
        let counting_reader =
            CountingFileRead::new(parquet_file.reader().await?, Arc::clone(bytes_read));
        Self::build_parquet_reader(
            Box::new(counting_reader),
            file_size_in_bytes,
            parquet_read_options,
        )
        .await
    }

    async fn build_parquet_reader(
        parquet_reader: Box<dyn FileRead>,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
    ) -> Result<(ArrowFileReader, ArrowReaderMetadata)> {
        let mut reader = ArrowFileReader::new(
            FileMetadata {
                size: file_size_in_bytes,
            },
            parquet_reader,
        )
        .with_parquet_read_options(parquet_read_options);

        let arrow_metadata = ArrowReaderMetadata::load_async(&mut reader, Default::default())
            .await
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata").with_source(e)
            })?;

        Ok((reader, arrow_metadata))
    }
}

/// A fully-materialized file served from cached bytes: every ranged read is
/// a zero-copy slice of the local copy.
struct CachedWholeFileRead {
    bytes: bytes::Bytes,
}

#[async_trait::async_trait]
impl FileRead for CachedWholeFileRead {
    async fn read(&self, range: std::ops::Range<u64>) -> Result<bytes::Bytes> {
        let (start, end) = (range.start as usize, range.end as usize);
        if start > end || end > self.bytes.len() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "range {start}..{end} out of bounds for cached file of {} bytes",
                    self.bytes.len()
                ),
            ));
        }
        Ok(self.bytes.slice(start..end))
    }
}

/// Fold any SHREDDED top-level variant columns back into the canonical unshredded
/// `Struct([metadata: Binary, value: Binary])` shape.
///
/// A shredded variant column is physically a `StructArray` carrying a `typed_value`
/// sub-field alongside `metadata` (and optionally `value`). The downstream
/// `RecordBatchTransformer` maps columns by Parquet field id to iceberg's canonical
/// variant Arrow type (`Struct([metadata: Binary, value: Binary])`, no field ids on
/// the sub-fields). It can't consume the shredded physical layout, so we run
/// arrow-rs's tested [`unshred_variant`] kernel here, which folds `typed_value`
/// (including nested) back into `value`.
///
/// Columns that are NOT shredded variants pass through untouched: an unshredded
/// variant (`metadata` + `value`, no `typed_value`) and any other struct/primitive
/// are left exactly as read.
///
/// The replacement column preserves the original top-level field's name,
/// `PARQUET_FIELD_ID_META_KEY` metadata, and nullability — only the inner struct
/// changes (drop `typed_value`, force `metadata`/`value` to `Binary`).
///
/// TODO: nested variant columns (a variant nested inside another Arrow struct/list)
/// are not unshredded here — only top-level variant columns are handled.
fn unshred_variant_columns(batch: RecordBatch, skip: &HashSet<String>) -> Result<RecordBatch> {
    let schema = batch.schema();

    // Detect which columns need unshredding before allocating anything.
    // A column in `skip` (shredded passthrough) is deliberately left in its
    // physical shredded shape.
    let needs_unshred = |field: &Field| -> bool {
        !skip.contains(field.name())
            && matches!(field.data_type(), DataType::Struct(sub)
            if sub.iter().any(|f| f.name() == "metadata")
                && sub.iter().any(|f| f.name() == "typed_value"))
    };

    if !schema.fields().iter().any(|f| needs_unshred(f)) {
        return Ok(batch);
    }

    let mut new_fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

    for (idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(idx);
        if needs_unshred(field) {
            // `unshred_variant` returns a variant whose `metadata`/`value` are
            // BinaryView; cast them to Binary to match iceberg's canonical type.
            let variant = VariantArray::try_new(col.as_ref())?;
            let unshredded = unshred_variant(&variant)?;
            let inner: StructArray = unshredded.into_inner();

            let metadata = inner
                .column_by_name("metadata")
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "unshredded variant is missing its 'metadata' field",
                    )
                })?
                .clone();
            let value = inner
                .column_by_name("value")
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "unshredded variant is missing its 'value' field",
                    )
                })?
                .clone();

            let metadata = arrow_cast::cast(metadata.as_ref(), &DataType::Binary)?;
            let value = arrow_cast::cast(value.as_ref(), &DataType::Binary)?;

            // Canonical variant sub-fields: metadata + value, both Binary,
            // non-null, NO field ids (variant internals).
            let sub_fields = Fields::from(vec![
                Field::new("metadata", DataType::Binary, false),
                Field::new("value", DataType::Binary, false),
            ]);
            let new_struct = StructArray::new(
                sub_fields.clone(),
                vec![metadata, value],
                // Preserve the top-level column's null buffer.
                col.nulls().cloned(),
            );

            // Keep the original top-level field's name + field-id metadata + nullability.
            let new_field = Field::new(
                field.name(),
                DataType::Struct(sub_fields),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone());

            new_fields.push(new_field);
            new_columns.push(Arc::new(new_struct) as ArrayRef);
        } else {
            new_fields.push(field.as_ref().clone());
            new_columns.push(Arc::clone(col));
        }
    }

    let new_schema =
        Arc::new(ArrowSchema::new(new_fields).with_metadata(schema.metadata().clone()));
    Ok(RecordBatch::try_new(new_schema, new_columns)?)
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{Array, ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use crate::Runtime;
    use crate::arrow::ArrowReaderBuilder;
    use crate::io::FileIO;
    use crate::scan::{FileScanTask, FileScanTaskStream};
    use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    // INT96 encoding: [nanos_low_u32, nanos_high_u32, julian_day_u32]
    // Julian day 2_440_588 = Unix epoch (1970-01-01)
    const UNIX_EPOCH_JULIAN: i64 = 2_440_588;
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    // Noon on 3333-01-01 (Julian day 2_953_529) — outside the i64 nanosecond range (~1677-2262).
    const INT96_TEST_NANOS_WITHIN_DAY: u64 = 43_200_000_000_000;
    const INT96_TEST_JULIAN_DAY: u32 = 2_953_529;

    fn make_int96_test_value() -> (parquet::data_type::Int96, i64) {
        let mut val = parquet::data_type::Int96::new();
        val.set_data(
            (INT96_TEST_NANOS_WITHIN_DAY & 0xFFFFFFFF) as u32,
            (INT96_TEST_NANOS_WITHIN_DAY >> 32) as u32,
            INT96_TEST_JULIAN_DAY,
        );
        let expected_micros = (INT96_TEST_JULIAN_DAY as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY
            + (INT96_TEST_NANOS_WITHIN_DAY / 1_000) as i64;
        (val, expected_micros)
    }

    async fn read_int96_batches(
        file_path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
    ) -> Vec<RecordBatch> {
        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let file_size = std::fs::metadata(file_path).unwrap().len();
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(file_size)
            .with_start(0)
            .with_length(file_size)
            .with_data_file_path(file_path.to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema)
            .with_project_field_ids(project_field_ids)
            .with_case_sensitive(false)
            .build();

        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap()
    }

    // ArrowWriter cannot write INT96, so we use SerializedFileWriter directly.
    fn write_int96_parquet_file(
        table_location: &str,
        filename: &str,
        with_field_ids: bool,
    ) -> (String, Vec<i64>) {
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::{Int32Type, Int96, Int96Type};
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let file_path = format!("{table_location}/{filename}");

        let mut ts_builder = SchemaType::primitive_type_builder("ts", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL);
        let mut id_builder = SchemaType::primitive_type_builder("id", PhysicalType::INT32)
            .with_repetition(Repetition::REQUIRED);

        if with_field_ids {
            ts_builder = ts_builder.with_id(Some(1));
            id_builder = id_builder.with_id(Some(2));
        }

        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(ts_builder.build().unwrap()),
                Arc::new(id_builder.build().unwrap()),
            ])
            .build()
            .unwrap();

        // Dates outside the i64 nanosecond range (~1677-2262) overflow without coercion.
        const NOON_NANOS: u64 = INT96_TEST_NANOS_WITHIN_DAY;
        const JULIAN_3333: u32 = INT96_TEST_JULIAN_DAY;
        const JULIAN_2100: u32 = 2_488_070;

        let test_data: Vec<(u32, u32, u32, i64)> = vec![
            // 3333-01-01 00:00:00
            (
                0,
                0,
                JULIAN_3333,
                (JULIAN_3333 as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY,
            ),
            // 3333-01-01 12:00:00
            (
                (NOON_NANOS & 0xFFFFFFFF) as u32,
                (NOON_NANOS >> 32) as u32,
                JULIAN_3333,
                (JULIAN_3333 as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY
                    + (NOON_NANOS / 1_000) as i64,
            ),
            // 2100-01-01 00:00:00
            (
                0,
                0,
                JULIAN_2100,
                (JULIAN_2100 as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY,
            ),
        ];

        let int96_values: Vec<Int96> = test_data
            .iter()
            .map(|(lo, hi, day, _)| {
                let mut v = Int96::new();
                v.set_data(*lo, *hi, *day);
                v
            })
            .collect();

        let id_values: Vec<i32> = (0..test_data.len() as i32).collect();
        let expected_micros: Vec<i64> = test_data.iter().map(|(_, _, _, m)| *m).collect();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(schema), Default::default()).unwrap();

        let mut row_group = writer.next_row_group().unwrap();
        {
            // def=1: ts is OPTIONAL and present. No repetition levels (top-level columns).
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&int96_values, Some(&vec![1; test_data.len()]), None)
                .unwrap();
            col.close().unwrap();
        }
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int32Type>()
                .write_batch(&id_values, None, None)
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        (file_path, expected_micros)
    }

    async fn assert_int96_read_matches(
        file_path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
        expected_micros: &[i64],
    ) {
        use arrow_array::TimestampMicrosecondArray;

        let batches = read_int96_batches(file_path, schema, project_field_ids).await;

        assert_eq!(batches.len(), 1);
        let ts_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray");

        for (i, expected) in expected_micros.iter().enumerate() {
            assert_eq!(
                ts_array.value(i),
                *expected,
                "Row {i}: got {}, expected {expected}",
                ts_array.value(i)
            );
        }
    }

    /// Test that concurrency=1 reads all files correctly and in deterministic order.
    /// This verifies the fast-path optimization for single concurrency.
    #[tokio::test]
    async fn test_read_with_concurrency_one() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "file_num", Type::Primitive(PrimitiveType::Int))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("file_num", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        // Create 3 parquet files with different data
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        for file_num in 0..3 {
            let id_data = Arc::new(Int32Array::from_iter_values(
                file_num * 10..(file_num + 1) * 10,
            )) as ArrayRef;
            let file_num_data = Arc::new(Int32Array::from(vec![file_num; 10])) as ArrayRef;

            let to_write =
                RecordBatch::try_new(arrow_schema.clone(), vec![id_data, file_num_data]).unwrap();

            let file = File::create(format!("{table_location}/file_{file_num}.parquet")).unwrap();
            let mut writer =
                ArrowWriter::try_new(file, to_write.schema(), Some(props.clone())).unwrap();
            writer.write(&to_write).expect("Writing batch");
            writer.close().unwrap();
        }

        // Read with concurrency=1 (fast-path)
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current())
            .with_data_file_concurrency_limit(1)
            .build();

        // Create tasks in a specific order: file_0, file_1, file_2
        let tasks = vec![
            Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/file_0.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/file_0.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .build()),
            Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/file_1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/file_1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .build()),
            Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/file_2.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/file_2.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .build()),
        ];

        let tasks_stream = Box::pin(futures::stream::iter(tasks)) as FileScanTaskStream;

        let result = reader
            .read(tasks_stream)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        // Verify we got all 30 rows (10 from each file)
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 30, "Should have 30 total rows");

        // Collect all ids and file_nums to verify data
        let mut all_ids = Vec::new();
        let mut all_file_nums = Vec::new();

        for batch in &result {
            let id_col = batch
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let file_num_col = batch
                .column(1)
                .as_primitive::<arrow_array::types::Int32Type>();

            for i in 0..batch.num_rows() {
                all_ids.push(id_col.value(i));
                all_file_nums.push(file_num_col.value(i));
            }
        }

        assert_eq!(all_ids.len(), 30);
        assert_eq!(all_file_nums.len(), 30);

        // With concurrency=1 and sequential processing, files should be processed in order
        // file_0: ids 0-9, file_num=0
        // file_1: ids 10-19, file_num=1
        // file_2: ids 20-29, file_num=2
        for i in 0..10 {
            assert_eq!(all_file_nums[i], 0, "First 10 rows should be from file_0");
            assert_eq!(all_ids[i], i as i32, "IDs should be 0-9");
        }
        for i in 10..20 {
            assert_eq!(all_file_nums[i], 1, "Next 10 rows should be from file_1");
            assert_eq!(all_ids[i], i as i32, "IDs should be 10-19");
        }
        for i in 20..30 {
            assert_eq!(all_file_nums[i], 2, "Last 10 rows should be from file_2");
            assert_eq!(all_ids[i], i as i32, "IDs should be 20-29");
        }
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_with_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let (file_path, expected_micros) =
            write_int96_parquet_file(&table_location, "with_ids.parquet", true);

        assert_int96_read_matches(&file_path, schema, vec![1, 2], &expected_micros).await;
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_without_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let (file_path, expected_micros) =
            write_int96_parquet_file(&table_location, "no_ids.parquet", false);

        assert_int96_read_matches(&file_path, schema, vec![1, 2], &expected_micros).await;
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_in_struct() {
        use arrow_array::{StructArray, TimestampMicrosecondArray};
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::Int96Type;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/struct_int96.parquet");

        let ts_type = SchemaType::primitive_type_builder("ts", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(2))
            .build()
            .unwrap();

        let struct_type = SchemaType::group_type_builder("data")
            .with_repetition(Repetition::REQUIRED)
            .with_id(Some(1))
            .with_fields(vec![Arc::new(ts_type)])
            .build()
            .unwrap();

        let parquet_schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(struct_type)])
            .build()
            .unwrap();

        let (int96_val, expected_micros) = make_int96_test_value();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(parquet_schema), Default::default()).unwrap();

        // def=1: struct is REQUIRED so no level, ts is OPTIONAL and present (1).
        // No repetition levels needed (no repeated groups).
        let mut row_group = writer.next_row_group().unwrap();
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&[int96_val], Some(&[1]), None)
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "data",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::optional(
                                2,
                                "ts",
                                Type::Primitive(PrimitiveType::Timestamp),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let batches = read_int96_batches(&file_path, iceberg_schema, vec![1]).await;

        assert_eq!(batches.len(), 1);
        let struct_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("Expected StructArray");
        let ts_array = struct_array
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray inside struct");

        assert_eq!(
            ts_array.value(0),
            expected_micros,
            "INT96 in struct: got {}, expected {expected_micros}",
            ts_array.value(0)
        );
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_in_list() {
        use arrow_array::{ListArray, TimestampMicrosecondArray};
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::Int96Type;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/list_int96.parquet");

        // 3-level LIST encoding:
        //   optional group timestamps (LIST) {
        //     repeated group list {
        //       optional int96 element;
        //     }
        //   }
        let element_type = SchemaType::primitive_type_builder("element", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(2))
            .build()
            .unwrap();

        let list_group = SchemaType::group_type_builder("list")
            .with_repetition(Repetition::REPEATED)
            .with_fields(vec![Arc::new(element_type)])
            .build()
            .unwrap();

        let list_type = SchemaType::group_type_builder("timestamps")
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(1))
            .with_logical_type(Some(parquet::basic::LogicalType::List))
            .with_fields(vec![Arc::new(list_group)])
            .build()
            .unwrap();

        let parquet_schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(list_type)])
            .build()
            .unwrap();

        let (int96_val, expected_micros) = make_int96_test_value();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(parquet_schema), Default::default()).unwrap();

        // Write a single row with a list containing one INT96 element.
        // def=3: list present (1) + repeated group (2) + element present (3)
        // rep=0: start of a new list
        let mut row_group = writer.next_row_group().unwrap();
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&[int96_val], Some(&[3]), Some(&[0]))
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(
                        1,
                        "timestamps",
                        Type::List(crate::spec::ListType {
                            element_field: NestedField::optional(
                                2,
                                "element",
                                Type::Primitive(PrimitiveType::Timestamp),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let batches = read_int96_batches(&file_path, iceberg_schema, vec![1]).await;

        assert_eq!(batches.len(), 1);
        let list_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray");
        let ts_array = list_array
            .values()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray inside list");

        assert_eq!(
            ts_array.value(0),
            expected_micros,
            "INT96 in list: got {}, expected {expected_micros}",
            ts_array.value(0)
        );
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_in_map() {
        use arrow_array::{MapArray, TimestampMicrosecondArray};
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::{ByteArrayType, Int96Type};
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/map_int96.parquet");

        // MAP encoding:
        //   optional group ts_map (MAP) {
        //     repeated group key_value {
        //       required binary key (UTF8);
        //       optional int96 value;
        //     }
        //   }
        let key_type = SchemaType::primitive_type_builder("key", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(Some(parquet::basic::LogicalType::String))
            .with_id(Some(2))
            .build()
            .unwrap();

        let value_type = SchemaType::primitive_type_builder("value", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(3))
            .build()
            .unwrap();

        let key_value_group = SchemaType::group_type_builder("key_value")
            .with_repetition(Repetition::REPEATED)
            .with_fields(vec![Arc::new(key_type), Arc::new(value_type)])
            .build()
            .unwrap();

        let map_type = SchemaType::group_type_builder("ts_map")
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(1))
            .with_logical_type(Some(parquet::basic::LogicalType::Map))
            .with_fields(vec![Arc::new(key_value_group)])
            .build()
            .unwrap();

        let parquet_schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(map_type)])
            .build()
            .unwrap();

        let (int96_val, expected_micros) = make_int96_test_value();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(parquet_schema), Default::default()).unwrap();

        // Write a single row with a map containing one key-value pair.
        // rep=0 for both columns: start of a new map.
        // key def=2: map present (1) + key_value entry present (2), key is REQUIRED.
        // value def=3: map present (1) + key_value entry present (2) + value present (3).
        let mut row_group = writer.next_row_group().unwrap();
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<ByteArrayType>()
                .write_batch(
                    &[parquet::data_type::ByteArray::from("event_time")],
                    Some(&[2]),
                    Some(&[0]),
                )
                .unwrap();
            col.close().unwrap();
        }
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&[int96_val], Some(&[3]), Some(&[0]))
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(
                        1,
                        "ts_map",
                        Type::Map(crate::spec::MapType {
                            key_field: NestedField::required(
                                2,
                                "key",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            value_field: NestedField::optional(
                                3,
                                "value",
                                Type::Primitive(PrimitiveType::Timestamp),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let batches = read_int96_batches(&file_path, iceberg_schema, vec![1]).await;

        assert_eq!(batches.len(), 1);
        let map_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("Expected MapArray");
        let ts_array = map_array
            .values()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray as map values");

        assert_eq!(
            ts_array.value(0),
            expected_micros,
            "INT96 in map: got {}, expected {expected_micros}",
            ts_array.value(0)
        );
    }

    /// A SHREDDED variant column (`{metadata, value, typed_value}`) is folded back
    /// into the canonical unshredded `Struct([metadata: Binary, value: Binary])`,
    /// dropping `typed_value`, preserving row count + field id + name, and the
    /// folded `value` decodes to the scalar that was in `typed_value`.
    #[test]
    fn test_unshred_variant_columns_folds_typed_value() {
        use arrow_array::{BinaryArray, Int32Array, StructArray};
        use parquet::variant::{EMPTY_VARIANT_METADATA_BYTES, Variant, VariantArray};

        use super::unshred_variant_columns;

        // Valid empty-variant metadata header (empty dictionary), one per row.
        let empty_meta = EMPTY_VARIANT_METADATA_BYTES;
        let metadata = Arc::new(BinaryArray::from(vec![empty_meta, empty_meta])) as ArrayRef;
        // Shredded: the payload lives in typed_value, so value is null for both rows.
        let value = Arc::new(BinaryArray::from(vec![None as Option<&[u8]>, None])) as ArrayRef;
        let typed_value = Arc::new(Int32Array::from(vec![7, 42])) as ArrayRef;

        let shredded_fields = arrow_schema::Fields::from(vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::Binary, true),
            Field::new("typed_value", DataType::Int32, true),
        ]);
        let shredded = Arc::new(StructArray::new(
            shredded_fields.clone(),
            vec![metadata, value, typed_value],
            None,
        )) as ArrayRef;

        // Top-level variant field carries name "v", field id 2, non-nullable.
        let v_field = Field::new("v", DataType::Struct(shredded_fields), false).with_metadata(
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]),
        );
        let id = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )]));
        let schema = Arc::new(ArrowSchema::new(vec![id_field, v_field]));
        let batch = RecordBatch::try_new(schema, vec![id, shredded]).unwrap();

        let out = unshred_variant_columns(batch, &HashSet::new()).expect("unshred must succeed");
        assert_eq!(out.num_rows(), 2);

        // Sibling 'id' column passes through untouched.
        let out_id = out
            .column_by_name("id")
            .expect("'id' dropped")
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(out_id.values(), &[1, 2]);

        // The variant column is now unshredded: Struct(metadata: Binary, value: Binary), no typed_value.
        let v = out.column_by_name("v").expect("'v' dropped");
        let v_field_out = out.schema().field_with_name("v").unwrap().clone();
        assert!(!v_field_out.is_nullable());
        assert_eq!(
            v_field_out.metadata().get(PARQUET_FIELD_ID_META_KEY),
            Some(&"2".to_string()),
            "variant field id must be preserved"
        );
        let s = v.as_struct();
        assert_eq!(
            s.fields().len(),
            2,
            "typed_value must be folded away: {:?}",
            s.fields()
        );
        assert!(s.column_by_name("typed_value").is_none());
        let m = s.column_by_name("metadata").expect("missing metadata");
        let val = s.column_by_name("value").expect("missing value");
        assert_eq!(m.data_type(), &DataType::Binary);
        assert_eq!(val.data_type(), &DataType::Binary);

        // The folded value decodes back to the scalars that were in typed_value.
        let va = VariantArray::try_new(v.as_ref()).expect("reparse unshredded variant");
        assert_eq!(va.value(0), Variant::from(7i32));
        assert_eq!(va.value(1), Variant::from(42i32));
    }

    /// An UNshredded variant column (`{metadata, value}`, no `typed_value`) and a
    /// plain non-variant struct both pass through `unshred_variant_columns` untouched.
    #[test]
    fn test_unshred_variant_columns_passthrough_unshredded() {
        use arrow_array::{BinaryArray, Int32Array, StructArray};

        use super::unshred_variant_columns;

        let metadata = Arc::new(BinaryArray::from(vec![&b"\x01"[..]])) as ArrayRef;
        let value = Arc::new(BinaryArray::from(vec![&b"\x0a"[..]])) as ArrayRef;
        let unshredded_fields = arrow_schema::Fields::from(vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::Binary, false),
        ]);
        let v = Arc::new(StructArray::new(
            unshredded_fields.clone(),
            vec![metadata, value],
            None,
        )) as ArrayRef;
        let v_field = Field::new("v", DataType::Struct(unshredded_fields), false);

        // A plain struct that happens to NOT be a variant (no metadata field).
        let plain_fields =
            arrow_schema::Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let plain = Arc::new(StructArray::new(
            plain_fields.clone(),
            vec![Arc::new(Int32Array::from(vec![9])) as ArrayRef],
            None,
        )) as ArrayRef;
        let plain_field = Field::new("p", DataType::Struct(plain_fields), false);

        let schema = Arc::new(ArrowSchema::new(vec![v_field, plain_field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![v, plain]).unwrap();

        let out =
            unshred_variant_columns(batch, &HashSet::new()).expect("passthrough must succeed");
        // Schema is byte-for-byte identical: nothing was rewritten.
        assert_eq!(out.schema(), schema);
        assert_eq!(out.num_rows(), 1);
    }

    /// A SHREDDED variant column named in the skip set (shredded passthrough)
    /// is NOT folded — it keeps its physical `{metadata, value, typed_value}`
    /// shape byte-for-byte, while a sibling shredded column NOT in the skip
    /// set still folds to canonical.
    #[test]
    fn test_unshred_variant_columns_skip_set_passthrough() {
        use arrow_array::{BinaryArray, Int32Array, StructArray};
        use parquet::variant::EMPTY_VARIANT_METADATA_BYTES;

        use super::unshred_variant_columns;

        let make_shredded = || {
            let empty_meta = EMPTY_VARIANT_METADATA_BYTES;
            let metadata = Arc::new(BinaryArray::from(vec![empty_meta])) as ArrayRef;
            let value = Arc::new(BinaryArray::from(vec![None as Option<&[u8]>])) as ArrayRef;
            let typed_value = Arc::new(Int32Array::from(vec![7])) as ArrayRef;
            let fields = arrow_schema::Fields::from(vec![
                Field::new("metadata", DataType::Binary, false),
                Field::new("value", DataType::Binary, true),
                Field::new("typed_value", DataType::Int32, true),
            ]);
            let arr = Arc::new(StructArray::new(
                fields.clone(),
                vec![metadata, value, typed_value],
                None,
            )) as ArrayRef;
            (arr, fields)
        };

        let (kept, kept_fields) = make_shredded();
        let (folded, folded_fields) = make_shredded();
        let kept_field = Field::new("kept", DataType::Struct(kept_fields.clone()), false);
        let folded_field = Field::new("folded", DataType::Struct(folded_fields), false);
        let schema = Arc::new(ArrowSchema::new(vec![kept_field, folded_field]));
        let batch = RecordBatch::try_new(schema, vec![kept, folded]).unwrap();

        let skip: HashSet<String> = HashSet::from(["kept".to_string()]);
        let out = unshred_variant_columns(batch, &skip).expect("skip fold must succeed");

        // "kept" retains its shredded physical type.
        let kept_out = out.schema().field_with_name("kept").unwrap().clone();
        assert_eq!(kept_out.data_type(), &DataType::Struct(kept_fields));
        assert!(
            out.column_by_name("kept")
                .unwrap()
                .as_struct()
                .column_by_name("typed_value")
                .is_some(),
            "skip-set column must keep typed_value"
        );

        // "folded" was folded to canonical {metadata, value}.
        let folded_out = out.column_by_name("folded").unwrap().as_struct();
        assert!(folded_out.column_by_name("typed_value").is_none());
        assert_eq!(folded_out.fields().len(), 2);
    }
}
