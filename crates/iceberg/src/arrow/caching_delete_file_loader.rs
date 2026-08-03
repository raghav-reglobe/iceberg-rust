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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, LargeStringArray, StringArray, StructArray};
use futures::{StreamExt, TryStreamExt};
use tokio::sync::oneshot::{Receiver, channel};

use super::delete_filter::{DeleteFilter, PosDelLoadAction};
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::arrow::scan_metrics::ScanMetrics;
use crate::arrow::{arrow_primitive_to_literal, arrow_schema_to_schema};
use crate::delete_vector::DeleteVector;
use crate::io::FileIO;
use crate::puffin::PuffinReader;
use crate::runtime::Runtime;
use crate::scan::{ArrowRecordBatchStream, FileScanTaskDeleteFile};
use crate::spec::{
    DataContentType, DataFileFormat, Datum, ListType, MapType, NestedField, NestedFieldRef,
    PartnerAccessor, PrimitiveType, Schema, SchemaRef, SchemaWithPartnerVisitor, StructType, Type,
    VariantType, visit_schema_with_partner,
};
use crate::{Error, ErrorKind, Result};

/// A composite key for equality delete lookups. Each element corresponds to one
/// equality_id column. For single-column deletes this contains one element.
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub(crate) struct EqDeleteKey(pub(crate) Vec<Option<Datum>>);

/// Bundles the hash set of delete keys with the field metadata needed to extract
/// matching keys from data record batches.
#[derive(Debug, Clone)]
pub(crate) struct EqDeleteSet {
    /// Delete key tuples to filter out of data batches.
    pub(crate) keys: HashSet<EqDeleteKey>,
    /// Ordered list of (field_name, field_id) used to locate the key columns in
    /// data record batches. The order matches the element order in `EqDeleteKey`.
    pub(crate) fields: Vec<(String, i32)>,
}

impl EqDeleteSet {
    fn new(fields: Vec<(String, i32)>) -> Self {
        Self {
            keys: HashSet::new(),
            fields,
        }
    }

    /// Returns true when the set contains no delete keys.
    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Merge another set (with the same field layout) into this one.
    pub(crate) fn union(&mut self, other: &EqDeleteSet) {
        self.keys.extend(other.keys.iter().cloned());
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CachingDeleteFileLoader {
    basic_delete_file_loader: BasicDeleteFileLoader,
    concurrency_limit_data_files: usize,
    /// Shared filter state to allow caching loaded deletes across multiple
    /// calls to `load_deletes` (e.g., across multiple file scan tasks).
    delete_filter: DeleteFilter,
    runtime: Runtime,
}

// Intermediate context during processing of a delete file task.
enum DeleteFileContext {
    ExistingEqDel,
    ExistingPosDel,
    PosDels {
        file_path: String,
        stream: ArrowRecordBatchStream,
    },
    /// A V3 deletion vector; the blob located by `content_offset`/`content_size`
    /// is read and parsed into a `DeleteVector` in the parse phase.
    DelVec {
        file_path: String,
        referenced_data_file: String,
        content_offset: u64,
        content_size: u64,
        file_io: FileIO,
    },
    FreshEqDel {
        batch_stream: ArrowRecordBatchStream,
        equality_ids: HashSet<i32>,
        sender: tokio::sync::oneshot::Sender<Arc<EqDeleteSet>>,
    },
}

// Final result of the processing of a delete file task before
// results are fully merged into the DeleteFileManager's state
enum ParsedDeleteFileContext {
    DelVecs {
        file_path: String,
        /// The load-unit identity to finish: `Some(blob offset)` for a V3
        /// deletion vector (a Puffin container may pack multiple blobs, each
        /// an independent load unit), `None` for a whole-file Parquet
        /// positional-delete stream. Must match the key used at
        /// `try_start_pos_del_load`.
        content_offset: Option<u64>,
        results: HashMap<String, DeleteVector>,
    },
    EqDel,
    ExistingPosDel,
}

#[allow(unused_variables)]
impl CachingDeleteFileLoader {
    pub(crate) fn new(
        file_io: FileIO,
        concurrency_limit_data_files: usize,
        runtime: Runtime,
    ) -> Self {
        let scan_metrics = ScanMetrics::new();
        CachingDeleteFileLoader {
            basic_delete_file_loader: BasicDeleteFileLoader::new(file_io, scan_metrics),
            concurrency_limit_data_files,
            delete_filter: DeleteFilter::new(runtime.clone()),
            runtime,
        }
    }

    pub(crate) fn with_scan_metrics(mut self, scan_metrics: ScanMetrics) -> Self {
        self.basic_delete_file_loader = BasicDeleteFileLoader::new(
            self.basic_delete_file_loader.file_io().clone(),
            scan_metrics,
        );
        self
    }

    /// Initiates loading of all deletes for all the specified tasks
    ///
    /// Returned future completes once all positional deletes and delete vectors
    /// have loaded. EQ deletes are not waited for in this method but the returned
    /// DeleteFilter will await their loading when queried for them.
    ///
    ///  * Create a single stream of all delete file tasks irrespective of type,
    ///    so that we can respect the combined concurrency limit
    ///  * We then process each in two phases: load and parse.
    ///  * for positional deletes the load phase instantiates an ArrowRecordBatchStream to
    ///    stream the file contents out
    ///  * for eq deletes, we first check if the EQ delete is already loaded or being loaded by
    ///    another concurrently processing data file scan task. If it is, we skip it.
    ///    If not, the DeleteFilter is updated to contain a notifier to prevent other data file
    ///    tasks from starting to load the same equality delete file. We spawn a task to load
    ///    the EQ delete's record batch stream, convert it to an `EqDeleteSet` (hash set of
    ///    delete key tuples), update the delete filter, and notify any task that was waiting
    ///    for it.
    ///  * for delete vectors (V3, stored in Puffin) the load phase records the Puffin file
    ///    path + referenced data file; the parse phase reads the deletion-vector-v1 blob.
    ///  * The parse phase parses each record batch stream according to its associated data type.
    ///    The result of this is a map of data file paths to delete vectors for the positional
    ///    delete tasks and the deletion-vector tasks. For equality delete
    ///    file tasks, this results in an `EqDeleteSet` (hash set of delete key tuples).
    ///  * The `EqDeleteSet`s resulting from equality deletes are sent to their associated oneshot
    ///    channel to store them in the right place in the delete file manager's state.
    ///  * The results of all of these futures are awaited on in parallel with the specified
    ///    level of concurrency and collected into a vec. We then combine all the delete
    ///    vector maps that resulted from any positional delete or delete vector files into a
    ///    single map and persist it in the state.
    ///
    ///
    ///  Conceptually, the data flow is like this:
    /// ```none
    ///                                          FileScanTaskDeleteFile
    ///                                                     |
    ///                                             Skip Started EQ Deletes
    ///                                                     |
    ///                                                     |
    ///                                       [load recordbatch stream / puffin]
    ///                                             DeleteFileContext
    ///                                                     |
    ///                                                     |
    ///                       +-----------------------------+--------------------------+
    ///                     Pos Del                    Del Vec                     EQ Del
    ///                       |                             |                          |
    ///              [parse pos del stream]         [parse del vec puffin]       [parse eq del]
    ///          HashMap<String, RoaringTreeMap> HashMap<String, RoaringTreeMap>  (EqDeleteSet, Sender)
    ///                       |                             |                          |
    ///                       |                             |                 [persist to state]
    ///                       |                             |                          ()
    ///                       |                             |                          |
    ///                       +-----------------------------+--------------------------+
    ///                                                     |
    ///                                             [buffer unordered]
    ///                                                     |
    ///                                            [combine del vectors]
    ///                                        HashMap<String, RoaringTreeMap>
    ///                                                     |
    ///                                        [persist del vectors to state]
    ///                                                    ()
    ///                                                    |
    ///                                                    |
    ///                                                 [join!]
    /// ```
    pub(crate) fn load_deletes(
        &self,
        delete_file_entries: &[FileScanTaskDeleteFile],
        schema: SchemaRef,
    ) -> Receiver<Result<DeleteFilter>> {
        let (tx, rx) = channel();

        let stream_items = delete_file_entries
            .iter()
            .map(|t| {
                (
                    t.clone(),
                    self.basic_delete_file_loader.clone(),
                    self.delete_filter.clone(),
                    schema.clone(),
                )
            })
            .collect::<Vec<_>>();
        let task_stream = futures::stream::iter(stream_items);

        let del_filter = self.delete_filter.clone();
        let concurrency_limit_data_files = self.concurrency_limit_data_files;
        let basic_delete_file_loader = self.basic_delete_file_loader.clone();
        self.runtime.io().spawn(async move {
            let result = async move {
                let mut del_filter = del_filter;
                let basic_delete_file_loader = basic_delete_file_loader.clone();

                let mut results_stream = task_stream
                    .map(move |(task, file_io, del_filter, schema)| {
                        let basic_delete_file_loader = basic_delete_file_loader.clone();
                        async move {
                            Self::load_file_for_task(
                                &task,
                                basic_delete_file_loader.clone(),
                                del_filter,
                                schema,
                            )
                            .await
                        }
                    })
                    .map(move |ctx| {
                        Ok(async { Self::parse_file_content_for_task(ctx.await?).await })
                    })
                    .try_buffer_unordered(concurrency_limit_data_files);

                while let Some(item) = results_stream.next().await {
                    let item = item?;
                    if let ParsedDeleteFileContext::DelVecs {
                        file_path,
                        content_offset,
                        results,
                    } = item
                    {
                        for (data_file_path, delete_vector) in results.into_iter() {
                            del_filter.upsert_delete_vector(data_file_path, delete_vector);
                        }
                        // Mark the positional delete load unit as fully loaded so waiters can proceed
                        del_filter.finish_pos_del_load(&file_path, content_offset);
                    }
                }

                Ok(del_filter)
            }
            .await;

            let _ = tx.send(result);
        });

        rx
    }

    async fn load_file_for_task(
        task: &FileScanTaskDeleteFile,
        basic_delete_file_loader: BasicDeleteFileLoader,
        del_filter: DeleteFilter,
        schema: SchemaRef,
    ) -> Result<DeleteFileContext> {
        match task.file_type {
            DataContentType::PositionDeletes => {
                // Load-unit identity: for a Puffin container, the DV blob's
                // content offset (one container may pack multiple blobs, each
                // referencing a different data file — every blob must load);
                // None for a whole-file Parquet positional-delete stream.
                let load_offset = if task.file_format == DataFileFormat::Puffin {
                    task.content_offset.map(|o| o as u64)
                } else {
                    None
                };
                match del_filter.try_start_pos_del_load(&task.file_path, load_offset) {
                    PosDelLoadAction::AlreadyLoaded => Ok(DeleteFileContext::ExistingPosDel),
                    PosDelLoadAction::WaitFor(notified) => {
                        // Positional deletes are accessed synchronously by ArrowReader.
                        // We must wait here to ensure the data is ready before returning,
                        // otherwise ArrowReader might get an empty/partial result.
                        notified.await;
                        Ok(DeleteFileContext::ExistingPosDel)
                    }
                    PosDelLoadAction::Load => {
                        if task.file_format == DataFileFormat::Puffin {
                            // V3 deletion vector — the blob located by
                            // content_offset/content_size is read + parsed in the
                            // parse phase (not a Parquet positional-delete stream).
                            Ok(DeleteFileContext::DelVec {
                                file_path: task.file_path.clone(),
                                referenced_data_file: task
                                    .referenced_data_file
                                    .clone()
                                    .ok_or_else(|| {
                                        Error::new(
                                            ErrorKind::DataInvalid,
                                            "deletion vector is missing referenced_data_file",
                                        )
                                    })?,
                                content_offset: task.content_offset.ok_or_else(|| {
                                    Error::new(
                                        ErrorKind::DataInvalid,
                                        "deletion vector is missing content_offset",
                                    )
                                })? as u64,
                                content_size: task.content_size_in_bytes.ok_or_else(|| {
                                    Error::new(
                                        ErrorKind::DataInvalid,
                                        "deletion vector is missing content_size_in_bytes",
                                    )
                                })? as u64,
                                file_io: basic_delete_file_loader.file_io().clone(),
                            })
                        } else {
                            Ok(DeleteFileContext::PosDels {
                                file_path: task.file_path.clone(),
                                stream: basic_delete_file_loader
                                    .parquet_to_batch_stream(
                                        &task.file_path,
                                        task.file_size_in_bytes,
                                        task.key_metadata.as_deref(),
                                    )
                                    .await?,
                            })
                        }
                    }
                }
            }

            DataContentType::EqualityDeletes => {
                let Some(notify) = del_filter.try_start_eq_del_load(&task.file_path) else {
                    return Ok(DeleteFileContext::ExistingEqDel);
                };

                let (sender, receiver) = channel();
                del_filter.insert_equality_delete(&task.file_path, receiver);

                // Per the Iceberg spec, equality_ids is required for equality delete files.
                // Evolve schema only for the equality_ids columns, not all table columns.
                let equality_ids_vec = task.equality_ids.clone().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "equality_ids is required for equality delete file '{}' but was not set",
                            task.file_path
                        ),
                    )
                })?;
                let evolved_stream = BasicDeleteFileLoader::evolve_schema(
                    basic_delete_file_loader
                        .parquet_to_batch_stream(
                            &task.file_path,
                            task.file_size_in_bytes,
                            task.key_metadata.as_deref(),
                        )
                        .await?,
                    schema,
                    &equality_ids_vec,
                )
                .await?;

                Ok(DeleteFileContext::FreshEqDel {
                    batch_stream: evolved_stream,
                    sender,
                    equality_ids: HashSet::from_iter(equality_ids_vec),
                })
            }

            DataContentType::Data => Err(Error::new(
                ErrorKind::Unexpected,
                "tasks with files of type Data not expected here",
            )),
        }
    }

    async fn parse_file_content_for_task(
        ctx: DeleteFileContext,
    ) -> Result<ParsedDeleteFileContext> {
        match ctx {
            DeleteFileContext::ExistingEqDel => Ok(ParsedDeleteFileContext::EqDel),
            DeleteFileContext::ExistingPosDel => Ok(ParsedDeleteFileContext::ExistingPosDel),
            DeleteFileContext::PosDels { file_path, stream } => {
                let del_vecs = Self::parse_positional_deletes_record_batch_stream(stream).await?;
                Ok(ParsedDeleteFileContext::DelVecs {
                    file_path,
                    content_offset: None,
                    results: del_vecs,
                })
            }
            DeleteFileContext::DelVec {
                file_path,
                referenced_data_file,
                content_offset,
                content_size,
                file_io,
            } => {
                let delete_vector =
                    Self::read_deletion_vector(&file_io, &file_path, content_offset, content_size)
                        .await?;
                let mut results = HashMap::default();
                results.insert(referenced_data_file, delete_vector);
                Ok(ParsedDeleteFileContext::DelVecs {
                    file_path,
                    content_offset: Some(content_offset),
                    results,
                })
            }
            DeleteFileContext::FreshEqDel {
                sender,
                batch_stream,
                equality_ids,
            } => {
                let eq_delete_set =
                    Self::parse_equality_deletes_record_batch_stream(batch_stream, equality_ids)
                        .await?;

                sender
                    .send(Arc::new(eq_delete_set))
                    .map_err(|err| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "Could not send eq delete set to state",
                        )
                    })
                    .map(|_| ParsedDeleteFileContext::EqDel)
            }
        }
    }

    /// Reads a V3 deletion-vector-v1 blob located at `content_offset` (length
    /// `content_size`) within `file_path` and parses it into a `DeleteVector`.
    ///
    /// The blob is addressed by the manifest entry's content offset/size rather
    /// than by parsing a Puffin footer, so this works whether the deletion vector
    /// is a standalone blob file (e.g. written by DuckDB) or stored inside a
    /// Puffin container.
    async fn read_deletion_vector(
        file_io: &FileIO,
        file_path: &str,
        content_offset: u64,
        content_size: u64,
    ) -> Result<DeleteVector> {
        let bytes = file_io.new_input(file_path)?.read().await?;
        let start = content_offset as usize;
        let end = start
            .checked_add(content_size as usize)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "deletion vector blob range {content_offset}..+{content_size} is out of bounds for {file_path} ({} bytes)",
                        bytes.len()
                    ),
                )
            })?;

        // Fast path: per the Iceberg spec a `deletion-vector-v1` blob is stored
        // uncompressed ("Omit compression-codec; deletion-vector-v1 is not
        // compressed"), so the bytes located by content_offset/content_size are the
        // serialized blob directly. This also covers container-less blob files
        // (e.g. written by DuckDB) that have no Puffin footer.
        if let Ok(delete_vector) = DeleteVector::from_serialized_bytes(&bytes[start..end]) {
            return Ok(delete_vector);
        }

        // Fallback: a deletion vector blob that is compressed (which the spec
        // forbids, but a non-conforming writer could produce) can only be
        // decompressed using the codec recorded in the Puffin footer. If this file
        // is a valid Puffin file, read the blob through its footer, which honors
        // the per-blob compression codec.
        let reader = PuffinReader::new(file_io.new_input(file_path)?);
        let metadata = reader.file_metadata().await.map_err(|err| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion vector at offset {content_offset} in {file_path} is neither a valid uncompressed deletion-vector-v1 blob nor a Puffin container"
                ),
            )
            .with_source(err)
        })?;
        let blob_metadata = metadata
            .blobs()
            .iter()
            .find(|blob| blob.offset() == content_offset)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Puffin file {file_path} has no blob at content_offset {content_offset}"
                    ),
                )
            })?;
        let blob = reader.blob(blob_metadata).await?;
        DeleteVector::from_puffin_blob(blob)
    }

    /// Parses a record batch stream coming from positional delete files
    ///
    /// Returns a map of data file path to a delete vector
    async fn parse_positional_deletes_record_batch_stream(
        mut stream: ArrowRecordBatchStream,
    ) -> Result<HashMap<String, DeleteVector>> {
        let mut result: HashMap<String, DeleteVector> = HashMap::default();

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let schema = batch.schema();
            let columns = batch.columns();

            // file_path decodes as Utf8 or LargeUtf8 depending on the reader's
            // offset widening — accept both.
            let file_paths: Box<dyn Iterator<Item = Option<&str>>> =
                if let Some(a) = columns[0].as_any().downcast_ref::<StringArray>() {
                    Box::new(a.iter())
                } else if let Some(a) = columns[0].as_any().downcast_ref::<LargeStringArray>() {
                    Box::new(a.iter())
                } else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "Could not downcast file paths array to a string array",
                    ));
                };
            let Some(positions) = columns[1].as_any().downcast_ref::<Int64Array>() else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast positions array to Int64Array",
                ));
            };

            for (file_path, pos) in file_paths.zip(positions.iter()) {
                let (Some(file_path), Some(pos)) = (file_path, pos) else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "null values in delete file",
                    ));
                };
                if pos < 0 {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("negative position in delete file {file_path}: {pos}"),
                    ));
                }

                result
                    .entry(file_path.to_string())
                    .or_default()
                    .insert(pos as u64);
            }
        }

        Ok(result)
    }

    /// Parses equality delete record batches into a hash-based delete set.
    ///
    /// We collect delete key tuples into a `HashSet` for O(1) per-row lookups.
    async fn parse_equality_deletes_record_batch_stream(
        mut stream: ArrowRecordBatchStream,
        equality_ids: HashSet<i32>,
    ) -> Result<EqDeleteSet> {
        let mut batch_schema_iceberg: Option<Schema> = None;
        let accessor = EqDelRecordBatchPartnerAccessor;
        // Discover field metadata from the first non-empty batch.
        let mut eq_delete_set: Option<EqDeleteSet> = None;

        while let Some(record_batch) = stream.next().await {
            let record_batch = record_batch?;

            if record_batch.num_columns() == 0 {
                return Ok(EqDeleteSet::new(Vec::new()));
            }

            let schema = match &batch_schema_iceberg {
                Some(schema) => schema,
                None => {
                    let schema = arrow_schema_to_schema(record_batch.schema().as_ref())?;
                    batch_schema_iceberg = Some(schema);
                    batch_schema_iceberg.as_ref().unwrap()
                }
            };

            let root_array: ArrayRef = Arc::new(StructArray::from(record_batch));

            let mut processor = EqDelColumnProcessor::new(&equality_ids);
            visit_schema_with_partner(schema, &root_array, &mut processor, &accessor)?;

            let mut datum_columns_with_names = processor.finish()?;
            if datum_columns_with_names.is_empty() {
                continue;
            }

            // Lazily initialize the EqDeleteSet with field metadata from the
            // first batch that has columns. The field order is stable across
            // batches because it comes from the delete file schema.
            //
            // Null semantics (Iceberg spec, Equality Delete Files): key tuples
            // are Option<Datum>, so a null delete value matches only a null
            // data value (None == None) and a null data value never matches a
            // non-null delete value — the same StructLikeSet semantics as the
            // Java implementation, and equivalent to the keep-predicate form
            // this replaced (`col IS NULL OR col != v` / `col IS NOT NULL`).
            let delete_set = eq_delete_set.get_or_insert_with(|| {
                let fields = datum_columns_with_names
                    .iter()
                    .map(|(_, name, field_id)| (name.clone(), *field_id))
                    .collect();
                EqDeleteSet::new(fields)
            });

            // Collect delete key tuples by iterating all columns in lockstep.
            #[allow(clippy::len_zero)]
            while datum_columns_with_names[0].0.len() > 0 {
                let mut key_values = Vec::with_capacity(datum_columns_with_names.len());
                for (column, _, _) in &mut datum_columns_with_names {
                    if let Some(item) = column.next() {
                        key_values.push(item?);
                    }
                }
                delete_set.keys.insert(EqDeleteKey(key_values));
            }
        }

        Ok(eq_delete_set.unwrap_or_else(|| EqDeleteSet::new(Vec::new())))
    }
}

struct EqDelColumnProcessor<'a> {
    equality_ids: &'a HashSet<i32>,
    collected_columns: Vec<(ArrayRef, String, i32, Type)>,
}

impl<'a> EqDelColumnProcessor<'a> {
    fn new(equality_ids: &'a HashSet<i32>) -> Self {
        Self {
            equality_ids,
            collected_columns: Vec::with_capacity(equality_ids.len()),
        }
    }

    /// Produces per-column Datum iterators alongside (field_name, field_id) metadata.
    #[allow(clippy::type_complexity)]
    fn finish(
        self,
    ) -> Result<
        Vec<(
            Box<dyn ExactSizeIterator<Item = Result<Option<Datum>>>>,
            String,
            i32,
        )>,
    > {
        self.collected_columns
            .into_iter()
            .map(|(array, field_name, field_id, field_type)| {
                let primitive_type = field_type
                    .as_primitive_type()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::Unexpected, "field is not a primitive type")
                    })?
                    .clone();

                let lit_vec = arrow_primitive_to_literal(&array, &field_type)?;
                let datum_iterator: Box<dyn ExactSizeIterator<Item = Result<Option<Datum>>>> =
                    Box::new(lit_vec.into_iter().map(move |c| {
                        c.map(|literal| {
                            literal
                                .as_primitive_literal()
                                .map(|primitive_literal| {
                                    Datum::new(primitive_type.clone(), primitive_literal)
                                })
                                .ok_or(Error::new(
                                    ErrorKind::Unexpected,
                                    "failed to convert to primitive literal",
                                ))
                        })
                        .transpose()
                    }));

                Ok((datum_iterator, field_name, field_id))
            })
            .collect::<Result<Vec<_>>>()
    }
}

impl SchemaWithPartnerVisitor<ArrayRef> for EqDelColumnProcessor<'_> {
    type T = ();

    fn schema(&mut self, _schema: &Schema, _partner: &ArrayRef, _value: ()) -> Result<()> {
        Ok(())
    }

    fn field(&mut self, field: &NestedFieldRef, partner: &ArrayRef, _value: ()) -> Result<()> {
        if self.equality_ids.contains(&field.id) && field.field_type.as_primitive_type().is_some() {
            self.collected_columns.push((
                partner.clone(),
                field.name.clone(),
                field.id,
                field.field_type.as_ref().clone(),
            ));
        }
        Ok(())
    }

    fn r#struct(
        &mut self,
        _struct: &StructType,
        _partner: &ArrayRef,
        _results: Vec<()>,
    ) -> Result<()> {
        Ok(())
    }

    fn list(&mut self, _list: &ListType, _partner: &ArrayRef, _value: ()) -> Result<()> {
        Ok(())
    }

    fn map(
        &mut self,
        _map: &MapType,
        _partner: &ArrayRef,
        _key_value: (),
        _value: (),
    ) -> Result<()> {
        Ok(())
    }

    fn primitive(&mut self, _primitive: &PrimitiveType, _partner: &ArrayRef) -> Result<()> {
        Ok(())
    }

    fn variant(&mut self, _v: &VariantType, _partner: &ArrayRef) -> Result<()> {
        Ok(())
    }
}

struct EqDelRecordBatchPartnerAccessor;

impl PartnerAccessor<ArrayRef> for EqDelRecordBatchPartnerAccessor {
    fn struct_partner<'a>(&self, schema_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Ok(schema_partner)
    }

    fn field_partner<'a>(
        &self,
        struct_partner: &'a ArrayRef,
        field: &NestedField,
    ) -> Result<&'a ArrayRef> {
        let Some(struct_array) = struct_partner.as_any().downcast_ref::<StructArray>() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Expected struct array for field extraction",
            ));
        };

        // Find the field by name within the struct
        for (i, field_def) in struct_array.fields().iter().enumerate() {
            if field_def.name() == &field.name {
                return Ok(struct_array.column(i));
            }
        }

        Err(Error::new(
            ErrorKind::Unexpected,
            format!("Field {} not found in parent struct", field.name),
        ))
    }

    fn list_element_partner<'a>(&self, _list_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "List columns are unsupported in equality deletes",
        ))
    }

    fn map_key_partner<'a>(&self, _map_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Map columns are unsupported in equality deletes",
        ))
    }

    fn map_value_partner<'a>(&self, _map_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Map columns are unsupported in equality deletes",
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        ArrayRef, BinaryArray, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::delete_filter::tests::setup;
    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{DataContentType, Schema};

    #[tokio::test]
    async fn test_delete_file_loader_parse_equality_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().as_os_str().to_str().unwrap();
        let file_io = FileIO::new_with_fs();

        let eq_delete_file_path = setup_write_equality_delete_file_1(table_location);

        let basic_delete_file_loader =
            BasicDeleteFileLoader::new(file_io.clone(), ScanMetrics::new());
        let record_batch_stream = basic_delete_file_loader
            .parquet_to_batch_stream(
                &eq_delete_file_path,
                std::fs::metadata(&eq_delete_file_path).unwrap().len(),
                None,
            )
            .await
            .expect("could not get batch stream");

        let eq_ids = HashSet::from_iter(vec![2, 3, 4, 6, 8]);

        let parsed_eq_delete = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            record_batch_stream,
            eq_ids,
        )
        .await
        .expect("error parsing batch stream");

        // The delete file has 2 rows, so we expect 2 keys in the set
        assert_eq!(parsed_eq_delete.keys.len(), 2);

        // Field metadata should list the 5 equality columns (y, z, a, sa, b)
        assert_eq!(parsed_eq_delete.fields.len(), 5);
        let field_names: Vec<&str> = parsed_eq_delete
            .fields
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(field_names.contains(&"y"));
        assert!(field_names.contains(&"z"));
        assert!(field_names.contains(&"a"));
        assert!(field_names.contains(&"sa"));
        assert!(field_names.contains(&"b"));

        // Row 1: y=1, z=100, a="HELP", sa=4, b=binary_data
        let row1 = EqDeleteKey(vec![
            Some(Datum::long(1)),
            Some(Datum::long(100)),
            Some(Datum::string("HELP")),
            Some(Datum::int(4)),
            Some(Datum::binary(b"binary_data".to_vec())),
        ]);
        assert!(
            parsed_eq_delete.keys.contains(&row1),
            "Row 1 should be in delete set"
        );

        // Row 2: y=2, z=NULL, a=NULL, sa=5, b=NULL
        let row2 = EqDeleteKey(vec![
            Some(Datum::long(2)),
            None,
            None,
            Some(Datum::int(5)),
            None,
        ]);
        assert!(
            parsed_eq_delete.keys.contains(&row2),
            "Row 2 should be in delete set"
        );

        // A non-existent key should not be in the set
        let non_existent = EqDeleteKey(vec![
            Some(Datum::long(999)),
            Some(Datum::long(0)),
            Some(Datum::string("NOPE")),
            Some(Datum::int(0)),
            Some(Datum::binary(b"nope".to_vec())),
        ]);
        assert!(!parsed_eq_delete.keys.contains(&non_existent));
    }

    // An equality delete keyed on a nullable column must not delete rows whose value in that
    // column is null: per the Iceberg spec (Equality Delete Files), a null matches only a null
    // delete value. Mirrors Iceberg-Java's
    // TestSparkReaderDeletes.testEqualityDeleteWithSchemaEvolution.
    #[tokio::test]
    async fn test_equality_delete_predicate_preserves_null_rows() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![simple_field(
            "status",
            DataType::Utf8,
            true,
            "3",
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![
                Arc::new(StringArray::from(vec![Some("INACTIVE")])) as ArrayRef,
            ])
            .unwrap();
        let stream: ArrowRecordBatchStream = futures::stream::iter(vec![Ok(batch)]).boxed();

        let eq_set = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            stream,
            HashSet::from_iter(vec![3]),
        )
        .await
        .expect("error parsing equality delete stream");

        // The non-null delete value is a key; a null data value (key None)
        // is NOT in the set, so null rows are kept.
        assert!(
            eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::string("INACTIVE"))]))
        );
        assert!(!eq_set.keys.contains(&EqDeleteKey(vec![None])));
    }

    // A delete row with a null value in the column matches only rows whose value is null (Iceberg
    // spec, Equality Delete Files), so the keep predicate is `col IS NOT NULL`.
    #[tokio::test]
    async fn test_equality_delete_predicate_matches_null_delete_value() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![simple_field(
            "status",
            DataType::Utf8,
            true,
            "3",
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec![
            None as Option<&str>,
        ])) as ArrayRef])
        .unwrap();
        let stream: ArrowRecordBatchStream = futures::stream::iter(vec![Ok(batch)]).boxed();

        let eq_set = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            stream,
            HashSet::from_iter(vec![3]),
        )
        .await
        .expect("error parsing equality delete stream");

        // A null delete value is the key (None,) — it matches only null data
        // values; non-null data values are not in the set and are kept.
        assert!(eq_set.keys.contains(&EqDeleteKey(vec![None])));
        assert!(
            !eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::string("ACTIVE"))]))
        );
    }

    // A delete row with several equality columns keeps a data row that differs in any one of them,
    // so the per-column keep predicates are OR-ed.
    #[tokio::test]
    async fn test_equality_delete_predicate_multiple_columns() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            simple_field("id", DataType::Int64, true, "1"),
            simple_field("status", DataType::Utf8, true, "3"),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("X")])) as ArrayRef,
        ])
        .unwrap();
        let stream: ArrowRecordBatchStream = futures::stream::iter(vec![Ok(batch)]).boxed();

        let eq_set = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            stream,
            HashSet::from_iter(vec![1, 3]),
        )
        .await
        .expect("error parsing equality delete stream");

        // The delete key is the full tuple — a row differing in ANY one
        // column is not in the set and is kept.
        assert!(eq_set.keys.contains(&EqDeleteKey(vec![
            Some(Datum::long(1)),
            Some(Datum::string("X"))
        ])));
        assert!(!eq_set.keys.contains(&EqDeleteKey(vec![
            Some(Datum::long(1)),
            Some(Datum::string("Y"))
        ])));
    }

    // A data row is kept only if it matches none of the delete rows, so the per-row keep
    // predicates are AND-ed.
    #[tokio::test]
    async fn test_equality_delete_predicate_multiple_delete_rows() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![simple_field(
            "status",
            DataType::Utf8,
            true,
            "3",
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec![
            Some("A"),
            Some("B"),
        ])) as ArrayRef])
        .unwrap();
        let stream: ArrowRecordBatchStream = futures::stream::iter(vec![Ok(batch)]).boxed();

        let eq_set = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            stream,
            HashSet::from_iter(vec![3]),
        )
        .await
        .expect("error parsing equality delete stream");

        // Every delete row becomes its own key — a data row is dropped when
        // it matches ANY key, kept when it matches none.
        assert_eq!(eq_set.keys.len(), 2);
        assert!(
            eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::string("A"))]))
        );
        assert!(
            eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::string("B"))]))
        );
        assert!(
            !eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::string("C"))]))
        );
    }

    /// Create a simple field with metadata.
    fn simple_field(name: &str, ty: DataType, nullable: bool, value: &str) -> Field {
        Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            value.to_string(),
        )]))
    }

    fn setup_write_equality_delete_file_1(table_location: &str) -> String {
        let col_y_vals = vec![1, 2];
        let col_y = Arc::new(Int64Array::from(col_y_vals)) as ArrayRef;

        let col_z_vals = vec![Some(100), None];
        let col_z = Arc::new(Int64Array::from(col_z_vals)) as ArrayRef;

        let col_a_vals = vec![Some("HELP"), None];
        let col_a = Arc::new(StringArray::from(col_a_vals)) as ArrayRef;

        let col_s = Arc::new(StructArray::from(vec![
            (
                Arc::new(simple_field("sa", DataType::Int32, false, "6")),
                Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
            ),
            (
                Arc::new(simple_field("sb", DataType::Utf8, true, "7")),
                Arc::new(StringArray::from(vec![Some("x"), None])) as ArrayRef,
            ),
        ]));

        let col_b_vals = vec![Some(&b"binary_data"[..]), None];
        let col_b = Arc::new(BinaryArray::from(col_b_vals)) as ArrayRef;

        let equality_delete_schema = {
            let struct_field = DataType::Struct(Fields::from(vec![
                simple_field("sa", DataType::Int32, false, "6"),
                simple_field("sb", DataType::Utf8, true, "7"),
            ]));

            let fields = vec![
                Field::new("y", DataType::Int64, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "2".to_string(),
                )])),
                Field::new("z", DataType::Int64, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "3".to_string(),
                )])),
                Field::new("a", DataType::Utf8, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "4".to_string(),
                )])),
                simple_field("s", struct_field, false, "5"),
                simple_field("b", DataType::Binary, true, "8"),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };

        let equality_deletes_to_write = RecordBatch::try_new(equality_delete_schema.clone(), vec![
            col_y, col_z, col_a, col_s, col_b,
        ])
        .unwrap();

        let path = format!("{}/equality-deletes-1.parquet", &table_location);

        let file = File::create(&path).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(
            file,
            equality_deletes_to_write.schema(),
            Some(props.clone()),
        )
        .unwrap();

        writer
            .write(&equality_deletes_to_write)
            .expect("Writing batch");

        // writer must be closed to write footer
        writer.close().unwrap();

        path
    }

    #[tokio::test]
    async fn test_caching_delete_file_loader_load_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader =
            CachingDeleteFileLoader::new(file_io.clone(), 10, Runtime::current());

        let file_scan_tasks = setup(table_location);

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();

        // union of pos dels from pos del file 1 and 2, ie
        // [0, 1, 3, 5, 6, 8, 1022, 1023] | [0, 1, 3, 5, 20, 21, 22, 23]
        // = [0, 1, 3, 5, 6, 8, 20, 21, 22, 23, 1022, 1023]
        assert_eq!(result.lock().unwrap().len(), 12);

        let result = delete_filter.get_delete_vector(&file_scan_tasks[1]);
        assert!(result.is_none()); // no pos dels for file 3
    }

    #[tokio::test]
    async fn test_parse_positional_deletes_rejects_negative_positions() {
        let schema = crate::arrow::delete_filter::tests::create_pos_del_schema();
        let file_path_col = Arc::new(StringArray::from_iter_values(vec!["data.parquet"]));
        let pos_col = Arc::new(Int64Array::from_iter_values(vec![-1i64]));
        let batch = RecordBatch::try_new(schema, vec![file_path_col, pos_col]).unwrap();
        let stream = futures::stream::iter(vec![Ok(batch)]).boxed();

        let err = CachingDeleteFileLoader::parse_positional_deletes_record_batch_stream(stream)
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("negative position"));
    }

    /// Verifies that evolve_schema on partial-schema equality deletes works correctly
    /// when only equality_ids columns are evolved, not all table columns.
    ///
    /// Per the [Iceberg spec](https://iceberg.apache.org/spec/#equality-delete-files),
    /// equality delete files can contain only a subset of columns.
    #[tokio::test]
    async fn test_partial_schema_equality_deletes_evolve_succeeds() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().as_os_str().to_str().unwrap();

        // Create table schema with REQUIRED fields
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Write equality delete file with PARTIAL schema (only 'data' column)
        let delete_file_path = {
            let data_vals = vec!["a", "d", "g"];
            let data_col = Arc::new(StringArray::from(data_vals)) as ArrayRef;

            let delete_schema = Arc::new(arrow_schema::Schema::new(vec![simple_field(
                "data",
                DataType::Utf8,
                false,
                "2", // field ID
            )]));

            let delete_batch = RecordBatch::try_new(delete_schema.clone(), vec![data_col]).unwrap();

            let path = format!("{}/partial-eq-deletes.parquet", &table_location);
            let file = File::create(&path).unwrap();
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();
            let mut writer =
                ArrowWriter::try_new(file, delete_batch.schema(), Some(props)).unwrap();
            writer.write(&delete_batch).expect("Writing batch");
            writer.close().unwrap();
            path
        };

        let file_io = FileIO::new_with_fs();
        let basic_delete_file_loader =
            BasicDeleteFileLoader::new(file_io.clone(), ScanMetrics::new());

        let batch_stream = basic_delete_file_loader
            .parquet_to_batch_stream(
                &delete_file_path,
                std::fs::metadata(&delete_file_path).unwrap().len(),
                None,
            )
            .await
            .unwrap();

        // Only evolve the equality_ids columns (field 2), not all table columns
        let equality_ids = vec![2];
        let evolved_stream =
            BasicDeleteFileLoader::evolve_schema(batch_stream, table_schema, &equality_ids)
                .await
                .unwrap();

        let result = evolved_stream.try_collect::<Vec<_>>().await;

        assert!(
            result.is_ok(),
            "Expected success when evolving only equality_ids columns, got error: {:?}",
            result.err()
        );

        let batches = result.unwrap();
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 1); // Only 'data' column

        // Verify the actual values are preserved after schema evolution
        let data_col = batch.column(0).as_string::<i64>();
        assert_eq!(data_col.value(0), "a");
        assert_eq!(data_col.value(1), "d");
        assert_eq!(data_col.value(2), "g");
    }

    /// Test loading a FileScanTask with BOTH positional and equality deletes.
    /// Verifies the fix for the inverted condition that caused "Missing predicate for equality delete file" errors.
    #[tokio::test]
    async fn test_load_deletes_with_mixed_types() {
        use crate::scan::FileScanTask;
        use crate::spec::{DataFileFormat, Schema};

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        // Create the data file schema
        let data_file_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Write positional delete file
        let positional_delete_schema = crate::arrow::delete_filter::tests::create_pos_del_schema();
        let file_path_values =
            vec![format!("{}/data-1.parquet", table_location.to_str().unwrap()); 4];
        let file_path_col = Arc::new(StringArray::from_iter_values(&file_path_values));
        let pos_col = Arc::new(Int64Array::from_iter_values(vec![0i64, 1, 2, 3]));

        let positional_deletes_to_write =
            RecordBatch::try_new(positional_delete_schema.clone(), vec![
                file_path_col,
                pos_col,
            ])
            .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let pos_del_path = format!("{}/pos-del-mixed.parquet", table_location.to_str().unwrap());
        let file = File::create(&pos_del_path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            positional_deletes_to_write.schema(),
            Some(props.clone()),
        )
        .unwrap();
        writer.write(&positional_deletes_to_write).unwrap();
        writer.close().unwrap();

        // Write equality delete file
        let eq_delete_path = setup_write_equality_delete_file_1(table_location.to_str().unwrap());

        // Create FileScanTask with BOTH positional and equality deletes
        let pos_del = FileScanTaskDeleteFile::builder()
            .with_file_path(pos_del_path.clone())
            .with_file_size_in_bytes(std::fs::metadata(&pos_del_path).unwrap().len())
            .with_file_type(DataContentType::PositionDeletes)
            .with_partition_spec_id(0)
            .build();

        let eq_del = FileScanTaskDeleteFile::builder()
            .with_file_path(eq_delete_path.clone())
            .with_file_size_in_bytes(std::fs::metadata(&eq_delete_path).unwrap().len())
            .with_file_type(DataContentType::EqualityDeletes)
            .with_partition_spec_id(0)
            .with_equality_ids(Some(vec![2, 3])) // Only use field IDs that exist in both schemas
            .build();

        let file_scan_task = FileScanTask::builder()
            .with_file_size_in_bytes(0)
            .with_start(0)
            .with_length(0)
            .with_data_file_path(format!(
                "{}/data-1.parquet",
                table_location.to_str().unwrap()
            ))
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(data_file_schema.clone())
            .with_project_field_ids(vec![2, 3])
            .with_deletes(vec![pos_del, eq_del])
            .with_case_sensitive(false)
            .build();

        // Load the deletes - should handle both types without error
        let delete_file_loader =
            CachingDeleteFileLoader::new(file_io.clone(), 10, Runtime::current());
        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_task.deletes, file_scan_task.schema_ref())
            .await
            .unwrap()
            .unwrap();

        // Verify both delete types can be processed together
        let result = delete_filter
            .build_equality_delete_sets(&file_scan_task)
            .await;
        assert!(
            result.is_ok(),
            "Failed to build equality delete sets: {:?}",
            result.err()
        );
        // The equality delete sets should contain delete keys
        let eq_sets = result.unwrap();
        assert!(
            !eq_sets.is_empty(),
            "Expected at least one equality delete set"
        );
    }

    /// Ported from apache/iceberg-go `TestFilterByDeletionVector` /
    /// `TestReadAllDeletionVectors`: a V3 deletion vector (Puffin) that references a
    /// data file is loaded during scan and its marked positions become the data
    /// file's delete vector. Before DV-applied scan support, the loader tried to read
    /// the Puffin file as a Parquet positional-delete and the marked rows were not
    /// excluded.
    #[tokio::test]
    async fn test_load_deletes_applies_v3_deletion_vector() {
        use crate::scan::FileScanTask;
        use crate::spec::{DataFileFormat, Schema, Struct};

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let data_file_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let data_file_path = format!("{}/data-dv.parquet", table_location.to_str().unwrap());

        // Build a deletion vector marking positions 1 and 3 as deleted and write it to
        // a Puffin file via the DV-write path.
        let mut dv = DeleteVector::default();
        dv.insert(1);
        dv.insert(3);
        let dv_puffin_path = format!("{}/dv-1.puffin", table_location.to_str().unwrap());
        let dv_data_file = dv
            .write_to_puffin_file(
                &file_io,
                dv_puffin_path.clone(),
                data_file_path.clone(),
                Struct::empty(),
                0,
            )
            .await
            .unwrap();

        let dv_del = FileScanTaskDeleteFile {
            key_metadata: None,
            file_path: dv_data_file.file_path().to_string(),
            file_size_in_bytes: dv_data_file.file_size_in_bytes(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            content_offset: dv_data_file.content_offset(),
            content_size_in_bytes: dv_data_file.content_size_in_bytes(),
            file_format: DataFileFormat::Puffin,
            referenced_data_file: dv_data_file.referenced_data_file(),
        };

        let file_scan_task = FileScanTask {
            key_metadata: None,
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: data_file_path.clone(),
            data_file_format: DataFileFormat::Parquet,
            schema: data_file_schema.clone(),
            project_field_ids: vec![2],
            predicate: None,
            deletes: vec![dv_del],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
        };

        let delete_file_loader =
            CachingDeleteFileLoader::new(file_io.clone(), 10, Runtime::current());
        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_task.deletes, file_scan_task.schema_ref())
            .await
            .unwrap()
            .unwrap();

        // The DV's marked positions are surfaced as the data file's delete vector.
        let dv_handle = delete_filter
            .get_delete_vector(&file_scan_task)
            .expect("expected a deletion vector for the data file");
        let mut positions: Vec<u64> = dv_handle.lock().unwrap().iter().collect();
        positions.sort();
        assert_eq!(positions, vec![1, 3]);
    }

    /// Regression: ONE Puffin container packing MULTIPLE `deletion-vector-v1`
    /// blobs (Doris and iceberg-java's DVFileWriter both write this shape) —
    /// N delete-file entries share the container's `file_path`, distinguished
    /// only by `content_offset`, each referencing a DIFFERENT data file. The
    /// load-state dedupe used to key on `file_path` alone, so only the first
    /// blob loaded; the rest returned AlreadyLoaded and their deletes were
    /// silently unapplied (deleted rows resurfaced). Every blob must load.
    #[tokio::test]
    async fn test_multi_blob_puffin_container_loads_every_dv() {
        use crate::delete_vector::{
            DELETION_VECTOR_PROPERTY_CARDINALITY, DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE,
        };
        use crate::puffin::{CompressionCodec, PuffinWriter};
        use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Type};

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let data_file_1 = format!("{}/data-1.parquet", table_location.to_str().unwrap());
        let data_file_2 = format!("{}/data-2.parquet", table_location.to_str().unwrap());

        let mut dv1 = DeleteVector::default();
        dv1.insert(1);
        dv1.insert(3);
        let mut dv2 = DeleteVector::default();
        dv2.insert(0);
        dv2.insert(2);

        let dv_properties = |referenced: &str, cardinality: u64| {
            HashMap::from([
                (
                    DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                    cardinality.to_string(),
                ),
                (
                    DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                    referenced.to_string(),
                ),
            ])
        };
        let blob1 = dv1.to_puffin_blob(dv_properties(&data_file_1, 2)).unwrap();
        let blob2 = dv2.to_puffin_blob(dv_properties(&data_file_2, 2)).unwrap();

        // ONE container holding BOTH blobs.
        let container_path = format!(
            "{}/delete_dv_multi.puffin",
            table_location.to_str().unwrap()
        );
        let output_file = file_io.new_output(&container_path).unwrap();
        let mut writer = PuffinWriter::new(&output_file, HashMap::new(), false)
            .await
            .unwrap();
        writer.add(blob1, CompressionCodec::None).await.unwrap();
        writer.add(blob2, CompressionCodec::None).await.unwrap();
        let result = writer.close_with_metadata().await.unwrap();
        let file_size = result.file_size_in_bytes;
        assert_eq!(result.blobs_metadata.len(), 2);

        // Two delete-file entries: same file_path, distinct content offsets,
        // different referenced data files — the manifest shape a packing
        // writer produces.
        let entries: Vec<FileScanTaskDeleteFile> = result
            .blobs_metadata
            .iter()
            .zip([data_file_1.clone(), data_file_2.clone()])
            .map(|(blob_meta, referenced)| FileScanTaskDeleteFile {
                key_metadata: None,
                file_path: container_path.clone(),
                file_size_in_bytes: file_size,
                file_type: DataContentType::PositionDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                content_offset: Some(blob_meta.offset() as i64),
                content_size_in_bytes: Some(blob_meta.length() as i64),
                file_format: DataFileFormat::Puffin,
                referenced_data_file: Some(referenced),
            })
            .collect();

        let delete_file_loader =
            CachingDeleteFileLoader::new(file_io.clone(), 10, Runtime::current());
        let delete_filter = delete_file_loader
            .load_deletes(&entries, schema)
            .await
            .unwrap()
            .unwrap();

        // BOTH referenced data files must have their delete vectors — keying
        // the load state by path alone drops whichever blob loses the race.
        let dv1_handle = delete_filter
            .get_delete_vector_for_path(&data_file_1)
            .expect("expected a deletion vector for data file 1");
        let mut positions_1: Vec<u64> = dv1_handle.lock().unwrap().iter().collect();
        positions_1.sort();
        assert_eq!(positions_1, vec![1, 3]);

        let dv2_handle = delete_filter
            .get_delete_vector_for_path(&data_file_2)
            .expect("expected a deletion vector for data file 2 — the second blob in the container must load");
        let mut positions_2: Vec<u64> = dv2_handle.lock().unwrap().iter().collect();
        positions_2.sort();
        assert_eq!(positions_2, vec![0, 2]);
    }

    #[tokio::test]
    async fn test_large_equality_delete_batch_stack_overflow() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().as_os_str().to_str().unwrap();
        let file_io = FileIO::new_with_fs();

        // Create a large batch of equality deletes
        let num_rows = 20_000;
        let col_y_vals: Vec<i64> = (0..num_rows).collect();
        let col_y = Arc::new(Int64Array::from(col_y_vals)) as ArrayRef;

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("y", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let record_batch = RecordBatch::try_new(schema.clone(), vec![col_y]).unwrap();

        // Write to file
        let path = format!("{}/large-eq-deletes.parquet", &table_location);
        let file = File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&record_batch).unwrap();
        writer.close().unwrap();

        let basic_delete_file_loader =
            BasicDeleteFileLoader::new(file_io.clone(), ScanMetrics::new());
        let record_batch_stream = basic_delete_file_loader
            .parquet_to_batch_stream(&path, std::fs::metadata(&path).unwrap().len(), None)
            .await
            .expect("could not get batch stream");

        let eq_ids = HashSet::from_iter(vec![2]);

        let result = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            record_batch_stream,
            eq_ids,
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_caching_delete_file_loader_caches_results() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader =
            CachingDeleteFileLoader::new(file_io.clone(), 10, Runtime::current());

        let file_scan_tasks = setup(table_location);

        // Load deletes for the first time
        let delete_filter_1 = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        // Load deletes for the second time (same task/files)
        let delete_filter_2 = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let dv1 = delete_filter_1
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();
        let dv2 = delete_filter_2
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();

        // Verify that the delete vectors point to the same memory location,
        // confirming that the second load reused the result from the first.
        assert!(Arc::ptr_eq(&dv1, &dv2));
    }
}
