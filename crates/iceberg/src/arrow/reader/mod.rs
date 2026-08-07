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

//! Parquet file data reader

use std::collections::HashSet;
use std::sync::Arc;

use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
use crate::arrow::scan_memory_gate::ScanMemoryGate;
use crate::cache::DataBytesCache;
use crate::io::FileIO;
use crate::runtime::Runtime;
use crate::util::available_parallelism;

/// Default gap between byte ranges below which they are coalesced into a
/// single request. Matches object_store's `OBJECT_STORE_COALESCE_DEFAULT`.
const DEFAULT_RANGE_COALESCE_BYTES: u64 = 1024 * 1024;

/// Default maximum number of coalesced byte ranges fetched concurrently.
/// Matches object_store's `OBJECT_STORE_COALESCE_PARALLEL`.
const DEFAULT_RANGE_FETCH_CONCURRENCY: usize = 10;

/// Default number of bytes to prefetch when parsing Parquet footer metadata.
/// Matches DataFusion's default `ParquetOptions::metadata_size_hint`.
const DEFAULT_METADATA_SIZE_HINT: usize = 512 * 1024;

mod file_reader;
mod options;
mod pipeline;
mod positional_deletes;
mod predicate_visitor;
mod projection;
mod row_filter;
pub use file_reader::ArrowFileReader;
pub(crate) use options::ParquetReadOptions;
use predicate_visitor::{CollectFieldIdVisitor, PredicateConverter};
use projection::{add_fallback_field_ids_to_arrow_schema, apply_name_mapping_to_arrow_schema};

/// Builder to create ArrowReader
pub struct ArrowReaderBuilder {
    batch_size: Option<usize>,
    file_io: FileIO,
    concurrency_limit_data_files: usize,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    parquet_read_options: ParquetReadOptions,
    runtime: Runtime,
    shredded_passthrough: Option<Arc<HashSet<String>>>,
    data_bytes_cache: Option<DataBytesCache>,
    scan_memory_gate: Option<Arc<dyn ScanMemoryGate>>,
}

impl ArrowReaderBuilder {
    /// Create a new ArrowReaderBuilder
    pub fn new(file_io: FileIO, runtime: Runtime) -> Self {
        let num_cpus = available_parallelism().get();

        ArrowReaderBuilder {
            batch_size: None,
            file_io,
            concurrency_limit_data_files: num_cpus,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
            parquet_read_options: ParquetReadOptions::builder().build(),
            runtime,
            shredded_passthrough: None,
            data_bytes_cache: None,
            scan_memory_gate: None,
        }
    }

    /// Back DATA-FILE reads with a WHOLE-FILE read-through cache (see
    /// [`crate::cache::DataBytesCache`]): the first read of a file fetches
    /// the entire object once into the shared store; every ranged read of
    /// that file (footer, column chunks, byte-range splits) is then served
    /// from the local copy. Files above the cache's `max_file_bytes` bypass
    /// it. Without this, reads go directly to storage — byte-identical to
    /// the uncached behavior.
    pub fn with_data_bytes_cache(mut self, data_bytes_cache: DataBytesCache) -> Self {
        self.data_bytes_cache = Some(data_bytes_cache);
        self
    }

    /// Account each file's estimated decode working set against `gate` before
    /// its decode begins (see [`crate::arrow::ScanMemoryGate`]). Without a
    /// gate, reads are unaccounted — the pre-existing behavior.
    pub fn with_scan_memory_gate(mut self, gate: Arc<dyn ScanMemoryGate>) -> Self {
        self.scan_memory_gate = Some(gate);
        self
    }

    /// Shredded variant passthrough: keep the named variant columns in their
    /// physical SHREDDED shape — whatever per-file `typed_value` layout the
    /// file carries — instead of folding them back to the canonical
    /// `{metadata, value}` form (a per-row blob reconstruction). Decided per
    /// file: canonical files pass through canonically as always, and each
    /// shredded file's batches carry that FILE's own layout, so batches from
    /// different files may disagree on the column's arrow type. Consumers
    /// must accept the shredded struct types verbatim AND tolerate that
    /// per-file variance (the MoR merge writer under
    /// `write.parquet.shred-variants` does: it routes appends to per-layout
    /// writers, so the fold + re-shred round-trip is skipped entirely).
    /// Names should be VARIANT columns of the table schema; non-variant
    /// columns are ignored by the per-file gate.
    pub fn with_shredded_passthrough(mut self, columns: HashSet<String>) -> Self {
        self.shredded_passthrough = Some(Arc::new(columns));
        self
    }

    /// Sets the max number of in flight data files that are being fetched
    pub fn with_data_file_concurrency_limit(mut self, val: usize) -> Self {
        self.concurrency_limit_data_files = val;
        self
    }

    /// Sets the desired size of batches in the response
    /// to something other than the default
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Determines whether to enable row group filtering.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Determines whether to enable row selection.
    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    /// Provide a hint as to the number of bytes to prefetch for parsing the Parquet metadata
    ///
    /// This hint can help reduce the number of fetch requests. For more details see the
    /// [ParquetMetaDataReader documentation](https://docs.rs/parquet/latest/parquet/file/metadata/struct.ParquetMetaDataReader.html#method.with_prefetch_hint).
    pub fn with_metadata_size_hint(mut self, metadata_size_hint: usize) -> Self {
        self.parquet_read_options.metadata_size_hint = Some(metadata_size_hint);
        self
    }

    /// Sets the gap threshold for merging nearby byte ranges into a single request.
    /// Ranges with gaps smaller than this value will be coalesced.
    ///
    /// Defaults to 1 MiB, matching object_store's OBJECT_STORE_COALESCE_DEFAULT.
    pub fn with_range_coalesce_bytes(mut self, range_coalesce_bytes: u64) -> Self {
        self.parquet_read_options.range_coalesce_bytes = range_coalesce_bytes;
        self
    }

    /// Caps parquet's per-row-group predicate cache (bytes) — the async
    /// decoder's cache of filter-decoded pages reused for output
    /// materialization (only effective when a row filter is present AND the
    /// filter columns are also projected). Parquet's own default is 100 MB
    /// PER ROW GROUP, which is real memory on decode-bounded pods; `0`
    /// disables the cache. Unset = parquet's default.
    pub fn with_max_predicate_cache_size(mut self, max_predicate_cache_bytes: usize) -> Self {
        self.parquet_read_options.max_predicate_cache_bytes = Some(max_predicate_cache_bytes);
        self
    }

    /// Sets the maximum number of merged byte ranges to fetch concurrently.
    ///
    /// Defaults to 10, matching object_store's OBJECT_STORE_COALESCE_PARALLEL.
    pub fn with_range_fetch_concurrency(mut self, range_fetch_concurrency: usize) -> Self {
        self.parquet_read_options.range_fetch_concurrency = range_fetch_concurrency;
        self
    }

    /// Build the ArrowReader.
    pub fn build(self) -> ArrowReader {
        ArrowReader {
            batch_size: self.batch_size,
            file_io: self.file_io.clone(),
            delete_file_loader: CachingDeleteFileLoader::new(
                self.file_io.clone(),
                self.concurrency_limit_data_files,
                self.runtime.clone(),
            ),
            concurrency_limit_data_files: self.concurrency_limit_data_files,
            row_group_filtering_enabled: self.row_group_filtering_enabled,
            row_selection_enabled: self.row_selection_enabled,
            parquet_read_options: self.parquet_read_options,
            runtime: self.runtime,
            shredded_passthrough: self.shredded_passthrough,
            data_bytes_cache: self.data_bytes_cache,
            scan_memory_gate: self.scan_memory_gate,
        }
    }
}

/// Reads data from Parquet files
#[derive(Clone)]
pub struct ArrowReader {
    batch_size: Option<usize>,
    file_io: FileIO,
    delete_file_loader: CachingDeleteFileLoader,

    /// the maximum number of data files that can be fetched at the same time
    concurrency_limit_data_files: usize,

    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    parquet_read_options: ParquetReadOptions,
    /// Runtime the per-file decode tasks are spawned onto (CPU handle) so
    /// files decode in PARALLEL across cores, not merely overlapped on the
    /// consuming task.
    runtime: Runtime,
    /// See [`ArrowReaderBuilder::with_shredded_passthrough`].
    shredded_passthrough: Option<Arc<HashSet<String>>>,
    /// See [`ArrowReaderBuilder::with_data_bytes_cache`].
    data_bytes_cache: Option<DataBytesCache>,
    /// See [`ArrowReaderBuilder::with_scan_memory_gate`].
    scan_memory_gate: Option<Arc<dyn ScanMemoryGate>>,
}
