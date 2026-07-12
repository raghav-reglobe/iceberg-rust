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

//! Shred-preserving variant rewrites — the compaction face of the shared
//! `iceberg::arrow::variant_shred` support (derivation, kernel, output-type
//! contract all live there; this module only owns the compaction-shaped IO:
//! footer loading from a planned `Group` and the batch-vector walk).

use std::collections::HashMap;

use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use iceberg::arrow::ArrowFileReader;
use iceberg::arrow::variant_shred::{shred_record_batch, shred_types_from_file_schema};
use iceberg::io::FileMetadata;
use iceberg::table::Table;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;

use crate::planner::Group;

/// Derive per-variant-column plain shredding types from the FIRST input
/// file's parquet footer. Columns not shredded in the input (or whose
/// shredding we can't derive) are absent from the map — they stay canonical.
pub(crate) async fn input_shred_types(
    table: &Table,
    group: &Group,
) -> Result<HashMap<String, DataType>> {
    let Some(task) = group.tasks.first() else {
        return Ok(HashMap::new());
    };
    let input = table.file_io().new_input(&task.data_file_path)?;
    let reader = input.reader().await?;
    let mut reader = ArrowFileReader::new(
        FileMetadata {
            size: task.file_size_in_bytes,
        },
        reader,
    );
    let meta = ArrowReaderMetadata::load_async(&mut reader, Default::default())
        .await
        .with_context(|| format!("loading parquet footer of {}", task.data_file_path))?;

    Ok(shred_types_from_file_schema(
        meta.schema(),
        table.metadata().current_schema(),
    ))
}

/// Shred the named variant columns of every (canonical) batch. Returns the
/// transformed batches plus each column's ACTUAL shredded arrow type — the
/// writer override that keeps the parquet schema in lockstep with the data.
pub(crate) fn shred_batches(
    batches: Vec<RecordBatch>,
    plain_types: &HashMap<String, DataType>,
) -> Result<(Vec<RecordBatch>, HashMap<String, DataType>)> {
    if plain_types.is_empty() || batches.is_empty() {
        return Ok((batches, HashMap::new()));
    }
    let mut overrides: HashMap<String, DataType> = HashMap::new();
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let shredded = shred_record_batch(&batch, plain_types)?;
        let schema = shredded.schema();
        for name in plain_types.keys() {
            let field = schema.field_with_name(name)?;
            overrides.insert(name.clone(), field.data_type().clone());
        }
        out.push(shredded);
    }
    Ok((out, overrides))
}
