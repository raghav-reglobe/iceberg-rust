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

//! Shred-preserving variant rewrites.
//!
//! The read path folds SHREDDED variant columns back to the canonical
//! `{metadata, value}` layout (the fold is what makes any downstream consumer
//! see one stable type), so a plain rewrite would silently UN-shred the
//! output — correct, but it discards the columnar `typed_value` subtree that
//! query engines prune into. With `Config::shred_variants` the rewrite
//! re-shreds on write instead:
//!
//! 1. derive each variant column's shredding schema from the FIRST input
//!    file's parquet footer (shred-preserving: the output keeps the input's
//!    layout; a column that isn't shredded in the input stays canonical);
//! 2. run the canonical batches through the `shred_variant` kernel;
//! 3. hand the writer the shredded arrow types so the parquet schema carries
//!    the `typed_value` subtree.
//!
//! Values that don't match the shredding schema stay in the binary `value`
//! (partial shredding) — semantics are unchanged either way, only the
//! physical layout differs.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use iceberg::arrow::ArrowFileReader;
use iceberg::io::FileMetadata;
use iceberg::spec::Type;
use iceberg::table::Table;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::variant::{VariantArray, shred_variant};

use crate::planner::Group;

/// Is this fields-set an OBJECT-shredding node set — every child a
/// `{value?, typed_value?}` group (the wrapper `shred_variant` puts around
/// each shredded object field)?
fn is_object_node_set(children: &arrow_schema::Fields) -> bool {
    !children.is_empty()
        && children.iter().all(|c| {
            matches!(c.data_type(), DataType::Struct(node)
                if !node.is_empty()
                    && node.iter().all(|f| f.name() == "value" || f.name() == "typed_value"))
        })
}

/// Recover the PLAIN requested type from a file's `typed_value` tree — the
/// inverse of the `{value, typed_value}` wrapping `shred_variant` applies.
/// Fields shredded value-only (no `typed_value`) are skipped: re-shredding
/// simply leaves their values in the binary `value`, which is still correct.
/// Returns `None` when nothing typed survives (column stays canonical).
fn unwrap_typed_value(t: &DataType) -> Option<DataType> {
    match t {
        DataType::Struct(children) if is_object_node_set(children) => {
            let fields: Vec<Field> = children
                .iter()
                .filter_map(|c| {
                    let DataType::Struct(node) = c.data_type() else {
                        return None;
                    };
                    let tv = node.iter().find(|f| f.name() == "typed_value")?;
                    let plain = unwrap_typed_value(tv.data_type())?;
                    Some(Field::new(c.name(), plain, true))
                })
                .collect();
            (!fields.is_empty()).then(|| DataType::Struct(fields.into()))
        }
        // Array shredding is not derived yet — the column falls back to
        // canonical output, which is always correct (just less prunable).
        DataType::List(_) | DataType::LargeList(_) => None,
        other => Some(other.clone()),
    }
}

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

    let table_schema = table.metadata().current_schema();
    let mut out = HashMap::new();
    for field in meta.schema().fields() {
        let Some(iceberg_field) = table_schema.field_by_name(field.name()) else {
            continue;
        };
        if !matches!(iceberg_field.field_type.as_ref(), Type::Variant(_)) {
            continue;
        }
        let DataType::Struct(children) = field.data_type() else {
            continue;
        };
        let Some(tv) = children.iter().find(|c| c.name() == "typed_value") else {
            continue; // canonical in the input -> canonical in the output
        };
        if let Some(plain) = unwrap_typed_value(tv.data_type()) {
            out.insert(field.name().to_string(), plain);
        }
    }
    Ok(out)
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
        let schema = batch.schema();
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for (name, plain) in plain_types {
            let idx = schema
                .index_of(name)
                .with_context(|| format!("variant column {name} missing from batch"))?;
            let variant = VariantArray::try_new(&columns[idx])
                .with_context(|| format!("column {name} is not a variant array"))?;
            let shredded = shred_variant(&variant, plain)
                .with_context(|| format!("shredding column {name}"))?;
            let arr = ArrayRef::from(shredded);
            overrides.insert(name.clone(), arr.data_type().clone());
            fields[idx] = Arc::new(
                fields[idx]
                    .as_ref()
                    .clone()
                    .with_data_type(arr.data_type().clone()),
            );
            columns[idx] = arr;
        }
        let new_schema = Arc::new(ArrowSchema::new_with_metadata(
            fields,
            schema.metadata().clone(),
        ));
        out.push(RecordBatch::try_new(new_schema, columns)?);
    }
    Ok((out, overrides))
}
