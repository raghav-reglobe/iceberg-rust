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

//! ATOMIC PARTITION-SCOPED REPLACE — the append-only CDC "full-load chunk
//! re-backfill" primitive: one `RowDelta` snapshot that (i) equality-deletes
//! the PRIOR rows of a key chunk inside one identity partition (top-level,
//! pk-only equality ids, partition-scoped — so same-key rows in other
//! partitions are structurally out of scope) and (ii) appends the
//! replacement rows. The delete tuples ARE the replacement rows' keys.
//!
//! Concurrency: `validate_from_snapshot` OCC; concurrent PURE APPENDS (a
//! streaming sink) pass under snapshot isolation, conflicting delete
//! commits fail closed, and a conflict is retried BY RERUN — the
//! already-written data + delete files are re-committed from a fresh base
//! (equality-delete sequence numbers come from the commit, so a re-commit
//! stays correct).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use parquet::variant::{VariantArrayBuilder, json_to_variant};
use uuid::Uuid;

use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use crate::spec::{DataFileFormat, Literal, PartitionKey, Struct, Transform};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

/// The result of an [`atomic_partition_replace`].
#[derive(Debug, Clone)]
pub struct ReplaceOutcome {
    /// The committed snapshot id (the pre-existing current snapshot for a
    /// no-op or dry run).
    pub snapshot_id: Option<i64>,
    /// Rows appended (= rows in the input batches).
    pub rows_appended: u64,
    /// Equality-delete tuples written (= rows; an ESTIMATE of rows deleted —
    /// tuples with no prior match delete nothing).
    pub delete_tuples: u64,
    /// Commit attempts used (0 for dry runs and no-ops).
    pub attempts: u32,
}

/// Atomically replace one key-chunk of an identity partition: equality-delete
/// the chunk's PRIOR rows (`partition_column = partition_value`, pk in the
/// batches' keys) and append the batches, as ONE snapshot.
///
/// Contracts:
/// - `equality_columns` must be TOP-LEVEL, non-float table columns (the
///   partition scope replaces any nested discriminator column — see the
///   sync-engine RFC).
/// - the table's default partition spec must be a single IDENTITY partition
///   on `partition_column`; the replacement rows must all belong to
///   `partition_value`'s partition.
/// - input batches are conformed to the table schema BY NAME (field-id
///   metadata attached, primitives cast, structs realigned recursively);
///   string columns targeting a VARIANT column are parsed as JSON via the
///   arrow variant kernel.
/// - `dry_run` stops after validation + conformance: nothing is written,
///   nothing committed; the outcome carries the would-be counts.
#[allow(clippy::too_many_arguments)]
pub async fn atomic_partition_replace(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    partition_column: &str,
    partition_value: Literal,
    equality_columns: &[String],
    batches: Vec<RecordBatch>,
    max_attempts: u32,
    dry_run: bool,
) -> Result<ReplaceOutcome> {
    let table = catalog.load_table(ident).await?;
    let schema = table.metadata().current_schema().clone();

    // Equality ids: top-level, by name, against the LOADED schema (create
    // paths reassign field ids — never trust creation-time constants).
    let mut equality_ids = Vec::with_capacity(equality_columns.len());
    for col in equality_columns {
        let field_id = schema.field_id_by_name(col).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("equality column `{col}` not found in table schema"),
            )
        })?;
        if schema.as_struct().field_by_id(field_id).is_none() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "equality column `{col}` is not a TOP-LEVEL column — nested equality ids \
                     are not supported (scope by partition instead)"
                ),
            ));
        }
        equality_ids.push(field_id);
    }

    // The identity partition this replace is scoped to.
    let spec = table.metadata().default_partition_spec().clone();
    let ok_spec = spec.fields().len() == 1
        && spec.fields()[0].transform == Transform::Identity
        && schema
            .field_id_by_name(partition_column)
            .is_some_and(|id| id == spec.fields()[0].source_id);
    if !ok_spec {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "atomic replace requires a single-field IDENTITY partition on \
                 `{partition_column}`; the table's default spec is {spec:?}"
            ),
        ));
    }
    let partition_key = PartitionKey::new(
        spec.as_ref().clone(),
        schema.clone(),
        Struct::from_iter(vec![Some(partition_value)]),
    );

    // Conform the input to the table's arrow schema (names -> field ids,
    // casts, struct realignment, JSON -> VARIANT).
    let target_arrow: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema)?);
    let batches = batches
        .into_iter()
        .filter(|b| b.num_rows() > 0)
        .map(|b| conform_batch(&b, &target_arrow))
        .collect::<Result<Vec<_>>>()?;
    let rows_appended: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();

    if dry_run || rows_appended == 0 {
        return Ok(ReplaceOutcome {
            snapshot_id: table.metadata().current_snapshot_id(),
            rows_appended,
            delete_tuples: rows_appended,
            attempts: 0,
        });
    }

    // Write the replacement data files (into the scoped partition).
    let run = Uuid::now_v7();
    let data_rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(writer_properties(&table), schema.clone()),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata())?,
        DefaultFileNameGenerator::new(format!("replace-{run}"), None, DataFileFormat::Parquet),
    );
    let mut data_writer = DataFileWriterBuilder::new(data_rolling)
        .build(Some(partition_key.clone()))
        .await?;
    for batch in &batches {
        data_writer.write(batch.clone()).await?;
    }
    let data_files = data_writer.close().await?;

    // Write the partition-scoped equality-delete file: the delete tuples are
    // the replacement rows' own keys (the writer projects the equality-id
    // columns). Its parquet schema is the PROJECTED equality-id schema.
    let eq_config = EqualityDeleteWriterConfig::new(equality_ids, schema.clone())?;
    let delete_schema = Arc::new(arrow_schema_to_schema(
        eq_config.projected_arrow_schema_ref(),
    )?);
    let eq_rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            delete_schema,
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata())?,
        DefaultFileNameGenerator::new(format!("replace-del-{run}"), None, DataFileFormat::Parquet),
    );
    let mut eq_writer = EqualityDeleteFileWriterBuilder::new(eq_rolling, eq_config)
        .build(Some(partition_key))
        .await?;
    for batch in &batches {
        eq_writer.write(batch.clone()).await?;
    }
    let delete_files = eq_writer.close().await?;

    // Commit as ONE RowDelta snapshot; on commit conflict, RERUN with the
    // SAME files from a fresh base.
    let max_attempts = max_attempts.max(1);
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let table = catalog.load_table(ident).await?;
        let tx = Transaction::new(&table);
        let mut action = tx
            .row_delta()
            .add_data_files(data_files.clone())
            .add_delete_files(delete_files.clone());
        action = match table.metadata().current_snapshot_id() {
            Some(base) => action.validate_from_snapshot(base),
            None => action.validate_from_empty_table(),
        };
        match async { action.apply(tx)?.commit(catalog).await }.await {
            Ok(committed) => {
                return Ok(ReplaceOutcome {
                    snapshot_id: committed.metadata().current_snapshot_id(),
                    rows_appended,
                    delete_tuples: rows_appended,
                    attempts,
                });
            }
            Err(e) if attempts < max_attempts && is_commit_conflict(&e) => {
                continue; // rerun from a fresh base with the same files
            }
            Err(e) => return Err(e),
        }
    }
}

/// A retryable optimistic-concurrency conflict: the catalog's commit
/// conflict (HTTP 409 class) or the RowDelta validator's concurrent-commit
/// rejection.
fn is_commit_conflict(e: &Error) -> bool {
    e.kind() == ErrorKind::CatalogCommitConflicts
        || e.to_string()
            .contains("Found conflicting concurrent commit")
}

/// Conform `batch` to the table's arrow schema: resolve columns BY NAME,
/// attach the target fields (field-id metadata), realign struct children
/// recursively, cast primitives, and parse string columns targeting VARIANT
/// as JSON via the arrow variant kernel.
pub fn conform_batch(batch: &RecordBatch, target: &ArrowSchemaRef) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(target.fields().len());
    for field in target.fields() {
        let idx = batch.schema().index_of(field.name()).map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("input batch is missing column `{}`", field.name()),
            )
        })?;
        columns.push(conform_array(batch.column(idx), field)?);
    }
    RecordBatch::try_new(Arc::new(ArrowSchema::new(target.fields().clone())), columns).map_err(
        |e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("conforming input batch to the table schema: {e}"),
            )
        },
    )
}

fn conform_array(array: &ArrayRef, target: &Field) -> Result<ArrayRef> {
    if array.data_type() == target.data_type() {
        return Ok(Arc::clone(array));
    }
    // JSON string -> VARIANT.
    if is_variant_type(target.data_type())
        && matches!(
            array.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    {
        return json_strings_to_variant(array, target);
    }
    // Struct: realign children by name recursively.
    if let (DataType::Struct(target_fields), DataType::Struct(_)) =
        (target.data_type(), array.data_type())
    {
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("checked struct");
        let mut children = Vec::with_capacity(target_fields.len());
        for child_field in target_fields {
            let child = struct_array
                .column_by_name(child_field.name())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("input struct is missing field `{}`", child_field.name()),
                    )
                })?;
            children.push(conform_array(child, child_field)?);
        }
        let out = StructArray::try_new(
            target_fields.clone(),
            children,
            struct_array.nulls().cloned(),
        )
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("conforming struct column: {e}"),
            )
        })?;
        return Ok(Arc::new(out));
    }
    // Primitives and everything else: arrow cast.
    arrow_cast::cast(array.as_ref(), target.data_type()).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "cannot convert input column `{}` from {:?} to {:?}: {e}",
                target.name(),
                array.data_type(),
                target.data_type()
            ),
        )
    })
}

/// The iceberg-arrow canonical VARIANT rendering: a struct of exactly
/// `metadata: Binary` + `value: Binary`.
fn is_variant_type(dt: &DataType) -> bool {
    match dt {
        DataType::Struct(fields) => {
            fields.len() == 2
                && fields.iter().any(|f| f.name() == "metadata")
                && fields.iter().any(|f| f.name() == "value")
                && fields.iter().all(|f| f.data_type() == &DataType::Binary)
        }
        _ => false,
    }
}

/// Parse a string column of JSON documents into the canonical VARIANT
/// struct layout (`metadata`/`value` Binary), preserving nulls, via the
/// arrow variant kernel — the same kernel the `json_to_variant` SQL UDF
/// wraps.
fn json_strings_to_variant(array: &ArrayRef, target: &Field) -> Result<ArrayRef> {
    let strings = arrow_cast::cast(array.as_ref(), &DataType::Utf8)
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("casting JSON input: {e}")))?;
    let variant = json_to_variant(&strings).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("parsing JSON as variant: {e}"),
        )
    })?;
    // Re-emit through the builder for a canonical layout, then coerce the
    // metadata/value children to plain Binary under the TARGET fields.
    let variant = {
        let mut builder = VariantArrayBuilder::new(variant.len());
        for i in 0..variant.len() {
            if variant.is_null(i) {
                builder.append_null();
            } else {
                builder.append_variant(variant.value(i));
            }
        }
        builder.build()
    };
    let inner: StructArray = variant.into_inner();
    let metadata = inner
        .column_by_name("metadata")
        .ok_or_else(|| Error::new(ErrorKind::Unexpected, "variant missing `metadata`"))?;
    let value = inner
        .column_by_name("value")
        .ok_or_else(|| Error::new(ErrorKind::Unexpected, "variant missing `value`"))?;
    let metadata = arrow_cast::cast(metadata.as_ref(), &DataType::Binary)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("variant metadata cast: {e}")))?;
    let value = arrow_cast::cast(value.as_ref(), &DataType::Binary)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("variant value cast: {e}")))?;
    let DataType::Struct(target_fields) = target.data_type() else {
        unreachable!("checked variant type");
    };
    let out = StructArray::try_new(
        target_fields.clone(),
        vec![metadata, value],
        inner.nulls().cloned(),
    )
    .map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("building variant column: {e}"),
        )
    })?;
    Ok(Arc::new(out))
}

/// Writer properties honoring the table's `write.parquet.compression-codec`
/// (defaulting to ZSTD, the iceberg convention — parquet-rs's own default is
/// UNCOMPRESSED, which would silently inflate output ~20x).
fn writer_properties(table: &Table) -> parquet::file::properties::WriterProperties {
    use parquet::basic::{Compression, GzipLevel, ZstdLevel};
    let codec = table
        .metadata()
        .properties()
        .get("write.parquet.compression-codec")
        .map(|s| s.to_ascii_lowercase());
    let compression = match codec.as_deref() {
        Some("uncompressed") => Compression::UNCOMPRESSED,
        Some("snappy") => Compression::SNAPPY,
        Some("gzip") => Compression::GZIP(GzipLevel::default()),
        Some("lz4") => Compression::LZ4,
        _ => Compression::ZSTD(ZstdLevel::default()),
    };
    parquet::file::properties::WriterProperties::builder()
        .set_compression(compression)
        .build()
}
