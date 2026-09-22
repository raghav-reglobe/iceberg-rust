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

use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize, Serializer};
use typed_builder::TypedBuilder;

use crate::Result;
use crate::expr::BoundPredicate;
use crate::spec::{
    DataContentType, DataFileFormat, ManifestEntryRef, NameMapping, NestedField, PartitionSpec,
    Schema, SchemaRef, Struct, Type,
};

/// A stream of [`FileScanTask`].
pub type FileScanTaskStream = BoxStream<'static, Result<FileScanTask>>;

/// Serialization helper that always returns NotImplementedError.
/// Used for fields that should not be serialized but we want to be explicit about it.
fn serialize_not_implemented<S, T>(_: &T, _: S) -> std::result::Result<S::Ok, S::Error>
where S: Serializer {
    Err(serde::ser::Error::custom(
        "Serialization not implemented for this field",
    ))
}

/// Deserialization helper that always returns NotImplementedError.
/// Used for fields that should not be deserialized but we want to be explicit about it.
fn deserialize_not_implemented<'de, D, T>(_: D) -> std::result::Result<T, D::Error>
where D: serde::Deserializer<'de> {
    Err(serde::de::Error::custom(
        "Deserialization not implemented for this field",
    ))
}

/// A task to scan part of file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TypedBuilder)]
#[builder(field_defaults(setter(prefix = "with_")))]
pub struct FileScanTask {
    /// The total size of the data file in bytes, from the manifest entry.
    /// Used to skip a stat/HEAD request when reading Parquet footers.
    pub file_size_in_bytes: u64,
    /// The start offset of the file to scan.
    pub start: u64,
    /// The length of the file to scan.
    pub length: u64,
    /// The number of records in the file to scan.
    ///
    /// This is an optional field, and only available if we are
    /// reading the entire data file.
    #[builder(default)]
    pub record_count: Option<u64>,

    /// Per-field compressed sizes the manifest recorded for the file
    /// (`column_sizes`), keyed by field id; `None` when the writer recorded
    /// none. Lets the reader estimate how much of the file a projection
    /// reads before opening it — see [`Self::projected_byte_share`].
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub column_sizes: Option<HashMap<i32, u64>>,

    /// The data file path corresponding to the task.
    pub data_file_path: String,

    /// The format of the file to scan.
    pub data_file_format: DataFileFormat,

    /// The schema of the file to scan.
    pub schema: SchemaRef,
    /// The field ids to project.
    pub project_field_ids: Vec<i32>,
    /// The predicate to filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub predicate: Option<BoundPredicate>,

    /// The list of delete files that may need to be applied to this data file
    #[builder(default)]
    pub deletes: Vec<FileScanTaskDeleteFile>,

    /// Partition data from the manifest entry, used to identify which columns can use
    /// constant values from partition metadata vs. reading from the data file.
    /// Per the Iceberg spec, only identity-transformed partition fields should use constants.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_not_implemented")]
    #[serde(deserialize_with = "deserialize_not_implemented")]
    #[builder(default)]
    pub partition: Option<Struct>,

    /// The partition spec for this file, used to distinguish identity transforms
    /// (which use partition metadata constants) from non-identity transforms like
    /// bucket/truncate (which must read source columns from the data file).
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_not_implemented")]
    #[serde(deserialize_with = "deserialize_not_implemented")]
    #[builder(default)]
    pub partition_spec: Option<Arc<PartitionSpec>>,

    /// Name mapping from table metadata (property: schema.name-mapping.default),
    /// used to resolve field IDs from column names when Parquet files lack field IDs
    /// or have field ID conflicts.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_not_implemented")]
    #[serde(deserialize_with = "deserialize_not_implemented")]
    #[builder(default)]
    pub name_mapping: Option<Arc<NameMapping>>,

    /// Whether this scan task should treat column names as case-sensitive when binding predicates.
    pub case_sensitive: bool,

    /// Key metadata for encrypted data files (Parquet Modular Encryption).
    /// When present, the reader uses this to build `FileDecryptionProperties`.
    ///
    /// Note on the trust boundary: for the standard encryption scheme this
    /// carries `StandardKeyMetadata`, whose payload is the *plaintext* DEK.
    /// Because `FileScanTask` derives `Serialize`, that plaintext DEK is part
    /// of the serialized scan plan should these tasks ever be serialized and sent
    /// over the network.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub key_metadata: Option<Box<[u8]>>,

    /// Explicit file-absolute row positions to KEEP (sorted strictly
    /// ascending) — an externally-known keep set applied to the parquet
    /// reader as a `RowSelection` (page-index skip + row mask), intersected
    /// with any delete- or predicate-derived selection. The merge
    /// late-fetch sets it to the victim positions so only the pages
    /// containing victims decode. Positions keep the reader-emitted `_pos`
    /// coordinate space (file-absolute — unaffected by row-group pruning or
    /// row selection). A position in a pruned row group or past EOF errors
    /// loudly rather than silently dropping a row.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub row_selection_positions: Option<Arc<Vec<u64>>>,
}

impl FileScanTask {
    /// Returns the data file path of this file scan task.
    pub fn data_file_path(&self) -> &str {
        &self.data_file_path
    }

    /// Returns the project field id of this file scan task.
    pub fn project_field_ids(&self) -> &[i32] {
        &self.project_field_ids
    }

    /// Returns the predicate of this file scan task.
    pub fn predicate(&self) -> Option<&BoundPredicate> {
        self.predicate.as_ref()
    }

    /// Returns the schema of this file scan task as a reference
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Returns the schema of this file scan task as a SchemaRef
    pub fn schema_ref(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// The share of the file's bytes this task's projection will read,
    /// estimated from the manifest's per-field `column_sizes`: each
    /// projected field contributes its recorded size plus that of every
    /// field nested beneath it (a struct rolls up its leaves; a list its
    /// element; a map its key and value).
    ///
    /// A variant column's sub-columns carry no field ids of their own, so a
    /// writer may record no size for the column at all — the file's bytes
    /// the manifest leaves unrecorded are then exactly those. A projected
    /// variant with no recorded size is therefore assumed to own the
    /// unrecorded remainder, erring toward "reads a lot". Field ids the
    /// schema does not know (reader-added metadata columns) contribute
    /// nothing. `None` when no sizes were recorded or the file size is
    /// unknown.
    pub fn projected_byte_share(&self) -> Option<f64> {
        let sizes = self.column_sizes.as_ref().filter(|s| !s.is_empty())?;
        if self.file_size_in_bytes == 0 {
            return None;
        }
        let recorded: u64 = sizes.values().sum();
        let unrecorded = self.file_size_in_bytes.saturating_sub(recorded);
        let mut projected: u64 = 0;
        let mut owns_unrecorded = false;
        let mut ids = Vec::new();
        for id in &self.project_field_ids {
            let Some(field) = self.schema.field_by_id(*id) else {
                continue;
            };
            ids.clear();
            collect_field_ids(field, &mut ids);
            let bytes: u64 = ids.iter().filter_map(|i| sizes.get(i)).sum();
            projected += bytes;
            if field.field_type.is_variant() && !ids.iter().any(|i| sizes.contains_key(i)) {
                owns_unrecorded = true;
            }
        }
        if owns_unrecorded {
            projected += unrecorded;
        }
        Some(projected.min(self.file_size_in_bytes) as f64 / self.file_size_in_bytes as f64)
    }
}

/// `field`'s id followed by the ids of every field nested beneath it.
fn collect_field_ids(field: &NestedField, out: &mut Vec<i32>) {
    out.push(field.id);
    match field.field_type.as_ref() {
        Type::Struct(s) => s.fields().iter().for_each(|f| collect_field_ids(f, out)),
        Type::List(l) => collect_field_ids(&l.element_field, out),
        Type::Map(m) => {
            collect_field_ids(&m.key_field, out);
            collect_field_ids(&m.value_field, out);
        }
        Type::Primitive(_) | Type::Variant(_) => {}
    }
}

#[derive(Debug)]
pub(crate) struct DeleteFileContext {
    pub(crate) manifest_entry: ManifestEntryRef,
    pub(crate) partition_spec_id: i32,
}

impl From<&DeleteFileContext> for FileScanTaskDeleteFile {
    fn from(ctx: &DeleteFileContext) -> Self {
        FileScanTaskDeleteFile::builder()
            .with_file_path(ctx.manifest_entry.file_path().to_string())
            .with_file_size_in_bytes(ctx.manifest_entry.file_size_in_bytes())
            .with_file_type(ctx.manifest_entry.content_type())
            .with_partition_spec_id(ctx.partition_spec_id)
            .with_equality_ids(ctx.manifest_entry.data_file.equality_ids.clone())
            .with_file_format(ctx.manifest_entry.data_file().file_format())
            .with_referenced_data_file(ctx.manifest_entry.data_file().referenced_data_file())
            .with_content_offset(ctx.manifest_entry.data_file().content_offset())
            .with_content_size_in_bytes(ctx.manifest_entry.data_file().content_size_in_bytes())
            .with_key_metadata(
                ctx.manifest_entry
                    .data_file
                    .key_metadata
                    .as_deref()
                    .map(Box::from),
            )
            .build()
    }
}

/// A task to scan part of file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TypedBuilder)]
#[builder(field_defaults(setter(prefix = "with_")))]
pub struct FileScanTaskDeleteFile {
    /// The delete file path
    pub file_path: String,

    /// The total size of the delete file in bytes, from the manifest entry.
    pub file_size_in_bytes: u64,

    /// delete file type
    pub file_type: DataContentType,

    /// partition id
    pub partition_spec_id: i32,

    /// equality ids for equality deletes (null for anything other than equality-deletes)
    #[builder(default)]
    pub equality_ids: Option<Vec<i32>>,

    /// Format of the delete file. `Puffin` marks a V3 deletion vector
    /// (as opposed to a Parquet positional/equality delete file).
    #[builder(default = DataFileFormat::Parquet)]
    pub file_format: DataFileFormat,

    /// The data file a V3 deletion vector applies to (the manifest entry's
    /// `referenced_data_file`). `None` for positional/equality delete files.
    #[builder(default)]
    pub referenced_data_file: Option<String>,

    /// Offset of the deletion-vector blob within the file (the manifest entry's
    /// `content_offset`). `None` for positional/equality delete files.
    #[builder(default)]
    pub content_offset: Option<i64>,

    /// Size in bytes of the deletion-vector blob (the manifest entry's
    /// `content_size_in_bytes`). `None` for positional/equality delete files.
    #[builder(default)]
    pub content_size_in_bytes: Option<i64>,
    /// Key metadata for encrypted delete files (Parquet Modular Encryption).
    /// When present, the reader uses this to build `FileDecryptionProperties`.
    ///
    /// Same plaintext-DEK trust boundary as [`FileScanTask::key_metadata`]:
    /// this is serialized into the scan plan and crosses the planner -> worker
    /// channel in the clear for the standard encryption scheme.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub key_metadata: Option<Box<[u8]>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ListType, MapType, PrimitiveType, StructType, VariantType};

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::required(3, "a", Type::Primitive(PrimitiveType::Long))
                                .into(),
                            NestedField::optional(4, "b", Type::Primitive(PrimitiveType::String))
                                .into(),
                        ])),
                    )
                    .into(),
                    NestedField::optional(
                        5,
                        "tags",
                        Type::List(ListType::new(
                            NestedField::list_element(6, Type::Primitive(PrimitiveType::String), true)
                                .into(),
                        )),
                    )
                    .into(),
                    NestedField::optional(
                        7,
                        "m",
                        Type::Map(MapType::new(
                            NestedField::map_key_element(8, Type::Primitive(PrimitiveType::String))
                                .into(),
                            NestedField::map_value_element(
                                9,
                                Type::Primitive(PrimitiveType::Long),
                                true,
                            )
                            .into(),
                        )),
                    )
                    .into(),
                    NestedField::optional(10, "v", Type::Variant(VariantType)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn task(
        file_size: u64,
        project: Vec<i32>,
        column_sizes: Option<HashMap<i32, u64>>,
    ) -> FileScanTask {
        FileScanTask::builder()
            .with_file_size_in_bytes(file_size)
            .with_start(0)
            .with_length(file_size)
            .with_data_file_path("f.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema())
            .with_project_field_ids(project)
            .with_column_sizes(column_sizes)
            .with_case_sensitive(false)
            .build()
    }

    /// Leaves only: the variant column (10) has no entry — its bytes are
    /// the 500 the file holds beyond the 500 recorded.
    fn sizes() -> HashMap<i32, u64> {
        HashMap::from([(1, 100), (3, 150), (4, 50), (6, 100), (8, 40), (9, 60)])
    }

    fn share(project: Vec<i32>) -> f64 {
        task(1000, project, Some(sizes())).projected_byte_share().unwrap()
    }

    #[test]
    fn primitive_is_its_own_recorded_size() {
        assert_eq!(share(vec![1]), 0.1);
    }

    #[test]
    fn nested_fields_roll_up_to_the_projected_field() {
        assert_eq!(share(vec![2]), 0.2, "struct = its leaves");
        assert_eq!(share(vec![3]), 0.15, "a nested leaf projected directly");
        assert_eq!(share(vec![5]), 0.1, "list = its element");
        assert_eq!(share(vec![7]), 0.1, "map = key + value");
    }

    #[test]
    fn variant_without_a_recorded_size_owns_the_unrecorded_bytes() {
        assert_eq!(share(vec![10]), 0.5);
        assert_eq!(share(vec![1, 10]), 0.6);
        assert_eq!(share(vec![1, 2, 5, 7, 10]), 1.0, "the full width");
    }

    #[test]
    fn variant_with_a_recorded_size_is_counted_like_any_field() {
        let mut sizes = sizes();
        sizes.insert(10, 480);
        let t = task(1000, vec![10], Some(sizes));
        assert_eq!(t.projected_byte_share(), Some(0.48));
    }

    #[test]
    fn unknown_field_ids_contribute_nothing() {
        assert_eq!(share(vec![i32::MAX - 2]), 0.0);
        assert_eq!(share(vec![1, i32::MAX - 2]), 0.1);
    }

    #[test]
    fn share_never_exceeds_one() {
        let t = task(300, vec![1, 2, 5, 7], Some(sizes()));
        assert_eq!(t.projected_byte_share(), Some(1.0));
    }

    #[test]
    fn no_sizes_or_no_file_size_means_unknown() {
        assert_eq!(task(1000, vec![1], None).projected_byte_share(), None);
        assert_eq!(
            task(1000, vec![1], Some(HashMap::new())).projected_byte_share(),
            None
        );
        assert_eq!(task(0, vec![1], Some(sizes())).projected_byte_share(), None);
    }
}
