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

use typed_builder::TypedBuilder;

use super::{FormatVersion, ManifestContentType, PartitionSpec, Schema};
use crate::error::Result;
use crate::spec::{PartitionField, SchemaId, SchemaRef, TableMetadataRef};
use crate::{Error, ErrorKind};

/// Meta data of a manifest that is stored in the key-value metadata of the Avro file
#[derive(Debug, PartialEq, Clone, Eq, TypedBuilder)]
pub struct ManifestMetadata {
    /// The table schema at the time the manifest
    /// was written
    pub schema: SchemaRef,
    /// ID of the schema used to write the manifest as a string
    pub schema_id: SchemaId,
    /// The partition spec used to write the manifest
    pub partition_spec: PartitionSpec,
    /// Table format version number of the manifest as a string
    pub format_version: FormatVersion,
    /// Type of content files tracked by the manifest: “data” or “deletes”
    pub content: ManifestContentType,
}

impl ManifestMetadata {
    /// Parse from metadata in avro file.
    pub fn parse(meta: &HashMap<String, Vec<u8>>) -> Result<Self> {
        Self::parse_with(meta, None)
    }

    /// Parse from the avro file's key-value metadata, preferring the table
    /// metadata's schema and partition spec (looked up by the manifest's
    /// `schema-id` / `partition-spec-id`) over the manifest's self-described
    /// `schema` / `partition-spec` keys.
    ///
    /// A manifest's embedded `schema` key is redundant with the authoritative
    /// table metadata, and some writers (e.g. duckdb-iceberg) store a
    /// non-conformant value there (the manifest_entry Avro record schema rather
    /// than the Iceberg table schema). When `table_metadata` is provided and
    /// contains the referenced schema and spec, those are used and the
    /// manifest's own `schema` / `partition-spec` keys are not parsed — mirroring
    /// iceberg-java's `ManifestReader(specsById)`, whose reading of the schema
    /// from manifest file metadata is deprecated. When `table_metadata` is
    /// `None` (or does not contain the referenced ids) the manifest's own
    /// metadata is parsed, preserving the previous self-describing behaviour.
    pub fn parse_with(
        meta: &HashMap<String, Vec<u8>>,
        table_metadata: Option<&TableMetadataRef>,
    ) -> Result<Self> {
        let schema_id: i32 = meta
            .get("schema-id")
            .map(|bs| {
                String::from_utf8_lossy(bs).parse().map_err(|err| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Fail to parse schema id in manifest metadata",
                    )
                    .with_source(err)
                })
            })
            .transpose()?
            .unwrap_or(0);
        let spec_id: i32 = meta
            .get("partition-spec-id")
            .map(|bs| {
                String::from_utf8_lossy(bs).parse().map_err(|err| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Fail to parse partition spec id in manifest metadata",
                    )
                    .with_source(err)
                })
            })
            .transpose()?
            .unwrap_or(0);
        let format_version = if let Some(bs) = meta.get("format-version") {
            serde_json::from_slice::<FormatVersion>(bs).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Fail to parse format version in manifest metadata",
                )
                .with_source(err)
            })?
        } else {
            FormatVersion::V1
        };
        let content = if let Some(v) = meta.get("content") {
            let v = String::from_utf8_lossy(v);
            v.parse()?
        } else {
            ManifestContentType::Data
        };

        // Prefer the authoritative table schema + partition spec when available,
        // bypassing the manifest's redundant (and sometimes non-conformant)
        // `schema` / `partition-spec` metadata keys.
        if let Some(table_metadata) = table_metadata
            && let (Some(schema), Some(partition_spec)) = (
                table_metadata.schema_by_id(schema_id),
                table_metadata.partition_spec_by_id(spec_id),
            )
        {
            // The `schema` stored here is used later to derive the partition
            // type (`partition_spec.partition_type(&schema)`) and to decode
            // each entry's partition values. It must therefore contain every
            // source column the partition spec references. A partition column
            // added by ALTER lives in `current_schema()` but may be absent from
            // the older `schema_by_id(schema_id)` the manifest was written with.
            // When that happens, resolve against the current schema instead —
            // it is the authoritative superset (Iceberg forbids incompatible
            // source-column type changes), matching iceberg-java's binding of
            // partition specs against the table schema.
            let schema = if schema_contains_spec_sources(schema, partition_spec.as_ref()) {
                schema.clone()
            } else {
                table_metadata.current_schema().clone()
            };
            return Ok(ManifestMetadata {
                schema,
                schema_id,
                partition_spec: partition_spec.as_ref().clone(),
                format_version,
                content,
            });
        }

        // Fallback. When table metadata is available (but the manifest's
        // recorded `schema-id` / `partition-spec-id` are not resolvable in it),
        // resolve the partition spec's source columns against the table's
        // CURRENT schema — the authoritative superset that contains every
        // column ever added, including a partition column added by ALTER. The
        // current schema is also used as `ManifestMetadata.schema` so the spec
        // and schema stay consistent for partition-type derivation + entry
        // decode. Only when no table metadata is available do we fall back to
        // the manifest's own self-describing `schema` key.
        let schema = if let Some(table_metadata) = table_metadata {
            table_metadata.current_schema().clone()
        } else {
            Arc::new({
                let bs = meta.get("schema").ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "schema is required in manifest metadata but not found",
                    )
                })?;
                serde_json::from_slice::<Schema>(bs).map_err(|err| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Fail to parse schema in manifest metadata",
                    )
                    .with_source(err)
                })?
            })
        };
        let fields = {
            let bs = meta.get("partition-spec").ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "partition-spec is required in manifest metadata but not found",
                )
            })?;
            serde_json::from_slice::<Vec<PartitionField>>(bs).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Fail to parse partition spec in manifest metadata",
                )
                .with_source(err)
            })?
        };
        let partition_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(spec_id)
            .add_unbound_fields(fields.into_iter().map(|f| f.into_unbound()))?
            .build()?;

        Ok(ManifestMetadata {
            schema,
            schema_id,
            partition_spec,
            format_version,
            content,
        })
    }

    /// Get the schema of table at the time manifest was written
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the ID of schema used to write the manifest
    pub fn schema_id(&self) -> SchemaId {
        self.schema_id
    }

    /// Get the partition spec used to write manifest
    pub fn partition_spec(&self) -> &PartitionSpec {
        &self.partition_spec
    }

    /// Get the table format version
    pub fn format_version(&self) -> &FormatVersion {
        &self.format_version
    }

    /// Get the type of content files tracked by manifest
    pub fn content(&self) -> &ManifestContentType {
        &self.content
    }
}

/// Returns true iff `schema` contains every source column referenced by the
/// partition spec's fields. Used to detect a stale schema that predates a
/// partition column added by ALTER (in which case the spec's `source_id` is
/// missing) so the caller can resolve against the current schema instead.
fn schema_contains_spec_sources(schema: &Schema, partition_spec: &PartitionSpec) -> bool {
    partition_spec
        .fields()
        .iter()
        .all(|f| schema.field_by_id(f.source_id).is_some())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::spec::{ManifestMetadata, TableMetadata};

    /// A `TableMetadata` whose CURRENT schema (id 1) has a `flag`
    /// boolean column (id 2) added by ALTER and is partitioned by it (identity,
    /// source-id 2), while an OLDER schema (id 0) lacks that column. The current
    /// partition spec is id 1; an older empty spec is id 0.
    fn table_metadata_with_added_partition_column() -> Arc<TableMetadata> {
        let json = r#"{
            "format-version": 2,
            "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
            "location": "s3://bucket/table",
            "last-sequence-number": 1,
            "last-updated-ms": 1602638573590,
            "last-column-id": 2,
            "current-schema-id": 1,
            "schemas": [
                {
                    "type": "struct",
                    "schema-id": 0,
                    "fields": [
                        {"id": 1, "name": "x", "required": true, "type": "long"}
                    ]
                },
                {
                    "type": "struct",
                    "schema-id": 1,
                    "fields": [
                        {"id": 1, "name": "x", "required": true, "type": "long"},
                        {"id": 2, "name": "flag", "required": false, "type": "boolean"}
                    ]
                }
            ],
            "default-spec-id": 1,
            "partition-specs": [
                {"spec-id": 0, "fields": []},
                {"spec-id": 1, "fields": [
                    {"name": "flag", "transform": "identity", "source-id": 2, "field-id": 1000}
                ]}
            ],
            "last-partition-id": 1000,
            "default-sort-order-id": 0,
            "sort-orders": [{"order-id": 0, "fields": []}],
            "properties": {},
            "current-snapshot-id": -1,
            "snapshots": [],
            "snapshot-log": [],
            "metadata-log": []
        }"#;
        Arc::new(serde_json::from_str::<TableMetadata>(json).unwrap())
    }

    /// Regression: a manifest written before `flag` was added embeds a
    /// stale `schema` (lacking id 2) and partitions by source-id 2. Its recorded
    /// `schema-id`/`partition-spec-id` are NOT resolvable in the table metadata
    /// (forcing the fallback). The fallback must resolve the partition spec —
    /// and derive its partition type — against the table's CURRENT schema, which
    /// DOES contain id 2.
    #[test]
    fn test_parse_with_resolves_added_partition_column_via_current_schema() {
        let table_metadata = table_metadata_with_added_partition_column();

        // schema-id / spec-id that the table metadata does NOT contain → fallback.
        let mut meta: HashMap<String, Vec<u8>> = HashMap::new();
        meta.insert("schema-id".to_string(), b"99".to_vec());
        meta.insert("partition-spec-id".to_string(), b"99".to_vec());
        meta.insert("format-version".to_string(), b"2".to_vec());
        meta.insert("content".to_string(), b"data".to_vec());
        // Manifest's embedded schema is stale: it lacks `flag` (id 2).
        meta.insert(
            "schema".to_string(),
            br#"{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"x","required":true,"type":"long"}]}"#
                .to_vec(),
        );
        // Manifest's embedded partition spec partitions by the added column.
        meta.insert(
            "partition-spec".to_string(),
            br#"[{"name":"flag","transform":"identity","source-id":2,"field-id":1000}]"#
                .to_vec(),
        );

        // Without table metadata, the manifest is self-describing and the stale
        // embedded schema can't resolve source-id 2 → spec build fails.
        assert!(ManifestMetadata::parse(&meta).is_err());

        // With table metadata, the fallback resolves against the current schema.
        let parsed = ManifestMetadata::parse_with(&meta, Some(&table_metadata)).unwrap();
        assert_eq!(parsed.partition_spec.spec_id(), 99);
        assert_eq!(parsed.partition_spec.fields().len(), 1);
        assert_eq!(parsed.partition_spec.fields()[0].source_id, 2);
        // The schema kept on the metadata must contain the spec's source column
        // so partition-value decode (partition_type) succeeds.
        assert!(parsed.schema.field_by_id(2).is_some());
        let part_type = parsed
            .partition_spec
            .partition_type(&parsed.schema)
            .unwrap();
        assert_eq!(part_type.fields().len(), 1);
    }

    /// The authoritative early-return path: when the manifest's `schema-id` /
    /// `partition-spec-id` ARE resolvable but the resolved (older) schema lacks
    /// the partition column added by ALTER, the kept schema must be swapped to
    /// the current schema so partition-type derivation succeeds.
    #[test]
    fn test_parse_with_authoritative_swaps_stale_schema_for_partition_type() {
        let table_metadata = table_metadata_with_added_partition_column();

        // schema-id 0 (the OLD schema, lacking `flag`) + spec-id 1 (the
        // NEW spec, partitioning by `flag`). Both resolve → authoritative
        // early-return, but the resolved schema is stale for the spec.
        let mut meta: HashMap<String, Vec<u8>> = HashMap::new();
        meta.insert("schema-id".to_string(), b"0".to_vec());
        meta.insert("partition-spec-id".to_string(), b"1".to_vec());
        meta.insert("format-version".to_string(), b"2".to_vec());
        meta.insert("content".to_string(), b"data".to_vec());

        let parsed = ManifestMetadata::parse_with(&meta, Some(&table_metadata)).unwrap();
        assert_eq!(parsed.schema_id, 0);
        assert_eq!(parsed.partition_spec.spec_id(), 1);
        // Schema kept for partition_type derivation must contain source-id 2.
        assert!(parsed.schema.field_by_id(2).is_some());
        parsed
            .partition_spec
            .partition_type(&parsed.schema)
            .expect("partition_type must resolve against the kept schema");
    }

    /// The table_metadata = None path stays self-describing: a normal manifest
    /// (no added column) parses from its own embedded `schema` / `partition-spec`.
    #[test]
    fn test_parse_without_table_metadata_is_self_describing() {
        let mut meta: HashMap<String, Vec<u8>> = HashMap::new();
        meta.insert("schema-id".to_string(), b"0".to_vec());
        meta.insert("partition-spec-id".to_string(), b"0".to_vec());
        meta.insert("format-version".to_string(), b"2".to_vec());
        meta.insert("content".to_string(), b"data".to_vec());
        meta.insert(
            "schema".to_string(),
            br#"{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"x","required":true,"type":"long"}]}"#
                .to_vec(),
        );
        meta.insert(
            "partition-spec".to_string(),
            br#"[{"name":"x","transform":"identity","source-id":1,"field-id":1000}]"#.to_vec(),
        );

        let parsed = ManifestMetadata::parse(&meta).unwrap();
        assert_eq!(parsed.schema_id, 0);
        assert_eq!(parsed.partition_spec.spec_id(), 0);
        assert_eq!(parsed.partition_spec.fields()[0].source_id, 1);
        parsed
            .partition_spec
            .partition_type(&parsed.schema)
            .unwrap();
    }

    /// Table metadata for a table whose `flag` partition column (boolean, field
    /// id 25) was ADDED by ALTER after earlier manifests were written. The
    /// pre-ALTER schema (schema-id 0) holds a variant `after`/`before` envelope
    /// plus a `meta` struct (sub-fields 21-24) — highest field id 24, NO `flag`.
    /// The current schema (schema-id 1) adds `flag` (25). Partition spec 1
    /// partitions by `source-id: 25` (identity). This is the realistic case: a
    /// manifest whose embedded schema predates an ALTER-added partition column.
    fn table_metadata_added_partition_with_variant() -> Arc<TableMetadata> {
        let meta_struct = r#"{"type":"struct","fields":[
            {"id":21,"name":"op","required":false,"type":"string"},
            {"id":22,"name":"offset","required":false,"type":"string"},
            {"id":23,"name":"ts","required":false,"type":"long"},
            {"id":24,"name":"target","required":false,"type":"string"}]}"#;
        // The pre-ALTER envelope fields, shared by both schema versions.
        let envelope = format!(
            r#"{{"id":1,"name":"op","required":false,"type":"string"}},
               {{"id":2,"name":"after","required":false,"type":"variant"}},
               {{"id":3,"name":"before","required":false,"type":"variant"}},
               {{"id":18,"name":"route","required":false,"type":"string"}},
               {{"id":19,"name":"ts_ms","required":false,"type":"long"}},
               {{"id":20,"name":"meta","required":false,"type":{meta_struct}}}"#
        );
        let json = format!(
            r#"{{
            "format-version": 3,
            "table-uuid": "a1b2c3d4-0000-4000-8000-000000000001",
            "location": "s3://bucket/table",
            "last-sequence-number": 1,
            "last-updated-ms": 1602638573590,
            "last-column-id": 25,
            "current-schema-id": 1,
            "schemas": [
                {{"type":"struct","schema-id":0,"fields":[{envelope}]}},
                {{"type":"struct","schema-id":1,"fields":[{envelope},
                    {{"id":25,"name":"flag","required":false,"type":"boolean"}}]}}
            ],
            "default-spec-id": 1,
            "partition-specs": [
                {{"spec-id": 0, "fields": []}},
                {{"spec-id": 1, "fields": [
                    {{"name":"flag","transform":"identity","source-id":25,"field-id":1000}}
                ]}}
            ],
            "last-partition-id": 1000,
            "next-row-id": 0,
            "default-sort-order-id": 0,
            "sort-orders": [{{"order-id": 0, "fields": []}}],
            "properties": {{}},
            "current-snapshot-id": -1,
            "snapshots": [],
            "snapshot-log": [],
            "metadata-log": []
        }}"#
        );
        Arc::new(serde_json::from_str::<TableMetadata>(&json).unwrap())
    }

    /// Regression: scanning a table partitioned by an ALTER-added column failed
    /// with `No column with source column id 25 in schema {... highest field id
    /// 24}` because the manifest's embedded schema predates the ALTER while its
    /// partition spec partitions by source-id 25. With the fix, the spec resolves
    /// against the table's CURRENT schema (which has id 25), so planning +
    /// partition-value decode succeed. Exercises a variant-bearing schema (the
    /// realistic case) via the fallback path (unresolvable schema-id/spec-id).
    #[test]
    fn test_parse_with_added_partition_col_variant_via_current_schema() {
        let table_metadata = table_metadata_added_partition_with_variant();

        let mut meta: HashMap<String, Vec<u8>> = HashMap::new();
        // schema-id / spec-id the table metadata does NOT carry -> forces the
        // fallback (the manifest's recorded ids weren't resolvable in metadata).
        meta.insert("schema-id".to_string(), b"99".to_vec());
        meta.insert("partition-spec-id".to_string(), b"99".to_vec());
        meta.insert("format-version".to_string(), b"2".to_vec());
        meta.insert("content".to_string(), b"data".to_vec());
        // Manifest's embedded schema is the STALE pre-ALTER envelope (no
        // `flag`; highest field id 24) — the case that triggered the bug.
        meta.insert(
            "schema".to_string(),
            br#"{"type":"struct","schema-id":0,"fields":[
                {"id":1,"name":"op","required":false,"type":"string"},
                {"id":2,"name":"after","required":false,"type":"variant"},
                {"id":3,"name":"before","required":false,"type":"variant"},
                {"id":18,"name":"route","required":false,"type":"string"},
                {"id":19,"name":"ts_ms","required":false,"type":"long"},
                {"id":20,"name":"meta","required":false,"type":{"type":"struct","fields":[
                    {"id":21,"name":"op","required":false,"type":"string"},
                    {"id":22,"name":"offset","required":false,"type":"string"},
                    {"id":23,"name":"ts","required":false,"type":"long"},
                    {"id":24,"name":"target","required":false,"type":"string"}]}}]}"#
                .to_vec(),
        );
        // Manifest partitions by the ALTER-added `flag` (source-id 25).
        meta.insert(
            "partition-spec".to_string(),
            br#"[{"name":"flag","transform":"identity","source-id":25,"field-id":1000}]"#
                .to_vec(),
        );

        // Without table metadata, the stale embedded schema can't resolve the
        // partition spec's source-id 25 -> the missing-source-column error the
        // bug produced (the exact wording varies by which validation fires).
        let err = ManifestMetadata::parse(&meta).unwrap_err().to_string();
        assert!(
            err.contains("25") && err.contains("source"),
            "expected a missing partition source-column-25 error, got: {err}"
        );

        // With table metadata, the spec resolves against the current schema.
        let parsed = ManifestMetadata::parse_with(&meta, Some(&table_metadata)).unwrap();
        assert_eq!(parsed.partition_spec.spec_id(), 99);
        assert_eq!(parsed.partition_spec.fields().len(), 1);
        assert_eq!(parsed.partition_spec.fields()[0].source_id, 25);
        // The kept schema must contain `flag` (25) so entry partition-
        // value decode (`partition_type(&schema)` in mod.rs) succeeds.
        assert!(parsed.schema.field_by_id(25).is_some());
        let part_type = parsed
            .partition_spec
            .partition_type(&parsed.schema)
            .unwrap();
        assert_eq!(part_type.fields().len(), 1);
    }
}
