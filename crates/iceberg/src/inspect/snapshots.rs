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

use arrow_array::RecordBatch;
use arrow_array::builder::{LargeStringBuilder, MapBuilder, MapFieldNames, PrimitiveBuilder};
use arrow_array::types::{Int64Type, TimestampMicrosecondType};
use arrow_schema::{DataType, Field};
use futures::{StreamExt, stream};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::Result;
use crate::arrow::{DEFAULT_MAP_FIELD_NAME, schema_to_arrow_schema};
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{
    MAP_KEY_FIELD_NAME, MAP_VALUE_FIELD_NAME, MapType, NestedField, PrimitiveType, Type,
};
use crate::table::Table;

/// Snapshots table.
pub struct SnapshotsTable<'a> {
    table: &'a Table,
}

impl<'a> SnapshotsTable<'a> {
    /// Create a new Snapshots table instance.
    pub fn new(table: &'a Table) -> Self {
        Self { table }
    }

    /// Returns the iceberg schema of the snapshots table.
    pub fn schema(&self) -> crate::spec::Schema {
        let fields = vec![
            NestedField::required(
                1,
                "committed_at",
                Type::Primitive(PrimitiveType::Timestamptz),
            ),
            NestedField::required(2, "snapshot_id", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(3, "parent_id", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(4, "operation", Type::Primitive(PrimitiveType::String)),
            NestedField::optional(5, "manifest_list", Type::Primitive(PrimitiveType::String)),
            NestedField::optional(
                6,
                "summary",
                Type::Map(MapType {
                    key_field: Arc::new(NestedField::map_key_element(
                        7,
                        Type::Primitive(PrimitiveType::String),
                    )),
                    value_field: Arc::new(NestedField::map_value_element(
                        8,
                        Type::Primitive(PrimitiveType::String),
                        false,
                    )),
                }),
            ),
        ];
        crate::spec::Schema::builder()
            .with_fields(fields.into_iter().map(|f| f.into()))
            .build()
            .unwrap()
    }

    /// Scans the snapshots table.
    pub async fn scan(&self) -> Result<ArrowRecordBatchStream> {
        let schema = schema_to_arrow_schema(&self.schema())?;

        let mut committed_at =
            PrimitiveBuilder::<TimestampMicrosecondType>::new().with_timezone("+00:00");
        let mut snapshot_id = PrimitiveBuilder::<Int64Type>::new();
        let mut parent_id = PrimitiveBuilder::<Int64Type>::new();
        // String columns build as LargeUtf8 to match the iceberg-derived
        // arrow schema (the LargeUtf8 flip).
        let mut operation = LargeStringBuilder::new();
        let mut manifest_list = LargeStringBuilder::new();
        let mut summary = MapBuilder::new(
            Some(MapFieldNames {
                entry: DEFAULT_MAP_FIELD_NAME.to_string(),
                key: MAP_KEY_FIELD_NAME.to_string(),
                value: MAP_VALUE_FIELD_NAME.to_string(),
            }),
            LargeStringBuilder::new(),
            LargeStringBuilder::new(),
        )
        .with_keys_field(Arc::new(
            Field::new(MAP_KEY_FIELD_NAME, DataType::LargeUtf8, false).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "7".to_string())]),
            ),
        ))
        .with_values_field(Arc::new(
            Field::new(MAP_VALUE_FIELD_NAME, DataType::LargeUtf8, true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "8".to_string())]),
            ),
        ));
        for snapshot in self.table.metadata().snapshots() {
            committed_at.append_value(snapshot.timestamp_ms() * 1000);
            snapshot_id.append_value(snapshot.snapshot_id());
            parent_id.append_option(snapshot.parent_snapshot_id());
            manifest_list.append_value(snapshot.manifest_list());
            operation.append_value(snapshot.summary().operation.as_str());
            for (key, value) in &snapshot.summary().additional_properties {
                summary.keys().append_value(key);
                summary.values().append_value(value);
            }
            summary.append(true)?;
        }

        let batch = RecordBatch::try_new(Arc::new(schema), vec![
            Arc::new(committed_at.finish()),
            Arc::new(snapshot_id.finish()),
            Arc::new(parent_id.finish()),
            Arc::new(operation.finish()),
            Arc::new(manifest_list.finish()),
            Arc::new(summary.finish()),
        ])?;

        Ok(stream::iter(vec![Ok(batch)]).boxed())
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use futures::TryStreamExt;

    use crate::scan::tests::TableTestFixture;
    use crate::test_utils::check_record_batches;

    #[tokio::test]
    async fn test_snapshots_table() {
        let table = TableTestFixture::new().table;

        let batch_stream = table.inspect().snapshots().scan().await.unwrap();

        check_record_batches(
            batch_stream.try_collect::<Vec<_>>().await.unwrap(),
            expect![[r#"
                Field { "committed_at": Timestamp(µs, "+00:00"), metadata: {"PARQUET:field_id": "1"} },
                Field { "snapshot_id": Int64, metadata: {"PARQUET:field_id": "2"} },
                Field { "parent_id": nullable Int64, metadata: {"PARQUET:field_id": "3"} },
                Field { "operation": nullable LargeUtf8, metadata: {"PARQUET:field_id": "4"} },
                Field { "manifest_list": nullable LargeUtf8, metadata: {"PARQUET:field_id": "5"} },
                Field { "summary": nullable Map("key_value": non-null Struct("key": non-null LargeUtf8, metadata: {"PARQUET:field_id": "7"}, "value": LargeUtf8, metadata: {"PARQUET:field_id": "8"}), unsorted), metadata: {"PARQUET:field_id": "6"} }"#]],
            expect![[r#"
                committed_at: PrimitiveArray<Timestamp(µs, "+00:00")>
                [
                  2018-01-04T21:22:35.770+00:00,
                  2019-04-12T20:29:15.770+00:00,
                ],
                snapshot_id: PrimitiveArray<Int64>
                [
                  3051729675574597004,
                  3055729675574597004,
                ],
                parent_id: PrimitiveArray<Int64>
                [
                  null,
                  3051729675574597004,
                ],
                operation: LargeStringArray
                [
                  "append",
                  "append",
                ],
                manifest_list: (skipped),
                summary: MapArray
                [
                  StructArray
                -- validity:
                [
                ]
                [
                -- child 0: "key" (LargeUtf8)
                LargeStringArray
                [
                ]
                -- child 1: "value" (LargeUtf8)
                LargeStringArray
                [
                ]
                ],
                  StructArray
                -- validity:
                [
                ]
                [
                -- child 0: "key" (LargeUtf8)
                LargeStringArray
                [
                ]
                -- child 1: "value" (LargeUtf8)
                LargeStringArray
                [
                ]
                ],
                ]"#]],
            &["manifest_list"],
            Some("committed_at"),
        );
    }
}
