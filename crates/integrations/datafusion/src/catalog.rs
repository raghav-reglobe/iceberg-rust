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

use datafusion::catalog::{CatalogProvider, SchemaProvider};
use futures::future::try_join_all;
use iceberg::{Catalog, NamespaceIdent, Result};

use crate::schema::IcebergSchemaProvider;

/// Provides an interface to manage and access multiple schemas
/// within an Iceberg [`Catalog`].
///
/// Acts as a centralized catalog provider that aggregates
/// multiple [`SchemaProvider`], each associated with distinct namespaces.
#[derive(Debug)]
pub struct IcebergCatalogProvider {
    /// A `HashMap` where keys are namespace names
    /// and values are dynamic references to objects implementing the
    /// [`SchemaProvider`] trait.
    schemas: HashMap<String, Arc<dyn SchemaProvider>>,
}

impl IcebergCatalogProvider {
    /// Asynchronously tries to construct a new [`IcebergCatalogProvider`]
    /// using the given client to fetch and initialize schema providers for
    /// each namespace in the Iceberg [`Catalog`].
    ///
    /// This method retrieves the list of namespace names
    /// attempts to create a schema provider for each namespace, and
    /// collects these providers into a `HashMap`.
    pub async fn try_new(client: Arc<dyn Catalog>) -> Result<Self> {
        // TODO:
        // Schemas and providers should be cached and evicted based on time
        // As of right now; schemas might become stale.
        let schema_names: Vec<_> = client
            .list_namespaces(None)
            .await?
            .iter()
            .flat_map(|ns| ns.as_ref().clone())
            .collect();

        let providers = try_join_all(
            schema_names
                .iter()
                .map(|name| {
                    IcebergSchemaProvider::try_new(
                        client.clone(),
                        NamespaceIdent::new(name.clone()),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .await?;

        let schemas: HashMap<String, Arc<dyn SchemaProvider>> = schema_names
            .into_iter()
            .zip(providers)
            .map(|(name, provider)| {
                let provider = Arc::new(provider) as Arc<dyn SchemaProvider>;
                (name, provider)
            })
            .collect();

        Ok(IcebergCatalogProvider { schemas })
    }

    /// Scoped mount: providers for ONLY the given `namespace -> table names`
    /// map — zero `list_namespaces`/`list_tables` calls and exactly one
    /// `load_table` per named table. Use when the SQL references a known
    /// table set (the merge doorway: one bronze source + one silver target);
    /// a full `try_new` of a production catalog loads EVERY table's metadata
    /// per call. A reference to any table outside the scope fails at
    /// planning ("table not found") — deliberately loud.
    pub async fn try_new_scoped(
        client: Arc<dyn Catalog>,
        scope: HashMap<String, Vec<String>>,
    ) -> Result<Self> {
        let entries: Vec<(String, Vec<String>)> = scope.into_iter().collect();
        let providers = try_join_all(
            entries
                .iter()
                .map(|(ns, tables)| {
                    IcebergSchemaProvider::try_new_scoped(
                        client.clone(),
                        NamespaceIdent::new(ns.clone()),
                        tables.clone(),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .await?;

        let schemas: HashMap<String, Arc<dyn SchemaProvider>> = entries
            .into_iter()
            .map(|(ns, _)| ns)
            .zip(providers)
            .map(|(name, provider)| (name, Arc::new(provider) as Arc<dyn SchemaProvider>))
            .collect();

        Ok(IcebergCatalogProvider { schemas })
    }

    /// Restrict `namespace.table`'s scans to the given data-file paths
    /// (externally planned file subset; deletes still apply). Errors if the
    /// namespace or table is not mounted — a mistyped identifier must fail
    /// loudly rather than silently scan the whole table.
    pub fn with_table_scan_file_allowlist(
        &self,
        namespace: &str,
        table: &str,
        files: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let schema = self.schemas.get(namespace).ok_or_else(|| {
            iceberg::Error::new(
                iceberg::ErrorKind::NamespaceNotFound,
                format!("namespace `{namespace}` not mounted (scan-file allowlist)"),
            )
        })?;
        let schema = (schema.as_ref() as &dyn std::any::Any)
            .downcast_ref::<IcebergSchemaProvider>()
            .ok_or_else(|| {
                iceberg::Error::new(
                    iceberg::ErrorKind::Unexpected,
                    format!("schema provider for `{namespace}` is not Iceberg-backed"),
                )
            })?;
        schema.set_scan_file_allowlist(table, files)
    }
}

impl CatalogProvider for IcebergCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).cloned()
    }
}
