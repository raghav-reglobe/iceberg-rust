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

use std::mem::size_of_val;
use std::sync::Arc;

use crate::cache::ObjectBytesCacheRef;
use crate::encryption::EncryptionManager;
use crate::io::FileIO;
use crate::spec::{
    FormatVersion, Manifest, ManifestFile, ManifestList, ManifestListReader, SchemaId, SnapshotRef,
    TableMetadataRef,
};
use crate::{Error, ErrorKind, Result};

const DEFAULT_CACHE_SIZE_BYTES: u64 = 32 * 1024 * 1024; // 32MB

#[derive(Clone, Debug)]
pub(crate) enum CachedItem {
    ManifestList(Arc<ManifestList>),
    Manifest(Arc<Manifest>),
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) enum CachedObjectKey {
    ManifestList((String, FormatVersion, SchemaId)),
    Manifest(String),
}

/// Where the parsed objects are stored.
#[derive(Clone, Debug)]
enum CacheBackend {
    /// A per-table in-process cache (the default).
    Local(moka::future::Cache<CachedObjectKey, CachedItem>),
    /// A caller-supplied, path-keyed store of RAW FILE BYTES shared across
    /// tables (a process' worth of Table instances). Manifest /
    /// manifest-list files are immutable by path, so sharing the bytes is
    /// correct by construction; fetches on miss always go through THIS
    /// table's `FileIO` (different tables/catalogs may carry different
    /// credentials), and parsing happens per call — cached bytes are
    /// context-free, so the store may be memory-, disk-, or hybrid-backed.
    Shared(ObjectBytesCacheRef),
    /// No caching: every read fetches and parses.
    Disabled,
}

/// Caches metadata objects deserialized from immutable files
#[derive(Clone, Debug)]
pub struct ObjectCache {
    backend: CacheBackend,
    file_io: FileIO,
    encryption_manager: Option<Arc<EncryptionManager>>,
}

impl ObjectCache {
    /// Creates a new [`ObjectCache`]
    /// with the default cache size
    pub(crate) fn new(file_io: FileIO, encryption_manager: Option<Arc<EncryptionManager>>) -> Self {
        Self::new_with_capacity(file_io, DEFAULT_CACHE_SIZE_BYTES, encryption_manager)
    }

    /// Creates a new [`ObjectCache`]
    /// with a specific cache size
    pub(crate) fn new_with_capacity(
        file_io: FileIO,
        cache_size_bytes: u64,
        encryption_manager: Option<Arc<EncryptionManager>>,
    ) -> Self {
        if cache_size_bytes == 0 {
            Self::with_disabled_cache(file_io, encryption_manager)
        } else {
            Self {
                backend: CacheBackend::Local(
                    moka::future::Cache::builder()
                        .weigher(|_, val: &CachedItem| match val {
                            CachedItem::ManifestList(item) => size_of_val(item.as_ref()),
                            CachedItem::Manifest(item) => size_of_val(item.as_ref()),
                        } as u32)
                        .max_capacity(cache_size_bytes)
                        .build(),
                ),
                file_io,
                encryption_manager,
            }
        }
    }

    /// Creates a new [`ObjectCache`] backed by a caller-supplied shared
    /// store of raw file bytes (see [`crate::cache::ObjectBytesCache`]).
    /// The store holds the fetched bytes; fetching, decryption and parsing
    /// still run through this table's own `FileIO` /
    /// [`EncryptionManager`] per call.
    pub(crate) fn with_shared_bytes_cache(
        file_io: FileIO,
        bytes_cache: ObjectBytesCacheRef,
        encryption_manager: Option<Arc<EncryptionManager>>,
    ) -> Self {
        Self {
            backend: CacheBackend::Shared(bytes_cache),
            file_io,
            encryption_manager,
        }
    }

    /// Creates a new [`ObjectCache`]
    /// with caching disabled
    pub(crate) fn with_disabled_cache(
        file_io: FileIO,
        encryption_manager: Option<Arc<EncryptionManager>>,
    ) -> Self {
        Self {
            backend: CacheBackend::Disabled,
            file_io,
            encryption_manager,
        }
    }

    /// Retrieves an Arc [`Manifest`] from the cache
    /// or retrieves one from FileIO and parses it if not present
    pub(crate) async fn get_manifest(
        &self,
        manifest_file: &ManifestFile,
        table_metadata: &TableMetadataRef,
    ) -> Result<Arc<Manifest>> {
        let cache = match &self.backend {
            CacheBackend::Disabled => {
                return manifest_file
                    .load_manifest_with(&self.file_io, Some(table_metadata))
                    .await
                    .map(Arc::new);
            }
            CacheBackend::Shared(bytes_cache) => {
                let path = &manifest_file.manifest_path;
                // The manifest list records each manifest's exact length —
                // hand it to the store as a fetch-size hint.
                let hint = u64::try_from(manifest_file.manifest_length)
                    .ok()
                    .filter(|v| *v > 0);
                let avro = match bytes_cache.get_with_size_hint(path, hint).await {
                    Some(bytes) => bytes,
                    None => {
                        let bytes = self.file_io.new_input(path)?.read().await?;
                        bytes_cache.set(path, bytes.clone()).await;
                        bytes
                    }
                };
                // Parse per call with the CALLER's table metadata (mirrors
                // `ManifestFile::load_manifest_with`): the cached bytes stay
                // context-free.
                let (metadata, mut entries) =
                    Manifest::try_from_avro_bytes_with(&avro, Some(table_metadata))?;
                for entry in &mut entries {
                    entry.inherit_data(manifest_file);
                }
                return Ok(Arc::new(Manifest::new(metadata, entries)));
            }
            CacheBackend::Local(cache) => cache,
        };

        let key = CachedObjectKey::Manifest(manifest_file.manifest_path.clone());

        let cache_entry = cache
            .entry_by_ref(&key)
            .or_try_insert_with(self.fetch_and_parse_manifest(manifest_file, table_metadata))
            .await
            .map_err(|err| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to load manifest {}", manifest_file.manifest_path),
                )
                .with_source(err)
            })?
            .into_value();

        match cache_entry {
            CachedItem::Manifest(arc_manifest) => Ok(arc_manifest),
            _ => Err(Error::new(
                ErrorKind::Unexpected,
                format!("cached object for key '{key:?}' is not a Manifest"),
            )),
        }
    }

    /// Retrieves an Arc [`ManifestList`] from the cache
    /// or retrieves one from FileIO and parses it if not present
    pub(crate) async fn get_manifest_list(
        &self,
        snapshot: &SnapshotRef,
        table_metadata: &TableMetadataRef,
    ) -> Result<Arc<ManifestList>> {
        let cache = match &self.backend {
            CacheBackend::Disabled => {
                return ManifestListReader::new(
                    snapshot.clone(),
                    self.file_io.clone(),
                    table_metadata.clone(),
                    self.encryption_manager.clone(),
                )
                .load()
                .await
                .map(Arc::new);
            }
            CacheBackend::Shared(bytes_cache) => {
                // An ENCRYPTED manifest list bypasses the shared cache:
                // decrypted bytes must never land in a shareable (possibly
                // disk-backed) store, and the raw ciphertext would need the
                // per-snapshot key anyway.
                if snapshot.encryption_key_id().is_some() {
                    return ManifestListReader::new(
                        snapshot.clone(),
                        self.file_io.clone(),
                        table_metadata.clone(),
                        self.encryption_manager.clone(),
                    )
                    .load()
                    .await
                    .map(Arc::new);
                }
                // Path-keyed: a manifest-list path is written once, by one
                // snapshot of one table, and both the format version and the
                // snapshot's schema are fixed for that path.
                let path = snapshot.manifest_list();
                let bytes = match bytes_cache.get(path).await {
                    Some(bytes) => bytes,
                    None => {
                        let bytes = self.file_io.new_input(path)?.read().await?;
                        bytes_cache.set(path, bytes.clone()).await;
                        bytes
                    }
                };
                return ManifestList::parse_with_version(&bytes, table_metadata.format_version)
                    .map(Arc::new);
            }
            CacheBackend::Local(cache) => cache,
        };

        let key = CachedObjectKey::ManifestList((
            snapshot.manifest_list().to_string(),
            table_metadata.format_version,
            snapshot.schema_id().unwrap(),
        ));
        let cache_entry = cache
            .entry_by_ref(&key)
            .or_try_insert_with(self.fetch_and_parse_manifest_list(snapshot, table_metadata))
            .await
            .map_err(|err| {
                Arc::try_unwrap(err).unwrap_or_else(|err| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to load manifest list in cache",
                    )
                    .with_source(err)
                })
            })?
            .into_value();

        match cache_entry {
            CachedItem::ManifestList(arc_manifest_list) => Ok(arc_manifest_list),
            _ => Err(Error::new(
                ErrorKind::Unexpected,
                format!("cached object for path '{key:?}' is not a manifest list"),
            )),
        }
    }

    async fn fetch_and_parse_manifest(
        &self,
        manifest_file: &ManifestFile,
        table_metadata: &TableMetadataRef,
    ) -> Result<CachedItem> {
        let manifest = manifest_file
            .load_manifest_with(&self.file_io, Some(table_metadata))
            .await?;

        Ok(CachedItem::Manifest(Arc::new(manifest)))
    }

    async fn fetch_and_parse_manifest_list(
        &self,
        snapshot: &SnapshotRef,
        table_metadata: &TableMetadataRef,
    ) -> Result<CachedItem> {
        let manifest_list = ManifestListReader::new(
            snapshot.clone(),
            self.file_io.clone(),
            table_metadata.clone(),
            self.encryption_manager.clone(),
        )
        .load()
        .await?;

        Ok(CachedItem::ManifestList(Arc::new(manifest_list)))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use minijinja::value::Value;
    use minijinja::{AutoEscape, Environment, context};
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::TableIdent;
    use crate::io::{FileIO, OutputFile};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry,
        ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Struct, TableMetadata,
    };
    use crate::table::Table;
    use crate::test_utils::test_runtime;

    fn render_template(template: &str, ctx: Value) -> String {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.render_str(template, ctx).unwrap()
    }

    pub(super) struct TableTestFixture {
        table_location: String,
        pub(super) table: Table,
    }

    impl TableTestFixture {
        pub(super) fn new() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.as_os_str().to_str().unwrap())
                .runtime(test_runtime())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        fn next_manifest_file(&self) -> OutputFile {
            self.table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_{}.avro",
                    self.table_location,
                    Uuid::new_v4()
                ))
                .unwrap()
        }

        pub(super) async fn setup_manifest_files(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            // Write data files
            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(100)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(100))]))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            // Write to manifest list
            let manifest_list_writer = self
                .table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap()
                .writer()
                .await
                .unwrap();
            let mut manifest_list_write = ManifestListWriter::v2(
                manifest_list_writer,
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_get_manifest_list_and_manifest_from_disabled_cache() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let object_cache = ObjectCache::with_disabled_cache(fixture.table.file_io().clone(), None);

        let result_manifest_list = object_cache
            .get_manifest_list(
                fixture.table.metadata().current_snapshot().unwrap(),
                &fixture.table.metadata_ref(),
            )
            .await
            .unwrap();

        assert_eq!(result_manifest_list.entries().len(), 1);

        let manifest_file = result_manifest_list.entries().first().unwrap();
        let result_manifest = object_cache
            .get_manifest(manifest_file, &fixture.table.metadata_ref())
            .await
            .unwrap();

        assert_eq!(
            result_manifest
                .entries()
                .first()
                .unwrap()
                .file_path()
                .split("/")
                .last()
                .unwrap(),
            "1.parquet"
        );
    }

    #[tokio::test]
    async fn test_get_manifest_list_and_manifest_from_default_cache() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let object_cache = ObjectCache::new(fixture.table.file_io().clone(), None);

        // not in cache
        let result_manifest_list = object_cache
            .get_manifest_list(
                fixture.table.metadata().current_snapshot().unwrap(),
                &fixture.table.metadata_ref(),
            )
            .await
            .unwrap();

        assert_eq!(result_manifest_list.entries().len(), 1);

        // retrieve cached version
        let result_manifest_list = object_cache
            .get_manifest_list(
                fixture.table.metadata().current_snapshot().unwrap(),
                &fixture.table.metadata_ref(),
            )
            .await
            .unwrap();

        assert_eq!(result_manifest_list.entries().len(), 1);

        let manifest_file = result_manifest_list.entries().first().unwrap();

        // not in cache
        let result_manifest = object_cache
            .get_manifest(manifest_file, &fixture.table.metadata_ref())
            .await
            .unwrap();

        assert_eq!(
            result_manifest
                .entries()
                .first()
                .unwrap()
                .file_path()
                .split("/")
                .last()
                .unwrap(),
            "1.parquet"
        );

        // retrieve cached version
        let result_manifest = object_cache
            .get_manifest(manifest_file, &fixture.table.metadata_ref())
            .await
            .unwrap();

        assert_eq!(
            result_manifest
                .entries()
                .first()
                .unwrap()
                .file_path()
                .split("/")
                .last()
                .unwrap(),
            "1.parquet"
        );
    }

    #[test]
    fn test_manifest_metadata_parse_prefers_table_metadata_over_bad_schema() {
        use std::collections::HashMap;

        use crate::spec::ManifestMetadata;

        let fixture = TableTestFixture::new();
        let table_metadata = fixture.table.metadata_ref();
        let schema_id = table_metadata.current_schema().schema_id();
        let spec_id = table_metadata.default_partition_spec().spec_id();

        // Manifest key-value metadata whose `schema` value is non-conformant
        // (as written by some engines, e.g. duckdb-iceberg, which serialize the
        // manifest_entry Avro schema there using Avro type names like `array`),
        // but whose `schema-id` / `partition-spec-id` are valid.
        let mut meta: HashMap<String, Vec<u8>> = HashMap::new();
        meta.insert("schema-id".to_string(), schema_id.to_string().into_bytes());
        meta.insert(
            "partition-spec-id".to_string(),
            spec_id.to_string().into_bytes(),
        );
        meta.insert("format-version".to_string(), b"2".to_vec());
        meta.insert("content".to_string(), b"data".to_vec());
        meta.insert(
            "schema".to_string(),
            br#"{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"x","required":true,"type":{"type":"array","items":"int"}}]}"#
                .to_vec(),
        );

        // Parsing from the manifest's own metadata rejects the non-conformant schema.
        assert!(ManifestMetadata::parse(&meta).is_err());

        // With table metadata available, the authoritative schema/spec are used
        // (looked up by id) and the manifest's `schema` key is not parsed.
        let parsed = ManifestMetadata::parse_with(&meta, Some(&table_metadata)).unwrap();
        assert_eq!(parsed.schema.schema_id(), schema_id);
        assert_eq!(parsed.partition_spec.spec_id(), spec_id);
    }

    #[test]
    fn test_manifest_metadata_parse_self_describes_when_ids_not_recorded() {
        use std::collections::HashMap;

        use crate::spec::{ManifestMetadata, Type};

        let fixture = TableTestFixture::new();
        let table_metadata = fixture.table.metadata_ref();

        // A manifest written WITHOUT `schema-id` / `partition-spec-id` keys
        // (some writers omit them): the recorded-id LOOKUP must not fire (a
        // defaulted id 0 would mistype column bounds after a type promotion).
        //
        // FORK semantics diverge from the upstream PR here: with table
        // metadata available the fork resolves against the CURRENT schema
        // (the authoritative superset — platform schema evolution is add-only
        // and partition-spec sources must resolve; see parse_with's fallback
        // comment), while the pure self-describing path applies only without
        // table metadata.
        let mut meta: HashMap<String, Vec<u8>> = HashMap::new();
        meta.insert("format-version".to_string(), b"2".to_vec());
        meta.insert("content".to_string(), b"data".to_vec());
        meta.insert(
            "schema".to_string(),
            br#"{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"foo","required":false,"type":"long"}]}"#
                .to_vec(),
        );
        meta.insert("partition-spec".to_string(), b"[]".to_vec());

        // With table metadata: current schema wins (NOT a defaulted-id lookup,
        // NOT the embedded schema).
        let parsed = ManifestMetadata::parse_with(&meta, Some(&table_metadata)).unwrap();
        assert_eq!(
            parsed.schema.as_ref().as_struct(),
            table_metadata.current_schema().as_struct(),
            "fork fallback resolves against the current schema"
        );

        // Without table metadata: the manifest self-describes.
        let parsed = ManifestMetadata::parse_with(&meta, None).unwrap();
        let field = parsed.schema.field_by_id(1).unwrap();
        assert_eq!(field.name, "foo");
        assert_eq!(
            *field.field_type,
            Type::Primitive(crate::spec::PrimitiveType::Long)
        );
    }
}

#[cfg(test)]
mod shared_bytes_cache_tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use bytes::Bytes;

    use super::tests::TableTestFixture;
    use super::*;
    use crate::cache::{ObjectBytesCache, ObjectBytesCacheRef};
    use crate::spec::ManifestListWriter;

    /// A path-keyed bytes store that counts hits and sets — every fetch
    /// through IO is followed by exactly one `set`, so an unchanged
    /// set-count proves ZERO manifest / manifest-list reads went through
    /// `FileIO`.
    #[derive(Debug, Default)]
    struct CountingBytesCache {
        map: Mutex<HashMap<String, Bytes>>,
        hits: AtomicUsize,
        sets: AtomicUsize,
    }

    #[async_trait]
    impl ObjectBytesCache for CountingBytesCache {
        async fn get(&self, path: &str) -> Option<Bytes> {
            let out = self.map.lock().unwrap().get(path).cloned();
            if out.is_some() {
                self.hits.fetch_add(1, Ordering::SeqCst);
            }
            out
        }

        async fn set(&self, path: &str, bytes: Bytes) {
            self.sets.fetch_add(1, Ordering::SeqCst);
            self.map.lock().unwrap().insert(path.to_string(), bytes);
        }
    }

    fn shared_cache_for(
        fixture_table: &crate::table::Table,
        bytes_cache: ObjectBytesCacheRef,
    ) -> ObjectCache {
        ObjectCache::with_shared_bytes_cache(fixture_table.file_io().clone(), bytes_cache, None)
    }

    /// Two Table-loads' worth of caches sharing one bytes store: the second
    /// load performs ZERO manifest-list / manifest reads through IO and
    /// parses identical objects out of the cached bytes.
    #[tokio::test]
    async fn second_load_hits_shared_cache_without_io() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;
        let store = Arc::new(CountingBytesCache::default());
        let metadata = fixture.table.metadata_ref();
        let snapshot = fixture.table.metadata().current_snapshot().unwrap();

        // "First Table load": cold store — both objects fetched.
        let cache1 = shared_cache_for(&fixture.table, store.clone());
        let list1 = cache1.get_manifest_list(snapshot, &metadata).await.unwrap();
        let manifest_file = list1.entries().first().unwrap();
        let manifest1 = cache1.get_manifest(manifest_file, &metadata).await.unwrap();
        assert_eq!(store.sets.load(Ordering::SeqCst), 2);
        assert_eq!(store.hits.load(Ordering::SeqCst), 0);

        // "Second Table load": a FRESH per-table cache over the same shared
        // store — zero new sets (=> zero IO fetches), pure hits, equal
        // parsed objects.
        let cache2 = shared_cache_for(&fixture.table, store.clone());
        let list2 = cache2.get_manifest_list(snapshot, &metadata).await.unwrap();
        let manifest2 = cache2.get_manifest(manifest_file, &metadata).await.unwrap();
        assert_eq!(
            store.sets.load(Ordering::SeqCst),
            2,
            "second load must not re-fetch anything through IO"
        );
        assert_eq!(store.hits.load(Ordering::SeqCst), 2);
        assert_eq!(
            list1.entries(),
            list2.entries(),
            "same parsed manifest list"
        );
        assert_eq!(
            manifest1.entries().len(),
            manifest2.entries().len(),
            "same parsed manifest"
        );
        assert_eq!(
            manifest1.entries().first().unwrap().file_path(),
            manifest2.entries().first().unwrap().file_path()
        );
    }

    /// Freshness: a DIFFERENT snapshot's manifest-list pointer is a
    /// different path — the shared cache must fetch it, never serve the
    /// other snapshot's bytes.
    #[tokio::test]
    async fn different_snapshot_manifest_list_is_fetched() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Write a (distinct, empty) manifest list for the OTHER snapshot.
        let other_snapshot = fixture
            .table
            .metadata()
            .snapshots()
            .find(|s| {
                s.snapshot_id()
                    != fixture
                        .table
                        .metadata()
                        .current_snapshot()
                        .unwrap()
                        .snapshot_id()
            })
            .unwrap()
            .clone();
        let writer = fixture
            .table
            .file_io()
            .new_output(other_snapshot.manifest_list())
            .unwrap()
            .writer()
            .await
            .unwrap();
        let mut list_writer = ManifestListWriter::v2(
            writer,
            other_snapshot.snapshot_id(),
            other_snapshot.parent_snapshot_id(),
            other_snapshot.sequence_number(),
        );
        list_writer.add_manifests(vec![].into_iter()).unwrap();
        list_writer.close().await.unwrap();

        let store = Arc::new(CountingBytesCache::default());
        let metadata = fixture.table.metadata_ref();
        let current = fixture.table.metadata().current_snapshot().unwrap();

        let cache = shared_cache_for(&fixture.table, store.clone());
        let current_list = cache.get_manifest_list(current, &metadata).await.unwrap();
        assert_eq!(store.sets.load(Ordering::SeqCst), 1);

        // The other pointer MUST be fetched (a set, not a hit)...
        let other_list = cache
            .get_manifest_list(&other_snapshot, &metadata)
            .await
            .unwrap();
        assert_eq!(
            store.sets.load(Ordering::SeqCst),
            2,
            "a different manifest-list pointer must be fetched"
        );
        assert_eq!(store.hits.load(Ordering::SeqCst), 0);
        assert_ne!(current_list.entries().len(), other_list.entries().len());

        // ...and only repeats of the SAME pointer hit.
        let again = cache
            .get_manifest_list(&other_snapshot, &metadata)
            .await
            .unwrap();
        assert_eq!(store.sets.load(Ordering::SeqCst), 2);
        assert_eq!(store.hits.load(Ordering::SeqCst), 1);
        assert_eq!(other_list.entries().len(), again.entries().len());
    }
}
