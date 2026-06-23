//! Engine entry — load a table, plan its compaction, rewrite each group.
//!
//! `plan_table` (scan -> planner) is the real read+plan path; `current_data_files`
//! enumerates the current-snapshot DataFiles (what the rewrite removes);
//! `compact_table` ties them to the rewrite loop. The rewrite's read -> sort ->
//! write internals are the remaining net-new pieces (`rewrite.rs`); the plan,
//! manifest-enum, and commit boundaries here are wired and correct.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use futures::TryStreamExt;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataContentType, DataFile, ManifestContentType, ManifestList};
use iceberg::table::Table;
use iceberg::{Catalog, TableIdent};

use crate::config::Config;
use crate::planner::{plan_compaction, Plan};
use crate::rewrite::{commit_rewrite, read_sort_write};

/// Scan a table's current data files and bin-pack the candidates into a `Plan`.
pub async fn plan_table(table: &Table, cfg: &Config) -> Result<Plan> {
    let tasks: Vec<FileScanTask> = table
        .scan()
        .select_all()
        .build()?
        .plan_files()
        .await?
        .try_collect()
        .await?;
    Ok(plan_compaction(tasks, cfg))
}

/// Enumerate the table's current-snapshot DATA files, keyed by file path.
///
/// Walks the current snapshot's manifest list -> data manifests -> live entries,
/// using only iceberg-rust's public API (`ManifestList::parse_with_version`,
/// `ManifestFile::load_manifest`, `ManifestEntry::{is_alive, content_type,
/// data_file}`). Delete-file manifests and dead/delete entries are skipped.
/// These DataFiles are what `rewrite_group` removes via RowDelta.
pub async fn current_data_files(table: &Table) -> Result<HashMap<String, DataFile>> {
    let mut out = HashMap::new();
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(out); // empty table — nothing to compact
    };
    let bytes = table
        .file_io()
        .new_input(snapshot.manifest_list())?
        .read()
        .await?;
    let manifest_list =
        ManifestList::parse_with_version(&bytes, table.metadata().format_version())?;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue; // skip delete-file manifests
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                let df = entry.data_file().clone();
                out.insert(df.file_path().to_string(), df);
            }
        }
    }
    Ok(out)
}

/// Enumerate the current-snapshot DELETE files (V3 deletion vectors), keyed by
/// **each delete file's own path**. Which deletes to reabsorb is decided from the
/// scan's per-task binding (`FileScanTask.deletes`, see `compact_table`) — matching
/// iceberg-go (`CollectSafeDeletionVectors(group.Tasks)`) and iceberg-java
/// (`RewriteFileGroup.danglingDVs()` = `tasks.flatMap(t -> t.deletes())`); this map
/// only resolves a bound delete's path back to its full `DataFile` (needed to mark
/// it removed). Keyed by the delete's OWN path, NOT `referenced_data_file`:
/// iceberg-rust's `referenced_data_file()` accessor returns None for some
/// cross-engine (duckdb-written) DVs, so a referenced_data_file map silently misses
/// them and the rewrite leaves them dangling (the multi-DV corruption). The delete's
/// own `file_path` is always populated, and it's what the scan task carries.
pub async fn current_delete_files(table: &Table) -> Result<HashMap<String, DataFile>> {
    let mut out = HashMap::new();
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(out);
    };
    let bytes = table
        .file_io()
        .new_input(snapshot.manifest_list())?
        .read()
        .await?;
    let manifest_list =
        ManifestList::parse_with_version(&bytes, table.metadata().format_version())?;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue; // delete-file manifests only
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() {
                let df = entry.data_file().clone();
                out.insert(df.file_path().to_string(), df); // key by the delete file's OWN path
            }
        }
    }
    Ok(out)
}

/// Compact one table end-to-end: load -> enumerate current data + delete files
/// -> plan -> read+sort+write each group, then swap them all in via ONE commit.
///
/// Each group is read/sorted/written independently (bounded memory), but every
/// group's new files + the old files they replace + the DVs they reabsorb are
/// **accumulated and committed in a single `RewriteFiles`** (one atomic `Replace`
/// snapshot). Committing per group instead re-ran the manifest carry-forward over
/// each prior snapshot, which duplicated data files. All reads are against the
/// loaded snapshot, so the once-built `files`/`delete_files` maps stay valid.
pub async fn compact_table(catalog: &dyn Catalog, ident: &TableIdent, cfg: &Config) -> Result<()> {
    let table = catalog.load_table(ident).await?;
    let files = current_data_files(&table).await?;
    let delete_files = current_delete_files(&table).await?;
    let plan = plan_table(&table, cfg).await?;

    let mut all_removed: Vec<DataFile> = Vec::new();
    let mut all_removed_deletes: Vec<DataFile> = Vec::new();
    let mut all_added: Vec<DataFile> = Vec::new();
    let mut seen_delete_paths: HashSet<String> = HashSet::new();
    for group in &plan.groups {
        let added = read_sort_write(&table, group).await?;
        if added.is_empty() {
            continue; // no live rows to write (e.g. fully-deleted group) — never remove without replacement
        }
        all_removed.extend(group.tasks.iter().filter_map(|t| files.get(&t.data_file_path).cloned()));
        // Delete files (DVs) the SCAN bound to these rewritten data files — now
        // dangling, so reabsorbed in the same commit. Sourced from `task.deletes`
        // (the scan's per-file binding, the same the read applies), NOT a
        // referenced_data_file lookup — matching iceberg-go (CollectSafeDeletionVectors)
        // and iceberg-java (RewriteFileGroup.danglingDVs). `delete_files` only
        // resolves each bound delete's path -> its DataFile; dedup by path.
        for t in &group.tasks {
            for d in &t.deletes {
                if seen_delete_paths.insert(d.file_path.clone()) {
                    if let Some(dv) = delete_files.get(&d.file_path) {
                        all_removed_deletes.push(dv.clone());
                    }
                }
            }
        }
        all_added.extend(added);
    }

    if all_added.is_empty() {
        return Ok(()); // nothing to compact
    }
    commit_rewrite(&table, catalog, all_removed, all_removed_deletes, all_added).await?;
    Ok(())
}
