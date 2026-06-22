//! Engine entry — load a table, plan its compaction, rewrite each group.
//!
//! `plan_table` (scan -> planner) is the real read+plan path; `current_data_files`
//! enumerates the current-snapshot DataFiles (what the rewrite removes);
//! `compact_table` ties them to the rewrite loop. The rewrite's read -> sort ->
//! write internals are the remaining net-new pieces (`rewrite.rs`); the plan,
//! manifest-enum, and commit boundaries here are wired and correct.

use std::collections::HashMap;

use anyhow::Result;
use futures::TryStreamExt;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataContentType, DataFile, ManifestContentType, ManifestList};
use iceberg::table::Table;
use iceberg::{Catalog, TableIdent};

use crate::config::Config;
use crate::planner::{plan_compaction, Plan};
use crate::rewrite::rewrite_group;

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
/// the data file each one references. When a data file is rewritten, its DV is
/// reabsorbed (removed) in the same RowDelta commit so it doesn't linger
/// orphaned. Equality deletes (no `referenced_data_file`) are skipped.
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
                if let Some(referenced) = df.referenced_data_file() {
                    out.insert(referenced, df);
                }
            }
        }
    }
    Ok(out)
}

/// Compact one table end-to-end: load -> enumerate current data + delete files
/// -> plan -> rewrite each group (swap data files + reabsorb their DVs).
///
/// Each `rewrite_group` commits (RowDelta) and returns the new table snapshot,
/// which threads into the next group's rewrite so later groups operate on the
/// already-committed state. (Groups are disjoint by file, so the once-built
/// maps stay valid for per-group lookups.)
pub async fn compact_table(catalog: &dyn Catalog, ident: &TableIdent, cfg: &Config) -> Result<()> {
    let mut table = catalog.load_table(ident).await?;
    let files = current_data_files(&table).await?;
    let delete_files = current_delete_files(&table).await?;
    let plan = plan_table(&table, cfg).await?;
    for group in plan.groups {
        let removed: Vec<DataFile> = group
            .tasks
            .iter()
            .filter_map(|t| files.get(&t.data_file_path).cloned())
            .collect();
        // Deletion vectors bound to the rewritten data files — reabsorbed in the
        // same commit so no delete file references the removed data afterward.
        let removed_deletes: Vec<DataFile> = group
            .tasks
            .iter()
            .filter_map(|t| delete_files.get(&t.data_file_path).cloned())
            .collect();
        table = rewrite_group(table, catalog, group, removed, removed_deletes).await?;
    }
    Ok(())
}
