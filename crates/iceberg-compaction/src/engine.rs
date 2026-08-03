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
use crate::planner::{Plan, plan_compaction};
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
    let mut all_added: Vec<DataFile> = Vec::new();
    let mut candidate_delete_paths: HashSet<String> = HashSet::new();
    for group in &plan.groups {
        let added = read_sort_write(&table, group, cfg).await?;
        if added.is_empty() {
            // The group produced no live rows. That is LEGITIMATE when every
            // row of every input file is masked by the deletes the scan bound
            // to it — the fully-superseded DELETE-then-INSERT shape, where an
            // old file is 100% covered by a later equality delete. Those files
            // must still be REMOVED (and their deletes thereby become
            // reabsorbable), or the delete pile is pinned forever: the old
            // "never remove without replacement" rule skipped these groups on
            // every pass, retaining both the dead files and every delete file
            // bound to them.
            //
            // Safety gate: only proceed when EVERY input file carried at
            // least one scan-bound delete. A live file with no deletes must
            // produce its rows, so an empty read from such a group signals a
            // read-side anomaly — leave it untouched rather than risk
            // deleting data a broken read failed to surface.
            if !group.tasks.iter().all(|t| !t.deletes.is_empty()) {
                continue;
            }
        }
        all_removed.extend(
            group
                .tasks
                .iter()
                .filter_map(|t| files.get(&t.data_file_path).cloned()),
        );
        // Delete files the SCAN bound to these rewritten data files become
        // removal CANDIDATES. Sourced from `task.deletes` (the scan's
        // per-file binding, the same the read applies), NOT a
        // referenced_data_file lookup — matching iceberg-go
        // (CollectSafeDeletionVectors) and iceberg-java
        // (RewriteFileGroup.danglingDVs).
        for t in &group.tasks {
            for d in &t.deletes {
                candidate_delete_paths.insert(d.file_path.clone());
            }
        }
        all_added.extend(added);
    }

    if all_added.is_empty() && all_removed.is_empty() {
        return Ok(()); // nothing to compact
    }

    // A delete file is removed ONLY when every data file the scan bound it
    // to was rewritten in THIS pass. An EQUALITY delete applies to many
    // files (its partition, lower sequence) — removing it after rewriting
    // only some of them would RESURRECT its deleted rows in the files left
    // behind. For positional DVs (bound to exactly one file) this subset
    // rule reduces to the previous behavior. A retained delete file keeps
    // applying to the remaining old files and ages out naturally once no
    // lower-sequence data remains.
    let rewritten_paths: HashSet<&str> = all_removed.iter().map(|f| f.file_path()).collect();
    let mut all_removed_deletes: Vec<DataFile> = Vec::new();
    for path in &candidate_delete_paths {
        let fully_covered = plan
            .delete_applicability
            .get(path)
            .is_some_and(|applies_to| {
                applies_to
                    .iter()
                    .all(|f| rewritten_paths.contains(f.as_str()))
            });
        if fully_covered {
            if let Some(df) = delete_files.get(path) {
                all_removed_deletes.push(df.clone());
            }
        }
    }

    commit_rewrite(&table, catalog, all_removed, all_removed_deletes, all_added).await?;
    Ok(())
}

/// Outcome of a path-scoped rewrite ([`compact_files`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct CompactFilesOutcome {
    /// Input data files removed (rewritten).
    pub rewritten: usize,
    /// Output data files added.
    pub added: usize,
    /// Delete files (DVs) reabsorbed alongside the rewrite.
    pub reabsorbed_deletes: usize,
}

/// Rewrite EXACTLY the given data files (no candidacy, no size bands),
/// reabsorbing the deletion vectors bound to them — the merge path's inline
/// DV micro-reabsorb entry.
///
/// Differences from [`compact_table`]:
///
/// * The scan is path-scoped (`with_data_file_path_filter`), and the plan is
///   forced to `rewrite_all` + `min_input_files = 1` so every scoped file is
///   rewritten regardless of size.
/// * Delete-file removal is restricted to POSITIONAL deletes whose
///   `referenced_data_file` is one of the rewritten paths. A path-scoped scan
///   cannot see an equality delete's full applicability (files OUTSIDE the
///   scope may still need it), so equality deletes — and positional deletes
///   without a recorded referenced file — are always RETAINED; they keep
///   applying to the files left behind and age out under the table-wide
///   compaction pass.
///
/// The commit anchors `validate_rebase_from` at the planning snapshot (via
/// [`commit_rewrite`]): a concurrent snapshot that added or removed delete
/// files against the inputs aborts NON-retryably — callers treat that as
/// "skipped, re-fires later", never retry in-place.
pub async fn compact_files(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    paths: &HashSet<String>,
    cfg: &Config,
) -> Result<CompactFilesOutcome> {
    if paths.is_empty() {
        return Ok(CompactFilesOutcome::default());
    }
    let table = catalog.load_table(ident).await?;
    let files = current_data_files(&table).await?;
    let delete_files = current_delete_files(&table).await?;

    let tasks: Vec<FileScanTask> = table
        .scan()
        .select_all()
        .with_data_file_path_filter(paths.iter().cloned())
        .build()?
        .plan_files()
        .await?
        .try_collect()
        .await?;
    let mut cfg = cfg.clone();
    cfg.rewrite_all = true;
    cfg.min_input_files = 1;
    let plan = plan_compaction(tasks, &cfg);

    let mut all_removed: Vec<DataFile> = Vec::new();
    let mut all_added: Vec<DataFile> = Vec::new();
    let mut candidate_delete_paths: HashSet<String> = HashSet::new();
    for group in &plan.groups {
        let added = read_sort_write(&table, group, &cfg).await?;
        if added.is_empty() {
            // Same fully-superseded gate as `compact_table`: an empty read is
            // legitimate only when every input carried at least one bound
            // delete (a 100%-dead file); otherwise leave the group untouched.
            if !group.tasks.iter().all(|t| !t.deletes.is_empty()) {
                continue;
            }
        }
        all_removed.extend(
            group
                .tasks
                .iter()
                .filter_map(|t| files.get(&t.data_file_path).cloned()),
        );
        for t in &group.tasks {
            for d in &t.deletes {
                candidate_delete_paths.insert(d.file_path.clone());
            }
        }
        all_added.extend(added);
    }

    if all_added.is_empty() && all_removed.is_empty() {
        return Ok(CompactFilesOutcome::default());
    }

    // Positional-and-referenced only (see doc): the scoped scan is blind to
    // an equality delete's applicability outside `paths`.
    let rewritten_paths: HashSet<&str> = all_removed.iter().map(|f| f.file_path()).collect();
    let mut all_removed_deletes: Vec<DataFile> = Vec::new();
    for path in &candidate_delete_paths {
        if let Some(df) = delete_files.get(path)
            && df.content_type() == DataContentType::PositionDeletes
            && df
                .referenced_data_file()
                .is_some_and(|f| rewritten_paths.contains(f.as_str()))
        {
            all_removed_deletes.push(df.clone());
        }
    }

    let outcome = CompactFilesOutcome {
        rewritten: all_removed.len(),
        added: all_added.len(),
        reabsorbed_deletes: all_removed_deletes.len(),
    };
    commit_rewrite(&table, catalog, all_removed, all_removed_deletes, all_added).await?;
    Ok(outcome)
}

/// Read-only diagnostic of where `removed_deletes` would come from — for pinning
/// the real-table `rdel=NULL` failure WITHOUT writing files or committing. Does NOT
/// call `read_sort_write` (which would write orphan parquet) and never commits; only
/// the same read-side manifest walks + scan plan as `compact_table`. Safe against a
/// production table.
#[derive(Debug)]
pub struct DryRunReport {
    /// current-snapshot DATA files (manifest walk)
    pub data_files: usize,
    /// current-snapshot DELETE files (DVs) found by the manifest walk, keyed by path
    pub delete_files: usize,
    /// a few DV paths from the delete-manifest walk
    pub sample_dv_paths: Vec<String>,
    /// those DVs' `referenced_data_file()` accessor result (Some/None) — tests whether
    /// the old referenced_data_file-keyed map would even have found them
    pub sample_dv_referenced: Vec<Option<String>>,
    /// planned compaction groups
    pub groups: usize,
    /// candidate data files across all groups (= scan tasks the plan would rewrite)
    pub candidate_tasks: usize,
    /// of those tasks, how many carry ≥1 scan-bound delete (`FileScanTask.deletes`)
    pub tasks_with_bound_deletes: usize,
    /// total scan-bound deletes across the plan's tasks (0 ⇒ the SCAN binds no DVs)
    pub total_bound_deletes: usize,
    /// bound deletes that resolve to a DataFile in the delete-manifest map
    /// (= what `compact_table` would actually reabsorb; 0 ⇒ `rdel=NULL` reproduced)
    pub removed_deletes_resolved: usize,
    /// sample of bound-delete paths NOT found in the manifest map (resolution mismatch)
    pub sample_unresolved_bound_paths: Vec<String>,
    /// per-group read result: tasks, delete-bound tasks, and rows that READ
    /// (deletes applied). A delete-bearing group reading 0 rows is the
    /// fully-superseded shape: `compact_table` removes it (and its deletes)
    /// when every input file carried ≥1 bound delete, and skips it otherwise
    /// (empty read from an undeleted file = read-side anomaly).
    pub group_reads: Vec<String>,
}

/// See [`DryRunReport`]. Loads the table, walks current data/delete manifests, plans
/// the compaction, and computes what `removed_deletes` would be — no writes, no commit.
pub async fn dry_run_inspect(
    catalog: &dyn Catalog,
    ident: &TableIdent,
    cfg: &Config,
) -> Result<DryRunReport> {
    let table = catalog.load_table(ident).await?;
    let files = current_data_files(&table).await?;
    let delete_files = current_delete_files(&table).await?;
    let plan = plan_table(&table, cfg).await?;

    let sample_dv_paths: Vec<String> = delete_files.keys().take(5).cloned().collect();
    let sample_dv_referenced: Vec<Option<String>> = sample_dv_paths
        .iter()
        .filter_map(|p| delete_files.get(p))
        .map(|df| df.referenced_data_file().map(|s| s.to_string()))
        .collect();

    let mut candidate_tasks = 0usize;
    let mut tasks_with_bound_deletes = 0usize;
    let mut total_bound_deletes = 0usize;
    let mut removed_deletes_resolved = 0usize;
    let mut seen: HashSet<String> = HashSet::new();
    let mut sample_unresolved: Vec<String> = Vec::new();
    for group in &plan.groups {
        for t in &group.tasks {
            candidate_tasks += 1;
            if !t.deletes.is_empty() {
                tasks_with_bound_deletes += 1;
            }
            for d in &t.deletes {
                total_bound_deletes += 1;
                if seen.insert(d.file_path.clone()) {
                    if delete_files.contains_key(&d.file_path) {
                        removed_deletes_resolved += 1;
                    } else if sample_unresolved.len() < 5 {
                        sample_unresolved.push(d.file_path.clone());
                    }
                }
            }
        }
    }

    // Per-group READ (DVs applied) — does a DV-bearing group read empty (→ skipped)?
    let mut group_reads: Vec<String> = Vec::new();
    for (i, group) in plan.groups.iter().enumerate() {
        let tasks = group.tasks.len();
        let dv_tasks = group.tasks.iter().filter(|t| !t.deletes.is_empty()).count();
        let rows = match crate::rewrite::read_group(&table, group.tasks.clone()).await {
            Ok(stream) => match stream.try_collect::<Vec<_>>().await {
                Ok(batches) => batches
                    .iter()
                    .map(|b| b.num_rows())
                    .sum::<usize>()
                    .to_string(),
                Err(e) => format!("READ_ERR({e})"),
            },
            Err(e) => format!("PLAN_ERR({e})"),
        };
        group_reads.push(format!(
            "group{i}: tasks={tasks} dv_tasks={dv_tasks} rows_read={rows}"
        ));
    }

    Ok(DryRunReport {
        data_files: files.len(),
        delete_files: delete_files.len(),
        sample_dv_paths,
        sample_dv_referenced,
        groups: plan.groups.len(),
        candidate_tasks,
        tasks_with_bound_deletes,
        total_bound_deletes,
        removed_deletes_resolved,
        sample_unresolved_bound_paths: sample_unresolved,
        group_reads,
    })
}
