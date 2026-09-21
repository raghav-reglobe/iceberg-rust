//! Compaction planner — mirrors iceberg-go's `Config.PlanCompaction`: group scan
//! tasks by partition, classify each as candidate (undersized or delete-pressure)
//! or skipped (optimal / oversized-without-deletes), bin-pack candidates to
//! `target_file_size_bytes`, and drop bins below `min_input_files` — and, beyond
//! iceberg-go, a lone file whose rewrite would remove nothing (iceberg-java's
//! `group.size() > 1` rule; see `is_noop_bin`).

use std::collections::{HashMap, HashSet};

use iceberg::scan::FileScanTask;

use crate::config::Config;

/// A bin-packed, single-partition set of data files to rewrite into ~one
/// target-sized output (per-partition output via the rewrite's FanoutWriter).
#[derive(Default)]
pub struct Group {
    /// The partition all files in this group share (groups never span partitions).
    pub partition_key: String,
    pub tasks: Vec<FileScanTask>,
    pub total_size_bytes: u64,
    pub delete_file_count: usize,
}

/// The output of planning: the groups to rewrite + what was left alone.
#[derive(Default)]
pub struct Plan {
    pub groups: Vec<Group>,
    /// Files left as-is: optimal/oversized, or in a sub-`min_input_files` bin.
    pub skipped_files: usize,
    pub total_input_files: usize,
    pub total_input_bytes: u64,
    pub est_output_files: usize,
    /// Delete-file APPLICABILITY over the WHOLE scan (candidates and skipped
    /// files alike): delete file path -> every data file path the scan bound
    /// it to. A delete file may only be removed by a rewrite when EVERY data
    /// file it applies to is rewritten in the same pass — an EQUALITY delete
    /// binds to many files (its partition, lower sequence), so removing it
    /// after rewriting only some of them would resurrect deleted rows in the
    /// rest. (A positional DV binds to exactly one file, so the same subset
    /// rule degenerates to the old behavior for DVs.)
    pub delete_applicability: HashMap<String, HashSet<String>>,
}

impl Group {
    /// Read cost this group's rewrite REMOVES, in objects a scan opens: the
    /// delete files bound to its inputs (reabsorbed) plus the data files the
    /// bin-pack folds away. A lone delete-free file scores 0 — rewriting it
    /// changes nothing a reader sees, so only a `rewrite_all` plan contains
    /// such a group (`is_noop_bin`).
    pub fn objects_removed(&self, target_file_size_bytes: u64) -> usize {
        objects_removed(
            self.tasks.len(),
            self.delete_file_count,
            self.total_size_bytes,
            target_file_size_bytes,
        )
    }
}

/// Pure scoring behind [`Group::objects_removed`] (unit-testable without
/// constructing `FileScanTask`s).
fn objects_removed(files: usize, deletes: usize, total_size_bytes: u64, target: u64) -> usize {
    let est_outputs = total_size_bytes.div_ceil(target.max(1)).max(1) as usize;
    deletes + files.saturating_sub(est_outputs)
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Group indices in EXECUTION order: most read cost removed first, plan
    /// order among equals (stable). An unbounded pass rewrites the same set
    /// either way; a budget-bounded pass spends its budget on the groups
    /// that matter and leaves the no-op tail for last.
    pub fn execution_order(&self, target_file_size_bytes: u64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.groups.len()).collect();
        order.sort_by_key(|&i| {
            std::cmp::Reverse(self.groups[i].objects_removed(target_file_size_bytes))
        });
        order
    }

    /// Total input data files across all groups (what the rewrite will replace).
    pub fn input_file_count(&self) -> usize {
        self.groups.iter().map(|g| g.tasks.len()).sum()
    }
}

/// A file is a candidate if it has delete-pressure (>= threshold delete files —
/// reabsorbing them is worthwhile) OR is undersized (combine). Oversized files
/// with few deletes, and optimally-sized files, are skipped. Pure policy —
/// mirrors iceberg-go `isCandidate`.
fn is_candidate(size: u64, delete_count: usize, cfg: &Config) -> bool {
    if delete_count >= cfg.delete_file_threshold {
        return true; // delete-pressure: always compact (reabsorb), any size
    }
    if size > cfg.max_file_size_bytes {
        return false; // oversized, few deletes: skip (rewrite = write-amplification)
    }
    if size >= cfg.min_file_size_bytes {
        return false; // right-sized: skip (optimal)
    }
    true // undersized: candidate
}

/// A bin whose rewrite removes nothing a reader opens: ONE file with no bound
/// delete. It folds into nothing and reabsorbs nothing, so the output is the
/// input again — and because it is still undersized afterwards, a
/// `min_input_files = 1` plan would re-select and re-write it on EVERY pass
/// (one such file per partition: the bin-pack's remainder). iceberg-java
/// never plans it either (`SizeBasedFileRewritePlanner.enoughInputFiles`
/// requires `group.size() > 1`); iceberg-go only avoids it through its
/// `MinInputFiles` default of 5. A lone file WITH a bound delete stays
/// planned — reabsorbing the delete is the point of `delete_file_threshold =
/// 1` — and `rewrite_all` means what it says.
fn is_noop_bin(files: usize, bound_deletes: usize, cfg: &Config) -> bool {
    files == 1 && bound_deletes == 0 && !cfg.rewrite_all
}

/// Call-site policy: `rewrite_all` bypasses candidacy (Spark `rewrite-all`
/// parity); otherwise the iceberg-go candidate rules apply.
fn should_rewrite(size: u64, delete_count: usize, cfg: &Config) -> bool {
    cfg.rewrite_all || is_candidate(size, delete_count, cfg)
}

/// Greedy bin-pack `items` into bins whose summed `weight` stays ~`target`.
/// Generic so the packing is unit-testable without constructing `FileScanTask`s.
fn bin_pack<T>(items: Vec<T>, target: u64, weight: impl Fn(&T) -> u64) -> Vec<Vec<T>> {
    let mut bins: Vec<Vec<T>> = Vec::new();
    let mut current: Vec<T> = Vec::new();
    let mut current_size = 0u64;
    for item in items {
        let w = weight(&item);
        if !current.is_empty() && current_size + w > target {
            bins.push(std::mem::take(&mut current));
            current_size = 0;
        }
        current_size += w;
        current.push(item);
    }
    if !current.is_empty() {
        bins.push(current);
    }
    bins
}

/// Partition identity for grouping. `FileScanTask` exposes `partition: Option<Struct>`;
/// its debug form is a stable per-partition key (same partition value → same key).
fn partition_key(task: &FileScanTask) -> String {
    format!("{:?}", task.partition)
}

/// Plan a compaction: group candidates by partition (preserving first-seen
/// order), bin-pack each partition to target, and drop bins below
/// `min_input_files`. Mirrors iceberg-go `Config.PlanCompaction`.
pub fn plan_compaction(tasks: Vec<FileScanTask>, cfg: &Config) -> Plan {
    let mut delete_applicability: HashMap<String, HashSet<String>> = HashMap::new();
    for t in &tasks {
        for d in &t.deletes {
            delete_applicability
                .entry(d.file_path.clone())
                .or_default()
                .insert(t.data_file_path.clone());
        }
    }
    let mut plan = Plan {
        total_input_files: tasks.len(),
        total_input_bytes: tasks.iter().map(|t| t.file_size_in_bytes).sum(),
        delete_applicability,
        ..Default::default()
    };

    // Group candidates by partition (preserve order); non-candidates are skipped.
    let mut buckets: Vec<(String, Vec<FileScanTask>)> = Vec::new();
    for t in tasks {
        if !should_rewrite(t.file_size_in_bytes, t.deletes.len(), cfg) {
            plan.skipped_files += 1;
            continue;
        }
        let key = partition_key(&t);
        match buckets.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(t),
            None => buckets.push((key, vec![t])),
        }
    }

    let target = cfg.target_file_size_bytes.max(1);
    for (key, candidates) in buckets {
        if candidates.len() < cfg.min_input_files {
            plan.skipped_files += candidates.len(); // too few to justify a rewrite
            continue;
        }
        for bin in bin_pack(candidates, target, |t| t.file_size_in_bytes) {
            if bin.len() < cfg.min_input_files {
                plan.skipped_files += bin.len();
                continue;
            }
            let total_size_bytes: u64 = bin.iter().map(|t| t.file_size_in_bytes).sum();
            let delete_file_count: usize = bin.iter().map(|t| t.deletes.len()).sum();
            if is_noop_bin(bin.len(), delete_file_count, cfg) {
                plan.skipped_files += 1; // nothing to fold, nothing to reabsorb
                continue;
            }
            plan.est_output_files += total_size_bytes.div_ceil(target).max(1) as usize;
            plan.groups.push(Group {
                partition_key: key.clone(),
                tasks: bin,
                total_size_bytes,
                delete_file_count,
            });
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::{
        Group, Plan, bin_pack, is_candidate, is_noop_bin, objects_removed, should_rewrite,
    };
    use crate::config::Config;

    // --- a lone file whose rewrite removes nothing is not planned ---

    #[test]
    fn lone_delete_free_bin_is_a_noop_unless_rewrite_all() {
        let aggressive = Config {
            min_input_files: 1,
            delete_file_threshold: 1,
            ..Config::default()
        };
        assert!(is_noop_bin(1, 0, &aggressive), "one file, nothing bound");
        assert!(!is_noop_bin(1, 1, &aggressive), "its delete is reabsorbed");
        assert!(!is_noop_bin(2, 0, &aggressive), "two files fold into one");
        // Below the delete THRESHOLD is still a bound delete: the rule is
        // about what the rewrite removes, not about candidacy.
        let high_threshold = Config {
            delete_file_threshold: 1000,
            ..aggressive.clone()
        };
        assert!(!is_noop_bin(1, 1, &high_threshold));
        let all = Config {
            rewrite_all: true,
            ..aggressive
        };
        assert!(!is_noop_bin(1, 0, &all), "rewrite_all means every file");
    }

    // --- execution order: most read cost removed first ---

    #[test]
    fn objects_removed_counts_reabsorbed_deletes_and_folded_files() {
        let target = 128 * 1024 * 1024;
        // 17 small files + 2 DVs folding into one output: 2 + 16
        assert_eq!(objects_removed(17, 2, 100 * 1024 * 1024, target), 18);
        // a lone delete-free undersized file: the rewrite is a no-op for readers
        assert_eq!(objects_removed(1, 0, 10 * 1024 * 1024, target), 0);
        // a lone delete-bearing file: its DV is reabsorbed
        assert_eq!(objects_removed(1, 1, 10 * 1024 * 1024, target), 1);
        // an oversized split never scores below zero
        assert_eq!(objects_removed(1, 0, 10 * target, target), 0);
    }

    #[test]
    fn execution_order_is_value_first_and_stable() {
        let g = |deletes: usize| Group {
            delete_file_count: deletes,
            ..Default::default()
        };
        let plan = Plan {
            groups: vec![g(0), g(3), g(0), g(7), g(3)],
            ..Default::default()
        };
        assert_eq!(plan.execution_order(128 * 1024 * 1024), vec![3, 1, 4, 0, 2]);
    }

    // --- is_candidate policy (mirrors iceberg-go isCandidate cases) ---

    #[test]
    fn undersized_is_candidate() {
        let c = Config::default();
        assert!(is_candidate(c.min_file_size_bytes - 1, 0, &c));
    }

    #[test]
    fn optimal_is_skipped() {
        let c = Config::default();
        assert!(!is_candidate(c.target_file_size_bytes, 0, &c));
        assert!(!is_candidate(c.min_file_size_bytes, 0, &c)); // exactly min = optimal
    }

    #[test]
    fn oversized_without_deletes_is_skipped() {
        let c = Config::default();
        assert!(!is_candidate(
            c.max_file_size_bytes + 1,
            c.delete_file_threshold - 1,
            &c
        ));
    }

    #[test]
    fn oversized_with_deletes_is_candidate() {
        let c = Config::default();
        assert!(is_candidate(
            c.max_file_size_bytes + 1,
            c.delete_file_threshold,
            &c
        ));
    }

    #[test]
    fn delete_pressure_overrides_any_size() {
        let c = Config::default();
        assert!(is_candidate(
            c.target_file_size_bytes,
            c.delete_file_threshold,
            &c
        ));
    }

    #[test]
    fn rewrite_all_bypasses_candidacy() {
        let mut c = Config::default();
        c.rewrite_all = true;
        // optimal and oversized-without-deletes are normally skipped
        assert!(should_rewrite(c.target_file_size_bytes, 0, &c));
        assert!(should_rewrite(c.max_file_size_bytes + 1, 0, &c));
        c.rewrite_all = false;
        assert!(!should_rewrite(c.target_file_size_bytes, 0, &c));
    }

    // --- bin-packing ---

    #[test]
    fn bin_pack_splits_at_target() {
        // 30+30+30 = 90 <= 100; +30 = 120 > 100 -> new bin.
        let bins = bin_pack(vec![30u64, 30, 30, 30], 100, |w| *w);
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0], vec![30, 30, 30]);
        assert_eq!(bins[1], vec![30]);
    }

    #[test]
    fn bin_pack_oversized_item_stands_alone() {
        // each 60 alone (2*60 > 100).
        let bins = bin_pack(vec![60u64, 60, 60], 100, |w| *w);
        assert_eq!(bins.len(), 3);
    }

    #[test]
    fn bin_pack_empty() {
        let bins = bin_pack(Vec::<u64>::new(), 100, |w| *w);
        assert!(bins.is_empty());
    }
}
