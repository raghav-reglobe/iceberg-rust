//! Compaction planner — mirrors iceberg-go's `Config.PlanCompaction`: group scan
//! tasks by partition, classify each as candidate (undersized or delete-pressure)
//! or skipped (optimal / oversized-without-deletes), bin-pack candidates to
//! `target_file_size_bytes`, and drop bins below `min_input_files`.

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
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
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
    let mut plan = Plan {
        total_input_files: tasks.len(),
        total_input_bytes: tasks.iter().map(|t| t.file_size_in_bytes).sum(),
        ..Default::default()
    };

    // Group candidates by partition (preserve order); non-candidates are skipped.
    let mut buckets: Vec<(String, Vec<FileScanTask>)> = Vec::new();
    for t in tasks {
        if !is_candidate(t.file_size_in_bytes, t.deletes.len(), cfg) {
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
    use super::{bin_pack, is_candidate};
    use crate::config::Config;

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
        assert!(!is_candidate(c.max_file_size_bytes + 1, c.delete_file_threshold - 1, &c));
    }

    #[test]
    fn oversized_with_deletes_is_candidate() {
        let c = Config::default();
        assert!(is_candidate(c.max_file_size_bytes + 1, c.delete_file_threshold, &c));
    }

    #[test]
    fn delete_pressure_overrides_any_size() {
        let c = Config::default();
        assert!(is_candidate(c.target_file_size_bytes, c.delete_file_threshold, &c));
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
