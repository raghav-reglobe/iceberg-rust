//! Compaction config — mirrors iceberg-go's `table/compaction.Config`, with
//! a 128 MB default target (iceberg-go defaults to 512 MB).

use anyhow::{Result, bail};

/// Tunables for a compaction pass.
#[derive(Debug, Clone)]
pub struct Config {
    /// Desired output file size. Default 128 MB.
    pub target_file_size_bytes: u64,
    /// Files smaller than this are compaction candidates (combine). ~75% of target.
    pub min_file_size_bytes: u64,
    /// Files larger than this are NOT rewritten unless they exceed the delete
    /// threshold (rewriting an already-large file is write-amplification). ~180%.
    pub max_file_size_bytes: u64,
    /// Minimum candidate files in a (partition) bin to justify a rewrite.
    /// iceberg-go `DefaultMinInputFiles` = 5. Set to 1 for aggressive DV reabsorb
    /// (when every delete-bearing file should be rewritten).
    pub min_input_files: usize,
    /// A data file with at least this many delete files is a candidate
    /// regardless of size (delete-pressure). iceberg-go default: 5.
    pub delete_file_threshold: usize,
    /// Upper bound (in estimated ARROW bytes) on how much of a group is
    /// buffered and sorted at once. A group whose decoded data exceeds this
    /// is rewritten as several independently-sorted runs instead of one —
    /// bounded memory on wide-row (multi-KB TEXT) tables whose 128 MB
    /// parquet groups decode to GBs. Groups under the budget (the common
    /// case) still produce one fully-sorted run. Default 1 GiB.
    pub sort_chunk_bytes: usize,
    /// Upper bound (in estimated ARROW bytes) on each record batch handed
    /// to the writer. Bounds the per-slice working set AND keeps every
    /// output array far below arrow's i32 string-offset range (~2 GiB
    /// cumulative string bytes), which one whole-group concatenation
    /// overflowed on wide-TEXT tables. Default 32 MiB.
    pub write_batch_bytes: usize,
    /// Preserve the input files' variant SHREDDING on rewrite. When true,
    /// each variant column's shredding schema is derived from the first input
    /// file's parquet footer and the output is re-shredded to match (columns
    /// canonical in the input stay canonical). When false (default) the
    /// rewrite emits the canonical `{metadata, value}` layout — semantically
    /// identical, but query engines lose `typed_value` pruning on the
    /// rewritten files.
    pub shred_variants: bool,
    /// Sort key for the rewrite. `None` (the default) rewrites rows in read
    /// order; a key the table does not have also passes through unsorted. The
    /// caller names the key: table sort-order metadata is NOT read yet.
    pub sort_column: Option<String>,
    /// Plan EVERY live data file as a rewrite candidate, bypassing the
    /// size/delete candidacy policy (Spark `rewrite-all` parity). Grouping
    /// and bin-packing still apply.
    pub rewrite_all: bool,
    /// Cooperative deadline. Checked between groups, between read batches
    /// and between sort-chunk flushes; when exceeded the pass aborts with a
    /// "compaction deadline exceeded" error BEFORE the commit — the commit
    /// itself, once entered, always runs to completion (cancelling a REST
    /// commit in flight leaves the outcome unknown). `None` (default) =
    /// unbounded.
    pub deadline: Option<std::time::Instant>,
    /// Cooperative BUDGET — the soft twin of `deadline`. Checked only BETWEEN
    /// groups: once a group has run, the pass refuses to START another one
    /// that, at the slowest group pace this pass has measured, would cross the
    /// budget — and COMMITS the groups it finished (one `RewriteFiles`, the
    /// same partial-rewrite delete rule as always: a delete file is removed
    /// only when every data file it applies to was rewritten). The first
    /// group always runs, so every bounded pass makes progress; the groups
    /// are ordered by the read cost they remove, so the progress is the most
    /// valuable part of the plan. A table too large for one pass converges
    /// over several instead of never committing. `None` (default) = every
    /// planned group.
    ///
    /// The budget bounds when groups START, nothing else: the group in
    /// flight at the stop and the commit both run past it. A caller that
    /// also has a hard limit (`deadline`, or its own kill) must leave a
    /// reserve between the two for one group plus the commit — a budget
    /// equal to the deadline loses the whole pass to the deadline.
    pub budget: Option<std::time::Instant>,
    /// How many groups are rewritten at once (iceberg-java
    /// `max-concurrent-file-group-rewrites`). A group's work is CPU-bound and
    /// single-threaded, so one group leaves most of a multi-core pod idle;
    /// each group in flight also holds its own working set — up to
    /// `sort_chunk_bytes` of decoded rows, a write slice and the writer's
    /// column buffers — so the right number is bounded by MEMORY, not cores,
    /// and is the caller's to derive from what it has measured. Groups still
    /// commit together, in the one `RewriteFiles`. Default 1 (one at a time);
    /// 0 is read as 1.
    pub max_concurrent_groups: usize,
}

/// The instant `secs` after `now`, for turning a caller's seconds into a
/// [`Config::budget`] or [`Config::deadline`]. A negative value is already
/// spent (`now`); a value too large to represent, including infinity, is no
/// bound at all (`None`) rather than a panic. NaN is a caller bug.
pub fn instant_after(now: std::time::Instant, secs: f64) -> Result<Option<std::time::Instant>> {
    if secs.is_nan() {
        bail!("a time bound in seconds must not be NaN");
    }
    Ok(std::time::Duration::try_from_secs_f64(secs.max(0.0))
        .ok()
        .and_then(|d| now.checked_add(d)))
}

impl Default for Config {
    fn default() -> Self {
        let target: u64 = 128 * 1024 * 1024; // 128 MB
        Self {
            target_file_size_bytes: target,
            min_file_size_bytes: target * 3 / 4, // 75%
            max_file_size_bytes: target * 9 / 5, // 180%
            min_input_files: 5,
            delete_file_threshold: 5,
            sort_chunk_bytes: 1024 * 1024 * 1024, // 1 GiB
            write_batch_bytes: 32 * 1024 * 1024,  // 32 MiB
            shred_variants: false,
            sort_column: None,
            rewrite_all: false,
            deadline: None,
            budget: None,
            max_concurrent_groups: 1,
        }
    }
}

impl Config {
    /// Validate the size bounds (mirrors iceberg-go `Config.Validate`):
    /// `0 < min < max` and `target ∈ [min, max]`.
    pub fn validate(&self) -> Result<()> {
        if self.max_file_size_bytes == 0 {
            bail!("max_file_size_bytes must be positive");
        }
        if self.min_file_size_bytes >= self.max_file_size_bytes {
            bail!(
                "min_file_size_bytes ({}) must be < max_file_size_bytes ({})",
                self.min_file_size_bytes,
                self.max_file_size_bytes
            );
        }
        if self.target_file_size_bytes < self.min_file_size_bytes
            || self.target_file_size_bytes > self.max_file_size_bytes
        {
            bail!(
                "target_file_size_bytes ({}) must be within [{}, {}]",
                self.target_file_size_bytes,
                self.min_file_size_bytes,
                self.max_file_size_bytes
            );
        }
        if self.sort_chunk_bytes == 0 || self.write_batch_bytes == 0 {
            bail!("sort_chunk_bytes and write_batch_bytes must be positive");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, instant_after};

    #[test]
    fn instant_after_never_panics_on_extreme_seconds() {
        let now = std::time::Instant::now();
        let secs = |s: f64| instant_after(now, s).unwrap();
        assert_eq!(secs(0.0), Some(now));
        assert_eq!(secs(-5.0), Some(now), "a negative bound is already spent");
        assert_eq!(
            secs(1.5),
            Some(now + std::time::Duration::from_millis(1500))
        );
        // Unrepresentable = unbounded, never a panic.
        assert_eq!(secs(f64::INFINITY), None);
        assert_eq!(secs(1e300), None);
        assert_eq!(secs(u64::MAX as f64), None);
        assert!(instant_after(now, f64::NAN).is_err());
    }

    #[test]
    fn default_matches_iceberg_go_ratios() {
        let c = Config::default();
        assert_eq!(c.target_file_size_bytes, 128 * 1024 * 1024);
        assert_eq!(c.min_file_size_bytes, c.target_file_size_bytes * 3 / 4); // 75%
        assert_eq!(c.max_file_size_bytes, c.target_file_size_bytes * 9 / 5); // 180%
        assert_eq!(c.min_input_files, 5);
        assert_eq!(c.delete_file_threshold, 5);
        c.validate().unwrap();
    }

    #[test]
    fn validate_rejects_bad_bounds() {
        let mut c = Config::default();
        c.min_file_size_bytes = c.max_file_size_bytes; // min >= max
        assert!(c.validate().is_err());

        let mut c = Config::default();
        c.target_file_size_bytes = c.max_file_size_bytes + 1; // target > max
        assert!(c.validate().is_err());
    }
}
