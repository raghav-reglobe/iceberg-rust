//! Compaction config — mirrors iceberg-go's `table/compaction.Config`, with
//! pulse's 128 MB target (iceberg-go defaults to 512 MB).

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
    use super::Config;

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
