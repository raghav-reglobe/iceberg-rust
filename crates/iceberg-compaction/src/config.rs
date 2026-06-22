//! Compaction config — mirrors iceberg-go's `table/compaction.Config`, with
//! pulse's 128 MB target (iceberg-go defaults to 512 MB).

use anyhow::{bail, Result};

/// Tunables for a compaction pass.
#[derive(Debug, Clone)]
pub struct Config {
    /// Desired output file size. Pulse: 128 MB (matches the Spark worker +
    /// `compaction_app_settings.target_file_size_mb`).
    pub target_file_size_bytes: u64,
    /// Files smaller than this are compaction candidates (combine). ~75% of target.
    pub min_file_size_bytes: u64,
    /// Files larger than this are NOT rewritten unless they exceed the delete
    /// threshold (rewriting an already-large file is write-amplification). ~180%.
    pub max_file_size_bytes: u64,
    /// Minimum candidate files in a (partition) bin to justify a rewrite.
    /// iceberg-go `DefaultMinInputFiles` = 5. Set to 1 for aggressive DV reabsorb
    /// (our Silver/Gold want every delete-bearing file reabsorbed).
    pub min_input_files: usize,
    /// A data file with at least this many delete files is a candidate
    /// regardless of size (delete-pressure). iceberg-go default: 5.
    pub delete_file_threshold: usize,
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
