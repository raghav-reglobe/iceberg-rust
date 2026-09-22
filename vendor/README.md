# vendor/

Crates carried as a local copy because the fix they need is not in a
release yet. Each directory is the crates.io tarball of the named version
with its registry metadata files removed, plus the patch described here;
`[patch.crates-io]` in the workspace `Cargo.toml` points the whole
dependency graph at the copy. Its manifest keeps the published, registry
resolved dependencies, so every other crate in the graph stays the
registry build — nothing else is duplicated.

## parquet-variant-compute 59.0.0

Shredding must be lossless. The kernel converted a variant decimal into a
`typed_value` decimal of a different scale through the casting path, which
rounds when reducing scale (and rounds doubles), so a `typed_value` derived
at scale 1 silently turned every later `0.68` into `0.7`. The patch adds an
exact conversion (`type_conversion::variant_to_unscaled_decimal_exact`), an
`exact` switch on the decimal row builder
(`PrimitiveVariantToArrowRowBuilder::require_exact_numeric`), and shredding
turns it on: a value that cannot be represented exactly in `typed_value`
stays a variant `value`. Plain casts (`variant_get`) keep rounding
semantics. Tests: `shred_variant::tests::{inexact_decimals_are_not_shredded,
plain_casts_still_round}`. Files touched: `src/type_conversion.rs`,
`src/variant_to_arrow.rs`, `src/shred_variant.rs`.
