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

## parquet 59.0.0

A shredded VARIANT decimal must land in the physical type the shredding spec
prescribes: INT32 for every precision <= 9, INT64 up to 18, otherwise
FIXED_LEN_BYTE_ARRAY. The released arrow-to-parquet schema mapping
(`src/arrow/schema/mod.rs`, `arrow_to_parquet_type`) guarded INT32 with
`precision > 1 && precision <= 9`, so a `DECIMAL(1, s)` typed_value - every
`0.9`-shaped grade in the mongo quotes - was written as INT64. Readers that
enforce the spec (Doris' native parquet reader, file scanner v2) reject such a
file as corruption: `Parquet Variant DECIMAL precision 1 does not match
physical type 2 (INT64)`. The patch drops the `> 1` guard (one line); the
logical type keeps the declared precision and scale, so a spec-tolerant reader
sees the same values. Test: `arrow::schema::tests::test_decimal_precision_one_is_int32`
(run from this directory with `--features arrow`). Files touched:
`src/arrow/schema/mod.rs`. Existing files written before the patch are NOT
rewritten by it - the fix is for what the writers produce from here on.

