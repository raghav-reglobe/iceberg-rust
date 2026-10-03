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

## parquet-variant-json 59.0.0

`json_to_variant` typed every non-integer JSON number as an IEEE double — the
released encoder parses into a `serde_json::Value`, whose `Number` has
already lost the digits the number was written with, so it could not know a
decimal's scale (`12.30` became the double 12.3; a 30-digit decimal became a
17-digit double; an integer past int64 became a double). The crate's own
decimal expectations were present but `#[ignore]`d for exactly that reason.

The copy parses through `serde_json::value::RawValue` (feature `raw_value`,
additive — it adds a type and changes nothing about `Number`), so every value
is seen as its text, and types numbers the way Spark's `VariantBuilder`
does: an integer that fits int64 → the narrowest int8/16/32/64; plain
decimal notation (sign, digits, one `.`; no exponent) with at most 38
significant digits and a scale of at most 38 → decimal4/8/16 by the
narrowest width whose precision and scale limits both hold, keeping the
text's scale; anything else → double. `-0.0` becomes decimal4(0, 1). Objects
keep a repeated key's last value and sort keys, as before. Each nesting
level re-validates its own subtree, so parsing costs the document size times
its depth. The fifteen upstream decimal tests are enabled;
`test_json_to_variant_double_precision` (29 digits at scale 29, within the
decimal16 limits) now expects a decimal16 and is named accordingly. Files
touched: `src/from_json.rs`, `Cargo.toml` (the feature).

## parquet-variant-compute 59.0.0 — float leaves (second change)

`PrimitiveVariantToArrowRowBuilder::require_exact_numeric` also covers
Float32/Float64 targets: an integer past the float's mantissa, a decimal the
float does not reproduce (its shortest round-trip rendering differs
numerically), or a double a Float32 cannot hold, is reported as not
converted — shredding then keeps it a variant `value`. Plain casts
(`variant_get`) keep rounding. Trait `type_conversion::ExactFloatFromVariant`,
`VariantToFloatArrowRowBuilder` in `src/variant_to_arrow.rs`; tests
`shred_variant::tests::{inexact_numerics_are_not_shredded_into_float_leaves,
plain_float_casts_still_round}`.
