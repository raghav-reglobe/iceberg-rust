<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# RFC: sync-worker engine on the doorway (atomic bronze replace, parity reads, variant writes)

Status: **design + prototypes**. Companions:
`crates/iceberg/tests/equality_delete_replace_test.rs` (prototype 1),
`crates/integrations/datafusion/tests/sql_parity_probe_test.rs` (prototype 2).

The JVM-free sync worker currently embeds a separate SQL engine for three
jobs: (1) atomically REPLACING a key-chunk of full-load (`op=R`) rows in an
append-only CDC bronze table, (2) row-hash PARITY reads over silver against
the source database, (3) writing VARIANT columns from ExtendedJSON. This RFC
maps each job onto this fork's doorway and marks what exists, what was
proven by prototype, and what needs new code.

## 1. Bronze atomic replace — FEASIBLE TODAY (prototyped)

### Recommended mechanism: partition-scoped, pk-only equality deletes

One `RowDelta` transaction commits, as **one snapshot**:

- an **equality-delete file** with `equality_ids = [pk]` (top-level column
  only), **partition-scoped to the full-load identity partition**
  (`_is_backfill=true`), whose delete tuples are exactly the replacement
  rows' keys — "delete the prior full-load row for every key I am about to
  rewrite";
- the replacement parquet data files (same partition);
- `validate_from_snapshot(base)` for OCC.

The partition scope replaces the `op='r'` predicate **structurally**: the
full-load rows live alone in the `_is_backfill=true` identity partition
(CDC event rows carry null), so pk-only tuples cannot touch CDC rows with
the same keys, and equality-delete sequence semantics guarantee rows
appended after the replace (later chunks, re-runs) are unaffected.

Proven end-to-end by the prototype against a V3 table:

- delete + append commit as ONE snapshot;
- prior full-load rows of the chunk are gone, CDC rows with the SAME keys
  survive (partition scoping), full-load rows outside the chunk survive;
- rows appended after the replace (higher data sequence number) survive;
- the equality deletes **coexist with positional deletes (V3 DVs)** on the
  same table — the transitional cross-engine state — and both apply in one
  read.

Also proven along the way (write-path contracts): the equality-delete
writer requires its `ParquetWriterBuilder` to be built over the **projected
equality-id schema**, not the table schema; and all write-side schemas/ids
must be derived from the **created table's** schema (`create_table`
reassigns field ids — top-level fields first, then nested — so creation-time
constants diverge).

### Rejected: `op` (nested) equality ids

The first prototype iteration used `equality_ids = [_cdc.op, pk]`. The
WRITER handles nested ids (the projector traverses structs) and `RowDelta`
commits — but the fork's **read path fails** ("field not found"): the
equality-delete schema evolution (`RecordBatchTransformer` over the
equality-id projection) only resolves top-level fields. Nested equality ids
are also a cross-engine hazard (uneven support in external engines).
Partition scoping makes them unnecessary. Do not fix the transformer for
this; keep the constraint "equality ids are top-level columns".

### Fallback (kept in reserve): positional deletes from a scan

Plan the affected partition's files with the pk-range predicate, read
matched `(_file, _pos)` narrowly, write consolidated DVs (union with prior
DVs, remove superseded — exactly the merge write node's proven logic), and
commit the same `RowDelta`. Correct but read-amplifying (a scan per chunk)
and it collides with the same files the merge path touches; use only if
partition-scoped equality deletes hit an external-reader gap.

### OCC / concurrency

- Bronze runs default **snapshot isolation**: the validator admits
  concurrent **pure appends** (the streaming sink) and rejects concurrent
  delete-file commits (our added equality delete is a wildcard in the
  conflict set — fail closed). Sink appends landing between base and commit
  are safe even though the delete's sequence number exceeds theirs: sink
  rows are CDC-partition rows, out of the delete's partition scope.
- On `CatalogCommitConflicts`: **retry-by-rerun** — the already-written
  data + delete files are re-committed in a fresh transaction from the new
  base (no rewrite needed); bounded attempts.

### New fork code needed

| Piece | Effort |
|---|---|
| `bronze_replace` doorway fn (binding): Arrow batches in (pyarrow C-data), partition-aware data writer + equality-delete writer + `RowDelta` + retry loop | ~3–5 days |
| **Compaction × equality-delete gap (required before compacting such tables)**: `compact_table` removes every delete file bound to a rewritten task — an equality-delete file applies to MANY files, so a PARTIAL rewrite would remove it while unrewritten files still need it (row resurrection). Fix: remove an equality delete only when ALL data files it can apply to (its partition, lower sequence) are rewritten in the same pass; otherwise leave it (it ages out once no older-sequence data remains). | ~1–2 days |
| Cross-engine soak item (platform side): external engines' scan paths applying partition-scoped equality deletes over these tables | soak, no code |

## 2. Parity reads via `sql_collect` — FEASIBLE (one tiny UDF + a contract)

### Filter correctness (the boolean-stats hazard class)

`sql_collect` runs through the DataFusion provider, which pushes ALL
filters as **Inexact** — the scan prunes best-effort, and DataFusion
re-applies the predicate row-exactly on the output. Correctness of the bare
truthy shape (`_is_current AND _is_deleted IS NOT TRUE`) therefore never
depends on file statistics; missing/broken boolean bounds on old files cost
pruning, not answers.

### The hash scheme

Source side: `SUM(CONV(SUBSTRING(MD5(CONCAT_WS('|', cols...)), 1, 15), 16, 10))`
per key-slice. Probed facts (locked by `sql_parity_probe_test.rs` so an
engine upgrade that would silently break parity fails there first):

- `md5()` is byte-identical to the source's (lowercase hex, UTF-8 bytes);
- `concat_ws()` **skips NULL args** exactly like the source's `CONCAT_WS`
  (and empty string ≠ NULL) — so NULLs must be canonicalized to a sentinel
  on BOTH sides or absent-vs-null becomes indistinguishable;
- `CONV(hex,16,10)` has **no built-in** — a ~20-line `conv16(prefix)` UDF
  registered at the session seam (like the variant functions) closes the
  gap; `substr(md5(x),1,15)` is already exact.

### The canonicalization contract (byte-equality per column type)

| Type | Source renders | DataFusion renders | Canonicalization (both sides unless noted) |
|---|---|---|---|
| Integer / VARCHAR | `123` / verbatim | same | none |
| DECIMAL(38,6) | full scale `1.500000` | full scale `1.500000` (proven) | `CAST(x AS DECIMAL(38,6))` before rendering |
| FLOAT/DOUBLE | engine-specific shortest form | ryu-style; scientific forms can diverge | **never hash raw floats** — `CAST(x AS DECIMAL(38,6))` on both sides |
| BOOLEAN | `1` / `0` | `true` / `false` (proven) | `CAST(x AS INT)` on the lakehouse side |
| TIMESTAMP | `2026-01-02 03:04:05` | `2026-01-02T03:04:05` (proven) | `to_char(x, '%Y-%m-%d %H:%M:%S')` on the lakehouse side (proven exact) |
| NULL (any) | skipped by CONCAT_WS | skipped by concat_ws (proven) | `COALESCE(rendered, '\N')` before concat, both sides |
| Sub-second timestamps / dates | `.%f` when scale > 0 / `%Y-%m-%d` | same via `to_char` | fix the format per column scale, identically on both sides |

Golden to run platform-side (needs a source container, not reproducible
here): one table exercising every row of the table above, comparing the
slice-hash SQL verbatim; the fork-side building blocks are already pinned.

## 3. VARIANT writes — LOW RISK (existing machinery)

- The write path does not go through SQL: the `bronze_replace` doorway
  receives Arrow batches with ExtendedJSON **strings** and converts them
  with the same arrow-rs `json_to_variant` kernel the registered UDF wraps
  — array-level, no SQL context needed.
- Writing VARIANT columns through `ParquetWriterBuilder` is already proven
  by the merge and compaction suites (canonical output, correct parquet
  VARIANT annotation, null-faithfulness, shred-preserving rewrites).
  Canonical output is sufficient for the replace path — compaction
  re-shreds later; external-engine readability of fork-written variant is
  already exercised in production by the compaction rewrites.
- Table creation with VARIANT columns goes through the existing catalog
  create path (unchanged).

## Verdict summary

| Piece | Verdict | New code |
|---|---|---|
| Bronze atomic replace | **feasible today** — partition-scoped equality deletes + `RowDelta`, prototype green | doorway fn (~3–5 d) + compaction eq-delete guard (~1–2 d) |
| Parity reads | **feasible** — probes green | `conv16` UDF (~hours) + platform-side golden |
| Variant writes | **low risk** — kernel + writer proven | none beyond the doorway fn |

Recommended sequence: land the compaction guard first (it is a latent
correctness bug for ANY equality-delete-bearing table this engine
compacts), then the `bronze_replace` doorway + `conv16`, then the
platform-side parity golden and cross-engine soak.
