//! Sort compaction inputs by `_valid_from` — a common universal sort key that
//! gives query engines file-skipping on time filters. Tables without the column
//! are passed through unsorted.
//!
//! Wide-row safety: the sort NEVER concatenates the input into one combined
//! batch. A whole-group concat of multi-KB TEXT rows both (a) overflows
//! arrow's i32 string-array offsets once cumulative string bytes cross ~2 GiB
//! (`Offset overflow error`), and (b) triples the group's decoded footprint
//! (input + concat + take copies) — the giant-TEXT compaction OOM. Instead,
//! the sorted order is computed over just the KEY column and the output is
//! emitted as BOUNDED batches via per-slice `interleave` — peak memory is the
//! (chunk's) input plus one output slice, and no output array can approach
//! the i32 offset range.

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_ord::sort::sort_to_indices;
use arrow_select::concat::concat;
use arrow_select::interleave::interleave_record_batch;

/// One chunk's rows in sorted order, emitted as bounded batches. Pull-based:
/// each [`SortedChunk::next_batch`] materializes ONE output slice (via
/// `interleave` across the retained input batches) of at most ~
/// `max_batch_bytes` estimated arrow bytes.
pub(crate) struct SortedChunk {
    batches: Vec<RecordBatch>,
    /// (batch index, row index) in output order. Empty when the input is
    /// passed through unsorted (no `_valid_from`).
    order: Vec<(usize, usize)>,
    /// Exclusive end offsets into `order`, one per output slice.
    cuts: Vec<usize>,
    next_slice: usize,
    /// Passthrough mode (no sort key): emit the input batches unchanged.
    passthrough: bool,
    next_passthrough: usize,
}

impl SortedChunk {
    pub(crate) fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.passthrough {
            let out = self.batches.get(self.next_passthrough).cloned();
            self.next_passthrough += 1;
            return Ok(out);
        }
        if self.next_slice >= self.cuts.len() {
            return Ok(None);
        }
        let start = if self.next_slice == 0 {
            0
        } else {
            self.cuts[self.next_slice - 1]
        };
        let end = self.cuts[self.next_slice];
        self.next_slice += 1;
        let refs: Vec<&RecordBatch> = self.batches.iter().collect();
        let out = interleave_record_batch(&refs, &self.order[start..end])?;
        Ok(Some(out))
    }
}

/// Sort `batches` (one buffered chunk) by `_valid_from` ascending and return
/// a puller of bounded output batches (~`max_batch_bytes` each). If the sort
/// column is absent the input batches pass through unchanged (they are
/// already reader-bounded).
pub(crate) fn sort_chunk(batches: Vec<RecordBatch>, max_batch_bytes: usize) -> Result<SortedChunk> {
    let non_empty: Vec<RecordBatch> = batches.into_iter().filter(|b| b.num_rows() > 0).collect();
    let passthrough = |batches: Vec<RecordBatch>| SortedChunk {
        batches,
        order: Vec::new(),
        cuts: Vec::new(),
        next_slice: 0,
        passthrough: true,
        next_passthrough: 0,
    };
    if non_empty.is_empty() {
        return Ok(passthrough(non_empty));
    }
    let schema = non_empty[0].schema();
    let Ok(key_idx) = schema.index_of("_valid_from") else {
        return Ok(passthrough(non_empty));
    };

    // Global order over the chunk from JUST the key column (primitive-cheap;
    // never the wide payload).
    let key_arrays: Vec<&dyn arrow_array::Array> = non_empty
        .iter()
        .map(|b| b.column(key_idx).as_ref())
        .collect();
    let keys = concat(&key_arrays)?;
    let indices = sort_to_indices(keys.as_ref(), None, None)?;

    // Flat index -> (batch, row) + a per-row byte estimate per source batch.
    let mut batch_starts = Vec::with_capacity(non_empty.len());
    let mut row_bytes = Vec::with_capacity(non_empty.len());
    let mut total = 0usize;
    for b in &non_empty {
        batch_starts.push(total);
        total += b.num_rows();
        row_bytes.push((b.get_array_memory_size() / b.num_rows().max(1)).max(1));
    }
    let locate = |flat: usize| -> (usize, usize) {
        let bi = match batch_starts.binary_search(&flat) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (bi, flat - batch_starts[bi])
    };

    let mut order = Vec::with_capacity(total);
    let mut cuts = Vec::new();
    let mut slice_bytes = 0usize;
    for flat in indices.values() {
        let (bi, ri) = locate(*flat as usize);
        // Cut BEFORE adding when the running slice has reached the budget —
        // every slice holds at least one row.
        if slice_bytes >= max_batch_bytes && order.len() > cuts.last().copied().unwrap_or(0) {
            cuts.push(order.len());
            slice_bytes = 0;
        }
        order.push((bi, ri));
        slice_bytes += row_bytes[bi];
    }
    cuts.push(order.len());

    Ok(SortedChunk {
        batches: non_empty,
        order,
        cuts,
        next_slice: 0,
        passthrough: false,
        next_passthrough: 0,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::sort_chunk;

    fn drain(mut chunk: super::SortedChunk) -> Vec<RecordBatch> {
        let mut out = Vec::new();
        while let Some(b) = chunk.next_batch().unwrap() {
            out.push(b);
        }
        out
    }

    #[test]
    fn sorts_rows_by_valid_from_across_batches() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_valid_from", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let b1 = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int64Array::from(vec![3, 1])),
            Arc::new(StringArray::from(vec!["c", "a"])),
        ])
        .unwrap();
        let b2 = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int64Array::from(vec![2])),
            Arc::new(StringArray::from(vec!["b"])),
        ])
        .unwrap();

        let out = drain(sort_chunk(vec![b1, b2], usize::MAX).unwrap());
        assert_eq!(out.len(), 1);
        let vf = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let pl = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(vf.values(), &[1, 2, 3]); // sorted ascending across batches
        assert_eq!(pl.value(0), "a"); // payload moved with its key
        assert_eq!(pl.value(2), "c");
    }

    #[test]
    fn coalesces_when_no_valid_from() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![2, 1]))]).unwrap();
        let out = drain(sort_chunk(vec![b], usize::MAX).unwrap());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num_rows(), 2); // unchanged, just passed through
    }

    #[test]
    fn output_batches_are_byte_bounded_and_globally_sorted() {
        // ~1 KB rows; a 4 KB budget must split the sorted output into
        // several small batches whose concatenation is globally sorted.
        let schema = Arc::new(Schema::new(vec![
            Field::new("_valid_from", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let mut batches = Vec::new();
        for b in 0..8i64 {
            let keys: Vec<i64> = (0..32).map(|r| (r * 8 + b) % 97).collect();
            let payloads: Vec<String> = keys
                .iter()
                .map(|k| format!("{k:04}-").repeat(200))
                .collect();
            batches.push(
                RecordBatch::try_new(schema.clone(), vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(StringArray::from(
                        payloads.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    )),
                ])
                .unwrap(),
            );
        }
        let out = drain(sort_chunk(batches, 4 * 1024).unwrap());
        assert!(
            out.len() > 4,
            "small budget must yield many slices: {}",
            out.len()
        );
        let mut all_keys = Vec::new();
        for b in &out {
            assert!(
                b.get_array_memory_size() < 64 * 1024,
                "slice must stay near the budget"
            );
            let vf = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            all_keys.extend(vf.values().iter().copied());
        }
        assert_eq!(all_keys.len(), 8 * 32);
        assert!(all_keys.windows(2).all(|w| w[0] <= w[1]), "globally sorted");
    }

    /// The production `Offset overflow error: 2158186044` class: >2 GiB of
    /// cumulative string bytes in one sort unit. The old whole-group
    /// `concat_batches` provably overflows arrow's i32 string offsets on
    /// this input; the chunked interleave sort handles it with bounded
    /// output slices. (Transiently allocates a few GiB for the input + the
    /// failing concat attempt.)
    #[test]
    fn over_2gib_of_strings_sorts_without_offset_overflow() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_valid_from", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        // 72 batches x 4096 rows x ~8 KB strings ~= 2.3 GiB of string bytes.
        let rows_per_batch = 4096usize;
        let n_batches = 72usize;
        let mut batches = Vec::new();
        for b in 0..n_batches {
            let keys: Vec<i64> = (0..rows_per_batch)
                .map(|r| ((r * n_batches + b) % 10_007) as i64)
                .collect();
            let payload_template = "x".repeat(8_192);
            let payloads: Vec<&str> = (0..rows_per_batch)
                .map(|_| payload_template.as_str())
                .collect();
            batches.push(
                RecordBatch::try_new(schema.clone(), vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(StringArray::from(payloads)),
                ])
                .unwrap(),
            );
        }

        // The OLD path (whole-group concat) fails on exactly this input.
        let err = arrow_select::concat::concat_batches(&schema, &batches)
            .expect_err("a single concatenated batch must overflow i32 offsets");
        assert!(
            err.to_string().contains("Offset overflow"),
            "expected the production overflow class, got: {err}"
        );

        // The chunked sort emits bounded slices — drop each after checking.
        let mut chunk = sort_chunk(batches, 64 * 1024 * 1024).unwrap();
        let mut rows = 0usize;
        let mut last_key = i64::MIN;
        while let Some(b) = chunk.next_batch().unwrap() {
            assert!(
                b.get_array_memory_size() < 256 * 1024 * 1024,
                "every output slice stays far below the i32 offset range"
            );
            let vf = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            for k in vf.values() {
                assert!(*k >= last_key, "globally sorted across slices");
                last_key = *k;
            }
            rows += b.num_rows();
        }
        assert_eq!(rows, rows_per_batch * n_batches);
    }
}
