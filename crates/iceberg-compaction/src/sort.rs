//! Sort compaction inputs by `_valid_from` — a common universal sort key that
//! gives query engines file-skipping on time filters. Tables without the column
//! are coalesced unsorted.

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_ord::sort::sort_to_indices;
use arrow_select::concat::concat_batches;
use arrow_select::take::take;

/// Concatenate `batches` and sort the result by `_valid_from` ascending.
/// If the column is absent, returns the coalesced batch unsorted.
/// Returns a single combined batch (the rewrite writes one sorted run per group).
pub(crate) fn sort_by_valid_from(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(batches);
    }
    let schema = batches[0].schema();
    let combined = concat_batches(&schema, &batches)?;
    let Ok(idx) = schema.index_of("_valid_from") else {
        return Ok(vec![combined]); // no sort key — just coalesce
    };
    let indices = sort_to_indices(combined.column(idx).as_ref(), None, None)?;
    let sorted_cols = combined
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(vec![RecordBatch::try_new(schema, sorted_cols)?])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::sort_by_valid_from;

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

        let out = sort_by_valid_from(vec![b1, b2]).unwrap();
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
        let out = sort_by_valid_from(vec![b]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num_rows(), 2); // unchanged, just coalesced
    }
}
