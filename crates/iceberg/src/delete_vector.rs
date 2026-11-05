// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Iceberg V3 deletion vectors (`deletion-vector-v1`): a roaring-bitmap-backed set
//! of deleted row positions, serialized to and from Puffin blobs and files.

use std::collections::HashMap;
use std::io::Cursor;
use std::ops::BitOrAssign;

use bytes::Buf;
use roaring::bitmap::Iter;
use roaring::treemap::BitmapIter;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::io::FileIO;
use crate::puffin::{Blob, CompressionCodec, DELETION_VECTOR_V1, PuffinWriter};
use crate::spec::{DataContentType, DataFile, DataFileBuilder, DataFileFormat, Struct};
use crc32fast::Hasher;
use crate::{Error, ErrorKind, Result};

/// Magic bytes prefixing a serialized `deletion-vector-v1` bitmap, per the Iceberg Puffin spec.
/// Iceberg-Java stores these as the little-endian int 1681511377 (0x6439D3D1).
const DV_MAGIC: [u8; 4] = [0xD1, 0xD3, 0x39, 0x64];
const DV_LENGTH_PREFIX_BYTES: usize = 4;
const DV_MAGIC_BYTES: usize = 4;
const DV_CRC_BYTES: usize = 4;
const DV_MIN_BLOB_BYTES: usize = DV_LENGTH_PREFIX_BYTES + DV_MAGIC_BYTES + DV_CRC_BYTES;

/// Iceberg `deletion-vector-v1` Puffin blob magic bytes (fork write path).
const DELETION_VECTOR_MAGIC_BYTES: [u8; 4] = [0xD1, 0xD3, 0x39, 0x64];
/// Minimum blob size: u32 length (4) + magic (4) + u32 crc (4).
const MIN_SERIALIZED_DELETION_VECTOR_BLOB: usize = 12;
/// Puffin blob property: deletion vector cardinality (number of deleted positions).
pub(crate) const DELETION_VECTOR_PROPERTY_CARDINALITY: &str = "cardinality";
/// Puffin blob property: referenced data file path the DV applies to.
pub(crate) const DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE: &str = "referenced-data-file";

/// A set of deleted row positions, the in-memory form of an Iceberg V3
/// `deletion-vector-v1` bitmap.
#[derive(Debug, Default, Clone)]
pub struct DeleteVector {
    inner: RoaringTreemap,
}

impl DeleteVector {
    /// Creates a delete vector that wraps an existing roaring treemap of positions.
    #[allow(unused)]
    pub fn new(roaring_treemap: RoaringTreemap) -> DeleteVector {
        DeleteVector {
            inner: roaring_treemap,
        }
    }

    /// Consumes the delete vector, returning the underlying roaring treemap
    /// of deleted row positions.
    pub fn into_inner(self) -> RoaringTreemap {
        self.inner
    }

    /// Returns an iterator over the deleted row positions in ascending order.
    pub fn iter(&self) -> DeleteVectorIterator<'_> {
        let outer = self.inner.bitmaps();
        DeleteVectorIterator { outer, inner: None }
    }

    /// Marks row position `pos` as deleted; returns `true` if it was newly added.
    pub fn insert(&mut self, pos: u64) -> bool {
        self.inner.insert(pos)
    }

    /// Marks the given `positions` as deleted and returns the number of elements appended.
    ///
    /// The input slice must be strictly ordered in ascending order, and every value must be greater than all existing values already in the set.
    ///
    /// # Errors
    ///
    /// Returns an error if the precondition is not met.
    pub fn insert_positions(&mut self, positions: &[u64]) -> Result<usize> {
        if let Err(err) = self.inner.append(positions.iter().copied()) {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "failed to marks rows as deleted".to_string(),
            )
            .with_source(err));
        }
        Ok(positions.len())
    }

    /// Returns the number of deleted row positions.
    pub fn len(&self) -> u64 {
        self.inner.len()
    }

    /// Parses a `deletion-vector-v1` Puffin blob into a `DeleteVector`.
    ///
    /// The layout, defined by the Iceberg Puffin spec and matching Iceberg-Java's
    /// `BitmapPositionDeleteIndex`, is:
    ///
    /// ```text
    /// [length: u32 big-endian][magic: D1 D3 39 64][vector][crc: u32 big-endian]
    /// ```
    ///
    /// `length` counts the magic and vector bytes (not itself or the CRC). The CRC-32 is
    /// computed over the magic and vector. `vector` is a roaring bitmap in the portable 64-bit
    /// format: a directory of 32-bit key / 32-bit roaring bitmap pairs, ordered by unsigned
    /// comparison of the keys, one bitmap per key.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::DataInvalid`] if the blob is shorter than the minimum, the length
    /// prefix or CRC does not match, the magic is wrong, the roaring bitmap count exceeds the
    /// portable format's maximum, the roaring directory's keys are not ordered by unsigned
    /// comparison, or the roaring payload fails to decode.
    pub fn deserialize(blob: &[u8]) -> Result<Self> {
        if blob.len() < DV_MIN_BLOB_BYTES {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion-vector-v1 blob is {} bytes, shorter than the {DV_MIN_BLOB_BYTES}-byte minimum",
                    blob.len()
                ),
            ));
        }

        // The magic and vector, i.e. the bytes covered by both the length prefix and the CRC.
        let body = &blob[DV_LENGTH_PREFIX_BYTES..blob.len() - DV_CRC_BYTES];
        verify_length_prefix(&blob[..DV_LENGTH_PREFIX_BYTES], body)?;

        // Verify the CRC before interpreting any bytes so a corrupt blob yields a single clear
        // error rather than an opaque roaring decode failure.
        verify_crc(body, &blob[blob.len() - DV_CRC_BYTES..])?;

        let (magic, vector) = body.split_at(DV_MAGIC_BYTES);
        verify_magic(magic)?;

        let inner = decode_roaring_directory(vector)?;

        Ok(DeleteVector { inner })
    }
}

fn verify_length_prefix(mut prefix: &[u8], body: &[u8]) -> Result<()> {
    let declared_len = prefix.try_get_u32().map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "failed to read the deletion-vector-v1 length prefix",
        )
        .with_source(e)
    })? as usize;
    if declared_len != body.len() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "deletion-vector-v1 length prefix is {declared_len}, expected {}",
                body.len()
            ),
        ));
    }
    Ok(())
}

fn verify_crc(body: &[u8], mut crc_bytes: &[u8]) -> Result<()> {
    let stored_crc = crc_bytes.try_get_u32().map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "failed to read the deletion-vector-v1 CRC",
        )
        .with_source(e)
    })?;
    let computed_crc = crc32fast::hash(body);
    if computed_crc != stored_crc {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "deletion-vector-v1 CRC mismatch: computed {computed_crc:#010x}, stored {stored_crc:#010x}"
            ),
        ));
    }
    Ok(())
}

fn verify_magic(magic: &[u8]) -> Result<()> {
    if magic != DV_MAGIC {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("deletion-vector-v1 magic mismatch: {magic:02x?}, expected {DV_MAGIC:02x?}"),
        ));
    }
    Ok(())
}

// The Puffin spec defines the roaring directory as the bitmaps "ordered by unsigned comparison
// of the 32-bit keys", with one bitmap per key. Walk it ourselves (rather than
// `RoaringTreemap::deserialize_from`, which stores keys in a `BTreeMap` via a plain insert and
// would silently accept a stream with duplicate or out-of-order keys, discarding the earlier
// bitmap on a duplicate) so a non-conformant blob is rejected instead of decoded into a value
// that doesn't match what was actually written.
fn decode_roaring_directory(mut reader: &[u8]) -> Result<RoaringTreemap> {
    let bitmap_count = reader.try_get_u64_le().map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "failed to decode deletion-vector-v1 roaring payload",
        )
        .with_source(e)
    })?;
    // The roaring portable format restricts the bitmap count to [0, 2^32 - 1] (it is stored as a
    // u64 with the upper 32 bits reserved as zero padding).
    if bitmap_count > u32::MAX as u64 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "deletion-vector-v1 roaring bitmap count {bitmap_count} exceeds the {}-key maximum",
                u32::MAX
            ),
        ));
    }

    let mut bitmaps = Vec::new();
    let mut last_key: Option<u32> = None;
    for _ in 0..bitmap_count {
        let key = reader.try_get_u32_le().map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "failed to decode deletion-vector-v1 roaring payload",
            )
            .with_source(e)
        })?;
        if let Some(last) = last_key
            && key <= last
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion-vector-v1 roaring keys must be ordered by unsigned comparison, got key {key} after {last}"
                ),
            ));
        }
        last_key = Some(key);

        let bitmap = RoaringBitmap::deserialize_from(&mut reader).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "failed to decode deletion-vector-v1 roaring payload",
            )
            .with_source(e)
        })?;
        bitmaps.push((key, bitmap));
    }

    // `bitmaps` is already sorted by key, but `roaring` has no constructor that accepts
    // pre-sorted pairs without re-sorting them; revisit if that changes upstream.
    Ok(RoaringTreemap::from_bitmaps(bitmaps))
}

// Ideally, we'd just wrap `roaring::RoaringTreemap`'s iterator, `roaring::treemap::Iter` here.
// But right now, it does not have a corresponding implementation of `roaring::bitmap::Iter::advance_to`,
// which is very handy in ArrowReader::build_deletes_row_selection.
// There is a PR open on roaring to add this (https://github.com/RoaringBitmap/roaring-rs/pull/314)
// and if that gets merged then we can simplify `DeleteVectorIterator` here, refactoring `advance_to`
// to just a wrapper around the underlying iterator's method.
/// Iterator over the deleted row positions of a [`DeleteVector`], in ascending order.
pub struct DeleteVectorIterator<'a> {
    // NB: `BitMapIter` was only exposed publicly in https://github.com/RoaringBitmap/roaring-rs/pull/316
    // which is not yet released. As a consequence our Cargo.toml temporarily uses a git reference for
    // the roaring dependency.
    outer: BitmapIter<'a>,
    inner: Option<DeleteVectorIteratorInner<'a>>,
}

struct DeleteVectorIteratorInner<'a> {
    high_bits: u32,
    bitmap_iter: Iter<'a>,
}

impl Iterator for DeleteVectorIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(inner) = &mut self.inner
            && let Some(inner_next) = inner.bitmap_iter.next()
        {
            return Some(u64::from(inner.high_bits) << 32 | u64::from(inner_next));
        }

        if let Some((high_bits, next_bitmap)) = self.outer.next() {
            self.inner = Some(DeleteVectorIteratorInner {
                high_bits,
                bitmap_iter: next_bitmap.iter(),
            })
        } else {
            return None;
        }

        self.next()
    }
}

impl DeleteVectorIterator<'_> {
    /// Advances the iterator so the next yielded position is `>= pos`.
    pub fn advance_to(&mut self, pos: u64) {
        let hi = (pos >> 32) as u32;
        let lo = pos as u32;

        let Some(ref mut inner) = self.inner else {
            return;
        };

        while inner.high_bits < hi {
            let Some((next_hi, next_bitmap)) = self.outer.next() else {
                return;
            };

            *inner = DeleteVectorIteratorInner {
                high_bits: next_hi,
                bitmap_iter: next_bitmap.iter(),
            }
        }

        inner.bitmap_iter.advance_to(lo);
    }
}

impl BitOrAssign for DeleteVector {
    fn bitor_assign(&mut self, other: Self) {
        self.inner.bitor_assign(&other.inner);
    }
}

// Reproduces Iceberg-Java's `deletion-vector-v1` framing so tests can round-trip through
// `deserialize` without a Java writer, and so other test modules can build blob fixtures.
// Cross-implementation golden fixtures produced by Iceberg-Java are tracked separately; this
// only checks that our decode matches our encode.
#[cfg(test)]
pub(crate) fn frame_dv_blob(vector: &[u8]) -> Vec<u8> {
    let body_len = DV_MAGIC_BYTES + vector.len();
    let mut blob = Vec::with_capacity(DV_LENGTH_PREFIX_BYTES + body_len + DV_CRC_BYTES);
    blob.extend_from_slice(&(body_len as u32).to_be_bytes());
    blob.extend_from_slice(&DV_MAGIC);
    blob.extend_from_slice(vector);
    let crc = crc32fast::hash(&blob[DV_LENGTH_PREFIX_BYTES..]);
    blob.extend_from_slice(&crc.to_be_bytes());
    blob
}


/// Fork additions: Puffin `deletion-vector-v1` WRITE (blob + file) and the blob READ
/// kept beside upstream's `deserialize` for callers that hold a `Blob`.
impl DeleteVector {

    /// Returns `true` if there are no deleted positions in this vector.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Serialize this delete vector into an Iceberg V3 `deletion-vector-v1` Puffin blob.
    ///
    /// Blob layout (Iceberg Puffin spec): `length(u32 BE = magic + bitmap) ‖ magic ‖
    /// portable 64-bit RoaringTreemap ‖ crc32(u32 BE over magic + bitmap)`.
    /// `properties` must contain `cardinality` + `referenced-data-file`.
    /// Ported from risingwavelabs/iceberg-rust #113 (design reference only).
    pub fn to_puffin_blob(&self, properties: HashMap<String, String>) -> Result<Blob> {
        Self::check_properties(&properties)?;

        let serialized_bitmap_size = self.inner.serialized_size();
        let combined_length = (DELETION_VECTOR_MAGIC_BYTES.len() + serialized_bitmap_size) as u32;
        let mut data = Vec::with_capacity(
            size_of_val(&combined_length)
                + DELETION_VECTOR_MAGIC_BYTES.len()
                + serialized_bitmap_size
                + 4,
        );

        data.extend_from_slice(&combined_length.to_be_bytes());
        data.extend_from_slice(&DELETION_VECTOR_MAGIC_BYTES);

        let bitmap_start = data.len();
        data.resize(bitmap_start + serialized_bitmap_size, 0);
        {
            let mut cursor = Cursor::new(&mut data[bitmap_start..]);
            self.inner.serialize_into(&mut cursor).map_err(|err| {
                Error::new(
                    ErrorKind::Unexpected,
                    "failed to serialize deletion vector bitmap".to_string(),
                )
                .with_source(err)
            })?;
        }

        let mut hasher = Hasher::new();
        hasher.update(&data[4..]);
        let crc = hasher.finalize();
        data.extend_from_slice(&crc.to_be_bytes());

        Ok(Blob::builder()
            .r#type(DELETION_VECTOR_V1.to_string())
            .fields(vec![])
            .snapshot_id(-1)
            .sequence_number(-1)
            .data(data)
            .properties(properties)
            .build())
    }

    /// Deserialize a delete vector from an Iceberg `deletion-vector-v1` Puffin blob.
    pub fn from_puffin_blob(blob: Blob) -> Result<Self> {
        if blob.blob_type() != DELETION_VECTOR_V1 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("unsupported puffin blob type: {}", blob.blob_type()),
            ));
        }
        Self::from_serialized_bytes(blob.data())
    }

    /// Parse a serialized `deletion-vector-v1` blob payload:
    /// `[u32 BE length][magic 0xD1D33964][serialized RoaringTreemap][u32 BE CRC32]`.
    ///
    /// These are exactly the bytes located by a manifest entry's `content_offset`
    /// and `content_size_in_bytes`, so this works whether the deletion vector is a
    /// standalone blob file (e.g. written by DuckDB) or stored inside a Puffin
    /// container (the manifest offset points past the container framing).
    pub fn from_serialized_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < MIN_SERIALIZED_DELETION_VECTOR_BLOB {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "serialized deletion vector blob too small".to_string(),
            ));
        }

        let magic = &data[4..8];
        if magic != DELETION_VECTOR_MAGIC_BYTES {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "invalid deletion vector magic bytes".to_string(),
            ));
        }

        let combined_length = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let expected_len = size_of_val(&combined_length) + combined_length as usize + 4;
        if expected_len != data.len() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "serialized deletion vector length mismatch: expected {expected_len}, actual {}",
                    data.len()
                ),
            ));
        }

        let bitmap_end = data.len() - 4;
        let bitmap_data = &data[8..bitmap_end];

        let mut hasher = Hasher::new();
        hasher.update(&data[4..bitmap_end]);
        let expected_crc = hasher.finalize();
        let stored_crc = u32::from_be_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]);
        if expected_crc != stored_crc {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("deletion vector crc mismatch: expected {expected_crc}, got {stored_crc}"),
            ));
        }

        let bitmap =
            RoaringTreemap::deserialize_from(&mut Cursor::new(bitmap_data)).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "failed to deserialize deletion vector bitmap".to_string(),
                )
                .with_source(err)
            })?;

        Ok(DeleteVector::new(bitmap))
    }

    fn check_properties(properties: &HashMap<String, String>) -> Result<()> {
        if !properties.contains_key(DELETION_VECTOR_PROPERTY_CARDINALITY) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion vector blob missing required property: {DELETION_VECTOR_PROPERTY_CARDINALITY}"
                ),
            ));
        }
        if !properties.contains_key(DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion vector blob missing required property: {DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE}"
                ),
            ));
        }
        Ok(())
    }

    /// Write this delete vector to a `deletion-vector-v1` Puffin file at `location` and
    /// return the V3 `DataFile{content=PositionDeletes, …}` to feed
    /// `RowDeltaAction::add_delete_files`. Connects DV serialization → Puffin file →
    /// delete-file metadata (offset/length) in one step (cf. RW `deletion_vector_writer.rs`).
    pub async fn write_to_puffin_file(
        &self,
        file_io: &FileIO,
        location: String,
        referenced_data_file: String,
        partition: Struct,
        partition_spec_id: i32,
    ) -> Result<DataFile> {
        let cardinality = self.len();
        let properties = HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                cardinality.to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                referenced_data_file.clone(),
            ),
        ]);
        let blob = self.to_puffin_blob(properties)?;

        let output_file = file_io.new_output(&location)?;
        let mut writer = PuffinWriter::new(&output_file, HashMap::new(), false).await?;
        writer.add(blob, CompressionCodec::None).await?;
        let result = writer.close_with_metadata().await?;
        let file_size = result.file_size_in_bytes;
        let blob_metadata = result.blobs_metadata.first().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "puffin metadata is empty after writing deletion vector",
            )
        })?;

        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(location)
            .file_format(DataFileFormat::Puffin)
            .partition(partition)
            .partition_spec_id(partition_spec_id)
            .record_count(cardinality)
            .file_size_in_bytes(file_size)
            .referenced_data_file(Some(referenced_data_file))
            .content_offset(Some(blob_metadata.offset() as i64))
            .content_size_in_bytes(Some(blob_metadata.length() as i64))
            .build()
            .map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("failed to build deletion vector data file: {err}"),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insertion_and_iteration() {
        let mut dv = DeleteVector::default();
        assert!(dv.insert(42));
        assert!(dv.insert(100));
        assert!(!dv.insert(42));

        let mut items: Vec<u64> = dv.iter().collect();
        items.sort();
        assert_eq!(items, vec![42, 100]);
        assert_eq!(dv.len(), 2);
    }

    #[test]
    fn test_successful_insert_positions() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 2, 3, 1000, 1 << 33];
        assert_eq!(dv.insert_positions(&positions).unwrap(), 5);

        let mut collected: Vec<u64> = dv.iter().collect();
        collected.sort();
        assert_eq!(collected, positions);
    }

    /// Testing scenario: bulk insertion fails because input positions are not strictly increasing.
    #[test]
    fn test_failed_insertion_unsorted_elements() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5, 4];
        let res = dv.insert_positions(&positions);
        assert!(res.is_err());
    }

    /// Testing scenario: bulk insertion fails because input positions have intersection with existing ones.
    #[test]
    fn test_failed_insertion_with_intersection() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5];
        assert_eq!(dv.insert_positions(&positions).unwrap(), 3);

        let res = dv.insert_positions(&[2, 4]);
        assert!(res.is_err());
    }

    /// Testing scenario: bulk insertion fails because input positions have duplicates.
    #[test]
    fn test_failed_insertion_duplicate_elements() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5, 5];
        let res = dv.insert_positions(&positions);
        assert!(res.is_err());
    }

    fn encode_dv_blob(dv: &DeleteVector) -> Vec<u8> {
        let mut vector = Vec::with_capacity(dv.inner.serialized_size());
        dv.inner.serialize_into(&mut vector).unwrap();
        frame_dv_blob(&vector)
    }

    fn dv_of(positions: impl IntoIterator<Item = u64>) -> DeleteVector {
        let mut dv = DeleteVector::default();
        for pos in positions {
            dv.insert(pos);
        }
        dv
    }

    fn sorted(dv: &DeleteVector) -> Vec<u64> {
        let mut positions: Vec<u64> = dv.iter().collect();
        positions.sort_unstable();
        positions
    }

    #[test]
    fn test_deserialize_roundtrip_empty() {
        let blob = encode_dv_blob(&DeleteVector::default());
        assert_eq!(DeleteVector::deserialize(&blob).unwrap().len(), 0);
    }

    #[test]
    fn test_deserialize_roundtrip_small() {
        let positions = [0u64, 5, 100, 1000];
        let dv = DeleteVector::deserialize(&encode_dv_blob(&dv_of(positions))).unwrap();
        assert_eq!(sorted(&dv), positions);
    }

    #[test]
    fn test_deserialize_roundtrip_spanning_64bit_keys() {
        let positions = [1u64, 1 << 33, (1 << 33) + 5, 1 << 34];
        let dv = DeleteVector::deserialize(&encode_dv_blob(&dv_of(positions))).unwrap();
        assert_eq!(sorted(&dv), positions);
    }

    // Java run-optimizes every deletion vector before writing, so real blobs carry RUN
    // containers, which use the SERIAL_COOKIE roaring layout. Force that layout so decode
    // exercises the run-container path rather than only array and bitmap containers.
    #[test]
    fn test_deserialize_roundtrip_run_optimized() {
        let mut dv = dv_of(0..10_000);
        assert!(
            dv.inner.optimize(),
            "expected a dense range to run-length encode"
        );
        let decoded = DeleteVector::deserialize(&encode_dv_blob(&dv)).unwrap();
        assert_eq!(decoded.len(), 10_000);
        let positions = sorted(&decoded);
        assert_eq!(positions.first(), Some(&0));
        assert_eq!(positions.last(), Some(&9_999));
    }

    #[test]
    fn test_deserialize_rejects_short_blob() {
        let err = DeleteVector::deserialize(&[0u8; DV_MIN_BLOB_BYTES - 1]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_deserialize_rejects_bad_magic() {
        let mut blob = encode_dv_blob(&dv_of([1]));
        blob[DV_LENGTH_PREFIX_BYTES] ^= 0xFF;
        // Recompute the CRC so the magic check, not the CRC check, is what fails.
        let end = blob.len() - DV_CRC_BYTES;
        let crc = crc32fast::hash(&blob[DV_LENGTH_PREFIX_BYTES..end]);
        blob[end..].copy_from_slice(&crc.to_be_bytes());
        let err = DeleteVector::deserialize(&blob).unwrap_err();
        assert!(err.message().contains("magic mismatch"), "got: {err}");
    }

    #[test]
    fn test_deserialize_rejects_bad_crc() {
        let mut blob = encode_dv_blob(&dv_of([1, 2, 3]));
        let end = blob.len() - DV_CRC_BYTES;
        blob[end] ^= 0xFF;
        let err = DeleteVector::deserialize(&blob).unwrap_err();
        assert!(err.message().contains("CRC mismatch"), "got: {err}");
    }

    #[test]
    fn test_deserialize_rejects_length_prefix_mismatch() {
        let mut blob = encode_dv_blob(&dv_of([1]));
        let declared = u32::from_be_bytes(blob[..DV_LENGTH_PREFIX_BYTES].try_into().unwrap());
        blob[..DV_LENGTH_PREFIX_BYTES].copy_from_slice(&(declared + 1).to_be_bytes());
        let err = DeleteVector::deserialize(&blob).unwrap_err();
        assert!(err.message().contains("length prefix"), "got: {err}");
    }

    // Crafts a raw roaring treemap directory (bitmap count header + key/bitmap entries) so tests
    // can exercise key-ordering violations that `dv_of`/`encode_dv_blob` can never produce, since
    // `DeleteVector::insert` always keeps keys unique and ascending.
    fn raw_roaring_vector(entries: &[(u32, &[u32])]) -> Vec<u8> {
        let mut vector = Vec::new();
        vector.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, positions) in entries {
            let mut bitmap = RoaringBitmap::new();
            for &pos in *positions {
                bitmap.insert(pos);
            }
            vector.extend_from_slice(&key.to_le_bytes());
            bitmap.serialize_into(&mut vector).unwrap();
        }
        vector
    }

    // `RoaringTreemap::deserialize_from` stores keys in a `BTreeMap` via a plain insert, so
    // without our own ordering check, decoding this would silently keep only the second bitmap
    // for key 5 (position 2) and drop the first (position 1).
    #[test]
    fn test_deserialize_rejects_duplicate_keys() {
        let vector = raw_roaring_vector(&[(5, &[1]), (5, &[2])]);
        let err = DeleteVector::deserialize(&frame_dv_blob(&vector)).unwrap_err();
        assert!(err.message().contains("unsigned comparison"), "got: {err}");
    }

    // The Puffin spec requires the roaring directory's keys to be "ordered by unsigned
    // comparison"; a stream with unique but out-of-order keys is not a conformant blob even
    // though `BTreeMap` would happily reorder it into a correct-looking result.
    #[test]
    fn test_deserialize_rejects_out_of_order_keys() {
        let vector = raw_roaring_vector(&[(5, &[1]), (3, &[2])]);
        let err = DeleteVector::deserialize(&frame_dv_blob(&vector)).unwrap_err();
        assert!(err.message().contains("unsigned comparison"), "got: {err}");
    }

    // The roaring portable format stores the bitmap count as a u64 with the upper 32 bits
    // reserved as zero padding, restricting it to [0, 2^32 - 1]; a value above that is not a
    // conformant blob, regardless of whether any key/bitmap entries follow.
    #[test]
    fn test_deserialize_rejects_bitmap_count_overflow() {
        let vector = (u32::MAX as u64 + 1).to_le_bytes().to_vec();
        let err = DeleteVector::deserialize(&frame_dv_blob(&vector)).unwrap_err();
        assert!(err.message().contains("exceeds the"), "got: {err}");
    }
}

#[cfg(test)]
mod fork_tests {
    use super::*;

    #[test]
    fn test_insertion_and_iteration() {
        let mut dv = DeleteVector::default();
        assert!(dv.insert(42));
        assert!(dv.insert(100));
        assert!(!dv.insert(42));

        let mut items: Vec<u64> = dv.iter().collect();
        items.sort();
        assert_eq!(items, vec![42, 100]);
        assert_eq!(dv.len(), 2);
    }

    #[test]
    fn test_successful_insert_positions() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 2, 3, 1000, 1 << 33];
        assert_eq!(dv.insert_positions(&positions).unwrap(), 5);

        let mut collected: Vec<u64> = dv.iter().collect();
        collected.sort();
        assert_eq!(collected, positions);
    }

    /// Testing scenario: bulk insertion fails because input positions are not strictly increasing.
    #[test]
    fn test_failed_insertion_unsorted_elements() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5, 4];
        let res = dv.insert_positions(&positions);
        assert!(res.is_err());
    }

    /// Testing scenario: bulk insertion fails because input positions have intersection with existing ones.
    #[test]
    fn test_failed_insertion_with_intersection() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5];
        assert_eq!(dv.insert_positions(&positions).unwrap(), 3);

        let res = dv.insert_positions(&[2, 4]);
        assert!(res.is_err());
    }

    /// Testing scenario: bulk insertion fails because input positions have duplicates.
    #[test]
    fn test_failed_insertion_duplicate_elements() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5, 5];
        let res = dv.insert_positions(&positions);
        assert!(res.is_err());
    }

    fn dv_props() -> HashMap<String, String> {
        HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                "0".to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                "s3://bucket/data/f.parquet".to_string(),
            ),
        ])
    }

    /// Self round-trip: serialize → Puffin blob → deserialize recovers the positions,
    /// validating the frame (length, magic, crc) and serialize/deserialize symmetry.
    #[test]
    fn test_dv_puffin_blob_roundtrip() {
        let positions = [1u64, 5, 42, 100, 1 << 33, (1u64 << 33) + 7];
        let mut dv = DeleteVector::default();
        for p in positions {
            dv.insert(p);
        }
        let blob = dv.to_puffin_blob(dv_props()).unwrap();
        assert_eq!(blob.blob_type(), DELETION_VECTOR_V1);

        let restored = DeleteVector::from_puffin_blob(blob).unwrap();
        let mut got: Vec<u64> = restored.iter().collect();
        got.sort();
        assert_eq!(got, positions.to_vec());
    }

    /// Spark-compatibility proxy: parse the serialized bitmap with the EXACT algorithm
    /// pyiceberg's `_deserialize_bitmap` uses — `[u64 LE bucket count]` then per bucket
    /// `[u32 LE high-key + 32-bit portable RoaringBitmap]`. If this recovers the
    /// positions, the bytes are Iceberg/Spark portable (no Spark needed for the signal).
    #[test]
    fn test_dv_blob_is_iceberg_portable() {
        use std::io::Read;

        use roaring::RoaringBitmap;

        let positions = [3u64, 7, 100, (1u64 << 33) + 5];
        let mut dv = DeleteVector::default();
        for p in positions {
            dv.insert(p);
        }
        let blob = dv.to_puffin_blob(dv_props()).unwrap();
        let data = blob.data();

        // Frame (certain): [u32 BE len][magic][bitmap][u32 BE crc]
        assert_eq!(&data[4..8], &DELETION_VECTOR_MAGIC_BYTES);
        let bitmap = &data[8..data.len() - 4];

        // pyiceberg portable parse
        let mut cur = Cursor::new(bitmap);
        let mut count_buf = [0u8; 8];
        cur.read_exact(&mut count_buf).unwrap();
        let n_buckets = u64::from_le_bytes(count_buf);

        let mut recovered: Vec<u64> = Vec::new();
        for _ in 0..n_buckets {
            let mut key_buf = [0u8; 4];
            cur.read_exact(&mut key_buf).unwrap();
            let hi = u32::from_le_bytes(key_buf) as u64;
            let bm = RoaringBitmap::deserialize_from(&mut cur).unwrap();
            for lo in bm.iter() {
                recovered.push((hi << 32) | u64::from(lo));
            }
        }
        recovered.sort();
        assert_eq!(
            recovered,
            positions.to_vec(),
            "serialized bitmap is NOT Iceberg-portable — roaring serialize_into header \
             differs from pyiceberg layout; switch to hand-rolled portable framing"
        );
    }

    /// Piece 2 — full Puffin-FILE round-trip in Rust: write a DV blob to a real
    /// Puffin file via `PuffinWriter`, read it back via `PuffinReader`, and recover
    /// the deleted positions. Proves the Puffin file framing, not just the blob bytes.
    #[tokio::test]
    async fn test_dv_puffin_file_roundtrip() {
        use tempfile::TempDir;

        use crate::io::FileIO;
        use crate::puffin::{CompressionCodec, PuffinReader, PuffinWriter};

        let positions = [2u64, 9, 256, (1u64 << 33) + 11];
        let mut dv = DeleteVector::default();
        for p in positions {
            dv.insert(p);
        }
        assert!(!dv.is_empty());

        let mut props = dv_props();
        props.insert(
            DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
            dv.len().to_string(),
        );
        let blob = dv.to_puffin_blob(props).unwrap();

        let tmp = TempDir::new().unwrap();
        let path_buf = tmp.path().join("dv.puffin");
        let path = path_buf.to_str().unwrap();

        let file_io = FileIO::new_with_fs();
        let output = file_io.new_output(path).unwrap();
        let mut writer = PuffinWriter::new(&output, HashMap::new(), false)
            .await
            .unwrap();
        writer.add(blob, CompressionCodec::None).await.unwrap();
        writer.close().await.unwrap();

        let input = output.to_input_file();
        let reader = PuffinReader::new(input);
        let meta = reader.file_metadata().await.unwrap().clone();
        assert_eq!(meta.blobs.len(), 1);
        let read_blob = reader.blob(meta.blobs.first().unwrap()).await.unwrap();

        let restored = DeleteVector::from_puffin_blob(read_blob).unwrap();
        let mut got: Vec<u64> = restored.iter().collect();
        got.sort();
        assert_eq!(got, positions.to_vec());
    }

    /// Piece 2.5 — glue: write a DV to a Puffin file and get back a V3
    /// `DataFile{PositionDeletes}` (offset/size/referenced-file filled), then read the
    /// written file back and recover the positions.
    #[tokio::test]
    async fn test_dv_write_to_puffin_file() {
        use tempfile::TempDir;

        use crate::puffin::PuffinReader;

        let positions = [4u64, 11, 512, (1u64 << 33) + 3];
        let mut dv = DeleteVector::default();
        for p in positions {
            dv.insert(p);
        }

        let tmp = TempDir::new().unwrap();
        let path_buf = tmp.path().join("dv2.puffin");
        let location = path_buf.to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let data_file = dv
            .write_to_puffin_file(
                &file_io,
                location.clone(),
                "s3://bucket/data/x.parquet".to_string(),
                Struct::empty(),
                0,
            )
            .await
            .unwrap();

        assert_eq!(data_file.content_type(), DataContentType::PositionDeletes);
        assert_eq!(
            data_file.referenced_data_file().as_deref(),
            Some("s3://bucket/data/x.parquet")
        );
        assert!(data_file.content_offset().is_some());
        assert!(data_file.content_size_in_bytes().is_some());

        // The written Puffin file reads back to the same positions.
        let input = file_io.new_input(&location).unwrap();
        let reader = PuffinReader::new(input);
        let meta = reader.file_metadata().await.unwrap().clone();
        let blob = reader.blob(meta.blobs.first().unwrap()).await.unwrap();
        let restored = DeleteVector::from_puffin_blob(blob).unwrap();
        let mut got: Vec<u64> = restored.iter().collect();
        got.sort();
        assert_eq!(got, positions.to_vec());
    }

    /// Cross-implementation byte-parity with Apache Iceberg-Java: the serialized
    /// `deletion-vector-v1` payload for positions {1,3,5,7,9} must be byte-identical
    /// to the Java-produced golden fixture (lifted from apache/iceberg test resources
    /// `small-alternating-values-position-index.bin` via apache/iceberg-go). Proves our
    /// roaring serialization + framing exactly match the Iceberg-Java reference.
    #[test]
    fn test_dv_payload_byte_identical_to_java_golden() {
        let golden: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/puffin/deletion-vector-v1-payload.bin"
        ));

        let mut dv = DeleteVector::default();
        for p in [1u64, 3, 5, 7, 9] {
            dv.insert(p);
        }
        let props = HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                "5".to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                "data/test.parquet".to_string(),
            ),
        ]);
        let blob = dv.to_puffin_blob(props).unwrap();

        assert_eq!(
            blob.data(),
            golden,
            "DV payload must be byte-identical to the apache/iceberg Java golden fixture"
        );
    }

    /// Empty deletion vector round-trips (mirrors iceberg-go `TestSerializeDVEmpty`).
    #[test]
    fn test_dv_empty_roundtrip() {
        let dv = DeleteVector::default();
        let props = HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                "0".to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                "data/empty.parquet".to_string(),
            ),
        ]);
        let blob = dv.to_puffin_blob(props).unwrap();
        let restored = DeleteVector::from_puffin_blob(blob).unwrap();
        assert!(restored.is_empty());
        assert_eq!(restored.len(), 0);
    }

    /// Positions straddling the 2^31 (Java-signed) and 2^32 (roaring bucket)
    /// boundaries round-trip (mirrors iceberg-go `TestSerializeDVLargePositions`).
    #[test]
    fn test_dv_boundary_positions_roundtrip() {
        let positions = [100u64, 101, 2_147_483_747, 2_147_483_748, (1u64 << 32) | 42];
        let mut dv = DeleteVector::default();
        for p in positions {
            dv.insert(p);
        }
        let props = HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                "5".to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                "data/boundary.parquet".to_string(),
            ),
        ]);
        let blob = dv.to_puffin_blob(props).unwrap();
        let restored = DeleteVector::from_puffin_blob(blob).unwrap();
        let mut got: Vec<u64> = restored.iter().collect();
        got.sort();
        assert_eq!(got, positions.to_vec());
    }

    /// Byte-parity with Apache Iceberg-Java: the **empty** DV payload
    /// (`empty-position-index.bin` from apache/iceberg test resources).
    #[test]
    fn test_dv_payload_byte_identical_to_java_empty() {
        let golden: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/puffin/empty-position-index.bin"
        ));
        let dv = DeleteVector::default();
        let props = HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                "0".to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                "data/test.parquet".to_string(),
            ),
        ]);
        let blob = dv.to_puffin_blob(props).unwrap();
        assert_eq!(
            blob.data(),
            golden,
            "empty DV payload must be byte-identical to the Java golden fixture"
        );
    }

    /// Byte-parity with Apache Iceberg-Java: small + large positions spanning two
    /// 16-bit roaring containers — {100, 101, INT_MAX+100, INT_MAX+101} per
    /// `small-and-large-values-position-index.bin` from apache/iceberg test resources.
    #[test]
    fn test_dv_payload_byte_identical_to_java_small_and_large() {
        let golden: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/puffin/small-and-large-values-position-index.bin"
        ));
        let mut dv = DeleteVector::default();
        for p in [100u64, 101, 2_147_483_747, 2_147_483_748] {
            dv.insert(p);
        }
        let props = HashMap::from([
            (
                DELETION_VECTOR_PROPERTY_CARDINALITY.to_string(),
                "4".to_string(),
            ),
            (
                DELETION_VECTOR_PROPERTY_REFERENCED_DATA_FILE.to_string(),
                "data/test.parquet".to_string(),
            ),
        ]);
        let blob = dv.to_puffin_blob(props).unwrap();
        assert_eq!(
            blob.data(),
            golden,
            "small+large DV payload must be byte-identical to the Java golden fixture"
        );
    }
}
