//! The only file that touches zrip. To add another backend (dictionary zstd,
//! brotli, ...) add a `Codec` variant in `format.rs` and dispatch here.

use crate::error::{Error, Result};

/// zrip supports levels -8..=4 (older releases -7..=4); clamp to the common range.
pub fn compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let level = level.clamp(-7, 4);
    zrip::compress(data, level as _).map_err(|e| Error::Codec(format!("{e:?}")))
}

/// Decompresses a single zstd frame, expecting exactly `expected` bytes.
///
/// Decompression-bomb guard: if the frame header declares a content size it
/// must match `expected` *before* any allocation happens. Frames without a
/// declared size are still accepted, but the output length is verified after.
pub fn decompress(frame: &[u8], expected: usize) -> Result<Vec<u8>> {
    if let Some(n) = frame_content_size(frame) {
        if n != expected as u64 {
            return Err(Error::Corrupt("zstd frame content size mismatch"));
        }
    }
    let out = zrip::decompress(frame).map_err(|e| Error::Codec(format!("{e:?}")))?;
    if out.len() != expected {
        return Err(Error::Corrupt("decompressed size mismatch"));
    }
    Ok(out)
}

/// Parses the Frame_Content_Size field of a zstd frame header (RFC 8878 §3.1.1.1).
fn frame_content_size(f: &[u8]) -> Option<u64> {
    if f.len() < 6 || f[..4] != [0x28, 0xB5, 0x2F, 0xFD] {
        return None;
    }
    let fhd = f[4];
    let fcs_flag = fhd >> 6;
    let single_segment = ((fhd >> 5) & 1) == 1;
    let dict_id_flag = (fhd & 3) as usize;

    let mut p = 5usize;
    if !single_segment {
        p += 1; // window descriptor
    }
    p += [0usize, 1, 2, 4][dict_id_flag];

    let n = match fcs_flag {
        0 if single_segment => 1,
        0 => return None,
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let b = f.get(p..p + n)?;
    Some(match n {
        1 => b[0] as u64,
        2 => u16::from_le_bytes([b[0], b[1]]) as u64 + 256,
        4 => u32::from_le_bytes(b.try_into().ok()?) as u64,
        _ => u64::from_le_bytes(b.try_into().ok()?),
    })
}
