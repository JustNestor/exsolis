//! On-disk format. All integers are little-endian and every struct has
//! alignment 1 (zerocopy `little_endian` types), so the mmap can be cast
//! directly on any platform with no alignment concerns.
//!
//! ```text
//! offset 0      Header (128 B)
//! offset 128    data blobs   (raw blobs aligned to `raw_align`, zstd blobs to 8)
//! entries_off   [Entry; entry_count]      40 B each, sorted by (ext, path)
//! slots_off     [u32; slot_count]         open-addressing hash table, 0 = empty, else entry_idx+1
//! names_off     names blob                UTF-8 paths, concatenated, no separators
//! ```
//!
//! The index (entries + slots + names) is contiguous at the end of the file and
//! covered by one xxh3-64 checksum stored in the header.

use std::cmp::Ordering;

use xxhash_rust::xxh3::xxh3_64;
use zerocopy::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::error::{Error, Result};

pub const MAGIC: [u8; 8] = *b"RPAK\r\n\x1a\n";
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 128;
pub const ENTRY_SIZE: usize = 40;

#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: U32,
    pub flags: U32,
    pub entry_count: U64,
    pub entries_off: U64,
    pub slots_off: U64,
    pub slot_count: U64,
    pub names_off: U64,
    pub names_len: U64,
    /// xxh3-64 over `entries_off..total_len`.
    pub index_checksum: U64,
    /// Exact file length; detects truncation.
    pub total_len: U64,
    pub reserved: [u8; 48],
}

#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub struct Entry {
    /// xxh3-64 of the path bytes (hash-table key).
    pub hash: U64,
    /// Absolute file offset of the stored blob.
    pub offset: U64,
    pub stored_len: U32,
    /// Uncompressed length.
    pub raw_len: U32,
    /// xxh3-64 of the *uncompressed* bytes.
    pub checksum: U64,
    /// Offset into the names blob.
    pub name_off: U32,
    pub name_len: U16,
    pub codec: u8,
    pub flags: u8,
}

const _: () = assert!(core::mem::size_of::<Header>() == HEADER_SIZE);
const _: () = assert!(core::mem::size_of::<Entry>() == ENTRY_SIZE);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    Raw = 0,
    Zstd = 1,
}

impl Codec {
    pub fn from_u8(v: u8) -> Option<Codec> {
        match v {
            0 => Some(Codec::Raw),
            1 => Some(Codec::Zstd),
            _ => None,
        }
    }
}

#[inline]
pub fn name_hash(name: &[u8]) -> u64 {
    xxh3_64(name)
}

/// Extension of the last path component, without the dot ("" if none).
/// A leading dot (".gitignore") does not count as an extension.
pub fn ext_of(path: &str) -> &str {
    let file = path.rsplit('/').next().unwrap_or(path);
    match file.rfind('.') {
        Some(i) if i > 0 => &file[i + 1..],
        _ => "",
    }
}

/// ASCII-case-insensitive lexicographic comparison of extensions.
pub fn ext_cmp(a: &str, b: &str) -> Ordering {
    a.bytes()
        .map(|c| c.to_ascii_lowercase())
        .cmp(b.bytes().map(|c| c.to_ascii_lowercase()))
}

/// Canonical entry order: by (extension, path). Entries of the same type end up
/// contiguous both in the index and in the data section.
pub fn cmp_entries(a: &str, b: &str) -> Ordering {
    ext_cmp(ext_of(a), ext_of(b)).then_with(|| a.cmp(b))
}

/// Normalizes a path to the canonical form used inside archives:
/// `/` separators, no leading slash, no `.`/empty components, no `..`.
pub fn normalize_path(p: &str) -> Result<String> {
    let invalid = || Error::InvalidPath(p.to_owned());
    if p.contains('\0') {
        return Err(invalid());
    }
    let mut out = String::with_capacity(p.len());
    for part in p.split(|c: char| c == '/' || c == '\\') {
        match part {
            "" | "." => continue,
            ".." => return Err(invalid()),
            _ => {}
        }
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(part);
    }
    if out.is_empty() || out.len() > u16::MAX as usize {
        return Err(invalid());
    }
    Ok(out)
}
