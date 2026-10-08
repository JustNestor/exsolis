use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{BufWriter, Seek, SeekFrom, Write};

use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zerocopy::little_endian::{U16, U32, U64};
use zerocopy::IntoBytes;

use crate::error::{Error, Result};
use crate::format::*;
use crate::util::align_up;

/// One ready-to-write asset (already compressed or raw).
pub(crate) struct Packed<'a> {
    pub name: &'a str,
    pub codec: Codec,
    pub raw_len: u32,
    pub stored: &'a [u8],
    /// xxh3-64 of the uncompressed bytes.
    pub checksum: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PackStats {
    pub entries: u64,
    /// Sum of uncompressed sizes of all entries.
    pub raw_bytes: u64,
    /// Blob bytes actually written (excludes deduplicated entries).
    pub stored_bytes: u64,
    /// Entries that reused an identical, already-written blob.
    pub deduped: u64,
    pub file_size: u64,
}

type DedupKey = (u128, u64, u32, u8);

/// Streams data blobs as they arrive and writes header + index on `finish`.
/// Entries MUST be added in canonical `(ext, path)` order.
pub(crate) struct PakWriter<W: Write + Seek> {
    out: BufWriter<W>,
    pos: u64,
    entries: Vec<Entry>,
    names: Vec<u8>,
    last: String,
    has_last: bool,
    dedup: Option<HashMap<DedupKey, (u64, u32)>>,
    raw_align: u64,
    stats: PackStats,
}

impl<W: Write + Seek> PakWriter<W> {
    pub fn new(w: W, raw_align: u32, dedup: bool) -> Result<Self> {
        let mut out = BufWriter::with_capacity(1 << 20, w);
        out.write_all(&[0u8; HEADER_SIZE])?; // placeholder, rewritten in finish()
        Ok(Self {
            out,
            pos: HEADER_SIZE as u64,
            entries: Vec::new(),
            names: Vec::new(),
            last: String::new(),
            has_last: false,
            dedup: dedup.then(HashMap::new),
            raw_align: (raw_align.max(1) as u64).next_power_of_two(),
            stats: PackStats::default(),
        })
    }

    fn pad(&mut self, align: u64) -> Result<()> {
        const ZEROS: [u8; 4096] = [0; 4096];
        let mut left = (align_up(self.pos, align) - self.pos) as usize;
        self.pos += left as u64;
        while left > 0 {
            let k = left.min(ZEROS.len());
            self.out.write_all(&ZEROS[..k])?;
            left -= k;
        }
        Ok(())
    }

    pub fn add(&mut self, p: Packed<'_>) -> Result<()> {
        if self.has_last {
            match cmp_entries(&self.last, p.name) {
                Ordering::Less => {}
                Ordering::Equal => return Err(Error::DuplicatePath(p.name.to_owned())),
                Ordering::Greater => return Err(Error::Internal("entries added out of order")),
            }
        }
        if p.name.is_empty() || p.name.len() > u16::MAX as usize {
            return Err(Error::InvalidPath(p.name.to_owned()));
        }
        if p.stored.len() > u32::MAX as usize {
            return Err(Error::TooLarge(format!("asset {:?} exceeds 4 GiB", p.name)));
        }
        if self.entries.len() + 2 >= u32::MAX as usize
            || self.names.len() + p.name.len() > u32::MAX as usize
        {
            return Err(Error::TooLarge("too many entries".into()));
        }

        let stored_len = p.stored.len() as u32;
        let key: DedupKey = (xxh3_128(p.stored), p.checksum, p.raw_len, p.codec as u8);
        let reuse = self
            .dedup
            .as_ref()
            .and_then(|m| m.get(&key))
            .filter(|&&(_, len)| len == stored_len)
            .map(|&(off, _)| off);

        let offset = match reuse {
            Some(off) => {
                self.stats.deduped += 1;
                off
            }
            None => {
                let align = if p.codec == Codec::Raw {
                    self.raw_align
                } else {
                    8
                };
                self.pad(align)?;
                let off = self.pos;
                self.out.write_all(p.stored)?;
                self.pos += p.stored.len() as u64;
                self.stats.stored_bytes += p.stored.len() as u64;
                if let Some(m) = &mut self.dedup {
                    m.insert(key, (off, stored_len));
                }
                off
            }
        };

        let name_off = self.names.len() as u32;
        self.names.extend_from_slice(p.name.as_bytes());
        self.entries.push(Entry {
            hash: U64::new(name_hash(p.name.as_bytes())),
            offset: U64::new(offset),
            stored_len: U32::new(stored_len),
            raw_len: U32::new(p.raw_len),
            checksum: U64::new(p.checksum),
            name_off: U32::new(name_off),
            name_len: U16::new(p.name.len() as u16),
            codec: p.codec as u8,
            flags: 0,
        });
        self.last.clear();
        self.last.push_str(p.name);
        self.has_last = true;
        self.stats.raw_bytes += p.raw_len as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<(W, PackStats)> {
        let n = self.entries.len();

        // Hash table: load factor <= 0.5, linear probing, slot = entry_idx + 1.
        let slot_count = (n.max(4) * 2).next_power_of_two();
        let mask = (slot_count - 1) as u64;
        let mut slots = vec![U32::new(0); slot_count];
        for (i, e) in self.entries.iter().enumerate() {
            let mut s = (e.hash.get() & mask) as usize;
            while slots[s].get() != 0 {
                s = (s + 1) & (slot_count - 1);
            }
            slots[s] = U32::new(i as u32 + 1);
        }

        self.pad(8)?;
        let entries_off = self.pos;
        let entries_bytes = self.entries.as_slice().as_bytes();
        let slots_bytes = slots.as_slice().as_bytes();
        let slots_off = entries_off + entries_bytes.len() as u64;
        let names_off = slots_off + slots_bytes.len() as u64;
        let total_len = names_off + self.names.len() as u64;

        let mut h = Xxh3::new();
        h.update(entries_bytes);
        h.update(slots_bytes);
        h.update(&self.names);
        let index_checksum = h.digest();

        self.out.write_all(entries_bytes)?;
        self.out.write_all(slots_bytes)?;
        self.out.write_all(&self.names)?;

        let header = Header {
            magic: MAGIC,
            version: U32::new(VERSION),
            flags: U32::new(0),
            entry_count: U64::new(n as u64),
            entries_off: U64::new(entries_off),
            slots_off: U64::new(slots_off),
            slot_count: U64::new(slot_count as u64),
            names_off: U64::new(names_off),
            names_len: U64::new(self.names.len() as u64),
            index_checksum: U64::new(index_checksum),
            total_len: U64::new(total_len),
            reserved: [0; 48],
        };
        self.out.seek(SeekFrom::Start(0))?;
        self.out.write_all(header.as_bytes())?;
        self.out.flush()?;

        let mut stats = self.stats;
        stats.entries = n as u64;
        stats.file_size = total_len;
        let inner = self
            .out
            .into_inner()
            .map_err(|e| Error::Io(e.into_error()))?;
        Ok((inner, stats))
    }
}
