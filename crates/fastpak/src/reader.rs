use std::borrow::Cow;
use std::cmp::Ordering;
use std::fs::File;
use std::ops::Range;
use std::path::Path;

use memmap2::Mmap;
use xxhash_rust::xxh3::xxh3_64;
use zerocopy::little_endian::U32;
use zerocopy::FromBytes;

use crate::codec;
use crate::error::{Error, Result};
use crate::format::*;
use crate::util::{par_map, usz};

#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Validate header, index checksum, every entry's bounds/name and every hash slot at
    /// open time (O(entries), no data blobs are touched). Default: true.
    /// Accessors are bounds-checked even when this is off; it only front-loads the errors.
    pub validate: bool,
    /// Verify the xxh3 checksum of the uncompressed bytes on every read. Default: false.
    pub verify_reads: bool,
    /// Refuse to decode assets larger than this (bomb guard for untrusted UGC paks).
    /// Default: 1 GiB.
    pub max_asset_size: u64,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { validate: true, verify_reads: false, max_asset_size: 1 << 30 }
    }
}

/// Stable index of an entry inside one archive. Resolve once, read many times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntryId(u32);

impl EntryId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EntryInfo<'a> {
    pub id: EntryId,
    pub path: &'a str,
    pub raw_len: u32,
    pub stored_len: u32,
    pub codec: Codec,
    /// xxh3-64 of the uncompressed bytes.
    pub checksum: u64,
    /// Absolute file offset of the stored blob.
    pub offset: u64,
}

#[derive(Clone, Debug)]
struct Layout {
    entries: Range<usize>,
    slots: Range<usize>,
    names: Range<usize>,
    data_end: usize,
}

/// A memory-mapped archive. `Send + Sync`; share it freely across threads.
pub struct Pak {
    mmap: Mmap,
    layout: Layout,
    opts: OpenOptions,
}

impl Pak {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, OpenOptions::default())
    }

    /// Opens and maps the file.
    ///
    /// The mapping is only sound while nobody truncates or rewrites the file in
    /// place. Writers in this crate always create a new file and rename it over
    /// the old one, which is safe on Unix; on Windows replacing a mapped file fails.
    pub fn open_with(path: impl AsRef<Path>, opts: OpenOptions) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: see doc comment above; the file is treated as immutable.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap, opts)
    }

    fn from_mmap(mmap: Mmap, opts: OpenOptions) -> Result<Self> {
        let layout = {
            let bytes: &[u8] = &mmap;
            if bytes.len() < HEADER_SIZE {
                return Err(Error::Corrupt("file too small"));
            }
            let h = *Header::ref_from_bytes(&bytes[..HEADER_SIZE])
                .map_err(|_| Error::Corrupt("bad header"))?;
            if h.magic != MAGIC {
                return Err(Error::Corrupt("bad magic"));
            }
            if h.version.get() != VERSION {
                return Err(Error::Unsupported(format!("pak version {}", h.version.get())));
            }
            let total = usz(h.total_len.get())?;
            if total != bytes.len() {
                return Err(Error::Corrupt("file length mismatch (truncated?)"));
            }
            let n = usz(h.entry_count.get())?;
            if n >= u32::MAX as usize - 1 {
                return Err(Error::Corrupt("too many entries"));
            }
            let entries_off = usz(h.entries_off.get())?;
            let slots_off = usz(h.slots_off.get())?;
            let slot_count = usz(h.slot_count.get())?;
            let names_off = usz(h.names_off.get())?;
            let names_len = usz(h.names_len.get())?;

            let ovf = || Error::Corrupt("index size overflow");
            let entries_end = n
                .checked_mul(ENTRY_SIZE)
                .and_then(|b| entries_off.checked_add(b))
                .ok_or_else(ovf)?;
            let slots_end = slot_count
                .checked_mul(4)
                .and_then(|b| slots_off.checked_add(b))
                .ok_or_else(ovf)?;
            let names_end = names_off.checked_add(names_len).ok_or_else(ovf)?;

            if entries_off < HEADER_SIZE
                || slots_off != entries_end
                || names_off != slots_end
                || names_end != total
            {
                return Err(Error::Corrupt("inconsistent index layout"));
            }
            if !slot_count.is_power_of_two() || slot_count <= n {
                return Err(Error::Corrupt("bad hash table size"));
            }

            let layout = Layout {
                entries: entries_off..entries_end,
                slots: slots_off..slots_end,
                names: names_off..names_end,
                data_end: entries_off,
            };

            // Structural casts must succeed regardless of `validate`: accessors rely on them.
            let entries = <[Entry]>::ref_from_bytes(&bytes[layout.entries.clone()])
                .map_err(|_| Error::Corrupt("bad entry table"))?;
            let slots = <[U32]>::ref_from_bytes(&bytes[layout.slots.clone()])
                .map_err(|_| Error::Corrupt("bad slot table"))?;

            if opts.validate {
                if xxh3_64(&bytes[entries_off..total]) != h.index_checksum.get() {
                    return Err(Error::Corrupt("index checksum mismatch"));
                }
                let names = &bytes[layout.names.clone()];
                for e in entries {
                    let codec = Codec::from_u8(e.codec)
                        .ok_or_else(|| Error::Unsupported(format!("codec id {}", e.codec)))?;
                    let off = usz(e.offset.get())?;
                    let end = off.checked_add(e.stored_len.get() as usize).ok_or_else(ovf)?;
                    if off < HEADER_SIZE || end > layout.data_end {
                        return Err(Error::Corrupt("blob out of range"));
                    }
                    if codec == Codec::Raw && e.raw_len.get() != e.stored_len.get() {
                        return Err(Error::Corrupt("raw entry length mismatch"));
                    }
                    if e.raw_len.get() as u64 > opts.max_asset_size {
                        return Err(Error::TooLarge("asset exceeds max_asset_size".into()));
                    }
                    let no = e.name_off.get() as usize;
                    let nl = e.name_len.get() as usize;
                    let nb = no
                        .checked_add(nl)
                        .and_then(|end| names.get(no..end))
                        .ok_or(Error::Corrupt("name out of range"))?;
                    if nl == 0 || std::str::from_utf8(nb).is_err() {
                        return Err(Error::Corrupt("invalid entry name"));
                    }
                }
                for s in slots {
                    if s.get() as usize > n {
                        return Err(Error::Corrupt("hash slot out of range"));
                    }
                }
            }
            layout
        };
        Ok(Self { mmap, layout, opts })
    }

    // ---- raw table access (validated structurally in open) -----------------------------

    #[inline]
    fn entries(&self) -> &[Entry] {
        <[Entry]>::ref_from_bytes(&self.mmap[self.layout.entries.clone()])
            .expect("entry table validated at open")
    }

    #[inline]
    fn slots(&self) -> &[U32] {
        <[U32]>::ref_from_bytes(&self.mmap[self.layout.slots.clone()])
            .expect("slot table validated at open")
    }

    fn name_bytes<'a>(&'a self, e: &Entry) -> Option<&'a [u8]> {
        let names = &self.mmap[self.layout.names.clone()];
        let o = e.name_off.get() as usize;
        names.get(o..o.checked_add(e.name_len.get() as usize)?)
    }

    fn entry_path<'a>(&'a self, e: &Entry) -> &'a str {
        self.name_bytes(e)
            .and_then(|b| std::str::from_utf8(b).ok())
            .unwrap_or("")
    }

    fn entry_ext<'a>(&'a self, e: &Entry) -> &'a str {
        ext_of(self.entry_path(e))
    }

    fn entry(&self, id: EntryId) -> Result<&Entry> {
        self.entries().get(id.0 as usize).ok_or(Error::Corrupt("bad entry id"))
    }

    fn blob(&self, e: &Entry) -> Result<&[u8]> {
        let off = usz(e.offset.get())?;
        let end = off
            .checked_add(e.stored_len.get() as usize)
            .ok_or(Error::Corrupt("blob overflow"))?;
        if off < HEADER_SIZE || end > self.layout.data_end {
            return Err(Error::Corrupt("blob out of range"));
        }
        Ok(&self.mmap[off..end])
    }

    // ---- public API --------------------------------------------------------------------

    pub fn len(&self) -> usize {
        self.layout.entries.len() / ENTRY_SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total archive size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// O(1) average lookup. `path` must be in canonical form (`/`-separated, no leading
    /// slash); see [`normalize_path`](crate::normalize_path).
    pub fn find(&self, path: &str) -> Option<EntryId> {
        let key = path.as_bytes();
        let h = name_hash(key);
        let slots = self.slots();
        let entries = self.entries();
        let mask = slots.len() - 1;
        let mut s = (h & mask as u64) as usize;
        for _ in 0..slots.len() {
            let v = slots[s].get();
            if v == 0 {
                return None;
            }
            let i = (v - 1) as usize;
            if let Some(e) = entries.get(i) {
                if e.hash.get() == h && self.name_bytes(e) == Some(key) {
                    return Some(EntryId(i as u32));
                }
            }
            s = (s + 1) & mask;
        }
        None
    }

    pub fn contains(&self, path: &str) -> bool {
        self.find(path).is_some()
    }

    pub fn info(&self, id: EntryId) -> Option<EntryInfo<'_>> {
        let e = self.entries().get(id.0 as usize)?;
        let path = std::str::from_utf8(self.name_bytes(e)?).ok()?;
        Some(EntryInfo {
            id,
            path,
            raw_len: e.raw_len.get(),
            stored_len: e.stored_len.get(),
            codec: Codec::from_u8(e.codec)?,
            checksum: e.checksum.get(),
            offset: e.offset.get(),
        })
    }

    /// Iterates all entries in layout order (sorted by extension, then path).
    pub fn iter(&self) -> impl Iterator<Item = EntryInfo<'_>> + '_ {
        (0..self.len()).filter_map(move |i| self.info(EntryId(i as u32)))
    }

    /// Index range of all entries with this extension (case-insensitive, with or
    /// without leading dot). O(log n): entries are sorted by extension.
    pub fn ext_range(&self, ext: &str) -> Range<usize> {
        let ext = ext.trim_start_matches('.');
        let es = self.entries();
        let lo = es.partition_point(|e| ext_cmp(self.entry_ext(e), ext) == Ordering::Less);
        let hi = es.partition_point(|e| ext_cmp(self.entry_ext(e), ext) != Ordering::Greater);
        lo..hi.max(lo)
    }

    pub fn iter_ext(&self, ext: &str) -> impl Iterator<Item = EntryInfo<'_>> + '_ {
        self.ext_range(ext).filter_map(move |i| self.info(EntryId(i as u32)))
    }

    /// Hints the OS to page in every asset of this extension with one sequential
    /// read-ahead (they are contiguous on disk). Useful before loading a level's textures.
    #[cfg(unix)]
    pub fn prefetch_ext(&self, ext: &str) -> Result<()> {
        let r = self.ext_range(ext);
        if r.is_empty() {
            return Ok(());
        }
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for e in &self.entries()[r] {
            lo = lo.min(e.offset.get());
            hi = hi.max(e.offset.get() + e.stored_len.get() as u64);
        }
        let start = lo & !0xFFFF; // page-aligned (64 KiB covers 4/16/64 KiB pages)
        self.mmap
            .advise_range(memmap2::Advice::WillNeed, start as usize, (hi - start) as usize)?;
        Ok(())
    }

    /// Stored (possibly compressed) bytes, exactly as on disk.
    pub fn stored_bytes(&self, id: EntryId) -> Result<&[u8]> {
        self.blob(self.entry(id)?)
    }

    /// Reads one asset. Uncompressed assets are returned borrowed straight from the
    /// mmap (zero-copy, no allocation); compressed ones are decoded into a new Vec.
    /// Only this asset's bytes are touched.
    pub fn get(&self, path: &str) -> Result<Cow<'_, [u8]>> {
        let id = self.find(path).ok_or_else(|| Error::NotFound(path.to_owned()))?;
        self.get_id(id)
    }

    pub fn get_id(&self, id: EntryId) -> Result<Cow<'_, [u8]>> {
        let e = self.entry(id)?;
        let stored = self.blob(e)?;
        let raw_len = e.raw_len.get() as usize;
        if raw_len as u64 > self.opts.max_asset_size {
            return Err(Error::TooLarge(format!("asset {:?}", self.entry_path(e))));
        }
        let codec = Codec::from_u8(e.codec)
            .ok_or_else(|| Error::Unsupported(format!("codec id {}", e.codec)))?;
        let out = match codec {
            Codec::Raw => {
                if stored.len() != raw_len {
                    return Err(Error::Corrupt("raw entry length mismatch"));
                }
                Cow::Borrowed(stored)
            }
            Codec::Zstd => Cow::Owned(codec::decompress(stored, raw_len)?),
        };
        if self.opts.verify_reads {
            self.check(e, &out)?;
        }
        Ok(out)
    }

    fn check(&self, e: &Entry, data: &[u8]) -> Result<()> {
        if xxh3_64(data) != e.checksum.get() {
            return Err(Error::ChecksumMismatch(self.entry_path(e).to_owned()));
        }
        Ok(())
    }

    /// Batch read. Results are returned in input order; internally requests are
    /// executed in on-disk order so the kernel sees mostly sequential access.
    pub fn get_many<S: AsRef<str> + Sync>(&self, paths: &[S]) -> Vec<Result<Cow<'_, [u8]>>> {
        self.get_many_par(paths, 1)
    }

    /// Like [`get_many`](Self::get_many) but decompresses on `threads` threads.
    pub fn get_many_par<S: AsRef<str> + Sync>(
        &self,
        paths: &[S],
        threads: usize,
    ) -> Vec<Result<Cow<'_, [u8]>>> {
        let ids: Vec<Option<EntryId>> = paths.iter().map(|p| self.find(p.as_ref())).collect();
        let entries = self.entries();
        let mut order: Vec<usize> = (0..paths.len()).filter(|&i| ids[i].is_some()).collect();
        order.sort_by_key(|&i| {
            ids[i]
                .and_then(|id| entries.get(id.0 as usize))
                .map_or(0, |e| e.offset.get())
        });

        let decoded = par_map(order.len(), threads, |k| {
            let id = ids[order[k]].expect("filtered to Some");
            self.get_id(id)
        });

        let mut out: Vec<Option<Result<Cow<'_, [u8]>>>> = (0..paths.len()).map(|_| None).collect();
        for (k, r) in decoded.into_iter().enumerate() {
            out[order[k]] = Some(r);
        }
        out.into_iter()
            .enumerate()
            .map(|(i, r)| {
                r.unwrap_or_else(|| Err(Error::NotFound(paths[i].as_ref().to_owned())))
            })
            .collect()
    }

    /// Full integrity check: decodes every asset and compares its checksum.
    pub fn verify(&self, threads: usize) -> Result<()> {
        let results = par_map(self.len(), threads, |i| {
            let id = EntryId(i as u32);
            let e = self.entry(id)?;
            let data = self.get_id(id)?;
            self.check(e, &data)
        });
        results.into_iter().collect()
    }
}
