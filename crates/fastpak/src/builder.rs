use std::borrow::Cow;
use std::collections::HashSet;
use std::fs;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

use xxhash_rust::xxh3::xxh3_64;

use crate::codec;
use crate::error::{Error, Result};
use crate::format::*;
use crate::util::{atomic_write, default_threads, par_map};
use crate::writer::{PackStats, PakWriter, Packed};

/// Compression intent. Maps onto zrip levels (zrip's ceiling is level 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// No compression; blobs are aligned and served zero-copy from the mmap.
    Store,
    /// Lowest latency (zrip level -3): pack on-demand while a client waits.
    Fast,
    /// Good ratio at still-high speed (zrip level 2). Default.
    Balanced,
    /// Best ratio zrip offers (level 4): offline release builds / CDN publishing.
    Max,
    /// Explicit zrip level (clamped to -7..=4).
    Level(i32),
}

impl Intent {
    pub fn level(self) -> Option<i32> {
        match self {
            Intent::Store => None,
            Intent::Fast => Some(-3),
            Intent::Balanced => Some(2),
            Intent::Max => Some(4),
            Intent::Level(l) => Some(l),
        }
    }
}

#[derive(Clone, Copy)]
struct Settings {
    intent: Intent,
    min_compress_size: usize,
    min_saving_percent: u32,
    compress_precompressed: bool,
}

enum Source {
    Bytes(Vec<u8>),
    File(PathBuf),
}

struct Pending {
    path: String,
    source: Source,
    intent: Option<Intent>,
    size_hint: u64,
}

pub struct PakBuilder {
    items: Vec<Pending>,
    seen: HashSet<String>,
    settings: Settings,
    threads: usize,
    dedup: bool,
    raw_align: u32,
}

impl Default for PakBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Formats that are already entropy-coded; compressing them again is wasted CPU.
fn is_precompressed(ext: &str) -> bool {
    const EXTS: &[&str] = &[
        "png", "jpg", "jpeg", "webp", "avif", "gif", "ogg", "oga", "opus", "mp3", "aac", "flac",
        "mp4", "webm", "mkv", "woff2", "zip", "gz", "zst", "xz", "bz2", "7z", "br", "lz4",
    ];
    EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext))
}

// Bound peak memory: compress in batches, then stream to disk.
const BATCH_BYTES: u64 = 128 << 20;
const BATCH_COUNT: usize = 4096;

impl PakBuilder {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            seen: HashSet::new(),
            settings: Settings {
                intent: Intent::Balanced,
                min_compress_size: 64,
                min_saving_percent: 3,
                compress_precompressed: false,
            },
            threads: default_threads(),
            dedup: true,
            raw_align: 64,
        }
    }

    /// Default intent for assets added without an explicit one.
    pub fn intent(&mut self, i: Intent) -> &mut Self {
        self.settings.intent = i;
        self
    }

    /// Worker threads used for compression (default: all cores).
    pub fn threads(&mut self, n: usize) -> &mut Self {
        self.threads = n.max(1);
        self
    }

    /// Store identical blobs once (default: on).
    pub fn dedup(&mut self, on: bool) -> &mut Self {
        self.dedup = on;
        self
    }

    /// Alignment of uncompressed blobs (power of two; default 64, use 4096 for page-aligned).
    pub fn raw_align(&mut self, a: u32) -> &mut Self {
        self.raw_align = a.max(1);
        self
    }

    /// Assets smaller than this are stored raw (default 64 B).
    pub fn min_compress_size(&mut self, n: usize) -> &mut Self {
        self.settings.min_compress_size = n;
        self
    }

    /// Keep the compressed form only if it saves at least this many percent (default 3).
    pub fn min_saving_percent(&mut self, p: u32) -> &mut Self {
        self.settings.min_saving_percent = p.min(99);
        self
    }

    /// Also try compressing png/jpg/ogg/... (default: skipped).
    pub fn compress_precompressed(&mut self, on: bool) -> &mut Self {
        self.settings.compress_precompressed = on;
        self
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn push(&mut self, path: &str, source: Source, intent: Option<Intent>, size_hint: u64) -> Result<&mut Self> {
        let path = normalize_path(path)?;
        if !self.seen.insert(path.clone()) {
            return Err(Error::DuplicatePath(path));
        }
        self.items.push(Pending { path, source, intent, size_hint });
        Ok(self)
    }

    pub fn add_bytes(&mut self, path: &str, data: Vec<u8>) -> Result<&mut Self> {
        let n = data.len() as u64;
        self.push(path, Source::Bytes(data), None, n)
    }

    pub fn add_bytes_with(&mut self, path: &str, data: Vec<u8>, intent: Intent) -> Result<&mut Self> {
        let n = data.len() as u64;
        self.push(path, Source::Bytes(data), Some(intent), n)
    }

    /// Adds a file lazily: it is read by a worker thread only when its batch is processed.
    pub fn add_file(&mut self, path: &str, fs_path: impl AsRef<Path>) -> Result<&mut Self> {
        let fs_path = fs_path.as_ref().to_path_buf();
        let n = fs::metadata(&fs_path)?.len();
        self.push(path, Source::File(fs_path), None, n)
    }

    /// Recursively adds every regular file under `root` (symlinks are skipped).
    /// Archive paths are relative to `root`. Returns the number of files added.
    pub fn add_dir(&mut self, root: impl AsRef<Path>) -> Result<usize> {
        let root = root.as_ref();
        let mut count = 0;
        let mut stack = vec![(root.to_path_buf(), String::new())];
        while let Some((dir, prefix)) = stack.pop() {
            for ent in fs::read_dir(&dir)? {
                let ent = ent?;
                let name = ent
                    .file_name()
                    .into_string()
                    .map_err(|n| Error::InvalidPath(format!("{n:?}")))?;
                let rel = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
                let ft = ent.file_type()?;
                if ft.is_dir() {
                    stack.push((ent.path(), rel));
                } else if ft.is_file() {
                    self.add_file(&rel, ent.path())?;
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Sorts, compresses in parallel and writes `path` atomically (tmp file + rename).
    pub fn build(self, path: impl AsRef<Path>) -> Result<PackStats> {
        atomic_write(path.as_ref(), |file| self.build_to(file))
    }

    /// Same as [`build`](Self::build) but into any `Write + Seek` sink.
    pub fn build_to<W: Write + Seek>(mut self, w: W) -> Result<(W, PackStats)> {
        let mut items = std::mem::take(&mut self.items);
        // Layout optimization: group by extension, then by path (directory locality).
        items.sort_by(|a, b| cmp_entries(&a.path, &b.path));

        let threads = self.threads;
        let settings = self.settings;
        let mut writer = PakWriter::new(w, self.raw_align, self.dedup)?;

        let mut iter = items.into_iter();
        loop {
            let mut batch: Vec<Pending> = Vec::new();
            let mut bytes = 0u64;
            for p in iter.by_ref() {
                bytes += p.size_hint;
                batch.push(p);
                if bytes >= BATCH_BYTES || batch.len() >= BATCH_COUNT {
                    break;
                }
            }
            if batch.is_empty() {
                break;
            }
            let results = par_map(batch.len(), threads, |i| process(&batch[i], &settings));
            for r in results {
                let d = r?;
                writer.add(Packed {
                    name: d.name,
                    codec: d.codec,
                    raw_len: d.raw_len,
                    stored: &d.stored,
                    checksum: d.checksum,
                })?;
            }
        }
        writer.finish()
    }
}

struct Done<'a> {
    name: &'a str,
    codec: Codec,
    raw_len: u32,
    stored: Cow<'a, [u8]>,
    checksum: u64,
}

fn process<'a>(p: &'a Pending, s: &Settings) -> Result<Done<'a>> {
    let data: Cow<'a, [u8]> = match &p.source {
        Source::Bytes(b) => Cow::Borrowed(b.as_slice()),
        Source::File(f) => Cow::Owned(fs::read(f)?),
    };
    if data.len() > u32::MAX as usize {
        return Err(Error::TooLarge(format!("asset {:?} exceeds 4 GiB", p.path)));
    }
    let checksum = xxh3_64(&data);
    let intent = p.intent.unwrap_or(s.intent);

    let skip = intent.level().is_none()
        || data.len() < s.min_compress_size
        || (!s.compress_precompressed && is_precompressed(ext_of(&p.path)));

    if !skip {
        let level = intent.level().unwrap_or(0);
        let c = codec::compress(&data, level)?;
        let limit = data.len() as u64 * (100 - s.min_saving_percent as u64);
        if (c.len() as u64) * 100 <= limit {
            return Ok(Done {
                name: &p.path,
                codec: Codec::Zstd,
                raw_len: data.len() as u32,
                stored: Cow::Owned(c),
                checksum,
            });
        }
    }
    Ok(Done {
        name: &p.path,
        codec: Codec::Raw,
        raw_len: data.len() as u32,
        stored: data,
        checksum,
    })
}
