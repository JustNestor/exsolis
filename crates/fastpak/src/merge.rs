use std::collections::hash_map::Entry as MapEntry;
use std::collections::HashMap;
use std::io::{Seek, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::format::cmp_entries;
use crate::reader::{EntryId, Pak};
use crate::util::atomic_write;
use crate::writer::{PackStats, PakWriter, Packed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conflict {
    /// Fail if a path exists in more than one input.
    Error,
    /// Keep the entry from the earliest input.
    FirstWins,
    /// Keep the entry from the latest input (overlay / mod semantics). Default.
    LastWins,
}

#[derive(Clone, Debug)]
pub struct MergeOptions {
    pub on_conflict: Conflict,
    /// Store identical blobs once. Default: false (inputs may be untrusted and xxh3 is not
    /// collision-resistant against an adversary).
    pub dedup: bool,
    pub raw_align: u32,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self { on_conflict: Conflict::LastWins, dedup: false, raw_align: 64 }
    }
}

/// Merges `inputs` into a new archive at `out`. Blobs are copied as-is: nothing is
/// decompressed or recompressed, so cost is one sequential read + write of the data.
/// The result is re-sorted into canonical (extension, path) order.
pub fn merge(inputs: &[&Pak], out: impl AsRef<Path>, opts: &MergeOptions) -> Result<PackStats> {
    atomic_write(out.as_ref(), |file| merge_to(inputs, file, opts))
}

pub fn merge_to<W: Write + Seek>(
    inputs: &[&Pak],
    w: W,
    opts: &MergeOptions,
) -> Result<(W, PackStats)> {
    let mut map: HashMap<&str, (usize, EntryId)> = HashMap::new();
    for (pi, pak) in inputs.iter().enumerate() {
        for info in pak.iter() {
            match map.entry(info.path) {
                MapEntry::Occupied(mut o) => match opts.on_conflict {
                    Conflict::Error => return Err(Error::DuplicatePath(info.path.to_owned())),
                    Conflict::FirstWins => {}
                    Conflict::LastWins => {
                        o.insert((pi, info.id));
                    }
                },
                MapEntry::Vacant(v) => {
                    v.insert((pi, info.id));
                }
            }
        }
    }

    let mut list: Vec<(&str, (usize, EntryId))> = map.into_iter().collect();
    list.sort_by(|a, b| cmp_entries(a.0, b.0));

    let mut writer = PakWriter::new(w, opts.raw_align, opts.dedup)?;
    for (name, (pi, id)) in list {
        let pak = inputs[pi];
        let info = pak.info(id).ok_or(Error::Corrupt("bad entry during merge"))?;
        writer.add(Packed {
            name,
            codec: info.codec,
            raw_len: info.raw_len,
            stored: pak.stored_bytes(id)?,
            checksum: info.checksum,
        })?;
    }
    writer.finish()
}
