//! Palette-compressed chunk block storage, shared by server and client.
//!
//! Wire layout (bincode, fixint):
//!   palette         : Vec<u16>  distinct block ids used by this chunk
//!   bits_per_block  : u8        0 = uniform chunk (only 1 palette entry, no data)
//!   data            : Vec<u64>  packed palette indices, `64 / bits` per word,
//!                               never straddling a word boundary
//!
//! Ids are [`BlockId`]s into a [`BlockRegistry`]. `unpack` only checks
//! structural validity (index inside the palette); unknown ids resolve at
//! the use site via the registry so old clients never panic on new blocks.

use serde::{Deserialize, Serialize};

use crate::blocks::BlockId;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkBlocks {
    palette: Vec<BlockId>,
    bits_per_block: u8,
    data: Vec<u64>,
}

impl ChunkBlocks {
    /// Number of bits needed to index a palette of `len` entries.
    /// 0 for a single-entry palette (uniform chunk).
    fn bits_for(len: usize) -> u8 {
        if len <= 1 {
            0
        } else {
            (usize::BITS - (len - 1).leading_zeros()) as u8
        }
    }

    /// Compress a full `CHUNK_VOLUME` block array (x + y*32 + z*32*32 order).
    pub fn pack(blocks: &[BlockId]) -> Self {
        assert_eq!(blocks.len(), CHUNK_VOLUME, "chunk must be 32^3 blocks");

        // block id -> palette index
        let mut lookup: HashMap<BlockId, u8> = HashMap::new();
        let mut palette: Vec<BlockId> = Vec::new();
        for &b in blocks {
            if !lookup.contains_key(&b) {
                // Palette is u8-indexed (max 256 distinct ids per chunk);
                // a 32^3 chunk with more distinct blocks repacks via fallback below.
                if palette.len() >= 256 {
                    break;
                }
                lookup.insert(b, palette.len() as u8);
                palette.push(b);
            }
        }
        // Degenerate overflow (should never happen with sane registries):
        // fall back to full-width packing via palette-per-block is overkill;
        // just extend palette and widen bits (indices still fit u8 only to 256).
        // If we overflowed, rebuild with truncation-free map (max 256 enforced
        // by registry size in practice).
        if palette.len() >= 256 && blocks.iter().any(|b| !lookup.contains_key(b)) {
            // Widen: allow up to 65536 distinct via 16-bit indices packed as u16 words.
            // Simplest correct fallback: store raw ids as palette of all distinct.
            let mut full: Vec<BlockId> = Vec::new();
            let mut map: HashMap<BlockId, usize> = HashMap::new();
            for &b in blocks {
                if !map.contains_key(&b) {
                    map.insert(b, full.len());
                    full.push(b);
                }
            }
            // Store as uniform-width 16-bit data: reuse data words with 16 bits each.
            let mut data = vec![0u64; CHUNK_VOLUME.div_ceil(4)];
            for (i, &b) in blocks.iter().enumerate() {
                let idx = map[&b] as u64;
                data[i / 4] |= idx << ((i % 4) * 16);
            }
            return Self {
                palette: full,
                bits_per_block: 16,
                data,
            };
        }

        let bits = Self::bits_for(palette.len());
        if bits == 0 {
            return Self {
                palette,
                bits_per_block: 0,
                data: Vec::new(),
            };
        }

        let bits_us = bits as usize;
        let per_word = 64 / bits_us;
        let mut data = vec![0u64; CHUNK_VOLUME.div_ceil(per_word)];
        for (i, &b) in blocks.iter().enumerate() {
            let idx = lookup[&b] as u64;
            data[i / per_word] |= idx << ((i % per_word) * bits_us);
        }

        Self {
            palette,
            bits_per_block: bits,
            data,
        }
    }

    /// Decompress. Returns `None` if the payload is malformed (wrong data
    /// length, index outside the palette) instead of panicking — this is
    /// data that came off the network.
    pub fn unpack(&self) -> Option<Vec<BlockId>> {
        if self.palette.is_empty() {
            return None;
        }
        if self.bits_per_block == 0 {
            return Some(vec![self.palette[0]; CHUNK_VOLUME]);
        }

        let bits = self.bits_per_block as usize;
        if bits > 16 {
            return None;
        }
        let per_word = 64 / bits;
        if self.data.len() != CHUNK_VOLUME.div_ceil(per_word) {
            return None;
        }

        let mask = if bits == 64 {
            u64::MAX
        } else if bits == 16 {
            0xFFFF
        } else {
            (1u64 << bits) - 1
        };
        let mut out = Vec::with_capacity(CHUNK_VOLUME);
        for i in 0..CHUNK_VOLUME {
            let idx = ((self.data[i / per_word] >> ((i % per_word) * bits)) & mask) as usize;
            out.push(*self.palette.get(idx)?);
        }
        Some(out)
    }

    /// Replace one block and repack. Unpacks the full 32^3 array, writes the
    /// one block, and repacks. Cheap enough for player-driven edit rates.
    ///
    /// `idx` is `x + y*32 + z*32*32`. Returns `None` if `idx` is out of range
    /// or the existing data is malformed.
    pub fn set_block(&self, idx: usize, block: BlockId) -> Option<Self> {
        if idx >= CHUNK_VOLUME {
            return None;
        }
        let mut blocks = self.unpack()?;
        blocks[idx] = block;
        Some(Self::pack(&blocks))
    }

    /// True if the whole chunk is a single block type (e.g. all air).
    pub fn is_uniform(&self) -> bool {
        self.bits_per_block == 0
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::BlockRegistry;

    #[test]
    fn default_registry_ids_are_frozen() {
        let r = BlockRegistry::default_registry();
        for (name, expect) in [
            ("exsolis:air", 0),
            ("exsolis:stone", 1),
            ("exsolis:dirt", 2),
            ("exsolis:grass", 3),
            ("exsolis:sand", 4),
            ("exsolis:water", 5),
            ("exsolis:snow", 6),
            ("exsolis:glass", 7),
            ("exsolis:log", 8),
            ("exsolis:leaves", 9),
            ("exsolis:chest", 10),
        ] {
            assert_eq!(r.id(name), Some(expect), "registry order moved for {name}");
        }
    }

    #[test]
    fn uniform_roundtrip() {
        let blocks = vec![0; CHUNK_VOLUME];
        let packed = ChunkBlocks::pack(&blocks);
        assert!(packed.is_uniform());
        assert_eq!(packed.unpack().unwrap(), blocks);
    }

    #[test]
    fn mixed_roundtrip() {
        let mut blocks = vec![0; CHUNK_VOLUME];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = (i * 7 + i / 13) as u16 % 11;
        }
        let packed = ChunkBlocks::pack(&blocks);
        assert_eq!(packed.unpack().unwrap(), blocks);
    }

    #[test]
    fn rejects_garbage() {
        let bad = ChunkBlocks {
            palette: vec![],
            bits_per_block: 0,
            data: vec![],
        };
        assert!(bad.unpack().is_none());
    }
}
