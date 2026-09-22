//! Palette-compressed chunk block storage, shared by server and client.
//!
//! Wire layout (bincode, fixint):
//!   palette         : Vec<u8>   distinct block ids used by this chunk
//!   bits_per_block  : u8        0 = uniform chunk (only 1 palette entry, no data)
//!   data            : Vec<u64>  packed palette indices, `64 / bits` per word,
//!                               never straddling a word boundary
//!
//! Typical sizes for a 32^3 chunk (was ~131 KB as Vec<BlockType> in bincode 1,
//! because serde writes every enum variant as a u32 tag no matter what
//! `#[repr(u8)]` says):
//!   all air / all stone      ->  ~40 bytes
//!   3 block types (2 bits)   ->  ~8 KB
//!   5..=16 types (3-4 bits)  ->  ~12-16 KB

use serde::{Deserialize, Serialize};

use crate::BlockType;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkBlocks {
    palette: Vec<u8>,
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
    pub fn pack(blocks: &[BlockType]) -> Self {
        assert_eq!(blocks.len(), CHUNK_VOLUME, "chunk must be 32^3 blocks");

        // block id -> palette index
        let mut lookup: [Option<u8>; 256] = [None; 256];
        let mut palette: Vec<u8> = Vec::new();
        for &b in blocks {
            let id = b as u8;
            if lookup[id as usize].is_none() {
                lookup[id as usize] = Some(palette.len() as u8);
                palette.push(id);
            }
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
            let idx = lookup[b as u8 as usize].unwrap() as u64;
            data[i / per_word] |= idx << ((i % per_word) * bits_us);
        }

        Self {
            palette,
            bits_per_block: bits,
            data,
        }
    }

    /// Decompress. Returns `None` if the payload is malformed (bad block id,
    /// wrong data length, index outside the palette) instead of panicking —
    /// this is data that came off the network.
    pub fn unpack(&self) -> Option<Vec<BlockType>> {
        let palette: Vec<BlockType> = self
            .palette
            .iter()
            .map(|&id| BlockType::from_u8(id))
            .collect::<Option<_>>()?;

        if self.bits_per_block == 0 {
            return Some(vec![*palette.first()?; CHUNK_VOLUME]);
        }

        let bits = self.bits_per_block as usize;
        if bits > 8 {
            return None;
        }
        let per_word = 64 / bits;
        if self.data.len() != CHUNK_VOLUME.div_ceil(per_word) {
            return None;
        }

        let mask = (1u64 << bits) - 1;
        let mut out = Vec::with_capacity(CHUNK_VOLUME);
        for i in 0..CHUNK_VOLUME {
            let idx = ((self.data[i / per_word] >> ((i % per_word) * bits)) & mask) as usize;
            out.push(*palette.get(idx)?);
        }
        Some(out)
    }

    /// Replace one block and repack. There's no in-place bit-packed write
    /// (palette indices are variable-width; growing the palette mid-edit
    /// would require re-laying out every word), so this unpacks the full
    /// 32^3 array, writes the one block, and repacks. Cheap enough for
    /// player-driven edit rates (one call per placed/broken block); would
    /// need a real mutable representation for bulk/worldgen edits.
    ///
    /// `idx` is `x + y*32 + z*32*32` (see the module doc / `pack`'s
    /// ordering contract). Returns `None` if `idx` is out of range or the
    /// existing data is malformed (mirrors `unpack`'s error handling,
    /// since this always unpacks first).
    pub fn set_block(&self, idx: usize, block: BlockType) -> Option<Self> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_ids_match_table() {
        for (i, b) in BlockType::ALL.iter().enumerate() {
            assert_eq!(
                *b as usize, i,
                "BlockType::ALL out of order with discriminants"
            );
        }
    }

    #[test]
    fn uniform_roundtrip() {
        let blocks = vec![BlockType::Air; CHUNK_VOLUME];
        let packed = ChunkBlocks::pack(&blocks);
        assert!(packed.is_uniform());
        assert_eq!(packed.unpack().unwrap(), blocks);
    }

    #[test]
    fn mixed_roundtrip() {
        let mut blocks = vec![BlockType::Air; CHUNK_VOLUME];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = BlockType::ALL[(i * 7 + i / 13) % BlockType::ALL.len()];
        }
        let packed = ChunkBlocks::pack(&blocks);
        assert_eq!(packed.unpack().unwrap(), blocks);
    }

    #[test]
    fn rejects_garbage() {
        let bad = ChunkBlocks {
            palette: vec![200],
            bits_per_block: 0,
            data: vec![],
        };
        assert!(bad.unpack().is_none());
    }
}
