//! # fastpak
//!
//! Memory-mapped asset archive for textures, models, shaders, animations, ...
//!
//! * **O(1) random access** – an on-disk open-addressing hash table (xxh3-64 of the path)
//!   maps a path to a 40-byte entry; reading an asset touches only that asset's bytes.
//! * **Zero-copy for raw assets** – uncompressed blobs are 64-byte aligned and handed out
//!   as `&[u8]` borrowed directly from the mmap (`Cow::Borrowed`).
//! * **Per-asset compression** – every asset is its own zstd frame (via `zrip`), so random
//!   access never needs to decode neighbours. Incompressible assets (png, ogg, ...) are
//!   stored raw automatically.
//! * **Extension-grouped layout** – entries *and* data are sorted by `(extension, path)`;
//!   all `.ktx2` files are one contiguous range (`ext_range`, `iter_ext`, `prefetch_ext`).
//! * **Batch reads** – `get_many` executes in on-disk order; `get_many_par` also
//!   decompresses in parallel.
//! * **Cheap merge** – `merge` copies compressed blobs without recompressing; overlay
//!   semantics (`Conflict::LastWins`) for base game + UGC packs.
//! * **Hardened open** – header/index checksums, bounds-checked entries, decompression-bomb
//!   guard; safe to open untrusted packs.
//!
//! ```no_run
//! use fastpak::{Intent, PakBuilder, Pak};
//!
//! let mut b = PakBuilder::new();
//! b.intent(Intent::Max);
//! b.add_dir("assets")?;
//! b.build("game.pak")?;
//!
//! let pak = Pak::open("game.pak")?;
//! let tex = pak.get("textures/grass.ktx2")?;          // one asset
//! let many = pak.get_many_par(&["a.glb", "b.glb"], 4); // batch
//! # Ok::<(), fastpak::Error>(())
//! ```
//!
//! See [`format`] docs in `format.rs` for the byte-level layout.

mod builder;
mod codec;
mod error;
mod format;
mod merge;
mod reader;
mod util;
mod writer;

pub use builder::{Intent, PakBuilder};
pub use error::{Error, Result};
pub use format::{cmp_entries, ext_of, normalize_path, Codec};
pub use merge::{merge, merge_to, Conflict, MergeOptions};
pub use reader::{EntryId, EntryInfo, OpenOptions, Pak};
pub use writer::PackStats;

// use fastpak::{Intent, Pak, PakBuilder};
use std::io::{self, Write};
use std::time::Instant;

#[test]
fn test_pack_and_fetch_goldclient() {
    let source_dir = "/home/nestor/Games/c-s/drive_c/Games/CS 1.6 GoldClient";
    let pak_path = "goldclient_test.pak";

    // 1. Measure packing latency
    let pack_start = Instant::now();
    let mut builder = PakBuilder::new();
    builder.intent(Intent::Store);
    builder
        .add_dir(source_dir)
        .expect("Failed to read source directory");
    builder.build(pak_path).expect("Failed to build archive");
    let pack_duration = pack_start.elapsed();

    // 2. Measure fetching latency (dynamically locating the nested file)
    let fetch_start = Instant::now();
    let pak = Pak::open(pak_path).expect("Failed to memory-map archive");

    // Search the index layout for the demo file regardless of its subdirectory depth
    let target_entry = pak
        .iter()
        .find(|e| e.path.ends_with("vbios_de_dust2_2x2.dem"))
        .expect("vbios_de_dust2_2x2.dem not found in any subdirectory");

    println!("Resolved relative path in archive: {}", target_entry.path);

    // O(1) random-access read using its EntryId[cite: 3, 5]
    let asset_data = pak
        .get_id(target_entry.id)
        .expect("Failed to read asset data");
    let fetch_duration = fetch_start.elapsed();

    fn to_mb(bytes: usize) -> String {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }

    // 3. Print metrics and flush stdout
    println!("\n[PERFORMANCE METRICS]");
    println!("Packing duration:  {:?}", pack_duration);
    println!(
        "Fetching duration: {:?} (size: {} bytes)",
        fetch_duration,
        to_mb(asset_data.len())
    );

    io::stdout().flush().unwrap();

    // Cleanup
    // let _ = std::fs::remove_file(pak_path);
}
