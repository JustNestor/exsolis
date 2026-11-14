# Assets

Assets are the textures, models, shaders, UI templates and other files the game loads at runtime. They are delivered as a **memory-mapped archive** over HTTP, with the game channel used only for the manifest and the control messages.

## The big picture

```
server                                       client
──────                                       ──────
scan assets/ → catalog
  (path + xxh3-64 checksum)
      │
      ├──── AssetManifest ─────────────────▶  compare against the local cache
      │                                        request what is missing
      │◀──── AssetRequest ───────────────────┘
      │
      ├──── AssetDownload { url } ─────────▶  GET the .pak delta over HTTP
      │                                        verify checksums, merge into assets.pak
      │◀──── AssetSyncComplete ─────────────┘
      │                                        the player is created now
```

**Asset bytes never travel over the UDP game channel.** Renet carries the manifest, the missing-asset requests, the download tickets, the preload instructions and the hot-reload notifications. The bulk data goes over the Axum HTTP service.

## The server side

`server/src/assets.rs` owns the catalog.

1. The server scans its own `assets/` source directory (override with `EXSOLIS_ASSET_DIR`) and builds a manifest of **normalized paths** and **fastpak xxh3-64 content checksums**.
2. The HTTP listener starts when the first Renet client connects, and shuts down after **30 seconds with no connected clients**.
3. The server polls asset file metadata during development. Added or changed files are delivered as small fastpak deltas; removed files are dropped from future manifests.

Defaults: `0.0.0.0:5001` for the listener, `http://<EXSOLIS_PUBLIC_ADDR host>:5001` for client URLs.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EXSOLIS_ASSET_DIR` | `assets/` | The server-side asset source directory. |
| `EXSOLIS_ASSET_BIND_ADDR` | `0.0.0.0:5001` | The address the HTTP listener binds to. |
| `EXSOLIS_ASSET_BASE_URL` | `http://<public host>:5001` | The public origin clients download from. |
| `EXSOLIS_ASSET_CACHE_DIR` | `~/.exso/downloaded/` | The client's persistent cache. |

### HTTPS

Terminate TLS at a reverse proxy and point `EXSOLIS_ASSET_BASE_URL` at that HTTPS origin; forward `/assets/*` to the configured listener. The Axum server itself speaks plain HTTP.

## The client side

Clients **never** read the server's `assets/` source directory. They keep a persistent cache outside the repository:

| Path | Contents |
|---|---|
| `~/.exso/downloaded/assets.pak` | The merged pak. |
| `~/.exso/downloaded/last_delta.pak` | The most recent verified server transfer. |
| `~/.exso/downloaded/assets/` | The extracted assets. |

On join, the client compares the server manifest with its pak and requests only the missing or changed entries. The merged pak stays on disk for the next session, so the second join downloads nothing.

**The server does not create the player until synchronization completes.** A failed transfer is logged and the client proceeds with whatever it already has in its cache.

### The handler registry

The client's `AssetHandlerRegistry` dispatches resident-load requests by file extension. The default PNG handler decodes image bytes into memory; unknown extensions are retained as bytes. Additional handlers can be registered for shaders, models, audio, or backend-specific GPU uploads.

Resident values currently stay alive for the process lifetime.

## The `.pak` format

`crates/fastpak` is a memory-mapped asset archive. Its design goals, in the crate's own words:

| Property | How |
|---|---|
| **O(1) random access** | An on-disk open-addressing hash table (xxh3-64 of the path) maps a path to a 40-byte entry. Reading an asset touches only that asset's bytes. |
| **Zero-copy for raw assets** | Uncompressed blobs are 64-byte aligned and handed out as `&[u8]` borrowed directly from the mmap (`Cow::Borrowed`). |
| **Per-asset compression** | Every asset is its own zstd frame (via `zrip`), so random access never has to decode neighbours. Incompressible assets (png, ogg, …) are stored raw automatically. |
| **Extension-grouped layout** | Entries *and* data are sorted by `(extension, path)`, so all `.ktx2` files are one contiguous range (`ext_range`, `iter_ext`, `prefetch_ext`). |
| **Batch reads** | `get_many` executes in on-disk order; `get_many_par` also decompresses in parallel. |
| **Cheap merge** | `merge` copies compressed blobs without recompressing; overlay semantics (`Conflict::LastWins`) for a base game pak plus UGC packs. |
| **Hardened open** | Header and index checksums, bounds-checked entries, and a decompression-bomb guard. Safe to open an untrusted pack. |

### Why it is a separate crate

`fastpak` is not part of the game — it is a reusable piece. It has its own README, its own tests and its own release profile (`lto = "thin"`, `codegen-units = 1`). It is published under MIT OR Apache-2.0.

### Library use

```rust
use fastpak::{Intent, PakBuilder, Pak};

let mut b = PakBuilder::new();
b.intent(Intent::Max);
b.add_dir("assets")?;
b.build("game.pak")?;

let pak = Pak::open("game.pak")?;
let tex = pak.get("textures/grass.ktx2")?;          // one asset
let many = pak.get_many_par(&["a.glb", "b.glb"], 4); // batch
```

## Hot reload

While the server runs it polls the asset directory's metadata. When something changes:

- **Added or changed files** are packed into a small fastpak delta and pushed to the clients as `AssetUpdate`. The client downloads it, merges it into its pak and re-registers the affected resources.
- **Removed files** are no longer referenced by future manifests, and clients are told via `AssetRemove`. Resident copies are **retained for the current process**, because the GPU backend does not yet support safe unloading.

On the client, a texture replacement invalidates the chunk meshes that referenced the old texture id, so they are rebuilt against the new one.

## Integrity, not security

The xxh3 checksums detect **accidental corruption**, not malicious modification. A pak is opened with bounds checks and a decompression-bomb guard, so an untrusted pack cannot read out of bounds or exhaust memory — but the checksum is not a signature.
