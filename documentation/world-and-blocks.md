# World, blocks and inventory

This is how the world is stored, generated, edited and given to players.

## Chunks

A chunk is a **32³ cube** of blocks — 32 768 of them. Cubic rather than the classic 16×16×256 column, because the chunk data is streamed and meshed as a unit and a cube keeps the mesh boundary symmetric in all three axes.

```rust
pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
```

### The wire encoding

Chunks travel palette-compressed (`common/src/chunk_blocks.rs`):

```
palette         : Vec<u16>   the distinct block ids used in this chunk
bits_per_block  : u8         0 means "uniform chunk" — one palette entry, no data
data            : Vec<u64>   packed palette indices, `64 / bits` per word
```

The bits needed are derived from the palette size, so a chunk with three distinct blocks costs 2 bits per block, and a chunk of solid stone costs **34 bytes total** (palette + header, no data at all). A typical terrain chunk with a dozen distinct blocks lands around 8 KB. Index bits never straddle a `u64` boundary, which keeps the decode branch-free.

`unpack()` returns `None` rather than panicking on malformed input — this data came off the network, and a corrupt chunk must not take the client down.

### Chunk coordinates and indexing

A `ChunkCoord` is the chunk's world origin divided by 32. Inside a chunk, blocks are indexed **x-major**:

```
index = x + y * 32 + z * 32 * 32
```

The client uses the same formula (`client/src/chunk.rs`), which is what lets the server generate a chunk and the client mesh it without agreeing on anything else. World→chunk conversion uses `div_euclid`/`rem_euclid` so negative coordinates behave.

### Streaming

Each player has a `ChunkRenderState` component holding:

- `render_distance` — the radius, default 5, max 32 (a client asking for more is clamped).
- `current_chunk` — the chunk the player is standing in.
- `desired` — the set of chunks that *should* be on their client, recomputed when the player crosses a border or changes the radius.
- `loaded` — the set that actually is.

The server computes the desired set, queues the missing chunks for generation, and streams them out as they finish.

Two details matter:

- **Vertical layers.** `VERTICAL_RENDER_DISTANCE = 2` keeps two chunk layers loaded above and below the player. Terrain lives near y=0 while a player may be anywhere vertically, so a flat ring at `center.y` used to stream nothing but empty sky. Extra layers are cheap because uniform chunks are 34 bytes.
- **A trickle, not a burst.** `MAX_CHUNK_SENDS_PER_PLAYER_PER_TICK = 16` caps how many chunks go to one player per tick. Without it, a fresh join at render distance 5 wanted ~81 chunks at once, which blew through Renet's reliable-channel memory budget and got the connection killed with *"reliable channel memory usage was exhausted"*. A `RESERVE_BYTES` check (512 KB of headroom) in `stream_chunks_to_players` is the real safety net.

`BlockUpdate` messages are only sent to players who actually have that chunk loaded — broadcasting them to everyone would leak block data outside a player's render distance.

## The block registry

Blocks are **data, not an enum**. `common/src/blocks.rs` defines a `BlockDef` per block:

```rust
pub struct BlockDef {
    pub name: String,          // "exsolis:stone"
    pub solid: bool,           // blocks movement
    pub transparent: bool,     // not drawn opaque
    pub block_entity: bool,    // has an entity attached (chest, furnace)
    pub placeable: bool,       // items and parts are not directly placeable
    pub hidden: bool,          // parts are hidden from the creative list
    pub multiblock: Option<MultiblockLayout>,  // beds, doors
    pub part_of: Option<String>,               // the base a part drops as
}
```

Two identifiers exist for every block:

- **The numeric `BlockId`** (`u16`, `0` is always air) goes into chunk palettes and the GPU world buffer.
- **The name** (`"exsolis:stone"`) is what mods, Lua and the wire protocol use.

### The frozen built-ins

The 11 original block ids are frozen for wire compatibility and are covered by a unit test:

| id | name | id | name |
|---:|---|---:|---|
| 0 | `exsolis:air` | 6 | `exsolis:snow` |
| 1 | `exsolis:stone` | 7 | `exsolis:glass` |
| 2 | `exsolis:dirt` | 8 | `exsolis:log` |
| 3 | `exsolis:grass` | 9 | `exsolis:leaves` |
| 4 | `exsolis:sand` | 10 | `exsolis:chest` |
| 5 | `exsolis:water` | | |

**Never renumber or reorder these.** Append new blocks and extend `ALL`.

### Mod-defined blocks

The server scans an optional Minecraft asset pack at startup (`EXSOLIS_MC_ASSETS`, default `temp/minecraft`) and registers every `blockstates/*.json` stem as `minecraft:<stem>`, sorted. The client performs **the same scan with the same sort**, so both sides derive the same numeric ids without a sync message.

On top of that the server registers:

- **Thaumcraft props** — `assets/models/thaumcraft/*.bbmodel` become `thaumcraft:<stem>`.
- **Multiblock parts** — `_door` gets a `_top` part at `[0,1,0]`, `_bed` gets a `_head` part at `[0,0,-1]`.
- **Orientation parts** — stairs, slabs, pillars and wall torches get view-dependent variants.
- **Items** — `assets/minecraft/models/item/*.json` become `minecraft:<item>`.

Each of these is registered as a *hidden, unplaceable* registry entry derived from its base block, so the creative list and the place path stay clean.

> This mirroring scheme is the thing a registry-sync message will eventually replace — right now both sides must scan identically.

### Placement and orientation

`resolve_placement` turns a master block plus a clicked face plus the player's yaw into a concrete block:

- The clicked face and the player's facing pick the **orientation part** (stair half, slab top/bottom, pillar axis, wall torch side).
- The eye position picks the **top/bottom half** of a stair or slab.
- If nothing applies, the master itself is placed.

Every multiblock footprint cell must be **loaded and air** before anything is placed — a partially blocked bed is never half-placed into the void.

## World generation

The default generator is a simple heightfield (`server/src/chunks.rs::generate_chunk`):

```rust
let height = ((world_x as f32 * 0.1).sin() * 4.0
    + (world_z as f32 * 0.1).cos() * 4.0
    + 16.0) as i32;
```

Below the height is stone, the height cell itself is grass, above is air.

### The advanced generator (opt-in)

A full noise-based generator lives behind the `advanced_worldgen` cargo feature (`server/src/terrain.rs`). It is driven by a **hot-reloadable Lua config** — `mods/worldgen.lua` — so terrain can be retuned while the server runs:

| Section | Controls |
|---|---|
| `seed` | The world seed. |
| `terrain` | Continentalness, erosion and peaks-and-valleys noise layers plus `base_height` / `height_scale`. |
| `caves` | 3D cave noise, carver tunnels and large caverns, with thresholds. |
| `biomes` | Temperature and humidity noise and their weights. |
| `surface` | Which block ids form the surface, plus `water_level` and `beach_depth`. |

Every noise layer is OpenSimplex2 with fBm:

| Field | Meaning |
|---|---|
| `frequency` | How zoomed-in the noise is. `0.0005` = thousands of blocks; `0.1` = a few blocks. |
| `amplitude` | Strength of the layer's contribution (usually 1.0). |
| `octaves` | How many layers are combined. 2–3 is natural, 4–6 is detailed. |
| `persistence` | How much each octave contributes relative to the previous (`0.5` is standard). |
| `lacunarity` | Frequency multiplier per octave (`2.0` is standard). |

## Placing and breaking

Both are **requests**, never facts (`server/src/interaction.rs`). The server validates:

1. The player has an entity, a position and an inventory.
2. The target is within `MAX_REACH` (8 blocks) **of the server's own record of the player's position** — not wherever the client says it is aiming. The distance is measured to the block *centre*.
3. The target chunk is loaded and ready server-side.
4. For a **place**: the target cell is air, the selected hotbar slot holds a placeable item, and every multiblock footprint cell is loaded air.
5. For a **break**: the target cell is not air.

The client's claimed block kind is never trusted. The server places **its own** selected slot's item; a mismatch is logged at debug level and the server's item wins.

On success the server mutates the chunk, broadcasts a `BlockUpdate` to the players who have that chunk, and — for a container block — spawns or despawns the matching block entity.

### Breaking a container

Breaking a chest hands its contents to the breaker (`Inventory::absorb`) rather than dropping them. If the overflow does not fit, the remainder is reported in the log instead of being silently destroyed.

## The inventory

`common/src/inventory.rs` — 36 slots: **0–8 are the hotbar**, 9–35 are main storage. `MAX_STACK` is 64 and is enforced by `add_item`, not just by the UI.

The server owns the inventory absolutely. The client renders `InventorySync` and sends intent. A `SelectHotbar` with an out-of-range index is ignored rather than panicking the server.

### Click semantics

`Inventory::click(index, right)` implements Minecraft-style click handling:

| Slot | Cursor | Result |
|---|---|---|
| empty | empty | nothing |
| full | empty | pick up the stack (right = half, rounded up) |
| empty | full | drop the stack in (right = one item) |
| full | full, same type | merge up to the stack limit (right = one item) |
| full | full, different type | swap (left button only) |

`stash_held` returns the cursor stack to storage when a screen closes, dropping the overflow with a count. `grab` is the creative variant: a full stack to the cursor.

## Creative mode

`server/src/creative.rs` handles `CreativeGive { name }`: the server validates the name against the registry and decides the slot. The client never chooses where an item goes. After any change the server sends a fresh `InventorySync` containing the hotbar contents and the selected index.

The full block list is sent once per join as `BlockList`, so the creative tab renders exactly what the server registered.
