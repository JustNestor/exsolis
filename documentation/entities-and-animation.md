# Entities and animation

Everything in the world that is not a block is an **entity**. Players are entities too — that is what lets a remote avatar and a mod-spawned prop travel the exact same render path.

## The entity model

```rust
pub enum EntityKind {
    Player,   // a connected player
    Block,    // bound to a BlockPos: a chest, a furnace, ...
    Item,     // a dropped item stack, with simple gravity
    Prop,     // a mod-spawned decorative prop with a model but no behaviour
}
```

`EntityKind` is a bincode enum, so **new kinds must be appended** — inserting one shifts every later index and desyncs the client's decode.

### Ids

`EntityId` is a `u64` allocated monotonically and **never reused** within a session. The reasoning is in the registry itself:

> A free-list would be tempting, but reusing an id lets a late `EntityStateUpdate` from a dropped packet animate a *different* entity. Ids are never reused; `MAX_ENTITIES` bounds the session instead.

`MAX_ENTITIES` is 4096 — hitting it drops the spawn rather than the server, so a mod that spawns in a loop cannot exhaust memory or flood the reliable channel.

### The registry

`server/src/entities.rs` owns an `EntityRegistry` that maps `EntityId` to an ECS entity plus its record. It is the single place that decides who exists.

- `records()` returns the live records **sorted by id**, so mod-facing listings do not reshuffle between ticks.
- `at_block(pos)` finds the entity bound to a block position.
- `id_of(entity)` / `entity_of(id)` translate between the wire id and the Bevy `Entity`.

`PlayerMap` still exists, but only because input handling wants a `PlayerId ↔ Entity` lookup. It is now a *view* over the registry, not a second source of truth.

## The wire protocol

| Message | Channel | Meaning |
|---|---|---|
| `SpawnEntity { entity_id, kind, state, model, scale, player_id }` | reliable | Bring an entity into existence. |
| `DespawnEntity { entity_id }` | reliable | Remove it. Unknown ids are ignored. |
| `EntityStateUpdate { entity_id, state }` | unreliable | It moved. |
| `BlockAnimState { entity_id, block, state }` | reliable | A block entity's animation state changed server-side; clients cross-fade to the clip the state names. |

Client → server, for the same system: `ClientMessage::UseBlock { pos, action }`
(reliable ordered) is the *intent*; the server resolves the action against the
block's animation set and owns the resulting state. See
[Block animation states](#block-animation-states-server-authoritative).

`model` is a catalog asset path (`models/girl.glb`) or `None` for entities that render nothing on their own — a block entity renders *its block*, not a model. `player_id` is set only for `EntityKind::Player`, so the client can correlate the avatar with `YourPlayerId` and with mod effects that address players by id.

`EntityState` carries more than a position:

```rust
pub struct EntityState {
    pub x: f32, pub y: f32, pub z: f32,
    pub yaw: f32, pub pitch: f32,
    pub speed: f32,      // blocks/second along the horizontal facing
    pub airborne: bool,
}
```

`speed` and `airborne` exist because the server already knows the authoritative speed. Deriving it client-side from unreliable position deltas is noisy, and the client needs it to pick an animation clip.

## Block entities

A block entity is an entity whose lifetime is tied to a block position rather than to physics:

- Placing a container block spawns one (`EntityKind::Block`, `model: None`, bound to the `BlockPos`).
- Breaking that block despawns it, and its contents are handed to the breaker — a chest can never be destroyed to destroy its loot.
- Its transform is the block's own position. Yaw is derived from the position hash (`(x * 7 + z * 13) & 3`), so a chest consistently faces one of four directions instead of every chest pointing the same way.

Block entities also carry a `Container` component — the same zero-based slot container the player's `Inventory` uses, so mod code treats them identically.

## The client side

`client/src/entities.rs` mirrors the server's registry. The client learns about entities only through the three messages above:

- `SpawnEntity` brings something into being, `EntityStateUpdate` moves it, `DespawnEntity` removes it.
- The client interpolates between updates, so a dropped packet costs smoothness but never correctness.

### The model yaw offset

The avatar model faces local **+Z** while the camera looks along **-Z**, so entity yaw has to be turned by π to make the body face its direction of travel:

```rust
pub const MODEL_YAW_OFFSET: f32 = std::f32::f32::consts::PI;
```

The same correction is applied to the local player in `render::gpu::render_frame`.

### The player avatar

The server assigns every player the same model and scale:

```rust
pub const PLAYER_MODEL: &str = "models/girl.glb";
pub const PLAYER_MODEL_SCALE: f32 = 0.565;
```

The scale takes the avatar's authored height (a ~3.19 unit figure with its feet near y=0) to the ~1.8 blocks a player occupies. It is measured from the file, and `client/src/anim/rig.rs` has a bounds test that guards it.

## Animation

`client/src/anim/` is a hand-written glTF animation system — not Bevy's.

### Building the skeleton

`client/src/anim/rig.rs` turns a glTF document into a `Skeleton`:

1. Walk the scene graph, composing every node's rest world matrix.
2. Collect the **joint set**: nodes driven by some clip, plus every ancestor needed to compose through them. For `models/girl.glb` that is 62 nodes out of 1908.
3. Rewrite each mesh primitive's vertices into its joint's local space, baking the static part of the hierarchy into them.
4. Merge primitives that share a material into one primitive.

### Why the skinning is simple

The rig is **rigid**: every primitive is a child of exactly one bone. Each vertex therefore gets a single influence (`JOINTS_0 = j, WEIGHTS_0 = 1`). Joint indices still go into the vertex so the GPU can skin, but there is no weight blend to compute and the batch stays a single draw call.

Joint index 0 is reserved as `STATIC_JOINT`: geometry that no clip can move. Its palette entry is always the identity matrix, so static geometry needs no special case in the shader — it just binds to joint 0.

### Clips

Animation clips are extracted from the glTF document and classified into a **role**, not addressed by name:

```rust
pub enum ClipRole {
    Rest,        // a held pose; the state before anything else is known
    Idle,        // standing still, looping
    Walk,        // walking
    WalkSlow,    // same topology as Walk, played slower
    Run,         // distinguished from Walk by root-motion amplitude
    TailSway,    // additive layer over whatever body clip is active
    EmoteHands,  // a long hand/body emote
    Sit, SitAlt, RestAlt,
    Unknown(u16),
}
```

The reason for roles rather than names is in the source:

> Clip names in shipped assets are frequently destroyed (mojibake, or absent), so gameplay never keys off them — it keys off the role this clip was classified as.

`is_locomotion()` selects the walking family; `is_additive()` marks the clips that must never be entered through the transition graph because they layer on top of another clip. A clip's natural speed comes from its **root motion**, not from its name.

The locomotion clip is chosen from the server's `speed` and `airborne` flags.

### Block animation states (server-authoritative)

Block entities can be posed by a small state machine instead of a locomotion
clip. The player only ever asks; the server owns the answer:

```
right-click an animated block
 → ClientMessage::UseBlock { pos, action }            (reliable, action ≤ 32 B)
 → server: resolve the block's set, map the action, validate the state
 → ServerMessage::BlockAnimState { entity_id, block, state }   (broadcast, reliable)
 → client: EntityAnimState component → drive_block_animations
```

The catalog is `assets/animations/blocks.json`, read by **both** sides
(`BlockAnimCatalog::from_json`), keyed by block name — the multiblock master
for structures like doors, since parts resolve to it (`break_cells` gives the
master cell, which is where the block entity lives):

```json
{
  "minecraft:oak_door": {
    "states": ["closed", "open"],
    "clips": {
      "closed": { "name": "door_closed", "duration": 0.2 },
      "open":   { "name": "door_open",   "duration": 0.4 }
    },
    "transitions": [
      { "from": "closed", "to": "open", "duration": 0.25 },
      { "from": "open", "to": "closed", "duration": 0.25, "mask": "lower" }
    ]
  }
}
```

`states` is the set of legal poses (`BlockAnimState`, appended-only wire enum);
`clips` names the authored clip that poses each state; `transitions` gives the
cross-fade duration and an optional `mask` (`full` | `upper` | `lower`, default
`full`) so a background branch can keep animating. Authored block clips are
looked up by **name** (`Skeleton::clip_index_by_name`), which is why block
sets carry their own `overrides` map instead of the gameplay `ClipRole` table.

Prerequisites for a block actually animating: it must have a block entity
(the state is stored on the entity record and addressed by `entity_id`), and
its render model must be a glTF whose clips carry the names used in `clips`.

## The first-person viewmodel

`PLAYER_VIEWMODEL` points at `assets/models/player/girl_base.glb` — an arms-only copy of the avatar, edited from `girl.bbmodel`. It is a **working copy**: first-person tuning (arm proportions, sleeve length, hand pose) never touches what other players see. Same rig, so the arm cut and the bone palette match.

It is drawn with **its own camera near/far range**, so the hands are never clipped by the world near plane. An additive arm swing lives on the *viewmodel instance matrix* rather than on the bone palette.
