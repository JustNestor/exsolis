# Networking

The client and server talk over [Renet](https://github.com/renet-rs/renet) — UDP with a netcode layer on top. The transport is deliberately boring: two channels, a bincode payload, and a single door in each process that every other system goes through.

## The transport

| Property | Value |
|---|---|
| Transport | UDP via Renet + `bevy_renet` netcode |
| Default address | `127.0.0.1:5000` (`SERVER_ADDR` in `common`) |
| Tick rate | 20 Hz (`TICK_RATE`) |
| Protocol id | 6 (`PROTOCOL_ID`) |
| Serialization | `bincode`, fixed-int, little-endian |

`PROTOCOL_ID` is checked during the netcode handshake. A mismatch aborts the connection rather than letting two different versions decode each other's streams — the enum discriminants are wire-visible, so a desync would be silent and nasty.

### The two channels

| Channel | Used for |
|---|---|
| `DefaultChannel::ReliableOrdered` | Everything that must arrive and must arrive in order: chunk data, block updates, entity spawns, UI screens, inventory syncs, asset manifests. |
| `DefaultChannel::Unreliable` | Everything that is superseded by the next update: player positions, entity transforms, `YourPosition`. |

The reliable channel's memory budget is raised to 64 MB on both sides, because a chunk burst is a real thing (see [world-and-blocks.md](world-and-blocks.md)).

### The single door

`server/src/net.rs` and `client/src/net.rs` are the only modules that touch `RenetServer`/`RenetClient`.

- **Inbound:** the drain reads every channel for every client and deserializes each packet into one `IncomingPacket { player_id, msg }` message. Many systems may read it; each sees every packet.
- **Outbound:** gameplay writes `OutgoingPacket { target, payload, channel }`, and one flush at the end of the frame writes them all.

This exists for a concrete reason, documented in `net.rs`:

> `receive_message` is a *consume*, not a broadcast. If two systems both drained the same channel, one would silently steal the other's packets.

Centralizing the drain means that bug cannot happen as more systems start caring about client packets — they all just read `IncomingPacket`.

### Ordering inside the frame

```
NetSet::Receive  →  GameplaySet  →  NetSet::Send
   drain_incoming     feature systems     flush_outgoing
```

Feature systems declare `.in_set(net::GameplaySet)` and `.after(...)` when they need to run after a specific router. `OutgoingPacket` written in `GameplaySet` is flushed in `NetSet::Send`.

`OutgoingPacket::raw` exists for the one case where a sender needs the encoded length *before* committing to the send — chunk streaming checks the remaining channel capacity against `RESERVE_BYTES` before it starts writing, and it does not want to serialize twice.

## The protocol

Both message enums live in `common/src/lib.rs` and are `#[non_exhaustive]`.

> **Adding a variant:** append it to the *end* of the enum. Bincode encodes a variant as its index, so inserting a variant in the middle shifts every later index and silently breaks old clients.

### Client → server

| Message | Channel | Meaning |
|---|---|---|
| `MoveAndLook { x, y, z, yaw, pitch }` | reliable | Movement input direction plus the look angle. |
| `Look { yaw, pitch }` | reliable | Look only. |
| `PlaceBlock { pos, kind, face }` | reliable | Placement request. `face` is the clicked face of the targeted block. |
| `BreakBlock { pos }` | reliable | Break request. |
| `SelectHotbar(u8)` | reliable | Hotbar selection. |
| `SetRenderDistance(u8)` | reliable | Request a different render distance (clamped server-side). |
| `UiAction { ui_id, element_id, action, data }` | reliable | A UI widget was activated. |
| `UiDrag { ui_id, source_id, target_id, data }` | reliable | A draggable widget was dropped on a droppable one. |
| `KeybindAction { id, state }` | reliable | A keybind was pressed or released. `id` is `"namespace:action"`. |
| `AssetRequest { assets }` | reliable | The client is missing these assets. |
| `AssetSyncComplete` | reliable | The client has persisted and verified the requested assets. |
| `AssetLoadComplete { paths }` | reliable | A preload finished. |
| `CreativeGive { name }` | reliable | Give one stack of a registered block. |

`UiAction` and `UiDrag` carry **untrusted** payloads. The server must treat `ui_id`, `element_id`, `action` and `data` as hostile input — they are strings the client chose.

### Server → client

| Message | Channel | Meaning |
|---|---|---|
| `YourPlayerId(PlayerId)` | reliable | The session-stable id of the receiving client. |
| `PlayerJoined` / `PlayerLeft(PlayerId)` | reliable | Peer presence. |
| `PlayerMoved { … }` | unreliable | A peer's position. |
| `YourPosition { x, y, z }` | unreliable | The authoritative position of *you*. |
| `ChunkData { coord, blocks }` | reliable | A chunk. |
| `ChunkUnload { coords }` | reliable | Drop a chunk. |
| `BlockUpdate { pos, kind }` | reliable | One block changed. |
| `SpawnEntity { … }` | reliable | Bring an entity into existence. |
| `DespawnEntity { entity_id }` | reliable | Remove an entity. |
| `EntityStateUpdate { entity_id, state }` | unreliable | An entity moved. |
| `UiShow { id, xml }` | reliable | Mount a UI screen. |
| `UiHide { id }` | reliable | Unmount a UI screen. |
| `RegisterKeybind` / `UnregisterKeybind` | reliable | Keybind registry changes. |
| `AssetManifest { assets, base_url }` | reliable | The catalog for this session. |
| `AssetDownload { url, assets }` | reliable | An authenticated download URL. |
| `AssetUpdate { url, assets }` | reliable | New or changed assets. |
| `AssetRemove { paths }` | reliable | Files removed from the catalog. |
| `AssetLoad { paths }` | reliable | Make these assets resident. |
| `AssetDownloadFailed { reason }` | reliable | The server could not build the pak. |
| `ModAction { id, payload_json }` | reliable | A mod asked clients to run a visual action. |
| `BlockList { blocks }` | reliable | Every registered block name, once per join. |
| `InventorySync { slots, selected }` | reliable | The authoritative hotbar. |

### Why position is unreliable

Position updates are sent unreliably and **repeated**. `PositionSync` in `server/src/movement.rs` keeps the last sent position and a `repeats_left` counter: when the position changes, it sends and sets `repeats_left = 3`, then keeps sending the same value three more times. A single dropped packet therefore cannot leave a client parked at a stale spot — the next tick repeats it.

The client interpolates between entity updates, so a dropped packet costs smoothness but never correctness.

### Join sequence

The order matters, and it is enforced by the asset handshake:

1. The client connects over Renet.
2. The server sends the `AssetManifest` and the asset base URL.
3. The client compares the manifest against its local cache, downloads what is missing over HTTP, verifies the checksums, and replies `AssetSyncComplete`.
4. **Only then** does the server create the player: `YourPlayerId`, `BlockList`, the peer roster, `SpawnEntity`, and the asset HTTP listener is shut down after 30 idle seconds.

The player entity is not created at connect time, because a client that cannot render the assets has nothing useful to show.

## Limits

The server and the client share the same constants (`common/src/lib.rs`), so validation never disagrees with the parser:

| Constant | Value | Guards |
|---|---|---|
| `UI_MAX_XML_BYTES` | 8 MiB | One `UiShow` payload. |
| `UI_MAX_ID_LEN` | 64 | `ui_id` / `element_id`. |
| `UI_MAX_ACTION_LEN` | 256 | `UiAction.action` / `.data`. |
| `UI_MAX_OPEN` | 32 | Screens mounted at once. |
| `MOD_ACTION_MAX_ID_BYTES` | 128 | A mod action id. |
| `MOD_ACTION_MAX_PAYLOAD_BYTES` | 4096 | A mod action JSON payload. |
| `MODEL_PATH_MAX_LEN` | 256 | A model path. |
| `GLB_MAX_BYTES` | 64 MiB | One decoded model. |

## Naming conventions

Mod-visible identifiers are namespaced `"namespace:name"`:

- Keybind ids: `"mymod:cast_spell"`.
- UI actions: `"inventory:click"`, `"mymod:open_panel"`.
- Mod visual actions: `"example:player_fly"`.

`is_valid_mod_action_id` enforces the shape: a non-empty namespace and name, both limited to alphanumerics plus `_`, `-`, `.` (and `/` allowed in the name). Namespacing is what keeps one mod's keybind from colliding with another's.
