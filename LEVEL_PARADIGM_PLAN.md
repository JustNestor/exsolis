# Level Paradigm Plan: Multi-Level Server Architecture

## Executive Summary

Transform the single-world server into a **multi-level server** where:
- One process runs multiple independent "levels" (survival worlds, mini-games, etc.)
- Each level has its own chunk system, entity registry, mod VM, and tick rate
- Clients are **level-agnostic** — they just receive chunk/entity data; level is purely server-side
- Levels are created/destroyed dynamically via mod API
- Mods are **stateless per-tick**; long-lived data lives in ECS storage (global + per-level)
- Global assets served once; mods transition from Lua → TS/Cranelift JIT later

---

## 1. Core Architecture

### 1.1 Level Resource
Each level is a Bevy `Resource` containing its isolated simulation state:

```rust
pub struct Level {
    pub id: LevelId,
    pub name: String,
    pub level_type: LevelType,  // Survival, MiniGame, Creative, etc.
    pub tick_rate: f32,         // Configurable per level (e.g., 20, 60, 10 TPS)
    
    // Isolated simulation state
    pub chunk_world: ChunkWorld,
    pub entity_registry: EntityRegistry,
    pub script_runtime: ScriptRuntime,  // Lua now, TS JIT later
    
    // Per-level persistent storage (mods cannot hold long-lived data)
    pub storage: LevelStorage,
    
    // Runtime state
    pub players: HashSet<PlayerId>,  // Players currently in this level
    pub created_at: SystemTime,
    pub last_tick: Instant,
    pub tick_accumulator: f32,
}
```

### 1.2 Level Manager (Global Resource)
```rust
pub struct LevelManager {
    pub levels: HashMap<LevelId, Level>,
    pub default_level: LevelId,
    pub next_level_id: LevelId,
    // Global persistent storage (survives level destroy)
    pub global_storage: GlobalStorage,
}
```

### 1.3 Player-Level Association
```rust
#[derive(Component)]
pub struct PlayerLevel {
    pub level_id: LevelId,
    // Transition state (for smooth handoff)
    pub transition: Option<LevelTransition>,
}

pub struct LevelTransition {
    pub from_level: LevelId,
    pub to_level: LevelId,
    pub step: TransitionStep,
    pub entity_mapping: HashMap<EntityId, EntityId>,  // For entity transfer
}
```

---

## 2. Scheduling & Tick Rates

### Approach: Single App, Multiple Fixed-Timestep Schedules

**Why**: Bevy's `ScheduleRunnerPlugin` supports multiple schedules with different timesteps. Each level gets its own `Schedule` + `ScheduleRunnerPlugin`.

```rust
// In main.rs - create one schedule per level
let level_schedule = Schedule::new(LevelScheduleLabel(level_id));
app.add_schedule(level_schedule);
app.add_plugins(ScheduleRunnerPlugin::run_loop(
    Duration::from_secs_f64(1.0 / level.tick_rate as f64)
).with_schedule(level_schedule));
```

**Level-specific systems** run on their level's schedule:
- Chunk generation/streaming
- Entity simulation
- Mod ticking (`api.on_tick`)
- Physics/movement

**Global systems** (run on `Main` schedule at max rate or fixed rate):
- Network I/O (RenetServer drain/flush)
- Asset HTTP server
- Level manager (creation/destruction)
- Player connection/disconnection
- Cross-level transfer logic

### Schedule Labels
```rust
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct LevelScheduleLabel(pub LevelId);

// System sets per level
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct LevelGameplaySet(pub LevelId);
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct LevelModSet(pub LevelId);
```

---

## 3. Networking: Single Connection, Level-Agnostic Client

### Key Insight
> "Client doesn't even know about levels — level switch is just different chunk data"

### Packet Flow
```
Client → RenetServer → IncomingPacket (global) 
  → dispatch_incoming (global) 
  → Route to player's current level schedule via Message/Event
  → Level-specific systems process
  → OutgoingPacket (global) 
  → flush_outgoing (global)
```

### Implementation
1. **Global `dispatch_incoming`** reads `IncomingPacket`, looks up player's `PlayerLevel`, writes level-specific `Message` events
2. **Level-specific systems** read their level's messages, process, write `OutgoingPacket`
3. **Global `flush_outgoing`** sends all packets (no level awareness needed)

### Player Level Switch
```rust
fn switch_level(player_id: PlayerId, new_level_id: LevelId) {
    // 1. Despawn player entity from old level's EntityRegistry
    // 2. Spawn player entity in new level's EntityRegistry (preserve inventory, stats)
    // 3. Update PlayerLevel component
    // 4. Send new level's chunks to client (client just receives ChunkData)
    // 5. Send new level's entities (SpawnEntity)
}
```

---

## 4. Mod System: Stateless Per-Tick, ECS-Backed Storage

### 4.1 Mod API Design Principle
> "Mods disallowed to have data inside them longer than one tick; storage in ECS for long-lived data"

### 4.2 Storage API (Global + Per-Level)
```rust
// Global storage - persists across level destroy, shared by all levels
#[derive(Resource, Default)]
pub struct GlobalStorage {
    data: HashMap<String, StorageValue>,  // Mod-scoped keys
}

// Per-level storage - destroyed with level
#[derive(Resource, Default)]
pub struct LevelStorage {
    data: HashMap<String, StorageValue>,
}

// Mod-facing API (Lua/TS)
api.storage.global.get(key: string): any
api.storage.global.set(key: string, value: any)
api.storage.level.get(key: string): any
api.storage.level.set(key: string, value: any)
```

### 4.3 Mod Registration Model
```rust
// Global mods: loaded once, run on ALL level schedules
// Level mods: loaded per-level, run only on that level's schedule

pub enum ModScope {
    Global,           // Runs on every level's schedule
    Level(LevelId),   // Runs only on specific level
}

// Mod registry tracks which mods run where
#[derive(Resource)]
pub struct ModRegistry {
    global_mods: Vec<ModHandle>,
    level_mods: HashMap<LevelId, Vec<ModHandle>>,
}
```

### 4.4 Future: TS/Cranelift JIT
- Current Lua `ScriptRuntime` → swap for TS JIT runtime
- Same API surface: `install_api`, `dispatch`, `tick`, `take_requests`
- Storage API remains identical (ECS-backed)
- Mod isolation: each level gets its own JIT context or shared with sandboxing

---

## 5. Entity Transfer Between Levels

### Transfer Types
```rust
pub enum TransferPolicy {
    PlayerOnly,           // Just the player (inventory, stats, position)
    PlayerAndVehicle,     // Player + mounted entity
    PlayerAndPets,        // Player + owned entities (tagged)
    FullSnapshot,         // Serialize entire entity graph, remap IDs
}
```

### Transfer Process
```rust
fn transfer_player(player_id: PlayerId, from: LevelId, to: LevelId, policy: TransferPolicy) {
    let from_level = level_manager.levels.get_mut(&from).unwrap();
    let to_level = level_manager.levels.get_mut(&to).unwrap();
    
    // 1. Extract player data from from_level
    let player_data = from_level.entity_registry.extract_player(player_id, policy);
    
    // 2. Despawn from from_level (sends DespawnEntity to clients in that level)
    from_level.entity_registry.despawn_player(player_id);
    
    // 3. Spawn in to_level with preserved data
    let new_entity_id = to_level.entity_registry.spawn_player(player_data);
    
    // 4. Update PlayerLevel component
    // 5. Client receives new chunks + entities seamlessly
}
```

---

## 6. Level Lifecycle (Dynamic, Mod-Controlled)

### Mod API for Level Management
```lua
-- Create a new level
api.level.create({
    name = "survival_1",
    type = "survival",
    tick_rate = 20,
    mods = { "survival_core", "economy" },
    config = { seed = 12345, ... }
})

-- Destroy a level (after grace period)
api.level.destroy(level_id)

-- List levels
api.level.list() -> { {id, name, type, player_count, ...} }

-- Get/set level config
api.level.config(level_id) -> table
api.level.set_config(level_id, config)

-- Teleport player
api.level.teleport(player_id, target_level_id, policy)
```

### Auto-Cleanup Policy (Configurable per LevelType)
```rust
pub enum CleanupPolicy {
    Never,                    // Persistent (survival worlds)
    WhenEmpty(Duration),      // Mini-games: destroy after empty for N seconds
    OnLastPlayerLeave,        // Immediate cleanup
    ManualOnly,               // Admin/mod controlled only
}
```

---

## 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        BEVY APP (Single Process)                 │
├─────────────────────────────────────────────────────────────────┤
│  GLOBAL SCHEDULE (Main)                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Net: drain_incoming → dispatch_incoming → flush_outgoing  │  │
│  │ Assets: AssetHttpService, AssetCatalog refresh            │  │
│  │ LevelManager: create/destroy levels, global storage       │  │
│  │ Player connections: connect/disconnect, level assignment  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  LEVEL SCHEDULE (LevelId=1, 20 TPS)  ← ScheduleRunnerPlugin     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Chunks: stream, generate, unload                          │  │
│  │ Entities: spawn/despawn, sync, physics                    │  │
│  │ Mods: on_tick, event dispatch, request drain              │  │
│  │ Storage: LevelStorage (per-level persistent data)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  LEVEL SCHEDULE (LevelId=2, 60 TPS)  ← ScheduleRunnerPlugin     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ (Same systems, different tick rate, isolated state)       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  GLOBAL RESOURCES                                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ RenetServer (single)                                      │  │
│  │ AssetCatalog, AssetHttpService (shared)                   │  │
│  │ LevelManager, GlobalStorage                               │  │
│  │ ModRegistry (global mods loaded once)                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Phases

### Phase 1: Core Level Infrastructure (Week 1-2)
- [ ] `Level` and `LevelManager` resources
- [ ] `PlayerLevel` component + transition logic
- [ ] Dynamic schedule creation/destruction per level
- [ ] Global `dispatch_incoming` routes to level-specific messages
- [ ] Level-aware `ChunkWorld`, `EntityRegistry`, `ScriptRuntime` as fields of `Level`

### Phase 2: Storage & Mod API (Week 2-3)
- [ ] `GlobalStorage` and `LevelStorage` resources + ECS integration
- [ ] Mod API: `api.storage.global`, `api.storage.level`
- [ ] Mod registration: `Global` vs `Level(LevelId)` scope
- [ ] Per-level mod loading/unloading (dynamic via mod API)

### Phase 3: Level Lifecycle & Transitions (Week 3-4)
- [ ] Mod API: `api.level.create/destroy/list/teleport`
- [ ] Player transfer logic (preserve inventory, stats)
- [ ] Entity transfer policies (PlayerOnly, PlayerAndPets, etc.)
- [ ] Cleanup policies per LevelType

### Phase 4: Polish & TS JIT Prep (Week 4-5)
- [ ] Refactor `ScriptRuntime` trait for Lua/TS swap
- [ ] Benchmark multi-level performance
- [ ] Config system for level definitions
- [ ] Admin/mod commands for level management

---

## 9. Key Files to Modify/Create

### New Files
```
server/src/level/
├── mod.rs              // Level, LevelManager, LevelType, LevelStorage
├── manager.rs          // Level creation/destruction, scheduling
├── transfer.rs         // Player/entity transfer logic
├── storage.rs          // GlobalStorage, LevelStorage, StorageValue
└── mod_api.rs          // api.level.*, api.storage.*

server/src/scheduling/
├── mod.rs              // LevelScheduleLabel, LevelGameplaySet, setup
└── runner.rs           // Custom schedule runner if needed
```

### Modified Files
```
server/src/main.rs           // Initialize LevelManager, create default level
server/src/events.rs         // dispatch_incoming → route to level messages
server/src/net.rs            // No changes (level-agnostic)
server/src/chunks.rs         // Systems take Level as parameter/resource
server/src/entities.rs       // EntityRegistry moves into Level
server/src/scripting.rs      // ScriptRuntime moves into Level; global mod support
server/src/assets.rs         // No changes (global)
common/src/lib.rs            // Add LevelId, LevelType, TransferPolicy
mods/                        // Example mods using new level API
```

---

## 10. Open Questions for Final Plan

1. **Default Level**: Created at startup from config, or first `api.level.create`?
2. **Level ID Allocation**: Sequential u32, or UUID for persistence across restarts?
3. **Chunk Generation**: Same generator for all levels, or per-level generator config?
4. **Global Mods Communication**: Do global mods need cross-level messaging (e.g., global chat, economy)?
5. **Player Spawn Position**: Per-level spawn config, or global?
6. **Resource Limits**: Max levels? Max players per level? Total entity cap across levels?

---

## 11. Migration Strategy

1. **Wrap existing globals** into `Level` struct (ChunkWorld, EntityRegistry, ScriptRuntime)
2. **Create `LevelManager`** with single "default" level containing current state
3. **Update systems** to read `Level` from `LevelManager` via `PlayerLevel` component
4. **Add dynamic schedule creation** for new levels
5. **Expose mod API** for level management
6. **Test** with 2 levels (survival + mini-game) simultaneously

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Schedule explosion (many levels) | Pool schedules; reuse for same tick rate; max level limit |
| Memory bloat (many ChunkWorlds) | Unload chunks for empty levels; compress/persist to disk |
| Mod API breaking changes | Version mod API; Lua/TS abstraction layer |
| Network congestion | Per-level chunk send limits already exist; global backpressure |
| Entity ID collision across levels | EntityId allocated per-level; client sees level-scoped IDs (or global remap) |

---

*This plan is ready for review. Once approved, I'll create detailed implementation tasks per phase.*