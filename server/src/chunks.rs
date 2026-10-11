use super::*;
use crate::net::{self, OutgoingPacket, SendTarget};
use crate::scripting::ScriptRuntime;
#[cfg(feature = "advanced_worldgen")]
use crate::terrain::WorldGen;
use bevy_renet::RenetServer;
use bevy_renet::renet::DefaultChannel;
use bevy_tasks::futures_lite::future;
use bevy_tasks::{AsyncComputeTaskPool, Task};
use exsolis_common::{BlockType, ChunkBlocks, ChunkCoord};
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// Sent unless the client overrides it via `ClientMessage::SetRenderDistance`.
pub const DEFAULT_RENDER_DISTANCE: u8 = 5;

/// Hard cap so a malicious/buggy client can't ask for a radius that melts the server.
const MAX_RENDER_DISTANCE: u8 = 32;

/// Chunk layers kept loaded above and below the player's chunk. Terrain
/// lives in y=0 while a player may be anywhere vertically, so a flat ring
/// at `center.y` used to stream nothing but empty sky. All-air / all-stone
/// chunks are 34 bytes on the wire, so extra layers are cheap.
const VERTICAL_RENDER_DISTANCE: i32 = 2;

/// Chunk packets sent to any one player per tick, across ALL their pending
/// chunks combined. Without this, a fresh join (or a big render-distance
/// bump) computes dozens of desired chunks that can all finish generating
/// around the same tick, and stream_chunks_to_players used to dump every
/// one of them onto DefaultChannel::ReliableOrdered in a single frame —
/// at render distance 5 that's 81 chunks * 32KB ~= 2.6MB in one burst,
/// which blows straight through renet's reliable channel memory budget
/// and gets the connection killed with "reliable channel memory usage
/// was exhausted". Keeping this modest turns the burst into a steady
/// trickle across many ticks instead.
///
/// Chunks are palette-compressed now (~8 KB terrain, 34 B for uniform
/// chunks), so 16/tick at 20 TPS is a few hundred KB/s at worst. The
/// `RESERVE_BYTES` check in `stream_chunks_to_players` is still the real
/// safety net against exhausting the channel.
const MAX_CHUNK_SENDS_PER_PLAYER_PER_TICK: usize = 16;

pub struct ChunksPlugin;

impl Plugin for ChunksPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkWorld>()
            .init_resource::<GenerationTasks>()
            .add_systems(
                Startup,
                init_world_gen.in_set(net::GameplaySet),
            )
            .add_systems(
                Update,
                (
                    handle_render_distance_packets,
                    init_player_render_state,
                    compute_desired_chunks,
                    queue_missing_chunks,
                    poll_generation_tasks,
                    stream_chunks_to_players,
                    unload_out_of_range_chunks,
                    #[cfg(feature = "advanced_worldgen")]
                    reload_worldgen_config,
                )
                    .chain()
                    .in_set(net::GameplaySet),
            );
    }
}

#[cfg(feature = "advanced_worldgen")]
/// Initialize the world generator after scripting is ready
fn init_world_gen(mut commands: Commands, runtime: NonSend<ScriptRuntime>) {
    let world_gen = WorldGen::from_lua(runtime.lua());
    commands.insert_resource(world_gen);
}

#[cfg(not(feature = "advanced_worldgen"))]
/// Initialize the world generator (simple generator - no Lua config needed)
fn init_world_gen(mut commands: Commands, _runtime: NonSend<ScriptRuntime>) {
    let world_gen = SimpleWorldGen::new();
    commands.insert_resource(world_gen);
}

#[cfg(feature = "advanced_worldgen")]
/// Reload worldgen config when mods are reloaded
fn reload_worldgen_config(mut commands: Commands, runtime: NonSend<ScriptRuntime>) {
    let world_gen = WorldGen::from_lua(runtime.lua());
    commands.insert_resource(world_gen);
}

/// Simple world generator (original sine-wave based) - used when advanced_worldgen feature is disabled
#[derive(bevy_ecs::resource::Resource, Clone)]
pub struct SimpleWorldGen;

impl SimpleWorldGen {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_chunk(&self, coord: ChunkCoord) -> ChunkBlocks {
        const SIZE: usize = 32;
        let mut blocks = vec![BlockType::Air; SIZE * SIZE * SIZE];

        for x in 0..SIZE {
            for z in 0..SIZE {
                let world_x = coord.x * SIZE as i32 + x as i32;
                let world_z = coord.z * SIZE as i32 + z as i32;
                let height = ((world_x as f32 * 0.1).sin() * 4.0
                    + (world_z as f32 * 0.1).cos() * 4.0
                    + 16.0) as i32;

                for y in 0..SIZE {
                    let world_y = coord.y * SIZE as i32 + y as i32;
                    let idx = x + y * SIZE + z * SIZE * SIZE;
                    blocks[idx] = if world_y < height {
                        BlockType::Stone
                    } else if world_y == height {
                        BlockType::Grass
                    } else {
                        BlockType::Air
                    };
                }
            }
        }

        ChunkBlocks::pack(&blocks)
    }
}

/// Per-player streaming state. Lives alongside `Player` on the player entity.
#[derive(Component, Debug)]
pub struct ChunkRenderState {
    pub render_distance: u8,
    pub current_chunk: ChunkCoord,
    /// Chunks this player currently has loaded on their client.
    pub loaded: HashSet<ChunkCoord>,
    /// Chunks that should be loaded on their client right now (recomputed each tick
    /// from current_chunk + render_distance).
    pub desired: HashSet<ChunkCoord>,
    /// (center chunk, radius) that `desired` was last built for. Lets
    /// `compute_desired_chunks` rebuild when the radius changes, not only
    /// when the player crosses a chunk border.
    desired_key: Option<(ChunkCoord, u8)>,
}

impl Default for ChunkRenderState {
    fn default() -> Self {
        Self {
            render_distance: DEFAULT_RENDER_DISTANCE,
            // sentinel; overwritten the first time we know the player's real position
            current_chunk: ChunkCoord::new(i32::MIN, i32::MIN, i32::MIN),
            loaded: HashSet::new(),
            desired: HashSet::new(),
            desired_key: None,
        }
    }
}

/// Server-wide chunk storage + generation bookkeeping. This is what lets multiple
/// players with overlapping render distances share work: a chunk is only ever
/// generated once, no matter how many players' radii cover it.
#[derive(Resource, Default)]
pub struct ChunkWorld {
    pub ready: HashMap<ChunkCoord, ChunkBlocks>,
    /// Coords currently being generated (task in flight) OR already ready.
    /// Used as the single source of truth for "has generation already been requested".
    pub known: HashSet<ChunkCoord>,
}

#[derive(Resource, Default)]
pub struct GenerationTasks {
    pub in_flight: HashMap<ChunkCoord, Task<ChunkBlocks>>,
}

/// Route SetRenderDistance packets into each player's ChunkRenderState.
/// Bridged in from `events.rs::dispatch_incoming` as `RenderDistanceRequested`.
fn handle_render_distance_packets(
    mut query: Query<(&Player, &mut ChunkRenderState)>,
    mut events: MessageReader<RenderDistanceRequested>,
) {
    for ev in events.read() {
        for (player, mut state) in &mut query {
            if player.0 == ev.player_id {
                state.render_distance = ev.radius.min(MAX_RENDER_DISTANCE);
            }
        }
    }
}

/// Message bridged in from events.rs::dispatch_incoming when a
/// ClientMessage::SetRenderDistance arrives.
#[derive(Message)]
pub struct RenderDistanceRequested {
    pub player_id: PlayerId,
    pub radius: u8,
}

/// Ensure every player entity has render state, defaulting to DEFAULT_RENDER_DISTANCE.
fn init_player_render_state(
    mut commands: Commands,
    query: Query<Entity, (With<Player>, Without<ChunkRenderState>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(ChunkRenderState::default());
    }
}

/// Recompute each player's desired chunk set from their current position + radius.
/// Only rebuilds when the player's chunk or render distance actually changed.
fn compute_desired_chunks(mut query: Query<(&PlayerPosition, &mut ChunkRenderState)>) {
    for (pos, mut state) in &mut query {
        let center = world_to_chunk(pos.0);
        let key = (center, state.render_distance);
        if state.desired_key == Some(key) {
            continue;
        }
        state.desired_key = Some(key);
        state.current_chunk = center;

        let radius = state.render_distance as i32;
        let columns = ((radius * 2 + 1).pow(2)) as usize;
        let layers = (VERTICAL_RENDER_DISTANCE * 2 + 1) as usize;
        let mut desired = HashSet::with_capacity(columns * layers);

        // Circular in x/z, a fixed number of layers in y.
        for dy in -VERTICAL_RENDER_DISTANCE..=VERTICAL_RENDER_DISTANCE {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    if dx * dx + dz * dz <= radius * radius {
                        desired.insert(ChunkCoord::new(
                            center.x + dx,
                            center.y + dy,
                            center.z + dz,
                        ));
                    }
                }
            }
        }
        state.desired = desired;
    }
}

/// Look at the union of every player's desired set, diff against ChunkWorld.known,
/// and kick off async generation tasks for anything genuinely new. This is the
/// dedup step: two overlapping players never cause double generation.
#[cfg(feature = "advanced_worldgen")]
fn queue_missing_chunks(
    players: Query<&ChunkRenderState>,
    mut world: ResMut<ChunkWorld>,
    mut tasks: ResMut<GenerationTasks>,
    world_gen: Res<WorldGen>,
) {
    let pool = AsyncComputeTaskPool::get();

    for state in &players {
        for &coord in &state.desired {
            if world.known.contains(&coord) {
                continue; // already generated or already in flight
            }
            world.known.insert(coord);

            let generator = world_gen.clone();
            let task = pool.spawn(async move { generator.generate_chunk(coord) });
            tasks.in_flight.insert(coord, task);
        }
    }
}

#[cfg(not(feature = "advanced_worldgen"))]
fn queue_missing_chunks(
    players: Query<&ChunkRenderState>,
    mut world: ResMut<ChunkWorld>,
    mut tasks: ResMut<GenerationTasks>,
    world_gen: Res<SimpleWorldGen>,
) {
    let pool = AsyncComputeTaskPool::get();

    for state in &players {
        for &coord in &state.desired {
            if world.known.contains(&coord) {
                continue; // already generated or already in flight
            }
            world.known.insert(coord);

            let generator = world_gen.clone();
            let task = pool.spawn(async move { generator.generate_chunk(coord) });
            tasks.in_flight.insert(coord, task);
        }
    }
}

/// Non-blocking poll of in-flight generation tasks. Only chunks that actually
/// finished this tick get moved into ChunkWorld.ready; everything else is left
/// alone and picked up on a later tick. Never blocks the server loop.
fn poll_generation_tasks(mut world: ResMut<ChunkWorld>, mut tasks: ResMut<GenerationTasks>) {
    let mut finished = Vec::new();

    for (coord, task) in tasks.in_flight.iter_mut() {
        if let Some(blocks) = future::block_on(future::poll_once(task)) {
            finished.push((*coord, blocks));
        }
    }

    for (coord, blocks) in finished {
        tasks.in_flight.remove(&coord);
        world.ready.insert(coord, blocks);
    }
}

const RESERVE_BYTES: usize = 512 * 1024; // keep headroom for other reliable traffic

/// Reads `RenetServer` only to check `channel_available_memory` (a
/// capacity query, not a send/receive) — the actual send goes through
/// `OutgoingPacket` like everywhere else, flushed centrally by `net.rs`.
fn stream_chunks_to_players(
    mut query: Query<(&Player, &mut ChunkRenderState)>,
    world: Res<ChunkWorld>,
    server: Res<RenetServer>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for (player, mut state) in &mut query {
        let center = state.current_chunk;

        // Ready-but-not-yet-sent chunks, nearest to the player first
        // (HashSet iteration order is arbitrary, which would otherwise
        // stream the horizon before the ground under their feet).
        let mut coords: Vec<ChunkCoord> = state
            .desired
            .iter()
            .filter(|c| !state.loaded.contains(*c) && world.ready.contains_key(*c))
            .copied()
            .collect();
        coords.sort_unstable_by_key(|c| {
            let (dx, dy, dz) = (c.x - center.x, c.y - center.y, c.z - center.z);
            dx * dx + dy * dy + dz * dz
        });

        let mut sent = 0;
        for coord in coords {
            if sent >= MAX_CHUNK_SENDS_PER_PLAYER_PER_TICK {
                break;
            }

            let blocks = world.ready[&coord].clone();
            let bytes = match bincode::serialize(&ServerMessage::ChunkData { coord, blocks }) {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("failed to serialize ChunkData for {coord:?}: {e}");
                    continue;
                }
            };

            let available =
                server.channel_available_memory(player.0, DefaultChannel::ReliableOrdered);
            if (available as usize) < bytes.len() + RESERVE_BYTES {
                break; // channel is full; resume next tick once acks free space
            }

            debug!("chunk {coord:?}: {} bytes", bytes.len());

            // Already serialized above to check its length against the
            // channel's remaining capacity — pass the bytes straight
            // through instead of paying to serialize the same message
            // again in net::flush_outgoing.
            outgoing.write(OutgoingPacket::raw(
                SendTarget::One(player.0),
                DefaultChannel::ReliableOrdered,
                bytes,
            ));
            state.loaded.insert(coord);
            sent += 1;
        }
    }
}

/// Tell each player's client to drop chunks that fell outside their radius
/// (e.g. after they moved or lowered their render distance).
fn unload_out_of_range_chunks(
    mut query: Query<(&Player, &mut ChunkRenderState)>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for (player, mut state) in &mut query {
        let to_unload: Vec<ChunkCoord> = state
            .loaded
            .iter()
            .filter(|c| !state.desired.contains(*c))
            .copied()
            .collect();

        for coord in to_unload {
            outgoing.write(OutgoingPacket::to(
                player.0,
                ServerMessage::ChunkUnload { coords: coord },
            ));
            state.loaded.remove(&coord);
        }
    }
}

fn world_to_chunk(pos: Vec3) -> ChunkCoord {
    const CHUNK_SIZE: f32 = 32.0; // matches your 32x32x32 cubic chunks
    ChunkCoord::new(
        (pos.x / CHUNK_SIZE).floor() as i32,
        (pos.y / CHUNK_SIZE).floor() as i32,
        (pos.z / CHUNK_SIZE).floor() as i32,
    )
}

/// Written by events.rs::dispatch_incoming from ClientMessage::Move, and
/// inserted at spawn time in events.rs::on_server_event so every
/// connected player has one from the start (see PlayerJoined handling).
#[derive(Component, Debug, Clone, Copy)]
pub struct PlayerPosition(pub Vec3);
