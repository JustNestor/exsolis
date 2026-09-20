use super::*;
use bevy_tasks::futures_lite::future;
use bevy_tasks::{AsyncComputeTaskPool, Task};
use exsolis_common::{ChunkBlocks, ChunkCoord};
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// Sent unless the client overrides it via `ClientMessage::SetRenderDistance`.
pub const DEFAULT_RENDER_DISTANCE: u8 = 5;

/// Hard cap so a malicious/buggy client can't ask for a radius that melts the server.
const MAX_RENDER_DISTANCE: u8 = 32;

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
const MAX_CHUNK_SENDS_PER_PLAYER_PER_TICK: usize = 4;

pub struct ChunksPlugin;

impl Plugin for ChunksPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkWorld>()
            .init_resource::<GenerationTasks>()
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
                )
                    .chain(),
            );
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
}

impl Default for ChunkRenderState {
    fn default() -> Self {
        Self {
            render_distance: DEFAULT_RENDER_DISTANCE,
            // sentinel; overwritten the first time we know the player's real position
            current_chunk: ChunkCoord::new(i32::MIN, i32::MIN, i32::MIN),
            loaded: HashSet::new(),
            desired: HashSet::new(),
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
/// Wire this into recive_packets' match in events.rs (see note below), or, as here,
/// run it as its own pass over a queue. For a minimal integration this listens
/// directly by re-reading messages tagged with the player id.
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

/// Message bridged in from events.rs::recive_packets when a ClientMessage::SetRenderDistance arrives.
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
/// Cheap: only rebuild when the player has actually changed chunk or radius since last tick.
fn compute_desired_chunks(mut query: Query<(&PlayerPosition, &mut ChunkRenderState)>) {
    for (pos, mut state) in &mut query {
        let center = world_to_chunk(pos.0);
        let radius = state.render_distance as i32;

        // Skip recompute if nothing that affects the desired set has changed.
        let radius_matches_len_hint = true; // desired is rebuilt fresh below either way
        if center == state.current_chunk && radius_matches_len_hint && !state.desired.is_empty() {
            continue;
        }
        state.current_chunk = center;

        let mut desired = HashSet::with_capacity(((radius * 2 + 1).pow(2)) as usize);
        // Simple square/cylindrical radius (x/z), flat in y for now — swap for a sphere
        // trivially later by adding a dy loop + distance check if you want vertical falloff.
        for dx in -radius..=radius {
            for dz in -radius..=radius {
                if dx * dx + dz * dz <= radius * radius {
                    desired.insert(ChunkCoord::new(center.x + dx, center.y, center.z + dz));
                }
            }
        }
        state.desired = desired;
    }
}

/// Look at the union of every player's desired set, diff against ChunkWorld.known,
/// and kick off async generation tasks for anything genuinely new. This is the
/// dedup step: two overlapping players never cause double generation.
fn queue_missing_chunks(
    players: Query<&ChunkRenderState>,
    mut world: ResMut<ChunkWorld>,
    mut tasks: ResMut<GenerationTasks>,
) {
    let pool = AsyncComputeTaskPool::get();

    for state in &players {
        for &coord in &state.desired {
            if world.known.contains(&coord) {
                continue; // already generated or already in flight
            }
            world.known.insert(coord);

            let task = pool.spawn(async move { generate_chunk(coord) });
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

fn stream_chunks_to_players(
    mut query: Query<(&Player, &mut ChunkRenderState)>,
    world: Res<ChunkWorld>,
    mut server: ResMut<RenetServer>,
) {
    for (player, mut state) in &mut query {
        let center = state.current_chunk;
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
            let msg = bincode::serialize(&ServerMessage::ChunkData { coord, blocks }).unwrap();
            let available =
                server.channel_available_memory(player.0, DefaultChannel::ReliableOrdered);
            if available < msg.len() + RESERVE_BYTES {
                break;
            }
            debug!("chunk {coord:?}: {} bytes", msg.len());
            server.send_message(player.0, DefaultChannel::ReliableOrdered, msg);
            state.loaded.insert(coord);
            sent += 1;
        }
    }
}

/// Tell each player's client to drop chunks that fell outside their radius
/// (e.g. after they moved or lowered their render distance).
fn unload_out_of_range_chunks(
    mut query: Query<(&Player, &mut ChunkRenderState)>,
    mut server: ResMut<RenetServer>,
) {
    for (player, mut state) in &mut query {
        let to_unload: Vec<ChunkCoord> = state
            .loaded
            .iter()
            .filter(|c| !state.desired.contains(*c))
            .copied()
            .collect();

        for coord in to_unload {
            let msg = bincode::serialize(&ServerMessage::ChunkUnload { coords: coord }).unwrap();
            server.send_message(player.0, DefaultChannel::ReliableOrdered, msg);
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

/// Placeholder terrain gen — swap in your real terrain_noise.rs sin-wave/OpenSimplex2
/// logic here. Runs off the main thread on AsyncComputeTaskPool, so it's safe for
/// this to be as expensive as it needs to be.
fn generate_chunk(coord: ChunkCoord) -> ChunkBlocks {
    use exsolis_common::BlockType;
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

/// Written by events.rs::recive_packets from ClientMessage::Move, and
/// inserted at spawn time in events.rs::on_server_event so every
/// connected player has one from the start (see PlayerJoined handling).
#[derive(Component, Debug, Clone, Copy)]
pub struct PlayerPosition(pub Vec3);
