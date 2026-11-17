// src/interaction.rs

use bevy_app::prelude::*;
use bevy_ecs::message::{MessageReader, MessageWriter};
use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::keyboard::KeyCode;
use bevy_input::mouse::{MouseButton, MouseScrollUnit, MouseWheel};
use bevy_log::{debug, info, warn};
use bevy_time::{Time, Timer, TimerMode};
use bevy_transform::prelude::Transform;
use exsolis_common::{AIR_ID, BlockId, BlockPos, ClientMessage, HOTBAR_SIZE};
use glam::{IVec3, Vec3};

use crate::blocks::{ClientAnimCatalog, ClientBlocks};
use crate::chunk::{CHUNK_SIZE_I32, Chunk};
use crate::chunk_registry::ChunkRegistry;
use crate::net::{GameplaySet, NetConnected, OutgoingPacket};
use crate::player::{FlyCamera, LocalPlayer};
use crate::ui::gameplay_input_allowed;

const REACH: f32 = 8.0;
const RAYCAST_STEP_LIMIT: i32 = 64;

#[derive(Resource)]
pub struct InteractionCooldown {
    pub break_timer: Timer,
    pub place_timer: Timer,
}

impl Default for InteractionCooldown {
    fn default() -> Self {
        Self {
            break_timer: Timer::from_seconds(0.2, TimerMode::Once),
            place_timer: Timer::from_seconds(0.2, TimerMode::Once),
        }
    }
}

#[derive(Component, Default)]
pub struct BlockTargeting {
    pub target: Option<TargetedBlock>,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetedBlock {
    pub pos: IVec3,
    pub place_pos: IVec3,
}

#[derive(Resource, Debug)]
pub struct ClientInventory {
    pub slots: [Option<(BlockId, u8)>; HOTBAR_SIZE],
    pub selected: u8,
}

impl Default for ClientInventory {
    // Starts empty on purpose: contents are server state (join
    // InventorySync fills them). A seeded local default would show blocks
    // the server never gave, including while offline.
    fn default() -> Self {
        Self {
            slots: [None; HOTBAR_SIZE],
            selected: 0,
        }
    }
}

impl ClientInventory {
    pub fn selected_item(&self) -> Option<BlockId> {
        self.slots[self.selected as usize].map(|(item, _)| item)
    }
}

pub struct InteractionPlugin;

/// Swing the first-person arms on use.
///
/// Separate from the handlers above so the animation plays whether or not
/// the action itself succeeded — clicking a block you cannot reach should
/// still move your hand.
fn drive_use_swing(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut swing: ResMut<crate::anim::UseSwing>,
) {
    if mouse.just_pressed(MouseButton::Left) || mouse.just_pressed(MouseButton::Right) {
        swing.trigger();
    }
    swing.advance(time.delta_secs());
}

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientInventory>()
            .init_resource::<InteractionCooldown>()
            .init_resource::<crate::anim::UseSwing>()
            .add_systems(
                Update,
                (
                    ensure_targeting_component,
                    (
                        select_hotbar_by_number_input,
                        select_hotbar_by_wheel,
                        update_targeting,
                        handle_break_input,
                        handle_place_input,
                    )
                        .run_if(gameplay_input_allowed),
                    drive_use_swing.run_if(gameplay_input_allowed),
                )
                    .in_set(GameplaySet),
            );
    }
}

fn ensure_targeting_component(
    mut commands: Commands,
    query: Query<Entity, (With<LocalPlayer>, Without<BlockTargeting>)>,
) {
    for entity in &query {
        info!(
            "[Targeting] Attaching BlockTargeting component to LocalPlayer entity {:?}",
            entity
        );
        commands.entity(entity).insert(BlockTargeting::default());
    }
}

const DIGIT_KEYS: [KeyCode; HOTBAR_SIZE] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];

fn select_hotbar_by_number_input(
    keys: Res<ButtonInput<KeyCode>>,
    connected: Res<NetConnected>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    let Some(index) = DIGIT_KEYS.iter().position(|k| keys.just_pressed(*k)) else {
        return;
    };
    if !connected.get() {
        warn!("[SelectHotbar] Key digit press detected, but not connected");
        return;
    }
    info!("[SelectHotbar] Sending SelectHotbar({}) packet", index);
    outgoing.write(OutgoingPacket::reliable(ClientMessage::SelectHotbar(
        index as u8,
    )));
}

/// Mouse wheel cycles the hotbar selection (same gates as place/break via
/// `gameplay_input_allowed`). Scroll up steps back, scroll down steps
/// forward; the local slot updates immediately and the server is told via
/// `SelectHotbar` (authoritative state comes back through `InventorySync`).
fn select_hotbar_by_wheel(
    mut wheels: MessageReader<MouseWheel>,
    connected: Res<NetConnected>,
    mut inventory: ResMut<ClientInventory>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    let mut steps = 0i32;
    for ev in wheels.read() {
        steps += match ev.unit {
            MouseScrollUnit::Line => ev.y.signum() as i32,
            MouseScrollUnit::Pixel => (ev.y / 50.0).signum() as i32,
        };
    }
    if steps == 0 {
        return;
    }
    if !connected.get() {
        warn!("[SelectHotbar] Mouse wheel scroll detected, but not connected");
        return;
    }
    let n = HOTBAR_SIZE as i32;
    // Positive `steps` (scroll up) moves toward slot 0, negative down.
    let next = (inventory.selected as i32 - steps).rem_euclid(n) as u8;
    inventory.selected = next;
    debug!("[SelectHotbar] Wheel scrolled {steps}, selecting slot {next}");
    outgoing.write(OutgoingPacket::reliable(ClientMessage::SelectHotbar(next)));
}

fn read_block(registry: &ChunkRegistry, chunks: &Query<&Chunk>, world: IVec3) -> Option<BlockId> {
    let coord = exsolis_common::ChunkCoord::new(
        world.x.div_euclid(CHUNK_SIZE_I32),
        world.y.div_euclid(CHUNK_SIZE_I32),
        world.z.div_euclid(CHUNK_SIZE_I32),
    );
    let Some(entity) = registry.entity_at(coord) else {
        return None;
    };
    let Ok(chunk) = chunks.get(entity) else {
        warn!(
            "[read_block] Chunk entity {:?} registered at {:?}, but missing Chunk component",
            entity, coord
        );
        return None;
    };
    let lx = world.x.rem_euclid(CHUNK_SIZE_I32) as usize;
    let ly = world.y.rem_euclid(CHUNK_SIZE_I32) as usize;
    let lz = world.z.rem_euclid(CHUNK_SIZE_I32) as usize;
    Some(chunk.get(lx, ly, lz))
}

fn raycast_voxels(
    registry: &ChunkRegistry,
    chunks: &Query<&Chunk>,
    origin: Vec3,
    dir: Vec3,
) -> Option<TargetedBlock> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut cell = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );
    let mut prev_cell = cell;

    let step = IVec3::new(
        dir.x.signum() as i32,
        dir.y.signum() as i32,
        dir.z.signum() as i32,
    );

    let t_delta = Vec3::new(
        if dir.x != 0.0 {
            (1.0 / dir.x).abs()
        } else {
            f32::INFINITY
        },
        if dir.y != 0.0 {
            (1.0 / dir.y).abs()
        } else {
            f32::INFINITY
        },
        if dir.z != 0.0 {
            (1.0 / dir.z).abs()
        } else {
            f32::INFINITY
        },
    );

    let next_boundary = |o: f32, c: i32, s: i32| -> f32 {
        if s > 0 {
            (c + 1) as f32 - o
        } else {
            o - c as f32
        }
    };
    let mut t_max = Vec3::new(
        if dir.x != 0.0 {
            next_boundary(origin.x, cell.x, step.x) / dir.x.abs()
        } else {
            f32::INFINITY
        },
        if dir.y != 0.0 {
            next_boundary(origin.y, cell.y, step.y) / dir.y.abs()
        } else {
            f32::INFINITY
        },
        if dir.z != 0.0 {
            next_boundary(origin.z, cell.z, step.z) / dir.z.abs()
        } else {
            f32::INFINITY
        },
    );

    for _ in 0..RAYCAST_STEP_LIMIT {
        let block = read_block(registry, chunks, cell)?;
        if block != AIR_ID {
            return Some(TargetedBlock {
                pos: cell,
                place_pos: prev_cell,
            });
        }
        prev_cell = cell;

        if t_max.x < t_max.y && t_max.x < t_max.z {
            if t_max.x > REACH {
                return None;
            }
            cell.x += step.x;
            t_max.x += t_delta.x;
        } else if t_max.y < t_max.z {
            if t_max.y > REACH {
                return None;
            }
            cell.y += step.y;
            t_max.y += t_delta.y;
        } else {
            if t_max.z > REACH {
                return None;
            }
            cell.z += step.z;
            t_max.z += t_delta.z;
        }
    }
    None
}

fn update_targeting(
    registry: Res<ChunkRegistry>,
    chunks: Query<&Chunk>,
    mut query: Query<(&Transform, &FlyCamera, &mut BlockTargeting), With<LocalPlayer>>,
) {
    for (transform, cam, mut targeting) in &mut query {
        let origin = cam.eye_position(transform);
        // The look direction has the full pitch, which lives on the camera,
        // not on the body's yaw-only transform.
        let dir = cam.forward();
        targeting.target = raycast_voxels(&registry, &chunks, origin, dir);
    }
}

fn handle_break_input(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cooldown: ResMut<InteractionCooldown>,
    connected: Res<NetConnected>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    query: Query<&BlockTargeting, With<LocalPlayer>>,
) {
    if !mouse.pressed(MouseButton::Left) {
        cooldown.break_timer.reset();
        return;
    }

    cooldown.break_timer.tick(time.delta());

    if !mouse.just_pressed(MouseButton::Left) && !cooldown.break_timer.is_finished() {
        return;
    }

    cooldown.break_timer.reset();

    if !connected.get() {
        warn!("[BreakBlock] Left click pressed, but not connected!");
        return;
    }

    if query.is_empty() {
        warn!(
            "[BreakBlock] Left click pressed, but query for LocalPlayer + BlockTargeting returned NO entities!"
        );
        return;
    }

    for targeting in &query {
        let Some(target) = targeting.target else {
            warn!(
                "[BreakBlock] Left click pressed, but BlockTargeting.target is None (raycast hit air, unloaded chunk, or reached max distance)"
            );
            continue;
        };

        info!(
            "[BreakBlock] Sending BreakBlock packet for pos {:?}",
            target.pos
        );
        outgoing.write(OutgoingPacket::reliable(ClientMessage::BreakBlock {
            pos: BlockPos::new(target.pos.x, target.pos.y, target.pos.z),
        }));
    }
}

fn handle_place_input(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cooldown: ResMut<InteractionCooldown>,
    connected: Res<NetConnected>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    inventory: Res<ClientInventory>,
    blocks: Option<Res<ClientBlocks>>,
    anims: Res<ClientAnimCatalog>,
    registry: Res<ChunkRegistry>,
    chunks: Query<&Chunk>,
    query: Query<&BlockTargeting, With<LocalPlayer>>,
) {
    if !mouse.pressed(MouseButton::Right) {
        cooldown.place_timer.reset();
        return;
    }

    cooldown.place_timer.tick(time.delta());

    if !mouse.just_pressed(MouseButton::Right) && !cooldown.place_timer.is_finished() {
        return;
    }

    cooldown.place_timer.reset();

    if !connected.get() {
        warn!("[PlaceBlock] Right click pressed, but not connected!");
        return;
    }

    if query.is_empty() {
        warn!(
            "[PlaceBlock] Right click pressed, but query for LocalPlayer + BlockTargeting returned NO entities!"
        );
        return;
    }

    for targeting in &query {
        let Some(target) = targeting.target else {
            warn!(
                "[PlaceBlock] Right click pressed, but BlockTargeting.target is None (raycast hit air, unloaded chunk, or reached max distance)"
            );
            continue;
        };

        // A right-click that lands on an existing animating block (a door, a
        // bed, ...) is a *use*, not a placement: drive the block's animation
        // state through `UseBlock` instead of placing one. Only on the initial
        // press — holding the button must not keep toggling the block.
        let use_action = read_block(&registry, &chunks, target.pos).and_then(|id| {
            let blocks = blocks.as_ref()?;
            let mut name = blocks.0.def(id)?.name.clone();
            // Multiblock parts resolve to their master's animation set.
            if let Some(master) = blocks.0.master_for_part(&name)
                && let Some(master_name) = blocks.0.def(master)
            {
                name = master_name.name.clone();
            }
            anims
                .get(&name)
                .filter(|s| s.states.len() > 1)
                .map(|_| "toggle".to_string())
        });
        if let Some(action) = use_action {
            if !mouse.just_pressed(MouseButton::Right) {
                continue;
            }
            info!(
                "[UseBlock] Sending UseBlock packet for pos {:?} action {action}",
                target.pos
            );
            outgoing.write(OutgoingPacket::reliable(ClientMessage::UseBlock {
                pos: BlockPos::new(target.pos.x, target.pos.y, target.pos.z),
                action,
            }));
            continue;
        }

        // An empty hand can still *use* a block (above), just not place one.
        let Some(item) = inventory.selected_item() else {
            warn!(
                "[PlaceBlock] Right click pressed, but ClientInventory selected slot {} is empty/None! (slots: {:?})",
                inventory.selected, inventory.slots
            );
            continue;
        };

        info!(
            "[PlaceBlock] Sending PlaceBlock packet for item {:?} at place_pos {:?}",
            item, target.place_pos
        );
        let kind = blocks
            .as_ref()
            .and_then(|b| b.0.def(item).map(|d| d.name.clone()))
            .unwrap_or_else(|| "exsolis:stone".to_string());
        // Clicked face = placement cell minus the targeted solid cell.
        let delta = target.place_pos - target.pos;
        let face = if delta.y > 0 {
            exsolis_common::Face::Top
        } else if delta.y < 0 {
            exsolis_common::Face::Bottom
        } else if delta.x > 0 {
            exsolis_common::Face::Right
        } else if delta.x < 0 {
            exsolis_common::Face::Left
        } else if delta.z > 0 {
            exsolis_common::Face::Front
        } else {
            exsolis_common::Face::Back
        };
        outgoing.write(OutgoingPacket::reliable(ClientMessage::PlaceBlock {
            pos: BlockPos::new(target.place_pos.x, target.place_pos.y, target.place_pos.z),
            kind,
            face,
        }));
    }
}
