//! Authoritative registry for everything in the world that is not a block.
//!
//! One `EntityId` per live entity, allocated monotonically and never reused
//! within a session. Clients only ever learn about entities through
//! `SpawnEntity` / `DespawnEntity` / `EntityStateUpdate`; the registry is the
//! single place that decides who exists.
//!
//! Players are entities too: `spawn_joined_player` allocates an id and
//! registers it, so first- and third-party avatars take exactly the same
//! render path. `PlayerMap` still exists because input handling wants a
//! `PlayerId <-> Entity` lookup, but it is now a view over this registry.

use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{debug, warn};
use bevy_time::Time;
use glam::Vec3;

use crate::net::{GameplaySet, OutgoingPacket};

use exsolis_common::{
    BlockAnimState, BlockId, BlockPos, BlockRegistry, EntityId, EntityKind, EntityState,
    MODEL_PATH_MAX_LEN, PlayerId, ServerMessage,
};

/// Model path used for every player avatar, and the scale that makes it the
/// right size. Declared in the lib because the client needs the same pair to
/// build the local first-person viewmodel.
pub use exsolis_server::{PLAYER_MODEL, PLAYER_MODEL_SCALE};

/// Upper bound on live entities. Bounded so a mod that spawns in a loop
/// cannot exhaust memory or flood the reliable channel; hitting it drops the
/// spawn rather than the server.
pub const MAX_ENTITIES: usize = 4096;

/// Stable server-side id stamped onto an entity's ECS entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityIdComponent(pub EntityId);

/// What this entity is, mirrored from the spawn message.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityKindComponent(pub EntityKind);

/// Model the client draws for this entity, if any.
#[derive(Component, Debug, Clone)]
pub struct EntityModel {
    pub path: String,
    pub scale: f32,
}

/// Authoritative transform, broadcast on change.
#[derive(Component, Debug, Clone, Copy)]
pub struct EntityTransform(pub EntityState);

/// A block entity: an entity whose lifetime is tied to a block position
/// rather than to physics. Breaking the block despawns the entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockEntity {
    pub pos: BlockPos,
    pub block: BlockId,
}

/// A block type whose placed form owns an entity. Resolved through the
/// registry so mod-registered container blocks work the same way.
pub fn is_block_entity(block: BlockId, registry: &BlockRegistry) -> bool {
    registry.is_block_entity(block)
}

/// One live entity.
pub struct EntityRecord {
    /// The registry key. Duplicated here rather than recovered from a
    /// component because `sync_player_entities` holds a `&mut` borrow on the
    /// record and must not need a second `Query` lookup to name itself.
    pub id: EntityId,
    pub kind: EntityKind,
    pub entity: Entity,
    pub state: EntityState,
    pub last_sent: Option<EntityState>,
    /// Where this entity lives, for block entities.
    pub block_pos: Option<BlockPos>,
    /// Server-authoritative animation state for block entities (doors open
    /// / closed, beds, ...). `None` means the at-rest pose until a use
    /// interaction changes it.
    pub anim: Option<BlockAnimState>,
}

#[derive(Resource, Default)]
pub struct EntityRegistry {
    records: HashMap<EntityId, EntityRecord>,
    /// Free-list would be tempting, but reusing an id lets a late
    /// `EntityStateUpdate` from a dropped packet animate a *different*
    /// entity. Ids are never reused; `MAX_ENTITIES` bounds the session
    /// instead.
    next_id: EntityId,
}

impl EntityRegistry {
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Every live record, in stable id order so mod-facing listings do not
    /// reshuffle between ticks.
    pub fn records(&self) -> Vec<&EntityRecord> {
        let mut out: Vec<&EntityRecord> = self.records.values().collect();
        out.sort_by_key(|r| r.id);
        out
    }

    pub fn get(&self, id: EntityId) -> Option<&EntityRecord> {
        self.records.get(&id)
    }

    pub fn entity_of(&self, id: EntityId) -> Option<Entity> {
        self.records.get(&id).map(|r| r.entity)
    }

    /// The registry id of an ECS entity, if it is a registered entity.
    /// Used by paths that hold a Bevy `Entity` (input routing, disconnect
    /// handling) and need the wire id without a second lookup structure.
    pub fn id_of(&self, entity: Entity) -> Option<EntityId> {
        self.records
            .values()
            .find(|r| r.entity == entity)
            .map(|r| r.id)
    }

    pub fn kind_of(&self, id: EntityId) -> Option<EntityKind> {
        self.records.get(&id).map(|r| r.kind)
    }

    pub fn ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.records.keys().copied()
    }

    /// The entity currently bound to a block position, if any.
    pub fn at_block(&self, pos: BlockPos) -> Option<EntityId> {
        self.records
            .iter()
            .find(|(_, r)| r.block_pos == Some(pos))
            .map(|(id, _)| *id)
    }

    /// The animation state of the block entity bound at `pos`, if any. A
    /// missing record (or a never-animated one) reads as the at-rest pose.
    pub fn anim_at_block(&self, pos: BlockPos) -> Option<(EntityId, BlockAnimState)> {
        let id = self.at_block(pos)?;
        let record = self.get(id)?;
        Some((id, record.anim.unwrap_or(BlockAnimState::Closed)))
    }

    /// Set the animation state of the block entity bound at `pos`.
    pub fn set_anim_at_block(&mut self, pos: BlockPos, state: BlockAnimState) -> Option<EntityId> {
        let id = self.at_block(pos)?;
        if let Some(record) = self.records.get_mut(&id) {
            record.anim = Some(state);
        }
        Some(id)
    }

    fn allocate(&mut self) -> Option<EntityId> {
        if self.records.len() >= MAX_ENTITIES {
            return None;
        }
        // 0 is not a valid id, so a default-initialized client-side id can
        // never alias a real entity.
        self.next_id += 1;
        Some(self.next_id)
    }
}

/// Request to bring an entity into existence. Emitted by joins, block
/// placement and the mod API; drained by [`spawn_requested`].
#[derive(Message, Debug, Clone)]
pub struct SpawnEntityRequest {
    pub kind: EntityKind,
    pub state: EntityState,
    pub model: Option<String>,
    pub scale: f32,
    pub block_pos: Option<BlockPos>,
    /// Extra components the spawner must attach (a player's inventory and
    /// input state, a block entity's container). Kept out of the Lua-facing
    /// API so a mod cannot smuggle arbitrary components through it.
    pub extra: SpawnExtra,
    /// Set for player spawns: the `PlayerId` this entity belongs to. The
    /// spawner fills `PlayerMap` from it once an `EntityId` exists, which
    /// is how input handling finds the entity for a joining client.
    pub player_token: Option<PlayerId>,
}

#[derive(Message, Debug, Clone, Copy)]
pub enum SpawnExtra {
    None,
    Player,
    BlockEntity(BlockId),
}

#[derive(Message, Debug, Clone, Copy)]
pub struct DespawnEntityRequest(pub EntityId);

pub struct EntityPlugin;

impl Plugin for EntityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EntityRegistry>()
            .add_message::<SpawnEntityRequest>()
            .add_message::<DespawnEntityRequest>()
            .add_systems(
                Update,
                (spawn_requested, despawn_requested, sync_player_entities)
                    .chain()
                    .in_set(GameplaySet),
            );
    }
}

/// Reject model paths we would not be able to serve. The client also
/// re-validates (it must, the value is untrusted) but refusing here keeps
/// the broadcast from carrying junk.
fn validate_model(path: &Option<String>) -> Option<String> {
    let path = path.as_ref()?;
    if !path.ends_with(".glb") || path.len() > MODEL_PATH_MAX_LEN {
        warn!("entity spawn: rejected model path {path:?}");
        return None;
    }
    Some(path.clone())
}

pub fn spawn_requested(
    mut requests: MessageReader<SpawnEntityRequest>,
    mut registry: ResMut<EntityRegistry>,
    mut player_map: ResMut<crate::PlayerMap>,
    mut commands: Commands,
    mut outgoing: MessageWriter<OutgoingPacket>,
    blocks: Res<crate::blocks::ServerBlocks>,
) {
    for req in requests.read() {
        let Some(id) = registry.allocate() else {
            warn!("entity cap ({MAX_ENTITIES}) reached; dropping spawn");
            continue;
        };
        let model = validate_model(&req.model);
        let scale = if req.scale.is_finite() && req.scale > 0.0 {
            req.scale
        } else {
            1.0
        };

        let entity = commands
            .spawn((
                EntityIdComponent(id),
                EntityKindComponent(req.kind),
                EntityTransform(req.state),
                EntityModel {
                    path: model.clone().unwrap_or_default(),
                    scale,
                },
            ))
            .id();

        if let Some(block) = req.extra.block() {
            commands.entity(entity).insert((
                BlockEntity {
                    pos: req.block_pos.unwrap_or(BlockPos::new(0, 0, 0)),
                    block,
                },
                crate::Container::default(),
            ));
        }
        if req.extra.is_player() {
            if let Some(token) = req.player_token {
                commands.entity(entity).insert(crate::Player(token));
                player_map.playerid_to_entity.insert(token, entity);
                player_map.entity_to_playerid.insert(entity, token);
            }
            crate::events::attach_player_components(&mut commands, entity, &blocks.0);
        }

        registry.records.insert(
            id,
            EntityRecord {
                id,
                kind: req.kind,
                entity,
                state: req.state,
                last_sent: None,
                block_pos: req.block_pos,
                anim: None,
            },
        );

        if req.kind == EntityKind::Player {
            debug!("player entity {id} for player {:?}", req.player_token);
        }
        outgoing.write(OutgoingPacket::broadcast(ServerMessage::SpawnEntity {
            entity_id: id,
            kind: req.kind,
            state: req.state,
            model,
            scale,
            player_id: req.player_token,
        }));
    }
}

impl SpawnExtra {
    fn is_player(self) -> bool {
        matches!(self, SpawnExtra::Player)
    }

    fn block(self) -> Option<BlockId> {
        match self {
            SpawnExtra::BlockEntity(block) => Some(block),
            _ => None,
        }
    }
}

pub fn despawn_requested(
    mut requests: MessageReader<DespawnEntityRequest>,
    mut registry: ResMut<EntityRegistry>,
    mut commands: Commands,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for DespawnEntityRequest(id) in requests.read() {
        // `remove` returns None for an unknown or already-despawned id; the
        // client is told to despawn unconditionally anyway because a
        // duplicated request is harmless and a missed one is not.
        let Some(record) = registry.records.remove(&id) else {
            continue;
        };
        commands.entity(record.entity).despawn();
        outgoing.write(OutgoingPacket::broadcast(ServerMessage::DespawnEntity {
            entity_id: record.id,
        }));
    }
}

/// Push transforms for player entities, and compute the authoritative
/// horizontal speed the client uses to pick a locomotion clip.
///
/// Speed is measured here rather than left to the client because position
/// updates go out on the unreliable channel: differencing two client-side
/// samples of a dropped stream yields a speed that jitters with packet loss,
/// and a jittering speed makes the walk/run blend flicker.
fn sync_player_entities(
    mut registry: ResMut<EntityRegistry>,
    positions: Query<(&crate::chunks::PlayerPosition, &crate::movement::PlayerLook)>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    time: Res<Time>,
) {
    let now = time.delta_secs();
    for record in registry.records.values_mut() {
        if record.kind != EntityKind::Player {
            continue;
        }
        let Ok((pos, look)) = positions.get(record.entity) else {
            continue;
        };
        let horizontal = Vec3::new(pos.0.x, 0.0, pos.0.z);
        let mut state = record.state;
        state.x = pos.0.x;
        state.y = pos.0.y;
        state.z = pos.0.z;
        state.yaw = look.yaw;
        state.pitch = look.pitch;
        // Exponential smoothing: the raw per-tick delta is far too spiky to
        // drive a clip blend directly. Must match `SPEED_SMOOTHING` in the
        // client's `player::drive_local_animator`, or the local player's
        // hands and everyone else's avatars would cross the walk/run
        // threshold at different speeds.
        const SPEED_SMOOTHING: f32 = 8.0;
        let instant = if now > f32::EPSILON {
            horizontal.length() / now
        } else {
            state.speed
        };
        let alpha = (1.0 - (-SPEED_SMOOTHING * now).exp()).clamp(0.0, 1.0);
        state.speed += (instant - state.speed) * alpha;
        // No gravity/ground query yet; the client treats a large downward
        // step between updates as airborne, so leave this false here rather
        // than guessing.
        state.airborne = false;

        if record.last_sent.is_some_and(|last| same_state(last, state)) {
            continue;
        }
        record.last_sent = Some(state);
        record.state = state;
        outgoing.write(OutgoingPacket::broadcast_unreliable(
            ServerMessage::EntityStateUpdate {
                entity_id: record.id,
                state,
            },
        ));
    }
}

fn same_state(a: EntityState, b: EntityState) -> bool {
    const POS_EPS: f32 = 1e-3;
    const ANG_EPS: f32 = 1e-3;
    (a.x - b.x).abs() < POS_EPS
        && (a.y - b.y).abs() < POS_EPS
        && (a.z - b.z).abs() < POS_EPS
        && (a.yaw - b.yaw).abs() < ANG_EPS
        && (a.pitch - b.pitch).abs() < ANG_EPS
        && (a.speed - b.speed).abs() < 0.05
        && a.airborne == b.airborne
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_never_zero_and_never_reused() {
        let mut reg = EntityRegistry::default();
        let a = reg.allocate().unwrap();
        let b = reg.allocate().unwrap();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert_ne!(a, b);
        // Simulate a despawn: the id is gone, and must not come back.
        assert_eq!(reg.allocate(), Some(a + 2));
    }

    #[test]
    fn allocation_stops_at_the_cap() {
        let mut reg = EntityRegistry::default();
        // The cap counts *live* entities, not ids handed out, so the
        // registry has to actually hold the records for the cap to bite.
        let mut next = 0u64;
        while reg.len() < MAX_ENTITIES {
            let id = reg.allocate().expect("under the cap");
            next = next.max(id);
            reg.records.insert(
                id,
                EntityRecord {
                    id,
                    kind: EntityKind::Prop,
                    entity: Entity::from_raw_u32(0).unwrap(),
                    state: EntityState::default(),
                    last_sent: None,
                    block_pos: None,
                    anim: None,
                },
            );
        }
        assert!(next as usize <= MAX_ENTITIES);
        assert!(reg.allocate().is_none(), "cap must refuse further spawns");
    }

    #[test]
    fn model_paths_are_validated() {
        assert!(validate_model(&Some("models/girl.glb".into())).is_some());
        assert!(validate_model(&Some("models/girl.gltf".into())).is_none());
        let too_long = format!("{}.glb", "x".repeat(MODEL_PATH_MAX_LEN));
        assert!(validate_model(&Some(too_long)).is_none());
        // `None` means "renders nothing of its own", which is legal.
        assert!(validate_model(&None).is_none());
    }

    #[test]
    fn identical_states_are_not_resent() {
        let a = EntityState {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            ..Default::default()
        };
        assert!(same_state(a, a));
        assert!(!same_state(a, EntityState { x: 1.5, ..a }));
    }
}
