//! Client-side entity registry.
//!
//! Mirrors the server's [`EntityRegistry`](exsolis_common::EntityId) space:
//! `SpawnEntity` brings something into being, `EntityStateUpdate` moves it,
//! `DespawnEntity` removes it. Players arrive through this path, so a remote
//! avatar and a mod-spawned prop are the same kind of object here as they
//! are on the server.

use bevy_app::prelude::*;
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::*;
use bevy_log::{debug, warn};
use bevy_time::Time;
use bevy_transform::prelude::{GlobalTransform, Transform};
use exsolis_common::{
    BlockAnimMask, BlockAnimState, EntityId, EntityKind, EntityState, PlayerId, ServerMessage,
};
use glam::{EulerRot, Quat, Vec3};

use crate::anim::{Animator, Easing, Mask, Transition};
use crate::blocks::ClientAnimCatalog;
use crate::gltf::{GltfModelInstance, GltfModels};

/// Hide third-person player avatars while viewmodel/inventory work is in
/// progress. Flip back when avatars should render again.
const HIDE_PLAYER_MODELS: bool = true;
use crate::net::{GameplaySet, IncomingPacket};
use crate::player::ServerTarget;
use crate::streaming::LocalPlayerId;

/// The avatar model faces local +Z while the camera looks along -Z, so
/// entity yaw has to be turned by PI to make the body face its direction of
/// travel. The same correction is applied to the local player in
/// `render::gpu::render_frame`.
pub const MODEL_YAW_OFFSET: f32 = std::f32::consts::PI;

/// Server-assigned identity of an entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityIdComponent(pub EntityId);

/// What this entity is.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityKindComponent(pub EntityKind);

/// Uniform scale the server asked for, kept apart from `Transform` so the
/// interpolated transform never drifts the size.
#[derive(Component, Debug, Clone, Copy)]
pub struct EntityScale(pub f32);

/// For player entities: the session-stable `PlayerId` from the server, which
/// is what mod effects address players by.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerIdComponent(pub PlayerId);

/// The most recent authoritative state received for an entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct EntityNetState(pub EntityState);

/// Where the entity is heading, smoothed toward by `follow_entity_targets`.
///
/// Entity updates ride the unreliable channel, so the entity's `Transform`
/// chases this instead of snapping to whatever the last packet said.
#[derive(Component, Debug, Clone, Copy)]
pub struct EntityTarget(pub EntityState);

/// Animation state the server assigned to an entity's block — doors opening,
/// beds being slept in, and whatever else the block animation config knows.
/// Inserted from [`ServerMessage::BlockAnimState`]; the block animation
/// driver in this module poses the owning entity's `Animator` from it.
#[derive(Component, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntityAnimState {
    /// Block (master) the state belongs to, registry name.
    pub block: String,
    /// Pose the block's model should be in.
    pub state: BlockAnimState,
}

/// Entity ids keyed by the ECS entity that represents them.
#[derive(Resource, Default)]
pub struct ClientEntities {
    by_id: std::collections::HashMap<EntityId, Entity>,
}

impl ClientEntities {
    pub fn entity_of(&self, id: EntityId) -> Option<Entity> {
        self.by_id.get(&id).copied()
    }

    pub fn id_of(&self, entity: Entity) -> Option<EntityId> {
        self.by_id
            .iter()
            .find(|(_, e)| **e == entity)
            .map(|(id, _)| *id)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// How fast an entity's rendered transform chases its authoritative target.
const FOLLOW_RATE: f32 = 15.0;
/// Beyond this the entity teleports instead of interpolating; a dropped run
/// of unreliable updates should not leave it sliding across the map.
const SNAP_DISTANCE: f32 = 8.0;

pub struct EntityClientPlugin;

impl Plugin for EntityClientPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientEntities>().add_systems(
            Update,
            (
                receive_entity_messages,
                hide_local_player_body,
                follow_entity_targets,
                drive_block_animations,
            )
                .chain()
                .in_set(GameplaySet),
        );
    }
}

/// Spawn, update and despawn entities from the server's messages.
///
/// Replace semantics on spawn: re-spawning a live id is treated as an
/// update rather than creating a duplicate, because a reliable-ordered
/// channel can still replay a spawn the client already applied.
fn receive_entity_messages(
    mut commands: Commands,
    mut incoming: MessageReader<IncomingPacket>,
    mut entities: ResMut<ClientEntities>,
    local_player: Res<LocalPlayerId>,
) {
    for packet in incoming.read() {
        match &packet.0 {
            ServerMessage::SpawnEntity {
                entity_id,
                kind,
                state,
                model,
                scale,
                player_id,
            } => {
                if let Some(entity) = entities.entity_of(*entity_id) {
                    debug!("spawn_entity: {entity_id} already live, refreshing state");
                    commands
                        .entity(entity)
                        .insert((EntityNetState(*state), EntityTarget(*state)));
                    continue;
                }
                if !scale.is_finite() || *scale <= 0.0 {
                    warn!("spawn_entity {entity_id}: rejecting scale {scale}");
                    continue;
                }
                let entity = commands
                    .spawn((
                        EntityIdComponent(*entity_id),
                        EntityKindComponent(*kind),
                        EntityScale(*scale),
                        EntityNetState(*state),
                        EntityTarget(*state),
                        entity_transform(*state),
                        Animator::new(),
                    ))
                    .id();
                if let Some(player_id) = player_id {
                    commands
                        .entity(entity)
                        .insert(PlayerIdComponent(*player_id));
                }
                // Only entities with a model draw anything; a block entity
                // renders through its block, not a mesh.
                //
                // The local player's own entity deliberately gets no body
                // mesh: the camera sits at its eye position, so a body drawn
                // at its feet surrounds the viewpoint and the player ends up
                // staring at the inside of their own skull. Their avatar is
                // the first-person viewmodel instance created at startup,
                // which draws the arms only.
                // Player avatars are hidden for now (see HIDE_PLAYER_MODELS):
                // remote bodies add visual noise while the viewmodel and
                // inventory work is in progress, and the local player
                // deliberately gets no body mesh anyway (see below).
                let is_local_player = player_id.is_some() && *player_id == local_player.0;
                if let Some(path) = model
                    && !is_local_player
                    && !(*kind == EntityKind::Player && HIDE_PLAYER_MODELS)
                {
                    commands.entity(entity).insert((
                        GltfModelInstance::new(path.clone()).with_scale(*scale),
                        GlobalTransform::default(),
                    ));
                }
                entities.by_id.insert(*entity_id, entity);
            }
            ServerMessage::EntityStateUpdate { entity_id, state } => {
                let Some(entity) = entities.entity_of(*entity_id) else {
                    // An update for an entity we never saw spawn. Spawns are
                    // reliable-ordered and updates are not, so this only
                    // happens when the client joins mid-session; wait for the
                    // spawn rather than inventing an entity.
                    continue;
                };
                commands
                    .entity(entity)
                    .insert((EntityNetState(*state), EntityTarget(*state)));
            }
            ServerMessage::BlockAnimState {
                entity_id,
                block,
                state,
            } => {
                let Some(entity) = entities.entity_of(*entity_id) else {
                    // Same reasoning as EntityStateUpdate above: a state for an
                    // entity we never saw spawn only happens on mid-session
                    // joins. Wait for the spawn.
                    continue;
                };
                debug!("block anim state update: entity {entity_id} block {block} -> {:?}", state);
                commands.entity(entity).insert(EntityAnimState {
                    block: block.clone(),
                    state: *state,
                });
            }
            ServerMessage::DespawnEntity { entity_id } => {
                if let Some(entity) = entities.entity_of(*entity_id) {
                    commands.entity(entity).despawn();
                    entities.by_id.remove(entity_id);
                }
            }
            _ => {}
        }
    }
}

/// Strip the body mesh from the local player's entity.
///
/// `receive_entity_messages` only knows the local `PlayerId` if it arrived
/// before the player's own `SpawnEntity`. When it arrives later the body gets
/// a model anyway, so the check is repeated every frame: any non-viewmodel
/// model instance on the local player's entity is removed. The first-person
/// viewmodel (arms) is a separate instance and is left alone.
fn hide_local_player_body(
    mut commands: Commands,
    local_player: Res<LocalPlayerId>,
    bodies: Query<(Entity, &PlayerIdComponent, &GltfModelInstance)>,
) {
    let Some(local_id) = local_player.0 else {
        return;
    };
    for (entity, player_id, instance) in &bodies {
        if player_id.0 == local_id && !instance.first_person {
            commands.entity(entity).try_remove::<GltfModelInstance>();
        }
    }
}

fn entity_transform(state: EntityState) -> Transform {
    let mut transform = Transform::from_translation(Vec3::new(state.x, state.y, state.z));
    transform.rotation = Quat::from_euler(
        EulerRot::YXZ,
        state.yaw + MODEL_YAW_OFFSET,
        -state.pitch,
        0.0,
    );
    transform
}

/// Smooth each entity's transform toward its authoritative target and feed
/// the animator the speed the server reported.
fn follow_entity_targets(
    time: Res<Time>,
    mut query: Query<(
        &mut Transform,
        &mut EntityTarget,
        &mut Animator,
        Option<&EntityKindComponent>,
    )>,
) {
    let dt = time.delta_secs();
    for (mut transform, target, mut animator, kind) in &mut query {
        let desired = Vec3::new(target.0.x, target.0.y, target.0.z);
        let delta = desired - transform.translation;
        if delta.length_squared() > SNAP_DISTANCE * SNAP_DISTANCE {
            transform.translation = desired;
        } else {
            let alpha = 1.0 - (-FOLLOW_RATE * dt).exp();
            transform.translation += delta * alpha;
        }
        transform.rotation = Quat::from_euler(
            EulerRot::YXZ,
            target.0.yaw + MODEL_YAW_OFFSET,
            -target.0.pitch,
            0.0,
        );
        // The server's smoothed speed drives clip selection; the client
        // never derives it from its own interpolation.
        animator.speed = target.0.speed;
        animator.airborne = target.0.airborne;
        if let Some(kind) = kind
            && kind.0 != EntityKind::Player
        {
            // Non-players have no locomotion clips worth selecting; hold the
            // rest pose instead of playing a walk cycle in place.
            animator.speed = 0.0;
        }
    }
}

/// Cross-fade duration used when a block anim set declares no transition for
/// a state change. Matches the default transition the clip graph uses.
const BLOCK_TRANSITION_DURATION: f32 = 0.25;

/// Drive block-entity animations from the server-authoritative state.
///
/// The server owns *which* state a block entity is in (see
/// [`ServerMessage::BlockAnimState`]); this system resolves that state
/// through the block animation catalog to the clip that poses it and asks the
/// entity's `Animator` to play it, cross-fading along the transition the
/// block's anim set declared (falling back to a default settle).
fn drive_block_animations(
    mut query: Query<(&mut Animator, Option<&EntityAnimState>, &GltfModelInstance)>,
    mut models: ResMut<GltfModels>,
    anims: Res<ClientAnimCatalog>,
) {
    for (mut animator, state, instance) in &mut query {
        let Some(state) = state else {
            continue;
        };
        let Some(set) = anims.get(&state.block) else {
            continue;
        };
        let Some(clip) = set.clip_for(state.state) else {
            continue;
        };
        let Some(skeleton) = models.skeletons.get_mut(&instance.model_path) else {
            continue;
        };
        let Some(to_clip) = skeleton.clip_index_by_name(&clip.name) else {
            warn!(
                "gltf {}: no clip named {:?} for block anim {:?} (block {})",
                instance.model_path, clip.name, state.state, state.block
            );
            continue;
        };
        let from_clip = animator.current.clip;
        if to_clip == from_clip {
            continue;
        }
        let from_name = skeleton.clips.get(from_clip as usize).map(|c| c.name.clone());
        let from_state = from_name
            .as_deref()
            .and_then(|n| set.state_for_clip_name(n))
            .unwrap_or(state.state);
        let authored = set.transition(from_state, state.state);
        let duration = authored
            .map(|t| t.duration)
            .unwrap_or(BLOCK_TRANSITION_DURATION);
        let mask = match authored.map(|t| t.mask).unwrap_or_default() {
            BlockAnimMask::Full => Mask::Full,
            BlockAnimMask::Upper => Mask::UpperBody,
            BlockAnimMask::Lower => Mask::LowerBody,
        };
        let transition = Transition {
            duration,
            easing: Easing::CubicInOut,
            mask,
        };
        // `Animator::play` ignores its transition argument; the renderer asks
        // the skeleton for the blend along `(from_clip, to_clip)`, so the
        // authored duration/mask has to live in `overrides`.
        skeleton.overrides.insert((from_clip, to_clip), transition);
        animator.play(to_clip, transition);
    }
}
