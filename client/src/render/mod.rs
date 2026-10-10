//! Chunk mesh building + the Vulkano task-graph pipeline that draws them.
//!
//! Split so the GPU allocator, the 3-stage mesher, the frame submit, and
//! player avatars can change independently:
//! - [`allocator`] — `offset-allocator` sub-alloc of the vertex/index buffers
//! - [`gpu`] — pipeline, buffers, task graph, draw
//! - [`meshing`] — invalidate / queue / poll / upload
//! - [`avatars`] — one local-space head, instanced by model matrix

pub mod allocator;
pub(crate) mod effects;
mod effects_gpu;
pub mod gpu;
mod meshing;
pub mod ui_pass;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::texture::TexturePlugin;

pub use allocator::{GpuMeshAllocation, MeshBufferAllocator};
pub use gpu::RenderState;

pub struct RendererPlugin;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(TexturePlugin)
            .init_resource::<crate::gltf::GltfModels>()
            .add_message::<effects::ModActionEvent>()
            .add_systems(
                Update,
                (
                    gpu::setup_renderer,
                    effects::route_mod_actions,
                    effects_gpu::prepare_gpu_effects,
                    meshing::invalidate_stale_meshes.run_if(resource_exists::<RenderState>),
                    meshing::invalidate_for_asset_change,
                    meshing::queue_chunk_meshing.run_if(resource_exists::<RenderState>),
                    meshing::collect_finished_meshes.run_if(resource_exists::<RenderState>),
                    meshing::upload_ready_chunk_meshes.run_if(resource_exists::<RenderState>),
                    meshing::cleanup_removed_chunk_meshes.run_if(resource_exists::<RenderState>),
                    crate::gltf::upload_gltf_assets.run_if(resource_exists::<RenderState>),
                    crate::gltf::sync_gltf_instances.run_if(resource_exists::<RenderState>),
                    gpu::render_frame.run_if(resource_exists::<RenderState>),
                )
                    .chain()
                    .in_set(crate::net::GameplaySet),
            );
    }
}
