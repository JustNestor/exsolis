//! Vulkan device/swapchain bring-up only.
//!
//! Deliberately stops short of building a task graph: doing that needs
//! the actual render task (vertex/index buffers, depth image, pipeline),
//! which is `renderer.rs`'s job, not this module's. Splitting it this
//! way means `setup_vulkan` here runs once and never has to change again
//! as the render task graph evolves.

use std::sync::Arc;

use bevy_window::{PrimaryWindow, RawHandleWrapperHolder, Window};
pub use vulkano::{
    VulkanError, VulkanLibrary,
    buffer::Buffer,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    format::Format,
    image::ImageUsage,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::AllocationCreateInfo,
    pipeline::graphics::viewport::Viewport,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};
pub use vulkano_taskgraph::{
    Id,
    descriptor_set::BindlessContext,
    resource::{AccessTypes, Flight, Resources, ResourcesCreateInfo},
};

use bevy_ecs::prelude::*;
use vulkano::image::sampler::{Filter, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano_taskgraph::descriptor_set::SamplerId;

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;
pub const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

/// Everything device/swapchain bring-up produces, held as a single
/// resource. `renderer.rs`'s `setup_renderer` reads this to build the
/// actual task graph, buffers, and pipeline once it runs.
#[derive(Resource)]
pub struct VulkanContext {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub resources: Arc<Resources>,
    pub flight_id: Id<Flight>,
    pub swapchain_id: Id<Swapchain>,
    pub swapchain_format: Format,
    pub viewport: Viewport,
    /// Bindless sampler for block textures (chunk meshes, item icons):
    /// NEAREST magnification for the crisp pixel-art look up close, linear
    /// min + linear mip blend so distant terrain doesn't shimmer. Uses the
    /// mip chains `texture.rs` generates.
    pub block_sampler_id: SamplerId,
    /// Bindless sampler for smooth content: SDF text (`text.rs`) needs
    /// linear magnification or glyph edges come out blocky. Clamps instead
    /// of repeating so out-of-range UVs can't pull texels from the opposite
    /// edge (edge bleed / dark contour seams).
    pub linear_sampler_id: SamplerId,
    /// Bindless sampler for pixel-art content, and the default for glTF
    /// models: NEAREST everywhere (mag, min and mip) + ClampToEdge, no
    /// anisotropy. Nothing is averaged, so neighbouring texels of an atlas
    /// never bleed into each other and cube seams stay crisp.
    pub pixel_sampler_id: SamplerId,
    /// Set when a resize is detected or the swapchain comes back
    /// `OutOfDate` from a failed present; `renderer.rs`'s frame system
    /// clears it after recreating the swapchain + depth image.
    pub recreate_swapchain: bool,
}

pub fn setup_vulkan(
    mut commands: Commands,
    windows: Query<(Entity, &Window, &RawHandleWrapperHolder), With<PrimaryWindow>>,
) {
    let Ok((_, window, handle)) = windows.single() else {
        return;
    };
    let raw_handle = handle.0.lock().unwrap();

    let Some(raw_handle_ref) = raw_handle.as_ref().map(|x| unsafe { x.get_handle() }) else {
        return;
    };

    let library = unsafe { VulkanLibrary::new() }.unwrap();

    let required_extensions = Surface::required_extensions(&raw_handle_ref);

    bevy_log::info!("Native window handle acquired, initializing Vulkan...");

    let instance = Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: &required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..BindlessContext::required_extensions(&instance)
    };
    // Enable sampler_anisotropy for anisotropic texture filtering
    let mut device_features = BindlessContext::required_features(&instance);
    device_features.sampler_anisotropy = true;

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .into_iter()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.supported_features().contains(&device_features)
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, &raw_handle_ref)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no suitable physical device found");

    let (device, queues) = Device::new(
        &physical_device,
        &DeviceCreateInfo {
            enabled_extensions: &device_extensions,
            enabled_features: &device_features,
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.into_iter().next().unwrap();

    let resources = Resources::new(
        &device,
        &ResourcesCreateInfo {
            bindless_context: Some(&Default::default()),
            ..Default::default()
        },
    )
    .unwrap();

    let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

    // Samplers live in the bindless global set; shaders receive them as a
    // `SamplerId` push constant. Anisotropic filtering is now enabled via
    // the `sampler_anisotropy` feature.
    let (block_sampler_id, linear_sampler_id, pixel_sampler_id) = {
        let global_set = resources.bindless_context().unwrap().global_set();
        let block = global_set
            .create_sampler(&SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                anisotropy: Some(16.0),
                ..Default::default()
            })
            .unwrap();
        // Linear magnification for smooth content (mod effect textures,
        // SDF text). ClampToEdge: a Repeat sampler would blend the last
        // texel with the opposite edge on bilinear fetches just outside
        // [0, 1], drawing dark seams along contours.
        let linear = global_set
            .create_sampler(&SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                anisotropy: Some(16.0),
                ..Default::default()
            })
            .unwrap();
        // Pixel-art sampler, the glTF default: everything Nearest so no
        // interpolation (and no atlas edge bleed) can happen at all.
        let pixel = global_set
            .create_sampler(&SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                anisotropy: None,
                ..Default::default()
            })
            .unwrap();
        (block, linear, pixel)
    };

    let surface = unsafe { Surface::from_window_ref(&instance, &raw_handle_ref).unwrap() };

    let window_size = [
        window.resolution.width() as u32,
        window.resolution.height() as u32,
    ];

    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, &Default::default())
        .unwrap();

    let (swapchain_format, _) = device
        .physical_device()
        .surface_formats(&surface, &Default::default())
        .unwrap()[0];

    let swapchain_id = resources
        .create_swapchain(
            &surface,
            &SwapchainCreateInfo {
                min_image_count: surface_capabilities
                    .min_image_count
                    .max(MIN_SWAPCHAIN_IMAGES),
                image_format: swapchain_format,
                image_extent: window_size,
                present_mode: vulkano::swapchain::PresentMode::Mailbox,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap();

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [window_size[0] as f32, window_size[1] as f32],
        min_depth: 0.0,
        max_depth: 1.0,
    };

    commands.insert_resource(VulkanContext {
        instance,
        device,
        queue,
        resources,
        flight_id,
        swapchain_id,
        swapchain_format,
        viewport,
        block_sampler_id,
        linear_sampler_id,
        pixel_sampler_id,
        recreate_swapchain: false,
    });
}
