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

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;
pub const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

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
    let device_features = BindlessContext::required_features(&instance);

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
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

    let (device, mut queues) = Device::new(
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

    let queue = queues.next().unwrap();

    let resources = Resources::new(
        &device,
        &ResourcesCreateInfo {
            bindless_context: Some(&Default::default()),
            ..Default::default()
        },
    )
    .unwrap();

    let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

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
        recreate_swapchain: false,
    });
}
