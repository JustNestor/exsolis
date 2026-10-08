#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_imports
)]

use std::{
    net::{SocketAddr, UdpSocket},
    time::{Duration, SystemTime},
};

use bevy_a11y::AccessibilityPlugin;
use bevy_app::prelude::*;
use bevy_asset::AssetPlugin;
use bevy_ecs::prelude::*;
use bevy_image::ImagePlugin;
use bevy_internal::DefaultPlugins;
use bevy_log::{LogPlugin, prelude::*};
use bevy_picking::events::{Click, Pointer};
use bevy_renet::{
    RenetClient, RenetClientPlugin, client_connected,
    netcode::{
        ClientAuthentication, NetcodeClientPlugin, NetcodeClientTransport, NetcodeErrorEvent,
    },
    renet::{ChannelConfig, ConnectionConfig, DefaultChannel, SendType},
};

use bevy_window::{Window, WindowPlugin, WindowResolution};
use exsolis_common::{PROTOCOL_ID, SERVER_ADDR};

mod assets;
mod block_textures;
mod chunk;
mod chunk_registry;
mod interaction;
mod keybinds;
mod mesh;
mod net;
mod player;
mod render;
mod streaming;
mod texture;
mod ui;
mod vulkan_prelude;

use player::PlayerPlugin;
use render::*;
use streaming::StreamingPlugin;
use vulkan_prelude::*;

use crate::net::NetPlugin;

pub fn connection_config() -> ConnectionConfig {
    let mut config = ConnectionConfig::default();

    for channel in config
        .server_channels_config
        .iter_mut()
        .chain(config.client_channels_config.iter_mut())
    {
        // channel_id 2 == DefaultChannel::ReliableOrdered
        if channel.channel_id == DefaultChannel::ReliableOrdered as u8 {
            channel.max_memory_usage_bytes = 64 * 1024 * 1024;
        }
    }

    config
}

fn main() {
    App::new()
        .add_plugins((
            // AccessibilityPlugin,
            // LogPlugin::default(),
            // AssetPlugin::default(),
            // ImagePlugin::default(),
            // bevy_ui::UiPlugin,
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: bevy_window::PresentMode::Mailbox,
                    title: "Exsolis".to_string(),
                    resolution: WindowResolution::new(1920, 1080).with_scale_factor_override(1.0),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        ))
        // .add_plugins(bevy_winit::WinitPlugin::default())
        .add_plugins((RenetClientPlugin, NetcodeClientPlugin))
        .insert_resource(RenetClient::new(connection_config()))
        .add_observer(log_netcode_error)
        .add_plugins(NetPlugin)
        .add_plugins(assets::AssetClientPlugin)
        .add_plugins(ui::UiPlugin)
        .add_plugins(StreamingPlugin)
        .add_plugins(crate::interaction::InteractionPlugin)
        .add_plugins((RendererPlugin, PlayerPlugin))
        .add_plugins(keybinds::KeybindPlugin)
        .add_systems(
            Update,
            setup_vulkan.run_if(not(resource_exists::<VulkanContext>)),
        )
        .run();
}

fn log_netcode_error(error: On<NetcodeErrorEvent>) {
    warn!("netcode error: {}", *error);
}
