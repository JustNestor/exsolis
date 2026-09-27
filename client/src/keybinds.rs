use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::keyboard::KeyCode;
use exsolis_common::{ClientMessage, KeybindState, ServerMessage};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct KeybindEntry {
    pub id: String,
    pub bound_key: KeyCode,
    pub default_key: KeyCode,
    pub description: String,
    pub category: String,
}

#[derive(Resource, Default)]
pub struct KeybindRegistry {
    pub binds: HashMap<String, KeybindEntry>,
    /// Local user remappings stored locally ("namespace:action" -> KeyCode)
    pub local_overrides: HashMap<String, KeyCode>,
}

pub struct KeybindPlugin;

impl Plugin for KeybindPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<KeybindRegistry>().add_systems(
            Update,
            (handle_incoming_keybind_registration, process_keybind_input),
        );
    }
}

fn parse_keycode(name: &str) -> Option<KeyCode> {
    // Basic mapping strategy or rely on serde/strum for KeyCode
    match name {
        "KeyF" => Some(KeyCode::KeyF),
        "KeyE" => Some(KeyCode::KeyE),
        "Space" => Some(KeyCode::Space),
        "ShiftLeft" => Some(KeyCode::ShiftLeft),
        _ => None,
    }
}

fn handle_incoming_keybind_registration(
    mut incoming: MessageReader<crate::net::IncomingPacket>,
    mut registry: ResMut<KeybindRegistry>,
) {
    for packet in incoming.read() {
        if let ServerMessage::RegisterKeybind {
            id,
            default_key,
            description,
            category,
        } = &packet.0
        {
            if let Some(parsed_default) = parse_keycode(default_key) {
                let bound_key = registry
                    .local_overrides
                    .get(id)
                    .copied()
                    .unwrap_or(parsed_default);
                registry.binds.insert(
                    id.clone(),
                    KeybindEntry {
                        id: id.clone(),
                        bound_key,
                        default_key: parsed_default,
                        description: description.clone(),
                        category: category.clone(),
                    },
                );
            }
        }
    }
}

fn process_keybind_input(
    keys: Res<ButtonInput<KeyCode>>,
    registry: Res<KeybindRegistry>,
    mut outgoing: MessageWriter<crate::net::OutgoingPacket>,
    // Optional: add UI input focus check here
) {
    for entry in registry.binds.values() {
        if keys.just_pressed(entry.bound_key) {
            outgoing.write(crate::net::OutgoingPacket::reliable(
                ClientMessage::KeybindAction {
                    id: entry.id.clone(),
                    state: KeybindState::Pressed,
                },
            ));
        } else if keys.just_released(entry.bound_key) {
            outgoing.write(crate::net::OutgoingPacket::reliable(
                ClientMessage::KeybindAction {
                    id: entry.id.clone(),
                    state: KeybindState::Released,
                },
            ));
        }
    }
}
