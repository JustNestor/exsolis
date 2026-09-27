//! UI input -> `OutgoingPacket(UiAction)`.

use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::keyboard::KeyCode;
use bevy_input::mouse::MouseButton;
use bevy_log::info;
use bevy_picking::prelude::*;
use bevy_ui::Interaction;
use bevy_ui::prelude::*;
use bevy_window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use exsolis_common::{ClientMessage, UI_MAX_ACTION_LEN};

use super::UiManager;
use super::spawn::{UiClickAction, UiElement, UiHoverStyle};
use crate::net::OutgoingPacket;
use crate::ui::spawn::UiPointerEnterAction;
use crate::ui::spawn::UiPointerLeaveAction;

/// Reserved action: closes the screen locally, nothing is sent. Handy for
/// "X" buttons so a lagging server can't leave the player stuck in a menu.
pub const CLIENT_CLOSE: &str = "client:close";

/// Attached per widget via `.observe(...)` in spawn.rs.
pub fn on_ui_click(
    mut click: On<Pointer<Click>>,
    targets: Query<(&UiElement, &UiClickAction)>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    mut out: MessageWriter<OutgoingPacket>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    info!("click");

    if let Ok(cursor) = windows.single() {
        if cursor.grab_mode != CursorGrabMode::None || !cursor.visible {
            return;
        }
    }

    info!("click 2");

    // ASSUMPTION: `event_target()` is the entity this observer is attached
    // to (the widget), not the original picked child. Adjust if 0.19 renamed it.
    let Ok((el, act)) = targets.get(click.event_target()) else {
        return;
    };
    if click.event().button != PointerButton::Primary {
        return;
    }
    click.propagate(false);

    if act.action == CLIENT_CLOSE {
        ui.hide(&mut commands, &el.ui_id);
        return;
    }
    if act.action.len() > UI_MAX_ACTION_LEN
        || act
            .data
            .as_ref()
            .is_some_and(|d| d.len() > UI_MAX_ACTION_LEN)
    {
        bevy_log::warn!("ui `{}`: action/data too long, not sent", el.ui_id);
        return;
    }

    out.write(OutgoingPacket::reliable(ClientMessage::UiAction {
        ui_id: el.ui_id.clone(),
        element_id: el.element_id.clone(),
        action: act.action.clone(),
        data: act.data.clone(),
    }));
}

/// Button hover/pressed feedback, purely client-side (no server round trip).
pub fn update_hover_styles(
    mut q: Query<(&Interaction, &UiHoverStyle, &mut BackgroundColor), Changed<Interaction>>,
) {
    for (interaction, style, mut bg) in &mut q {
        bg.0 = match interaction {
            Interaction::Pressed => style.pressed,
            Interaction::Hovered => style.hover,
            Interaction::None => style.normal,
        };
    }
}

pub fn build_action_data(
    custom_data: Option<&str>,
    keyboard: &ButtonInput<KeyCode>,
    mouse: &ButtonInput<MouseButton>,
) -> Option<String> {
    let mut parts = Vec::new();

    if keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight) {
        parts.push("ctrl");
    }
    if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
        parts.push("shift");
    }
    if keyboard.pressed(KeyCode::AltLeft) || keyboard.pressed(KeyCode::AltRight) {
        parts.push("alt");
    }
    if keyboard.pressed(KeyCode::SuperLeft) || keyboard.pressed(KeyCode::SuperRight) {
        parts.push("super");
    }

    if mouse.pressed(MouseButton::Left) {
        parts.push("left");
    }
    if mouse.pressed(MouseButton::Right) {
        parts.push("right");
    }
    if mouse.pressed(MouseButton::Middle) {
        parts.push("middle");
    }

    if let Some(d) = custom_data {
        if !d.is_empty() {
            parts.push(d);
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

pub fn on_ui_pointer_enter(
    mut enter: On<Pointer<Over>>,
    targets: Query<(&UiElement, &UiPointerEnterAction)>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut out: MessageWriter<OutgoingPacket>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    if let Ok(cursor) = windows.single() {
        if cursor.grab_mode != CursorGrabMode::None || !cursor.visible {
            return;
        }
    }

    let Ok((el, act)) = targets.get(enter.event_target()) else {
        return;
    };
    enter.propagate(false);

    if act.action == CLIENT_CLOSE {
        ui.hide(&mut commands, &el.ui_id);
        return;
    }

    let data = build_action_data(act.data.as_deref(), &keyboard, &mouse);

    if act.action.len() > UI_MAX_ACTION_LEN
        || data.as_ref().is_some_and(|d| d.len() > UI_MAX_ACTION_LEN)
    {
        bevy_log::warn!("ui `{}`: action/data too long, not sent", el.ui_id);
        return;
    }

    out.write(OutgoingPacket::reliable(ClientMessage::UiAction {
        ui_id: el.ui_id.clone(),
        element_id: el.element_id.clone(),
        action: act.action.clone(),
        data,
    }));
}

pub fn on_ui_pointer_leave(
    mut leave: On<Pointer<Out>>,
    targets: Query<(&UiElement, &UiPointerLeaveAction)>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut out: MessageWriter<OutgoingPacket>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    if let Ok(cursor) = windows.single() {
        if cursor.grab_mode != CursorGrabMode::None || !cursor.visible {
            return;
        }
    }

    let Ok((el, act)) = targets.get(leave.event_target()) else {
        return;
    };
    leave.propagate(false);

    if act.action == CLIENT_CLOSE {
        ui.hide(&mut commands, &el.ui_id);
        return;
    }

    let data = build_action_data(act.data.as_deref(), &keyboard, &mouse);

    if act.action.len() > UI_MAX_ACTION_LEN
        || data.as_ref().is_some_and(|d| d.len() > UI_MAX_ACTION_LEN)
    {
        bevy_log::warn!("ui `{}`: action/data too long, not sent", el.ui_id);
        return;
    }

    out.write(OutgoingPacket::reliable(ClientMessage::UiAction {
        ui_id: el.ui_id.clone(),
        element_id: el.element_id.clone(),
        action: act.action.clone(),
        data,
    }));
}
