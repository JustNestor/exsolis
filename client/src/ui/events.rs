//! UI input -> `OutgoingPacket(UiAction)`.

use bevy_ecs::prelude::*;
use bevy_picking::prelude::*;
use bevy_ui::Interaction;
use bevy_ui::prelude::*;
use exsolis_common::{ClientMessage, UI_MAX_ACTION_LEN};

use super::UiManager;
use super::spawn::{UiClickAction, UiElement, UiHoverStyle};
use crate::net::OutgoingPacket;

/// Reserved action: closes the screen locally, nothing is sent. Handy for
/// "X" buttons so a lagging server can't leave the player stuck in a menu.
pub const CLIENT_CLOSE: &str = "client:close";

/// Attached per widget via `.observe(...)` in spawn.rs.
pub fn on_ui_click(
    mut click: On<Pointer<Click>>,
    targets: Query<(&UiElement, &UiClickAction)>,
    mut out: MessageWriter<OutgoingPacket>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
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
