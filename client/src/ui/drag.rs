//! Drag & drop for SDUI widgets.
//!
//! Mirrors `events.rs`'s click handling: everything here is client-side
//! *reporting*, not authority. The client never moves an item between
//! slots itself — it draws a ghost that follows the cursor, figures out
//! which `droppable` widget (if any) the cursor was over on release, and
//! sends one `ClientMessage::UiDrag { ui_id, source_id, target_id, data }`.
//! The server owns inventory state and decides whether the drag was legal;
//! see the doc comment on `UiDrag` in `exsolis_common` and the server-side
//! note at the bottom of this file.
//!
//! Widget markup:
//! ```xml
//! <div id="slot0" droppable="true" drag-group="hotbar" bg="#333a">
//!   <div id="item0" draggable="true" drag-group="hotbar" data="0" bg="#a83">Wood</div>
//! </div>
//! ```
//!
//! `drag-group` is optional; a draggable/droppable with no group matches
//! anything. Set it when a screen has multiple incompatible drop zones
//! (e.g. "hotbar" slots vs a "trash" icon that only accepts drags from a
//! different group) so `on_drag_drop` can reject the mismatch client-side
//! before ever bothering the server.
//!
//! Uses `bevy_picking`'s own `Pointer<DragStart>/Drag/DragEnd/DragDrop`
//! (the same event family `events.rs` already taps for `Pointer<Click>`),
//! so no new input plumbing is needed — `bevy_picking` handles the
//! press-move-release state machine and hands us bubbled events per entity.

use bevy_color::{Alpha, Color};
use bevy_ecs::prelude::*;
use bevy_picking::prelude::*;
use bevy_ui::prelude::*;
use bevy_ui::{GlobalZIndex, PositionType};
use bevy_window::{PrimaryWindow, Window};
use exsolis_common::{ClientMessage, UI_DRAG_MAX_ID_LEN};

use super::UiManager;
use super::spawn::UiElement;
use crate::net::OutgoingPacket;

/// Above normal UI (see `spawn::UI_Z`), so the dragged item always renders
/// on top of every screen while it's in flight.
const DRAG_GHOST_Z: i32 = 200;
/// Ghost is a fixed-size square following the cursor; real content varies
/// wildly in size and re-measuring it here isn't worth it for a drag cue.
const GHOST_SIZE_PX: f32 = 40.0;
/// Ghost is drawn slightly offset and translucent so it reads as "lifted",
/// not as a second real widget.
const GHOST_ALPHA: f32 = 0.75;

/// Marks a widget as a legal drag source. Carries what gets reported as
/// `UiDrag.data` and what color the ghost should use.
#[derive(Component, Debug, Clone)]
pub struct Draggable {
    pub data: Option<String>,
    pub ghost_bg: Option<Color>,
}

/// Marks a widget as a legal drop target. Presence alone is the flag;
/// group-matching is via the sibling `DragGroup` component.
#[derive(Component, Debug, Clone, Copy)]
pub struct Droppable;

/// `drag-group` attribute, shared by both `Draggable` and `Droppable`
/// widgets. `None` matches any group on the other side.
#[derive(Component, Debug, Clone, Default)]
pub struct DragGroup(pub Option<String>);

/// The one active drag, if any. A drag also carries which `ui_id` it
/// started in (screens can't drag-and-drop into each other) and the
/// dragged widget's `UiElement.element_id` for the outgoing packet.
#[derive(Resource, Default)]
pub struct ActiveDrag(pub Option<DragState>);

pub struct DragState {
    pub ui_id: String,
    pub source_id: String,
    pub data: Option<String>,
    pub group: Option<String>,
    pub ghost: Entity,
}

/// `Pointer<DragStart>` on any `Draggable` widget: spawn the ghost and
/// remember what's being dragged. Bubbles like clicks do, so dragging a
/// label inside a draggable div still starts the div's drag.
pub fn on_drag_start(
    mut ev: On<Pointer<DragStart>>,
    q: Query<(&UiElement, &Draggable, &DragGroup)>,
    mut active: ResMut<ActiveDrag>,
    mut commands: Commands,
) {
    let Ok((el, draggable, group)) = q.get(ev.event_target()) else {
        return;
    };
    if ev.event().button != PointerButton::Primary {
        return;
    }
    ev.propagate(false);

    // A drag already in flight (e.g. a stray DragStart while another drag
    // is being finalized) — ignore rather than leak the old ghost.
    if active.0.is_some() {
        return;
    }

    let ghost_color = draggable
        .ghost_bg
        .unwrap_or(Color::srgba(1.0, 1.0, 1.0, 0.3))
        .with_alpha(GHOST_ALPHA);

    let ghost = commands
        .spawn((
            Name::new(format!("ui-drag-ghost:{}", el.element_id)),
            Node {
                position_type: PositionType::Absolute,
                width: Val::Px(GHOST_SIZE_PX),
                height: Val::Px(GHOST_SIZE_PX),
                ..Default::default()
            },
            BackgroundColor(ghost_color),
            GlobalZIndex(DRAG_GHOST_Z),
            // Ghost must not itself be pickable, or it would intercept the
            // very DragDrop/pointer-over events we need to hit the real
            // droppable underneath it.
            Pickable::IGNORE,
        ))
        .id();

    active.0 = Some(DragState {
        ui_id: el.ui_id.clone(),
        source_id: el.element_id.clone(),
        data: draggable.data.clone(),
        group: group.0.clone(),
        ghost,
    });
}

/// `Pointer<Drag>` fires every frame the pointer moves while held. Just
/// re-centers the ghost on the cursor; all the drop-target logic happens
/// at `DragEnd`.
pub fn on_drag_move(
    ev: On<Pointer<Drag>>,
    active: Res<ActiveDrag>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut ghosts: Query<&mut Node>,
) {
    let Some(drag) = &active.0 else { return };
    let Ok(mut node) = ghosts.get_mut(drag.ghost) else {
        return;
    };
    let Ok(window) = windows.single() else { return };

    // ASSUMPTION: using window cursor position (absolute, no drift over a
    // long drag) rather than accumulating `ev.event().delta` frame over
    // frame. `Pointer<Drag>` definitely carries `delta`; it may *also*
    // carry an absolute `pointer_location.position` on this bevy_picking
    // version, which would be equivalent and slightly cheaper (skips the
    // `Window` query) — if so, swap to that. Not relying on it blind since
    // this codebase already flags exactly this kind of cross-version
    // uncertainty elsewhere (see `events.rs`'s note on `event_target()`).
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let _ = &ev; // event confirms *that* a drag is happening; position comes from the window

    node.left = Val::Px(cursor.x - GHOST_SIZE_PX * 0.5);
    node.top = Val::Px(cursor.y - GHOST_SIZE_PX * 0.5);
}

/// `Pointer<DragDrop>` fires on the widget the pointer was released over,
/// *before* that widget's own `DragEnd`/the source's `DragEnd`. This is
/// where we actually know the target, so this is the system that sends
/// the packet; `on_drag_end` below only ever cleans up.
pub fn on_drag_drop(
    mut ev: On<Pointer<DragDrop>>,
    targets: Query<(&UiElement, &DragGroup), With<Droppable>>,
    mut active: ResMut<ActiveDrag>,
    mut out: MessageWriter<OutgoingPacket>,
    mut commands: Commands,
) {
    let Ok((target_el, target_group)) = targets.get(ev.event_target()) else {
        return;
    };
    ev.propagate(false);

    let Some(drag) = active.0.take() else {
        return;
    };

    // Despawn the ghost unconditionally — whether or not the drop is
    // accepted, the drag gesture itself is over.
    commands.entity(drag.ghost).despawn();

    // Can't drop into a different screen, and dropping a widget onto
    // itself is a no-op, not a swap.
    if target_el.ui_id != drag.ui_id || target_el.element_id == drag.source_id {
        return;
    }
    // Group mismatch: both sides specified a group and they differ.
    if let (Some(a), Some(b)) = (&drag.group, &target_group.0) {
        if a != b {
            return;
        }
    }
    if drag.source_id.len() > UI_DRAG_MAX_ID_LEN || target_el.element_id.len() > UI_DRAG_MAX_ID_LEN
    {
        bevy_log::warn!("ui `{}`: drag id too long, not sent", drag.ui_id);
        return;
    }

    out.write(OutgoingPacket::reliable(ClientMessage::UiDrag {
        ui_id: drag.ui_id,
        source_id: drag.source_id,
        target_id: target_el.element_id.clone(),
        data: drag.data,
    }));
    // Observers (this fn included) run as soon as bevy_picking triggers the
    // event — earlier in the frame than `Update`. `flush_outgoing` runs
    // later, `in_set(NetSet::Send)`, so this message is safely queued
    // before the flush regardless of ordering here. Same reasoning already
    // applies to `on_ui_click` in `events.rs`.
}

/// `Pointer<DragEnd>` fires on the *source* widget regardless of whether a
/// `DragDrop` happened. If a drop already consumed the active drag,
/// `active.0` is `None` here and this is a no-op; if the pointer was
/// released over nothing droppable, this is what cleans up the ghost so
/// it doesn't linger.
pub fn on_drag_end(
    ev: On<Pointer<DragEnd>>,
    mut active: ResMut<ActiveDrag>,
    mut commands: Commands,
) {
    let _ = ev;
    if let Some(drag) = active.0.take() {
        commands.entity(drag.ghost).despawn();
    }
}

/// Screens are server state (see `mod.rs::hide_on_disconnect`): if a
/// screen gets hidden mid-drag (server sent `UiHide` while the player was
/// dragging), the ghost would otherwise outlive the widgets it referenced.
pub fn cancel_drag_if_ui_hidden(
    ui: Res<UiManager>,
    mut active: ResMut<ActiveDrag>,
    mut commands: Commands,
) {
    let Some(drag) = &active.0 else { return };
    if !ui.is_open(&drag.ui_id) {
        commands.entity(drag.ghost).despawn();
        active.0 = None;
    }
}

pub struct UiDragPlugin;

impl bevy_app::prelude::Plugin for UiDragPlugin {
    fn build(&self, app: &mut bevy_app::prelude::App) {
        // Plain `Update`, not `GameplaySet`: this only reacts to
        // `UiManager` state (screens open/closed), it doesn't touch
        // `IncomingPacket`/`OutgoingPacket` ordering, so it doesn't need
        // to sit between net receive and net send like the packet handler
        // in `mod.rs` does.
        app.init_resource::<ActiveDrag>()
            .add_systems(bevy_app::prelude::Update, cancel_drag_if_ui_hidden);
    }
}
