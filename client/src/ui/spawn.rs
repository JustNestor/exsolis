//! `ParsedUi` -> `bevy_ui` entity subtree.
//!
//! Every mounted screen looks like:
//!
//! ```text
//! UiRoot (full-screen wrapper, click-through, anchors the content)
//! └── XML root widget
//!     └── ...
//! ```
//!
//! `color` and `font-size` are inherited down the tree like CSS.

use bevy_color::Color;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_picking::prelude::*;
use bevy_text::{TextColor, TextFont};
use bevy_ui::prelude::*;
use bevy_ui::{GlobalZIndex, Interaction};

use crate::ui::events::{on_ui_pointer_enter, on_ui_pointer_leave};

use super::drag::{
    DragGroup, Draggable, Droppable, on_drag_drop, on_drag_end, on_drag_move, on_drag_start,
};
use super::events::on_ui_click;
use super::parse::{ParsedUi, WidgetDef, WidgetKind};
use super::style::{self, ResolvedStyle};

const DEFAULT_TEXT_COLOR: Color = Color::srgb(1.0, 1.0, 1.0);
const DEFAULT_FONT_SIZE: f32 = 16.0;
/// Above the world, below any future debug overlay.
const UI_Z: i32 = 100;

/// Marks the wrapper entity of one mounted screen.
#[derive(Component, Debug, Clone)]
pub struct UiRoot {
    pub ui_id: String,
}

/// Attached to every widget: which screen and which element it is.
#[derive(Component, Debug, Clone)]
pub struct UiElement {
    pub ui_id: String,
    pub element_id: String,
}

/// From `on-click` / `data`.
#[derive(Component, Debug, Clone)]
pub struct UiClickAction {
    pub primary: Option<String>,
    pub secondary: Option<String>,
    pub data: Option<String>,
}

#[derive(Component, Debug, Clone)]
pub struct UiPointerEnterAction {
    pub action: String,
    pub data: Option<String>,
}

/// From `on-pointer-leave` (or `on-pointer-left`) / `data`.
#[derive(Component, Debug, Clone)]
pub struct UiPointerLeaveAction {
    pub action: String,
    pub data: Option<String>,
}

/// Button background per interaction state (see `events::update_hover_styles`).
#[derive(Component, Debug, Clone, Copy)]
pub struct UiHoverStyle {
    pub normal: Color,
    pub hover: Color,
    pub pressed: Color,
}

#[derive(Clone, Copy)]
struct Inherited {
    color: Color,
    font_size: f32,
}

/// Spawns the whole screen and returns the wrapper entity.
pub fn spawn_ui(commands: &mut Commands, ui_id: &str, parsed: &ParsedUi) -> Entity {
    let mut warnings = parsed.warnings.clone();

    let wrapper_node = style::root_wrapper(
        parsed.root.attrs.get("anchor").map(String::as_str),
        &mut warnings,
    );
    let inherited = Inherited {
        color: DEFAULT_TEXT_COLOR,
        font_size: DEFAULT_FONT_SIZE,
    };
    let content = spawn_widget(commands, ui_id, &parsed.root, "0", inherited, &mut warnings);

    let wrapper = commands
        .spawn((
            Name::new(format!("ui:{ui_id}")),
            UiRoot {
                ui_id: ui_id.to_owned(),
            },
            wrapper_node,
            GlobalZIndex(UI_Z),
            // The wrapper covers the screen but must not eat world clicks.
            Pickable::IGNORE,
        ))
        .add_children(&[content])
        .id();

    for w in warnings {
        warn!("ui `{ui_id}`: {w}");
    }
    wrapper
}

fn spawn_widget(
    commands: &mut Commands,
    ui_id: &str,
    def: &WidgetDef,
    path: &str,
    inherited: Inherited,
    warnings: &mut Vec<String>,
) -> Entity {
    let ResolvedStyle {
        node,
        bg,
        border_color,
        color,
        font_size,
        hover_bg,
        pressed_bg,
        drag,
    } = style::resolve(def.kind, &def.attrs, warnings);

    let inh = Inherited {
        color: color.unwrap_or(inherited.color),
        font_size: font_size.unwrap_or(inherited.font_size),
    };

    // Auto ids are structural paths ("#0.1.2"): stable as long as the XML
    // shape is, so servers can still route un-id'd widgets if they must.
    let element_id = def.id.clone().unwrap_or_else(|| format!("#{path}"));

    let mut e = commands.spawn((
        Name::new(format!("ui:{ui_id}/{element_id}")),
        node,
        UiElement {
            ui_id: ui_id.to_owned(),
            element_id,
        },
    ));

    match def.kind {
        WidgetKind::Div => {}
        WidgetKind::Label => {
            e.insert((
                Text::new(def.text.clone().unwrap_or_default()),
                TextFont {
                    font_size: inh.font_size.into(),
                    ..Default::default()
                },
                TextColor(inh.color),
                Pickable::IGNORE,
            ));
        }
        WidgetKind::Button => {
            // `Button` requires `Interaction`, which drives hover styling.
            e.insert(Button);
            if let Some(normal) = bg {
                e.insert(UiHoverStyle {
                    normal,
                    hover: hover_bg.unwrap_or(normal),
                    pressed: pressed_bg.or(hover_bg).unwrap_or(normal),
                });
            }
        }
    }
    if let Some(c) = bg {
        e.insert(BackgroundColor(c));
    }
    if let Some(c) = border_color {
        // ASSUMPTION: Bevy >= 0.17 per-side `BorderColor`; on older
        // versions this is `BorderColor(c)`.
        e.insert(BorderColor::all(c));
    }
    if def.on_click.is_some() || def.on_secondary_click.is_some() {
        e.insert(UiClickAction {
            primary: def.on_click.clone(),
            secondary: def.on_secondary_click.clone(),
            data: def.data.clone(),
        });
        e.observe(on_ui_click);
    }
    if let Some(action) = &def.on_pointer_enter {
        e.insert(UiPointerEnterAction {
            action: action.clone(),
            data: def.data.clone(),
        });
        e.observe(on_ui_pointer_enter);
    }
    if let Some(action) = &def.on_pointer_leave {
        e.insert(UiPointerLeaveAction {
            action: action.clone(),
            data: def.data.clone(),
        });
        e.observe(on_ui_pointer_leave);
    }

    // Drag source: any widget with `draggable="true"`, regardless of kind
    // (a div slot, a label, a button — inventory slots are usually plain
    // divs with a background icon, not buttons).
    if drag.draggable {
        e.insert((
            Draggable {
                data: def.data.clone(),
                ghost_bg: drag.ghost_bg.or(bg),
            },
            DragGroup(drag.group.clone()),
        ));
        e.observe(on_drag_start);
        e.observe(on_drag_move);
        e.observe(on_drag_end);
    }
    // Drop target: separate flag so a widget can be *both* (e.g. reordering
    // a hotbar — every slot is drag source and drop target at once).
    if drag.droppable {
        e.insert(Droppable);
        // Only insert DragGroup once; draggable+droppable widgets already
        // have it from the block above.
        if !drag.draggable {
            e.insert(DragGroup(drag.group.clone()));
        }
        e.observe(on_drag_drop);
    }

    let id = e.id();

    let kids: Vec<Entity> = def
        .children
        .iter()
        .enumerate()
        .map(|(i, child)| {
            spawn_widget(
                commands,
                ui_id,
                child,
                &format!("{path}.{i}"),
                inh,
                warnings,
            )
        })
        .collect();
    if !kids.is_empty() {
        commands.entity(id).add_children(&kids);
    }
    id
}
