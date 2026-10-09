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
use bevy_ecs::observer::On;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_math::Vec2;
use bevy_picking::events::{Pointer, Scroll};
use bevy_picking::prelude::*;
use bevy_text::{TextColor, TextFont};
use bevy_ui::prelude::*;
use bevy_ui::{GlobalZIndex, Interaction};

use crate::ui::events::{on_ui_pointer_enter, on_ui_pointer_leave};

use super::drag::{
    DragGroup, Draggable, Droppable, on_drag_drop, on_drag_end, on_drag_move, on_drag_start,
};
use super::events::on_ui_click;
use super::img_assets;
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

/// On `<img>` widgets whose file was found: which `assets/ui/` file to draw.
/// `render_bridge` uploads it to the GPU once per distinct `src`.
#[derive(Component, Debug, Clone)]
pub struct UiImageSrc {
    pub src: String,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct UiScrollArea;

#[derive(Component, Debug, Clone, Copy)]
pub struct UiScrollbarThumb {
    pub area: Entity,
    pub horizontal: bool,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct UiScrollbarRail {
    pub area: Entity,
}

pub fn update_scrollbar_thumbs(
    mut bars: Query<(
        &mut Node,
        Option<&UiScrollbarThumb>,
        Option<&UiScrollbarRail>,
    )>,
    areas: Query<(&ComputedNode, &bevy_ui::ScrollPosition)>,
) {
    for (mut node, thumb, rail) in &mut bars {
        let Some(area) = thumb
            .map(|thumb| thumb.area)
            .or_else(|| rail.map(|rail| rail.area))
        else {
            continue;
        };
        let Ok((computed, position)) = areas.get(area) else {
            continue;
        };
        if rail.is_some() {
            // The parent scroll offset moves every child; offset the rail by
            // the same amount so it stays pinned to the viewport edge.
            node.left = Val::Px(position.x);
            node.top = Val::Px(position.y);
            continue;
        }
        let Some(thumb) = thumb else { continue };
        let visible_size = computed.size() * computed.inverse_scale_factor;
        let content = computed.content_size() * computed.inverse_scale_factor;
        let (visible, content, offset) = if thumb.horizontal {
            (visible_size.x, content.x, position.x)
        } else {
            (visible_size.y, content.y, position.y)
        };
        let range = (content - visible).max(0.0);
        let ratio = if content > 0.0 {
            (visible / content).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let track = visible;
        let length = (track * ratio).max(18.0).min(track);
        let travel = (track - length).max(0.0);
        let offset = if range > 0.0 {
            offset / range * travel
        } else {
            0.0
        };
        if thumb.horizontal {
            node.width = Val::Px(length);
            node.left = Val::Px(offset);
        } else {
            node.height = Val::Px(length);
            node.top = Val::Px(offset);
        }
    }
}

fn spawn_scrollbar(
    commands: &mut Commands,
    area: Entity,
    horizontal: bool,
    thumb_color: Color,
    track_color: Color,
) -> Entity {
    let rail = if horizontal {
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Percent(100.0),
            height: Val::Px(6.0),
            ..Default::default()
        }
    } else {
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            width: Val::Px(6.0),
            height: Val::Percent(100.0),
            ..Default::default()
        }
    };
    let rail_id = commands
        .spawn((
            rail,
            BackgroundColor(track_color),
            ZIndex(100),
            UiScrollbarRail { area },
        ))
        .id();
    let thumb = if horizontal {
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            width: Val::Px(24.0),
            height: Val::Percent(100.0),
            ..Default::default()
        }
    } else {
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            width: Val::Percent(100.0),
            height: Val::Px(24.0),
            ..Default::default()
        }
    };
    let thumb_id = commands
        .spawn((
            thumb,
            BackgroundColor(thumb_color),
            ZIndex(101),
            UiScrollbarThumb { area, horizontal },
        ))
        .id();
    commands.entity(rail_id).add_child(thumb_id);
    rail_id
}

pub fn on_scroll(
    mut event: On<Pointer<Scroll>>,
    mut areas: Query<(&Node, &ComputedNode, &mut ScrollPosition), With<UiScrollArea>>,
) {
    if let Ok((node, computed, mut position)) = areas.get_mut(event.entity) {
        event.propagate(false);
        let factor = match event.event().unit {
            bevy_input::mouse::MouseScrollUnit::Line => {
                bevy_input::mouse::MouseScrollUnit::SCROLL_UNIT_CONVERSION_FACTOR
            }
            bevy_input::mouse::MouseScrollUnit::Pixel => 1.0,
        };
        let visible = computed.size() * computed.inverse_scale_factor;
        let content = computed.content_size() * computed.inverse_scale_factor;
        let range = (content - visible).max(Vec2::ZERO);
        if node.overflow.x == OverflowAxis::Scroll {
            position.x = (position.x - event.event().x * factor).clamp(0.0, range.x);
        }
        if node.overflow.y == OverflowAxis::Scroll {
            position.y = (position.y - event.event().y * factor).clamp(0.0, range.y);
        }
    }
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
        mut node,
        bg,
        border_color,
        color,
        font_size,
        hover_bg,
        pressed_bg,
        drag,
        z_index,
        background_image,
        scrollbar,
        scrollbar_color,
        scrollbar_track,
    } = style::resolve(def.kind, &def.attrs, warnings);
    let is_scroll_area =
        node.overflow.x == OverflowAxis::Scroll || node.overflow.y == OverflowAxis::Scroll;
    let scrolls_x = node.overflow.x == OverflowAxis::Scroll;
    let scrolls_y = node.overflow.y == OverflowAxis::Scroll;

    let inh = Inherited {
        color: color.unwrap_or(inherited.color),
        font_size: font_size.unwrap_or(inherited.font_size),
    };

    // `<img>`: size is resolved before spawning (`node` is moved into the
    // spawn below). Unreadable file => `image_src` is None and the widget
    // falls back to its `alt` text.
    let image_src: Option<&str> = if def.kind == WidgetKind::Image {
        match def.src.as_deref().map(|s| (s, img_assets::probe_size(s))) {
            Some((s, Some((w, h)))) => {
                let (w, h) = (w as f32, h as f32);
                node.aspect_ratio = Some(w / h);
                if node.width == Val::Auto && node.height == Val::Auto {
                    node.width = Val::Px(w);
                    node.height = Val::Px(h);
                } else {
                    // Don't let `align: stretch` fight the aspect ratio.
                    node.align_self = AlignSelf::FlexStart;
                }
                Some(s)
            }
            _ => {
                warnings.push(format!(
                    "<img>: cannot load {:?} from assets/ui; showing alt text",
                    def.src
                ));
                None
            }
        }
    } else {
        None
    };
    let background_image = background_image.filter(|src| {
        if super::parse::is_valid_img_src(src) && img_assets::probe_size(src).is_some() {
            true
        } else {
            warnings.push(format!("invalid or unreadable `background-image` {src:?}"));
            false
        }
    });

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
    if let Some(z) = z_index {
        e.insert(ZIndex(z));
    }
    if is_scroll_area {
        e.insert((UiScrollArea, bevy_ui::ScrollPosition::default()));
    }

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
        WidgetKind::Image => {
            // Non-interactive icons let pointer events fall through to the
            // parent (slot, button, ...), same as labels do.
            let interactive = def.on_click.is_some()
                || def.on_secondary_click.is_some()
                || def.on_pointer_enter.is_some()
                || def.on_pointer_leave.is_some()
                || drag.draggable
                || drag.droppable;
            if !interactive {
                e.insert(Pickable::IGNORE);
            }
            match image_src {
                Some(src) => {
                    e.insert(UiImageSrc {
                        src: src.to_owned(),
                    });
                }
                None => {
                    e.insert((
                        Text::new(def.alt.clone().unwrap_or_default()),
                        TextFont {
                            font_size: inh.font_size.into(),
                            ..Default::default()
                        },
                        TextColor(inh.color),
                    ));
                }
            }
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
    if def.kind != WidgetKind::Image {
        if let Some(src) = background_image {
            e.insert(UiImageSrc { src });
        }
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
    if is_scroll_area && scrollbar {
        let mut bars = Vec::new();
        let thumb_color = scrollbar_color.unwrap_or(Color::srgba(0.86, 0.89, 0.94, 0.8));
        let track_color = scrollbar_track.unwrap_or(Color::srgba(0.0, 0.0, 0.0, 0.24));
        if scrolls_x {
            bars.push(spawn_scrollbar(
                commands,
                id,
                true,
                thumb_color,
                track_color,
            ));
        }
        if scrolls_y {
            bars.push(spawn_scrollbar(
                commands,
                id,
                false,
                thumb_color,
                track_color,
            ));
        }
        if !bars.is_empty() {
            commands.entity(id).add_children(&bars);
        }
    }
    id
}
