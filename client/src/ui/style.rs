//! Attribute strings -> `bevy_ui` values.
//!
//! This is the ONLY file that should need touching when bevy_ui's style
//! API shifts between Bevy versions (`border_radius`, `BorderColor`, ...).
//! Bad values never fail: they push a message into `warnings` and fall
//! back to the default.

use bevy_color::Color;
use bevy_ui::prelude::*;

use super::parse::{Attrs, WidgetKind};

// ------------------------------------------------------------------ values

/// `12`, `12px`, `50%`, `10vw`, `10vh`, `auto`.
pub fn parse_val(s: &str) -> Option<Val> {
    let s = s.trim();
    if s == "auto" {
        return Some(Val::Auto);
    }
    let num = |n: &str| n.trim().parse::<f32>().ok().filter(|v| v.is_finite());
    if let Some(n) = s.strip_suffix('%') {
        return num(n).map(Val::Percent);
    }
    if let Some(n) = s.strip_suffix("vw") {
        return num(n).map(Val::Vw);
    }
    if let Some(n) = s.strip_suffix("vh") {
        return num(n).map(Val::Vh);
    }
    num(s.strip_suffix("px").unwrap_or(s)).map(Val::Px)
}

/// CSS shorthand: 1 value = all, 2 = vertical horizontal,
/// 3 = top horizontal bottom, 4 = top right bottom left.
pub fn parse_rect(s: &str) -> Option<UiRect> {
    let v: Vec<Val> = s.split_whitespace().map(parse_val).collect::<Option<_>>()?;
    Some(match v.as_slice() {
        [a] => UiRect::all(*a),
        [v, h] => UiRect::axes(*h, *v),
        [t, h, b] => UiRect {
            top: *t,
            left: *h,
            right: *h,
            bottom: *b,
        },
        [t, r, b, l] => UiRect {
            top: *t,
            right: *r,
            bottom: *b,
            left: *l,
        },
        _ => return None,
    })
}

/// `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`, plus `transparent`/`white`/`black`.
pub fn parse_color(s: &str) -> Option<Color> {
    let s = s.trim();
    match s {
        "transparent" => return Some(Color::srgba_u8(0, 0, 0, 0)),
        "white" => return Some(Color::srgb_u8(255, 255, 255)),
        "black" => return Some(Color::srgb_u8(0, 0, 0)),
        _ => {}
    }
    let d: Vec<u8> = s
        .strip_prefix('#')?
        .chars()
        .map(|c| c.to_digit(16).map(|d| d as u8))
        .collect::<Option<_>>()?;
    let (r, g, b, a) = match d.len() {
        3 => (d[0] * 17, d[1] * 17, d[2] * 17, 255),
        4 => (d[0] * 17, d[1] * 17, d[2] * 17, d[3] * 17),
        6 => (d[0] * 16 + d[1], d[2] * 16 + d[3], d[4] * 16 + d[5], 255),
        8 => (
            d[0] * 16 + d[1],
            d[2] * 16 + d[3],
            d[4] * 16 + d[5],
            d[6] * 16 + d[7],
        ),
        _ => return None,
    };
    Some(Color::srgba_u8(r, g, b, a))
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

fn parse_direction(s: &str) -> Option<FlexDirection> {
    Some(match s.trim() {
        "column" => FlexDirection::Column,
        "row" => FlexDirection::Row,
        "column-reverse" => FlexDirection::ColumnReverse,
        "row-reverse" => FlexDirection::RowReverse,
        _ => return None,
    })
}

fn parse_align(s: &str) -> Option<AlignItems> {
    Some(match s.trim() {
        "start" => AlignItems::FlexStart,
        "center" => AlignItems::Center,
        "end" => AlignItems::FlexEnd,
        "stretch" => AlignItems::Stretch,
        _ => return None,
    })
}

fn parse_justify(s: &str) -> Option<JustifyContent> {
    Some(match s.trim() {
        "start" => JustifyContent::FlexStart,
        "center" => JustifyContent::Center,
        "end" => JustifyContent::FlexEnd,
        "between" => JustifyContent::SpaceBetween,
        "around" => JustifyContent::SpaceAround,
        "evenly" => JustifyContent::SpaceEvenly,
        _ => return None,
    })
}

// ----------------------------------------------------------------- resolve

/// Everything `spawn.rs` needs from a widget's attributes.
#[derive(Debug, Clone)]
pub struct ResolvedStyle {
    pub node: Node,
    pub bg: Option<Color>,
    pub border_color: Option<Color>,
    /// Text color; inherited by descendants when set.
    pub color: Option<Color>,
    /// Font size; inherited by descendants when set.
    pub font_size: Option<f32>,
    pub hover_bg: Option<Color>,
    pub pressed_bg: Option<Color>,
    pub drag: DragConfig,
}

/// Drag/drop knobs for one widget. `None`/empty group means "matches any
/// group" on both sides, so simple single-purpose UIs (one inventory, one
/// set of slots) don't need `drag-group` at all.
#[derive(Debug, Clone, Default)]
pub struct DragConfig {
    pub draggable: bool,
    pub droppable: bool,
    pub group: Option<String>,
    /// Background of the floating ghost while this widget is being
    /// dragged. Falls back to the widget's own `bg` (dimmed) when unset.
    pub ghost_bg: Option<Color>,
}

fn get<T>(
    key: &str,
    parse: fn(&str) -> Option<T>,
    attrs: &Attrs,
    warnings: &mut Vec<String>,
) -> Option<T> {
    let raw = attrs.get(key)?;
    let v = parse(raw);
    if v.is_none() {
        warnings.push(format!("invalid `{key}` value {raw:?}"));
    }
    v
}

pub fn resolve(kind: WidgetKind, attrs: &Attrs, warnings: &mut Vec<String>) -> ResolvedStyle {
    // Per-kind defaults.
    let mut node = match kind {
        WidgetKind::Div => Node {
            flex_direction: FlexDirection::Column,
            ..Default::default()
        },
        WidgetKind::Label => Node::default(),
        WidgetKind::Button => Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..Default::default()
        },
    };

    // `get` reads + parses one attribute, warning on bad values.
    // let mut get = |key: &str, parse: fn(&str) -> Option<_>| -> Option<_> {
    //     let raw = attrs.get(key)?;
    //     let v = parse(raw);
    //     if v.is_none() {
    //         warnings.push(format!("invalid `{key}` value {raw:?}"));
    //     }
    //     v
    // };

    if let Some(v) = get(
        "layout",
        |s| match s.trim() {
            "column" => Some((Display::Flex, FlexDirection::Column)),
            "row" => Some((Display::Flex, FlexDirection::Row)),
            "column-reverse" => Some((Display::Flex, FlexDirection::ColumnReverse)),
            "row-reverse" => Some((Display::Flex, FlexDirection::RowReverse)),
            "grid" => Some((Display::Grid, FlexDirection::Row)),
            _ => None,
        },
        attrs,
        warnings,
    ) {
        node.display = v.0;
        node.flex_direction = v.1;
    }

    if let Some(cols) = get("cols", parse_grid_tracks, attrs, warnings) {
        node.display = Display::Grid;
        node.grid_template_columns = cols;
    }
    if let Some(rows) = get("rows", parse_grid_tracks, attrs, warnings) {
        node.display = Display::Grid;
        node.grid_template_rows = rows;
    }
    if let Some(flow) = get("grid-flow", parse_grid_flow, attrs, warnings) {
        node.display = Display::Grid;
        node.grid_auto_flow = flow;
    }
    if let Some(v) = get("gap", parse_val, attrs, warnings) {
        node.row_gap = v;
        node.column_gap = v;
    }
    if let Some(v) = get("padding", parse_rect, attrs, warnings) {
        node.padding = v;
    }
    if let Some(v) = get("margin", parse_rect, attrs, warnings) {
        node.margin = v;
    }
    if let Some(v) = get("border", parse_rect, attrs, warnings) {
        node.border = v;
    }
    if let Some(v) = get("width", parse_val, attrs, warnings) {
        node.width = v;
    }
    if let Some(v) = get("height", parse_val, attrs, warnings) {
        node.height = v;
    }
    if let Some(v) = get("min-width", parse_val, attrs, warnings) {
        node.min_width = v;
    }
    if let Some(v) = get("min-height", parse_val, attrs, warnings) {
        node.min_height = v;
    }
    if let Some(v) = get("max-width", parse_val, attrs, warnings) {
        node.max_width = v;
    }
    if let Some(v) = get("max-height", parse_val, attrs, warnings) {
        node.max_height = v;
    }
    if let Some(v) = get(
        "grow",
        |s| s.trim().parse::<f32>().ok().filter(|v| v.is_finite()),
        attrs,
        warnings,
    ) {
        node.flex_grow = v.max(0.0);
    }
    if get("wrap", parse_bool, attrs, warnings) == Some(true) {
        node.flex_wrap = FlexWrap::Wrap;
    }
    if let Some(v) = get("align", parse_align, attrs, warnings) {
        node.align_items = v;
    }
    if let Some(v) = get("justify", parse_justify, attrs, warnings) {
        node.justify_content = v;
    }
    if let Some(v) = get(
        "position",
        |s| match s.trim() {
            "absolute" => Some(PositionType::Absolute),
            "relative" => Some(PositionType::Relative),
            _ => None,
        },
        attrs,
        warnings,
    ) {
        node.position_type = v;
    }
    if let Some(v) = get("left", parse_val, attrs, warnings) {
        node.left = v;
    }
    if let Some(v) = get("right", parse_val, attrs, warnings) {
        node.right = v;
    }
    if let Some(v) = get("top", parse_val, attrs, warnings) {
        node.top = v;
    }
    if let Some(v) = get("bottom", parse_val, attrs, warnings) {
        node.bottom = v;
    }
    if get("hidden", parse_bool, attrs, warnings) == Some(true) {
        node.display = Display::None;
    }
    if let Some(Val::Px(r)) = get("radius", parse_val, attrs, warnings) {
        // ASSUMPTION: border radius lives on `Node` (Bevy >= 0.17). If your
        // 0.19 still has a separate `BorderRadius` component, return it in
        // `ResolvedStyle` and insert it in spawn.rs instead.
        node.border_radius = BorderRadius::all(Val::Px(r));
    }

    ResolvedStyle {
        node,
        bg: get("bg", parse_color, attrs, warnings),
        border_color: get("border-color", parse_color, attrs, warnings),
        color: get("color", parse_color, attrs, warnings),
        font_size: get(
            "font-size",
            |s| {
                s.trim()
                    .trim_end_matches("px")
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && *v > 0.0)
            },
            attrs,
            warnings,
        ),
        hover_bg: get("hover-bg", parse_color, attrs, warnings),
        pressed_bg: get("pressed-bg", parse_color, attrs, warnings),
        drag: DragConfig {
            draggable: get("draggable", parse_bool, attrs, warnings).unwrap_or(false),
            droppable: get("droppable", parse_bool, attrs, warnings).unwrap_or(false),
            group: attrs.get("drag-group").cloned(),
            ghost_bg: get("drag-ghost-bg", parse_color, attrs, warnings),
        },
    }
}

// ------------------------------------------------------------------ anchor

/// The screen-filling wrapper every mounted UI lives in. `anchor` decides
/// where the XML root sits: `center` (default), `top-left`, `top`,
/// `top-right`, `left`, `right`, `bottom-left`, `bottom`, `bottom-right`.
pub fn root_wrapper(anchor: Option<&str>, warnings: &mut Vec<String>) -> Node {
    use AlignItems as A;
    use JustifyContent as J;
    // Wrapper is a row: justify = horizontal, align = vertical.
    let (j, a) = match anchor.map(str::trim).unwrap_or("center") {
        "center" => (J::Center, A::Center),
        "top-left" => (J::FlexStart, A::FlexStart),
        "top" => (J::Center, A::FlexStart),
        "top-right" => (J::FlexEnd, A::FlexStart),
        "left" => (J::FlexStart, A::Center),
        "right" => (J::FlexEnd, A::Center),
        "bottom-left" => (J::FlexStart, A::FlexEnd),
        "bottom" => (J::Center, A::FlexEnd),
        "bottom-right" => (J::FlexEnd, A::FlexEnd),
        other => {
            warnings.push(format!("invalid `anchor` value {other:?}"));
            (J::Center, A::Center)
        }
    };
    Node {
        position_type: PositionType::Absolute,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        justify_content: j,
        align_items: a,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vals() {
        assert_eq!(parse_val("12"), Some(Val::Px(12.0)));
        assert_eq!(parse_val("12px"), Some(Val::Px(12.0)));
        assert_eq!(parse_val("50%"), Some(Val::Percent(50.0)));
        assert_eq!(parse_val("auto"), Some(Val::Auto));
        assert_eq!(parse_val("nan"), None);
        assert_eq!(parse_val("abc"), None);
    }

    #[test]
    fn rects() {
        assert_eq!(parse_rect("4"), Some(UiRect::all(Val::Px(4.0))));
        let r = parse_rect("1 2 3 4").unwrap();
        assert_eq!(
            (r.top, r.right, r.bottom, r.left),
            (Val::Px(1.0), Val::Px(2.0), Val::Px(3.0), Val::Px(4.0))
        );
        assert_eq!(parse_rect("1 2 3 4 5"), None);
    }

    #[test]
    fn colors() {
        assert_eq!(parse_color("#f00"), Some(Color::srgba_u8(255, 0, 0, 255)));
        assert_eq!(
            parse_color("#11223344"),
            Some(Color::srgba_u8(0x11, 0x22, 0x33, 0x44))
        );
        assert_eq!(parse_color("#12"), None);
        assert_eq!(parse_color("red"), None);
    }
}

fn parse_grid_flow(s: &str) -> Option<GridAutoFlow> {
    Some(match s.trim() {
        "row" => GridAutoFlow::Row,
        "column" => GridAutoFlow::Column,
        "row-dense" => GridAutoFlow::RowDense,
        "column-dense" => GridAutoFlow::ColumnDense,
        _ => return None,
    })
}

fn parse_grid_tracks(s: &str) -> Option<Vec<RepeatedGridTrack>> {
    let s = s.trim();
    // Integer shortcut: `cols="4"` -> 4 equal flexible columns
    if let Ok(count) = s.parse::<u16>() {
        if count > 0 {
            return Some(vec![RepeatedGridTrack::flex(count, 1.0)]);
        }
    }

    // Explicit space-separated track list: `cols="100px 1fr auto"`
    let tracks: Vec<GridTrack> = s
        .split_whitespace()
        .map(|part| {
            if part == "auto" {
                Some(GridTrack::auto())
            } else if let Some(n) = part.strip_suffix("fr") {
                n.parse::<f32>().ok().map(GridTrack::flex)
            } else if let Some(val) = parse_val(part) {
                match val {
                    Val::Px(px) => Some(GridTrack::px(px)),
                    Val::Percent(p) => Some(GridTrack::percent(p)),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect::<Option<_>>()?;

    if tracks.is_empty() {
        None
    } else {
        Some(
            tracks
                .into_iter()
                .map(|t| RepeatedGridTrack::repeat_many(1, vec![t]))
                .collect(),
        )
    }
}
