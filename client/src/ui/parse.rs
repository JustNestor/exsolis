//! Pure XML -> AST parser. No Bevy types in here, so it is trivially
//! unit-testable and can later be shared with a server-side validator.
//!
//! ```xml
//! <div id="inv" layout="column" anchor="center" bg="#222d" padding="8" gap="4" modal="true">
//!   <label font-size="20" color="#fff">Inventory</label>
//!   <div layout="row" gap="4">
//!     <button id="close" on-click="core:close" bg="#a33" hover-bg="#c44">Close</button>
//!   </div>
//! </div>
//! ```
//!
//! `<img src="icons/sword.png" alt="Sword" width="32" height="32"/>` draws a
//! PNG/JPEG from `assets/ui/`. `src` must be a relative path inside that
//! folder (no `..`, no absolute paths); `alt` is shown if the file can't load.
//!
//! Strictness policy: structural problems (bad XML, unknown element, too
//! big, too deep) are hard errors — the screen isn't mounted. Attribute
//! problems (typos, unknown attributes) are *warnings* — the widget still
//! spawns, so one bad attribute can't blank a whole screen.

use std::collections::HashMap;
use std::fmt;

use exsolis_common::UI_MAX_XML_BYTES;

pub const MAX_DEPTH: usize = 32;
/// roxmltree counts every node (elements, text, attributes).
pub const MAX_XML_NODES: u32 = 4096;

pub type Attrs = HashMap<String, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetKind {
    Div,
    Label,
    Button,
    Image,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WidgetDef {
    pub kind: WidgetKind,
    /// `id="..."`. Used as `element_id` in `UiAction`.
    pub id: Option<String>,
    /// `on-click="namespace:action"`.
    pub on_click: Option<String>,
    pub on_secondary_click: Option<String>,
    pub on_pointer_enter: Option<String>,
    /// `on-pointer-leave="namespace:action"` or `on-pointer-left="namespace:action"`.
    pub on_pointer_leave: Option<String>,
    /// `data="..."`, forwarded verbatim in `UiAction.data`.
    pub data: Option<String>,
    /// `<img src="...">`, relative to `assets/ui/`. Validated by [`is_valid_img_src`].
    pub src: Option<String>,
    /// `<img alt="...">`, shown when the image can't be loaded.
    pub alt: Option<String>,
    /// Layout/style attributes, interpreted later by `style.rs`.
    pub attrs: Attrs,
    /// Label text (whitespace-collapsed). `None` for non-labels.
    pub text: Option<String>,
    pub children: Vec<WidgetDef>,
}

#[derive(Debug, Clone)]
pub struct ParsedUi {
    pub root: WidgetDef,
    /// Non-fatal problems (unknown attributes, ...). Logged by the caller.
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub enum ParseError {
    TooLarge(usize),
    Xml(roxmltree::Error),
    UnknownElement { tag: String, line: u32 },
    TooDeep,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge(n) => write!(f, "xml too large ({n} bytes, max {UI_MAX_XML_BYTES})"),
            Self::Xml(e) => write!(f, "invalid xml: {e}"),
            Self::UnknownElement { tag, line } => {
                write!(f, "unknown element <{tag}> at line {line}")
            }
            Self::TooDeep => write!(f, "tree deeper than {MAX_DEPTH} levels"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Attributes understood by `style.rs` (kept here so typos can be reported
/// at parse time without the parser depending on Bevy).
const STYLE_ATTRS: &[&str] = &[
    // layout
    "layout",
    "cols",
    "rows",
    "grid-flow",
    "gap",
    "padding",
    "margin",
    "width",
    "height",
    "min-width",
    "min-height",
    "max-width",
    "max-height",
    "grow",
    "wrap",
    "align",
    "justify",
    "position",
    "left",
    "right",
    "top",
    "bottom",
    "hidden",
    // root only
    "anchor",
    "modal",
    // styling
    "bg",
    "color",
    "border",
    "border-color",
    "radius",
    "font-size",
    "hover-bg",
    "pressed-bg",
    // drag & drop
    "draggable",
    "droppable",
    "drag-group",
    "drag-ghost-bg",
];

pub fn parse_ui(xml: &str) -> Result<ParsedUi, ParseError> {
    if xml.len() > UI_MAX_XML_BYTES {
        return Err(ParseError::TooLarge(xml.len()));
    }
    let opts = roxmltree::ParsingOptions {
        allow_dtd: false, // no entity-expansion games
        nodes_limit: MAX_XML_NODES,
        ..Default::default()
    };
    let doc = roxmltree::Document::parse_with_options(xml, opts).map_err(ParseError::Xml)?;
    let mut warnings = Vec::new();
    let root = convert(doc.root_element(), &doc, 0, &mut warnings)?;
    Ok(ParsedUi { root, warnings })
}

fn convert(
    node: roxmltree::Node,
    doc: &roxmltree::Document,
    depth: usize,
    warnings: &mut Vec<String>,
) -> Result<WidgetDef, ParseError> {
    if depth > MAX_DEPTH {
        return Err(ParseError::TooDeep);
    }
    let tag = node.tag_name().name();
    let kind = match tag {
        "div" => WidgetKind::Div,
        "label" => WidgetKind::Label,
        "button" => WidgetKind::Button,
        "img" => WidgetKind::Image,
        other => {
            return Err(ParseError::UnknownElement {
                tag: other.to_owned(),
                line: doc.text_pos_at(node.range().start).row,
            });
        }
    };

    let mut def = WidgetDef {
        kind,
        id: None,
        on_click: None,
        on_secondary_click: None,
        data: None,
        attrs: Attrs::new(),
        text: None,
        children: Vec::new(),
        on_pointer_enter: None,
        on_pointer_leave: None,
        src: None,
        alt: None,
    };

    for a in node.attributes() {
        let (name, value) = (a.name(), a.value());
        match name {
            "id" => def.id = Some(value.to_owned()),
            "on-click" => def.on_click = Some(value.to_owned()),
            "on-secondary-click" | "on-right-click" => {
                def.on_secondary_click = Some(value.to_owned());
            }
            "data" => def.data = Some(value.to_owned()),
            "src" => def.src = Some(value.to_owned()),
            "alt" => def.alt = Some(value.to_owned()),
            "on-pointer-enter" => def.on_pointer_enter = Some(value.to_owned()),
            "on-pointer-leave" => {
                def.on_pointer_leave = Some(value.to_owned());
            }
            n if STYLE_ATTRS.contains(&n) => {
                def.attrs.insert(n.to_owned(), value.to_owned());
            }
            other => warnings.push(format!("<{tag}>: unknown attribute `{other}`")),
        }
    }

    if kind != WidgetKind::Image && (def.src.is_some() || def.alt.is_some()) {
        warnings.push(format!("<{tag}>: `src`/`alt` only apply to <img>"));
        def.src = None;
        def.alt = None;
    }

    if kind == WidgetKind::Image {
        // An image is a leaf, like a label.
        if node.children().any(|c| c.is_element()) {
            warnings.push("<img>: child elements are ignored".into());
        }
        match def.src.take() {
            Some(s) if is_valid_img_src(&s) => def.src = Some(s),
            Some(s) => warnings.push(format!(
                "<img>: invalid `src` {s:?} (needs a relative .png/.jpg/.jpeg path inside assets/ui)"
            )),
            None => warnings.push("<img>: missing `src`".into()),
        }
        return Ok(def);
    }

    if kind == WidgetKind::Label {
        // A label is a leaf: all descendant text is its content.
        if node.children().any(|c| c.is_element()) {
            warnings.push("<label>: child elements are ignored".into());
        }
        let raw: String = node
            .children()
            .filter(|c| c.is_text())
            .filter_map(|c| c.text())
            .collect();
        def.text = Some(collapse_ws(&raw));
        return Ok(def);
    }

    for child in node.children() {
        if child.is_element() {
            def.children.push(convert(child, doc, depth + 1, warnings)?);
        } else if child.is_text() {
            // Bare text inside <div>/<button> becomes an implicit label that
            // inherits color/font-size from its parent.
            let t = collapse_ws(child.text().unwrap_or(""));
            if !t.is_empty() {
                def.children.push(WidgetDef {
                    kind: WidgetKind::Label,
                    id: None,
                    on_click: None,
                    data: None,
                    attrs: Attrs::new(),
                    text: Some(t),
                    children: Vec::new(),
                    on_pointer_enter: None,
                    on_pointer_leave: None,
                    on_secondary_click: None,
                    src: None,
                    alt: None,
                });
            }
        }
    }
    Ok(def)
}

/// `src` comes from the server, so it is sandboxed to `assets/ui/`: only
/// `[A-Za-z0-9_-./]`, no absolute paths, no empty/hidden/`..` segments, and
/// only formats we decode (png, jpg, jpeg).
pub fn is_valid_img_src(src: &str) -> bool {
    if src.is_empty() || src.len() > 128 {
        return false;
    }
    if !src
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
    {
        return false;
    }
    if src
        .split('/')
        .any(|seg| seg.is_empty() || seg.starts_with('.'))
    {
        return false;
    }
    match src.rsplit_once('.') {
        Some((_, ext)) => matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"),
        None => false,
    }
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tree_and_implicit_labels() {
        let ui = parse_ui(
            r##"<div id="a" layout="row" bg="#fff">
                  <button id="b" on-click="mod:go" data="7">  Go   now </button>
                  <label>Hi
                     there</label>
                </div>"##,
        )
        .unwrap();
        assert!(ui.warnings.is_empty());
        let r = ui.root;
        assert_eq!(r.id.as_deref(), Some("a"));
        assert_eq!(r.attrs["layout"], "row");
        assert_eq!(r.children.len(), 2);
        let b = &r.children[0];
        assert_eq!(b.kind, WidgetKind::Button);
        assert_eq!(b.on_click.as_deref(), Some("mod:go"));
        assert_eq!(b.data.as_deref(), Some("7"));
        assert_eq!(b.children[0].text.as_deref(), Some("Go now"));
        assert_eq!(r.children[1].text.as_deref(), Some("Hi there"));
    }

    #[test]
    fn parses_img() {
        let ui = parse_ui(r#"<div><img src="icons/a.png" alt="A" width="8"/></div>"#).unwrap();
        assert!(ui.warnings.is_empty());
        let i = &ui.root.children[0];
        assert_eq!(i.kind, WidgetKind::Image);
        assert_eq!(i.src.as_deref(), Some("icons/a.png"));
        assert_eq!(i.alt.as_deref(), Some("A"));
        assert_eq!(i.attrs["width"], "8");
    }

    #[test]
    fn img_src_is_sandboxed() {
        for bad in [
            "../x.png",
            "/etc/x.png",
            "a//b.png",
            "a\\b.png",
            ".hidden.png",
            "x.gif",
            "png",
            "",
        ] {
            assert!(!is_valid_img_src(bad), "{bad}");
        }
        assert!(is_valid_img_src("a/b_c-1.PNG"));
        let ui = parse_ui(r#"<img src="../x.png"/>"#).unwrap();
        assert_eq!(ui.warnings.len(), 1);
        assert!(ui.root.src.is_none());
    }

    #[test]
    fn unknown_element_is_error() {
        assert!(matches!(
            parse_ui("<div><img/></div>"),
            Err(ParseError::UnknownElement { .. })
        ));
    }

    #[test]
    fn unknown_attr_is_warning() {
        let ui = parse_ui(r#"<div colour="red"/>"#).unwrap();
        assert_eq!(ui.warnings.len(), 1);
    }

    #[test]
    fn rejects_dtd_and_bad_xml() {
        assert!(parse_ui(r#"<!DOCTYPE a [<!ENTITY x "y">]><div/>"#).is_err());
        assert!(parse_ui("<div>").is_err());
    }

    #[test]
    fn rejects_deep_and_large() {
        let deep = "<div>".repeat(MAX_DEPTH + 2) + &"</div>".repeat(MAX_DEPTH + 2);
        assert!(matches!(parse_ui(&deep), Err(ParseError::TooDeep)));
        let big = format!("<div>{}</div>", "x".repeat(UI_MAX_XML_BYTES));
        assert!(matches!(parse_ui(&big), Err(ParseError::TooLarge(_))));
    }
}
