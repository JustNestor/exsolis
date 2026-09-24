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
}

#[derive(Debug, Clone, PartialEq)]
pub struct WidgetDef {
    pub kind: WidgetKind,
    /// `id="..."`. Used as `element_id` in `UiAction`.
    pub id: Option<String>,
    /// `on-click="namespace:action"`.
    pub on_click: Option<String>,
    /// `data="..."`, forwarded verbatim in `UiAction.data`.
    pub data: Option<String>,
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
        data: None,
        attrs: Attrs::new(),
        text: None,
        children: Vec::new(),
    };

    for a in node.attributes() {
        let (name, value) = (a.name(), a.value());
        match name {
            "id" => def.id = Some(value.to_owned()),
            "on-click" => def.on_click = Some(value.to_owned()),
            "data" => def.data = Some(value.to_owned()),
            n if STYLE_ATTRS.contains(&n) => {
                def.attrs.insert(n.to_owned(), value.to_owned());
            }
            other => warnings.push(format!("<{tag}>: unknown attribute `{other}`")),
        }
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
                });
            }
        }
    }
    Ok(def)
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
