//! Server-side UI XML templates.
//!
//! Templates are ordinary XML plus three directives, expanded **on the
//! server** before the string is sent to the client as a normal `UiShow`.
//! The client never sees `v-for` / `v-if` / `{{…}}`.
//!
//! # Directives
//!
//! - **`{{path}}`** in text or attribute values — dotted lookup into the
//!   data table (and `v-for` locals). Missing → empty string.
//! - **`v-if="path"`** on an element — element omitted when the value is
//!   nil / false / empty string / 0.
//! - **`v-for="item in path"`** on an element — element repeated once per
//!   array entry; `item` is bound for the body (and nested directives).
//!
//! `v-for` is applied before `v-if` on the same node (each iteration gets
//! its own `v-if` check). Both attributes are stripped from the output.
//!
//! # Data (from Lua)
//!
//! Nested tables: sequential integer keys `1..n` become arrays; other keys
//! become string-keyed maps. Numbers, strings, bools, and nil only.
//!
//! # Integration
//!
//! ```rust
//! // in your server crate root / lib.rs:
//! pub mod ui_template;
//! ```
//!
//! Lua: `api.render_xml("inventory", { title = "...", slots = { ... } })`

use std::collections::HashMap;
use std::fmt::Write as _;

use roxmltree::{Document, Node};

// ---------------------------------------------------------------------
// Public data model (Lua → Rust)
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub enum DataValue {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<DataValue>),
    Map(HashMap<String, DataValue>),
}

impl DataValue {
    pub fn is_truthy(&self) -> bool {
        match self {
            DataValue::Nil => false,
            DataValue::Bool(b) => *b,
            DataValue::Int(i) => *i != 0,
            DataValue::Float(f) => *f != 0.0 && !f.is_nan(),
            DataValue::Str(s) => !s.is_empty(),
            DataValue::Array(a) => !a.is_empty(),
            DataValue::Map(m) => !m.is_empty(),
        }
    }

    pub fn as_array(&self) -> Option<&[DataValue]> {
        match self {
            DataValue::Array(a) => Some(a),
            _ => None,
        }
    }

    fn get_path(&self, path: &str) -> DataValue {
        let mut cur = self.clone();
        for part in path.split('.').filter(|p| !p.is_empty()) {
            cur = match cur {
                DataValue::Map(m) => m.get(part).cloned().unwrap_or(DataValue::Nil),
                DataValue::Array(a) => {
                    if let Ok(i) = part.parse::<usize>() {
                        a.get(i).cloned().unwrap_or(DataValue::Nil)
                    } else {
                        DataValue::Nil
                    }
                }
                _ => DataValue::Nil,
            };
        }
        cur
    }

    fn display(&self) -> String {
        match self {
            DataValue::Nil => String::new(),
            DataValue::Bool(b) => b.to_string(),
            DataValue::Int(i) => i.to_string(),
            DataValue::Float(f) => {
                if f.fract() == 0.0 && f.abs() < (i64::MAX as f64) {
                    (*f as i64).to_string()
                } else {
                    f.to_string()
                }
            }
            DataValue::Str(s) => s.clone(),
            DataValue::Array(_) | DataValue::Map(_) => String::new(),
        }
    }
}

/// Convert an `mlua::Value` (typically a table passed from a mod) into
/// [`DataValue`]. Sequential tables become arrays; mixed/hash tables become
/// maps. Non-convertible values become `Nil`.
pub fn data_from_lua(v: &mlua::Value) -> mlua::Result<DataValue> {
    Ok(match v {
        mlua::Value::Nil => DataValue::Nil,
        mlua::Value::Boolean(b) => DataValue::Bool(*b),
        mlua::Value::Integer(i) => DataValue::Int(*i),
        mlua::Value::Number(n) => {
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                DataValue::Int(*n as i64)
            } else {
                DataValue::Float(*n)
            }
        }
        mlua::Value::String(s) => DataValue::Str(s.to_str()?.to_owned()),
        mlua::Value::Table(t) => table_from_lua(t)?,
        _ => DataValue::Nil,
    })
}

fn table_from_lua(t: &mlua::Table) -> mlua::Result<DataValue> {
    let mut pairs: Vec<(mlua::Value, mlua::Value)> = Vec::new();
    for pair in t.pairs::<mlua::Value, mlua::Value>() {
        pairs.push(pair?);
    }

    // Array if keys are exactly 1..n (Lua sequence).
    let mut max_i: i64 = 0;
    let mut all_seq = !pairs.is_empty();
    for (k, _) in &pairs {
        match k {
            mlua::Value::Integer(i) if *i >= 1 => {
                max_i = max_i.max(*i);
            }
            mlua::Value::Number(n) if *n >= 1.0 && n.fract() == 0.0 => {
                max_i = max_i.max(*n as i64);
            }
            _ => {
                all_seq = false;
                break;
            }
        }
    }
    if all_seq && max_i as usize == pairs.len() {
        let mut arr = vec![DataValue::Nil; max_i as usize];
        for (k, v) in pairs {
            let i = match k {
                mlua::Value::Integer(i) => i as usize,
                mlua::Value::Number(n) => n as usize,
                _ => unreachable!(),
            };
            arr[i - 1] = data_from_lua(&v)?;
        }
        return Ok(DataValue::Array(arr));
    }

    let mut map = HashMap::new();
    for (k, v) in pairs {
        let key = match k {
            mlua::Value::String(s) => s.to_str()?.to_owned(),
            mlua::Value::Integer(i) => i.to_string(),
            mlua::Value::Number(n) => n.to_string(),
            mlua::Value::Boolean(b) => b.to_string(),
            _ => continue,
        };
        map.insert(key, data_from_lua(&v)?);
    }
    Ok(DataValue::Map(map))
}

// ---------------------------------------------------------------------
// Owned template AST
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TplNode {
    tag: String,
    attrs: Vec<(String, String)>,
    children: Vec<TplChild>,
}

#[derive(Clone, Debug)]
enum TplChild {
    Text(String),
    Element(TplNode),
}

#[derive(Debug)]
pub enum TemplateError {
    Xml(String),
    Expand(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Xml(s) | Self::Expand(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for TemplateError {}

// ---------------------------------------------------------------------
// Parse XML → TplNode
// ---------------------------------------------------------------------

fn parse_tpl(xml: &str) -> Result<TplNode, TemplateError> {
    let doc = Document::parse(xml).map_err(|e| TemplateError::Xml(e.to_string()))?;
    convert_node(doc.root_element())
}

fn convert_node(node: Node<'_, '_>) -> Result<TplNode, TemplateError> {
    let tag = node.tag_name().name().to_owned();
    let mut attrs = Vec::new();
    for a in node.attributes() {
        attrs.push((a.name().to_owned(), a.value().to_owned()));
    }

    let mut children = Vec::new();
    for child in node.children() {
        if child.is_element() {
            children.push(TplChild::Element(convert_node(child)?));
        } else if child.is_text() {
            let t = child.text().unwrap_or("");
            if !t.is_empty() {
                children.push(TplChild::Text(t.to_owned()));
            }
        }
    }

    Ok(TplNode {
        tag,
        attrs,
        children,
    })
}

// ---------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------

struct Scope<'a> {
    root: &'a DataValue,
    /// Innermost last. Each frame is `(loop_var_name, value)`.
    locals: Vec<(String, DataValue)>,
}

impl Scope<'_> {
    fn resolve(&self, path: &str) -> DataValue {
        let path = path.trim();
        if path.is_empty() {
            return DataValue::Nil;
        }
        let mut parts = path.split('.');
        let first = parts.next().unwrap_or("");

        let mut cur = DataValue::Nil;
        for (name, val) in self.locals.iter().rev() {
            if name == first {
                cur = val.clone();
                break;
            }
        }
        if matches!(cur, DataValue::Nil) {
            cur = self.root.get_path(first);
        }
        for part in parts {
            cur = cur.get_path(part);
        }
        cur
    }
}

fn attr<'a>(node: &'a TplNode, name: &str) -> Option<&'a str> {
    node.attrs
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

fn expand_nodes(node: &TplNode, scope: &Scope<'_>) -> Result<Vec<TplNode>, TemplateError> {
    if let Some(spec) = attr(node, "v-for") {
        let (var, list_path) = parse_v_for(spec)?;
        let list_val = scope.resolve(list_path);
        let items = list_val.as_array().ok_or_else(|| {
            TemplateError::Expand(format!(
                "v-for: `{list_path}` is not an array (got {list_val:?})"
            ))
        })?;

        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let mut nested = Scope {
                root: scope.root,
                locals: scope.locals.clone(),
            };
            nested.locals.push((var.to_owned(), item.clone()));
            let body = strip_attr(node, "v-for");
            out.extend(expand_nodes_no_for(&body, &nested)?);
        }
        return Ok(out);
    }

    expand_nodes_no_for(node, scope)
}

fn expand_nodes_no_for(node: &TplNode, scope: &Scope<'_>) -> Result<Vec<TplNode>, TemplateError> {
    if let Some(cond) = attr(node, "v-if") {
        if !scope.resolve(cond).is_truthy() {
            return Ok(Vec::new());
        }
    }

    let node = strip_attr(node, "v-if");
    let node = strip_attr(&node, "v-for");

    let mut attrs = Vec::with_capacity(node.attrs.len());
    for (k, v) in &node.attrs {
        if k == "v-if" || k == "v-for" {
            continue;
        }
        attrs.push((k.clone(), interpolate(v, scope)?));
    }

    let mut children = Vec::new();
    for child in &node.children {
        match child {
            TplChild::Text(t) => {
                let s = interpolate(t, scope)?;
                if !s.is_empty() {
                    children.push(TplChild::Text(s));
                }
            }
            TplChild::Element(el) => {
                for expanded in expand_nodes(el, scope)? {
                    children.push(TplChild::Element(expanded));
                }
            }
        }
    }

    Ok(vec![TplNode {
        tag: node.tag.clone(),
        attrs,
        children,
    }])
}

fn strip_attr(node: &TplNode, name: &str) -> TplNode {
    TplNode {
        tag: node.tag.clone(),
        attrs: node
            .attrs
            .iter()
            .filter(|(k, _)| k != name)
            .cloned()
            .collect(),
        children: node.children.clone(),
    }
}

fn parse_v_for(spec: &str) -> Result<(&str, &str), TemplateError> {
    let spec = spec.trim();
    let mut parts = spec.split_whitespace();
    let var = parts
        .next()
        .ok_or_else(|| TemplateError::Expand(format!("v-for: bad spec `{spec}`")))?;
    let in_kw = parts.next();
    let path = parts.next();
    if in_kw != Some("in") || path.is_none() || parts.next().is_some() {
        return Err(TemplateError::Expand(format!(
            "v-for: expected `item in path`, got `{spec}`"
        )));
    }
    if !var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(TemplateError::Expand(format!(
            "v-for: invalid variable name `{var}`"
        )));
    }
    Ok((var, path.unwrap()))
}

fn interpolate(input: &str, scope: &Scope<'_>) -> Result<String, TemplateError> {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(end) = input[i + 2..].find("}}") {
                let path = input[i + 2..i + 2 + end].trim();
                out.push_str(&scope.resolve(path).display());
                i = i + 2 + end + 2;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// Serialize → XML string
// ---------------------------------------------------------------------

fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn serialize(node: &TplNode) -> String {
    let mut out = String::new();
    serialize_into(node, &mut out);
    out
}

fn serialize_into(node: &TplNode, out: &mut String) {
    out.push('<');
    out.push_str(&node.tag);
    for (k, v) in &node.attrs {
        let _ = write!(out, " {}=\"{}\"", k, escape_attr(v));
    }
    if node.children.is_empty() {
        out.push_str("/>");
        return;
    }
    out.push('>');
    for child in &node.children {
        match child {
            TplChild::Text(t) => out.push_str(&escape_text(t)),
            TplChild::Element(el) => serialize_into(el, out),
        }
    }
    out.push('<');
    out.push('/');
    out.push_str(&node.tag);
    out.push('>');
}

// ---------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------

/// Expand a template XML string with `data` and return client-ready XML.
pub fn render(template_xml: &str, data: &DataValue) -> Result<String, TemplateError> {
    let root = parse_tpl(template_xml)?;
    let scope = Scope {
        root: data,
        locals: Vec::new(),
    };
    let expanded = expand_nodes(&root, &scope)?;
    if expanded.len() != 1 {
        return Err(TemplateError::Expand(format!(
            "root expanded to {} nodes (need exactly 1; check v-for/v-if on the root)",
            expanded.len()
        )));
    }
    Ok(serialize(&expanded[0]))
}

/// Load `assets/xml/{name}.xml` (same path rules as `api.load_xml`) and render.
pub fn render_file(name: &str, data: &DataValue) -> Result<String, TemplateError> {
    let rel = std::path::Path::new(name);
    if rel
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(TemplateError::Expand(
            "parent directory navigation ('..') is not allowed".into(),
        ));
    }
    let mut path = std::path::PathBuf::from("assets/xml").join(rel);
    if path.extension().is_none() {
        path.set_extension("xml");
    }
    let src = std::fs::read_to_string(&path)
        .map_err(|e| TemplateError::Xml(format!("failed to read '{}': {e}", path.display())))?;
    render(&src, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, DataValue)]) -> DataValue {
        DataValue::Map(
            entries
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn interp_and_vif() {
        let tpl = r#"<div><label>{{title}}</label><div v-if="show">X</div></div>"#;
        let data = map(&[
            ("title", DataValue::Str("Hi".into())),
            ("show", DataValue::Bool(false)),
        ]);
        let out = render(tpl, &data).unwrap();
        assert_eq!(out, "<div><label>Hi</label></div>");
    }

    #[test]
    fn vfor_slots() {
        let tpl = r#"
            <div id="inv">
              <div v-for="s in slots" id="slot_{{s.index}}">
                <div v-if="s.item">{{s.item.id}} x{{s.item.count}}</div>
              </div>
            </div>
        "#;
        let slots = DataValue::Array(vec![
            map(&[
                ("index", DataValue::Int(0)),
                (
                    "item",
                    map(&[
                        ("id", DataValue::Str("wood".into())),
                        ("count", DataValue::Int(12)),
                    ]),
                ),
            ]),
            map(&[("index", DataValue::Int(1))]),
        ]);
        let data = map(&[("slots", slots)]);
        let out = render(tpl, &data).unwrap();
        assert!(out.contains(r#"id="slot_0""#));
        assert!(out.contains("wood x12"));
        assert!(out.contains(r#"id="slot_1""#));
    }
}
