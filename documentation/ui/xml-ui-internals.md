# XML UI — how it works

The whole pipeline, from a mod calling `api.render_xml` to a widget being clicked.

```
  mod                    server                                  client
  ───                    ──────                                  ──────
  render_xml(file, data)
      │                  ui_template::render_file
      │                    ├─ parse the template XML
      │                    ├─ expand {{…}} / v-if / v-for
      │                    └─ serialize back to XML
      │                         │
      │                         └── ServerMessage::UiShow { id, xml }
      │                                        │  (ReliableOrdered)
      │                                        ▼
      │                                   ui::parse_ui  ──► WidgetDef tree
      │                                        │
      │                                   ui::style::resolve ──► bevy_ui Style
      │                                        │
      │                                   ui::spawn::spawn_ui ──► entity tree
      │                                        │
      │                                   render_bridge::sync ──► Vulkano
      │                                        │
      │◀── UiAction { ui_id, element_id, action, data } ─────────┘
      │
  api.on("ui_action", …)
```

---

## 1. The server side — the template renderer

`server/src/ui_template.rs`.

### The data model

The renderer works on `DataValue`:

```rust
pub enum DataValue {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<DataValue>),
    Map(HashMap<String, DataValue>),
}
```

`data_from_lua` converts an `mlua::Value`. The interesting rule is in `table_from_lua`:

```rust
// Array if keys are exactly 1..n (Lua sequence).
let mut max_i: i64 = 0;
let mut all_seq = !pairs.is_empty();
…
if all_seq && max_i as usize == pairs.len() {
    // → DataValue::Array
}
// otherwise → DataValue::Map
```

An **empty** table has no keys, so `all_seq` is false and it becomes an empty `Map`. That is why `v-for` over an empty Lua list is an error rather than an empty render.

### Path resolution

`Scope` holds the root data plus a stack of `v-for` locals:

```rust
struct Scope<'a> {
    root: &'a DataValue,
    /// Innermost last. Each frame is `(loop_var_name, value)`.
    locals: Vec<(String, DataValue)>,
}
```

`Scope::resolve` splits the path on `.`. The **first** segment is looked up in the locals (innermost first), then in the root map. The remaining segments are plain lookups: a map key, or a `usize` index into an array.

```rust
DataValue::Array(a) => {
    if let Ok(i) = part.parse::<usize>() {
        a.get(i).cloned().unwrap_or(DataValue::Nil)
    } else {
        DataValue::Nil
    }
}
```

So `{{s.item.count}}` works, `{{slots.0}}` works, `{{slots.first}}` does not.

### Interpolation

`interpolate` scans the string for `{{`, finds the closing `}}`, trims the path, resolves it, and appends `DataValue::display()`:

| Value | Displayed as |
|---|---|
| `Nil` | `""` |
| `Bool` | `"true"` / `"false"` |
| `Int` | the decimal |
| `Float` | the decimal, without `.0` when integral |
| `Str` | as-is |
| `Array` / `Map` | `""` |

Interpolation happens in **both** text nodes and attribute values, and the result is XML-escaped on the way out (`escape_text` / `escape_attr`), so a value containing `&` or `<` cannot break the document.

### `v-if`

```rust
if let Some(cond) = attr(node, "v-if") {
    if !scope.resolve(cond).is_truthy() {
        return Ok(Vec::new());
    }
}
```

`is_truthy` is false for `Nil`, `false`, `0`, `0.0`, `NaN`, the empty string, the empty array and the empty map.

### `v-for`

`parse_v_for` accepts exactly `<ident> in <path>` — three whitespace-separated tokens, the variable matching `[A-Za-z0-9_]+`.

The path must resolve to an array:

```rust
let items = list_val.as_array().ok_or_else(|| {
    TemplateError::Expand(format!(
        "v-for: `{list_path}` is not an array (got {list_val:?})"
    ))
})?;
```

Each iteration gets a **nested scope** with the loop variable pushed onto the locals stack. Because the stack is searched innermost-first, nested loops shadow correctly.

### The order of the two directives

`expand_nodes` handles `v-for` by recursing into `expand_nodes_no_for` for each iteration, and `expand_nodes_no_for` handles `v-if`. So:

> **`v-for` is applied before `v-if` on the same node — each iteration gets its own `v-if` check.**

That is why `<div v-for="s in slots"><div v-if="s.item">…</div></div>` renders an item box only for the slots that have an item.

### Serialization

The expanded tree is written back out as XML. Attributes are emitted in their original order, values are escaped, and an element with no children is written self-closing (`<img …/>`).

`render` requires the expansion to produce **exactly one root node**:

```rust
if expanded.len() != 1 {
    return Err(TemplateError::Expand(format!(
        "root expanded to {} nodes (need exactly 1; check v-for/v-if on the root)", … )));
}
```

`render_file` loads `assets/xml/{name}.xml`, appends `.xml` if the name has no extension, and **rejects `..`**.

### Entry points

| Caller | Data |
|---|---|
| `api.render_xml(file_or_template, data)` (Lua) | The mod's table. |
| `api.render_xml(file)` (TypeScript) | **Empty** — the TS backend renders with `Default::default()`. |
| `ui_template::render_file("hud", …)` (Rust) | Built in Rust, for the hotbar HUD. |

`api.render_xml` accepts a **file name** or an **inline template** — if the first argument starts with `<`, it is treated as the template itself.

---

## 2. The wire

Two messages, both on the reliable channel (`common/src/lib.rs`):

```rust
UiShow { id: String, xml: String },
UiHide { id: String },
```

`UiShow` is *mount or atomically replace*. The client keeps the previously mounted screen if the new XML fails to parse.

The server also tracks which screens are open per player in `OpenScreens` (`server/src/ui.rs`), fed by a `MessageReader<OutgoingPacket>` that watches for `UiShow`/`UiHide` addressed to one player. Broadcast shows are **not** tracked.

---

## 3. The client side — the parser

`client/src/ui/parse.rs`. It is a pure XML → AST parser with no Bevy types, so it is trivially unit-testable and could be shared with a server-side validator.

### The strictness policy

This is the key design decision, and it is stated at the top of the file:

> Structural problems (bad XML, unknown element, too big, too deep) are hard errors — the screen isn't mounted. Attribute problems (typos, unknown attributes) are *warnings* — the widget still spawns, so one bad attribute can't blank a whole screen.

| Problem | Result |
|---|---|
| Invalid XML, or a DTD present | `ParseError::Xml` → not mounted |
| Unknown element tag | `ParseError::UnknownElement { tag, line }` → not mounted |
| XML > 8 MiB | `ParseError::TooLarge` |
| Deeper than 32 levels | `ParseError::TooDeep` |
| More than 32 768 roxmltree nodes | `ParseError::Xml` (the `nodes_limit` option) |
| Unknown attribute | warning, widget still spawns |
| Bad attribute value | warning, attribute ignored |
| `src`/`alt` on a non-`<img>` | warning, both dropped |
| Child elements in a `<label>`/`<img>` | warning, ignored |

DTDs are rejected explicitly:

```rust
let opts = roxmltree::ParsingOptions {
    allow_dtd: false, // no entity-expansion games
    nodes_limit: MAX_XML_NODES,
    ..Default::default()
};
```

### The AST

```rust
pub enum WidgetKind { Div, Label, Button, Image, Cursor }

pub struct WidgetDef {
    pub kind: WidgetKind,
    pub id: Option<String>,
    pub on_click: Option<String>,
    pub on_secondary_click: Option<String>,
    pub on_pointer_enter: Option<String>,
    pub on_pointer_leave: Option<String>,
    pub data: Option<String>,
    pub src: Option<String>,
    pub alt: Option<String>,
    pub attrs: Attrs,          // the style attributes
    pub text: Option<String>,
    pub children: Vec<WidgetDef>,
}
```

The behavioral attributes are pulled into typed fields; the style attributes stay in an `Attrs` map for `style.rs`. Everything else becomes a warning.

### Leaf handling

`<label>` collects **all** descendant text and ignores child elements. `<img>` ignores children entirely. Bare text inside a `<div>` or `<button>` becomes a synthetic `<label>` child, so it inherits the parent's color and font size.

Label text is whitespace-collapsed: `"  Go   now "` → `"Go now"`.

### The image sandbox

`src` comes from the server, so it is sandboxed:

```rust
pub fn is_valid_img_src(src: &str) -> bool {
    if is_block_preview_src(src) || is_item_preview_src(src) { return true; }
    if src.is_empty() || src.len() > 128 { return false; }
    if !src.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/')) { return false; }
    if src.split('/').any(|seg| seg.is_empty() || seg.starts_with('.')) { return false; }
    match src.rsplit_once('.') {
        Some((_, ext)) => matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"),
        None => false,
    }
}
```

No absolute paths, no `..`, no hidden segments, no other formats.

---

## 4. The client side — style resolution

`client/src/ui/style.rs`. This is the **only** file that needs touching when Bevy's UI style API shifts between versions.

Bad values never fail — they push a warning and fall back to the default:

```rust
fn get<T>(key: &str, parse: fn(&str) -> Option<T>, attrs: &Attrs, warnings: &mut Vec<String>) -> Option<T> {
    let raw = attrs.get(key)?;
    let v = parse(raw);
    if v.is_none() {
        warnings.push(format!("invalid `{key}` value {raw:?}"));
    }
    v
}
```

### Per-kind defaults

| Kind | Default `Node` |
|---|---|
| `Div`, `Cursor` | `flex_direction: Column` |
| `Label` | `Node::default()` |
| `Image` | `flex_shrink: 0.0` — icons keep their size inside flex containers |
| `Button` | `flex_direction: Row`, `align_items: Center`, `justify_content: Center` |

### The root wrapper

`root_wrapper` builds the screen-filling wrapper and maps `anchor` onto it:

| `anchor` | Horizontal | Vertical |
|---|---|---|
| `center` (default) | center | center |
| `top-left` | start | start |
| `top` | center | start |
| `top-right` | end | start |
| `left` | start | center |
| `right` | end | center |
| `bottom-left` | start | end |
| `bottom` | center | end |
| `bottom-right` | end | end |

### The parsers

| Function | Accepts |
|---|---|
| `parse_val` | `auto`, `50%`, `10vw`, `10vh`, `12`, `12px` |
| `parse_rect` | 1, 2, 3 or 4 values (CSS shorthand). Any other count → `None`. |
| `parse_color` | `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`, `transparent`, `white`, `black` |
| `parse_bool` | `true`/`1`/`yes`, `false`/`0`/`no` |

Named CSS colors such as `red` are **not** supported.

### Grid tracks

`cols` and `rows` accept either an integer shorthand (`cols="4"` → four equal flexible columns) or a track list:

```
cols="100px 1fr auto"
```

### The `radius` quirk

```rust
if let Some(Val::Px(r)) = get("radius", parse_val, attrs, warnings) {
    node.border_radius = BorderRadius::all(Val::Px(r));
}
```

Only a pixel value is applied; a percentage is silently dropped.

---

## 5. The client side — spawning

`client/src/ui/spawn.rs`.

### The wrapper

```rust
let wrapper = commands.spawn((
    Name::new(format!("ui:{ui_id}")),
    UiRoot { ui_id: ui_id.to_owned() },
    wrapper_node,
    GlobalZIndex(UI_Z),    // 100
    Pickable::IGNORE,      // the wrapper covers the screen but must not eat world clicks
)).id();
```

`Pickable::IGNORE` is important: the wrapper covers the whole viewport, but a click on empty space must still reach the world.

### Element ids

```rust
// Auto ids are structural paths ("#0.1.2"): stable as long as the XML
// structure is stable.
let element_id = def.id.clone().unwrap_or_else(|| format!("#{path}"));
```

The path is the chain of child indices from the root, and the index is taken from the **unfiltered** child list — so `<cursor>` children shift the numbering of their later siblings.

### Inheritance

`color` and `font-size` are inherited. `spawn_widget` threads an `Inherited { color, font_size }` value down the tree, and a child only overrides it if it sets its own.

### Images

An `<img>` is sized **before** the entity is spawned (because `node` is moved into the spawn). `img_assets::probe_size` returns the natural size, the aspect ratio is set, and if neither dimension was given the widget takes the natural size.

For the synthetic sources, `probe_size` reports 48×48, so the layout reserves that aspect ratio. If one dimension *was* given, `align_self = FlexStart` is forced so `align: stretch` cannot fight the aspect ratio.

If a file cannot be read, the widget falls back to its `alt` text.

### Cursor overlays

`<cursor>` subtrees are **hoisted** to the wrapper's children:

```rust
// The custom renderer walks child order and does not apply Bevy's ZIndex
// stack, so pointer-following overlays must come after the screen tree.
commands.entity(wrapper).add_children(&cursor_overlays);
```

This is the same reason `z-index` has no effect on the custom renderer — it draws in child order and never consults Bevy's `ZIndex` component.

### Actions

```rust
if def.on_click.is_some() || def.on_secondary_click.is_some() {
    e.insert(UiClickAction { primary: …, secondary: …, data: … });
    e.observe(on_ui_click);
}
```

The observers are Bevy picking observers attached to the entity, so the picking backend decides what was hit.

---

## 6. The client side — actions

`client/src/ui/events.rs`.

### Routing

```rust
fn route(el, action, custom_data, keyboard, mouse, ui, commands, out, local) {
    if action == CLIENT_CLOSE {
        ui.hide(commands, &el.ui_id);
        return;                        // nothing is sent
    }
    let data = build_action_data(custom_data, keyboard, mouse);
    …
    if el.ui_id.starts_with(LOOPBACK_UI_PREFIX) {
        local.write(LocalUiAction { … });     // client-owned screen
    } else {
        out.write(OutgoingPacket::reliable(ClientMessage::UiAction { … }));
    }
}
```

Three routes:

1. **`client:close`** — handled entirely in the client. Nothing is sent.
2. **`clientui:` screens** — an in-process `LocalUiAction` message. Never hits the network. A server that tries to use a `clientui:` id has the message dropped in `client/src/net.rs`.
3. **Everything else** — `ClientMessage::UiAction` over the reliable channel.

### The data payload

```rust
pub fn build_action_data(custom_data, keyboard, mouse) -> Option<String> {
    let mut parts = Vec::new();
    if keyboard.pressed(ControlLeft) || keyboard.pressed(ControlRight) { parts.push("ctrl"); }
    if keyboard.pressed(ShiftLeft)   || keyboard.pressed(ShiftRight)   { parts.push("shift"); }
    if keyboard.pressed(AltLeft)     || keyboard.pressed(AltRight)     { parts.push("alt"); }
    if keyboard.pressed(SuperLeft)   || keyboard.pressed(SuperRight)   { parts.push("super"); }
    if mouse.pressed(MouseButton::Left)   { parts.push("left"); }
    if mouse.pressed(MouseButton::Right)  { parts.push("right"); }
    if mouse.pressed(MouseButton::Middle) { parts.push("middle"); }
    if let Some(d) = custom_data { if !d.is_empty() { parts.push(d); } }
    if parts.is_empty() { None } else { Some(parts.join(" ")) }
}
```

The order is fixed and the widget's own `data` is **last**. A ctrl+click on slot 7 reports `"ctrl left 7"`.

That is why the server-side inventory handler does not compare the whole string:

```rust
/// Parse click data ("3", "left 12", "right 7") into slot + button.
fn parse_click(data: Option<&str>) -> Option<(usize, bool)> {
    for token in data.split_whitespace() {
        if token.eq_ignore_ascii_case("right") { right = true; }
        else if let Ok(i) = token.parse::<usize>() { index = Some(i); }
    }
    …
}
```

### Pointer gating

Every pointer observer starts with:

```rust
if !cursor_free(&windows) { return; }
```

`cursor_free` is true only when the cursor is **not** grabbed. So no UI events fire while the player is looking around.

---

## 7. The server side — validation and dispatch

`server/src/ui.rs`.

### Validation

The server does **generic** validation only:

1. `action == "client:close"` → dropped. The reserved local-close action never reaches a mod.
2. `ui_id` must be non-empty, ≤ 64 characters, **and actually open for this player**.
3. `element_id` must be non-empty and ≤ 64 characters.
4. `action` and `data` must each be ≤ 256 characters.

For `UiDrag` the same `ui_id` check applies, plus `source_id`/`target_id` non-empty and ≤ 64 characters, `source_id != target_id`, and `data` ≤ 256 characters.

Everything **semantic** — which slot a click refers to, whether an item fits, whether a chest may be unlocked — belongs to the mod.

### The tick order

```
net::drain_incoming            (NetSet::Receive)
  → ui::dispatch_ui_packets    (GameplaySet)      ← UiActionRequested / UiDragRequested
      → creative::apply_inventory_clicks   (.after(dispatch_ui_packets).before(forward_ui_action))
      → scripting::forward_ui_action / forward_ui_drag
  → scripting::run_on_tick
  → scripting::drain_script_requests        (.before(net::flush_outgoing))
  → net::flush_outgoing                     (NetSet::Send)
```

The inventory mechanics live in **Rust**, deliberately ahead of the Lua forwarder, so the Lua handler only re-renders the snapshot:

```rust
match ev.action.as_str() {
    "inventory:click" => { inventory.click(index, right); send_inventory_sync(…); }
    "inventory:creative" => { inventory.grab(block); send_inventory_sync(…); }
    _ => {}
}
```

### What the mod receives

```lua
api.on("ui_action", function(ev)
    -- ev.player_id, ev.ui_id, ev.element_id, ev.action, ev.data
end)

api.on("ui_drag", function(ev)
    -- ev.player_id, ev.ui_id, ev.source_id, ev.target_id, ev.data
end)
```

---

## 8. Drag and drop

`client/src/ui/drag.rs`.

| Component | Meaning |
|---|---|
| `Draggable { data, ghost_bg }` | A drag source. |
| `Droppable` | A drop target. |
| `DragGroup(Option<String>)` | `None` matches **any** group. |
| `ActiveDrag(Option<DragState>)` | The drag in flight. |

The flow:

1. **`on_drag_start`** (`Pointer<DragStart>`) — primary button only. `ev.propagate(false)` so dragging a label inside a draggable div still starts the div's drag. A **ghost** is spawned: a 40×40 absolutely-positioned square with `GlobalZIndex(200)` and `Pickable::IGNORE`, so it cannot steal the drop event.
2. **`on_drag_move`** (`Pointer<Drag>`) — recenters the ghost on the cursor position.
3. **`on_drag_drop`** (`Pointer<DragDrop>`) — the packet is sent here:

```rust
if target_el.ui_id != drag.ui_id || target_el.element_id == drag.source_id { return; }
if let (Some(a), Some(b)) = (&drag.group, &target_group.0) { if a != b { return; } }
out.write(OutgoingPacket::reliable(ClientMessage::UiDrag {
    ui_id: drag.ui_id, source_id: drag.source_id, target_id: target_el.element_id.clone(), data: drag.data,
}));
```

4. **`on_drag_end`** — cleanup. `cancel_drag_if_ui_hidden` cleans up if the screen was hidden mid-drag.

The client only reports geometry. It never moves an item.

---

## 9. Block and item previews

`client/src/ui/block_preview.rs`.

`block#namespace:name` renders a **48×48 isometric cube preview**, rasterized on the CPU:

- The model is resolved from the baked block models, with an `exsolis:* → minecraft:*` legacy fallback.
- The texture comes from the pack directory; a documented quirk collapses one doubled `block/block/` segment and retries.
- The projection is a classic 2:1 isometric (`((x - z) * COS30, (x + z) * SIN30 - y)`), fitted into the 48×48 box with 3 px padding.
- Faces are culled with `dot(normal, [1,1,1]) > 0` and shaded: top 1.0, bottom 0.55, ±X 0.8, ±Z 0.7, otherwise 0.85.
- Grayscale `tintindex` faces get the plains biome tint (`#91BD59`).
- Rasterization is two triangles per quad with barycentric interpolation, a z-buffer, nearest-neighbour sampling and an alpha cutout at 128.

`item#namespace:name` renders a flat 48×48 icon — the `layer0` texture of the item model.

Both are uploaded like any other `<img>`.

---

## 10. Image uploads

`client/src/ui/render_bridge.rs`.

- `sync_ui_images` walks every mounted `UiImageSrc` and uploads whatever is not registered yet.
- At most **4 new images per frame** (`MAX_IMAGE_UPLOADS_PER_FRAME`), so a screen with many new icons trickles in over several frames.
- The format is `R8G8B8A8_SRGB` when the swapchain is sRGB, else `R8G8B8A8_UNORM`.
- A failure is recorded and **never retried**.
- Bindless slots are **never freed** by design — a long session with many distinct icons accumulates slots.

---

## 11. Screen lifecycle

`client/src/ui/mod.rs`.

### Mounting

```rust
pub fn show(&mut self, commands: &mut Commands, ui_id: &str, xml: &str) {
    if ui_id.is_empty() || ui_id.len() > UI_MAX_ID_LEN { warn!(…); return; }
    if !self.roots.contains_key(ui_id) && self.roots.len() >= UI_MAX_OPEN { warn!(…); return; }
    let parsed = match parse::parse_ui(xml) {
        Ok(p) => p,
        Err(e) => { warn!("ui `{ui_id}`: {e}; keeping previous version"); return; }
    };
    self.hide(commands, ui_id);            // replace semantics
    let root = spawn::spawn_ui(commands, ui_id, &parsed);
    self.roots.insert(ui_id.to_owned(), root);
    if parsed.root.attrs.get("modal").is_some_and(|v| v == "true") { self.modal.insert(ui_id.to_owned()); }
}
```

### Input capture and the cursor

`escape_and_cursor` owns the Escape key and the cursor grab:

- While a modal is mounted, the cursor is **released** (visible, ungrabbed).
- When the last modal closes, the cursor is **re-grabbed** (hidden, locked).
- `Escape`: a client modal re-mounts the main menu; a remote modal closes all remote modals; otherwise the grab toggles.
- `gameplay_input_allowed` is the gate gameplay code consults — it also requires the previous frame to have been allowed, so the state does not flap.

### Disconnect

`hide_on_disconnect` hides every **remote** screen when the connection drops. Client-owned screens survive.

---

## 12. Limits

| Constant | Value | Where |
|---|---|---|
| `UI_MAX_XML_BYTES` | 8 MiB | `common/src/lib.rs` |
| `UI_MAX_ID_LEN` | 64 | `common/src/lib.rs` |
| `UI_MAX_ACTION_LEN` | 256 | `common/src/lib.rs` |
| `UI_MAX_OPEN` | 32 | `common/src/lib.rs` |
| `UI_DRAG_MAX_ID_LEN` | 64 | `common/src/lib.rs` |
| `MAX_DEPTH` | 32 | `client/src/ui/parse.rs` |
| `MAX_XML_NODES` | 32 768 | `client/src/ui/parse.rs` |
| `MAX_IMAGE_DIM` | 4096 | `client/src/ui/img_assets.rs` |
| `MAX_IMAGE_UPLOADS_PER_FRAME` | 4 | `client/src/ui/render_bridge.rs` |
| `src` length | 128 | `client/src/ui/parse.rs` |
| `block#`/`item#` name length | 160 | `client/src/ui/parse.rs` |

The important ones are enforced **twice** — the client checks them when mounting, the server checks them when dispatching — because the client is not trusted.

---

## 13. Known limitations

1. **`virtual`, `item-height` and `overscan` are unreachable.** `style.rs` reads them, but the parser's `STYLE_ATTRS` list does not contain them, so they are dropped as unknown attributes before `style.rs` sees them. Virtualization is therefore always on with an auto-measured item height. See [scroll_virtualization_plan.md](../plans/scroll_virtualization_plan.md).

2. **`font-weight` does not exist.** `assets/xml/inventory.xml` uses it; it warns and does nothing.

3. **`on-pointer-left` is documented in a comment but not implemented.**

4. **An empty Lua table is a Map, not an empty Array**, so `v-for` over an empty list is a hard error rather than an empty render.

5. **`z-index` has no rendering effect** — the custom renderer draws in child order.

6. **`client:close` and Esc-to-close are client-local.** The server's `OpenScreens` still lists the screen as open, and the mod's own state is not updated. A later `UiShow` for the same id re-mounts it.

7. **The `data` payload mixes token kinds.** A handler that treats `data` as a verbatim value can be broken by a modifier token.

8. **TypeScript mods get no template data**, so a data-driven template fails in the TS backend.

9. **Bindless GPU slots for images and glyph atlases are never freed.**

10. **The creative catalog is close to the node budget** — ~1200 cells × ~18 nodes each approaches `MAX_XML_NODES`.

11. **The server and client parsers differ on DTD handling** — the client sets `allow_dtd: false`; the server renderer uses the roxmltree defaults.
