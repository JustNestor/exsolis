# XML UI — how to use it

The UI is **server-driven**. The server sends an XML document; the client parses it into a widget tree and reports clicks back. The client never decides what a button does.

```
server                                     client
──────                                     ──────
api.render_xml("inventory", data)
  → expands {{…}}, v-if, v-for
  → sends UiShow { id, xml }
                                          → parses the XML
                                          → spawns the widget tree
                                          → draws it
user clicks a slot
  ← UiAction { ui_id, element_id, action, data }
api.on("ui_action", handler)
```

There are two halves to learn:

- **The XML vocabulary** — the elements and attributes the client understands. This page.
- **The template language** — `{{…}}`, `v-if` and `v-for`, expanded on the server.

---

## Quick start

A minimal screen:

```xml
<div id="panel" anchor="center" layout="column" gap="8" padding="16" bg="#222831ee">
  <label font-size="20" color="#ffffff">Hello</label>
  <button id="ok" on-click="mymod:ok" bg="#3a6ea5" padding="6 12" radius="6">
    <label color="#ffffff">OK</label>
  </button>
</div>
```

Render it from a Lua mod and send it:

```lua
api.on("player_join", function(ev)
    api.send(ev.player_id, "ui_show", "panel", api.render_xml("mypanel", {
        title = "Hello",
    }))
end)
```

Or from a TypeScript mod:

```ts
export function on_player_join(id: string): void {
    send(id, "ui_show", "panel", render_xml("mypanel"));
}
```

---

## The elements

There are exactly **five** elements. Anything else is a hard error and the screen is not mounted.

| Element | What it is | Notes |
|---|---|---|
| `<div>` | A container. | The workhorse: layout, background, border, and children. |
| `<label>` | A run of text. | A **leaf** — child elements are ignored. |
| `<button>` | A clickable element. | Behaves like a div, plus hover/pressed feedback and click actions. |
| `<img>` | An image. | A **leaf** — child elements are ignored. |
| `<cursor>` | A subtree that follows the pointer. | Hidden until its parent drag source moves. |

### Leaf rules

`<label>` and `<img>` are leaves. If they contain child elements, the client logs a warning and ignores the children — it does not fail the screen.

Bare text inside a `<div>` or `<button>` becomes an **implicit label** that inherits `color` and `font-size` from its parent:

```xml
<button on-click="mymod:go">Go now</button>
```

is equivalent to:

```xml
<button on-click="mymod:go">
  <label>Go now</label>
</button>
```

---

## Layout

The layout is flexbox, with a CSS-like vocabulary.

| Attribute | Values | Default |
|---|---|---|
| `layout` | `column`, `row`, `column-reverse`, `row-reverse`, `grid` | `column` for div/cursor, `row` for button |
| `cols` | `4` (N equal columns) or `"100px 1fr auto"` | — |
| `rows` | Same as `cols` | — |
| `grid-flow` | `row`, `column`, `row-dense`, `column-dense` | — |
| `gap` | A length | `0` |
| `align` | `start`, `center`, `end`, `stretch` | `stretch` |
| `justify` | `start`, `center`, `end`, `between`, `around`, `evenly` | `start` |
| `wrap` | `true` / `false` | `false` |
| `grow` | A number (flex-grow) | `0` |
| `position` | `absolute`, `relative` | `relative` |
| `left` / `right` / `top` / `bottom` | A length | `auto` |
| `hidden` | `true` / `false` | `false` |

### Sizing

| Attribute | Meaning |
|---|---|
| `width`, `height` | The preferred size. |
| `min-width`, `min-height`, `max-width`, `max-height` | The size bounds. |
| `padding` | Inner spacing. Also a rect shorthand. |
| `margin` | Outer spacing. Also a rect shorthand. |

### The root anchor

`anchor` decides where the screen sits. It only applies to the **root** element:

| Value | Position |
|---|---|
| `center` | The default — dead centre. |
| `top-left`, `top`, `top-right` | Along the top edge. |
| `left`, `right` | Vertically centred, at one edge. |
| `bottom-left`, `bottom`, `bottom-right` | Along the bottom edge. |

```xml
<div anchor="bottom" layout="column" align="center" gap="4" padding="0 0 10">
```

---

## Values

### Lengths

A length is a number, optionally suffixed:

| Form | Meaning |
|---|---|
| `12` or `12px` | Pixels. |
| `50%` | Percent of the parent. |
| `10vw` / `10vh` | Percent of the viewport width/height. |
| `auto` | Let the layout decide. |

### Rects

`padding`, `margin` and `border` accept a CSS-style shorthand:

| Form | Meaning |
|---|---|
| `4` | All four sides. |
| `1 2` | Vertical, horizontal. |
| `1 2 3` | Top, horizontal, bottom. |
| `1 2 3 4` | Top, right, bottom, left. |

### Colors

`bg`, `color`, `border-color`, `hover-bg`, `pressed-bg`, `scrollbar-color`, `scrollbar-track`, `drag-ghost-bg`:

| Form | Meaning |
|---|---|
| `#RGB` | `#f00` → opaque red. |
| `#RGBA` | `#f008` → red at 53% alpha. |
| `#RRGGBB` | `#ff0000` |
| `#RRGGBBAA` | `#ff000080` → red at 50% alpha. |
| `transparent`, `white`, `black` | Named. |

> The alpha byte is what makes the glassy panels work: `bg="#0f1220cc"` is a dark blue at 80%.

### Booleans

`true` / `false`, or `1` / `0`, or `yes` / `no`.

---

## Appearance

| Attribute | Applies to | Meaning |
|---|---|---|
| `bg` | any | The background color. |
| `color` | any | The **text** color. Inherited by descendants that do not set their own. |
| `font-size` | any | The text size. Inherited the same way. |
| `border` | any | The border width, as a rect shorthand. |
| `border-color` | any | The border color. |
| `radius` | any | The corner radius, in **pixels only**. |
| `hover-bg` | button | The background while hovered. Only applied if the button has a `bg`. |
| `pressed-bg` | button | The background while pressed. Falls back to `hover-bg`, then `bg`. |
| `background-image` | any | A background image path (relative to `assets/ui/`). |
| `z-index` | any | The stacking order within the screen. |
| `scrollbar` | scroll areas | `false` hides the scrollbar. Default `true`. |
| `scrollbar-color` | scroll areas | The scrollbar thumb. |
| `scrollbar-track` | scroll areas | The scrollbar track. |

`color` and `font-size` are **inherited**: a `<label>` inside a `<div color="#a8b4e0">` uses that color unless it sets its own.

> **There is no `font-weight`.** `assets/xml/inventory.xml` uses `font-weight="bold"`, but no such attribute exists — it logs an "unknown attribute" warning and the text is *not* bold.

---

## Overflow and scrolling

| Attribute | Meaning |
|---|---|
| `overflow` | `visible`, `clip`, `hidden`, `scroll` (both axes), `scroll-x`, `scroll-y`, `auto` |

A scrollable panel:

```xml
<div layout="column" gap="4" padding="6" bg="#0f1426" overflow="scroll" height="320">
  ...
</div>
```

The scroll offset is remembered per screen, so a screen that is re-mounted (the inventory re-renders on every click) keeps its scroll position instead of jumping back to the top.

---

## Images

### From disk

```xml
<img src="icons/sword.png" alt="Sword" width="32" height="32"/>
```

`src` is relative to `assets/ui/` and is **sandboxed**: only `[A-Za-z0-9_-./]`, no absolute paths, no empty or hidden (`..`, `.`) segments, and only `.png`, `.jpg`, `.jpeg`. Anything else is rejected with a warning and the image falls back to its `alt` text.

Images are read from the **client's asset cache**, not from the server's source directory. A base-repo asset at `assets/ui/icons/x.png` is referenced as `src="icons/x.png"`. A mod's asset at `mods/mymod/assets/ui/icons/x.png` is referenced as `src="mods/mymod/ui/icons/x.png"` — the mod name is part of the path.

If both `width` and `height` are unset, the image takes its natural size. If one is set, the aspect ratio is preserved.

### Block and item previews

Two **synthetic** sources render game content instead of reading a file:

```xml
<img src="block#exsolis:stone" width="40" height="40"/>
<img src="item#minecraft:diamond" width="40" height="40"/>
```

| Prefix | Renders |
|---|---|
| `block#` | A 48×48 isometric 3D cube preview, rasterized on the CPU from the block's baked model. |
| `item#` | A 48×48 flat item icon (the layer-0 texture). |

The name after the prefix is `namespace:name`, using the same charset as a mod action id. These sources never touch disk.

This is how the creative grid and the inventory render icons without shipping a texture per block:

```lua
local prefix = "block#"
if name and not api.is_placeable(name) then prefix = "item#" end
```

---

## The template language

Templates are ordinary XML plus **three directives**, expanded on the server before the string is sent. The client never sees them.

| Directive | Where | Meaning |
|---|---|---|
| `{{path}}` | Text and attribute values | A dotted lookup into the data table. |
| `v-if="path"` | On an element | The element is omitted when the value is falsy. |
| `v-for="item in path"` | On an element | The element is repeated once per array entry. |

### `{{path}}`

A dotted path into the data. Missing values render as the **empty string**.

```xml
<label font-size="20" color="#e6ecff">{{title}}</label>
<div bg="{{slot.bg}}" border-color="{{slot.border}}">...</div>
```

Inside a `v-for`, the loop variable is in scope and shadows the root:

```xml
<div v-for="s in slots" id="slot_{{s.index}}" data="{{s.index}}">
  <label>{{s.item.name}} × {{s.item.count}}</label>
</div>
```

### `v-if`

An element is omitted when the value is `nil`, `false`, `0`, an empty string, an empty array or an empty map.

```xml
<div v-if="s.item" id="item_{{s.index}}">
  <img v-if="s.item.img" src="{{s.item.img}}"/>
  <label>{{s.item.count}}</label>
</div>
```

### `v-for`

`v-for="item in path"` repeats the element once per entry of the array at `path`. The variable name must be alphanumeric or `_`.

```xml
<div v-for="c in creative" layout="row" gap="4">
  <div v-for="cell in c.cells" on-click="inventory:creative" data="{{cell.name}}">
    <img src="{{cell.img}}"/>
  </div>
</div>
```

Nested loops work, and each loop variable shadows the outer one of the same name.

**Order matters:** `v-for` is applied **before** `v-if` on the same node, so each iteration gets its own `v-if` check. That is why the inner `<div v-if="s.item">` above can be inside a `v-for` and still vary per iteration.

Both attributes are stripped from the output.

### The data model

The data comes from the mod as a table of scalars, arrays and maps:

| Lua | Rendered as |
|---|---|
| `nil` | empty string, falsy |
| `true` / `false` | `true` / `false`, falsy when false |
| `1` | `1` |
| `1.5` | `1.5` |
| `"text"` | `text` |
| `{ "a", "b" }` | an array (keys are exactly `1..n`) |
| `{ key = "value" }` | a map (any other keys) |

A float with no fractional part renders as an integer (`2.0` → `2`).

> **An empty Lua table is a map, not an empty array.** The array test is "the keys are exactly `1..n`", and an empty table has no keys. So `{}` becomes an empty **map**: it is falsy for `v-if` (good), but **`v-for` over it fails** with *"v-for: `list` is not an array"*. If a list may legitimately be empty, either keep one entry and filter it in the template, or pass a flag alongside it.

### What the template language does *not* have

There is **no expression evaluation**. No arithmetic, no comparisons, no function calls, no filters, no string methods. `{{a + b}}` renders the literal text `a + b` — the path lookup simply fails and produces an empty string, because `a + b` is not a valid path.

There is also no `v-else`, `v-else-if`, `v-show`, `v-bind`, `v-model`, `v-html`, no includes or partials, and no built-in loop index.

Logic belongs in the mod; the template only does lookups, conditionals and repetition.

---

## Actions

An element reports back to the server through its action attributes.

| Attribute | Fires on |
|---|---|
| `on-click` | Primary (left) click. |
| `on-secondary-click` (or `on-right-click`) | Secondary (right) click. |
| `on-pointer-enter` | The pointer enters the element. |
| `on-pointer-leave` | The pointer leaves the element. |

> `on-pointer-left` appears in a source comment but is **not** implemented — only `on-pointer-leave` is accepted.

The value is a **namespaced action id**: `"namespace:action"`.

```xml
<button id="close" on-click="inventory:close" bg="#ff4d6d">✕</button>
<div id="slot_{{s.index}}" on-click="inventory:click" data="{{s.index}}">
```

### The `data` attribute

`data` is forwarded **verbatim** in the action payload. It is how the server knows *which* slot was clicked:

```xml
<div id="slot_{{s.index}}" on-click="inventory:click" data="{{s.index}}">
```

```lua
api.on("ui_action", function(ev)
    if ev.ui_id ~= "inv" then return end
    if ev.action == "inventory:click" then
        local index = tonumber(ev.data)
        -- apply the click to slot `index`
    end
end)
```

### Modifier keys

The client automatically appends the held modifier keys and mouse buttons to the data string, space-separated, **before** your custom `data`:

```
ctrl shift alt super left right middle <your data>
```

Only the ones that are actually held appear. So a ctrl+click on a slot reports `data = "ctrl left 12"` where `12` is your `data="{{s.index}}"`.

### The reserved `client:close` action

```xml
<button on-click="client:close">Close</button>
```

This one is handled **locally** — the client unmounts the screen itself and nothing is sent to the server.

---

## The element id

Every action carries an `element_id`. If the element has an explicit `id`, that is used:

```xml
<div id="slot_{{s.index}}" ...>
```

If it does not, the client generates a **structural path** — `"#0.2.1"`, the chain of child indices from the root:

```
#0        the root
#0.1      its second child
#0.1.0    that child's first child
```

The paths are stable as long as the XML structure is stable, so they are usable as identifiers, but an explicit `id` is always better.

---

## Modal screens

```xml
<div id="inv" modal="true" ...>
```

A modal screen **captures input**: the mouse cursor is released from the window and gameplay input (movement, looking) is suspended while it is mounted.

`Escape` closes remote modals. `Escape` on a non-modal screen toggles the mouse grab instead.

> `modal` must be exactly the string `"true"`. `"1"` and `"yes"` do **not** count.

---

## The cursor widget

A `<cursor>` element is a subtree that follows the mouse pointer while it is present. It is hidden until its parent drag source is moved.

```xml
<cursor v-if="held_item" width="44" height="44" layout="column"
        bg="{{held_item.color}}" border="1" border-color="#ffffff"
        padding="4" align="center" justify="center" radius="8">
  <img v-if="held_item.img" src="{{held_item.img}}" alt="?" width="34" height="34"/>
  <label font-size="11" color="#ffffff">{{held_item.short}}</label>
  <label font-size="9" color="#ffffff">{{held_item.count}}</label>
</cursor>
```

This is how the inventory shows the stack under the cursor while dragging it.

---

## Drag and drop

| Attribute | Meaning |
|---|---|
| `draggable="true"` | The element can be picked up. |
| `droppable="true"` | The element can receive a drop. |
| `drag-group` | Only elements in the same group interact. Optional. |
| `drag-ghost-bg` | The background of the floating ghost while dragging. |

```xml
<div id="slotB" droppable="true" drag-group="test" bg="#333a" width="64" height="64">
  <div id="itemA" draggable="true" drag-group="test" data="A" bg="#a83"
       drag-ghost-bg="#fc5" width="48" height="48">A</div>
</div>
```

When a draggable element is dropped on a droppable one, the client sends a `UiDrag` message with the source id, the target id and the source's `data`.

If neither side specifies a `drag-group`, they match anything. If one side specifies a group, both must specify the **same** group.

The client only reports geometry — it never moves inventory items itself. The server decides what a drop means.

---

## Worked examples

### A status panel

```xml
<div id="status" anchor="top-right" layout="column" gap="4" padding="10"
     bg="#0f1220cc" border="1" border-color="#4a5a8a55" radius="10">
  <label font-size="14" color="#e6ecff">{{title}}</label>
  <label font-size="12" color="#a8b4e0">Health: {{health}}</label>
  <label font-size="12" color="#a8b4e0">Coins: {{coins}}</label>
  <button on-click="mymod:refresh" bg="#3a6ea5" hover-bg="#4d84c2"
          pressed-bg="#2c527d" padding="4 10" radius="6">
    <label font-size="12" color="#ffffff">Refresh</label>
  </button>
</div>
```

### A list with an empty state

```xml
<div id="quest_log" anchor="top-left" layout="column" gap="6" padding="12"
     bg="#0f1220cc" border="1" border-color="#4a5a8a55" radius="12" width="240">
  <label font-size="16" color="#e6ecff" font-weight="bold">Quests</label>

  <div v-for="q in quests" layout="column" gap="2" padding="6"
       bg="#1e2440" radius="6">
    <label font-size="13" color="#ffffff">{{q.title}}</label>
    <label font-size="11" color="#a8b4e0">{{q.description}}</label>
  </div>

  <label v-if="quests" font-size="11" color="#6f8f55">Done!</label>
</div>
```

> `v-if="quests"` is truthy for a **non-empty array**. An empty quest list is falsy, so "Done!" only shows when there is at least one quest.

### A grid of buttons

```xml
<div id="menu" anchor="center" layout="column" gap="12" padding="24"
     bg="#1a1f33ee" border="1" border-color="#6b7aa855" radius="14">
  <label font-size="22" color="#e6ecff" font-weight="bold">{{title}}</label>
  <div layout="grid" cols="3" gap="8">
    <button v-for="entry in entries" on-click="mymod:pick"
            data="{{entry.id}}" bg="#1e2440" border="1" border-color="#3a4670"
            width="96" height="40" radius="8" align="center" justify="center">
      <label font-size="13" color="#ffffff">{{entry.label}}</label>
    </button>
  </div>
</div>
```

---

## Limits

| Limit | Value | Why |
|---|---|---|
| XML size | 8 MiB | Sized for the creative tab (~1200 cells). |
| Tree depth | 32 levels | Structural sanity. |
| XML node count | 32 768 | `roxmltree` counts every element, text node and attribute. |
| Screens mounted | 32 at once | |
| `ui_id` / `element_id` length | 64 characters | |
| Action / data length | 256 characters | |
| `src` length | 128 characters | |
| `block#` / `item#` name length | 160 characters | |
| Image dimension | 4096 px | |
| Image uploads per frame | 4 | New icons trickle in over several frames. |

## Gotchas

**Structural problems are hard errors.** Bad XML, an unknown element, a too-deep tree or a too-large document means the screen is **not mounted** and any previously mounted version of that screen is kept.

**Attribute problems are warnings.** A typo or an unknown attribute is logged and the widget still spawns, so one bad attribute cannot blank a whole screen.

The rest, in the order you are likely to hit them:

- **DTDs are rejected** — no entity-expansion games.
- **The root must expand to exactly one node.** A `v-for` or `v-if` on the root element is an error.
- **`v-for` needs an array, and an empty Lua table is not one.** See [The data model](#the-data-model).
- **There is no `font-weight`.** It warns and does nothing.
- **`z-index` has no rendering effect.** The custom renderer draws in child order and never consults Bevy's `ZIndex` stack, so a higher `z-index` does not raise a widget above a later sibling.
- **`radius` is pixels only.** `radius="50%"` is silently ignored.
- **`overflow="auto"` behaves like `scroll`** on both axes.
- **Hover and pressed feedback need a `bg`.** A button without `bg` gets no hover or pressed color.
- **The modifier tokens are mixed into `data`.** If the server treats `data` as a verbatim value (a block name, say), a click that also carries a modifier key will not match. Scan for the token you need instead of comparing the whole string.
- **`client:close` is local-only.** The client unmounts the screen and nothing is sent, so the server's record of which screens are open is not updated, and the mod's own state is not updated either.
- **The `virtual`, `item-height` and `overscan` attributes are not wired up yet.** The style resolver understands them, but the parser does not forward them, so they are currently inert (and log an "unknown attribute" warning). See [scroll_virtualization_plan.md](../plans/scroll_virtualization_plan.md).
- **TypeScript mods get no template data.** `render_xml` in the TS backend renders with empty data, so a `v-for` or `v-if` over data fails there. Use the Lua backend for data-driven templates.
