# Lua modding

Lua mods live in `mods/**/*.lua` and are the fastest way to change gameplay. They are dynamic, event-driven, and hot-reloaded: edit a file and the server picks it up on the next tick.

The Lua bridge lives in `server/src/scripting.rs`. Its design constraints are stated at the top of that file and are worth understanding before you write anything:

> - one long-lived `Lua` VM, ticked once per `Update`
> - mods subscribe to a fixed set of Bevy `Message` types by name (`api.on("place_block", fn(ev) ... end)`) and get called with a plain Lua table per event, once per tick, in the order the engine emitted them
> - mods call back into the engine via a small, explicit `api.*` surface that queues a request; a Bevy system drains the queue after the VM tick and turns it into real ECS messages / `OutgoingPacket`s

Three rules follow from that:

1. **A mod never gets a live reference into the world.** No `&mut World`, no `Entity`, no Rust type. Every value crossing the boundary is a number, a string, a bool, or a flat table of those.
2. **A mod callback never mutates the ECS directly.** It reads a table and calls `api.*`. The translation from "mod asked for X" to "ECS mutation" happens entirely in `drain_script_requests`, in plain Rust, after the Lua call has returned.
3. **Hot paths are pull, not push.** There is no `on("player_moved")` firing 20×/second to every mod whether it cares or not. Only discrete, low-frequency events are pushed; a mod that wants per-tick behaviour registers `api.on_tick` and pulls what it needs from inside it.

## Writing a mod

```lua
-- mods/hello.lua
api.on("player_join", function(ev)
    api.send(ev.player_id, "chat", "Hello, " .. ev.player_id .. "!")
end)
```

`api.on(event, fn)` registers a handler. `api.on_tick(fn)` registers a per-tick handler that takes no arguments.

## Events

These are the events a mod can subscribe to with `api.on`:

| Event | Payload | Fired when |
|---|---|---|
| `player_join` | `{ player_id }` | A player connected. |
| `player_leave` | `{ player_id }` | A player disconnected. |
| `place_block` | `{ player_id, x, y, z, block, block_id }` | A player placed a block. `block` is the block **name**, `block_id` the numeric id. |
| `break_block` | `{ player_id, x, y, z, block, block_id }` | A player broke a block. |
| `select_hotbar` | `{ player_id, index }` | A player changed their hotbar slot. |
| `keybind` | `{ player_id, id, pressed, position }` | A keybind changed state. `position` is `{ x, y, z }` — the server's record of the player's position. |
| `ui_action` | `{ player_id, ui_id, element_id, action, data }` | A UI widget was activated. |
| `ui_drag` | `{ player_id, ui_id, source_id, target_id, data }` | A draggable widget was dropped on a droppable one. |

`player_id` is a **string** (the decimal id), because Lua numbers cannot represent every `u64`.

Handlers are called in registration order. **An error in one handler is logged and does not stop the other handlers or the server** — a bug in one mod's script must never take the game down.

## The `api.*` surface

### Messaging

| Function | Meaning |
|---|---|
| `api.send(player_id, kind, ...)` | Send a message to one player. Extra arguments are scalars. |
| `api.broadcast(kind, ...)` | Send a message to every player. |
| `api.broadcast_except(player_id, kind, ...)` | Send to everyone except one player. |
| `api.broadcast_action(id, payload)` | Send a reliable, ordered `ModAction` packet — the visual-effects protocol. |
| `api.broadcast_action_unreliable(id, payload)` | The unreliable variant, for frequent updates the next one supersedes. |

`api.broadcast_action` takes a **namespaced** id (`"example:player_fly"`) and a JSON-ish table payload. Nesting, value count, id length and encoded payload size are all bounded. Player ids should be passed as **strings**, because Lua numbers cannot represent every `u64`.

### The world

| Function | Meaning |
|---|---|
| `api.get_block(x, y, z)` | The block id at a position (0 if the chunk is not loaded). |
| `api.set_block(x, y, z, block)` | Place a block. `0` is air. |
| `api.blocks()` | The list of every registered block name. |
| `api.block_id(name)` | The numeric id for a block name, or nil. |
| `api.is_placeable(name)` | `false` for items and multiblock parts. |
| `api.get_block_entity(x, y, z)` | The block entity at a position, if any. |

### Players and entities

| Function | Meaning |
|---|---|
| `api.get_player_pos(player_id)` | The server's authoritative position of a player. |
| `api.list_players()` | The connected player ids. |
| `api.list_entities()` | The live entities. |
| `api.spawn_entity(kind, opts)` | Spawn an entity. `kind` is `"player"`, `"block"`, `"item"` or `"prop"`; `opts` may carry `x`, `y`, `z`, `yaw`, `scale` and `model`. |
| `api.despawn_entity(entity_id)` | Remove one. |
| `api.set_entity_state(entity_id, opts)` | Move one. `opts` must carry finite `x`, `y`, `z` and `yaw`. |

### The player inventory

| Function | Meaning |
|---|---|
| `api.get_inventory(player_id)` | The authoritative snapshot: `{ slots = {...}, selected = n, held = {...} }`. |
| `api.stash_held(player_id)` | Return the cursor stack to storage. |
| `api.give_block(player_id, name, count)` | Give a block to a player. |

The inventory is **Rust-authoritative**. Lua renders a snapshot that is written before the Lua tick, and Lua owns no items itself — clicks, gives and held stacks all mutate server-side. That is what guarantees the E-screen, the HUD and placement can never disagree.

### UI

| Function | Meaning |
|---|---|
| `api.render_xml(file_or_template, data)` | Expand a template to client-ready XML. |
| `api.load_xml(file)` | Read a template file verbatim. `..` is rejected. |
| `api.register_keybind(config)` | Register a keybind on the client. |

`api.render_xml` accepts either a file name (resolved under `assets/xml`, `.xml` appended if missing) or an inline template string starting with `<`. See [the XML UI guide](ui/xml-ui.md) for the template language.

`api.register_keybind` takes a table:

```lua
api.register_keybind({
    player_id = ev.player_id,
    id = "inv:toggle",
    key = "KeyE",
    description = "Toggle Inventory",
    category = "Inventory",
})
```

### Keybinds

A keybind handler receives the key event:

```lua
api.on("keybind", function(ev)
    if ev.id == "inv:toggle" and ev.pressed then
        -- do the thing
    end
end)
```

`ev.pressed` is true on the press and false on the release. `ev.position` is the server's current position of that player.

## A complete example — the inventory screen

`mods/main.lua` is the real, working example. Its shape is:

```lua
local IsInvOpen = {}

local function item_view(item)
    if not item then return nil end
    -- Items render flat (`item#`), blocks isometric (`block#`).
    local prefix = "block#"
    if item.name and not api.is_placeable(item.name) then prefix = "item#" end
    return {
        id = item.id,
        count = item.count,
        short = ...,
        color = "#6f8f55",
        img = item.name and (prefix .. item.name) or nil,
    }
end

local function inv_data(player_id)
    local inv = api.get_inventory(player_id)
    if not inv then inv = { slots = {}, selected = 0 } end
    local slots = {}
    for i = 9, 35 do                       -- main storage (Lua is 1-based)
        slots[#slots + 1] = { index = i, item = item_view(inv.slots[i + 1]) }
    end
    local slots2 = {}                      -- the hotbar
    for i = 0, 8 do
        slots2[#slots2 + 1] = {
            index = i,
            item = item_view(inv.slots[i + 1]),
            selected_bg = inv.selected == i and "#b5b5b5" or "#777777",
        }
    end
    local creative = {}                    -- the creative grid
    for _, name in ipairs(api.blocks()) do
        ...
    end
    return {
        title = "Inventory",
        slots = slots,
        slots2 = slots2,
        selected = inv.selected,
        held_item = item_view(inv.held),
        creative = creative,
    }
end

local function show_inv(player_id)
    local xml = api.render_xml("inventory", inv_data(player_id))
    api.send(player_id, "ui_show", "inv", xml)
    IsInvOpen[player_id] = true
end
```

Note the creative grid: it is built from `api.blocks()` — the **server's** registry — so the tab always shows exactly what the server registered, including mod blocks.

## Error handling

- A handler that throws is logged (`warn!`) and skipped; other handlers still run.
- A mod file that fails to parse is logged (`error!`) and skipped; the other mods still load.
- A `api.*` call with a bad argument raises a Lua error inside the handler, which is caught by the same mechanism.

There is no fuel budget for Lua — it is an interpreter, and the Lua VM is bounded by the server's process. Use the TypeScript backend when you need a hard CPU bound or typed ECS access.

## Lua vs TypeScript

| | Lua | TypeScript |
|---|---|---|
| Speed | Interpreted | Compiled to native code |
| ECS access | None — only the `api.*` surface | Typed `Query` / `Res` / `Mut` / `Commands` |
| CPU bound | No | Yes — fuel budget, traps |
| Getting started | Easier | Every boundary needs a type annotation |
| Use it for | Gameplay rules, UI, events | Per-tick systems, data processing |

Both backends run **at the same time** and share the same request translator, so a Lua mod and a TypeScript mod can cooperate freely.
