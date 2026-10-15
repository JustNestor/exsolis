local SLOT_COUNT = 36
IsInvOpen = IsInvOpen or {}

-- Inventory rendering reads the Rust authority through
-- `api.get_inventory()` (snapshot written before Lua renders). Lua owns no
-- items: clicks, gives and held stacks all mutate server-side, so the
-- E-screen, the HUD and placement always agree.

local function item_view(item)
    if not item then return nil end
    local name = item.name
    -- Items render flat (`item#`), blocks isometric (`block#`).
    local prefix = "block#"
    if name and not api.is_placeable(name) then prefix = "item#" end
    local short = "?"
    if name then
        local base = string.match(name, "[^:]+$") or name
        short = string.upper(string.sub(base, 1, 1))
    end
    return {
        id = item.id,
        count = item.count,
        short = short,
        color = "#6f8f55",
        img = name and (prefix .. name) or nil,
    }
end

local function inv_data(player_id)
    local inv = api.get_inventory(player_id)
    if not inv then
        inv = { slots = {}, selected = 0 }
    end
    local selected = inv.selected or 0
    local slots = {}
    local slots2 = {}
    -- Main storage follows the 9 hotbar slots (Lua 1-based).
    for i = 9, 35 do
        slots[#slots + 1] = { index = i, item = item_view(inv.slots[i + 1]) }
    end
    for i = 0, 8 do
        slots2[#slots2 + 1] = {
            index = i,
            item = item_view(inv.slots[i + 1]),
            selected_bg = selected == i and "#b5b5b5" or "#777777",
        }
    end
    -- Creative grid: every registered block, wheel-scrollable rows of 9.
    -- Clicks and gives apply server-side; Lua only re-renders.
    local creative = {}
    for _, name in ipairs(api.blocks()) do
        local last = creative[#creative]
        if not last or #last.cells >= 9 then
            last = { cells = {} }
            creative[#creative + 1] = last
        end
        local prefix = "block#"
        if not api.is_placeable(name) then prefix = "item#" end
        last.cells[#last.cells + 1] = { name = name, img = prefix .. name }
    end
    return {
        title = "Inventory",
        slots = slots,
        slots2 = slots2,
        selected = selected,
        held_item = item_view(inv.held),
        creative = creative,
    }
end

local function show_inv(player_id)
    local xml = api.render_xml("inventory", inv_data(player_id))
    api.send(player_id, "ui_show", "inv", xml)
    IsInvOpen[player_id] = true
end

local function hide_inv(player_id)
    api.send(player_id, "ui_hide", "inv")
    IsInvOpen[player_id] = false
end

local function close_inv(player_id)
    api.stash_held(player_id)
    hide_inv(player_id)
end

local function toggle_inv(player_id)
    if IsInvOpen[player_id] then
        close_inv(player_id)
    else
        show_inv(player_id)
    end
end

-- Реєстрація кейбінду та ініціалізація стану при підключенні гравця
api.on("player_join", function(ev)
    api.register_keybind({
        player_id = ev.player_id,
        id = "inv:toggle",
        key = "KeyE",
        description = "Toggle Inventory",
        category = "Inventory"
    })

    IsInvOpen[ev.player_id] = false
end)

api.on("player_leave", function(ev)
    IsInvOpen[ev.player_id] = nil
end)



api.on("keybind", function(ev)
    if ev.id == "inv:toggle" and ev.pressed then
        toggle_inv(ev.player_id)
    end
end)

api.on("ui_action", function(ev)
    if ev.ui_id ~= "inv" then return end
    if ev.action == "inventory:close" then
        close_inv(ev.player_id)
    elseif ev.action == "inventory:click" or ev.action == "inventory:creative" then
        -- Applied server-side before this dispatch; render the snapshot.
        if IsInvOpen[ev.player_id] then show_inv(ev.player_id) end
    end
end)

api.on("select_hotbar", function(ev)
    if IsInvOpen[ev.player_id] then show_inv(ev.player_id) end
end)

-- Place/break need no Lua mirror: the next render reads Rust state.
-- Reshow while open so counts update live.
api.on("place_block", function(ev)
    if IsInvOpen[ev.player_id] then show_inv(ev.player_id) end
end)

api.on("break_block", function(ev)
    if IsInvOpen[ev.player_id] then show_inv(ev.player_id) end
end)
