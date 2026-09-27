local SLOT_COUNT = 27
Inventories = Inventories or {}

local function ensure(player_id)
    if not Inventories[player_id] then
        Inventories[player_id] = {}
        for i = 0, SLOT_COUNT - 1 do
            Inventories[player_id][i] = nil
        end
        Inventories[player_id][0] = { id = "wood", count = 12 }
        Inventories[player_id][2] = { id = "stone", count = 3 }
    end
    return Inventories[player_id]
end

local function inv_data(player_id)
    local inv = ensure(player_id)
    local slots = {}
    for i = 0, SLOT_COUNT - 1 do
        -- Lua arrays are 1-based for template v-for lists
        slots[i + 1] = {
            index = i,
            item = inv[i], -- nil → v-if="s.item" hides the child
        }
    end
    return {
        title = "Inventory",
        slots = slots,
    }
end

local function show_inv(player_id)
    local xml = api.render_xml("inventory", inv_data(player_id))
    api.send(player_id, "ui_show", "inv", xml)
end

api.on("break_block", function(ev)
    show_inv(ev.player_id)
end)

api.on("place_block", function(ev)
    api.send(ev.player_id, "ui_hide", "inv")
end)

api.on("ui_drag", function(ev)
    if ev.ui_id ~= "inv" then return end

    local from = tonumber(ev.data) or tonumber(string.match(ev.source_id or "", "^item_(%d+)$"))
    local to = tonumber(string.match(ev.target_id or "", "^slot_(%d+)$"))
    if from == nil or to == nil or from == to then return end
    if from < 0 or from >= SLOT_COUNT or to < 0 or to >= SLOT_COUNT then return end

    local inv = ensure(ev.player_id)
    inv[from], inv[to] = inv[to], inv[from]
    show_inv(ev.player_id)
end)
