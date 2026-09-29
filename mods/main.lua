local SLOT_COUNT = 27
Inventories = Inventories or {}
IsInvOpen = IsInvOpen or {}

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
    local slots2 = {}
    for i = 0, SLOT_COUNT - 1 do
        slots[i + 1] = {
            index = i,
            item = inv[i],
        }
    end
    for i = 0, 9 - 1 do
        slots2[i + 1] = {
            index = i,
        }
    end
    return {
        title = "Inventory",
        slots = slots,
        slots2=slots2,
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

local function toggle_inv(player_id)
    if IsInvOpen[player_id] then
        hide_inv(player_id)
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

    ensure(ev.player_id)
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
