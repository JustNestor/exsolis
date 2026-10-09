local SLOT_COUNT = 36
IsInvOpen = IsInvOpen or {}

local function ensure(player_id)
    if not PlayerInventory.players[player_id] then
        local inventory = PlayerInventory.create(player_id)
        Container.set(inventory.container, 0, { id = "wood", count = 12 })
        Container.set(inventory.container, 2, { id = "stone", count = 3 })
    end
    return PlayerInventory.get(player_id)
end

local function item_view(item)
    if not item then return nil end
    local short_names = { wood = "W", stone = "S" }
    local colors = { wood = "#8a5a32", stone = "#777777" }
    return {
        id = item.id,
        count = item.count,
        short = short_names[item.id] or string.upper(string.sub(item.id, 1, 1)),
        color = colors[item.id] or "#6f8f55",
    }
end

local function return_held_item(player_id)
    local inventory = ensure(player_id)
    local held = inventory.held_item
    if not held then return end
    local remaining = PlayerInventory.add(player_id, held.id, held.count)
    if remaining == 0 then
        inventory.held_item = nil
    else
        held.count = remaining
    end
end

local function inv_data(player_id)
    local inv = ensure(player_id).container
    local selected = ensure(player_id).selected
    local slots = {}
    local slots2 = {}
    -- The 27 main-inventory slots follow the 9 hotbar slots in storage.
    for i = 0, 26 do
        local index = i + 9
        slots[i + 1] = {
            index = index,
            item = item_view(Container.get(inv, index)),
        }
    end
    for i = 0, 8 do
        local item = Container.get(inv, i)
        slots2[i + 1] = {
            index = i,
            item = item_view(item),
            selected_bg = selected == i and "#b5b5b5" or "#777777",
        }
    end
    return {
        title = "Inventory",
        slots = slots,
        slots2 = slots2,
        selected = selected,
        held_item = item_view(ensure(player_id).held_item),
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
    return_held_item(player_id)
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

    ensure(ev.player_id)
    IsInvOpen[ev.player_id] = false
end)

api.on("player_leave", function(ev)
    IsInvOpen[ev.player_id] = nil
    PlayerInventory.remove_player(ev.player_id)
end)



api.on("keybind", function(ev)
    if ev.id == "inv:toggle" and ev.pressed then
        toggle_inv(ev.player_id)
    end
end)

local function click_slot(player_id, index, right_click)
    if index == nil or index < 0 or index >= SLOT_COUNT then return end
    local inventory = ensure(player_id)
    local container = inventory.container
    local slot = Container.get(container, index)
    local held = inventory.held_item

    if not held then
        if not slot then return end
        if right_click then
            local taken = math.floor((slot.count + 1) / 2)
            inventory.held_item = { id = slot.id, count = taken }
            slot.count = slot.count - taken
            if slot.count == 0 then Container.set(container, index, nil) end
        else
            inventory.held_item = slot
            Container.set(container, index, nil)
        end
    elseif not slot then
        if right_click then
            Container.set(container, index, { id = held.id, count = 1 })
            held.count = held.count - 1
        else
            Container.set(container, index, held)
            inventory.held_item = nil
        end
        if inventory.held_item and inventory.held_item.count == 0 then
            inventory.held_item = nil
        end
    elseif slot.id == held.id and slot.count < PlayerInventory.max_stack then
        local amount = right_click and 1 or math.min(held.count, PlayerInventory.max_stack - slot.count)
        slot.count = slot.count + amount
        held.count = held.count - amount
        if held.count == 0 then inventory.held_item = nil end
    elseif not right_click then
        Container.set(container, index, held)
        inventory.held_item = slot
    else
        return
    end

    show_inv(player_id)
end

api.on("ui_action", function(ev)
    if ev.ui_id ~= "inv" then return end
    if ev.action == "inventory:close" then
        close_inv(ev.player_id)
    elseif ev.action == "inventory:click" then
        local payload = ev.data or ""
        local index = tonumber(string.match(payload, "(%d+)$"))
        local right_click = string.find(payload, "right", 1, true) ~= nil
        click_slot(ev.player_id, index, right_click)
    end
end)

api.on("select_hotbar", function(ev)
    if ev.index >= 0 and ev.index < 9 then
        PlayerInventory.get(ev.player_id).selected = ev.index
        if IsInvOpen[ev.player_id] then show_inv(ev.player_id) end
    end
end)
