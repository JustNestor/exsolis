-- Player inventory policy layered on the reusable generic Container.
-- Kept in Lua so item rules and future equipment/crafting slots can evolve
-- independently from the wire protocol and entity storage.
PlayerInventory = PlayerInventory or {}
PlayerInventory.players = PlayerInventory.players or {}
PlayerInventory.size = 27
PlayerInventory.max_stack = 64

function PlayerInventory.create(player_id)
    local inventory = { container = Container.new(PlayerInventory.size), selected = 0 }
    PlayerInventory.players[player_id] = inventory
    return inventory
end

function PlayerInventory.get(player_id)
    return PlayerInventory.players[player_id] or PlayerInventory.create(player_id)
end

function PlayerInventory.remove_player(player_id)
    PlayerInventory.players[player_id] = nil
end

function PlayerInventory.selected_item(player_id)
    local inventory = PlayerInventory.get(player_id)
    return Container.get(inventory.container, inventory.selected)
end

function PlayerInventory.add(player_id, item_id, amount)
    local inventory = PlayerInventory.get(player_id)
    local container = inventory.container
    for index, item in Container.each(container) do
        if amount <= 0 then break end
        if item and item.id == item_id and item.count < PlayerInventory.max_stack then
            local added = math.min(amount, PlayerInventory.max_stack - item.count)
            item.count = item.count + added
            amount = amount - added
        end
    end
    for index, item in Container.each(container) do
        if amount <= 0 then break end
        if not item then
            local added = math.min(amount, PlayerInventory.max_stack)
            Container.set(container, index, { id = item_id, count = added })
            amount = amount - added
        end
    end
    return amount
end

function PlayerInventory.remove_one(player_id, item_id)
    local inventory = PlayerInventory.get(player_id)
    local slot = inventory.selected
    local item = Container.get(inventory.container, slot)
    if not item or item.id ~= item_id then return false end
    item.count = item.count - 1
    if item.count <= 0 then Container.set(inventory.container, slot, nil) end
    return true
end

function PlayerInventory.swap(player_id, first, second)
    return Container.swap(PlayerInventory.get(player_id).container, first, second)
end
