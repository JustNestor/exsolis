-- Example mod for the entity API.
--
-- Small on purpose: it exists so the shape of the API is discoverable, not to
-- be useful.

-- Ids come back from every API that hands one out as a decimal *string*, not
-- a number: a Lua number cannot represent every u64, and the truncation would
-- be silent.
local function entity_id_of_newest_prop()
    local newest = nil
    for _, e in ipairs(api.list_entities()) do
        if e.kind == "prop" and (newest == nil or tonumber(e.id) > tonumber(newest.id)) then
            newest = e
        end
    end
    return newest
end

local function spawn_a_prop(player_id)
    local pos = api.get_player_pos(player_id)
    if not pos then return end

    -- `spawn_entity` returns nothing: the server allocates ids so a mod cannot
    -- pick a colliding one. Find what you made with `list_entities`.
    api.spawn_entity("prop", {
        x = pos.x + 2,
        y = pos.y,
        z = pos.z,
        model = "models/duck.glb",
        scale = 1.0,
    })
end

local function spawn_a_chest(player_id)
    local pos = api.get_player_pos(player_id)
    if not pos then return end

    -- A block entity needs an actual container block underneath it, so place
    -- one first and then ask for the entity that owns it.
    local bx = math.floor(pos.x) + 3
    local by = math.floor(pos.y)
    local bz = math.floor(pos.z)

    -- Block names (see exsolis_common::BlockRegistry::default_registry).
    api.set_block(bx, by, bz, "exsolis:chest")

    local id = api.get_block_entity(bx, by, bz)
    if id then
        print("chest entity is " .. id .. " at " .. bx .. "," .. by .. "," .. bz)
    else
        print("no block entity at " .. bx .. "," .. by .. "," .. bz)
    end
end

local function move_the_prop(player_id)
    local prop = entity_id_of_newest_prop()
    if not prop then
        print("no props to move")
        return
    end
    local pos = api.get_player_pos(player_id)
    if not pos then return end
    api.set_entity_state(prop.id, { x = pos.x, y = pos.y, z = pos.z, yaw = 0 })
end

local function report(player_id)
    local by_kind = {}
    for _, e in ipairs(api.list_entities()) do
        by_kind[e.kind] = (by_kind[e.kind] or 0) + 1
    end
    local parts = {}
    for kind, n in pairs(by_kind) do
        parts[#parts + 1] = kind .. "=" .. n
    end
    table.sort(parts)
    print(player_id .. ": " .. #parts .. " kinds alive (" ..
        table.concat(parts, ", ") .. ")")
end

local function clear_props()
    for _, e in ipairs(api.list_entities()) do
        if e.kind == "prop" then
            api.despawn_entity(e.id)
        end
    end
end

api.on("player_join", function(ev)
    api.register_keybind({
        player_id = ev.player_id,
        id = "entities:spawn",
        key = "KeyP",
        description = "Spawn entities",
        category = "Entities",
    })
    api.register_keybind({
        player_id = ev.player_id,
        id = "entities:move",
        key = "KeyO",
        description = "Move the newest prop onto the player",
        category = "Entities",
    })
    api.register_keybind({
        player_id = ev.player_id,
        id = "entities:report",
        key = "KeyI",
        description = "Report live entities",
        category = "Entities",
    })
    api.register_keybind({
        player_id = ev.player_id,
        id = "entities:clear",
        key = "Delete",
        description = "Despawn every prop",
        category = "Entities",
    })
end)

api.on("keybind", function(ev)
    if not ev.pressed then return end

    if ev.id == "entities:spawn" then
        spawn_a_prop(ev.player_id)
        spawn_a_chest(ev.player_id)
    elseif ev.id == "entities:move" then
        move_the_prop(ev.player_id)
    elseif ev.id == "entities:report" then
        report(ev.player_id)
    elseif ev.id == "entities:clear" then
        clear_props()
    end
end)