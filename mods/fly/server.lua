-- One-shot effect: a burst that stays where it happened until it dies.
local aura_active = {}

api.on("player_join", function(_ev)
    api.register_keybind({
        id = "example:fly",
        key = "Space",
        description = "Fly burst",
        category = "Movement",
    })
    api.register_keybind({
        id = "example:aura",
        key = "KeyF",
        description = "Toggle aura",
        category = "Effects",
    })
    api.register_keybind({
        id = "example:aura_boost",
        key = "KeyQ",
        description = "Boost aura",
        category = "Effects",
    })
end)

api.on("keybind", function(ev)
    if not ev.pressed then return end

    if ev.id == "example:fly" then
        if not ev.position then return end
        api.broadcast_action("example:player_fly", {
            player_id = ev.player_id,
            position = ev.position,
            strength = 1.0,
        })
    elseif ev.id == "example:aura" then
        -- Continuous effect keyed by player: start it, or stop (with fade-out).
        if aura_active[ev.player_id] then
            aura_active[ev.player_id] = nil
            api.broadcast_action("example:aura_stop", { player_id = ev.player_id })
        else
            aura_active[ev.player_id] = true
            api.broadcast_action("example:aura_start", {
                player_id = ev.player_id,
                intensity = 1.0,
            })
        end
    elseif ev.id == "example:aura_boost" and aura_active[ev.player_id] then
        -- Updates the running instance in place; nothing new is spawned.
        api.broadcast_action("example:aura_update", {
            player_id = ev.player_id,
            intensity = 2.0,
        })
    end
end)
