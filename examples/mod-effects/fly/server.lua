api.on("player_join", function(_ev)
    api.register_keybind({
        id = "example:fly",
        key = "Space",
        description = "Fly burst",
        category = "Movement",
    })
end)

api.on("keybind", function(ev)
    if ev.id == "example:fly" and ev.pressed then
        if not ev.position then return end
        api.broadcast_action("example:player_fly", {
            player_id = ev.player_id,
            position = ev.position,
            strength = 1.0,
        })
    end
end)
