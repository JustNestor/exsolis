api.register_keybind({
    id = "example:fly",
    key = "Space",
    description = "Fly burst",
    category = "Movement",
})

api.on("keybind", function(ev)
    if ev.id == "example:fly" and ev.pressed then
        api.broadcast_action("example:player_fly", {
            player_id = ev.player_id,
            strength = 1.0,
        })
    end
end)
