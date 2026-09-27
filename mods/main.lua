api.on("break_block", function(ev)
    api.send(ev.player_id, "ui_show","test", api.load_xml("ui.xml"))
end
)

api.on("place_block", function(ev)
    api.send(ev.player_id, "ui_hide", "test")
end
)
