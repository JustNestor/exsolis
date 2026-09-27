api.on("break_block", function(ev)
    api.send(ev.player_id, "ui_show","test", api.load_xml("ui.xml"))
end
)

api.on("place_block", function(ev)
    api.send(ev.player_id, "ui_hide", "test")
end
)

api.on("ui_drag", function(ev)
    -- ev.player_id, ev.ui_id, ev.source_id, ev.target_id, ev.data
    -- e.g. rebuild XML reflecting the new location of the item and
    -- api.send(ev.player_id, "ui_show", ev.ui_id, new_xml)
    api.send(ev.player_id, "ui_show", ev.ui_id, api.load_xml("ui2"))
end)
