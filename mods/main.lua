api.on("break_block", function(ev)
    api.send(ev.player_id, "ui_show","test", "<label font-size=\"22\" color=\"#ffffff\">SDUI dev panel</label>")
end
)
