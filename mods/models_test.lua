-- TEMPORARY GLTF spawn test (see GLTF_INTEGRATION_PLAN.md).
-- Spawns the test models next to each joining player.
-- Duck is ~1 unit; the avocado sample is modeled in meters, hence scale 10.

api.on("player_join", function(ev)
    api.send(ev.player_id, "spawn_model", 1, "models/duck.glb", 2.0, 27.0, 0.0, 1.5)
    api.send(ev.player_id, "spawn_model", 2, "models/avocado.glb", -3.0, 27.0, 3.0, 10.0)
    api.send(ev.player_id, "spawn_model", 3, "models/dragon/result.glb", 7.0, 27.0, 0.0, 1.0)
    api.send(ev.player_id, "spawn_model", 4, "models/girl/result.glb", -8.0, 27.0, -2.0, 1.0)
end)
