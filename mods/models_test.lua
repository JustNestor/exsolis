-- TEMPORARY GLTF spawn test (see GLTF_INTEGRATION_PLAN.md).
-- Spawns the test models next to each joining player.
-- Scales normalize measured model bounds to ~2-3 world units:
-- duck spans 165u, avocado 0.04u, dragon 51u, girl 2.7u.

api.on("player_join", function(ev)
    -- api.send(ev.player_id, "spawn_model", 1, "models/duck.glb", 4.0, 27.0, 0.0, 0.02)
    -- api.send(ev.player_id, "spawn_model", 2, "models/avocado.glb", -4.0, 26.0, 3.0, 50.0)
    -- api.send(ev.player_id, "spawn_model", 3, "models/dragon/result.glb", 10.0, 27.0, 0.0, 0.5)
    api.send(ev.player_id, "spawn_model", 4, "models/girl/result.glb", -10.0, 28.0, -4.0, 1.0)
    
    -- api.send(ev.player_id, "spawn_model", 4, "models/labs.glb", -10.0, 30.0, -4.0, 1.0)
end)
