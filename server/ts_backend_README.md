# TypeScript mod backend

`server/src/ts_backend.rs` runs `mods/**/*.ts` next to the Lua mods
(`mods/**/*.lua`). Both are live at the same time: Lua keeps its directory and
its behaviour, TS gets its own plugin, and both push the same
`ScriptRequest` shape through the same translator (`scripting::apply_script_requests`).

## Pipeline per tick

1. `ts_reload_system` recompiles mods whose source changed (a failing mod keeps
   its last good build)
2. `ts_tick_system` (exclusive `fn(&mut World)`) snapshots player positions,
   registers the packed row buffer with every mod, installs the pull context
3. each mod's exported `on_tick` runs: fuel armed per call, traps logged
4. row edits are written back into the ECS (`apply_player_positions`)
5. `ts_drain_system` applies queued requests through the shared Lua translator

## Safety

- fuel kills runaway loops/recursion, traps return instead of unwinding
- rows are host memory behind bounds-checked accessors; a mod never holds a
  `&mut` component across a call
- region ids are per module and start at 1, so "no region" (0) is unambiguous
- layout cross-check: `check_row_layout(db)` compares `PlayerRow` against the
  `glam::Vec3` offsets exported by `just export-types`, so a drift on the Rust
  side is an error rather than a silent misread
