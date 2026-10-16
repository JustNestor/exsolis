//! Focused comparison: native Bevy query vs BMS-driven `world.query` from a
//! script, same workload everywhere (N rows, `x += v * dt`).
//!
//! Rust side defines two reflected components, spawns N entities, and loads a
//! script whose `bench()` integrates one step. Timing calls `bench()` directly
//! on the loaded context (upstream's own bench methodology). Correctness is
//! verified by reading positions back in Rust afterwards.
//!
//! Backend is selected by cargo feature (`bms-luajit` / `bms-luau` /
//! `bms-lua54`).
//!
//! Usage: `cargo run --release --manifest-path crates/bms-bench/Cargo.toml
//! --features bms-luajit` (etc.)

use std::time::{Duration, Instant};

use bevy_app::App;
use bevy_ecs::prelude::{Component, ReflectComponent};
use bevy_mod_scripting::bindings::{CurrentScriptAttachment, WorldExtensions};
use bevy_mod_scripting::core::pipeline::PipelineRun;
use bevy_mod_scripting::core::script::ScriptContexts;
use bevy_mod_scripting::lua::{LuaScriptingPlugin, mlua};
use bevy_mod_scripting::prelude::{BMSPlugin, ScriptAttachment, ScriptComponent};
use bevy_mod_scripting_world::WorldAccessGuard;
use bevy_reflect::Reflect;

#[cfg(feature = "bms-luajit")]
const BACKEND: &str = "bms_luajit_ecs";
#[cfg(feature = "bms-luau")]
const BACKEND: &str = "bms_luau_ecs";
#[cfg(feature = "bms-lua54")]
const BACKEND: &str = "bms_lua54_ecs";
#[cfg(not(any(
    feature = "bms-luajit",
    feature = "bms-luau",
    feature = "bms-lua54"
)))]
compile_error!("pick one backend feature: bms-luajit, bms-luau or bms-lua54");

#[derive(Component, Reflect)]
#[reflect(Component)]
struct Pos {
    x: f32,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct Vel {
    x: f32,
}

const DT: f64 = 0.016;
const WARMUP: u32 = 5;
const STEPS: u32 = 50;

/// Workload variants, selected by first CLI arg:
/// - `full` (default): integrate `x += v*dt` with field reads+writes.
/// - `iter`: build the query and iterate, empty body (upstream's shape).
/// - `build`: build the query only, no iteration.
fn workload() -> (String, usize) {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "full".to_owned());
    let n: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    (mode, n)
}

fn script_source(mode: &str) -> String {
    let body = match mode {
        "iter" => {
            "for _, _ in pairs(world.query():component(Pos):component(Vel):build()) do end"
                .to_owned()
        }
        "build" => "world.query():component(Pos):component(Vel):build()".to_owned(),
        _ => {
            r#"for _, result in pairs(world.query():component(Pos):component(Vel):build()) do
        local comps = result:components()
        local p = comps[1]
        local v = comps[2]
        p.x = p.x + v.x * 0.016
    end"#
                .to_owned()
        }
    };
    format!(
        r#"
local Pos = world.get_type_by_name("{}")
local Vel = world.get_type_by_name("{}")
function bench()
    {body}
end
"#,
        std::any::type_name::<Pos>(),
        std::any::type_name::<Vel>(),
    )
}

fn main() {
    let (mode, n) = workload();
    let asset_dir = std::env::temp_dir().join("bms-bench-assets");
    let asset_dir = asset_dir.to_string_lossy().into_owned();
    std::fs::create_dir_all(&asset_dir).expect("bench assets dir");
    std::fs::write(format!("{asset_dir}/bench.lua"), script_source(&mode))
        .expect("bench script written");

    let mut app = App::new();
    // Same headless plugin set as upstream's integration tests.
    app.add_plugins((
        bevy_app::TaskPoolPlugin::default(),
        bevy_app::ScheduleRunnerPlugin::default(),
        bevy_asset::AssetPlugin {
            file_path: asset_dir,
            ..Default::default()
        },
        bevy_time::TimePlugin,
        bevy_diagnostic::FrameCountPlugin,
        bevy_diagnostic::DiagnosticsPlugin,
        bevy_log::LogPlugin {
            level: bevy_log::Level::ERROR,
            ..Default::default()
        },
    ));
    app.add_plugins(BMSPlugin);
    app.register_type::<Pos>();
    app.register_type::<Vel>();

    // The workload entities, spawned from Rust like a real game would.
    for i in 0..n {
        app.world_mut()
            .spawn((Pos { x: i as f32 * 0.1 }, Vel { x: 1.0 }));
    }

    let handle = app
        .world()
        .resource::<bevy_asset::AssetServer>()
        .load::<bevy_mod_scripting::asset::ScriptAsset>("bench.lua");
    let script_id = handle.id();
    let t0 = Instant::now();
    loop {
        app.update();
        match app
            .world()
            .resource::<bevy_asset::AssetServer>()
            .load_state(script_id)
        {
            bevy_asset::LoadState::Loaded => break,
            bevy_asset::LoadState::Failed(e) => {
                panic!("bench script failed to load: {e}")
            }
            _ => {}
        }
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "bench script never loaded"
        );
    }
    let entity = app
        .world_mut()
        .spawn(ScriptComponent(vec![handle.clone()]))
        .id();
    let key = ScriptAttachment::EntityScript(entity, handle);

    // The component-sync systems turn the spawn into an attach event; this
    // needs real updates, the pipeline drain below only finishes queued work.
    let t0 = Instant::now();
    loop {
        app.update();
        let n = app
            .world_mut()
            .get_resource_or_init::<ScriptContexts<LuaScriptingPlugin>>()
            .read()
            .all_residents_len();
        if n > 0 {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "script context never created"
        );
    }
    app.update_until_all_scripts_processed::<LuaScriptingPlugin>();

    let contexts = app
        .world_mut()
        .get_resource_or_init::<ScriptContexts<LuaScriptingPlugin>>()
        .clone();
    let contexts = contexts.read();
    let ctxt_arc = contexts
        .get_context(&key)
        .expect("script context created");
    let ctxt = ctxt_arc
        .as_loaded()
        .expect("script context loaded");
    drop(contexts);
    let ctxt = ctxt.lock();

    let bench: mlua::Function = ctxt
        .globals()
        .get("bench")
        .expect("bench function defined");
    drop(ctxt);

    // `world.query()` inside the script resolves the world through the thread
    // container: install the guard exactly like upstream's benches do.
    let cache = <WorldAccessGuard as WorldExtensions>::setup_cache(
        app.world(),
        CurrentScriptAttachment(Some(key.clone())),
    );
    let guard = WorldAccessGuard::new_exclusive(app.world_mut(), cache);
    let total_ns = WorldAccessGuard::with_existing_static_guard(guard, |guard| {
        guard.set_current_attachment(key.clone());
        for _ in 0..WARMUP {
            bench.call::<()>(()).expect("bench runs");
        }
        let t0 = Instant::now();
        for _ in 0..STEPS {
            bench.call::<()>(()).expect("bench runs");
        }
        t0.elapsed().as_nanos() as f64
    });

    // Correctness (full mode only): every row advanced by
    // (WARMUP + STEPS) * DT * v (v == 1.0).
    if mode == "full" {
        let mut query = app.world_mut().query::<(&Pos, &Vel)>();
        let mut checked = 0;
        for (i, (p, _)) in query.iter(app.world()).enumerate() {
            let want = i as f32 * 0.1 + (f64::from(WARMUP + STEPS) * DT) as f32;
            assert!(
                (p.x - want).abs() < 1e-2,
                "row {i} drifted: got {}, want {want}",
                p.x
            );
            checked += 1;
        }
        assert_eq!(checked, n, "all rows visited");
    }

    println!(
        "{{\"backend\":\"{BACKEND}\",\"mode\":\"{mode}\",\"ecs_n\":{n},\"ecs_steps\":{STEPS},\
        \"ecs_ns_per_row_per_step\":{:.3}}}",
        total_ns / (f64::from(STEPS) * n as f64),
    );
}
