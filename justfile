default:
    #!/usr/bin/env bash
    cargo run --bin exsolis_server_bin &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT INT TERM
    cargo run --bin exsolis

# Export Rust type layouts from the debug-info of the server binary into
# types.tdb, which the server loads at startup (see server/src/typedb.rs).
# NOTE: the workspace dev profile is `debug = "line-tables-only"`, which has no
# type info in DWARF. Rebuild with full debuginfo so type layouts exist.
export-types:
    RUSTFLAGS="-C debuginfo=2" cargo build --bin exsolis_server_bin
    cargo run -p tesjit-dwarf -- target/debug/exsolis_server_bin -o types.tdb

# Focused comparison: native Bevy query vs BMS-driven `world.query` scripts.
# One workload everywhere: 10k entities, `x += v * dt` per step.
# (Raw Lua-loop runners in crates/lua-bench-* are for JIT loop work, not this
# table — they measure Lua tables, not Bevy queries.)
bench-scripting:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "== BMS backends (isolated graphs, one Lua backend per build) =="
    BMS_LUAJIT=$(cargo run --release --quiet --manifest-path crates/bms-bench/Cargo.toml --no-default-features --features bms-luajit)
    echo "$BMS_LUAJIT"
    BMS_LUAU=$(cargo run --release --quiet --manifest-path crates/bms-bench/Cargo.toml --no-default-features --features bms-luau)
    echo "$BMS_LUAU"
    BMS_LUA54=$(cargo run --release --quiet --manifest-path crates/bms-bench/Cargo.toml --no-default-features --features bms-lua54)
    echo "$BMS_LUA54"
    echo "== native rust / tesjit / bevy (criterion) =="
    CRIT=$(cargo bench -p tesjit --bench scripting -- ecs_step 2>&1 | grep -E "^ecs_step_10k/")
    echo "$CRIT"
    echo ""
    echo "== ecs_step, N=10000, x += v*dt: ns per row per step =="
    python3 - "$BMS_LUAJIT" "$BMS_LUAU" "$BMS_LUA54" "$CRIT" <<'EOF'
    import json, re, sys
    rows = []
    for doc in sys.argv[1:4]:
        d = json.loads(doc)
        rows.append((d["backend"], float(d["ecs_ns_per_row_per_step"])))
    for line in sys.argv[4].splitlines():
        m = re.match(r"ecs_step_10k/(\S+)\s+time:\s+\[.*?\s+([\d.]+)\s*(µs|ns|ms)", line)
        if m:
            name, val, unit = m.group(1), float(m.group(2)), m.group(3)
            scale = {"ns": 1.0, "µs": 1e3, "ms": 1e6}[unit]
            rows.append((name, val * scale / 10000.0))
    base = dict(rows).get("rust_vec", 1.0)
    print(f"{'backend':<16} {'ns/row':>10} {'vs native':>10}")
    for name, ns in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<16} {ns:>10.2f} {ns / base:>9.1f}x")
    EOF
