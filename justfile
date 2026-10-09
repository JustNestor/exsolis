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
