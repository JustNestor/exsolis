default:
    #!/usr/bin/env bash
    cargo run --bin exsolis_server_bin &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT INT TERM
    cargo run --bin exsolis
