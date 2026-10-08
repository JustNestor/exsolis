# Asset system handoff

## Current state

Work is in progress in the common protocol, client, and server. The latest user direction is: persistent client asset cache across sessions; use HTTP(S) for asset bytes (Axum on the server, no UDP asset payloads); start the asset listener on demand and stop it after 30 seconds with no connected clients; preserve the generic handler design; continue implementation rather than limiting the work to an MVP.

Implemented so far:

- `common/src/lib.rs`: shared `AssetHash` and Renet manifest/request/download/preload/update messages.
- `server/src/assets.rs`: fastpak catalog, Axum ticket downloads, 30-second idle shutdown, metadata polling/hot-reload, and push/load helper functions.
- `server/src/events.rs`: player spawn waits for client asset-sync completion.
- `client/src/assets.rs`: persistent `.exsolis` pak cache, serialized background worker, HTTP(S) download, fastpak checksum verification/merge, extension-based resident handlers, preload commands, and a local syncing overlay.
- Client block textures and UI images read from the extracted cache and refresh on updates.
- `ASSETS.md` documents environment variables and HTTPS reverse-proxy setup.
- `.gitignore` now ignores `.exsolis/`.

## Next steps

1. Finish cache extraction recovery: when the pak is valid but an extracted file is missing, restore required manifest entries from the pak. Avoid extracting stale entries removed from the current manifest.
2. Review the edited files for API/type errors. `rustfmt` parsed the edited Rust files, but no build or tests have been run.
3. Inspect the worktree carefully. It had pre-existing staged/unstaged changes (including graft files and fastpak changes). `cargo fmt --all` also ran earlier, so check unrelated formatting changes and preserve user work.
4. Regenerate `Cargo.lock` for the new Axum, Tokio, reqwest, and bytes dependencies when approved/appropriate.
5. Build the workspace and fix compiler errors; then verify join sync, persistent-cache reuse, preload, hot reload, and HTTP idle shutdown.

## Important implementation notes

- HTTPS is supported through `EXSOLIS_ASSET_BASE_URL` pointing at a TLS reverse proxy; Axum currently serves plain HTTP behind it.
- Defaults: bind `0.0.0.0:5001`, cache `.exsolis/`, asset source `assets/`; see `ASSETS.md` for overrides.
- Checksum is fastpak xxh3-64 of decoded asset bytes. It detects accidental corruption, not malicious modification.
- `fastpack::merge` overlays deltas. GPU asset unloading is intentionally deferred; removed assets are retained for the process lifetime.
- Current hot-reload implementation scans asset metadata on the server and rebuilds the catalog synchronously when files change.
- Do not revert unrelated worktree changes when reviewing or cleaning up.
