# UI Update Strategy — Incremental Screen Updates (proposal)

- **Status:** Proposal (draft)
- **Date:** 2026-11-14
- **Scope:** Server-driven UI (`ServerMessage::UiShow` → client `UiManager::show` → `parse` → `style` → `spawn` → `render_bridge`)
- **Related:** [`ui/xml-ui-internals.md`](../ui/xml-ui-internals.md), [`ui/xml-ui.md`](../ui/xml-ui.md), [`plans/scroll_virtualization_plan.md`](scroll_virtualization_plan.md)

---

## TL;DR

Today, **every** inventory state change re-renders the whole screen and re-mounts the whole UI tree:

```
slot click
 → ClientMessage::UiAction
 → server: apply click (creative::apply_inventory_clicks)
 → Lua: api.render_xml("inventory", inv_data(pid))   // read file + parse + v-for-expand + serialize
 → api.send(pid, "ui_show", "inv", xml)              // ServerMessage::UiShow { id, xml }  (reliable ordered)
 → client: UiManager::show → parse → hide(old) → spawn(new)   // full unmount + remount
 → thousands of widgets re-parsed / re-styled / re-spawned / re-probed
```

The user-visible cost: with a few thousand items on screen (the creative tab alone is ~1200 block cells and sits close to the `MAX_XML_NODES` / `MAX_XML_BYTES` budgets), "add hovering item / change slot / reselect hotbar" pays the full render + full remount tax every time.

**Proposal:** keep the wire format and the server-side template engine exactly as they are, and make the **client** diff the newly parsed screen against the currently mounted one, patching in place instead of always unmounting + remounting. Structural changes (widgets appearing/disappearing) fall back to a full remount; identical screens become a no-op.

This delivers ~95% of the benefit of the fancier "send template + data, client builds the UI" idea with a fraction of the risk.

## 1. The cost we are trying to remove

| Stage | What happens per click today | Cost |
|---|---|---|
| Server (Lua mod) | `mods/main.lua::show_inv` re-runs `inv_data` → `api.render_xml` | `server/src/ui_template.rs::render_file` re-reads `assets/xml/*.xml` from disk, re-parses with roxmltree, re-expands `v-for`, re-serializes. Cheap-ish for the file itself, but multiplied by the whole creative grid on every click. |
| Server → client | Full XML string over reliable-ordered UDP (`UiShow`, cap `UI_MAX_XML_BYTES` = 8 MiB, sized for the creative tab) | One big payload per click (~100s of KB with the creative grid). |
| Client | `client/src/ui/mod.rs::UiManager::show` → `parse::parse_ui` → `spawn::spawn_ui` | Always `hide(old)` + `spawn(new)`: full despawn + full re-create of the entity tree, re-run `style::resolve` per widget, re-probe images, trickle uploads at 4/frame (`render_bridge::sync_ui_images`). |

Grounded in code:

- `ServerMessage::UiShow { id, xml }` / `UiHide` — `common/src/lib.rs:317`.
- `UI_MAX_XML_BYTES = 8 MiB`, `MAX_XML_NODES = 32 768` — `common/src/lib.rs:436`, `client/src/ui/parse.rs`.
- Client keeps the previous screen on parse error, replaces on success — `client/src/ui/mod.rs:94-118`.
- Scroll restoration for re-mounted screens — `client/src/ui/mod.rs:46-52, 285-340` (`scroll_memory`).
- Creative tab node-budget risk — `documentation/ui/xml-ui-internals.md:669-670`.

## 2. On the idea "send template and data, client makes the full UI"

**Assessment:** it's the right long-term direction (this is server-driven UI / SDUI in embryo), and the user-visible answer to "will it be user friendly" is *yes* — but it is not the right first step, and as proposed it has real problems:

1. **The template engine would need porting to the client.** The `v-for` / `v-if` / `{{…}}` expansion logic (`server/src/ui_template.rs:253-492`) would become a second source of truth next to the server copy (which has the unit tests). The client parser (`client/src/ui/parse.rs`) is strict XML → AST and is *not* an expansion engine.
2. **The wire format would need a structured data payload** (Lua table → `DataValue` → JSON-ish for the protocol). `DataValue` lives only in the server crate today.
3. **Error semantics move.** "Root must expand to exactly one node", "empty Lua table is a Map not an Array", strictness policy — all become client-side concerns.
4. **The TypeScript backend gap becomes load-bearing.** TS mods currently render with empty data — the known limitation (`documentation/ui/xml-ui.md:567`) would stop being an edge case and a blocker.

**Verdict:** adopt the *spirit* of the idea (server owns template + data; client renders), but deliver it incrementally via client-side diffing first. The incremental path gets the user-visible win (no re-mount per slot change) without re-architecting the engine, the wire format, or the mod API.

## 3. The proposal: incremental screen updates

### 3.1 Client side (the main win)

- `UiManager` keeps, per mounted `ui_id`, the last-mounted widget tree (`WidgetDef` root) alongside the entity root.
- On `ServerMessage::UiShow` for a screen that is **already mounted**: parse the new XML, diff old vs new trees.
  - **Equal** → no-op (don't touch entities at all).
  - **Patched** → update only what changed (attrs, text, handlers, `data`, `src`) via Bevy component updates on the existing entities; insert/remove children where structure changed. Entity identity is preserved, so hover/pointer/drag state stays coherent.
  - **Not safely diffable** (auto structural ids `#0.2.1`, deep structural rewrites) → fall back to the existing unmount + remount.
- The diff key is the widget's element id (`def.id` or auto structural path) — stable for `v-for`-generated ids because generated ids are deterministic (`slot_{{s.index}}` → `slot_12`).

### 3.2 Server side (cheap follow-up)

- Cache parsed templates per file (`assets/xml/{name}.xml` → `TplNode`) so per-click re-renders don't re-read/re-parse/re-expand from scratch; expansion still runs per data change, serialization is the remaining work.

### 3.3 Wire protocol

- **No change needed for phase 1.** `UiShow` already means "mount or atomically replace"; the client decides *replace* vs *patch* by diffing. Mods, `OpenScreens` tracking, and server-side validation stay untouched.

## 4. Why this is the right first move (and what it deliberately skips)

- It fixes the actual complaint — full re-render + full remount per slot change — with a contained change.
- It keeps **scripting convenient**: mods keep writing plain Lua (`api.render_xml` + `api.send(..., "ui_show", ...)`), the fast path is transparent, and zero new Lua API is required for the common case. Existing mods keep compiling/running unchanged.
- It preserves the server-authority model (client still only ever renders what the server sent; the client is not trusted with data it shouldn't see — diffs are computed over trees the client itself parsed from server input).

## 5. User-friendliness from the user's perspective

- **Should be neutral-to-better.** A full remount currently costs a hitch per click; with in-place patching the frame-1 cost drops to a handful of component updates. Perceived latency on "add hovering item / change slot" goes away rather than shifting elsewhere.
- Scroll: with patching, scroll containers keep their `ScrollPosition` naturally (better than today, where re-mount + `scroll_memory` restore is needed). When a full remount *is* required, the existing `scroll_memory` path already handles it.
- Input continuity: because entities are updated in place rather than despawned/re-spawned, hover styles, active drags (`UiDrag` state), and cursor overlays stay consistent by construction.

## 6. Phased rollout

1. **Phase 0 — client-side diff + patch.** Add mounted-tree cache + diff/patch in `client/src/ui/` (parse output already holds the full tree, so no parser changes needed). Unit tests on the diff. Fall back to remount when not safely diffable.
2. **Phase 1 — server-side template cache.** `render_file` → cached `TplNode` + per-render expand/serialize.
3. **Phase 2 — data-driven updates (the "template + data" end state).** Once Phase 0/1 are in, revisit sending template + data (`UiShow` carrying a template ref + data) as the long-term SDUI message. Only if the incremental path turns out insufficient.
4. **Phase 3 — TypeScript backend template data.** Removes the `render_xml`-with-empty-data limitation so data-driven templates work for TS mods too.

## 7. Open questions

- Element-id stability for auto-generated ids under heavy structural change — diff must fall back to remount then (cheap safety, rare path).
- In-flight drags when a patch removes a source/target widget — guard before applying structural removals.
- Image re-uploads on patch: registry is keyed by `src` and only fills unregistered entries, so patched screens with unchanged icons cost nothing; new icons trickle at 4/frame as before.

## 8. Recommendation

Go with **Phase 0** now (client-side incremental diff over the existing full-XML `UiShow`), paired with **Phase 1** (server template cache) as the small enabling follow-up. Revisit full template+data SDUI (Phase 2) after the incremental path has proven its numbers.

## 9. Status (2026-11-15)

Landed ahead of Phase 0, because these were cheap and unambiguous:

- **`ServerMessage::InventorySyncDiff`** (`common/src/lib.rs`) + `HotbarSyncLog`
  (`server/src/creative.rs`): the authoritative hotbar now sends only the
  changed slots (and only the selection when it moved) after a give / place /
  break / click / select. No change at all = no packet **and** no HUD
  re-render (`send_hud` is skipped; previously every sync re-sent the HUD).
  The first sync after join is still a full `InventorySync` baseline.
- **Client no-op on identical `UiShow`** (`client/src/ui/mod.rs`): a screen
  that is already mounted with byte-identical XML is left untouched — no
  re-parse, no despawn/respawn, so hover/scroll/drag state survives. A screen
  closed locally (`client:close`) has no cached XML and still re-opens.
- Diff unit tests in `server/src/creative.rs` (`hotbar_diff_*`) pin the
  selection-only, content-change, slot-emptied and no-change cases.

Still open (unchanged by the above): the inventory/creative screens a Lua mod
re-renders still re-send the **whole** XML and still re-mount the whole tree
on a real change. That is Phase 0's job.

