# XML Display on Items — ideas for the future

- **Status:** brainstorm — one idea, kept separate from the executable plan
- **Date:** 2026-11-17
- **Related:** [`plans/ecs-items-and-abilities-plan.md`](plans/ecs-items-and-abilities-plan.md)

---

## How to read this

This file is about a single idea: **items with a server-authored XML display that appears on hover.**

It has two halves:

- **Part I — My own view** is the vision and the reasons I care about it.
- **Part II — Propositions** is the engineering sketch of how it could work here.

They are kept separate so I can accept, reject or change any part of the proposal without losing the original thought. My view wins.

---

# Part I — My own view

## The spark

Items should have a component — call it `display`, or something with a better name — that holds **XML the user sees when they hover over the item.**

That sounds small, but it opens up things Minecraft players couldn't even imagine:

- adding an enchantment can **change the item's name** in that card;
- it can add a **GIF animation** or **border effects**;
- the card can contain **buttons or other settings** — imagine a rune-socket item whose tooltip has real buttons, or an enchant that offers a "reforge" action right there.

The tooltip stops being static text and becomes a little server-driven app attached to the item. This is the part I'm most excited about, and I want the design to leave room for it.

## Why it matters to me

This is the proof that the modding system is truly flexible. If a mod can attach a tiny interactive UI to an item without touching engine code — and another mod can add its own piece to the same card — then modders are not filling in numbers in a data file. They are building real features.

It also fits the way I like to work: I don't want to answer twenty compact questions. I want descriptive prose I can read and react to. If I don't like something in Part II, I'll say so.

---

# Part II — Propositions (proposed design, not yet decided)

## 1. Naming

`Display` is too generic — it tends to mean "rendering". My pick is **`HoverCard`**: it says "rich thing that appears on hover", which leaves room for buttons and animation without lying. Runner-ups: `ItemView`, `ItemPresentation`.

## 2. Structure — fragments, not one template

The item entity carries a `HoverCard { fragments: [...] }` component. Each enchant contributes **its own fragment** when added, so no mod has to fight over a single shared template:

```ts
export function on_lightning_added(
  items: Query<[Entity, Added<Lightning>, Mut<HoverCard>]>,
): void {
  for (const [e, ench, card] of items) {
    card.fragments.push(`<text class="name-suffix">of Lightning</text>
      <img src="mymod:lightning_border" class="animated-border"/>
      <button action="mymod:attune_rune">Attune Rune…</button>`);
  }
}
```

The engine assembles the fragments in priority order into one XML document. The item's **name stays a derived characteristic** (`DisplayName`, written by the `Added<Lightning>` system), so the HUD, chat links and the card all agree automatically — the card is a *view*, not a second source of truth.

This composes for the same reason the editable `HitContext` does: different mods write different pieces, never the same field.

## 3. Flow — instant first, full card on demand

- Whenever a card's components change, a small **presentation snapshot** (name, icon, a generic "has visual effects" flag, short description, "has interactive card" flag) rides along in the existing `InventorySyncDiff`. Hover shows it instantly — no round trip over UDP.
- If the snapshot says the card is interactive, the client sends an "inspect this item" intent. The server renders the full XML from fragments and replies with `UiShow`; the client mounts it near the cursor and `UiHide`s on real leave.
- Buttons inside use the existing `UiAction` round trip with a **server-side handle** in `ui_id` — never trust a client-supplied entity id, and validate that the player can actually see or hold that item.

## 4. What this makes possible

- An enchant that renames the item, adds an animated border, and offers a reforge button right in the tooltip.
- A rune-socket item whose card exposes real buttons to insert and remove runes — the rune system's UI, with zero engine UI code.
- Per-item settings panels, animated effects, or any other small server-driven app the modder can imagine.

## 5. Hard parts to solve before promising it

1. **Input conflict** — clicking a card button must not also break or place the block under the cursor. Pointer-over-card has to consume the click.
2. **Latency and flicker** — debounce hover; keep the panel mounted while the pointer moves into it; unmount only on true leave.
3. **Budgets** — a per-card cap (e.g. 16 KB / 64 nodes) so stacked fragments can't mount a huge panel. Reuse the `UI_MAX_*` idea with tighter limits.
4. **Animated images** — the client image path is KTX2-based; GIF/APNG may need a client-side format decision. CSS-style animated borders are cheap; animated textures are not.
5. **Positioning** — confirm the XML UI can do a cursor-anchored floating panel; if not, that is a small client addition.
6. **Element ids** — namespace fragment ids so two mods can't collide on `id=`.

## 6. Choices left open

- **Fragments** (proposed) vs **one shared per-item data dictionary** that mods append to. Fragments are more independent; a shared dictionary gives a consistent look but forces key agreement.
- **Instant snapshot → full card on demand** (proposed) vs **always render the full card on the server**. The latter is simpler, but hover will feel laggy over UDP.

## 7. Where it fits

The hover card depends on two things from the main plan: **item entities** (so a card can live on the item) and **`Added` change detection in TypeScript** (so an enchant can contribute its fragment when applied). It therefore belongs after steps 1–3 of the execution plan, not before. Nothing in the engine needs to know what a "lightning" card looks like.

---

## Mod capabilities — user-granted permissions (future work, not in stage 2)

**Vision:** a mod must be able to *ask* for dangerous abilities instead of silently having them. The player (or server owner) sees a clear request and grants or denies it. This is the line between "mods are sandboxed code" and "mods are trusted programs".

Examples of capabilities a mod might request:
- **Filesystem** — read/write a save directory for its own persistent state (tool XP, rune collections).
- **Network** — make outbound HTTP requests (fetch a texture pack, post a score to a leaderboard).
- **Host commands** — spawn entities, change blocks, send UI (today these are implicitly allowed through the `ScriptRequest` surface; in a capability model they would be explicit grants).
- **Cross-mod data** — read another mod's component data (relevant to the rune-on-enchanted-tool case).

How it should feel:
- A mod declares its requested capabilities in a manifest or at load time.
- On first join/load, the user sees a plain-language prompt: "mymod wants to: read/write its save folder, make network requests" with Allow / Deny.
- Grants are remembered per server/mod, revocable later from the launcher UI.
- Denied capability → the mod still loads but that specific host function returns a clean error (or `None`), never a crash or silent data theft.

Why it is **not** in stage 2:
- Stage 2 already introduces cross-mod type imports, item entities and the `creative.rs` split. Capabilities add a second trust/permission axis and would double the blast radius.
- The TS runtime is already sandboxed with fuel budgets and bounds-checked memory, so there is no urgent security hole — this is a *policy* feature, not a stop-the-bleeding fix.
- It interacts with the free-client/server-discovery business model: grants belong to the player's launcher profile, so the design should be done once the launcher UI exists.

Open questions to settle when this is picked up:
- Who grants on a multiplayer server — the player for their client, or the server owner for the server-side script? Likely both, for different capabilities.
- Are capabilities per-mod, per-mod-version, or per-transaction?
- How do authored game bundles declare their needs without prompting on every join?

---

## Engine hygiene: remove `creative.rs` from the core (queued, not urgent)

`server/src/creative.rs` is misnamed. It started as "creative tab" but absorbed all inventory-authority logic because that is where UI actions landed. It currently mixes three unrelated things:

| What is there | Where it belongs |
|---|---|
| `HotbarSyncLog`, `send_inventory_sync`, `hotbar_diff`, `sync_new_inventories` | Engine protocol core (`common`/`net`) — the hotbar delta contract every game uses. |
| `apply_inventory_clicks`, `give_stack` | General inventory mechanics (`interaction`) — shared by every game, not just creative. |
| `handle_creative_give`, `send_hud`, `sorted_block_names` | Game/content layer — with Lua removed these are **scripts of the first game bundle**, not engine code. |

Why it hurts the target architecture:
- games are scripts, so a creative mode is just one game/re among many — `CreativePlugin` does not belong in `main.rs`;
- inventory sync is a contract all games depend on, so hiding it inside a module named "creative" creates fake coupling (this is what caused the `HotbarSyncLog` / `apply_script_requests` commit conflicts during the change-detection work);
- `main.rs` should not know that "creative mode" exists.

**Proposed split:**
1. Move `HotbarSyncLog` + `send_inventory_sync` + `hotbar_diff` + `sync_new_inventories` into a neutral `inventory_sync` module (or `common`).
2. Move `apply_inventory_clicks` / `give_stack` into `interaction` (or a shared inventory mechanics module).
3. Delete `CreativePlugin`, `handle_creative_give`, `send_hud`, `sorted_block_names` from the core — reimplement as TS systems + XML in the first authored bundle. Keep `ClientMessage::CreativeGive` in the protocol only if a script needs it.

Do this **after** change detection is stable — it touches `events.rs`, `interaction.rs`, `scripting.rs`, `ts_backend.rs` and the client inventory path, so it is a refactor with a real blast radius, not a cleanup.

---

## Reconcile before building

- My vision says modding flexibility is the point — if the fragment design ever hardens into a closed vocabulary, cut it.
- The two choices in §6 need my decision before this enters the executable plan.
- Ask me with prose, not question lists.
