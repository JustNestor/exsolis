# Ideas for the future

- **Status:** brainstorm — a place to keep the big picture, separate from the executable plan
- **Date:** 2026-11-17
- **Related:** [`plans/ecs-items-and-abilities-plan.md`](plans/ecs-items-and-abilities-plan.md), [`entities-and-animation.md`](entities-and-animation.md), [`world-and-blocks.md`](world-and-blocks.md)

---

## How to read this

This file has two halves that must not be confused:

- **Part I — My own view** is the *vision and instincts* I (the project owner) brought to the conversation: what I want the game and its modding to feel like, what I rejected, and the spark of an idea I want to keep alive.
- **Part II — Propositions** is the concrete engineering design proposed in response: how those desires could actually be built on top of this codebase.

Where the two disagree or leave a choice open, that is deliberate — it is a conversation, not a settled spec. My view wins.

---

# Part I — My own view

## 1. What the game needs

We need proper **entities, block entities and items**. I looked at what exists and saw that the first two have a solid foundation, but items are still just numbers in an array — no real item identity, no dropped items, no way for a mod to define a tool or a resource that isn't a block.

When asked, I chose:

- a **real `ItemRegistry`** with namespaced ids, not "items are just non-placeable blocks";
- **world drops**, not direct transfer — broken blocks and overflow should produce item entities with gravity and pickup.

## 2. The real reason this matters: modders

I don't want items and enchantments just for their own sake. I want a **common ground** where modders can build their own systems:

- an **enchanting system**, with enchantments that make an item deal more damage, pierce armor, or mine a 3×3 area;
- a **rune system**, where runes can affect something that already exists *or define something completely new*;
- **tools that level up the more you use them**.

And crucially: **new mods must be able to modify an existing attribute or react to it in a new way**, without editing the mod that defined it. That openness is the point. Enchantments and runes are just two examples — the system must not know about them specifically.

I like the typed way and I'm fine with a big refactor to get it.

## 3. What I rejected, and why

- **A central registry of abilities/attributes addressed by string ids.** It smells like Bedrock: data files declare behaviour, mods are limited to filling in numbers, and new systems need engine changes. I don't like the lack of flexible modding. JSON must never be a ceiling on what a modder can do.
- **`resolve_lightning()`-style functions.** I don't want a dedicated resolver for each trait. I want the world to tell us *when something changed* and have systems react.
- **typedb.** It is deprecated; types are defined by hand now. Any design must live with that.
- **Terse interview questions.** When I can't answer twenty compact questions, give me descriptive prose I can react to instead.

## 4. How scripts should react — the ECS way

We have done a lot of work to bring typed ECS to TypeScript. Scripts should react *through that*, not around it. Concretely, the things I asked for:

- **Mutable `HitContext`, processed in posttick.** Don't call a function when a hit happens. Let mods mutate a hit context, and let all hits be processed at the end of the tick.
- **`Added` for characteristics.** When an enchantment is added to an item, that should change the item's damage, its name, or any other aspect that other systems query — driven by change detection, not polling.
- **`Added` / `Changed` / `Removed` and the ability to put those labels ourselves.** Scripts need to be able to mark things as added, changed or removed explicitly.
- **Importing other mods' types**, so one mod can query components defined by another mod.

## 5. The hover card — a spark I want to keep

Items should have a component — call it `display`, or something with a better name — that holds **XML the user sees when they hover over the item**.

That sounds small, but it opens up things Minecraft players couldn't even imagine:

- adding an enchantment can **change the item's name** in that card;
- it can add a **GIF animation** or **border effects**;
- the card can contain **buttons or other settings** — imagine a rune-socket item whose tooltip has real buttons, or an enchant that offers a "reforge" action right there.

The tooltip stops being static text and becomes a little server-driven app attached to the item. This is the idea I'm most excited about, and I want the design to leave room for it.

## 6. Things I said explicitly in the conversation

- I don't understand terse questions — prefer expressive descriptions.
- I want a plan doc with checkboxes showing what's already done, detailed enough that another agent can pick up a task.
- I want work on its own branch, committed and pushed.
- "If I don't like something I'll say so" — so keep proposals clearly labelled and easy to reject.

---

# Part II — Propositions (proposed design, not yet decided)

## A. The core reframe: ECS *is* the extension point

The first design I proposed had a registry of attributes with string ids, a `combine` rule, and host functions like `stack_attr(handle, attr_f32("..."))`. On reflection that is exactly the Bedrock-shaped thing you rejected: it would make mods data-entry clerks and leave all our typed-ECS work decorative.

**Proposition:** there is no attribute registry. Components are the vocabulary and systems are the behaviour:

- an item stack is an **ECS entity**;
- an attribute is a **component type** (`EffectiveDamage`, `MiningSpeed`, `BreakOffsets`);
- a mod's enchant is **its own component** (`Lightning { level }`) — the type is the identity, no string id;
- "combining" is just systems writing to the effective component in a defined phase — no global `combine` enum;
- JSON only *assembles components into prefabs*; it never defines behaviour.

## B. Change detection: making `Added` the superpower

Bevy already keeps per-component change ticks. The server even stamps them when handing mutable columns to scripts (`server/src/ts_ecs.rs:742`) — but TypeScript mods cannot query on them yet.

**Proposition:** finish that plumbing:

- each JIT system gets a `last_run` tick, advanced each tick;
- `Added<T>` / `Changed<T>` become query filters comparing the cell's ticks against `last_run`;
- `Removed<T>` becomes a separate region backed by Bevy's removed-component buffer, cleared at end of posttick;
- manual labels become host functions `mark_added`, `mark_changed`, `mark_removed(entity, component_id)` using the existing `component_access_id`. These drive observers; they don't physically remove storage unless paired with a real `commands.remove`.

This is what makes "add lightning → sword gains damage and name" work with no resolver function. A system with `Query<[Entity, Added<Lightning>]>` writes the derived characteristics once.

## C. Editable hit events, applied once

**Proposition:** a hit spawns a short-lived entity with a `HitContext` component:

- engine-owned identity fields: attacker, weapon, target, entity id;
- mod-writable fields: damage, flags, a bounded list of effects (max 8).

During a `React` phase, any number of systems take `Query<[Mut<HitContext>]>` and write their own field. One engine system in `PostTick` reads every finished context, applies it once, and despawns the event entity. This uses the ECS you built, composes without a callback-ordering registry, and is inspectable like any other entity.

*(Alternative I noted but did not recommend: Bevy observers, `On<Add, Hit>` — more reactive, but more JIT-to-world wiring.)*

## D. Scripts talking to each other

**Proposition:** mods share hand-written type packages. Each mod ships `types/<mod-id>.d.ts` plus a manifest declaring which `namespace:name` component ids it owns and its version. Another mod writes `import type { Lightning } from "@mymod/lightning"`; the declaration is resolved at compile time, and only the component name survives at runtime. This is like a C header: types for the compiler, a name for the runtime.

Supporting pieces: a global component-name registry with namespace ownership, dependency declarations with a hard load error on missing deps (silence hides bugs), semver ranges, and explicit layout sizes validated at load — that last part is how we stay safe now that typedb is gone.

## E. JSON's job

**Proposition:** JSON is content authoring, never behaviour. A prefab maps a name to a component bundle; applying it inserts those components. A content-only mod can be pure JSON; a behaviour mod writes systems and can create the same components from code. No conditions, no reactions, no combine rules in JSON.

## F. The hover card, designed

### Naming

`Display` is too generic; my pick is **`HoverCard`** — it says "rich thing that appears on hover", which accommodates buttons and animation. Runner-ups: `ItemView`, `ItemPresentation`.

### Structure

The item entity carries `HoverCard { fragments: [...] }`. Each enchant contributes **its own fragment** when added, so no mod fights over one template:

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

The engine assembles fragments in priority order into one XML document. The item's name remains a derived characteristic (`DisplayName`), so HUD, chat and card agree.

### Flow (hybrid, to keep hover instant)

- When a card's components change, a small **presentation snapshot** (name, icon, glint, short description, "has interactive card" flag) rides along in the existing `InventorySyncDiff`. Hover shows it instantly, no round trip.
- If the snapshot says the card is interactive, the client sends an "inspect this item" intent; the server renders the full XML from fragments and replies with `UiShow`; the client mounts it near the cursor and `UiHide`s on leave.
- Card buttons use the existing `UiAction` round trip with a **server-side handle** in `ui_id` — never trust a client-supplied entity id, and validate the player can see/hold the item.

### Why this is powerful

An enchant can rename the item, add an animated border, and offer a reforge button. A rune-socket item can expose actual UI to insert/remove runes. None of that needs engine UI code — it is XML plus a host action, authored by mods.

### Hard parts to solve before promising it

1. **Input conflict** — clicking a card button must not also break/place the block under the cursor.
2. **Latency & flicker** — debounce hover; keep the panel mounted while the pointer moves into it.
3. **Budgets** — a per-card cap (e.g. 16 KB / 64 nodes) so stacked fragments can't mount a huge panel.
4. **Animated images** — the client image path is KTX2-based; GIF/APNG may need a format decision. CSS-style animated borders are cheap; animated textures are not.
5. **Positioning** — confirm the XML UI can do a cursor-anchored floating panel; if not, small client addition.
6. **Element ids** — namespace fragment ids so mods don't collide.

### Two choices left open

- **Fragments** (proposed) vs a single shared per-item data dictionary that mods append to. Fragments are more independent; a shared dictionary gives a consistent look but forces key agreement.
- **Instant snapshot → full card on demand** (proposed) vs always rendering the full card on the server (simpler, but laggy over UDP).

## G. Decisions I made so you wouldn't have to answer twenty questions

These are defaults, not laws. Flip any of them.

- **D1** V1 TS surface: change ticks, `Added`/`Changed` filters, `mark_*` labels, `world.get<T>` first.
- **D2** Hit fields: engine owns identity; mods write damage/flags/effects (max 8).
- **D3** Phases: `Seed → React → PostTick → Apply`, commutative writes + priority escape hatch.
- **D4** Stackability: same `ItemId` + equal component set, derived components excluded.
- **D5** Container break: transfer what fits, spill the rest as drops.
- **D6** No stable stack instance id yet; reactors stay within a tick.
- **D7** Enchants are components on the item entity, not separate source entities.
- **D8** Characteristic removal rebuilds from scratch rather than reversing writes.
- **D9** Prefabs insert several components at once.
- **D10** Missing mod dependency = hard load error with a clear message.
- **D11** Semver ranges in manifests.
- **D12** One agent per execution step, with review between.
- **D13** Bump the protocol and require matching clients.

## H. Execution order (proposed)

1. Change detection + cross-mod imports → exit test: an `Added<Lightning>` system writes damage; a second mod imports the type and queries it.
2. Item entities + wire changes → exit test: place/break/drop/pickup round-trip without losing stack identity.
3. Prefabs + `Added` characteristics + editable hits + posttick → exit test: lightning adds damage on add, two mods edit one context, damage applied once.
4. Dropped-item lifecycle + rendering → exit test: no overflow loss; drops render and can be picked up.
5. Lua parity, then persistence. **HoverCard** belongs alongside step 3–5 once the item entity and `Added` characteristics exist.

---

## Where the two halves must be reconciled

- My view wants modding flexibility above all; the propositions must never harden into a closed vocabulary. If any part of Part II starts feeling like Bedrock, it should be cut.
- The hover card is my spark; Part II only sketches it. Its two open choices (fragments vs shared dictionary; snapshot vs full render) need my decision before it enters the executable plan.
- I want descriptive prose, not question lists, when I'm asked to decide.
