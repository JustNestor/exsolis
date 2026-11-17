# Items, enchantments and reactive abilities — the design, in plain language

- **Status:** proposal for discussion — no engine code has been changed
- **Date:** 2026-11-17
- **Related:** [`../entities-and-animation.md`](../entities-and-animation.md), [`../world-and-blocks.md`](../world-and-blocks.md), [`../scripting/typescript-modding.md`](../scripting/typescript-modding.md)

---

## How to read this

This document explains the design as a story instead of a list of questions. Each section says what the idea is, what it feels like for a modder, and what we give up. Where I made a choice for you, I wrote it as a statement ending in **"If you want it the other way, say so"** — that is your steering wheel. Nothing here is code yet.

Checkbox legend: `[x]` means "already in the code today", `[ ]` means "we still have to build it".

---

## 1. The whole idea, told as a story

Imagine a mod author wants a **Sword of Lightning**: it hits harder, and every hit calls down a lightning arc. Here is its life in the engine we are designing.

1. **An author writes a small content file.** It says: "the thing called `mymod:lightning` is made of these components — a `Lightning { level }` component and a display-name suffix." This is JSON, but JSON only *assembles parts*. It never says what lightning *does*.

2. **Applying it to a sword is an ordinary ECS operation.** The server inserts those components onto the sword's entity. That's it — no registry lookup, no "register your enchantment with the engine" call. The components are now simply part of what that sword *is*.

3. **The sword gains its characteristics because something was *added*.** A system in a mod asks the world: "show me every entity that just received a `Lightning` component." It finds the sword and writes onto it: damage +1.5 per level, the name "Sword of Lightning", a glowing flag. Nobody wrote a function called `resolve_lightning`. The sword changed because the world told us it changed, and we reacted.

4. **The player swings at a zombie.** The engine spawns a short-lived *event entity* holding a `HitContext`: who attacked, which weapon, which target, the base damage, and an empty list of effects. Think of it as a shared form that mods fill in before it is submitted.

5. **Mods edit the form.** During a `React` phase, every interested system takes a typed query over `Mut<HitContext>` and writes its own field. The lightning mod sets "arc radius = 3". A rune mod appends a burning effect. An armor-pierce mod flips a boolean. They never talk to each other; they just edit different fields of the same form.

6. **Once, at the end of the tick, the engine submits the form.** A single `PostTick` system reads the finished `HitContext`, applies the damage and effects, and throws the event entity away. No mod can apply damage twice, because only the engine submits.

7. **If the sword is dropped, it is the same entity.** The `InInventory` component comes off, a `Transform` goes on, gravity starts. Pick it back up and the reverse happens. The lightning, the damage bonus, the name — all still there, because they were never stored in a separate "item data" blob. They are the entity.

That is the entire vision. Every step is a normal Bevy ECS operation, visible and modifiable through the typed TypeScript queries we already built. There is no special "ability subsystem" sitting off to the side.

**If you want it the other way, say so** — for example if you'd rather enchant components live on *their own* entities attached to the item, or if you want hits to be plain function calls instead of editable entities.

---

## 2. The design we rejected, in plain language

Earlier we sketched a "registry of abilities": one big table where every attribute (`damage_bonus`, `armor_pierce`, `mining_radius`) has a string name, a value type, and a rule for combining contributions. Enchantments and runes would be JSON entries that add numbers to those attributes, and scripts would call host functions like `stack_attr(handle, attr_f32("exsolis:damage_bonus"))` to read them.

The more we looked at it, the more it felt wrong, and now I can say why:

- **It turns modders into data-entry clerks.** The engine owns a closed vocabulary; mods only fill in numbers under ids the engine already knows. That is the Bedrock behaviour-pack model, and you already said flexible modding is the whole point.
- **It throws away our best asset.** We spent a lot of work giving TypeScript real typed ECS — `Query<[...]>`, `Mut<T>`, components, commands. Under the registry design, the main gameplay extension point would bypass all of it and use string keys and helper calls instead. That work would become decoration.
- **JSON would become a ceiling.** If all semantics live in definitions, a mod can only express what the definition format supports. Anything new (a conditional effect, a weird damage type) would need an engine change.

So we are not building that. There is no central attribute table, no `combine` enum, and no "enchantment registry". Components and systems *are* the extension point.

**If you actually liked parts of it** (say, a shared list of standard attribute names for mods to agree on) we can keep that as a small set of documented *Rust component types* — not a string registry.

---

## 3. The world model

### 3.1 Today: items are just numbers in an array

Right now an inventory slot is `Slot { item: BlockId, amount: u8 }` (`common/src/inventory.rs:21`). Items are aliases of blocks — a pickaxe can't exist, only a block with `placeable: false`. A dropped item is a completely different thing: an `EntityKind::Item` entity that no system actually spawns or moves. Breaking a chest hands its contents straight to the player; overflow is deleted with a warning.

### 3.2 Target: an item is an entity

A stack becomes a Bevy entity with components:

```
Sword entity
  ItemStack { item, amount }        <- engine core
  InInventory { owner, slot }       <- where it lives
  Lightning { level: 3 }            <- a mod's enchantment
  RuneSocket { id }                 <- another mod's rune
  EffectiveDamage { value: 12.5 }   <- derived characteristic
  DisplayName { suffix }            <- derived characteristic
```

Inventory slots and chests hold *entity handles* instead of `Slot` values. This is a real refactor of `Inventory`, `Container`, and the wire messages — but it buys something big: one representation for items everywhere. Inventory, chest contents, and dropped items become the same kind of thing.

**If you want it the other way, say so** — the cheaper alternative is to keep `Slot` as a value and hang components off it in a side table, but then dropped items and inventory items stay two different systems and mods get a messier API.

### 3.3 An attribute is just a component

`EffectiveDamage`, `MiningSpeed`, and `BreakOffsets` are ordinary components that the engine's combat and mining systems read. A mod's `Lightning` is an ordinary component too. There is no table that says "attribute #17 is damage"; the Rust type *is* the identity.

The only shared vocabulary is a short list of **contract components** the engine itself consumes — damage, mining speed, break offsets, armor pierce. Mods write into those; they don't redefine them. Everything else belongs to mods entirely.

### 3.4 Combining is done by systems, not a rulebook

With no `combine` field, how do three damage bonuses add up? Each contributing system writes to `EffectiveDamage` during a `Resolve`-style part of the tick. Sharpness adds 1.5, a rune adds 3, a tool tier adds 2. The final component value is what combat reads. The convention is "everyone adds" — if a mod wants multiplicative behaviour, it writes a different component or a different field. Ordering within that phase must be commutative (or use a documented priority), so mod load order never changes the result.

**If you want it the other way, say so** — e.g. an explicit "damage = (base + flat) × multiplier" pair of contract components with engine-defined fields. That's more structured but less open.

---

## 4. Change detection: the "Added" superpower

Bevy keeps a little "last touched" timestamp on every component. A query can ask for only the rows where a component was *just added*, or *changed since I last looked*, or *removed*. We already have half of this in the server: `server/src/ts_ecs.rs:742` stamps those ticks when handing mutable columns to a script — but TypeScript mods cannot ask for them yet.

Finishing that is the first piece of work, and it is what makes the whole design work:

- **`Added<T>`** — a system with `Query<[Entity, Added<Lightning>]>` runs once when an enchant is applied. This is how a sword *gains* its damage, name, and glow without a resolver function.
- **`Changed<T>`** — "this component was touched since my last check". The end-of-tick damage system uses `Changed<HitContext>` to skip hits nobody edited.
- **`Removed<T>`** — a component went away. Removing the enchant triggers a rebuild of the sword's characteristics. Because there is no row left to filter, this is exposed as a small separate query region of "things that were removed".
- **Manual labels** — sometimes a script changes something *derived* without writing the field directly. It needs to say "treat this as changed". We'll add host functions `mark_added`, `mark_changed`, `mark_removed(entity, component)` using the existing `component_access_id`. Important detail: these labels drive other systems' queries; they do not physically remove a component unless the script also issues a real `commands.remove`.

You also asked how we can "put these labels ourselves" — that's exactly the `mark_*` functions above. **If you want it the other way, say so** — for example a `Dirty<T>` wrapper type where writing `dirty.set()` marks the change implicitly.

---

## 5. Events you can edit

A hit is not a function call; it is a short-lived entity with a `HitContext` component:

```
attacker, weapon, target        <- identity (engine-owned, mods shouldn't rewrite)
damage, pierce_armor, effects   <- the writable form
```

Mods take `Query<[Mut<HitContext>, ...]>` and write their fields. Then one engine system in `PostTick` reads every finished context, applies the result once, and despawns the event entity.

Why this beats an `api.on("hit")` callback list:

- **It uses the ECS you already built.** The same typed query machinery, no parallel subscription system.
- **It composes by default.** Two mods editing different fields never step on each other; there's no callback ordering registry to maintain.
- **It's inspectable.** You can query pending hits, log them, or add a debugging UI — event entities are just entities.

**If you want it the other way, say so** — e.g. Bevy *observers* (`On<Add, Hit>`), which are even more reactive, but need more wiring between the JIT and the world. I'd ship editable event entities first.

---

## 6. Scripts talking to each other

If a lightning mod defines `Lightning`, how does a separate "lifesteal" mod query it without owning it? Answer: **mods share hand-written type packages**.

Each mod ships a small `types/<mod-id>.d.ts` file (the "by hand" types you chose instead of typedb) and a manifest saying "I own the component `mymod:Lightning`, version 1.2". Another mod writes:

```ts
import type { Lightning } from "@mymod/lightning";
```

At compile time the import is resolved and the type declaration is pulled in. At runtime only the component *name* survives (`component_access_id("mymod:Lightning")`), so the JIT doesn't need a module system. Think of it like a C header file: it tells the compiler the shape, and it names the thing at link time.

Supporting pieces:

- A global component-name registry, with namespace ownership enforced (only `mymod` may register `mymod:*`).
- Dependency declarations in the manifest; a missing dependency is a **hard load error with a clear message**, not silence — silent empty queries hide bugs.
- Semver ranges so a mod can say "works with lightning 1.x".
- Because typedb is gone, safety comes from an explicit hand-written size/field layout in the type package, validated when the mod loads. A mismatch refuses to load instead of risking memory corruption.

**If you want it the other way, say so** — e.g. global ambient declarations instead of explicit imports, or "degrade gracefully" when a dependency is missing.

---

## 7. JSON is for content, never behaviour

A file like `assets/items/prefabs.json` maps a name to a bundle of components:

```json
{
  "mymod:lightning": {
    "components": [
      { "Lightning": { "level": 1 } },
      { "DisplayName": { "suffix": "of Lightning" } }
    ]
  }
}
```

Applying the prefab inserts those components. That is all JSON does. It cannot describe a condition, a reaction, or a combine rule. A content-only mod can be pure JSON; a behaviour mod writes systems in TypeScript or Lua and can build the exact same components from code whenever it likes. JSON is a shortcut, never a fence.

**If you want it the other way, say so** — for instance if you'd prefer no JSON at all and everything registered from code.

---

## 8. What exists today, and what we build

### Already there `[x]`

- [x] Entity ids, kinds, spawn/despawn/state-update messages, server registry and client mirror — `common/src/lib.rs:148,377`, `server/src/entities.rs:89`, `client/src/entities.rs`.
- [x] Block entities: flag on blocks, spawned on place, despawned on break, inventory-shaped container, server-authoritative door/bed animation — `common/src/blocks.rs:27`, `server/src/interaction.rs:466,758`.
- [x] Inventory click mechanics and hotbar sync messages — `common/src/inventory.rs:77-236`, `common/src/lib.rs:410,433`.
- [x] Hand-written TypeScript API with typed queries, mutable row helpers, component-id lookup, and handler exports — `mods/exsolis.d.ts`, `server/src/ts_ecs.rs`, `server/src/ts_backend.rs`.
- [x] JSON assets already read by both sides — `assets/animations/blocks.json`.

### To build `[ ]`

- [ ] Change detection exposed to TypeScript: ticks per system, `Added`/`Changed` filters, `Removed` region, manual `mark_*` labels.
- [ ] Typed random access to another entity's components (`world.get<T>`), insert/remove commands, generic queries over mod-defined components.
- [ ] Mod type packages with imports, ownership, dependency checks, layout validation.
- [ ] An item entity model: `ItemId` registry, `ItemStack` entity, `InInventory`, stack merge/split rules.
- [ ] Dropped-item lifecycle: gravity, pickup, merge, TTL, and rendering; replace all "dropped with a warning" paths.
- [ ] Prefab JSON loader; `Added`-driven characteristic systems; removal rebuild.
- [ ] `HitContext` event entities, `React` phase, single `PostTick` application.
- [ ] Wire protocol changes (bump `PROTOCOL_ID` to 7) and thin-client presentation data.
- [ ] Lua parity for the same model, then persistence later.

---

## 9. Choices I made so you don't have to answer twenty questions

I picked sensible defaults for everything we discussed. Read them as statements, not questions — tell me which one you want changed.

- **D1 — V1 TS surface.** First land: per-system change ticks, `Added`/`Changed` filters, manual `mark_*` labels, and `world.get<T>`. Generic mod-defined queries and named phases come next but are not blockers. *Why: this is the minimum that makes `Added`-driven characteristics possible.*
- **D2 — Hit field ownership.** Engine owns attacker/weapon/target/entity id; mods may write damage, flags, and a bounded list of effects (max 8). *Why: mods can't corrupt the identity of an event.*
- **D3 — Phases.** `Seed → React → PostTick → Apply`. Ordering inside a phase is by convention (write distinct fields) with a priority escape hatch. No full ordering graph. *Why: simple and enough for the examples.*
- **D4 — Stackability.** Two stacks merge if their `ItemId` matches and their component sets are equal, excluding derived components. *Why: safe default — unknown components make things unstackable rather than silently merging.*
- **D5 — Container break.** Transfer what fits into the breaker's inventory; spill the remainder as dropped items. *Why: preserves the "a chest can't be destroyed to destroy its loot" invariant.*
- **D6 — No stack instance id yet.** Event handling stays inside one tick. *Why: stable ids across ticks would force a bigger `Inventory` refactor for no current need.*
- **D7 — Enchants live on the item entity** as components, not as separate source entities. *Why: covers runes, enchantments and tool tiers with the simplest model.*
- **D8 — Removal rebuilds from scratch.** When a characteristic-affecting component disappears, systems recompute all effective components rather than trying to undo their old writes. *Why: reversal is fragile when mods load and unload.*
- **D9 — Prefabs insert several components at once.** *Why: one file per enchant instead of one per component.*
- **D10 — Missing mod dependency = hard load error with a clear message.** *Why: silent empty queries are the worst kind of bug.*
- **D11 — Semver ranges in manifests.** *Why: strict equality makes mod updates brittle.*
- **D12 — One agent per execution step** (§10), with review between steps. *Why: steps 1 and 2 are independent enough to parallelize once this design is agreed.*
- **D13 — Bump the protocol and require matching clients.** *Why: this project already treats decode desync as a hard error.*

**Any of these can be flipped. Just say "D4 should be X" and I'll update the document.**

---

## 10. Execution order

1. **Change detection + cross-mod imports.** Exit test: one TS mod with `Query<[Entity, Added<Lightning>]>` writes damage, and a second mod imports the first mod's `Lightning` type and queries it.
2. **Item entities + wire changes.** Exit test: place, break, drop and pick up an item round-trip without losing stack identity.
3. **Prefabs + `Added` characteristics + editable hits + posttick.** Exit test: lightning enchant adds damage on add, two mods edit one `HitContext`, damage applied once.
4. **Dropped-item lifecycle and client rendering.** Exit test: no item loss on overflow; dropped items render and can be picked up.
5. **Lua parity, then persistence.**

Do not start step 3 before step 1 — `Added`-driven characteristics are useless without change detection in TypeScript.

---

## 11. Ground rules for whoever implements

- No central string-keyed attribute registry, no JSON-defined behaviour. Components + systems only.
- New wire enums are append-only; new message fields get `#[serde(default)]`; bump `PROTOCOL_ID`.
- The client never resolves attributes or trusts mod component bytes — it gets presentation data and optional `ModAction`/UI payloads.
- When a decision above changes, edit that decision and every section it touches, and note the date.
