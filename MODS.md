# TypeScript mods

TypeScript mods live under `mods/ts/` and are compiled by tesjit. The server
prefers explicitly registered component schemas and temporarily uses
`types.tdb` as a fallback for types not yet migrated. Mods can use registered
schemas even when the database is absent. After a mod is loaded, the server
writes `mods/exsolis.generated.d.ts` for editor completion; that generated file
is managed by the server.

## ECS queries

An exported `on_tick` may take one or more typed query parameters. Each query
iterates the matching Bevy table rows directly; component changes write through
to the world without a snapshot or copy-back:

```ts
export function on_tick(
  query: Query<
    [Entity, Mut<Position>],
    [With<PlayerTag>, Without<Disabled>]
  >,
  config: Res<ModConfig>,
): void {
  for (const [entity, position] of query) {
    position.x += config.speed;
  }
}
```

The component types and fields available in a mod are listed in the generated
declarations. Query terms may be `Entity`, a read-only component type, or
`Mut<Component>` for writable access. Tick systems may also take
`Res<Resource>` for a read-only resource or `Mut<Resource>` for a mutable
resource. The resource must be present when the system is linked; if removed
later, that system is skipped until it returns. Read-only component/resource
writes are rejected at compile time. A `Mut` borrow updates Bevy's changed tick
when acquired, even if the mod leaves its fields unchanged.

Query systems execute synchronously during the server's exclusive tick system.
They cannot add or remove components while query pointers are live. Query
matching supports table-stored component terms and `With<Component>` /
`Without<Component>` filters (including zero-sized marker components).
Sparse-set query terms and filters are not supported. `Commands.spawn` accepts
registered, supported component values and stages a new entity:

```ts
export function on_tick(commands: Commands): void {
  const position: Position = { x: 1.5, y: 9 };
  const marker: PlayerTag = {};
  commands.spawn(position, marker);
}
```

Component object literals need an explicit component type. Spawned entities are
added after every mod has run, so they are visible to queries on the next tick,
not later in the current tick. `spawn` returns `void`; `despawn`, `insert`, and
other command operations are not implemented. Components must have a registered
layout with supported fields; sparse-set components and fields with unsupported
representations cannot be spawned. Each table chunk is charged against the
mod's fuel budget before any row in that chunk runs.

Existing zero-argument `on_tick()` mods and event handlers remain supported.
An export other than `on_tick` cannot take query parameters.
