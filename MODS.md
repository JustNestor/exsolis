# TypeScript mods

TypeScript mods live under `mods/ts/` and are compiled by tesjit. The server
loads the current component layouts from `types.tdb`; after a mod is first
loaded, it writes `mods/exsolis.generated.d.ts` for editor completion. That
generated file is managed by the server. If component types are missing from
the database, regenerate it with `just export-types`.

## ECS queries

An exported `on_tick` may take one or more typed query parameters. Each query
iterates the matching Bevy table rows directly; component changes write through
to the world without a snapshot or copy-back:

```ts
export function on_tick(
  query: Query<[Entity, Mut<Position>]>,
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
matching currently supports table-stored components and requires every listed
term; filters and `Commands` are not exposed yet. Each table chunk is charged
against the mod's fuel budget before any row in that chunk runs.

Existing zero-argument `on_tick()` mods and event handlers remain supported.
An export other than `on_tick` cannot take query parameters.
