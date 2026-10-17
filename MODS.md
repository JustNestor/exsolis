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
): void {
  for (const [entity, position] of query) {
    position.x += 1.0;
  }
}
```

The component types and fields available in a mod are listed in the generated
declarations. Query terms may be `Entity`, a read-only component type, or
`Mut<Component>` for writable access. Read-only component writes are rejected
at compile time. A `Mut` query updates Bevy's changed tick for every matching
row, including rows whose fields the mod leaves unchanged.

Query systems execute synchronously during the server's exclusive tick system.
They cannot add or remove components while query pointers are live. Query
matching currently supports table-stored components and requires every listed
term; filters, resources, and `Commands` are not exposed yet. Each table chunk
is charged against the mod's fuel budget before any row in that chunk runs.

Existing zero-argument `on_tick()` mods and event handlers remain supported.
An export other than `on_tick` cannot take query parameters.
