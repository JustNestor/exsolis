# Scroll Virtualization Plan

## Pass 1: Seams & flags
- Identify scroll areas in client/src/ui/spawn.rs and mod.rs.
- Add `UiVirtualList` marker component carrying item_count, item_height, viewport_items.
- Keep all data in ECS, render a window + overscan.

## Pass 2: Creative grid pilot
- Virtualize creative grid v-for rows; measure row heights.
- Map scroll offset to visible row range.

## Pass 3: Generalization
- Handle nested scrolls and dynamic item sizes.
- Add tests/snapshots.
