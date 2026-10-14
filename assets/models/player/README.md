# First-person viewmodel working copies

`girl_base.glb` / `girl_base.bbmodel` are copies of the avatar
(`../girl.glb`, source `../girl.bbmodel`) reserved for first-person tuning.

- The game loads this copy for the local arms (`PLAYER_VIEWMODEL` in
  `server/src/lib.rs`); `models/girl.glb` stays the third-person avatar.
- Same rig: the arms cut (`ARM_ROOT_PREFIXES` in `client/src/anim/rig.rs`)
  and `VIEWMODEL_OFFSET` (`client/src/player.rs`) apply unchanged.
- Edit `girl_base.bbmodel` in Blockbench (desktop + MCP plugin, see
  `docs/blockbench/`), re-export to `girl_base.glb`. The
  `girl_base_copy_matches_girl` test pins the copy until adaptation starts —
  update it when the first intentional change lands.
- `VIEWMODEL_OFFSET` is the number to tune by eye (arms in lower frame).
