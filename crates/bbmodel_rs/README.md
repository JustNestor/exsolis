# bbmodel_rs

Blockbench `.bbmodel` model loading and animation summaries, baked to the
engine's voxel format.

This crate used to live inline in the client crate as
`client/src/loader/bbmodel.rs` (+ the shared baked-geometry types in
`client/src/loader/baked.rs`). It was extracted so the bake is a standalone,
testable, Bevy-free library: the engine mesher, the block-preview rasterizer
and mod tooling all end up consuming the same pipeline instead of a copy of it.

## Modules

| Module | Contents |
|---|---|
| [`baked`] | The baked geometry types the loader emits: `BakedModel`, `BakedQuad`, `BakedTri`, `RenderType`, the `BlockPack` trait, `face_quad`, `rotate_point`. |
| [`bbmodel`] | The Blockbench project JSON loader: `load_bbmodel`, `parse_bbmodel`, textures, and animation summaries (`BbAnimInfo`). |

## Usage

```rust
use bbmodel_rs::{load_bbmodel, parse_bbmodel};

let (model, report) = load_bbmodel(std::path::Path::new("girl.bbmodel"))?;
// model.baked: quads + tris in 0-1 block space
// report.animations: Vec<BbAnimInfo> (bone -> channels -> keyframe counts)
```

## Animation scope

The loader parses `animations[]` into [`BbAnimInfo`] summaries (per-animation
loop mode, length, bone → channel/keyframe map). Full Bedrock bone sampling is
deferred: the runtime currently plays glTF clips, so the baked summaries are the
integration point for future animation work.

## Tests

```sh
cargo test -p bbmodel_rs
```
