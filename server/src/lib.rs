//! Shared server-side code used by both the server binary and the client.
//!
//! The client depends on this crate (it already did, for `gltf_pack`), so
//! anything both sides must agree on belongs here rather than in a binary
//! module. At the moment that is the player's avatar model and its scale:
//! the server names the model when it spawns a player entity and the client
//! needs the same path to instantiate the matching local viewmodel.

pub mod gltf_pack;

/// Model the server assigns to every player entity. A catalog asset path.
pub const PLAYER_MODEL: &str = "models/girl.glb";

/// Scale that takes the avatar's authored height to the ~1.8 blocks a
/// player occupies.
///
/// The model is a ~3.19 unit tall figure with its feet near y=0, so this is
/// about 1.8 / 3.19. Measured from the file; see
/// `client/src/anim/rig.rs`'s bounds test.
pub const PLAYER_MODEL_SCALE: f32 = 0.565;
