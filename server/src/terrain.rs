use exsolis_common::{BlockId, ChunkCoord, CHUNK_SIZE};
use opensimplex2::fast::{noise2, noise3_Fallback};

/// World generation configuration loaded from Lua
#[derive(Debug, Clone)]
pub struct WorldGenConfig {
    pub seed: u64,
    pub terrain: TerrainConfig,
    pub caves: CavesConfig,
    pub biomes: BiomesConfig,
    pub surface: SurfaceConfig,
}

#[derive(Debug, Clone)]
pub struct TerrainConfig {
    pub continentalness: NoiseLayerConfig,
    pub erosion: NoiseLayerConfig,
    pub peaks_valleys: NoiseLayerConfig,
    pub base_height: f32,
    pub height_scale: f32,
}

#[derive(Debug, Clone)]
pub struct CavesConfig {
    pub enabled: bool,
    pub cave_noise: NoiseLayerConfig,
    pub carver_noise: NoiseLayerConfig,
    pub cave_threshold: f32,
    pub carver_threshold: f32,
    pub large_cave_threshold: f32,
    pub carver_frequency: f32,
}

#[derive(Debug, Clone)]
pub struct BiomesConfig {
    pub temperature: NoiseLayerConfig,
    pub humidity: NoiseLayerConfig,
    #[allow(dead_code)]
    pub continentalness_weight: f32,
    #[allow(dead_code)]
    pub erosion_weight: f32,
}

#[derive(Debug, Clone)]
pub struct SurfaceConfig {
    #[allow(dead_code)]
    pub top_block: BlockId,
    #[allow(dead_code)]
    pub filler_block: BlockId,
    #[allow(dead_code)]
    pub stone_block: BlockId,
    pub water_level: i32,
    pub beach_depth: i32,
}

#[derive(Debug, Clone)]
pub struct NoiseLayerConfig {
    pub frequency: f64,
    pub amplitude: f64,
    pub octaves: usize,
    pub persistence: f64,
    pub lacunarity: f64,
}

impl Default for WorldGenConfig {
    fn default() -> Self {
        Self {
            seed: 12345,
            terrain: TerrainConfig {
                continentalness: NoiseLayerConfig {
                    frequency: 0.001,
                    amplitude: 1.0,
                    octaves: 4,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                erosion: NoiseLayerConfig {
                    frequency: 0.002,
                    amplitude: 1.0,
                    octaves: 4,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                peaks_valleys: NoiseLayerConfig {
                    frequency: 0.005,
                    amplitude: 1.0,
                    octaves: 3,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                base_height: 64.0,
                height_scale: 64.0,
            },
            caves: CavesConfig {
                enabled: true,
                cave_noise: NoiseLayerConfig {
                    frequency: 0.02,
                    amplitude: 1.0,
                    octaves: 2,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                carver_noise: NoiseLayerConfig {
                    frequency: 0.005,
                    amplitude: 1.0,
                    octaves: 3,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                cave_threshold: 0.4,
                carver_threshold: 0.3,
                large_cave_threshold: 0.6,
                carver_frequency: 0.003,
            },
            biomes: BiomesConfig {
                temperature: NoiseLayerConfig {
                    frequency: 0.0005,
                    amplitude: 1.0,
                    octaves: 3,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                humidity: NoiseLayerConfig {
                    frequency: 0.0007,
                    amplitude: 1.0,
                    octaves: 3,
                    persistence: 0.5,
                    lacunarity: 2.0,
                },
                continentalness_weight: 1.0,
                erosion_weight: 1.0,
            },
            surface: SurfaceConfig {
                top_block: exsolis_common::legacy_ids::GRASS as BlockId,
                filler_block: exsolis_common::legacy_ids::DIRT as BlockId,
                stone_block: exsolis_common::legacy_ids::STONE as BlockId,
                water_level: 63,
                beach_depth: 4,
            },
        }
    }
}

/// Noise sampler using OpenSimplex2 (function-based API)
#[derive(Clone, Copy)]
pub struct NoiseSampler {
    seed: i64,
    salt: u64,
}

impl NoiseSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            seed: seed as i64,
            salt: 0,
        }
    }

    /// Create a derived noise sampler with a different salt
    pub fn derived(&self, salt: u64) -> Self {
        Self {
            seed: self.seed,
            salt: self.salt.wrapping_add(salt),
        }
    }

    /// Sample 2D noise with fractal Brownian motion (fBm)
    pub fn sample_2d(&self, x: f64, z: f64, config: &NoiseLayerConfig) -> f64 {
        let mut value = 0.0;
        let mut amplitude = config.amplitude;
        let mut frequency = config.frequency;
        let mut max_amplitude = 0.0;

        for _ in 0..config.octaves {
            let sample = noise2(self.seed, x * frequency, z * frequency) as f64;
            value += sample * amplitude;
            max_amplitude += amplitude;
            amplitude *= config.persistence;
            frequency *= config.lacunarity;
        }

        if max_amplitude > 0.0 {
            value / max_amplitude
        } else {
            0.0
        }
    }

    /// Sample 3D noise with fractal Brownian motion
    pub fn sample_3d(&self, x: f64, y: f64, z: f64, config: &NoiseLayerConfig) -> f64 {
        let mut value = 0.0;
        let mut amplitude = config.amplitude;
        let mut frequency = config.frequency;
        let mut max_amplitude = 0.0;

        for _ in 0..config.octaves {
            let sample = noise3_Fallback(self.seed, x * frequency, y * frequency, z * frequency) as f64;
            value += sample * amplitude;
            max_amplitude += amplitude;
            amplitude *= config.persistence;
            frequency *= config.lacunarity;
        }

        if max_amplitude > 0.0 {
            value / max_amplitude
        } else {
            0.0
        }
    }
}

/// Biome types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Biome {
    Ocean,
    Beach,
    Plains,
    Forest,
    Desert,
    Mountains,
    SnowyPlains,
    SnowyMountains,
    Swamp,
    Jungle,
    Taiga,
}

impl Biome {
    pub fn surface_block(&self) -> BlockId {
        use exsolis_common::legacy_ids as ids;
        match self {
            Biome::Ocean | Biome::Beach => ids::SAND as BlockId,
            Biome::Desert => ids::SAND as BlockId,
            Biome::Swamp => ids::DIRT as BlockId,
            Biome::Jungle => ids::GRASS as BlockId,
            Biome::SnowyPlains | Biome::SnowyMountains => ids::SNOW as BlockId,
            _ => ids::GRASS as BlockId,
        }
    }

    pub fn filler_block(&self) -> BlockId {
        use exsolis_common::legacy_ids as ids;
        match self {
            Biome::Ocean | Biome::Beach | Biome::Desert => ids::SAND as BlockId,
            Biome::Swamp => ids::DIRT as BlockId,
            Biome::SnowyPlains | Biome::SnowyMountains => ids::SNOW as BlockId,
            _ => ids::DIRT as BlockId,
        }
    }

    pub fn stone_block(&self) -> BlockId {
        exsolis_common::legacy_ids::STONE as BlockId
    }
}

/// Complete world generator with all noise layers
#[derive(Clone)]
pub struct WorldGenerator {
    config: WorldGenConfig,
    continentalness_noise: NoiseSampler,
    erosion_noise: NoiseSampler,
    peaks_valleys_noise: NoiseSampler,
    temperature_noise: NoiseSampler,
    humidity_noise: NoiseSampler,
    cave_noise: NoiseSampler,
    carver_noise: NoiseSampler,
    large_cave_noise: NoiseSampler,
}

impl WorldGenerator {
    pub fn new(config: WorldGenConfig) -> Self {
        let continentalness_noise = NoiseSampler::new(config.seed);
        let erosion_noise = continentalness_noise.derived(0x9e3779b97f4a7c15);
        let peaks_valleys_noise = erosion_noise.derived(0x9e3779b97f4a7c15);
        let temperature_noise = peaks_valleys_noise.derived(0x9e3779b97f4a7c15);
        let humidity_noise = temperature_noise.derived(0x9e3779b97f4a7c15);
        let cave_noise = humidity_noise.derived(0x9e3779b97f4a7c15);
        let carver_noise = cave_noise.derived(0x9e3779b97f4a7c15);
        let large_cave_noise = carver_noise.derived(0x9e3779b97f4a7c15);

        Self {
            config,
            continentalness_noise,
            erosion_noise,
            peaks_valleys_noise,
            temperature_noise,
            humidity_noise,
            cave_noise,
            carver_noise,
            large_cave_noise,
        }
    }

    /// Generate a chunk using all noise layers
    pub fn generate_chunk(&self, coord: ChunkCoord) -> Vec<BlockId> {
        use exsolis_common::AIR_ID;
        const SIZE: usize = CHUNK_SIZE as usize;
        let mut blocks = vec![AIR_ID; SIZE * SIZE * SIZE];

        for x in 0..SIZE {
            for z in 0..SIZE {
                let world_x = coord.x * SIZE as i32 + x as i32;
                let world_z = coord.z * SIZE as i32 + z as i32;

                // Sample all 2D noise layers at this column
                let continentalness = self.continentalness_noise.sample_2d(
                    world_x as f64,
                    world_z as f64,
                    &self.config.terrain.continentalness,
                );
                let erosion = self.erosion_noise.sample_2d(
                    world_x as f64,
                    world_z as f64,
                    &self.config.terrain.erosion,
                );
                let peaks_valleys = self.peaks_valleys_noise.sample_2d(
                    world_x as f64,
                    world_z as f64,
                    &self.config.terrain.peaks_valleys,
                );
                let temperature = self.temperature_noise.sample_2d(
                    world_x as f64,
                    world_z as f64,
                    &self.config.biomes.temperature,
                );
                let humidity = self.humidity_noise.sample_2d(
                    world_x as f64,
                    world_z as f64,
                    &self.config.biomes.humidity,
                );

                // Calculate terrain height
                let height = self.calculate_height(continentalness, erosion, peaks_valleys);

                // Determine biome
                let biome = self.determine_biome(
                    continentalness,
                    erosion,
                    temperature,
                    humidity,
                    height as f64,
                );

                // Generate vertical column
                self.generate_column(
                    &mut blocks,
                    x,
                    z,
                    coord.y,
                    height,
                    biome,
                    world_x,
                    world_z,
                );
            }
        }

        blocks
    }

    fn calculate_height(&self, continentalness: f64, erosion: f64, peaks_valleys: f64) -> i32 {
        let t = &self.config.terrain;
        
        // Map continentalness from [-1, 1] to [0, 1]
        let c = (continentalness + 1.0) * 0.5;
        // Map erosion from [-1, 1] to [0, 1]
        let e = (erosion + 1.0) * 0.5;
        // Peaks and valleys
        let pv = peaks_valleys;

        // Base continentalness determines general elevation
        let base = t.base_height as f64;
        
        // Continentalness: positive = land, negative = ocean
        let continental_factor = c * t.height_scale as f64 * 0.8;
        
        // Erosion: smooths terrain, creates valleys
        let erosion_factor = (1.0 - e) * t.height_scale as f64 * 0.4;
        
        // Peaks and valleys: adds mountain ridges and deep valleys
        let pv_factor = pv * t.height_scale as f64 * 0.5;

        let height = base + continental_factor + erosion_factor + pv_factor;
        height.clamp(0.0, 255.0) as i32
    }

    fn determine_biome(
        &self,
        continentalness: f64,
        erosion: f64,
        temperature: f64,
        humidity: f64,
        height: f64,
    ) -> Biome {
        // Map noise values from [-1, 1] to [0, 1]
        let c = (continentalness + 1.0) * 0.5;
        let _e = (erosion + 1.0) * 0.5; // Unused but kept for potential future biome logic
        let t = (temperature + 1.0) * 0.5;
        let h = (humidity + 1.0) * 0.5;

        // Ocean check (very low continentalness)
        if c < 0.1 {
            return Biome::Ocean;
        }

        // Beach (transition zone)
        if c < 0.2 || (c < 0.3 && height < self.config.surface.water_level as f64 + 5.0) {
            return Biome::Beach;
        }

        // High elevation biomes
        if height > 140.0 {
            if t < 0.3 {
                return Biome::SnowyMountains;
            }
            return Biome::Mountains;
        }

        // Temperature/humidity based biomes
        match (t, h) {
            // Cold
            (t, _) if t < 0.2 => {
                if height > 100.0 {
                    Biome::SnowyMountains
                } else {
                    Biome::SnowyPlains
                }
            }
            // Hot & dry
            (t, h) if t > 0.7 && h < 0.3 => Biome::Desert,
            // Hot & wet
            (t, h) if t > 0.7 && h > 0.7 => Biome::Jungle,
            // Temperate & wet
            (t, h) if t > 0.4 && t < 0.7 && h > 0.6 => Biome::Swamp,
            // Temperate & medium
            (t, h) if t > 0.4 && t < 0.7 && h > 0.3 => Biome::Forest,
            // Cold & medium
            (t, h) if t < 0.4 && h > 0.4 => Biome::Taiga,
            // Default
            _ => Biome::Plains,
        }
    }

    fn generate_column(
        &self,
        blocks: &mut [BlockId],
        x: usize,
        z: usize,
        chunk_y: i32,
        surface_height: i32,
        biome: Biome,
        world_x: i32,
        world_z: i32,
    ) {
        const SIZE: usize = CHUNK_SIZE as usize;
        let water_level = self.config.surface.water_level;
        let beach_depth = self.config.surface.beach_depth;

        for y in 0..SIZE {
            let world_y = chunk_y * SIZE as i32 + y as i32;
            let idx = x + y * SIZE + z * SIZE * SIZE;

            // Check for caves
            if self.config.caves.enabled && world_y > 0 && world_y < 255 {
                if self.is_cave(world_x, world_y, world_z) {
                    blocks[idx] = if world_y <= water_level {
                        exsolis_common::legacy_ids::WATER as BlockId
                    } else {
                        exsolis_common::AIR_ID
                    };
                    continue;
                }
            }

            if world_y < surface_height - beach_depth {
                // Deep underground: stone
                blocks[idx] = biome.stone_block();
            } else if world_y < surface_height {
                // Filler layer
                blocks[idx] = biome.filler_block();
            } else if world_y == surface_height {
                // Surface block
                if world_y <= water_level {
                    blocks[idx] = exsolis_common::legacy_ids::WATER as BlockId;
                } else {
                    blocks[idx] = biome.surface_block();
                }
            } else if world_y <= water_level {
                // Below water level but above surface: water
                blocks[idx] = exsolis_common::legacy_ids::WATER as BlockId;
            } else {
                // Above surface: air
                blocks[idx] = exsolis_common::AIR_ID;
            }
        }
    }

    fn is_cave(&self, x: i32, y: i32, z: i32) -> bool {
        let cave = self.cave_noise.sample_3d(
            x as f64,
            y as f64,
            z as f64,
            &self.config.caves.cave_noise,
        );
        
        let carver = self.carver_noise.sample_3d(
            x as f64 * self.config.caves.carver_frequency as f64,
            y as f64 * self.config.caves.carver_frequency as f64,
            z as f64 * self.config.caves.carver_frequency as f64,
            &self.config.caves.carver_noise,
        );

        let large_cave = self.large_cave_noise.sample_3d(
            x as f64 * 0.5,
            y as f64 * 0.5,
            z as f64 * 0.5,
            &self.config.caves.cave_noise,
        );

        // Combine cave systems
        cave > self.config.caves.cave_threshold as f64
            || carver > self.config.caves.carver_threshold as f64
            || large_cave > self.config.caves.large_cave_threshold as f64
    }
}

/// World generator resource - shared across threads for chunk generation
#[derive(bevy_ecs::resource::Resource, Clone)]
pub struct WorldGen {
    generator: WorldGenerator,
    #[allow(dead_code)]
    config: WorldGenConfig,
}

impl WorldGen {
    /// Create a new WorldGen from a Lua state
    pub fn from_lua(lua: &mlua::Lua) -> Self {
        let config = load_worldgen_config(lua).unwrap_or_default();
        let generator = WorldGenerator::new(config.clone());
        Self {
            generator,
            config,
        }
    }

    pub fn generate_chunk(&self, coord: ChunkCoord) -> exsolis_common::ChunkBlocks {
        let blocks = self.generator.generate_chunk(coord);
        exsolis_common::ChunkBlocks::pack(&blocks)
    }
}

/// Load world generation config from Lua
pub fn load_worldgen_config(lua: &mlua::Lua) -> mlua::Result<WorldGenConfig> {
    let globals = lua.globals();
    let config_table: mlua::Table = globals.get("worldgen").unwrap_or_else(|_| lua.create_table().unwrap());

    let seed: u64 = config_table.get("seed").unwrap_or(12345);

    let terrain_table: mlua::Table = config_table.get("terrain").unwrap_or_else(|_| lua.create_table().unwrap());
    let caves_table: mlua::Table = config_table.get("caves").unwrap_or_else(|_| lua.create_table().unwrap());
    let biomes_table: mlua::Table = config_table.get("biomes").unwrap_or_else(|_| lua.create_table().unwrap());
    let surface_table: mlua::Table = config_table.get("surface").unwrap_or_else(|_| lua.create_table().unwrap());

    let terrain = TerrainConfig {
        continentalness: load_noise_layer(&terrain_table, "continentalness", &NoiseLayerConfig {
            frequency: 0.001,
            amplitude: 1.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        erosion: load_noise_layer(&terrain_table, "erosion", &NoiseLayerConfig {
            frequency: 0.002,
            amplitude: 1.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        peaks_valleys: load_noise_layer(&terrain_table, "peaks_valleys", &NoiseLayerConfig {
            frequency: 0.005,
            amplitude: 1.0,
            octaves: 3,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        base_height: terrain_table.get("base_height").unwrap_or(64.0),
        height_scale: terrain_table.get("height_scale").unwrap_or(64.0),
    };

    let caves = CavesConfig {
        enabled: caves_table.get("enabled").unwrap_or(true),
        cave_noise: load_noise_layer(&caves_table, "cave_noise", &NoiseLayerConfig {
            frequency: 0.02,
            amplitude: 1.0,
            octaves: 2,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        carver_noise: load_noise_layer(&caves_table, "carver_noise", &NoiseLayerConfig {
            frequency: 0.005,
            amplitude: 1.0,
            octaves: 3,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        cave_threshold: caves_table.get("cave_threshold").unwrap_or(0.4),
        carver_threshold: caves_table.get("carver_threshold").unwrap_or(0.3),
        large_cave_threshold: caves_table.get("large_cave_threshold").unwrap_or(0.6),
        carver_frequency: caves_table.get("carver_frequency").unwrap_or(0.003),
    };

    let biomes = BiomesConfig {
        temperature: load_noise_layer(&biomes_table, "temperature", &NoiseLayerConfig {
            frequency: 0.0005,
            amplitude: 1.0,
            octaves: 3,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        humidity: load_noise_layer(&biomes_table, "humidity", &NoiseLayerConfig {
            frequency: 0.0007,
            amplitude: 1.0,
            octaves: 3,
            persistence: 0.5,
            lacunarity: 2.0,
        }),
        continentalness_weight: biomes_table.get("continentalness_weight").unwrap_or(1.0),
        erosion_weight: biomes_table.get("erosion_weight").unwrap_or(1.0),
    };

    let surface = SurfaceConfig {
        top_block: read_block_id(&surface_table, "top_block", exsolis_common::legacy_ids::GRASS),
        filler_block: read_block_id(
            &surface_table,
            "filler_block",
            exsolis_common::legacy_ids::DIRT,
        ),
        stone_block: read_block_id(
            &surface_table,
            "stone_block",
            exsolis_common::legacy_ids::STONE,
        ),
        water_level: surface_table.get("water_level").unwrap_or(63),
        beach_depth: surface_table.get("beach_depth").unwrap_or(4),
    };

    Ok(WorldGenConfig {
        seed,
        terrain,
        caves,
        biomes,
        surface,
    })
}

/// Read a legacy u8 block id from a Lua config table, falling back to
/// `default` when missing or unknown. Frozen M0 ids match `BlockId`
/// numerically, so the cast is a no-op for the built-in set.
fn read_block_id(table: &mlua::Table, key: &str, default: u8) -> BlockId {
    let raw: u8 = table.get(key).unwrap_or(default);
    if raw <= exsolis_common::legacy_ids::CHEST {
        raw as BlockId
    } else {
        default as BlockId
    }
}

fn load_noise_layer(table: &mlua::Table, key: &str, default: &NoiseLayerConfig) -> NoiseLayerConfig {
    let layer_table: Option<mlua::Table> = table.get(key).ok();
    match layer_table {
        Some(t) => NoiseLayerConfig {
            frequency: t.get("frequency").unwrap_or(default.frequency),
            amplitude: t.get("amplitude").unwrap_or(default.amplitude),
            octaves: t.get("octaves").unwrap_or(default.octaves),
            persistence: t.get("persistence").unwrap_or(default.persistence),
            lacunarity: t.get("lacunarity").unwrap_or(default.lacunarity),
        },
        None => default.clone(),
    }
}