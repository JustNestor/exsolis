-- World Generation Configuration
-- This file is hot-reloadable! Edit it while the server is running to see changes.
-- All noise values use OpenSimplex2 with fractal Brownian motion (fBm).

worldgen = {
    -- Global seed for the world. Change this to get a completely different world.
    seed = 12345,

    -- Terrain height generation using layered noise (Minecraft-style)
    terrain = {
        -- Continentalness: large-scale land vs ocean distribution
        -- Low values = ocean, high values = continents
        continentalness = {
            frequency = 0.001,      -- How zoomed out the noise is (lower = larger features)
            amplitude = 1.0,        -- Strength of this noise layer
            octaves = 4,            -- Number of noise layers (more = more detail)
            persistence = 0.5,      -- How much each octave contributes (0-1)
            lacunarity = 2.0,       -- Frequency multiplier per octave
        },

        -- Erosion: creates valleys and smooths terrain
        -- Low values = eroded/valleys, high values = rugged
        erosion = {
            frequency = 0.002,
            amplitude = 1.0,
            octaves = 4,
            persistence = 0.5,
            lacunarity = 2.0,
        },

        -- Peaks and Valleys: adds mountain ridges and deep valleys
        -- Positive = peaks, negative = valleys
        peaks_valleys = {
            frequency = 0.005,
            amplitude = 1.0,
            octaves = 3,
            persistence = 0.5,
            lacunarity = 2.0,
        },

        -- Base height (blocks) - the "sea level" baseline
        base_height = 64.0,

        -- Height scale - how much noise affects final height
        height_scale = 64.0,
    },

    -- 3D Cave generation
    caves = {
        enabled = true,

        -- Main cave noise (3D) - creates spherical/irregular caves
        cave_noise = {
            frequency = 0.02,
            amplitude = 1.0,
            octaves = 2,
            persistence = 0.5,
            lacunarity = 2.0,
        },

        -- Carver noise (3D) - creates long winding tunnels/canyons
        carver_noise = {
            frequency = 0.005,
            amplitude = 1.0,
            octaves = 3,
            persistence = 0.5,
            lacunarity = 2.0,
        },

        -- Thresholds: noise values above these create caves
        -- Lower threshold = more caves
        cave_threshold = 0.4,         -- Main caves
        carver_threshold = 0.3,       -- Tunnel carvers
        large_cave_threshold = 0.6,   -- Rare large caverns

        -- Carver frequency multiplier (makes carvers larger/wider)
        carver_frequency = 0.003,
    },

    -- Biome determination (temperature, humidity, etc.)
    biomes = {
        -- Temperature noise (2D) - hot vs cold regions
        temperature = {
            frequency = 0.0005,
            amplitude = 1.0,
            octaves = 3,
            persistence = 0.5,
            lacunarity = 2.0,
        },

        -- Humidity noise (2D) - wet vs dry regions
        humidity = {
            frequency = 0.0007,
            amplitude = 1.0,
            octaves = 3,
            persistence = 0.5,
            lacunarity = 2.0,
        },

        -- How much continentalness/erosion affect biome selection
        continentalness_weight = 1.0,
        erosion_weight = 1.0,
    },

    -- Surface blocks and water level
    surface = {
        -- Block names (see exsolis_common::BlockRegistry::default_registry).
        -- Numeric ids stay frozen (0=Air, 1=Stone, ...) for worldgen compat.
        top_block = 3,        -- Surface block (Grass)
        filler_block = 2,     -- Sub-surface block (Dirt)
        stone_block = 1,      -- Deep stone block (Stone)

        water_level = 63,     -- Sea level (blocks at/below this become water in oceans)
        beach_depth = 4,      -- How many blocks of sand below water level at beaches
    },
}

--[[
Noise Parameter Guide:
---------------------
frequency:  How "zoomed in" the noise is.
  - 0.0005 = huge features (thousands of blocks)
  - 0.001  = large features (hundreds of blocks)
  - 0.01   = medium features (tens of blocks)
  - 0.1    = small features (few blocks)

amplitude:  Strength of the noise contribution (usually 1.0)

octaves:    Number of noise layers combined.
  - 1 = smooth, simple
  - 2-3 = natural terrain
  - 4-6 = very detailed, rugged
  - More octaves = more CPU time

persistence: How much each octave contributes vs the previous.
  - 0.5 = standard (each octave half strength)
  - Higher = more small detail
  - Lower = smoother

lacunarity:  Frequency multiplier per octave.
  - 2.0 = standard (each octave doubles frequency)
  - Higher = more fine detail
  - Lower = smoother transitions

Biome System:
-------------
Biomes are determined by combining:
1. Continentalness (land vs ocean)
2. Erosion (rugged vs smooth)
3. Temperature (hot vs cold)
4. Humidity (wet vs dry)
5. Height (elevation)

Biome types: Ocean, Beach, Plains, Forest, Desert, Mountains,
             SnowyPlains, SnowyMountains, Swamp, Jungle, Taiga

Cave System:
------------
Three cave systems layered together:
1. Cave noise (spherical caves) - threshold 0.4
2. Carver noise (winding tunnels) - threshold 0.3
3. Large cave noise (rare huge caverns) - threshold 0.6

Adjust thresholds to control cave density:
- Lower threshold = more caves
- Higher threshold = fewer caves
--]]