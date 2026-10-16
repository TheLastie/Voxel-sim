use glam::Vec3;

pub const WORLD_SIDE_M: f32 = 10.0;
pub const WORLD_CELL_M: f32 = 0.1;
pub const WORLD_SIDE_CELLS: usize = (WORLD_SIDE_M / WORLD_CELL_M) as usize;

#[derive(Clone, Copy, Debug)]
pub struct TerrainVoxel {
    pub center: Vec3,
    pub size: Vec3,
    pub color: [f32; 4],
}

#[derive(Debug)]
pub struct GeneratedWorld {
    pub voxels: Vec<TerrainVoxel>,
    terrain_voxels: Vec<TerrainVoxel>,
    plants: Vec<Plant>,
    seed_hash: u64,
    pub age_years: f32,
    generated_year: u32,
}

#[derive(Clone, Copy, Debug)]
struct Plant {
    x: f32,
    y: f32,
    soil_z: f32,
    moisture: f32,
    fertility: f32,
    species: TreeSpecies,
}

#[derive(Clone, Copy, Debug)]
enum TreeSpecies {
    Oak,
    Birch,
    Acacia,
}

impl GeneratedWorld {
    /// Build a deterministic 10 m × 10 m terrain heightfield from the seed.
    /// These coarse visual voxels are not attached to the centimeter physics lattice.
    pub fn from_seed(seed: &str) -> Self {
        let seed_hash = hash_seed(seed);
        let mut terrain_voxels = Vec::with_capacity(WORLD_SIDE_CELLS * WORLD_SIDE_CELLS);
        for y in 0..WORLD_SIDE_CELLS {
            for x in 0..WORLD_SIDE_CELLS {
                let large = value_noise(seed_hash, x as f32 / 34.0, y as f32 / 34.0);
                let medium = value_noise(seed_hash ^ 0xA076_1D64_78BD_642F, x as f32 / 13.0, y as f32 / 13.0);
                let fine = value_noise(seed_hash ^ 0xE703_7ED1_A0B4_28DB, x as f32 / 5.0, y as f32 / 5.0);
                let height = (0.55 + large * 0.72 + medium * 0.26 + fine * 0.08).max(0.16);
                let variation = (large * 0.5 + medium * 0.35 + fine * 0.15).clamp(-1.0, 1.0);
                let color = [
                    0.31 + variation * 0.055,
                    0.34 + variation * 0.08,
                    0.19 + variation * 0.045,
                    1.0,
                ];
                terrain_voxels.push(TerrainVoxel {
                    center: Vec3::new(
                        (x as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                        (y as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                        height * 0.5,
                    ),
                    size: Vec3::new(WORLD_CELL_M, WORLD_CELL_M, height),
                    color,
                });
            }
        }
        let mut world = Self {
            voxels: Vec::new(),
            terrain_voxels,
            plants: make_plants(seed_hash),
            seed_hash,
            age_years: 8.0,
            generated_year: u32::MAX,
        };
        world.rebuild_vegetation();
        world
    }

    /// Advance the ecological preview clock. One wall second represents one
    /// procedural growth year; physics remains paused and is not time-scaled.
    pub fn advance_growth(&mut self, elapsed_seconds: f32) {
        self.age_years = (self.age_years + elapsed_seconds.max(0.0)).min(100.0);
        let year = self.age_years.floor() as u32;
        if year != self.generated_year {
            self.rebuild_vegetation();
        }
    }

    pub fn vegetation_summary(&self) -> String {
        let mut oak = 0;
        let mut birch = 0;
        let mut acacia = 0;
        for plant in &self.plants {
            match plant.species {
                TreeSpecies::Oak => oak += 1,
                TreeSpecies::Birch => birch += 1,
                TreeSpecies::Acacia => acacia += 1,
            }
        }
        format!("OAK {oak} BIRCH {birch} ACACIA {acacia}")
    }

    fn rebuild_vegetation(&mut self) {
        self.voxels.clone_from(&self.terrain_voxels);
        for plant in &self.plants {
            self.voxels
                .extend(plant.voxels_at_age(self.age_years, self.seed_hash));
        }
        self.generated_year = self.age_years.floor() as u32;
    }
}

impl Plant {
    fn voxels_at_age(self, age_years: f32, seed: u64) -> Vec<TerrainVoxel> {
        let growth = (age_years / 100.0).clamp(0.0, 1.0);
        let maturity = 0.08 + 0.92 * growth.powf(0.72);
        let (max_height, trunk_radius, crown_shape, bark_color, leaf_color) = match self.species {
            TreeSpecies::Oak => (
                7.0 * (0.82 + self.fertility * 0.36) * (0.90 + self.moisture * 0.20),
                0.38 * maturity,
                [1.9, 1.75, 1.35],
                [0.36, 0.22, 0.12, 1.0],
                [0.16, 0.39, 0.12, 1.0],
            ),
            TreeSpecies::Birch => (
                8.5 * (0.82 + self.fertility * 0.36),
                0.16 * maturity,
                [0.9, 0.95, 1.9],
                [0.76, 0.72, 0.59, 1.0],
                [0.24, 0.48, 0.18, 1.0],
            ),
            TreeSpecies::Acacia => (
                6.0 * (0.85 + self.fertility * 0.30),
                0.27 * maturity,
                [2.25, 1.95, 0.65],
                [0.39, 0.25, 0.13, 1.0],
                [0.30, 0.43, 0.15, 1.0],
            ),
        };
        let height = max_height * maturity;
        let trunk_height = height * 0.76;
        let trunk_cells = (trunk_height / WORLD_CELL_M).ceil() as i32;
        let max_radius_cells = (trunk_radius / WORLD_CELL_M).ceil().max(1.0) as i32;
        let mut result = Vec::new();
        for z_cell in 0..trunk_cells {
            let z = (z_cell as f32 + 0.5) * WORLD_CELL_M;
            let taper = 0.72 + 0.28 * (1.0 - z / trunk_height.max(WORLD_CELL_M));
            let radius = trunk_radius * taper;
            let radius_cells = (radius / WORLD_CELL_M).ceil().max(1.0) as i32;
            for dy in -max_radius_cells..=max_radius_cells {
                for dx in -max_radius_cells..=max_radius_cells {
                    let local_x = dx as f32 * WORLD_CELL_M;
                    let local_y = dy as f32 * WORLD_CELL_M;
                    if local_x * local_x + local_y * local_y
                        > radius_cells as f32 * radius_cells as f32 * WORLD_CELL_M.powi(2)
                    {
                        continue;
                    }
                    let color = if matches!(self.species, TreeSpecies::Birch)
                        && lattice_hash(seed, dx + z_cell * 13, dy) % 9 == 0
                    {
                        [0.31, 0.29, 0.23, 1.0]
                    } else {
                        bark_color
                    };
                    result.push(TerrainVoxel {
                        center: Vec3::new(self.x + local_x, self.y + local_y, self.soil_z + z),
                        size: Vec3::splat(WORLD_CELL_M),
                        color,
                    });
                }
            }
        }

        let crown_scale = maturity;
        let crown_center = Vec3::new(
            self.x,
            self.y,
            self.soil_z + height * if matches!(self.species, TreeSpecies::Acacia) { 0.90 } else { 0.78 },
        );
        let radii = Vec3::from_array(crown_shape) * crown_scale;
        let cells = (radii / WORLD_CELL_M).ceil().as_ivec3();
        for z in -cells.z..=cells.z {
            for y in -cells.y..=cells.y {
                for x in -cells.x..=cells.x {
                    let offset = Vec3::new(x as f32, y as f32, z as f32) * WORLD_CELL_M;
                    let normalized = offset / radii.max(Vec3::splat(WORLD_CELL_M));
                    if normalized.length_squared() > 1.0 {
                        continue;
                    }
                    let hash = lattice_hash(seed ^ 0xD6E8_FEB8_6659_FD93, x + 17, y - z * 3);
                    if hash % 11 == 0 {
                        continue;
                    }
                    let shade = ((hash >> 24) as f32 / 16_777_215.0 - 0.5) * 0.08;
                    result.push(TerrainVoxel {
                        center: crown_center + offset,
                        size: Vec3::splat(WORLD_CELL_M),
                        color: [
                            (leaf_color[0] + shade).clamp(0.0, 1.0),
                            (leaf_color[1] + shade).clamp(0.0, 1.0),
                            (leaf_color[2] + shade).clamp(0.0, 1.0),
                            1.0,
                        ],
                    });
                }
            }
        }
        result
    }
}

fn make_plants(seed: u64) -> Vec<Plant> {
    [(-2.7, -1.7), (0.2, 0.8), (2.8, -1.2)]
        .into_iter()
        .enumerate()
        .map(|(index, (x, y))| {
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let moisture = (0.52
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx / 28.0, ny / 28.0) * 0.34)
                .clamp(0.0, 1.0);
            let temperature = (0.52
                + value_noise(seed ^ 0x5899_65CC_7537_4CC3, nx / 31.0, ny / 31.0) * 0.38)
                .clamp(0.0, 1.0);
            let fertility = (0.58
                + value_noise(seed ^ index as u64, nx / 20.0, ny / 20.0) * 0.28)
                .clamp(0.1, 1.0);
            let species = if temperature > 0.69 && moisture < 0.43 {
                TreeSpecies::Acacia
            } else if temperature < 0.35 || moisture > 0.78 {
                TreeSpecies::Birch
            } else {
                TreeSpecies::Oak
            };
            Plant {
                x,
                y,
                soil_z: terrain_height(seed, nx as usize, ny as usize),
                moisture,
                fertility,
                species,
            }
        })
        .collect()
}

fn terrain_height(seed: u64, x: usize, y: usize) -> f32 {
    let x = x as f32;
    let y = y as f32;
    let large = value_noise(seed, x / 34.0, y / 34.0);
    let medium = value_noise(seed ^ 0xA076_1D64_78BD_642F, x / 13.0, y / 13.0);
    let fine = value_noise(seed ^ 0xE703_7ED1_A0B4_28DB, x / 5.0, y / 5.0);
    (0.55 + large * 0.72 + medium * 0.26 + fine * 0.08).max(0.16)
}

fn hash_seed(seed: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in seed.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn lattice_hash(seed: u64, x: i32, y: i32) -> u64 {
    let mut value = seed
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = smooth(x - x.floor());
    let ty = smooth(y - y.floor());
    let a = unit_noise(lattice_hash(seed, x0, y0));
    let b = unit_noise(lattice_hash(seed, x0 + 1, y0));
    let c = unit_noise(lattice_hash(seed, x0, y0 + 1));
    let d = unit_noise(lattice_hash(seed, x0 + 1, y0 + 1));
    lerp(lerp(a, b, tx), lerp(c, d, tx), ty)
}

fn unit_noise(value: u64) -> f32 {
    (value >> 40) as f32 / 8_388_607.5 - 1.0
}

fn smooth(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
