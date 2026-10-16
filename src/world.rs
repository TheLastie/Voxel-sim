use std::collections::HashSet;

use glam::Vec3;

pub const WORLD_SIDE_M: f32 = 10.0;
pub const WORLD_CELL_M: f32 = 0.1;
pub const WORLD_SIDE_CELLS: usize = (WORLD_SIDE_M / WORLD_CELL_M) as usize;
/// Resolution of the generated REV lattice. This is a description grid; it is
/// not a claim that every generated cell is currently simulated by the solver.
pub const WORLD_REV_M: f32 = 0.01;
pub const WORLD_REV_SIDE: i32 = (WORLD_SIDE_M / WORLD_REV_M) as i32;
pub const MAX_WORLD_SAMPLE_CELLS: usize = 65_536;
const WORLD_SAMPLE_HEIGHT_M: f32 = 16.0;
const WORLD_SAMPLE_HEIGHT_CELLS: i32 = (WORLD_SAMPLE_HEIGHT_M / WORLD_REV_M) as i32;

/// Geometric identity of a generated 1 cm cell. These tags distinguish plant
/// anatomy, but do not claim that matching physical material parameters exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeneratedCellContent {
    Air,
    Terrain,
    OakWood,
    OakFoliage,
    BirchWood,
    BirchFoliage,
    AcaciaWood,
    AcaciaFoliage,
    Grass,
    Fern,
    Mushroom,
}

impl GeneratedCellContent {
    fn priority(self) -> u8 {
        match self {
            Self::OakWood | Self::BirchWood | Self::AcaciaWood => 0,
            Self::OakFoliage | Self::BirchFoliage | Self::AcaciaFoliage => 1,
            Self::Mushroom => 2,
            Self::Fern => 3,
            Self::Grass => 4,
            Self::Terrain => 5,
            Self::Air => 6,
        }
    }
}

/// Seed-derived state at one 1 cm world coordinate. This describes generated
/// content and ecology; physical mass, energy, and material response are added
/// when this representation is connected to the engine's `Voxel` state.
#[derive(Clone, Copy, Debug)]
pub struct GeneratedCell {
    pub cell: [i32; 3],
    pub content: GeneratedCellContent,
    pub terrain_height_m: f32,
    pub soil_moisture: f32,
    pub fertility: f32,
    pub temperature_kelvin: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TerrainVoxel {
    pub center: Vec3,
    pub size: Vec3,
    pub color: [f32; 4],
    pub content: GeneratedCellContent,
}

#[derive(Debug)]
pub struct GeneratedWorld {
    pub voxels: Vec<TerrainVoxel>,
    terrain_voxels: Vec<TerrainVoxel>,
    plants: Vec<Plant>,
    plant_branches: Vec<Vec<BranchSegment>>,
    seed_hash: u64,
    pub age_years: f32,
    geometry_age_years: f32,
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
struct BranchSegment {
    start: Vec3,
    end: Vec3,
    radius_m: f32,
}

#[derive(Clone, Copy, Debug)]
enum TreeSpecies {
    Oak,
    Birch,
    Acacia,
}

impl TreeSpecies {
    fn generated_wood(self) -> GeneratedCellContent {
        match self {
            Self::Oak => GeneratedCellContent::OakWood,
            Self::Birch => GeneratedCellContent::BirchWood,
            Self::Acacia => GeneratedCellContent::AcaciaWood,
        }
    }

    fn generated_foliage(self) -> GeneratedCellContent {
        match self {
            Self::Oak => GeneratedCellContent::OakFoliage,
            Self::Birch => GeneratedCellContent::BirchFoliage,
            Self::Acacia => GeneratedCellContent::AcaciaFoliage,
        }
    }
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
                    content: GeneratedCellContent::Terrain,
                });
            }
        }
        let mut world = Self {
            voxels: Vec::new(),
            terrain_voxels,
            plants: make_plants(seed_hash),
            plant_branches: Vec::new(),
            seed_hash,
            age_years: 8.0,
            geometry_age_years: 8.0,
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

    /// Sample one cell on the deterministic 1 cm lattice. Coordinates start
    /// at the south-west world corner; valid horizontal indices are 0..1000,
    /// and the vertical sample domain is 0..1600. Cells represent the current
    /// generated geometry, not dynamic solver state.
    pub fn sample_cell(&self, cell: [i32; 3]) -> Option<GeneratedCell> {
        let [x, y, z] = cell;
        if !(0..WORLD_REV_SIDE).contains(&x)
            || !(0..WORLD_REV_SIDE).contains(&y)
            || !(0..WORLD_SAMPLE_HEIGHT_CELLS).contains(&z)
        {
            return None;
        }

        let x_m = (x as f32 + 0.5) * WORLD_REV_M;
        let y_m = (y as f32 + 0.5) * WORLD_REV_M;
        let z_m = (z as f32 + 0.5) * WORLD_REV_M;
        let field_x = x_m / WORLD_CELL_M;
        let field_y = y_m / WORLD_CELL_M;
        let terrain_height_m = terrain_height(self.seed_hash, field_x, field_y);
        let (soil_moisture, fertility) = ecology_at(self.seed_hash, field_x, field_y);
        let temperature_kelvin = climate_temperature(self.seed_hash, field_x, field_y, z_m);

        let content = if z_m <= terrain_height_m {
            GeneratedCellContent::Terrain
        } else {
            self.vegetation_at_point(
                cell,
                Vec3::new(x_m - WORLD_SIDE_M * 0.5, y_m - WORLD_SIDE_M * 0.5, z_m),
            )
            .unwrap_or(GeneratedCellContent::Air)
        };

        Some(GeneratedCell {
            cell,
            content,
            terrain_height_m,
            soil_moisture,
            fertility,
            temperature_kelvin,
        })
    }

    /// Sample a half-open integer-cell region, subject to a hard work limit.
    /// A request larger than `MAX_WORLD_SAMPLE_CELLS` returns `None` instead
    /// of allocating or iterating an unexpectedly large fraction of the world.
    pub fn sample_region(
        &self,
        min_cell: [i32; 3],
        max_cell_exclusive: [i32; 3],
    ) -> Option<Vec<GeneratedCell>> {
        let extent = [
            max_cell_exclusive[0].checked_sub(min_cell[0])?,
            max_cell_exclusive[1].checked_sub(min_cell[1])?,
            max_cell_exclusive[2].checked_sub(min_cell[2])?,
        ];
        if extent.iter().any(|&n| n <= 0) {
            return None;
        }
        let count = extent
            .into_iter()
            .try_fold(1usize, |total, n| total.checked_mul(n as usize))?;
        if count > MAX_WORLD_SAMPLE_CELLS {
            return None;
        }

        let mut cells = Vec::with_capacity(count);
        for z in min_cell[2]..max_cell_exclusive[2] {
            for y in min_cell[1]..max_cell_exclusive[1] {
                for x in min_cell[0]..max_cell_exclusive[0] {
                    cells.push(self.sample_cell([x, y, z])?);
                }
            }
        }
        Some(cells)
    }

    fn vegetation_at_point(&self, cell: [i32; 3], point: Vec3) -> Option<GeneratedCellContent> {
        self.plants
            .iter()
            .zip(&self.plant_branches)
            .filter_map(|(plant, branches)| {
                plant.content_at_point(
                    self.geometry_age_years,
                    point,
                    self.seed_hash,
                    cell,
                    branches,
                )
            })
            .chain(ground_flora_at_point(
                self.seed_hash,
                self.geometry_age_years,
                point,
            ))
            .min_by_key(|content| content.priority())
    }

    /// Build the displayed LOD from candidate geometry, retaining a preview
    /// voxel only when its center's implicit 1 cm sample has matching content.
    fn rebuild_vegetation(&mut self) {
        self.geometry_age_years = self.age_years.floor();
        self.plant_branches = self
            .plants
            .iter()
            .map(|plant| plant.branch_segments(self.geometry_age_years, self.seed_hash))
            .collect();
        self.voxels.clone_from(&self.terrain_voxels);
        let mut candidates = make_ground_flora(self.seed_hash, self.geometry_age_years);
        for (plant, branches) in self.plants.iter().zip(&self.plant_branches) {
            candidates.extend(plant.voxels_at_age(
                self.geometry_age_years,
                self.seed_hash,
                branches,
            ));
        }
        self.voxels.reserve(candidates.len());
        let mut seen_preview_cells = HashSet::new();
        for voxel in candidates {
            let Some(cell) = cell_containing_point(voxel.center) else {
                continue;
            };
            let Some(sample) = self.sample_cell(cell) else {
                continue;
            };
            if sample.content == voxel.content {
                let key = (
                    ((voxel.center.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).floor() as i32,
                    ((voxel.center.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).floor() as i32,
                    (voxel.center.z / WORLD_CELL_M).floor() as i32,
                    voxel.content,
                );
                if seen_preview_cells.insert(key) {
                    self.voxels.push(voxel);
                }
            }
        }
        self.generated_year = self.age_years.floor() as u32;
    }
}

fn cell_containing_point(point: Vec3) -> Option<[i32; 3]> {
    let cell = [
        ((point.x + WORLD_SIDE_M * 0.5) / WORLD_REV_M - 0.5).round() as i32,
        ((point.y + WORLD_SIDE_M * 0.5) / WORLD_REV_M - 0.5).round() as i32,
        (point.z / WORLD_REV_M - 0.5).round() as i32,
    ];
    if (0..WORLD_REV_SIDE).contains(&cell[0])
        && (0..WORLD_REV_SIDE).contains(&cell[1])
        && (0..WORLD_SAMPLE_HEIGHT_CELLS).contains(&cell[2])
    {
        Some(cell)
    } else {
        None
    }
}

fn make_ground_flora(seed: u64, age_years: f32) -> Vec<TerrainVoxel> {
    let growth = (age_years / 4.0).clamp(0.0, 1.0);
    let mut result = Vec::new();
    // Fixed 0.5 m sampling keeps density seed-locked and cheap to rebuild.
    for gy in 0..20 {
        for gx in 0..20 {
            let cell_hash = lattice_hash(seed ^ 0xA24B_AED4_963E_E407, gx, gy);
            let x = -4.75 + gx as f32 * 0.5;
            let y = -4.75 + gy as f32 * 0.5;
            let nx = ((x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M) as usize;
            let ny = ((y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M) as usize;
            let moisture = (0.5
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx as f32 / 27.0, ny as f32 / 27.0)
                    * 0.38)
                .clamp(0.0, 1.0);
            let fertility = (0.55
                + value_noise(seed ^ 0x5899_65CC_7537_4CC3, nx as f32 / 22.0, ny as f32 / 22.0)
                    * 0.35)
                .clamp(0.1, 1.0);
            let ground_z = terrain_height(seed, nx as f32, ny as f32);

            if cell_hash % 5 != 0 {
                continue;
            }
            let plant_count = 2 + ((cell_hash >> 12) % 4) as usize;
            let is_fern = moisture > 0.58;
            for blade in 0..plant_count {
                let blade_hash = lattice_hash(cell_hash, blade as i32, gy - gx);
                let angle = (blade_hash >> 40) as f32 / 16_777_215.0 * std::f32::consts::TAU;
                let radius = 0.035 + ((blade_hash >> 20) & 0xff) as f32 / 255.0 * 0.12;
                let height = (0.12 + fertility * 0.24) * (0.12 + growth * 0.88);
                let center = Vec3::new(
                    x + angle.cos() * radius,
                    y + angle.sin() * radius,
                    ground_z + height * 0.5,
                );
                let color = if is_fern {
                    [0.11, 0.33 + fertility * 0.10, 0.13, 1.0]
                } else {
                    [0.26, 0.36 + fertility * 0.10, 0.12, 1.0]
                };
                result.push(TerrainVoxel {
                    center,
                    size: Vec3::new(0.035, 0.035, height),
                    color,
                    content: if is_fern {
                        GeneratedCellContent::Fern
                    } else {
                        GeneratedCellContent::Grass
                    },
                });
            }
        }
    }

    // Fruiting bodies appear only where the seeded soil-moisture field is high.
    for gy in 0..25 {
        for gx in 0..25 {
            let x = -4.8 + gx as f32 * 0.4;
            let y = -4.8 + gy as f32 * 0.4;
            let nx = ((x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M) as usize;
            let ny = ((y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M) as usize;
            let moisture = (0.5
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx as f32 / 27.0, ny as f32 / 27.0)
                    * 0.38)
                .clamp(0.0, 1.0);
            let spot = lattice_hash(seed ^ 0xDB4F_0B91_75AE_2165, gx, gy);
            if moisture < 0.64 || spot % 13 != 0 || age_years < 1.0 {
                continue;
            }
            let height = 0.06 + 0.08 * growth * (0.5 + moisture * 0.5);
            let ground_z = terrain_height(seed, nx as f32, ny as f32);
            let stem_height = height * 0.62;
            result.push(TerrainVoxel {
                center: Vec3::new(x, y, ground_z + stem_height * 0.5),
                size: Vec3::new(0.035, 0.035, stem_height),
                color: [0.82, 0.78, 0.62, 1.0],
                content: GeneratedCellContent::Mushroom,
            });
            let cap_color = if spot & 1 == 0 {
                [0.58, 0.20, 0.12, 1.0]
            } else {
                [0.48, 0.32, 0.16, 1.0]
            };
            result.push(TerrainVoxel {
                center: Vec3::new(x, y, ground_z + stem_height + height * 0.12),
                size: Vec3::new(0.12, 0.12, 0.04),
                color: cap_color,
                content: GeneratedCellContent::Mushroom,
            });
        }
    }
    result
}

/// Evaluate seeded grass, fern, and mushroom geometry directly at a point on
/// the REV lattice. This does not inspect or depend on the renderer's LOD list.
fn ground_flora_at_point(
    seed: u64,
    age_years: f32,
    point: Vec3,
) -> Option<GeneratedCellContent> {
    let mut found = None;
    let mut record = |content: GeneratedCellContent| {
        if found.is_none_or(|current: GeneratedCellContent| {
            content.priority() < current.priority()
        }) {
            found = Some(content);
        }
    };

    let gx0 = ((point.x + 4.75) / 0.5).floor() as i32;
    let gy0 = ((point.y + 4.75) / 0.5).floor() as i32;
    for gy in (gy0 - 1).max(0)..=(gy0 + 1).min(19) {
        for gx in (gx0 - 1).max(0)..=(gx0 + 1).min(19) {
            let cell_hash = lattice_hash(seed ^ 0xA24B_AED4_963E_E407, gx, gy);
            if cell_hash % 5 != 0 {
                continue;
            }
            let x = -4.75 + gx as f32 * 0.5;
            let y = -4.75 + gy as f32 * 0.5;
            if (point.x - x).abs() > 0.155 || (point.y - y).abs() > 0.155 {
                continue;
            }
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let moisture = (0.5
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx / 27.0, ny / 27.0) * 0.38)
                .clamp(0.0, 1.0);
            let fertility = (0.55
                + value_noise(seed ^ 0x5899_65CC_7537_4CC3, nx / 22.0, ny / 22.0) * 0.35)
                .clamp(0.1, 1.0);
            let ground_z = terrain_height(seed, nx, ny);
            let height = (0.12 + fertility * 0.24)
                * (0.12 + (age_years / 4.0).clamp(0.0, 1.0) * 0.88);
            let plant_count = 2 + ((cell_hash >> 12) % 4) as usize;
            let content = if moisture > 0.58 {
                GeneratedCellContent::Fern
            } else {
                GeneratedCellContent::Grass
            };
            for blade in 0..plant_count {
                let blade_hash = lattice_hash(cell_hash, blade as i32, gy - gx);
                let angle = (blade_hash >> 40) as f32 / 16_777_215.0 * std::f32::consts::TAU;
                let radius = 0.035 + ((blade_hash >> 20) & 0xff) as f32 / 255.0 * 0.12;
                let center_x = x + angle.cos() * radius;
                let center_y = y + angle.sin() * radius;
                if (point.x - center_x).abs() <= 0.0175
                    && (point.y - center_y).abs() <= 0.0175
                    && point.z >= ground_z
                    && point.z <= ground_z + height
                {
                    record(content);
                }
            }
        }
    }

    let mx0 = ((point.x + 4.8) / 0.4).floor() as i32;
    let my0 = ((point.y + 4.8) / 0.4).floor() as i32;
    for gy in (my0 - 1).max(0)..=(my0 + 1).min(24) {
        for gx in (mx0 - 1).max(0)..=(mx0 + 1).min(24) {
            let x = -4.8 + gx as f32 * 0.4;
            let y = -4.8 + gy as f32 * 0.4;
            if (point.x - x).abs() > 0.06 || (point.y - y).abs() > 0.06 {
                continue;
            }
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let moisture = (0.5
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx / 27.0, ny / 27.0) * 0.38)
                .clamp(0.0, 1.0);
            let spot = lattice_hash(seed ^ 0xDB4F_0B91_75AE_2165, gx, gy);
            if moisture < 0.64 || spot % 13 != 0 || age_years < 1.0 {
                continue;
            }
            let height = 0.06 + 0.08 * (age_years / 4.0).clamp(0.0, 1.0) * (0.5 + moisture * 0.5);
            let ground_z = terrain_height(seed, nx, ny);
            let stem_height = height * 0.62;
            let stem = (point.x - x).abs() <= 0.0175
                && (point.y - y).abs() <= 0.0175
                && point.z >= ground_z
                && point.z <= ground_z + stem_height;
            let cap = (point.x - x).abs() <= 0.06
                && (point.y - y).abs() <= 0.06
                && (point.z - (ground_z + stem_height + height * 0.12)).abs() <= 0.02;
            if stem || cap {
                record(GeneratedCellContent::Mushroom);
            }
        }
    }
    found
}

impl Plant {
    fn branch_segments(self, age_years: f32, seed: u64) -> Vec<BranchSegment> {
        let maturity = 0.08 + 0.92 * (age_years / 100.0).clamp(0.0, 1.0).powf(0.72);
        let (max_height, trunk_radius_factor, crown_radius, tiers, branches_per_tier, slope, width) =
            match self.species {
                TreeSpecies::Oak => (
                    7.0 * (0.82 + self.fertility * 0.36) * (0.90 + self.moisture * 0.20),
                    0.38,
                    1.9,
                    3,
                    4,
                    0.22,
                    0.34,
                ),
                TreeSpecies::Birch => (
                    8.5 * (0.82 + self.fertility * 0.36),
                    0.16,
                    0.9,
                    4,
                    3,
                    0.62,
                    0.40,
                ),
                TreeSpecies::Acacia => (
                    6.0 * (0.85 + self.fertility * 0.30),
                    0.27,
                    2.25,
                    2,
                    4,
                    0.04,
                    0.42,
                ),
            };
        let height = max_height * maturity;
        let trunk_height = height * 0.76;
        let trunk_radius = trunk_radius_factor * maturity;
        let plant_x = ((self.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32;
        let plant_y = ((self.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32;
        let phase_hash = lattice_hash(seed ^ 0x7F4A_7C15_9E37_79B9, plant_x, plant_y);
        let phase = (unit_noise(phase_hash) + 1.0) * std::f32::consts::PI;
        let mut segments = Vec::with_capacity(tiers * branches_per_tier * 3);

        for tier in 0..tiers {
            let tier_fraction = match self.species {
                TreeSpecies::Oak => 0.60 + tier as f32 * 0.10,
                TreeSpecies::Birch => 0.62 + tier as f32 * 0.08,
                TreeSpecies::Acacia => 0.82 + tier as f32 * 0.065,
            };
            let start_z = self.soil_z + trunk_height * tier_fraction;
            for branch in 0..branches_per_tier {
                let index = (tier * branches_per_tier + branch) as i32;
                let hash = lattice_hash(seed ^ phase_hash, index, tier as i32);
                let jitter = unit_noise(hash) * 0.20;
                let angle = phase
                    + std::f32::consts::TAU * branch as f32 / branches_per_tier as f32
                    + jitter;
                let length_scale = 0.62 + unit_noise(hash ^ 0xD6E8_FEB8_6659_FD93) * 0.24;
                let length_m = crown_radius * maturity * length_scale;
                let start = Vec3::new(self.x, self.y, start_z);
                let horizontal = Vec3::new(angle.cos(), angle.sin(), 0.0);
                let end = start + horizontal * length_m + Vec3::Z * (length_m * slope);
                let radius_m = trunk_radius
                    * width
                    * (0.72 + unit_noise(hash.rotate_left(17)).abs() * 0.28);
                let primary = BranchSegment { start, end, radius_m };
                segments.push(primary);

                let midpoint = start.lerp(end, 0.58);
                for side in [-1.0_f32, 1.0] {
                    let split_angle = angle + side * 0.72;
                    let secondary_length = length_m * 0.42;
                    let secondary_end = midpoint
                        + Vec3::new(split_angle.cos(), split_angle.sin(), slope.max(0.18))
                            * secondary_length;
                    segments.push(BranchSegment {
                        start: midpoint,
                        end: secondary_end,
                        radius_m: radius_m * 0.55,
                    });
                }
            }
        }
        segments
    }

    /// Sample the tree's species-specific 1 cm geometry independently of the
    /// 10 cm preview voxels. The trunk is tapered and the crown is an ellipsoid.
    fn content_at_point(
        self,
        age_years: f32,
        point: Vec3,
        seed: u64,
        cell: [i32; 3],
        branches: &[BranchSegment],
    ) -> Option<GeneratedCellContent> {
        let growth = (age_years / 100.0).clamp(0.0, 1.0);
        let maturity = 0.08 + 0.92 * growth.powf(0.72);
        let (max_height, radius_base, crown_shape, crown_z_fraction) = match self.species {
            TreeSpecies::Oak => (
                7.0 * (0.82 + self.fertility * 0.36) * (0.90 + self.moisture * 0.20),
                0.38,
                Vec3::new(1.9, 1.75, 1.35),
                0.78,
            ),
            TreeSpecies::Birch => (
                8.5 * (0.82 + self.fertility * 0.36),
                0.16,
                Vec3::new(0.9, 0.95, 1.9),
                0.78,
            ),
            TreeSpecies::Acacia => (
                6.0 * (0.85 + self.fertility * 0.30),
                0.27,
                Vec3::new(2.25, 1.95, 0.65),
                0.90,
            ),
        };
        let height = max_height * maturity;
        let trunk_height = height * 0.76;
        let trunk_radius = radius_base * maturity;
        let local_z = point.z - self.soil_z;
        let local_x = point.x - self.x;
        let local_y = point.y - self.y;
        if (0.0..trunk_height).contains(&local_z) {
            let taper = 0.72 + 0.28 * (1.0 - local_z / trunk_height.max(WORLD_REV_M));
            let radius = trunk_radius * taper;
            if local_x * local_x + local_y * local_y <= radius * radius {
                return Some(self.species.generated_wood());
            }
        }

        for branch in branches {
            let segment = branch.end - branch.start;
            let length_squared = segment.length_squared();
            if length_squared <= f32::EPSILON {
                continue;
            }
            let along = ((point - branch.start).dot(segment) / length_squared).clamp(0.0, 1.0);
            let closest = branch.start + segment * along;
            if point.distance_squared(closest) <= branch.radius_m.powi(2) {
                return Some(self.species.generated_wood());
            }
        }

        let crown_center_z = self.soil_z + height * crown_z_fraction;
        let radii = crown_shape * maturity;
        let normalized = Vec3::new(
            local_x / radii.x,
            local_y / radii.y,
            (point.z - crown_center_z) / radii.z,
        );
        if normalized.length_squared() <= 1.0 {
            let canopy_seed = seed
                ^ 0xD6E8_FEB8_6659_FD93
                ^ (cell[2] as u32 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87);
            if lattice_hash(canopy_seed, cell[0], cell[1]) % 11 != 0 {
                return Some(self.species.generated_foliage());
            }
        }
        None
    }

    fn voxels_at_age(
        self,
        age_years: f32,
        seed: u64,
        branches: &[BranchSegment],
    ) -> Vec<TerrainVoxel> {
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
                        content: self.species.generated_wood(),
                    });
                }
            }
        }

        for branch in branches {
            let segment = branch.end - branch.start;
            let min = branch.start.min(branch.end) - Vec3::splat(branch.radius_m);
            let max = branch.start.max(branch.end) + Vec3::splat(branch.radius_m);
            let min_cell = [
                ((min.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).floor() as i32,
                ((min.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).floor() as i32,
                (min.z / WORLD_CELL_M).floor() as i32,
            ];
            let max_cell = [
                ((max.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).ceil() as i32,
                ((max.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).ceil() as i32,
                (max.z / WORLD_CELL_M).ceil() as i32,
            ];
            let length_squared = segment.length_squared();
            for z in min_cell[2].max(0)..=max_cell[2].min((WORLD_SAMPLE_HEIGHT_M / WORLD_CELL_M) as i32) {
                for y in min_cell[1].max(0)..=max_cell[1].min(WORLD_SIDE_CELLS as i32 - 1) {
                    for x in min_cell[0].max(0)..=max_cell[0].min(WORLD_SIDE_CELLS as i32 - 1) {
                        let center = Vec3::new(
                            (x as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                            (y as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                            (z as f32 + 0.5) * WORLD_CELL_M,
                        );
                        let along = ((center - branch.start).dot(segment) / length_squared)
                            .clamp(0.0, 1.0);
                        let closest = branch.start + segment * along;
                        let voxel_half_diagonal = WORLD_CELL_M * 0.5 * 3.0_f32.sqrt();
                        if center.distance_squared(closest)
                            <= (branch.radius_m + voxel_half_diagonal).powi(2)
                        {
                            result.push(TerrainVoxel {
                                center,
                                size: Vec3::splat(WORLD_CELL_M),
                                color: bark_color,
                                content: self.species.generated_wood(),
                            });
                        }
                    }
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
                        content: self.species.generated_foliage(),
                    });
                }
            }
        }
        result
    }
}

fn make_plants(seed: u64) -> Vec<Plant> {
    let mut plants = Vec::with_capacity(10);
    for gy in 0..4 {
        for gx in 0..4 {
            let site_hash = lattice_hash(seed ^ 0xA24B_AED4_963E_E407, gx, gy);
            let x = -3.75 + gx as f32 * 2.5 + unit_noise(site_hash) * 0.35;
            let y = -3.75 + gy as f32 * 2.5 + unit_noise(site_hash.rotate_left(29)) * 0.35;
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let soil_z = terrain_height(seed, nx, ny);
            let (moisture, fertility) = ecology_at(seed, nx, ny);
            let site_roll = (unit_noise(site_hash.rotate_left(53)) + 1.0) * 0.5;
            let establishment_chance = 0.30 + fertility * 0.28 + moisture * 0.12;
            if site_roll > establishment_chance {
                continue;
            }
            let temperature_kelvin = climate_temperature(seed, nx, ny, soil_z);
            let climate_index =
                (0.52 + (temperature_kelvin - 288.15) * (0.38 / 11.4)).clamp(0.0, 1.0);
            let species = if climate_index > 0.69 && moisture < 0.43 {
                TreeSpecies::Acacia
            } else if climate_index < 0.35 || moisture > 0.78 {
                TreeSpecies::Birch
            } else {
                TreeSpecies::Oak
            };
            plants.push(Plant {
                x,
                y,
                soil_z,
                moisture,
                fertility,
                species,
            });
        }
    }
    plants
}

fn terrain_height(seed: u64, x: f32, y: f32) -> f32 {
    let large = value_noise(seed, x / 34.0, y / 34.0);
    let medium = value_noise(seed ^ 0xA076_1D64_78BD_642F, x / 13.0, y / 13.0);
    let fine = value_noise(seed ^ 0xE703_7ED1_A0B4_28DB, x / 5.0, y / 5.0);
    (0.55 + large * 0.72 + medium * 0.26 + fine * 0.08).max(0.16)
}

fn ecology_at(seed: u64, x: f32, y: f32) -> (f32, f32) {
    let moisture = (0.5
        + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, x / 27.0, y / 27.0) * 0.38)
        .clamp(0.0, 1.0);
    let fertility = (0.55
        + value_noise(seed ^ 0x5899_65CC_7537_4CC3, x / 22.0, y / 22.0) * 0.35)
        .clamp(0.1, 1.0);
    (moisture, fertility)
}

fn climate_temperature(seed: u64, x: f32, y: f32, altitude_m: f32) -> f32 {
    // The current growth preview uses a seed-derived climate field. This value
    // is ecological input only and is not yet coupled to the thermal solver.
    288.15 + value_noise(seed ^ 0x5899_65CC_7537_4CC3, x / 31.0, y / 31.0) * 11.4
        - altitude_m * 0.0065
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
