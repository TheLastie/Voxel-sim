use std::collections::{HashMap, HashSet, VecDeque};

use rayon::prelude::*;

use crate::{
    scene::{Scene, SceneConfig, VoxelSpawn},
    voxel::{Composition, Microstructure},
};
use glam::{DVec3, Quat, Vec3};

/// A square 10 km² map. The centimeter REV field is generated on demand from
/// the seed; the full map is never expanded into an in-memory voxel array.
pub const WORLD_SIDE_M: f32 = 3162.2776;
pub const WORLD_CELL_M: f32 = 0.1;
pub const WORLD_SIDE_CELLS: usize = (WORLD_SIDE_M / WORLD_CELL_M) as usize;
const WORLD_OVERVIEW_SIDE_CELLS: usize = 100;
const WORLD_OVERVIEW_CELL_M: f32 = WORLD_SIDE_M / WORLD_OVERVIEW_SIDE_CELLS as f32;
/// Resolution of the generated REV lattice. This is a description grid; it is
/// not a claim that every generated cell is currently simulated by the solver.
pub const WORLD_REV_M: f32 = 0.01;
pub const WORLD_REV_SIDE: i32 = (WORLD_SIDE_M / WORLD_REV_M) as i32;
const WORLD_REVS_PER_RENDER_CELL: i32 = 10;
pub const MAX_WORLD_SAMPLE_CELLS: usize = 65_536;
/// Maximum fine REVs accepted in one independently allocated physical patch.
pub const MAX_PHYSICAL_PATCH_REVS: usize = 1_000_000;
const WORLD_SAMPLE_HEIGHT_M: f32 = 512.0;
const WORLD_SAMPLE_HEIGHT_CELLS: i32 = (WORLD_SAMPLE_HEIGHT_M / WORLD_REV_M) as i32;
const SIMULATION_YEAR_SECONDS: f32 = 365.25 * 24.0 * 60.0 * 60.0;
const PLANT_QUERY_BUCKET_M: f32 = 4.0;
const LEAF_CLUSTER_RADIUS_M: f32 = 0.42;

/// Geometric identity of a generated 1 cm cell. These tags distinguish plant
/// anatomy, but do not claim that matching physical material parameters exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeneratedCellContent {
    Air,
    Terrain,
    Seed,
    PlantWood,
    PlantFoliage,
    Grass,
    Fern,
    Wildflower,
    Mushroom,
    Bolete,
    Puffball,
    Mycelium,
}

impl GeneratedCellContent {
    fn priority(self) -> u8 {
        match self {
            Self::Seed => 0,
            Self::PlantWood => 0,
            Self::PlantFoliage => 1,
            Self::Mushroom | Self::Bolete | Self::Puffball | Self::Mycelium => 2,
            Self::Fern | Self::Wildflower => 3,
            Self::Grass => 4,
            Self::Terrain => 5,
            Self::Air => 6,
        }
    }
}

/// Seed-derived state at one 1 cm world coordinate. This describes generated
/// content, ecology, and the local wood-fiber direction; physical mass, energy,
/// and material response are added when connected to the engine's `Voxel`
/// state.
#[derive(Clone, Copy, Debug)]
pub struct GeneratedCell {
    pub cell: [i32; 3],
    pub content: GeneratedCellContent,
    /// Genetic state carried by a dormant tree seed in this cell.
    pub seed_genome: Option<SeedGenome>,
    /// Principal wood-fiber axis in world coordinates; absent for non-wood
    /// cells.
    pub fiber_direction: Option<Vec3>,
    pub terrain_height_m: f32,
    pub soil_moisture: f32,
    pub fertility: f32,
    pub temperature_kelvin: f32,
}

/// Heritable parameters stored on an individual tree seed. Trait values are
/// deterministic for a world seed and seed identity. Mutation is reserved for
/// offspring once reproduction and calibrated rates are implemented.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeedGenome {
    pub individual_id: u64,
    pub morphotype_id: u64,
    pub maximum_height_m: f32,
    pub crown_spread_m: f32,
    pub mature_trunk_radius_m: f32,
    pub shade_tolerance: f32,
    pub water_demand: f32,
    pub nutrient_demand: f32,
    pub longevity_years: f32,
    pub moisture_optimum: f32,
    pub temperature_optimum_kelvin: f32,
    pub root_count: u8,
    pub root_extent_m: f32,
    pub root_depth_factor: f32,
    pub trunk_height_fraction: f32,
    pub branching_rate: f32,
    pub apical_dominance: f32,
    pub branch_spacing_m: f32,
    pub branch_slope: f32,
    pub branch_width: f32,
    pub leaf_retention: f32,
    pub mycorrhizal_affinity: f32,
    pub germination_delay_years: f32,
    pub mutation_probability: f32,
}

/// Caller-supplied physical material state for one generated cell. A scene
/// derives mass and internal energy from the composition and temperature;
/// constraints remain independent of the cell's finite mass.
#[derive(Clone, Debug)]
pub struct GeneratedCellPhysics {
    pub composition: Composition,
    pub anisotropy: f32,
    pub velocity: Vec3,
    pub kinematic: bool,
}

/// A bounded physical world region whose solver coordinates are local to an
/// integer-aligned world-cell origin. Global identity is retained separately,
/// so REV spacing does not depend on distance from the world origin.
pub struct PhysicalWorldPatch {
    pub origin_cell: [i32; 3],
    pub origin_world_m: DVec3,
    /// Stable generated-world identity aligned with `scene.voxels` and
    /// retained even after those physical cells move.
    pub source_cells: Vec<[i32; 3]>,
    pub scene: Scene,
}

impl PhysicalWorldPatch {
    pub fn world_position(&self, local_position_m: DVec3) -> DVec3 {
        self.origin_world_m + local_position_m
    }

    pub fn local_position(&self, world_position_m: DVec3) -> DVec3 {
        world_position_m - self.origin_world_m
    }
}

impl GeneratedCell {
    /// Convert this generated 1 cm cell into a world-coordinate spawn input.
    /// The map origin is not aligned to the engine's global REV lattice, so use
    /// `to_local_voxel_spawn` when constructing a solver scene. Composition is
    /// explicit; the scene derives mass and internal energy from it and the
    /// sampled temperature.
    pub fn to_voxel_spawn(
        self,
        composition: Composition,
        anisotropy: f32,
        velocity: Vec3,
        kinematic: bool,
    ) -> Result<VoxelSpawn, &'static str> {
        let cell_size = crate::coordinates::CELL_M;
        let half_map_m = f64::from(WORLD_SIDE_M) * 0.5;
        let center = DVec3::new(
            (f64::from(self.cell[0]) + 0.5) * cell_size - half_map_m,
            (f64::from(self.cell[1]) + 0.5) * cell_size - half_map_m,
            f64::from(self.cell[2]) * cell_size + cell_size * 0.5,
        );
        self.to_voxel_spawn_at(center, composition, anisotropy, velocity, kinematic)
    }

    /// Convert a generated cell to a REV position relative to an integer cell
    /// origin. Subtracting cell indices before converting to floating point
    /// preserves adjacent-centimetre coordinates at any supported world offset.
    pub fn to_local_voxel_spawn(
        self,
        origin_cell: [i32; 3],
        composition: Composition,
        anisotropy: f32,
        velocity: Vec3,
        kinematic: bool,
    ) -> Result<VoxelSpawn, &'static str> {
        let cell_size = crate::coordinates::CELL_M;
        let delta = std::array::from_fn::<_, 3, _>(|axis| {
            i64::from(self.cell[axis]) - i64::from(origin_cell[axis])
        });
        let center = DVec3::new(
            delta[0] as f64 * cell_size + cell_size * 0.5,
            delta[1] as f64 * cell_size + cell_size * 0.5,
            delta[2] as f64 * cell_size + cell_size * 0.5,
        );
        self.to_voxel_spawn_at(center, composition, anisotropy, velocity, kinematic)
    }

    fn to_voxel_spawn_at(
        self,
        center: DVec3,
        composition: Composition,
        anisotropy: f32,
        velocity: Vec3,
        kinematic: bool,
    ) -> Result<VoxelSpawn, &'static str> {
        if !anisotropy.is_finite() || !(0.0..=1.0).contains(&anisotropy) {
            return Err("material anisotropy must be finite and between zero and one");
        }
        if !velocity.is_finite() {
            return Err("generated voxel velocity must be finite");
        }
        let orientation = if anisotropy == 0.0 {
            Quat::IDENTITY
        } else {
            let direction = self
                .fiber_direction
                .ok_or("anisotropic generated cell has no fiber direction")?;
            if !direction.is_finite() || direction.length_squared() < 1e-8 {
                return Err("generated fiber direction must be finite and nonzero");
            }
            Quat::from_rotation_arc(Vec3::Z, direction.normalize())
        };
        Ok(VoxelSpawn {
            center,
            composition,
            velocity,
            microstructure: Microstructure {
                orientation,
                anisotropy,
            },
            kinematic,
            temperature_kelvin: self.temperature_kelvin,
        })
    }
}

fn world_cell_origin_m(cell: [i32; 3]) -> DVec3 {
    let cell_size = crate::coordinates::CELL_M;
    let half_map_m = f64::from(WORLD_SIDE_M) * 0.5;
    DVec3::new(
        f64::from(cell[0]) * cell_size - half_map_m,
        f64::from(cell[1]) * cell_size - half_map_m,
        f64::from(cell[2]) * cell_size,
    )
}

fn build_physical_spawns<F>(
    cells: Vec<GeneratedCell>,
    local_origin: Option<[i32; 3]>,
    mut physics_for_cell: F,
) -> Result<(Vec<VoxelSpawn>, Vec<[i32; 3]>), String>
where
    F: FnMut(GeneratedCell) -> Result<Option<GeneratedCellPhysics>, String>,
{
    let mut spawns = Vec::with_capacity(cells.len());
    let mut source_cells = Vec::with_capacity(cells.len());
    for cell in cells {
        let Some(state) = physics_for_cell(cell).map_err(|reason| {
            format!(
                "material policy rejected {:?} at {:?}: {reason}",
                cell.content, cell.cell
            )
        })?
        else {
            continue;
        };
        let spawn = match local_origin {
            Some(origin) => cell.to_local_voxel_spawn(
                origin,
                state.composition,
                state.anisotropy,
                state.velocity,
                state.kinematic,
            ),
            None => cell.to_voxel_spawn(
                state.composition,
                state.anisotropy,
                state.velocity,
                state.kinematic,
            ),
        }
        .map_err(|reason| {
            format!(
                "cannot create physical {:?} at {:?}: {reason}",
                cell.content, cell.cell
            )
        })?;
        spawns.push(spawn);
        source_cells.push(cell.cell);
    }
    Ok((spawns, source_cells))
}

fn assemble_physical_patch<F>(
    config: SceneConfig,
    origin_cell: [i32; 3],
    cells: Vec<GeneratedCell>,
    physics_for_cell: F,
) -> Result<PhysicalWorldPatch, String>
where
    F: FnMut(GeneratedCell) -> Result<Option<GeneratedCellPhysics>, String>,
{
    let (spawns, source_cells) = build_physical_spawns(cells, Some(origin_cell), physics_for_cell)?;
    let scene = Scene::from_voxels(config, spawns)?;
    Ok(PhysicalWorldPatch {
        origin_cell,
        origin_world_m: world_cell_origin_m(origin_cell),
        source_cells,
        scene,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct TerrainVoxel {
    pub center: Vec3,
    pub size: Vec3,
    pub color: [f32; 4],
    pub content: GeneratedCellContent,
    /// Local wood-fiber axis used to keep branch direction when geometry is
    /// reduced to a coarser render LOD.
    pub fiber_direction: Option<Vec3>,
}

#[derive(Clone, Debug)]
pub struct GeneratedWorld {
    /// Optional visible/accessible experiment footprint; the seeded field is unchanged.
    pub view_bounds: Option<(Vec3, Vec3)>,
    pub voxels: Vec<TerrainVoxel>,
    terrain_voxels: Vec<TerrainVoxel>,
    plants: Vec<Plant>,
    morphotypes: Vec<Morphotype>,
    plant_growth: Vec<PlantGrowth>,
    plant_branches: Vec<Vec<BranchSegment>>,
    plant_spatial_index: PlantSpatialIndex,
    mycelium_segments: Vec<MyceliumSegment>,
    mycelium_buckets: Vec<Vec<usize>>,
    seed_hash: u64,
    pub age_years: f32,
    calendar_years: f64,
    geometry_age_years: f32,
    season_phase_offset: f32,
    generated_growth_quarter: u32,
    generated_calendar_quarter: u64,
    geometry_revision: u64,
}

/// Schedule biological growth without running vegetation rebuilds in the
/// window/event thread. Elapsed increments are replayed in order, including
/// increments received while a previous rebuild is still running.
#[derive(Default)]
pub struct GrowthScheduler {
    queued: VecDeque<f32>,
    in_flight: Option<(std::sync::mpsc::Receiver<GeneratedWorld>, f32)>,
    source_key: Option<(u64, u64, u32, u64)>,
}

impl GrowthScheduler {
    fn key(world: &GeneratedWorld) -> (u64, u64, u32, u64) {
        (
            world.seed_hash,
            world.geometry_revision,
            world.age_years.to_bits(),
            world.calendar_years.to_bits(),
        )
    }

    pub fn is_busy(&self) -> bool {
        self.in_flight.is_some() || !self.queued.is_empty()
    }

    /// Poll completed work and retain all valid elapsed simulation time.
    /// Call with zero while paused to finish updates requested before pausing.
    pub fn advance(&mut self, world: &mut GeneratedWorld, elapsed_seconds: f32) {
        let key = Self::key(world);
        if self.source_key.is_some() && self.source_key != Some(key) {
            // The caller replaced or independently advanced the world.
            self.queued.clear();
            self.in_flight = None;
        }
        self.source_key = Some(key);
        if elapsed_seconds.is_finite() && elapsed_seconds > 0.0 {
            self.queued.push_back(elapsed_seconds);
        }
        if let Some((receiver, elapsed)) = &self.in_flight {
            match receiver.try_recv() {
                Ok(next_world) => {
                    *world = next_world;
                    self.in_flight = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.queued.push_front(*elapsed);
                    self.in_flight = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
            }
        }
        while let Some(elapsed) = self.queued.pop_front() {
            let years = elapsed / SIMULATION_YEAR_SECONDS;
            let age = (world.age_years + years).min(100.0);
            let calendar = world.calendar_years + f64::from(years);
            let growth_quarter = (age * 4.0).floor() as u32;
            let calendar_quarter =
                ((calendar + f64::from(world.season_phase_offset)) * 4.0).floor() as u64;
            if growth_quarter == world.generated_growth_quarter
                && calendar_quarter == world.generated_calendar_quarter
            {
                world.advance_growth(elapsed);
                continue;
            }
            let mut snapshot = world.clone();
            let (sender, receiver) = std::sync::mpsc::channel();
            match std::thread::Builder::new()
                .name("world-growth".into())
                .spawn(move || {
                    snapshot.advance_growth(elapsed);
                    let _ = sender.send(snapshot);
                }) {
                Ok(_) => self.in_flight = Some((receiver, elapsed)),
                Err(error) => {
                    self.queued.push_front(elapsed);
                    eprintln!("growth worker could not start: {error}");
                }
            }
            break;
        }
        self.source_key = Some(Self::key(world));
    }
}

/// Gameplay character state in SI units. This is a kinematic controller used
/// for walking; it does not pretend the avatar is a dynamically solved body.
#[derive(Clone, Copy, Debug)]
pub struct WorldCharacter {
    /// Bottom-center position of the character capsule, in metres.
    pub position: Vec3,
    /// Heading in radians, measured around the world Z axis.
    pub heading: f32,
    /// Distance walked, used only to animate the rendered limbs.
    pub walk_distance_m: f32,
    vertical_velocity_mps: f32,
    grounded: bool,
    crouched: bool,
}

impl WorldCharacter {
    pub const RADIUS_M: f32 = 0.28;
    pub const HEIGHT_M: f32 = 1.75;
    pub const CROUCH_HEIGHT_M: f32 = 0.95;
    const GRAVITY_MPS2: f32 = 9.81;
    const JUMP_SPEED_MPS: f32 = 5.1;
    const MAX_STEP_UP_M: f32 = 0.32;

    pub fn spawn(world: &GeneratedWorld) -> Self {
        let position = Vec3::new(0.0, 0.0, world.terrain_height_at(0.0, 0.0));
        Self {
            position,
            heading: 0.0,
            walk_distance_m: 0.0,
            vertical_velocity_mps: 0.0,
            grounded: true,
            crouched: false,
        }
    }

    pub fn eye_position(self) -> Vec3 {
        self.position + Vec3::Z * if self.crouched { 0.78 } else { 1.42 }
    }

    pub fn height(self) -> f32 {
        if self.crouched {
            Self::CROUCH_HEIGHT_M
        } else {
            Self::HEIGHT_M
        }
    }

    /// Release Ctrl to stand only when the complete standing volume is clear.
    /// The feet and horizontal radius are unchanged by changing posture.
    pub fn set_crouching(&mut self, world: &GeneratedWorld, requested: bool) {
        if requested {
            self.crouched = true;
        } else if !world.character_overlaps_wood(self.position, Self::RADIUS_M, Self::HEIGHT_M) {
            self.crouched = false;
        }
    }

    fn inside_bounds(world: &GeneratedWorld, x: f32, y: f32) -> bool {
        if let Some((min, max)) = world.view_bounds {
            return x >= min.x + Self::RADIUS_M
                && x <= max.x - Self::RADIUS_M
                && y >= min.y + Self::RADIUS_M
                && y <= max.y - Self::RADIUS_M;
        }
        let limit = WORLD_SIDE_M * 0.5 - Self::RADIUS_M;
        x.abs() <= limit && y.abs() <= limit
    }

    /// Advance movement in bounded substeps so a slow rendered frame does not
    /// let the capsule tunnel through terrain or generated wood.
    pub fn advance(
        &mut self,
        world: &GeneratedWorld,
        input: Vec3,
        camera_yaw: f32,
        speed_mps: f32,
        sprint: bool,
        jump: bool,
        elapsed_seconds: f32,
    ) {
        let total = elapsed_seconds.clamp(0.0, 0.1);
        if total == 0.0 || !total.is_finite() {
            return;
        }
        let steps = (total / (1.0 / 60.0)).ceil().max(1.0) as usize;
        let dt = total / steps as f32;
        let mut move_input = input;
        move_input.z = 0.0;
        if move_input.length_squared() > 1.0 {
            move_input = move_input.normalize();
        }
        let forward = Vec3::new(-camera_yaw.cos(), -camera_yaw.sin(), 0.0);
        let right = Vec3::new(forward.y, -forward.x, 0.0);
        let planar = (right * move_input.x + forward * move_input.y).normalize_or_zero();
        let speed = speed_mps.max(0.0)
            * if self.crouched {
                0.45
            } else if sprint {
                1.7
            } else {
                1.0
            };

        for _ in 0..steps {
            if jump && self.grounded {
                self.vertical_velocity_mps = Self::JUMP_SPEED_MPS;
                self.grounded = false;
            }

            let displacement = planar * (speed * dt);
            if displacement.length_squared() > 0.0 {
                let proposed = self.position + displacement;
                let terrain_z = world.terrain_height_at(proposed.x, proposed.y);
                let can_step_up = terrain_z <= self.position.z + Self::MAX_STEP_UP_M;
                if can_step_up
                    && Self::inside_bounds(world, proposed.x, proposed.y)
                    && !world.character_overlaps_wood(
                        Vec3::new(proposed.x, proposed.y, self.position.z),
                        Self::RADIUS_M,
                        self.height(),
                    )
                {
                    self.position.x = proposed.x;
                    self.position.y = proposed.y;
                    if self.grounded && terrain_z <= self.position.z + Self::MAX_STEP_UP_M {
                        self.position.z = terrain_z;
                    }
                    self.walk_distance_m += displacement.length();
                    self.heading = planar.y.atan2(planar.x);
                } else {
                    // Slide along a trunk or steep rise when one movement axis
                    // remains clear, as standard capsule controllers do.
                    for axis in [Vec3::X, Vec3::Y] {
                        let partial =
                            Vec3::new(displacement.x * axis.x, displacement.y * axis.y, 0.0);
                        let candidate = self.position + partial;
                        if Self::inside_bounds(world, candidate.x, candidate.y)
                            && world.terrain_height_at(candidate.x, candidate.y)
                                <= self.position.z + Self::MAX_STEP_UP_M
                            && !world.character_overlaps_wood(
                                Vec3::new(candidate.x, candidate.y, self.position.z),
                                Self::RADIUS_M,
                                self.height(),
                            )
                        {
                            self.position.x = candidate.x;
                            self.position.y = candidate.y;
                        }
                    }
                }
            }

            let floor_z = world.terrain_height_at(self.position.x, self.position.y);
            if self.grounded && self.position.z <= floor_z + 0.02 {
                self.position.z = floor_z;
                self.vertical_velocity_mps = 0.0;
                continue;
            }
            self.vertical_velocity_mps -= Self::GRAVITY_MPS2 * dt;
            let next_z = self.position.z + self.vertical_velocity_mps * dt;
            if self.vertical_velocity_mps > 0.0
                && world.character_overlaps_wood(
                    Vec3::new(self.position.x, self.position.y, next_z),
                    Self::RADIUS_M,
                    self.height(),
                )
            {
                self.vertical_velocity_mps = 0.0;
            } else {
                self.position.z = next_z;
            }
            if self.position.z <= floor_z {
                self.position.z = floor_z;
                self.vertical_velocity_mps = 0.0;
                self.grounded = true;
            } else {
                self.grounded = false;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Morphotype {
    pub id: u64,
    pub latin_name: String,
    pub population: usize,
}

#[derive(Clone, Copy, Debug)]
struct Plant {
    genome: SeedGenome,
    x: f32,
    y: f32,
    soil_z: f32,
    moisture: f32,
    fertility: f32,
    temperature_kelvin: f32,
    germination_age_years: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct PlantGrowth {
    stage: GrowthStage,
    light: f32,
    structural_light: f32,
    water: f32,
    nutrients: f32,
    development_age_years: f32,
    vigor: f32,
    height_m: f32,
    trunk_height_m: f32,
    trunk_radius_m: f32,
    crown_spread_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum GrowthStage {
    #[default]
    Seedling,
    Sapling,
    Pole,
    Mature,
    Senescent,
}

impl GrowthStage {
    fn from_development_age(age_years: f32) -> Self {
        match age_years {
            age if age < 2.0 => Self::Seedling,
            age if age < 10.0 => Self::Sapling,
            age if age < 25.0 => Self::Pole,
            age if age < 80.0 => Self::Mature,
            _ => Self::Senescent,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FungusForm {
    content: GeneratedCellContent,
    height_scale: f32,
    stem_fraction: f32,
    stem_radius_m: f32,
    cap_width_m: f32,
    cap_thickness_m: f32,
    cap_color: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct BranchSegment {
    start: Vec3,
    end: Vec3,
    radius_m: f32,
    is_root: bool,
}

#[derive(Clone, Copy, Debug)]
enum PlantSpatialRef {
    Trunk(usize),
    Branch {
        plant_index: usize,
        branch_index: usize,
    },
}

impl PlantSpatialRef {
    fn plant_index(self) -> usize {
        match self {
            Self::Trunk(index)
            | Self::Branch {
                plant_index: index, ..
            } => index,
        }
    }
}

/// Sparse broadphase for implicit 1 cm cell queries. Buckets contain only
/// plant parts whose generated bounds can cover a query point.
#[derive(Clone, Debug, Default)]
struct PlantSpatialIndex {
    canopy: HashMap<(i32, i32), Vec<PlantSpatialRef>>,
    roots: HashMap<(i32, i32), Vec<PlantSpatialRef>>,
    seed_cells: HashMap<[i32; 3], usize>,
    tree_hosts: HashMap<(i32, i32), Vec<usize>>,
}

impl PlantSpatialIndex {
    fn build(
        plants: &[Plant],
        growth: &[PlantGrowth],
        branches: &[Vec<BranchSegment>],
        age_years: f32,
    ) -> Self {
        let mut index = Self::default();
        for (plant_index, plant) in plants.iter().enumerate() {
            let Some(&plant_growth) = growth.get(plant_index) else {
                continue;
            };
            let plant_branches = branches.get(plant_index).map(Vec::as_slice).unwrap_or(&[]);
            let seed_cell = if age_years < plant.germination_age_years {
                plant.seed_cell()
            } else {
                plant.apical_seed_cell(plant_growth)
            };
            index.seed_cells.entry(seed_cell).or_insert(plant_index);

            if plant.genome.mycorrhizal_affinity >= 0.35 {
                insert_spatial_bounds(
                    &mut index.tree_hosts,
                    plant.x - 2.5,
                    plant.y - 2.5,
                    plant.x + 2.5,
                    plant.y + 2.5,
                    plant_index,
                );
            }

            if plant_growth.trunk_height_m > 0.0 {
                let radius = plant_growth.trunk_radius_m;
                insert_spatial_bounds(
                    &mut index.canopy,
                    plant.x - radius,
                    plant.y - radius,
                    plant.x + radius,
                    plant.y + radius,
                    PlantSpatialRef::Trunk(plant_index),
                );
            }

            for (branch_index, branch) in plant_branches.iter().enumerate() {
                let query_radius = if branch.is_root {
                    branch.radius_m
                } else {
                    branch.radius_m.max(LEAF_CLUSTER_RADIUS_M)
                };
                let minimum = branch.start.min(branch.end) - Vec3::splat(query_radius);
                let maximum = branch.start.max(branch.end) + Vec3::splat(query_radius);
                let spatial_ref = PlantSpatialRef::Branch {
                    plant_index,
                    branch_index,
                };
                if branch.is_root {
                    insert_spatial_bounds(
                        &mut index.roots,
                        minimum.x,
                        minimum.y,
                        maximum.x,
                        maximum.y,
                        spatial_ref,
                    );
                } else {
                    insert_spatial_bounds(
                        &mut index.canopy,
                        minimum.x,
                        minimum.y,
                        maximum.x,
                        maximum.y,
                        spatial_ref,
                    );
                }
            }
        }
        index
    }
}

fn plant_spatial_bucket(x: f32, y: f32) -> (i32, i32) {
    (
        (x / PLANT_QUERY_BUCKET_M).floor() as i32,
        (y / PLANT_QUERY_BUCKET_M).floor() as i32,
    )
}

fn insert_spatial_bounds<T: Copy>(
    buckets: &mut HashMap<(i32, i32), Vec<T>>,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    value: T,
) {
    let (min_x, min_y) = plant_spatial_bucket(min_x, min_y);
    let (max_x, max_y) = plant_spatial_bucket(max_x, max_y);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            buckets.entry((x, y)).or_default().push(value);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MyceliumSegment {
    start: Vec3,
    end: Vec3,
    radius_m: f32,
}

impl GeneratedWorld {
    /// Build a deterministic 10 km² seeded map. Centimeter cells are generated
    /// implicitly; these 100×100 tiles are overview LOD only and have no physics.
    pub fn from_seed(seed: &str) -> Self {
        let seed_hash = hash_seed(seed);
        // A seed-derived calendar epoch keeps the initial season reproducible.
        let season_phase_offset = (unit_noise(seed_hash ^ 0x6A09_E667_F3BC_C909) + 1.0) * 0.5;
        let mut terrain_voxels =
            Vec::with_capacity(WORLD_OVERVIEW_SIDE_CELLS * WORLD_OVERVIEW_SIDE_CELLS);
        for y in 0..WORLD_OVERVIEW_SIDE_CELLS {
            for x in 0..WORLD_OVERVIEW_SIDE_CELLS {
                let field_x = x as f32 + 0.5;
                let field_y = y as f32 + 0.5;
                let world_x = field_x * WORLD_OVERVIEW_CELL_M - WORLD_SIDE_M * 0.5;
                let world_y = field_y * WORLD_OVERVIEW_CELL_M - WORLD_SIDE_M * 0.5;
                let large = value_noise(seed_hash, world_x / 2_200.0, world_y / 2_200.0);
                let medium = value_noise(
                    seed_hash ^ 0xA076_1D64_78BD_642F,
                    world_x / 700.0,
                    world_y / 700.0,
                );
                let fine = value_noise(
                    seed_hash ^ 0xE703_7ED1_A0B4_28DB,
                    world_x / 220.0,
                    world_y / 220.0,
                );
                let height = terrain_height(seed_hash, world_x, world_y);
                let variation = (large * 0.5 + medium * 0.35 + fine * 0.15).clamp(-1.0, 1.0);
                // Visual soil/grass tint follows the deterministic terrain
                // field. This is display color only; it does not create or
                // alter physical material state.
                let vegetation_tint = (0.52 + variation * 0.42).clamp(0.08, 0.94);
                let color = [
                    0.40 - vegetation_tint * 0.19,
                    0.29 + vegetation_tint * 0.23,
                    0.16 + vegetation_tint * 0.05,
                    1.0,
                ];
                terrain_voxels.push(TerrainVoxel {
                    center: Vec3::new(
                        (x as f32 + 0.5) * WORLD_OVERVIEW_CELL_M - WORLD_SIDE_M * 0.5,
                        (y as f32 + 0.5) * WORLD_OVERVIEW_CELL_M - WORLD_SIDE_M * 0.5,
                        height * 0.5,
                    ),
                    size: Vec3::new(WORLD_OVERVIEW_CELL_M, WORLD_OVERVIEW_CELL_M, height),
                    color,
                    content: GeneratedCellContent::Terrain,
                    fiber_direction: None,
                });
            }
        }
        let (plants, morphotypes) = make_plants(seed_hash);
        let mut world = Self {
            view_bounds: None,
            voxels: Vec::new(),
            terrain_voxels,
            plants,
            morphotypes,
            plant_growth: Vec::new(),
            plant_branches: Vec::new(),
            plant_spatial_index: PlantSpatialIndex::default(),
            mycelium_segments: Vec::new(),
            mycelium_buckets: vec![Vec::new(); 25 * 25],
            seed_hash,
            age_years: 0.0,
            calendar_years: 0.0,
            geometry_age_years: 0.0,
            season_phase_offset,
            generated_growth_quarter: u32::MAX,
            generated_calendar_quarter: u64::MAX,
            geometry_revision: 0,
        };
        world.rebuild_vegetation();
        world
    }

    /// Advance the world's simulation clock by simulation seconds. At a 1x
    /// multiplier, one real second advances one world second; plant age and
    /// seasonal phase are derived from the elapsed calendar years.
    pub fn advance_growth(&mut self, elapsed_simulation_seconds: f32) {
        let elapsed_simulation_seconds = if elapsed_simulation_seconds.is_finite() {
            elapsed_simulation_seconds.max(0.0)
        } else {
            0.0
        };
        let elapsed_years = elapsed_simulation_seconds / SIMULATION_YEAR_SECONDS;
        self.calendar_years += f64::from(elapsed_years);
        self.age_years = (self.age_years + elapsed_years).min(100.0);
        let growth_quarter = (self.age_years * 4.0).floor() as u32;
        let calendar_quarter = self.calendar_quarter();
        if growth_quarter != self.generated_growth_quarter
            || calendar_quarter != self.generated_calendar_quarter
        {
            self.rebuild_vegetation();
        }
    }

    pub fn vegetation_summary(&self) -> String {
        let mut stages = [0; 5];
        let mut mean_light = 0.0;
        let mut mean_water = 0.0;
        let mut mean_nutrients = 0.0;
        let dormant_seeds = self
            .plants
            .iter()
            .filter(|plant| self.age_years < plant.germination_age_years)
            .count();
        for growth in &self.plant_growth {
            stages[growth.stage as usize] += 1;
            mean_light += growth.light;
            mean_water += growth.water;
            mean_nutrients += growth.nutrients;
        }
        let plant_count = self.plant_growth.len().max(1) as f32;
        let names = self
            .morphotypes
            .iter()
            .take(3)
            .map(|morphotype| format!("{}({})", morphotype.latin_name, morphotype.population))
            .collect::<Vec<_>>()
            .join(", ");
        let remaining = self.morphotypes.len().saturating_sub(3);
        format!(
            "TREES {} SEEDS {dormant_seeds} | MORPHOTYPES {} [{names}{}] | SEEDLING {} SAPLING {} POLE {} MATURE {} OLD {} | L {:.2} W {:.2} N {:.2}",
            self.plants.len(),
            self.morphotypes.len(),
            if remaining > 0 { format!(", +{remaining}") } else { String::new() },
            stages[GrowthStage::Seedling as usize],
            stages[GrowthStage::Sapling as usize],
            stages[GrowthStage::Pole as usize],
            stages[GrowthStage::Mature as usize],
            stages[GrowthStage::Senescent as usize],
            mean_light / plant_count,
            mean_water / plant_count,
            mean_nutrients / plant_count,
        )
    }

    /// Build centimeter wood cells only for the camera's local detail region.
    /// The full world keeps its compact overview geometry; this temporary list
    /// is a render refinement and never enters the simulated material state.
    pub fn plant_wood_voxels_near(&self, center: Vec3, radius_m: f32) -> Vec<TerrainVoxel> {
        if !center.is_finite() || !radius_m.is_finite() || radius_m <= 0.0 {
            return Vec::new();
        }
        let radius_squared = radius_m * radius_m;
        let region_min = center - Vec3::splat(radius_m);
        let region_max = center + Vec3::splat(radius_m);
        let season_phase = self.season_phase();
        let mut result = Vec::new();
        let mut occupied = HashSet::new();

        for ((plant, growth), branches) in self
            .plants
            .iter()
            .zip(&self.plant_growth)
            .zip(&self.plant_branches)
        {
            if self.age_years < plant.germination_age_years
                || !plant.wood_intersects_region(*growth, branches, region_min, region_max, false)
            {
                continue;
            }

            for voxel in plant
                .voxels_at_age(
                    *growth,
                    self.seed_hash,
                    branches,
                    season_phase,
                    self.geometry_age_years,
                    WORLD_REV_M,
                    Some((center, radius_m)),
                    true,
                    false,
                    None,
                )
                .expect("unbounded render-shell generation")
            {
                if voxel.content != GeneratedCellContent::PlantWood
                    || voxel.center.distance_squared(center) > radius_squared
                {
                    continue;
                }
                let cell = (
                    ((voxel.center.x + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
                    ((voxel.center.y + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
                    (voxel.center.z / WORLD_REV_M).floor() as i32,
                );
                if occupied.insert(cell) {
                    result.push(voxel);
                }
            }
        }
        result
    }

    /// Enumerate complete 1 cm plant-wood volume near a point for physical
    /// activation. Unlike `plant_wood_voxels_near`, this includes interior and
    /// root REVs; the render-only shell must never be used as a physical body.
    pub fn plant_wood_rev_cells_near(
        &self,
        center: Vec3,
        radius_m: f32,
        max_cells: usize,
    ) -> Result<Vec<[i32; 3]>, String> {
        let candidates = self.plant_wood_rev_candidates_near(center, radius_m, max_cells)?;
        Ok(candidates
            .into_iter()
            .filter(|&cell| {
                self.sample_cell(cell)
                    .is_some_and(|sample| sample.content == GeneratedCellContent::PlantWood)
            })
            .collect())
    }

    /// Return the same exact physical wood cells with their generated state
    /// already sampled. Pass these directly to
    /// `build_physical_patch_from_generated_cells` to avoid sampling every REV
    /// a second time during activation.
    pub fn plant_wood_generated_cells_near(
        &self,
        center: Vec3,
        radius_m: f32,
        max_cells: usize,
    ) -> Result<Vec<GeneratedCell>, String> {
        let candidates = self.plant_wood_rev_candidates_near(center, radius_m, max_cells)?;
        Ok(candidates
            .into_iter()
            .filter_map(|cell| {
                self.sample_cell(cell)
                    .filter(|sample| sample.content == GeneratedCellContent::PlantWood)
            })
            .collect())
    }

    fn plant_wood_rev_candidates_near(
        &self,
        center: Vec3,
        radius_m: f32,
        max_cells: usize,
    ) -> Result<Vec<[i32; 3]>, String> {
        if !center.is_finite() || !radius_m.is_finite() || radius_m <= 0.0 {
            return Err("physical wood query requires a finite center and positive radius".into());
        }
        if !(radius_m * radius_m).is_finite() {
            return Err("physical wood query radius is too large".into());
        }
        if max_cells == 0 || max_cells > MAX_PHYSICAL_PATCH_REVS {
            return Err(format!(
                "physical wood query limit must be between 1 and {MAX_PHYSICAL_PATCH_REVS} REVs"
            ));
        }

        let radius_squared = radius_m * radius_m;
        let region_min = center - Vec3::splat(radius_m);
        let region_max = center + Vec3::splat(radius_m);
        let season_phase = self.season_phase();
        let mut cells = Vec::new();
        let mut occupied = HashSet::new();
        for ((plant, growth), branches) in self
            .plants
            .iter()
            .zip(&self.plant_growth)
            .zip(&self.plant_branches)
        {
            if self.age_years < plant.germination_age_years
                || !plant.wood_intersects_region(*growth, branches, region_min, region_max, true)
            {
                continue;
            }
            for voxel in plant
                .voxels_at_age(
                    *growth,
                    self.seed_hash,
                    branches,
                    season_phase,
                    self.geometry_age_years,
                    WORLD_REV_M,
                    Some((center, radius_m)),
                    false,
                    true,
                    Some(max_cells),
                )
                .map_err(str::to_owned)?
            {
                if voxel.content != GeneratedCellContent::PlantWood
                    || voxel.center.distance_squared(center) > radius_squared
                {
                    continue;
                }
                let cell = [
                    ((voxel.center.x + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
                    ((voxel.center.y + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
                    (voxel.center.z / WORLD_REV_M).floor() as i32,
                ];
                if occupied.insert(cell) {
                    if cells.len() == max_cells {
                        return Err(format!(
                            "physical wood query exceeds its {max_cells}-REV limit"
                        ));
                    }
                    cells.push(cell);
                }
            }
        }
        cells.sort_unstable();
        Ok(cells)
    }

    /// Stable identity for renderer caches; geometry changes only when a
    /// growth-stage or seasonal rebuild completes.
    pub fn render_cache_key(&self) -> (u64, u64) {
        (self.seed_hash, self.geometry_revision)
    }

    /// Sample the deterministic terrain elevation in world-centered metres.
    pub fn terrain_height_at(&self, x_m: f32, y_m: f32) -> f32 {
        terrain_height(self.seed_hash, x_m, y_m)
    }

    pub fn set_view_bounds(&mut self, bounds: Option<(Vec3, Vec3)>) {
        self.view_bounds = bounds;
        self.geometry_revision += 1;
    }

    /// Conservative capsule-vs-wood check against the current generated plant
    /// geometry. Candidate trees are culled by their seeded crown radius first.
    fn character_overlaps_wood(&self, foot: Vec3, radius: f32, height: f32) -> bool {
        for (plant, segments) in self.plants.iter().zip(&self.plant_branches) {
            let reach = plant
                .genome
                .crown_spread_m
                .max(plant.genome.maximum_height_m * 0.35)
                + radius;
            if (plant.x - foot.x).abs() > reach || (plant.y - foot.y).abs() > reach {
                continue;
            }
            for segment in segments {
                let low_z = segment.start.z.min(segment.end.z) - segment.radius_m;
                let high_z = segment.start.z.max(segment.end.z) + segment.radius_m;
                if high_z < foot.z || low_z > foot.z + height {
                    continue;
                }
                let dx = segment.end.x - segment.start.x;
                let dy = segment.end.y - segment.start.y;
                let length_sq = dx * dx + dy * dy;
                let t = if length_sq > 1e-10 {
                    (((foot.x - segment.start.x) * dx + (foot.y - segment.start.y) * dy)
                        / length_sq)
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let nearest_x = segment.start.x + dx * t;
                let nearest_y = segment.start.y + dy * t;
                let gap_x = foot.x - nearest_x;
                let gap_y = foot.y - nearest_y;
                let clearance = radius + segment.radius_m;
                if gap_x * gap_x + gap_y * gap_y < clearance * clearance {
                    return true;
                }
            }
        }
        false
    }

    /// Build the camera-local terrain surface on the 10 cm render lattice.
    /// The centimeter REV world remains implicit; this is only its visible surface.
    pub fn terrain_patch(&self, center: Vec3, radius_m: f32) -> Vec<TerrainVoxel> {
        let radius = radius_m.max(WORLD_CELL_M);
        // The REV grid stays at 1 cm and terrain height at 10 cm increments.
        // Only render-cell footprints grow with distance from the camera.
        let rings = [
            (0.0, 8.0, 0.1),
            (8.0, 32.0, 0.4),
            (32.0, 128.0, 1.6),
            (128.0, 512.0, 6.4),
            (512.0, radius, 25.6),
        ];
        let mut patch = Vec::with_capacity(82_000);
        let fade = |start: f32, end: f32, value: f32| {
            let t = ((value - start) / (end - start)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        for (inner_radius, outer_radius, cell) in rings {
            if inner_radius >= radius {
                continue;
            }
            let outer = outer_radius.min(radius);
            let inner_fade_width = inner_radius * 0.2;
            let outer_fade_width = outer * 0.2;
            let render_inner = if inner_radius > 0.0 {
                inner_radius - inner_fade_width
            } else {
                0.0
            };
            let render_outer = if outer_radius < radius {
                (outer + outer_fade_width).min(radius)
            } else {
                outer
            };
            let extent = render_outer + cell * 0.5;
            let min_x = ((center.x - extent + WORLD_SIDE_M * 0.5) / cell).floor() as i32;
            let max_x = ((center.x + extent + WORLD_SIDE_M * 0.5) / cell).ceil() as i32;
            let min_y = ((center.y - extent + WORLD_SIDE_M * 0.5) / cell).floor() as i32;
            let max_y = ((center.y + extent + WORLD_SIDE_M * 0.5) / cell).ceil() as i32;
            let inner_squared = render_inner * render_inner;
            let outer_squared = render_outer * render_outer;
            let half_cell = cell * 0.5;
            for iy in min_y..max_y {
                let y = (iy as f32 + 0.5) * cell - WORLD_SIDE_M * 0.5;
                if !(0.0..WORLD_SIDE_M).contains(&(y + WORLD_SIDE_M * 0.5)) {
                    continue;
                }
                for ix in min_x..max_x {
                    let x = (ix as f32 + 0.5) * cell - WORLD_SIDE_M * 0.5;
                    if let Some((min, max)) = self.view_bounds {
                        if x + half_cell <= min.x
                            || x - half_cell >= max.x
                            || y + half_cell <= min.y
                            || y - half_cell >= max.y
                        {
                            continue;
                        }
                    }
                    let dx = x - center.x;
                    let dy = y - center.y;
                    let distance = (dx * dx + dy * dy).sqrt();
                    // Select a tile when its footprint intersects this ring.
                    // Center-only clipping left thin uncovered arcs wherever
                    // neighboring LOD grids had different cell widths.
                    let nearest_x = (dx.abs() - half_cell).max(0.0);
                    let nearest_y = (dy.abs() - half_cell).max(0.0);
                    let nearest_distance_squared = nearest_x * nearest_x + nearest_y * nearest_y;
                    let farthest_x = dx.abs() + half_cell;
                    let farthest_y = dy.abs() + half_cell;
                    let farthest_distance_squared =
                        farthest_x * farthest_x + farthest_y * farthest_y;
                    if nearest_distance_squared > outer_squared
                        || farthest_distance_squared < inner_squared
                        || !(0.0..WORLD_SIDE_M).contains(&(x + WORLD_SIDE_M * 0.5))
                    {
                        continue;
                    }
                    let height = terrain_height(self.seed_hash, x, y);
                    let variation =
                        value_noise(self.seed_hash ^ 0x4D59_5DF4_D0F3_3173, x / 35.0, y / 35.0);
                    let vegetation_tint = (0.52 + variation * 0.40).clamp(0.08, 0.94);
                    let mut opacity = 1.0;
                    if inner_radius > 0.0 {
                        opacity *= fade(
                            inner_radius - inner_fade_width,
                            inner_radius + inner_fade_width,
                            distance,
                        );
                    }
                    let color = [
                        (0.40 - vegetation_tint * 0.19).clamp(0.0, 1.0),
                        (0.29 + vegetation_tint * 0.23).clamp(0.0, 1.0),
                        (0.16 + vegetation_tint * 0.05).clamp(0.0, 1.0),
                        opacity,
                    ];
                    patch.push(TerrainVoxel {
                        center: Vec3::new(x, y, height * 0.5),
                        size: Vec3::new(cell, cell, height),
                        color,
                        content: GeneratedCellContent::Terrain,
                        fiber_direction: None,
                    });
                }
            }
        }
        patch
    }

    /// Return the plant's inherited genome at its seed while dormant and at
    /// its apical growth cell after germination.
    pub fn seed_genome_at_cell(&self, cell: [i32; 3]) -> Option<SeedGenome> {
        self.plant_spatial_index
            .seed_cells
            .get(&cell)
            .and_then(|&index| self.plants.get(index))
            .map(|plant| plant.genome)
    }

    /// Sample one cell on the deterministic 1 cm lattice. Coordinates start
    /// at the south-west world corner; valid horizontal indices are
    /// `0..WORLD_REV_SIDE`, and the vertical domain is
    /// `0..WORLD_SAMPLE_HEIGHT_CELLS`. Cells represent generated geometry, not
    /// dynamic solver state.
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
        let (terrain_x, terrain_y) =
            terrain_field_coordinates(x_m - WORLD_SIDE_M * 0.5, y_m - WORLD_SIDE_M * 0.5);
        let terrain_height_m = terrain_height(self.seed_hash, terrain_x, terrain_y);
        let (soil_moisture, fertility) = ecology_at(self.seed_hash, field_x, field_y);
        let temperature_kelvin = climate_temperature(self.seed_hash, field_x, field_y, z_m);

        let point = Vec3::new(x_m - WORLD_SIDE_M * 0.5, y_m - WORLD_SIDE_M * 0.5, z_m);
        let seed_genome = self.seed_genome_at_cell(cell);
        let (vegetation, fiber_direction) = if seed_genome.is_some() {
            (Some(GeneratedCellContent::Seed), None)
        } else if z_m <= terrain_height_m {
            if let Some((content, direction)) = self.root_content_at_point(point) {
                (Some(content), Some(direction))
            } else {
                (self.mycelium_content_at_point(point), None)
            }
        } else {
            let sampled = self.vegetation_at_point(cell, point);
            let direction = sampled.and_then(|(_, direction)| direction);
            (sampled.map(|(content, _)| content), direction)
        };
        let generated_soil_content = matches!(
            vegetation,
            Some(GeneratedCellContent::PlantWood | GeneratedCellContent::Mycelium)
        );
        let content = if z_m <= terrain_height_m && !generated_soil_content {
            GeneratedCellContent::Terrain
        } else {
            vegetation.unwrap_or(GeneratedCellContent::Air)
        };

        Some(GeneratedCell {
            cell,
            content,
            seed_genome,
            fiber_direction,
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

    /// Legacy absolute-coordinate scene builder. The generated map origin is
    /// not aligned to the engine's global REV lattice; use
    /// `build_physical_patch` for solver-compatible local coordinates.
    #[deprecated(
        note = "the generated map is not aligned to the global REV lattice; use build_physical_patch"
    )]
    pub fn build_physical_region<F>(
        &self,
        config: SceneConfig,
        min_cell: [i32; 3],
        max_cell_exclusive: [i32; 3],
        physics_for_cell: F,
    ) -> Result<Scene, String>
    where
        F: FnMut(GeneratedCell) -> Result<Option<GeneratedCellPhysics>, String>,
    {
        let cells = self
            .sample_region(min_cell, max_cell_exclusive)
            .ok_or_else(|| {
                format!(
                    "invalid physical world region {min_cell:?}..{max_cell_exclusive:?} or sample limit exceeded"
                )
            })?;
        let (spawns, _) = build_physical_spawns(cells, None, physics_for_cell)?;
        Scene::from_voxels(config, spawns)
    }

    /// Build a bounded physical patch with local solver coordinates and a
    /// retained world origin. Unlike a render LOD, every included 1 cm sample
    /// becomes an independent REV with its own state in the returned Scene.
    pub fn build_physical_patch<F>(
        &self,
        config: SceneConfig,
        min_cell: [i32; 3],
        max_cell_exclusive: [i32; 3],
        physics_for_cell: F,
    ) -> Result<PhysicalWorldPatch, String>
    where
        F: FnMut(GeneratedCell) -> Result<Option<GeneratedCellPhysics>, String>,
    {
        let cells = self
            .sample_region(min_cell, max_cell_exclusive)
            .ok_or_else(|| {
                format!(
                    "invalid physical world region {min_cell:?}..{max_cell_exclusive:?} or sample limit exceeded"
                )
            })?;
        let (spawns, source_cells) =
            build_physical_spawns(cells, Some(min_cell), physics_for_cell)?;
        let scene = Scene::from_voxels(config, spawns)?;
        Ok(PhysicalWorldPatch {
            origin_cell: min_cell,
            origin_world_m: world_cell_origin_m(min_cell),
            source_cells,
            scene,
        })
    }

    /// Build a physical patch from an explicit sparse set of generated 1 cm
    /// cells. Duplicate identities are rejected to prevent double-counting mass.
    pub fn build_physical_patch_from_cells<F, I>(
        &self,
        config: SceneConfig,
        origin_cell: [i32; 3],
        source_cells: I,
        physics_for_cell: F,
    ) -> Result<PhysicalWorldPatch, String>
    where
        F: FnMut(GeneratedCell) -> Result<Option<GeneratedCellPhysics>, String>,
        I: IntoIterator<Item = [i32; 3]>,
    {
        let mut bounded_source_cells = Vec::new();
        for cell in source_cells {
            if bounded_source_cells.len() == MAX_PHYSICAL_PATCH_REVS {
                return Err(format!(
                    "physical world patch exceeds its {MAX_PHYSICAL_PATCH_REVS}-REV limit"
                ));
            }
            bounded_source_cells.push(cell);
        }
        let mut source_cells = bounded_source_cells;
        if source_cells.is_empty() {
            return Err("physical world patch requires at least one source cell".into());
        }
        source_cells.sort_unstable();
        if source_cells.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("physical world patch contains duplicate source cells".into());
        }

        let cells = source_cells
            .iter()
            .map(|&cell| {
                self.sample_cell(cell)
                    .ok_or_else(|| format!("physical patch cell {cell:?} is outside the world"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        assemble_physical_patch(config, origin_cell, cells, physics_for_cell)
    }

    /// Build a physical patch from cells already sampled from this world.
    /// This keeps exact generated state and avoids a second world lookup for
    /// every REV when activating geometry returned by a generated-cell query.
    pub fn build_physical_patch_from_generated_cells<F, I>(
        &self,
        config: SceneConfig,
        origin_cell: [i32; 3],
        generated_cells: I,
        physics_for_cell: F,
    ) -> Result<PhysicalWorldPatch, String>
    where
        F: FnMut(GeneratedCell) -> Result<Option<GeneratedCellPhysics>, String>,
        I: IntoIterator<Item = GeneratedCell>,
    {
        let mut cells = Vec::new();
        for cell in generated_cells {
            if cells.len() == MAX_PHYSICAL_PATCH_REVS {
                return Err(format!(
                    "physical world patch exceeds its {MAX_PHYSICAL_PATCH_REVS}-REV limit"
                ));
            }
            if !(0..WORLD_REV_SIDE).contains(&cell.cell[0])
                || !(0..WORLD_REV_SIDE).contains(&cell.cell[1])
                || !(0..WORLD_SAMPLE_HEIGHT_CELLS).contains(&cell.cell[2])
            {
                return Err(format!(
                    "physical patch cell {:?} is outside the world",
                    cell.cell
                ));
            }
            cells.push(cell);
        }
        if cells.is_empty() {
            return Err("physical world patch requires at least one source cell".into());
        }
        cells.sort_unstable_by_key(|cell| cell.cell);
        if cells.windows(2).any(|pair| pair[0].cell == pair[1].cell) {
            return Err("physical world patch contains duplicate source cells".into());
        }

        assemble_physical_patch(config, origin_cell, cells, physics_for_cell)
    }

    fn vegetation_at_point(
        &self,
        cell: [i32; 3],
        point: Vec3,
    ) -> Option<(GeneratedCellContent, Option<Vec3>)> {
        let season_phase = self.season_phase();
        let mut best: Option<(GeneratedCellContent, Option<Vec3>)> = None;
        if let Some(candidates) = self
            .plant_spatial_index
            .canopy
            .get(&plant_spatial_bucket(point.x, point.y))
        {
            let mut start = 0;
            while start < candidates.len() {
                let plant_index = candidates[start].plant_index();
                let mut end = start + 1;
                while end < candidates.len() && candidates[end].plant_index() == plant_index {
                    end += 1;
                }
                if let (Some(plant), Some(growth), Some(branches)) = (
                    self.plants.get(plant_index),
                    self.plant_growth.get(plant_index),
                    self.plant_branches.get(plant_index),
                ) {
                    let sampled = plant.content_at_indexed_point(
                        *growth,
                        point,
                        self.seed_hash,
                        cell,
                        branches,
                        season_phase,
                        &candidates[start..end],
                    );
                    if let Some((content, _)) = sampled {
                        if best.is_none_or(|(current, _)| content.priority() < current.priority()) {
                            best = sampled;
                        }
                    }
                }
                start = end;
            }
        }
        if let Some(content) = ground_flora_at_point(
            self.seed_hash,
            self.geometry_age_years,
            season_phase,
            point,
            &self.plants,
            &self.plant_spatial_index,
        ) {
            if best.is_none_or(|(current, _)| content.priority() < current.priority()) {
                best = Some((content, None));
            }
        }
        best
    }

    fn root_content_at_point(&self, point: Vec3) -> Option<(GeneratedCellContent, Vec3)> {
        let candidates = self
            .plant_spatial_index
            .roots
            .get(&plant_spatial_bucket(point.x, point.y))?;
        let mut best: Option<(GeneratedCellContent, Vec3)> = None;
        let mut start = 0;
        while start < candidates.len() {
            let plant_index = candidates[start].plant_index();
            let mut end = start + 1;
            while end < candidates.len() && candidates[end].plant_index() == plant_index {
                end += 1;
            }
            if let (Some(plant), Some(branches)) = (
                self.plants.get(plant_index),
                self.plant_branches.get(plant_index),
            ) {
                let sampled = plant.root_content_at_indexed_point(
                    self.geometry_age_years,
                    point,
                    self.seed_hash,
                    branches,
                    &candidates[start..end],
                );
                if let Some((content, _)) = sampled {
                    if best.is_none_or(|(current, _)| content.priority() < current.priority()) {
                        best = sampled;
                    }
                }
            }
            start = end;
        }
        best
    }

    fn mycelium_content_at_point(&self, point: Vec3) -> Option<GeneratedCellContent> {
        let gx = ((point.x + 4.8) / 0.4).floor() as i32;
        let gy = ((point.y + 4.8) / 0.4).floor() as i32;
        if !(0..25).contains(&gx) || !(0..25).contains(&gy) {
            return None;
        }
        let bucket = (gy as usize) * 25 + gx as usize;
        for &index in &self.mycelium_buckets[bucket] {
            let segment = self.mycelium_segments[index];
            if point.z < segment.start.z.min(segment.end.z) - segment.radius_m
                || point.z > segment.start.z.max(segment.end.z) + segment.radius_m
            {
                continue;
            }
            let delta = segment.end - segment.start;
            let length_squared = delta.length_squared();
            if length_squared <= f32::EPSILON {
                continue;
            }
            let along = ((point - segment.start).dot(delta) / length_squared).clamp(0.0, 1.0);
            let closest = segment.start + delta * along;
            if point.distance_squared(closest) <= segment.radius_m.powi(2) {
                return Some(GeneratedCellContent::Mycelium);
            }
        }
        None
    }

    fn season_phase(&self) -> f32 {
        (self.calendar_years + f64::from(self.season_phase_offset)).fract() as f32
    }

    fn calendar_quarter(&self) -> u64 {
        ((self.calendar_years + f64::from(self.season_phase_offset)) * 4.0).floor() as u64
    }

    /// Build the displayed 10 cm LOD directly from candidate geometry. The
    /// implicit 1 cm lattice remains authoritative for physical cell queries.
    fn rebuild_vegetation(&mut self) {
        self.geometry_age_years = (self.age_years * 4.0).floor() * 0.25;
        let season_phase = self.season_phase();
        let growth_quarter = (self.age_years * 4.0).floor() as u32;
        if growth_quarter != self.generated_growth_quarter {
            let next_growth = compute_plant_growth(
                &self.plants,
                self.geometry_age_years,
                self.seed_hash,
                self.calendar_quarter(),
                &self.plant_growth,
            );
            self.plant_growth = next_growth;
            self.plant_branches = self
                .plants
                .par_iter()
                .zip(self.plant_growth.par_iter())
                .map(|(plant, growth)| plant.branch_segments(*growth, self.seed_hash))
                .collect();
            self.plant_spatial_index = PlantSpatialIndex::build(
                &self.plants,
                &self.plant_growth,
                &self.plant_branches,
                self.age_years,
            );
            (self.mycelium_segments, self.mycelium_buckets) =
                make_mycelium(self.seed_hash, self.geometry_age_years, &self.plants);
        }
        self.voxels.clone_from(&self.terrain_voxels);
        let mut candidates = make_ground_flora(
            self.seed_hash,
            self.geometry_age_years,
            season_phase,
            &self.plants,
        );
        let plant_candidates: Vec<Vec<TerrainVoxel>> = self
            .plants
            .par_iter()
            .zip(self.plant_growth.par_iter())
            .zip(self.plant_branches.par_iter())
            .map(|((plant, growth), branches)| {
                plant
                    .voxels_at_age(
                        *growth,
                        self.seed_hash,
                        branches,
                        season_phase,
                        self.geometry_age_years,
                        WORLD_CELL_M,
                        None,
                        false,
                        false,
                        None,
                    )
                    .expect("unbounded overview geometry generation")
            })
            .collect();
        for mut plant_voxels in plant_candidates {
            candidates.append(&mut plant_voxels);
        }
        self.voxels.reserve(candidates.len());
        let mut preview_cells: HashMap<(i32, i32, i32), usize> =
            HashMap::with_capacity(candidates.len().min(262_144));
        let mut terrain_height_by_column = HashMap::with_capacity(candidates.len().min(65_536));
        for mut voxel in candidates {
            // All displayed 10 cm cubes use one world-aligned lattice. Plant
            // candidate centers can be offset (for example trunk-local cells);
            // snapping ensures that candidates occupy the same render lattice.
            let ix = ((voxel.center.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).floor() as i32;
            let iy = ((voxel.center.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).floor() as i32;
            let iz = (voxel.center.z / WORLD_CELL_M).floor() as i32;
            let preview_cell = (ix, iy, iz);
            let aligned_center = Vec3::new(
                (ix as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                (iy as f32 + 0.5) * WORLD_CELL_M - WORLD_SIDE_M * 0.5,
                (iz as f32 + 0.5) * WORLD_CELL_M,
            );
            let terrain_z = *terrain_height_by_column.entry([ix, iy]).or_insert_with(|| {
                terrain_height(
                    self.seed_hash,
                    terrain_field_coordinates(aligned_center.x, aligned_center.y).0,
                    terrain_field_coordinates(aligned_center.x, aligned_center.y).1,
                )
            });
            let seed_preview_matches = voxel.content == GeneratedCellContent::Seed
                && self.plants.iter().any(|plant| {
                    if self.age_years >= plant.germination_age_years {
                        return false;
                    }
                    let cell = plant.seed_cell();
                    cell[0] / WORLD_REVS_PER_RENDER_CELL == ix
                        && cell[1] / WORLD_REVS_PER_RENDER_CELL == iy
                        && cell[2] / WORLD_REVS_PER_RENDER_CELL == iz
                });
            if voxel.content == GeneratedCellContent::Seed && !seed_preview_matches {
                continue;
            }
            if voxel.content == GeneratedCellContent::Seed {
                // A seed is one real 1 cm state cell, not a 10 cm preview cube.
                // Keep its REV-centered position and let the nearby renderer
                // display it at native scale; coarse LOD intentionally omits it.
                self.voxels.push(voxel);
                continue;
            }
            if aligned_center.z - voxel.size.z * 0.5 < terrain_z - 1.0e-4 {
                continue;
            }
            voxel.center = aligned_center;
            if let Some(&existing_index) = preview_cells.get(&preview_cell) {
                if voxel.content.priority() < self.voxels[existing_index].content.priority() {
                    self.voxels[existing_index] = voxel;
                }
            } else {
                preview_cells.insert(preview_cell, self.voxels.len());
                self.voxels.push(voxel);
            }
        }
        self.generated_growth_quarter = growth_quarter;
        self.generated_calendar_quarter = self.calendar_quarter();
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
    }
}

fn make_ground_flora(
    seed: u64,
    age_years: f32,
    season_phase: f32,
    plants: &[Plant],
) -> Vec<TerrainVoxel> {
    let growth = (age_years / 4.0).clamp(0.0, 1.0);
    let mut result = Vec::new();
    // Fixed 0.5 m sampling keeps density seed-locked and cheap to rebuild.
    for gy in 0..20 {
        for gx in 0..20 {
            let cell_hash = lattice_hash(seed ^ 0xA24B_AED4_963E_E407, gx, gy);
            let x = -4.75 + gx as f32 * 0.5;
            let y = -4.75 + gy as f32 * 0.5;
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let moisture = (0.5
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx / 27.0, ny / 27.0) * 0.38)
                .clamp(0.0, 1.0);
            let fertility = (0.55
                + value_noise(seed ^ 0x5899_65CC_7537_4CC3, nx / 22.0, ny / 22.0) * 0.35)
                .clamp(0.1, 1.0);
            let (terrain_x, terrain_y) = terrain_field_coordinates(x, y);
            let ground_z = terrain_height(seed, terrain_x, terrain_y);

            if cell_hash % 5 != 0 {
                continue;
            }
            let plant_count = 2 + ((cell_hash >> 12) % 4) as usize;
            let is_wildflower = wildflower_site(cell_hash, moisture, fertility);
            let is_fern = !is_wildflower && moisture > 0.58;
            let content = if is_wildflower {
                GeneratedCellContent::Wildflower
            } else if is_fern {
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
                let height = (0.12 + fertility * 0.24) * (0.12 + growth * 0.88);
                let center = Vec3::new(center_x, center_y, ground_z + height * 0.5);
                let color = if is_fern {
                    [0.11, 0.33 + fertility * 0.10, 0.13, 1.0]
                } else {
                    [0.26, 0.36 + fertility * 0.10, 0.12, 1.0]
                };
                result.push(TerrainVoxel {
                    center,
                    size: Vec3::new(0.035, 0.035, height),
                    color,
                    content,
                    fiber_direction: None,
                });
                if is_wildflower && wildflowers_blooming(season_phase) {
                    let bloom_color = match blade_hash % 3 {
                        0 => [0.95, 0.72, 0.16, 1.0],
                        1 => [0.92, 0.38, 0.56, 1.0],
                        _ => [0.62, 0.48, 0.88, 1.0],
                    };
                    result.push(TerrainVoxel {
                        center: Vec3::new(center_x, center_y, ground_z + height * 0.94),
                        size: Vec3::splat(0.05),
                        color: bloom_color,
                        content,
                        fiber_direction: None,
                    });
                }
            }
        }
    }

    // Seeded fruiting sites select a soil, tree-associated, or open-ground form.
    for gy in 0..25 {
        for gx in 0..25 {
            let x = -4.8 + gx as f32 * 0.4;
            let y = -4.8 + gy as f32 * 0.4;
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let moisture = (0.5
                + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, nx / 27.0, ny / 27.0) * 0.38)
                .clamp(0.0, 1.0);
            let fertility = (0.55
                + value_noise(seed ^ 0x5899_65CC_7537_4CC3, nx / 22.0, ny / 22.0) * 0.35)
                .clamp(0.1, 1.0);
            let spot = lattice_hash(seed ^ 0xDB4F_0B91_75AE_2165, gx, gy);
            let has_tree_host = plants.iter().any(|plant| {
                plant.genome.mycorrhizal_affinity >= 0.35
                    && (plant.x - x).powi(2) + (plant.y - y).powi(2) <= 2.5_f32.powi(2)
            });
            let Some(form) = fungus_form(
                spot,
                moisture,
                fertility,
                has_tree_host,
                age_years,
                season_phase,
            ) else {
                continue;
            };
            let height = (0.06 + 0.08 * growth * (0.5 + moisture * 0.5)) * form.height_scale;
            let (terrain_x, terrain_y) = terrain_field_coordinates(x, y);
            let ground_z = terrain_height(seed, terrain_x, terrain_y);
            let stem_height = height * form.stem_fraction;
            if stem_height > 0.0 {
                result.push(TerrainVoxel {
                    center: Vec3::new(x, y, ground_z + stem_height * 0.5),
                    size: Vec3::new(
                        form.stem_radius_m * 2.0,
                        form.stem_radius_m * 2.0,
                        stem_height,
                    ),
                    color: [0.82, 0.78, 0.62, 1.0],
                    content: form.content,
                    fiber_direction: Some(Vec3::Z),
                });
            }
            result.push(TerrainVoxel {
                center: Vec3::new(x, y, ground_z + stem_height + form.cap_thickness_m * 0.5),
                size: Vec3::new(form.cap_width_m, form.cap_width_m, form.cap_thickness_m),
                color: form.cap_color,
                content: form.content,
                fiber_direction: None,
            });
        }
    }
    result
}

fn wildflower_site(site_hash: u64, moisture: f32, fertility: f32) -> bool {
    site_hash % 3 == 0 && fertility >= 0.68 && (0.32..=0.68).contains(&moisture)
}

fn wildflowers_blooming(season_phase: f32) -> bool {
    (0.18..0.72).contains(&season_phase)
}

/// Evaluate seeded grass, fern, wildflower, and fungal geometry directly at a point on
/// the REV lattice. This does not inspect or depend on the renderer's LOD list.
fn ground_flora_at_point(
    seed: u64,
    age_years: f32,
    season_phase: f32,
    point: Vec3,
    plants: &[Plant],
    plant_spatial_index: &PlantSpatialIndex,
) -> Option<GeneratedCellContent> {
    let mut found = None;
    let mut record = |content: GeneratedCellContent| {
        if found.is_none_or(|current: GeneratedCellContent| content.priority() < current.priority())
        {
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
            let (terrain_x, terrain_y) = terrain_field_coordinates(x, y);
            let ground_z = terrain_height(seed, terrain_x, terrain_y);
            let height =
                (0.12 + fertility * 0.24) * (0.12 + (age_years / 4.0).clamp(0.0, 1.0) * 0.88);
            let plant_count = 2 + ((cell_hash >> 12) % 4) as usize;
            let is_wildflower = wildflower_site(cell_hash, moisture, fertility);
            let content = if is_wildflower {
                GeneratedCellContent::Wildflower
            } else if moisture > 0.58 {
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
                if is_wildflower && wildflowers_blooming(season_phase) {
                    let bloom_z = ground_z + height * 0.94;
                    if (point.x - center_x).abs() <= 0.025
                        && (point.y - center_y).abs() <= 0.025
                        && (point.z - bloom_z).abs() <= 0.025
                    {
                        record(GeneratedCellContent::Wildflower);
                    }
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
            let fertility = (0.55
                + value_noise(seed ^ 0x5899_65CC_7537_4CC3, nx / 22.0, ny / 22.0) * 0.35)
                .clamp(0.1, 1.0);
            let spot = lattice_hash(seed ^ 0xDB4F_0B91_75AE_2165, gx, gy);
            let has_tree_host = plant_spatial_index
                .tree_hosts
                .get(&plant_spatial_bucket(x, y))
                .is_some_and(|hosts| {
                    hosts.iter().any(|&plant_index| {
                        let plant = &plants[plant_index];
                        (plant.x - x).powi(2) + (plant.y - y).powi(2) <= 2.5_f32.powi(2)
                    })
                });
            let Some(form) = fungus_form(
                spot,
                moisture,
                fertility,
                has_tree_host,
                age_years,
                season_phase,
            ) else {
                continue;
            };
            let growth = (age_years / 4.0).clamp(0.0, 1.0);
            let height = (0.06 + 0.08 * growth * (0.5 + moisture * 0.5)) * form.height_scale;
            let (terrain_x, terrain_y) = terrain_field_coordinates(x, y);
            let ground_z = terrain_height(seed, terrain_x, terrain_y);
            let stem_height = height * form.stem_fraction;
            let stem = form.stem_fraction > 0.0
                && (point.x - x).abs() <= form.stem_radius_m
                && (point.y - y).abs() <= form.stem_radius_m
                && point.z >= ground_z
                && point.z <= ground_z + stem_height;
            let cap = (point.x - x).abs() <= form.cap_width_m * 0.5
                && (point.y - y).abs() <= form.cap_width_m * 0.5
                && (point.z - (ground_z + stem_height + form.cap_thickness_m * 0.5)).abs()
                    <= form.cap_thickness_m * 0.5;
            if stem || cap {
                record(form.content);
            }
        }
    }
    found
}

fn fungi_fruiting(season_phase: f32) -> bool {
    (0.55..0.90).contains(&season_phase)
}

fn fungus_form(
    site_hash: u64,
    moisture: f32,
    fertility: f32,
    has_tree_host: bool,
    age_years: f32,
    season_phase: f32,
) -> Option<FungusForm> {
    if moisture < 0.64 || age_years < 1.0 || site_hash % 13 != 0 || !fungi_fruiting(season_phase) {
        return None;
    }

    let form = match site_hash % 5 {
        0 | 1 => FungusForm {
            content: GeneratedCellContent::Mushroom,
            height_scale: 1.0,
            stem_fraction: 0.62,
            stem_radius_m: 0.0175,
            cap_width_m: 0.12,
            cap_thickness_m: 0.04,
            cap_color: if site_hash & 1 == 0 {
                [0.58, 0.20, 0.12, 1.0]
            } else {
                [0.48, 0.32, 0.16, 1.0]
            },
        },
        2 if has_tree_host && moisture >= 0.68 && fertility >= 0.45 => FungusForm {
            content: GeneratedCellContent::Bolete,
            height_scale: 1.15,
            stem_fraction: 0.68,
            stem_radius_m: 0.024,
            cap_width_m: 0.15,
            cap_thickness_m: 0.055,
            cap_color: [0.48, 0.32, 0.16, 1.0],
        },
        3 if !has_tree_host && fertility >= 0.55 => FungusForm {
            content: GeneratedCellContent::Puffball,
            height_scale: 0.75,
            stem_fraction: 0.0,
            stem_radius_m: 0.0,
            cap_width_m: 0.08,
            cap_thickness_m: 0.075,
            cap_color: [0.78, 0.75, 0.63, 1.0],
        },
        _ => return None,
    };
    Some(form)
}

fn make_mycelium(
    seed: u64,
    age_years: f32,
    plants: &[Plant],
) -> (Vec<MyceliumSegment>, Vec<Vec<usize>>) {
    const SITE_SIDE: usize = 25;
    let mut segments = Vec::new();
    let mut buckets = vec![Vec::new(); SITE_SIDE * SITE_SIDE];
    if age_years < 1.0 {
        return (segments, buckets);
    }

    let colony_growth = ((age_years - 1.0) / 40.0).clamp(0.0, 1.0);
    for gy in 0..SITE_SIDE {
        for gx in 0..SITE_SIDE {
            let x = -4.8 + gx as f32 * 0.4;
            let y = -4.8 + gy as f32 * 0.4;
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let (moisture, fertility) = ecology_at(seed, nx, ny);
            let site_hash = lattice_hash(seed ^ 0xDB4F_0B91_75AE_2165, gx as i32, gy as i32);
            if site_hash % 13 != 0 || moisture < 0.58 {
                continue;
            }
            let has_tree_host = plants.iter().any(|plant| {
                plant.genome.mycorrhizal_affinity >= 0.35
                    && (plant.x - x).powi(2) + (plant.y - y).powi(2) <= 2.5_f32.powi(2)
            });
            if !has_tree_host && fertility < 0.48 {
                continue;
            }

            let arm_count = 4 + ((site_hash >> 12) % 4) as usize;
            let spread = (0.12 + colony_growth * 0.72)
                * (0.65 + moisture * 0.5)
                * if has_tree_host { 1.12 } else { 1.0 };
            let (terrain_x, terrain_y) = terrain_field_coordinates(x, y);
            let soil_z = terrain_height(seed, terrain_x, terrain_y) - 0.06;
            let center = Vec3::new(x, y, soil_z);
            let radius = 0.012 + moisture * 0.006;

            for arm in 0..arm_count {
                let arm_hash = lattice_hash(
                    site_hash ^ 0xA24B_AED4_963E_E407,
                    arm as i32,
                    gy as i32 - gx as i32,
                );
                let angle = std::f32::consts::TAU * arm as f32 / arm_count as f32
                    + unit_noise(arm_hash) * 0.30;
                let direction = Vec3::new(angle.cos(), angle.sin(), 0.0);
                let length = spread * (0.45 + (unit_noise(arm_hash.rotate_left(17)) + 1.0) * 0.275);
                let depth = 0.04
                    + colony_growth * 0.12
                    + (unit_noise(arm_hash.rotate_left(31)).abs() * 0.04);
                let shoulder = center + direction * (length * 0.52) - Vec3::Z * (depth * 0.36);
                let tip = center + direction * length - Vec3::Z * depth;
                insert_mycelium_segment(
                    MyceliumSegment {
                        start: center,
                        end: shoulder,
                        radius_m: radius,
                    },
                    &mut segments,
                    &mut buckets,
                );
                insert_mycelium_segment(
                    MyceliumSegment {
                        start: shoulder,
                        end: tip,
                        radius_m: radius * 0.78,
                    },
                    &mut segments,
                    &mut buckets,
                );

                let side_angle = angle + if arm % 2 == 0 { 0.82 } else { -0.82 };
                let side_direction = Vec3::new(side_angle.cos(), side_angle.sin(), 0.0);
                let side_tip =
                    shoulder + side_direction * (length * 0.34) - Vec3::Z * (depth * 0.30);
                insert_mycelium_segment(
                    MyceliumSegment {
                        start: shoulder,
                        end: side_tip,
                        radius_m: radius * 0.62,
                    },
                    &mut segments,
                    &mut buckets,
                );
            }
        }
    }
    (segments, buckets)
}

fn insert_mycelium_segment(
    segment: MyceliumSegment,
    segments: &mut Vec<MyceliumSegment>,
    buckets: &mut [Vec<usize>],
) {
    let segment_index = segments.len();
    segments.push(segment);
    let min_x =
        (((segment.start.x.min(segment.end.x) - segment.radius_m + 4.8) / 0.4).floor()) as i32;
    let max_x =
        (((segment.start.x.max(segment.end.x) + segment.radius_m + 4.8) / 0.4).floor()) as i32;
    let min_y =
        (((segment.start.y.min(segment.end.y) - segment.radius_m + 4.8) / 0.4).floor()) as i32;
    let max_y =
        (((segment.start.y.max(segment.end.y) + segment.radius_m + 4.8) / 0.4).floor()) as i32;
    if max_x < 0 || max_y < 0 || min_x >= 25 || min_y >= 25 {
        return;
    }
    for gy in min_y.max(0)..=max_y.min(24) {
        for gx in min_x.max(0)..=max_x.min(24) {
            buckets[gy as usize * 25 + gx as usize].push(segment_index);
        }
    }
}

impl Plant {
    fn seed_cell(self) -> [i32; 3] {
        [
            ((self.x + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
            ((self.y + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
            (self.soil_z / WORLD_REV_M).ceil() as i32,
        ]
    }

    fn apical_seed_cell(self, growth: PlantGrowth) -> [i32; 3] {
        [
            ((self.x + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
            ((self.y + WORLD_SIDE_M * 0.5) / WORLD_REV_M).floor() as i32,
            ((self.soil_z + growth.height_m) / WORLD_REV_M).floor() as i32,
        ]
    }

    fn root_profile(self) -> (usize, f32, f32) {
        (
            usize::from(self.genome.root_count),
            self.genome.root_extent_m,
            self.genome.root_depth_factor,
        )
    }

    fn root_content_at_indexed_point(
        self,
        age_years: f32,
        point: Vec3,
        seed: u64,
        branches: &[BranchSegment],
        candidates: &[PlantSpatialRef],
    ) -> Option<(GeneratedCellContent, Vec3)> {
        let maturity = 0.08 + 0.92 * (age_years / 100.0).clamp(0.0, 1.0).powf(0.72);
        let vigor = self.growth_vigor(seed);
        let (_, lateral_extent, depth_factor) = self.root_profile();
        let root_extent = lateral_extent * maturity * vigor * (0.85 + self.fertility * 0.25);
        let root_depth = depth_factor * maturity * vigor * (1.10 - self.moisture * 0.50);
        let root_margin = 0.10;
        if (point.x - self.x).abs() > root_extent + root_margin
            || (point.y - self.y).abs() > root_extent + root_margin
            || point.z > self.soil_z + root_margin
            || point.z < self.soil_z - root_depth - root_margin
        {
            return None;
        }

        for candidate in candidates {
            let PlantSpatialRef::Branch { branch_index, .. } = *candidate else {
                continue;
            };
            let Some(branch) = branches.get(branch_index) else {
                continue;
            };
            if !branch.is_root {
                continue;
            }
            let segment = branch.end - branch.start;
            let length_squared = segment.length_squared();
            if length_squared <= f32::EPSILON {
                continue;
            }
            let along = ((point - branch.start).dot(segment) / length_squared).clamp(0.0, 1.0);
            let closest = branch.start + segment * along;
            if point.distance_squared(closest) <= branch.radius_m.powi(2) {
                return Some((GeneratedCellContent::PlantWood, segment.normalize()));
            }
        }
        None
    }

    /// Deterministic individual growth response from the seed and local habitat.
    /// This affects generated shape only, not physical material state.
    fn growth_vigor(self, seed: u64) -> f32 {
        let moisture_fit =
            (1.0 - ((self.moisture - self.genome.moisture_optimum) / 0.42).abs()).clamp(0.25, 1.0);
        let temperature_fit = (1.0
            - ((self.temperature_kelvin - self.genome.temperature_optimum_kelvin) / 10.0).abs())
        .clamp(0.25, 1.0);
        let site_fit = 0.50
            + self.fertility.clamp(0.0, 1.0) * 0.25
            + moisture_fit * 0.15
            + temperature_fit * 0.10;
        let individual_variation =
            0.94 + (unit_noise(self.genome.individual_id ^ seed) + 1.0) * 0.06;
        (0.85 + site_fit * 0.25) * individual_variation
    }

    fn branch_segments(self, growth: PlantGrowth, seed: u64) -> Vec<BranchSegment> {
        let vigor = growth.vigor;
        if growth.development_age_years <= 0.0 {
            return Vec::new();
        }
        let trunk_height_fraction = self.genome.trunk_height_fraction;
        let crown_spread_factor = 1.0;
        let apical_dominance = self.genome.apical_dominance;
        let branching_rate = self.genome.branching_rate;
        let slope = self.genome.branch_slope;
        let width = self.genome.branch_width;
        let height = growth.height_m;
        let trunk_height = height * trunk_height_fraction;
        let trunk_radius = growth.trunk_radius_m;
        let crown_radius = growth.crown_spread_m * crown_spread_factor;
        let plant_x = ((self.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32;
        let plant_y = ((self.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32;
        let phase_hash = lattice_hash(seed ^ 0x7F4A_7C15_9E37_79B9, plant_x, plant_y);
        let phase = (unit_noise(phase_hash) + 1.0) * std::f32::consts::PI;
        let mut segments = Vec::new();
        if trunk_height > 0.2 {
            // Juvenile shoots form shorter internodes than an established
            // crown. Let spacing grow smoothly with the plant's realized size
            // so seedlings can develop lateral shoots before the sapling stage.
            let relative_size = (height / self.genome.maximum_height_m).clamp(0.0, 1.0);
            let node_spacing = self.genome.branch_spacing_m * (0.40 + 0.38 * relative_size.sqrt());
            let node_count = (trunk_height / node_spacing).floor().max(1.0) as usize;
            for node in 0..node_count {
                let node_hash = lattice_hash(seed ^ phase_hash, node as i32, 0x51A7);
                // Real shoots do not make perfectly level branch whorls. Keep
                // node positions deterministic while allowing inherited trees
                // to form alternating, slightly irregular branch levels.
                let node_jitter = unit_noise(node_hash.rotate_left(23)) * node_spacing * 0.22;
                let start_z = (self.soil_z + node_spacing * (node as f32 + 1.0) + node_jitter)
                    .min(self.soil_z + trunk_height * 0.96);
                let height_fraction =
                    ((start_z - self.soil_z) / trunk_height.max(WORLD_REV_M)).clamp(0.0, 1.0);
                let bud_hash = node_hash;
                let buds = 2 + (bud_hash as usize % 2);
                for bud in 0..buds {
                    let hash = lattice_hash(bud_hash, bud as i32, node as i32);
                    // Bud identity is seed-locked. Light controls subsequent
                    // branch extension, so changing shade cannot make a limb
                    // appear and disappear from one growth rebuild to another.
                    let bud_success = (0.55 + branching_rate * 0.40).clamp(0.0, 1.0);
                    if unit_noise(hash).mul_add(0.5, 0.5) > bud_success {
                        continue;
                    }
                    let angle = phase
                        + node as f32 * 2.399_963_1
                        + std::f32::consts::TAU * (bud as f32 / buds as f32)
                        + unit_noise(hash.rotate_left(11)) * 0.45;
                    let length_m = crown_radius
                        * (0.35 + 0.35 * growth.structural_light)
                        * (0.72 + unit_noise(hash.rotate_left(5)).abs() * 0.28)
                        * (0.72 + unit_noise(hash ^ 0xD6E8_FEB8_6659_FD93).abs() * 0.28);
                    let start = Vec3::new(self.x, self.y, start_z);
                    let horizontal = Vec3::new(angle.cos(), angle.sin(), 0.0);
                    let upward =
                        0.16 + slope * 0.78 + apical_dominance * (1.0 - height_fraction) * 0.40;
                    let end = start + horizontal * length_m + Vec3::Z * (length_m * upward);
                    let radius_m = (trunk_radius
                        * width
                        * 0.68
                        * (0.72 + unit_noise(hash.rotate_left(17)).abs() * 0.28))
                        .max(WORLD_REV_M * 0.55);
                    segments.push(BranchSegment {
                        start,
                        end,
                        radius_m,
                        is_root: false,
                    });

                    // Established shoots form smaller axes where light and
                    // resource supply can support them. Hashes fix bud identity
                    // across growth updates; seed traits shape the probability.
                    let side_probability =
                        (branching_rate * (0.70 + 0.30 * vigor) * (1.0 - apical_dominance * 0.25))
                            .clamp(0.0, 1.0);
                    if matches!(
                        growth.stage,
                        GrowthStage::Pole | GrowthStage::Mature | GrowthStage::Senescent
                    ) && unit_noise(hash.rotate_left(29)).mul_add(0.5, 0.5) < side_probability
                    {
                        let split_fraction = 0.48 + unit_noise(hash.rotate_left(31)).abs() * 0.28;
                        let midpoint = start.lerp(end, split_fraction);
                        let split_angle = angle + unit_noise(hash.rotate_left(37)) * 1.3;
                        let side_length = length_m * (0.28 + 0.18 * growth.structural_light);
                        let side_end = midpoint
                            + Vec3::new(split_angle.cos(), split_angle.sin(), 0.32 + slope * 0.38)
                                * side_length;
                        let side_radius = radius_m * 0.42;
                        segments.push(BranchSegment {
                            start: midpoint,
                            end: side_end,
                            radius_m: side_radius,
                            is_root: false,
                        });

                        // Fine outer shoots divide again near their tips,
                        // creating a branching canopy instead of bare, evenly
                        // spaced spokes. Their abundance follows vigor.
                        let twig_hash =
                            lattice_hash(hash ^ 0xA076_1D64_78BD_642F, node as i32, bud as i32);
                        let twig_probability = branching_rate * vigor * 0.80;
                        if unit_noise(twig_hash).mul_add(0.5, 0.5) < twig_probability {
                            let twig_start = midpoint.lerp(side_end, 0.62);
                            let twig_count = 1 + ((twig_hash >> 8) % 3) as usize;
                            for twig in 0..twig_count {
                                let side = if twig == 0 { -1.0 } else { 1.0 };
                                let twig_angle = split_angle
                                    + side
                                        * (0.48
                                            + unit_noise(twig_hash.rotate_left(17)).abs() * 0.36);
                                let twig_length =
                                    side_length * (0.46 + 0.16 * growth.structural_light);
                                let twig_end = twig_start
                                    + Vec3::new(
                                        twig_angle.cos(),
                                        twig_angle.sin(),
                                        0.42 + slope * 0.32,
                                    ) * twig_length;
                                segments.push(BranchSegment {
                                    start: twig_start,
                                    end: twig_end,
                                    radius_m: side_radius * 0.36,
                                    is_root: false,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Preserve the apical leader above the main trunk. It is the growing
        // shoot axis where new leaves and lateral buds form, not an empty gap
        // above a truncated stem. Its relative strength follows apical dominance.
        if height > trunk_height + WORLD_REV_M {
            let leader_start = Vec3::new(self.x, self.y, self.soil_z + trunk_height * 0.92);
            let leader_end = Vec3::new(self.x, self.y, self.soil_z + height);
            segments.push(BranchSegment {
                start: leader_start,
                end: leader_end,
                radius_m: (trunk_radius * (0.18 + apical_dominance * 0.10)).max(WORLD_REV_M * 0.55),
                is_root: false,
            });
        }

        // Root architecture follows each seed's inherited continuous traits.
        let (root_count, lateral_extent, depth_factor) = self.root_profile();
        let root_stage_scale = match growth.stage {
            GrowthStage::Seedling => 0.12,
            GrowthStage::Sapling => 0.35,
            GrowthStage::Pole => 0.68,
            GrowthStage::Mature => 1.0,
            GrowthStage::Senescent => 0.92,
        };
        let root_extent =
            lateral_extent * root_stage_scale * vigor * (0.85 + self.fertility * 0.25);
        let root_depth = depth_factor * root_stage_scale * vigor * (1.10 - self.moisture * 0.50);
        let base = Vec3::new(self.x, self.y, self.soil_z);
        for root in 0..root_count {
            let root_hash =
                lattice_hash(seed ^ 0xDB4F_0B91_75AE_2165, plant_x + root as i32, plant_y);
            let angle = phase
                + std::f32::consts::TAU * root as f32 / root_count as f32
                + unit_noise(root_hash) * 0.16;
            let direction = Vec3::new(angle.cos(), angle.sin(), 0.0);
            let length = root_extent * (0.72 + unit_noise(root_hash.rotate_left(19)).abs() * 0.28);
            let depth = root_depth * (0.72 + unit_noise(root_hash.rotate_left(37)).abs() * 0.28);
            let shoulder = base + direction * (length * 0.48) - Vec3::Z * (depth * 0.28);
            let tip = base + direction * length - Vec3::Z * depth;
            segments.push(BranchSegment {
                start: base,
                end: shoulder,
                radius_m: (trunk_radius * 0.20).max(WORLD_REV_M * 0.55),
                is_root: true,
            });
            segments.push(BranchSegment {
                start: shoulder,
                end: tip,
                radius_m: (trunk_radius * 0.085).max(WORLD_REV_M * 0.45),
                is_root: true,
            });
        }
        segments
    }

    /// Sample seed-specific, age- and environment-dependent shoots at the
    /// implicit 1 cm REV resolution; the 10 cm preview is only a render LOD.
    fn content_at_indexed_point(
        self,
        growth: PlantGrowth,
        point: Vec3,
        seed: u64,
        cell: [i32; 3],
        branches: &[BranchSegment],
        season_phase: f32,
        candidates: &[PlantSpatialRef],
    ) -> Option<(GeneratedCellContent, Option<Vec3>)> {
        let trunk_height = growth.trunk_height_m;
        let trunk_radius = growth.trunk_radius_m;
        let local_z = point.z - self.soil_z;
        let local_x = point.x - self.x;
        let local_y = point.y - self.y;
        let has_trunk_candidate = candidates
            .iter()
            .any(|candidate| matches!(candidate, PlantSpatialRef::Trunk(_)));
        if has_trunk_candidate && (0.0..trunk_height).contains(&local_z) {
            let taper = 0.72 + 0.28 * (1.0 - local_z / trunk_height.max(WORLD_REV_M));
            let radius = trunk_radius * taper;
            if local_x * local_x + local_y * local_y <= radius * radius {
                return Some((GeneratedCellContent::PlantWood, Some(Vec3::Z)));
            }
        }

        for candidate in candidates {
            let PlantSpatialRef::Branch { branch_index, .. } = *candidate else {
                continue;
            };
            let Some(branch) = branches.get(branch_index) else {
                continue;
            };
            if branch.is_root {
                continue;
            }
            let segment = branch.end - branch.start;
            let length_squared = segment.length_squared();
            if length_squared <= f32::EPSILON {
                continue;
            }
            let along = ((point - branch.start).dot(segment) / length_squared).clamp(0.0, 1.0);
            let closest = branch.start + segment * along;
            if point.distance_squared(closest) <= branch.radius_m.powi(2) {
                return Some((GeneratedCellContent::PlantWood, Some(segment.normalize())));
            }
        }

        let in_leaf_season = (0.20..0.84).contains(&season_phase);
        if in_leaf_season || self.genome.leaf_retention >= 0.85 {
            let leaf_radius =
                (0.20 + growth.crown_spread_m * 0.055).clamp(0.22, LEAF_CLUSTER_RADIUS_M);
            for candidate in candidates {
                let PlantSpatialRef::Branch { branch_index, .. } = *candidate else {
                    continue;
                };
                let Some(branch) = branches.get(branch_index) else {
                    continue;
                };
                if branch.is_root {
                    continue;
                }
                for (cluster_index, center) in [branch.start.lerp(branch.end, 0.62), branch.end]
                    .into_iter()
                    .enumerate()
                {
                    if point.distance_squared(center) > leaf_radius * leaf_radius {
                        continue;
                    }
                    let cluster_key = (branch_index as i32)
                        .wrapping_mul(37)
                        .wrapping_add((cluster_index as i32).wrapping_mul(101));
                    let leaf_hash = lattice_hash(
                        seed ^ 0xD6E8_FEB8_6659_FD93,
                        cell[0] ^ cell[2] ^ cluster_key,
                        cell[1] + (center.z / WORLD_CELL_M) as i32 + cluster_key,
                    );
                    if leaf_hash % 10 != 0 {
                        return Some((GeneratedCellContent::PlantFoliage, None));
                    }
                }
            }
        }
        None
    }

    fn wood_intersects_region(
        self,
        growth: PlantGrowth,
        branches: &[BranchSegment],
        region_min: Vec3,
        region_max: Vec3,
        include_roots: bool,
    ) -> bool {
        let trunk_radius = growth.trunk_radius_m;
        let trunk_min = Vec3::new(self.x - trunk_radius, self.y - trunk_radius, self.soil_z);
        let trunk_max = Vec3::new(
            self.x + trunk_radius,
            self.y + trunk_radius,
            self.soil_z + growth.trunk_height_m,
        );
        let overlaps = |min: Vec3, max: Vec3| {
            min.x <= region_max.x
                && max.x >= region_min.x
                && min.y <= region_max.y
                && max.y >= region_min.y
                && min.z <= region_max.z
                && max.z >= region_min.z
        };
        overlaps(trunk_min, trunk_max)
            || branches
                .iter()
                .filter(|branch| include_roots || !branch.is_root)
                .any(|branch| {
                    let radius = branch.radius_m + WORLD_REV_M;
                    let extent = Vec3::splat(radius);
                    overlaps(
                        branch.start.min(branch.end) - extent,
                        branch.start.max(branch.end) + extent,
                    )
                })
    }

    fn voxels_at_age(
        self,
        growth: PlantGrowth,
        seed: u64,
        branches: &[BranchSegment],
        season_phase: f32,
        world_age_years: f32,
        wood_cell_m: f32,
        wood_region: Option<(Vec3, f32)>,
        surface_only: bool,
        include_roots: bool,
        max_wood_cells: Option<usize>,
    ) -> Result<Vec<TerrainVoxel>, &'static str> {
        if world_age_years < self.germination_age_years {
            let [seed_x, seed_y, seed_z] = self.seed_cell();
            let pigment = (unit_noise(self.genome.individual_id) + 1.0) * 0.5;
            return Ok(vec![TerrainVoxel {
                center: Vec3::new(
                    (seed_x as f32 + 0.5) * WORLD_REV_M - WORLD_SIDE_M * 0.5,
                    (seed_y as f32 + 0.5) * WORLD_REV_M - WORLD_SIDE_M * 0.5,
                    (seed_z as f32 + 0.5) * WORLD_REV_M,
                ),
                size: Vec3::splat(WORLD_REV_M),
                color: [0.36 + pigment * 0.16, 0.24 + pigment * 0.10, 0.10, 1.0],
                content: GeneratedCellContent::Seed,
                fiber_direction: None,
            }]);
        }
        let pigment = (unit_noise(self.genome.individual_id ^ seed) + 1.0) * 0.5;
        let bark_color = [
            0.28 + pigment * 0.32,
            0.17 + self.genome.mature_trunk_radius_m * 0.18,
            0.09 + self.genome.moisture_optimum * 0.12,
            1.0,
        ];
        let leaf_color = [
            0.08 + pigment * 0.10 + self.genome.leaf_retention * 0.04,
            0.36 + self.genome.shade_tolerance * 0.18 + self.moisture * 0.12,
            0.035 + pigment * 0.045,
            1.0,
        ];
        let trunk_height = growth.trunk_height_m;
        let trunk_radius = growth.trunk_radius_m;
        let side_cells = (WORLD_SIDE_M / wood_cell_m).ceil() as i32;
        let height_cells = (WORLD_SAMPLE_HEIGHT_M / wood_cell_m).ceil() as i32;
        let trunk_half_cell = wood_cell_m * 0.5;
        let mut trunk_min_x = (((self.x - trunk_radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor()
            as i32)
            .clamp(0, side_cells - 1);
        let mut trunk_max_x = (((self.x + trunk_radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor()
            as i32)
            .clamp(0, side_cells - 1);
        let mut trunk_min_y = (((self.y - trunk_radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor()
            as i32)
            .clamp(0, side_cells - 1);
        let mut trunk_max_y = (((self.y + trunk_radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor()
            as i32)
            .clamp(0, side_cells - 1);
        let mut trunk_min_z = (self.soil_z / wood_cell_m).floor() as i32;
        let mut trunk_max_z = ((self.soil_z + trunk_height) / wood_cell_m).floor() as i32;
        if let Some((center, radius)) = wood_region {
            trunk_min_x = trunk_min_x
                .max(((center.x - radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
            trunk_max_x = trunk_max_x
                .min(((center.x + radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
            trunk_min_y = trunk_min_y
                .max(((center.y - radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
            trunk_max_y = trunk_max_y
                .min(((center.y + radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
            trunk_min_z = trunk_min_z.max(((center.z - radius) / wood_cell_m).floor() as i32);
            trunk_max_z = trunk_max_z.min(((center.z + radius) / wood_cell_m).floor() as i32);
        }
        trunk_min_x = trunk_min_x.max(0);
        trunk_min_y = trunk_min_y.max(0);
        trunk_min_z = trunk_min_z.max(0);
        trunk_max_x = trunk_max_x.min(side_cells - 1);
        trunk_max_y = trunk_max_y.min(side_cells - 1);
        trunk_max_z = trunk_max_z.min(height_cells - 1);
        // Render callers may request only boundary cells. Physics requests the
        // full occupied volume so hidden REVs still carry mass and bonds.
        let trunk_cell_occupied = |cell: [i32; 3]| {
            if cell[0] < 0
                || cell[0] >= side_cells
                || cell[1] < 0
                || cell[1] >= side_cells
                || cell[2] < 0
                || cell[2] >= height_cells
            {
                return false;
            }
            let center = plant_cell_center(cell, wood_cell_m);
            let z = center.z - self.soil_z;
            if z + trunk_half_cell <= 0.0 || z - trunk_half_cell >= trunk_height {
                return false;
            }
            let taper =
                0.72 + 0.28 * (1.0 - z.clamp(0.0, trunk_height) / trunk_height.max(wood_cell_m));
            let radius = trunk_radius * taper;
            let dx = ((center.x - self.x).abs() - trunk_half_cell).max(0.0);
            let dy = ((center.y - self.y).abs() - trunk_half_cell).max(0.0);
            dx * dx + dy * dy <= radius * radius
        };
        let mut result = Vec::new();
        for z_cell in trunk_min_z..=trunk_max_z {
            let center_z = (z_cell as f32 + 0.5) * wood_cell_m;
            let local_z = center_z - self.soil_z;
            if local_z + trunk_half_cell <= 0.0 || local_z - trunk_half_cell >= trunk_height {
                continue;
            }
            let taper = 0.72
                + 0.28 * (1.0 - local_z.clamp(0.0, trunk_height) / trunk_height.max(wood_cell_m));
            let radius = trunk_radius * taper;
            for y_cell in trunk_min_y..=trunk_max_y {
                let center_y = (y_cell as f32 + 0.5) * wood_cell_m - WORLD_SIDE_M * 0.5;
                let dy = ((center_y - self.y).abs() - trunk_half_cell).max(0.0);
                for x_cell in trunk_min_x..=trunk_max_x {
                    let center_x = (x_cell as f32 + 0.5) * wood_cell_m - WORLD_SIDE_M * 0.5;
                    let dx = ((center_x - self.x).abs() - trunk_half_cell).max(0.0);
                    if dx * dx + dy * dy > radius * radius {
                        continue;
                    }
                    if surface_only
                        && cell_has_six_neighbours([x_cell, y_cell, z_cell], &trunk_cell_occupied)
                    {
                        continue;
                    }
                    if max_wood_cells.is_some_and(|limit| result.len() >= limit) {
                        return Err("physical plant wood exceeds its REV limit");
                    }
                    result.push(TerrainVoxel {
                        center: Vec3::new(center_x, center_y, center_z),
                        size: Vec3::splat(wood_cell_m),
                        color: bark_color,
                        content: GeneratedCellContent::PlantWood,
                        fiber_direction: Some(Vec3::Z),
                    });
                }
            }
        }

        for branch in branches
            .iter()
            .filter(|branch| include_roots || !branch.is_root)
        {
            let segment = branch.end - branch.start;
            let cell_half = wood_cell_m * 0.5;
            let extent = Vec3::splat(branch.radius_m + cell_half);
            let min = branch.start.min(branch.end) - extent;
            let max = branch.start.max(branch.end) + extent;
            let mut min_cell = [
                ((min.x + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32,
                ((min.y + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32,
                (min.z / wood_cell_m).floor() as i32,
            ];
            let mut max_cell = [
                ((max.x + WORLD_SIDE_M * 0.5) / wood_cell_m).ceil() as i32,
                ((max.y + WORLD_SIDE_M * 0.5) / wood_cell_m).ceil() as i32,
                (max.z / wood_cell_m).ceil() as i32,
            ];
            if let Some((center, radius)) = wood_region {
                min_cell[0] = min_cell[0]
                    .max(((center.x - radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
                min_cell[1] = min_cell[1]
                    .max(((center.y - radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
                min_cell[2] = min_cell[2].max(((center.z - radius) / wood_cell_m).floor() as i32);
                max_cell[0] = max_cell[0]
                    .min(((center.x + radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
                max_cell[1] = max_cell[1]
                    .min(((center.y + radius + WORLD_SIDE_M * 0.5) / wood_cell_m).floor() as i32);
                max_cell[2] = max_cell[2].min(((center.z + radius) / wood_cell_m).floor() as i32);
            }
            min_cell[0] = min_cell[0].max(0);
            min_cell[1] = min_cell[1].max(0);
            min_cell[2] = min_cell[2].max(0);
            max_cell[0] = max_cell[0].min(side_cells - 1);
            max_cell[1] = max_cell[1].min(side_cells - 1);
            max_cell[2] = max_cell[2].min(height_cells - 1);
            let mut exceeded_limit = false;
            visit_branch_cells(
                branch.start,
                branch.end,
                branch.radius_m,
                wood_cell_m,
                min_cell,
                max_cell,
                surface_only,
                |cell| {
                    if max_wood_cells.is_some_and(|limit| result.len() >= limit) {
                        exceeded_limit = true;
                        return false;
                    }
                    result.push(TerrainVoxel {
                        center: plant_cell_center(cell, wood_cell_m),
                        size: Vec3::splat(wood_cell_m),
                        color: bark_color,
                        content: GeneratedCellContent::PlantWood,
                        fiber_direction: Some(segment.normalize_or_zero()),
                    });
                    true
                },
            );
            if exceeded_limit {
                return Err("physical plant wood exceeds its REV limit");
            }
        }

        if wood_cell_m < WORLD_CELL_M {
            return Ok(result);
        }

        // Foliage grows in small clusters at branch tips. A closed crown shape
        // is deliberately not imposed; branching and local resources determine
        // the silhouette.
        // Larger foliage clusters keep the canopy continuous at render scale;
        // their size follows the grown crown rather than a fixed tree template.
        let leaf_radius = (0.20 + growth.crown_spread_m * 0.055).clamp(0.22, 0.42);
        let leaf_cells = (leaf_radius / WORLD_CELL_M).ceil() as i32;
        if (0.20..0.84).contains(&season_phase) || self.genome.leaf_retention >= 0.85 {
            for (branch_index, branch) in branches
                .iter()
                .enumerate()
                .filter(|(_, branch)| !branch.is_root)
            {
                for (cluster_index, center) in [branch.start.lerp(branch.end, 0.62), branch.end]
                    .into_iter()
                    .enumerate()
                {
                    let center_cell = [
                        ((center.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32,
                        ((center.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32,
                        (center.z / WORLD_CELL_M).round() as i32,
                    ];
                    let cluster_key = (branch_index as i32)
                        .wrapping_mul(37)
                        .wrapping_add((cluster_index as i32).wrapping_mul(101));
                    for dz in -leaf_cells..=leaf_cells {
                        for dy in -leaf_cells..=leaf_cells {
                            for dx in -leaf_cells..=leaf_cells {
                                let offset =
                                    Vec3::new(dx as f32, dy as f32, dz as f32) * WORLD_CELL_M;
                                if offset.length_squared() > leaf_radius * leaf_radius {
                                    continue;
                                }
                                let hash = lattice_hash(
                                    seed ^ 0xD6E8_FEB8_6659_FD93,
                                    center_cell[0] + dx ^ cluster_key,
                                    center_cell[1] + dy + cluster_key,
                                );
                                if hash % 10 == 0 {
                                    continue;
                                }
                                let shade = ((hash >> 24) as f32 / 16_777_215.0 - 0.5) * 0.08;
                                result.push(TerrainVoxel {
                                    center: center + offset,
                                    size: Vec3::splat(WORLD_CELL_M),
                                    color: [
                                        (leaf_color[0] + shade).clamp(0.0, 1.0),
                                        (leaf_color[1] + shade).clamp(0.0, 1.0),
                                        (leaf_color[2] + shade).clamp(0.0, 1.0),
                                        1.0,
                                    ],
                                    content: GeneratedCellContent::PlantFoliage,
                                    fiber_direction: None,
                                });
                            }
                        }
                    }
                }
            }
        }
        if growth.height_m > 0.0 {
            let [seed_x, seed_y, seed_z] = self.apical_seed_cell(growth);
            result.push(TerrainVoxel {
                center: Vec3::new(
                    (seed_x as f32 + 0.5) * WORLD_REV_M - WORLD_SIDE_M * 0.5,
                    (seed_y as f32 + 0.5) * WORLD_REV_M - WORLD_SIDE_M * 0.5,
                    (seed_z as f32 + 0.5) * WORLD_REV_M,
                ),
                size: Vec3::splat(WORLD_REV_M),
                color: [0.42 + pigment * 0.16, 0.29 + pigment * 0.10, 0.12, 1.0],
                content: GeneratedCellContent::Seed,
                fiber_direction: None,
            });
        }
        Ok(result)
    }
}

fn plant_cell_center(cell: [i32; 3], cell_m: f32) -> Vec3 {
    Vec3::new(
        (cell[0] as f32 + 0.5) * cell_m - WORLD_SIDE_M * 0.5,
        (cell[1] as f32 + 0.5) * cell_m - WORLD_SIDE_M * 0.5,
        (cell[2] as f32 + 0.5) * cell_m,
    )
}

fn cell_has_six_neighbours(cell: [i32; 3], occupied: impl Fn([i32; 3]) -> bool) -> bool {
    for axis in 0..3 {
        for offset in [-1, 1] {
            let mut neighbour = cell;
            neighbour[axis] += offset;
            if !occupied(neighbour) {
                return false;
            }
        }
    }
    true
}

/// Slice a capsule along its longest axis. The final capsule/AABB predicate
/// matches the exhaustive bounding-box rasterizer; only empty space is pruned.
fn visit_branch_cells(
    start: Vec3,
    end: Vec3,
    radius_m: f32,
    cell_m: f32,
    min_cell: [i32; 3],
    max_cell: [i32; 3],
    surface_only: bool,
    mut emit: impl FnMut([i32; 3]) -> bool,
) {
    if (0..3).any(|axis| min_cell[axis] > max_cell[axis]) {
        return;
    }
    let direction = end - start;
    let abs_direction = direction.abs();
    let axis = if abs_direction.x >= abs_direction.y && abs_direction.x >= abs_direction.z {
        0
    } else if abs_direction.y >= abs_direction.z {
        1
    } else {
        2
    };
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let origin = Vec3::new(-WORLD_SIDE_M * 0.5, -WORLD_SIDE_M * 0.5, 0.0);
    // Conservative guard for pruning only: the occupancy test is unmodified.
    let extent = radius_m + cell_m * (0.5 + 0.001);
    let half_extent = Vec3::splat(cell_m * 0.5);
    let side_cells = (WORLD_SIDE_M / cell_m).ceil() as i32;
    let height_cells = (WORLD_SAMPLE_HEIGHT_M / cell_m).ceil() as i32;
    let occupied = |cell: [i32; 3]| {
        cell[0] >= 0
            && cell[0] < side_cells
            && cell[1] >= 0
            && cell[1] < side_cells
            && cell[2] >= 0
            && cell[2] < height_cells
            && segment_aabb_distance_squared(
                start,
                end,
                plant_cell_center(cell, cell_m),
                half_extent,
            ) <= radius_m * radius_m
    };
    'slices: for slice in min_cell[axis]..=max_cell[axis] {
        let center_axis = (slice as f32 + 0.5) * cell_m + origin[axis];
        let (t_min, t_max) = if abs_direction[axis] > 1e-12 {
            let a = (center_axis - extent - start[axis]) / direction[axis];
            let b = (center_axis + extent - start[axis]) / direction[axis];
            if a.min(b) > 1.0 || a.max(b) < 0.0 {
                continue;
            }
            (a.min(b).max(0.0), a.max(b).min(1.0))
        } else {
            (0.0, 1.0)
        };
        let a = start + direction * t_min;
        let b = start + direction * t_max;
        let low = a.min(b) - Vec3::splat(extent);
        let high = a.max(b) + Vec3::splat(extent);
        let lo_u = (((low[u] - origin[u]) / cell_m).floor() as i32).max(min_cell[u]);
        let hi_u = (((high[u] - origin[u]) / cell_m).ceil() as i32).min(max_cell[u]);
        let lo_v = (((low[v] - origin[v]) / cell_m).floor() as i32).max(min_cell[v]);
        let hi_v = (((high[v] - origin[v]) / cell_m).ceil() as i32).min(max_cell[v]);
        for cv in lo_v..=hi_v {
            for cu in lo_u..=hi_u {
                let mut cell = [0; 3];
                cell[axis] = slice;
                cell[u] = cu;
                cell[v] = cv;
                if occupied(cell) && (!surface_only || !cell_has_six_neighbours(cell, &occupied)) {
                    if !emit(cell) {
                        break 'slices;
                    }
                }
            }
        }
    }
}

/// Exact piecewise-quadratic segment/AABB distance. There are at most six
/// face crossings; between consecutive crossings the squared distance is a
/// quadratic, whose minimum can be computed directly. Double intermediates
/// avoid cancellation when a long branch passes close to a centimetre cell.
fn segment_aabb_distance_squared(start: Vec3, end: Vec3, center: Vec3, half_extent: Vec3) -> f32 {
    let relative = start.as_dvec3() - center.as_dvec3();
    let direction = end.as_dvec3() - start.as_dvec3();
    let half = half_extent.as_dvec3();
    let distance_at = |t: f64| {
        ((relative + direction * t).abs() - half)
            .max(glam::DVec3::ZERO)
            .length_squared()
    };
    let length_squared = direction.length_squared();
    if length_squared == 0.0 {
        return distance_at(0.0) as f32;
    }
    let center_projection = (-relative.dot(direction) / length_squared).clamp(0.0, 1.0);
    let mut best = distance_at(center_projection);
    if best == 0.0 {
        return 0.0;
    }
    let mut crossings = [0.0_f64; 8];
    crossings[1] = 1.0;
    let mut count = 2;
    for axis in 0..3 {
        if direction[axis] == 0.0 {
            continue;
        }
        for face in [-half[axis], half[axis]] {
            let t = (face - relative[axis]) / direction[axis];
            if t > 0.0 && t < 1.0 {
                crossings[count] = t;
                count += 1;
            }
        }
    }
    crossings[..count].sort_unstable_by(f64::total_cmp);
    for interval in crossings[..count].windows(2) {
        let lower = interval[0];
        let upper = interval[1];
        let midpoint = (lower + upper) * 0.5;
        let point = relative + direction * midpoint;
        let mut quadratic = 0.0;
        let mut linear = 0.0;
        for axis in 0..3 {
            let offset = if point[axis] < -half[axis] {
                relative[axis] + half[axis]
            } else if point[axis] > half[axis] {
                relative[axis] - half[axis]
            } else {
                continue;
            };
            quadratic += direction[axis] * direction[axis];
            linear += direction[axis] * offset;
        }
        let t = if quadratic > 0.0 {
            (-linear / quadratic).clamp(lower, upper)
        } else {
            midpoint
        };
        best = best.min(distance_at(t));
        if best == 0.0 {
            return 0.0;
        }
    }
    best as f32
}

/// Allocate finite normalized soil water and nutrient availability among
/// overlapping root footprints, then limit developmental age by light and
/// resource sufficiency. Ecology values are dimensionless world fields; this
/// is a competitive growth model, not yet a calibrated soil mass balance.
fn compute_plant_growth(
    plants: &[Plant],
    age_years: f32,
    seed: u64,
    _calendar_quarter: u64,
    previous: &[PlantGrowth],
) -> Vec<PlantGrowth> {
    let potential_heights: Vec<f32> = plants
        .iter()
        .map(|plant| {
            let plant_age = (age_years - plant.germination_age_years).max(0.0);
            plant.genome.maximum_height_m * (1.0 - (-plant_age / 34.0).exp())
        })
        .collect();

    plants
        .iter()
        .enumerate()
        .map(|(index, plant)| {
            let genome = plant.genome;
            let plant_age = (age_years - plant.germination_age_years).max(0.0);
            let max_height = genome.maximum_height_m;
            let max_crown = genome.crown_spread_m;
            let max_trunk_radius = genome.mature_trunk_radius_m;
            let shade_tolerance = genome.shade_tolerance;
            let water_demand = genome.water_demand;
            let nutrient_demand = genome.nutrient_demand;
            let longevity = genome.longevity_years;
            let (_, root_extent, _) = plant.root_profile();
            let mut shared_water_demand = water_demand;
            let mut shared_nutrient_demand = nutrient_demand;
            for (other_index, other) in plants.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let (_, other_root_extent, _) = other.root_profile();
                let distance = Vec3::new(plant.x - other.x, plant.y - other.y, 0.0).length();
                let overlap = ((root_extent + other_root_extent - distance)
                    / (root_extent + other_root_extent).max(WORLD_CELL_M))
                .clamp(0.0, 1.0);
                let shared_fraction = overlap * overlap * 0.55;
                shared_water_demand += other.genome.water_demand * shared_fraction;
                shared_nutrient_demand += other.genome.nutrient_demand * shared_fraction;
            }
            let water_sufficiency =
                (plant.moisture * 1.10 / shared_water_demand.max(0.05)).clamp(0.08, 1.0);
            let nutrient_sufficiency =
                (plant.fertility * 1.10 / shared_nutrient_demand.max(0.05)).clamp(0.08, 1.0);

            // Eight annual daylight sectors approximate a moving sun. A taller
            // neighbor blocks a sector only when its projected shadow reaches
            // this crown; each ray has the same deterministic seed phase.
            let azimuth_phase = unit_noise(seed ^ 0x4CF5_AD43_2745_937F) * std::f32::consts::PI;
            let own_height = potential_heights[index];
            let mut lit_sectors = 0.0;
            for ray in 0..8 {
                let angle = azimuth_phase + std::f32::consts::TAU * ray as f32 / 8.0;
                let sun = Vec3::new(angle.cos(), angle.sin(), 0.0);
                let mut blocked = false;
                for (other_index, other) in plants.iter().enumerate() {
                    if other_index == index {
                        continue;
                    }
                    let offset = Vec3::new(other.x - plant.x, other.y - plant.y, 0.0);
                    let along_sun = offset.dot(sun);
                    if along_sun <= 0.0 {
                        continue;
                    }
                    let lateral = (offset - sun * along_sun).length();
                    let other_crown = other.genome.crown_spread_m;
                    if lateral > other_crown + max_crown * 0.35 {
                        continue;
                    }
                    let sun_elevation = 0.62_f32;
                    let shadow_top = potential_heights[other_index] - along_sun * sun_elevation;
                    if shadow_top > own_height * 0.72 {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    lit_sectors += 1.0;
                }
            }
            let sky_light = lit_sectors / 8.0;
            let light_sufficiency =
                (sky_light + shade_tolerance * (1.0 - sky_light) * 0.55).clamp(0.08, 1.0);

            let temperature_sufficiency = (1.0
                - ((plant.temperature_kelvin - genome.temperature_optimum_kelvin) / 16.0).abs())
            .clamp(0.20, 1.0);
            let limiting_factor = light_sufficiency
                .min(water_sufficiency)
                .min(nutrient_sufficiency)
                .min(temperature_sufficiency);
            let prior = previous.get(index).copied().unwrap_or_default();
            let structural_light = light_sufficiency.max(prior.structural_light);
            let development_age_years =
                (plant_age * limiting_factor).max(prior.development_age_years);
            let stage = if plant_age >= longevity * 0.82 || prior.stage == GrowthStage::Senescent {
                GrowthStage::Senescent
            } else {
                GrowthStage::from_development_age(development_age_years).max(prior.stage)
            };
            let height_fraction =
                (1.0 - (-development_age_years / 34.0).exp()) / (1.0 - (-100.0_f32 / 34.0).exp());
            let individual_hash = lattice_hash(
                seed ^ 0x94D0_49BB_1331_11EB,
                ((plant.x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32,
                ((plant.y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M).round() as i32,
            );
            let individual_variation = 0.92 + (unit_noise(individual_hash) + 1.0) * 0.08;
            let vigor = limiting_factor * individual_variation;
            let height_m = max_height * height_fraction * individual_variation;
            let trunk_height_fraction = genome.trunk_height_fraction;
            let trunk_radius_m = max_trunk_radius * height_fraction.sqrt() * individual_variation;
            // Crown radius follows accumulated height continuously. Discrete
            // stage multipliers made young trees unnaturally narrow and caused
            // a visible jump in crown size at each stage boundary.
            let crown_spread_m = max_crown
                * height_fraction.powf(0.72)
                * (0.55 + 0.45 * structural_light)
                * individual_variation;
            PlantGrowth {
                stage,
                light: light_sufficiency,
                structural_light,
                water: water_sufficiency,
                nutrients: nutrient_sufficiency,
                development_age_years,
                vigor,
                height_m,
                trunk_height_m: height_m * trunk_height_fraction,
                trunk_radius_m,
                crown_spread_m,
            }
        })
        .collect()
}

fn make_plants(seed: u64) -> (Vec<Plant>, Vec<Morphotype>) {
    const TREE_SITE_SIDE: i32 = 9;
    let mut plants = Vec::with_capacity((TREE_SITE_SIDE * TREE_SITE_SIDE) as usize);
    let site_spacing = WORLD_SIDE_M / TREE_SITE_SIDE as f32;
    for gy in 0..TREE_SITE_SIDE {
        for gx in 0..TREE_SITE_SIDE {
            let site_hash = lattice_hash(seed ^ 0xA24B_AED4_963E_E407, gx, gy);
            let x = -WORLD_SIDE_M * 0.5
                + (gx as f32 + 0.5) * site_spacing
                + unit_noise(site_hash) * site_spacing * 0.12;
            let y = -WORLD_SIDE_M * 0.5
                + (gy as f32 + 0.5) * site_spacing
                + unit_noise(site_hash.rotate_left(29)) * site_spacing * 0.12;
            let nx = (x + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let ny = (y + WORLD_SIDE_M * 0.5) / WORLD_CELL_M;
            let (terrain_x, terrain_y) = terrain_field_coordinates(x, y);
            let soil_z = terrain_height(seed, terrain_x, terrain_y);
            let (moisture, fertility) = ecology_at(seed, nx, ny);
            let site_roll = (unit_noise(site_hash.rotate_left(53)) + 1.0) * 0.5;
            let establishment_chance = 0.30 + fertility * 0.28 + moisture * 0.12;
            if site_roll > establishment_chance {
                continue;
            }

            let temperature_kelvin = climate_temperature(seed, nx, ny, soil_z);
            // Seed traits use a bounded normal distribution: ordinary values
            // are common, while the tails still allow unusual morphologies.
            // Each trait is deterministic for this seed and individual.
            let trait_normal = |trait_id: u32| {
                normal_noise(lattice_hash(
                    site_hash ^ 0xD6E8_FEB8_6659_FD93,
                    trait_id as i32,
                    0x5345_4544,
                ))
            };
            let trait_value = |trait_id: u32| (0.5 + trait_normal(trait_id) / 6.0).clamp(0.0, 1.0);
            let height_z = trait_normal(0);
            let crown_z = 0.65 * height_z + 0.76 * trait_normal(1);
            let trunk_z = 0.75 * height_z + 0.66 * trait_normal(2);
            let genome = SeedGenome {
                individual_id: site_hash,
                morphotype_id: 0,
                maximum_height_m: (18.0 + height_z * 8.0).clamp(2.0, 60.0),
                crown_spread_m: (5.0 + crown_z * 2.4).clamp(0.8, 28.0),
                mature_trunk_radius_m: (0.34 + trunk_z * 0.10).clamp(0.08, 1.4),
                shade_tolerance: 0.24 + trait_value(3) * 0.54,
                water_demand: 0.28 + trait_value(4) * 0.46,
                nutrient_demand: 0.26 + trait_value(5) * 0.42,
                longevity_years: 50.0 + trait_value(6) * 90.0,
                moisture_optimum: 0.30 + trait_value(7) * 0.48,
                temperature_optimum_kelvin: 280.0 + trait_value(8) * 15.0,
                root_count: 5 + (trait_value(9) * 6.0) as u8,
                root_extent_m: 0.85 + trait_value(10) * 0.95,
                root_depth_factor: 0.30 + trait_value(11) * 0.85,
                trunk_height_fraction: 0.62 + trait_value(12) * 0.25,
                branching_rate: 0.38 + trait_value(13) * 0.52,
                apical_dominance: 0.18 + trait_value(14) * 0.72,
                branch_spacing_m: 0.55 + trait_value(15) * 0.65,
                branch_slope: 0.12 + trait_value(16) * 0.55,
                branch_width: 0.22 + trait_value(17) * 0.24,
                leaf_retention: trait_value(18),
                mycorrhizal_affinity: trait_value(19),
                germination_delay_years: 0.15 + trait_value(20) * 0.85,
                // Provisional per-locus rate; reproduction is not implemented yet.
                mutation_probability: 0.00001,
            };
            plants.push(Plant {
                genome,
                x,
                y,
                soil_z,
                moisture,
                fertility,
                temperature_kelvin,
                germination_age_years: genome.germination_delay_years,
            });
        }
    }

    let morphotypes = assign_morphotypes(&mut plants);
    (plants, morphotypes)
}

fn morphotype_traits(genome: SeedGenome) -> [f32; 8] {
    [
        genome.maximum_height_m / 24.0,
        genome.crown_spread_m / 4.5,
        genome.shade_tolerance,
        genome.water_demand,
        genome.branching_rate,
        genome.apical_dominance,
        genome.leaf_retention,
        genome.root_extent_m / 2.0,
    ]
}

fn morphotype_distance(left: [f32; 8], right: [f32; 8]) -> f32 {
    (left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / left.len() as f32)
        .sqrt()
}

fn pseudo_latin_name(genome: SeedGenome, id: u64, group_number: usize) -> String {
    let genus_roots = [
        "Sylvara", "Nemora", "Lignora", "Viridia", "Frondara", "Arboria",
    ];
    let height = if genome.maximum_height_m > 17.0 {
        "altissima"
    } else if genome.maximum_height_m < 12.0 {
        "humilis"
    } else {
        "media"
    };
    let branching = if genome.branching_rate > 0.68 {
        "ramosa"
    } else if genome.apical_dominance > 0.62 {
        "stricta"
    } else {
        "patula"
    };
    let genus = genus_roots[(id as usize) % genus_roots.len()];
    format!(
        "{genus} {height} {branching} morph. {}",
        roman_numeral(group_number)
    )
}

fn roman_numeral(mut value: usize) -> String {
    const NUMERALS: &[(usize, &str)] = &[
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut result = String::new();
    for &(amount, numeral) in NUMERALS {
        while value >= amount {
            result.push_str(numeral);
            value -= amount;
        }
    }
    result
}

fn assign_morphotypes(plants: &mut [Plant]) -> Vec<Morphotype> {
    const SIMILARITY_RADIUS: f32 = 0.20;
    let mut order: Vec<usize> = (0..plants.len()).collect();
    order.sort_by_key(|&index| plants[index].genome.individual_id);
    let mut representatives: Vec<[f32; 8]> = Vec::new();
    let mut representative_genomes: Vec<SeedGenome> = Vec::new();
    let mut member_indices: Vec<Vec<usize>> = Vec::new();

    for index in order {
        let traits = morphotype_traits(plants[index].genome);
        let nearest = representatives
            .iter()
            .enumerate()
            .map(|(cluster, representative)| {
                (cluster, morphotype_distance(traits, *representative))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1));
        let cluster = match nearest {
            Some((cluster, distance)) if distance <= SIMILARITY_RADIUS => cluster,
            _ => {
                representatives.push(traits);
                representative_genomes.push(plants[index].genome);
                member_indices.push(Vec::new());
                representatives.len() - 1
            }
        };
        member_indices[cluster].push(index);
    }

    let morphotypes: Vec<Morphotype> = representative_genomes
        .iter()
        .zip(&member_indices)
        .enumerate()
        .map(|(index, (genome, members))| Morphotype {
            id: genome.individual_id,
            latin_name: pseudo_latin_name(*genome, genome.individual_id, index + 1),
            population: members.len(),
        })
        .collect();
    for (cluster_index, morphotype) in morphotypes.iter().enumerate() {
        for &plant_index in &member_indices[cluster_index] {
            plants[plant_index].genome.morphotype_id = morphotype.id;
        }
    }
    morphotypes
}

fn terrain_field_coordinates(x_m: f32, y_m: f32) -> (f32, f32) {
    (x_m, y_m)
}

fn terrain_height(seed: u64, x: f32, y: f32) -> f32 {
    let large = value_noise(seed, x / 2_200.0, y / 2_200.0);
    let medium = value_noise(seed ^ 0xA076_1D64_78BD_642F, x / 700.0, y / 700.0);
    let fine = value_noise(seed ^ 0xE703_7ED1_A0B4_28DB, x / 220.0, y / 220.0);
    // Domain-warped rolling hills and shallow drainage depressions. These
    // wavelengths keep local walking relief gentle while breaking up the
    // formerly uniform hundreds-of-metres slopes. Heights remain 10 cm steps.
    let warp_x = value_noise(seed ^ 0xB65A_37CF_4E29_810D, x / 190.0, y / 190.0) * 22.0;
    let warp_y = value_noise(seed ^ 0x71CA_58D2_903E_B46F, x / 190.0, y / 190.0) * 22.0;
    let hills = value_noise(
        seed ^ 0x9A35_F82D_614C_E70B,
        (x + warp_x) / 65.0,
        (y + warp_y) / 65.0,
    );
    let swale = value_noise(
        seed ^ 0x43D8_6AB9_E715_C02F,
        (x + warp_x) / 130.0,
        (y + warp_y) / 130.0,
    )
    .abs();
    let channel = (1.0 - swale / 0.16).clamp(0.0, 1.0);
    let small = value_noise(seed ^ 0xD138_69FA_275B_C04E, x / 13.0, y / 13.0);
    // Elevation varies over hundreds of metres horizontally; the local relief
    // is a continuous seeded height field sampled on the 10 cm render lattice.
    let elevation_m = (48.0 + large * 48.0 + medium * 18.0 + fine * 4.0 + hills * 2.6
        - channel * channel * 0.65
        + small * 0.16)
        .max(2.0);
    (elevation_m / WORLD_CELL_M).round() * WORLD_CELL_M
}

fn ecology_at(seed: u64, x: f32, y: f32) -> (f32, f32) {
    let moisture = (0.5 + value_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, x / 27.0, y / 27.0) * 0.38)
        .clamp(0.0, 1.0);
    let fertility = (0.55 + value_noise(seed ^ 0x5899_65CC_7537_4CC3, x / 22.0, y / 22.0) * 0.35)
        .clamp(0.1, 1.0);
    (moisture, fertility)
}

fn climate_temperature(seed: u64, x: f32, y: f32, altitude_m: f32) -> f32 {
    // The seed selects a broad regional climate; spatial noise and elevation
    // then create local variation. This is ecological input only and is not
    // yet coupled to the thermal solver.
    let regional_offset_k = unit_noise(seed ^ 0x243F_6A88_85A3_08D3) * 4.0;
    let local_variation_k = value_noise(seed ^ 0x5899_65CC_7537_4CC3, x / 31.0, y / 31.0) * 11.4;
    288.15 + regional_offset_k + local_variation_k - altitude_m * 0.0065
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

/// Deterministic standard-normal sample used for heritable variation.
/// Box-Muller converts two independent hash samples into a bell-shaped value;
/// clipping at four standard deviations keeps generated traits finite.
fn normal_noise(value: u64) -> f32 {
    const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
    let second_hash = lattice_hash(value ^ 0xA076_1D64_78BD_642F, 0x4E4F_524D, 0x414C);
    let u1 = ((value >> 11) as f64 + 0.5) / TWO_POW_53;
    let u2 = (second_hash >> 11) as f64 / TWO_POW_53;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()).clamp(-4.0, 4.0) as f32
}

#[cfg(test)]
mod render_geometry_tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn physical_world_patch_preserves_centimetre_spacing_at_map_scale() {
        let world = GeneratedWorld::from_seed("1001");
        let min_cell = [10, 20, 10];
        let max_cell = [12, 21, 11];
        let oak = crate::material::Material::component_id("oak").unwrap();
        let mut expected_world_positions = Vec::new();
        let patch = world
            .build_physical_patch(SceneConfig::default(), min_cell, max_cell, |cell| {
                expected_world_positions.push(
                    cell.to_voxel_spawn(Composition::pure(oak), 0.0, Vec3::ZERO, false)
                        .unwrap()
                        .center,
                );
                Ok(Some(GeneratedCellPhysics {
                    composition: Composition::pure(oak),
                    anisotropy: 0.0,
                    velocity: Vec3::ZERO,
                    kinematic: false,
                }))
            })
            .unwrap();

        assert_eq!(patch.origin_cell, min_cell);
        assert_eq!(patch.source_cells, vec![[10, 20, 10], [11, 20, 10]]);
        assert_eq!(patch.scene.positions.len(), 2);
        assert!(
            (patch.scene.positions[1].x - patch.scene.positions[0].x - crate::coordinates::CELL_M)
                .abs()
                < 1e-12
        );
        for (local, expected) in patch.scene.positions.iter().zip(expected_world_positions) {
            assert!(
                (patch.world_position(*local) - expected)
                    .abs()
                    .max_element()
                    < 1e-9
            );
        }
    }

    #[test]
    fn analytic_segment_box_distance_matches_refined_reference() {
        let mut state = 0x1345_7623_u32;
        let mut noise = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32
        };
        for case in 0..2048 {
            let start = Vec3::new(noise() * 20.0, noise() * 20.0, noise() * 20.0);
            let mut direction = Vec3::new(noise() - 0.5, noise() - 0.5, noise() - 0.5) * 20.0;
            if case % 31 == 0 {
                direction = Vec3::ZERO;
            } else if case % 7 == 0 {
                direction[case % 3] = 0.0;
            }
            let end = start + direction;
            let center = start + Vec3::new(noise() - 0.5, noise() - 0.5, noise() - 0.5) * 4.0;
            let half = Vec3::new(noise(), noise(), noise()) * 0.2 + Vec3::splat(0.005);
            let relative = start.as_dvec3() - center.as_dvec3();
            let delta = end.as_dvec3() - start.as_dvec3();
            let distance_at = |t: f64| {
                ((relative + delta * t).abs() - half.as_dvec3())
                    .max(glam::DVec3::ZERO)
                    .length_squared()
            };
            let mut lower = 0.0;
            let mut upper = 1.0;
            for _ in 0..110 {
                let third = (upper - lower) / 3.0;
                let a = lower + third;
                let b = upper - third;
                if distance_at(a) <= distance_at(b) {
                    upper = b;
                } else {
                    lower = a;
                }
            }
            let reference = distance_at((lower + upper) * 0.5);
            let actual = f64::from(segment_aabb_distance_squared(start, end, center, half));
            let tolerance = (reference.abs() * 2e-6).max(1e-10);
            assert!(
                (actual - reference).abs() <= tolerance,
                "case {case}: analytic={actual} reference={reference}"
            );
        }
    }

    #[test]
    fn growth_worker_replays_all_elapsed_increments_in_order() {
        let mut expected = GeneratedWorld::from_seed("1001");
        expected.advance_growth(0.24 * SIMULATION_YEAR_SECONDS);
        let mut actual = expected.clone();
        let mut scheduler = GrowthScheduler::default();
        for years in [0.02, 0.001, 0.03, 0.24, 0.005] {
            let elapsed = years * SIMULATION_YEAR_SECONDS;
            expected.advance_growth(elapsed);
            scheduler.advance(&mut actual, elapsed);
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while scheduler.is_busy() {
            assert!(
                std::time::Instant::now() < deadline,
                "growth worker did not finish"
            );
            scheduler.advance(&mut actual, 0.0);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(actual.age_years.to_bits(), expected.age_years.to_bits());
        assert_eq!(
            actual.calendar_years.to_bits(),
            expected.calendar_years.to_bits()
        );
        assert_eq!(actual.render_cache_key(), expected.render_cache_key());
        assert_eq!(actual.voxels.len(), expected.voxels.len());
        for (a, b) in actual.voxels.iter().zip(&expected.voxels) {
            assert_eq!(a.center, b.center);
            assert_eq!(a.size, b.size);
            assert_eq!(a.color, b.color);
            assert_eq!(a.content, b.content);
        }
    }

    #[test]
    fn cloned_world_refines_mature_wood_on_a_worker() {
        let mut world = GeneratedWorld::from_seed("1001");
        world.advance_growth(40.0 * SIMULATION_YEAR_SECONDS);
        let plant = world.plants[0];
        let camera = Vec3::new(plant.x + 1.0, plant.y, plant.soil_z + 1.42);
        let before = (
            world.render_cache_key(),
            world.age_years,
            world.voxels.len(),
        );
        let snapshot_start = std::time::Instant::now();
        let snapshot = world.clone();
        let snapshot_ms = snapshot_start.elapsed().as_secs_f64() * 1_000.0;
        let (elapsed, cells) = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let cells = snapshot.plant_wood_voxels_near(camera, 10.8);
            (start.elapsed(), cells)
        })
        .join()
        .unwrap();
        assert!(!cells.is_empty());
        let mut occupied = HashSet::new();
        for voxel in &cells {
            assert_eq!(voxel.content, GeneratedCellContent::PlantWood);
            assert_eq!(voxel.size, Vec3::splat(WORLD_REV_M));
            assert!(voxel.center.distance_squared(camera) <= 10.8 * 10.8);
            let cell = ((voxel.center + Vec3::new(WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5, 0.0))
                / WORLD_REV_M)
                .floor()
                .as_ivec3()
                .to_array();
            assert!(occupied.insert(cell), "duplicate centimetre render cell");
        }
        assert_eq!(
            before,
            (
                world.render_cache_key(),
                world.age_years,
                world.voxels.len()
            )
        );
        eprintln!("mature seed 1001 render refinement: snapshot_ms={snapshot_ms:.3} detail_ms={:.3} cells={}",
            elapsed.as_secs_f64() * 1_000.0, cells.len());
    }

    #[test]
    fn sliced_branch_rasterizer_preserves_occupied_and_surface_cells() {
        let mut exhaustive_time = std::time::Duration::ZERO;
        let mut sliced_time = std::time::Duration::ZERO;
        let mut full_count = 0;
        let mut surface_count = 0;
        for cell_m in [WORLD_REV_M, WORLD_CELL_M] {
            let base = [
                (WORLD_SIDE_M * 0.5 / cell_m).floor() as i32,
                (WORLD_SIDE_M * 0.5 / cell_m).floor() as i32,
                (0.4 / cell_m).floor() as i32,
            ];
            let start = plant_cell_center(base, cell_m) + Vec3::new(0.003, -0.002, 0.001);
            for direction in [
                Vec3::ZERO,
                Vec3::new(0.24, 0.18, 0.21),
                Vec3::new(-0.24, 0.18, 0.09),
                Vec3::new(0.03, -0.24, 0.16),
                Vec3::new(0.01, 0.02, -0.24),
                Vec3::new(0.0, 0.0, 0.24),
            ] {
                let end = start + direction;
                let radius = cell_m * 1.8;
                let extent = Vec3::splat(radius + cell_m * 0.5);
                let origin = Vec3::new(-WORLD_SIDE_M * 0.5, -WORLD_SIDE_M * 0.5, 0.0);
                let min = ((start.min(end) - extent - origin) / cell_m)
                    .floor()
                    .as_ivec3()
                    .to_array();
                let max = ((start.max(end) + extent - origin) / cell_m)
                    .ceil()
                    .as_ivec3()
                    .to_array();
                for clipped in [false, true] {
                    let mut lo = min;
                    let mut hi = max;
                    if clipped {
                        lo[0] += 1;
                        hi[1] -= 1;
                    }
                    let occupied = |cell| {
                        segment_aabb_distance_squared(
                            start,
                            end,
                            plant_cell_center(cell, cell_m),
                            Vec3::splat(cell_m * 0.5),
                        ) <= radius * radius
                    };
                    let timer = std::time::Instant::now();
                    let mut expected = BTreeSet::new();
                    for z in lo[2]..=hi[2] {
                        for y in lo[1]..=hi[1] {
                            for x in lo[0]..=hi[0] {
                                if occupied([x, y, z]) {
                                    expected.insert([x, y, z]);
                                }
                            }
                        }
                    }
                    exhaustive_time += timer.elapsed();
                    let timer = std::time::Instant::now();
                    let mut actual = BTreeSet::new();
                    visit_branch_cells(start, end, radius, cell_m, lo, hi, false, |cell| {
                        assert!(actual.insert(cell), "duplicate branch cell");
                        true
                    });
                    sliced_time += timer.elapsed();
                    assert_eq!(
                        actual, expected,
                        "slice changed occupied cells: {direction:?} {cell_m} clipped={clipped}"
                    );
                    let expected_surface: BTreeSet<_> = expected
                        .iter()
                        .copied()
                        .filter(|&cell| !cell_has_six_neighbours(cell, &occupied))
                        .collect();
                    let mut surface = BTreeSet::new();
                    visit_branch_cells(start, end, radius, cell_m, lo, hi, true, |cell| {
                        surface.insert(cell);
                        true
                    });
                    assert_eq!(
                        surface, expected_surface,
                        "surface filtering changed the boundary"
                    );
                    full_count += actual.len();
                    surface_count += surface.len();
                }
            }
        }
        eprintln!("synthetic branch rasterization only: exhaustive_ms={:.3} sliced_ms={:.3} full_cells={full_count} surface_cells={surface_count}",
            exhaustive_time.as_secs_f64() * 1000.0, sliced_time.as_secs_f64() * 1000.0);
    }
}

fn smooth(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
