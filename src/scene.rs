use crate::{
    bond::{Bond, BondWorkspace, EMPTY},
    material::{Material, PhaseState},
    voxel::{Composition, Microstructure, Voxel},
};
use glam::{Quat, Vec3};
use serde::Deserialize;
use std::{collections::HashMap, fs};

pub const CELL: f32 = 0.01;
pub const AREA: f32 = CELL * CELL;

/// CPU preconditioner used by the coupled elastic PCG solver.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CpuPreconditioner {
    /// Block incomplete Cholesky with zero fill; reduces iterations but has
    /// sequential triangular application passes.
    #[default]
    IncompleteCholesky0,
    /// Independent per-REV 3×3 diagonal blocks; parallel application can be
    /// faster on multicore CPUs at the cost of potentially more PCG iterations.
    BlockJacobi,
}

/// Inclusive world-space bounds for REV centres (metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldBounds {
    pub min: Vec3,
    pub max: Vec3,
}

/// Temperature, impact, pressure, and dwell thresholds for automatic bonding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalBondingRule {
    pub component: u16,
    pub minimum_mass_fraction: f32,
    pub minimum_temperature_kelvin: f32,
    pub minimum_impact_energy_j: f64,
    /// Normal contact impulse divided by fixed-step duration and REV face area.
    pub minimum_contact_pressure_pa: f32,
    /// Continuous time for which the contact pressure threshold is maintained.
    pub minimum_dwell_seconds: f32,
}

/// Automatic fusion criterion evaluated from real contact impulses and dwell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalFusionRule {
    pub minimum_temperature_kelvin: f32,
    pub minimum_contact_pressure_pa: f32,
    pub minimum_dwell_seconds: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct BondingContactState {
    dwell_seconds: f32,
    impact_qualified: bool,
    last_seen_substep: u64,
}

#[derive(Default)]
struct ThermalScratch {
    edges: Vec<(usize, usize, f64)>,
    conductance_sum: Vec<f64>,
    temperature: Vec<f64>,
    energy_delta: Vec<f64>,
}

/// Cumulative material and energy removed by world-boundary deletion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundaryRemovalLedger {
    pub voxel_count: u64,
    pub mass_kg: f64,
    pub internal_energy_j: f64,
    pub kinetic_energy_j: f64,
    pub pending_bond_energy_j: f64,
    pub momentum_kg_m_s: [f64; 3],
}

/// User-supplied initial state for one 1 cm REV.
#[derive(Clone, Debug)]
pub struct VoxelSpawn {
    pub center: Vec3,
    pub composition: Composition,
    pub velocity: Vec3,
    pub microstructure: Microstructure,
    pub kinematic: bool,
    pub temperature_kelvin: f32,
}

impl VoxelSpawn {
    pub fn new(center: Vec3, component: u16) -> Self {
        Self {
            center,
            composition: Composition::pure(component),
            velocity: Vec3::ZERO,
            microstructure: Microstructure {
                orientation: Quat::IDENTITY,
                anisotropy: 0.0,
            },
            kinematic: false,
            temperature_kelvin: 293.15,
        }
    }

    /// Construct a REV at integer cell coordinates; `(0, 0, 0)` is centred at
    /// `(0.5 cm, 0.5 cm, 0.5 cm)`.
    pub fn at_cell(cell: [i32; 3], component: u16) -> Self {
        let cell = Vec3::new(cell[0] as f32, cell[1] as f32, cell[2] as f32);
        Self::new(cell * CELL + Vec3::splat(CELL * 0.5), component)
    }

    pub fn with_composition(mut self, composition: Composition) -> Self {
        self.composition = composition;
        self
    }

    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    pub fn fixed(mut self) -> Self {
        self.kinematic = true;
        self
    }

    pub fn at_temperature(mut self, kelvin: f32) -> Self {
        self.temperature_kelvin = kelvin;
        self
    }

    pub fn with_microstructure(mut self, microstructure: Microstructure) -> Self {
        self.microstructure = microstructure;
        self
    }
}

/// General scene authoring API for arbitrary voxel arrangements.
pub struct SceneBuilder {
    config: SceneConfig,
    voxels: Vec<VoxelSpawn>,
}

impl SceneBuilder {
    pub fn new(config: SceneConfig) -> Self {
        Self {
            config,
            voxels: Vec::new(),
        }
    }

    /// Start an authorable scene with engine defaults and no prebuilt objects.
    pub fn standard() -> Self {
        Self::new(SceneConfig::default())
    }

    pub fn add_voxel(mut self, voxel: VoxelSpawn) -> Self {
        self.voxels.push(voxel);
        self
    }

    pub fn add_component_voxel(self, center: Vec3, component: u16) -> Self {
        self.add_voxel(VoxelSpawn::new(center, component))
    }

    pub fn add_cell_voxel(self, cell: [i32; 3], component: u16) -> Self {
        self.add_voxel(VoxelSpawn::at_cell(cell, component))
    }

    pub fn add_named_voxel(self, center: Vec3, material: &str) -> Result<Self, String> {
        let component = Material::component_id(material)
            .ok_or_else(|| format!("unknown material: {material}"))?;
        Ok(self.add_component_voxel(center, component))
    }

    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.config.gravity = gravity.to_array();
        self
    }

    pub fn with_timestep(mut self, fixed_dt: f32, substeps: usize) -> Self {
        self.config.fixed_dt = fixed_dt;
        self.config.substeps = substeps;
        self
    }

    pub fn with_solver_iterations(mut self, iterations: usize) -> Self {
        self.config.iterations = iterations;
        self
    }

    /// Set the CUDA PCG iteration cap independently of the CPU solver cap.
    pub fn with_cuda_solver_iterations(mut self, iterations: usize) -> Self {
        self.config.cuda_iterations = iterations;
        self
    }

    /// Choose the CPU PCG preconditioner for the scene.
    pub fn with_cpu_preconditioner(mut self, preconditioner: CpuPreconditioner) -> Self {
        self.config.cpu_preconditioner = preconditioner;
        self
    }

    pub fn build(self) -> Result<Scene, String> {
        Scene::from_voxels(self.config, self.voxels)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    pub dimensions: [usize; 3],
    pub branch_dimensions: [usize; 3],
    pub ground_dimensions: [usize; 3],
    pub tool_dimensions: [usize; 3],
    pub fibers: [f32; 3],
    pub anisotropy: f32,
    pub gravity: [f32; 3],
    pub fixed_dt: f32,
    pub substeps: usize,
    pub iterations: usize,
    /// Maximum CUDA PCG iterations; CPU PCG keeps `iterations` as its cap.
    #[serde(default = "default_cuda_iterations")]
    pub cuda_iterations: usize,
    /// CPU PCG preconditioner; GPU currently uses its block-Jacobi kernel.
    #[serde(default)]
    pub cpu_preconditioner: CpuPreconditioner,
    pub solver_relative_tolerance: f32,
    pub corotation_tolerance: f32,
    pub corotation_iterations: usize,
    pub fracture: crate::fracture::FractureConfig,

    pub initial_speed: f32,
    pub tool_start: [f32; 3],
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            dimensions: [1, 1, 1],
            branch_dimensions: [0; 3],
            ground_dimensions: [1, 1, 1],
            tool_dimensions: [2; 3],
            fibers: [0.0, 0.0, 1.0],
            anisotropy: 0.0,
            gravity: [0.0, 0.0, -9.81],
            fixed_dt: 0.001,
            substeps: 4,
            iterations: 80,
            cuda_iterations: default_cuda_iterations(),
            cpu_preconditioner: CpuPreconditioner::default(),
            solver_relative_tolerance: 1e-5,
            corotation_tolerance: 1e-5,
            corotation_iterations: 2,
            fracture: Default::default(),
            initial_speed: 0.0,
            tool_start: [0.0; 3],
        }
    }
}

impl SceneConfig {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let result: Self = ron::from_str(&fs::read_to_string(path)?)?;
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        let result = self;
        if result.dimensions.contains(&0)
            || result.ground_dimensions.contains(&0)
            || result.tool_dimensions.iter().any(|&n| n < 2)
            || !(0.0..=1.0).contains(&result.anisotropy)
            || !Vec3::from_array(result.gravity).is_finite()
            || !Vec3::from_array(result.tool_start).is_finite()
            || !result.initial_speed.is_finite()
            || result.initial_speed < 0.0
            || !Vec3::from_array(result.fibers).is_finite()
            || !result.fixed_dt.is_finite()
            || result.fixed_dt <= 0.0
            || result.substeps == 0
            || result.iterations == 0
            || result.cuda_iterations == 0
            || result.corotation_iterations == 0
            || !result.solver_relative_tolerance.is_finite()
            || result.solver_relative_tolerance <= 0.0
            || !result.corotation_tolerance.is_finite()
            || result.corotation_tolerance <= 0.0
            || !result.fracture.validate()
            || Vec3::from_array(result.fibers).length_squared() < 1e-8
        {
            return Err("Invalid scene parameters");
        }
        Ok(())
    }
}

fn default_cuda_iterations() -> usize {
    160
}
/// Positions are separate arrays; material identity exists only in composition.
pub struct Scene {
    /// Capacity-reusing workspace retained between physical steps.
    pub(crate) solver_scratch: crate::solver::SolverScratch,
    /// Capacity-reusing thermal transport workspace retained between substeps.
    thermal_scratch: ThermalScratch,
    /// Capacity-reusing spatial data for broadphase collision detection.
    pub(crate) collision_scratch: crate::collision::CollisionScratch,
    pub config: SceneConfig,
    pub voxels: Vec<Voxel>,
    /// Persistent per-voxel forces in newtons, retained until explicitly reset.
    pub external_forces: Vec<Vec3>,
    /// Persistent per-voxel acceleration inputs in m/s².
    pub external_accelerations: Vec<Vec3>,
    /// Persistent world-coordinate constraints; None leaves an axis free.
    pub axis_targets: Vec<[Option<f32>; 3]>,
    pub positions: Vec<Vec3>,
    pub rest_positions: Vec<Vec3>,
    /// Stress-free temperature of each REV when the scene was created (K).
    pub(crate) reference_temperatures_kelvin: Vec<f32>,
    pub previous: Vec<Vec3>,
    pub bonds: [Vec<Bond>; 3],
    pub workspace: [Vec<BondWorkspace>; 3],
    pub colors: [Vec<usize>; 6],
    pub materials: Vec<Material>,
    pub properties: Vec<Material>,
    pub rotations: Vec<Quat>,
    /// Derived connectivity labels; rebuilt only when topology changes.
    pub clusters: Vec<u32>,
    pub broken: usize,
    pub cluster_count: usize,
    pub ground_top: f32,
    pub energy: crate::fracture::EnergyLedger,
    pub releases: Vec<crate::fracture::Release>,
    pub substep_index: u64,
    pub solver_residual: f32,
    world_bounds: Option<WorldBounds>,
    thermal_bonding_rule: Option<ThermalBondingRule>,
    bonding_contacts: HashMap<(usize, usize), BondingContactState>,
    thermal_fusion_rule: Option<ThermalFusionRule>,
    fusion_contacts: HashMap<(usize, usize), BondingContactState>,
    pending_fusion: Option<(usize, usize)>,
    boundary_removals: BoundaryRemovalLedger,
}
impl Scene {
    /// Advance one fixed physical step without a rigid tool.
    pub fn tick(&mut self) {
        crate::solver::step(self, &mut None);
    }

    /// Advance one fixed physical step with an optional rigid tool.
    pub fn tick_with_tool(&mut self, tool: &mut Option<crate::tool::Tool>) {
        crate::solver::step(self, tool);
    }

    /// Configure the pre-melt strength-softening onset for a built-in material.
    /// The ratio is relative to that material's melting temperature; 1 disables
    /// progressive softening. Existing REV derived properties are refreshed.
    pub fn set_material_strength_softening(
        &mut self,
        material_name: &str,
        onset_ratio: f32,
    ) -> Result<(), String> {
        if !onset_ratio.is_finite() || !(0.0..=1.0).contains(&onset_ratio) {
            return Err("strength softening onset ratio must be finite and in [0, 1]".into());
        }
        let id = Material::component_id(material_name)
            .ok_or_else(|| format!("unknown material: {material_name}"))? as usize;
        self.materials[id].strength_softening_start_ratio = onset_ratio;
        for (voxel, properties) in self.voxels.iter().zip(&mut self.properties) {
            *properties = Material::mixed(&voxel.composition, &self.materials);
        }
        Ok(())
    }

    /// Set a built-in material's liquid dynamic viscosity in pascal-seconds.
    /// Mixtures use a mass-fraction weighted estimate; current values are
    /// constant across temperature and are not material-calibrated curves.
    pub fn set_material_dynamic_viscosity(
        &mut self,
        material_name: &str,
        viscosity_pa_s: f32,
    ) -> Result<(), String> {
        if !viscosity_pa_s.is_finite() || viscosity_pa_s < 0.0 {
            return Err("dynamic viscosity must be finite and nonnegative".into());
        }
        let id = Material::component_id(material_name)
            .ok_or_else(|| format!("unknown material: {material_name}"))? as usize;
        self.materials[id].dynamic_viscosity_pa_s = viscosity_pa_s;
        for (voxel, properties) in self.voxels.iter().zip(&mut self.properties) {
            *properties = Material::mixed(&voxel.composition, &self.materials);
        }
        Ok(())
    }

    /// Set a built-in material's isotropic linear thermal expansion coefficient (1/K).
    /// The scene's stress-free reference temperatures remain the creation temperatures.
    pub fn set_material_thermal_expansion(
        &mut self,
        material_name: &str,
        coefficient_per_kelvin: f32,
    ) -> Result<(), String> {
        self.set_material_thermal_expansion_anisotropic(
            material_name,
            coefficient_per_kelvin,
            coefficient_per_kelvin,
        )
    }

    /// Set separate linear thermal expansion coefficients along and across the
    /// material's principal fiber axis, in 1/K. For isotropic materials use
    /// `set_material_thermal_expansion`.
    pub fn set_material_thermal_expansion_anisotropic(
        &mut self,
        material_name: &str,
        parallel_per_kelvin: f32,
        perpendicular_per_kelvin: f32,
    ) -> Result<(), String> {
        if !parallel_per_kelvin.is_finite()
            || parallel_per_kelvin < 0.0
            || !perpendicular_per_kelvin.is_finite()
            || perpendicular_per_kelvin < 0.0
        {
            return Err("thermal expansion coefficients must be finite and nonnegative".into());
        }
        let id = Material::component_id(material_name)
            .ok_or_else(|| format!("unknown material: {material_name}"))? as usize;
        self.materials[id].thermal_expansion_parallel = parallel_per_kelvin;
        self.materials[id].thermal_expansion_perpendicular = perpendicular_per_kelvin;
        for (voxel, properties) in self.voxels.iter().zip(&mut self.properties) {
            *properties = Material::mixed(&voxel.composition, &self.materials);
        }
        Ok(())
    }

    /// Create a scene from arbitrary REV states. Bond topology is derived from
    /// adjacent 1 cm lattice cells; no built-in object layout is inserted.
    pub fn from_voxels(config: SceneConfig, spawns: Vec<VoxelSpawn>) -> Result<Self, String> {
        config.validate().map_err(str::to_owned)?;
        let materials = Material::table();
        let mut scene = Self {
            config,
            solver_scratch: Default::default(),
            thermal_scratch: Default::default(),
            collision_scratch: Default::default(),
            voxels: Vec::with_capacity(spawns.len()),
            external_forces: Vec::with_capacity(spawns.len()),
            external_accelerations: Vec::with_capacity(spawns.len()),
            axis_targets: Vec::with_capacity(spawns.len()),
            positions: Vec::with_capacity(spawns.len()),
            rest_positions: Vec::with_capacity(spawns.len()),
            reference_temperatures_kelvin: Vec::with_capacity(spawns.len()),
            previous: Vec::with_capacity(spawns.len()),
            bonds: std::array::from_fn(|_| Vec::with_capacity(spawns.len())),
            workspace: std::array::from_fn(|_| Vec::with_capacity(spawns.len())),
            colors: std::array::from_fn(|_| Vec::new()),
            materials,
            properties: Vec::with_capacity(spawns.len()),
            rotations: Vec::with_capacity(spawns.len()),
            clusters: Vec::with_capacity(spawns.len()),
            broken: 0,
            cluster_count: 0,
            ground_top: 0.0,
            energy: Default::default(),
            releases: Vec::new(),
            substep_index: 0,
            solver_residual: 0.0,
            world_bounds: None,
            thermal_bonding_rule: None,
            bonding_contacts: HashMap::new(),
            thermal_fusion_rule: None,
            fusion_contacts: HashMap::new(),
            pending_fusion: None,
            boundary_removals: BoundaryRemovalLedger::default(),
        };
        let mut occupancy = HashMap::with_capacity(spawns.len());
        let mut keys = Vec::with_capacity(spawns.len());
        for (index, spawn) in spawns.into_iter().enumerate() {
            if !spawn.center.is_finite()
                || !spawn.velocity.is_finite()
                || !spawn.temperature_kelvin.is_finite()
                || spawn.temperature_kelvin < 0.0
                || !spawn.microstructure.orientation.is_finite()
                || spawn.microstructure.orientation.length_squared() < 1e-8
                || !spawn.microstructure.anisotropy.is_finite()
                || !(0.0..=1.0).contains(&spawn.microstructure.anisotropy)
            {
                return Err(format!("invalid initial state for voxel {index}"));
            }
            spawn
                .composition
                .validate(scene.materials.len())
                .map_err(|reason| format!("invalid composition for voxel {index}: {reason}"))?;
            let lattice = spawn.center / CELL - Vec3::splat(0.5);
            let rounded = lattice.round();
            if (lattice - rounded).abs().max_element() > 1e-3 {
                return Err(format!(
                    "voxel {index} centre must lie on the 1 cm REV lattice"
                ));
            }
            let key = [rounded.x as i32, rounded.y as i32, rounded.z as i32];
            if occupancy.insert(key, index).is_some() {
                return Err(format!(
                    "duplicate voxel lattice cell at {:?}",
                    spawn.center
                ));
            }
            let material = Material::mixed(&spawn.composition, &scene.materials);
            let mass = material.density * CELL.powi(3);
            if !mass.is_finite() || mass <= 0.0 {
                return Err(format!("voxel {index} has invalid derived mass"));
            }
            scene.positions.push(spawn.center);
            scene.previous.push(spawn.center);
            scene.rest_positions.push(spawn.center);
            scene
                .reference_temperatures_kelvin
                .push(spawn.temperature_kelvin);
            scene.voxels.push(Voxel {
                mass,
                kinematic: spawn.kinematic,
                composition: spawn.composition,
                internal_energy: material.energy_at_temperature(mass, spawn.temperature_kelvin),
                internal_energy_remainder_j: 0.0,
                velocity: spawn.velocity,
                microstructure: Microstructure {
                    orientation: spawn.microstructure.orientation.normalize(),
                    anisotropy: spawn.microstructure.anisotropy,
                },
            });
            scene.external_forces.push(Vec3::ZERO);
            scene.external_accelerations.push(Vec3::ZERO);
            scene.axis_targets.push([None; 3]);
            scene.properties.push(material);
            scene.rotations.push(Quat::IDENTITY);
            scene.clusters.push(0);
            keys.push(key);
        }

        let count = scene.voxels.len();
        for axis in 0..3 {
            scene.bonds[axis].resize(count, Bond::default());
            scene.workspace[axis].resize(
                count,
                BondWorkspace {
                    rest_neighbor: EMPTY,
                    ..Default::default()
                },
            );
        }
        for (i, key) in keys.iter().enumerate() {
            for axis in 0..3 {
                let mut neighbor = *key;
                neighbor[axis] += 1;
                if let Some(&j) = occupancy.get(&neighbor) {
                    if scene.voxels[i].kinematic == scene.voxels[j].kinematic {
                        scene.bonds[axis][i].neighbor = j as u32;
                        scene.workspace[axis][i].rest_neighbor = j as u32;
                        scene.colors[axis * 2 + key[axis].rem_euclid(2) as usize].push(i);
                    }
                }
            }
        }
        crate::solver::rebuild_clusters(&mut scene);
        Ok(scene)
    }

    /// Set world bounds. REV centres outside the inclusive box are removed now
    /// and after each subsequent physical step.
    pub fn set_world_bounds(&mut self, min: Vec3, max: Vec3) -> Result<usize, &'static str> {
        if !min.is_finite() || !max.is_finite() || min.cmpgt(max).any() {
            return Err("world bounds must be finite and min <= max on every axis");
        }
        self.world_bounds = Some(WorldBounds { min, max });
        Ok(self.enforce_world_bounds())
    }

    /// Remove world barriers. Removed REV are not restored.
    pub fn clear_world_bounds(&mut self) {
        self.world_bounds = None;
    }

    /// Enable thermal bonding for a material. Both endpoints must meet the
    /// component and temperature thresholds. A qualifying new impact arms a
    /// pair; normal impulse divided by substep duration and face area estimates
    /// contact pressure, which must remain above its threshold for the dwell
    /// duration. Set impact energy to zero for pressure/dwell-only bonding.
    pub fn set_thermal_bonding_rule(
        &mut self,
        material_name: &str,
        minimum_mass_fraction: f32,
        minimum_temperature_kelvin: f32,
        minimum_impact_energy_j: f64,
        minimum_contact_pressure_pa: f32,
        minimum_dwell_seconds: f32,
    ) -> Result<(), String> {
        if !minimum_mass_fraction.is_finite()
            || !(0.0..=1.0).contains(&minimum_mass_fraction)
            || minimum_mass_fraction == 0.0
            || !minimum_temperature_kelvin.is_finite()
            || minimum_temperature_kelvin < 0.0
            || !minimum_impact_energy_j.is_finite()
            || minimum_impact_energy_j < 0.0
            || !minimum_contact_pressure_pa.is_finite()
            || minimum_contact_pressure_pa < 0.0
            || !minimum_dwell_seconds.is_finite()
            || minimum_dwell_seconds < 0.0
        {
            return Err(
                "thermal bonding thresholds must be finite and nonnegative (fraction in (0, 1])"
                    .into(),
            );
        }
        let component = Material::component_id(material_name)
            .ok_or_else(|| format!("unknown material: {material_name}"))?;
        self.thermal_bonding_rule = Some(ThermalBondingRule {
            component,
            minimum_mass_fraction,
            minimum_temperature_kelvin,
            minimum_impact_energy_j,
            minimum_contact_pressure_pa,
            minimum_dwell_seconds,
        });
        self.bonding_contacts.clear();
        Ok(())
    }

    /// Disable automatic thermal/impact bond formation.
    pub fn clear_thermal_bonding_rule(&mut self) {
        self.thermal_bonding_rule = None;
        self.bonding_contacts.clear();
    }

    pub fn thermal_bonding_rule(&self) -> Option<ThermalBondingRule> {
        self.thermal_bonding_rule
    }

    /// Advance contact pressure and dwell state after the contact impulse solve.
    pub(crate) fn form_thermal_bonds(
        &mut self,
        contacts: &[crate::collision::Contact],
        impact_energies_j: &[f64],
        normal_impulses_ns: &[f32],
        elapsed_seconds: f32,
    ) -> usize {
        let Some(rule) = self.thermal_bonding_rule else {
            return 0;
        };
        let mut formed = 0;
        let current_substep = self.substep_index;
        for (index, contact) in contacts.iter().enumerate() {
            let i = contact.i;
            let j = contact.j;
            if i >= self.voxels.len() || j >= self.voxels.len() {
                continue;
            }
            let key = if i < j { (i, j) } else { (j, i) };
            if self.voxels[i].kinematic != self.voxels[j].kinematic
                || self
                    .bonds
                    .iter()
                    .any(|axis| axis[i].neighbor == j as u32 || axis[j].neighbor == i as u32)
            {
                self.bonding_contacts.remove(&key);
                continue;
            }
            let qualifies =
                |voxel: usize, scene: &Scene| {
                    scene.voxels[voxel].composition.components.iter().any(
                        |&(component, fraction)| {
                            component == rule.component && fraction >= rule.minimum_mass_fraction
                        },
                    ) && scene.properties[voxel]
                        .phase_state(
                            scene.voxels[voxel].mass,
                            scene.voxels[voxel].thermal_energy_f32(),
                        )
                        .temperature_kelvin
                        >= rule.minimum_temperature_kelvin
                };
            if !qualifies(i, self) || !qualifies(j, self) {
                self.bonding_contacts.remove(&key);
                continue;
            }
            let impact_energy_j = impact_energies_j.get(index).copied().unwrap_or(0.0);
            self.bonding_contacts
                .entry(key)
                .or_default()
                .last_seen_substep = current_substep;
            if rule.minimum_impact_energy_j == 0.0 {
                self.bonding_contacts
                    .entry(key)
                    .or_default()
                    .impact_qualified = true;
            }
            if contact.new_impact && impact_energy_j >= rule.minimum_impact_energy_j {
                self.bonding_contacts
                    .entry(key)
                    .or_default()
                    .impact_qualified = true;
            }
            let pressure_pa = normal_impulses_ns.get(index).copied().unwrap_or(0.0)
                / (elapsed_seconds.max(f32::MIN_POSITIVE) * CELL * CELL);
            let state = self.bonding_contacts.entry(key).or_default();
            if pressure_pa >= rule.minimum_contact_pressure_pa {
                state.dwell_seconds += elapsed_seconds;
            } else {
                state.dwell_seconds = 0.0;
            }
            let ready = state.impact_qualified
                && state.dwell_seconds + f32::EPSILON >= rule.minimum_dwell_seconds;
            if ready {
                let connected = self.connect_adjacent_voxels(i, j).is_ok();
                self.bonding_contacts.remove(&key);
                formed += usize::from(connected);
            }
        }
        self.bonding_contacts
            .retain(|_, state| state.last_seen_substep == current_substep);
        formed
    }

    pub fn world_bounds(&self) -> Option<WorldBounds> {
        self.world_bounds
    }

    pub fn boundary_removals(&self) -> BoundaryRemovalLedger {
        self.boundary_removals
    }

    /// Fix or release a REV through the engine API. A transition clears its
    /// velocity to avoid injecting a positional impulse into the solver.
    pub fn set_voxel_kinematic(&mut self, voxel: usize, fixed: bool) -> Result<(), &'static str> {
        self.set_voxels_kinematic(&[voxel], fixed)
    }

    /// Fix or release several REVs as one scene mutation. Validates every
    /// index before changing state, then invalidates dependent caches once.
    pub fn set_voxels_kinematic(
        &mut self,
        voxels: &[usize],
        fixed: bool,
    ) -> Result<(), &'static str> {
        if voxels.iter().any(|&voxel| voxel >= self.voxels.len()) {
            return Err("voxel index out of range");
        }
        let changed: Vec<_> = voxels
            .iter()
            .copied()
            .filter(|&voxel| self.voxels[voxel].kinematic != fixed)
            .collect();
        if changed.is_empty() {
            return Ok(());
        }
        for voxel in changed {
            self.voxels[voxel].kinematic = fixed;
            self.voxels[voxel].velocity = Vec3::ZERO;
            self.previous[voxel] = self.positions[voxel];
        }
        self.solver_scratch = Default::default();
        self.collision_scratch = Default::default();
        crate::solver::rebuild_clusters(self);
        Ok(())
    }

    pub fn fix_voxels(&mut self, voxels: &[usize]) -> Result<(), &'static str> {
        self.set_voxels_kinematic(voxels, true)
    }

    pub fn release_voxels(&mut self, voxels: &[usize]) -> Result<(), &'static str> {
        self.set_voxels_kinematic(voxels, false)
    }

    pub fn fix_voxel(&mut self, voxel: usize) -> Result<(), &'static str> {
        self.set_voxel_kinematic(voxel, true)
    }

    pub fn release_voxel(&mut self, voxel: usize) -> Result<(), &'static str> {
        self.set_voxel_kinematic(voxel, false)
    }

    /// Form an intact, initially stress-free bond between two adjacent REVs.
    ///
    /// This is the engine topology operation used by adhesion, welding, or
    /// healing systems. The caller decides when those material processes
    /// occur; the bond thereafter uses the ordinary constitutive and fracture
    /// rules. The pair must be in face contact within 10 micrometres and must
    /// have no queued fracture-energy release. The bond records the local
    /// stress-free offset at formation, so temperature, fiber orientation, and
    /// prior deformation do not create an instantaneous spring impulse.
    /// Adjacency is measured from the
    /// original REV lattice positions, so deformation does not change which
    /// pair this API connects.
    pub fn connect_adjacent_voxels(&mut self, a: usize, b: usize) -> Result<(), &'static str> {
        if a >= self.voxels.len() || b >= self.voxels.len() {
            return Err("voxel index out of range");
        }
        if a == b {
            return Err("cannot connect a voxel to itself");
        }
        if self.voxels[a].kinematic != self.voxels[b].kinematic {
            return Err("bond endpoints must have matching kinematic constraints");
        }

        let lattice_delta = (self.rest_positions[b] - self.rest_positions[a]) / CELL;
        let rounded = lattice_delta.round();
        if (lattice_delta - rounded).abs().max_element() > 1e-3
            || rounded.abs().element_sum() != 1.0
        {
            return Err("bond endpoints must be adjacent on the REV lattice");
        }
        let axis = (0..3)
            .find(|&axis| rounded[axis].abs() == 1.0)
            .ok_or("bond endpoints must be adjacent on the REV lattice")?;
        let (lower, upper) = if rounded[axis] > 0.0 { (a, b) } else { (b, a) };
        let existing_neighbor = self.bonds[axis][lower].neighbor;
        if existing_neighbor != EMPTY {
            return if existing_neighbor as usize == upper {
                Ok(())
            } else {
                Err("bond slot already connects a different voxel")
            };
        }
        if self.releases.iter().any(|release| {
            (release.i == lower && release.j == upper) || (release.i == upper && release.j == lower)
        }) {
            return Err("cannot connect a voxel pair with a pending fracture release");
        }
        let actual = self.positions[upper] - self.positions[lower];
        if !actual.is_finite() || (actual.length() - CELL).abs() > CELL * 1e-3 {
            return Err("voxel pair must be in face contact to form a bond");
        }
        let formation_offset = crate::solver::bond_formation_offset(self, lower, upper, axis);
        if formation_offset.iter().any(|value| !value.is_finite()) {
            return Err("bond formation offset is not finite");
        }
        self.bonds[axis][lower] = Bond {
            neighbor: upper as u32,
            formation_offset,
            ..Bond::default()
        };
        self.workspace[axis][lower] = BondWorkspace {
            rest_neighbor: upper as u32,
            ..BondWorkspace::default()
        };
        self.bonding_contacts
            .remove(&(lower.min(upper), lower.max(upper)));
        let cell_coordinate = (self.rest_positions[lower][axis] / CELL - 0.5).round() as i32;
        let color = axis * 2 + cell_coordinate.rem_euclid(2) as usize;
        if !self.colors[color].contains(&lower) {
            self.colors[color].push(lower);
        }
        self.solver_scratch = Default::default();
        self.thermal_scratch = Default::default();
        self.collision_scratch = Default::default();
        crate::solver::rebuild_clusters(self);
        Ok(())
    }

    /// Fuse two free, face-adjacent REVs into one compressed representative
    /// volume. Mass-weighted composition, linear momentum, and total energy are
    /// conserved; relative kinetic energy becomes internal heat. The fused REV
    /// retains one cell's geometry, so this operation models local compaction,
    /// not expansion into a two-cell-shaped body. Both inputs must be free of
    /// bonds, pending fracture releases, and kinematic constraints.
    pub fn fuse_adjacent_voxels(&mut self, a: usize, b: usize) -> Result<usize, &'static str> {
        if a >= self.voxels.len() || b >= self.voxels.len() {
            return Err("voxel index out of range");
        }
        if a == b {
            return Err("cannot fuse a voxel with itself");
        }
        if self.voxels[a].kinematic || self.voxels[b].kinematic {
            return Err("fusion requires unconstrained voxels");
        }
        if self.axis_targets[a].iter().any(Option::is_some)
            || self.axis_targets[b].iter().any(Option::is_some)
        {
            return Err("fusion requires unconstrained voxels");
        }
        let lattice_delta = (self.rest_positions[b] - self.rest_positions[a]) / CELL;
        let rounded = lattice_delta.round();
        if (lattice_delta - rounded).abs().max_element() > 1e-3
            || rounded.abs().element_sum() != 1.0
        {
            return Err("fused voxels must be adjacent on the REV lattice");
        }
        let face_normal = rounded.normalize_or_zero();
        let separation = self.positions[b] - self.positions[a];
        let normal_gap = separation.dot(face_normal).abs();
        let tangential_offset = separation - face_normal * separation.dot(face_normal);
        if (normal_gap - CELL).abs() > CELL * 1e-3 || tangential_offset.abs().max_element() >= CELL
        {
            return Err("fused voxels must be in face contact");
        }
        let incident_bond = |voxel: usize| {
            self.bonds.iter().any(|axis| {
                axis.iter().enumerate().any(|(i, bond)| {
                    bond.neighbor != EMPTY && (i == voxel || bond.neighbor as usize == voxel)
                })
            })
        };
        if incident_bond(a) || incident_bond(b) {
            return Err("fusion currently requires unbonded input voxels");
        }
        if self
            .releases
            .iter()
            .any(|release| release.i == a || release.j == a || release.i == b || release.j == b)
        {
            return Err("cannot fuse voxels with pending fracture releases");
        }

        let (survivor, consumed) = if a < b { (a, b) } else { (b, a) };
        let mass_a = self.voxels[survivor].mass as f64;
        let mass_b = self.voxels[consumed].mass as f64;
        let mass_total = mass_a + mass_b;
        let merged_mass = mass_total as f32;
        if !merged_mass.is_finite() || merged_mass <= 0.0 {
            return Err("fused mass is outside the supported range");
        }
        let va = self.voxels[survivor].velocity;
        let vb = self.voxels[consumed].velocity;
        let kinetic_before =
            0.5 * mass_a * va.length_squared() as f64 + 0.5 * mass_b * vb.length_squared() as f64;
        let position = (self.positions[survivor] * mass_a as f32
            + self.positions[consumed] * mass_b as f32)
            / mass_total as f32;
        let velocity = (va * mass_a as f32 + vb * mass_b as f32) / mass_total as f32;
        let kinetic_after = 0.5 * merged_mass as f64 * velocity.length_squared() as f64;
        let fusion_heat_j = (kinetic_before - kinetic_after).max(0.0);

        let mut fractions = std::collections::BTreeMap::<u16, f64>::new();
        for &(component, fraction) in &self.voxels[survivor].composition.components {
            *fractions.entry(component).or_default() += mass_a * fraction as f64;
        }
        for &(component, fraction) in &self.voxels[consumed].composition.components {
            *fractions.entry(component).or_default() += mass_b * fraction as f64;
        }
        let mut composition = Composition {
            components: Default::default(),
        };
        for (component, component_mass) in fractions {
            composition
                .components
                .push((component, (component_mass / mass_total) as f32));
        }
        composition
            .validate(self.materials.len())
            .map_err(|_| "merged composition is invalid")?;

        let specific_b = mass_b / mass_total;
        let internal_b = self.voxels[consumed].total_internal_energy_j();
        let reference_temperature = ((mass_a * self.reference_temperatures_kelvin[survivor] as f64
            + mass_b * self.reference_temperatures_kelvin[consumed] as f64)
            / mass_total) as f32;
        let merged_microstructure = Microstructure {
            orientation: self.voxels[survivor]
                .microstructure
                .orientation
                .slerp(
                    self.voxels[consumed].microstructure.orientation,
                    specific_b as f32,
                )
                .normalize(),
            anisotropy: ((mass_a * self.voxels[survivor].microstructure.anisotropy as f64
                + mass_b * self.voxels[consumed].microstructure.anisotropy as f64)
                / mass_total) as f32,
        };
        let merged_material = Material::mixed(&composition, &self.materials);

        {
            let merged = &mut self.voxels[survivor];
            merged.mass = merged_mass;
            merged.composition = composition;
            merged.velocity = velocity;
            merged.microstructure = merged_microstructure;
            merged.add_internal_energy_j(internal_b + kinetic_before - kinetic_after);
        }
        self.positions[survivor] = position;
        self.previous[survivor] = position;
        self.rest_positions[survivor] = position;
        self.reference_temperatures_kelvin[survivor] = reference_temperature;
        let merged_force = self.external_forces[survivor] + self.external_forces[consumed];
        self.external_forces[survivor] = merged_force;
        self.external_accelerations[survivor] = (self.external_accelerations[survivor]
            * mass_a as f32
            + self.external_accelerations[consumed] * mass_b as f32)
            / mass_total as f32;
        self.properties[survivor] = merged_material;
        self.energy.fusion_heat += fusion_heat_j;
        self.rotations[survivor] = self.rotations[survivor]
            .slerp(self.rotations[consumed], specific_b as f32)
            .normalize();

        let old_count = self.voxels.len();
        let keep: Vec<bool> = (0..old_count).map(|i| i != consumed).collect();
        let mut remap = vec![EMPTY; old_count];
        let mut next = 0u32;
        for i in 0..old_count {
            if keep[i] {
                remap[i] = next;
                next += 1;
            }
        }
        for release in &mut self.releases {
            release.i = remap[release.i] as usize;
            release.j = remap[release.j] as usize;
        }
        for axis in 0..3 {
            let old_bonds = std::mem::take(&mut self.bonds[axis]);
            let old_workspace = std::mem::take(&mut self.workspace[axis]);
            for i in 0..old_count {
                if !keep[i] {
                    continue;
                }
                let mut bond = old_bonds[i];
                let mut workspace = old_workspace[i];
                if bond.neighbor != EMPTY {
                    bond.neighbor = remap[bond.neighbor as usize];
                }
                if workspace.rest_neighbor != EMPTY {
                    workspace.rest_neighbor = remap[workspace.rest_neighbor as usize];
                }
                self.bonds[axis].push(bond);
                self.workspace[axis].push(workspace);
            }
        }
        self.bonding_contacts = std::mem::take(&mut self.bonding_contacts)
            .into_iter()
            .filter_map(|((i, j), state)| {
                (keep[i] && keep[j]).then_some(((remap[i] as usize, remap[j] as usize), state))
            })
            .collect();
        for color in &mut self.colors {
            color.retain(|&i| keep[i]);
            for i in color {
                *i = remap[*i] as usize;
            }
        }
        macro_rules! compact {
            ($field:ident) => {{
                self.$field = std::mem::take(&mut self.$field)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, value)| keep[i].then_some(value))
                    .collect();
            }};
        }
        compact!(voxels);
        compact!(positions);
        compact!(rest_positions);
        compact!(reference_temperatures_kelvin);
        compact!(previous);
        compact!(external_forces);
        compact!(external_accelerations);
        compact!(axis_targets);
        compact!(properties);
        compact!(rotations);
        compact!(clusters);
        self.solver_scratch = Default::default();
        self.thermal_scratch = Default::default();
        self.collision_scratch = Default::default();
        crate::solver::rebuild_clusters(self);
        Ok(remap[survivor] as usize)
    }

    /// Compatibility API for partial world-space constraints.
    pub fn fix_voxel_axes(
        &mut self,
        voxel: usize,
        coordinates: [Option<f32>; 3],
    ) -> Result<(), &'static str> {
        crate::solver::fix_voxel_axes(self, voxel, coordinates)
    }

    /// Compact all per-REV arrays and reindex topology after boundary crossings.
    /// Returns the number of removed REV.
    pub fn enforce_world_bounds(&mut self) -> usize {
        let Some(bounds) = self.world_bounds else {
            return 0;
        };
        let old_count = self.voxels.len();
        let keep: Vec<bool> = self
            .positions
            .iter()
            .map(|p| p.cmpge(bounds.min).all() && p.cmple(bounds.max).all())
            .collect();
        let removed_count = keep.iter().filter(|&&kept| !kept).count();
        if removed_count == 0 {
            return 0;
        }

        let mut remap = vec![EMPTY; old_count];
        let mut next = 0u32;
        for (old, &kept) in keep.iter().enumerate() {
            if kept {
                remap[old] = next;
                next += 1;
            }
        }

        self.bonding_contacts = std::mem::take(&mut self.bonding_contacts)
            .into_iter()
            .filter_map(|((i, j), state)| {
                (keep[i] && keep[j]).then_some(((remap[i] as usize, remap[j] as usize), state))
            })
            .collect();

        for i in 0..old_count {
            if keep[i] {
                continue;
            }
            let voxel = &self.voxels[i];
            self.boundary_removals.voxel_count += 1;
            self.boundary_removals.mass_kg += voxel.mass as f64;
            self.boundary_removals.internal_energy_j += voxel.total_internal_energy_j();
            self.boundary_removals.kinetic_energy_j +=
                0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64;
            self.boundary_removals.momentum_kg_m_s[0] +=
                voxel.mass as f64 * voxel.velocity.x as f64;
            self.boundary_removals.momentum_kg_m_s[1] +=
                voxel.mass as f64 * voxel.velocity.y as f64;
            self.boundary_removals.momentum_kg_m_s[2] +=
                voxel.mass as f64 * voxel.velocity.z as f64;
        }

        let mut kept_releases = Vec::with_capacity(self.releases.len());
        for mut release in self.releases.drain(..) {
            if keep[release.i] && keep[release.j] {
                release.i = remap[release.i] as usize;
                release.j = remap[release.j] as usize;
                kept_releases.push(release);
            } else {
                self.boundary_removals.pending_bond_energy_j += release.energy;
                self.energy.pending_release -= release.energy;
            }
        }
        self.releases = kept_releases;

        for axis in 0..3 {
            let old_bonds = std::mem::take(&mut self.bonds[axis]);
            let old_workspace = std::mem::take(&mut self.workspace[axis]);
            let mut bonds = Vec::with_capacity(old_count - removed_count);
            let mut workspace = Vec::with_capacity(old_count - removed_count);
            for i in 0..old_count {
                if !keep[i] {
                    continue;
                }
                let mut bond = old_bonds[i];
                let mut work = old_workspace[i];
                if bond.neighbor != EMPTY {
                    let neighbor = remap[bond.neighbor as usize];
                    bond.neighbor = neighbor;
                }
                if work.rest_neighbor != EMPTY {
                    work.rest_neighbor = remap[work.rest_neighbor as usize];
                }
                bonds.push(bond);
                workspace.push(work);
            }
            self.bonds[axis] = bonds;
            self.workspace[axis] = workspace;
        }

        let mut old_index = 0;
        self.voxels.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.positions.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.rest_positions.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.reference_temperatures_kelvin.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.previous.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.external_forces.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.external_accelerations.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.axis_targets.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.properties.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.rotations.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.clusters.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        for color in &mut self.colors {
            color.retain(|index| keep[*index]);
            for index in color {
                *index = remap[*index] as usize;
            }
        }

        self.solver_scratch = Default::default();
        self.collision_scratch = Default::default();
        crate::solver::rebuild_clusters(self);
        removed_count
    }

    /// Physical mass of every REV, including kinematic boundary material.
    pub fn total_mass_kg(&self) -> f64 {
        self.voxels.iter().map(|voxel| voxel.mass as f64).sum()
    }

    /// Temperature derived from mass, internal energy, and current composition.
    pub fn temperature_kelvin(&self, voxel: usize) -> Option<f32> {
        let state = self.voxels.get(voxel)?;
        let material = Material::mixed(&state.composition, &self.materials);
        Some(material.temperature(state.mass, state.thermal_energy_f32()))
    }

    /// Phase fractions derived from conserved mass, energy, and composition.
    pub fn phase_state(&self, voxel: usize) -> Option<PhaseState> {
        let state = self.voxels.get(voxel)?;
        let material = Material::mixed(&state.composition, &self.materials);
        Some(material.phase_state(state.mass, state.thermal_energy_f32()))
    }

    /// Add heat to a REV; temperature and phase fractions update as derived state.
    pub fn add_heat_joules(&mut self, voxel: usize, joules: f32) -> bool {
        if !joules.is_finite() || self.voxels.get(voxel).is_none() {
            return false;
        }
        self.voxels[voxel].add_internal_energy_j(joules as f64);
        true
    }

    /// Conduct heat across intact voxel bonds. Pairwise fluxes are equal and
    /// opposite, and automatic subcycling keeps the explicit update stable.
    /// The returned value is the number of thermal substeps used.
    pub fn conduct_heat(&mut self, elapsed_seconds: f32) -> Result<usize, &'static str> {
        if !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
            return Err("thermal elapsed time must be finite and nonnegative");
        }
        if elapsed_seconds == 0.0 || self.voxels.len() < 2 {
            return Ok(0);
        }

        let voxel_count = self.voxels.len();
        let scratch = &mut self.thermal_scratch;
        scratch.edges.clear();
        scratch.conductance_sum.resize(voxel_count, 0.0);
        scratch.conductance_sum.fill(0.0);
        scratch.temperature.resize(voxel_count, 0.0);
        scratch.energy_delta.resize(voxel_count, 0.0);
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY {
                    continue;
                }
                let j = bond.neighbor as usize;
                let ki = self.properties[i].thermal_cond.max(0.0) as f64;
                let kj = self.properties[j].thermal_cond.max(0.0) as f64;
                if ki == 0.0 || kj == 0.0 {
                    continue;
                }
                let harmonic_k = 2.0 * ki * kj / (ki + kj);
                let conductance =
                    harmonic_k * CELL as f64 * (1.0 - bond.damage as f64).clamp(0.0, 1.0);
                if conductance == 0.0 {
                    continue;
                }
                scratch.edges.push((i, j, conductance));
                scratch.conductance_sum[i] += conductance;
                scratch.conductance_sum[j] += conductance;
            }
        }
        if scratch.edges.is_empty() {
            return Ok(0);
        }

        let maximum_rate = self
            .voxels
            .iter()
            .zip(&self.properties)
            .zip(&scratch.conductance_sum)
            .map(|((voxel, material), conductance)| {
                conductance
                    / (voxel.mass.max(1e-20) as f64 * material.heat_capacity.max(1e-12) as f64)
            })
            .fold(0.0f64, f64::max);
        let thermal_steps = ((elapsed_seconds as f64 * maximum_rate / 0.45).ceil() as usize).max(1);
        if thermal_steps > 100_000 {
            return Err("thermal update requires more than 100000 stability substeps");
        }
        let dt = elapsed_seconds as f64 / thermal_steps as f64;
        for _ in 0..thermal_steps {
            for i in 0..self.voxels.len() {
                scratch.temperature[i] = self.properties[i]
                    .temperature(self.voxels[i].mass, self.voxels[i].thermal_energy_f32())
                    as f64;
            }
            scratch.energy_delta.fill(0.0);
            for &(i, j, conductance) in &scratch.edges {
                let heat = conductance * (scratch.temperature[j] - scratch.temperature[i]) * dt;
                scratch.energy_delta[i] += heat;
                scratch.energy_delta[j] -= heat;
            }
            for (voxel, delta) in self.voxels.iter_mut().zip(&scratch.energy_delta) {
                voxel.add_internal_energy_j(*delta);
            }
        }
        Ok(thermal_steps)
    }

    /// Derived display data for all incident bonds, including the negative-axis
    /// endpoint. Original outside faces are not damage. No visual state is stored.
    pub fn incident_bond_damage(&self) -> Vec<(f32, u8)> {
        let mut result = vec![(0.0f32, 0u8); self.voxels.len()];
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                let original = self.workspace[axis][i].rest_neighbor;
                if original == EMPTY {
                    continue;
                }
                let j = original as usize;
                let broken = bond.neighbor == EMPTY;
                let damage = if broken { 1.0 } else { bond.damage };
                for n in [i, j] {
                    result[n].0 = result[n].0.max(damage);
                    result[n].1 += u8::from(broken);
                }
            }
        }
        result
    }

    pub fn new(config: SceneConfig) -> Self {
        let materials = Material::table();
        let mut voxels = Vec::new();
        let mut positions = Vec::new();
        let mut keys = Vec::new();
        let mut occupancy = HashMap::new();
        let fibers = Vec3::from_array(config.fibers).normalize();
        let orientation = Quat::from_rotation_arc(Vec3::Z, fibers);
        let mut add =
            |key: [i32; 3], component: u16, anisotropy: f32, fixed: bool, orientation: Quat| {
                if occupancy.contains_key(&key) {
                    return;
                }
                let m = &materials[component as usize];
                let mass = m.density * CELL.powi(3);
                occupancy.insert(key, voxels.len());
                keys.push(key);
                positions.push(
                    Vec3::new(key[0] as f32, key[1] as f32, key[2] as f32) * CELL
                        + Vec3::splat(CELL * 0.5),
                );
                voxels.push(Voxel {
                    mass,
                    kinematic: fixed,
                    composition: Composition::pure(component),
                    internal_energy: m.energy_at_temperature(mass, 293.15),
                    internal_energy_remainder_j: 0.0,
                    velocity: Vec3::ZERO,
                    microstructure: Microstructure {
                        orientation,
                        anisotropy,
                    },
                });
            };
        let [nx, ny, nz] = config.dimensions;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    add(
                        [x as i32 - nx as i32 / 2, y as i32 - ny as i32 / 2, z as i32],
                        1,
                        config.anisotropy,
                        false,
                        orientation,
                    );
                }
            }
        }
        let [bx, by, bz] = config.branch_dimensions;
        for z in 0..bz {
            for y in 0..by {
                for x in 0..bx {
                    add(
                        [
                            nx as i32 / 2 + x as i32,
                            y as i32 - by as i32 / 2,
                            (nz * 2 / 3 + z) as i32,
                        ],
                        1,
                        config.anisotropy,
                        false,
                        Quat::from_rotation_arc(Vec3::Z, Vec3::X),
                    );
                }
            }
        }
        let [gx, gy, gz] = config.ground_dimensions;
        for z in 0..gz {
            for y in 0..gy {
                for x in 0..gx {
                    add(
                        [
                            x as i32 - gx as i32 / 2,
                            y as i32 - gy as i32 / 2,
                            -(z as i32) - 1,
                        ],
                        3,
                        0.0,
                        true,
                        Quat::IDENTITY,
                    );
                }
            }
        }
        let n = voxels.len();
        let mut bonds: [Vec<Bond>; 3] = std::array::from_fn(|_| vec![Bond::default(); n]);
        let mut workspace: [Vec<BondWorkspace>; 3] = std::array::from_fn(|_| {
            vec![
                BondWorkspace {
                    rest_neighbor: EMPTY,
                    ..Default::default()
                };
                n
            ]
        });
        let mut colors: [Vec<usize>; 6] = std::array::from_fn(|_| Vec::new());
        for (i, key) in keys.iter().enumerate() {
            for axis in 0..3 {
                let mut neighbor = *key;
                neighbor[axis] += 1;
                if let Some(&j) = occupancy.get(&neighbor) {
                    // Ground supports by contact; do not glue different bodies together.
                    if voxels[i].kinematic == voxels[j].kinematic {
                        bonds[axis][i].neighbor = j as u32;
                        workspace[axis][i].rest_neighbor = j as u32;
                        colors[axis * 2 + key[axis].rem_euclid(2) as usize].push(i);
                    }
                }
            }
        }
        let properties = voxels
            .iter()
            .map(|v| Material::mixed(&v.composition, &materials))
            .collect();
        Self {
            config,
            solver_scratch: Default::default(),
            thermal_scratch: Default::default(),
            collision_scratch: Default::default(),
            external_forces: vec![Vec3::ZERO; n],
            external_accelerations: vec![Vec3::ZERO; n],
            axis_targets: vec![[None; 3]; n],
            previous: positions.clone(),
            rest_positions: positions.clone(),
            reference_temperatures_kelvin: vec![293.15; n],
            positions,
            voxels,
            bonds,
            workspace,
            colors,
            materials,
            properties,
            rotations: vec![Quat::IDENTITY; n],
            clusters: vec![0; n],
            broken: 0,
            cluster_count: 1,
            ground_top: 0.0,
            energy: Default::default(),
            releases: Vec::new(),
            substep_index: 0,
            solver_residual: 0.0,
            world_bounds: None,
            thermal_bonding_rule: None,
            bonding_contacts: HashMap::new(),
            boundary_removals: BoundaryRemovalLedger::default(),
        }
    }
}

#[cfg(test)]
mod damage_display_tests {
    use super::*;

    #[test]
    fn original_boundary_is_intact_and_both_crack_endpoints_show_damage() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        let mut scene = Scene::new(config);
        assert_eq!(scene.incident_bond_damage(), vec![(0.0, 0); 2]);
        scene.bonds[0][0].damage = 0.4;
        assert_eq!(scene.incident_bond_damage(), vec![(0.4, 0); 2]);
        scene.bonds[0][0].neighbor = EMPTY;
        assert_eq!(scene.incident_bond_damage(), vec![(1.0, 1); 2]);
    }
}

#[cfg(test)]
mod world_bounds_tests {
    use super::*;

    #[test]
    fn boundary_removal_compacts_bonds_events_and_all_voxel_arrays() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = Scene::from_voxels(
            SceneConfig::default(),
            vec![
                VoxelSpawn::at_cell([0, 0, 0], iron)
                    .with_velocity(Vec3::new(-1.0, 2.0, 3.0))
                    .at_temperature(300.0),
                VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(310.0),
                VoxelSpawn::at_cell([2, 0, 0], iron).at_temperature(320.0),
            ],
        )
        .unwrap();
        assert_eq!(scene.bonds[0][0].neighbor, 1);
        assert_eq!(scene.bonds[0][1].neighbor, 2);

        scene.releases = vec![
            crate::fracture::Release {
                i: 0,
                j: 1,
                energy: 0.25,
                due: 7,
            },
            crate::fracture::Release {
                i: 1,
                j: 2,
                energy: 0.75,
                due: 9,
            },
        ];
        scene.energy.pending_release = 1.0;
        let removed = &scene.voxels[0];
        let expected_mass = removed.mass as f64;
        let expected_internal = removed.internal_energy as f64;
        let expected_kinetic = 0.5 * removed.mass as f64 * removed.velocity.length_squared() as f64;
        let expected_momentum = [
            removed.mass as f64 * removed.velocity.x as f64,
            removed.mass as f64 * removed.velocity.y as f64,
            removed.mass as f64 * removed.velocity.z as f64,
        ];

        let removed_count = scene
            .set_world_bounds(Vec3::new(0.01, 0.0, 0.0), Vec3::new(0.03, 0.01, 0.01))
            .unwrap();

        assert_eq!(removed_count, 1);
        assert_eq!(scene.voxels.len(), 2);
        assert!((scene.positions[0] - VoxelSpawn::at_cell([1, 0, 0], iron).center).length() < 1e-7);
        assert!((scene.positions[1] - VoxelSpawn::at_cell([2, 0, 0], iron).center).length() < 1e-7);
        assert_eq!(scene.bonds[0].len(), 2);
        assert_eq!(scene.workspace[0].len(), 2);
        assert_eq!(scene.bonds[0][0].neighbor, 1);
        assert_eq!(scene.reference_temperatures_kelvin, vec![310.0, 320.0]);
        assert_eq!(scene.workspace[0][0].rest_neighbor, 1);
        assert_eq!(scene.bonds[0][1].neighbor, EMPTY);
        for axis in 0..3 {
            assert_eq!(scene.bonds[axis].len(), 2);
            assert_eq!(scene.workspace[axis].len(), 2);
        }
        assert_eq!(scene.voxels.len(), scene.previous.len());
        assert_eq!(scene.voxels.len(), scene.rest_positions.len());
        assert_eq!(scene.voxels.len(), scene.external_forces.len());
        assert_eq!(scene.voxels.len(), scene.external_accelerations.len());
        assert_eq!(scene.voxels.len(), scene.axis_targets.len());
        assert_eq!(scene.voxels.len(), scene.properties.len());
        assert_eq!(scene.voxels.len(), scene.rotations.len());
        assert_eq!(scene.voxels.len(), scene.clusters.len());
        assert!(scene
            .colors
            .iter()
            .flatten()
            .all(|&index| index < scene.voxels.len()));
        assert_eq!(scene.cluster_count, 1);
        assert_eq!(scene.releases.len(), 1);
        assert_eq!(scene.releases[0].i, 0);
        assert_eq!(scene.releases[0].j, 1);
        assert_eq!(scene.releases[0].due, 9);
        assert_eq!(scene.energy.pending_release, 0.75);

        let ledger = scene.boundary_removals();
        assert_eq!(ledger.voxel_count, 1);
        assert!((ledger.mass_kg - expected_mass).abs() < 1e-12);
        assert!((ledger.internal_energy_j - expected_internal).abs() < 1e-9);
        assert!((ledger.kinetic_energy_j - expected_kinetic).abs() < 1e-12);
        assert_eq!(ledger.momentum_kg_m_s, expected_momentum);
        assert_eq!(ledger.pending_bond_energy_j, 0.25);
    }

    #[test]
    fn boundary_compaction_reindexes_bonding_dwell_pairs() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([2, 0, 0], iron))
            .build()
            .unwrap();
        scene.bonding_contacts.insert(
            (1, 2),
            BondingContactState {
                dwell_seconds: 0.25,
                impact_qualified: true,
                last_seen_substep: 7,
            },
        );
        assert_eq!(
            scene.set_world_bounds(Vec3::new(0.01, 0.0, 0.0), Vec3::splat(1.0)),
            Ok(1)
        );
        assert_eq!(scene.bonding_contacts.len(), 1);
        assert_eq!(scene.bonding_contacts[&(0, 1)].dwell_seconds, 0.25);
        assert!(scene
            .set_world_bounds(Vec3::new(0.02, 0.0, 0.0), Vec3::splat(1.0))
            .is_ok());
        assert!(scene.bonding_contacts.is_empty());
    }
}

#[cfg(test)]
mod thermal_transport_tests {
    use super::*;

    #[test]
    fn connected_hot_and_cold_revs_exchange_heat_conservatively() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).at_temperature(450.0))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(300.0))
            .build()
            .unwrap();
        assert_eq!(scene.bonds[0][0].neighbor, 1);

        let initial_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| voxel.internal_energy as f64)
            .sum();
        let initial_hot = scene.temperature_kelvin(0).unwrap();
        let initial_cold = scene.temperature_kelvin(1).unwrap();

        let thermal_substeps = scene.conduct_heat(1.0).unwrap();

        let final_hot = scene.temperature_kelvin(0).unwrap();
        let final_cold = scene.temperature_kelvin(1).unwrap();
        let final_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| voxel.internal_energy as f64)
            .sum();
        assert!(thermal_substeps > 0);
        assert!(final_hot < initial_hot);
        assert!(final_cold > initial_cold);
        assert!(final_hot >= initial_cold && final_cold <= initial_hot);
        assert!((final_energy - initial_energy).abs() < 1e-3);
    }

    #[test]
    fn material_softening_api_validates_and_refreshes_derived_properties() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .build()
            .unwrap();
        assert!(scene.set_material_strength_softening("iron", 0.6).is_ok());
        assert_eq!(
            scene.materials[iron as usize].strength_softening_start_ratio,
            0.6
        );
        assert_eq!(scene.properties[0].strength_softening_start_ratio, 0.6);
        assert!(scene
            .set_material_strength_softening("iron", f32::NAN)
            .is_err());
        assert!(scene
            .set_material_strength_softening("unknown", 0.5)
            .is_err());
        assert_eq!(scene.properties[0].strength_softening_start_ratio, 0.6);
    }

    #[test]
    fn thermal_expansion_api_validates_and_refreshes_derived_properties() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .build()
            .unwrap();
        assert!(scene
            .set_material_thermal_expansion("iron", 18.0e-6)
            .is_ok());
        assert_eq!(scene.properties[0].thermal_expansion_parallel, 18.0e-6);
        assert_eq!(scene.properties[0].thermal_expansion_perpendicular, 18.0e-6);
        assert!(scene
            .set_material_thermal_expansion_anisotropic("iron", 12.0e-6, 20.0e-6)
            .is_ok());
        assert_eq!(scene.properties[0].thermal_expansion_parallel, 12.0e-6);
        assert_eq!(scene.properties[0].thermal_expansion_perpendicular, 20.0e-6);
        assert!(scene
            .set_material_thermal_expansion("iron", f32::NAN)
            .is_err());
        assert!(scene
            .set_material_thermal_expansion("iron", -1.0e-6)
            .is_err());
        assert!(scene
            .set_material_thermal_expansion_anisotropic("iron", 12.0e-6, f32::INFINITY)
            .is_err());
        assert!(scene
            .set_material_thermal_expansion("unknown", 12.0e-6)
            .is_err());
        assert_eq!(scene.properties[0].thermal_expansion_parallel, 12.0e-6);
        assert_eq!(scene.properties[0].thermal_expansion_perpendicular, 20.0e-6);
    }
}

#[cfg(test)]
mod topology_mutation_tests {
    use super::*;

    #[test]
    fn connecting_adjacent_revs_rebuilds_topology_and_accepts_either_order() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(
                VoxelSpawn::at_cell([0, 0, 0], iron)
                    .fixed()
                    .at_temperature(450.0),
            )
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(300.0))
            .build()
            .unwrap();
        assert_eq!(scene.cluster_count, 1);
        scene.release_voxel(0).unwrap();
        assert_eq!(scene.cluster_count, 2);
        scene.connect_adjacent_voxels(1, 0).unwrap();
        assert_eq!(scene.bonds[0][0].neighbor, 1);
        assert_eq!(scene.workspace[0][0].rest_neighbor, 1);
        assert_eq!(scene.cluster_count, 1);
        assert!(scene.colors.iter().any(|color| color.contains(&0)));

        // Existing bonds are idempotent and retain their state.
        scene.bonds[0][0].damage = 0.25;
        scene.positions[1].x += 0.001;
        scene.connect_adjacent_voxels(0, 1).unwrap();
        assert_eq!(scene.bonds[0][0].damage, 0.25);
        assert_eq!(scene.cluster_count, 1);

        let hot_before = scene.temperature_kelvin(0).unwrap();
        let cold_before = scene.temperature_kelvin(1).unwrap();
        scene.conduct_heat(0.01).unwrap();
        assert!(scene.temperature_kelvin(0).unwrap() < hot_before);
        assert!(scene.temperature_kelvin(1).unwrap() > cold_before);
    }

    #[test]
    fn fusion_mixes_composition_and_conserves_mass_momentum_and_energy() {
        let iron = Material::component_id("iron").unwrap();
        let copper = Material::component_id("copper").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(
                VoxelSpawn::at_cell([0, 0, 0], iron)
                    .fixed()
                    .with_velocity(Vec3::Y * 0.01)
                    .at_temperature(500.0),
            )
            .add_voxel(
                VoxelSpawn::at_cell([1, 0, 0], copper)
                    .with_velocity(Vec3::NEG_Y * 0.02)
                    .at_temperature(700.0),
            )
            .add_voxel(VoxelSpawn::at_cell([3, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([4, 0, 0], iron))
            .build()
            .unwrap();
        scene.release_voxel(0).unwrap();
        scene.positions[1].y += 0.002;
        scene.previous[1].y += 0.002;
        scene.voxels[0].velocity = Vec3::Y * 0.01;
        scene.external_forces[0] = Vec3::X;
        scene.external_forces[1] = Vec3::NEG_X * 0.5;
        scene.external_accelerations[0] = Vec3::Y;
        scene.external_accelerations[1] = Vec3::Y * 3.0;
        let expected_iron_fraction = scene.voxels[0].mass as f64
            / (scene.voxels[0].mass as f64 + scene.voxels[1].mass as f64);
        let expected_acceleration_y = (scene.voxels[0].mass as f64
            + 3.0 * scene.voxels[1].mass as f64)
            / (scene.voxels[0].mass as f64 + scene.voxels[1].mass as f64);
        let initial_mass = scene.total_mass_kg();
        let initial_momentum: Vec3 = scene
            .voxels
            .iter()
            .fold(Vec3::ZERO, |sum, voxel| sum + voxel.velocity * voxel.mass);
        let initial_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| {
                voxel.total_internal_energy_j()
                    + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
            })
            .sum();
        scene.energy.pending_release = 0.5;
        scene.releases.push(crate::fracture::Release {
            i: 2,
            j: 3,
            energy: 0.5,
            due: 100,
        });

        let fused_index = scene.fuse_adjacent_voxels(1, 0).unwrap();

        assert_eq!(fused_index, 0);
        assert_eq!(scene.voxels.len(), 3);
        assert!((scene.total_mass_kg() - initial_mass).abs() < 1e-9);
        let final_momentum: Vec3 = scene
            .voxels
            .iter()
            .fold(Vec3::ZERO, |sum, voxel| sum + voxel.velocity * voxel.mass);
        assert!((final_momentum - initial_momentum).length() < 1e-9);
        let final_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| {
                voxel.total_internal_energy_j()
                    + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
            })
            .sum();
        assert!((final_energy - initial_energy).abs() < 1e-9);
        assert!(scene.energy.fusion_heat > 0.0);
        assert_eq!(scene.releases[0].i, 1);
        assert_eq!(scene.releases[0].j, 2);
        assert_eq!(scene.bonds[0][1].neighbor, 2);
        assert_eq!(scene.workspace[0][1].rest_neighbor, 2);
        assert_eq!(scene.cluster_count, 2);
        assert_eq!(scene.energy.pending_release, 0.5);
        let iron_mass_fraction = scene.voxels[0]
            .composition
            .components
            .iter()
            .find(|(component, _)| *component == iron)
            .unwrap()
            .1;
        assert!((iron_mass_fraction as f64 - expected_iron_fraction).abs() < 1e-6);
        assert!((scene.external_forces[0] - Vec3::X * 0.5).length() < 1e-6);
        assert!((scene.external_accelerations[0].y as f64 - expected_acceleration_y).abs() < 1e-6);
        assert_eq!(scene.axis_targets.len(), scene.voxels.len());
        assert_eq!(scene.properties.len(), scene.voxels.len());
        assert!(scene
            .colors
            .iter()
            .flatten()
            .all(|&i| i < scene.voxels.len()));
    }

    #[test]
    fn fusion_rejects_bonds_constraints_and_nonadjacent_voxels_without_mutation() {
        let iron = Material::component_id("iron").unwrap();
        let mut bonded = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        assert!(bonded.fuse_adjacent_voxels(0, 1).is_err());
        assert_eq!(bonded.voxels.len(), 2);

        let mut constrained = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        constrained.release_voxel(0).unwrap();
        constrained
            .fix_voxel_axes(0, [Some(0.005), None, None])
            .unwrap();
        assert!(constrained.fuse_adjacent_voxels(0, 1).is_err());
        assert_eq!(constrained.voxels.len(), 2);

        let mut distant = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([2, 0, 0], iron))
            .build()
            .unwrap();
        distant.release_voxel(0).unwrap();
        assert!(distant.fuse_adjacent_voxels(0, 1).is_err());
        assert_eq!(distant.voxels.len(), 2);

        let mut pending = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        pending.release_voxel(0).unwrap();
        pending.releases.push(crate::fracture::Release {
            i: 0,
            j: 1,
            energy: 0.25,
            due: 1,
        });
        assert!(pending.fuse_adjacent_voxels(0, 1).is_err());
        assert_eq!(pending.voxels.len(), 2);
        assert_eq!(pending.releases.len(), 1);
    }

    #[test]
    fn liquid_viscosity_api_validates_and_refreshes_derived_properties() {
        let water = Material::component_id("water").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], water))
            .build()
            .unwrap();
        assert!((scene.properties[0].dynamic_viscosity_pa_s - 1.002e-3).abs() < 1e-8);
        scene.set_material_dynamic_viscosity("water", 0.25).unwrap();
        assert_eq!(scene.properties[0].dynamic_viscosity_pa_s, 0.25);
        assert!(scene.set_material_dynamic_viscosity("water", -1.0).is_err());
        assert!(scene
            .set_material_dynamic_viscosity("water", f32::INFINITY)
            .is_err());
        assert!(scene
            .set_material_dynamic_viscosity("unknown", 0.1)
            .is_err());
    }

    #[test]
    fn connecting_rejects_nonadjacent_or_constraint_mismatched_revs() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([3, 0, 0], iron).fixed())
            .build()
            .unwrap();
        assert!(scene.connect_adjacent_voxels(0, 1).is_err());
        assert!(scene.connect_adjacent_voxels(0, 2).is_err());
        assert!(scene.connect_adjacent_voxels(0, 0).is_err());
        assert!(scene.connect_adjacent_voxels(0, 9).is_err());
        assert_eq!(scene.cluster_count, 1);
    }

    #[test]
    fn connection_rejects_gap_and_pending_fracture_energy() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        scene.release_voxel(0).unwrap();
        scene.positions[1].x += 0.001;
        assert!(scene.connect_adjacent_voxels(0, 1).is_err());
        scene.positions[1].x -= 0.001;
        scene.releases.push(crate::fracture::Release {
            i: 0,
            j: 1,
            energy: 0.5,
            due: 10,
        });
        scene.energy.pending_release = 0.5;
        assert!(scene.connect_adjacent_voxels(0, 1).is_err());
        assert_eq!(scene.bonds[0][0].neighbor, EMPTY);
        assert_eq!(scene.energy.pending_release, 0.5);
    }

    #[test]
    fn hot_contact_can_connect_without_energy_or_impulse_jump() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .with_timestep(0.001, 1)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        scene.release_voxel(0).unwrap();
        scene
            .set_material_thermal_expansion("iron", 12.0e-6)
            .unwrap();
        for i in 0..2 {
            scene.voxels[i].internal_energy =
                scene.properties[i].energy_at_temperature(scene.voxels[i].mass, 363.15);
        }
        let free_span = CELL * (1.0 + 12.0e-6 * (363.15 - 293.15));
        let energy_before: f64 = scene
            .voxels
            .iter()
            .map(|v| v.total_internal_energy_j())
            .sum();
        scene.connect_adjacent_voxels(0, 1).unwrap();
        assert!((scene.bonds[0][0].formation_offset[0] - (CELL - free_span)).abs() < 1e-9);
        scene.tick();
        let energy_after: f64 = scene
            .voxels
            .iter()
            .map(|v| v.total_internal_energy_j())
            .sum();
        assert_eq!(
            scene.positions[0],
            VoxelSpawn::at_cell([0, 0, 0], iron).center
        );
        assert!((scene.positions[1].x - (scene.positions[0].x + CELL)).abs() < 1e-6);
        assert_eq!(energy_after, energy_before);
    }

    fn impact_pair(temperature_kelvin: f32) -> Scene {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .with_timestep(0.01, 2)
            .add_voxel(
                VoxelSpawn::at_cell([0, 0, 0], iron)
                    .fixed()
                    .with_velocity(Vec3::X * 0.01)
                    .at_temperature(temperature_kelvin),
            )
            .add_voxel(
                VoxelSpawn::at_cell([1, 0, 0], iron)
                    .with_velocity(Vec3::NEG_X * 0.01)
                    .at_temperature(temperature_kelvin),
            )
            .build()
            .unwrap();
        scene.positions[1].x += 0.005;
        scene.previous[1] = scene.positions[1];
        scene.release_voxels(&[0, 1]).unwrap();
        scene.voxels[0].velocity = Vec3::X * 0.01;
        scene
    }

    #[test]
    fn thermal_bond_rule_automatically_connects_hot_impact() {
        let mut scene = impact_pair(363.15);
        scene
            .set_thermal_bonding_rule("iron", 1.0, 350.0, 1.0e-8, 0.0, 0.0)
            .unwrap();
        for _ in 0..40 {
            scene.tick();
            if scene.bonds[0][0].neighbor == 1 {
                break;
            }
        }
        assert_eq!(scene.bonds[0][0].neighbor, 1);
        assert_eq!(scene.cluster_count, 1);
    }

    #[test]
    fn thermal_bond_rule_leaves_cold_impact_unbonded() {
        let mut scene = impact_pair(293.15);
        scene
            .set_thermal_bonding_rule("iron", 1.0, 350.0, 1.0e-8, 0.0, 0.0)
            .unwrap();
        for _ in 0..40 {
            scene.tick();
        }
        assert_eq!(scene.bonds[0][0].neighbor, EMPTY);
        assert_eq!(scene.cluster_count, 2);
    }

    #[test]
    fn thermal_bond_rule_requires_pressure_dwell_after_qualifying_impact() {
        let mut scene = impact_pair(363.15);
        scene.positions[1] = scene.positions[0] + Vec3::X * CELL;
        scene.previous[1] = scene.positions[1];
        scene
            .set_thermal_bonding_rule("iron", 1.0, 350.0, 1e-8, 1.0e6, 0.02)
            .unwrap();
        let first = crate::collision::Contact {
            i: 0,
            j: 1,
            normal: Vec3::X,
            new_impact: true,
        };
        assert_eq!(scene.form_thermal_bonds(&[first], &[1e-5], &[0.5], 0.01), 0);
        assert_eq!(scene.bonds[0][0].neighbor, EMPTY);
        let resting = crate::collision::Contact {
            new_impact: false,
            ..first
        };
        assert_eq!(
            scene.form_thermal_bonds(&[resting], &[0.0], &[1.0], 0.01),
            0
        );
        assert_eq!(
            scene.form_thermal_bonds(&[resting], &[0.0], &[1.0], 0.01),
            1
        );
        assert_eq!(scene.bonds[0][0].neighbor, 1);
    }

    #[test]
    fn thermal_bond_rule_validates_and_can_be_cleared() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .build()
            .unwrap();
        assert!(scene
            .set_thermal_bonding_rule("unknown", 1.0, 300.0, 1e-8, 0.0, 0.0)
            .is_err());
        assert!(scene
            .set_thermal_bonding_rule("iron", 0.0, 300.0, 1e-8, 0.0, 0.0)
            .is_err());
        assert!(scene
            .set_thermal_bonding_rule("iron", 1.0, f32::NAN, 1e-8, 0.0, 0.0)
            .is_err());
        assert!(scene
            .set_thermal_bonding_rule("iron", 1.0, 300.0, 0.0, -1.0, 0.0)
            .is_err());
        scene
            .set_thermal_bonding_rule("iron", 0.75, 350.0, 1e-8, 2.0e6, 0.5)
            .unwrap();
        assert_eq!(scene.thermal_bonding_rule().unwrap().component, iron);
        assert_eq!(
            scene.thermal_bonding_rule().unwrap().minimum_dwell_seconds,
            0.5
        );
        scene.clear_thermal_bonding_rule();
        assert_eq!(scene.thermal_bonding_rule(), None);
    }
}
