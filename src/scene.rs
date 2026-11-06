use crate::{
    aggregation::{AggregationError, ConstituentState, RigidAggregate, RigidMemberTranslationPose},
    bond::{Bond, BondWorkspace, EMPTY},
    material::{Material, PhaseState},
    voxel::{Composition, Microstructure, Voxel},
};
use glam::{DVec3, Quat, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs};

pub const CELL: f32 = crate::coordinates::CELL_M as f32;
pub const DCELL: f64 = crate::coordinates::CELL_M;
pub const AREA: f32 = CELL * CELL;

/// Uniform ambient dry air. Pressure and temperature determine density through
/// the ideal-gas law; the force model uses a measured bluff-body drag
/// coefficient for the exposed voxel surface.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AtmosphereConfig {
    pub pressure_pa: f32,
    pub temperature_kelvin: f32,
    pub wind_velocity_m_s: [f32; 3],
    pub drag_coefficient: f32,
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            pressure_pa: 101_325.0,
            temperature_kelvin: 293.15,
            wind_velocity_m_s: [0.0; 3],
            drag_coefficient: 0.65,
        }
    }
}

impl AtmosphereConfig {
    pub fn density_kg_m3(self) -> f32 {
        const DRY_AIR_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 287.05;
        self.pressure_pa / (DRY_AIR_SPECIFIC_GAS_CONSTANT_J_KG_K * self.temperature_kelvin)
    }

    fn validate(self) -> bool {
        self.pressure_pa.is_finite()
            && self.pressure_pa >= 0.0
            && self.temperature_kelvin.is_finite()
            && self.temperature_kelvin > 0.0
            && Vec3::from_array(self.wind_velocity_m_s).is_finite()
            && self.drag_coefficient.is_finite()
            && self.drag_coefficient >= 0.0
    }
}

/// CPU preconditioner used by the coupled elastic PCG solver.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CpuPreconditioner {
    /// Block incomplete Cholesky with zero fill; reduces iterations but has
    /// sequential triangular application passes.
    #[default]
    IncompleteCholesky0,
    /// Independent per-REV 3×3 diagonal blocks; parallel application can be
    /// faster on multicore CPUs at the cost of potentially more PCG iterations.
    BlockJacobi,
    /// Connected algebraic aggregates, Galerkin coarse operators, symmetric
    /// block smoothing and an exact small bottom solve. Fine physics is retained.
    AggregationMultigrid,
}

/// Inclusive world-space bounds for REV centres (metres).
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct WorldBounds {
    pub min: DVec3,
    pub max: DVec3,
}

/// Temperature, impact, pressure, and dwell thresholds for automatic bonding.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq)]
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
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct ThermalFusionRule {
    pub minimum_temperature_kelvin: f32,
    pub minimum_contact_pressure_pa: f32,
    pub minimum_dwell_seconds: f32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, Default)]
struct BondingContactState {
    dwell_seconds: f32,
    impact_qualified: bool,
    last_seen_substep: u64,
}

#[derive(Clone, Copy)]
struct ThermalEdge {
    a: u32,
    b: u32,
    conductance: f32,
}

#[derive(Clone, Copy)]
struct ThermalFace {
    lower: [u32; 9],
    upper: [u32; 9],
    axis: u8,
    bond_index: u32,
    base_conductance: f64,
}

#[derive(Default)]
struct ThermalScratch {
    /// REV-internal edges are cached for the current active thermal zone.
    internal_edges: Vec<ThermalEdge>,
    cached_active_voxels: Vec<usize>,
    cached_conductivity: Vec<f32>,
    internal_conductance_sum: Vec<f64>,
    /// Cross-REV edges depend on current bond topology and damage.
    edges: Vec<ThermalEdge>,
    /// Face-cell connectivity and undamaged conductance, rebuilt only when
    /// the active thermal set, conductivity, or bond topology changes.
    interface_faces: Vec<ThermalFace>,
    cached_bond_neighbors: Vec<[u32; 3]>,
    cached_interface_voxels: Vec<usize>,
    cached_interface_conductivity: Vec<u32>,
    macro_temperature: Vec<f32>,
    active_mask: Vec<bool>,
    active_seed: Vec<bool>,
    active_voxels: Vec<usize>,
    active_slot: Vec<usize>,
    conductance_sum: Vec<f64>,
    temperature: Vec<f64>,
    energy_delta: Vec<f64>,
    cell_energy: Vec<f64>,
}

/// Cumulative material and energy removed by world-boundary deletion.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundaryRemovalLedger {
    pub voxel_count: u64,
    pub mass_kg: f64,
    pub internal_energy_j: f64,
    pub kinetic_energy_j: f64,
    /// Elastic strain energy in intact bonds removed with the escaping REV.
    pub elastic_bond_energy_j: f64,
    /// Plastic work still stored in removed bond states (joules).
    pub stored_plastic_bond_energy_j: f64,
    /// Energy already queued for delayed bond release when an endpoint leaves.
    pub pending_bond_energy_j: f64,
    pub momentum_kg_m_s: [f64; 3],
}

/// Independent accuracy diagnostics for the most recent frozen mechanical solve.
#[derive(Clone, Copy, Debug)]
pub struct SolverResidualAudit {
    /// Independent f64 b-Ax evaluation for the returned displacement vector.
    pub correction_relative: f64,
    /// The same residual after applying x0+dx to authoritative positions.
    pub applied_relative: f64,
    /// L2 norm of displacement lost when applying dx to stored coordinates (m).
    pub position_rounding_l2_m: f64,
}

/// User-supplied initial state for one 1 cm REV.
#[derive(Clone, Debug)]
pub struct VoxelSpawn {
    pub center: DVec3,
    pub composition: Composition,
    pub velocity: Vec3,
    pub microstructure: Microstructure,
    pub kinematic: bool,
    pub temperature_kelvin: f32,
}

impl VoxelSpawn {
    pub fn new(center: impl Into<DVec3>, component: u16) -> Self {
        Self::new_precise(center.into(), component)
    }

    /// Author a physical centre without rounding through render coordinates.
    pub fn new_precise(center: DVec3, component: u16) -> Self {
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
        let cell = DVec3::new(cell[0] as f64, cell[1] as f64, cell[2] as f64);
        Self::new_precise(
            cell * crate::coordinates::CELL_M + DVec3::splat(crate::coordinates::CELL_M * 0.5),
            component,
        )
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    pub dimensions: [usize; 3],
    pub branch_dimensions: [usize; 3],
    pub ground_dimensions: [usize; 3],
    pub tool_dimensions: [usize; 3],
    pub fibers: [f32; 3],
    pub anisotropy: f32,
    pub gravity: [f32; 3],
    #[serde(default)]
    pub atmosphere: AtmosphereConfig,
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
    /// Travel direction for the tool and its applied force; normalized on use.
    #[serde(default = "default_tool_direction")]
    pub tool_direction: [f32; 3],
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
            atmosphere: AtmosphereConfig::default(),
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
            tool_direction: default_tool_direction(),
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
            || !result.atmosphere.validate()
            || !Vec3::from_array(result.tool_start).is_finite()
            || !Vec3::from_array(result.tool_direction).is_finite()
            || Vec3::from_array(result.tool_direction).length_squared() < 1e-8
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
    640
}
fn default_tool_direction() -> [f32; 3] {
    [1.0, 0.0, 0.0]
}
/// Runtime observations, never part of an authored scene's positional codec.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StepDiagnostics {
    pub scope: &'static str,
    pub fixed_dt_seconds: f64,
    pub mechanical_substeps: usize,
    pub active_fine_revs_last_solve: usize,
    pub pcg_iterations_total: usize,
    /// All projected contact solves, including restitution corrections.
    pub contact_pcg_calls: usize,
    pub contact_pcg_iterations_including_restitution: usize,
    pub contact_gpu_operator_calls: usize,
    pub contact_gpu_preconditioner_calls: usize,
    pub contact_gpu_operator_fallbacks: usize,
    pub contact_pcg_unconverged_calls: usize,
    pub contact_pcg_max_true_residual: Option<f32>,
    pub relative_solver_residual: f32,
    pub contacts_peak: usize,
    pub new_impact_observations: usize,
    /// Timings overlap: assembly/factor/PCG are parts of mechanics_total.
    pub timings_ms: std::collections::BTreeMap<&'static str, f64>,
}

mod checkpoint;
mod beam_region;

/// Positions are separate arrays; material identity exists only in composition.
pub struct Scene {
    /// Physical time advanced by the fixed-step solver; never wall-clock time.
    pub simulated_seconds: f64,
    pub last_step_diagnostics: Option<StepDiagnostics>,
    /// Runtime scheduler policy; kept out of the positional document format.
    pub(crate) adaptive_substeps: bool,
    pub(crate) last_mechanical_substeps: usize,
    /// Capacity-reusing workspace retained between physical steps.
    pub(crate) solver_scratch: crate::solver::SolverScratch,
    /// Capacity-reusing thermal transport workspace retained between substeps.
    thermal_scratch: ThermalScratch,
    /// Capacity-reusing spatial data for broadphase collision detection.
    pub(crate) collision_scratch: crate::collision::CollisionScratch,
    /// Conservative rigid-body tiers for closed REV components outside the
    /// active interaction zone. `aggregate_of` remains separate from
    /// `Voxel::kinematic`: aggregated matter keeps its physical mass.
    pub(crate) rigid_aggregates: Vec<Option<RigidAggregate>>,
    pub(crate) aggregate_of: Vec<Option<usize>>,
    pub config: SceneConfig,
    pub voxels: Vec<Voxel>,
    /// Optional generated-world identity for each REV, compacted with topology.
    pub(crate) source_cells: Vec<Option<[i32; 3]>>,
    /// Persistent per-voxel forces in newtons, retained until explicitly reset.
    pub external_forces: Vec<Vec3>,
    /// Transient skeleton-driven motors, applied through the ordinary solver.
    pub skeletal_actuators: Vec<crate::skeletal_actuator::SkeletalActuator>,
    /// Persistent per-voxel acceleration inputs in m/s².
    pub external_accelerations: Vec<Vec3>,
    /// Persistent world-coordinate constraints; None leaves an axis free.
    pub axis_targets: Vec<[Option<f64>; 3]>,
    pub positions: Vec<DVec3>,
    pub rest_positions: Vec<DVec3>,
    /// Stress-free temperature of each REV when the scene was created (K).
    pub(crate) reference_temperatures_kelvin: Vec<f32>,
    pub previous: Vec<DVec3>,
    pub bonds: [Vec<Bond>; 3],
    pub workspace: [Vec<BondWorkspace>; 3],
    pub colors: [Vec<usize>; 6],
    pub materials: Vec<Material>,
    pub properties: Vec<Material>,
    pub rotations: Vec<Quat>,
    /// Run material reactions (combustion, tanning, decay) at every fixed-step
    /// boundary. Off by default so scenes with no chemistry do not pay for a scan
    /// of every REV; a scene that burns or rots turns it on.
    pub chemistry_enabled: bool,
    /// Simulated time (s) between reaction updates while `chemistry_enabled`;
    /// fixed-step time is accumulated until it is reached. `0` (default) reacts
    /// at every fixed step, right for fast chemistry such as combustion. Slow
    /// chemistry (rot, tanning) changes a REV by less than the `f32` resolution
    /// of its composition in one millisecond step, so such a scene sets an
    /// interval of seconds to minutes; see `Scene::set_chemistry_interval`.
    pub(crate) chemistry_interval_s: f64,
    pub(crate) chemistry_pending_s: f64,
    /// Recipes used by the fixed-step hook; `None` is the built-in registry.
    pub(crate) chemistry_registry: Option<crate::recipes::RecipeRegistry>,
    /// Slender parts simulated as beams (tier T2); see `scene/beam_region.rs`.
    pub(crate) beam_regions: Vec<Option<beam_region::BeamRegion>>,
    /// Whether this step solves cell rotations; decided at the start of each step.
    pub(crate) cell_dofs: bool,
    /// Double-precision cell rotations; `rotations` holds the same values as f32.
    pub(crate) precise_rotations: Vec<glam::DQuat>,
    /// Normal reaction of a fixed support on each REV in the last substep, newtons.
    /// Zero for REVs that do not rest on a fixed support. Derived, not saved.
    pub support_load_n: Vec<f32>,
    /// Angular velocity of each cell about its own centre, rad/s, world axes.
    pub angular_velocities: Vec<glam::DVec3>,
    /// Derived connectivity labels; rebuilt only when topology changes.
    pub clusters: Vec<u32>,
    pub broken: usize,
    pub cluster_count: usize,
    pub ground_top: f32,
    pub energy: crate::fracture::EnergyLedger,
    pub releases: Vec<crate::fracture::Release>,
    pub substep_index: u64,
    /// Monotonic revision for data consumed by the renderer.
    pub render_revision: u64,
    /// Independently recomputed relative residual of the last fine operator.
    pub solver_residual: f32,
    /// Last independently audited solve; enabled by VOXEL_AUDIT_PCG_RESIDUAL.
    /// The verified operator residual alone does not measure position-write loss.
    pub solver_residual_audit: Option<SolverResidualAudit>,
    /// Worst independent relative residual over every solve in this fixed step.
    pub solver_max_audited_residual: Option<f64>,
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

    /// Notify render caches after direct mutation of public render-related arrays.
    pub fn mark_render_changed(&mut self) {
        self.render_revision = self.render_revision.wrapping_add(1);
    }

    /// Register a conservative aggregate candidate for one closed bonded
    /// component with no unresolved internal velocity mode. Fine REV data
    /// remains authoritative; fixed-step mechanics advance this candidate
    /// coarsely until a load, constraint, contact, thermal change, topology
    /// change, or world-bound event wakes its REVs.
    pub fn register_rigid_aggregate_candidate(
        &mut self,
        members: &[usize],
    ) -> Result<usize, AggregationError> {
        let aggregate = self.rigid_aggregate_snapshot(members)?;
        if aggregate.has_unresolved_internal_motion() {
            return Err(AggregationError::UnresolvedInternalMotion);
        }
        for &index in members {
            if self.aggregate_of[index].is_some() {
                return Err(AggregationError::AlreadyAggregated(index));
            }
        }

        let id = self.rigid_aggregates.len();
        self.rigid_aggregates.push(Some(aggregate));
        for &index in members {
            self.aggregate_of[index] = Some(id);
        }
        Ok(id)
    }

    /// Register members already extracted as a complete component from the
    /// current cluster labels. This avoids rescanning the entire bond graph
    /// once per component during an automatic coarse-tier update.
    pub(crate) fn register_rigid_aggregate_candidate_in_cluster(
        &mut self,
        members: &[usize],
        cluster_id: u32,
        expected_len: usize,
    ) -> Result<usize, AggregationError> {
        if members.is_empty() {
            return Err(AggregationError::Empty);
        }
        if self.clusters.len() != self.voxels.len()
            || members.len() != expected_len
            || cluster_id as usize >= self.cluster_count
        {
            return Err(AggregationError::DisconnectedSet);
        }
        for &index in members {
            if index >= self.voxels.len() {
                return Err(AggregationError::IndexOutOfBounds(index));
            }
            if self.clusters[index] != cluster_id {
                return Err(AggregationError::DisconnectedSet);
            }
            if self.aggregate_of[index].is_some() {
                return Err(AggregationError::AlreadyAggregated(index));
            }
        }

        let aggregate = self.rigid_aggregate_snapshot_impl(members, false)?;
        if aggregate.has_unresolved_internal_motion() {
            return Err(AggregationError::UnresolvedInternalMotion);
        }
        let id = self.rigid_aggregates.len();
        self.rigid_aggregates.push(Some(aggregate));
        for &index in members {
            self.aggregate_of[index] = Some(id);
        }
        Ok(id)
    }

    pub fn aggregate_for_voxel(&self, voxel: usize) -> Option<usize> {
        self.aggregate_of.get(voxel).copied().flatten()
    }

    pub fn registered_rigid_aggregate_candidate(&self, id: usize) -> Option<&RigidAggregate> {
        self.rigid_aggregates.get(id)?.as_ref()
    }

    pub(crate) fn has_registered_rigid_candidates(&self) -> bool {
        self.rigid_aggregates.iter().any(Option::is_some)
    }

    /// Refine a registered aggregate candidate back into its original 1 cm
    /// REV state. Membership is cleared only if transactional refinement
    /// succeeds.
    pub fn refine_registered_rigid_candidate(
        &mut self,
        id: usize,
    ) -> Result<usize, AggregationError> {
        let aggregate = self
            .rigid_aggregates
            .get_mut(id)
            .and_then(Option::take)
            .ok_or(AggregationError::InactiveAggregate(id))?;
        match self.refine_rigid_aggregate(&aggregate) {
            Ok(refined_count) => {
                for index in aggregate.member_indices() {
                    if self.aggregate_of.get(index).copied().flatten() == Some(id) {
                        self.aggregate_of[index] = None;
                    }
                }
                Ok(refined_count)
            }
            Err(error) => {
                self.rigid_aggregates[id] = Some(aggregate);
                Err(error)
            }
        }
    }

    /// Wake aggregate candidates touched by a fine-tier interaction. The
    /// current REV arrays contain the just-computed contact impulse or contact
    /// projection, so refresh the conservative snapshot from those arrays
    /// before refining; this preserves transferred momentum, heat, and pose.
    /// A candidate whose topology has gone stale is discarded while the fine
    /// scene state remains authoritative.
    pub(crate) fn wake_rigid_candidates_for_voxels(&mut self, voxels: &[usize]) -> usize {
        let mut ids = Vec::new();
        ids.extend(
            voxels
                .iter()
                .filter_map(|&voxel| self.aggregate_for_voxel(voxel)),
        );
        ids.sort_unstable();
        ids.dedup();

        let mut woken = 0;
        for id in ids {
            let members: Vec<_> = self
                .registered_rigid_aggregate_candidate(id)
                .into_iter()
                .flat_map(RigidAggregate::member_indices)
                .collect();
            if members.is_empty() {
                continue;
            }

            match self.rigid_aggregate_snapshot(&members) {
                Ok(refreshed) => self.rigid_aggregates[id] = Some(refreshed),
                Err(_) => {
                    self.discard_rigid_aggregate_candidate(id);
                    continue;
                }
            }
            if self.refine_registered_rigid_candidate(id).is_ok() {
                woken += 1;
            } else {
                self.discard_rigid_aggregate_candidate(id);
            }
        }
        woken
    }

    pub(crate) fn wake_all_rigid_candidates(&mut self) -> usize {
        let voxels: Vec<_> = self
            .rigid_aggregates
            .iter()
            .flatten()
            .flat_map(RigidAggregate::member_indices)
            .collect();
        self.wake_rigid_candidates_for_voxels(&voxels)
    }

    /// Heat and phase remain resolved on the original REVs. Wake a coarse
    /// mechanical body before altered thermal state can change its stiffness,
    /// expansion, or fracture thresholds.
    pub(crate) fn wake_rigid_candidates_with_thermal_changes(&mut self) -> usize {
        let mut changed_voxels = Vec::new();
        for aggregate in self.rigid_aggregates.iter().flatten() {
            for member in aggregate.refine() {
                let Some(voxel) = self.voxels.get(member.index) else {
                    changed_voxels.push(member.index);
                    continue;
                };
                let cells_match = member.thermal_cell_energy_j.as_deref()
                    == voxel.thermal_cell_energy_j.as_deref();
                if member.thermal_energy_j.to_bits() != voxel.total_internal_energy_j().to_bits()
                    || !cells_match
                {
                    changed_voxels.push(member.index);
                }
            }
        }
        self.wake_rigid_candidates_for_voxels(&changed_voxels)
    }

    /// Advance registered coarse bodies by their exact resultant force and
    /// torque, then synchronize their rigid pose to the retained REV arrays.
    /// The caller must wake bodies with nonuniform nodal loading or active
    /// constraints before calling this method.
    pub(crate) fn advance_registered_rigid_candidates(
        &mut self,
        dt_s: f64,
        gravity_m_s2: DVec3,
        nodal_loads_n: &[Vec3],
        reference_step_s: Option<f64>,
    ) -> Result<usize, AggregationError> {
        let mut advanced = 0;
        let aggregate_count = self.rigid_aggregates.len();
        let has_nodal_loads = nodal_loads_n.iter().any(|load| *load != Vec3::ZERO);
        for id in 0..aggregate_count {
            let Some(view) = self.rigid_aggregates[id].as_ref() else {
                continue;
            };
            let (force, torque, total_mass) = if !has_nodal_loads {
                // Uniform gravity acts through the aggregate center of mass:
                // its exact resultant is m*g with zero torque. Avoid walking
                // every retained REV to re-sum that identity and its
                // roundoff-only moment.
                let mass = view.mass_kg();
                (gravity_m_s2 * mass, DVec3::ZERO, mass)
            } else {
                let center = view.center_of_mass_m();
                let mut force = DVec3::ZERO;
                let mut torque = DVec3::ZERO;
                let mut total_mass = 0.0;
                for member in view.member_poses() {
                    let nodal = nodal_loads_n
                        .get(member.index)
                        .copied()
                        .unwrap_or(Vec3::ZERO);
                    let member_force = gravity_m_s2 * member.mass_kg
                        + DVec3::new(nodal.x as f64, nodal.y as f64, nodal.z as f64);
                    force += member_force;
                    torque += (member.position_m - center).cross(member_force);
                    total_mass += member.mass_kg;
                }
                (force, torque, total_mass)
            };

            // Boundary removal is performed on fine REVs at the fixed-step
            // boundary. Refine before a coarse body's conservative swept
            // bounds can cross that boundary, so removal and its mass/energy
            // ledger always see the constituent state.
            let boundary_members = if let Some(bounds) = self.world_bounds {
                let acceleration = force / total_mass;
                let (mut swept_min, mut swept_max) =
                    view.swept_bounds_under_acceleration(dt_s, acceleration)?;
                if let Some(h) = reference_step_s {
                    let drift_bias = acceleration * (0.5 * dt_s * h);
                    swept_min += drift_bias.min(DVec3::ZERO);
                    swept_max += drift_bias.max(DVec3::ZERO);
                }
                let bounds_min = DVec3::new(
                    bounds.min.x as f64,
                    bounds.min.y as f64,
                    bounds.min.z as f64,
                );
                let bounds_max = DVec3::new(
                    bounds.max.x as f64,
                    bounds.max.y as f64,
                    bounds.max.z as f64,
                );
                if swept_min.cmplt(bounds_min).any() || swept_max.cmpgt(bounds_max).any() {
                    Some(view.member_indices().collect::<Vec<_>>())
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(member_indices) = boundary_members {
                self.wake_rigid_candidates_for_voxels(&member_indices);
                continue;
            }

            let mut aggregate = self.rigid_aggregates[id]
                .take()
                .expect("active aggregate id collected above");
            let orientation_before = aggregate.orientation();
            let advance = match reference_step_s {
                Some(h) => aggregate
                    .advance_under_wrench_matching_inertial_step(force, torque, dt_s, h),
                None => aggregate.advance_under_wrench(force, torque, dt_s),
            };
            if let Err(error) = advance {
                self.rigid_aggregates[id] = Some(aggregate);
                return Err(error);
            }
            let write_result = if aggregate.has_zero_angular_momentum()
                && aggregate.orientation() == orientation_before
            {
                self.write_rigid_aggregate_translation_pose(&aggregate)
            } else {
                self.write_rigid_aggregate_pose(&aggregate)
            };
            if let Err(error) = write_result {
                self.rigid_aggregates[id] = Some(aggregate);
                return Err(error);
            }
            self.rigid_aggregates[id] = Some(aggregate);
            advanced += 1;
        }
        if advanced > 0 {
            self.mark_render_changed();
        }
        Ok(advanced)
    }

    fn write_rigid_aggregate_pose(
        &mut self,
        aggregate: &RigidAggregate,
    ) -> Result<(), AggregationError> {
        let f32_max = f32::MAX as f64;
        let fits_f32 =
            |value: DVec3| value.is_finite() && value.to_array().iter().all(|x| x.abs() <= f32_max);
        for state in aggregate.member_poses() {
            if state.index >= self.voxels.len() {
                return Err(AggregationError::IndexOutOfBounds(state.index));
            }
            let orientation = state.orientation;
            if !fits_f32(state.position_m)
                || !fits_f32(state.rest_position_m)
                || !fits_f32(state.velocity_m_s)
                || !orientation.is_finite()
                || !orientation.length_squared().is_finite()
                || orientation.length_squared() <= f64::EPSILON
                || state
                    .bond_last_force_n
                    .iter()
                    .any(|force| !fits_f32(*force))
            {
                return Err(AggregationError::InvalidState(state.index));
            }
        }

        let to_vec3 = |value: DVec3| Vec3::new(value.x as f32, value.y as f32, value.z as f32);
        for state in aggregate.member_poses() {
            let i = state.index;
            self.positions[i] = state.position_m;
            self.rest_positions[i] = state.rest_position_m;
            self.voxels[i].velocity = to_vec3(state.velocity_m_s);
            self.rotations[i] = Quat::from_xyzw(
                state.orientation.x as f32,
                state.orientation.y as f32,
                state.orientation.z as f32,
                state.orientation.w as f32,
            )
            .normalize();
            for axis in 0..3 {
                self.workspace[axis][i].last_force = to_vec3(state.bond_last_force_n[axis]);
            }
        }
        Ok(())
    }

    fn write_rigid_aggregate_translation_pose(
        &mut self,
        aggregate: &RigidAggregate,
    ) -> Result<(), AggregationError> {
        let f32_max = f32::MAX as f64;
        let fits_f32 =
            |value: DVec3| value.is_finite() && value.to_array().iter().all(|x| x.abs() <= f32_max);
        for state in aggregate.translation_member_poses() {
            if state.index >= self.voxels.len() {
                return Err(AggregationError::IndexOutOfBounds(state.index));
            }
            if !fits_f32(state.position_m)
                || !fits_f32(state.rest_position_m)
                || !fits_f32(state.velocity_m_s)
            {
                return Err(AggregationError::InvalidState(state.index));
            }
        }

        let to_vec3 = |value: DVec3| Vec3::new(value.x as f32, value.y as f32, value.z as f32);
        for RigidMemberTranslationPose {
            index,
            position_m,
            rest_position_m,
            velocity_m_s,
        } in aggregate.translation_member_poses()
        {
            self.positions[index] = position_m;
            self.rest_positions[index] = rest_position_m;
            self.voxels[index].velocity = to_vec3(velocity_m_s);
        }
        Ok(())
    }

    fn discard_rigid_aggregate_candidate(&mut self, id: usize) {
        let Some(slot) = self.rigid_aggregates.get_mut(id) else {
            return;
        };
        if let Some(aggregate) = slot.take() {
            for voxel in aggregate.member_indices() {
                if self.aggregate_of.get(voxel).copied().flatten() == Some(id) {
                    self.aggregate_of[voxel] = None;
                }
            }
        }
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
    /// Mixtures use a mass-fraction weighted reference estimate. The value is
    /// evaluated at each material's configured reference temperature.
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

    /// Configure the Arrhenius temperature law for a built-in material's
    /// liquid viscosity. The existing `dynamic_viscosity_pa_s` value remains
    /// the viscosity at `reference_temperature_kelvin`; activation energy is
    /// in J/mol. Built-in values are illustrative and require calibration.
    pub fn set_material_viscosity_temperature_model(
        &mut self,
        material_name: &str,
        reference_temperature_kelvin: f32,
        activation_energy_j_mol: f32,
    ) -> Result<(), String> {
        if !reference_temperature_kelvin.is_finite()
            || reference_temperature_kelvin <= 0.0
            || !activation_energy_j_mol.is_finite()
            || activation_energy_j_mol < 0.0
        {
            return Err("viscosity reference temperature must be positive and activation energy nonnegative".into());
        }
        let id = Material::component_id(material_name)
            .ok_or_else(|| format!("unknown material: {material_name}"))? as usize;
        self.materials[id].viscosity_reference_temperature_kelvin = reference_temperature_kelvin;
        self.materials[id].viscosity_activation_energy_j_mol = activation_energy_j_mol;
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
        Self::from_voxels_with_materials(config, Material::table(), spawns)
    }

    /// Create a scene from separate voxel objects. Adjacent REVs bond only
    /// inside their source object; contacts between different objects remain
    /// separate mechanical interactions rather than being silently fused.
    pub fn from_voxel_objects(
        config: SceneConfig,
        objects: Vec<Vec<VoxelSpawn>>,
    ) -> Result<Self, String> {
        Self::from_voxel_objects_with_materials(config, Material::table(), objects)
    }

    /// Variant of [`Scene::from_voxel_objects`] for authored material tables.
    pub fn from_voxel_objects_with_materials(
        config: SceneConfig,
        materials: Vec<Material>,
        objects: Vec<Vec<VoxelSpawn>>,
    ) -> Result<Self, String> {
        let voxel_count = objects
            .iter()
            .try_fold(0usize, |total, object| total.checked_add(object.len()))
            .ok_or_else(|| "scene object voxel count overflow".to_owned())?;
        let mut spawns = Vec::with_capacity(voxel_count);
        let mut bonding_groups = Vec::with_capacity(voxel_count);
        for (object_index, object) in objects.into_iter().enumerate() {
            let group = u32::try_from(object_index)
                .map_err(|_| "scene has too many separate voxel objects".to_owned())?;
            bonding_groups.extend(std::iter::repeat_n(group, object.len()));
            spawns.extend(object);
        }
        Self::from_voxels_with_materials_and_groups(config, materials, spawns, bonding_groups)
    }

    pub(crate) fn from_voxels_with_materials(
        config: SceneConfig,
        materials: Vec<Material>,
        spawns: Vec<VoxelSpawn>,
    ) -> Result<Self, String> {
        let bonding_groups = vec![0; spawns.len()];
        Self::from_voxels_with_materials_and_groups(config, materials, spawns, bonding_groups)
    }

    fn from_voxels_with_materials_and_groups(
        config: SceneConfig,
        mut materials: Vec<Material>,
        spawns: Vec<VoxelSpawn>,
        bonding_groups: Vec<u32>,
    ) -> Result<Self, String> {
        Material::restore_builtin_chemistry(&mut materials);
        config.validate().map_err(str::to_owned)?;
        if materials.is_empty() {
            return Err("scene material table must not be empty".into());
        }
        if bonding_groups.len() != spawns.len() {
            return Err("voxel object membership does not match the REV count".into());
        }
        let mut scene = Self {
            config,
            simulated_seconds: 0.0,
            last_step_diagnostics: None,
            solver_scratch: Default::default(),
            adaptive_substeps: false,
            last_mechanical_substeps: 0,
            thermal_scratch: Default::default(),
            collision_scratch: Default::default(),
            rigid_aggregates: Vec::new(),
            aggregate_of: vec![None; spawns.len()],
            voxels: Vec::with_capacity(spawns.len()),
            source_cells: vec![None; spawns.len()],
            external_forces: Vec::with_capacity(spawns.len()),
            skeletal_actuators: Vec::new(),
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
            chemistry_enabled: false,
            chemistry_interval_s: 0.0,
            chemistry_pending_s: 0.0,
            chemistry_registry: None,
            cell_dofs: false,
            beam_regions: Vec::new(),
            support_load_n: Vec::new(),
            precise_rotations: Vec::with_capacity(spawns.len()),
            angular_velocities: Vec::with_capacity(spawns.len()),
            clusters: Vec::with_capacity(spawns.len()),
            broken: 0,
            cluster_count: 0,
            ground_top: 0.0,
            energy: Default::default(),
            releases: Vec::new(),
            substep_index: 0,
            render_revision: 0,
            solver_residual: 0.0,
            solver_residual_audit: None,
            solver_max_audited_residual: None,
            world_bounds: None,
            thermal_bonding_rule: None,
            bonding_contacts: HashMap::new(),
            thermal_fusion_rule: None,
            fusion_contacts: HashMap::new(),
            pending_fusion: None,
            boundary_removals: BoundaryRemovalLedger::default(),
        };
        // Voxel lattice keys are engine-generated integer coordinates, so
        // randomized collision resistance is unnecessary here. FxHashMap keeps
        // the exact same occupancy semantics while reducing the cost of the
        // three neighbor lookups per REV in large physical patches.
        let mut occupancy =
            rustc_hash::FxHashMap::with_capacity_and_hasher(spawns.len(), Default::default());
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
            let lattice = spawn.center / DCELL - DVec3::splat(0.5);
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
                thermal_cell_energy_j: None,
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
            scene.precise_rotations.push(glam::DQuat::IDENTITY);
            scene.angular_velocities.push(glam::DVec3::ZERO);
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
                    if bonding_groups[i] == bonding_groups[j]
                        && scene.voxels[i].kinematic == scene.voxels[j].kinematic
                    {
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

    /// Restore saved per-REV state while rebuilding only the scene's derived
    /// topology and workspaces. Constraint and animation runtime state is
    /// restored by the corresponding editor systems when those systems exist.
    pub(crate) fn from_saved_voxel_states(
        config: SceneConfig,
        materials: Vec<Material>,
        snapshots: Vec<(DVec3, DVec3, Voxel)>,
    ) -> Result<Self, String> {
        let bonding_groups = vec![0; snapshots.len()];
        Self::from_saved_voxel_states_with_groups(config, materials, snapshots, bonding_groups)
    }

    /// Restore per-REV state from separate authored objects while keeping
    /// their internal bonds and object-to-object contacts distinct.
    pub(crate) fn from_saved_voxel_objects(
        config: SceneConfig,
        materials: Vec<Material>,
        objects: Vec<Vec<(DVec3, DVec3, Voxel)>>,
    ) -> Result<Self, String> {
        let voxel_count = objects
            .iter()
            .try_fold(0usize, |total, object| total.checked_add(object.len()))
            .ok_or_else(|| "scene object voxel count overflow".to_owned())?;
        let mut snapshots = Vec::with_capacity(voxel_count);
        let mut bonding_groups = Vec::with_capacity(voxel_count);
        for (object_index, object) in objects.into_iter().enumerate() {
            let group = u32::try_from(object_index)
                .map_err(|_| "scene has too many separate voxel objects".to_owned())?;
            bonding_groups.extend(std::iter::repeat_n(group, object.len()));
            snapshots.extend(object);
        }
        Self::from_saved_voxel_states_with_groups(config, materials, snapshots, bonding_groups)
    }

    fn from_saved_voxel_states_with_groups(
        config: SceneConfig,
        mut materials: Vec<Material>,
        snapshots: Vec<(DVec3, DVec3, Voxel)>,
        bonding_groups: Vec<u32>,
    ) -> Result<Self, String> {
        Material::restore_builtin_chemistry(&mut materials);
        if materials.is_empty() {
            return Err("scene material table must not be empty".into());
        }
        if bonding_groups.len() != snapshots.len() {
            return Err("saved object membership does not match the REV count".into());
        }
        let spawns = snapshots
            .iter()
            .map(|(position, rest_position, state)| {
                state.composition.validate(materials.len())?;
                if !position.is_finite()
                    || !rest_position.is_finite()
                    || !state.mass.is_finite()
                    || state.mass <= 0.0
                    || !state.velocity.is_finite()
                    || !state.internal_energy.is_finite()
                    || state.internal_energy < 0.0
                    || !state.internal_energy_remainder_j.is_finite()
                    || !state.microstructure.orientation.is_finite()
                    || state.microstructure.orientation.length_squared() < 1e-8
                    || !state.microstructure.anisotropy.is_finite()
                    || !(0.0..=1.0).contains(&state.microstructure.anisotropy)
                {
                    return Err("saved voxel has invalid physical state".to_owned());
                }
                let mut thermal_cells = [0.0; crate::voxel::THERMAL_CELL_COUNT];
                state.copy_thermal_cell_energies_j(&mut thermal_cells);
                let thermal_total: f64 = thermal_cells.iter().sum();
                if thermal_cells
                    .iter()
                    .any(|energy| !energy.is_finite() || *energy < 0.0)
                    || (thermal_total - state.total_internal_energy_j()).abs()
                        > 1e-9 * thermal_total.abs().max(1.0)
                {
                    return Err("saved voxel has inconsistent thermal energy".to_owned());
                }
                let material = Material::mixed(&state.composition, &materials);
                Ok(VoxelSpawn {
                    center: *position,
                    composition: state.composition.clone(),
                    velocity: state.velocity,
                    microstructure: state.microstructure,
                    kinematic: state.kinematic,
                    temperature_kelvin: material
                        .phase_state(state.mass, state.thermal_energy_f32())
                        .temperature_kelvin,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut scene =
            Self::from_voxels_with_materials_and_groups(config, materials, spawns, bonding_groups)?;
        for (index, (position, rest_position, state)) in snapshots.into_iter().enumerate() {
            if !position.is_finite() || !rest_position.is_finite() {
                return Err(format!("saved voxel {index} has a non-finite position"));
            }
            let properties = Material::mixed(&state.composition, &scene.materials);
            scene.reference_temperatures_kelvin[index] = properties
                .phase_state(state.mass, state.thermal_energy_f32())
                .temperature_kelvin;
            scene.positions[index] = position;
            scene.previous[index] = position;
            scene.rest_positions[index] = rest_position;
            scene.properties[index] = properties;
            scene.voxels[index] = state;
        }
        Ok(scene)
    }

    /// Set world bounds. REV centres outside the inclusive box are removed now
    /// and after each subsequent physical step.
    pub fn set_world_bounds(
        &mut self,
        min: impl Into<DVec3>,
        max: impl Into<DVec3>,
    ) -> Result<usize, &'static str> {
        let min = min.into();
        let max = max.into();
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

    /// Configure automatic fusion for hot adjacent REVs under sustained
    /// contact pressure. Fusion is committed at the fixed-step boundary after
    /// the complete contact solve, so solver arrays never change mid-substep.
    pub fn set_thermal_fusion_rule(
        &mut self,
        minimum_temperature_kelvin: f32,
        minimum_contact_pressure_pa: f32,
        minimum_dwell_seconds: f32,
    ) -> Result<(), &'static str> {
        if !minimum_temperature_kelvin.is_finite()
            || minimum_temperature_kelvin < 0.0
            || !minimum_contact_pressure_pa.is_finite()
            || minimum_contact_pressure_pa < 0.0
            || !minimum_dwell_seconds.is_finite()
            || minimum_dwell_seconds < 0.0
        {
            return Err("fusion thresholds must be finite and nonnegative");
        }
        self.thermal_fusion_rule = Some(ThermalFusionRule {
            minimum_temperature_kelvin,
            minimum_contact_pressure_pa,
            minimum_dwell_seconds,
        });
        self.fusion_contacts.clear();
        self.pending_fusion = None;
        Ok(())
    }

    pub fn clear_thermal_fusion_rule(&mut self) {
        self.thermal_fusion_rule = None;
        self.fusion_contacts.clear();
        self.pending_fusion = None;
    }

    pub fn thermal_fusion_rule(&self) -> Option<ThermalFusionRule> {
        self.thermal_fusion_rule
    }

    /// Update automatic fusion contact pressure/dwell after impulse resolution.
    pub(crate) fn form_thermal_fusions(
        &mut self,
        contacts: &[crate::collision::Contact],
        normal_impulses_ns: &[f32],
        elapsed_seconds: f32,
        targets: &[[Option<f64>; 3]],
    ) {
        let Some(rule) = self.thermal_fusion_rule else {
            return;
        };
        if self.pending_fusion.is_some() {
            return;
        }
        let current_substep = self.substep_index;
        for (index, contact) in contacts.iter().enumerate() {
            let i = contact.i;
            let j = contact.j;
            let key = if i < j { (i, j) } else { (j, i) };
            let has_bond = self.bonds.iter().any(|axis| {
                axis.iter().enumerate().any(|(endpoint, bond)| {
                    bond.neighbor != EMPTY
                        && (endpoint == i
                            || bond.neighbor as usize == i
                            || endpoint == j
                            || bond.neighbor as usize == j)
                })
            });
            let constrained = self.voxels[i].kinematic
                || self.voxels[j].kinematic
                || self.axis_targets[i].iter().any(Option::is_some)
                || self.axis_targets[j].iter().any(Option::is_some)
                || targets
                    .get(i)
                    .is_some_and(|target| target.iter().any(Option::is_some))
                || targets
                    .get(j)
                    .is_some_and(|target| target.iter().any(Option::is_some));
            let pending_release = self.releases.iter().any(|release| {
                release.i == i || release.j == i || release.i == j || release.j == j
            });
            let lattice_delta = (self.rest_positions[j] - self.rest_positions[i]) / DCELL;
            let rounded = lattice_delta.round();
            let lattice_adjacent = (lattice_delta - rounded).abs().max_element() <= 1e-3
                && rounded.abs().element_sum() == 1.0;
            let face_normal = rounded.normalize_or_zero();
            let separation = self.positions[j] - self.positions[i];
            let projected = separation.dot(face_normal);
            let tangent = separation - face_normal * projected;
            let face_contact = (projected.abs() - DCELL).abs() <= DCELL * 1e-3
                && tangent.abs().max_element() < DCELL;
            let temperature_ok = self.temperature_kelvin(i).unwrap_or(0.0)
                >= rule.minimum_temperature_kelvin
                && self.temperature_kelvin(j).unwrap_or(0.0) >= rule.minimum_temperature_kelvin;
            if has_bond
                || constrained
                || pending_release
                || !temperature_ok
                || !lattice_adjacent
                || !face_contact
            {
                self.fusion_contacts.remove(&key);
                continue;
            }
            let impulse = normal_impulses_ns.get(index).copied().unwrap_or(0.0);
            let pressure_pa = impulse / (elapsed_seconds.max(f32::MIN_POSITIVE) * CELL * CELL);
            let ready = {
                let state = self.fusion_contacts.entry(key).or_default();
                state.last_seen_substep = current_substep;
                if pressure_pa >= rule.minimum_contact_pressure_pa {
                    state.dwell_seconds += elapsed_seconds;
                } else {
                    state.dwell_seconds = 0.0;
                }
                state.dwell_seconds + f32::EPSILON >= rule.minimum_dwell_seconds
            };
            if ready {
                self.pending_fusion = Some(key);
                break;
            }
        }
        self.fusion_contacts
            .retain(|_, state| state.last_seen_substep == current_substep);
    }

    pub(crate) fn take_pending_fusion(&mut self) -> Option<(usize, usize)> {
        self.pending_fusion.take()
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

    /// Current recoverable elastic strain energy stored in intact bonds (joules).
    pub fn elastic_bond_energy_j(&self) -> f64 {
        self.elastic_bond_energy_at_positions_j(&self.positions)
    }

    /// Elastic energy of bonds with at least one non-kinematic end. Bonds between
    /// two fixed REVs are excluded, so differences within one substep match
    /// `elastic_bond_energy_j` while the cost follows the moving part of a scene.
    pub(crate) fn mobile_elastic_bond_energy_j(&self) -> f64 {
        let mut energy_j = 0.0;
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor != EMPTY
                    && !(self.voxels[i].kinematic && self.voxels[bond.neighbor as usize].kinematic)
                {
                    energy_j += crate::solver::bond_stored_elastic_energy_at_positions_j(
                        self,
                        &self.positions,
                        i,
                        bond.neighbor as usize,
                        axis,
                    );
                }
            }
        }
        energy_j
    }

    /// Elastic strain energy evaluated against an earlier positional snapshot.
    /// Used only when a previously unconstrained solve discovers a new contact.
    pub(crate) fn elastic_bond_energy_at_positions_j(&self, positions: &[DVec3]) -> f64 {
        if positions.len() != self.voxels.len() {
            return 0.0;
        }
        let mut energy_j = 0.0;
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor != EMPTY {
                    energy_j += crate::solver::bond_stored_elastic_energy_at_positions_j(
                        self,
                        positions,
                        i,
                        bond.neighbor as usize,
                        axis,
                    );
                }
            }
        }
        energy_j
    }

    /// Current plastic work retained as stored energy in bond history (joules).
    pub fn stored_plastic_bond_energy_j(&self) -> f64 {
        let retained_fraction = 1.0 - self.config.fracture.plastic_heat_fraction as f64;
        self.bonds
            .iter()
            .flatten()
            .map(|bond| bond.plastic_work.max(0.0) as f64 * retained_fraction)
            .sum()
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

        let lattice_delta = (self.rest_positions[b] - self.rest_positions[a]) / DCELL;
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
        if !actual.is_finite() || (actual.length() - DCELL).abs() > DCELL * 1e-3 {
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
        let cell_coordinate = (self.rest_positions[lower][axis] / DCELL - 0.5).round() as i32;
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

    /// Default stress-free length of a manually authored bond (m).
    pub const DEFAULT_MANUAL_BOND_REST_LENGTH_M: f64 = DCELL;

    /// Create an explicit bond between two REVs that need not be face
    /// neighbours. `rest_length_m` is the stress-free centre distance along the
    /// bond axis (default 1 cm); `pretension_n` is an additional axial force the
    /// bond carries at creation (positive pulls the ends together). The stress
    /// is stored through `Bond::formation_offset`, so force and energy follow the
    /// ordinary constitutive law. Energy stored at creation (stretch plus
    /// pretension) is recorded as external work in `energy.external_work`.
    /// The bond axis is the lattice axis best aligned with the separation. The
    /// graph holds one bond per axis per REV on each side, so a request whose
    /// slot is occupied is rejected without changing the scene.
    pub fn connect_manual_bond(
        &mut self,
        a: usize,
        b: usize,
        rest_length_m: f64,
        pretension_n: f64,
    ) -> Result<(), String> {
        let count = self.voxels.len();
        if a >= count || b >= count {
            return Err(format!("manual bond REV index out of range ({a}, {b}; {count} REVs)"));
        }
        if a == b {
            return Err("manual bond cannot connect a REV to itself".into());
        }
        if !rest_length_m.is_finite() || rest_length_m <= 0.0 {
            return Err("manual bond rest length must be finite and positive".into());
        }
        if !pretension_n.is_finite() {
            return Err("manual bond pretension must be finite".into());
        }
        let separation = self.positions[b] - self.positions[a];
        if !separation.is_finite() || separation.length() < 1e-9 {
            return Err("manual bond endpoints coincide".into());
        }
        let frame_rotation = self.rotations[a].slerp(self.rotations[b], 0.5);
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        let (axis, projection) = (0..3)
            .map(|axis| (axis, separation.dot((frame_rotation * axes[axis]).as_dvec3())))
            .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
            .expect("three axes");
        let (lower, upper) = if projection >= 0.0 { (a, b) } else { (b, a) };
        let axis_name = ["x", "y", "z"][axis];
        if self.bonds[axis][lower].neighbor != EMPTY {
            return Err(format!(
                "REV {lower} already has a bond on its +{axis_name} side; the bond graph allows one bond per axis per REV"
            ));
        }
        if self.bonds[axis].iter().any(|bond| bond.neighbor as usize == upper) {
            return Err(format!(
                "REV {upper} already has a bond on its -{axis_name} side; the bond graph allows one bond per axis per REV"
            ));
        }
        if self.releases.iter().any(|release| {
            (release.i == lower && release.j == upper) || (release.i == upper && release.j == lower)
        }) {
            return Err("cannot connect a REV pair with a pending fracture release".into());
        }

        // Install with the stretch-free local offset (a 1 cm rest length equals
        // the lattice spacing), then add the stress implied by the rest length
        // and pretension. Axial offset shortens the unstressed bond when the
        // rest length is below the lattice spacing.
        let previous = self.bonds[axis][lower];
        self.bonds[axis][lower] = Bond {
            neighbor: upper as u32,
            formation_offset: [rest_length_m - DCELL, 0.0, 0.0],
            ..Bond::default()
        };
        let frame = glam::Mat3::from_cols(
            frame_rotation * axes[axis],
            frame_rotation * axes[(axis + 1) % 3],
            frame_rotation * axes[(axis + 2) % 3],
        );
        let stiffness = crate::solver::stiffness_matrix(self, lower, upper, frame);
        let mut offset = [rest_length_m - DCELL, 0.0, 0.0];
        if pretension_n != 0.0 {
            if stiffness == glam::Mat3::ZERO || stiffness.determinant().abs() < 1e-30 {
                self.bonds[axis][lower] = previous;
                return Err("manual bond endpoints have no stiffness to carry pretension".into());
            }
            let extra = stiffness.inverse() * Vec3::new(pretension_n as f32, 0.0, 0.0);
            offset[0] -= f64::from(extra.x);
            offset[1] -= f64::from(extra.y);
            offset[2] -= f64::from(extra.z);
        }
        if offset.iter().any(|value| !value.is_finite()) {
            self.bonds[axis][lower] = previous;
            return Err("manual bond formation offset is not finite".into());
        }
        self.bonds[axis][lower].formation_offset = offset;
        self.workspace[axis][lower] = BondWorkspace {
            rest_neighbor: upper as u32,
            ..BondWorkspace::default()
        };
        self.bonding_contacts
            .remove(&(lower.min(upper), lower.max(upper)));
        let cell_coordinate = (self.rest_positions[lower][axis] / DCELL - 0.5).round() as i32;
        let color = axis * 2 + cell_coordinate.rem_euclid(2) as usize;
        if !self.colors[color].contains(&lower) {
            self.colors[color].push(lower);
        }
        if !(self.voxels[lower].kinematic && self.voxels[upper].kinematic) {
            self.energy.external_work +=
                crate::solver::bond_stored_elastic_energy_j(self, lower, upper, axis);
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
        let lattice_delta = (self.rest_positions[b] - self.rest_positions[a]) / DCELL;
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
        if (normal_gap - DCELL).abs() > DCELL * 1e-3
            || tangential_offset.abs().max_element() >= DCELL
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
        let position =
            (self.positions[survivor] * mass_a + self.positions[consumed] * mass_b) / mass_total;
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
        self.fusion_contacts = std::mem::take(&mut self.fusion_contacts)
            .into_iter()
            .filter_map(|((i, j), state)| {
                (keep[i] && keep[j]).then_some(((remap[i] as usize, remap[j] as usize), state))
            })
            .collect();
        self.pending_fusion = None;
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
        coordinates: [Option<f64>; 3],
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
        // Removal shifts REV indices, which a beam region keeps.
        self.expand_all_beams();

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
        self.fusion_contacts = std::mem::take(&mut self.fusion_contacts)
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

        // Bond strain energy belongs to the topology being removed. Count each
        // stored directed bond once, even when both endpoints leave together.
        for axis in 0..3 {
            for i in 0..old_count {
                let bond = self.bonds[axis][i];
                let intact_neighbor = bond.neighbor;
                let rest_neighbor = self.workspace[axis][i].rest_neighbor;
                let removed_with_bond = !keep[i]
                    || (intact_neighbor != EMPTY && !keep[intact_neighbor as usize])
                    || (rest_neighbor != EMPTY && !keep[rest_neighbor as usize]);
                if !removed_with_bond {
                    continue;
                }
                if intact_neighbor != EMPTY {
                    let j = intact_neighbor as usize;
                    self.boundary_removals.elastic_bond_energy_j +=
                        crate::solver::bond_stored_elastic_energy_j(self, i, j, axis);
                }
                let retained_fraction = 1.0 - self.config.fracture.plastic_heat_fraction as f64;
                self.boundary_removals.stored_plastic_bond_energy_j +=
                    bond.plastic_work.max(0.0) as f64 * retained_fraction;
            }
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
        self.source_cells.retain(|_| {
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
        self.support_load_n.clear();
        self.precise_rotations.retain(|_| {
            let keep = keep[old_index];
            old_index += 1;
            keep
        });
        old_index = 0;
        self.angular_velocities.retain(|_| {
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

    /// Enable conservative mechanical time refinement. `config.substeps` stays
    /// the upper bound used for interactions. This never changes `fixed_dt`.
    pub fn set_adaptive_substeps(&mut self, enabled: bool) {
        if self.adaptive_substeps != enabled {
            self.wake_all_rigid_candidates();
            self.solver_scratch = Default::default();
            self.adaptive_substeps = enabled;
        }
    }

    pub fn adaptive_substeps(&self) -> bool {
        self.adaptive_substeps
    }

    pub fn last_mechanical_substeps(&self) -> usize {
        self.last_mechanical_substeps
    }

    /// Physical mass of every REV, including kinematic boundary material.
    pub fn total_mass_kg(&self) -> f64 {
        self.voxels.iter().map(|voxel| voxel.mass as f64).sum()
    }

    /// Build a conservative rigid-body snapshot for one closed, unconstrained
    /// bonded component. This does not mutate or deactivate the fine REVs;
    /// callers use it as the state transfer when the world-physics manager
    /// moves that component into its far-field aggregate tier.
    pub fn rigid_aggregate_snapshot(
        &self,
        members: &[usize],
    ) -> Result<RigidAggregate, AggregationError> {
        self.rigid_aggregate_snapshot_impl(members, true)
    }

    fn rigid_aggregate_snapshot_impl(
        &self,
        members: &[usize],
        validate_bond_graph: bool,
    ) -> Result<RigidAggregate, AggregationError> {
        if members.is_empty() {
            return Err(AggregationError::Empty);
        }
        let mut member_set =
            validate_bond_graph.then(|| std::collections::HashSet::with_capacity(members.len()));
        for &index in members {
            if index >= self.voxels.len() {
                return Err(AggregationError::IndexOutOfBounds(index));
            }
            if member_set
                .as_mut()
                .is_some_and(|members| !members.insert(index))
            {
                return Err(AggregationError::DuplicateIndex(index));
            }
            if self.voxels[index].kinematic || self.axis_targets[index].iter().any(Option::is_some)
            {
                return Err(AggregationError::ConstrainedMember(index));
            }
        }

        if let Some(member_set) = member_set.as_ref() {
            // The public snapshot API validates connectivity against the live
            // bond graph. The automatic path above instead trusts the freshly
            // rebuilt component labels and validates membership there.
            let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::with_capacity(members.len());
            for axis in 0..3 {
                for (i, bond) in self.bonds[axis].iter().enumerate() {
                    if bond.neighbor == EMPTY {
                        continue;
                    }
                    let j = bond.neighbor as usize;
                    if j >= self.voxels.len() {
                        return Err(AggregationError::CrossingBond);
                    }
                    let inside_i = member_set.contains(&i);
                    let inside_j = member_set.contains(&j);
                    if inside_i != inside_j {
                        return Err(AggregationError::CrossingBond);
                    }
                    if inside_i {
                        adjacency.entry(i).or_default().push(j);
                        adjacency.entry(j).or_default().push(i);
                    }
                }
            }
            let mut visited = std::collections::HashSet::with_capacity(members.len());
            let mut stack = vec![members[0]];
            while let Some(index) = stack.pop() {
                if visited.insert(index) {
                    if let Some(neighbors) = adjacency.get(&index) {
                        stack.extend(neighbors.iter().copied());
                    }
                }
            }
            if visited.len() != member_set.len() {
                return Err(AggregationError::DisconnectedSet);
            }
        }

        let constituents: Vec<_> = members
            .iter()
            .map(|&index| {
                let voxel = &self.voxels[index];
                let position = self.positions[index];
                let rest_position = self.rest_positions[index];
                let previous_position = self.previous[index];
                let orientation = self.rotations[index];
                ConstituentState {
                    index,
                    mass_kg: voxel.mass as f64,
                    position_m: DVec3::new(position.x as f64, position.y as f64, position.z as f64),
                    rest_position_m: DVec3::new(
                        rest_position.x as f64,
                        rest_position.y as f64,
                        rest_position.z as f64,
                    ),
                    previous_position_m: DVec3::new(
                        previous_position.x as f64,
                        previous_position.y as f64,
                        previous_position.z as f64,
                    ),
                    velocity_m_s: DVec3::new(
                        voxel.velocity.x as f64,
                        voxel.velocity.y as f64,
                        voxel.velocity.z as f64,
                    ),
                    orientation: glam::DQuat::from_xyzw(
                        orientation.x as f64,
                        orientation.y as f64,
                        orientation.z as f64,
                        orientation.w as f64,
                    ),
                    bond_last_force_n: std::array::from_fn(|axis| {
                        let force = self.workspace[axis][index].last_force;
                        DVec3::new(force.x as f64, force.y as f64, force.z as f64)
                    }),
                    thermal_energy_j: voxel.total_internal_energy_j(),
                    thermal_cell_energy_j: voxel.thermal_cell_energy_j.clone(),
                }
            })
            .collect();
        RigidAggregate::from_constituents(&constituents)
    }

    /// Restore one refined aggregate into the existing 1 cm REV arrays.
    /// Validation is completed for the whole component before any scene state
    /// changes, so stale indices or a changed topology cannot partially move a
    /// body. Material, bond histories, and external inputs remain on the REVs.
    pub fn refine_rigid_aggregate(
        &mut self,
        aggregate: &RigidAggregate,
    ) -> Result<usize, AggregationError> {
        let refined = aggregate.refine();
        let mut member_set = std::collections::HashSet::with_capacity(refined.len());
        let f32_limit = f32::MAX as f64;
        let vector_f32_representable = |value: DVec3| {
            value.is_finite()
                && value
                    .to_array()
                    .iter()
                    .all(|component| component.abs() <= f32_limit)
        };
        for state in &refined {
            if state.index >= self.voxels.len() {
                return Err(AggregationError::IndexOutOfBounds(state.index));
            }
            if !member_set.insert(state.index) {
                return Err(AggregationError::DuplicateIndex(state.index));
            }
            let voxel = &self.voxels[state.index];
            if (voxel.mass as f64) != state.mass_kg {
                return Err(AggregationError::InvalidMass(state.index));
            }
            if voxel.kinematic || self.axis_targets[state.index].iter().any(Option::is_some) {
                return Err(AggregationError::ConstrainedMember(state.index));
            }
            if !vector_f32_representable(state.position_m)
                || !vector_f32_representable(state.rest_position_m)
                || !vector_f32_representable(state.previous_position_m)
                || !vector_f32_representable(state.velocity_m_s)
                || !state.orientation.is_finite()
                || state.orientation.length_squared() <= 1e-24
                || state
                    .bond_last_force_n
                    .iter()
                    .any(|force| !vector_f32_representable(*force))
            {
                return Err(AggregationError::InvalidState(state.index));
            }
            if !state.thermal_energy_j.is_finite() || state.thermal_energy_j < 0.0 {
                return Err(AggregationError::InvalidThermalEnergy(state.index));
            }
            if state.thermal_cell_energy_j.as_ref().is_some_and(|cells| {
                cells
                    .iter()
                    .any(|energy| !energy.is_finite() || *energy < 0.0)
            }) {
                return Err(AggregationError::InvalidThermalCells(state.index));
            }
        }

        // Revalidate current topology, not just the topology present when the
        // aggregate was created: internal fracture or a new boundary bond
        // invalidates the old rigid representation.
        let current_members: Vec<_> = refined.iter().map(|state| state.index).collect();
        self.rigid_aggregate_snapshot(&current_members)?;

        let to_vec3 = |value: DVec3| Vec3::new(value.x as f32, value.y as f32, value.z as f32);
        for state in refined {
            let index = state.index;
            self.positions[index] = state.position_m;
            self.rest_positions[index] = state.rest_position_m;
            self.previous[index] = state.previous_position_m;
            self.voxels[index].velocity = to_vec3(state.velocity_m_s);
            self.rotations[index] = Quat::from_xyzw(
                state.orientation.x as f32,
                state.orientation.y as f32,
                state.orientation.z as f32,
                state.orientation.w as f32,
            )
            .normalize();
            for axis in 0..3 {
                self.workspace[axis][index].last_force = to_vec3(state.bond_last_force_n[axis]);
            }
            self.voxels[index].thermal_cell_energy_j = state.thermal_cell_energy_j;
            let energy = state.thermal_energy_j.min(f32::MAX as f64);
            self.voxels[index].internal_energy = energy as f32;
            self.voxels[index].internal_energy_remainder_j =
                energy - self.voxels[index].internal_energy as f64;
        }
        self.mark_render_changed();
        Ok(member_set.len())
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

    /// Conduct heat through 3×3×3 thermal control volumes inside each
    /// mechanical REV. Internal cell faces remain conductive when the REV is
    /// mechanically detached; cross-REV faces respect bond damage. Pairwise
    /// fluxes are equal and opposite, and subcycling enforces the explicit
    /// diffusion stability bound.
    /// The returned value is the number of thermal substeps used.
    pub fn conduct_heat(&mut self, elapsed_seconds: f32) -> Result<usize, &'static str> {
        if !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
            return Err("thermal elapsed time must be finite and nonnegative");
        }
        if elapsed_seconds == 0.0 || self.voxels.is_empty() {
            return Ok(0);
        }

        let voxel_count = self.voxels.len();
        let cells_per_voxel = crate::voxel::THERMAL_CELL_COUNT;
        let scratch = &mut self.thermal_scratch;
        scratch.macro_temperature.resize(voxel_count, 0.0);
        scratch.active_mask.resize(voxel_count, false);
        scratch.active_seed.resize(voxel_count, false);
        scratch.active_mask.fill(false);
        for (i, voxel) in self.voxels.iter().enumerate() {
            scratch.macro_temperature[i] =
                self.properties[i].temperature(voxel.mass, voxel.thermal_energy_f32());
            scratch.active_mask[i] = voxel.thermal_cell_energy_j.is_some();
        }

        // Seed the thermal workset only where there is an actual temperature
        // gradient or a retained intra-REV gradient. Uniform, equilibrated
        // regions need no per-cell updates. This keeps the 27-cell model
        // tractable in large worlds without changing any active flux.
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY
                    || bond.damage >= 1.0
                    || self.properties[i].thermal_cond <= 0.0
                {
                    continue;
                }
                let j = bond.neighbor as usize;
                if self.properties[j].thermal_cond <= 0.0 {
                    continue;
                }
                if scratch.macro_temperature[i] != scratch.macro_temperature[j] {
                    scratch.active_mask[i] = true;
                    scratch.active_mask[j] = true;
                }
            }
        }
        if !scratch.active_mask.iter().any(|&active| active) {
            return Ok(0);
        }

        // Include one connected layer around retained gradients, so the next
        // finite-volume flux can advance naturally across REV boundaries.
        scratch.active_seed.clone_from(&scratch.active_mask);
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY
                    || bond.damage >= 1.0
                    || self.properties[i].thermal_cond <= 0.0
                {
                    continue;
                }
                let j = bond.neighbor as usize;
                if self.properties[j].thermal_cond <= 0.0 {
                    continue;
                }
                if scratch.active_seed[i] || scratch.active_seed[j] {
                    scratch.active_mask[i] = true;
                    scratch.active_mask[j] = true;
                }
            }
        }
        scratch.active_voxels.clear();
        scratch
            .active_voxels
            .extend((0..voxel_count).filter(|&i| scratch.active_mask[i]));
        let active_voxel_count = scratch.active_voxels.len();
        let Some(thermal_cell_count) = active_voxel_count.checked_mul(cells_per_voxel) else {
            return Err("thermal active region is too large");
        };
        if thermal_cell_count > u32::MAX as usize {
            return Err("thermal grid exceeds the 32-bit edge index range");
        }
        scratch.active_slot.resize(voxel_count, usize::MAX);
        scratch.active_slot.fill(usize::MAX);
        for (slot, &voxel) in scratch.active_voxels.iter().enumerate() {
            scratch.active_slot[voxel] = slot;
            self.voxels[voxel].reconcile_thermal_cell_energy();
        }
        scratch.cell_energy.resize(thermal_cell_count, 0.0);
        for (slot, &voxel) in scratch.active_voxels.iter().enumerate() {
            let first = slot * cells_per_voxel;
            self.voxels[voxel].copy_thermal_cell_energies_j(
                &mut scratch.cell_energy[first..first + cells_per_voxel],
            );
        }

        let cells_per_axis = crate::voxel::THERMAL_CELLS_PER_AXIS;
        let subcell_length = CELL as f64 / cells_per_axis as f64;
        let axis_stride = [1usize, cells_per_axis, cells_per_axis * cells_per_axis];
        scratch
            .internal_conductance_sum
            .resize(thermal_cell_count, 0.0);
        let internal_graph_changed = scratch.cached_active_voxels != scratch.active_voxels
            || scratch.cached_conductivity.len() != active_voxel_count
            || scratch
                .cached_conductivity
                .iter()
                .zip(&scratch.active_voxels)
                .any(|(cached, &voxel)| {
                    cached.to_bits() != self.properties[voxel].thermal_cond.to_bits()
                });
        if internal_graph_changed {
            scratch.internal_edges.clear();
            scratch
                .internal_edges
                .reserve(active_voxel_count.saturating_mul(54));
            scratch.internal_conductance_sum.fill(0.0);
            for (slot, &voxel) in scratch.active_voxels.iter().enumerate() {
                let conductivity = self.properties[voxel].thermal_cond.max(0.0);
                if conductivity == 0.0 {
                    continue;
                }
                let conductance = conductivity * subcell_length as f32;
                let node_start = slot * cells_per_voxel;
                for z in 0..cells_per_axis {
                    for y in 0..cells_per_axis {
                        for x in 0..cells_per_axis {
                            let cell = x + cells_per_axis * (y + cells_per_axis * z);
                            let coordinates = [x, y, z];
                            for axis in 0..3 {
                                if coordinates[axis] + 1 == cells_per_axis {
                                    continue;
                                }
                                let a = node_start + cell;
                                let b = a + axis_stride[axis];
                                scratch.internal_edges.push(ThermalEdge {
                                    a: a as u32,
                                    b: b as u32,
                                    conductance,
                                });
                                scratch.internal_conductance_sum[a] += conductance as f64;
                                scratch.internal_conductance_sum[b] += conductance as f64;
                            }
                        }
                    }
                }
            }
            scratch
                .cached_active_voxels
                .clone_from(&scratch.active_voxels);
            scratch.cached_conductivity.clear();
            scratch.cached_conductivity.extend(
                scratch
                    .active_voxels
                    .iter()
                    .map(|&voxel| self.properties[voxel].thermal_cond),
            );
        }
        scratch.conductance_sum.resize(thermal_cell_count, 0.0);
        scratch.temperature.resize(thermal_cell_count, 0.0);
        scratch.energy_delta.resize(thermal_cell_count, 0.0);
        scratch
            .conductance_sum
            .copy_from_slice(&scratch.internal_conductance_sum);
        scratch.edges.clear();
        let edge_capacity = active_voxel_count.saturating_mul(27);
        if scratch.edges.capacity() < edge_capacity {
            scratch.edges.reserve(edge_capacity);
        }

        let interface_topology_changed = scratch.cached_interface_voxels != scratch.active_voxels
            || scratch.cached_interface_conductivity.len() != active_voxel_count
            || scratch
                .cached_interface_conductivity
                .iter()
                .zip(&scratch.active_voxels)
                .any(|(cached, &voxel)| *cached != self.properties[voxel].thermal_cond.to_bits())
            || scratch.cached_bond_neighbors.len() != voxel_count
            || scratch
                .cached_bond_neighbors
                .iter()
                .enumerate()
                .any(|(i, cached)| {
                    *cached
                        != [
                            self.bonds[0][i].neighbor,
                            self.bonds[1][i].neighbor,
                            self.bonds[2][i].neighbor,
                        ]
                });
        if interface_topology_changed {
            scratch.interface_faces.clear();
            for axis in 0..3 {
                let side_axes: [usize; 2] = match axis {
                    0 => [1, 2],
                    1 => [0, 2],
                    _ => [0, 1],
                };
                for (i, bond) in self.bonds[axis].iter().enumerate() {
                    if bond.neighbor == EMPTY {
                        continue;
                    }
                    let j = bond.neighbor as usize;
                    let slot_i = scratch.active_slot[i];
                    let slot_j = scratch.active_slot[j];
                    if slot_i == usize::MAX || slot_j == usize::MAX {
                        continue;
                    }
                    let ki = self.properties[i].thermal_cond.max(0.0) as f64;
                    let kj = self.properties[j].thermal_cond.max(0.0) as f64;
                    if ki == 0.0 || kj == 0.0 {
                        continue;
                    }
                    let base_conductance = (2.0 * ki * kj / (ki + kj)) * subcell_length;
                    let mut lower_nodes = [0u32; 9];
                    let mut upper_nodes = [0u32; 9];
                    let mut pair = 0;
                    for a in 0..cells_per_axis {
                        for b in 0..cells_per_axis {
                            let mut lower = [0usize; 3];
                            lower[axis] = cells_per_axis - 1;
                            lower[side_axes[0]] = a;
                            lower[side_axes[1]] = b;
                            let mut upper = lower;
                            upper[axis] = 0;
                            let lower_cell =
                                lower[0] + cells_per_axis * (lower[1] + cells_per_axis * lower[2]);
                            let upper_cell =
                                upper[0] + cells_per_axis * (upper[1] + cells_per_axis * upper[2]);
                            lower_nodes[pair] = (slot_i * cells_per_voxel + lower_cell) as u32;
                            upper_nodes[pair] = (slot_j * cells_per_voxel + upper_cell) as u32;
                            pair += 1;
                        }
                    }
                    scratch.interface_faces.push(ThermalFace {
                        lower: lower_nodes,
                        upper: upper_nodes,
                        axis: axis as u8,
                        bond_index: i as u32,
                        base_conductance,
                    });
                }
            }
            scratch
                .cached_interface_voxels
                .clone_from(&scratch.active_voxels);
            scratch.cached_interface_conductivity.clear();
            scratch.cached_interface_conductivity.extend(
                scratch
                    .active_voxels
                    .iter()
                    .map(|&voxel| self.properties[voxel].thermal_cond.to_bits()),
            );
            scratch.cached_bond_neighbors.clear();
            scratch
                .cached_bond_neighbors
                .extend((0..voxel_count).map(|i| {
                    [
                        self.bonds[0][i].neighbor,
                        self.bonds[1][i].neighbor,
                        self.bonds[2][i].neighbor,
                    ]
                }));
        }

        // Connectivity and undamaged conductance are cached. Current bond
        // damage remains live and scales all nine face fluxes every update.
        for face in &scratch.interface_faces {
            let damage = self.bonds[face.axis as usize][face.bond_index as usize].damage;
            let conductance =
                (face.base_conductance * (1.0 - damage as f64).clamp(0.0, 1.0)) as f32;
            if conductance == 0.0 {
                continue;
            }
            for pair in 0..9 {
                let lower = face.lower[pair] as usize;
                let upper = face.upper[pair] as usize;
                scratch.edges.push(ThermalEdge {
                    a: lower as u32,
                    b: upper as u32,
                    conductance,
                });
                scratch.conductance_sum[lower] += conductance as f64;
                scratch.conductance_sum[upper] += conductance as f64;
            }
        }

        if scratch.edges.is_empty() && scratch.internal_edges.is_empty() {
            return Ok(0);
        }

        let mut maximum_rate = 0.0f64;
        for (slot, &world_voxel) in scratch.active_voxels.iter().enumerate() {
            let cell_mass =
                self.voxels[world_voxel].mass.max(1e-20) as f64 / cells_per_voxel as f64;
            let cell_capacity =
                cell_mass * self.properties[world_voxel].heat_capacity.max(1e-12) as f64;
            let first_node = slot * cells_per_voxel;
            for node in first_node..first_node + cells_per_voxel {
                maximum_rate = maximum_rate.max(scratch.conductance_sum[node] / cell_capacity);
            }
        }
        let thermal_steps = ((elapsed_seconds as f64 * maximum_rate / 0.45).ceil() as usize).max(1);
        if thermal_steps > 100_000 {
            return Err("thermal update requires more than 100000 stability substeps");
        }
        let dt = elapsed_seconds as f64 / thermal_steps as f64;
        for _ in 0..thermal_steps {
            if active_voxel_count >= 128 {
                let active_voxels = &scratch.active_voxels;
                let cell_energies = &scratch.cell_energy;
                let temperatures = &mut scratch.temperature[..thermal_cell_count];
                let voxels = &self.voxels;
                let properties = &self.properties;
                temperatures
                    .par_chunks_mut(cells_per_voxel)
                    .enumerate()
                    .for_each(|(slot, output)| {
                        let voxel = active_voxels[slot];
                        let cell_mass = voxels[voxel].mass / cells_per_voxel as f32;
                        let property = &properties[voxel];
                        let first_node = slot * cells_per_voxel;
                        for (cell, temperature) in output.iter_mut().enumerate() {
                            let energy =
                                cell_energies[first_node + cell].clamp(0.0, f32::MAX as f64) as f32;
                            *temperature = property.temperature(cell_mass, energy) as f64;
                        }
                    });
            } else {
                for (slot, &voxel) in scratch.active_voxels.iter().enumerate() {
                    let cell_mass = self.voxels[voxel].mass / cells_per_voxel as f32;
                    let property = &self.properties[voxel];
                    let first_node = slot * cells_per_voxel;
                    for node in first_node..first_node + cells_per_voxel {
                        let energy = scratch.cell_energy[node].clamp(0.0, f32::MAX as f64) as f32;
                        scratch.temperature[node] = property.temperature(cell_mass, energy) as f64;
                    }
                }
            }
            scratch.energy_delta.fill(0.0);
            for edge in scratch.internal_edges.iter().chain(&scratch.edges) {
                let i = edge.a as usize;
                let j = edge.b as usize;
                let conductance = edge.conductance as f64;
                let heat = conductance * (scratch.temperature[j] - scratch.temperature[i]) * dt;
                scratch.energy_delta[i] += heat;
                scratch.energy_delta[j] -= heat;
            }
            for (energy, delta) in scratch.cell_energy.iter_mut().zip(&scratch.energy_delta) {
                *energy += delta;
            }
        }
        for (slot, &voxel) in scratch.active_voxels.iter().enumerate() {
            let first = slot * cells_per_voxel;
            self.voxels[voxel]
                .set_thermal_cell_energies_j(&scratch.cell_energy[first..first + cells_per_voxel]);
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
        let oak_component = Material::component_id("oak").expect("built-in oak profile");
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
                    DVec3::new(key[0] as f64, key[1] as f64, key[2] as f64)
                        * crate::coordinates::CELL_M
                        + DVec3::splat(crate::coordinates::CELL_M * 0.5),
                );
                voxels.push(Voxel {
                    mass,
                    kinematic: fixed,
                    composition: Composition::pure(component),
                    internal_energy: m.energy_at_temperature(mass, 293.15),
                    internal_energy_remainder_j: 0.0,
                    thermal_cell_energy_j: None,
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
                        oak_component,
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
                        oak_component,
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
            simulated_seconds: 0.0,
            last_step_diagnostics: None,
            adaptive_substeps: false,
            last_mechanical_substeps: 0,
            thermal_scratch: Default::default(),
            collision_scratch: Default::default(),
            rigid_aggregates: Vec::new(),
            aggregate_of: vec![None; n],
            source_cells: vec![None; n],
            external_forces: vec![Vec3::ZERO; n],
            skeletal_actuators: Vec::new(),
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
            chemistry_enabled: false,
            chemistry_interval_s: 0.0,
            chemistry_pending_s: 0.0,
            chemistry_registry: None,
            cell_dofs: false,
            beam_regions: Vec::new(),
            support_load_n: Vec::new(),
            precise_rotations: vec![glam::DQuat::IDENTITY; n],
            angular_velocities: vec![glam::DVec3::ZERO; n],
            clusters: vec![0; n],
            broken: 0,
            cluster_count: 1,
            ground_top: 0.0,
            energy: Default::default(),
            releases: Vec::new(),
            substep_index: 0,
            render_revision: 0,
            solver_residual: 0.0,
            solver_residual_audit: None,
            solver_max_audited_residual: None,
            world_bounds: None,
            thermal_bonding_rule: None,
            bonding_contacts: HashMap::new(),
            thermal_fusion_rule: None,
            fusion_contacts: HashMap::new(),
            pending_fusion: None,
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
mod rigid_aggregate_translation_pose_tests {
    use super::*;

    #[test]
    fn pure_translation_updates_only_translational_rev_state() {
        let oak = Material::component_id("oak").unwrap();
        let velocity = Vec3::new(0.25, -0.5, 0.75);
        let mut builder = SceneBuilder::standard().with_gravity(Vec3::ZERO);
        let mut members = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let cell = [x, y, z];
                    members.push(members.len());
                    builder =
                        builder.add_voxel(VoxelSpawn::at_cell(cell, oak).with_velocity(velocity));
                }
            }
        }
        let mut scene = builder.build().unwrap();

        // These are retained material/frame quantities. A pure translation
        // must not transform or rewrite either array.
        for index in 0..scene.voxels.len() {
            scene.rotations[index] = Quat::from_rotation_y(index as f32 * 0.07 + 0.1);
            for axis in 0..3 {
                scene.workspace[axis][index].last_force =
                    Vec3::new(index as f32 + 1.0, axis as f32 + 2.0, 0.5);
            }
        }
        scene.register_rigid_aggregate_candidate(&members).unwrap();

        let positions_before = scene.positions.clone();
        let rest_positions_before = scene.rest_positions.clone();
        let rotations_before = scene.rotations.clone();
        let last_forces_before: Vec<_> = scene
            .workspace
            .iter()
            .map(|axis| axis.iter().map(|bond| bond.last_force).collect::<Vec<_>>())
            .collect();
        let dt_s = 0.01;
        let gravity = DVec3::new(0.0, 0.0, -9.81);
        let orientation_before = scene
            .registered_rigid_aggregate_candidate(0)
            .unwrap()
            .orientation();

        assert_eq!(
            scene
                .advance_registered_rigid_candidates(dt_s, gravity, &[], None)
                .unwrap(),
            1
        );

        let displacement = DVec3::new(velocity.x as f64, velocity.y as f64, velocity.z as f64)
            * dt_s
            + gravity * (0.5 * dt_s * dt_s);
        let expected_velocity = velocity + Vec3::new(0.0, 0.0, (gravity.z * dt_s) as f32);
        for index in 0..scene.voxels.len() {
            assert!(
                (scene.positions[index] - (positions_before[index] + displacement)).length()
                    < 1e-14
            );
            assert!(
                (scene.rest_positions[index] - (rest_positions_before[index] + displacement))
                    .length()
                    < 1e-14
            );
            assert_eq!(scene.voxels[index].velocity, expected_velocity);
            assert_eq!(scene.rotations[index], rotations_before[index]);
        }
        let aggregate = scene.registered_rigid_aggregate_candidate(0).unwrap();
        assert_eq!(aggregate.invariants().angular_momentum_kg_m2_s, DVec3::ZERO);
        assert_eq!(aggregate.orientation(), orientation_before);
        for (axis, forces) in scene.workspace.iter().zip(last_forces_before) {
            assert!(axis
                .iter()
                .zip(forces)
                .all(|(bond, before)| bond.last_force == before));
        }
    }
}

#[cfg(test)]
mod world_bounds_tests {
    use super::*;

    #[test]
    fn fixed_step_removes_crossing_voxel_and_records_boundary_ledger() {
        let iron = Material::component_id("iron").unwrap();
        let spawn = VoxelSpawn::at_cell([0, 0, 0], iron)
            .with_velocity(Vec3::new(2.0, 0.0, 0.0))
            .at_temperature(325.0);
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .with_timestep(0.001, 1)
            .add_voxel(spawn.clone())
            .build()
            .unwrap();
        // The removal ledger records the state at exit, after this step's
        // real forces (including ambient drag), not the initial launch state.
        // Compare with the same physical step without a removal boundary.
        let mut reference = Scene::from_voxels(scene.config.clone(), vec![spawn]).unwrap();
        reference.tick();
        let voxel = &reference.voxels[0];
        let expected_mass = voxel.mass as f64;
        let expected_internal = voxel.total_internal_energy_j();
        let expected_kinetic = 0.5 * expected_mass * voxel.velocity.length_squared() as f64;
        let expected_momentum = voxel.velocity.to_array().map(|v| expected_mass * v as f64);
        scene
            .set_world_bounds(Vec3::splat(-1.0), Vec3::new(0.006, 1.0, 1.0))
            .unwrap();

        scene.tick();

        assert!(scene.voxels.is_empty());
        let ledger = scene.boundary_removals();
        assert_eq!(ledger.voxel_count, 1);
        assert!((ledger.mass_kg - expected_mass).abs() < 1e-12);
        assert!((ledger.internal_energy_j - expected_internal).abs() < 1e-9);
        assert!((ledger.kinetic_energy_j - expected_kinetic).abs() < 1e-12);
        assert_eq!(ledger.momentum_kg_m_s, expected_momentum);
    }

    #[test]
    fn boundary_removal_transfers_intact_bond_strain_energy_once() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([2, 0, 0], iron))
            .build()
            .unwrap();
        scene.positions[1].x += 0.001;
        scene.positions[2].x += 0.002;
        scene.bonds[0][0].damage = 0.25;
        scene.bonds[0][1].damage = 0.25;
        scene.bonds[0][0].plastic_work = 1.25;
        scene.bonds[0][1].plastic_work = 2.5;
        let k = crate::solver::stiffness_matrix(&scene, 0, 1, glam::Mat3::IDENTITY);
        let strain_a = glam::Vec3::X * (scene.positions[1].x - scene.positions[0].x - DCELL) as f32;
        let strain_b = glam::Vec3::X * (scene.positions[2].x - scene.positions[1].x - DCELL) as f32;
        let expected_energy =
            (0.5 * 0.75 * (strain_a.dot(k * strain_a) + strain_b.dot(k * strain_b))).max(0.0)
                as f64;
        assert!(expected_energy > 0.0);
        assert!((scene.elastic_bond_energy_j() - expected_energy).abs() < expected_energy * 1e-5);
        let expected_stored_plastic =
            3.75 * (1.0 - scene.config.fracture.plastic_heat_fraction as f64);
        assert!((scene.stored_plastic_bond_energy_j() - expected_stored_plastic).abs() < 1e-12);

        assert_eq!(
            scene.set_world_bounds(Vec3::new(0.02, 0.0, 0.0), Vec3::splat(1.0)),
            Ok(2)
        );

        assert_eq!(scene.boundary_removals().voxel_count, 2);
        assert_eq!(scene.voxels.len(), 1);
        assert!(
            (scene.boundary_removals().elastic_bond_energy_j - expected_energy).abs()
                < expected_energy * 1.0e-5,
            "ledger={} expected={expected_energy}",
            scene.boundary_removals().elastic_bond_energy_j
        );
        assert_eq!(scene.elastic_bond_energy_j(), 0.0);
        assert_eq!(scene.stored_plastic_bond_energy_j(), 0.0);
        assert!(
            (scene.boundary_removals().stored_plastic_bond_energy_j - expected_stored_plastic)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn boundary_removal_transfers_stored_plastic_work_from_broken_bond() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        scene.bonds[0][0].neighbor = EMPTY;
        scene.bonds[0][0].damage = 1.0;
        scene.bonds[0][0].plastic_work = 2.0;
        assert!((scene.stored_plastic_bond_energy_j() - 0.2).abs() < 1e-7);

        assert_eq!(
            scene.set_world_bounds(Vec3::new(0.01, 0.0, 0.0), Vec3::splat(1.0)),
            Ok(1)
        );

        assert_eq!(scene.boundary_removals().elastic_bond_energy_j, 0.0);
        assert!((scene.boundary_removals().stored_plastic_bond_energy_j - 0.2).abs() < 1e-7);
        assert_eq!(scene.stored_plastic_bond_energy_j(), 0.0);
    }

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

        // Give each survivor distinguishable indexed state so this regression
        // checks the remap, not only the array lengths.
        scene.rest_positions[1] += DVec3::new(0.0, 0.001, 0.0);
        scene.rest_positions[2] += DVec3::new(0.0, 0.002, 0.0);
        scene.previous[1] += DVec3::new(0.0, 0.003, 0.0);
        scene.previous[2] += DVec3::new(0.0, 0.004, 0.0);
        scene.external_forces[1] = Vec3::new(1.0, 2.0, 3.0);
        scene.external_forces[2] = Vec3::new(4.0, 5.0, 6.0);
        scene.external_accelerations[1] = Vec3::new(7.0, 8.0, 9.0);
        scene.external_accelerations[2] = Vec3::new(10.0, 11.0, 12.0);
        scene.axis_targets[1] = [Some(0.011), None, Some(0.013)];
        scene.axis_targets[2] = [None, Some(0.022), Some(0.023)];
        scene.rotations[1] = glam::Quat::from_rotation_z(0.2);
        scene.rotations[2] = glam::Quat::from_rotation_y(-0.3);
        scene.properties[1].thermal_cond = 123.0;
        scene.properties[2].thermal_cond = 456.0;
        let survivor_rest_positions = [scene.rest_positions[1], scene.rest_positions[2]];
        let survivor_previous = [scene.previous[1], scene.previous[2]];
        let survivor_rotations = [scene.rotations[1], scene.rotations[2]];
        let survivor_conductivities = [
            scene.properties[1].thermal_cond,
            scene.properties[2].thermal_cond,
        ];

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
        assert_eq!(scene.rest_positions, survivor_rest_positions);
        assert_eq!(scene.previous, survivor_previous);
        assert_eq!(
            scene.external_forces,
            vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)]
        );
        assert_eq!(
            scene.external_accelerations,
            vec![Vec3::new(7.0, 8.0, 9.0), Vec3::new(10.0, 11.0, 12.0)]
        );
        assert_eq!(scene.axis_targets[0], [Some(0.011), None, Some(0.013)]);
        assert_eq!(scene.axis_targets[1], [None, Some(0.022), Some(0.023)]);
        assert_eq!(scene.rotations, survivor_rotations);
        assert_eq!(
            [
                scene.properties[0].thermal_cond,
                scene.properties[1].thermal_cond
            ],
            survivor_conductivities
        );
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
    fn boundary_compaction_rebuilds_collision_candidates_in_new_index_space() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = Scene::from_voxels(
            SceneConfig::default(),
            vec![
                VoxelSpawn::at_cell([0, 0, 0], iron),
                VoxelSpawn::at_cell([1, 0, 0], iron),
                VoxelSpawn::at_cell([2, 0, 0], iron),
            ],
        )
        .unwrap();
        // Separate the second pair mechanically so broadphase must retain its
        // touching contact before and after the first REV is removed.
        scene.bonds[0][1].neighbor = EMPTY;
        let before = crate::collision::project(&mut scene, &[]);
        assert!(before
            .iter()
            .any(|contact| (contact.i, contact.j) == (1, 2)));

        assert_eq!(
            scene.set_world_bounds(Vec3::new(0.01, 0.0, 0.0), Vec3::splat(1.0)),
            Ok(1)
        );
        let after = crate::collision::project(&mut scene, &[]);
        assert!(after.iter().any(|contact| (contact.i, contact.j) == (0, 1)));
        assert!(after
            .iter()
            .all(|contact| contact.i < scene.voxels.len() && contact.j < scene.voxels.len()));
    }

    #[test]
    fn boundary_removal_rebuilds_clusters_after_an_interior_path_exits() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 1, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 2, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([0, 2, 0], iron))
            .build()
            .unwrap();
        assert_eq!(scene.cluster_count, 1);

        // The two endpoint REV stay in bounds while the three-REV detour
        // leaves through X+, severing the only path that connects them.
        assert_eq!(
            scene.set_world_bounds(Vec3::new(0.004, -1.0, -1.0), Vec3::new(0.006, 1.0, 1.0)),
            Ok(3)
        );
        assert_eq!(scene.voxels.len(), 2);
        assert_eq!(scene.boundary_removals().voxel_count, 3);
        assert_eq!(scene.cluster_count, 2);
        assert_ne!(scene.clusters[0], scene.clusters[1]);
        assert!(scene
            .bonds
            .iter()
            .all(|axis| axis.iter().all(|bond| bond.neighbor == EMPTY)));
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
        scene.fusion_contacts.insert(
            (1, 2),
            BondingContactState {
                dwell_seconds: 0.125,
                impact_qualified: false,
                last_seen_substep: 7,
            },
        );
        assert_eq!(
            scene.set_world_bounds(Vec3::new(0.01, 0.0, 0.0), Vec3::splat(1.0)),
            Ok(1)
        );
        assert_eq!(scene.bonding_contacts.len(), 1);
        assert_eq!(scene.bonding_contacts[&(0, 1)].dwell_seconds, 0.25);
        assert_eq!(scene.fusion_contacts.len(), 1);
        assert_eq!(scene.fusion_contacts[&(0, 1)].dwell_seconds, 0.125);
        assert!(scene
            .set_world_bounds(Vec3::new(0.02, 0.0, 0.0), Vec3::splat(1.0))
            .is_ok());
        assert!(scene.bonding_contacts.is_empty());
        assert!(scene.fusion_contacts.is_empty());
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
    fn thermal_subcells_resolve_face_gradients_and_conserve_energy() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).at_temperature(450.0))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(300.0))
            .build()
            .unwrap();
        let initial_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum();

        let thermal_steps = scene.conduct_heat(0.01).unwrap();

        let hot_face = [2usize, 5, 8];
        let hot_interior = [0usize, 3, 6];
        let cold_face = [0usize, 3, 6];
        let cold_interior = [2usize, 5, 8];
        let mean_cell_energy = |voxel: usize, cells: &[usize]| {
            cells
                .iter()
                .map(|&cell| scene.voxels[voxel].thermal_cell_energy_at(cell))
                .sum::<f64>()
                / cells.len() as f64
        };
        let final_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum();

        assert!(thermal_steps > 0);
        assert!(mean_cell_energy(0, &hot_face) < mean_cell_energy(0, &hot_interior));
        assert!(mean_cell_energy(1, &cold_face) > mean_cell_energy(1, &cold_interior));
        assert!((final_energy - initial_energy).abs() < 1e-6);
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
    fn thermal_fusion_rule_requires_hot_pressurized_dwell_contact() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(
                VoxelSpawn::at_cell([0, 0, 0], iron)
                    .fixed()
                    .at_temperature(400.0),
            )
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(400.0))
            .build()
            .unwrap();
        scene.release_voxel(0).unwrap();
        assert!(scene.set_thermal_fusion_rule(f32::NAN, 0.0, 0.0).is_err());
        assert!(scene.set_thermal_fusion_rule(350.0, -1.0, 0.0).is_err());
        assert!(scene
            .set_thermal_fusion_rule(350.0, 1000.0, f32::INFINITY)
            .is_err());
        scene.set_thermal_fusion_rule(350.0, 1000.0, 0.02).unwrap();
        let contact = crate::collision::Contact {
            i: 0,
            j: 1,
            normal: Vec3::X,
            new_impact: false,
        };

        let targets = [[None; 3]; 2];
        let transient_targets = [[Some(0.005), None, None], [None; 3]];
        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &transient_targets);
        assert_eq!(scene.take_pending_fusion(), None);
        assert!(scene.fusion_contacts.is_empty());
        scene.form_thermal_fusions(&[contact], &[0.0005], 0.01, &targets); // 500 Pa
        assert_eq!(scene.take_pending_fusion(), None);
        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &targets); // 1500 Pa, 10 ms
        assert_eq!(scene.take_pending_fusion(), None);
        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &targets); // 1500 Pa, 20 ms
        assert_eq!(scene.take_pending_fusion(), Some((0, 1)));
        assert_eq!(scene.voxels.len(), 2);
        assert_eq!(
            scene
                .thermal_fusion_rule()
                .unwrap()
                .minimum_contact_pressure_pa,
            1000.0
        );
        scene.clear_thermal_fusion_rule();
        assert_eq!(scene.thermal_fusion_rule(), None);
        assert!(scene.fusion_contacts.is_empty());
    }

    #[test]
    fn thermal_fusion_dwell_resets_on_contact_loss_or_cooling() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).at_temperature(400.0))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(400.0))
            .build()
            .unwrap();
        scene.bonds[0][0] = Bond::default();
        scene.workspace[0][0] = BondWorkspace::default();
        scene.set_thermal_fusion_rule(350.0, 1000.0, 0.02).unwrap();
        let contact = crate::collision::Contact {
            i: 0,
            j: 1,
            normal: Vec3::X,
            new_impact: false,
        };
        let targets = [[None; 3]; 2];

        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &targets);
        assert_eq!(scene.take_pending_fusion(), None);
        assert_eq!(scene.fusion_contacts[&(0, 1)].dwell_seconds, 0.01);

        scene.substep_index += 1;
        scene.form_thermal_fusions(&[], &[], 0.01, &targets);
        assert!(scene.fusion_contacts.is_empty());
        scene.substep_index += 1;
        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &targets);
        assert_eq!(scene.take_pending_fusion(), None);
        assert_eq!(scene.fusion_contacts[&(0, 1)].dwell_seconds, 0.01);

        scene.substep_index += 1;
        scene.voxels[0].internal_energy = 0.0;
        scene.voxels[0].internal_energy_remainder_j = 0.0;
        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &targets);
        assert_eq!(scene.take_pending_fusion(), None);
        assert!(scene.fusion_contacts.is_empty());
    }

    #[test]
    fn thermal_bond_formation_takes_precedence_over_fusion() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).at_temperature(400.0))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(400.0))
            .build()
            .unwrap();
        scene.bonds[0][0] = Bond::default();
        scene.workspace[0][0] = BondWorkspace::default();
        crate::solver::rebuild_clusters(&mut scene);
        scene
            .set_thermal_bonding_rule("iron", 0.5, 350.0, 0.0, 1000.0, 0.0)
            .unwrap();
        scene.set_thermal_fusion_rule(350.0, 1000.0, 0.0).unwrap();
        let contact = crate::collision::Contact {
            i: 0,
            j: 1,
            normal: Vec3::X,
            new_impact: true,
        };
        let formed = scene.form_thermal_bonds(&[contact], &[0.0], &[0.0015], 0.01);
        assert_eq!(formed, 1);
        scene.form_thermal_fusions(&[contact], &[0.0015], 0.01, &[[None; 3]; 2]);
        assert_eq!(scene.take_pending_fusion(), None);
        assert_eq!(scene.bonds[0][0].neighbor, 1);
        assert_eq!(scene.voxels.len(), 2);
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
    fn viscosity_temperature_model_validates_and_refreshes_material_properties() {
        let water = Material::component_id("water").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], water).at_temperature(293.15))
            .build()
            .unwrap();
        assert_eq!(scene.properties[0].viscosity_activation_energy_j_mol, 0.0);
        scene
            .set_material_viscosity_temperature_model("water", 293.15, 17_000.0)
            .unwrap();
        let properties = &scene.properties[0];
        assert_eq!(properties.viscosity_reference_temperature_kelvin, 293.15);
        assert_eq!(properties.viscosity_activation_energy_j_mol, 17_000.0);
        assert!(
            properties.liquid_viscosity_at_temperature(333.15) < properties.dynamic_viscosity_pa_s
        );
        assert!(scene
            .set_material_viscosity_temperature_model("water", 0.0, 17_000.0)
            .is_err());
        assert!(scene
            .set_material_viscosity_temperature_model("water", 293.15, -1.0)
            .is_err());
        assert!(scene
            .set_material_viscosity_temperature_model("water", f32::INFINITY, 17_000.0)
            .is_err());
        assert!(scene
            .set_material_viscosity_temperature_model("unknown", 293.15, 17_000.0)
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
        assert!((scene.bonds[0][0].formation_offset[0] - (CELL - free_span) as f64).abs() < 1e-9);
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
        assert!((scene.positions[1].x - (scene.positions[0].x + DCELL)).abs() < 1e-6);
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
        scene.positions[1] = scene.positions[0] + DVec3::X * DCELL;
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

    fn manual_bond_scene(cells: &[[i32; 3]]) -> Scene {
        let iron = Material::component_id("iron").unwrap();
        let mut builder = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .with_timestep(0.0005, 2);
        for cell in cells {
            builder = builder.add_voxel(VoxelSpawn::at_cell(*cell, iron));
        }
        builder.build().unwrap()
    }

    fn ledger_energy_j(scene: &Scene) -> f64 {
        let voxels: f64 = scene
            .voxels
            .iter()
            .map(|v| {
                v.total_internal_energy_j()
                    + 0.5 * f64::from(v.mass) * f64::from(v.velocity.length_squared())
            })
            .sum();
        voxels + scene.elastic_bond_energy_j() + scene.stored_plastic_bond_energy_j()
            + scene.energy.pending_release
    }

    #[test]
    fn stretched_manual_bond_pulls_free_revs_together_within_external_work() {
        // Two free iron REVs 2 cm apart (centres) with a 1.98 cm rest length:
        // a 0.2 mm stretch that stays below the fracture limit. (The 1 cm
        // default would over-stretch iron here; see the default-rest test.)
        let mut scene = manual_bond_scene(&[[0, 0, 0], [2, 0, 0]]);
        let before = ledger_energy_j(&scene);
        let gap_before = (scene.positions[1] - scene.positions[0]).length();
        scene.connect_manual_bond(0, 1, 0.0198, 0.0).unwrap();
        let work = scene.energy.external_work;
        assert!(work > 0.0, "stretch must be booked as external work");
        let elastic = scene.elastic_bond_energy_j();
        assert!((elastic - work).abs() <= 1e-9 * work.max(1.0));
        for _ in 0..40 {
            scene.tick();
        }
        assert_eq!(scene.broken, 0);
        assert_ne!(scene.bonds[0][0].neighbor, EMPTY);
        let gap_after = (scene.positions[1] - scene.positions[0]).length();
        assert!(gap_after < gap_before, "{gap_after} !< {gap_before}");
        let momentum = scene.voxels[0].velocity * scene.voxels[0].mass
            + scene.voxels[1].velocity * scene.voxels[1].mass;
        assert!(momentum.length() < 1e-4 * scene.voxels[0].mass, "{momentum:?}");
        let after = ledger_energy_j(&scene);
        let external = scene.energy.external_work;
        // f32 voxel state carries ~1e-7 relative error on the ~2 kJ thermal total.
        let tolerance = 1e-6 * before;
        assert!(
            after - before <= external + tolerance,
            "total {after} - {before} exceeds external work {external}"
        );
        assert!(
            after - before >= external - tolerance,
            "energy lost beyond ledger"
        );
    }

    #[test]
    fn manual_bond_pretension_is_external_work_and_default_rest_is_one_cm() {
        assert_eq!(Scene::DEFAULT_MANUAL_BOND_REST_LENGTH_M, 0.01);
        let mut scene = manual_bond_scene(&[[0, 0, 0], [2, 0, 0]]);
        scene.connect_manual_bond(0, 1, 0.02, 50.0).unwrap();
        let k_work = scene.energy.external_work;
        assert!(k_work > 0.0);
        assert!((scene.elastic_bond_energy_j() - k_work).abs() < 1e-9 * k_work);
        // A bond with zero stretch and no pretension stores nothing.
        let mut relaxed = manual_bond_scene(&[[0, 0, 0], [2, 0, 0]]);
        relaxed.connect_manual_bond(0, 1, 0.02, 0.0).unwrap();
        assert_eq!(relaxed.energy.external_work, 0.0);
    }

    #[test]
    fn manual_bond_rejects_invalid_requests_without_changing_the_scene() {
        let mut scene = manual_bond_scene(&[[0, 0, 0], [3, 0, 0], [6, 0, 0], [-3, 0, 0]]);
        for (a, b, length, pretension) in [
            (0, 0, 0.01, 0.0),
            (0, 9, 0.01, 0.0),
            (0, 1, f64::NAN, 0.0),
            (0, 1, -0.01, 0.0),
            (0, 1, 0.0, 0.0),
            (0, 1, f64::INFINITY, 0.0),
            (0, 1, 0.01, f64::NAN),
        ] {
            assert!(
                scene.connect_manual_bond(a, b, length, pretension).is_err(),
                "{a} {b} {length} {pretension}"
            );
        }
        assert_eq!(scene.bonds[0][0].neighbor, EMPTY);
        assert_eq!(scene.energy.external_work, 0.0);
        scene.connect_manual_bond(0, 1, 0.03, 0.0).unwrap();
        // +x slot of REV 0 is taken, and REV 1's -x slot is taken.
        let slot = scene.connect_manual_bond(0, 2, 0.03, 0.0).unwrap_err();
        assert!(slot.contains("one bond per axis"), "{slot}");
        let incoming = scene.connect_manual_bond(3, 1, 0.03, 0.0).unwrap_err();
        assert!(incoming.contains("one bond per axis"), "{incoming}");
        assert_eq!(scene.bonds[0][0].neighbor, 1);
    }
}
