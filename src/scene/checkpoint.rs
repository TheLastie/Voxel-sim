//! Versioned on-disk checkpoint of a running physical scene.
//!
//! A `.vxstate` file holds everything a stopped simulation needs to continue:
//! positions, velocities, internal energy and thermal cells, bonds with damage and
//! plastic history, contact and friction state, ledgers, actuators and aggregates.
//! An authored `.vxscene` describes how a scene starts; this describes where a run
//! is now. Capacity-only workspaces and solver factors are rebuilt on the next step,
//! so a continued run can differ within solver tolerance from an uninterrupted one.
use super::*;
use crate::solver::SolverWarmState;
use std::{fs, io::Write, path::Path};

const MAGIC: &[u8; 8] = b"VXSTATE\0";
const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct ActuatorState {
    model: crate::documents::VoxelModelDocument,
    voxel_range: (usize, usize),
    total_scene_voxels: usize,
    instance_rotation: Quat,
    runtime: crate::skeletal_actuator::ActuatorRuntime,
}

/// Complete physical state of a scene. Fields follow `Scene`.
#[derive(Serialize, Deserialize)]
struct SceneCheckpoint {
    simulated_seconds: f64,
    adaptive_substeps: bool,
    aggregates: Vec<Option<RigidAggregate>>,
    aggregate_of: Vec<Option<usize>>,
    config: SceneConfig,
    voxels: Vec<Voxel>,
    source_cells: Vec<Option<[i32; 3]>>,
    external_forces: Vec<Vec3>,
    external_accelerations: Vec<Vec3>,
    axis_targets: Vec<[Option<f64>; 3]>,
    positions: Vec<DVec3>,
    rest_positions: Vec<DVec3>,
    reference_temperatures_kelvin: Vec<f32>,
    previous: Vec<DVec3>,
    bonds: [Vec<Bond>; 3],
    colors: [Vec<usize>; 6],
    materials: Vec<Material>,
    properties: Vec<Material>,
    precise_rotations: Vec<glam::DQuat>,
    angular_velocities: Vec<DVec3>,
    clusters: Vec<u32>,
    broken: usize,
    cluster_count: usize,
    ground_top: f32,
    energy: crate::fracture::EnergyLedger,
    releases: Vec<crate::fracture::Release>,
    substep_index: u64,
    solver_residual: f32,
    world_bounds: Option<WorldBounds>,
    thermal_bonding_rule: Option<ThermalBondingRule>,
    bonding_contacts: Vec<((usize, usize), BondingContactState)>,
    thermal_fusion_rule: Option<ThermalFusionRule>,
    fusion_contacts: Vec<((usize, usize), BondingContactState)>,
    pending_fusion: Option<(usize, usize)>,
    boundary_removals: BoundaryRemovalLedger,
    actuators: Vec<ActuatorState>,
    solver_warm: SolverWarmState,
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

impl Scene {
    fn to_checkpoint(&self) -> SceneCheckpoint {
        SceneCheckpoint {
            simulated_seconds: self.simulated_seconds,
            adaptive_substeps: self.adaptive_substeps,
            aggregates: self.rigid_aggregates.clone(),
            aggregate_of: self.aggregate_of.clone(),
            config: self.config.clone(),
            voxels: self.voxels_with_sleepers_awake(),
            source_cells: self.source_cells.clone(),
            external_forces: self.external_forces.clone(),
            external_accelerations: self.external_accelerations.clone(),
            axis_targets: self.axis_targets.clone(),
            positions: self.positions.clone(),
            rest_positions: self.rest_positions.clone(),
            reference_temperatures_kelvin: self.reference_temperatures_kelvin.clone(),
            previous: self.previous.clone(),
            bonds: self.bonds.clone(),
            colors: self.colors.clone(),
            materials: self.materials.clone(),
            properties: self.properties.clone(),
            precise_rotations: self
                .rotations
                .iter()
                .zip(self.precise_rotations.iter().chain(std::iter::repeat(&glam::DQuat::IDENTITY)))
                .map(|(single, precise)| {
                    if precise.as_quat() == *single { *precise } else { single.as_dquat().normalize() }
                })
                .collect(),
            angular_velocities: self.angular_velocities.clone(),
            clusters: self.clusters.clone(),
            broken: self.broken,
            cluster_count: self.cluster_count,
            ground_top: self.ground_top,
            energy: self.energy.clone(),
            releases: self.releases.clone(),
            substep_index: self.substep_index,
            solver_residual: self.solver_residual,
            world_bounds: self.world_bounds,
            thermal_bonding_rule: self.thermal_bonding_rule,
            bonding_contacts: self.bonding_contacts.iter().map(|(k, v)| (*k, *v)).collect(),
            thermal_fusion_rule: self.thermal_fusion_rule,
            fusion_contacts: self.fusion_contacts.iter().map(|(k, v)| (*k, *v)).collect(),
            pending_fusion: self.pending_fusion,
            boundary_removals: self.boundary_removals,
            actuators: self
                .skeletal_actuators
                .iter()
                .map(|actor| ActuatorState {
                    model: actor.model().clone(),
                    voxel_range: (actor.voxel_range().start, actor.voxel_range().end),
                    total_scene_voxels: actor.total_scene_voxels(),
                    instance_rotation: actor.instance_rotation_f32(),
                    runtime: actor.runtime(),
                })
                .collect(),
            solver_warm: self.solver_scratch.warm_state(),
        }
    }

    fn from_checkpoint(state: SceneCheckpoint) -> Result<Self, String> {
        let n = state.voxels.len();
        let per_rev = [
            ("positions", state.positions.len()),
            ("rest_positions", state.rest_positions.len()),
            ("previous", state.previous.len()),
            ("precise_rotations", state.precise_rotations.len()),
            ("angular_velocities", state.angular_velocities.len()),
            ("properties", state.properties.len()),
            ("clusters", state.clusters.len()),
            ("external_forces", state.external_forces.len()),
            ("external_accelerations", state.external_accelerations.len()),
            ("axis_targets", state.axis_targets.len()),
            ("aggregate_of", state.aggregate_of.len()),
            ("source_cells", state.source_cells.len()),
            ("reference_temperatures", state.reference_temperatures_kelvin.len()),
        ];
        if let Some((name, len)) = per_rev.iter().find(|(_, len)| *len != n) {
            return Err(format!("checkpoint {name} has {len} entries for {n} REVs"));
        }
        if state.bonds.iter().any(|axis| axis.len() != n) {
            return Err("checkpoint bond arrays do not match the REV count".into());
        }
        if state.positions.iter().any(|p| !p.is_finite())
            || state.voxels.iter().any(|v| !v.velocity.is_finite() || !v.has_finite_thermal_state())
        {
            return Err("checkpoint contains a non-finite physical state".into());
        }
        let mut actuators = Vec::with_capacity(state.actuators.len());
        for actor in state.actuators {
            let mut restored = crate::skeletal_actuator::SkeletalActuator::new(
                actor.model,
                actor.voxel_range.0..actor.voxel_range.1,
                actor.instance_rotation,
                actor.total_scene_voxels,
            )?;
            restored.restore_runtime(actor.runtime);
            actuators.push(restored);
        }
        let rotations = state.precise_rotations.iter().map(|q| q.as_quat()).collect();
        let mut scene = Scene::from_voxels(state.config.clone(), Vec::new())?;
        scene.simulated_seconds = state.simulated_seconds;
        scene.adaptive_substeps = state.adaptive_substeps;
        scene.rigid_aggregates = state.aggregates;
        scene.aggregate_of = state.aggregate_of;
        scene.config = state.config;
        scene.voxels = state.voxels;
        scene.source_cells = state.source_cells;
        scene.external_forces = state.external_forces;
        scene.external_accelerations = state.external_accelerations;
        scene.axis_targets = state.axis_targets;
        scene.positions = state.positions;
        scene.rest_positions = state.rest_positions;
        scene.reference_temperatures_kelvin = state.reference_temperatures_kelvin;
        scene.previous = state.previous;
        scene.workspace = std::array::from_fn(|axis| vec![Default::default(); state.bonds[axis].len()]);
        scene.bonds = state.bonds;
        scene.colors = state.colors;
        scene.materials = state.materials;
        scene.properties = state.properties;
        scene.rotations = rotations;
        scene.precise_rotations = state.precise_rotations;
        scene.angular_velocities = state.angular_velocities;
        scene.clusters = state.clusters;
        scene.broken = state.broken;
        scene.cluster_count = state.cluster_count;
        scene.ground_top = state.ground_top;
        scene.energy = state.energy;
        scene.releases = state.releases;
        scene.substep_index = state.substep_index;
        scene.solver_residual = state.solver_residual;
        scene.world_bounds = state.world_bounds;
        scene.thermal_bonding_rule = state.thermal_bonding_rule;
        scene.bonding_contacts = state.bonding_contacts.into_iter().collect();
        scene.thermal_fusion_rule = state.thermal_fusion_rule;
        scene.fusion_contacts = state.fusion_contacts.into_iter().collect();
        scene.pending_fusion = state.pending_fusion;
        scene.boundary_removals = state.boundary_removals;
        scene.skeletal_actuators = actuators;
        scene.solver_scratch.restore_warm_state(state.solver_warm);
        scene.mark_render_changed();
        Ok(scene)
    }

    /// Write the complete physical state, replacing `path` only after a full write.
    pub fn save_checkpoint(&self, path: &Path) -> Result<(), String> {
        if path.extension().and_then(|e| e.to_str()) != Some("vxstate") {
            return Err("checkpoint path must end in .vxstate".into());
        }
        if self.beam_regions.iter().any(Option::is_some) {
            return Err("expand the beam regions before saving the physical state".into());
        }
        let payload = bincode::serialize(&self.to_checkpoint())
            .map_err(|error| format!("encode checkpoint: {error}"))?;
        let temporary = path.with_extension("vxstate.tmp");
        let mut file = fs::File::create(&temporary)
            .map_err(|error| format!("create checkpoint: {error}"))?;
        file.write_all(MAGIC)
            .and_then(|_| file.write_all(&VERSION.to_le_bytes()))
            .and_then(|_| file.write_all(&crc32(&payload).to_le_bytes()))
            .and_then(|_| file.write_all(&payload))
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("write checkpoint: {error}"))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|error| format!("finish checkpoint: {error}"))
    }

    /// Load a checkpoint written by `save_checkpoint`, validating it before use.
    pub fn load_checkpoint(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| format!("read checkpoint: {error}"))?;
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err("not a voxel checkpoint file".into());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(format!("unsupported checkpoint version {version}"));
        }
        let stored_crc = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if crc32(&bytes[16..]) != stored_crc {
            return Err("checkpoint is damaged: checksum mismatch".into());
        }
        let state: SceneCheckpoint = bincode::deserialize(&bytes[16..])
            .map_err(|error| format!("decode checkpoint: {error}"))?;
        Self::from_checkpoint(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    fn impact_scene() -> Scene {
        let iron = Material::component_id("iron").unwrap();
        let stone = Material::component_id("stone").unwrap();
        let mut config = SceneConfig::default();
        config.fixed_dt = 0.01;
        config.substeps = 4;
        config.gravity = [1.5, 0.0, -9.81];
        let mut spawns = Vec::new();
        for x in -12..12 {
            for y in -4..4 {
                spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
            }
        }
        for z in 1..4 {
            for x in -2..2 {
                for y in -2..2 {
                    spawns.push(
                        VoxelSpawn::at_cell([x, y, z], iron).with_velocity(Vec3::new(0.3, 0.0, 0.0)),
                    );
                }
            }
        }
        Scene::from_voxels(config, spawns).unwrap()
    }

    fn checkpoint_path(name: &str) -> std::path::PathBuf {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("checkpoint-tests");
        fs::create_dir_all(&directory).unwrap();
        directory.join(format!("{name}-{}.vxstate", std::process::id()))
    }

    #[test]
    fn continued_run_matches_uninterrupted_run() {
        let mut uninterrupted = impact_scene();
        for _ in 0..20 {
            uninterrupted.tick();
        }
        let path = checkpoint_path("continue");
        uninterrupted.save_checkpoint(&path).unwrap();
        let mut restored = Scene::load_checkpoint(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(restored.simulated_seconds, uninterrupted.simulated_seconds);
        assert_eq!(restored.positions, uninterrupted.positions);
        assert_eq!(restored.rotations, uninterrupted.rotations);
        assert_eq!(restored.angular_velocities, uninterrupted.angular_velocities);
        for (a, b) in restored.voxels.iter().zip(&uninterrupted.voxels) {
            assert_eq!(a.velocity, b.velocity);
            assert_eq!(a.total_internal_energy_j(), b.total_internal_energy_j());
        }

        for _ in 0..20 {
            uninterrupted.tick();
            restored.tick();
        }
        let worst = restored
            .positions
            .iter()
            .zip(&uninterrupted.positions)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0, f64::max);
        assert!(worst < 1.0e-6, "continued run drifted {worst} m");
        let heat = |scene: &Scene| scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum::<f64>();
        assert!((heat(&restored) - heat(&uninterrupted)).abs() < 1.0e-3);
    }

    #[test]
    fn damaged_or_foreign_files_are_rejected() {
        let path = checkpoint_path("damaged");
        impact_scene().save_checkpoint(&path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        assert!(Scene::load_checkpoint(&path).err().unwrap().contains("checksum"));
        fs::write(&path, b"not a checkpoint").unwrap();
        assert!(Scene::load_checkpoint(&path).is_err());
        fs::remove_file(&path).unwrap();
        assert!(impact_scene().save_checkpoint(Path::new("state.vxscene")).is_err());
    }
}
