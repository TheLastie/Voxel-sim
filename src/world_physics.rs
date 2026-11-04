//! Local, explicitly activated centimetre REVs inside an implicit world.
//! A worker owns the complete solver state; drawing never waits for a step.
use crate::{
    scene::{Scene, SceneConfig, VoxelSpawn, CELL},
    solver,
    world::{GeneratedWorld, PhysicalWorldPatch},
};
use glam::{DVec3, Quat, Vec3};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::{any::Any, thread::JoinHandle};

const GRAB_STIFFNESS_N_PER_M: f32 = 1_200.0;
const GRAB_DAMPING_N_S_PER_M: f32 = 24.0;
const GRAB_MAX_FORCE_N: f32 = 150.0;
const MAX_WORKER_BATCH_WALL_TIME: std::time::Duration = std::time::Duration::from_millis(8);

static NEXT_PHYSICS_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Publish bounded measurements from a real scene. Geometry never crosses MCP.
/// Observer time is reported separately from the solver's per-step time.
pub fn publish_scene_diagnostics(scene: &Scene, context: serde_json::Value) {
    if !crate::runtime_metrics::is_active() {
        return;
    }
    let start = std::time::Instant::now();
    let mut metrics = crate::physics_metrics::collect_runtime_metrics(scene, 32);
    let components = metrics
        .as_object_mut()
        .and_then(|data| data.remove("components"));
    crate::runtime_metrics::publish(
        "objects",
        serde_json::json!({
            "context": context,
            "simulated_seconds": scene.simulated_seconds,
            "topology": metrics["topology"],
            "component_count": metrics["topology"]["component_count"],
            "components": components.unwrap_or_else(|| serde_json::json!([])),
            "identity_scope": "current_material_connectivity_not_authored_model_ids",
        }),
    );
    metrics["context"] = context;
    metrics["simulated_seconds"] = serde_json::json!(scene.simulated_seconds);
    metrics["fixed_dt_seconds"] = serde_json::json!(scene.config.fixed_dt);
    metrics["configured_substeps"] = serde_json::json!(scene.config.substeps);
    metrics["solver_tolerance"] = serde_json::json!(scene.config.solver_relative_tolerance);
    metrics["last_fixed_step"] =
        serde_json::to_value(&scene.last_step_diagnostics).unwrap_or_default();
    metrics["collection_ms"] = serde_json::json!(start.elapsed().as_secs_f64() * 1000.0);
    crate::runtime_metrics::publish("physics", metrics);
}

enum PhysicsWorkerRequest {
    Steps(usize),
    ReturnScene,
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "non-string panic payload".to_owned(),
        },
    }
}

/// In-memory transfer of an active physical patch between worker lifetimes.
/// This preserves the complete runtime Scene, including private solver and
/// thermal history, without pretending to be a disk checkpoint format.
pub struct PhysicalSceneSnapshot {
    pub origin_cell: Option<[i32; 3]>,
    pub origin_world_m: DVec3,
    pub simulated_seconds: f64,
    pub pending_seconds: f64,
    pub scene: Scene,
}

impl PhysicalSceneSnapshot {
    pub fn source_cells(&self) -> &[Option<[i32; 3]>] {
        &self.scene.source_cells
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalCellPose {
    /// Current solver index, used only when this scene has no generated-cell identity.
    pub scene_index: usize,
    /// Stable generated-world cell identity; absent for standalone experiments.
    pub source_cell: Option<[i32; 3]>,
    /// Kept in f64 world coordinates until the render boundary.
    pub world_position_m: DVec3,
    pub rotation: Quat,
    pub temperature_kelvin: f32,
    pub liquid_fraction: f32,
    pub gas_fraction: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PhysicalCellThermalState {
    pub scene_index: usize,
    pub temperature_kelvin: f32,
    pub liquid_fraction: f32,
    pub gas_fraction: f32,
}

/// A hand anchor acting on one REV through the ordinary external-force path.
#[derive(Clone, Copy, Debug)]
pub struct GrabTarget {
    pub source_cell: Option<[i32; 3]>,
    pub scene_index: usize,
    pub target_world_m: DVec3,
}

#[derive(Clone, Debug, Default)]
pub struct PhysicsFrame {
    /// Acknowledged fixed steps in this publication, not rendered frames.
    pub completed_steps: usize,
    pub cells: Vec<PhysicalCellPose>,
    /// Thermal snapshots for fixed REVs, which conduct heat but have no dynamic pose.
    pub fixed_thermal: Vec<PhysicalCellThermalState>,
    /// Distinguishes a restarted worker from an earlier frame at the same step.
    pub generation: u64,
    /// Advances only when the worker publishes a new solver state.
    pub revision: u64,
    pub simulated_seconds: f64,
    pub step_ms: f64,
    /// Latest completed worker-batch cost normalized to 10 ms of simulation.
    /// Includes pose capture and any metrics publish performed by that batch.
    pub full_worker_ms_per_10ms_simulated: Option<f64>,
    pub broken: usize,
}

/// Ordered clock increments are retained even when the solver runs slower
/// than real time. At most one time-budgeted batch is outstanding. Several
/// fixed steps can finish between redraws, without uploading a pose each step.
pub struct PhysicsWorker {
    requests: Sender<PhysicsWorkerRequest>,
    frames: Receiver<PhysicsFrame>,
    worker: Option<JoinHandle<PhysicalSceneSnapshot>>,
    pending_seconds: f64,
    fixed_dt: f64,
    in_flight: bool,
    in_flight_steps: usize,
    failure: Option<String>,
    grab_target: Arc<Mutex<Option<GrabTarget>>>,
    pub frame: PhysicsFrame,
}

impl PhysicsFrame {
    fn capture(
        scene: &Scene,
        origin: DVec3,
        simulated_seconds: f64,
        step_ms: f64,
        generation: u64,
        revision: u64,
    ) -> Self {
        Self {
            cells: scene
                .positions
                .iter()
                .zip(&scene.rotations)
                .zip(&scene.voxels)
                .enumerate()
                .filter(|(_, (_, voxel))| !voxel.kinematic)
                .map(|(index, ((&position, &rotation), _))| {
                    let phase = scene
                        .phase_state(index)
                        .expect("captured voxel must have a phase state");
                    PhysicalCellPose {
                        scene_index: index,
                        source_cell: scene.source_cells.get(index).copied().flatten(),
                        world_position_m: position + origin,
                        rotation,
                        temperature_kelvin: phase.temperature_kelvin,
                        liquid_fraction: phase.liquid_fraction,
                        gas_fraction: phase.gas_fraction,
                    }
                })
                .collect(),
            fixed_thermal: scene
                .voxels
                .iter()
                .enumerate()
                .filter(|(_, voxel)| voxel.kinematic)
                .map(|(scene_index, _)| {
                    let phase = scene
                        .phase_state(scene_index)
                        .expect("captured voxel must have a phase state");
                    PhysicalCellThermalState {
                        scene_index,
                        temperature_kelvin: phase.temperature_kelvin,
                        liquid_fraction: phase.liquid_fraction,
                        gas_fraction: phase.gas_fraction,
                    }
                })
                .collect(),
            generation,
            revision,
            completed_steps: 0,
            simulated_seconds,
            step_ms,
            full_worker_ms_per_10ms_simulated: None,
            broken: scene.broken,
        }
    }
}

impl PhysicsWorker {
    pub fn new(scene: Scene) -> Result<Self, std::io::Error> {
        Self::at_origin(scene, DVec3::ZERO)
    }

    /// Start an authored scene with one persistent external force per REV.
    /// Loads are in newtons and are passed through the same solver force path
    /// as other `Scene::external_forces` on every fixed step.
    pub fn with_loads(mut scene: Scene, loads: Vec<Vec3>) -> Result<Self, std::io::Error> {
        if loads.len() != scene.voxels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "authored load count must match the scene REV count",
            ));
        }
        if loads.iter().any(|load| !load.is_finite()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "authored loads must contain only finite force values",
            ));
        }
        scene.external_forces = loads;
        Self::at_origin(scene, DVec3::ZERO)
    }

    /// Keep mechanical coordinates near zero; add the lattice-aligned world
    /// origin only to render snapshots. Translating a region changes no forces.
    pub fn at_origin(scene: Scene, origin: DVec3) -> Result<Self, std::io::Error> {
        let mut scene = scene;
        scene.source_cells = vec![None; scene.voxels.len()];
        Self::from_snapshot(PhysicalSceneSnapshot {
            origin_cell: None,
            origin_world_m: origin,
            simulated_seconds: 0.0,
            pending_seconds: 0.0,
            scene,
        })
    }

    /// Start simulation from a generated-world patch without dropping the
    /// source cell mapping or rounding its large world origin to f32.
    pub fn from_patch(patch: PhysicalWorldPatch) -> Result<Self, std::io::Error> {
        if patch.source_cells.len() != patch.scene.voxels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "physical world patch source-cell mapping does not match its REV count",
            ));
        }
        let mut scene = patch.scene;
        scene.source_cells = patch.source_cells.into_iter().map(Some).collect();
        Self::from_snapshot(PhysicalSceneSnapshot {
            origin_cell: Some(patch.origin_cell),
            origin_world_m: patch.origin_world_m,
            simulated_seconds: 0.0,
            pending_seconds: 0.0,
            scene,
        })
    }

    pub fn from_snapshot(snapshot: PhysicalSceneSnapshot) -> Result<Self, std::io::Error> {
        let PhysicalSceneSnapshot {
            origin_cell,
            origin_world_m,
            simulated_seconds,
            pending_seconds,
            mut scene,
        } = snapshot;
        if scene.source_cells.len() != scene.voxels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "physical scene snapshot source-cell mapping does not match its REV count",
            ));
        }
        if !simulated_seconds.is_finite()
            || simulated_seconds < 0.0
            || !pending_seconds.is_finite()
            || pending_seconds < 0.0
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "physical scene snapshot has an invalid simulation clock",
            ));
        }
        let fixed_dt = f64::from(scene.config.fixed_dt);
        let generation = NEXT_PHYSICS_GENERATION.fetch_add(1, Ordering::Relaxed);
        let frame = PhysicsFrame::capture(
            &scene,
            origin_world_m,
            simulated_seconds,
            0.0,
            generation,
            0,
        );
        let (requests, input) = mpsc::channel();
        let (output, frames) = mpsc::channel();
        let grab_target = Arc::new(Mutex::new(None::<GrabTarget>));
        let worker_grab_target = Arc::clone(&grab_target);
        let worker = std::thread::Builder::new()
            .name("world-physics".into())
            .spawn(move || {
                let mut time = simulated_seconds;
                let mut revision = 0u64;
                let mut tool = None;
                let mut applied_grab: Option<(usize, Vec3)> = None;
                let mut remaining_batch_steps = 0usize;
                let mut performance = crate::physics_metrics::PerformanceWindow::default();
                let mut previous_observer_ms = 0.0;
                let mut last_metrics = std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(1)).unwrap();
                loop {
                    match input.recv() {
                        Ok(PhysicsWorkerRequest::Steps(requested_steps)) => {
                            let start = std::time::Instant::now();
                            let broken_before = scene.broken;
                            let mut batch_contacts_peak = 0usize;
                            let mut batch_impacts = 0usize;
                            let mut batch_pcg_iterations = 0usize;
                            let mut batch_residual_max = 0.0_f32;
                            let mut batch_non_finite_residuals = 0usize;
                            let mut batch_active_fine_peak = 0usize;
                            let mut completed_steps = 0usize;
                            for _ in 0..requested_steps {
                                if let Some((index, force)) = applied_grab.take() {
                                    if let Some(applied) = scene.external_forces.get_mut(index) {
                                        *applied -= force;
                                    }
                                }
                                let requested_grab = worker_grab_target
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .to_owned();
                                if let Some(grab) = requested_grab {
                                    let index = match grab.source_cell {
                                        Some(cell) => scene
                                            .source_cells
                                            .iter()
                                            .position(|source| *source == Some(cell)),
                                        None => Some(grab.scene_index),
                                    };
                                    if let Some(index) = index.filter(|&index| {
                                        index < scene.voxels.len() && !scene.voxels[index].kinematic
                                    }) {
                                        let position = scene.positions[index] + origin_world_m;
                                        let displacement = (grab.target_world_m - position).as_vec3();
                                        let force = displacement * GRAB_STIFFNESS_N_PER_M
                                            - scene.voxels[index].velocity * GRAB_DAMPING_N_S_PER_M;
                                        let force = if force.length() > GRAB_MAX_FORCE_N {
                                            force.normalize() * GRAB_MAX_FORCE_N
                                        } else {
                                            force
                                        };
                                        scene.external_forces[index] += force;
                                        applied_grab = Some((index, force));
                                    }
                                }
                                solver::step(&mut scene, &mut tool);
                                if !scene.solver_residual.is_finite() {
                                    batch_non_finite_residuals += 1;
                                }
                                if let Some(diagnostics) = &scene.last_step_diagnostics {
                                    batch_contacts_peak = batch_contacts_peak.max(diagnostics.contacts_peak);
                                    batch_impacts += diagnostics.new_impact_observations;
                                    batch_pcg_iterations += diagnostics.pcg_iterations_total;
                                    batch_residual_max = batch_residual_max.max(diagnostics.relative_solver_residual);
                                    batch_active_fine_peak = batch_active_fine_peak.max(diagnostics.active_fine_revs_last_solve);
                                }
                                time += fixed_dt;
                                revision = revision.wrapping_add(1);
                                completed_steps += 1;
                                if start.elapsed() >= MAX_WORKER_BATCH_WALL_TIME { break; }
                            }
                            remaining_batch_steps = requested_steps - completed_steps;
                            let batch_ms = start.elapsed().as_secs_f64()*1000.0;
                            let capture_start = std::time::Instant::now();
                            let mut frame = PhysicsFrame::capture(
                                    &scene,
                                    origin_world_m,
                                    time,
                                    batch_ms / completed_steps.max(1) as f64,
                                    generation,
                                    revision,
                                );
                            frame.completed_steps = completed_steps;
                            let pose_capture_ms = capture_start.elapsed().as_secs_f64()*1000.0;
                            let new_broken = scene.broken.saturating_sub(broken_before);
                            let interesting = new_broken>0 || batch_impacts>0
                                || batch_non_finite_residuals > 0
                                || batch_residual_max > scene.config.solver_relative_tolerance*10.0;
                            if interesting && crate::runtime_metrics::is_active() {
                                crate::runtime_metrics::publish("event", serde_json::json!({
                                    "generation":generation,"revision":revision,"simulated_seconds":time,
                                    "new_broken_bonds":new_broken,"new_impact_observations":batch_impacts,
                                    "contacts_peak":batch_contacts_peak,"solver_residual_max":batch_residual_max,
                                    "solver_residual_is_finite":scene.solver_residual.is_finite(),
                                    "non_finite_solver_residual_steps_in_batch":batch_non_finite_residuals,
                                    "fixed_dt_seconds":fixed_dt,"configured_substeps":scene.config.substeps,
                                }));
                            }
                            // Keep full-scene aggregation bounded even during impact storms.
                            if last_metrics.elapsed() >= std::time::Duration::from_millis(100)
                                && crate::runtime_metrics::is_active() {
                                let observer_start = std::time::Instant::now();
                                publish_scene_diagnostics(&scene, serde_json::json!({
                                    "source":"worker", "generation":generation, "revision":revision,
                                    "completed_steps":completed_steps,
                                    "requested_steps":requested_steps,
                                    "remaining_requested_steps":remaining_batch_steps,
                                    "batch_wall_ms":batch_ms,
                                    "mean_step_ms":frame.step_ms,
                                    "pose_capture_ms":pose_capture_ms,
                                    "performance":performance.to_json(),
                                    "performance_scope":"previous_completed_batches; includes_observer_and_pose_capture",
                                    "observer_collection_and_publish_ms_previous":previous_observer_ms,
                                    "worker_cost_scope":"performance_includes_observer; batch_wall_ms_is_solver_only",
                                    "active_count_scope":"peak_of_last_solve_in_each_completed_step; quiescent_fast_path_is_zero",
                                    "wall_budget_ms":MAX_WORKER_BATCH_WALL_TIME.as_secs_f64()*1000.0,
                                    "contacts_peak_in_batch":batch_contacts_peak,
                                    "new_impact_observations_in_batch":batch_impacts,
                                    "pcg_iterations_in_batch":batch_pcg_iterations,
                                    "solver_residual_max_in_batch":batch_residual_max,
                                    "non_finite_solver_residual_steps_in_batch":batch_non_finite_residuals,
                                    "new_broken_bonds_in_batch":new_broken,
                                }));
                                previous_observer_ms = observer_start.elapsed().as_secs_f64()*1000.0;
                                last_metrics = std::time::Instant::now();
                            }
                            let completed_simulated_seconds = completed_steps as f64 * fixed_dt;
                            let full_worker_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                            if performance.push_batch(
                                batch_active_fine_peak,
                                completed_simulated_seconds,
                                batch_ms,
                                full_worker_wall_ms,
                            ) {
                                frame.full_worker_ms_per_10ms_simulated = Some(
                                    full_worker_wall_ms * 0.01 / completed_simulated_seconds,
                                );
                            }
                            if output.send(frame).is_err() {
                                break;
                            }
                        }
                        Ok(PhysicsWorkerRequest::ReturnScene) | Err(_) => {
                            if let Some((index, force)) = applied_grab {
                                if let Some(applied) = scene.external_forces.get_mut(index) {
                                    *applied -= force;
                                }
                            }
                            return PhysicalSceneSnapshot {
                                origin_cell,
                                origin_world_m,
                                simulated_seconds: time,
                                pending_seconds: remaining_batch_steps as f64 * fixed_dt,
                                scene,
                            };
                        }
                    }
                }
                PhysicalSceneSnapshot {
                    origin_cell,
                    origin_world_m,
                    simulated_seconds: time,
                    pending_seconds: remaining_batch_steps as f64 * fixed_dt,
                    scene,
                }
            })?;
        Ok(Self {
            requests,
            frames,
            worker: Some(worker),
            pending_seconds,
            fixed_dt,
            in_flight: false,
            in_flight_steps: 0,
            failure: None,
            grab_target,
            frame,
        })
    }

    /// Replace the hand target atomically; the worker applies it at the next fixed-step boundary.
    pub fn set_grab_target(&self, target: Option<GrabTarget>) {
        *self
            .grab_target
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = target;
    }

    /// Whether the worker stopped unexpectedly. The last published frame is
    /// retained unchanged so callers can show a paused/error state safely.
    pub fn has_failure(&self) -> bool {
        self.failure.is_some()
    }

    /// A stable error message suitable for a status panel.
    pub fn failure_message(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    fn collect_finished_worker_failure(&mut self) {
        let finished = self.worker.as_ref().is_some_and(JoinHandle::is_finished);
        if !finished {
            return;
        }
        let Some(worker) = self.worker.take() else {
            return;
        };
        if let Err(payload) = worker.join() {
            self.failure = Some(format!(
                "physics worker panicked: {}",
                panic_payload_message(payload)
            ));
        }
    }

    fn mark_worker_failed(&mut self, message: &str) {
        self.in_flight = false;
        self.in_flight_steps = 0;
        self.pending_seconds = 0.0;
        self.failure.get_or_insert_with(|| message.to_owned());
        // `is_finished` makes this join nonblocking on the UI thread.
        self.collect_finished_worker_failure();
    }

    pub fn advance(&mut self, elapsed: f64, paused: bool) {
        if self.failure.is_some() {
            self.collect_finished_worker_failure();
            return;
        }
        match self.frames.try_recv() {
            Ok(frame) => {
                if self.in_flight {
                    let remaining = self.in_flight_steps.saturating_sub(frame.completed_steps);
                    self.pending_seconds += remaining as f64 * self.fixed_dt;
                }
                self.frame = frame;
                self.in_flight = false;
                self.in_flight_steps = 0;
            }
            Err(TryRecvError::Disconnected) => {
                self.mark_worker_failed("physics worker stopped without publishing a frame");
                return;
            }
            Err(TryRecvError::Empty) => {}
        }
        if elapsed.is_finite() && elapsed > 0.0 && !paused {
            self.pending_seconds += elapsed;
        }
        if !paused && !self.in_flight && self.pending_seconds >= self.fixed_dt {
            let steps = (self.pending_seconds / self.fixed_dt).floor() as usize;
            if self
                .requests
                .send(PhysicsWorkerRequest::Steps(steps))
                .is_ok()
            {
                self.pending_seconds -= steps as f64 * self.fixed_dt;
                self.in_flight = true;
                self.in_flight_steps = steps;
            } else {
                self.mark_worker_failed("physics worker request channel disconnected");
            }
        }
    }

    pub fn backlog_seconds(&self) -> f64 {
        self.pending_seconds + self.in_flight_steps as f64 * self.fixed_dt
    }

    /// Stop this worker after its bounded batch, returning the
    /// authoritative runtime scene for a later reactivation without copying it.
    pub fn into_snapshot(mut self) -> Result<PhysicalSceneSnapshot, std::io::Error> {
        self.requests
            .send(PhysicsWorkerRequest::ReturnScene)
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "physics worker has stopped")
            })?;
        let mut snapshot = self
            .worker
            .take()
            .ok_or_else(|| std::io::Error::other("physics worker handle is missing"))?
            .join()
            .map_err(|_| {
                std::io::Error::other("physics worker panicked before returning its scene")
            })?;
        // Unfinished requested steps live on the worker until their frame is
        // collected; otherwise they have already returned to pending_seconds.
        snapshot.pending_seconds = self.pending_seconds
            + if self.in_flight {
                snapshot.pending_seconds
            } else {
                0.0
            };
        Ok(snapshot)
    }
}

/// A first sparse activation experiment. Only the visible support patch is a
/// constrained material reservoir; there is no invisible collision plane.
/// The rest of the 20 m square remains implicit and is not yet dynamic.
pub struct PhysicalRegion {
    pub scene: Scene,
    pub origin: Vec3,
}

pub fn contact_experiment(
    world: &GeneratedWorld,
    center: Vec3,
    mut config: SceneConfig,
) -> Result<PhysicalRegion, String> {
    let stone = crate::material::Material::component_id("stone").ok_or("stone material missing")?;
    let wood = crate::material::Material::component_id("oak").ok_or("oak material missing")?;
    config.fixed_dt = 0.010;
    config.substeps = 40;
    config.gravity = [0.0, 0.0, -9.81];
    let cx = (center.x / CELL).floor() as i32;
    let cy = (center.y / CELL).floor() as i32;
    let cz = (world.terrain_height_at(center.x, center.y) / CELL).round() as i32;
    let origin = Vec3::new(cx as f32, cy as f32, cz as f32) * CELL;
    let mut spawns = Vec::new();
    let mut top = f32::NEG_INFINITY;
    for x in cx - 20..cx + 20 {
        for y in cy - 20..cy + 20 {
            let height = world.terrain_height_at((x as f32 + 0.5) * CELL, (y as f32 + 0.5) * CELL);
            let z = (height / CELL).round() as i32;
            top = top.max(z as f32 * CELL);
            for layer in 1..=2 {
                spawns.push(VoxelSpawn::at_cell([x - cx, y - cy, z - layer - cz], stone).fixed());
            }
        }
    }
    let bottom = ((top + 0.35) / CELL).ceil() as i32;
    for x in cx - 12..cx + 12 {
        for y in cy - 3..cy + 3 {
            for z in bottom..bottom + 6 {
                spawns.push(
                    VoxelSpawn::at_cell([x - cx, y - cy, z - cz], wood).with_microstructure(
                        crate::voxel::Microstructure {
                            orientation: Quat::from_rotation_arc(Vec3::Z, Vec3::X),
                            anisotropy: 1.0,
                        },
                    ),
                );
            }
        }
    }
    let mut scene = Scene::from_voxels(config, spawns)?;
    scene
        .set_world_bounds(
            Vec3::new(center.x - 10.0, center.y - 10.0, 0.0) - origin,
            Vec3::new(center.x + 10.0, center.y + 10.0, 512.0) - origin,
        )
        .map_err(str::to_owned)?;
    Ok(PhysicalRegion { scene, origin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{voxel::Composition, world::GeneratedCellPhysics};

    fn mock_worker_clock(
        fixed_dt: f64,
    ) -> (
        PhysicsWorker,
        Receiver<PhysicsWorkerRequest>,
        Sender<PhysicsFrame>,
    ) {
        let (requests, input) = mpsc::channel();
        let (output, frames) = mpsc::channel();
        (
            PhysicsWorker {
                requests,
                frames,
                worker: None,
                pending_seconds: 0.0,
                fixed_dt,
                in_flight: false,
                in_flight_steps: 0,
                failure: None,
                grab_target: Arc::new(Mutex::new(None)),
                frame: PhysicsFrame::default(),
            },
            input,
            output,
        )
    }

    #[test]
    fn redraw_requests_all_due_fixed_steps_instead_of_one() {
        let (mut worker, input, _output) = mock_worker_clock(0.001);
        worker.advance(0.016, false);
        assert!(matches!(
            input.try_recv().unwrap(),
            PhysicsWorkerRequest::Steps(16)
        ));
        assert_eq!(worker.in_flight_steps, 16);
        assert!((worker.backlog_seconds() - 0.016).abs() < 1.0e-12);
    }

    #[test]
    fn incomplete_batch_returns_time_to_backlog_and_pause_queues_nothing() {
        let (mut worker, input, output) = mock_worker_clock(0.001);
        worker.advance(0.016, false);
        input.try_recv().unwrap();
        output
            .send(PhysicsFrame {
                completed_steps: 3,
                simulated_seconds: 0.003,
                ..PhysicsFrame::default()
            })
            .unwrap();
        worker.advance(30.0, true);
        assert!(!worker.in_flight);
        assert!((worker.backlog_seconds() - 0.013).abs() < 1.0e-12);
        assert!(input.try_recv().is_err());
        worker.advance(0.0, false);
        assert!(matches!(input.try_recv().unwrap(),PhysicsWorkerRequest::Steps(steps) if steps>1));
        assert!((worker.backlog_seconds() - 0.013).abs() < 1.0e-12);
    }

    #[test]
    fn returning_scene_preserves_completed_and_unfinished_batch_time() {
        let mut scene = one_rev_scene();
        scene.voxels[0].velocity = Vec3::X;
        let mut worker = PhysicsWorker::new(scene).unwrap();
        let requested = worker.fixed_dt * 20.0;
        worker.advance(requested, false);
        let snapshot = worker.into_snapshot().unwrap();
        assert!(snapshot.simulated_seconds > 0.0);
        assert!(
            (snapshot.simulated_seconds + snapshot.pending_seconds - requested).abs() < 1.0e-12
        );
        let steps = (snapshot.simulated_seconds / f64::from(snapshot.scene.config.fixed_dt)).round()
            as usize;
        let mut reference = one_rev_scene();
        reference.voxels[0].velocity = Vec3::X;
        for _ in 0..steps {
            reference.tick();
        }
        assert_eq!(snapshot.scene.positions, reference.positions);
        assert_eq!(
            snapshot.scene.voxels[0].velocity,
            reference.voxels[0].velocity
        );
    }

    fn one_rev_scene() -> Scene {
        let material = crate::material::Material::component_id("oak").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        Scene::from_voxels(config, vec![VoxelSpawn::at_cell([0, 0, 0], material)]).unwrap()
    }

    #[test]
    fn authored_loads_reject_count_mismatch_and_non_finite_values() {
        let error = PhysicsWorker::with_loads(one_rev_scene(), Vec::new())
            .err()
            .unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);

        let error = PhysicsWorker::with_loads(one_rev_scene(), vec![Vec3::new(f32::NAN, 0.0, 0.0)])
            .err()
            .unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn authored_force_accelerates_a_free_rev_through_the_worker_solver() {
        let scene = one_rev_scene();
        let dt = scene.config.fixed_dt;
        let mass = scene.voxels[0].mass;
        let force = mass * 1.0;
        let mut worker = PhysicsWorker::with_loads(scene, vec![Vec3::X * force]).unwrap();
        worker.advance(f64::from(dt), false);

        let snapshot = worker.into_snapshot().unwrap();
        let velocity_x = snapshot.scene.voxels[0].velocity.x;
        assert!(velocity_x > 0.0);
        assert!((velocity_x - dt).abs() < dt * 0.1);
    }

    #[test]
    fn captured_frame_contains_current_iron_temperature_and_phase_fractions() {
        let iron = crate::material::Material::component_id("iron").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        let mut scene =
            Scene::from_voxels(config, vec![VoxelSpawn::at_cell([0, 0, 0], iron)]).unwrap();

        let cold = PhysicsFrame::capture(&scene, DVec3::ZERO, 0.0, 0.0, 1, 1);
        assert_eq!(cold.cells.len(), 1);
        assert!(cold.cells[0].temperature_kelvin < 9000.0);
        assert_eq!(cold.cells[0].liquid_fraction, 0.0);
        assert_eq!(cold.cells[0].gas_fraction, 0.0);

        let material =
            crate::material::Material::mixed(&scene.voxels[0].composition, &scene.materials);
        let target_energy = material.energy_at_temperature(scene.voxels[0].mass, 9000.0);
        let current_energy = scene.voxels[0].total_internal_energy_j() as f32;
        assert!(scene.add_heat_joules(0, target_energy - current_energy));
        let hot_phase = scene.phase_state(0).unwrap();
        assert!((hot_phase.temperature_kelvin - 9000.0).abs() < 0.1);

        let hot = PhysicsFrame::capture(&scene, DVec3::ZERO, 0.0, 0.0, 1, 2);
        assert!((hot.cells[0].temperature_kelvin - 9000.0).abs() < 0.1);
        assert_eq!(hot.cells[0].liquid_fraction, hot_phase.liquid_fraction);
        assert_eq!(hot.cells[0].gas_fraction, hot_phase.gas_fraction);
        assert!(hot.cells[0].temperature_kelvin > cold.cells[0].temperature_kelvin);
        assert!(hot.cells[0].gas_fraction > cold.cells[0].gas_fraction);
    }

    #[test]
    fn captured_frame_includes_heated_fixed_iron_without_dynamic_pose() {
        let iron = crate::material::Material::component_id("iron").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        let mut scene =
            Scene::from_voxels(config, vec![VoxelSpawn::at_cell([0, 0, 0], iron).fixed()]).unwrap();
        let material =
            crate::material::Material::mixed(&scene.voxels[0].composition, &scene.materials);
        let target_energy = material.energy_at_temperature(scene.voxels[0].mass, 9000.0);
        let current_energy = scene.voxels[0].total_internal_energy_j() as f32;
        assert!(scene.add_heat_joules(0, target_energy - current_energy));

        let frame = PhysicsFrame::capture(&scene, DVec3::ZERO, 0.0, 0.0, 1, 1);

        assert!(frame.cells.is_empty());
        assert_eq!(frame.fixed_thermal.len(), 1);
        assert_eq!(frame.fixed_thermal[0].scene_index, 0);
        assert!((frame.fixed_thermal[0].temperature_kelvin - 9000.0).abs() < 0.1);
        assert_eq!(
            frame.fixed_thermal[0].liquid_fraction,
            scene.phase_state(0).unwrap().liquid_fraction
        );
        assert_eq!(
            frame.fixed_thermal[0].gas_fraction,
            scene.phase_state(0).unwrap().gas_fraction
        );
    }

    #[test]
    fn panicked_worker_stops_backlog_and_preserves_last_published_frame() {
        let (requests, _input) = mpsc::channel();
        let (output, frames) = mpsc::channel();
        drop(output);
        let worker =
            std::thread::spawn(|| -> PhysicalSceneSnapshot { panic!("test panic payload") });
        while !worker.is_finished() {
            std::thread::yield_now();
        }
        let mut worker = PhysicsWorker {
            requests,
            frames,
            worker: Some(worker),
            pending_seconds: 0.75,
            fixed_dt: 0.1,
            in_flight: true,
            in_flight_steps: 1,
            failure: None,
            grab_target: Arc::new(Mutex::new(None)),
            frame: PhysicsFrame {
                generation: 4,
                revision: 9,
                simulated_seconds: 2.5,
                ..PhysicsFrame::default()
            },
        };

        worker.advance(10.0, false);

        assert!(worker.has_failure());
        assert!(worker
            .failure_message()
            .unwrap()
            .contains("test panic payload"));
        assert!(!worker.in_flight);
        assert_eq!(worker.backlog_seconds(), 0.0);
        assert_eq!(worker.frame.revision, 9);
        assert_eq!(worker.frame.simulated_seconds, 2.5);
    }

    #[test]
    fn patch_worker_keeps_world_cell_identity_and_f64_origin() {
        let world = GeneratedWorld::from_seed("patch-worker-origin");
        let wood = crate::material::Material::component_id("wood").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        let patch = world
            .build_physical_patch(config, [10, 20, 10], [12, 21, 11], |_| {
                Ok(Some(GeneratedCellPhysics {
                    composition: Composition::pure(wood),
                    anisotropy: 0.0,
                    velocity: Vec3::ZERO,
                    kinematic: false,
                }))
            })
            .unwrap();
        let expected_world_position = patch.origin_world_m + patch.scene.positions[0];
        let origin_cell = patch.origin_cell;
        let origin_world_m = patch.origin_world_m;

        let worker = PhysicsWorker::from_patch(patch).unwrap();

        assert_eq!(worker.frame.cells.len(), 2);
        assert_eq!(worker.frame.cells[0].source_cell, Some([10, 20, 10]));
        assert_eq!(
            worker.frame.cells[0].world_position_m,
            expected_world_position
        );

        let mut snapshot = worker.into_snapshot().unwrap();
        snapshot.simulated_seconds = 12.5;
        snapshot.pending_seconds = 0.004;
        snapshot.scene.voxels[0].velocity = Vec3::new(0.25, -0.5, 0.75);
        snapshot.scene.voxels[0].set_thermal_cell_energies_j(&[
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6,
        ]);
        snapshot.scene.bonds[0][0].damage = 0.375;
        let expected_mass = snapshot.scene.total_mass_kg();
        let expected_internal_energy = snapshot.scene.voxels[0].total_internal_energy_j();
        let expected_velocity = snapshot.scene.voxels[0].velocity;
        let mut expected_thermal_energy = [0.0; crate::voxel::THERMAL_CELL_COUNT];
        snapshot.scene.voxels[0].copy_thermal_cell_energies_j(&mut expected_thermal_energy);

        let reactivated = PhysicsWorker::from_snapshot(snapshot).unwrap();
        let resumed = reactivated.into_snapshot().unwrap();

        assert_eq!(resumed.origin_cell, Some(origin_cell));
        assert_eq!(resumed.origin_world_m, origin_world_m);
        assert_eq!(resumed.simulated_seconds, 12.5);
        assert_eq!(resumed.pending_seconds, 0.004);
        assert_eq!(
            resumed.source_cells(),
            &[Some([10, 20, 10]), Some([11, 20, 10])]
        );
        assert_eq!(resumed.scene.total_mass_kg(), expected_mass);
        assert_eq!(
            resumed.scene.voxels[0].total_internal_energy_j(),
            expected_internal_energy
        );
        assert_eq!(resumed.scene.voxels[0].velocity, expected_velocity);
        assert_eq!(resumed.scene.bonds[0][0].damage, 0.375);
        let mut restored_thermal_energy = [0.0; crate::voxel::THERMAL_CELL_COUNT];
        resumed.scene.voxels[0].copy_thermal_cell_energies_j(&mut restored_thermal_energy);
        assert_eq!(restored_thermal_energy, expected_thermal_energy);
    }

    #[test]
    fn source_cell_identity_compacts_with_boundary_removed_revs() {
        let world = GeneratedWorld::from_seed("patch-source-compaction");
        let wood = crate::material::Material::component_id("wood").unwrap();
        let patch = world
            .build_physical_patch(SceneConfig::default(), [10, 20, 10], [12, 21, 11], |_| {
                Ok(Some(GeneratedCellPhysics {
                    composition: Composition::pure(wood),
                    anisotropy: 0.0,
                    velocity: Vec3::ZERO,
                    kinematic: false,
                }))
            })
            .unwrap();
        let mut snapshot = PhysicsWorker::from_patch(patch)
            .unwrap()
            .into_snapshot()
            .unwrap();
        let keep_position = snapshot.scene.positions[1];
        snapshot
            .scene
            .set_world_bounds(
                keep_position - DVec3::splat(0.001),
                keep_position + DVec3::splat(0.001),
            )
            .unwrap();

        assert_eq!(snapshot.scene.voxels.len(), 1);
        assert_eq!(snapshot.source_cells(), &[Some([11, 20, 10])]);
    }

    #[test]
    fn worker_snapshot_waits_for_an_already_queued_fixed_step() {
        let world = GeneratedWorld::from_seed("patch-worker-drain");
        let wood = crate::material::Material::component_id("wood").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        let mut patch = world
            .build_physical_patch(config, [10, 20, 10], [12, 21, 11], |_| {
                Ok(Some(GeneratedCellPhysics {
                    composition: Composition::pure(wood),
                    anisotropy: 0.0,
                    velocity: Vec3::ZERO,
                    kinematic: true,
                }))
            })
            .unwrap();
        for voxel in &mut patch.scene.voxels {
            voxel.kinematic = true;
        }
        let mut worker = PhysicsWorker::from_patch(patch).unwrap();
        let fixed_dt = worker.fixed_dt;
        worker.advance(fixed_dt, false);

        let snapshot = worker.into_snapshot().unwrap();

        assert_eq!(snapshot.simulated_seconds, fixed_dt);
        assert_eq!(snapshot.pending_seconds, 0.0);
        assert!(snapshot.scene.substep_index > 0);
    }
}
