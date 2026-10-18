//! Local, explicitly activated centimetre REVs inside an implicit world.
//! A worker owns the complete solver state; drawing never waits for a step.
use crate::{scene::{Scene, SceneConfig, VoxelSpawn, CELL}, solver, world::GeneratedWorld};
use glam::{Quat, Vec3};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

#[derive(Clone, Debug, Default)]
pub struct PhysicsFrame {
    pub poses: Vec<(Vec3, Quat)>,
    pub simulated_seconds: f64,
    pub step_ms: f64,
    pub broken: usize,
}

/// Ordered clock increments are retained even when the solver runs slower
/// than real time. At most one solver step is outstanding on the worker.
pub struct PhysicsWorker {
    requests: Sender<()>,
    frames: Receiver<PhysicsFrame>,
    pending_seconds: f64,
    fixed_dt: f64,
    in_flight: bool,
    pub frame: PhysicsFrame,
}

impl PhysicsFrame {
    fn capture(scene: &Scene, origin: Vec3, simulated_seconds: f64, step_ms: f64) -> Self {
        Self {
            poses: scene.positions.iter().zip(&scene.rotations).zip(&scene.voxels)
                .filter(|(_, voxel)| !voxel.kinematic)
                .map(|((&p, &q), _)| ((p + origin.as_dvec3()).as_vec3(), q)).collect(),
            simulated_seconds, step_ms, broken: scene.broken,
        }
    }
}

impl PhysicsWorker {
    pub fn new(scene: Scene) -> Result<Self, std::io::Error> {
        Self::at_origin(scene, Vec3::ZERO)
    }

    /// Keep mechanical coordinates near zero; add the lattice-aligned world
    /// origin only to render snapshots. Translating a region changes no forces.
    pub fn at_origin(mut scene: Scene, origin: Vec3) -> Result<Self, std::io::Error> {
        let fixed_dt = f64::from(scene.config.fixed_dt);
        let frame = PhysicsFrame::capture(&scene, origin, 0.0, 0.0);
        let (requests, input) = mpsc::channel();
        let (output, frames) = mpsc::channel();
        std::thread::Builder::new().name("world-physics".into()).spawn(move || {
            let mut time = 0.0;
            let mut tool = None;
            while input.recv().is_ok() {
                let start = std::time::Instant::now();
                solver::step(&mut scene, &mut tool);
                time += fixed_dt;
                if output.send(PhysicsFrame::capture(&scene, origin, time,
                    start.elapsed().as_secs_f64() * 1_000.0)).is_err() { break; }
            }
        })?;
        Ok(Self { requests, frames, pending_seconds: 0.0, fixed_dt, in_flight: false, frame })
    }

    pub fn advance(&mut self, elapsed: f64, paused: bool) {
        if elapsed.is_finite() && elapsed > 0.0 && !paused { self.pending_seconds += elapsed; }
        match self.frames.try_recv() {
            Ok(frame) => { self.frame = frame; self.in_flight = false; }
            Err(TryRecvError::Disconnected) => return,
            Err(TryRecvError::Empty) => {}
        }
        if !paused && !self.in_flight && self.pending_seconds >= self.fixed_dt {
            if self.requests.send(()).is_ok() {
                self.pending_seconds -= self.fixed_dt;
                self.in_flight = true;
            }
        }
    }

    pub fn backlog_seconds(&self) -> f64 {
        self.pending_seconds + if self.in_flight { self.fixed_dt } else { 0.0 }
    }
}

/// A first sparse activation experiment. Only the visible support patch is a
/// constrained material reservoir; there is no invisible collision plane.
/// The rest of the 20 m square remains implicit and is not yet dynamic.
pub struct PhysicalRegion {
    pub scene: Scene,
    pub origin: Vec3,
}

pub fn contact_experiment(world: &GeneratedWorld, center: Vec3,
    mut config: SceneConfig) -> Result<PhysicalRegion, String> {
    let stone = crate::material::Material::component_id("stone").ok_or("stone material missing")?;
    let wood = crate::material::Material::component_id("oak").ok_or("oak material missing")?;
    config.fixed_dt = 0.010;
    config.substeps = 40;
    config.gravity = [0.0, 0.0, -9.81];
    let cx = (center.x / CELL).floor() as i32;
    let cy = (center.y / CELL).floor() as i32;
    let cz = (world.terrain_height_at(center.x,center.y)/CELL).round() as i32;
    let origin = Vec3::new(cx as f32,cy as f32,cz as f32)*CELL;
    let mut spawns = Vec::new();
    let mut top = f32::NEG_INFINITY;
    for x in cx-20..cx+20 {
        for y in cy-20..cy+20 {
            let height = world.terrain_height_at((x as f32+0.5)*CELL, (y as f32+0.5)*CELL);
            let z = (height / CELL).round() as i32;
            top = top.max(z as f32 * CELL);
            for layer in 1..=2 {
                spawns.push(VoxelSpawn::at_cell([x-cx,y-cy,z-layer-cz],stone).fixed());
            }
        }
    }
    let bottom = ((top + 0.35) / CELL).ceil() as i32;
    for x in cx-12..cx+12 {
        for y in cy-3..cy+3 {
            for z in bottom..bottom+6 {
                spawns.push(VoxelSpawn::at_cell([x-cx,y-cy,z-cz],wood)
                    .with_microstructure(crate::voxel::Microstructure {
                        orientation: Quat::from_rotation_arc(Vec3::Z,Vec3::X), anisotropy: 1.0,
                    }));
            }
        }
    }
    let mut scene = Scene::from_voxels(config,spawns)?;
    scene.set_world_bounds(Vec3::new(center.x-10.0,center.y-10.0,0.0)-origin,
        Vec3::new(center.x+10.0,center.y+10.0,512.0)-origin).map_err(str::to_owned)?;
    Ok(PhysicalRegion { scene, origin })
}
