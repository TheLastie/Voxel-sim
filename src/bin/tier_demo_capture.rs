//! Recording of the physics tiers in one real scene: an iron block settles on a stone
//! floor and falls asleep (T0), a clamped iron rod is put on the beam tier by the budget
//! governor (T2), and at t = 4 s an iron ball is released above the block; it wakes the
//! block before touching it, both settle and sleep again. Every frame is an actual solver
//! state; the wall time of each `solver::step` and the tier census are recorded.
//!
//! Usage: tier_demo_capture [out.csv]
use std::{fs::File, io::{BufWriter, Write}, time::Instant};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, SleepConfig, TierGovernorConfig, VoxelSpawn},
    solver,
};

const STEPS: usize = 1_760;
const EVERY: usize = 10;
const RELEASE_STEP: usize = 400;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "validation/tier-demo.csv".into());
    let iron = Material::component_id("iron").ok_or("iron")?;
    let stone = Material::component_id("stone").ok_or("stone")?;
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.atmosphere.pressure_pa = 0.0;
    config.fracture.friction = 3.0;
    let mut spawns = Vec::new();
    for x in -12..12 {
        for y in -12..12 {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    for z in 1..=8 {
        for x in -4..4 {
            for y in -4..4 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    // Clamped rod: 31 slices of 3x3 along +X, root slice fixed, well above the floor.
    let rod_start = spawns.len();
    for x in 14..45 {
        for y in -1..2 {
            for z in 14..17 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let rod_root: Vec<usize> = (rod_start..rod_start + 9).collect();
    let ball_start = spawns.len();
    for z in 40..43 {
        for x in -1..2 {
            for y in -1..2 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let ball: Vec<usize> = (ball_start..spawns.len()).collect();
    let mut scene = Scene::from_voxels(config, spawns)?;
    scene.fix_voxels(&rod_root)?;
    scene.fix_voxels(&ball)?;
    scene.enable_sleep(SleepConfig::default())?;
    scene.enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() })?;

    let mut out = BufWriter::new(File::create(&out_path)?);
    writeln!(out, "H,{},{STEPS},{EVERY},{RELEASE_STEP}", scene.config.fixed_dt)?;
    let write_state = |out: &mut BufWriter<File>, scene: &Scene, step: usize, ms: f64| -> std::io::Result<()> {
        let c = scene.tier_census();
        writeln!(out, "S,{step},{ms:.4},{},{},{},{},{}", c.sleeping, c.fixed, c.rigid, c.beam, c.cells)?;
        for (i, (p, t)) in scene.positions.iter().zip(scene.rev_tiers()).enumerate() {
            writeln!(out, "V,{step},{i},{:.6},{:.6},{:.6},{t}", p.x, p.y, p.z)?;
        }
        Ok(())
    };
    write_state(&mut out, &scene, 0, 0.0)?;
    let mut window_ms = Vec::new();
    let mut all_ms = Vec::with_capacity(STEPS);
    for step in 1..=STEPS {
        if step == RELEASE_STEP {
            scene.release_voxels(&ball)?;
        }
        let started = Instant::now();
        solver::step(&mut scene, &mut None);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        all_ms.push(ms);
        window_ms.push(ms);
        if !scene.positions.iter().all(|p| p.is_finite()) {
            return Err(format!("non-finite state at step {step}").into());
        }
        if step % EVERY == 0 {
            let mean = window_ms.iter().sum::<f64>() / window_ms.len() as f64;
            window_ms.clear();
            write_state(&mut out, &scene, step, mean)?;
        }
    }
    let mut sorted = all_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |f: f64| sorted[((sorted.len() - 1) as f64 * f) as usize];
    writeln!(out, "T,{:.4},{:.4},{:.4}", at(0.5), at(0.95), sorted[sorted.len() - 1])?;
    eprintln!(
        "tier_demo_capture REVs={} steps={STEPS} states={} p50={:.3} ms p95={:.3} ms max={:.1} ms final census {:?} -> {out_path}",
        scene.voxels.len(), STEPS / EVERY + 1, at(0.5), at(0.95), sorted[sorted.len() - 1], scene.tier_census()
    );
    Ok(())
}
