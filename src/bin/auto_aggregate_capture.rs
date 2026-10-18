//! CPU benchmark for conservative automatic rigid aggregation. A set of
//! stress-free oak blocks falls freely under uniform gravity; there are no
//! collisions, constraints, or material-specific forces in this workload.
//! Set `VOXEL_DISABLE_AUTO_AGGREGATION=1` for the matched fine-tier control.

use std::{env, time::Instant};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

const BODY_SIDE: usize = 4;
const BODY_COUNT_X: usize = 8;
const BODY_COUNT_Y: usize = 8;
const FIXED_STEPS: usize = 84;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let count_x: usize = args.get(1).map(|x| x.parse()).transpose()?.unwrap_or(BODY_COUNT_X);
    let count_y: usize = args.get(2).map(|x| x.parse()).transpose()?.unwrap_or(BODY_COUNT_Y);
    let steps: usize = args.get(3).map(|x| x.parse()).transpose()?.unwrap_or(FIXED_STEPS);
    if count_x == 0 || count_y == 0 || steps <= 64 {
        return Err("body counts must be positive and steps must exceed the 64-step discovery interval".into());
    }
    let auto_enabled = env::var_os("VOXEL_DISABLE_AUTO_AGGREGATION").is_none();
    let oak = Material::component_id("oak").ok_or("oak material is unavailable")?;
    let body_count = count_x * count_y;
    let mut spawns = Vec::with_capacity(body_count * BODY_SIDE.pow(3));
    for body_y in 0..count_y {
        for body_x in 0..count_x {
            for z in 0..BODY_SIDE {
                for y in 0..BODY_SIDE {
                    for x in 0..BODY_SIDE {
                        let cell = [
                            (body_x * (BODY_SIDE + 2) + x) as i32,
                            (body_y * (BODY_SIDE + 2) + y) as i32,
                            z as i32 + 8,
                        ];
                        spawns.push(VoxelSpawn::at_cell(cell, oak).at_temperature(293.15));
                    }
                }
            }
        }
    }

    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 1;
    config.gravity = [0.0, 0.0, -9.81];
    // An explicitly isolated reference workload. Ambient drag creates
    // differential surface loading and invalidates the rigid free-fall case.
    // This setting applies only to this benchmark, not the interactive world.
    config.atmosphere.pressure_pa = 0.0;
    config.cpu_preconditioner = voxel_physics_lab::scene::CpuPreconditioner::IncompleteCholesky0;
    let mut scene = Scene::from_voxels(config, spawns)?;
    let initial_positions = scene.positions.clone();
    let initial_momentum = scene
        .voxels
        .iter()
        .fold(glam::DVec3::ZERO, |sum, voxel| {
            sum + glam::DVec3::new(
                f64::from(voxel.mass * voxel.velocity.x),
                f64::from(voxel.mass * voxel.velocity.y),
                f64::from(voxel.mass * voxel.velocity.z),
            )
        });
    let mut durations_ms = Vec::with_capacity(steps);
    let mut scan_step_ms = 0.0;
    let mut tool = None;
    eprintln!("free-fall reference: bodies={body_count} revs={} atmosphere_pa=0 steps={steps} auto={auto_enabled}", scene.voxels.len());
    for step in 1..=steps {
        let start = Instant::now();
        solver::step(&mut scene, &mut tool);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        durations_ms.push(elapsed_ms);
        if step == 64 {
            scan_step_ms = elapsed_ms;
        }
        if step == 1 || step % 16 == 0 || step == steps {
            eprintln!("free-fall step={step}/{steps} step_ms={elapsed_ms:.3} broken={}", scene.broken);
        }
    }

    let elapsed_sim_s = f64::from(scene.config.fixed_dt) * steps as f64;
    let gravity_z = f64::from(scene.config.gravity[2]);
    let mut max_position_error_m = 0.0_f64;
    let mut max_velocity_error_m_s = 0.0_f64;
    let mut final_momentum = glam::DVec3::ZERO;
    for (index, voxel) in scene.voxels.iter().enumerate() {
        let expected_z = f64::from(initial_positions[index].z)
            + 0.5 * gravity_z * elapsed_sim_s * elapsed_sim_s;
        max_position_error_m = max_position_error_m.max(
            (f64::from(scene.positions[index].z) - expected_z).abs(),
        );
        max_velocity_error_m_s = max_velocity_error_m_s
            .max((f64::from(voxel.velocity.z) - gravity_z * elapsed_sim_s).abs());
        final_momentum += glam::DVec3::new(
            f64::from(voxel.mass * voxel.velocity.x),
            f64::from(voxel.mass * voxel.velocity.y),
            f64::from(voxel.mass * voxel.velocity.z),
        );
    }
    let expected_momentum = initial_momentum
        + glam::DVec3::new(0.0, 0.0, gravity_z * elapsed_sim_s)
            * scene.voxels.iter().map(|voxel| f64::from(voxel.mass)).sum::<f64>();
    let mut aggregate_members = 0usize;
    let mut aggregate_count = 0usize;
    for index in 0..scene.voxels.len() {
        if scene.aggregate_for_voxel(index).is_some() {
            aggregate_members += 1;
            if index == 0 || scene.aggregate_for_voxel(index - 1) != scene.aggregate_for_voxel(index) {
                aggregate_count += 1;
            }
        }
    }
    // Ignore the first 64 updates and the discovery scan when reporting the
    // steady-state throughput after candidate bodies have formed.
    let steady_median_ms = median(durations_ms[64..].to_vec());
    println!(
        "mode={} bodies={} revs={} auto_aggregated_revs={} aggregates={} scan_step_ms={:.3} steady_median_ms={:.3} broken={} max_free_fall_position_error_m={:.3e} max_free_fall_velocity_error_m_s={:.3e} momentum_error_kg_m_s={:.3e}",
        if auto_enabled { "automatic" } else { "control" },
        body_count,
        scene.voxels.len(),
        aggregate_members,
        aggregate_count,
        scan_step_ms,
        steady_median_ms,
        scene.broken,
        max_position_error_m,
        max_velocity_error_m_s,
        (final_momentum - expected_momentum).length(),
    );
    Ok(())
}
