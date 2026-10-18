//! Physical scaling workload: an iron cube loaded by gravity and supported
//! through contact with a kinematic bottom layer. No prescribed trajectories.
//! Arguments: side in 1 cm REVs, number of 1 ms updates, ic0|multigrid.

use std::{env, time::Instant};
use voxel_physics_lab::{
    material::Material,
    scene::{CpuPreconditioner, Scene, SceneConfig, VoxelSpawn},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let side: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(16);
    let steps: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    if side < 2 || steps == 0 {
        return Err("side must be at least 2 and steps must be positive".into());
    }
    let count = side.checked_pow(3).ok_or("voxel count overflow")?;
    if side > i32::MAX as usize {
        return Err("voxel coordinate exceeds i32".into());
    }
    let preconditioner = match args.get(3).map(String::as_str).unwrap_or("ic0") {
        "ic0" => CpuPreconditioner::IncompleteCholesky0,
        "multigrid" => CpuPreconditioner::AggregationMultigrid,
        _ => return Err("preconditioner must be ic0 or multigrid".into()),
    };
    let config = SceneConfig {
        fixed_dt: 0.001,
        substeps: 4,
        iterations: 1024,
        cpu_preconditioner: preconditioner,
        ..SceneConfig::default()
    };
    let iron = Material::component_id("iron").ok_or("iron material is unavailable")?;
    let mut spawns = Vec::with_capacity(count);
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let voxel = VoxelSpawn::at_cell([x as i32, y as i32, z as i32], iron);
                spawns.push(if z == 0 { voxel.fixed() } else { voxel });
            }
        }
    }
    let prepare = Instant::now();
    let mut scene = Scene::from_voxels(config, spawns)?;
    eprintln!("mechanics scale: side={side} revs={count} active={} fixed_dt_s=0.001 substeps=4 cpu_preconditioner={preconditioner:?} tolerance={} setup_s={:.3}",
        count-side*side, scene.config.solver_relative_tolerance, prepare.elapsed().as_secs_f64());
    println!("step,revs,active,wall_ms,residual,broken,mass_kg,kinetic_j,true_residual,applied_residual,position_rounding_l2_m,max_audited_residual");
    for step in 1..=steps {
        let clock = Instant::now();
        scene.tick();
        let wall_ms = clock.elapsed().as_secs_f64()*1000.0;
        let mass: f64 = scene.voxels.iter().map(|v| v.mass as f64).sum();
        let kinetic: f64 = scene.voxels.iter().map(|v|
            0.5 * v.mass as f64 * v.velocity.as_dvec3().length_squared()).sum();
        let audit = scene.solver_residual_audit.map(|a| format!("{:.9e},{:.9e},{:.9e}",
            a.correction_relative,a.applied_relative,a.position_rounding_l2_m))
            .unwrap_or_else(|| ",,".into());
        let worst_audit = scene.solver_max_audited_residual
            .map(|x| format!("{x:.9e}")).unwrap_or_default();
        println!("{step},{count},{},{wall_ms:.3},{:.9e},{},{mass:.9e},{kinetic:.9e},{audit},{worst_audit}",
            count-side*side, scene.solver_residual, scene.broken);
        if !scene.solver_residual.is_finite()
            || scene.solver_residual >= scene.config.solver_relative_tolerance
        {
            return Err("fine mechanical solve did not reach its unchanged residual tolerance".into());
        }
        if let Some(worst) = scene.solver_max_audited_residual {
            if !worst.is_finite() || worst >= scene.config.solver_relative_tolerance as f64 {
                return Err("independent fine-operator residual or applied-coordinate residual exceeds tolerance".into());
            }
        }
    }
    Ok(())
}
