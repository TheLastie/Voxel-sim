//! A tall iron block on a fixed floor under tilted gravity must tip about its edge.
//!
//! The rigid-body solution is the reference. A translation-only solver reaches
//! about 1e-4 of it because bond shear resists rotation; the cell solver must stay
//! within 25%. The large sizes are timing probes, run with `--ignored`.
use std::time::Instant;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

fn tip(width: i32, height: i32, steps: usize) -> (f64, f64, f64) {
    let iron = Material::component_id("iron").unwrap();
    let stone = Material::component_id("stone").unwrap();
    let gx = 1.2f64 * 9.81;
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.gravity = [gx as f32, 0.0, -9.81];
    config.atmosphere.pressure_pa = 0.0;
    config.fracture.friction = 3.0;
    let half = width / 2 + 8;
    let mut spawns = Vec::new();
    for x in -half..half {
        for y in -half..half {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let floor = spawns.len();
    let lo = -(width / 2);
    for z in 1..=height {
        for x in lo..lo + width {
            for y in lo..lo + width {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    for p in scene.properties.iter_mut() {
        p.thermal_expansion_parallel = 0.0;
        p.thermal_expansion_perpendicular = 0.0;
    }
    let count = (scene.voxels.len() - floor) as f64;
    let mean_height = |scene: &Scene| scene.positions[floor..].iter().map(|p| p.z).sum::<f64>() / count;
    let start = mean_height(&scene);
    let started = Instant::now();
    for _ in 0..steps {
        solver::step(&mut scene, &mut None);
        assert!(scene.solver_residual.is_finite());
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let p = &solver::CELL_PROFILE;
    let ms = |k: usize| p[k].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e6 / steps as f64;
    eprintln!(
        "  PHASES per step: assemble={:.0} build={:.0} cg={:.0} (matrix={:.0} precondition={:.0} of which bottom={:.0}) ms",
        ms(0), ms(1), ms(2), ms(5), ms(6), ms(7)
    );
    // Rigid block pivoting about its front bottom edge, integrated exactly.
    let (w, h) = (f64::from(width) * 0.01, f64::from(height) * 0.01);
    let (hw, hh) = (w * 0.5, h * 0.5);
    let inertia = (w * w + h * h) / 12.0 + hw * hw + hh * hh;
    let (mut angle, mut rate) = (0.0f64, 0.0f64);
    let dt = 1.0e-5;
    for _ in 0..(steps as f64 * 0.01 / dt) as usize {
        let x = -hw * angle.cos() + hh * angle.sin();
        let z = hw * angle.sin() + hh * angle.cos();
        rate += (gx * z + 9.81 * x) / inertia * dt;
        angle += rate * dt;
    }
    let rigid = hw * angle.sin() + hh * angle.cos() - hh;
    (mean_height(&scene) - start, rigid, wall_ms)
}

#[test]
fn small_block_tips_like_a_rigid_body() {
    let (rise, rigid, ms) = tip(6, 10, 10);
    eprintln!("TIP 6x6x10: rise={rise:.3e} rigid={rigid:.3e} ratio={:.3} step={ms:.0} ms", rise / rigid);
    assert!((rise / rigid - 1.0).abs() < 0.25, "rise {rise} against rigid {rigid}");
}

#[test]
#[ignore = "timing probe, several seconds"]
fn bulk_blocks_tip_at_every_size() {
    for (width, height) in [(10, 16), (14, 28), (20, 40)] {
        std::env::set_var("VOXEL_FACE_CELLS", "1");
        let (rise, rigid, ms) = tip(width, height, 6);
        eprintln!(
            "TIP {width}x{width}x{height}: rise={rise:.3e} rigid={rigid:.3e} ratio={:.3} step={ms:.0} ms",
            rise / rigid
        );
        assert!((rise / rigid - 1.0).abs() < 0.25, "{width}x{height}: ratio {}", rise / rigid);
    }
}
