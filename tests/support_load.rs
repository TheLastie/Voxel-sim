//! A body at rest on a fixed floor: the reported support loads must carry its weight.
use glam::Vec3;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

fn resting_block(width: i32, height: i32) -> (Scene, usize) {
    let iron = Material::component_id("iron").unwrap();
    let stone = Material::component_id("stone").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.gravity = [0.0, 0.0, -9.81];
    config.atmosphere.pressure_pa = 0.0;
    let mut spawns = Vec::new();
    for x in -8..8 {
        for y in -8..8 {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let first = spawns.len();
    for z in 1..=height {
        for x in 0..width {
            for y in 0..width {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    (Scene::from_voxels(config, spawns).unwrap(), first)
}

#[test]
fn support_loads_carry_the_weight_of_a_resting_block() {
    let (mut scene, first) = resting_block(4, 3);
    for _ in 0..120 {
        solver::step(&mut scene, &mut None);
    }
    let weight: f64 = scene.voxels[first..].iter().map(|v| f64::from(v.mass) * 9.81).sum();
    let carried: f64 = scene.support_load_n.iter().map(|&n| f64::from(n)).sum();
    let in_body: f64 = scene.support_load_n[first..].iter().map(|&n| f64::from(n)).sum();
    eprintln!("SUPPORT weight={weight:.4} N carried={carried:.4} N (body REVs {in_body:.4} N)");
    assert!((carried / weight - 1.0).abs() < 0.05, "support {carried} N against weight {weight} N");
    // Only the bottom layer touches the floor.
    let layer = 16;
    let lifted = scene.support_load_n[first + layer..].iter().filter(|&&n| n > 1.0e-6).count();
    assert_eq!(lifted, 0, "upper layers must not report a support load");
    let _ = Vec3::ZERO;
}

#[test]
fn a_block_in_free_fall_reports_no_support() {
    let (mut scene, first) = resting_block(2, 2);
    for p in scene.positions[first..].iter_mut() {
        p.z += 0.5;
    }
    solver::step(&mut scene, &mut None);
    assert!(scene.support_load_n.iter().all(|&n| n == 0.0));
}
