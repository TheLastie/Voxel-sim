//! The budget governor puts a clamped slender part of a scene on the beam tier when the
//! fine solver would not fit the budget, and takes it back when the beam stops being valid.
use glam::DVec3;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, TierGovernorConfig, VoxelSpawn},
    solver,
};

const GRAVITY: f32 = -2.0e3;

/// A 3x3x31 iron cantilever clamped at x = 0, plus one fixed REV two cells above its tip.
fn scene_with_cantilever() -> (Scene, Vec<usize>, usize) {
    let iron = Material::component_id("iron").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.gravity = [0.0, 0.0, GRAVITY];
    config.atmosphere.pressure_pa = 0.0;
    let (nx, side) = (31, 3);
    let mut spawns = Vec::new();
    for x in 0..nx {
        for y in 0..side {
            for z in 0..side {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    spawns.push(VoxelSpawn::at_cell([nx - 1, 0, 4], iron));
    let index = |x: i32, y: i32, z: i32| (x * side * side + y * side + z) as usize;
    let root: Vec<usize> = (0..side).flat_map(|y| (0..side).map(move |z| (y, z))).map(|(y, z)| index(0, y, z)).collect();
    let tip: Vec<usize> = root.iter().map(|&i| i + ((nx - 1) * side * side) as usize).collect();
    let extra = spawns.len() - 1;
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    scene.fix_voxels(&root).unwrap();
    scene.fix_voxels(&[extra]).unwrap();
    (scene, tip, extra)
}

fn mean_z(scene: &Scene, ids: &[usize]) -> f64 {
    ids.iter().map(|&i| scene.positions[i].z).sum::<f64>() / ids.len() as f64
}

#[test]
fn a_tight_budget_puts_the_cantilever_on_the_beam_tier() {
    let (mut scene, tip, _) = scene_with_cantilever();
    let rest = mean_z(&scene, &tip);
    scene
        .enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() })
        .unwrap();
    for _ in 0..3 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.governed_beam_count(), 1, "the cantilever should run as a beam");
    let plan = scene.tier_plan().expect("a plan").clone();
    assert!(plan.total_cost_ms <= plan.budget_ms + 1.0e-9, "plan {plan:?}");
    for _ in 0..100 {
        solver::step(&mut scene, &mut None);
    }
    let deflection = mean_z(&scene, &tip) - rest;
    eprintln!("GOVERNED beams={} tip deflection {deflection:.4e} plan cost {:.3} ms", scene.governed_beam_count(), plan.total_cost_ms);
    assert!(deflection < 0.0 && deflection > -5.0e-3, "deflection {deflection:e}");
    assert!(scene.positions.iter().all(|p| p.is_finite()));
}

#[test]
fn a_generous_budget_keeps_the_cells() {
    let (mut scene, _, _) = scene_with_cantilever();
    scene
        .enable_tier_governor(TierGovernorConfig { budget_ms: 1.0e4, ..Default::default() })
        .unwrap();
    for _ in 0..25 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.governed_beam_count(), 0);
}

#[test]
fn a_body_that_comes_near_forces_the_cells_back() {
    let (mut scene, tip, extra) = scene_with_cantilever();
    scene
        .enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() })
        .unwrap();
    for _ in 0..3 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.governed_beam_count(), 1);
    // Bring the fixed REV to within a cell of the tip.
    let tip_position = scene.positions[tip[0]];
    scene.positions[extra] = tip_position + DVec3::new(0.0, 0.0, 0.013);
    for _ in 0..12 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.governed_beam_count(), 0, "contact must return the region to cells");
    assert!(scene.positions.iter().all(|p| p.is_finite()));
}

#[test]
fn disabling_the_governor_returns_every_region() {
    let (mut scene, _, _) = scene_with_cantilever();
    scene
        .enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() })
        .unwrap();
    for _ in 0..3 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.governed_beam_count(), 1);
    scene.disable_tier_governor();
    assert_eq!(scene.governed_beam_count(), 0);
    assert!(scene.voxels.iter().enumerate().all(|(i, v)| v.kinematic == (i < 9 || i == 279)));
}

#[test]
fn the_census_counts_every_rev_once_in_the_tier_it_runs_at() {
    let (mut scene, _, _) = scene_with_cantilever();
    let before = scene.tier_census();
    assert_eq!(before.total(), scene.voxels.len());
    assert_eq!((before.fixed, before.cells, before.beam), (10, 270, 0));
    scene
        .enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() })
        .unwrap();
    for _ in 0..3 {
        solver::step(&mut scene, &mut None);
    }
    let after = scene.tier_census();
    assert_eq!(after.total(), scene.voxels.len());
    assert_eq!((after.fixed, after.cells, after.beam), (10, 0, 270));
}
