//! The physics tiers must work together in one scene, not only one at a time. These
//! reproduce what the tier demo recording showed on 7 November.
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, SleepConfig, TierGovernorConfig, VoxelSpawn},
    solver,
};

/// Stone floor, iron block on it, and a clamped iron rod 3x3x31 well above the floor.
fn demo_scene(with_floor_and_block: bool) -> Scene {
    let iron = Material::component_id("iron").unwrap();
    let stone = Material::component_id("stone").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.atmosphere.pressure_pa = 0.0;
    config.fracture.friction = 3.0;
    let mut spawns = Vec::new();
    if with_floor_and_block {
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
    }
    let rod_start = spawns.len();
    for x in 14..45 {
        for y in -1..2 {
            for z in 14..17 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    let root: Vec<usize> = (rod_start..rod_start + 9).collect();
    scene.fix_voxels(&root).unwrap();
    scene
}

#[test]
fn the_governor_reduces_the_rod_beside_other_bodies() {
    for with_others in [false, true] {
        let mut scene = demo_scene(with_others);
        scene.enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() }).unwrap();
        for _ in 0..3 {
            solver::step(&mut scene, &mut None);
        }
        eprintln!("with_others={with_others} census {:?} plan {:?}", scene.tier_census(), scene.tier_plan());
        assert_eq!(scene.governed_beam_count(), 1, "with_others={with_others}");
    }
}

#[test]
fn sleep_does_not_take_the_rod_before_the_governor() {
    let mut scene = demo_scene(true);
    scene.enable_sleep(SleepConfig::default()).unwrap();
    scene.enable_tier_governor(TierGovernorConfig { budget_ms: 1.0, ..Default::default() }).unwrap();
    for _ in 0..60 {
        solver::step(&mut scene, &mut None);
    }
    eprintln!("sleep+governor census {:?}", scene.tier_census());
    assert_eq!(scene.governed_beam_count(), 1);
}
