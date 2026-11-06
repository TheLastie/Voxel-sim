//! The sleeping tier freezes settled clusters without changing what the world does.
use glam::DVec3;
use std::time::Instant;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, SleepConfig, VoxelSpawn},
    solver,
};

/// A stone floor, an iron block resting on it, and an iron ball held still above it.
fn world() -> (Scene, Vec<usize>, Vec<usize>) {
    let iron = Material::component_id("iron").unwrap();
    let stone = Material::component_id("stone").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.atmosphere.pressure_pa = 0.0;
    config.fracture.friction = 3.0;
    let mut spawns = Vec::new();
    for x in -8..8 {
        for y in -8..8 {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let block_start = spawns.len();
    for z in 1..=6 {
        for x in -3..3 {
            for y in -3..3 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let block: Vec<usize> = (block_start..spawns.len()).collect();
    let ball_start = spawns.len();
    for z in 20..23 {
        for x in -1..1 {
            for y in -1..1 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let ball: Vec<usize> = (ball_start..spawns.len()).collect();
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    scene.fix_voxels(&ball).unwrap();
    (scene, block, ball)
}

fn mean(scene: &Scene, ids: &[usize]) -> DVec3 {
    ids.iter().map(|&i| scene.positions[i]).sum::<DVec3>() / ids.len() as f64
}

#[test]
fn a_settled_block_sleeps_and_costs_almost_nothing() {
    let (mut scene, block, _) = world();
    scene.enable_sleep(SleepConfig::default()).unwrap();
    for _ in 0..80 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.sleeping_rev_count(), block.len(), "census {:?}", scene.tier_census());
    let census = scene.tier_census();
    assert_eq!(census.total(), scene.voxels.len());
    assert_eq!(census.sleeping, block.len());
    let before = scene.positions.clone();
    let started = Instant::now();
    for _ in 0..20 {
        solver::step(&mut scene, &mut None);
    }
    let per_step_ms = started.elapsed().as_secs_f64() * 1000.0 / 20.0;
    eprintln!("SLEEPING {} REVs, {per_step_ms:.2} ms per step", scene.sleeping_rev_count());
    assert_eq!(scene.positions, before, "sleepers must not move");
}

#[test]
fn a_ball_dropped_on_a_sleeping_block_lands_as_it_would_awake() {
    let run = |sleep: bool| {
        let (mut scene, block, ball) = world();
        if sleep {
            scene.enable_sleep(SleepConfig::default()).unwrap();
        }
        for _ in 0..80 {
            solver::step(&mut scene, &mut None);
        }
        if sleep {
            assert_eq!(scene.sleeping_rev_count(), block.len());
        }
        scene.release_voxels(&ball).unwrap();
        let mut woke_at = None;
        for step in 0..120 {
            solver::step(&mut scene, &mut None);
            if woke_at.is_none() && sleep && scene.sleeping_rev_count() == 0 {
                woke_at = Some(step);
            }
        }
        assert!(scene.positions.iter().all(|p| p.is_finite()));
        (mean(&scene, &ball), mean(&scene, &block), woke_at)
    };
    let (ball_awake, block_awake, _) = run(false);
    let (ball_sleep, block_sleep, woke_at) = run(true);
    eprintln!(
        "DROP ball awake z={:.5} sleep z={:.5}; block awake z={:.5} sleep z={:.5}; woke at step {woke_at:?}",
        ball_awake.z, ball_sleep.z, block_awake.z, block_sleep.z
    );
    assert!(woke_at.is_some(), "the falling ball should have woken the block");
    assert!((ball_awake.z - ball_sleep.z).abs() < 5.0e-4, "ball {} against {}", ball_sleep.z, ball_awake.z);
    assert!((block_awake.z - block_sleep.z).abs() < 5.0e-4, "block {} against {}", block_sleep.z, block_awake.z);
}

#[test]
fn a_force_on_a_sleeper_wakes_it() {
    let (mut scene, block, _) = world();
    scene.enable_sleep(SleepConfig::default()).unwrap();
    for _ in 0..80 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.sleeping_rev_count(), block.len());
    voxel_physics_lab::solver::set_force(&mut scene, block[0], glam::Vec3::new(1.0, 0.0, 0.0)).unwrap();
    solver::step(&mut scene, &mut None);
    assert_eq!(scene.sleeping_rev_count(), 0);
}

#[test]
fn a_checkpoint_stores_sleepers_as_free() {
    let (mut scene, block, _) = world();
    scene.enable_sleep(SleepConfig::default()).unwrap();
    for _ in 0..80 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.sleeping_rev_count(), block.len());
    let path = std::env::temp_dir().join("voxel_sleep_checkpoint_test.vxstate");
    scene.save_checkpoint(&path).unwrap();
    let restored = Scene::load_checkpoint(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(block.iter().all(|&i| !restored.voxels[i].kinematic), "sleepers must not be stored as fixed");
}

#[test]
fn heat_reaching_a_sleeper_wakes_it() {
    let (mut scene, block, _) = world();
    scene.enable_sleep(SleepConfig::default()).unwrap();
    for _ in 0..80 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.sleeping_rev_count(), block.len());
    assert!(scene.add_heat_joules(block[0], 50.0));
    solver::step(&mut scene, &mut None);
    assert_eq!(scene.sleeping_rev_count(), 0, "a change of internal energy must wake the cluster");
}

#[test]
fn a_scene_where_everything_sleeps_only_advances_time() {
    let (mut scene, block, _) = world();
    scene.enable_sleep(SleepConfig::default()).unwrap();
    for _ in 0..80 {
        solver::step(&mut scene, &mut None);
    }
    assert_eq!(scene.sleeping_rev_count(), block.len());
    let (positions, substeps, seconds) = (scene.positions.clone(), scene.substep_index, scene.simulated_seconds);
    let started = Instant::now();
    for _ in 0..50 {
        solver::step(&mut scene, &mut None);
    }
    let per_step_ms = started.elapsed().as_secs_f64() * 1000.0 / 50.0;
    eprintln!("DORMANT step {per_step_ms:.3} ms");
    assert_eq!(scene.positions, positions);
    assert_eq!(scene.substep_index, substeps + 50 * 4);
    assert!((scene.simulated_seconds - seconds - 0.5).abs() < 1.0e-6, "time must keep running");
    assert_eq!(scene.sleeping_rev_count(), block.len());
}
