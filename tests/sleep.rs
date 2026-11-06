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
    let (scene, block, ball, _) = world_plus(None);
    (scene, block, ball)
}

/// The same, plus one free iron REV at `pebble` (a cell) when given.
fn world_plus(pebble: Option<[i32; 3]>) -> (Scene, Vec<usize>, Vec<usize>, Option<usize>) {
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
    let pebble_index = pebble.map(|cell| {
        spawns.push(VoxelSpawn::at_cell(cell, iron));
        spawns.len() - 1
    });
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    scene.fix_voxels(&ball).unwrap();
    (scene, block, ball, pebble_index)
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

/// Everything the scene holds per REV that a step may change.
fn state(scene: &Scene) -> Vec<f64> {
    let mut values = Vec::new();
    for i in 0..scene.voxels.len() {
        values.extend(scene.positions[i].to_array());
        values.extend(scene.previous[i].to_array());
        values.extend(scene.voxels[i].velocity.to_array().map(f64::from));
        values.push(scene.voxels[i].total_internal_energy_j());
        values.push(f64::from(scene.voxels[i].kinematic));
        values.push(f64::from(scene.support_load_n.get(i).copied().unwrap_or(0.0)));
        for axis in 0..3 {
            values.push(f64::from(scene.bonds[axis][i].damage));
        }
    }
    values.push(scene.energy.implicit_solver_heat);
    values.push(scene.energy.contact_friction_heat);
    values.push(scene.energy.contact_restitution_heat);
    values.push(scene.energy.external_work);
    values
}

#[test]
fn an_awake_rev_far_from_the_sleepers_moves_as_it_does_with_sleep_off() {
    let run = |sleep: bool| {
        let (mut scene, block, _, pebble) = world_plus(Some([-20, -20, 2000]));
        let pebble = pebble.unwrap();
        if sleep {
            scene.enable_sleep(SleepConfig::default()).unwrap();
        }
        let start = scene.positions[pebble];
        for _ in 0..90 {
            solver::step(&mut scene, &mut None);
        }
        if sleep {
            assert_eq!(scene.sleeping_rev_count(), block.len(), "census {:?}", scene.tier_census());
        }
        (start, scene.positions[pebble], scene.voxels[pebble].velocity)
    };
    let (start, awake_position, awake_velocity) = run(false);
    let (_, sleep_position, sleep_velocity) = run(true);
    eprintln!("PEBBLE awake {awake_position:?} sleep {sleep_position:?}");
    assert!((awake_position - sleep_position).abs().max_element() < 1.0e-9, "{awake_position:?} against {sleep_position:?}");
    assert!((awake_velocity - sleep_velocity).abs().max_element() < 1.0e-9);
    assert!(start.z - awake_position.z > 1.0, "the pebble should have fallen");
}

/// Sparse passes visit only awake REVs; the world must come out the same as when they
/// visit everything, through sleeping, contacts with the floor and with a sleeper, a
/// wake-up and a moving fixed body.
#[test]
fn sparse_passes_give_the_same_state_as_full_passes() {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Case {
        PebbleLandsOnFloor,
        BallWakesBlock,
        PebblePushedAcrossSleepingBlock,
        FixedBallMovesFarAway,
    }
    for case in [
        Case::PebbleLandsOnFloor,
        Case::BallWakesBlock,
        Case::PebblePushedAcrossSleepingBlock,
        Case::FixedBallMovesFarAway,
    ] {
        let run = |sparse: bool| {
            let pebble_cell = match case {
                Case::PebbleLandsOnFloor => [7, 7, 14],
                // On top of the block: in contact with a sleeper for the whole run.
                Case::PebblePushedAcrossSleepingBlock => [0, 0, 7],
                _ => [-20, -20, 2000],
            };
            let (mut scene, block, ball, pebble) = world_plus(Some(pebble_cell));
            let mut config = SleepConfig::default();
            config.sparse_passes = sparse;
            scene.enable_sleep(config).unwrap();
            let mut trace = Vec::new();
            let mut asleep_steps = 0;
            for step in 0..170 {
                if step == 90 && case == Case::PebblePushedAcrossSleepingBlock {
                    // Both have settled and sleep. A steady push wakes the pebble while it
                    // rests on sleepers; later the block wakes too. Both phases must match.
                    assert!(scene.sleeping_rev_count() > block.len(), "{case:?}: census {:?}", scene.tier_census());
                    solver::set_force(&mut scene, pebble.unwrap(), glam::Vec3::new(0.02, 0.0, 0.0)).unwrap();
                }
                if step == 90 && case == Case::BallWakesBlock {
                    assert_eq!(scene.sleeping_rev_count(), block.len(), "{case:?}: census {:?}", scene.tier_census());
                    scene.release_voxels(&ball).unwrap();
                }
                if step >= 90 && case == Case::FixedBallMovesFarAway {
                    for &i in &ball {
                        scene.positions[i].x += 2.0e-4;
                    }
                }
                solver::step(&mut scene, &mut None);
                asleep_steps += usize::from(scene.sleeping_rev_count() > 0);
                trace.push((state(&scene), scene.sleeping_rev_count()));
            }
            assert!(asleep_steps > 20, "{case:?}: something should have slept for a while");
            trace
        };
        let full = run(false);
        let sparse = run(true);
        let mut worst = 0.0_f64;
        for (step, ((a, a_asleep), (b, b_asleep))) in full.iter().zip(&sparse).enumerate() {
            assert_eq!(a_asleep, b_asleep, "{case:?}: step {step} sleeper count");
            assert_eq!(a.len(), b.len());
            for (k, (x, y)) in a.iter().zip(b).enumerate() {
                let diff = (x - y).abs();
                worst = worst.max(diff);
                assert!(diff <= 1.0e-12 * x.abs().max(1.0), "{case:?}: step {step} value {k}: {x} against {y}");
            }
        }
        eprintln!("SPARSE {case:?}: largest difference {worst:e}, sleepers at the end {}", sparse.last().unwrap().1);
    }
}
