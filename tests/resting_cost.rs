//! What does a body at rest cost per step? Probe for the "normal game scene" budget:
//! most of the world is still, and the solver should not pay the fine-solver price for
//! it. Run with `--ignored --nocapture`; the numbers are wall time of `solver::step` on
//! the machine running it, headless.
use std::time::Instant;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, SleepConfig, VoxelSpawn},
    solver,
};

fn resting(width: i32, height: i32, steps: usize, sleep: bool, pebble: bool) -> Vec<f64> {
    let iron = Material::component_id("iron").unwrap();
    let stone = Material::component_id("stone").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.atmosphere.pressure_pa = 0.0;
    config.fracture.friction = 3.0;
    let half = width / 2 + 4;
    let mut spawns = Vec::new();
    for x in -half..half {
        for y in -half..half {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let lo = -(width / 2);
    for z in 1..=height {
        for x in lo..lo + width {
            for y in lo..lo + width {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    if pebble {
        // One awake REV falling from 20 m for the whole run: the world is not dormant.
        spawns.push(VoxelSpawn::at_cell([lo - 6, lo - 6, 2000], iron));
    }
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    if sleep {
        scene.enable_sleep(SleepConfig::default()).unwrap();
    }
    let mut times = Vec::new();
    for _ in 0..steps {
        let started = Instant::now();
        solver::step(&mut scene, &mut None);
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let census = scene.tier_census();
    eprintln!("  census after {steps} steps: {census:?}");
    times
}

fn report(label: &str, times: &[f64]) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |f: f64| sorted[((sorted.len() - 1) as f64 * f) as usize];
    // Steps slower than 20 ms with their index: these are the sleep and wake events.
    let slow: Vec<(usize, f64)> = times.iter().enumerate().filter(|(_, t)| **t > 20.0).map(|(i, t)| (i, t.round())).collect();
    eprintln!("  slow steps (index, ms): {} of {}, last at {:?}", slow.len(), times.len(), slow.last().map(|s| s.0));
    if let Some(&(last, _)) = slow.last() {
        // Steps after the last sleep or wake event: the steady cost of the sleeping scene.
        let mut steady = times[last + 1..].to_vec();
        if !steady.is_empty() {
            steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let at = |f: f64| steady[((steady.len() - 1) as f64 * f) as usize];
            eprintln!(
                "  steady after the last event ({} steps): p50={:.2} ms p95={:.2} ms max={:.2} ms",
                steady.len(), at(0.5), at(0.95), steady[steady.len() - 1]
            );
        }
    }
    eprintln!(
        "{label}: first={:.1} ms p50={:.1} ms p95={:.1} ms max={:.1} ms last5={:?}",
        times[0],
        at(0.5),
        at(0.95),
        sorted[sorted.len() - 1],
        &times[times.len().saturating_sub(5)..].iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "timing probe, tens of seconds"]
fn a_block_at_rest_costs_little_when_it_sleeps() {
    // REST_ONLY_AWAKE=1 runs just the "sleep on + 1 awake" case (the other two are slow at 27k).
    let only_awake = std::env::var_os("REST_ONLY_AWAKE").is_some();
    for (width, height) in [(14, 14), (20, 25), (30, 30)] {
        let active = width * width * height;
        if !only_awake {
            report(&format!("rest {width}x{width}x{height} ({active} REVs) sleep off          "), &resting(width, height, 60, false, false));
            report(&format!("rest {width}x{width}x{height} ({active} REVs) sleep on, dormant  "), &resting(width, height, 60, true, false));
        }
        report(&format!("rest {width}x{width}x{height} ({active} REVs) sleep on + 1 awake "), &resting(width, height, 150, true, true));
    }
}
