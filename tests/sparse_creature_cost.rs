//! Cost of a step with one actuated creature among sleepers, sparse against full passes.
//! The saved spider scene gets metal cubes dropped on its ground; they settle and sleep
//! while the spider keeps walking. Run with `--ignored --nocapture`; wall time of
//! `solver::step`, headless, on the machine running it.
use std::path::{Path, PathBuf};
use std::time::Instant;
use voxel_physics_lab::{
    documents::{load_document, ModelInstanceDocument, SceneDocument},
    scene::{Scene, SleepConfig},
    solver,
};

const SCENE_PATH: &str = "scenes/spider_flat_physics_2026-11-04.vxscene";

fn spider_with_cubes(cubes: usize) -> Scene {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SCENE_PATH);
    let mut document: SceneDocument = load_document(&path).unwrap();
    let probe = document.build_scene(&path).unwrap();
    let actor = probe.skeletal_actuators[0].voxel_range();
    let (mut lo, mut hi) = (glam::DVec3::splat(f64::INFINITY), glam::DVec3::splat(f64::NEG_INFINITY));
    for (i, p) in probe.positions.iter().enumerate() {
        if probe.voxels[i].kinematic {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    for k in 0..cubes {
        let x = lo.x + 0.08 + 0.12 * k as f64;
        document.models.push(ModelInstanceDocument {
            asset_path: PathBuf::from("two_metal_cubes_slow-assets/cube.vxm"),
            translation_m: (glam::DVec3::new(x, lo.y + 0.08, hi.z + 0.06) * 100.0).round() / 100.0,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        });
    }
    let scene = document.build_scene(&path).unwrap();
    eprintln!(
        "scene: {} REVs, actor {:?} ({} REVs), ground x {:.2}..{:.2} y {:.2}..{:.2} top {:.3}",
        scene.voxels.len(), actor, actor.len(), lo.x, hi.x, lo.y, hi.y, hi.z
    );
    scene
}

fn run(cubes: usize, sparse: bool, steps: usize) -> (Vec<f64>, Vec<bool>, Vec<glam::DVec3>, usize) {
    let mut scene = spider_with_cubes(cubes);
    let mut config = SleepConfig::default();
    config.sparse_passes = sparse;
    scene.enable_sleep(config).unwrap();
    let mut times = Vec::new();
    let mut asleep = Vec::new();
    for _ in 0..steps {
        let started = Instant::now();
        solver::step(&mut scene, &mut None);
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        asleep.push(scene.sleeping_rev_count() > 0);
    }
    eprintln!("  census {:?}", scene.tier_census());
    (times, asleep, scene.positions.clone(), scene.sleeping_rev_count())
}

fn stats(times: &[f64]) -> String {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |f: f64| sorted[((sorted.len() - 1) as f64 * f) as usize];
    format!("p50={:.2} ms p95={:.2} ms max={:.2} ms", at(0.5), at(0.95), sorted[sorted.len() - 1])
}

#[test]
#[ignore = "timing probe, minutes"]
fn an_actuated_creature_among_sleepers() {
    let cubes = std::env::var("CUBES").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let steps = 200;
    let (full, full_asleep_steps, full_positions, full_asleep) = run(cubes, false, steps);
    let (sparse, sparse_asleep_steps, sparse_positions, sparse_asleep) = run(cubes, true, steps);
    assert_eq!(full_asleep_steps, sparse_asleep_steps);
    let worst = full_positions
        .iter()
        .zip(&sparse_positions)
        .map(|(a, b)| (*a - *b).abs().max_element())
        .fold(0.0, f64::max);
    // The cubes settle, then sleep and wake in turn (their landing heat still moves);
    // compare the steps in which they sleep.
    let pick = |times: &[f64]| -> Vec<f64> {
        times.iter().zip(&sparse_asleep_steps).filter(|(_, a)| **a).map(|(t, _)| *t).collect()
    };
    let slept = sparse_asleep_steps.iter().filter(|a| **a).count();
    eprintln!("CREATURE {slept} of {steps} steps with sleepers; sleepers at the end {full_asleep}/{sparse_asleep}");
    eprintln!("CREATURE full   (steps with sleepers): {}", stats(&pick(&full)));
    eprintln!("CREATURE sparse (steps with sleepers): {}", stats(&pick(&sparse)));
    eprintln!("CREATURE largest position difference {worst:e}");
    assert_eq!(full_asleep, sparse_asleep);
    assert!(worst <= 1.0e-12, "sparse and full passes differ by {worst:e}");
}
