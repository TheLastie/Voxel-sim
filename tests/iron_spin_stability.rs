//! A tiny seeded spin on a free iron cube must not grow, at any time step.
//!
//! Regression for a defect found on 2026-11-05: at 40 substeps of 0.25 ms the spin grew
//! from 1e-6 to 0.4 rad/s in 0.4 s while 4 substeps of 2.5 ms stayed flat. The causes
//! were fixed along the way (a hierarchy kept with stale masks, double-precision cell
//! rotations, one inertial target per substep). Energy is split into its parts so a
//! growing one would be visible; the bound of 1e-4 rad/s is 100 times the seed.
use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

struct Energy {
    translation: f64,
    spin: f64,
    elastic: f64,
    heat: f64,
}

fn energy(scene: &Scene) -> Energy {
    let cell = 0.01f64;
    let translation = scene
        .voxels
        .iter()
        .map(|v| 0.5 * f64::from(v.mass) * f64::from(v.velocity.length_squared()))
        .sum();
    let spin = scene
        .voxels
        .iter()
        .zip(&scene.angular_velocities)
        .map(|(v, w)| 0.5 * (f64::from(v.mass) * cell * cell / 6.0) * w.length_squared())
        .sum();
    Energy {
        translation,
        spin,
        elastic: scene.elastic_bond_energy_j(),
        heat: scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum(),
    }
}

fn run(material: &str, substeps: usize, steps: usize) -> (f64, f64) {
    let id = Material::component_id(material).unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = substeps;
    config.gravity = [0.0; 3];
    config.atmosphere.pressure_pa = 0.0;
    let mut spawns = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], id));
            }
        }
    }
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    for (i, w) in scene.angular_velocities.iter_mut().enumerate() {
        let f = i as f64;
        *w = DVec3::new((f * 1.3).sin(), (f * 2.1).sin(), (f * 3.7).sin()) * 1.0e-6;
    }
    let start = energy(&scene);
    let mechanical_start = start.translation + start.spin + start.elastic;
    let mut peak_spin = 0.0f64;
    for step in 1..=steps {
        solver::step(&mut scene, &mut None);
        let peak = scene.angular_velocities.iter().map(|w| w.length()).fold(0.0, f64::max);
        peak_spin = peak_spin.max(peak);
        if step % 10 == 0 {
            let e = energy(&scene);
            eprintln!(
                "SPIN {material} sub={substeps} step={step}: max_spin={peak:.2e} rad/s  E_trans={:.2e} E_spin={:.2e} E_elastic={:.2e} dHeat={:+.2e} J",
                e.translation,
                e.spin,
                e.elastic,
                e.heat - start.heat
            );
        }
    }
    let _ = Vec3::ZERO;
    let end = energy(&scene);
    (peak_spin, (end.translation + end.spin + end.elastic) - mechanical_start)
}

#[test]
fn iron_spin_is_flat_at_the_default_time_step() {
    let (peak, _) = run("iron", 4, 40);
    assert!(peak < 1.0e-4, "spin grew to {peak}");
}

#[test]
fn iron_spin_is_flat_at_a_small_time_step() {
    let (peak, gain) = run("iron", 40, 40);
    assert!(peak < 1.0e-4, "spin grew to {peak} rad/s; mechanical energy changed by {gain:+.3e} J");
}
