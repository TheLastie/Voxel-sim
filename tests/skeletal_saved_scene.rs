use std::path::Path;
use std::time::Instant;

use voxel_physics_lab::{
    documents::{load_document, SceneDocument},
    scene::Scene,
    solver,
};

const SCENE_PATH: &str = "scenes/spider_flat_physics_2026-11-04.vxscene";
fn requested_steps() -> usize {
    std::env::var("VOXEL_SAVED_SCENE_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|steps| *steps > 0)
        .unwrap_or(8)
}

fn load_spider() -> Scene {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SCENE_PATH);
    let document: SceneDocument = load_document(&path).expect("load saved spider scene document");
    let scene = document
        .build_scene(&path)
        .expect("restore saved spider scene and skeletal actor");
    assert_eq!(
        scene.skeletal_actuators.len(),
        1,
        "saved spider actor count"
    );
    assert_eq!(scene.voxels.len(), 9_594, "saved scene voxel count");
    let reverse_owner_bonds = scene
        .bonds
        .iter()
        .map(|bonds| {
            bonds
                .iter()
                .enumerate()
                .filter(|(i, bond)| bond.neighbor != u32::MAX && (bond.neighbor as usize) < *i)
                .count()
        })
        .sum::<usize>();
    eprintln!("SPIDER_IC0 reverse_owner_bonds={reverse_owner_bonds}");
    assert!(
        reverse_owner_bonds > 0,
        "fixture must retain bonds whose owner index follows its neighbor"
    );
    scene
}

fn state_is_finite(scene: &Scene) -> bool {
    scene.positions.iter().all(|position| position.is_finite())
        && scene
            .voxels
            .iter()
            .all(|voxel| voxel.velocity.is_finite() && voxel.has_finite_thermal_state())
        && scene.rotations.iter().all(|rotation| rotation.is_finite())
}

fn state_counts(scene: &Scene) -> (usize, usize, usize, usize) {
    (
        scene
            .positions
            .iter()
            .filter(|value| !value.is_finite())
            .count(),
        scene
            .voxels
            .iter()
            .filter(|value| !value.velocity.is_finite())
            .count(),
        scene
            .voxels
            .iter()
            .filter(|value| !value.has_finite_thermal_state())
            .count(),
        scene
            .rotations
            .iter()
            .filter(|value| !value.is_finite())
            .count(),
    )
}

fn report(label: &str, step: usize, scene: &Scene) {
    let actor = &scene.skeletal_actuators[0];
    let (bad_positions, bad_velocities, bad_thermal, bad_rotations) = state_counts(scene);
    let diagnostics = scene.last_step_diagnostics.as_ref();
    eprintln!(
        "SPIDER_STATE label={label} step={step} sim_s={:.6} actor_enabled={} actor_pose_s={:.6} actor_force_n={:.6e} bad_positions={bad_positions} bad_velocities={bad_velocities} bad_thermal={bad_thermal} bad_rotations={bad_rotations} contacts_peak={} contact_pcg_calls={} pcg_iters={} residual={:.6e}",
        scene.simulated_seconds,
        actor.enabled,
        actor.elapsed_seconds(),
        actor.last_force_norm_n(),
        diagnostics.map_or(0, |value| value.contacts_peak),
        diagnostics.map_or(0, |value| value.contact_pcg_calls),
        diagnostics.map_or(0, |value| value.pcg_iterations_total),
        scene.solver_residual,
    );
}

// A same-binary A/B diagnostic, not a portable checkpoint or an accuracy oracle.
// Voxel serialization includes all 27 thermal cells when they are allocated.
fn diagnostic_state_hash(scene: &Scene) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    let mut append = |bytes: &[u8]| {
        for byte in bytes {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    };
    append(
        &bincode::serialize(&(
            &scene.positions,
            &scene.previous,
            &scene.rotations,
            &scene.voxels,
            scene.simulated_seconds,
            scene.broken,
        ))
        .unwrap(),
    );
    for axis in &scene.bonds {
        for bond in axis {
            append(
                &bincode::serialize(&(
                    bond.neighbor,
                    bond.formation_offset,
                    bond.plastic_strain,
                    bond.damage,
                    bond.equivalent_plastic_strain,
                    bond.plastic_work,
                    bond.fatigue_damage,
                    bond.previous_stress_ratio,
                    bond.reversal_stress_ratio,
                    bond.loading_increment,
                ))
                .unwrap(),
            );
        }
    }
    append(format!("{:?}", scene.energy).as_bytes());
    hash
}

#[test]
fn saved_spider_remains_finite_with_actor_on_and_off() {
    let steps = requested_steps();
    for (label, actor_enabled) in [("actor_on", true), ("actor_off", false)] {
        let mut scene = load_spider();
        assert!(state_is_finite(&scene), "saved spider must start finite");
        assert_eq!(scene.config.fixed_dt, 0.01);
        assert_eq!(scene.config.substeps, 4);
        scene.skeletal_actuators[0].enabled = actor_enabled;
        let mut step_ms = Vec::with_capacity(steps);
        for tick in 1..=steps {
            let started = Instant::now();
            solver::step(&mut scene, &mut None);
            step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            assert!(
                state_is_finite(&scene),
                "saved spider became non-finite with {label} at tick {tick}; counts={:?}",
                state_counts(&scene)
            );
            if tick == 1 || tick == steps {
                report(label, tick, &scene);
            }
        }
        let internal_j = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum::<f64>();
        let kinetic_j = scene
            .voxels
            .iter()
            .map(|voxel| 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared()))
            .sum::<f64>();
        eprintln!("SPIDER_FINITE label={label} ticks={steps} sim_s={:.6} kinetic_j={kinetic_j:.9e} internal_j={internal_j:.9e} elastic_j={:.9e} external_work_j={:.9e} residual={:.6e}",
            scene.simulated_seconds, scene.elastic_bond_energy_j(), scene.energy.external_work, scene.solver_residual);
        let warmup = steps.saturating_sub(1).min(4);
        let mut measured = step_ms[warmup..].to_vec();
        measured.sort_by(f64::total_cmp);
        eprintln!("SPIDER_COMPARISON label={label} ticks={steps} state_hash={:016x} solver_only_median_ms={:.6} solver_only_max_ms={:.6} scope=headless_solver_excludes_worker_metrics_and_render warmup={warmup}",
            diagnostic_state_hash(&scene), measured[measured.len()/2], measured[measured.len()-1]);
    }
}
