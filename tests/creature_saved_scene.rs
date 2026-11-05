//! The creature stack (sensors, spiking controller, resource ledger) attached to
//! the saved spider scene and run through the ordinary solver for 40 steps.
//!
//! The network here is untrained. This test checks accounting and finiteness;
//! it does not show that the spider learned to walk.

use std::path::Path;

use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    creature::{genome_for_body, Creature, OrganParams, SensorLayout, SensorReading},
    documents::{load_document, SceneDocument},
    scene::Scene,
    solver,
};

const SCENE_PATH: &str = "scenes/spider_flat_physics_2026-11-04.vxscene";
const STEPS: usize = 40;

fn load_spider() -> Scene {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SCENE_PATH);
    let document: SceneDocument = load_document(&path).expect("load saved spider scene document");
    let scene = document
        .build_scene(&path)
        .expect("restore saved spider scene and skeletal actor");
    assert_eq!(scene.skeletal_actuators.len(), 1);
    assert_eq!(scene.voxels.len(), 9_594);
    scene
}

fn state_is_finite(scene: &Scene) -> bool {
    scene.positions.iter().all(|p| p.is_finite())
        && scene
            .voxels
            .iter()
            .all(|v| v.velocity.is_finite() && v.has_finite_thermal_state())
        && scene.rotations.iter().all(|r| r.is_finite())
}

#[test]
fn spider_runs_40_steps_with_network_and_ledger_attached() {
    let mut scene = load_spider();
    assert_eq!(scene.config.fixed_dt, 0.01);
    let range = scene.skeletal_actuators[0].voxel_range();
    let body_mass_kg: f64 = range
        .clone()
        .map(|i| f64::from(scene.voxels[i].mass))
        .sum();

    let feet = SensorLayout::lowest_revs(&scene, range.clone(), 0.005).unwrap();
    let sensors = SensorLayout::new(&scene.skeletal_actuators[0], feet).unwrap();
    let organs = OrganParams::from_body_mass_kg(body_mass_kg);
    let genome = genome_for_body("spider", &sensors, organs, 2026, 64, 4).unwrap();
    let mut creature = Creature::new(genome, None, &scene, 0).unwrap();
    let mut controller = creature.spiking_controller().unwrap();
    eprintln!(
        "CREATURE body_mass_kg={body_mass_kg:.4} feet={} joints={} inputs={} neurons={} synapses={}",
        sensors.foot_count(),
        sensors.joint_count(),
        sensors.input_count(),
        creature.genome().network.neuron_count(),
        creature.genome().network.synapses.len(),
    );
    assert!(state_is_finite(&scene), "saved spider must start finite");

    let first = SensorReading::read(&scene, &sensors, creature.ledger()).unwrap();
    eprintln!(
        "CREATURE initial feet_touching={}/{} joints_readable={}",
        first.foot_contact.iter().filter(|&&c| c).count(),
        first.foot_contact.len(),
        first.joints.iter().filter(|j| j.rotation_vector_rad.is_some()).count()
    );

    let stored0 = creature.ledger().stored_j();
    let (mut debited, mut positive_ledger, mut displacement_net, mut displacement_pos) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut peak_applied = 0.0f32;
    let mut advance = |s: &mut Scene| solver::step(s, &mut None);
    for tick in 1..=STEPS {
        let before: Vec<DVec3> = range.clone().map(|i| scene.positions[i]).collect();
        let report = creature
            .drive_step(&mut scene, &mut controller, &mut advance)
            .unwrap_or_else(|e| panic!("drive_step failed at tick {tick}: {e}"));
        assert!(state_is_finite(&scene), "non-finite state at tick {tick}");
        assert!(creature.ledger().stored_j() >= 0.0, "ledger negative at tick {tick}");
        assert!((0.0..=1.0).contains(&report.applied));
        peak_applied = peak_applied.max(report.applied);
        let reading = SensorReading::read(&scene, &sensors, creature.ledger()).unwrap();
        assert!(reading.is_finite(), "non-finite sensor reading at tick {tick}");

        // Independent measure of the actuator's work: the loads are constant
        // over a step, so work is exactly F . displacement.
        let loads: &[Vec3] = scene.skeletal_actuators[0].last_forces_n();
        for (k, i) in range.clone().enumerate() {
            let force = loads.get(k).copied().unwrap_or(Vec3::ZERO).as_dvec3();
            let w = force.dot(scene.positions[i] - before[k]);
            displacement_net += w;
            displacement_pos += w.max(0.0);
        }
        debited += report.outcome.debited_j;
        positive_ledger += report.outcome.positive_work_j;
    }

    let stored1 = creature.ledger().stored_j();
    let cost = controller.network().cost();
    eprintln!(
        "CREATURE steps={STEPS} peak_applied={peak_applied:.3} debited_j={debited:.6e} \
         ledger_positive_work_j={positive_ledger:.6e} displacement_net_work_j={displacement_net:.6e} \
         displacement_positive_work_j={displacement_pos:.6e} heat_j={:.6e} stored_j {stored0:.6e}->{stored1:.6e} \
         unfunded_j={:.3e}",
        creature.ledger().heat_total_j(),
        creature.ledger().unfunded_work_total_j(),
    );
    eprintln!(
        "CREATURE network steps={} neuron_updates={} spikes={} synaptic_events={} input_ops={} operations={}",
        cost.steps, cost.neuron_updates, cost.spikes, cost.synaptic_events, cost.input_ops, cost.operations()
    );

    assert!(peak_applied > 0.0, "the untrained network must drive the actuator at least once");
    assert!(positive_ledger > 0.0, "the actuator did mechanical work");
    assert!(
        debited >= displacement_net && debited >= displacement_pos,
        "debited {debited:e} J must cover the actuator's mechanical work {displacement_pos:e} J"
    );
    assert!(debited >= positive_ledger);
    assert!((stored0 - stored1 - debited).abs() <= 1e-9 * stored0);
    assert!(creature.ledger().conservation_residual_j().abs() <= 1e-9 * stored0);
    assert!(cost.steps > 0 && cost.spikes > 0);
}
