//! The creature stack (sensors read from the real scene, spiking controller,
//! resource ledger) attached to the saved spider scene and run through the
//! ordinary solver for 40 fixed steps.
//!
//! The network here is untrained. This test checks sensing, accounting,
//! finiteness and cost reporting; it does not show that the spider learned to
//! walk, and the spider is not shown to walk.

use std::path::Path;

use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    creature::{
        genome_for_body, Creature, OrganParams, PassiveController, SensorLayout, SensorReading,
        StepTiming,
    },
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

fn internal_energy_j(scene: &Scene, range: &std::ops::Range<usize>) -> f64 {
    range
        .clone()
        .map(|i| scene.voxels[i].total_internal_energy_j())
        .sum()
}

fn build_creature(scene: &Scene, body_mass_kg: f64, store_fraction: f64) -> Creature {
    let range = scene.skeletal_actuators[0].voxel_range();
    let feet = SensorLayout::lowest_revs(scene, range, 0.005).unwrap();
    let probe = SensorLayout::new(&scene.skeletal_actuators[0], feet).unwrap();
    let mut organs = OrganParams::from_body_mass_kg(body_mass_kg);
    organs.birth_store_fraction = store_fraction;
    let genome = genome_for_body("spider", &probe, organs, 2026, 64, 4).unwrap();
    Creature::new(genome, None, scene, 0).unwrap()
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

    let mut creature = build_creature(&scene, body_mass_kg, 0.5);
    let mut controller = creature.spiking_controller().unwrap();
    let catalog = creature.sensor_layout().catalog();
    eprintln!(
        "CREATURE body_mass_kg={body_mass_kg:.4} feet={} joints={} muscle_groups={} sensor_channels={} \
         inputs={} neurons={} synapses={}",
        creature.sensor_layout().foot_count(),
        creature.sensor_layout().joint_count(),
        creature.sensor_layout().muscle_count(),
        catalog.len(),
        creature.sensor_layout().input_count(),
        creature.genome().network.neuron_count(),
        creature.genome().network.synapses.len(),
    );
    assert!(state_is_finite(&scene), "saved spider must start finite");

    let first = creature.sense(&scene).unwrap();
    assert_eq!(first.values().len(), catalog.len());
    eprintln!(
        "CREATURE initial feet_touching={}/{} joints_readable={} com_height_m={:?} body_height_m={:?} tilt_rad={:?} pose_error_rms_m={:?}",
        first.foot_contact.iter().filter(|&&c| c).count(),
        first.foot_contact.len(),
        first.joints.iter().filter(|j| j.rotation_vector_rad.is_some()).count(),
        first.com_height_m,
        first.body_height_m,
        first.tilt_rad,
        first.pose_error_rms_m,
    );

    let stored0 = creature.ledger().stored_j();
    let (mut debited, mut positive_ledger, mut net_ledger) = (0.0f64, 0.0f64, 0.0f64);
    let (mut displacement_net, mut displacement_pos) = (0.0f64, 0.0f64);
    let (mut heat_ledger, mut heat_in_revs) = (0.0f64, 0.0f64);
    let mut peak_applied = 0.0f32;
    let mut timing = StepTiming::default();
    let mut max_stress_pa = 0.0f32;
    let mut touching_steps = 0usize;
    let mut supported_ratio: Vec<f32> = Vec::new();
    for tick in 1..=STEPS {
        let before: Vec<DVec3> = range.clone().map(|i| scene.positions[i]).collect();
        let mut after_solver_j = 0.0;
        let mut advance = |s: &mut Scene| {
            solver::step(s, &mut None);
            after_solver_j = internal_energy_j(s, &range);
        };
        let report = creature
            .drive_step(&mut scene, &mut controller, &mut advance)
            .unwrap_or_else(|e| panic!("drive_step failed at tick {tick}: {e}"));
        // Heat the creature put into its own REVs, seen from the scene itself.
        heat_in_revs += internal_energy_j(&scene, &range) - after_solver_j;
        assert!(state_is_finite(&scene), "non-finite state at tick {tick}");
        assert!(creature.ledger().stored_j() >= 0.0, "ledger negative at tick {tick}");
        assert!((0.0..=1.0).contains(&report.applied));
        peak_applied = peak_applied.max(report.applied);
        let reading = creature.sense(&scene).unwrap();
        assert!(reading.is_finite(), "non-finite sensor reading at tick {tick}");
        assert!(reading.values().iter().flatten().all(|v| v.is_finite()));
        max_stress_pa = max_stress_pa.max(
            reading
                .muscles
                .iter()
                .map(|m| m.peak_tissue_stress_pa)
                .fold(0.0, f32::max),
        );
        if reading.foot_contact.iter().any(|&c| c) {
            touching_steps += 1;
        }
        let load: f32 = reading.foot_normal_load_n.iter().sum();
        assert!((load - reading.support_load_n).abs() <= 1e-4 * reading.support_load_n.max(1e-6));
        if tick > 15 && reading.foot_contact.iter().any(|&c| c) {
            // Standing on the ground: the environment carries about the weight.
            let ratio = reading.support_load_n / reading.body_weight_n;
            assert!((0.6..1.4).contains(&ratio), "tick {tick}: support / weight = {ratio}");
            supported_ratio.push(ratio);
        }
        if tick % 10 == 0 {
            eprintln!(
                "CREATURE tick={tick} applied={:.3} feet_touching={}/{} support_n={load:.4} weight_n={:.4} \
                 com_height_m={:.5} body_height_m={:.5} tilt_rad={:.4} pose_error_rms_m={:.2e} \
                 peak_tissue_stress_pa={max_stress_pa:.3e}",
                report.applied,
                reading.foot_contact.iter().filter(|&&c| c).count(),
                reading.foot_contact.len(),
                reading.body_weight_n,
                reading.com_height_m.unwrap_or(f32::NAN),
                reading.body_height_m.unwrap_or(f32::NAN),
                reading.tilt_rad.unwrap_or(f32::NAN),
                reading.pose_error_rms_m.unwrap_or(f32::NAN),
            );
        }

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
        net_ledger += report.outcome.net_mechanical_work_j;
        heat_ledger += report.outcome.heat_per_rev_j.iter().sum::<f64>();
        timing.sensing_s += report.timing.sensing_s;
        timing.control_s += report.timing.control_s;
        timing.solver_s += report.timing.solver_s;
        timing.accounting_s += report.timing.accounting_s;
        assert_eq!(report.outcome.unfunded_work_j, 0.0, "the store paid for all work");
    }

    let stored1 = creature.ledger().stored_j();
    let cost = controller.network().cost();
    let sensor_cost = creature.sensor_cost_total();
    eprintln!(
        "CREATURE ledger steps={STEPS} peak_applied={peak_applied:.3} stored_drop_j={:.6e} \
         net_mechanical_work_j={net_ledger:.6e} heat_j={heat_ledger:.6e} heat_in_revs_j={heat_in_revs:.6e} \
         positive_work_j={positive_ledger:.6e} displacement_net_work_j={displacement_net:.6e} \
         displacement_positive_work_j={displacement_pos:.6e} stored_j {stored0:.6e}->{stored1:.6e}",
        stored0 - stored1,
    );
    eprintln!(
        "CREATURE network_cost steps={} neuron_updates={} spikes={} synaptic_events={} input_ops={} operations={}",
        cost.steps, cost.neuron_updates, cost.spikes, cost.synaptic_events, cost.input_ops, cost.operations()
    );
    eprintln!(
        "CREATURE sensor_cost reads={} values={} rev_reads={} bond_reads={} pair_tests={} operations={}",
        sensor_cost.reads, sensor_cost.values, sensor_cost.rev_reads, sensor_cost.bond_reads,
        sensor_cost.pair_tests, sensor_cost.operations()
    );
    let own_s = timing.sensing_s + timing.control_s + timing.accounting_s;
    eprintln!(
        "CREATURE wall_time_debug_profile solver_s={:.3} sensing_s={:.4} network_s={:.4} accounting_s={:.4} \
         creature_share_of_solver={:.3}% (debug build, one machine; not a performance measurement)",
        timing.solver_s, timing.sensing_s, timing.control_s, timing.accounting_s,
        100.0 * own_s / timing.solver_s
    );

    // The books: the store paid for the muscles, nothing else drew on it.
    let stored_drop = stored0 - stored1;
    assert!(peak_applied > 0.0, "the untrained network must drive the actuator at least once");
    assert!(positive_ledger > 0.0, "the actuator did mechanical work");
    assert!(
        debited >= displacement_net && debited >= displacement_pos,
        "debited {debited:e} J must cover the actuator's mechanical work {displacement_pos:e} J"
    );
    assert!(debited >= positive_ledger);
    assert!((stored_drop - debited).abs() <= 1e-9 * stored0);
    // Stored energy drop = mechanical work + heat, and the heat is really in the REVs.
    assert!((stored_drop - (net_ledger + heat_ledger)).abs() <= 1e-9 * stored_drop.abs().max(1e-12) + 1e-12);
    assert!((heat_in_revs - heat_ledger).abs() <= 1e-6 * heat_ledger + 1e-12);
    assert!((creature.ledger().heat_total_j() - heat_ledger).abs() <= 1e-9 * heat_ledger + 1e-12);
    assert!(creature.ledger().conservation_residual_j().abs() <= 1e-9 * stored0);
    // The ledger's work agrees with the displacement-based work within the
    // trapezoid-versus-end-of-step quadrature difference (loose bound).
    assert!(positive_ledger > 0.0 && displacement_pos > 0.0);
    assert!((positive_ledger / displacement_pos - 1.0).abs() < 0.25);

    // Cost is reported separately and is the one the specification asks for.
    // 1 ms network steps in each 10 ms physics step. The network's time
    // accumulator carries rounding of 0.01 s as f32, so one step may slip.
    assert!(
        (10 * STEPS as u64 - 1..=10 * STEPS as u64).contains(&cost.steps),
        "network steps {}",
        cost.steps
    );
    assert!(cost.spikes > 0 && cost.operations() > 0);
    assert_eq!(sensor_cost.reads, 2 * STEPS as u64 + 1, "one read per control step, one after it, one at the start");
    assert!(sensor_cost.bond_reads > 0 && sensor_cost.rev_reads > 0);
    assert!(touching_steps > 0 && !supported_ratio.is_empty(), "the spider reaches the ground");
    eprintln!(
        "CREATURE support/weight over {} standing steps: min {:.3} max {:.3}",
        supported_ratio.len(),
        supported_ratio.iter().copied().fold(f32::INFINITY, f32::min),
        supported_ratio.iter().copied().fold(0.0, f32::max)
    );
    assert!(max_stress_pa > 0.0, "tissue stress is read from the solver's bond forces");
}

#[test]
fn an_empty_store_leaves_the_spider_passive_and_a_relaxed_spider_costs_only_basal_power() {
    let mut scene = load_spider();
    let range = scene.skeletal_actuators[0].voxel_range();
    let body_mass_kg: f64 = range.clone().map(|i| f64::from(scene.voxels[i].mass)).sum();

    // Store below the vital reserve: the strength limit is zero.
    let mut empty = build_creature(&scene, body_mass_kg, 0.005);
    let mut controller = empty.spiking_controller().unwrap();
    let mut advance = |s: &mut Scene| solver::step(s, &mut None);
    for tick in 1..=4 {
        let r = empty.drive_step(&mut scene, &mut controller, &mut advance).unwrap();
        assert_eq!(r.applied, 0.0, "tick {tick}: an empty store gives no strength");
        assert_eq!(scene.skeletal_actuators[0].last_force_norm_n(), 0.0);
        assert_eq!(r.outcome.positive_work_j, 0.0);
        assert!(state_is_finite(&scene));
    }
    assert!(empty.ledger().stored_j() >= 0.0);

    // Relaxed muscles on a fresh spider: only the basal cost is paid.
    let mut scene = load_spider();
    let mut relaxed = build_creature(&scene, body_mass_kg, 0.5);
    let stored0 = relaxed.ledger().stored_j();
    for _ in 0..3 {
        let r = relaxed.drive_step(&mut scene, &mut PassiveController, &mut advance).unwrap();
        assert_eq!(r.applied, 0.0);
        assert_eq!(r.outcome.positive_work_j, 0.0);
    }
    let basal = relaxed.genome().organs.basal_power_w * 0.03;
    assert!((stored0 - relaxed.ledger().stored_j() - basal).abs() <= 1e-9 * stored0);
    let reading: SensorReading = relaxed.sense(&scene).unwrap();
    assert!(reading.is_finite());
}
