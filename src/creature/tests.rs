use super::*;
use crate::{
    documents::{
        BoneKeyframeDocument, EditorDocument, SkeletonDocument, VoxelBoneWeightDocument,
        VoxelModelDocument,
    },
    scene::{SceneBuilder, VoxelSpawn},
    solver,
};

// ----- fixtures ------------------------------------------------------------

const GROUND: usize = 8;

/// Eight creature REVs (the engine's gravity points along -z): bone 0 owns a 2x2x1 slab, bone 1 the next slab.
fn creature_cells() -> Vec<[i32; 3]> {
    vec![
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 0, 0],
        [3, 0, 0],
        [2, 1, 0],
        [3, 1, 0],
    ]
}

fn creature_model() -> VoxelModelDocument {
    creature_model_of(&creature_cells())
}

/// Same skeleton for any 8 cells: the first four REVs belong to the root bone,
/// the last four to the limb bone.
fn creature_model_of(cells: &[[i32; 3]]) -> VoxelModelDocument {
    let mut builder = SceneBuilder::standard();
    for &cell in cells {
        builder = builder.add_voxel(VoxelSpawn::at_cell(cell, 1));
    }
    let scene = builder.build().unwrap();
    let mut model = VoxelModelDocument::from_scene("creature-fixture", &scene);
    let mut skeleton = SkeletonDocument::with_root_bone("root", DVec3::new(0.01, 0.0, 0.0));
    skeleton
        .add_bone("limb", Some(0), DVec3::new(0.02, 0.0, 0.0))
        .unwrap();
    skeleton.voxel_weights = (0..8)
        .map(|i| VoxelBoneWeightDocument {
            voxel_index: i,
            bone_index: u32::from(i >= 4),
            weight: 1.0,
        })
        .collect();
    let clip = skeleton.add_animation_clip("bend", 1.0).unwrap();
    for (time_s, angle) in [(0.0, 0.0), (1.0, 0.3)] {
        skeleton
            .set_keyframe(
                clip,
                1,
                BoneKeyframeDocument {
                    time_s,
                    translation_m: DVec3::new(0.02, 0.0, 0.0),
                    rotation: Quat::from_rotation_z(angle),
                },
            )
            .unwrap();
    }
    model.skeleton = Some(skeleton);
    model.validate().unwrap();
    model
}

/// Fixed ground under the creature (REVs 0..8), creature REVs 8..16.
fn world() -> Scene {
    world_raised(0)
}

/// As `world`, with the creature `cells` REV heights above the ground.
fn world_raised(cells: i32) -> Scene {
    let mut builder = SceneBuilder::standard();
    for x in 0..4 {
        for y in 0..2 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, -1], 1).fixed());
        }
    }
    for [x, y, z] in creature_cells() {
        builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z + cells], 1));
    }
    let mut scene = builder.build().unwrap();
    assert_eq!(scene.voxels.len(), 16);
    let actuator =
        SkeletalActuator::new(creature_model(), GROUND..GROUND + 8, Quat::IDENTITY, 16).unwrap();
    scene.skeletal_actuators.push(actuator);
    scene
}

fn organs() -> OrganParams {
    OrganParams {
        store_capacity_j: 100.0,
        birth_store_fraction: 0.5,
        stomach_capacity_j: 20.0,
        digestion_power_w: 2.0,
        absorption_efficiency: 0.8,
        muscle_efficiency: 0.25,
        basal_power_w: 0.0,
        peak_mechanical_power_w: 1.0,
        weak_fraction: 0.1,
        exhaustion_fraction: 0.0,
    }
}

fn zero_basal(mut o: OrganParams) -> OrganParams {
    o.basal_power_w = 0.0;
    o
}

fn bound_layout(scene: &Scene, feet: Vec<usize>) -> SensorLayout {
    let mut layout = SensorLayout::new(&scene.skeletal_actuators[0], feet).unwrap();
    layout.bind(scene).unwrap();
    layout
}

// ----- 1. resource ledger --------------------------------------------------

#[test]
fn muscle_work_costs_work_over_efficiency_and_losses_become_heat() {
    let mut ledger = ResourceLedger::new(organs()).unwrap();
    let before = ledger.stored_j();
    // 2 N at 0.5 m/s for 10 ms is 0.01 J of work; efficiency 0.25 -> 0.04 J.
    let f = [Vec3::new(2.0, 0.0, 0.0)];
    let v = [Vec3::new(0.5, 0.0, 0.0)];
    let out = ledger.settle_muscle_step(0.01, &f, &v, &v).unwrap();
    assert!((out.positive_work_j - 0.01).abs() < 1e-12);
    assert!((out.debited_j - 0.04).abs() < 1e-12);
    assert!((before - ledger.stored_j() - 0.04).abs() < 1e-12);
    assert!((out.heat_per_rev_j[0] - 0.03).abs() < 1e-12);
    // debited = net work + heat: nothing appears or disappears.
    let books = out.net_mechanical_work_j + out.heat_per_rev_j.iter().sum::<f64>();
    assert!((books - out.debited_j).abs() < 1e-12);
    assert_eq!(out.unfunded_work_j, 0.0);
    assert!(ledger.conservation_residual_j().abs() < 1e-12);
}

#[test]
fn braking_work_costs_nothing_and_returns_to_the_muscle_as_heat() {
    let mut ledger = ResourceLedger::new(organs()).unwrap();
    let before = ledger.stored_j();
    let f = [Vec3::new(-2.0, 0.0, 0.0)];
    let v = [Vec3::new(0.5, 0.0, 0.0)];
    let out = ledger.settle_muscle_step(0.01, &f, &v, &v).unwrap();
    assert!((out.negative_work_j - 0.01).abs() < 1e-12);
    assert_eq!(out.debited_j, 0.0);
    assert_eq!(ledger.stored_j(), before);
    assert!((out.heat_per_rev_j[0] - 0.01).abs() < 1e-12);
}

#[test]
fn basal_cost_is_debited_without_motion_and_becomes_heat() {
    let mut p = organs();
    p.basal_power_w = 5.0;
    let mut ledger = ResourceLedger::new(p).unwrap();
    let z = [Vec3::ZERO; 2];
    let out = ledger.settle_muscle_step(0.01, &z, &z, &z).unwrap();
    assert!((out.debited_j - 0.05).abs() < 1e-12);
    assert!((out.heat_per_rev_j.iter().sum::<f64>() - 0.05).abs() < 1e-12);
    // With no muscle REV at all the heat is reported as unlocalised.
    let out = ledger.settle_muscle_step(0.01, &[], &[], &[]).unwrap();
    assert!((out.unlocalised_heat_j - 0.05).abs() < 1e-12);
}

#[test]
fn exhausted_creature_cannot_move_and_the_store_never_goes_negative() {
    let mut p = organs();
    p.birth_store_fraction = 0.0004; // 0.04 J
    let mut ledger = ResourceLedger::new(p).unwrap();
    assert!(ledger.strength_limit(0.01) < 1.0);
    // A load far beyond what the store can pay.
    let f = [Vec3::new(100.0, 0.0, 0.0)];
    let v = [Vec3::new(1.0, 0.0, 0.0)];
    let out = ledger.settle_muscle_step(0.01, &f, &v, &v).unwrap();
    assert!(ledger.stored_j() >= 0.0);
    assert!(out.unfunded_work_j > 0.0, "shortfall must be visible");
    assert_eq!(ledger.stored_j(), 0.0);
    assert_eq!(out.strength_limit, 0.0);
    assert_eq!(ledger.strength_limit(0.01), 0.0);
    assert!(ledger.conservation_residual_j().abs() < 1e-12);
    // Heat never exceeds what was debited plus absorbed braking work.
    let heat = out.heat_per_rev_j[0];
    assert!(heat >= 0.0 && heat <= out.debited_j + out.negative_work_j + 1e-12);
}

#[test]
fn strength_limit_weakens_as_the_store_drops() {
    let mut p = organs();
    p.peak_mechanical_power_w = 0.001;
    let mut ledger = ResourceLedger::new(p).unwrap();
    assert_eq!(ledger.strength_limit(0.01), 1.0);
    let mut last = 1.0;
    for fraction in [0.08, 0.05, 0.02, 0.0] {
        ledger.stored_j = 100.0 * fraction;
        let l = ledger.strength_limit(0.01);
        assert!(l <= last && (0.0..=1.0).contains(&l));
        last = l;
    }
    assert_eq!(last, 0.0);
}

#[test]
fn hunger_and_satiety_come_from_stored_energy_and_digestion() {
    let mut ledger = ResourceLedger::new(zero_basal(organs())).unwrap();
    let hungry0 = ledger.hunger();
    assert!((hungry0 - 0.5).abs() < 1e-12);
    assert_eq!(ledger.satiety(), 0.0);
    // Food in the stomach reduces hunger before it reaches the store.
    assert_eq!(ledger.ingest_food(10.0).unwrap(), 10.0);
    assert!(ledger.hunger() < hungry0);
    assert!((ledger.satiety() - 0.5).abs() < 1e-12);
    // Stomach capacity limits intake; the rest stays outside.
    assert_eq!(ledger.ingest_food(50.0).unwrap(), 10.0);
    assert!(ledger.ingest_food(f64::NAN).is_err());
    let stored = ledger.stored_j();
    for _ in 0..100 {
        ledger.advance_digestion(0.1);
    }
    assert!(ledger.stomach_j() < 1e-9, "digestion empties the stomach");
    assert!(ledger.satiety() < 1e-9);
    // 20 J eaten at 80 % absorption.
    assert!((ledger.stored_j() - stored - 16.0).abs() < 1e-9);
    assert!(ledger.conservation_residual_j().abs() < 1e-9);
    // Spending energy makes the creature hungrier with no flag involved.
    let h = ledger.hunger();
    let f = [Vec3::X * 10.0];
    let v = [Vec3::X];
    ledger.settle_muscle_step(0.5, &f, &v, &v).unwrap();
    assert!(ledger.hunger() > h);
}

#[test]
fn invalid_ledger_input_is_rejected() {
    let mut bad = organs();
    bad.muscle_efficiency = 1.5;
    assert!(ResourceLedger::new(bad).is_err());
    let mut ledger = ResourceLedger::new(organs()).unwrap();
    let nan = [Vec3::splat(f32::NAN)];
    let ok = [Vec3::ZERO];
    assert!(ledger.settle_muscle_step(0.01, &nan, &ok, &ok).is_err());
    assert!(ledger.settle_muscle_step(0.01, &ok, &ok, &[]).is_err());
    assert!(ledger.settle_muscle_step(0.0, &ok, &ok, &ok).is_err());
}

// ----- 2. sensors ----------------------------------------------------------

#[test]
fn horn_fit_recovers_a_known_rotation() {
    let rest = [
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(0.01, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 0.01),
        DVec3::new(0.01, 0.0, 0.01),
    ];
    let q = DQuat::from_axis_angle(DVec3::new(1.0, 2.0, -0.5).normalize(), 0.8);
    let cur: Vec<_> = rest.iter().map(|p| DVec3::new(1.0, 2.0, 3.0) + q * *p).collect();
    let fitted = fit_rotation(&rest, &cur, &[1.0, 2.0, 1.0, 3.0]).unwrap();
    assert!(fitted.angle_between(q) < 1e-9);
    // Collinear points have no defined rotation.
    let line = [DVec3::ZERO, DVec3::X * 0.01, DVec3::X * 0.02];
    assert!(fit_rotation(&line, &line, &[1.0; 3]).is_none());
}

#[test]
fn sensors_read_feet_orientation_joints_and_internal_state_from_the_scene() {
    let mut scene = world();
    let ledger = ResourceLedger::new(organs()).unwrap();
    let layout = bound_layout(&scene, vec![0, 4]);
    assert_eq!(layout.joint_count(), 1);
    assert_eq!(layout.input_count(), 2 * 2 + 3 + 3 + 1 + 4);

    let r = SensorReading::read(&scene, &layout, &ledger).unwrap();
    assert_eq!(r.foot_contact, vec![true, true], "feet rest on the ground");
    assert!(r.body_orientation.unwrap().angle_between(Quat::IDENTITY) < 1e-5);
    let up = r.up_in_body.unwrap();
    assert!((up - Vec3::Z).length() < 1e-5, "up={up:?}");
    assert!(r.joints[0].angle_rad().unwrap() < 1e-5);
    assert!((r.stored_energy_fraction - 0.5).abs() < 1e-6);
    assert!((r.hunger - 0.5).abs() < 1e-6);
    assert!(r.is_finite());

    // Lift one foot by 5 mm: it no longer touches anything. Its creature
    // neighbours do not count as contacts.
    scene.positions[GROUND + 4].z += 0.005;
    let r = SensorReading::read(&scene, &layout, &ledger).unwrap();
    assert_eq!(r.foot_contact, vec![true, false]);
    scene.positions[GROUND + 4].z -= 0.005;

    // Rotate the limb slab 0.3 rad about the vertical (z) axis through the hinge.
    let pivot = (scene.positions[GROUND + 1] + scene.positions[GROUND + 4]) * 0.5;
    let rot = DQuat::from_rotation_z(0.3);
    for i in GROUND + 4..GROUND + 8 {
        scene.positions[i] = pivot + rot * (scene.positions[i] - pivot);
    }
    let r = SensorReading::read(&scene, &layout, &ledger).unwrap();
    let joint = r.joints[0];
    assert_eq!((joint.bone_index, joint.reference_bone_index), (1, 0));
    assert!((joint.angle_rad().unwrap() - 0.3).abs() < 1e-4, "{joint:?}");
    let axis = joint.rotation_vector_rad.unwrap().normalize();
    assert!(axis.dot(Vec3::Z).abs() > 0.999);
    assert!(r.body_orientation.unwrap().angle_between(Quat::IDENTITY) < 1e-5);

    // Spin the whole body: angular velocity is measured from real velocities.
    for i in GROUND..GROUND + 8 {
        let rr = scene.positions[i] - scene.positions[GROUND];
        scene.voxels[i].velocity = DVec3::new(0.0, 0.0, 2.0).cross(rr).as_vec3();
    }
    let r = SensorReading::read(&scene, &layout, &ledger).unwrap();
    assert!((r.angular_velocity_rad_s.z - 2.0).abs() < 1e-3, "{:?}", r.angular_velocity_rad_s);

    let mut enc = Vec::new();
    r.encode(&mut enc);
    assert_eq!(enc.len(), layout.input_count());
    assert!(enc.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
}

#[test]
fn sensor_layout_rejects_stale_topology() {
    let mut scene = world();
    let ledger = ResourceLedger::new(organs()).unwrap();
    let layout = bound_layout(&scene, vec![0]);
    scene.positions.pop();
    assert!(SensorReading::read(&scene, &layout, &ledger).is_err());
    let scene = world();
    assert!(SensorLayout::new(&scene.skeletal_actuators[0], vec![8]).is_err());
}

#[test]
fn lowest_revs_choose_feet_from_the_initial_pose() {
    let scene = world();
    let feet = SensorLayout::lowest_revs(&scene, GROUND..GROUND + 8, 1e-4).unwrap();
    assert_eq!(feet, (0..8).collect::<Vec<_>>(), "the slab is one layer thick");
}

// ----- 3. spiking network --------------------------------------------------

fn tiny_layout() -> NetworkLayout {
    NetworkLayout {
        dt_s: 0.001,
        input_count: 1,
        neurons: vec![
            NeuronParams {
                leak_per_s: 0.0,
                threshold: 1.0,
                reset: 0.0,
                refractory_s: 0.005,
            };
            2
        ],
        synapses: vec![SynapseSpec {
            pre: 0,
            post: 1,
            delay_steps: 5,
        }],
        input_links: vec![InputLink { input: 0, post: 0 }],
        output_neurons: vec![1],
        rate_window_s: 0.05,
        max_rate_hz: 100.0,
        synaptic_weight_limit: 2.0,
        input_weight_limit: 100.0,
        bias_limit: 10.0,
        init_weight_fraction: 0.5,
        init_bias_fraction: 0.0,
        init_seed: 1,
    }
}

fn tiny_learned(input_w: f32, syn_w: f32) -> LearnedState {
    LearnedState {
        weights: NetworkWeights {
            synaptic: vec![syn_w],
            input: vec![input_w],
        },
        neuron_bias: vec![0.0; 2],
        lifetime_s: 0.0,
        trials: 0,
        accepted_updates: 0,
    }
}

#[test]
fn a_spike_arrives_after_exactly_its_delay() {
    let mut net = SpikingNetwork::new(tiny_layout(), &tiny_learned(100.0, 1.5)).unwrap();
    let (mut first_pre, mut first_post) = (None, None);
    for step in 0..60u64 {
        let out = net.step(0.001, &[1.0]).unwrap()[0];
        if first_pre.is_none() && net.cost().spikes > 0 {
            first_pre = Some(step);
        }
        if first_post.is_none() && out > 0.0 {
            first_post = Some(step);
        }
    }
    let (pre, post) = (first_pre.unwrap(), first_post.unwrap());
    assert_eq!(post - pre, 5, "pre={pre} post={post}");
}

#[test]
fn refractory_period_limits_the_firing_rate() {
    let mut net = SpikingNetwork::new(tiny_layout(), &tiny_learned(100.0, 0.0)).unwrap();
    for _ in 0..200 {
        net.step(0.001, &[1.0]).unwrap();
    }
    // Neuron 0 fires every >= 6 ms (5 ms refraction + 1 ms step): <= 34 spikes.
    let spikes = net.cost().spikes;
    assert!(spikes > 5 && spikes <= 34, "spikes={spikes}");
}

#[test]
fn outputs_are_bounded_and_cost_is_counted() {
    let mut layout = tiny_layout();
    layout.max_rate_hz = 1.0; // any spiking saturates the decoder
    let mut net = SpikingNetwork::new(layout, &tiny_learned(100.0, 1.5)).unwrap();
    for _ in 0..300 {
        let out = net.step(0.01, &[1.0]).unwrap()[0];
        assert!((0.0..=1.0).contains(&out));
    }
    let c = net.cost();
    assert_eq!(c.steps, 3000, "10 ms physical step = ten 1 ms network steps");
    assert_eq!(c.neuron_updates, 2 * c.steps);
    assert_eq!(c.input_ops, 300);
    assert!(c.spikes > 0 && c.synaptic_events > 0);
    assert_eq!(c.operations(), c.neuron_updates + c.synaptic_events + c.input_ops);
    net.reset_cost();
    assert_eq!(net.cost(), NetworkCost::default());
}

#[test]
fn fractional_time_steps_accumulate_and_bad_inputs_are_rejected() {
    let mut net = SpikingNetwork::new(tiny_layout(), &tiny_learned(100.0, 0.0)).unwrap();
    net.step(0.0004, &[0.0]).unwrap();
    assert_eq!(net.cost().steps, 0);
    net.step(0.0004, &[0.0]).unwrap();
    net.step(0.0004, &[0.0]).unwrap();
    assert_eq!(net.cost().steps, 1);
    assert!(net.step(0.001, &[]).is_err());
    assert!(net.step(0.001, &[f32::NAN]).is_err());
    assert!(net.step(-1.0, &[0.0]).is_err());
}

#[test]
fn network_is_deterministic_for_a_seed_and_serializes() {
    let build = || {
        let layout = NetworkLayout::random_recurrent(42, 4, 12, 3, 4, 4, 5).unwrap();
        let learned = LearnedState::from_scratch(&layout);
        SpikingNetwork::new(layout, &learned).unwrap()
    };
    let (mut a, mut b) = (build(), build());
    let input = |k: usize| [0.1 * (k % 10) as f32, 1.0, -0.5, 0.25];
    for k in 0..200 {
        assert_eq!(a.step(0.005, &input(k)).unwrap(), b.step(0.005, &input(k)).unwrap());
    }
    assert_eq!(a.cost(), b.cost());
    // A different seed differs.
    let layout = NetworkLayout::random_recurrent(43, 4, 12, 3, 4, 4, 5).unwrap();
    assert_ne!(layout, a.layout().clone());

    // Serialize mid-run and continue: the copy follows the original exactly.
    let json = serde_json::to_string(&a).unwrap();
    let mut c: SpikingNetwork = serde_json::from_str(&json).unwrap();
    for k in 200..300 {
        assert_eq!(a.step(0.005, &input(k)).unwrap(), c.step(0.005, &input(k)).unwrap());
    }
    assert_eq!(a.cost(), c.cost());
}

#[test]
fn network_size_is_capped_at_256_neurons() {
    assert!(NetworkLayout::random_recurrent(1, 2, 250, 6, 2, 2, 3).is_ok());
    assert!(NetworkLayout::random_recurrent(1, 2, 251, 6, 2, 2, 3).is_err());
    let mut layout = tiny_layout();
    layout.neurons = vec![layout.neurons[0]; 257];
    assert!(layout.validate().is_err());
}

// ----- 4. learning vs heredity ----------------------------------------------

fn follow_layout() -> NetworkLayout {
    NetworkLayout::random_recurrent(7, 1, 8, 1, 8, 3, 3).unwrap()
}

/// Reward: minus the mean absolute error between the decoded output and the
/// input level, after a settling time in each level.
fn follow_reward(layout: &NetworkLayout, learned: &LearnedState) -> f64 {
    let mut net = SpikingNetwork::new(layout.clone(), learned).unwrap();
    let levels = [0.1f32, 0.9, 0.5, 0.2, 0.7, 0.4];
    let (mut error, mut count) = (0.0f64, 0usize);
    for level in levels {
        for t in 0..300 {
            let y = net.step(0.001, &[level]).unwrap()[0];
            if t >= 150 {
                error += f64::from((y - level).abs());
                count += 1;
            }
        }
    }
    -error / count as f64
}

fn train(layout: &NetworkLayout, mut learned: LearnedState, seed: u64, trials: usize) -> (LearnedState, f64, f64) {
    let mut climber = HillClimber::new(seed, 0.1);
    let start = follow_reward(layout, &learned);
    climber.set_baseline(start);
    for _ in 0..trials {
        let candidate = climber.propose(&learned, layout);
        let reward = follow_reward(layout, &candidate);
        climber.judge(&mut learned, candidate, reward);
    }
    let end = follow_reward(layout, &learned);
    (learned, start, end)
}

#[test]
fn the_network_learns_to_make_its_output_follow_its_input() {
    let layout = follow_layout();
    let scratch = LearnedState::from_scratch(&layout);
    let (learned, start, end) = train(&layout, scratch.clone(), 99, 250);
    eprintln!("follow task: mean abs error {:.4} -> {:.4}, accepted {}/{}", -start, -end, learned.accepted_updates, learned.trials);
    assert!(end > start, "learning must improve the reward");
    assert!(-end < 0.12, "final mean error {} too large", -end);
    assert!(-start > 0.2, "task must be nontrivial at the start");
    assert!(learned.accepted_updates > 0 && learned.trials == 250);
    // Bounded: every learned number stays within the layout limits.
    learned.validate_against(&layout).unwrap();
    // Deterministic: same seed, same result bit for bit.
    let (again, _, end2) = train(&layout, scratch, 99, 250);
    assert_eq!(learned, again);
    assert_eq!(end, end2);
}

#[test]
fn a_rejected_trial_never_changes_the_learned_state() {
    let layout = follow_layout();
    let mut learned = LearnedState::from_scratch(&layout);
    let mut climber = HillClimber::new(5, 0.1);
    climber.set_baseline(0.0); // nothing can beat the best possible reward
    let before = learned.clone();
    let candidate = climber.propose(&learned, &layout);
    assert_ne!(candidate, before);
    assert!(!climber.judge(&mut learned, candidate, -1.0));
    assert!(!climber.judge(&mut learned, before.clone(), f64::NAN));
    assert_eq!(learned.weights, before.weights);
    assert_eq!(learned.neuron_bias, before.neuron_bias);
}

#[test]
fn learning_does_not_touch_the_genome_and_offspring_start_from_scratch() {
    let scene = world();
    let layout_s = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
    let genome = genome_for_body("creature-fixture", &layout_s, organs(), 3, 10, 2).unwrap();
    let frozen = serde_json::to_string(&genome).unwrap();

    // No gait: from scratch, equal to the genome's regenerated weights.
    let scratch = Creature::new(genome.clone(), None, &scene, 0).unwrap();
    assert_eq!(scratch.learned(), &LearnedState::from_scratch(&genome.network));

    // With a gait: starts from exactly that state and continues from it.
    let mut taught = LearnedState::from_scratch(&genome.network);
    taught.weights.input[0] = genome.network.input_weight_limit * 0.9;
    taught.accepted_updates = 17;
    let mut with_gait = Creature::new(genome.clone(), Some(taught.clone()), &scene, 0).unwrap();
    assert_eq!(with_gait.learned(), &taught);
    assert_ne!(with_gait.learned(), scratch.learned());

    // Learning one more step changes the learned state, never the genome.
    let layout = with_gait.genome().network.clone();
    let mut climber = HillClimber::new(1, 0.2);
    climber.set_baseline(f64::NEG_INFINITY);
    let candidate = climber.propose(with_gait.learned(), &layout);
    let mut state = with_gait.learned().clone();
    assert!(climber.judge(&mut state, candidate, -1.0));
    *with_gait.learned_mut() = state;
    assert_eq!(with_gait.learned().accepted_updates, 18, "continued from the gait");
    assert_ne!(with_gait.learned().weights, taught.weights);
    assert_eq!(serde_json::to_string(with_gait.genome()).unwrap(), frozen);

    // A mismatched gait is refused.
    let mut wrong = taught.clone();
    wrong.neuron_bias.pop();
    assert!(Creature::new(genome.clone(), Some(wrong), &scene, 0).is_err());

    // Offspring: deterministic per seed, valid, different, and learned from scratch.
    let a = genome.offspring(11).unwrap();
    assert_eq!(a, genome.offspring(11).unwrap());
    assert_ne!(a, genome.offspring(12).unwrap());
    assert_ne!(a.network.init_seed, genome.network.init_seed);
    assert_ne!(a.organs, genome.organs);
    assert!((a.organs.muscle_efficiency - genome.organs.muscle_efficiency).abs() < 0.25);
    let child = Creature::new(a.clone(), None, &scene, 0).unwrap();
    assert_eq!(child.learned(), &LearnedState::from_scratch(&a.network));
    assert_eq!(child.learned().accepted_updates, 0);
    assert_eq!(serde_json::to_string(&genome).unwrap(), frozen);

    // Both halves serialize on their own.
    let g2: Genome = serde_json::from_str(&frozen).unwrap();
    assert_eq!(g2, genome);
    let l2: LearnedState = serde_json::from_str(&serde_json::to_string(&taught).unwrap()).unwrap();
    assert_eq!(l2, taught);
}

// ----- 5. controller boundary ------------------------------------------------

fn reading(contacts: &[bool]) -> SensorReading {
    SensorReading {
        foot_contact: contacts.to_vec(),
        stored_energy_fraction: 1.0,
        ..SensorReading::empty(contacts.len())
    }
}

#[test]
fn all_controllers_share_one_boundary_and_commands_are_clamped_by_the_limit() {
    let mut recorded = RecordedAnimationController;
    let mut hand = HandWrittenController { standing_strength: 0.4 };
    let r = reading(&[true, false]);
    let a = recorded.control(0.01, &r, 1).unwrap();
    let b = hand.control(0.01, &r, 1).unwrap();
    assert_eq!(a.actuator_strength, vec![1.0]);
    assert!((b.actuator_strength[0] - 0.7).abs() < 1e-6);

    let mut scene = world();
    let cmd = ControlCommand { actuator_strength: vec![0.9] };
    apply_command(&cmd, 0.5, &mut scene.skeletal_actuators).unwrap();
    assert_eq!(scene.skeletal_actuators[0].strength_scale, 0.5);
    assert!(scene.skeletal_actuators[0].enabled);
    apply_command(&cmd, 0.0, &mut scene.skeletal_actuators).unwrap();
    assert!(!scene.skeletal_actuators[0].enabled, "exhausted creature is disabled");
    let wild = ControlCommand { actuator_strength: vec![f32::NAN] };
    apply_command(&wild, 1.0, &mut scene.skeletal_actuators).unwrap();
    assert_eq!(scene.skeletal_actuators[0].strength_scale, 0.0);
    let big = ControlCommand { actuator_strength: vec![7.0] };
    apply_command(&big, 1.0, &mut scene.skeletal_actuators).unwrap();
    assert_eq!(scene.skeletal_actuators[0].strength_scale, 1.0);
    assert!(apply_command(&ControlCommand { actuator_strength: vec![] }, 1.0, &mut scene.skeletal_actuators).is_err());
}

fn small_creature(scene: &Scene, capacity_j: f64, store_fraction: f64) -> Creature {
    small_creature_with_peak(scene, capacity_j, store_fraction, 5.0)
}

fn small_creature_with_peak(scene: &Scene, capacity_j: f64, store_fraction: f64, peak_w: f64) -> Creature {
    let mut p = organs();
    p.store_capacity_j = capacity_j;
    p.birth_store_fraction = store_fraction;
    p.peak_mechanical_power_w = peak_w;
    p.exhaustion_fraction = 0.01;
    let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
    let genome = genome_for_body("creature-fixture", &layout, p, 21, 16, 2).unwrap();
    Creature::new(genome, None, scene, 0).unwrap()
}

/// The creature slides over the ground at `speed` m/s so that the motor loads
/// (about 1 N per REV in this fixture) act along a real velocity.
fn start_sliding(scene: &mut Scene, speed: f32) {
    for i in GROUND..GROUND + 8 {
        scene.voxels[i].velocity = Vec3::new(speed, 0.0, 0.0);
    }
}

fn finite_scene(scene: &Scene) -> bool {
    scene.positions.iter().all(|p| p.is_finite())
        && scene.voxels.iter().all(|v| v.velocity.is_finite() && v.has_finite_thermal_state())
}

#[test]
fn driving_steps_debits_at_least_the_mechanical_work_and_keeps_the_ledger_non_negative() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    start_sliding(&mut scene, 0.3);
    let mut creature = small_creature(&scene, 1.0e6, 0.5);
    let mut controller = creature.spiking_controller().unwrap();
    let stored0 = creature.ledger().stored_j();
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let (mut debited, mut positive, mut net, mut heat) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..12 {
        let report = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
        assert!(finite_scene(&scene));
        assert!(creature.ledger().stored_j() >= 0.0);
        assert!((0.0..=1.0).contains(&report.applied));
        debited += report.outcome.debited_j;
        positive += report.outcome.positive_work_j;
        net += report.outcome.net_mechanical_work_j;
        heat += report.outcome.heat_per_rev_j.iter().sum::<f64>();
    }
    eprintln!("drive: debited {debited:.3e} positive {positive:.3e} net {net:.3e} heat {heat:.3e}");
    assert!(positive > 0.0, "the motors did real work");
    assert!(debited >= positive && debited >= net);
    assert!((debited - net - heat).abs() < 1e-9 * debited.max(1.0));
    assert!((stored0 - creature.ledger().stored_j() - debited).abs() < 1e-9);
    assert!(creature.ledger().conservation_residual_j().abs() < 1e-6);
    assert!(controller.network().cost().steps > 0);
}

#[test]
fn an_exhausted_creature_cannot_move_until_it_eats() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    // 5 mJ stored of 1 J capacity: below the 1 % vital reserve.
    let mut creature = small_creature(&scene, 1.0, 0.005);
    let mut controller = RecordedAnimationController;
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    for _ in 0..10 {
        let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
        assert_eq!(r.applied, 0.0);
        assert_eq!(scene.skeletal_actuators[0].last_force_norm_n(), 0.0);
        assert!(!scene.skeletal_actuators[0].enabled);
        assert_eq!(r.outcome.debited_j, 0.0);
    }
    assert!(finite_scene(&scene));
    creature.ledger_mut().ingest_food(10.0).unwrap();
    for _ in 0..100 {
        creature.ledger_mut().advance_digestion(0.1);
    }
    assert!(creature.ledger().strength_limit(0.01) > 0.0);
    let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
    assert!(r.applied > 0.0 && scene.skeletal_actuators[0].enabled);
    assert!(scene.skeletal_actuators[0].last_force_norm_n() > 0.0);
}

#[test]
fn a_draining_store_weakens_the_creature_without_overdraft() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    start_sliding(&mut scene, 0.3);
    // Design power 30 W (about what the fixture's motors draw while sliding).
    let mut creature = small_creature_with_peak(&scene, 20.0, 1.0, 30.0);
    let mut controller = RecordedAnimationController;
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let mut previous = creature.ledger().stored_j();
    let (mut applied, mut debits) = (Vec::new(), Vec::new());
    for _ in 0..60 {
        let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
        let stored = creature.ledger().stored_j();
        assert!(stored <= previous && stored >= 0.0);
        previous = stored;
        applied.push(r.applied);
        debits.push(r.outcome.debited_j);
        assert!(finite_scene(&scene));
    }
    eprintln!("draining store: applied {:?} final store {previous:.3e}", applied.iter().step_by(6).collect::<Vec<_>>());
    assert!(previous < 20.0, "the store was spent on motor work");
    assert!(applied.last().unwrap() < &applied[0], "muscles weaken as the store drains");
    let total: f64 = debits.iter().sum();
    assert!(creature.ledger().unfunded_work_total_j() < 0.1 * total, "shortfall stays small");
    // Once the limit reaches zero nothing more is debited.
    if let Some(first_zero) = applied.iter().position(|&a| a == 0.0) {
        assert!(debits[first_zero..].iter().all(|&d| d == 0.0));
    }
}

// ----- 6. sensors from real scene state ---------------------------------------

const COLUMN: usize = 4;

/// A 2 x 2 x 2 block (bottom layer = root bone, top layer = limb bone) resting
/// on fixed ground, `raised` REV heights up. Creature REVs are 4..12.
fn column_world(raised: i32) -> Scene {
    let cells: Vec<[i32; 3]> = [0, 1]
        .iter()
        .flat_map(|&z| [[0, 0], [1, 0], [0, 1], [1, 1]].map(|[x, y]| [x, y, z]))
        .collect();
    let mut builder = SceneBuilder::standard();
    for x in 0..2 {
        for y in 0..2 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, -1], 1).fixed());
        }
    }
    for [x, y, z] in &cells {
        builder = builder.add_voxel(VoxelSpawn::at_cell([*x, *y, z + raised], 1));
    }
    let mut scene = builder.build().unwrap();
    let actuator =
        SkeletalActuator::new(creature_model_of(&cells), COLUMN..COLUMN + 8, Quat::IDENTITY, 12).unwrap();
    scene.skeletal_actuators.push(actuator);
    scene.config.fixed_dt = 0.01;
    scene
}

fn creature_on(scene: &Scene, feet: Vec<usize>) -> Creature {
    let mut o = organs();
    o.store_capacity_j = 1.0e6;
    let layout = SensorLayout::new(&scene.skeletal_actuators[0], feet).unwrap();
    let genome = genome_for_body("sensor-fixture", &layout, o, 5, 8, 1).unwrap();
    Creature::new(genome, None, scene, 0).unwrap()
}

#[test]
fn every_sensor_has_a_stable_id_a_unit_and_a_cost_and_reads_are_noise_free() {
    let scene = world();
    let ledger = ResourceLedger::new(organs()).unwrap();
    let layout = bound_layout(&scene, vec![0, 4]);
    let catalog = layout.catalog();
    let a = SensorReading::read(&scene, &layout, &ledger).unwrap();
    let b = SensorReading::read(&scene, &layout, &ledger).unwrap();
    assert_eq!(a, b, "no noise and no hidden state");
    assert_eq!(a.values().len(), catalog.len());
    let labels: Vec<String> = catalog.iter().map(|d| d.id.label()).collect();
    let unique: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique.len(), labels.len(), "ids are unique: {labels:?}");
    assert!(labels.contains(&"foot_normal_load/rev=4".to_string()));
    assert!(labels.contains(&"support_load".to_string()));
    assert!(labels.contains(&"joint_angle/bone=1".to_string()));
    assert!(labels.contains(&"muscle_load/bone=0".to_string()));
    let unit_of = |label: &str| catalog.iter().find(|d| d.id.label() == label).unwrap().unit;
    assert_eq!(unit_of("foot_normal_load/rev=0"), "N");
    assert_eq!(unit_of("com_height"), "m");
    assert_eq!(unit_of("body_tilt"), "rad");
    assert_eq!(unit_of("tissue_stress/bone=1"), "Pa");
    assert_eq!(unit_of("angular_velocity/x"), "rad/s");

    // The id of a foot is its creature-local REV index, not a scene index.
    let mut builder = SceneBuilder::standard();
    for x in 0..4 {
        for y in 0..3 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, -1], 1).fixed());
        }
    }
    for cell in creature_cells() {
        builder = builder.add_voxel(VoxelSpawn::at_cell(cell, 1));
    }
    let mut shifted = builder.build().unwrap();
    let total = shifted.voxels.len();
    shifted.skeletal_actuators.push(
        SkeletalActuator::new(creature_model(), total - 8..total, Quat::IDENTITY, total).unwrap(),
    );
    assert_ne!(shifted.skeletal_actuators[0].voxel_range().start, GROUND);
    let moved_labels: Vec<String> = bound_layout(&shifted, vec![0, 4])
        .catalog()
        .iter()
        .map(|d| d.id.label())
        .collect();
    assert_eq!(moved_labels, labels, "ids follow the model's REVs, not the scene block");

    assert!(a.is_finite());
    assert_eq!(a.cost.reads, 1);
    assert_eq!(a.cost.values, catalog.len() as u64);
    assert!(a.cost.rev_reads > 0 && a.cost.bond_reads > 0 && a.cost.pair_tests > 0);
    assert_eq!(a.cost.operations(), a.cost.rev_reads + a.cost.bond_reads + a.cost.pair_tests);
    let mut unbound = SensorLayout::new(&scene.skeletal_actuators[0], vec![0]).unwrap();
    assert!(SensorReading::read(&scene, &unbound, &ledger).is_err(), "unbound layouts are refused");
    unbound.bind(&scene).unwrap();
    assert!(unbound.is_bound());
}

#[test]
fn foot_load_shares_the_support_measured_by_momentum_balance() {
    let mut passive = PassiveController;
    let mut step = |s: &mut Scene| solver::step(s, &mut None);

    // One layer on the ground: each of the 8 foot REVs carries its own weight.
    let mut scene = world_raised(3);
    scene.config.fixed_dt = 0.01;
    let mut creature = creature_on(&scene, (0..8).collect());
    let m_g = f64::from(scene.voxels[GROUND].mass) * 9.81;
    let mut readings = Vec::new();
    for _ in 0..20 {
        creature.drive_step(&mut scene, &mut passive, &mut step).unwrap();
        readings.push(creature.sense(&scene).unwrap());
    }
    let falling = &readings[2];
    assert!(falling.foot_contact.iter().all(|&c| !c));
    assert!(
        falling.support_load_n == 0.0 && falling.foot_normal_load_n.iter().all(|&l| l == 0.0),
        "free fall carries no load: {:?}",
        falling.foot_normal_load_n
    );
    let landing = readings.iter().map(|r| r.foot_normal_load_n[0]).fold(0.0f32, f32::max);
    assert!(f64::from(landing) > 3.0 * m_g, "landing impact load {landing} vs {m_g}");
    let rest = readings.last().unwrap();
    assert!(rest.foot_contact.iter().all(|&c| c));
    for load in &rest.foot_normal_load_n {
        assert!((f64::from(*load) / m_g - 1.0).abs() < 0.02, "foot load {load} vs m g {m_g}");
    }
    assert!((rest.support_load_n / rest.body_weight_n - 1.0).abs() < 0.02, "support carries the weight");
    assert!(rest.inertia_included);
    // The same body read without a measured acceleration (static assumption):
    // resting loads agree, a hanging body has no feet on anything and no load.
    let plain = SensorReading::read(&scene, creature.sensor_layout(), creature.ledger()).unwrap();
    assert!(!plain.inertia_included);
    assert!((plain.support_load_n / plain.body_weight_n - 1.0).abs() < 0.02);
    let hanging = world_raised(3);
    let plain = SensorReading::read(&hanging, creature.sensor_layout(), creature.ledger()).unwrap();
    assert_eq!(plain.support_load_n, 0.0, "nothing touches, nothing supports");

    // A two-layer block: the load of the top layer reaches the feet through bonds.
    let mut scene = column_world(0);
    let mut creature = creature_on(&scene, vec![0, 1, 2, 3]);
    let m_g = f64::from(scene.voxels[COLUMN].mass) * 9.81;
    for _ in 0..40 {
        creature.drive_step(&mut scene, &mut passive, &mut step).unwrap();
    }
    let r = creature.sense(&scene).unwrap();
    eprintln!(
        "column foot loads / (m g) = {:?}",
        r.foot_normal_load_n.iter().map(|l| f64::from(*l) / m_g).collect::<Vec<_>>()
    );
    for load in &r.foot_normal_load_n {
        assert!((f64::from(*load) / m_g - 2.0).abs() < 0.1, "column foot load {load} vs 2 m g {m_g}");
    }
    let total: f32 = r.foot_normal_load_n.iter().sum();
    assert!((total / r.body_weight_n - 1.0).abs() < 0.05, "feet carry the weight: {total} vs {}", r.body_weight_n);
}

#[test]
fn height_tilt_and_pose_error_follow_the_scene() {
    let mut scene = column_world(2);
    let ledger = ResourceLedger::new(organs()).unwrap();
    let layout = bound_layout(&scene, vec![0]);
    let r = SensorReading::read(&scene, &layout, &ledger).unwrap();
    // Layers at cells 2 and 3: centres 2.5 and 3.5 REVs up, so the block centre is at 3 REVs.
    assert!((r.com_height_m.unwrap() - 0.03).abs() < 1e-6, "{:?}", r.com_height_m);
    assert!((r.body_height_m.unwrap() - 0.005).abs() < 1e-6, "centre half a REV above the lowest REV");
    assert!(r.tilt_rad.unwrap() < 1e-6);
    assert!(r.pose_error_rms_m.unwrap() < 1e-9, "the rest pose is the commanded pose at t = 0");

    // Turn the block 0.2 rad about the horizontal y axis through its centre.
    let centre = (COLUMN..COLUMN + 8).map(|i| scene.positions[i]).sum::<DVec3>() / 8.0;
    let turn = DQuat::from_rotation_y(0.2);
    for i in COLUMN..COLUMN + 8 {
        scene.positions[i] = centre + turn * (scene.positions[i] - centre);
    }
    let r = SensorReading::read(&scene, &layout, &ledger).unwrap();
    assert!((r.tilt_rad.unwrap() - 0.2).abs() < 1e-4, "{:?}", r.tilt_rad);

    // Push one limb REV 3 mm out of the commanded pose: the pose error is the
    // distance to the commanded position in the comoving frame.
    let mut scene = column_world(0);
    scene.positions[COLUMN + 5].x += 0.003;
    let errors = scene.skeletal_actuators[0]
        .pose_error_m(&scene.positions, &scene.voxels, &scene.rotations)
        .unwrap();
    assert!((f64::from(errors[5]) - 0.003 * 7.0 / 8.0).abs() < 1e-6, "{}", errors[5]);
    assert!((f64::from(errors[0]) - 0.003 / 8.0).abs() < 1e-6);
    let r = SensorReading::read(&scene, &bound_layout(&scene, vec![0]), &ledger).unwrap();
    let limb = r.muscles.iter().find(|m| m.bone_index == 1).unwrap();
    let root = r.muscles.iter().find(|m| m.bone_index == 0).unwrap();
    assert!(limb.pose_error_rms_m.unwrap() > 0.001);
    assert!(limb.pose_error_rms_m.unwrap() > 2.0 * root.pose_error_rms_m.unwrap());
}

#[test]
fn muscle_load_and_tissue_stress_come_from_the_actuator_and_the_bonds() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    let mut creature = small_creature(&scene, 1.0e6, 0.5);
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let mut controller = RecordedAnimationController;
    for _ in 0..6 {
        creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
    }
    let r = creature.sense(&scene).unwrap();
    let actuator = &scene.skeletal_actuators[0];
    let l1: f32 = actuator.last_forces_n().iter().map(|f| f.length()).sum();
    let reported: f32 = r.muscles.iter().map(|m| m.actuator_load_n).sum();
    assert!(l1 > 0.0 && (reported - l1).abs() <= 1e-4 * l1, "muscle load {reported} vs actuator {l1}");

    // Independent largest bond stress over the bonds that touch the creature.
    let range = actuator.voxel_range();
    let mut expected = 0.0f32;
    for axis in 0..3 {
        for owner in 0..scene.voxels.len() {
            let j = scene.workspace[axis][owner].rest_neighbor;
            if j == crate::bond::EMPTY || scene.bonds[axis][owner].neighbor == crate::bond::EMPTY {
                continue;
            }
            if range.contains(&owner) || range.contains(&(j as usize)) {
                let force = scene.workspace[axis][owner].last_force.length();
                expected = expected.max(force / (CELL_M * CELL_M) as f32);
            }
        }
    }
    let peak = r.muscles.iter().map(|m| m.peak_tissue_stress_pa).fold(0.0f32, f32::max);
    assert!(expected > 0.0, "driving loads the bonds");
    assert!((peak - expected).abs() <= 1e-4 * expected, "stress {peak} vs {expected}");
    let strain = r.muscles.iter().map(|m| m.peak_tissue_strain).fold(0.0f32, f32::max);
    assert!(strain > 0.0 && strain < 0.1, "strain {strain}");

    // A relaxed body has no actuator load.
    let mut relaxed = world();
    relaxed.config.fixed_dt = 0.01;
    let mut c2 = small_creature(&relaxed, 1.0e6, 0.5);
    c2.drive_step(&mut relaxed, &mut PassiveController, &mut step).unwrap();
    let r2 = c2.sense(&relaxed).unwrap();
    assert!(r2.muscles.iter().all(|m| m.actuator_load_n == 0.0));
}

#[test]
fn a_driven_step_reports_the_cost_and_time_of_sensing_control_and_accounting() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    let mut creature = small_creature(&scene, 1.0e6, 0.5);
    let mut controller = creature.spiking_controller().unwrap();
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let mut spent = SensorCost::default();
    for _ in 0..3 {
        let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
        spent.accumulate(r.sensor_cost);
        assert!(r.timing.solver_s > 0.0);
    }
    assert_eq!(spent, creature.sensor_cost_total());
    assert_eq!(creature.sensor_cost_total().reads, 3);
    assert_eq!(controller.network().cost().steps, 30, "the network runs at its own 1 ms step");
}

// ----- 7. learning a simple task from real sensor frames ------------------------

/// Sensor frames of a real physical episode: the creature is dropped from
/// `cells` REV heights with relaxed muscles and lands. One frame per fixed step.
fn drop_frames(cells: i32, steps: usize) -> Vec<SensorReading> {
    let mut scene = world_raised(cells);
    scene.config.fixed_dt = 0.01;
    let mut creature = small_creature(&scene, 1.0e6, 0.5);
    let mut passive = PassiveController;
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let mut frames = Vec::new();
    for _ in 0..steps {
        creature.drive_step(&mut scene, &mut passive, &mut step).unwrap();
        frames.push(creature.sense(&scene).unwrap());
    }
    frames
}

fn learner(seed: u64) -> Creature {
    let scene = world();
    let mut o = organs();
    o.store_capacity_j = 1.0e6;
    let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
    // Two muscle channels; the task trains channel 1 only.
    let mut genome = genome_for_body("learner", &layout, o, seed, 12, 2).unwrap();
    // Start with most neurons tonically active: a silent network has a flat
    // reward landscape (no spikes, nothing to improve on).
    genome.network.init_bias_fraction = 0.8;
    Creature::new(genome, None, &scene, 0).unwrap()
}

fn loaded_task(frames: &[SensorReading]) -> ActivationTask {
    let weight_per_rev = frames[0].body_weight_n / 8.0;
    ActivationTask::new(
        1,
        SensorCondition::FootLoadAtLeast {
            foot: 0,
            newtons: 0.5 * weight_per_rev,
        },
    )
}

#[test]
fn a_chosen_muscle_learns_to_follow_a_sensor_condition_and_the_genome_is_unchanged() {
    let train = drop_frames(6, 40);
    let held_out = drop_frames(4, 40);
    let task = loaded_task(&train);
    let airborne = train.iter().filter(|f| !task.condition.holds(f)).count();
    eprintln!("train frames: {airborne} unloaded, {} loaded", train.len() - airborne);
    assert!(airborne > task.settle_frames + 3 && train.len() - airborne > 10, "both classes occur in the physics");
    assert!(task.condition.holds(train.last().unwrap()) && !task.condition.holds(&train[2]));

    let mut creature = learner(31);
    let genome_before = serde_json::to_string(creature.genome()).unwrap();
    let learned_before = creature.learned().clone();
    let layout = creature.genome().network.clone();
    let dt = 0.01;
    let held_before = task.reward(&layout, &learned_before, &held_out, dt).unwrap();

    let report = creature.learn_activation_task(&task, &train, dt, 7, 300).unwrap();
    let held_after = task.reward(&layout, creature.learned(), &held_out, dt).unwrap();
    eprintln!(
        "task reward {:.4} -> {:.4} (accepted {}/{}); held-out episode {:.4} -> {:.4}; a constant output scores -0.5",
        report.reward_before, report.reward_after, report.accepted, report.trials, held_before, held_after
    );
    assert_eq!(report.trials, 300);
    assert!(report.accepted > 0 && report.accepted <= 300);
    assert!(report.reward_after > report.reward_before, "learning improves the reward");
    assert!(report.reward_after > -0.35, "better than any constant activation (-0.5)");
    assert!(held_after > held_before, "the improvement carries to a new drop");
    assert!(held_after > -0.45, "held-out reward beats a constant activation");

    // Learning moved the learned state and nothing else.
    assert_ne!(creature.learned().weights, learned_before.weights);
    assert_eq!(creature.learned().accepted_updates as usize, report.accepted);
    assert_eq!(creature.learned().trials, 300);
    creature.learned().validate_against(&layout).unwrap();
    assert_eq!(serde_json::to_string(creature.genome()).unwrap(), genome_before);
    assert_eq!(creature.genome().network.init_seed, learned_before_seed(&creature));

    // Deterministic for a fixed seed; another seed takes another path.
    let mut again = learner(31);
    let report2 = again.learn_activation_task(&task, &train, dt, 7, 300).unwrap();
    assert_eq!(report2, report);
    assert_eq!(again.learned(), creature.learned());
    let mut other = learner(31);
    other.learn_activation_task(&task, &train, dt, 8, 300).unwrap();
    assert_ne!(other.learned().weights, creature.learned().weights);

    // A child inherits the genome, not what the parent learned.
    let child = Creature::new(creature.genome().offspring(5).unwrap(), None, &world(), 0).unwrap();
    assert_eq!(child.learned().accepted_updates, 0);
    assert_ne!(child.learned().weights, creature.learned().weights);
}

fn learned_before_seed(creature: &Creature) -> u64 {
    creature.genome().network.init_seed
}

#[test]
fn a_task_without_both_classes_or_with_a_bad_channel_is_refused() {
    let frames = drop_frames(6, 40);
    let creature = learner(3);
    let layout = creature.genome().network.clone();
    let learned = creature.learned().clone();
    let never = ActivationTask::new(1, SensorCondition::FootLoadAtLeast { foot: 0, newtons: 1.0e6 });
    assert!(never.reward(&layout, &learned, &frames, 0.01).is_err());
    let bad_channel = ActivationTask::new(5, SensorCondition::TiltAtLeast { rad: 0.1 });
    assert!(bad_channel.reward(&layout, &learned, &frames, 0.01).is_err());
    let missing_foot = ActivationTask::new(0, SensorCondition::FootLoadAtLeast { foot: 9, newtons: 0.0 });
    assert!(!missing_foot.condition.holds(&frames[0]));
}

// ----- 8. energy path through the scene -----------------------------------------

#[test]
fn muscle_losses_reach_the_muscle_revs_as_internal_energy_and_the_books_balance() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    start_sliding(&mut scene, 0.3);
    let mut creature = small_creature(&scene, 1.0e6, 0.5);
    let mut controller = creature.spiking_controller().unwrap();
    let internal = |s: &Scene| -> f64 {
        (GROUND..GROUND + 8).map(|i| s.voxels[i].total_internal_energy_j()).sum()
    };
    let stored0 = creature.ledger().stored_j();
    let (mut debited, mut net, mut heat, mut deposited) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..12 {
        // The closure notes the internal energy the moment the solver is done.
        let mut after_solver = 0.0;
        let mut step = |s: &mut Scene| {
            solver::step(s, &mut None);
            after_solver = internal(s);
        };
        let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
        deposited += internal(&scene) - after_solver;
        debited += r.outcome.debited_j;
        net += r.outcome.net_mechanical_work_j;
        heat += r.outcome.heat_per_rev_j.iter().sum::<f64>();
        assert_eq!(r.outcome.unfunded_work_j, 0.0);
    }
    assert!(heat > 0.0 && net.abs() < heat, "mostly losses: net {net:e} J, heat {heat:e} J");
    assert!((deposited - heat).abs() <= 1e-9 * heat, "heat in the REVs {deposited:e} vs ledger {heat:e}");
    assert!((stored0 - creature.ledger().stored_j() - debited).abs() <= 1e-12 * stored0);
    assert!((debited - (net + heat)).abs() <= 1e-9 * debited, "store drop = work + heat");
    assert!((creature.ledger().heat_total_j() - heat).abs() <= 1e-9 * heat);
}
