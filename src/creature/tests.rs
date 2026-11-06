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
fn foot_loads_are_the_solver_reactions_of_the_fixed_support() {
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
    eprintln!("landing peak foot load / (m g) = {:.2}", f64::from(landing) / m_g);
    assert!(f64::from(landing) > 1.5 * m_g, "landing impact load {landing} vs {m_g}");
    assert!(readings.iter().all(|r| r.load_source == LoadSource::SolverReaction));
    let rest = readings.last().unwrap();
    assert!(rest.foot_contact.iter().all(|&c| c));
    for load in &rest.foot_normal_load_n {
        assert!((f64::from(*load) / m_g - 1.0).abs() < 0.02, "foot load {load} vs m g {m_g}");
    }
    assert!((rest.support_load_n / rest.body_weight_n - 1.0).abs() < 0.02, "support carries the weight");
    assert!(rest.inertia_included);
    // The exact loads need no measured acceleration.
    let plain = SensorReading::read(&scene, creature.sensor_layout(), creature.ledger()).unwrap();
    assert_eq!(plain.load_source, LoadSource::SolverReaction);
    assert!((plain.support_load_n / plain.body_weight_n - 1.0).abs() < 0.02);
    let feet_sum: f32 = plain.foot_normal_load_n.iter().sum();
    assert!((feet_sum / plain.body_weight_n - 1.0).abs() < 0.02, "feet carry the weight");
    // Fallback: with no reaction vector the momentum-balance estimate is used
    // and agrees at rest (equal shares of the whole-body balance).
    scene.support_load_n.clear();
    let est = SensorReading::read(&scene, creature.sensor_layout(), creature.ledger()).unwrap();
    assert_eq!(est.load_source, LoadSource::MomentumEstimate);
    assert!(!est.inertia_included);
    assert!((est.support_load_n / est.body_weight_n - 1.0).abs() < 0.02);
    // A body that never stepped has no vector: fallback, nothing touches, no load.
    let hanging = world_raised(3);
    let plain = SensorReading::read(&hanging, creature.sensor_layout(), creature.ledger()).unwrap();
    assert_eq!(plain.load_source, LoadSource::MomentumEstimate);
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

// ----- 9. organ graph --------------------------------------------------------

/// The (axis, owner) of the bond between REVs `a` and `b`, in whichever direction it is stored.
fn bond_slot(scene: &Scene, a: usize, b: usize) -> (usize, usize) {
    for axis in 0..3 {
        if scene.bonds[axis][a].neighbor == b as u32 {
            return (axis, a);
        }
        if scene.bonds[axis][b].neighbor == a as u32 {
            return (axis, b);
        }
    }
    panic!("no bond between {a} and {b}");
}

fn cut(scene: &mut Scene, local_a: usize, local_b: usize) {
    let (axis, owner) = bond_slot(scene, GROUND + local_a, GROUND + local_b);
    scene.bonds[axis][owner].neighbor = crate::bond::EMPTY;
}

fn hurt(scene: &mut Scene, local_a: usize, local_b: usize, damage: f32) {
    let (axis, owner) = bond_slot(scene, GROUND + local_a, GROUND + local_b);
    scene.bonds[axis][owner].damage = damage;
}

/// Creature layout (local indices) in the fixture world, x = 0..3:
///   y=1:  2  3  6  7
///   y=0:  0  1  4  5
/// Brain on 6 and 7, stomach on 2, store on 3, foot sense on 0 and 1,
/// muscle on 4 and 5. The hinge between x=1 and x=2 holds the links 1-4 and 3-6.
fn anatomy(scene: &Scene) -> OrganGraph {
    let g = |locals: &[usize]| locals.iter().map(|l| GROUND + l).collect::<Vec<_>>();
    let plan = OrganPlan {
        brain: g(&[6, 7]),
        stomach: g(&[2]),
        store: g(&[3]),
        foot_sense: g(&[0, 1]),
        foot_indices: vec![0],
        body_sense: Vec::new(),
    };
    OrganGraph::standard(scene, GROUND..GROUND + 8, &plan, &[g(&[4, 5])], 100.0, 100.0).unwrap()
}

fn channel_of(graph: &OrganGraph, kind: ChannelKind, from: OrganRole, to: OrganRole) -> Channel {
    let f = graph.first_node(from).unwrap().id;
    let t = graph.first_node(to).unwrap().id;
    graph
        .channels()
        .iter()
        .find(|c| c.kind == kind && c.from == f && c.to == t)
        .unwrap()
        .clone()
}

fn organ_creature(scene: &Scene) -> Creature {
    let mut p = organs();
    p.store_capacity_j = 1.0e6;
    p.basal_power_w = 1.0;
    p.exhaustion_fraction = 0.01;
    let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
    let genome = genome_for_body("creature-fixture", &layout, p, 21, 16, 1).unwrap();
    let mut creature = Creature::new(genome, None, scene, 0).unwrap();
    creature.set_organs(anatomy(scene), 0.0).unwrap();
    creature
}

#[test]
fn organs_are_views_of_existing_revs_and_count_mass_and_energy_once() {
    let scene = world();
    let graph = anatomy(&scene);
    // Brain, stomach, store, feet and muscle together cover all eight REVs once.
    let all: f64 = (GROUND..GROUND + 8).map(|i| f64::from(scene.voxels[i].mass)).sum();
    assert!((graph.owned_mass_kg(&scene) - all).abs() <= 1e-12 * all);
    let energy: f64 = (GROUND..GROUND + 8).map(|i| scene.voxels[i].total_internal_energy_j()).sum();
    assert!((graph.owned_internal_energy_j(&scene) - energy).abs() <= 1e-12 * energy);
    // Sharing a REV between two nodes does not double the books.
    let mut shared = anatomy(&scene);
    shared
        .add_node(&scene, NodeSpec::new(OrganRole::Respiration, "lungs", vec![GROUND, GROUND + 6]))
        .unwrap();
    assert_eq!(shared.owned_mass_kg(&scene), graph.owned_mass_kg(&scene));
    assert_eq!(shared.owned_internal_energy_j(&scene), graph.owned_internal_energy_j(&scene));
    // The graph stores ids, not energy: heating a REV shows in the organ's energy.
    let mut warmed = world();
    warmed.voxels[GROUND + 4].add_internal_energy_j(5.0);
    assert!((graph.owned_internal_energy_j(&warmed) - energy - 5.0).abs() < 1e-3);
    // Tissue must belong to the body and exist.
    assert!(shared.add_node(&scene, NodeSpec::new(OrganRole::Store, "bad", vec![0])).is_err());
    assert!(shared.add_node(&scene, NodeSpec::new(OrganRole::Store, "bad", vec![])).is_err());
    // Intact body: every channel is fully available, every organ whole.
    for c in graph.channels() {
        assert_eq!(c.availability(), 1.0, "{c:?}");
        assert_eq!(c.throughput(), c.capacity);
    }
    for n in graph.nodes() {
        assert_eq!(n.integrity(), 1.0);
    }
}

#[test]
fn damage_to_the_path_of_a_channel_limits_its_throughput() {
    let mut scene = world();
    let mut graph = anatomy(&scene);
    let sense_to_brain = |g: &OrganGraph| channel_of(g, ChannelKind::Signal, OrganRole::SenseOrgan, OrganRole::Brain);
    let store_to_muscle = |g: &OrganGraph| channel_of(g, ChannelKind::Nutrient, OrganRole::Store, OrganRole::Muscle);
    let intact = sense_to_brain(&graph);
    assert_eq!((intact.availability(), intact.throughput()), (1.0, 1000.0));
    assert!(intact.delay_s() > 0.0 && intact.delay_s().is_finite(), "{intact:?}");

    // Both rows cross the hinge (links 1-4 and 3-6). Damage one: the signal
    // takes the other row at full strength.
    hurt(&mut scene, 1, 4, 0.7);
    graph.refresh(&scene);
    assert_eq!(sense_to_brain(&graph).availability(), 1.0, "detour through intact tissue");
    // Damage both: the better of the two limits the channel.
    hurt(&mut scene, 3, 6, 0.4);
    graph.refresh(&scene);
    let hurt_both = sense_to_brain(&graph);
    assert!((hurt_both.availability() - 0.6).abs() < 1e-6, "{hurt_both:?}");
    assert!((hurt_both.throughput() - 600.0).abs() < 1e-3);
    let nutrient = store_to_muscle(&graph);
    assert!((nutrient.availability() - 0.6).abs() < 1e-6, "{nutrient:?}");
    let muscle = graph.node(nutrient.to).unwrap().integrity();
    assert!((graph.strength_cap(0) - 0.6f64.min(muscle)).abs() < 1e-6, "the weakest link limits");
    // The stomach-to-store channel does not cross the hinge and is untouched.
    assert_eq!(channel_of(&graph, ChannelKind::Nutrient, OrganRole::Digestion, OrganRole::Store).availability(), 1.0);
    // A broken bond is worth zero health: break the better row (3-6).
    cut(&mut scene, 3, 6);
    graph.refresh(&scene);
    let c = sense_to_brain(&graph);
    assert!((c.availability() - 0.3).abs() < 1e-6, "only the 0.7-damaged row is left: {c:?}");
    // Cut the last connection: the channel is dead, signals and food do not cross.
    cut(&mut scene, 1, 4);
    graph.refresh(&scene);
    let c = sense_to_brain(&graph);
    assert_eq!((c.availability(), c.throughput()), (0.0, 0.0));
    assert!(c.delay_s().is_infinite() && c.route_ids().is_empty());
    assert_eq!(graph.strength_cap(0), 0.0);
    // Brain to muscle lies on the right side of the hinge and still works.
    assert_eq!(channel_of(&graph, ChannelKind::Signal, OrganRole::Brain, OrganRole::Muscle).availability(), 1.0);
    // The mechanical channel (muscle -> feet) crosses the hinge as well and is cut.
    assert_eq!(channel_of(&graph, ChannelKind::Mechanical, OrganRole::Muscle, OrganRole::SenseOrgan).availability(), 0.0);
    assert!(graph.cost().refreshes >= 5 && graph.cost().search_visits > 0);
}

#[test]
fn brain_damage_changes_the_controller_output_and_losing_it_leaves_a_passive_body() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    let mut creature = organ_creature(&scene);
    let mut controller = RecordedAnimationController;
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let healthy = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
    assert_eq!(healthy.commanded_each, vec![1.0]);
    assert_eq!(healthy.applied_each, vec![1.0]);

    // Wound the brain tissue (the 6-7 bond lies inside it): output falls with it.
    hurt(&mut scene, 6, 7, 0.5);
    let wounded = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
    let brain = creature.organs().unwrap().first_node(OrganRole::Brain).unwrap().integrity();
    assert!((brain - 0.5).abs() < 1e-6, "integrity {brain}");
    assert!((wounded.commanded_each[0] - 0.5).abs() < 1e-6, "{:?}", wounded.commanded_each);
    assert!(wounded.applied < healthy.applied);
    assert!(scene.skeletal_actuators[0].enabled);

    // Remove the brain REVs from the world (outside the bounds): central
    // control is gone, the body is not.
    let ids_before: Vec<u64> = (GROUND..GROUND + 8).map(|i| scene.rev_id(i).unwrap()).collect();
    let mass_before = scene.total_mass_kg();
    let positions_before: Vec<DVec3> = scene.positions.clone();
    scene.set_world_bounds(DVec3::splat(-100.0), DVec3::splat(100.0)).unwrap();
    for local in [6, 7] {
        scene.positions[GROUND + local].x = 1.0e3;
    }
    assert_eq!(scene.set_world_bounds(DVec3::splat(-100.0), DVec3::splat(100.0)).unwrap(), 2);
    assert_eq!(scene.voxels.len(), 14, "only the two brain REVs left; the body stays");
    let removed = scene.boundary_removals();
    assert_eq!(removed.voxel_count, 2);
    assert!(
        (scene.total_mass_kg() + removed.mass_kg - mass_before).abs() <= 1e-9 * mass_before,
        "mass is conserved"
    );

    // Stable ids: every surviving REV is still found as the same tissue.
    for local in 0..6 {
        let index = scene.rev_index(ids_before[local]).expect("surviving tissue keeps its id");
        assert_eq!(index, GROUND + local);
        assert_eq!(scene.positions[index], positions_before[GROUND + local]);
    }
    assert!(scene.rev_index(ids_before[6]).is_none() && scene.rev_index(ids_before[7]).is_none());

    let mut step_after = |s: &mut Scene| solver::step(s, &mut None);
    let lost = creature.drive_step(&mut scene, &mut controller, &mut step_after).unwrap();
    assert!(lost.topology_changed);
    assert_eq!(creature.organs().unwrap().brain_integrity(), 0.0);
    assert_eq!(lost.commanded_each, vec![0.0]);
    assert_eq!(lost.applied_each, vec![0.0]);
    assert!(finite_scene(&scene), "the body keeps simulating");
    assert_eq!(scene.voxels.len(), 14);
    assert!(creature.ledger().conservation_residual_j().abs() < 1e-6);
}

#[test]
fn ids_still_point_to_the_same_tissue_after_a_removal_in_the_middle() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    let mut creature = organ_creature(&scene);
    let (stomach, store, foot) = {
        let g = creature.organs().unwrap();
        (
            g.first_node(OrganRole::Digestion).unwrap().clone(),
            g.first_node(OrganRole::Store).unwrap().clone(),
            g.first_node(OrganRole::SenseOrgan).unwrap().clone(),
        )
    };
    let store_id = store.tissue_ids()[0];
    let store_rest = scene.rest_positions[GROUND + 3];
    let store_mass = scene.voxels[GROUND + 3].mass;
    // Detach REV 2 (the stomach) from its neighbours, throw it out of the world
    // inside the step, and let the solver remove it at the end of that step.
    scene.set_world_bounds(DVec3::splat(-100.0), DVec3::splat(100.0)).unwrap();
    for (a, b) in [(2, 3), (2, 0)] {
        cut(&mut scene, a, b);
    }
    let total_before = scene.total_mass_kg();
    let mut internal_after_solver = 0.0;
    let mut step = |s: &mut Scene| {
        s.positions[GROUND + 2].x = 1.0e3;
        solver::step(s, &mut None);
        internal_after_solver = s.voxels.iter().map(|v| v.total_internal_energy_j()).sum::<f64>();
    };
    let mut controller = RecordedAnimationController;
    creature.ledger_mut().ingest_food(10.0).unwrap();
    let report = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
    assert!(report.topology_changed, "the solver removed the REV during the step");
    assert_eq!(scene.voxels.len(), 15);
    let removed = scene.boundary_removals();
    assert!((scene.total_mass_kg() + removed.mass_kg - total_before).abs() <= 1e-9 * total_before);

    // The store sat behind the removed REV in the array; its id maps to a shifted index.
    let moved = scene.rev_index(store_id).unwrap();
    assert_eq!(moved, GROUND + 2, "indices shifted down by one");
    assert_eq!(scene.voxels[moved].mass, store_mass);
    assert_eq!(scene.rest_positions[moved], store_rest, "same lattice cell, same tissue");
    assert!(scene.rev_index(stomach.tissue_ids()[0]).is_none());

    // The graph follows the ids: the stomach is gone, the store and feet are whole.
    creature
        .drive_step(&mut scene, &mut controller, &mut |s: &mut Scene| solver::step(s, &mut None))
        .unwrap();
    let g = creature.organs().unwrap();
    assert_eq!(g.node(stomach.id).unwrap().integrity(), 0.0);
    assert!(g.node(store.id).unwrap().integrity() > 0.99);
    assert!(g.node(foot.id).unwrap().integrity() > 0.99);
    assert_eq!(g.digestion_throughput(), 0.0, "no stomach, no digestion");
    assert_eq!(creature.ledger().digestion_throughput(), 0.0);
    assert!(creature.ledger().stomach_j() > 0.0, "food stays put when digestion stops");
    assert!(creature.ledger().conservation_residual_j().abs() < 1e-9);
    // Energy books: the heat the ledger booked reached the surviving REVs.
    assert!(creature.ledger().heat_total_j() > 0.0);
    assert_eq!(creature.undeposited_heat_j(), 0.0);
    let internal_after: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
    assert!(internal_after > internal_after_solver, "heat from the metabolism reached the tissue");
}

#[test]
fn damaged_sense_tissue_degrades_the_sensor_inputs() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    let mut creature = organ_creature(&scene);
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    for _ in 0..20 {
        creature.drive_step(&mut scene, &mut PassiveController, &mut step).unwrap();
    }
    let healthy = creature.sense(&scene).unwrap();
    assert_eq!(healthy.foot_contact, vec![true, true]);
    assert!(healthy.foot_normal_load_n[0] > 0.0);
    // Hurt the tissue of the foot organ (its internal bond 0-1): foot 0 reads weaker.
    hurt(&mut scene, 0, 1, 0.8);
    creature.organs_mut().unwrap().refresh(&scene);
    let weak = creature.sense(&scene).unwrap();
    let integrity = creature.organs().unwrap().first_node(OrganRole::SenseOrgan).unwrap().integrity();
    assert!((integrity - 0.2).abs() < 1e-6);
    assert_eq!(weak.foot_contact, vec![false, true], "below the detection threshold the foot is silent");
    assert!((weak.foot_normal_load_n[0] - 0.2 * healthy.foot_normal_load_n[0]).abs() <= 1e-4 * healthy.foot_normal_load_n[0]);
    assert_eq!(weak.foot_normal_load_n[1], healthy.foot_normal_load_n[1], "other feet are untouched");
    assert_eq!(weak.joints, healthy.joints);
}

#[test]
fn damaged_stomach_slows_digestion_and_the_books_balance() {
    let mut scene = world();
    scene.config.fixed_dt = 0.01;
    let mut creature = organ_creature(&scene);
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    creature.ledger_mut().ingest_food(10.0).unwrap();
    let mut digest = |c: &mut Creature, scene: &mut Scene| {
        let before = c.ledger().stomach_j();
        for _ in 0..20 {
            c.drive_step(scene, &mut PassiveController, &mut step).unwrap();
        }
        before - c.ledger().stomach_j()
    };
    let full = digest(&mut creature, &mut scene);
    assert!((full - 0.4).abs() < 1e-6, "2 W for 0.2 s: {full}");
    // The stomach is REV 2 alone: its links are 2-0 and 2-3. Damage both by 0.5.
    hurt(&mut scene, 2, 0, 0.5);
    hurt(&mut scene, 2, 3, 0.5);
    let half = digest(&mut creature, &mut scene);
    assert!((half - 0.5 * full).abs() < 1e-6, "digestion at half: {half} vs {full}");
    assert!(creature.ledger().conservation_residual_j().abs() < 1e-9);
}

// ----- 10. several actuators -----------------------------------------------------

fn second_cells() -> Vec<[i32; 3]> {
    creature_cells().iter().map(|&[x, y, z]| [x + 6, y, z]).collect()
}

/// Two separate bodies on one floor: REVs 16..24 and 24..32, one skeletal
/// actuator each (scene actuators 0 and 1).
fn two_body_world() -> Scene {
    let mut builder = SceneBuilder::standard();
    for x in (0..4).chain(6..10) {
        for y in 0..2 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, -1], 1).fixed());
        }
    }
    for cells in [creature_cells(), second_cells()] {
        for [x, y, z] in cells {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], 1));
        }
    }
    let mut scene = builder.build().unwrap();
    assert_eq!(scene.voxels.len(), 32);
    for (range, cells) in [(16..24, creature_cells()), (24..32, second_cells())] {
        let actuator = SkeletalActuator::new(creature_model_of(&cells), range, Quat::IDENTITY, 32).unwrap();
        scene.skeletal_actuators.push(actuator);
    }
    scene.config.fixed_dt = 0.01;
    scene
}

struct Fixed(Vec<f32>);
impl Controller for Fixed {
    fn name(&self) -> &str {
        "fixed"
    }
    fn control(&mut self, _: f64, _: &SensorReading, n: usize) -> Result<ControlCommand, String> {
        assert_eq!(n, self.0.len());
        Ok(ControlCommand { actuator_strength: self.0.clone() })
    }
}

fn two_actuator_creature(scene: &Scene, outputs: usize) -> Creature {
    let mut p = organs();
    p.store_capacity_j = 1.0e6;
    p.peak_mechanical_power_w = 5.0;
    p.exhaustion_fraction = 0.01;
    let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
    let genome = genome_for_body("creature-fixture", &layout, p, 21, 16, outputs).unwrap();
    Creature::new_multi(genome, None, scene, &[0, 1]).unwrap()
}

#[test]
fn two_actuators_are_driven_independently_through_the_solver_and_the_ledger_balances() {
    let mut scene = two_body_world();
    for i in 16..32 {
        scene.voxels[i].velocity = Vec3::new(0.3, 0.0, 0.0);
    }
    let mut creature = two_actuator_creature(&scene, 2);
    assert_eq!(creature.actuator_count(), 2);
    let internal = |s: &Scene| -> f64 { (16..32).map(|i| s.voxels[i].total_internal_energy_j()).sum() };
    let stored0 = creature.ledger().stored_j();
    let (mut debited, mut net, mut heat, mut deposited) = (0.0, 0.0, 0.0, 0.0);
    let mut by_actuator = [0.0f64; 2];
    let schedule = [[1.0f32, 0.0], [0.0, 1.0], [0.25, 0.75], [1.0, 1.0]];
    let mut work_per_entry = Vec::new();
    for commands in schedule {
        // Keep both bodies sliding so that the loads act along a real velocity.
        for i in 16..32 {
            scene.voxels[i].velocity = Vec3::new(0.3, 0.0, 0.0);
        }
        let mut entry_work = [0.0f64; 2];
        let mut controller = Fixed(commands.to_vec());
        for _ in 0..4 {
            let mut after_solver = 0.0;
            let mut step = |s: &mut Scene| {
                solver::step(s, &mut None);
                after_solver = internal(s);
            };
            let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
            deposited += internal(&scene) - after_solver;
            assert_eq!(r.applied_each, commands.to_vec(), "each actuator gets its own factor");
            assert_eq!(r.commanded_each, commands.to_vec());
            for k in 0..2 {
                let a = &scene.skeletal_actuators[k];
                assert_eq!(a.strength_scale, commands[k]);
                assert_eq!(a.enabled, commands[k] > 0.0);
                by_actuator[k] += r.work_each_j[k];
                entry_work[k] += r.work_each_j[k];
                // A switched-off actuator emits no load and does no work.
                if commands[k] == 0.0 {
                    assert_eq!(a.last_force_norm_n(), 0.0);
                    assert_eq!(r.work_each_j[k], 0.0);
                }
            }
            let sum: f64 = r.work_each_j.iter().sum();
            assert!((sum - r.outcome.net_mechanical_work_j).abs() <= 1e-12 + 1e-9 * r.outcome.positive_work_j);
            debited += r.outcome.debited_j;
            net += r.outcome.net_mechanical_work_j;
            heat += r.outcome.heat_per_rev_j.iter().sum::<f64>();
            assert!(finite_scene(&scene));
        }
        work_per_entry.push(entry_work);
    }
    eprintln!("two actuators: work per schedule entry {work_per_entry:?}, total by actuator {by_actuator:?}");
    assert!(work_per_entry[0][0].abs() > 1e-6 && work_per_entry[0][1] == 0.0, "only actuator 0 works");
    assert!(work_per_entry[1][0] == 0.0 && work_per_entry[1][1].abs() > 1e-6, "only actuator 1 works");
    assert!(work_per_entry[2].iter().all(|w| w.abs() > 1e-6) && work_per_entry[3].iter().all(|w| w.abs() > 1e-6));
    // Ledger: the store paid net work plus heat, and the heat is in the REVs.
    assert!(by_actuator[0] != 0.0 && by_actuator[1] != 0.0);
    assert!((by_actuator.iter().sum::<f64>() - net).abs() <= 1e-9 * net.abs().max(1e-9));
    assert!((stored0 - creature.ledger().stored_j() - debited).abs() <= 1e-9 * stored0);
    assert!((debited - (net + heat)).abs() <= 1e-9 * debited, "debited {debited:e} vs net {net:e} + heat {heat:e}");
    assert!((deposited - heat).abs() <= 1e-9 * heat, "heat in the REVs {deposited:e} vs ledger {heat:e}");
    assert!(creature.ledger().conservation_residual_j().abs() < 1e-6);
    assert_eq!(creature.ledger().unfunded_work_total_j(), 0.0);
}

#[test]
fn a_spiking_network_drives_two_actuators_from_two_outputs_and_bad_setups_are_refused() {
    let mut scene = two_body_world();
    let mut creature = two_actuator_creature(&scene, 2);
    let mut controller = creature.spiking_controller().unwrap();
    let mut step = |s: &mut Scene| solver::step(s, &mut None);
    let mut differ = false;
    for _ in 0..12 {
        let r = creature.drive_step(&mut scene, &mut controller, &mut step).unwrap();
        assert_eq!(r.commanded_each.len(), 2);
        differ |= r.commanded_each[0] != r.commanded_each[1];
    }
    assert!(differ, "two outputs, two separate strengths");
    // One output cannot feed two actuators (the old averaging needs a multiple).
    let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
    let genome = genome_for_body("creature-fixture", &layout, organs(), 3, 8, 1).unwrap();
    let mut single_output = Creature::new_multi(genome.clone(), None, &scene, &[0, 1]).unwrap();
    let mut c = single_output.spiking_controller().unwrap();
    assert!(single_output.drive_step(&mut scene, &mut c, &mut step).is_err());
    assert!(Creature::new_multi(genome.clone(), None, &scene, &[0, 0]).is_err(), "same actuator twice overlaps");
    assert!(Creature::new_multi(genome, None, &scene, &[]).is_err());
}
