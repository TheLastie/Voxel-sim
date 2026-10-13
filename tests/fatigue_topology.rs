use glam::Vec3;
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{SceneBuilder, VoxelSpawn, CELL},
    solver,
};

#[test]
fn fatigue_failure_splits_clusters_and_released_rev_keeps_mass_and_motion() {
    let iron = Material::component_id("iron").unwrap();
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.01, 4)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
        .build()
        .unwrap();
    scene.config.fracture.fatigue_cycles = 30.0;
    scene.config.fracture.fatigue_exponent = 1.0;
    scene.config.fracture.fatigue_threshold = 0.1;

    let initial_mass = scene.total_mass_kg();
    let strength = scene.properties[0].tensile_strength;
    let young = scene.properties[0].young;
    let displacement = 0.4 * strength / young * CELL;
    let base = scene.positions[0];
    let end = scene.positions[1];
    assert_eq!(scene.cluster_count, 1);
    assert_eq!(scene.bonds[0][0].neighbor, 1);

    let mut tool = None;
    let mut break_step = None;
    let mut max_stress_ratio = 0.0f32;
    for step in 1..=1_100 {
        let phase = std::f32::consts::TAU * (step % 10) as f32 / 10.0;
        let targets = [
            base.to_array().map(Some),
            [
                Some(end.x + displacement * phase.sin()),
                Some(end.y),
                Some(end.z),
            ],
        ];
        solver::step_with_axis_targets(&mut scene, &mut tool, &targets);
        let strain = (scene.positions[1].x - scene.positions[0].x - CELL) / CELL;
        max_stress_ratio = max_stress_ratio.max((strain * young / strength).abs());
        if scene.bonds[0][0].neighbor == EMPTY {
            break_step = Some(step);
            break;
        }
    }

    let break_step = break_step.expect("subcritical cyclic load did not fatigue the bond");
    assert!(break_step < 1_100);
    assert!(
        max_stress_ratio < 1.0,
        "stress exceeded static failure threshold"
    );
    assert_eq!(scene.cluster_count, 2);
    assert_ne!(scene.clusters[0], scene.clusters[1]);
    assert!((scene.total_mass_kg() - initial_mass).abs() < 1e-12);

    let before_release_step = scene.positions[1];
    let targets = [base.to_array().map(Some), [None; 3]];
    solver::step_with_axis_targets(&mut scene, &mut tool, &targets);
    assert!(
        (scene.positions[1] - before_release_step).length() > 0.0,
        "the formerly constrained fragment did not resume its solver velocity"
    );
    assert_eq!(scene.cluster_count, 2);
    assert_ne!(scene.clusters[0], scene.clusters[1]);
    assert!((scene.total_mass_kg() - initial_mass).abs() < 1e-12);
}

#[test]
fn bond_release_pipeline_conserves_momentum_and_ledgered_energy() {
    let iron = Material::component_id("iron").unwrap();
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.001, 1)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).with_velocity(Vec3::new(1.0, 0.2, 0.0)))
        .add_voxel(VoxelSpawn::at_cell([3, 0, 0], iron).with_velocity(Vec3::new(-0.5, -0.1, 0.0)))
        .build()
        .unwrap();
    assert!(scene
        .bonds
        .iter()
        .flatten()
        .all(|bond| bond.neighbor == EMPTY));
    assert_eq!(scene.cluster_count, 2);

    let mass_before = scene.total_mass_kg();
    let momentum_before = scene.voxels.iter().fold(Vec3::ZERO, |momentum, voxel| {
        momentum + voxel.mass * voxel.velocity
    });
    let kinetic_before: f64 = scene
        .voxels
        .iter()
        .map(|voxel| 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64)
        .sum();
    let released_energy = 0.25;
    scene.releases.push(voxel_physics_lab::fracture::Release {
        i: 0,
        j: 1,
        energy: released_energy,
        due: 0,
    });
    scene.energy.pending_release = released_energy;

    solver::step(&mut scene, &mut None);

    let momentum_after = scene.voxels.iter().fold(Vec3::ZERO, |momentum, voxel| {
        momentum + voxel.mass * voxel.velocity
    });
    let kinetic_after: f64 = scene
        .voxels
        .iter()
        .map(|voxel| 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64)
        .sum();
    assert!((momentum_after - momentum_before).length() < 1e-5);
    assert!(((kinetic_after - kinetic_before) - released_energy).abs() < 1e-5);
    assert_eq!(scene.energy.pending_release, 0.0);
    assert!((scene.energy.kinetic_release - released_energy).abs() < 1e-12);
    assert!((scene.total_mass_kg() - mass_before).abs() < 1e-12);
}

#[test]
fn fatigue_generated_fragments_collide_through_solver_contact() {
    let iron = Material::component_id("iron").unwrap();
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.001, 1)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
        .add_voxel(VoxelSpawn::at_cell([2, 0, 0], iron))
        .build()
        .unwrap();
    scene.config.fracture.fatigue_cycles = 30.0;
    scene.config.fracture.fatigue_exponent = 1.0;
    scene.config.fracture.fatigue_threshold = 0.1;
    scene.config.fracture.restitution = 0.25;
    scene.config.fracture.friction = 0.0;
    assert_eq!(scene.cluster_count, 1);
    assert_eq!(scene.bonds[0][0].neighbor, 1);
    assert_eq!(scene.bonds[0][1].neighbor, 2);

    let strength = scene.properties[0].tensile_strength;
    let young = scene.properties[0].young;
    let displacement = 0.8 * strength / young * CELL;
    let base = scene.positions[0];
    let end = scene.positions[2];
    let mut tool = None;
    let mut fractured = false;
    for step in 1..=1_100 {
        let phase = std::f32::consts::TAU * (step % 10) as f32 / 10.0;
        let targets = [
            base.to_array().map(Some),
            [None; 3],
            [
                Some(end.x + displacement * phase.sin()),
                Some(end.y),
                Some(end.z),
            ],
        ];
        solver::step_with_axis_targets(&mut scene, &mut tool, &targets);
        if scene.bonds[0][1].neighbor == EMPTY {
            fractured = true;
            break;
        }
    }
    assert!(fractured, "cyclic loading did not fracture the 3-REV chain");
    assert_eq!(scene.bonds[0][0].neighbor, EMPTY);
    assert_eq!(scene.cluster_count, 3);
    assert_ne!(scene.clusters[0], scene.clusters[1]);
    assert_ne!(scene.clusters[1], scene.clusters[2]);
    assert_ne!(scene.clusters[0], scene.clusters[2]);

    // Launch the newly separated REV back into the remaining piece of the
    // original specimen; contact response remains in solver::step.
    let impulse = scene.voxels[2].mass * 2.0;
    solver::apply_impulse(&mut scene, 2, Vec3::NEG_X * impulse).unwrap();
    let momentum_before = scene
        .voxels
        .iter()
        .fold(Vec3::ZERO, |sum, voxel| sum + voxel.velocity * voxel.mass);
    let mass_before = scene.total_mass_kg();
    let total_energy_before: f64 = scene
        .voxels
        .iter()
        .map(|voxel| {
            voxel.total_internal_energy_j()
                + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
        })
        .sum();
    let mut contact_heat_before = scene.energy.contact_restitution_heat;
    for _ in 0..8 {
        solver::step(&mut scene, &mut None);
        if scene.energy.contact_restitution_heat > contact_heat_before {
            break;
        }
        contact_heat_before = scene.energy.contact_restitution_heat;
    }

    assert!(
        scene.energy.contact_restitution_heat > 0.0,
        "fracture-generated pieces did not produce a restitutive contact"
    );
    let momentum_after = scene
        .voxels
        .iter()
        .fold(Vec3::ZERO, |sum, voxel| sum + voxel.velocity * voxel.mass);
    let total_energy_after: f64 = scene
        .voxels
        .iter()
        .map(|voxel| {
            voxel.total_internal_energy_j()
                + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
        })
        .sum();
    assert!((momentum_after - momentum_before).length() < 1e-5);
    assert!((scene.total_mass_kg() - mass_before).abs() < 1e-12);
    assert!((total_energy_after - total_energy_before).abs() < 1e-6);
    assert_eq!(scene.cluster_count, 3);
}
