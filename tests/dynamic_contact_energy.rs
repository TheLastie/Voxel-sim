use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn, CELL},
};

const SIDE: usize = 4;
const DT: f32 = 1.0e-4;
const SPEED: f32 = 2.0;
const GAP: f64 = 0.01;
const STEPS: usize = 40;

struct ImpactSetup {
    scene: Scene,
    left: Vec<usize>,
    right: Vec<usize>,
}

fn make_scene(adaptive: bool) -> ImpactSetup {
    let iron = Material::component_id("iron").unwrap();
    let side = SIDE as i32;
    let gap_cells = (GAP / f64::from(CELL)).round() as i32;
    let right_start = side + gap_cells;
    let velocity = Vec3::new(SPEED, 0.0, 0.0);
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 4);
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([x, y, z], iron).with_velocity(velocity),
                );
            }
        }
    }
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([right_start + x, y, z], iron).with_velocity(-velocity),
                );
            }
        }
    }
    let mut scene = builder.build().unwrap();
    scene.config.atmosphere.pressure_pa = 0.0;

    // Classify the bodies from the constructed state instead of relying on
    // SceneBuilder preserving the spawn order.
    let mut xs: Vec<_> = scene.positions.iter().map(|p| p.x).collect();
    xs.sort_by(f64::total_cmp);
    let midpoint = (xs[SIDE.pow(3) - 1] + xs[SIDE.pow(3)]) * 0.5;
    let (left, right): (Vec<_>, Vec<_>) = scene
        .positions
        .iter()
        .enumerate()
        .partition(|(_, p)| p.x < midpoint);
    let left: Vec<_> = left.into_iter().map(|(i, _)| i).collect();
    let right: Vec<_> = right.into_iter().map(|(i, _)| i).collect();
    assert_eq!(left.len(), SIDE.pow(3));
    assert_eq!(right.len(), SIDE.pow(3));

    if adaptive {
        scene
            .register_rigid_aggregate_candidate(&left)
            .unwrap();
        scene
            .register_rigid_aggregate_candidate(&right)
            .unwrap();
        scene.set_adaptive_substeps(true);
    }
    ImpactSetup { scene, left, right }
}

fn momentum(scene: &Scene) -> DVec3 {
    scene.voxels.iter().fold(DVec3::ZERO, |sum, voxel| {
        sum + voxel.velocity.as_dvec3() * f64::from(voxel.mass)
    })
}

fn kinetic_energy_j(scene: &Scene) -> f64 {
    scene.voxels.iter().fold(0.0, |sum, voxel| {
        sum + 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared())
    })
}

fn internal_energy_j(scene: &Scene) -> f64 {
    scene.voxels.iter().map(|voxel| voxel.total_internal_energy_j()).sum()
}

fn center_velocity_x(scene: &Scene, members: &[usize]) -> f64 {
    let (px, mass) = members.iter().copied().fold((0.0, 0.0), |(px, mass), i| {
        let m = f64::from(scene.voxels[i].mass);
        (px + m * f64::from(scene.voxels[i].velocity.x), mass + m)
    });
    px / mass
}

fn x_surface_gap(scene: &Scene, left: &[usize], right: &[usize]) -> f64 {
    let left_max = left
        .iter()
        .map(|&i| scene.positions[i].x + f64::from(CELL) * 0.5)
        .fold(f64::NEG_INFINITY, f64::max);
    let right_min = right
        .iter()
        .map(|&i| scene.positions[i].x - f64::from(CELL) * 0.5)
        .fold(f64::INFINITY, f64::min);
    right_min - left_max
}

fn assert_impact_energy_is_bounded(adaptive: bool) {
    let ImpactSetup { mut scene, left, right } = make_scene(adaptive);
    let initial_momentum = momentum(&scene);
    let initial_kinetic = kinetic_energy_j(&scene);
    let initial_internal = internal_energy_j(&scene);
    let initial_elastic = scene.elastic_bond_energy_j();
    let initial_plastic = scene.stored_plastic_bond_energy_j();
    let initial_relative_vx = center_velocity_x(&scene, &left) - center_velocity_x(&scene, &right);
    let momentum_tolerance = (scene.total_mass_kg() * (2.0 * SPEED as f64) * 1.0e-4).max(1.0e-9);
    let energy_tolerance = (initial_kinetic * 0.10).max(1.0e-10);
    let initial_contact_heat =
        scene.energy.contact_friction_heat + scene.energy.contact_restitution_heat;
    let mut max_momentum_error = 0.0_f64;
    let mut minimum_gap = x_surface_gap(&scene, &left, &right);
    let mut max_energy_excess = f64::NEG_INFINITY;
    let mut max_energy_step = 0;

    assert!(initial_kinetic.is_finite() && initial_kinetic > 0.0);
    for step in 0..STEPS {
        scene.tick();
        assert!(
            scene.positions.iter().all(|p| p.is_finite())
                && scene.voxels.iter().all(|v| v.velocity.is_finite()),
            "nonfinite state in adaptive={adaptive} at step {}",
            step + 1
        );

        max_momentum_error = max_momentum_error.max((momentum(&scene) - initial_momentum).length());
        minimum_gap = minimum_gap.min(x_surface_gap(&scene, &left, &right));
        let accounted = kinetic_energy_j(&scene)
            + internal_energy_j(&scene)
            - initial_internal
            + scene.elastic_bond_energy_j()
            - initial_elastic
            + scene.stored_plastic_bond_energy_j()
            - initial_plastic
            + scene.energy.surface_work
            + scene.energy.pending_release;
        assert!(
            accounted.is_finite(),
            "nonfinite accounted energy in adaptive={adaptive} at step {}: {accounted}",
            step + 1
        );
        if accounted - initial_kinetic > max_energy_excess {
            max_energy_excess = accounted - initial_kinetic;
            max_energy_step = step + 1;
        }
    }

    let final_relative_vx = center_velocity_x(&scene, &left) - center_velocity_x(&scene, &right);
    let contact_heat = scene.energy.contact_friction_heat
        + scene.energy.contact_restitution_heat
        - initial_contact_heat;
    assert!(
        minimum_gap <= 1.0e-3
            && (final_relative_vx - initial_relative_vx).abs() > 1.0e-3
            && contact_heat > 0.0,
        "collision was not observed in adaptive={adaptive}: minimum_gap={minimum_gap:.6e} m relative_vx={initial_relative_vx:.6}->{final_relative_vx:.6} m/s contact_heat={contact_heat:.6e} J"
    );
    assert!(
        max_momentum_error <= momentum_tolerance,
        "momentum drift in adaptive={adaptive}: max_error={max_momentum_error:.9e} kg*m/s tolerance={momentum_tolerance:.9e} kg*m/s"
    );
    assert!(
        max_energy_excess <= energy_tolerance,
        "energy created in adaptive={adaptive} at step {max_energy_step}: input_kinetic={initial_kinetic:.9e} J max_excess={max_energy_excess:.9e} J tolerance={energy_tolerance:.9e} J"
    );
}

#[test]
fn fixed_step_iron_impact_conserves_momentum_and_does_not_create_energy() {
    assert_impact_energy_is_bounded(false);
}

#[test]
fn adaptive_step_iron_impact_conserves_momentum_and_does_not_create_energy() {
    assert_impact_energy_is_bounded(true);
}
