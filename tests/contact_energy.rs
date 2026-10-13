use glam::Vec3;
use voxel_physics_lab::{material::Material, scene::SceneBuilder, scene::VoxelSpawn, solver};

#[test]
fn normal_collision_converts_restitution_loss_into_internal_energy() {
    let iron = Material::component_id("iron").unwrap();
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.001, 1)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).with_velocity(Vec3::X * 0.1))
        .add_voxel(VoxelSpawn::at_cell([3, 0, 0], iron).with_velocity(Vec3::NEG_X * 0.1))
        .build()
        .unwrap();
    scene.config.fracture.friction = 0.0;
    scene.config.fracture.restitution = 0.25;

    let initial_mass = scene.total_mass_kg();
    let initial_momentum = total_momentum(&scene);
    let initial_total_energy = total_energy(&scene);
    assert!(scene
        .bonds
        .iter()
        .flatten()
        .all(|bond| bond.neighbor == u32::MAX));

    for _ in 0..150 {
        solver::step(&mut scene, &mut None);
    }

    assert!(scene.energy.contact_restitution_heat > 0.0);
    assert_eq!(scene.energy.contact_friction_heat, 0.0);
    assert!((total_momentum(&scene) - initial_momentum).length() < 1e-5);
    assert!((scene.total_mass_kg() - initial_mass).abs() < 1e-12);
    let energy_error = (total_energy(&scene) - initial_total_energy).abs();
    assert!(
        energy_error < initial_total_energy.abs() * 2e-5,
        "total energy error {energy_error:.8e} J; restitution heat {:.8e} J",
        scene.energy.contact_restitution_heat
    );
}

fn total_momentum(scene: &voxel_physics_lab::scene::Scene) -> Vec3 {
    scene
        .voxels
        .iter()
        .fold(Vec3::ZERO, |sum, voxel| sum + voxel.velocity * voxel.mass)
}

fn total_energy(scene: &voxel_physics_lab::scene::Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|voxel| {
            voxel.total_internal_energy_j()
                + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
        })
        .sum()
}
