use glam::Vec3;
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{SceneBuilder, VoxelSpawn},
    scene_welding::ThermalBondingSettings,
};

#[test]
fn hot_contact_welds_locally_while_pressured_cold_contact_stays_separate() {
    let iron = Material::component_id("iron").expect("iron material is registered");
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.01, 2)
        // Each pair is driven into face contact. The gap between pairs keeps
        // their collision and connectivity neighborhoods independent.
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], iron)
                .with_velocity(Vec3::X * 0.01)
                .at_temperature(500.0),
        )
        .add_voxel(
            VoxelSpawn::at_cell([1, 0, 0], iron)
                .with_velocity(Vec3::NEG_X * 0.01)
                .at_temperature(500.0),
        )
        .add_voxel(
            VoxelSpawn::at_cell([4, 0, 0], iron).with_velocity(Vec3::X * 0.01),
        )
        .add_voxel(
            VoxelSpawn::at_cell([5, 0, 0], iron).with_velocity(Vec3::NEG_X * 0.01),
        )
        .build()
        .expect("scene builds");

    // Begin with a small physical overlap, matching an ordinary contact
    // state after broadphase detection while retaining lattice-adjacent rest
    // positions for the new face bond.
    for right in [1, 3] {
        scene.positions[right].x -= 0.005;
        scene.previous[right] = scene.positions[right];
    }
    // SceneBuilder creates the authored lattice's ordinary structural bonds.
    // This regression starts from two unbonded contacting faces, so sever the
    // two fixture links before exercising the contact-driven welding rule.
    scene.bonds[0][0].neighbor = EMPTY;
    scene.bonds[0][2].neighbor = EMPTY;
    voxel_physics_lab::solver::rebuild_clusters(&mut scene);
    ThermalBondingSettings {
        material_name: "iron".into(),
        minimum_mass_fraction: 1.0,
        minimum_temperature_kelvin: 350.0,
        minimum_impact_energy_j: 0.0,
        minimum_contact_pressure_pa: 0.0,
        minimum_dwell_seconds: 0.0,
    }
    .apply(&mut scene)
    .expect("thermal bonding settings apply");
    assert!(scene.temperature_kelvin(2).unwrap() < 350.0);
    assert!(scene.temperature_kelvin(3).unwrap() < 350.0);

    let initial_voxel_count = scene.voxels.len();
    for _ in 0..8 {
        scene.tick();
        if scene.bonds[0][0].neighbor == 1 {
            break;
        }
    }

    assert_eq!(scene.bonds[0][0].neighbor, 1, "the hot face pair bonds");
    assert_eq!(scene.bonds[0][2].neighbor, EMPTY, "cold face pair stays unbonded");
    assert_eq!(scene.voxels.len(), initial_voxel_count, "bonding keeps both REVs");
    assert_eq!(scene.cluster_count, 3, "only the hot pair becomes connected");
}
