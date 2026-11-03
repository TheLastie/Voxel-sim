use glam::Vec3;
use voxel_physics_lab::{scene::{Scene, SceneBuilder, VoxelSpawn}, material::Material};

fn kinetic(scene: &Scene) -> f64 {
    scene.voxels.iter().map(|v| 0.5*f64::from(v.mass)*v.velocity.as_dvec3().length_squared()).sum()
}
fn internal(scene: &Scene) -> f64 {
    scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum()
}
fn vibration(translation: Vec3) -> Scene {
    let iron=Material::component_id("iron").unwrap();
    let mut scene=SceneBuilder::standard().with_gravity(Vec3::ZERO).with_timestep(0.00001,1)
        .add_voxel(VoxelSpawn::at_cell([0,0,0],iron).with_velocity(translation+Vec3::X*0.01))
        .add_voxel(VoxelSpawn::at_cell([1,0,0],iron).with_velocity(translation-Vec3::X*0.01))
        .build().unwrap();
    scene.config.atmosphere.pressure_pa=0.0;
    scene
}

#[test]
fn free_elastic_numerical_damping_is_accounted_as_local_heat() {
    let mut scene=vibration(Vec3::ZERO);
    let initial_mechanical=kinetic(&scene)+scene.elastic_bond_energy_j();
    let initial_internal=internal(&scene);
    for _ in 0..12 { scene.tick(); }
    let remaining=kinetic(&scene)+scene.elastic_bond_energy_j();
    let heat=internal(&scene)-initial_internal;
    let error=(remaining+heat-initial_mechanical).abs();
    eprintln!("free vibration: mechanical {initial_mechanical:.9e}->{remaining:.9e} J heat={heat:.9e} error={error:.9e} J");
    assert!(remaining < initial_mechanical*0.5, "fixture must exercise implicit damping");
    assert!(heat>0.0, "dissipation must reach the voxel thermal state");
    assert!(error < initial_mechanical*0.001, "unaccounted elastic energy: {error:.9e} J");
    assert_eq!(scene.broken,0);
}

#[test]
fn rigid_translation_does_not_create_implicit_heat() {
    let mut scene=vibration(Vec3::ZERO);
    for voxel in &mut scene.voxels { voxel.velocity=Vec3::new(1.0,-0.5,0.25); }
    for _ in 0..12 { scene.tick(); }
    assert!(scene.energy.implicit_solver_heat<1.0e-12);
}

#[test]
fn unrelated_resting_component_receives_no_vibration_heat() {
    let iron=Material::component_id("iron").unwrap();
    let mut builder=SceneBuilder::standard().with_gravity(Vec3::ZERO).with_timestep(0.00001,1);
    for (x,vx) in [(0,0.01),(1,-0.01),(5,0.0),(6,0.0)] {
        builder=builder.add_voxel(VoxelSpawn::at_cell([x,0,0],iron).with_velocity(Vec3::X*vx));
    }
    let mut scene=builder.build().unwrap();
    scene.config.atmosphere.pressure_pa=0.0;
    let resting:Vec<_>=scene.positions.iter().enumerate().filter_map(|(i,p)|(p.x>0.04).then_some(i)).collect();
    let before:f64=resting.iter().map(|&i|scene.voxels[i].total_internal_energy_j()).sum();
    for _ in 0..12 {scene.tick();}
    let after:f64=resting.iter().map(|&i|scene.voxels[i].total_internal_energy_j()).sum();
    assert_eq!(after,before);
    assert!(scene.energy.implicit_solver_heat>0.0);
}

#[test]
fn simultaneous_remote_impact_does_not_redirect_free_vibration_heat() {
    let mut isolated=vibration(Vec3::ZERO);
    let initial_isolated=internal(&isolated);
    let iron=Material::component_id("iron").unwrap();
    let mut builder=SceneBuilder::standard().with_gravity(Vec3::ZERO).with_timestep(0.00001,1);
    for (x,vx) in [(0,0.01),(1,-0.01),(5,0.02),(6,-0.02)] {
        builder=builder.add_voxel(VoxelSpawn::at_cell([x,0,0],iron).with_velocity(Vec3::X*vx));
    }
    let mut together=builder.build().unwrap();
    together.config.atmosphere.pressure_pa=0.0;
    let first= together.positions.iter().position(|p|(p.x-0.055).abs()<1.0e-8).unwrap();
    together.bonds[0][first].neighbor=voxel_physics_lab::bond::EMPTY;
    voxel_physics_lab::solver::rebuild_clusters(&mut together);
    let vibrating:Vec<_>=together.positions.iter().enumerate().filter_map(|(i,p)|(p.x<0.025).then_some(i)).collect();
    let initial_together:f64=vibrating.iter().map(|&i|together.voxels[i].total_internal_energy_j()).sum();
    for _ in 0..12 {isolated.tick(); together.tick();}
    let heat_alone=internal(&isolated)-initial_isolated;
    let heat_together=vibrating.iter().map(|&i|together.voxels[i].total_internal_energy_j()).sum::<f64>()-initial_together;
    assert!((heat_alone-heat_together).abs() < heat_alone*0.001,
        "unrelated contact changed free vibration heat: {heat_alone:.9e} -> {heat_together:.9e} J");
}
