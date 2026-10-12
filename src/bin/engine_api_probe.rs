use glam::Vec3;
use voxel_physics_lab::{
    material::Material,
    scene::{SceneBuilder, VoxelSpawn},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wood = Material::component_id("wood").ok_or("wood is not in the material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::new(0.0, 0.0, -9.81))
        .with_timestep(0.001, 4)
        .add_voxel(VoxelSpawn::new(Vec3::splat(0.005), wood).fixed())
        .add_voxel(VoxelSpawn::new(Vec3::new(0.015, 0.005, 0.005), wood))
        .build()?;

    scene.fix_voxel(0)?;
    println!(
        "custom scene: {} voxels, fixed[0]={}, total mass={:.6} kg",
        scene.voxels.len(),
        scene.voxels[0].is_kinematic(),
        scene.total_mass_kg()
    );

    let removed = scene.set_world_bounds(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.01, 0.1, 0.1))?;
    let ledger = scene.boundary_removals();
    println!(
        "world barrier: removed {removed} REV; remaining {}; removed mass={:.6} kg, escaped momentum={:?}",
        scene.voxels.len(),
        ledger.mass_kg,
        ledger.momentum_kg_m_s
    );
    Ok(())
}
