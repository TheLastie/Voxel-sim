use glam::Vec3;
use voxel_physics_lab::{
    material::Material,
    scene::{CpuPreconditioner, Scene, SceneBuilder, VoxelSpawn},
    solver,
};

fn loaded_lattice(preconditioner: CpuPreconditioner) -> Scene {
    let iron = Material::component_id("iron").expect("iron is a built-in material");
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.001, 1)
        .with_solver_iterations(256)
        .with_cpu_preconditioner(preconditioner);

    for x in 0..3 {
        for y in 0..2 {
            for z in 0..2 {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }

    let mut scene = builder.build().expect("build iron lattice");
    // Anchor one whole face so the elastic system has no rigid-body modes.
    scene.fix_voxels(&[0, 1, 2, 3]).expect("fix lattice face");

    // Keep only one world axis constrained on a dynamic REV. The other two
    // axes must remain in the PCG system for both preconditioners.
    let partially_fixed = 7;
    let fixed_x = scene.positions[partially_fixed].x;
    solver::fix_voxel_axes(&mut scene, partially_fixed, [Some(fixed_x), None, None])
        .expect("fix one axis");
    // Keep stress below iron's failure strength while making the elastic
    // displacement large enough to survive f32 position rounding.
    solver::set_force(&mut scene, 11, Vec3::new(0.0, 3_000.0, 2_000.0)).expect("load lattice");
    scene
}

fn run(preconditioner: CpuPreconditioner) -> (Vec<Vec3>, f32, f32) {
    let mut scene = loaded_lattice(preconditioner);
    let fixed_x = scene.axis_targets[7][0].expect("X axis remains fixed");
    let loaded_start = scene.positions[11];
    for _ in 0..8 {
        solver::step(&mut scene, &mut None);
        assert!(scene.positions.iter().all(|p| p.is_finite()));
        assert!(scene.solver_residual.is_finite());
        assert!(
            scene.solver_residual <= scene.config.solver_relative_tolerance,
            "{preconditioner:?} residual {} exceeds tolerance {}",
            scene.solver_residual,
            scene.config.solver_relative_tolerance
        );
        assert_eq!(scene.positions[7].x, fixed_x);
    }
    let residual = scene.solver_residual;
    let loaded_displacement = scene.positions[11].distance(loaded_start);
    (scene.positions, residual, loaded_displacement)
}

#[test]
fn block_jacobi_and_ic0_solve_the_same_partially_constrained_scene() {
    let (ic_positions, ic_residual, _) = run(CpuPreconditioner::IncompleteCholesky0);
    let (jacobi_positions, jacobi_residual, loaded_displacement) =
        run(CpuPreconditioner::BlockJacobi);

    assert!(ic_residual <= 1e-5);
    assert!(jacobi_residual <= 1e-5);
    assert!(loaded_displacement > 0.0, "the applied force moved no REV");
    let max_position_delta = ic_positions
        .iter()
        .zip(jacobi_positions)
        .map(|(ic, jacobi)| ic.distance(jacobi))
        .fold(0.0_f32, f32::max);
    assert!(
        max_position_delta < 2e-5,
        "preconditioners disagree by {max_position_delta} m"
    );
}
