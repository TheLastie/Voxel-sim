use glam::Vec3;
use voxel_physics_lab::{
    scene::{Scene, SceneConfig, CELL},
    solver,
    tool::{Shape, Tool},
};

fn scene(floor: bool) -> Scene {
    let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
    config.dimensions = [1; 3];
    config.branch_dimensions = [0; 3];
    config.ground_dimensions = if floor { [5, 5, 1] } else { [0; 3] };
    config.tool_dimensions = [2; 3];
    config.fixed_dt = 0.001;
    config.substeps = 2;
    Scene::new(config)
}

#[test]
fn persistent_axis_constraints_allow_gravity_and_floor_support() {
    let mut s = scene(true);
    let start = s.positions[0] + Vec3::Z * 0.03;
    solver::fix_voxel_at(&mut s, 0, start).unwrap();
    solver::fix_voxel_axis(&mut s, 0, 2, None).unwrap();
    solver::set_force(&mut s, 0, Vec3::new(100.0, -100.0, 0.0)).unwrap();
    let floor = s.positions[1..].to_vec();
    for _ in 0..250 {
        solver::step(&mut s, &mut None);
    }
    assert_eq!(s.positions[0].x, start.x);
    assert_eq!(s.positions[0].y, start.y);
    assert!(s.positions[0].z < start.z - 0.02);
    assert!((s.positions[0].z - CELL * 0.5).abs() < 2e-6);
    assert!(s.voxels[0].velocity.length() < 1e-4);
    assert_eq!(s.positions[1..], floor);
    solver::release_voxel(&mut s, 0).unwrap();
    solver::step(&mut s, &mut None);
    assert!(s.voxels[0].velocity.x > 0.0);
}

#[test]
fn constraint_inputs_are_atomic_and_impulses_respect_fixed_axes() {
    let mut s = scene(false);
    solver::fix_voxel(&mut s, 0, [true, false, true]).unwrap();
    let old = s.axis_targets[0];
    assert!(solver::fix_voxel_axes(&mut s, 0, [Some(1.0), Some(f32::NAN), None]).is_err());
    assert_eq!(s.axis_targets[0], old);
    assert!(solver::fix_voxel_axis(&mut s, 0, 3, None).is_err());
    assert!(solver::fix_voxel(&mut s, 100, [true; 3]).is_err());
    let mass = s.voxels[0].mass;
    solver::apply_impulse(&mut s, 0, Vec3::ONE * mass).unwrap();
    assert_eq!(s.voxels[0].velocity, Vec3::Y);
}

#[test]
fn axe_guide_carries_lateral_gravity_torque_and_contact_reactions() {
    let mut s = scene(false);
    let mut tool = Tool::new(Shape::Wedge, &s, Vec3::new(-0.05, 0.0, 0.005), Vec3::X, 1.0);
    let base = Vec3::new(-0.02, 0.0, 0.0);
    tool.mount_base(base, Vec3::X).unwrap();
    let mount = tool.mount.unwrap();
    tool.set_external_force(Vec3::new(0.0, 100.0, -100.0))
        .unwrap();
    tool.set_external_torque(Vec3::splat(100.0)).unwrap();
    let mut tool = Some(tool);
    // Fix the target so its reaction slows the sliding axe, instead of drifting away.
    solver::fix_voxel(&mut s, 0, [true; 3]).unwrap();
    for _ in 0..100 {
        solver::step(&mut s, &mut tool);
    }
    let t = tool.unwrap();
    let final_base = t.position + t.rotation * base;
    assert!((final_base.y - mount.world_base.y).abs() < 1e-7);
    assert!((final_base.z - mount.world_base.z).abs() < 1e-7);
    assert_eq!(t.rotation, mount.rotation);
    assert_eq!(t.angular_velocity, Vec3::ZERO);
    assert!(t.velocity.x.abs() < 0.01);
    assert_eq!(t.velocity.y, 0.0);
    assert_eq!(t.velocity.z, 0.0);
    assert!(t.position.x < -0.009);
}

#[test]
fn fully_fixed_base_stays_put_and_unmount_restores_gravity() {
    let mut s = scene(false);
    let mut t = Tool::new(Shape::Blade, &s, Vec3::splat(1.0), Vec3::X, 3.0);
    t.mount_base(Vec3::new(-0.01, 0.0, 0.0), Vec3::ZERO)
        .unwrap();
    let start = t.position;
    t.set_external_force(Vec3::splat(100.0)).unwrap();
    t.advance(&mut s, 0.05, &[]);
    assert_eq!(t.position, start);
    assert_eq!(t.velocity, Vec3::ZERO);
    t.clear_external_wrench();
    t.unmount();
    t.advance(&mut s, 0.01, &[]);
    assert!(t.position.z < start.z);
    assert!(t.velocity.z < -0.09);
}

#[test]
fn default_scene_has_explicit_floor_and_gravity() {
    let config = SceneConfig::load("scenes/mounted_axe.ron").unwrap();
    assert_eq!(config.gravity, [0.0, 0.0, -9.81]);
    let s = Scene::new(config);
    assert_eq!(s.voxels.iter().filter(|v| !v.is_kinematic()).count(), 4320);
    assert_eq!(s.voxels.iter().filter(|v| v.is_kinematic()).count(), 6400);
    assert!(s
        .positions
        .iter()
        .zip(&s.voxels)
        .filter(|(_, v)| v.is_kinematic())
        .all(|(p, _)| (p.z + CELL * 0.5).abs() < 1e-7));
}

#[test]
fn oblique_guide_preserves_its_line_and_projected_force() {
    let mut s = scene(false);
    s.config.gravity = [0.0; 3];
    let mut t = Tool::new(Shape::Wedge, &s, Vec3::splat(1.0), Vec3::X, 0.0);
    let direction = Vec3::new(1.0, 1.0, 0.0).normalize();
    let base = -Vec3::X * 0.01;
    t.mount_base(base, direction).unwrap();
    let mount = t.mount.unwrap();
    t.set_external_force(Vec3::X * t.mass * 4.0).unwrap();
    t.advance(&mut s, 0.05, &[]);
    assert!((t.velocity - Vec3::new(0.1, 0.1, 0.0)).length() < 1e-6);
    let displacement = t.position + t.rotation * base - mount.world_base;
    assert!(displacement.cross(direction).length() < 2e-7);
    let previous_mount = t.mount;
    assert!(t.mount_base(Vec3::splat(f32::NAN), direction).is_err());
    assert_eq!(
        t.mount.unwrap().world_base,
        previous_mount.unwrap().world_base
    );
}
