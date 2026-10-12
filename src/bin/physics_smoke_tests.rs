//! Fast, small-scale checks against the same solver used by the sandbox.
use glam::Vec3;
use voxel_physics_lab::{
    bond::EMPTY,
    scene::{Scene, SceneConfig, CELL},
    solver,
    tool::{Shape, Tool},
};

struct Check {
    name: &'static str,
    passed: bool,
    detail: String,
}

fn scene(dimensions: [usize; 3], ground: [usize; 3], gravity: Vec3) -> Scene {
    let mut config = SceneConfig::load("scenes/wood_chop.ron").expect("load base scene");
    config.dimensions = dimensions;
    config.branch_dimensions = [0; 3];
    config.ground_dimensions = ground;
    config.gravity = gravity.to_array();
    config.fixed_dt = 0.001;
    config.substeps = 1;
    config.iterations = 24;
    config.corotation_iterations = 1;
    Scene::new(config)
}

fn free_fall() -> Check {
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut s = scene([1, 1, 1], [0; 3], gravity);
    // Lift the REV so the ground plane does not enter this ballistic test.
    let start = s.positions[0];
    s.positions[0].z += 0.1;
    s.previous[0] = s.positions[0];
    s.rest_positions[0] = s.positions[0];
    let dt = s.config.fixed_dt;
    let frames = 100;
    let mut tool = None;
    for _ in 0..frames {
        solver::step(&mut s, &mut tool);
    }
    // The integrator applies gravity before advancing position (semi-implicit Euler).
    let expected = start.z + 0.1 + gravity.z * dt * dt * (frames * (frames + 1)) as f32 * 0.5;
    let actual = s.positions[0].z;
    let actual_velocity = s.voxels[0].velocity.z;
    let expected_velocity = gravity.z * dt * frames as f32;
    let error = (actual - expected).abs();
    Check {
        name: "free_fall_100ms",
        passed: error < 2e-6 && (actual_velocity - expected_velocity).abs() < 1e-5,
        detail: format!(
            "z={actual:.6} m expected={expected:.6} m error={error:.2e} m vz={actual_velocity:.6}/{expected_velocity:.6} m/s"
        ),
    }
}

fn ground_support() -> Check {
    let mut s = scene([1, 1, 1], [3, 3, 1], Vec3::new(0.0, 0.0, -9.81));
    let mut tool = None;
    for _ in 0..250 {
        solver::step(&mut s, &mut tool);
    }
    let bottom = s.positions[0].z - CELL * 0.5;
    let penetration = (s.ground_top - bottom).max(0.0);
    Check {
        name: "ground_support_250ms",
        passed: penetration <= 1e-6 && s.voxels[0].velocity.z.abs() < 1e-4,
        detail: format!(
            "bottom={bottom:.6} m ground={:.6} m penetration={penetration:.2e} m vz={:.4} m/s",
            s.ground_top, s.voxels[0].velocity.z
        ),
    }
}

fn constant_force() -> Check {
    let mut s = scene([1, 1, 1], [0; 3], Vec3::ZERO);
    let mass = s.voxels[0].mass;
    solver::set_force(&mut s, 0, Vec3::X * (10.0 * mass)).unwrap();
    let mut tool = None;
    for _ in 0..100 {
        solver::step(&mut s, &mut tool);
    }
    let expected = 10.0 * s.config.fixed_dt * 100.0;
    let actual = s.voxels[0].velocity.x;
    let error = (actual - expected).abs();
    Check {
        name: "constant_force_100ms",
        passed: error < 1e-4,
        detail: format!("vx={actual:.6} m/s expected={expected:.6} m/s error={error:.2e}"),
    }
}

fn head_on_collision() -> Check {
    let mut s = scene([2, 1, 1], [0; 3], Vec3::ZERO);
    s.bonds[0][0].neighbor = EMPTY;
    s.workspace[0][0].rest_neighbor = EMPTY;
    solver::rebuild_clusters(&mut s);
    solver::set_velocity(&mut s, 0, Vec3::X * 20.0).unwrap();
    solver::set_velocity(&mut s, 1, -Vec3::X * 20.0).unwrap();
    let momentum_before: f32 = s
        .voxels
        .iter()
        .filter(|v| !v.kinematic)
        .map(|v| v.mass * v.velocity.x)
        .sum();
    let mut tool = None;
    solver::step(&mut s, &mut tool);
    let momentum_after: f32 = s
        .voxels
        .iter()
        .filter(|v| !v.kinematic)
        .map(|v| v.mass * v.velocity.x)
        .sum();
    let gap = s.positions[1].x - s.positions[0].x;
    let separating_speed = s.voxels[1].velocity.x - s.voxels[0].velocity.x;
    let momentum_error = (momentum_after - momentum_before).abs();
    Check {
        name: "head_on_ccd_20mps",
        passed: gap >= CELL - 1e-6
            && separating_speed >= -1e-4
            && momentum_error <= momentum_before.abs().max(1e-5) * 1e-4,
        detail: format!(
            "gap={gap:.6} m relative_v={separating_speed:.3} m/s momentum_error={momentum_error:.2e} kg·m/s"
        ),
    }
}

fn kinetic_energy(s: &Scene, t: &Tool) -> f32 {
    let voxels: f32 = s
        .voxels
        .iter()
        .filter(|v| !v.kinematic)
        .map(|v| 0.5 * v.mass * v.velocity.length_squared())
        .sum();
    let inertia = t.inverse_inertia.recip();
    let rotation = 0.5
        * t.angular_velocity
            .dot(t.rotation * (inertia * (t.rotation.conjugate() * t.angular_velocity)));
    voxels + 0.5 * t.mass * t.velocity.length_squared() + rotation
}

fn rigid_tool_contact() -> Check {
    let mut s = scene([1, 1, 1], [0; 3], Vec3::ZERO);
    s.config.tool_dimensions = [2; 3];
    s.config.fracture.friction = 0.0;
    let mut tool = Some(Tool::new(
        Shape::Blade,
        &s,
        Vec3::new(0.0199, 0.005, 0.005),
        -Vec3::X,
        0.1,
    ));
    let energy_before = kinetic_energy(&s, tool.as_ref().unwrap());
    solver::step(&mut s, &mut tool);
    let t = tool.as_ref().unwrap();
    let energy_after = kinetic_energy(&s, t);
    let voxel = s.positions[0];
    let no_overlap = t
        .world_cells()
        .all(|center| (0..3).any(|axis| (center[axis] - voxel[axis]).abs() >= CELL - 2e-6));
    Check {
        name: "rigid_tool_contact",
        passed: no_overlap && energy_after <= energy_before * (1.0 + 1e-4),
        detail: format!(
            "tool_vx={:.4} m/s kinetic_energy={energy_before:.5e}→{energy_after:.5e} J overlap={}",
            t.velocity.x, !no_overlap
        ),
    }
}

fn main() {
    let checks = [
        free_fall(),
        ground_support(),
        constant_force(),
        head_on_collision(),
        rigid_tool_contact(),
    ];
    let mut failed = false;
    for c in checks {
        println!(
            "{} {} | {}",
            if c.passed { "PASS" } else { "FAIL" },
            c.name,
            c.detail
        );
        failed |= !c.passed;
    }
    if failed {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ballistic_motion_matches_integrator() {
        assert!(free_fall().passed);
    }

    #[test]
    fn ground_contact_prevents_sinking() {
        assert!(ground_support().passed);
    }

    #[test]
    fn constant_force_has_the_expected_mass_response() {
        assert!(constant_force().passed);
    }

    #[test]
    fn swept_contact_conserves_momentum_without_tunnelling() {
        assert!(head_on_collision().passed);
    }

    #[test]
    fn rigid_tool_stays_outside_voxel_without_energy_gain() {
        let result = rigid_tool_contact();
        assert!(result.passed, "{}", result.detail);
    }
}
