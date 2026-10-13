use glam::{Mat3, Quat, Vec3};
use voxel_physics_lab::{
    bond::EMPTY,
    scene::{Scene, SceneConfig},
    solver::{self, stiffness_matrix},
    tool::{Shape, Tool},
};

fn total_ke(scene: &Scene, tool: &Tool) -> f64 {
    let wood: f64 = scene
        .voxels
        .iter()
        .filter(|v| !v.kinematic)
        .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
        .sum();
    let inertia = tool.inverse_inertia.recip();
    let angular = 0.5
        * tool
            .angular_velocity
            .dot(tool.rotation * (inertia * (tool.rotation.conjugate() * tool.angular_velocity)))
            as f64;
    wood + 0.5 * tool.mass as f64 * tool.velocity.length_squared() as f64 + angular
}

fn elastic_energy(scene: &Scene) -> f64 {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let mut energy = 0.0f64;
    for axis in 0..3 {
        for i in 0..scene.voxels.len() {
            let bond = &scene.bonds[axis][i];
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            let q = scene.rotations[i].slerp(scene.rotations[j], 0.5);
            let frame = Mat3::from_cols(
                q * axes[axis],
                q * axes[(axis + 1) % 3],
                q * axes[(axis + 2) % 3],
            );
            let k = stiffness_matrix(scene, i, j, frame);
            let u = frame.transpose() * (scene.positions[j] - scene.positions[i])
                - Vec3::X * 0.01
                - Vec3::from_array(bond.plastic_strain) * 0.01;
            energy += (0.5 * (1.0 - bond.damage) * u.dot(k * u)).max(0.0) as f64;
        }
    }
    energy
}

fn run(bonded: bool, fractures: bool) {
    let mut c = SceneConfig::load("scenes/wood_chop.ron").unwrap();
    c.dimensions = [12, 12, 30];
    c.branch_dimensions = [0; 3];
    c.ground_dimensions = [0; 3];
    c.gravity = [0.; 3];
    c.fixed_dt = 0.001;
    c.substeps = 4;
    c.iterations = 120;
    c.corotation_iterations = 2;
    c.tool_dimensions = [10, 4, 14];
    let mut s = Scene::new(c);
    if !bonded {
        for axis in 0..3 {
            for b in &mut s.bonds[axis] {
                b.neighbor = EMPTY;
            }
        }
    }
    if !fractures {
        for m in &mut s.properties {
            m.tensile_strength = 1e20;
            m.tensile_strength_perp = 1e20;
            m.compressive_strength = 1e20;
            m.compressive_strength_perp = 1e20;
            m.shear_strength = 1e20;
            m.yield_strength = 1e20;
            m.failure_plastic_strain = 1e20;
        }
    }
    let mut t = Some(Tool::new(
        Shape::Blade,
        &s,
        Vec3::new(0.12, 0.0, 0.15),
        -Vec3::X,
        3.5,
    ));
    t.as_mut().unwrap().rotation = Quat::from_mat3(&Mat3::from_cols(-Vec3::X, Vec3::Z, Vec3::Y));
    let e0 = total_ke(&s, t.as_ref().unwrap());
    let label = match (bonded, fractures) {
        (false, _) => "free",
        (true, false) => "elastic",
        (true, true) => "fracturing",
    };
    println!("{label} start KE={e0:.6} J");
    for frame in 1..=50 {
        solver::step(&mut s, &mut t);
        if frame % 5 == 0 {
            println!(
                "{label} frame={frame} KE={:.6} J elastic={:.6} J released={:.6} J surface={:.6} J pending={:.6} J friction_heat={:.6} J restitution_heat={:.6} J broken={} residual={:.3e} vmax={:.3} m/s",
                total_ke(&s, t.as_ref().unwrap()),
                elastic_energy(&s),
                s.energy.kinetic_release,
                s.energy.surface_work,
                s.energy.pending_release,
                s.energy.contact_friction_heat,
                s.energy.contact_restitution_heat,
                s.broken,
                s.solver_residual,
                s.voxels.iter().filter(|v|!v.kinematic).map(|v|v.velocity.length()).fold(0.,f32::max)
            );
        }
    }
}
fn run_elastic_kick() {
    let mut c = SceneConfig::load("scenes/wood_chop.ron").unwrap();
    c.dimensions = [12, 12, 30];
    c.branch_dimensions = [0; 3];
    c.ground_dimensions = [0; 3];
    c.gravity = [0.; 3];
    c.fixed_dt = 0.001;
    c.substeps = 4;
    c.iterations = 120;
    c.corotation_iterations = 2;
    let mut s = Scene::new(c);
    for m in &mut s.properties {
        m.tensile_strength = 1e20;
        m.tensile_strength_perp = 1e20;
        m.compressive_strength = 1e20;
        m.compressive_strength_perp = 1e20;
        m.shear_strength = 1e20;
        m.yield_strength = 1e20;
        m.failure_plastic_strain = 1e20;
    }
    let i = 12 * 12 * 15 + 6 * 12 + 6;
    s.voxels[i].velocity = Vec3::X * 0.8;
    let e0 = 0.5 * s.voxels[i].mass as f64 * 0.8f64.powi(2);
    println!("elastic-kick start KE={e0:.9} J");
    for frame in 1..=50 {
        solver::step(&mut s, &mut None);
        if frame % 5 == 0 {
            let ke: f64 = s
                .voxels
                .iter()
                .filter(|v| !v.kinematic)
                .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
                .sum();
            println!("elastic-kick frame={frame} KE={ke:.9} J elastic={:.9} J total={:.9} J residual={:.3e}",elastic_energy(&s),ke+elastic_energy(&s),s.solver_residual);
        }
    }
}
fn main() {
    run(false, true);
    run_elastic_kick();
    run(true, false);
    run(true, true);
}
