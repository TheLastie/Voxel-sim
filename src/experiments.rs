//! Reproducible loading fixtures. They use the production solver, not tool rules.
use crate::{
    scene::{Scene, SceneConfig, AREA, CELL},
    solver,
    tool::{Shape, Tool},
};
use glam::{Mat3, Quat, Vec3};

pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}
fn wood_fragment_sizes(scene: &Scene) -> Vec<usize> {
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for (&cluster, voxel) in scene.clusters.iter().zip(&scene.voxels) {
        if !voxel.kinematic {
            *counts.entry(cluster).or_default() += 1;
        }
    }
    let mut sizes = counts.into_values().collect::<Vec<_>>();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}
fn blade_trailing_face_x(tool: &Tool) -> f32 {
    tool.position.x
        + tool
            .cells
            .iter()
            .map(|&p| (tool.rotation * p).x)
            .fold(f32::NEG_INFINITY, f32::max)
}
fn specimen(dimensions: [usize; 3], fiber: Vec3) -> Scene {
    let mut c = SceneConfig::load("scenes/wood_chop.ron").unwrap();
    c.dimensions = dimensions;
    c.branch_dimensions = [0; 3];
    c.ground_dimensions = [0; 3];
    c.fibers = fiber.to_array();
    c.gravity = [0.; 3];
    c.fixed_dt = 0.001;
    c.substeps = 1;
    c.iterations = 300;
    let mut s = Scene::new(c);
    for i in 0..s.positions.len() {
        s.positions[i] += Vec3::Z;
        s.previous[i] = s.positions[i];
        s.rest_positions[i] = s.positions[i];
    }
    s
}
fn reaction(s: &Scene, grip: &[bool]) -> f32 {
    let mut f = Vec3::ZERO;
    for axis in 0..3 {
        for i in 0..s.voxels.len() {
            let j = s.workspace[axis][i].rest_neighbor;
            if j == crate::bond::EMPTY {
                continue;
            }
            if grip[i] {
                f += s.workspace[axis][i].last_force;
            }
            if grip[j as usize] {
                f -= s.workspace[axis][i].last_force;
            }
        }
    }
    f.x.abs()
}
/// Pull/compress a 10³ sample along X, rotating fibers rather than fixture geometry.
fn axial(fiber: Vec3, sign: f32) -> (f32, usize) {
    let mut s = specimen([10, 10, 10], fiber);
    let min = s
        .rest_positions
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    let max = s
        .rest_positions
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let left: Vec<_> = s
        .rest_positions
        .iter()
        .map(|p| p.x < min + CELL * 0.1)
        .collect();
    let right: Vec<_> = s
        .rest_positions
        .iter()
        .map(|p| p.x > max - CELL * 0.1)
        .collect();
    let mut targets = vec![None; s.voxels.len()];
    let mut peak = 0f32;
    // Displacement controlled, slow monotonic ramp. Stop on first physical failure.
    for n in 1..=500 {
        let displacement = sign * (max - min) * 0.12 * n as f32 / 500.;
        for i in 0..targets.len() {
            targets[i] = if left[i] {
                Some(s.rest_positions[i])
            } else if right[i] {
                Some(s.rest_positions[i] + Vec3::X * displacement)
            } else {
                None
            };
        }
        solver::step_with_targets(&mut s, &mut None, &targets);
        peak = peak.max(reaction(&s, &right));
        if s.broken > 0 {
            return (peak, s.broken);
        }
    }
    (peak, 0)
}
pub fn tension() -> TestResult {
    let (parallel, b1) = axial(Vec3::X, 1.);
    let (perpendicular, b2) = axial(Vec3::Z, 1.);
    let ratio = parallel / perpendicular;
    TestResult{name:"tension_10x10x10".into(),passed:b1>0&&b2>0&&ratio>=10.,
        detail:format!("parallel={parallel:.3} N perpendicular={perpendicular:.3} N ratio={ratio:.3} first_breaks={b1}/{b2}")}
}
pub fn compression() -> TestResult {
    let (parallel, b1) = axial(Vec3::X, -1.);
    let (perpendicular, b2) = axial(Vec3::Z, -1.);
    let ratio = parallel / perpendicular;
    TestResult{name:"compression_10x10x10".into(),passed:b1>0&&b2>0&&ratio>=10.,
        detail:format!("parallel={parallel:.3} N perpendicular={perpendicular:.3} N ratio={ratio:.3} first_breaks={b1}/{b2}")}
}

/// Three-point bending: lower surface supports at the ends and a distributed
/// vertical load over a short central nose patch.
pub fn bending() -> TestResult {
    let mut s = specimen([30, 3, 3], Vec3::X);
    // A one-second static increment makes inertia negligible compared with bond stiffness.
    s.config.fixed_dt = 1.0;
    s.config.iterations = 500;
    let initial = s.rest_positions.clone();
    let min = initial.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let max = initial
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let middle = (min + max) * 0.5;
    let y_mid = initial.iter().map(|p| p.y).sum::<f32>() / initial.len() as f32;
    let z_min = initial.iter().map(|p| p.z).fold(f32::INFINITY, f32::min);
    let z_max = initial
        .iter()
        .map(|p| p.z)
        .fold(f32::NEG_INFINITY, f32::max);
    let y_min = initial.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let y_max = initial
        .iter()
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let closest = |x: f32| {
        (0..initial.len())
            .min_by(|&a, &b| {
                let da = (initial[a].x - x).abs()
                    + (initial[a].y - y_mid).abs()
                    + (initial[a].z - z_min).abs();
                let db = (initial[b].x - x).abs()
                    + (initial[b].y - y_mid).abs()
                    + (initial[b].z - z_min).abs();
                da.total_cmp(&db)
            })
            .unwrap()
    };
    let left_anchor = closest(min);
    let right_anchor = closest(max);
    let height = z_max - z_min + CELL;
    let support_nose_width = 9usize;
    let nose_weights: Vec<f32> = (0..support_nose_width)
        .map(|k| {
            (std::f32::consts::PI * (k as f32 + 0.5) / support_nose_width as f32)
                .sin()
                .powi(2)
        })
        .collect();
    let nose_weight_sum: f32 = nose_weights.iter().sum();
    let ny = 3.0;
    // Four times the nominal flexural strength allows for the ductile return map.
    let width = y_max - y_min + CELL;
    let max_moment = 4.0 * 80e6 * width * height * height / 6.0;
    let span = max - min;
    let max_total_force = 4.0 * max_moment / span;
    let max_force_per_node = max_total_force / (ny * nose_weight_sum);
    let center_index = ((middle - min) / CELL).round() as isize;
    let mut targets = vec![[None; 3]; s.voxels.len()];
    let mut loads = vec![Vec3::ZERO; s.voxels.len()];
    let mut max_cut = (0usize, 0usize);
    for n in 1..=500 {
        let force = max_force_per_node * n as f32 / 500.;
        for (i, &p) in initial.iter().enumerate() {
            targets[i] = [None; 3];
            loads[i] = Vec3::ZERO;
            let grid_x = ((p.x - min) / CELL).round() as usize;
            if (p.x - min).abs() < CELL * 0.1 || (p.x - max).abs() < CELL * 0.1 {
                targets[i][2] = Some(p.z);
            }
            if i == left_anchor {
                targets[i][0] = Some(p.x);
                targets[i][1] = Some(p.y);
                targets[i][2] = Some(p.z);
            }
            if i == right_anchor {
                targets[i][1] = Some(p.y);
            }
            let x_index = grid_x as isize;
            let nose_start = center_index - support_nose_width as isize / 2;
            if (nose_start..nose_start + support_nose_width as isize).contains(&x_index)
                && (p.z - z_max).abs() < CELL * 0.1
            {
                let w = nose_weights[(x_index - nose_start) as usize];
                loads[i].z = -force * w;
            }
        }
        solver::step_with_axis_targets_and_loads(&mut s, &mut None, &targets, &loads);
        // The force ramp is quasi-static; suppress residual inertial motion between solves.
        for voxel in &mut s.voxels {
            voxel.velocity = Vec3::ZERO;
        }
        let mut cuts = vec![0usize; 29];
        for i in 0..s.voxels.len() {
            if s.bonds[0][i].damage >= 1.0 {
                let plane = ((initial[i].x - min) / CELL).round() as usize;
                if plane < cuts.len() {
                    cuts[plane] += 1;
                }
            }
        }
        if let Some((plane, &count)) = cuts.iter().enumerate().max_by_key(|(_, n)| *n) {
            if count > max_cut.1 {
                max_cut = (plane, count);
            }
            let cut_x = min + (plane as f32 + 0.5) * CELL;
            let is_central = (cut_x - middle).abs() < span * 0.25;
            // Nine X-axis bonds cross each plane; require at least 75% to sever.
            if is_central && count * 4 >= 9 * 3 {
                return TestResult {
                    name: "bending_3x3x30".into(),
                    passed: true,
                    detail: format!(
                        "load={:.1} kN cut={count}/9 at x={cut_x:.3} m residual={:.3e}",
                        force * ny * nose_weight_sum / 1000.0,
                        s.solver_residual
                    ),
                };
            }
        }
    }
    let mut max_axial = 0.0f32;
    for i in 0..s.voxels.len() {
        let j = s.bonds[0][i].neighbor;
        if j == crate::bond::EMPTY {
            continue;
        }
        let j = j as usize;
        let q = s.rotations[i].slerp(s.rotations[j], 0.5);
        let frame = Mat3::from_cols(q * Vec3::X, q * Vec3::Y, q * Vec3::Z);
        let k = solver::stiffness_matrix(&s, i, j, frame);
        let u = frame.transpose() * (s.positions[j] - s.positions[i]) - Vec3::X * CELL;
        max_axial = max_axial.max((k * u / crate::scene::AREA).x.abs());
    }
    TestResult {
        name: "bending_3x3x30".into(),
        passed: false,
        detail: format!(
            "No central cross-section cut; max broken axial bonds in any plane={}/9 at x={:.3} m; max longitudinal bond stress={:.2} MPa",
            max_cut.1,
            min + (max_cut.0 as f32 + 0.5) * CELL,
            max_axial / 1e6
        ),
    }
}

/// A rigid wedge strikes the end grain of a compact Z-fiber wood specimen.
/// Geometry, iron composition, mass and contact impulses are shared with the app.
pub fn wedge_split() -> TestResult {
    let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
    config.dimensions = [12, 12, 30];
    config.branch_dimensions = [0; 3];
    config.ground_dimensions = [0; 3];
    config.fibers = Vec3::Z.to_array();
    config.gravity = [0.0, 0.0, -9.81];
    config.fixed_dt = 0.001;
    config.substeps = 4;
    config.iterations = 120;
    config.corotation_iterations = 2;
    config.tool_dimensions = [10, 2, 8];
    let mut scene = Scene::new(config);
    let bottom = scene
        .positions
        .iter()
        .map(|p| p.z)
        .fold(f32::INFINITY, f32::min);
    // Constrain only the rooted bottom layer vertically. The REV bonds remain
    // intact, so the trunk transfers the axe load into its support.
    let root_targets = scene
        .positions
        .iter()
        .map(|p| {
            if (p.z - bottom).abs() < CELL * 0.1 {
                [None, None, Some(p.z)]
            } else {
                [None; 3]
            }
        })
        .collect::<Vec<_>>();
    let mut tool = Some(Tool::new(
        Shape::Wedge,
        &scene,
        Vec3::new(0.0, 0.0, 0.38),
        -Vec3::Z,
        6.0,
    ));
    if let Some(tool) = &mut tool {
        tool.mount_base(Vec3::ZERO, -Vec3::Z)
            .expect("valid axe guide");
        tool.set_external_force(-Vec3::Z * 50.0)
            .expect("finite axe feed force");
    }
    let mut cut_step = None;
    let mut peak_speed = 0.0f32;
    for step in 1..=500 {
        if cut_step.is_some() {
            let speed = tool.as_ref().unwrap().velocity.z;
            let brake = if speed < -0.05 { 2_000.0 } else { 0.0 };
            tool.as_mut()
                .unwrap()
                .set_external_force(Vec3::Z * brake)
                .expect("finite axe brake");
        }
        solver::step_with_axis_targets(&mut scene, &mut tool, &root_targets);
        if cut_step.is_none() {
            let sizes = wood_fragment_sizes(&scene);
            if sizes.len() >= 2 && sizes[1] >= scene.voxels.len() / 10 {
                cut_step = Some(step);
            }
        }
        peak_speed = peak_speed.max(
            scene
                .voxels
                .iter()
                .map(|v| v.velocity.length())
                .fold(0.0, f32::max),
        );
        if cut_step.is_some_and(|cut| step >= cut + 100) {
            break;
        }
    }
    let mut sizes = wood_fragment_sizes(&scene);
    sizes.truncate(8);
    let passed = cut_step.is_some() && sizes.len() >= 2 && sizes[1] >= scene.voxels.len() / 10;
    TestResult {
        name: "wedge_split_12x12x30".into(),
        passed,
        detail: format!(
            "cut_step={cut_step:?} broken_bonds={} wood_clusters={} largest_fragment_sizes={sizes:?} peak_REV_speed={peak_speed:.3} m/s tool_z={:.3} m",
            scene.broken,
            scene.cluster_count,
            tool.as_ref().map_or(0.0, |t| t.position.z)
        ),
    }
}

/// The same iron impact path with a beveled blade loading Z-oriented wood fibers.
pub fn knife_cut() -> TestResult {
    let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
    config.dimensions = [12, 12, 30];
    config.branch_dimensions = [0; 3];
    config.ground_dimensions = [0; 3];
    config.fibers = Vec3::Z.to_array();
    config.gravity = [0.0; 3];
    config.fixed_dt = 0.001;
    config.substeps = 4;
    config.iterations = 120;
    config.corotation_iterations = 2;
    // The scene tool width sets a 4 cm wedge; Blade is a one-REV edge.
    // Match the blade-to-trunk proportion of the interactive 50 cm blade
    // against the 30 cm trunk: this reduced coupon uses a 20 cm blade.
    config.tool_dimensions = [20, 5, 14];
    config.fracture.energy_release = 0.005;
    config.fracture.damage_rate = 15.0;
    let mut scene = Scene::new(config);
    let mut tool = Some(Tool::new(
        Shape::Blade,
        &scene,
        Vec3::new(0.16, 0.0, 0.15),
        -Vec3::X,
        5.8,
    ));
    if let Some(tool) = &mut tool {
        // Local blade X advances through the trunk, Z spans its width, and Y
        // supplies the thin vertical cutting edge.
        tool.rotation = Quat::from_mat3(&Mat3::from_cols(-Vec3::X, Vec3::Z, Vec3::Y));
        // Drive the edge through the trunk on a straight guide. The previous
        // free impact bounced the blade off after making only a surface notch.
        // This is a force-driven cutter, not a prescribed fracture plane.
        tool.mount_base(Vec3::ZERO, -Vec3::X)
            .expect("valid blade guide");
        tool.set_external_force(-Vec3::X * 2_000.0)
            .expect("finite blade force");
    }
    let x_min = scene
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    let x_max = scene
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let y_min = scene
        .positions
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    let y_max = scene
        .positions
        .iter()
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let z_min = scene
        .positions
        .iter()
        .map(|p| p.z)
        .fold(f32::INFINITY, f32::min);
    let targets = scene
        .positions
        .iter()
        .map(|p| {
            let corner = (p.x - x_min).abs() < CELL * 0.1 || (p.x - x_max).abs() < CELL * 0.1;
            let side = (p.y - y_min).abs() < CELL * 0.1 || (p.y - y_max).abs() < CELL * 0.1;
            ((p.z - z_min).abs() < CELL * 0.1 && corner && side).then_some(*p)
        })
        .collect::<Vec<_>>();
    let specimen_x_min = scene
        .rest_positions
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    // Require the rear of the blade to clear the far face as well as two large
    // connected components. A contact shock that merely cracks the coupon is
    // not accepted as a through-cut.
    let mut severing_step = 0usize;
    for step in 1..=250 {
        solver::step_with_targets(&mut scene, &mut tool, &targets);
        severing_step = step;
        let sizes = wood_fragment_sizes(&scene);
        let blade_rear_x = blade_trailing_face_x(tool.as_ref().unwrap());
        if sizes.len() >= 2 && sizes[1] >= scene.voxels.len() / 10 && blade_rear_x < specimen_x_min
        {
            break;
        }
    }
    let sizes = wood_fragment_sizes(&scene);
    let blade_rear_x = blade_trailing_face_x(tool.as_ref().unwrap());
    let passed = blade_rear_x < specimen_x_min
        && sizes.len() >= 2
        && sizes[0] >= scene.voxels.len() / 10
        && sizes[1] >= scene.voxels.len() / 10;
    let mut largest_sizes = sizes;
    largest_sizes.truncate(8);
    TestResult {
        name: "knife_severs_trunk_12x12x30".into(),
        passed,
        detail: format!(
            "simulated_time_ms={} broken_bonds={} wood_clusters={} largest_fragment_sizes={:?} blade_rear_x={:.3} m specimen_far_face_x={:.3} m tool_position=({:.3},{:.3},{:.3}) m",
            severing_step,
            scene.broken,
            scene.cluster_count,
            largest_sizes,
            blade_rear_x,
            specimen_x_min,
            tool.as_ref().map_or(0.0, |t| t.position.x),
            tool.as_ref().map_or(0.0, |t| t.position.y),
            tool.as_ref().map_or(0.0, |t| t.position.z)
        ),
    }
}

/// Run the exact knife acceptance fixture and write sparse real solver states
/// for inspection and visualization. `F` rows are frame/tool diagnostics;
/// `V` rows are the corresponding voxel positions and damage state.
pub fn capture_knife(
    path: &str,
    stride: usize,
    speed: f32,
    frames: usize,
    attack_angle_degrees: f32,
    approach_seconds: f32,
    roll_degrees: f32,
    ramp_force_newtons: f32,
    ramp_seconds: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    capture_knife_internal(
        path,
        stride,
        speed,
        frames,
        attack_angle_degrees,
        approach_seconds,
        roll_degrees,
        ramp_force_newtons,
        ramp_seconds,
        None,
    )
}

/// Capture the reproducible force-driven severing fixture using the public
/// tool guide API. Feed stops and actively brakes the tool after full severing.
pub fn capture_guided_knife(
    path: &str,
    stride: usize,
    frames: usize,
    force_newtons: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    capture_knife_internal(
        path,
        stride,
        5.8,
        frames,
        0.0,
        0.0,
        0.0,
        force_newtons,
        0.001,
        Some(force_newtons),
    )
}

fn capture_knife_internal(
    path: &str,
    stride: usize,
    speed: f32,
    frames: usize,
    attack_angle_degrees: f32,
    approach_seconds: f32,
    roll_degrees: f32,
    ramp_force_newtons: f32,
    ramp_seconds: f32,
    guided_force_newtons: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        fs::File,
        io::{BufWriter, Write},
    };
    if !ramp_force_newtons.is_finite()
        || ramp_force_newtons < 0.0
        || !ramp_seconds.is_finite()
        || ramp_seconds <= 0.0
        || !approach_seconds.is_finite()
        || approach_seconds < 0.0
        || !speed.is_finite()
        || speed < 0.0
        || guided_force_newtons.is_some_and(|force| !force.is_finite() || force < 0.0)
    {
        return Err("capture requires finite nonnegative speed, approach and force, and positive ramp duration".into());
    }
    let mut config = SceneConfig::load("scenes/wood_chop.ron")?;
    config.dimensions = [12, 12, 30];
    config.branch_dimensions = [0; 3];
    config.ground_dimensions = [0; 3];
    config.fibers = Vec3::Z.to_array();
    config.fixed_dt = 0.001;
    config.substeps = 4;
    config.iterations = 120;
    config.corotation_iterations = 2;
    config.tool_dimensions = [20, 5, 14];
    config.fracture.energy_release = 0.005;
    config.fracture.damage_rate = 15.0;
    // Isolated contact fixture: gravity would drop a free blade by almost
    // five metres during a one-second horizontal approach.
    config.gravity = [0.0; 3];
    let mut scene = Scene::new(config);
    let specimen_x_max = scene
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut tool = Some(Tool::new(
        Shape::Blade,
        &scene,
        // Match knife_cut(): the leading face touches the specimen before impact.
        Vec3::new(0.16, 0.0, 0.15),
        -Vec3::X,
        speed,
    ));
    if let Some(t) = &mut tool {
        let base = Quat::from_mat3(&Mat3::from_cols(-Vec3::X, Vec3::Z, Vec3::Y));
        let pitch = Quat::from_rotation_y(attack_angle_degrees.to_radians());
        let roll = Quat::from_rotation_x(roll_degrees.to_radians());
        t.rotation = (pitch * roll * base).normalize();
        // The swing follows the pitched blade so a negative angle chops down
        // through the wood instead of merely rotating the silhouette.
        t.velocity = (pitch * -Vec3::X) * speed;
        let leading_extent = t
            .cells
            .iter()
            .map(|&p| -(t.rotation * p).x)
            .fold(0.0f32, f32::max);
        // `leading_extent` is measured between cell centers. Add one REV so
        // the blade edge starts at first contact instead of sharing centers
        // with the outer wood layer and being depenetrated before the strike.
        t.position.x = specimen_x_max + leading_extent + CELL;
        t.position -= t.velocity * approach_seconds;
        if let Some(force) = guided_force_newtons {
            t.mount_base(Vec3::ZERO, -Vec3::X)
                .map_err(std::io::Error::other)?;
            t.set_external_force(-Vec3::X * force)
                .map_err(std::io::Error::other)?;
        }
    }
    let x_min = scene
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    let x_max = scene
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let y_min = scene
        .positions
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    let y_max = scene
        .positions
        .iter()
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let z_min = scene
        .positions
        .iter()
        .map(|p| p.z)
        .fold(f32::INFINITY, f32::min);
    let targets = scene
        .positions
        .iter()
        .map(|p| {
            let corner = (p.x - x_min).abs() < CELL * 0.1 || (p.x - x_max).abs() < CELL * 0.1;
            let side = (p.y - y_min).abs() < CELL * 0.1 || (p.y - y_max).abs() < CELL * 0.1;
            ((p.z - z_min).abs() < CELL * 0.1 && corner && side).then_some(*p)
        })
        .collect::<Vec<_>>();
    let out = File::create(path)?;
    let mut out = BufWriter::new(out);
    let wall_start = std::time::Instant::now();
    writeln!(
        out,
        "H,2,{},{},{},{},{},{},{}",
        scene.config.fixed_dt,
        scene.config.dimensions[0],
        scene.config.dimensions[1],
        scene.config.dimensions[2],
        approach_seconds,
        roll_degrees,
        speed
    )?;
    for p in &tool.as_ref().unwrap().cells {
        writeln!(out, "T,{},{},{}", p.x, p.y, p.z)?;
    }
    let stride = stride.max(1);
    let drive_direction = Quat::from_rotation_y(attack_angle_degrees.to_radians()) * -Vec3::X;
    let mut drive_work = 0.0f64;
    let mut cut_frame = None;
    writeln!(
        out,
        "R,{ramp_force_newtons},{ramp_seconds},{approach_seconds}"
    )?;
    for frame in 0..=frames {
        if frame > 0 {
            // A real force through the blade COM, not a prescribed trajectory.
            // Midpoint sampling integrates a linear ramp without a frame bias.
            let time = (frame as f32 - 0.5) * scene.config.fixed_dt;
            let force = if let Some(force) = guided_force_newtons {
                if cut_frame.is_some() {
                    let feed_speed = tool.as_ref().unwrap().velocity.dot(drive_direction);
                    if feed_speed > 0.05 {
                        -force.max(2_000.0)
                    } else {
                        0.0
                    }
                } else {
                    force
                }
            } else {
                ramp_force_newtons * ((time - approach_seconds) / ramp_seconds).clamp(0.0, 1.0)
            };
            let before = tool.as_ref().unwrap().position;
            tool.as_mut()
                .unwrap()
                .set_external_force(drive_direction * force)?;
            solver::step_with_targets(&mut scene, &mut tool, &targets);
            drive_work +=
                (drive_direction * force).dot(tool.as_ref().unwrap().position - before) as f64;
            if guided_force_newtons.is_some() && cut_frame.is_none() {
                let sizes = wood_fragment_sizes(&scene);
                let rear_x = blade_trailing_face_x(tool.as_ref().unwrap());
                if sizes.len() >= 2 && sizes[1] >= scene.voxels.len() / 10 && rear_x < x_min {
                    cut_frame = Some(frame);
                    eprintln!(
                        "guided blade passed fully through coupon at {:.3}s: rear_x={:.3} m far_face_x={:.3} m broken={} clusters={} largest={:?}",
                        frame as f32 * scene.config.fixed_dt,
                        rear_x,
                        x_min,
                        scene.broken,
                        scene.cluster_count,
                        &sizes[..sizes.len().min(8)]
                    );
                }
            }
        }
        if frame % 100 == 0 {
            eprintln!("frame={frame}/{frames} sim={:.3}s wall={:.3}s broken={} clusters={} residual={:.3e}",
                frame as f32 * scene.config.fixed_dt, wall_start.elapsed().as_secs_f64(),
                scene.broken, scene.cluster_count, scene.solver_residual);
        }
        let impact_frame = (approach_seconds / scene.config.fixed_dt).round() as usize;
        let near_impact = frame >= impact_frame.saturating_sub(10) && frame <= impact_frame + 60;
        let near_cut =
            cut_frame.is_some_and(|cut| frame >= cut.saturating_sub(10) && frame <= cut + 60);
        if !near_impact && !near_cut && frame % stride != 0 && frame != frames {
            continue;
        }
        let t = tool.as_ref().unwrap();
        writeln!(out, "L,{frame},{},{drive_work}", t.external_force.length())?;
        writeln!(
            out,
            "Q,{frame},{},{},{},{}",
            t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
        )?;
        let mut elastic_energy = 0.0f64;
        for axis in 0..3 {
            for i in 0..scene.positions.len() {
                let b = &scene.bonds[axis][i];
                let original_j = scene.workspace[axis][i].rest_neighbor;
                if b.neighbor == crate::bond::EMPTY {
                    if original_j != crate::bond::EMPTY {
                        writeln!(out, "B,{frame},{i},{original_j}")?;
                    }
                    continue;
                }
                let j = b.neighbor as usize;
                let q = scene.rotations[i].slerp(scene.rotations[j], 0.5);
                let axes = [Vec3::X, Vec3::Y, Vec3::Z];
                let basis = Mat3::from_cols(
                    q * axes[axis],
                    q * axes[(axis + 1) % 3],
                    q * axes[(axis + 2) % 3],
                );
                let k = solver::stiffness_matrix(&scene, i, j, basis);
                let u = basis.transpose() * (scene.positions[j] - scene.positions[i])
                    - Vec3::X * CELL
                    - Vec3::from_array(b.plastic_strain) * CELL;
                elastic_energy += (0.5 * (1.0 - b.damage) * u.dot(k * u)).max(0.0) as f64;
            }
        }
        writeln!(
            out,
            "E,{frame},{},{elastic_energy},{}",
            frame as f32 * scene.config.fixed_dt,
            scene.solver_residual
        )?;
        let vmax = scene
            .voxels
            .iter()
            .filter(|v| !v.kinematic)
            .map(|v| v.velocity.length())
            .fold(0.0, f32::max);
        let wood_ke: f64 = scene
            .voxels
            .iter()
            .filter(|v| !v.kinematic)
            .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
            .sum();
        let tool_ke = 0.5 * t.mass as f64 * t.velocity.length_squared() as f64
            + 0.5
                * t.angular_velocity.dot(
                    t.rotation
                        * (t.inverse_inertia.recip()
                            * (t.rotation.conjugate() * t.angular_velocity)),
                ) as f64;
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        let mut max_stress_ratio = 0.0f32;
        let mut peak_stress = Vec3::ZERO;
        let mut peak_limits = [0.0f32; 3];
        for axis in 0..3 {
            for i in 0..scene.voxels.len() {
                let j = scene.bonds[axis][i].neighbor;
                if j == crate::bond::EMPTY {
                    continue;
                }
                let j = j as usize;
                let q = scene.rotations[i].slerp(scene.rotations[j], 0.5);
                let frame = Mat3::from_cols(
                    q * axes[axis],
                    q * axes[(axis + 1) % 3],
                    q * axes[(axis + 2) % 3],
                );
                let stress = frame.transpose() * scene.workspace[axis][i].last_force / AREA;
                let limits = solver::bond_limits(&scene, i, j, frame.x_axis);
                let normal = if stress.x >= 0.0 {
                    limits.tension
                } else {
                    limits.compression
                }
                .max(1.0);
                let ratio = ((stress.x / normal).powi(2)
                    + (stress.y * stress.y + stress.z * stress.z) / limits.shear.max(1.0).powi(2))
                .sqrt();
                if ratio > max_stress_ratio {
                    max_stress_ratio = ratio;
                    peak_stress = stress;
                    peak_limits = [limits.tension, limits.compression, limits.shear];
                }
            }
        }
        writeln!(
            out,
            "F,{frame},{tx},{ty},{tz},{vx},{vy},{vz},{ax},{ay},{az},{broken},{clusters},{vmax},{wood_ke},{tool_ke},{kinetic_release},{pending_release},{surface_work},{damage_heat},{plastic_heat},{stored_plastic},{max_stress_ratio},{stress_x},{stress_y},{stress_z},{tension_limit},{compression_limit},{shear_limit},{contact_friction_heat},{contact_restitution_heat}",
            tx = t.position.x,
            ty = t.position.y,
            tz = t.position.z,
            vx = t.velocity.x,
            vy = t.velocity.y,
            vz = t.velocity.z,
            ax = t.angular_velocity.x,
            ay = t.angular_velocity.y,
            az = t.angular_velocity.z,
            broken = scene.broken,
            clusters = scene.cluster_count,
            wood_ke = wood_ke,
            tool_ke = tool_ke,
            kinetic_release = scene.energy.kinetic_release,
            pending_release = scene.energy.pending_release,
            surface_work = scene.energy.surface_work,
            damage_heat = scene.energy.damage_heat,
            plastic_heat = scene.energy.plastic_heat,
            stored_plastic = scene.energy.stored_plastic,
            max_stress_ratio = max_stress_ratio,
            stress_x = peak_stress.x,
            stress_y = peak_stress.y,
            stress_z = peak_stress.z,
            tension_limit = peak_limits[0],
            compression_limit = peak_limits[1],
            shear_limit = peak_limits[2],
            contact_friction_heat = scene.energy.contact_friction_heat,
            contact_restitution_heat = scene.energy.contact_restitution_heat
        )?;
        for i in 0..scene.positions.len() {
            let mut d = 0.0f32;
            let mut lost = 0u8;
            for axis in 0..3 {
                let b = &scene.bonds[axis][i];
                if b.neighbor == crate::bond::EMPTY {
                    lost += 1;
                } else {
                    d = d.max(b.damage);
                }
            }
            writeln!(
                out,
                "V,{frame},{i},{},{},{},{d},{lost}",
                scene.positions[i].x, scene.positions[i].y, scene.positions[i].z
            )?;
        }
    }
    out.flush()?;
    eprintln!(
        "complete simulated={:.3}s wall={:.3}s",
        frames as f32 * scene.config.fixed_dt,
        wall_start.elapsed().as_secs_f64()
    );
    Ok(())
}
