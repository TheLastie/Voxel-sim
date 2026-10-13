use crate::{
    bond::EMPTY,
    collision,
    scene::{Scene, AREA, CELL},
    tool::Tool,
};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;
use std::time::{Duration, Instant};

// Launching Rayon jobs for tiny scenes costs more than the work. Keep small
// scenes local to the calling thread; large active zones still use the pool.
const PARALLEL_MIN_VOXELS: usize = 32_768;

fn basis(axis: usize, rotation: Quat) -> Mat3 {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    Mat3::from_cols(
        rotation * axes[axis],
        rotation * axes[(axis + 1) % 3],
        rotation * axes[(axis + 2) % 3],
    )
}
/// Symmetric positive-definite 3×3 approximation in the corotated bond frame.
/// Transverse shear remains finite. This is not a full orthotropic continuum law.
fn tensor_compliance(scene: &Scene, n: usize, frame: Mat3) -> Mat3 {
    let material = &scene.properties[n];
    let solid = material
        .phase_state(scene.voxels[n].mass, scene.voxels[n].internal_energy)
        .solid_fraction
        .clamp(1e-4, 1.0);
    let micro = scene.voxels[n].microstructure;
    let fiber = frame.transpose() * (scene.rotations[n] * micro.fibers());
    let transverse = material.young + micro.anisotropy * (material.young_perp - material.young);
    // Directional axial modulus plus coupled anisotropic tangential response.
    let normal = transverse + (material.young - transverse) * fiber.x * fiber.x;
    let diagonal = Vec3::new(normal, material.shear, material.shear) * solid;
    if !diagonal.is_finite() || diagonal.min_element() <= 0.0 {
        return Mat3::ZERO;
    }
    // Sherman–Morrison inverse of diag(D) + alpha * fiber * fiberᵀ.
    let inverse_diagonal = diagonal.recip();
    let scaled_fiber = fiber * inverse_diagonal;
    let alpha = material.shear * micro.anisotropy * 0.2;
    let rank_one = Mat3::from_cols(
        scaled_fiber * scaled_fiber.x,
        scaled_fiber * scaled_fiber.y,
        scaled_fiber * scaled_fiber.z,
    );
    Mat3::from_diagonal(inverse_diagonal)
        - rank_one * (alpha / (1.0 + alpha * fiber.dot(scaled_fiber)))
}

pub fn stiffness_matrix(scene: &Scene, i: usize, j: usize, frame: Mat3) -> Mat3 {
    let a_compliance = tensor_compliance(scene, i, frame);
    let b_compliance = tensor_compliance(scene, j, frame);
    if a_compliance == Mat3::ZERO || b_compliance == Mat3::ZERO {
        return Mat3::ZERO;
    }
    // Equal half-lengths in series; matrix harmonic mean, not entrywise average.
    (a_compliance + b_compliance).inverse() * (2.0 * AREA / CELL)
}
pub fn bond_limits(scene: &Scene, i: usize, j: usize, direction: Vec3) -> crate::fracture::Limits {
    let alignment =
        [i, j].map(|n| direction.dot(scene.rotations[n] * scene.voxels[n].microstructure.fibers()));
    let mut limits = crate::fracture::pair_limits(
        &scene.properties[i],
        &scene.properties[j],
        alignment,
        [
            scene.voxels[i].microstructure.anisotropy,
            scene.voxels[j].microstructure.anisotropy,
        ],
    );
    let solid = [i, j]
        .map(|n| {
            scene.properties[n]
                .phase_state(scene.voxels[n].mass, scene.voxels[n].internal_energy)
                .solid_fraction
                .clamp(0.0, 1.0)
        })
        .into_iter()
        .fold(1.0f32, f32::min);
    let strength_scale = solid * solid;
    limits.tension *= strength_scale;
    limits.compression *= strength_scale;
    limits.shear *= strength_scale;
    limits.yield_stress *= strength_scale;
    limits.hardening *= strength_scale;
    limits.fracture_energy *= strength_scale;
    limits
}
pub fn fracture_strength(scene: &Scene, i: usize, j: usize, direction: Vec3) -> f32 {
    bond_limits(scene, i, j, direction).tension
}

/// Set a world-space force that remains active until changed or cleared.
pub fn set_force(scene: &mut Scene, voxel: usize, force: Vec3) -> Result<(), &'static str> {
    let body = scene.voxels.get(voxel).ok_or("voxel index out of range")?;
    if body.kinematic {
        return Err("cannot apply a force to a fixed voxel");
    }
    if !force.is_finite() {
        return Err("force must be finite");
    }
    scene.external_forces[voxel] = force;
    Ok(())
}

/// Add to a persistent force. Repeated calls accumulate until cleared.
pub fn add_force(scene: &mut Scene, voxel: usize, force: Vec3) -> Result<(), &'static str> {
    let current = *scene
        .external_forces
        .get(voxel)
        .ok_or("voxel index out of range")?;
    set_force(scene, voxel, current + force)
}

/// Set a persistent acceleration input. It is converted to force using the
/// voxel's current mass on every solver step, so composition changes affect it.
pub fn set_acceleration(
    scene: &mut Scene,
    voxel: usize,
    acceleration: Vec3,
) -> Result<(), &'static str> {
    let body = scene.voxels.get(voxel).ok_or("voxel index out of range")?;
    let mass = body.mass;
    if body.kinematic {
        return Err("cannot accelerate a fixed voxel");
    }
    if !acceleration.is_finite() {
        return Err("acceleration must be finite");
    }
    if !(acceleration * mass).is_finite() {
        return Err("acceleration is too large for this voxel's mass");
    }
    scene.external_accelerations[voxel] = acceleration;
    Ok(())
}

/// Set a voxel's initial or current world-space velocity in m/s.
pub fn set_velocity(scene: &mut Scene, voxel: usize, velocity: Vec3) -> Result<(), &'static str> {
    let body = scene
        .voxels
        .get_mut(voxel)
        .ok_or("voxel index out of range")?;
    if body.kinematic {
        return Err("cannot set velocity on a fixed voxel");
    }
    if !velocity.is_finite() {
        return Err("velocity must be finite");
    }
    body.velocity = Vec3::from_array(std::array::from_fn(|a| {
        if scene.axis_targets[voxel][a].is_some() {
            0.0
        } else {
            velocity[a]
        }
    }));
    Ok(())
}

/// Apply a one-time impulse in N·s, conserving the corresponding momentum.
pub fn apply_impulse(scene: &mut Scene, voxel: usize, impulse: Vec3) -> Result<(), &'static str> {
    let body = scene
        .voxels
        .get_mut(voxel)
        .ok_or("voxel index out of range")?;
    if body.kinematic {
        return Err("cannot impulse a fixed voxel");
    }
    if !impulse.is_finite() {
        return Err("impulse must be finite");
    }
    let velocity = body.velocity + impulse / body.mass;
    if !velocity.is_finite() {
        return Err("impulse would produce a non-finite velocity");
    }
    body.velocity = Vec3::from_array(std::array::from_fn(|a| {
        if scene.axis_targets[voxel][a].is_some() {
            0.0
        } else {
            velocity[a]
        }
    }));
    Ok(())
}

/// Clear persistent force and acceleration inputs for one voxel.
pub fn clear_applied_actions(scene: &mut Scene, voxel: usize) -> Result<(), &'static str> {
    if voxel >= scene.voxels.len() {
        return Err("voxel index out of range");
    }
    scene.external_forces[voxel] = Vec3::ZERO;
    scene.external_accelerations[voxel] = Vec3::ZERO;
    Ok(())
}

/// Fix selected world axes at specified coordinates (metres). None releases an axis.
/// Coordinates are imposed immediately; the remaining axes retain mass and motion.
pub fn fix_voxel_axes(
    scene: &mut Scene,
    voxel: usize,
    coordinates: [Option<f32>; 3],
) -> Result<(), &'static str> {
    if voxel >= scene.voxels.len() {
        return Err("voxel index out of range");
    }
    if coordinates.iter().flatten().any(|p| !p.is_finite()) {
        return Err("coordinates must be finite");
    }
    scene.axis_targets[voxel] = coordinates;
    for axis in 0..3 {
        if let Some(p) = coordinates[axis] {
            scene.positions[voxel][axis] = p;
            scene.previous[voxel][axis] = p;
            scene.voxels[voxel].velocity[axis] = 0.0;
        }
    }
    Ok(())
}
/// Fix all axes at a world coordinate.
pub fn fix_voxel_at(scene: &mut Scene, voxel: usize, coordinate: Vec3) -> Result<(), &'static str> {
    fix_voxel_axes(scene, voxel, coordinate.to_array().map(Some))
}
/// Fix selected axes at their current coordinates; false leaves an axis free.
pub fn fix_voxel(scene: &mut Scene, voxel: usize, axes: [bool; 3]) -> Result<(), &'static str> {
    let p = *scene
        .positions
        .get(voxel)
        .ok_or("voxel index out of range")?;
    fix_voxel_axes(
        scene,
        voxel,
        std::array::from_fn(|a| axes[a].then_some(p[a])),
    )
}
/// Change one constraint without changing the other axes. None releases it.
pub fn fix_voxel_axis(
    scene: &mut Scene,
    voxel: usize,
    axis: usize,
    coordinate: Option<f32>,
) -> Result<(), &'static str> {
    if axis >= 3 {
        return Err("axis must be 0 (X), 1 (Y), or 2 (Z)");
    }
    let mut targets = *scene
        .axis_targets
        .get(voxel)
        .ok_or("voxel index out of range")?;
    targets[axis] = coordinate;
    fix_voxel_axes(scene, voxel, targets)
}
pub fn release_voxel(scene: &mut Scene, voxel: usize) -> Result<(), &'static str> {
    fix_voxel_axes(scene, voxel, [None; 3])
}

/// Local polar fits make the vector constraints invariant to rigid rotation.
/// Returns the largest quaternion angle changed by this fit.
fn fit_rotations(scene: &mut Scene, scratch: &mut SolverScratch) -> f32 {
    let n = scene.voxels.len();
    scratch.rotation_gradients.resize(n, Mat3::ZERO);
    scratch.rotation_gradients.fill(Mat3::ZERO);
    scratch.rotation_counts.resize(n, Vec3::ZERO);
    scratch.rotation_counts.fill(Vec3::ZERO);
    for axis in 0..3 {
        for i in 0..n {
            let j = scene.bonds[axis][i].neighbor;
            if j == EMPTY {
                continue;
            }
            let j = j as usize;
            let d = (scene.positions[j] - scene.positions[i]) / CELL;
            let mut m = Mat3::ZERO;
            match axis {
                0 => m.x_axis = d,
                1 => m.y_axis = d,
                _ => m.z_axis = d,
            };
            if !scene.voxels[i].kinematic {
                scratch.rotation_gradients[i] += m;
                scratch.rotation_counts[i][axis] += 1.0;
            }
            if !scene.voxels[j].kinematic {
                scratch.rotation_gradients[j] += m;
                scratch.rotation_counts[j][axis] += 1.0;
            }
        }
    }
    let gradients = &scratch.rotation_gradients;
    let counts = &scratch.rotation_counts;
    let fixed = &scene.voxels;
    let update_rotation = |(i, q): (usize, &mut Quat)| {
        if fixed[i].kinematic {
            return 0.0;
        }
        let previous = *q;
        let old = Mat3::from_quat(*q);
        let mut f = gradients[i];
        for axis in 0..3 {
            let column = if counts[i][axis] > 0.0 {
                f.col(axis) / counts[i][axis]
            } else {
                old.col(axis)
            };
            match axis {
                0 => f.x_axis = column,
                1 => f.y_axis = column,
                _ => f.z_axis = column,
            };
        }
        if f.determinant() <= 1e-6 {
            return 0.0;
        }
        for _ in 0..5 {
            let next = (f + f.inverse().transpose()) * 0.5;
            let delta = (next.x_axis - f.x_axis)
                .abs()
                .max_element()
                .max((next.y_axis - f.y_axis).abs().max_element())
                .max((next.z_axis - f.z_axis).abs().max_element());
            f = next;
            if delta <= 1e-6 {
                break;
            }
        }
        // Gram-Schmidt removes residual stretch and keeps a right-handed basis.
        let x = f.x_axis.normalize_or_zero();
        let y = (f.y_axis - x * x.dot(f.y_axis)).normalize_or_zero();
        if x.length_squared() > 0.5 && y.length_squared() > 0.5 {
            *q = Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y))).normalize();
        }
        2.0 * previous.dot(*q).abs().clamp(0.0, 1.0).acos()
    };
    if n < PARALLEL_MIN_VOXELS {
        scene
            .rotations
            .iter_mut()
            .enumerate()
            .map(update_rotation)
            .fold(0.0, f32::max)
    } else {
        scene
            .rotations
            .par_iter_mut()
            .enumerate()
            .map(update_rotation)
            .reduce(|| 0.0, f32::max)
    }
}

pub fn step(scene: &mut Scene, tool: &mut Option<Tool>) {
    step_with_targets(scene, tool, &[]);
}
pub fn step_with_targets(scene: &mut Scene, tool: &mut Option<Tool>, targets: &[Option<Vec3>]) {
    let axes: Vec<_> = targets
        .iter()
        .map(|p| p.map(|v| v.to_array().map(Some)).unwrap_or([None; 3]))
        .collect();
    step_with_axis_targets(scene, tool, &axes);
}
pub fn step_with_axis_targets(
    scene: &mut Scene,
    tool: &mut Option<Tool>,
    targets: &[[Option<f32>; 3]],
) {
    step_with_axis_targets_and_loads(scene, tool, targets, &[]);
}

/// Advance with optional per-step world-space nodal forces in newtons, in
/// addition to persistent inputs stored on `Scene`. All enter the same coupled
/// solve as bonds and are shared by every substep.
pub fn step_with_axis_targets_and_loads(
    scene: &mut Scene,
    tool: &mut Option<Tool>,
    targets: &[[Option<f32>; 3]],
    loads: &[Vec3],
) {
    assert!(targets.is_empty() || targets.len() == scene.voxels.len());
    assert!(loads.is_empty() || loads.len() == scene.voxels.len());
    // Explicit per-step coordinates override persistent constraints on that axis.
    let merged_targets: Vec<_> = scene
        .axis_targets
        .iter()
        .enumerate()
        .map(|(i, fixed)| std::array::from_fn(|a| targets.get(i).and_then(|t| t[a]).or(fixed[a])))
        .collect();
    assert!(merged_targets
        .iter()
        .flatten()
        .flatten()
        .all(|p| p.is_finite()));
    let targets = merged_targets.as_slice();
    let applied_loads: Vec<_> = scene
        .voxels
        .iter()
        .enumerate()
        .map(|(i, voxel)| {
            if !voxel.kinematic {
                scene.external_forces[i]
                    + scene.external_accelerations[i] * voxel.mass
                    + loads.get(i).copied().unwrap_or(Vec3::ZERO)
            } else {
                Vec3::ZERO
            }
        })
        .collect();
    let h = scene.config.fixed_dt / scene.config.substeps as f32;
    let gravity = Vec3::from_array(scene.config.gravity);
    let mut topology_changed = break_fully_melted_bonds(scene);
    let profile_step = std::env::var_os("VOXEL_PROFILE_STEP").is_some();
    let mut stage_time = [Duration::ZERO; 5];
    let mut total_pcg_iterations = 0usize;
    let mut solve_detail_time = [Duration::ZERO; 3];
    // Reuse large solver buffers across substeps and corotation corrections.
    let mut solver_scratch = std::mem::take(&mut scene.solver_scratch);
    for _ in 0..scene.config.substeps {
        let mut phase_start = Instant::now();
        scene.previous.clone_from(&scene.positions);
        let mut drift_velocity = vec![Vec3::ZERO; scene.voxels.len()];
        for (i, (p, v)) in scene
            .positions
            .iter_mut()
            .zip(&mut scene.voxels)
            .enumerate()
        {
            if !v.kinematic {
                v.velocity += gravity * h;
                for a in 0..3 {
                    if targets[i][a].is_some() {
                        v.velocity[a] = 0.0;
                    }
                }
                drift_velocity[i] = v.velocity;
                *p += v.velocity * h;
            }
        }
        for (i, target) in targets.iter().enumerate() {
            for axis in 0..3 {
                if let Some(p) = target[axis] {
                    scene.positions[i][axis] = p;
                }
            }
        }
        for work in &mut scene.workspace {
            for w in work {
                w.lambda = Vec3::ZERO;
            }
        }
        let contact_velocity = if let Some(t) = tool {
            t.advance(scene, h, targets)
        } else {
            vec![Vec3::ZERO; scene.positions.len()]
        };
        // Treat a contact impulse as acting near the middle of this substep.
        // Without this half-step drift, the new velocity is only integrated
        // on the next substep, after the implicit bond solve has already
        // damped the impact. That delayed stress made a sharp tool stop at the
        // surface without loading bonds enough to fail.
        for (p, &delta_v) in scene.positions.iter_mut().zip(&contact_velocity) {
            *p += delta_v * (0.5 * h);
        }
        let predicted = scene.positions.clone();
        fit_rotations(scene, &mut solver_scratch);
        stage_time[0] += phase_start.elapsed();
        phase_start = Instant::now();
        for iteration in 0..scene.config.corotation_iterations {
            let (iterations, detail_time) = solve_coupled(
                scene,
                h,
                targets,
                &applied_loads,
                &predicted,
                &mut solver_scratch,
                iteration > 0,
            );
            total_pcg_iterations += iterations;
            for index in 0..solve_detail_time.len() {
                solve_detail_time[index] += detail_time[index];
            }
            if iteration + 1 == scene.config.corotation_iterations {
                break;
            }
            let orientation_change = fit_rotations(scene, &mut solver_scratch);
            if orientation_change <= scene.config.corotation_tolerance
                && scene.solver_residual <= scene.config.solver_relative_tolerance
            {
                break;
            }
        }
        stage_time[1] += phase_start.elapsed();
        phase_start = Instant::now();
        // Preserve the accurately integrated drift velocity. Recover only the
        // velocity correction from the much smaller constraint displacement;
        // subtracting two large f32 positions every step otherwise accumulates
        // measurable kinetic error in simple free fall.
        let solved_positions = scene.positions.clone();
        let contacts = collision::project(scene, targets);
        stage_time[2] += phase_start.elapsed();
        phase_start = Instant::now();
        // Actual converged strain defines elastic energy and constitutive stress.
        // Reading an intermediate lambda would depend on sweep order and convergence.
        for constraint in &solver_scratch.constraints {
            let i = constraint.i;
            let j = constraint.j;
            let axis = constraint.axis;
            if scene.bonds[axis][i].neighbor == EMPTY || scene.voxels[i].kinematic {
                continue;
            }
            let frame = constraint.frame;
            let k = constraint.material_k;
            let limits = bond_limits(scene, i, j, frame.x_axis);
            let u = frame.transpose() * (scene.positions[j] - scene.positions[i]) - Vec3::X * CELL;
            scene.workspace[axis][i].last_force = frame
                * (k * (u - Vec3::from_array(scene.bonds[axis][i].plastic_strain) * CELL))
                * (1.0 - scene.bonds[axis][i].damage);
            let dissipation = crate::fracture::update(
                &mut scene.bonds[axis][i],
                u,
                k,
                limits,
                h,
                &scene.config.fracture,
            );
            scene.energy.record(dissipation);
            let heat = ((dissipation.plastic_heat + dissipation.damage_heat) * 0.5) as f32;
            scene.voxels[i].internal_energy += heat;
            scene.voxels[j].internal_energy += heat;
            if dissipation.released > 0.0 {
                scene.releases.push(crate::fracture::Release {
                    i,
                    j,
                    energy: dissipation.released,
                    due: scene.substep_index + scene.config.fracture.propagation_delay,
                });
            }
            if scene.bonds[axis][i].neighbor == EMPTY {
                scene.broken += 1;
                topology_changed = true;
            }
        }
        stage_time[3] += phase_start.elapsed();
        phase_start = Instant::now();
        for i in 0..scene.voxels.len() {
            if !scene.voxels[i].kinematic {
                let constraint_velocity = (solved_positions[i] - predicted[i]) / h;
                for axis in 0..3 {
                    if targets.get(i).and_then(|target| target[axis]).is_some() {
                        scene.voxels[i].velocity[axis] =
                            (scene.positions[i][axis] - scene.previous[i][axis]) / h;
                    } else {
                        scene.voxels[i].velocity[axis] = drift_velocity[i][axis]
                            + constraint_velocity[axis]
                            + contact_velocity[i][axis];
                    }
                }
            }
        }
        collision::resolve_velocities(scene, &contacts, targets);
        if let Some(t) = tool {
            // Structural correction can move material back into the blade.
            // Run a zero-time geometric/impulse pass before leaving the substep.
            let _ = t.advance(scene, 0.0, targets);
        }
        apply_releases(scene, targets);
        if let Err(reason) = scene.conduct_heat(h) {
            eprintln!("thermal transport skipped: {reason}");
        }
        scene.substep_index += 1;
        stage_time[4] += phase_start.elapsed();
    }
    if topology_changed {
        rebuild_clusters(scene);
    }
    scene.solver_scratch = solver_scratch;
    // World exits are a topology event at the fixed-step boundary. This keeps
    // every solver/contact array in one consistent index space for the step.
    scene.enforce_world_bounds();
    if profile_step {
        let ms = stage_time.map(|duration| duration.as_secs_f64() * 1000.0);
        eprintln!(
            "profile step ms: predict/tool={:.2} solve={:.2} [assemble={:.2} factor={:.2} pcg={:.2}] collision={:.2} fracture={:.2} finish={:.2} pcg_iters={total_pcg_iterations} residual={:.3e}",
            ms[0],
            ms[1],
            solve_detail_time[0].as_secs_f64() * 1000.0,
            solve_detail_time[1].as_secs_f64() * 1000.0,
            solve_detail_time[2].as_secs_f64() * 1000.0,
            ms[2],
            ms[3],
            ms[4],
            scene.solver_residual
        );
    }
}

/// Convert stored bond energy to heat and remove bonds that cannot exist in a
/// fully liquid REV. Called at the fixed-step boundary so all solver workspaces
/// see one consistent topology for the whole step.
fn break_fully_melted_bonds(scene: &mut Scene) -> bool {
    let melted: Vec<bool> = scene
        .voxels
        .iter()
        .zip(&scene.properties)
        .map(|(voxel, material)| {
            material
                .phase_state(voxel.mass, voxel.internal_energy)
                .solid_fraction
                <= 0.0
        })
        .collect();
    if !melted.iter().any(|&value| value) {
        return false;
    }

    let mut topology_changed = false;
    for axis in 0..3 {
        for i in 0..scene.voxels.len() {
            let bond = scene.bonds[axis][i];
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if !melted[i] && !melted[j] {
                continue;
            }

            let frame = basis(axis, scene.rotations[i]);
            let displacement = frame.transpose() * (scene.positions[j] - scene.positions[i])
                - Vec3::X * CELL
                - Vec3::from_array(bond.plastic_strain) * CELL;
            let stored = 0.5
                * (frame.transpose() * scene.workspace[axis][i].last_force)
                    .dot(displacement)
                    .max(0.0);
            let pending: f64 = scene
                .releases
                .iter()
                .filter(|release| release.i == i && release.j == j)
                .map(|release| release.energy)
                .sum();
            scene
                .releases
                .retain(|release| release.i != i || release.j != j);
            scene.energy.pending_release = (scene.energy.pending_release - pending).max(0.0);
            scene.energy.damage_heat += stored as f64 + pending;
            let heat = (stored as f64 + pending) as f32 * 0.5;
            scene.voxels[i].internal_energy += heat;
            scene.voxels[j].internal_energy += heat;

            scene.bonds[axis][i] = Default::default();
            scene.workspace[axis][i].lambda = Vec3::ZERO;
            scene.workspace[axis][i].last_force = Vec3::ZERO;
            scene.broken += 1;
            topology_changed = true;
        }
    }
    topology_changed
}

pub fn rebuild_clusters(scene: &mut Scene) {
    let n = scene.voxels.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for bonds in &scene.bonds {
        for (i, b) in bonds.iter().enumerate() {
            if b.neighbor != EMPTY {
                let a = root(&mut parent, i);
                let c = root(&mut parent, b.neighbor as usize);
                parent[c] = a;
            }
        }
    }
    let mut ids = std::collections::HashMap::new();
    for i in 0..n {
        if scene.voxels[i].kinematic {
            scene.clusters[i] = u32::MAX;
            continue;
        }
        let r = root(&mut parent, i);
        let next = ids.len() as u32;
        scene.clusters[i] = *ids.entry(r).or_insert(next);
    }
    scene.cluster_count = ids.len();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fracture_release_respects_fixed_axes_and_accounts_for_heat() {
        let mut s = scene();
        s.energy.pending_release = 0.25;
        s.releases.push(crate::fracture::Release {
            i: 0,
            j: 1,
            energy: 0.25,
            due: 0,
        });
        let targets = vec![[Some(0.0); 3]; s.voxels.len()];
        let e0 = s.voxels[0].internal_energy + s.voxels[1].internal_energy;
        apply_releases(&mut s, &targets);
        assert_eq!(s.voxels[0].velocity, Vec3::ZERO);
        assert_eq!(s.voxels[1].velocity, Vec3::ZERO);
        assert_eq!(s.energy.pending_release, 0.0);
        assert_eq!(s.energy.damage_heat, 0.25);
        assert_eq!(s.energy.kinetic_release, 0.0);
        assert!(
            (s.voxels[0].internal_energy + s.voxels[1].internal_energy - e0 - 0.25).abs() < 1e-4
        );
    }
    fn scene() -> Scene {
        let mut c = crate::scene::SceneConfig::load("scenes/wood_chop.ron").unwrap();
        c.dimensions = [2, 2, 2];
        c.branch_dimensions = [0, 0, 0];
        c.ground_dimensions = [2, 2, 1];
        c.gravity = [0.0; 3];
        Scene::new(c)
    }
    fn free_voxel() -> Scene {
        let mut c = crate::scene::SceneConfig::load("scenes/wood_chop.ron").unwrap();
        c.dimensions = [1, 1, 1];
        c.branch_dimensions = [0; 3];
        c.ground_dimensions = [0; 3];
        c.gravity = [0.0; 3];
        c.fixed_dt = 0.01;
        c.substeps = 1;
        c.iterations = 1;
        c.corotation_iterations = 1;
        Scene::new(c)
    }
    #[test]
    fn persistent_force_and_acceleration_use_voxel_mass() {
        let mut s = free_voxel();
        let mass = s.voxels[0].mass;
        set_force(&mut s, 0, Vec3::X * (3.0 * mass)).unwrap();
        set_acceleration(&mut s, 0, Vec3::Y * 2.0).unwrap();
        let mut tool = None;
        step(&mut s, &mut tool);
        assert!((s.voxels[0].velocity.x - 0.03).abs() < 1e-5);
        assert!((s.voxels[0].velocity.y - 0.02).abs() < 1e-5);
        step(&mut s, &mut tool);
        assert!((s.voxels[0].velocity.x - 0.06).abs() < 1e-5);
        assert!((s.voxels[0].velocity.y - 0.04).abs() < 1e-5);
        clear_applied_actions(&mut s, 0).unwrap();
        step(&mut s, &mut tool);
        assert!((s.voxels[0].velocity.x - 0.06).abs() < 1e-5);
        assert!((s.voxels[0].velocity.y - 0.04).abs() < 1e-5);
    }
    #[test]
    fn velocity_and_impulse_are_mass_consistent() {
        let mut s = free_voxel();
        let mass = s.voxels[0].mass;
        set_velocity(&mut s, 0, Vec3::X * 1.5).unwrap();
        apply_impulse(&mut s, 0, Vec3::Y * (mass * 2.0)).unwrap();
        assert!((s.voxels[0].velocity.x - 1.5).abs() < 1e-6);
        assert!((s.voxels[0].velocity.y - 2.0).abs() < 1e-6);
    }
    #[test]
    fn external_actions_reject_fixed_and_missing_voxels() {
        let mut s = scene();
        let fixed = s.voxels.iter().position(|v| v.kinematic).unwrap();
        assert!(set_force(&mut s, fixed, Vec3::X).is_err());
        assert!(set_acceleration(&mut s, fixed, Vec3::X).is_err());
        assert!(apply_impulse(&mut s, fixed, Vec3::X).is_err());
        assert!(set_velocity(&mut s, usize::MAX, Vec3::X).is_err());
        assert!(set_force(&mut s, 0, Vec3::splat(f32::NAN)).is_err());
    }
    #[test]
    fn rest_and_rigid_rotation_have_no_strain() {
        let mut s = scene();
        let q = Quat::from_rotation_z(0.7);
        for i in 0..s.positions.len() {
            s.positions[i] = q * s.positions[i];
        }
        let mut scratch = SolverScratch::default();
        fit_rotations(&mut s, &mut scratch);
        let frame = basis(0, s.rotations[0]);
        let j = s.bonds[0][0].neighbor as usize;
        let c = frame.transpose() * (s.positions[j] - s.positions[0]) - Vec3::X * CELL;
        assert!(c.length() < 1e-6);
    }
    #[test]
    fn stiffness_has_shear_and_is_symmetric() {
        let s = scene();
        let k = stiffness_matrix(&s, 0, 1, Mat3::IDENTITY);
        assert!(k.y_axis.y > 0.0 && k.z_axis.z > 0.0 && k.determinant() > 0.0);
        assert!((k - k.transpose())
            .to_cols_array()
            .iter()
            .all(|v| v.abs() < 1e-3));
    }
    #[test]
    fn actual_vertical_wood_bonds_are_stronger_than_horizontal_bonds() {
        let s = scene();
        let i = 0;
        let vertical_j = s.bonds[2][i].neighbor as usize;
        let horizontal_j = s.bonds[0][i].neighbor as usize;
        assert_ne!(s.bonds[2][i].neighbor, EMPTY);
        assert_ne!(s.bonds[0][i].neighbor, EMPTY);

        let vertical = bond_limits(&s, i, vertical_j, Vec3::Z);
        let horizontal = bond_limits(&s, i, horizontal_j, Vec3::X);
        assert!(vertical.tension >= horizontal.tension * 39.0);
        assert!(vertical.compression >= horizontal.compression * 24.0);
        assert!(vertical.shear >= horizontal.shear * 3.9);
        assert!((vertical.tension - 100e6).abs() < 1.0);
        assert!((horizontal.tension - 2.5e6).abs() < 1.0);
        assert!((vertical.compression - 50e6).abs() < 1.0);
        assert!((horizontal.compression - 2e6).abs() < 1.0);
        assert!((vertical.shear - 12e6).abs() < 1.0);
        assert!((horizontal.shear - 3e6).abs() < 1.0);

        let vertical_stiffness = stiffness_matrix(&s, i, vertical_j, basis(2, Quat::IDENTITY))
            .x_axis
            .x;
        let horizontal_stiffness = stiffness_matrix(&s, i, horizontal_j, basis(0, Quat::IDENTITY))
            .x_axis
            .x;
        assert!(vertical_stiffness > horizontal_stiffness * 15.0);
    }
    #[test]
    fn equilibrium_does_not_break() {
        let mut s = scene();
        let initial = s.positions.clone();
        step(&mut s, &mut None);
        assert_eq!(s.broken, 0);
        assert!(s
            .positions
            .iter()
            .zip(initial)
            .all(|(a, b)| a.distance(b) < 1e-6));
    }
    #[test]
    fn coupled_solve_has_uniform_axial_traction() {
        let mut c = crate::scene::SceneConfig::load("scenes/wood_chop.ron").unwrap();
        c.dimensions = [10, 2, 2];
        c.branch_dimensions = [0; 3];
        c.ground_dimensions = [0; 3];
        c.gravity = [0.; 3];
        c.fixed_dt = 0.001;
        c.substeps = 1;
        c.iterations = 100;
        let mut s = Scene::new(c);
        let mut targets = vec![None; s.voxels.len()];
        for i in 0..s.voxels.len() {
            if i % 10 == 0 {
                targets[i] = Some(s.positions[i]);
            }
            if i % 10 == 9 {
                targets[i] = Some(s.positions[i] + Vec3::X * 0.00002);
            }
        }
        step_with_targets(&mut s, &mut None, &targets);
        let forces: Vec<_> = s.bonds[0]
            .iter()
            .enumerate()
            .filter(|(_, b)| b.neighbor != EMPTY)
            .map(|(i, _)| s.workspace[0][i].last_force.x)
            .collect();
        let min = forces.iter().copied().fold(f32::INFINITY, f32::min);
        let max = forces.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min > 0.0 && max / min < 1.02, "min={min} max={max}");
        assert_eq!(s.broken, 0);
    }
    #[test]
    fn damage_splits_connectivity() {
        let mut s = scene();
        for a in 0..3 {
            for b in &mut s.bonds[a] {
                b.neighbor = EMPTY;
                b.damage = 1.0;
            }
        }
        rebuild_clusters(&mut s);
        assert_eq!(s.cluster_count, 8);
    }
}

fn apply_releases(scene: &mut Scene, targets: &[[Option<f32>; 3]]) {
    let releases = std::mem::take(&mut scene.releases);
    for r in releases {
        if r.due > scene.substep_index {
            scene.releases.push(r);
            continue;
        }
        let normal = (scene.positions[r.j] - scene.positions[r.i]).normalize_or_zero();
        let wi = if scene.voxels[r.i].kinematic {
            0.0
        } else {
            scene.voxels[r.i].mass.recip()
        };
        let wj = if scene.voxels[r.j].kinematic {
            0.0
        } else {
            scene.voxels[r.j].mass.recip()
        };
        let free = |i: usize| {
            Vec3::from_array(std::array::from_fn(|axis| {
                if scene.voxels[i].kinematic || targets.get(i).and_then(|t| t[axis]).is_some() {
                    0.0
                } else {
                    1.0
                }
            }))
        };
        let fi = free(r.i);
        let fj = free(r.j);
        let effective_i = wi * normal.dot(normal * fi);
        let effective_j = wj * normal.dot(normal * fj);
        let impulse = crate::fracture::release_impulse(
            scene.voxels[r.i].velocity * fi,
            scene.voxels[r.j].velocity * fj,
            effective_i,
            effective_j,
            normal,
            r.energy,
        );
        scene.voxels[r.i].velocity -= impulse * wi * fi;
        scene.voxels[r.j].velocity += impulse * wj * fj;
        scene.energy.pending_release -= r.energy;
        if effective_i + effective_j > 0.0 && normal.length_squared() > 0.5 {
            scene.energy.kinetic_release += r.energy;
        } else {
            scene.energy.damage_heat += r.energy;
            scene.voxels[r.i].internal_energy += (r.energy * 0.5) as f32;
            scene.voxels[r.j].internal_energy += (r.energy * 0.5) as f32;
        }
    }
}

struct Constraint {
    i: usize,
    j: usize,
    axis: usize,
    frame: Mat3,
    material_k: Mat3,
    k: Mat3,
    rest: Vec3,
}

#[derive(Clone, Copy)]
struct PcgRow {
    neighbors: [u32; 6],
}

impl Default for PcgRow {
    fn default() -> Self {
        Self {
            neighbors: [u32::MAX; 6],
        }
    }
}

#[derive(Default)]
pub(crate) struct SolverScratch {
    free: Vec<Vec3>,
    active: Vec<usize>,
    active_index: Vec<u32>,
    masses: Vec<f32>,
    incidence: Vec<[u32; 6]>,
    incidence_len: Vec<u8>,
    diagonal: Vec<Mat3>,
    factor_matrix: Vec<Mat3>,
    factor_diagonal: Vec<Mat3>,
    factor_edges: Vec<Mat3>,
    preconditioner_work: Vec<Vec3>,
    rhs: Vec<Vec3>,
    constraints: Vec<Constraint>,
    pcg_rows: Vec<PcgRow>,
    cuda_blocks: Vec<[f32; 54]>,
    cuda_diagonal: Vec<[f32; 9]>,
    cuda_inverse: Vec<[f32; 9]>,
    dx: Vec<Vec3>,
    ap: Vec<Vec3>,
    r: Vec<Vec3>,
    z: Vec<Vec3>,
    p: Vec<Vec3>,
    rotation_gradients: Vec<Mat3>,
    rotation_counts: Vec<Vec3>,
}
fn make_constraint(scene: &Scene, i: usize, axis: usize, j: usize) -> Constraint {
    let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
    let material_k = stiffness_matrix(scene, i, j, frame);
    let local = material_k * (1. - scene.bonds[axis][i].damage).max(0.001);
    let k = frame * local * frame.transpose();
    let bond = scene.bonds[axis][i];
    let rest = frame * (Vec3::X * CELL + Vec3::from_array(bond.plastic_strain) * CELL);
    Constraint {
        i,
        j,
        axis,
        frame,
        material_k,
        k,
        rest,
    }
}

fn cholesky_lower(m: Mat3) -> Mat3 {
    let l00 = m.x_axis.x.max(1e-20).sqrt();
    let l10 = m.x_axis.y / l00;
    let l20 = m.x_axis.z / l00;
    let l11 = (m.y_axis.y - l10 * l10).max(1e-20).sqrt();
    let l21 = (m.y_axis.z - l20 * l10) / l11;
    let l22 = (m.z_axis.z - l20 * l20 - l21 * l21).max(1e-20).sqrt();
    Mat3::from_cols(
        Vec3::new(l00, l10, l20),
        Vec3::new(0.0, l11, l21),
        Vec3::new(0.0, 0.0, l22),
    )
}

fn solve_lower(l: Mat3, b: Vec3) -> Vec3 {
    let x = b.x / l.x_axis.x;
    let y = (b.y - l.x_axis.y * x) / l.y_axis.y;
    let z = (b.z - l.x_axis.z * x - l.y_axis.z * y) / l.z_axis.z;
    Vec3::new(x, y, z)
}

fn solve_lower_transpose(l: Mat3, b: Vec3) -> Vec3 {
    let z = b.z / l.z_axis.z;
    let y = (b.y - l.y_axis.z * z) / l.y_axis.y;
    let x = (b.x - l.x_axis.y * y - l.x_axis.z * z) / l.x_axis.x;
    Vec3::new(x, y, z)
}

fn right_solve_lower_transpose(m: Mat3, l: Mat3) -> Mat3 {
    let row0 = solve_lower(l, Vec3::new(m.x_axis.x, m.y_axis.x, m.z_axis.x));
    let row1 = solve_lower(l, Vec3::new(m.x_axis.y, m.y_axis.y, m.z_axis.y));
    let row2 = solve_lower(l, Vec3::new(m.x_axis.z, m.y_axis.z, m.z_axis.z));
    Mat3::from_cols(
        Vec3::new(row0.x, row1.x, row2.x),
        Vec3::new(row0.y, row1.y, row2.y),
        Vec3::new(row0.z, row1.z, row2.z),
    )
}

// Block incomplete Cholesky (zero fill) uses the same face-neighbor graph as
// the elasticity matrix. It costs more than diagonal scaling per PCG pass,
// but should cut the pass count on the long-wavelength modes that dominate
// this structured REV lattice.
fn build_ic0(scene: &Scene, scratch: &mut SolverScratch) {
    let n = scene.voxels.len();
    scratch.factor_matrix.resize(n, Mat3::ZERO);
    scratch.factor_diagonal.resize(n, Mat3::IDENTITY);
    scratch
        .factor_edges
        .resize(scratch.constraints.len(), Mat3::ZERO);
    for i in 0..n {
        let projection = Mat3::from_diagonal(scratch.free[i]);
        scratch.factor_matrix[i] =
            projection * scratch.diagonal[i] * projection + Mat3::IDENTITY - projection;
    }
    let incidence = &scratch.incidence;
    let incidence_len = &scratch.incidence_len;
    let constraints = &scratch.constraints;
    for row in 0..n {
        if scratch.free[row] == Vec3::ZERO {
            scratch.factor_diagonal[row] = Mat3::IDENTITY;
            continue;
        }
        let row_projection = Mat3::from_diagonal(scratch.free[row]);
        let row_edges = &incidence[row][..incidence_len[row] as usize];
        for &edge in row_edges {
            let edge = edge as usize;
            let c = &constraints[edge];
            if c.j != row {
                continue;
            }
            let col = c.i;
            let col_projection = Mat3::from_diagonal(scratch.free[col]);
            let mut value = -row_projection * c.k * col_projection;
            let col_edges = &incidence[col][..incidence_len[col] as usize];
            for &row_shared in row_edges {
                let row_shared = row_shared as usize;
                let r = &constraints[row_shared];
                if r.j != row || r.i >= col {
                    continue;
                }
                for &col_shared in col_edges {
                    let col_shared = col_shared as usize;
                    let s = &constraints[col_shared];
                    if s.j == col && s.i == r.i {
                        value -= scratch.factor_edges[row_shared]
                            * scratch.factor_edges[col_shared].transpose();
                        break;
                    }
                }
            }
            scratch.factor_edges[edge] =
                right_solve_lower_transpose(value, scratch.factor_diagonal[col]);
            scratch.factor_matrix[row] -=
                scratch.factor_edges[edge] * scratch.factor_edges[edge].transpose();
        }
        scratch.factor_diagonal[row] = cholesky_lower(scratch.factor_matrix[row]);
    }
}

fn apply_ic0(
    constraints: &[Constraint],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    factor_diagonal: &[Mat3],
    factor_edges: &[Mat3],
    active: &[usize],
    active_index: &[u32],
    rhs: &[Vec3],
    work: &mut [Vec3],
    out: &mut [Vec3],
) {
    for (slot, &i) in active.iter().enumerate() {
        let mut value = rhs[slot];
        for &edge in &incidence[i][..incidence_len[i] as usize] {
            let edge = edge as usize;
            let c = &constraints[edge];
            if c.j == i {
                let neighbor = active_index[c.i] as usize;
                if active_index[c.i] != u32::MAX {
                    value -= factor_edges[edge] * work[neighbor];
                }
            }
        }
        work[slot] = solve_lower(factor_diagonal[i], value);
    }
    for slot in (0..active.len()).rev() {
        let i = active[slot];
        let mut value = work[slot];
        for &edge in &incidence[i][..incidence_len[i] as usize] {
            let edge = edge as usize;
            let c = &constraints[edge];
            if c.i == i {
                let neighbor = active_index[c.j] as usize;
                if active_index[c.j] != u32::MAX {
                    value -= factor_edges[edge].transpose() * out[neighbor];
                }
            }
        }
        out[slot] = solve_lower_transpose(factor_diagonal[i], value);
    }
}

/// Primal form of the coupled vector XPBD system for a frozen corotational frame.
/// Eliminates multipliers: (M/h² + Jᵀ K J) dx = -Jᵀ K C.
/// This is algebraically the same compliant solve, with global coupling retained.
fn solve_coupled(
    scene: &mut Scene,
    h: f32,
    targets: &[[Option<f32>; 3]],
    loads: &[Vec3],
    predicted: &[Vec3],
    scratch: &mut SolverScratch,
    reuse_preconditioner: bool,
) -> (usize, [Duration; 3]) {
    let assembly_start = Instant::now();
    let n = scene.voxels.len();
    scratch.free.clear();
    scratch
        .free
        .extend(scene.voxels.iter().enumerate().map(|(i, v)| {
            Vec3::from_array(std::array::from_fn(|a| {
                if v.kinematic || targets.get(i).and_then(|t| t[a]).is_some() {
                    0.
                } else {
                    1.
                }
            }))
        }));
    scratch.active.clear();
    scratch.active_index.resize(n, u32::MAX);
    scratch.active_index.fill(u32::MAX);
    for i in 0..n {
        if scratch.free[i] != Vec3::ZERO {
            scratch.active_index[i] =
                u32::try_from(scratch.active.len()).expect("active REV count exceeds u32");
            scratch.active.push(i);
        }
    }
    scratch.masses.clear();
    scratch
        .masses
        .extend(scene.voxels.iter().map(|v| v.mass / (h * h)));
    // The bond graph is face-connected, so each voxel can touch at most six
    // constraints. Fixed slots avoid millions of tiny heap allocations on a
    // large scene, while preserving the same per-node gather in the solver.
    scratch.incidence.resize(n, [u32::MAX; 6]);
    scratch.incidence.fill([u32::MAX; 6]);
    scratch.incidence_len.resize(n, 0);
    scratch.incidence_len.fill(0);
    scratch.diagonal.resize(n, Mat3::ZERO);
    scratch.rhs.resize(n, Vec3::ZERO);
    for i in 0..n {
        scratch.diagonal[i] = Mat3::from_diagonal(Vec3::splat(scratch.masses[i]));
        scratch.rhs[i] = (predicted[i] - scene.positions[i]) * scratch.masses[i]
            + loads.get(i).copied().unwrap_or(Vec3::ZERO);
    }
    scratch.constraints.clear();
    for axis in 0..3 {
        for i in 0..n {
            let bond = scene.bonds[axis][i];
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if scratch.free[i] == Vec3::ZERO
                && scratch.free[j] == Vec3::ZERO
                && scene.voxels[i].kinematic
            {
                continue;
            }
            let constraint = make_constraint(scene, i, axis, j);
            if scratch.free[i] == Vec3::ZERO && scratch.free[j] == Vec3::ZERO {
                // The solve has no free degree of freedom, but fracture still
                // needs the current undamaged stiffness and local frame.
                scratch.constraints.push(constraint);
                continue;
            }
            let force = constraint.k * (scene.positions[j] - scene.positions[i] - constraint.rest);
            scratch.rhs[i] += force;
            scratch.rhs[j] -= force;
            scratch.diagonal[i] += constraint.k;
            scratch.diagonal[j] += constraint.k;
            let edge = u32::try_from(scratch.constraints.len())
                .expect("bond constraint count exceeds the voxel index range");
            for vertex in [i, j] {
                let slot = scratch.incidence_len[vertex] as usize;
                debug_assert!(slot < 6, "face-connected voxel has more than six bonds");
                scratch.incidence[vertex][slot] = edge;
                scratch.incidence_len[vertex] += 1;
            }
            scratch.constraints.push(constraint);
        }
    }
    for i in 0..n {
        scratch.rhs[i] *= scratch.free[i];
        let projection = Mat3::from_diagonal(scratch.free[i]);
        scratch.diagonal[i] =
            projection * scratch.diagonal[i] * projection + Mat3::IDENTITY - projection;
    }
    let assembly_time = assembly_start.elapsed();
    let factor_start = Instant::now();
    if !reuse_preconditioner {
        build_ic0(scene, scratch);
    }
    let factor_time = factor_start.elapsed();
    let free = &scratch.free;
    let masses = &scratch.masses;
    let active = &scratch.active;
    let active_index = &scratch.active_index;
    let incidence = &scratch.incidence;
    let incidence_len = &scratch.incidence_len;
    let constraints = &scratch.constraints;
    scratch.pcg_rows.resize(active.len(), PcgRow::default());
    for (slot, &i) in active.iter().enumerate() {
        let row = &mut scratch.pcg_rows[slot];
        for (edge_slot, &edge) in incidence[i][..incidence_len[i] as usize].iter().enumerate() {
            let constraint = &constraints[edge as usize];
            let neighbor = if constraint.i == i {
                constraint.j
            } else {
                constraint.i
            };
            row.neighbors[edge_slot] = active_index[neighbor];
        }
    }
    let pcg_rows = &scratch.pcg_rows;
    let apply = |v: &[Vec3], out: &mut [Vec3]| {
        let apply_one = |(slot, out): (usize, &mut Vec3)| {
            let i = active[slot];
            let vi = v[slot];
            let mut value = vi * masses[i];
            for (edge_slot, &edge_id) in
                incidence[i][..incidence_len[i] as usize].iter().enumerate()
            {
                let neighbor = pcg_rows[slot].neighbors[edge_slot];
                let vj = if neighbor == u32::MAX {
                    Vec3::ZERO
                } else {
                    v[neighbor as usize]
                };
                value += constraints[edge_id as usize].k * (vi - vj);
            }
            *out = value * free[i];
        };
        if active.len() < PARALLEL_MIN_VOXELS {
            out.iter_mut()
                .enumerate()
                .for_each(|(slot, out)| apply_one((slot, out)));
        } else {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(slot, out)| apply_one((slot, out)));
        }
    };
    let dot = |a: &[Vec3], b: &[Vec3]| -> f64 {
        if a.len() < PARALLEL_MIN_VOXELS {
            a.iter().zip(b).map(|(x, y)| x.dot(*y) as f64).sum()
        } else {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| x.dot(*y) as f64)
                .sum()
        }
    };
    let active_count = active.len();
    scratch.dx.resize(active_count, Vec3::ZERO);
    scratch.dx.fill(Vec3::ZERO);
    scratch.ap.resize(active_count, Vec3::ZERO);
    scratch.r.resize(active_count, Vec3::ZERO);
    for (slot, &i) in active.iter().enumerate() {
        scratch.r[slot] = scratch.rhs[i];
    }
    scratch.z.resize(active_count, Vec3::ZERO);
    scratch.preconditioner_work.resize(active_count, Vec3::ZERO);
    let pcg_start = Instant::now();
    let backend = std::env::var("VOXEL_SOLVER_BACKEND").unwrap_or_else(|_| "auto".into());
    let mut cuda_solution = None;
    let try_cuda = match backend.as_str() {
        "cpu" => false,
        "cuda" => true,
        _ => active_count >= 2_048,
    };
    if try_cuda {
        scratch.cuda_blocks.resize(active_count, [0.0; 54]);
        scratch.cuda_blocks.fill([0.0; 54]);
        scratch.cuda_diagonal.resize(active_count, [0.0; 9]);
        scratch.cuda_inverse.resize(active_count, [0.0; 9]);
        for (slot, &i) in active.iter().enumerate() {
            let row_projection = Mat3::from_diagonal(scratch.free[i]);
            for (edge_slot, &edge) in incidence[i][..incidence_len[i] as usize].iter().enumerate() {
                let neighbor = pcg_rows[slot].neighbors[edge_slot];
                let column_projection = if neighbor == u32::MAX {
                    Mat3::ZERO
                } else {
                    Mat3::from_diagonal(scratch.free[active[neighbor as usize]])
                };
                let matrix = -row_projection * constraints[edge as usize].k * column_projection;
                let packed = &mut scratch.cuda_blocks[slot][edge_slot * 9..edge_slot * 9 + 9];
                packed.copy_from_slice(&[
                    matrix.x_axis.x,
                    matrix.y_axis.x,
                    matrix.z_axis.x,
                    matrix.x_axis.y,
                    matrix.y_axis.y,
                    matrix.z_axis.y,
                    matrix.x_axis.z,
                    matrix.y_axis.z,
                    matrix.z_axis.z,
                ]);
            }
            let diagonal = scratch.diagonal[i];
            scratch.cuda_diagonal[slot] = [
                diagonal.x_axis.x,
                diagonal.y_axis.x,
                diagonal.z_axis.x,
                diagonal.x_axis.y,
                diagonal.y_axis.y,
                diagonal.z_axis.y,
                diagonal.x_axis.z,
                diagonal.y_axis.z,
                diagonal.z_axis.z,
            ];
            let inverse = scratch.diagonal[i].inverse();
            scratch.cuda_inverse[slot] = [
                inverse.x_axis.x,
                inverse.y_axis.x,
                inverse.z_axis.x,
                inverse.x_axis.y,
                inverse.y_axis.y,
                inverse.z_axis.y,
                inverse.x_axis.z,
                inverse.y_axis.z,
                inverse.z_axis.z,
            ];
        }
        match crate::cuda_solver::solve_pcg(
            &scratch.r,
            &scratch
                .pcg_rows
                .iter()
                .map(|row| row.neighbors)
                .collect::<Vec<_>>(),
            &scratch.cuda_diagonal,
            &scratch.cuda_blocks,
            &scratch.cuda_inverse,
            scene.config.iterations,
            scene.config.solver_relative_tolerance,
        ) {
            Ok((correction, iterations, residual)) => {
                scratch.dx = correction;
                scene.solver_residual = residual;
                cuda_solution = Some(iterations);
            }
            Err(error) => {
                static REPORTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                REPORTED.get_or_init(|| eprintln!("CUDA PCG unavailable; using CPU PCG: {error}"));
            }
        }
    }
    let cuda_time = cuda_solution.map(|_| pcg_start.elapsed());
    if let Some(iterations) = cuda_solution {
        if std::env::var_os("VOXEL_PROFILE_STEP").is_some() {
            eprintln!(
                "CUDA PCG: active={} iterations={} residual={:.3e} elapsed={:.2} ms",
                active_count,
                iterations,
                scene.solver_residual,
                cuda_time.unwrap_or_default().as_secs_f64() * 1_000.0,
            );
        }
        for (slot, &i) in active.iter().enumerate() {
            scene.positions[i] += scratch.dx[slot];
        }
        for c in &scratch.constraints {
            let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest);
            scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h * h);
        }
        return (
            iterations,
            [assembly_time, factor_time, cuda_time.unwrap_or_default()],
        );
    }
    let cpu_pcg_start = Instant::now();
    apply_ic0(
        constraints,
        incidence,
        incidence_len,
        &scratch.factor_diagonal,
        &scratch.factor_edges,
        active,
        active_index,
        &scratch.r,
        &mut scratch.preconditioner_work,
        &mut scratch.z,
    );
    scratch.p.clone_from(&scratch.z);
    let dx = &mut scratch.dx;
    let ap = &mut scratch.ap;
    let r = &mut scratch.r;
    let z = &mut scratch.z;
    let p = &mut scratch.p;
    let mut rz = dot(r, z);
    let initial = dot(&r, &r).sqrt();
    scene.solver_residual = 0.;
    let mut iterations = 0;
    for _ in 0..scene.config.iterations {
        if initial <= 1e-12 {
            break;
        }
        apply(p, ap);
        iterations += 1;
        let denom = dot(&p, &ap);
        if denom <= 1e-30 {
            break;
        }
        let alpha = (rz / denom) as f32;
        if active_count < PARALLEL_MIN_VOXELS {
            for i in 0..active_count {
                dx[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
        } else {
            dx.par_iter_mut()
                .zip(r.par_iter_mut())
                .zip(p.par_iter().zip(ap.par_iter()))
                .for_each(|((dx, r), (p, ap))| {
                    *dx += alpha * *p;
                    *r -= alpha * *ap;
                });
        }
        scene.solver_residual = (dot(&r, &r).sqrt() / initial) as f32;
        if scene.solver_residual < scene.config.solver_relative_tolerance {
            break;
        }
        apply_ic0(
            constraints,
            incidence,
            incidence_len,
            &scratch.factor_diagonal,
            &scratch.factor_edges,
            active,
            active_index,
            r,
            &mut scratch.preconditioner_work,
            z,
        );
        let next = dot(&r, &z);
        let beta = (next / rz) as f32;
        if active_count < PARALLEL_MIN_VOXELS {
            for i in 0..active_count {
                p[i] = z[i] + beta * p[i];
            }
        } else {
            p.par_iter_mut()
                .zip(z.par_iter())
                .for_each(|(p, z)| *p = *z + beta * *p);
        }
        rz = next;
    }
    let pcg_time = cpu_pcg_start.elapsed();
    for (slot, &i) in active.iter().enumerate() {
        scene.positions[i] += dx[slot];
    }
    for c in constraints {
        let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest);
        scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h * h);
    }
    (iterations, [assembly_time, factor_time, pcg_time])
}
