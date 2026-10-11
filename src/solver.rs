use crate::{
    bond::EMPTY,
    collision,
    scene::{Scene, AREA, CELL},
    tool::Tool,
};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

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
pub fn stiffness_matrix(scene: &Scene, i: usize, j: usize, frame: Mat3) -> Mat3 {
    let tensor = |n: usize| {
        let m = &scene.properties[n];
        let micro = scene.voxels[n].microstructure;
        let f = frame.transpose() * (scene.rotations[n] * micro.fibers());
        let outer = Mat3::from_cols(f * f.x, f * f.y, f * f.z);
        let transverse = m.young + micro.anisotropy * (m.young_perp - m.young);
        // Directional axial modulus plus coupled anisotropic tangential response.
        let normal = transverse + (m.young - transverse) * f.x * f.x;
        Mat3::from_diagonal(Vec3::new(normal, m.shear, m.shear))
            + outer * (m.shear * micro.anisotropy * 0.2)
    };
    let a = tensor(i);
    let b = tensor(j);
    // Equal half-lengths in series; matrix harmonic mean, not entrywise average.
    if a.determinant() <= 0.0 || b.determinant() <= 0.0 {
        return Mat3::ZERO;
    }
    (a.inverse() + b.inverse()).inverse() * (2.0 * AREA / CELL)
}
pub fn bond_limits(scene: &Scene, i: usize, j: usize, direction: Vec3) -> crate::fracture::Limits {
    let alignment =
        [i, j].map(|n| direction.dot(scene.rotations[n] * scene.voxels[n].microstructure.fibers()));
    crate::fracture::pair_limits(
        &scene.properties[i],
        &scene.properties[j],
        alignment,
        [
            scene.voxels[i].microstructure.anisotropy,
            scene.voxels[j].microstructure.anisotropy,
        ],
    )
}
pub fn fracture_strength(scene: &Scene, i: usize, j: usize, direction: Vec3) -> f32 {
    bond_limits(scene, i, j, direction).tension
}

/// Set a world-space force that remains active until changed or cleared.
pub fn set_force(scene: &mut Scene, voxel: usize, force: Vec3) -> Result<(), &'static str> {
    let mass = scene
        .voxels
        .get(voxel)
        .ok_or("voxel index out of range")?
        .mass;
    if !mass.is_finite() {
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
    let mass = scene
        .voxels
        .get(voxel)
        .ok_or("voxel index out of range")?
        .mass;
    if !mass.is_finite() {
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
    if !body.mass.is_finite() {
        return Err("cannot set velocity on a fixed voxel");
    }
    if !velocity.is_finite() {
        return Err("velocity must be finite");
    }
    body.velocity = velocity;
    Ok(())
}

/// Apply a one-time impulse in N·s, conserving the corresponding momentum.
pub fn apply_impulse(scene: &mut Scene, voxel: usize, impulse: Vec3) -> Result<(), &'static str> {
    let body = scene
        .voxels
        .get_mut(voxel)
        .ok_or("voxel index out of range")?;
    if !body.mass.is_finite() {
        return Err("cannot impulse a fixed voxel");
    }
    if !impulse.is_finite() {
        return Err("impulse must be finite");
    }
    let velocity = body.velocity + impulse / body.mass;
    if !velocity.is_finite() {
        return Err("impulse would produce a non-finite velocity");
    }
    body.velocity = velocity;
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

/// Local polar fits make the vector constraints invariant to rigid rotation.
/// Returns the largest quaternion angle changed by this fit.
fn fit_rotations(scene: &mut Scene) -> f32 {
    let n = scene.voxels.len();
    let mut gradients = vec![Mat3::ZERO; n];
    let mut counts = vec![Vec3::ZERO; n];
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
            gradients[i] += m;
            gradients[j] += m;
            counts[i][axis] += 1.0;
            counts[j][axis] += 1.0;
        }
    }
    scene
        .rotations
        .par_iter_mut()
        .enumerate()
        .map(|(i, q)| {
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
                f = (f + f.inverse().transpose()) * 0.5;
            }
            // Gram-Schmidt removes residual stretch and keeps a right-handed basis.
            let x = f.x_axis.normalize_or_zero();
            let y = (f.y_axis - x * x.dot(f.y_axis)).normalize_or_zero();
            if x.length_squared() > 0.5 && y.length_squared() > 0.5 {
                *q = Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y))).normalize();
            }
            2.0 * previous.dot(*q).abs().clamp(0.0, 1.0).acos()
        })
        .reduce(|| 0.0, f32::max)
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
    let applied_loads: Vec<_> = scene
        .voxels
        .iter()
        .enumerate()
        .map(|(i, voxel)| {
            if voxel.mass.is_finite() {
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
    let mut topology_changed = false;
    // Reuse large solver buffers across substeps and corotation corrections.
    let mut solver_scratch = SolverScratch::default();
    for _ in 0..scene.config.substeps {
        scene.previous.clone_from(&scene.positions);
        let mut drift_velocity = vec![Vec3::ZERO; scene.voxels.len()];
        for (i, (p, v)) in scene
            .positions
            .iter_mut()
            .zip(&mut scene.voxels)
            .enumerate()
        {
            if v.mass.is_finite() {
                v.velocity += gravity * h;
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
        fit_rotations(scene);
        for iteration in 0..scene.config.corotation_iterations {
            solve_coupled(
                scene,
                h,
                targets,
                &applied_loads,
                &predicted,
                &mut solver_scratch,
            );
            if iteration + 1 == scene.config.corotation_iterations {
                break;
            }
            let orientation_change = fit_rotations(scene);
            if orientation_change <= scene.config.corotation_tolerance
                && scene.solver_residual <= scene.config.solver_relative_tolerance
            {
                break;
            }
        }
        // Preserve the accurately integrated drift velocity. Recover only the
        // velocity correction from the much smaller constraint displacement;
        // subtracting two large f32 positions every step otherwise accumulates
        // measurable kinetic error in simple free fall.
        let solved_positions = scene.positions.clone();
        let contacts = collision::project(scene, targets);
        // Actual converged strain defines elastic energy and constitutive stress.
        // Reading an intermediate lambda would depend on sweep order and convergence.
        for axis in 0..3 {
            for i in 0..scene.voxels.len() {
                let neighbor = scene.bonds[axis][i].neighbor;
                if neighbor == EMPTY || scene.voxels[i].mass.is_infinite() {
                    continue;
                }
                let j = neighbor as usize;
                let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
                let k = stiffness_matrix(scene, i, j, frame);
                if k.determinant() <= 0.0 {
                    continue;
                }
                let limits = bond_limits(scene, i, j, frame.x_axis);
                let u =
                    frame.transpose() * (scene.positions[j] - scene.positions[i]) - Vec3::X * CELL;
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
        }
        for i in 0..scene.voxels.len() {
            if scene.voxels[i].mass.is_finite() {
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
        scene.substep_index += 1;
    }
    if topology_changed {
        rebuild_clusters(scene);
    }
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
        if scene.voxels[i].mass.is_infinite() {
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
        s.releases.push(crate::fracture::Release { i: 0, j: 1, energy: 0.25, due: 0 });
        let targets = vec![[Some(0.0); 3]; s.voxels.len()];
        let e0 = s.voxels[0].internal_energy + s.voxels[1].internal_energy;
        apply_releases(&mut s, &targets);
        assert_eq!(s.voxels[0].velocity, Vec3::ZERO);
        assert_eq!(s.voxels[1].velocity, Vec3::ZERO);
        assert_eq!(s.energy.pending_release, 0.0);
        assert_eq!(s.energy.damage_heat, 0.25);
        assert_eq!(s.energy.kinetic_release, 0.0);
        assert!((s.voxels[0].internal_energy + s.voxels[1].internal_energy - e0 - 0.25).abs() < 1e-4);
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
        let fixed = s.voxels.iter().position(|v| !v.mass.is_finite()).unwrap();
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
        fit_rotations(&mut s);
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
        let wi = scene.voxels[r.i].mass.recip();
        let wj = scene.voxels[r.j].mass.recip();
        let free = |i: usize| Vec3::from_array(std::array::from_fn(|axis| {
            if targets.get(i).and_then(|t| t[axis]).is_some() { 0.0 } else { 1.0 }
        }));
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
    k: Mat3,
    rest: Vec3,
}
#[derive(Default)]
struct SolverScratch {
    free: Vec<Vec3>,
    masses: Vec<f32>,
    incidence: Vec<[u32; 6]>,
    incidence_len: Vec<u8>,
    diagonal: Vec<Mat3>,
    rhs: Vec<Vec3>,
    constraints: Vec<Constraint>,
    dx: Vec<Vec3>,
    ap: Vec<Vec3>,
    r: Vec<Vec3>,
    z: Vec<Vec3>,
    p: Vec<Vec3>,
}
fn make_constraint(scene: &Scene, i: usize, axis: usize, j: usize) -> Constraint {
    let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
    let local =
        stiffness_matrix(scene, i, j, frame) * (1. - scene.bonds[axis][i].damage).max(0.001);
    let k = frame * local * frame.transpose();
    let bond = scene.bonds[axis][i];
    let rest = frame * (Vec3::X * CELL + Vec3::from_array(bond.plastic_strain) * CELL);
    Constraint {
        i,
        j,
        axis,
        frame,
        k,
        rest,
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
) {
    let n = scene.voxels.len();
    scratch.free.clear();
    scratch
        .free
        .extend(scene.voxels.iter().enumerate().map(|(i, v)| {
            Vec3::from_array(std::array::from_fn(|a| {
                if v.mass.is_infinite() || targets.get(i).and_then(|t| t[a]).is_some() {
                    0.
                } else {
                    1.
                }
            }))
        }));
    scratch.masses.clear();
    scratch.masses.extend(scene.voxels.iter().map(|v| {
        if v.mass.is_finite() {
            v.mass / (h * h)
        } else {
            0.
        }
    }));
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
            if scratch.free[i] == Vec3::ZERO && scratch.free[j] == Vec3::ZERO {
                continue;
            }
            let constraint = make_constraint(scene, i, axis, j);
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
            (projection * scratch.diagonal[i] * projection + Mat3::IDENTITY - projection).inverse();
    }
    let free = &scratch.free;
    let masses = &scratch.masses;
    let incidence = &scratch.incidence;
    let incidence_len = &scratch.incidence_len;
    let diagonal = &scratch.diagonal;
    let constraints = &scratch.constraints;
    let apply = |v: &[Vec3], out: &mut [Vec3]| {
        out.par_iter_mut().enumerate().for_each(|(i, out)| {
            if free[i] == Vec3::ZERO {
                *out = Vec3::ZERO;
                return;
            }
            let vi = v[i];
            let mut value = vi * masses[i];
            for &edge in &incidence[i][..incidence_len[i] as usize] {
                let c = &constraints[edge as usize];
                let j = if c.i == i { c.j } else { c.i };
                value += c.k * (vi - v[j]);
            }
            *out = value * free[i];
        });
    };
    let dot =
        |a: &[Vec3], b: &[Vec3]| -> f64 { a.iter().zip(b).map(|(x, y)| x.dot(*y) as f64).sum() };
    scratch.dx.resize(n, Vec3::ZERO);
    scratch.dx.fill(Vec3::ZERO);
    scratch.ap.resize(n, Vec3::ZERO);
    scratch.r.clone_from(&scratch.rhs);
    scratch.z.resize(n, Vec3::ZERO);
    for i in 0..n {
        scratch.z[i] = diagonal[i] * scratch.r[i];
    }
    scratch.p.clone_from(&scratch.z);
    let dx = &mut scratch.dx;
    let ap = &mut scratch.ap;
    let r = &mut scratch.r;
    let z = &mut scratch.z;
    let p = &mut scratch.p;
    let mut rz = dot(&r, &z);
    let initial = dot(&r, &r).sqrt();
    scene.solver_residual = 0.;
    for _ in 0..scene.config.iterations {
        if initial <= 1e-12 {
            break;
        }
        apply(p, ap);
        let denom = dot(&p, &ap);
        if denom <= 1e-30 {
            break;
        }
        let alpha = (rz / denom) as f32;
        for i in 0..n {
            dx[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        scene.solver_residual = (dot(&r, &r).sqrt() / initial) as f32;
        if scene.solver_residual < scene.config.solver_relative_tolerance {
            break;
        }
        for i in 0..n {
            z[i] = diagonal[i] * r[i];
        }
        let next = dot(&r, &z);
        let beta = (next / rz) as f32;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = next;
    }
    for i in 0..n {
        scene.positions[i] += dx[i];
    }
    for c in constraints {
        let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest);
        scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h * h);
    }
}
