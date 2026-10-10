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

/// Local polar fits make the vector constraints invariant to rigid rotation.
fn fit_rotations(scene: &mut Scene) {
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
        .for_each(|(i, q)| {
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
                return;
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
        });
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

/// Advance with optional world-space nodal forces in newtons. Forces are applied
/// through the same coupled solve as bonds and are shared by every substep.
pub fn step_with_axis_targets_and_loads(
    scene: &mut Scene,
    tool: &mut Option<Tool>,
    targets: &[[Option<f32>; 3]],
    loads: &[Vec3],
) {
    assert!(targets.is_empty() || targets.len() == scene.voxels.len());
    assert!(loads.is_empty() || loads.len() == scene.voxels.len());
    let h = scene.config.fixed_dt / scene.config.substeps as f32;
    let gravity = Vec3::from_array(scene.config.gravity);
    let mut topology_changed = false;
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
        for _ in 0..scene.config.corotation_iterations {
            fit_rotations(scene);
            solve_coupled(scene, h, targets, loads, &predicted);
        }
        for (p, v) in scene.positions.iter_mut().zip(&scene.voxels) {
            if v.mass.is_finite() {
                p.z = p.z.max(scene.ground_top + CELL * 0.5);
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
        apply_releases(scene);
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
    fn scene() -> Scene {
        let mut c = crate::scene::SceneConfig::load("scenes/wood_chop.ron").unwrap();
        c.dimensions = [2, 2, 2];
        c.branch_dimensions = [0, 0, 0];
        c.ground_dimensions = [2, 2, 1];
        c.gravity = [0.0; 3];
        Scene::new(c)
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

fn apply_releases(scene: &mut Scene) {
    let releases = std::mem::take(&mut scene.releases);
    for r in releases {
        if r.due > scene.substep_index {
            scene.releases.push(r);
            continue;
        }
        let normal = (scene.positions[r.j] - scene.positions[r.i]).normalize_or_zero();
        let wi = scene.voxels[r.i].mass.recip();
        let wj = scene.voxels[r.j].mass.recip();
        let impulse = crate::fracture::release_impulse(
            scene.voxels[r.i].velocity,
            scene.voxels[r.j].velocity,
            wi,
            wj,
            normal,
            r.energy,
        );
        scene.voxels[r.i].velocity -= impulse * wi;
        scene.voxels[r.j].velocity += impulse * wj;
        scene.energy.pending_release -= r.energy;
        if wi + wj > 0.0 && normal.length_squared() > 0.5 {
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
/// Primal form of the coupled vector XPBD system for a frozen corotational frame.
/// Eliminates multipliers: (M/h² + Jᵀ K J) dx = -Jᵀ K C.
/// This is algebraically the same compliant solve, with global coupling retained.
fn solve_coupled(
    scene: &mut Scene,
    h: f32,
    targets: &[[Option<f32>; 3]],
    loads: &[Vec3],
    predicted: &[Vec3],
) {
    let n = scene.voxels.len();
    let free: Vec<_> = scene
        .voxels
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Vec3::from_array(std::array::from_fn(|a| {
                if v.mass.is_infinite() || targets.get(i).and_then(|t| t[a]).is_some() {
                    0.
                } else {
                    1.
                }
            }))
        })
        .collect();
    let masses: Vec<_> = scene
        .voxels
        .iter()
        .map(|v| {
            if v.mass.is_finite() {
                v.mass / (h * h)
            } else {
                0.
            }
        })
        .collect();
    let mut constraints = Vec::new();
    let mut incidence = vec![Vec::new(); n];
    let mut diagonal: Vec<_> = masses
        .iter()
        .map(|&m| Mat3::from_diagonal(Vec3::splat(m)))
        .collect();
    let mut rhs: Vec<_> = (0..n)
        .map(|i| {
            (predicted[i] - scene.positions[i]) * masses[i]
                + loads.get(i).copied().unwrap_or(Vec3::ZERO)
        })
        .collect();
    for axis in 0..3 {
        for i in 0..n {
            let bond = scene.bonds[axis][i];
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if free[i] == Vec3::ZERO && free[j] == Vec3::ZERO {
                continue;
            }
            let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
            let local = stiffness_matrix(scene, i, j, frame) * (1. - bond.damage).max(0.001);
            let k = frame * local * frame.transpose();
            let rest = frame * (Vec3::X * CELL + Vec3::from_array(bond.plastic_strain) * CELL);
            let force = k * (scene.positions[j] - scene.positions[i] - rest);
            rhs[i] += force;
            rhs[j] -= force;
            diagonal[i] += k;
            diagonal[j] += k;
            let edge = constraints.len();
            incidence[i].push(edge);
            incidence[j].push(edge);
            constraints.push(Constraint {
                i,
                j,
                axis,
                frame,
                k,
                rest,
            });
        }
    }
    for i in 0..n {
        rhs[i] *= free[i];
        let projection = Mat3::from_diagonal(free[i]);
        diagonal[i] =
            (projection * diagonal[i] * projection + Mat3::IDENTITY - projection).inverse();
    }
    let apply = |v: &[Vec3]| -> Vec<Vec3> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                if free[i] == Vec3::ZERO {
                    return Vec3::ZERO;
                }
                let mut out = v[i] * masses[i];
                for &edge in &incidence[i] {
                    let c = &constraints[edge];
                    let j = if c.i == i { c.j } else { c.i };
                    out += c.k * (v[i] - v[j]);
                }
                out * free[i]
            })
            .collect()
    };
    let dot =
        |a: &[Vec3], b: &[Vec3]| -> f64 { a.iter().zip(b).map(|(x, y)| x.dot(*y) as f64).sum() };
    let mut dx = vec![Vec3::ZERO; n];
    let mut r = rhs;
    let mut z: Vec<_> = r
        .iter()
        .enumerate()
        .map(|(i, &v)| diagonal[i] * v)
        .collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let initial = dot(&r, &r).sqrt();
    scene.solver_residual = 0.;
    for _ in 0..scene.config.iterations {
        if initial <= 1e-12 {
            break;
        }
        let ap = apply(&p);
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
