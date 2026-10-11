use crate::scene::{Scene, CELL};
use glam::Vec3;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub i: usize,
    pub j: usize,
    /// Unit normal pointing from i toward j.
    pub normal: Vec3,
    pub new_impact: bool,
}

/// Project voxel pairs apart and detect swept contacts before they can cross.
/// Bonded nearest neighbors are governed by the constitutive solver; every
/// other pair uses REV-sized, axis-aligned contact bounds.
pub fn project(scene: &mut Scene, targets: &[[Option<f32>; 3]]) -> Vec<Contact> {
    let starts = scene.previous.clone();
    let pairs = candidate_pairs(scene, &starts);
    let mut contacts = Vec::with_capacity(pairs.len());
    for _ in 0..6 {
        let mut changed = false;
        for &(i, j) in &pairs {
            if is_bonded(scene, i, j) {
                continue;
            }
            let sweep = swept_normal(
                &starts[i],
                &starts[j],
                &scene.positions[i],
                &scene.positions[j],
            );
            if let Some(normal) = sweep {
                let separation = (scene.positions[j] - scene.positions[i]).dot(normal);
                if separation < CELL {
                    let correction = CELL - separation + 1e-6;
                    changed |= separate(scene, i, j, normal, correction, targets);
                }
                contacts.push(Contact { i, j, normal,
                    new_impact: (starts[j]-starts[i]).dot(normal) > CELL + 2e-6 });
                continue;
            }
            if let Some((normal, penetration)) = overlap_normal(
                scene.positions[i],
                scene.positions[j],
                starts[j] - starts[i],
            ) {
                changed |= separate(scene, i, j, normal, penetration + 1e-6, targets);
                contacts.push(Contact { i, j, normal,
                    new_impact: (starts[j]-starts[i]).dot(normal) > CELL + 2e-6 });
            }
        }
        if !changed {
            break;
        }
    }
    contacts.sort_unstable_by_key(|c| (c.i, c.j));
    contacts.dedup_by_key(|c| (c.i, c.j));
    contacts
}

/// Remove closing normal velocity after position projection. Restitution is
/// applied once at the contact pair; separating motion is left unchanged.
pub fn resolve_velocities(scene: &mut Scene, contacts: &[Contact], targets: &[[Option<f32>; 3]]) {
    for iteration in 0..4 {
        let mut changed = false;
        for c in contacts {
            let axis = normal_axis(c.normal);
            let wi = inverse_mass(scene, c.i, axis, targets);
            let wj = inverse_mass(scene, c.j, axis, targets);
            let denom = wi + wj;
            if denom <= 0.0 {
                continue;
            }
            let closing = (scene.voxels[c.j].velocity - scene.voxels[c.i].velocity).dot(c.normal);
            if closing < -1e-6 {
                // A resting manifold must not bounce on each new gravity step.
                // Restitution applies only if this pair started with a gap.
                let restitution = if c.new_impact && iteration == 0 {
                    scene.config.fracture.restitution
                } else {
                    0.0
                };
                let impulse = -(1.0 + restitution) * closing / denom;
                scene.voxels[c.i].velocity -= c.normal * (impulse * wi);
                scene.voxels[c.j].velocity += c.normal * (impulse * wj);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn candidate_pairs(scene: &Scene, starts: &[Vec3]) -> Vec<(usize, usize)> {
    let mut bins: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for i in 0..scene.voxels.len() {
        let a = starts[i].min(scene.positions[i]) / CELL;
        let b = starts[i].max(scene.positions[i]) / CELL;
        let eps = 1e-5;
        let lo = (a - Vec3::splat(0.5 - eps)).floor().as_ivec3();
        let hi = (b + Vec3::splat(0.5 - eps)).floor().as_ivec3();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    bins.entry([x, y, z]).or_default().push(i);
                }
            }
        }
    }
    let mut unique = HashSet::new();
    for ids in bins.values() {
        for a in 0..ids.len() {
            for b in (a + 1)..ids.len() {
                let i = ids[a].min(ids[b]);
                let j = ids[a].max(ids[b]);
                if !scene.voxels[i].mass.is_finite() && !scene.voxels[j].mass.is_finite() {
                    continue;
                }
                unique.insert(((i as u64) << 32) | j as u64);
            }
        }
    }
    let mut pairs: Vec<_> = unique
        .into_iter()
        .map(|key| ((key >> 32) as usize, key as u32 as usize))
        .collect();
    pairs.sort_unstable();
    pairs
}

fn is_bonded(scene: &Scene, i: usize, j: usize) -> bool {
    scene
        .bonds
        .iter()
        .any(|axis| axis[i].neighbor == j as u32 || axis[j].neighbor == i as u32)
}

fn swept_normal(start_i: &Vec3, start_j: &Vec3, end_i: &Vec3, end_j: &Vec3) -> Option<Vec3> {
    let d0 = *start_j - *start_i;
    let di = *end_i - *start_i;
    let dj = *end_j - *start_j;
    let relative_step = dj - di;
    if d0.abs().cmple(Vec3::splat(CELL - 1e-6)).all() {
        return None;
    }
    let mut enter = 0.0f32;
    let mut exit = 1.0f32;
    let mut hit_axis = None;
    let mut hit_sign = 1.0f32;
    for axis in 0..3 {
        if relative_step[axis].abs() < 1e-12 {
            if d0[axis].abs() > CELL {
                return None;
            }
            continue;
        }
        let t0 = (-CELL - d0[axis]) / relative_step[axis];
        let t1 = (CELL - d0[axis]) / relative_step[axis];
        let near = t0.min(t1);
        let far = t0.max(t1);
        if near > enter || (hit_axis.is_none() && near >= enter) {
            enter = near;
            hit_axis = Some(axis);
            hit_sign = if d0[axis] + relative_step[axis] * enter >= 0.0 {
                1.0
            } else {
                -1.0
            };
        }
        exit = exit.min(far);
        if enter > exit {
            return None;
        }
    }
    if exit < 0.0 || enter > 1.0 {
        return None;
    }
    let axis = hit_axis?;
    let mut normal = Vec3::ZERO;
    normal[axis] = hit_sign;
    Some(normal)
}

fn overlap_normal(a: Vec3, b: Vec3, fallback: Vec3) -> Option<(Vec3, f32)> {
    let delta = b - a;
    let overlap = Vec3::splat(CELL) - delta.abs();
    if overlap.min_element() <= 0.0 {
        return None;
    }
    let axis = if overlap.x <= overlap.y && overlap.x <= overlap.z {
        0
    } else if overlap.y <= overlap.z {
        1
    } else {
        2
    };
    let sign_source = if delta[axis].abs() > 1e-8 {
        delta[axis]
    } else if fallback[axis].abs() > 1e-8 {
        fallback[axis]
    } else {
        1.0
    };
    let mut normal = Vec3::ZERO;
    normal[axis] = sign_source.signum();
    Some((normal, overlap[axis]))
}

fn normal_axis(normal: Vec3) -> usize {
    if normal.x.abs() > 0.5 {
        0
    } else if normal.y.abs() > 0.5 {
        1
    } else {
        2
    }
}

fn inverse_mass(scene: &Scene, i: usize, axis: usize, targets: &[[Option<f32>; 3]]) -> f32 {
    let v = &scene.voxels[i];
    if !v.mass.is_finite() || targets.get(i).and_then(|t| t[axis]).is_some() {
        0.0
    } else {
        v.mass.recip()
    }
}

fn separate(
    scene: &mut Scene,
    i: usize,
    j: usize,
    normal: Vec3,
    amount: f32,
    targets: &[[Option<f32>; 3]],
) -> bool {
    let axis = normal_axis(normal);
    let wi = inverse_mass(scene, i, axis, targets);
    let wj = inverse_mass(scene, j, axis, targets);
    let denom = wi + wj;
    if denom <= 0.0 {
        return false;
    }
    let di = -normal * (amount * wi / denom);
    let dj = normal * (amount * wj / denom);
    scene.positions[i] += di;
    scene.positions[j] += dj;
    // Split positional correction from velocity so it cannot create energy.
    scene.previous[i] += di;
    scene.previous[j] += dj;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bond::EMPTY, scene::SceneConfig, solver};

    #[test]
    fn free_fall_has_no_invisible_ground() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [1; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0, 0.0, -9.81];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        let mut scene = Scene::new(config);
        let start = scene.positions[0].z;
        for _ in 0..100 {
            solver::step(&mut scene, &mut None);
        }
        let expected = start - 9.81 * 0.001 * 0.001 * 5050.0;
        assert!((scene.positions[0].z - expected).abs() < 2e-6);
        assert!(scene.positions[0].z < 0.0);
    }

    #[test]
    fn moving_voxel_cannot_tunnel_through_fixed_voxel() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.voxels[1].mass = f32::INFINITY;
        scene.previous.clone_from(&scene.positions);
        let wall = scene.positions[1];
        scene.positions[0].x += 3.0 * CELL;
        let contacts = project(&mut scene, &[]);
        assert!(!contacts.is_empty());
        assert_eq!(scene.positions[1], wall);
        assert!(wall.x - scene.positions[0].x >= CELL - 1e-6);
    }

    #[test]
    fn unbonded_voxels_cannot_tunnel_through_each_other() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        config.iterations = 20;
        config.corotation_iterations = 1;
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.workspace[0][0].rest_neighbor = EMPTY;
        crate::solver::rebuild_clusters(&mut scene);
        scene.voxels[0].velocity = Vec3::X * 20.0;
        scene.voxels[1].velocity = -Vec3::X * 20.0;
        let mut tool = None;
        solver::step(&mut scene, &mut tool);
        assert!(
            scene.positions[1].x - scene.positions[0].x >= CELL - 1e-5,
            "positions {:?} {:?}, velocities {:?} {:?}",
            scene.positions[0],
            scene.positions[1],
            scene.voxels[0].velocity,
            scene.voxels[1].velocity
        );
        assert!((scene.voxels[1].velocity - scene.voxels[0].velocity).x >= -1e-4);
    }

    #[test]
    fn initially_overlapping_voxels_are_projected_to_touching() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        config.iterations = 20;
        config.corotation_iterations = 1;
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.workspace[0][0].rest_neighbor = EMPTY;
        crate::solver::rebuild_clusters(&mut scene);
        let midpoint = (scene.positions[0] + scene.positions[1]) * 0.5;
        scene.positions[0] = midpoint;
        scene.positions[1] = midpoint;
        scene.previous.clone_from(&scene.positions);
        let mut tool = None;
        solver::step(&mut scene, &mut tool);
        assert!(scene.positions[1].x - scene.positions[0].x >= CELL - 1e-5);
    }
}
