use crate::scene::{Scene, CELL};
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub i: usize,
    pub j: usize,
    /// Unit normal pointing from i toward j.
    pub normal: Vec3,
    pub new_impact: bool,
}

#[derive(Default)]
pub(crate) struct CollisionScratch {
    bins: HashMap<[i32; 3], Vec<usize>>,
    bin_pool: Vec<Vec<usize>>,
    static_bins: HashMap<[i32; 3], Vec<usize>>,
    static_indices: Vec<usize>,
    static_positions: Vec<Vec3>,
    static_ready: bool,
    unique: HashSet<u64>,
    pairs: Vec<(usize, usize)>,
    adjacency: Vec<[u32; 6]>,
}

/// Project voxel pairs apart and detect swept contacts before they can cross.
/// The intact local continuum stencil is governed by the constitutive solver.
/// Treating its edge/corner neighbors as independent AABBs creates artificial
/// strain under rigid rotation. Broken face neighbors and nonlocal folded
/// regions still use REV-sized contact bounds.
pub fn project(scene: &mut Scene, targets: &[[Option<f32>; 3]]) -> Vec<Contact> {
    let profiling = std::env::var_os("VOXEL_PROFILE_COLLISION").is_some();
    let mut phase_start = Instant::now();
    let starts = scene.previous.clone();
    let mut scratch = std::mem::take(&mut scene.collision_scratch);
    adjacency(scene, &mut scratch.adjacency);
    let setup_time = phase_start.elapsed();
    phase_start = Instant::now();
    candidate_pairs(scene, &starts, &mut scratch);
    let candidate_time = phase_start.elapsed();
    let candidate_count = scratch.pairs.len();
    phase_start = Instant::now();
    scratch
        .pairs
        .retain(|&(i, j)| !local_continuum_pair(scene, &scratch.adjacency, i, j));
    let pair_filter_time = phase_start.elapsed();
    let pair_count = scratch.pairs.len();
    let mut contacts = Vec::with_capacity(pair_count);
    phase_start = Instant::now();
    for _ in 0..6 {
        let mut changed = false;
        for &(i, j) in &scratch.pairs {
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
                contacts.push(Contact {
                    i,
                    j,
                    normal,
                    new_impact: (starts[j] - starts[i]).dot(normal) > CELL + 2e-6,
                });
                continue;
            }
            if let Some((normal, penetration)) = overlap_normal(
                scene.positions[i],
                scene.positions[j],
                starts[j] - starts[i],
            ) {
                changed |= separate(scene, i, j, normal, penetration + 1e-6, targets);
                contacts.push(Contact {
                    i,
                    j,
                    normal,
                    new_impact: (starts[j] - starts[i]).dot(normal) > CELL + 2e-6,
                });
            }
        }
        if !changed {
            break;
        }
    }
    let projection_time = phase_start.elapsed();
    phase_start = Instant::now();
    contacts.sort_unstable_by_key(|c| (c.i, c.j));
    contacts.dedup_by_key(|c| (c.i, c.j));
    if profiling {
        eprintln!(
            "collision profile ms: setup={:.2} broadphase={:.2} local_filter={:.2} project={:.2} dedup={:.2} candidates={candidate_count} pairs={} contacts={}",
            setup_time.as_secs_f64() * 1000.0,
            candidate_time.as_secs_f64() * 1000.0,
            pair_filter_time.as_secs_f64() * 1000.0,
            projection_time.as_secs_f64() * 1000.0,
            phase_start.elapsed().as_secs_f64() * 1000.0,
            pair_count,
            contacts.len()
        );
    }
    scene.collision_scratch = scratch;
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

fn candidate_pairs(scene: &Scene, starts: &[Vec3], scratch: &mut CollisionScratch) {
    ensure_static_bins(scene, starts, scratch);
    for (_, mut ids) in scratch.bins.drain() {
        ids.clear();
        scratch.bin_pool.push(ids);
    }
    scratch.unique.clear();
    scratch.pairs.clear();
    let bins = &mut scratch.bins;
    let bin_pool = &mut scratch.bin_pool;
    for i in 0..scene.voxels.len() {
        if scene.voxels[i].kinematic {
            continue;
        }
        let a = starts[i].min(scene.positions[i]) / CELL;
        let b = starts[i].max(scene.positions[i]) / CELL;
        let (lo, hi) = bin_bounds(a, b);
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    bins.entry([x, y, z])
                        .or_insert_with(|| bin_pool.pop().unwrap_or_default())
                        .push(i);
                }
            }
        }
    }
    let unique = &mut scratch.unique;
    let pairs = &mut scratch.pairs;
    let static_bins = &scratch.static_bins;
    for (key, ids) in bins.iter() {
        for a in 0..ids.len() {
            for b in (a + 1)..ids.len() {
                let i = ids[a].min(ids[b]);
                let j = ids[a].max(ids[b]);
                let key = ((i as u64) << 32) | j as u64;
                if unique.insert(key) {
                    pairs.push((i, j));
                }
            }
        }
        if let Some(fixed_ids) = static_bins.get(key) {
            for &i in ids {
                for &j in fixed_ids {
                    let (i, j) = (i.min(j), i.max(j));
                    let pair_key = ((i as u64) << 32) | j as u64;
                    if unique.insert(pair_key) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
}

fn bin_bounds(a: Vec3, b: Vec3) -> ([i32; 3], [i32; 3]) {
    let eps = 1e-5;
    let lo = (a - Vec3::splat(0.5 - eps)).floor().as_ivec3();
    let hi = (b + Vec3::splat(0.5 - eps)).floor().as_ivec3();
    (lo.to_array(), hi.to_array())
}

fn ensure_static_bins(scene: &Scene, starts: &[Vec3], scratch: &mut CollisionScratch) {
    let mut changed = !scratch.static_ready;
    let mut fixed_slot = 0;
    for (i, voxel) in scene.voxels.iter().enumerate() {
        if !voxel.kinematic {
            continue;
        }
        if scratch.static_indices.get(fixed_slot) != Some(&i)
            || scratch.static_positions.get(fixed_slot) != Some(&scene.positions[i])
            || starts[i] != scene.positions[i]
        {
            changed = true;
        }
        fixed_slot += 1;
    }
    if fixed_slot != scratch.static_indices.len() {
        changed = true;
    }
    if !changed {
        return;
    }
    scratch.static_bins.clear();
    scratch.static_indices.clear();
    scratch.static_positions.clear();
    for (i, voxel) in scene.voxels.iter().enumerate() {
        if !voxel.kinematic {
            continue;
        }
        scratch.static_indices.push(i);
        scratch.static_positions.push(scene.positions[i]);
        let a = starts[i].min(scene.positions[i]) / CELL;
        let b = starts[i].max(scene.positions[i]) / CELL;
        let (lo, hi) = bin_bounds(a, b);
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    scratch.static_bins.entry([x, y, z]).or_default().push(i);
                }
            }
        }
    }
    scratch.static_ready = true;
}

fn adjacency(scene: &Scene, result: &mut Vec<[u32; 6]>) {
    result.resize(scene.voxels.len(), [crate::bond::EMPTY; 6]);
    result.fill([crate::bond::EMPTY; 6]);
    for axis in 0..3 {
        for (i, bond) in scene.bonds[axis].iter().enumerate() {
            if bond.neighbor != crate::bond::EMPTY {
                let j = bond.neighbor as usize;
                result[i][2 * axis + 1] = j as u32;
                result[j][2 * axis] = i as u32;
            }
        }
    }
}

fn local_continuum_pair(scene: &Scene, adjacent: &[[u32; 6]], i: usize, j: usize) -> bool {
    let delta = (scene.rest_positions[j] - scene.rest_positions[i]) / CELL;
    let steps = delta.round();
    if (delta - steps).abs().max_element() > 1e-3 || steps.abs().max_element() > 1.0 {
        return false;
    }
    // Only monotone paths inside the original elementary lattice cell qualify.
    // A broken face cannot be bypassed by a longer path around a crack, and a
    // remote part of the same connected component retains self-contact.
    fn connected(adjacent: &[[u32; 6]], i: usize, j: usize, steps: Vec3) -> bool {
        if steps == Vec3::ZERO {
            return i == j;
        }
        for axis in 0..3 {
            if steps[axis] == 0.0 {
                continue;
            }
            let next = adjacent[i][2 * axis + usize::from(steps[axis] > 0.0)];
            if next == crate::bond::EMPTY {
                continue;
            }
            let mut remaining = steps;
            remaining[axis] = 0.0;
            if connected(adjacent, next as usize, j, remaining) {
                return true;
            }
        }
        false
    }
    connected(adjacent, i, j, steps)
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
    if v.kinematic || targets.get(i).and_then(|t| t[axis]).is_some() {
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
    fn rigidly_rotated_continuum_has_no_internal_box_contacts() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [3; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        let mut scene = Scene::new(config);
        let rotation = glam::Quat::from_rotation_z(0.37);
        for p in &mut scene.positions {
            *p = rotation * *p;
        }
        scene.previous.clone_from(&scene.positions);
        let before = scene.positions.clone();
        assert!(project(&mut scene, &[]).is_empty());
        assert_eq!(scene.positions, before);
    }

    #[test]
    fn local_exclusion_does_not_hide_a_crack_or_remote_self_contact() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [3, 2, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        let mut scene = Scene::new(config);
        let mut a = Vec::new();
        adjacency(&scene, &mut a);
        assert!(local_continuum_pair(&scene, &a, 0, 4));
        assert!(!local_continuum_pair(&scene, &a, 0, 2));
        scene.bonds[0][0].neighbor = EMPTY;
        let mut a = Vec::new();
        adjacency(&scene, &mut a);
        assert!(!local_continuum_pair(&scene, &a, 0, 1));
        scene.positions[1] = scene.positions[0] + Vec3::X * (CELL * 0.5);
        scene.previous.clone_from(&scene.positions);
        assert!(project(&mut scene, &[])
            .iter()
            .any(|c| c.i == 0 && c.j == 1));
    }

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
        scene.set_voxel_kinematic(1, true).unwrap();
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
