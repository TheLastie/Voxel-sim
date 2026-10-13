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
                continue;
            }
            if let Some(normal) = touching_normal(scene.positions[i], scene.positions[j]) {
                contacts.push(Contact {
                    i,
                    j,
                    normal,
                    new_impact: false,
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
pub fn resolve_velocities(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f32>; 3]],
) -> Vec<f32> {
    let pre_contact_velocities: Vec<_> = scene.voxels.iter().map(|voxel| voxel.velocity).collect();
    resolve_velocities_with_precontact(scene, contacts, targets, &pre_contact_velocities)
}

pub(crate) fn resolve_velocities_with_precontact(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f32>; 3]],
    pre_contact_velocities: &[Vec3],
) -> Vec<f32> {
    let mut normal_impulses = vec![0.0; contacts.len()];
    let mut physical_velocities = pre_contact_velocities.to_vec();
    for iteration in 0..4 {
        let mut changed = false;
        for (contact_index, c) in contacts.iter().enumerate() {
            let axis = normal_axis(c.normal);
            let wi = inverse_mass(scene, c.i, axis, targets);
            let wj = inverse_mass(scene, c.j, axis, targets);
            let denom = wi + wj;
            if denom <= 0.0 {
                continue;
            }
            let closing = (scene.voxels[c.j].velocity - scene.voxels[c.i].velocity).dot(c.normal);
            let mut normal_impulse = 0.0;
            if closing < -1e-6 {
                // A resting manifold must not bounce on each new gravity step.
                // Restitution applies only if this pair started with a gap.
                let restitution = if c.new_impact && iteration == 0 {
                    scene.config.fracture.restitution
                } else {
                    0.0
                };
                normal_impulse = -(1.0 + restitution) * closing / denom;
                normal_impulses[contact_index] += normal_impulse;
                let physical_closing =
                    (physical_velocities[c.j] - physical_velocities[c.i]).dot(c.normal);
                let physical_impulse = if c.new_impact {
                    normal_impulse
                } else if physical_closing < -1e-6 {
                    (-physical_closing / denom).min(normal_impulse)
                } else {
                    0.0
                };
                scene.voxels[c.i].velocity -= c.normal * (normal_impulse * wi);
                scene.voxels[c.j].velocity += c.normal * (normal_impulse * wj);
                // A persistent manifold can receive a solver impulse solely
                // to balance this substep's applied load. Track only the
                // incoming kinetic component that existed before that solve.
                let heat_closing = if c.new_impact {
                    closing
                } else {
                    physical_closing
                };
                let heat_impulse = physical_impulse;
                let restitution_heat = if heat_closing < -1e-6 && heat_impulse > 0.0 {
                    (heat_impulse as f64 * -(heat_closing as f64)
                        - 0.5 * (heat_impulse as f64).powi(2) * denom as f64)
                        .max(0.0)
                } else {
                    0.0
                };
                if restitution_heat > 0.0 {
                    let i_free = wi > 0.0;
                    let j_free = wj > 0.0;
                    let receivers = i_free as u8 + j_free as u8;
                    let heat_per_voxel = restitution_heat / receivers as f64;
                    if i_free {
                        scene.voxels[c.i].add_internal_energy_j(heat_per_voxel);
                    }
                    if j_free {
                        scene.voxels[c.j].add_internal_energy_j(heat_per_voxel);
                    }
                    scene.energy.contact_restitution_heat += restitution_heat;
                }
                physical_velocities[c.i] -= c.normal * (physical_impulse * wi);
                physical_velocities[c.j] += c.normal * (physical_impulse * wj);
                changed = true;
            }

            // Ground and voxel contacts use the same Coulomb friction rule as
            // tool contacts. The normal impulse bounds the tangential impulse;
            // per-axis inverse masses respect partially fixed scene constraints.
            if normal_impulse > 0.0 && scene.config.fracture.friction > 0.0 {
                let relative = scene.voxels[c.j].velocity - scene.voxels[c.i].velocity;
                let tangent = relative - c.normal * relative.dot(c.normal);
                let tangent_response: [f32; 3] = std::array::from_fn(|axis| {
                    inverse_mass(scene, c.i, axis, targets)
                        + inverse_mass(scene, c.j, axis, targets)
                });
                // Ignore tangential relative motion along axes that neither
                // endpoint can respond on (for example, a partially fixed REV).
                let responsive_tangent = Vec3::new(
                    if tangent_response[0] > 0.0 {
                        tangent.x
                    } else {
                        0.0
                    },
                    if tangent_response[1] > 0.0 {
                        tangent.y
                    } else {
                        0.0
                    },
                    if tangent_response[2] > 0.0 {
                        tangent.z
                    } else {
                        0.0
                    },
                );
                let tangent_speed = responsive_tangent.length();
                if tangent_speed > 1e-6 {
                    let direction = responsive_tangent / tangent_speed;
                    let tangent_weight = (0..3)
                        .map(|axis| direction[axis].powi(2) * tangent_response[axis])
                        .sum::<f32>();
                    if tangent_weight > 0.0 {
                        let impulse_magnitude = (tangent_speed / tangent_weight)
                            .min(scene.config.fracture.friction * normal_impulse);
                        // The relative tangential velocity is v_j - v_i, so
                        // the contact impulse on j points opposite that motion.
                        let impulse = -direction * impulse_magnitude;
                        for axis in 0..3 {
                            scene.voxels[c.i].velocity[axis] -=
                                impulse[axis] * inverse_mass(scene, c.i, axis, targets);
                            scene.voxels[c.j].velocity[axis] +=
                                impulse[axis] * inverse_mass(scene, c.j, axis, targets);
                        }
                        // The constrained relative inverse mass used above
                        // gives the exact kinetic loss for this tangential
                        // impulse. Convert that loss to heat on free endpoints.
                        let friction_heat = (impulse_magnitude as f64 * tangent_speed as f64
                            - 0.5 * (impulse_magnitude as f64).powi(2) * tangent_weight as f64)
                            .max(0.0);
                        if friction_heat > 0.0 {
                            let i_free = !scene.voxels[c.i].kinematic;
                            let j_free = !scene.voxels[c.j].kinematic;
                            let receivers = i_free as u8 + j_free as u8;
                            if receivers > 0 {
                                let heat_per_voxel = friction_heat / receivers as f64;
                                if i_free {
                                    scene.voxels[c.i].add_internal_energy_j(heat_per_voxel);
                                }
                                if j_free {
                                    scene.voxels[c.j].add_internal_energy_j(heat_per_voxel);
                                }
                                scene.energy.contact_friction_heat += friction_heat;
                            }
                        }
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    normal_impulses
}

/// Apply a viscous shear impulse to contacting liquid-bearing REVs. The
/// equal-and-opposite impulse conserves momentum; its exact kinetic loss is
/// returned to the free endpoints as internal heat.
pub(crate) fn resolve_liquid_viscosity(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f32>; 3]],
    elapsed_seconds: f32,
) {
    if elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return;
    }
    for contact in contacts {
        let i = contact.i;
        let j = contact.j;
        let mu_i = scene.properties[i].dynamic_viscosity_pa_s;
        let mu_j = scene.properties[j].dynamic_viscosity_pa_s;
        if mu_i <= 0.0 || mu_j <= 0.0 {
            continue;
        }
        let phase_i = scene.properties[i]
            .phase_state(scene.voxels[i].mass, scene.voxels[i].thermal_energy_f32());
        let phase_j = scene.properties[j]
            .phase_state(scene.voxels[j].mass, scene.voxels[j].thermal_energy_f32());
        let liquid_fraction = phase_i.liquid_fraction.min(phase_j.liquid_fraction);
        if liquid_fraction <= 0.0 {
            continue;
        }
        let viscosity = (2.0 * mu_i * mu_j / (mu_i + mu_j)) * liquid_fraction;
        let normal_axis = normal_axis(contact.normal);
        let delta = scene.positions[j] - scene.positions[i];
        let area = (0..3)
            .filter(|&axis| axis != normal_axis)
            .map(|axis| (CELL - delta[axis].abs()).clamp(0.0, CELL))
            .product::<f32>();
        if area <= 0.0 {
            continue;
        }
        let separation = delta.dot(contact.normal).abs().max(CELL);
        let relative = scene.voxels[j].velocity - scene.voxels[i].velocity;
        let tangent = relative - contact.normal * relative.dot(contact.normal);
        let tangent_speed = tangent.length();
        if tangent_speed <= 1e-9 {
            continue;
        }
        let mut direction = tangent / tangent_speed;
        let response: [f32; 3] = std::array::from_fn(|axis| {
            inverse_mass(scene, i, axis, targets) + inverse_mass(scene, j, axis, targets)
        });
        direction.x *= (response[0] > 0.0) as u8 as f32;
        direction.y *= (response[1] > 0.0) as u8 as f32;
        direction.z *= (response[2] > 0.0) as u8 as f32;
        let responsive_speed = direction.length() * tangent_speed;
        if responsive_speed <= 1e-9 {
            continue;
        }
        direction = direction.normalize();
        let inverse_effective_mass = (0..3)
            .map(|axis| direction[axis].powi(2) * response[axis])
            .sum::<f32>();
        if inverse_effective_mass <= 0.0 {
            continue;
        }
        let requested_impulse = viscosity * area / separation * elapsed_seconds * responsive_speed;
        let impulse_magnitude = requested_impulse.min(responsive_speed / inverse_effective_mass);
        let impulse = direction * impulse_magnitude;
        for axis in 0..3 {
            scene.voxels[i].velocity[axis] += impulse[axis] * inverse_mass(scene, i, axis, targets);
            scene.voxels[j].velocity[axis] -= impulse[axis] * inverse_mass(scene, j, axis, targets);
        }
        let heat_j = (impulse_magnitude as f64 * responsive_speed as f64
            - 0.5 * (impulse_magnitude as f64).powi(2) * inverse_effective_mass as f64)
            .max(0.0);
        if heat_j > 0.0 {
            let i_free = (0..3).any(|axis| inverse_mass(scene, i, axis, targets) > 0.0);
            let j_free = (0..3).any(|axis| inverse_mass(scene, j, axis, targets) > 0.0);
            let receivers = i_free as u8 + j_free as u8;
            if i_free {
                scene.voxels[i].add_internal_energy_j(heat_j / receivers as f64);
            }
            if j_free {
                scene.voxels[j].add_internal_energy_j(heat_j / receivers as f64);
            }
            scene.energy.liquid_viscous_heat += heat_j;
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
    // Expand each swept voxel by the same micrometre-scale shell used to
    // retain resting contacts after positional projection.
    let margin = 2e-4;
    let lo = (a - Vec3::splat(0.5 + margin)).floor().as_ivec3();
    let hi = (b + Vec3::splat(0.5 + margin)).floor().as_ivec3();
    (lo.to_array(), hi.to_array())
}

fn touching_normal(a: Vec3, b: Vec3) -> Option<Vec3> {
    const CONTACT_SLOP: f32 = 2.0e-6;
    let delta = b - a;
    let gaps = delta.abs() - Vec3::splat(CELL);
    let axis = if gaps.x >= gaps.y && gaps.x >= gaps.z {
        0
    } else if gaps.y >= gaps.z {
        1
    } else {
        2
    };
    if gaps[axis] > CONTACT_SLOP
        || (0..3).any(|other| other != axis && gaps[other] >= -CONTACT_SLOP)
    {
        return None;
    }
    let mut normal = Vec3::ZERO;
    normal[axis] = if delta[axis] >= 0.0 { 1.0 } else { -1.0 };
    Some(normal)
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
    fn sliding_contact_dissipates_tangent_energy_into_voxel_heat() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.fracture.friction = 0.35;
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.set_voxel_kinematic(1, true).unwrap();
        scene.previous.clone_from(&scene.positions);
        scene.voxels[0].velocity = Vec3::new(1.0, 2.0, 5.0);
        scene.positions[0].x += CELL * 0.2;
        let targets = [[None, None, Some(scene.positions[0].z)], [None; 3]];
        let contacts = project(&mut scene, &targets);
        assert!(contacts
            .iter()
            .any(|contact| contact.i == 0 && contact.j == 1));

        let mass = scene.voxels[0].mass as f64;
        let tangent_ke_before = 0.5 * mass * (scene.voxels[0].velocity.y as f64).powi(2);
        let internal_before = scene.voxels[0].total_internal_energy_j();
        resolve_velocities(&mut scene, &contacts, &targets);

        let tangent_ke_after = 0.5 * mass * (scene.voxels[0].velocity.y as f64).powi(2);
        let dissipated = tangent_ke_before - tangent_ke_after;
        let internal_gain = scene.voxels[0].total_internal_energy_j() - internal_before;
        let total_contact_heat =
            scene.energy.contact_friction_heat + scene.energy.contact_restitution_heat;
        let internal_storage_tolerance = 1e-12;
        assert!(scene.voxels[0].velocity.y.abs() < 2.0);
        assert_eq!(scene.voxels[0].velocity.z, 5.0);
        assert!(dissipated > 0.0);
        assert!((dissipated - scene.energy.contact_friction_heat).abs() < 1e-7);
        assert!(
            (internal_gain - total_contact_heat).abs() <= internal_storage_tolerance,
            "internal gain={internal_gain:.12e}, contact heat={total_contact_heat:.12e}, initial internal={internal_before:.12e}, tolerance={internal_storage_tolerance:.12e}"
        );
    }

    #[test]
    fn resting_contact_under_compression_does_not_accumulate_impact_heat() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.fixed_dt = 0.01;
        config.substeps = 2;
        config.iterations = 1;
        config.corotation_iterations = 1;
        config.fracture.friction = 0.0;
        config.fracture.restitution = 0.0;
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.voxels[0].velocity = Vec3::ZERO;
        scene.voxels[1].velocity = Vec3::ZERO;
        scene.workspace[0][0].rest_neighbor = EMPTY;
        solver::set_force(&mut scene, 0, Vec3::X * 0.1).unwrap();
        solver::set_force(&mut scene, 1, Vec3::NEG_X * 0.1).unwrap();
        for _ in 0..20 {
            solver::step(&mut scene, &mut None);
        }
        assert!((scene.positions[1] - scene.positions[0]).length() <= CELL + 2e-6);
        assert_eq!(scene.energy.contact_restitution_heat, 0.0);
    }

    #[test]
    fn normal_restitution_loss_is_converted_to_voxel_heat() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.fracture.friction = 0.0;
        config.fracture.restitution = 0.25;
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.workspace[0][0].rest_neighbor = EMPTY;
        scene.voxels[0].velocity = Vec3::X;
        scene.voxels[1].velocity = -Vec3::X;
        let initial_momentum = scene.voxels[0].mass * scene.voxels[0].velocity
            + scene.voxels[1].mass * scene.voxels[1].velocity;
        let initial_kinetic = scene
            .voxels
            .iter()
            .map(|voxel| 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64)
            .sum::<f64>();
        let initial_internal = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum::<f64>();
        let contact = Contact {
            i: 0,
            j: 1,
            normal: Vec3::X,
            new_impact: true,
        };

        resolve_velocities(&mut scene, &[contact], &[]);

        let final_momentum = scene.voxels[0].mass * scene.voxels[0].velocity
            + scene.voxels[1].mass * scene.voxels[1].velocity;
        let final_kinetic = scene
            .voxels
            .iter()
            .map(|voxel| 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64)
            .sum::<f64>();
        let internal_gain = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum::<f64>()
            - initial_internal;
        let dissipated = initial_kinetic - final_kinetic;

        assert!((final_momentum - initial_momentum).length() < 1e-6);
        assert!((scene.voxels[1].velocity.x - scene.voxels[0].velocity.x - 0.5).abs() < 1e-6);
        assert!(dissipated > 0.0);
        assert!((dissipated - scene.energy.contact_restitution_heat).abs() < 1e-8);
        let storage_tolerance = 1e-12;
        assert!(
            (internal_gain - scene.energy.contact_restitution_heat).abs() <= storage_tolerance,
            "internal gain={internal_gain:.12e}, restitution heat={:.12e}, tolerance={storage_tolerance:.12e}",
            scene.energy.contact_restitution_heat
        );
        assert_eq!(scene.energy.contact_friction_heat, 0.0);
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
    fn touching_contact_persists_after_projection_slop() {
        let iron = crate::material::Material::component_id("iron").unwrap();
        let mut scene = crate::scene::SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(crate::scene::VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(crate::scene::VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        scene.release_voxel(0).unwrap();
        scene.positions[1].x += 0.005;
        scene.previous[1] = scene.positions[1];
        scene.voxels[0].velocity = Vec3::X * 0.01;
        scene.voxels[1].velocity = Vec3::NEG_X * 0.01;
        scene.positions[0].x += 0.002503;
        scene.positions[1].x -= 0.002503;
        let targets = vec![[None; 3]; 2];

        let impact = project(&mut scene, &targets);
        assert_eq!(impact.len(), 1);
        assert!(impact[0].new_impact);
        resolve_velocities(&mut scene, &impact, &targets);
        scene.previous.clone_from(&scene.positions);

        let resting = project(&mut scene, &targets);
        assert_eq!(resting.len(), 1);
        assert!(!resting[0].new_impact);
        assert!((scene.positions[1] - scene.positions[0]).length() <= CELL + 2e-6);
    }

    #[test]
    fn liquid_contact_viscosity_conserves_momentum_and_converts_loss_to_heat() {
        let water = crate::material::Material::component_id("water").unwrap();
        let mut scene = crate::scene::SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(crate::scene::VoxelSpawn::at_cell([0, 0, 0], water))
            .add_voxel(crate::scene::VoxelSpawn::at_cell([1, 0, 0], water))
            .build()
            .unwrap();
        scene.set_material_dynamic_viscosity("water", 1.0).unwrap();
        scene.voxels[0].velocity = Vec3::Y * 0.01;
        scene.voxels[1].velocity = Vec3::NEG_Y * 0.01;
        let targets = vec![[None; 3]; 2];
        let contact = Contact {
            i: 0,
            j: 1,
            normal: Vec3::X,
            new_impact: false,
        };
        let momentum_before = scene.voxels[0].velocity * scene.voxels[0].mass
            + scene.voxels[1].velocity * scene.voxels[1].mass;
        let energy_before: f64 = scene
            .voxels
            .iter()
            .map(|voxel| {
                voxel.total_internal_energy_j()
                    + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
            })
            .sum();
        let relative_before = (scene.voxels[1].velocity - scene.voxels[0].velocity).length();

        resolve_liquid_viscosity(&mut scene, &[contact], &targets, 0.01);

        let momentum_after = scene.voxels[0].velocity * scene.voxels[0].mass
            + scene.voxels[1].velocity * scene.voxels[1].mass;
        let energy_after: f64 = scene
            .voxels
            .iter()
            .map(|voxel| {
                voxel.total_internal_energy_j()
                    + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
            })
            .sum();
        let relative_after = (scene.voxels[1].velocity - scene.voxels[0].velocity).length();
        assert!(relative_after < relative_before);
        assert!((momentum_after - momentum_before).length() < 1e-10);
        assert!((energy_after - energy_before).abs() < 1e-12);
        assert!(scene
            .voxels
            .iter()
            .any(|voxel| voxel.internal_energy_remainder_j > 0.0));
        assert!(scene.energy.liquid_viscous_heat > 0.0);
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
