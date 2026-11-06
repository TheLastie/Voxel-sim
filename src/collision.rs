use crate::scene::{sync_visited, visit, Scene, CELL, DCELL};
use glam::{DVec3, Vec3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

const COLLISION_BIN_SCALE: f32 = 1.05;
// Normal swept REV bounds occupy only a few dozen bins. Larger bounds are
// handled by the sparse sweep below so one fast or corrupted REV cannot make
// the hash broadphase allocate in proportion to the swept world volume.
const MAX_BINS_PER_REV: u128 = 4096;

fn collision_bin_scale() -> f32 {
    static SCALE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("VOXEL_COLLISION_BIN_SCALE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value >= 0.25 && *value <= 8.0)
            .unwrap_or(COLLISION_BIN_SCALE)
    })
}

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
    starts: Vec<DVec3>,
    bins: FxHashMap<[i32; 3], Vec<usize>>,
    bin_pool: Vec<Vec<usize>>,
    static_bins: FxHashMap<[i32; 3], Vec<usize>>,
    static_indices: Vec<usize>,
    static_positions: Vec<DVec3>,
    dynamic_lower_by_index: Vec<[i32; 3]>,
    dynamic_bounds_by_index: Vec<Option<([i32; 3], [i32; 3])>>,
    static_lower_by_index: Vec<[i32; 3]>,
    static_ready: bool,
    /// Slot of each fixed REV in `static_indices` (`u32::MAX` for the others), the visit
    /// list the bins were built for, and how many fixed REVs that list holds. A sparse
    /// step checks only those REVs: the fixed ones outside it are sleepers, which are
    /// frozen and cannot make the bins stale without a sleep or wake event.
    static_slot: Vec<u32>,
    static_live_id: u64,
    static_live_kinematic: usize,
    static_requires_sparse: bool,
    candidate_pairs_valid: bool,
    pairs: Vec<(usize, usize)>,
    adjacency: Vec<[u32; 6]>,
    adjacency_topology: [Vec<u32>; 3],
    /// The visit list whose bonds the adjacency cache was last compared against in full.
    adjacency_live_id: u64,
}

/// Project voxel pairs apart and detect swept contacts before they can cross.
/// The intact local continuum stencil is governed by the constitutive solver.
/// Treating its edge/corner neighbors as independent AABBs creates artificial
/// strain under rigid rotation. Broken face neighbors and nonlocal folded
/// regions still use REV-sized contact bounds.
pub fn project(scene: &mut Scene, targets: &[[Option<f64>; 3]]) -> Vec<Contact> {
    let mut contacts = Vec::new();
    project_into(scene, targets, &mut contacts);
    contacts
}

pub(crate) fn project_into(
    scene: &mut Scene,
    targets: &[[Option<f64>; 3]],
    contacts: &mut Vec<Contact>,
) {
    contacts_into(scene, targets, contacts, true);
}

/// Discover the manifold without adding a second, unaccounted displacement
/// after the implicit mechanical solve.
pub(crate) fn detect_into(scene: &mut Scene, targets: &[[Option<f64>; 3]], contacts: &mut Vec<Contact>) {
    contacts_into(scene, targets, contacts, false);
}

fn contacts_into(scene: &mut Scene, targets: &[[Option<f64>; 3]], contacts: &mut Vec<Contact>, project: bool) {
    let profiling = std::env::var_os("VOXEL_PROFILE_COLLISION").is_some();
    let mut phase_start = Instant::now();
    let mut scratch = std::mem::take(&mut scene.collision_scratch);
    let mut starts = std::mem::take(&mut scratch.starts);
    let live = scene.step_live.clone();
    sync_visited(&mut starts, &scene.previous, &live);
    if skip_separated_aggregate_collision(scene, &starts, &mut scratch, contacts) {
        scratch.starts = starts;
        scene.collision_scratch = scratch;
        return;
    }
    let topology_changed = adjacency(
        scene,
        &mut scratch.adjacency,
        &mut scratch.adjacency_topology,
        &mut scratch.adjacency_live_id,
    );
    let setup_time = phase_start.elapsed();
    phase_start = Instant::now();
    candidate_pairs(scene, &starts, &mut scratch, topology_changed);
    let candidate_time = phase_start.elapsed();
    let candidate_count = scratch.pairs.len();
    contacts.clear();
    contacts.reserve(candidate_count);
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
                if internal_kinematic_face_contact(
                    scene,
                    &starts,
                    targets,
                    &scratch.adjacency,
                    i,
                    j,
                    normal,
                ) {
                    continue;
                }
                let separation = (scene.positions[j] - scene.positions[i]).dot(normal.as_dvec3());
                if project && separation < DCELL {
                    // Do not add a visible separation gap to bonded material.
                    // A 1 um offset on a 1 cm stiff REV becomes real bond strain
                    // at fracture evaluation and creates repeated support work.
                    let correction = (DCELL - separation) as f32;
                    changed |= separate(scene, i, j, normal, correction, targets);
                }
                contacts.push(Contact {
                    i,
                    j,
                    normal,
                    new_impact: (starts[j] - starts[i]).dot(normal.as_dvec3()) > DCELL + 2e-6,
                });
                continue;
            }
            if let Some((normal, penetration)) = overlap_normal(
                scene.positions[i],
                scene.positions[j],
                starts[j] - starts[i],
            ) {
                if internal_kinematic_face_contact(
                    scene,
                    &starts,
                    targets,
                    &scratch.adjacency,
                    i,
                    j,
                    normal,
                ) {
                    continue;
                }
                if project { changed |= separate(scene, i, j, normal, penetration, targets); }
                contacts.push(Contact {
                    i,
                    j,
                    normal,
                    new_impact: (starts[j] - starts[i]).dot(normal.as_dvec3()) > DCELL + 2e-6,
                });
                continue;
            }
            if let Some(normal) = touching_normal(scene.positions[i], scene.positions[j]) {
                if internal_kinematic_face_contact(
                    scene,
                    &starts,
                    targets,
                    &scratch.adjacency,
                    i,
                    j,
                    normal,
                ) {
                    continue;
                }
                contacts.push(Contact {
                    i,
                    j,
                    normal,
                    new_impact: (starts[j] - starts[i]).dot(normal.as_dvec3()) > DCELL + 2e-6,
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
            "collision profile ms: setup={:.2} broadphase_and_filter={:.2} project={:.2} dedup={:.2} candidates={candidate_count} contacts={}",
            setup_time.as_secs_f64() * 1000.0,
            candidate_time.as_secs_f64() * 1000.0,
            projection_time.as_secs_f64() * 1000.0,
            phase_start.elapsed().as_secs_f64() * 1000.0,
            contacts.len()
        );
    }
    scratch.starts = starts;
    scene.collision_scratch = scratch;
}

#[derive(Clone, Copy)]
struct AggregateSweepBounds {
    representative: usize,
    min: DVec3,
    max: DVec3,
}

#[derive(Clone, Copy)]
struct StepSweepBounds {
    min: DVec3,
    max: DVec3,
}

/// Conservative whole-step contact forecast for the coarse rigid tier.
/// Current member AABBs are inflated for translation and bounded rotation.
pub(crate) fn rigid_candidates_clear_for_step(
    scene: &Scene,
    dt_s: f64,
    gravity_m_s2: DVec3,
    nodal_loads_n: &[Vec3],
) -> bool {
    if !dt_s.is_finite()
        || dt_s < 0.0
        || !gravity_m_s2.is_finite()
        || (!nodal_loads_n.is_empty() && nodal_loads_n.len() != scene.voxels.len())
        || scene.aggregate_of.len() != scene.voxels.len()
        || scene.positions.len() != scene.voxels.len()
        || scene.external_forces.len() != scene.voxels.len()
        || scene.external_accelerations.len() != scene.voxels.len()
        || scene.axis_targets.len() != scene.voxels.len()
    {
        return false;
    }

    let mut dynamic_count = 0usize;
    for (index, voxel) in scene.voxels.iter().enumerate() {
        if voxel.kinematic {
            if scene.aggregate_of[index].is_some() {
                return false;
            }
            continue;
        }
        dynamic_count += 1;
        if scene.aggregate_of[index].is_none()
            || scene.axis_targets[index].iter().any(Option::is_some)
            || !voxel.mass.is_finite()
            || voxel.mass <= 0.0
            || !scene.positions[index].is_finite()
        {
            return false;
        }
    }
    if dynamic_count == 0 {
        return false;
    }

    let mut total_members = 0usize;
    let mut body_bounds = Vec::with_capacity(scene.rigid_aggregates.len());
    for (id, aggregate) in scene.rigid_aggregates.iter().enumerate() {
        let Some(aggregate) = aggregate.as_ref() else {
            continue;
        };
        let center = aggregate.center_of_mass_m();
        let radius = aggregate.bounding_radius_m();
        if !center.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || aggregate.has_unresolved_internal_motion()
        {
            return false;
        }

        let mut max_speed = 0.0_f64;
        let mut max_acceleration = 0.0_f64;
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut torque = DVec3::ZERO;
        let mut torque_abs_sum = 0.0_f64;
        let mut member_count = 0usize;
        for member in aggregate.member_poses() {
            let index = member.index;
            if index >= scene.voxels.len()
                || scene.aggregate_of[index] != Some(id)
                || scene.voxels[index].kinematic
                || !member.position_m.is_finite()
                || !member.velocity_m_s.is_finite()
                || !member.mass_kg.is_finite()
                || member.mass_kg <= 0.0
                || scene.axis_targets[index].iter().any(Option::is_some)
            {
                return false;
            }
            let external_acceleration = scene.external_accelerations[index];
            let external_force = scene.external_forces[index];
            let nodal_force = nodal_loads_n.get(index).copied().unwrap_or(Vec3::ZERO);
            if !external_acceleration.is_finite()
                || !external_force.is_finite()
                || !nodal_force.is_finite()
            {
                return false;
            }

            // Persistent external fields are not part of the frozen wrench
            // used by advance_registered_rigid_candidates; require their
            // earlier solver wake path to refine this body first.
            if external_acceleration != Vec3::ZERO || external_force != Vec3::ZERO {
                return false;
            }
            let force = gravity_m_s2 * member.mass_kg
                + DVec3::new(
                    nodal_force.x as f64,
                    nodal_force.y as f64,
                    nodal_force.z as f64,
                );
            let acceleration = force / member.mass_kg;
            if !acceleration.is_finite() {
                return false;
            }
            max_speed = max_speed.max(member.velocity_m_s.length());
            max_acceleration = max_acceleration.max(acceleration.length());
            let arm = member.position_m - center;
            torque += arm.cross(force);
            torque_abs_sum += arm.length() * force.length();
            let half_extent = DVec3::splat(0.5 * DCELL as f64);
            min = min.min(member.position_m - half_extent);
            max = max.max(member.position_m + half_extent);
            member_count += 1;
        }
        if member_count == 0 || member_count != aggregate.member_count() {
            return false;
        }
        total_members = match total_members.checked_add(member_count) {
            Some(total) => total,
            None => return false,
        };

        let inverse_inertia_norm = aggregate.inverse_inertia_frobenius_norm();
        let angular_momentum = aggregate.angular_momentum_kg_m2_s();
        let torque_error = 128.0 * f64::EPSILON * torque_abs_sum;
        let torque_bound = torque.length() + torque_error;
        if !inverse_inertia_norm.is_finite()
            || inverse_inertia_norm <= 0.0
            || !angular_momentum.is_finite()
            || !torque_bound.is_finite()
        {
            return false;
        }
        let angular_speed_bound =
            inverse_inertia_norm * (angular_momentum.length() + torque_bound * dt_s);
        let rotation_travel = radius * (angular_speed_bound * dt_s).min(2.0);
        let translation_travel = max_speed * dt_s + max_acceleration * dt_s * dt_s;
        let travel = translation_travel + rotation_travel + DCELL as f64;
        if !travel.is_finite() || !min.is_finite() || !max.is_finite() {
            return false;
        }
        let inflation = DVec3::splat(travel);
        let swept_min = min - inflation;
        let swept_max = max + inflation;
        if !swept_min.is_finite() || !swept_max.is_finite() {
            return false;
        }
        body_bounds.push(StepSweepBounds {
            min: swept_min,
            max: swept_max,
        });
    }
    if total_members != dynamic_count || body_bounds.is_empty() {
        return false;
    }

    if let Some(world_bounds) = scene.world_bounds() {
        let bounds_min = DVec3::new(
            world_bounds.min.x as f64,
            world_bounds.min.y as f64,
            world_bounds.min.z as f64,
        );
        let bounds_max = DVec3::new(
            world_bounds.max.x as f64,
            world_bounds.max.y as f64,
            world_bounds.max.z as f64,
        );
        if body_bounds
            .iter()
            .any(|body| body.min.cmplt(bounds_min).any() || body.max.cmpgt(bounds_max).any())
        {
            return false;
        }
    }

    for (index, voxel) in scene.voxels.iter().enumerate() {
        if !voxel.kinematic {
            continue;
        }
        let center = scene.positions[index];
        if !center.is_finite() {
            return false;
        }
        let half_extent = DVec3::splat(0.5 * DCELL as f64);
        let fixed_min = center - half_extent;
        let fixed_max = center + half_extent;
        if body_bounds
            .iter()
            .any(|body| body.min.cmple(fixed_max).all() && body.max.cmpge(fixed_min).all())
        {
            return false;
        }
    }

    for index in 0..body_bounds.len() {
        for other in index + 1..body_bounds.len() {
            let a = body_bounds[index];
            let b = body_bounds[other];
            if a.min.cmple(b.max).all() && a.max.cmpge(b.min).all() {
                return false;
            }
        }
    }
    true
}

/// Avoid rebuilding per-REV collision bins when every dynamic REV belongs to
/// a coarse body and no swept body bounds can meet. Overlapping coarse bounds
/// wake those bodies first; the caller then runs the unchanged REV-level CCD.
fn skip_separated_aggregate_collision(
    scene: &mut Scene,
    starts: &[DVec3],
    scratch: &mut CollisionScratch,
    contacts: &mut Vec<Contact>,
) -> bool {
    if !scene.has_registered_rigid_candidates() || scene.voxels.is_empty() {
        return false;
    }
    let mut dynamic_count = 0usize;
    for (index, voxel) in scene.voxels.iter().enumerate() {
        if !voxel.kinematic {
            dynamic_count += 1;
            if scene.aggregate_of.get(index).copied().flatten().is_none() {
                return false;
            }
        }
    }

    // Retain static REV contacts too. The fixed-bin cache is already the
    // narrow broadphase for kinematic geometry, so query it with each coarse
    // swept bound instead of rebuilding bins for every dynamic REV.
    if dynamic_count < scene.voxels.len() {
        ensure_static_bins(scene, starts, scratch);
        if scratch.static_requires_sparse {
            return false;
        }
    }

    let half_extent = DVec3::splat(DCELL * 0.5 + 2.0e-6);
    let mut total_members = 0usize;
    let mut bounds = Vec::with_capacity(scene.rigid_aggregates.len());
    for (id, aggregate) in scene.rigid_aggregates.iter().enumerate() {
        let Some(aggregate) = aggregate.as_ref() else {
            continue;
        };
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut representative = None;
        for index in aggregate.member_indices() {
            if scene.aggregate_of.get(index).copied().flatten() != Some(id)
                || scene.voxels[index].kinematic
            {
                return false;
            }
            representative.get_or_insert(index);
            total_members += 1;
            for center in [starts[index], scene.positions[index]] {
                min = min.min(center - half_extent);
                max = max.max(center + half_extent);
            }
        }
        let Some(representative) = representative else {
            continue;
        };
        bounds.push(AggregateSweepBounds {
            representative,
            min,
            max,
        });
    }
    if total_members != dynamic_count {
        return false;
    }

    bounds.sort_unstable_by(|a, b| a.min.x.total_cmp(&b.min.x));
    let mut possibly_contacting = Vec::new();
    for (i, a) in bounds.iter().enumerate() {
        for b in bounds.iter().skip(i + 1) {
            if b.min.x > a.max.x {
                break;
            }
            if a.max.y >= b.min.y && b.max.y >= a.min.y && a.max.z >= b.min.z && b.max.z >= a.min.z
            {
                possibly_contacting.push(a.representative);
                possibly_contacting.push(b.representative);
            }
        }

        let (lo, hi) = bin_bounds(a.min / DCELL, a.max / DCELL);
        if bin_volume_exceeds_limit(lo, hi) {
            return false;
        }
        'query: for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    if scratch.static_bins.contains_key(&[x, y, z]) {
                        possibly_contacting.push(a.representative);
                        break 'query;
                    }
                }
            }
        }
    }

    if possibly_contacting.is_empty() {
        scratch.candidate_pairs_valid = false;
        scratch.pairs.clear();
        contacts.clear();
        return true;
    }

    scene.wake_rigid_candidates_for_voxels(&possibly_contacting);
    false
}

/// Remove closing normal velocity after position projection. Restitution is
/// applied once at the contact pair; separating motion is left unchanged.
pub fn resolve_velocities(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f64>; 3]],
) -> Vec<f32> {
    let pre_contact_velocities: Vec<_> = scene.voxels.iter().map(|voxel| voxel.velocity).collect();
    resolve_velocities_with_precontact(scene, contacts, targets, &pre_contact_velocities)
}

pub(crate) fn resolve_velocities_with_precontact(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f64>; 3]],
    pre_contact_velocities: &[Vec3],
) -> Vec<f32> {
    let mut normal_impulses = Vec::with_capacity(contacts.len());
    let mut physical_velocities = Vec::with_capacity(pre_contact_velocities.len());
    let mut friction_used = Vec::new();
    resolve_velocities_with_precontact_into(
        scene,
        contacts,
        targets,
        pre_contact_velocities,
        &mut normal_impulses,
        &mut physical_velocities,
        &[],
        &mut friction_used,
    );
    normal_impulses
}

pub(crate) fn resolve_velocities_with_precontact_into(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f64>; 3]],
    pre_contact_velocities: &[Vec3],
    normal_impulses: &mut Vec<f32>,
    physical_velocities: &mut Vec<Vec3>,
    implicit_friction: &[bool],
    friction_used: &mut Vec<f32>,
) {
    normal_impulses.resize(contacts.len(), 0.0);
    // One Coulomb budget per contact for the whole substep: the passes below
    // redistribute it between contacts, they must not grant it again.
    friction_used.clear();
    friction_used.resize(contacts.len(), 0.0);
    // Without a solved reaction the support load is estimated from a REV's own
    // closing speed; several contacts of that REV share the same load.
    let mut contacts_per_voxel_axis = FxHashMap::<(usize, usize), u32>::default();
    for (index, contact) in contacts.iter().enumerate() {
        if implicit_friction.get(index).copied().unwrap_or(false) {
            continue;
        }
        let axis = normal_axis(contact.normal);
        for voxel in [contact.i, contact.j] {
            if !scene.voxels[voxel].kinematic {
                *contacts_per_voxel_axis.entry((voxel, axis)).or_default() += 1;
            }
        }
    }
    sync_visited(physical_velocities, pre_contact_velocities, &scene.step_live.clone());
    for iteration in 0..4 {
        let mut changed = false;
        for (contact_index, c) in contacts.iter().enumerate() {
            let axis = normal_axis(c.normal);
            let wi = inverse_mass(scene, c.i, axis, targets);
            let wj = inverse_mass(scene, c.j, axis, targets);
            let denom = wi + wj;
            let closing = (scene.voxels[c.j].velocity - scene.voxels[c.i].velocity).dot(c.normal);
            let physical_closing =
                (physical_velocities[c.j] - physical_velocities[c.i]).dot(c.normal);
            // A positive impulse supplied by the coupled position/elastic
            // contact solve already handled this normal degree of freedom.
            // Keep it for friction and thermal welding, but do not apply a
            // second velocity impulse or count the same impact heat twice.
            let constrained_normal_impulse = normal_impulses[contact_index].max(0.0);
            let mut normal_impulse = constrained_normal_impulse;
            if constrained_normal_impulse <= 0.0 && denom > 0.0 && closing < -1e-6 {
                // A resting manifold must not bounce on each new gravity step.
                // Restitution applies only if this pair started with a gap.
                let restitution = if c.new_impact && iteration == 0 {
                    scene.config.fracture.restitution
                } else {
                    0.0
                };
                normal_impulse = -(1.0 + restitution) * closing / denom;
                normal_impulses[contact_index] = normal_impulse;
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
            let pair_friction = scene.config.fracture.friction
                * (scene.properties[c.i].surface_friction_coefficient
                    + scene.properties[c.j].surface_friction_coefficient);
            // A fixed support's friction was already solved in the implicit system.
            let friction_is_implicit = implicit_friction.get(contact_index).copied().unwrap_or(false);
            let mut friction_normal_impulse = if friction_is_implicit { 0.0 } else { normal_impulse };
            if !friction_is_implicit && friction_normal_impulse <= 0.0 && physical_closing < -1e-6 {
                // A resting floor contact is position constrained by the
                // mechanical solve. That removes its normal velocity before
                // this impulse pass, but gravity still creates a real support
                // load for Coulomb friction. Estimate only that load here;
                // the normal velocity is already constrained and must not be
                // corrected a second time.
                let unconstrained_inverse_mass = |voxel: usize| {
                    if scene.voxels[voxel].kinematic {
                        0.0
                    } else {
                        scene.voxels[voxel].mass.recip()
                    }
                };
                let support_denom =
                    unconstrained_inverse_mass(c.i) + unconstrained_inverse_mass(c.j);
                if support_denom > 0.0 {
                    let sharing = [c.i, c.j]
                        .iter()
                        .filter_map(|&voxel| contacts_per_voxel_axis.get(&(voxel, axis)))
                        .copied()
                        .max()
                        .unwrap_or(1)
                        .max(1);
                    friction_normal_impulse = -physical_closing / support_denom / sharing as f32;
                }
            }
            if friction_normal_impulse > 0.0 && pair_friction > 0.0 {
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
                    let friction_budget = (pair_friction * friction_normal_impulse
                        - friction_used[contact_index])
                        .max(0.0);
                    if tangent_weight > 0.0 && friction_budget > 0.0 {
                        let impulse_magnitude =
                            (tangent_speed / tangent_weight).min(friction_budget);
                        friction_used[contact_index] += impulse_magnitude;
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
}

/// Apply a viscous shear impulse to contacting liquid-bearing REVs. The
/// equal-and-opposite impulse conserves momentum; its exact kinetic loss is
/// returned to the free endpoints as internal heat.
pub(crate) fn resolve_liquid_viscosity(
    scene: &mut Scene,
    contacts: &[Contact],
    targets: &[[Option<f64>; 3]],
    elapsed_seconds: f32,
) {
    if elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return;
    }
    for contact in contacts {
        let i = contact.i;
        let j = contact.j;
        let phase_i = scene.properties[i]
            .phase_state(scene.voxels[i].mass, scene.voxels[i].thermal_energy_f32());
        let phase_j = scene.properties[j]
            .phase_state(scene.voxels[j].mass, scene.voxels[j].thermal_energy_f32());
        let liquid_i = phase_i.liquid_fraction.clamp(0.0, 1.0);
        let liquid_j = phase_j.liquid_fraction.clamp(0.0, 1.0);
        let viscosity = match (liquid_i > 0.0, liquid_j > 0.0) {
            (true, true) => {
                let mu_i =
                    scene.properties[i].liquid_viscosity_at_temperature(phase_i.temperature_kelvin);
                let mu_j =
                    scene.properties[j].liquid_viscosity_at_temperature(phase_j.temperature_kelvin);
                if mu_i <= 0.0 || mu_j <= 0.0 {
                    continue;
                }
                (2.0 * mu_i * mu_j / (mu_i + mu_j)) * liquid_i.min(liquid_j)
            }
            (true, false) => {
                scene.properties[i].liquid_viscosity_at_temperature(phase_i.temperature_kelvin)
                    * liquid_i
            }
            (false, true) => {
                scene.properties[j].liquid_viscosity_at_temperature(phase_j.temperature_kelvin)
                    * liquid_j
            }
            (false, false) => continue,
        };
        let normal_axis = normal_axis(contact.normal);
        let delta = (scene.positions[j] - scene.positions[i]).as_vec3();
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

fn candidate_pairs(
    scene: &Scene,
    starts: &[DVec3],
    scratch: &mut CollisionScratch,
    topology_changed: bool,
) {
    let live = scene.step_live.clone();
    let static_changed = ensure_static_bins(scene, starts, scratch);
    let bounds_len_changed = scratch.dynamic_bounds_by_index.len() != scene.voxels.len();
    scratch
        .dynamic_bounds_by_index
        .resize(scene.voxels.len(), None);
    scratch
        .dynamic_lower_by_index
        .resize(scene.voxels.len(), [0; 3]);
    let mut dynamic_bounds_changed = bounds_len_changed;
    // A sleeper has no dynamic bounds (`None`, lower corner zero) from the start.
    for i in visit(&live, scene.voxels.len()) {
        let bounds = if scene.voxels[i].kinematic {
            None
        } else {
            let a = starts[i].min(scene.positions[i]) / DCELL;
            let b = starts[i].max(scene.positions[i]) / DCELL;
            Some(bin_bounds(a, b))
        };
        if scratch.dynamic_bounds_by_index[i] != bounds {
            dynamic_bounds_changed = true;
        }
        scratch.dynamic_bounds_by_index[i] = bounds;
        scratch.dynamic_lower_by_index[i] = bounds.map_or([0; 3], |(lo, _)| lo);
    }
    if scratch.candidate_pairs_valid
        && !static_changed
        && !dynamic_bounds_changed
        && !topology_changed
    {
        return;
    }

    let requires_sparse = scratch.static_requires_sparse
        || visit(&live, scene.voxels.len())
            .filter_map(|i| scratch.dynamic_bounds_by_index[i])
            .any(|(lo, hi)| bin_volume_exceeds_limit(lo, hi));
    if requires_sparse {
        candidate_pairs_sparse(scene, starts, scratch);
        scratch.candidate_pairs_valid = true;
        return;
    }

    for (_, mut ids) in scratch.bins.drain() {
        ids.clear();
        scratch.bin_pool.push(ids);
    }
    scratch.pairs.clear();
    let bins = &mut scratch.bins;
    let bin_pool = &mut scratch.bin_pool;
    for i in visit(&live, scene.voxels.len()) {
        let Some((lo, hi)) = &scratch.dynamic_bounds_by_index[i] else {
            continue;
        };
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
    let generated_pairs = {
        let lower_by_index = &scratch.dynamic_lower_by_index;
        let static_lower_by_index = &scratch.static_lower_by_index;
        let adjacent = &scratch.adjacency;
        let static_bins = &scratch.static_bins;
        // The pairs are sorted below, so the order bins are visited in does not matter.
        // A handful of bins (one falling body) is not worth waking the thread pool for.
        let fold_bin = |mut local_pairs: Vec<(usize, usize)>, (key, ids): (&[i32; 3], &Vec<usize>)| {
                for a in 0..ids.len() {
                    for b in (a + 1)..ids.len() {
                        let i = ids[a].min(ids[b]);
                        let j = ids[a].max(ids[b]);
                        if scene.aggregate_of[i].is_some()
                            && scene.aggregate_of[i] == scene.aggregate_of[j]
                        {
                            // Relative geometry inside an aggregate is rigid
                            // and cannot develop a new self-contact while it
                            // remains coarse. Any pre-existing contact blocks
                            // aggregation, and later external contact wakes it.
                            continue;
                        }
                        let lo_i = lower_by_index[i];
                        let lo_j = lower_by_index[j];
                        let canonical = [
                            lo_i[0].max(lo_j[0]),
                            lo_i[1].max(lo_j[1]),
                            lo_i[2].max(lo_j[2]),
                        ];
                        if *key == canonical && !local_continuum_pair(scene, adjacent, i, j) {
                            local_pairs.push((i, j));
                        }
                    }
                }
                if let Some(fixed_ids) = static_bins.get(key) {
                    for &dynamic_index in ids {
                        for &static_index in fixed_ids {
                            let (i, j) = (
                                dynamic_index.min(static_index),
                                dynamic_index.max(static_index),
                            );
                            // Keep source arrays associated with voxel classes even
                            // after sorting the emitted pair by scene index.
                            let lo_i = lower_by_index[dynamic_index];
                            let lo_j = static_lower_by_index[static_index];
                            let canonical = [
                                lo_i[0].max(lo_j[0]),
                                lo_i[1].max(lo_j[1]),
                                lo_i[2].max(lo_j[2]),
                            ];
                            if *key == canonical && !local_continuum_pair(scene, adjacent, i, j) {
                                local_pairs.push((i, j));
                            }
                        }
                    }
                }
                local_pairs
        };
        if bins.len() <= 64 {
            bins.iter().fold(Vec::new(), fold_bin)
        } else {
            bins.par_iter()
                .fold(Vec::new, fold_bin)
                .reduce(Vec::new, |mut a, mut b| {
                    a.append(&mut b);
                    a
                })
        }
    };
    scratch.pairs.clear();
    scratch.pairs.extend(generated_pairs);
    scratch.pairs.sort_unstable();
    scratch.candidate_pairs_valid = true;
}

#[derive(Clone, Copy)]
struct SweepBound {
    index: usize,
    lower: [i32; 3],
    upper: [i32; 3],
    dynamic: bool,
}

fn bin_volume_exceeds_limit(lower: [i32; 3], upper: [i32; 3]) -> bool {
    let mut volume = 1u128;
    for axis in 0..3 {
        let span = upper[axis] as i64 - lower[axis] as i64 + 1;
        if span <= 0 {
            return true;
        }
        let Some(next) = volume.checked_mul(span as u128) else {
            return true;
        };
        volume = next;
        if volume > MAX_BINS_PER_REV {
            return true;
        }
    }
    false
}

/// Exact swept-AABB broadphase fallback for exceptional bounds. It retains
/// every overlapping dynamic/dynamic or dynamic/static pair while using
/// memory proportional to REV count and actual candidate pairs, not bin volume.
fn candidate_pairs_sparse(scene: &Scene, starts: &[DVec3], scratch: &mut CollisionScratch) {
    let mut bounds = Vec::with_capacity(scene.voxels.len());
    for (index, voxel) in scene.voxels.iter().enumerate() {
        let (lower, upper, dynamic) = if voxel.kinematic {
            let a = starts[index].min(scene.positions[index]) / DCELL;
            let b = starts[index].max(scene.positions[index]) / DCELL;
            let (lower, upper) = bin_bounds(a, b);
            (lower, upper, false)
        } else {
            let Some((lower, upper)) = scratch.dynamic_bounds_by_index[index] else {
                continue;
            };
            (lower, upper, true)
        };
        bounds.push(SweepBound {
            index,
            lower,
            upper,
            dynamic,
        });
    }
    bounds.sort_unstable_by(|a, b| {
        a.lower[0]
            .cmp(&b.lower[0])
            .then_with(|| a.upper[0].cmp(&b.upper[0]))
    });

    scratch.pairs.clear();
    for i in 0..bounds.len() {
        let a = bounds[i];
        for &b in &bounds[i + 1..] {
            if b.lower[0] > a.upper[0] {
                break;
            }
            if (!a.dynamic && !b.dynamic)
                || a.upper[1] < b.lower[1]
                || b.upper[1] < a.lower[1]
                || a.upper[2] < b.lower[2]
                || b.upper[2] < a.lower[2]
            {
                continue;
            }
            let (first, second) = (a.index.min(b.index), a.index.max(b.index));
            if scene.aggregate_of[first].is_some()
                && scene.aggregate_of[first] == scene.aggregate_of[second]
            {
                continue;
            }
            if !local_continuum_pair(scene, &scratch.adjacency, first, second) {
                scratch.pairs.push((first, second));
            }
        }
    }
    scratch.pairs.sort_unstable();
    scratch.pairs.dedup();
}

fn bin_bounds(a: DVec3, b: DVec3) -> ([i32; 3], [i32; 3]) {
    // Expand each swept voxel by the same micrometre-scale shell used to
    // retain resting contacts after positional projection.
    let margin = 2e-4;
    let scale = collision_bin_scale() as f64;
    let lo = ((a - DVec3::splat(0.5 + margin)) * scale)
        .floor()
        .as_ivec3();
    let hi = ((b + DVec3::splat(0.5 + margin)) * scale)
        .floor()
        .as_ivec3();
    (lo.to_array(), hi.to_array())
}

fn touching_normal(a: DVec3, b: DVec3) -> Option<Vec3> {
    const CONTACT_SLOP: f32 = 2.0e-6;
    let delta = (b - a).as_vec3();
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

fn vectors_match_roundoff(a: DVec3, b: DVec3) -> bool {
    let scale = a.abs().max(b.abs()).max_element().max(1.0);
    (a - b).abs().max_element() <= 64.0 * f64::EPSILON * scale
}

fn stable_kinematic_position(
    scene: &Scene,
    starts: &[DVec3],
    targets: &[[Option<f64>; 3]],
    index: usize,
) -> bool {
    if index >= scene.voxels.len()
        || !scene.voxels[index].kinematic
        || !vectors_match_roundoff(starts[index], scene.positions[index])
    {
        return false;
    }
    for axis in 0..3 {
        let position = scene.positions[index][axis];
        if scene.axis_targets[index][axis].is_some_and(|target| {
            !vectors_match_roundoff(DVec3::splat(target), DVec3::splat(position))
        }) || targets
            .get(index)
            .and_then(|target| target[axis])
            .is_some_and(|target| {
                !vectors_match_roundoff(DVec3::splat(target), DVec3::splat(position))
            })
        {
            return false;
        }
    }
    true
}

/// Ignore a contact against an internal face of a stable tiled kinematic
/// surface. The adjacent fixed REV must occupy the whole face at both ends of
/// the sweep; exposed, deformed, or moving faces keep normal contact handling.
fn internal_kinematic_face_contact(
    scene: &Scene,
    starts: &[DVec3],
    targets: &[[Option<f64>; 3]],
    adjacency: &[[u32; 6]],
    i: usize,
    j: usize,
    contact_normal: Vec3,
) -> bool {
    if scene.voxels[i].kinematic == scene.voxels[j].kinematic {
        return false;
    }
    let (fixed, direction) = if scene.voxels[i].kinematic {
        (i, contact_normal.as_dvec3())
    } else {
        (j, -contact_normal.as_dvec3())
    };
    if !stable_kinematic_position(scene, starts, targets, fixed) {
        return false;
    }

    let axis = normal_axis(direction.as_vec3());
    let sign = direction[axis].signum();
    if sign == 0.0
        || (0..3).any(|other| other != axis && direction[other].abs() > 1e-6)
        || (direction[axis].abs() - 1.0).abs() > 1e-6
    {
        return false;
    }
    let side = 2 * axis + usize::from(sign > 0.0);
    let Some(&neighbor_u32) = adjacency
        .get(fixed)
        .and_then(|directions| directions.get(side))
    else {
        return false;
    };
    if neighbor_u32 == crate::bond::EMPTY {
        return false;
    }
    let neighbor = neighbor_u32 as usize;
    if neighbor == i
        || neighbor == j
        || !stable_kinematic_position(scene, starts, targets, neighbor)
    {
        return false;
    }

    let offset = direction * DCELL;
    vectors_match_roundoff(starts[neighbor], starts[fixed] + offset)
        && vectors_match_roundoff(scene.positions[neighbor], scene.positions[fixed] + offset)
}

fn ensure_static_bins(scene: &Scene, starts: &[DVec3], scratch: &mut CollisionScratch) -> bool {
    let live = scene.step_live.clone();
    let live_id = live.as_ref().map_or(0, |live| live.id);
    let n = scene.voxels.len();
    let mut changed = !scratch.static_ready;
    if let Some(live) = live.as_ref().filter(|_| !changed) {
        if scratch.static_live_id != live_id || scratch.static_slot.len() != n {
            changed = true;
        } else {
            let mut fixed_in_list = 0;
            for &i in &live.list {
                let slot = scratch.static_slot[i];
                if !scene.voxels[i].kinematic {
                    // Awake now, but the bins still hold it as fixed.
                    changed |= slot != u32::MAX;
                    continue;
                }
                fixed_in_list += 1;
                if slot == u32::MAX
                    || scratch.static_positions[slot as usize] != scene.positions[i]
                    || starts[i] != scene.positions[i]
                {
                    changed = true;
                }
            }
            changed |= fixed_in_list != scratch.static_live_kinematic;
        }
    } else if live.is_none() {
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
    }
    if !changed {
        return false;
    }
    scratch.static_live_id = live_id;
    scratch.static_slot.clear();
    scratch.static_slot.resize(n, u32::MAX);
    scratch.static_live_kinematic = 0;
    scratch.static_bins.clear();
    scratch.static_requires_sparse = false;
    scratch.static_indices.clear();
    scratch.static_positions.clear();
    scratch
        .static_lower_by_index
        .resize(scene.voxels.len(), [0; 3]);
    scratch.static_lower_by_index.fill([0; 3]);
    for (i, voxel) in scene.voxels.iter().enumerate() {
        if !voxel.kinematic {
            continue;
        }
        scratch.static_slot[i] = scratch.static_indices.len() as u32;
        if live.as_ref().map_or(true, |live| live.in_list[i]) {
            scratch.static_live_kinematic += 1;
        }
        scratch.static_indices.push(i);
        scratch.static_positions.push(scene.positions[i]);
        let a = starts[i].min(scene.positions[i]) / DCELL;
        let b = starts[i].max(scene.positions[i]) / DCELL;
        let (lo, hi) = bin_bounds(a, b);
        scratch.static_lower_by_index[i] = lo;
        if bin_volume_exceeds_limit(lo, hi) {
            scratch.static_requires_sparse = true;
            continue;
        }
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    scratch.static_bins.entry([x, y, z]).or_default().push(i);
                }
            }
        }
    }
    if scratch.static_requires_sparse {
        scratch.static_bins.clear();
    }
    scratch.static_ready = true;
    true
}

fn adjacency(
    scene: &Scene,
    result: &mut Vec<[u32; 6]>,
    cached_topology: &mut [Vec<u32>; 3],
    compared_live_id: &mut u64,
) -> bool {
    // The bonds of fixed REVs outside the visit list are compared in full whenever the
    // list is new; after that only the listed REVs can have changed.
    let live = scene.step_live.clone().filter(|live| live.id == *compared_live_id);
    *compared_live_id = scene.step_live.as_ref().map_or(0, |live| live.id);
    let topology_unchanged = result.len() == scene.voxels.len()
        && scene
            .bonds
            .iter()
            .zip(cached_topology.iter())
            .all(|(bonds, cached)| {
                bonds.len() == cached.len()
                    && match &live {
                        // Bonds of sleepers away from awake REVs cannot have changed.
                        Some(live) => live.list.iter().all(|&i| bonds[i].neighbor == cached[i]),
                        None => bonds
                            .iter()
                            .zip(cached)
                            .all(|(bond, &neighbor)| bond.neighbor == neighbor),
                    }
            });
    if topology_unchanged {
        return false;
    }

    result.resize(scene.voxels.len(), [crate::bond::EMPTY; 6]);
    result.fill([crate::bond::EMPTY; 6]);
    for axis in 0..3 {
        cached_topology[axis].clear();
        cached_topology[axis].reserve(scene.bonds[axis].len());
        for (i, bond) in scene.bonds[axis].iter().enumerate() {
            cached_topology[axis].push(bond.neighbor);
            if bond.neighbor != crate::bond::EMPTY {
                let j = bond.neighbor as usize;
                result[i][2 * axis + 1] = j as u32;
                result[j][2 * axis] = i as u32;
            }
        }
    }
    true
}

fn local_continuum_pair(scene: &Scene, adjacent: &[[u32; 6]], i: usize, j: usize) -> bool {
    let delta = (scene.rest_positions[j] - scene.rest_positions[i]) / DCELL;
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
    connected(adjacent, i, j, steps.as_vec3())
}

fn swept_normal(start_i: &DVec3, start_j: &DVec3, end_i: &DVec3, end_j: &DVec3) -> Option<Vec3> {
    let d0 = (*start_j - *start_i).as_vec3();
    let di = (*end_i - *start_i).as_vec3();
    let dj = (*end_j - *start_j).as_vec3();
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
    // A pair that is exactly touching on an edge at t=0 but immediately
    // separating along that axis has no swept overlap interval. Treating the
    // zero-duration graze as a hit makes adjacent fixed floor REVs behave like
    // vertical walls and blocks horizontal sliding across the tiled surface.
    let interval_start = enter.max(0.0);
    if exit <= interval_start + 1e-6 {
        return None;
    }
    let axis = hit_axis?;
    let mut normal = Vec3::ZERO;
    normal[axis] = hit_sign;
    Some(normal)
}

fn overlap_normal(a: DVec3, b: DVec3, fallback: DVec3) -> Option<(Vec3, f32)> {
    let delta = (b - a).as_vec3();
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
        fallback[axis] as f32
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

fn inverse_mass(scene: &Scene, i: usize, axis: usize, targets: &[[Option<f64>; 3]]) -> f32 {
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
    targets: &[[Option<f64>; 3]],
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
    scene.positions[i] += di.as_dvec3();
    scene.positions[j] += dj.as_dvec3();
    // Split positional correction from velocity so it cannot create energy.
    scene.previous[i] += di.as_dvec3();
    scene.previous[j] += dj.as_dvec3();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bond::EMPTY,
        scene::{SceneConfig, VoxelSpawn},
        solver,
    };

    #[test]
    fn rigidly_rotated_continuum_has_no_internal_box_contacts() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [3; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        let mut scene = Scene::new(config);
        let rotation = glam::Quat::from_rotation_z(0.37);
        for p in &mut scene.positions {
            *p = rotation.as_dquat() * *p;
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
        let mut cached_topology: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
        adjacency(&scene, &mut a, &mut cached_topology, &mut 0);
        assert!(local_continuum_pair(&scene, &a, 0, 4));
        assert!(!local_continuum_pair(&scene, &a, 0, 2));
        scene.previous.clone_from(&scene.positions);
        assert!(project(&mut scene, &[]).is_empty());
        scene.bonds[0][0].neighbor = EMPTY;
        adjacency(&scene, &mut a, &mut cached_topology, &mut 0);
        assert!(!local_continuum_pair(&scene, &a, 0, 1));
        scene.previous.clone_from(&scene.positions);
        assert!(project(&mut scene, &[])
            .iter()
            .any(|c| c.i == 0 && c.j == 1));
        scene.positions[1] = scene.positions[0] + DVec3::X * (DCELL * 0.5);
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
        // The scene retains its configured atmosphere, so aerodynamic drag
        // makes the vacuum-only ballistic estimate slightly too large.
        assert!((scene.positions[0].z - expected).abs() < 2e-4);
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
        scene.positions[0].x += 3.0 * DCELL;
        let contacts = project(&mut scene, &[]);
        assert!(!contacts.is_empty());
        assert_eq!(scene.positions[1], wall);
        assert!(wall.x - scene.positions[0].x >= DCELL - 1e-6);
    }

    #[test]
    fn broadphase_finds_far_contact_when_fixed_voxel_has_lower_index() {
        let iron = crate::material::Material::component_id("iron").unwrap();
        for fixed_is_first in [false, true] {
            let low_cell = crate::scene::VoxelSpawn::at_cell([20, 0, 0], iron);
            let high_cell = crate::scene::VoxelSpawn::at_cell([21, 0, 0], iron);
            let spawns = if fixed_is_first {
                vec![low_cell.fixed(), high_cell]
            } else {
                vec![low_cell, high_cell.fixed()]
            };
            let mut scene = Scene::from_voxels(
                SceneConfig {
                    gravity: [0.0; 3],
                    ..SceneConfig::default()
                },
                spawns,
            )
            .unwrap();
            // A broken bond makes this a true contact candidate rather than an
            // intact local-continuum neighbor excluded from rigid-box collision.
            scene.bonds[0][0].neighbor = crate::bond::EMPTY;
            scene.workspace[0][0].rest_neighbor = crate::bond::EMPTY;

            let contacts = project(&mut scene, &[]);
            assert!(contacts
                .iter()
                .any(|contact| (contact.i, contact.j) == (0, 1)));
        }
    }

    #[test]
    fn aggregate_members_skip_internal_pairs_but_keep_fine_tier_contacts() {
        let oak = crate::material::Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(65);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], oak));
                }
            }
        }
        spawns.push(VoxelSpawn::at_cell([4, 0, 0], oak));
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        // The neighboring fine REV remains a separate body and a collision
        // candidate when its material bond is broken.
        scene.bonds[0][3].neighbor = EMPTY;
        scene.workspace[0][3].rest_neighbor = EMPTY;
        let members: Vec<_> = (0..64).collect();
        scene.register_rigid_aggregate_candidate(&members).unwrap();

        let contacts = project(&mut scene, &[]);

        assert!(!contacts
            .iter()
            .any(|contact| scene.aggregate_of[contact.i].is_some()
                && scene.aggregate_of[contact.i] == scene.aggregate_of[contact.j]));
        assert!(contacts
            .iter()
            .any(|contact| (contact.i, contact.j) == (3, 64)));
    }

    fn clearance_scene(
        origins: &[[i32; 3]],
        velocities: &[Vec3],
        fixed_cells: &[[i32; 3]],
    ) -> Scene {
        assert_eq!(origins.len(), velocities.len());
        let oak = crate::material::Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(origins.len() * 64 + fixed_cells.len());
        for (&origin, &velocity) in origins.iter().zip(velocities) {
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        spawns.push(
                            VoxelSpawn::at_cell([origin[0] + x, origin[1] + y, origin[2] + z], oak)
                                .with_velocity(velocity),
                        );
                    }
                }
            }
        }
        for &cell in fixed_cells {
            spawns.push(VoxelSpawn::at_cell(cell, oak).fixed());
        }
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        for group in 0..origins.len() {
            let start = group * 64;
            scene
                .register_rigid_aggregate_candidate(&(start..start + 64).collect::<Vec<_>>())
                .unwrap();
        }
        scene
    }

    #[test]
    fn rigid_candidates_clear_for_separated_moving_bodies() {
        let scene = clearance_scene(
            &[[0, 0, 0], [8, 0, 0]],
            &[Vec3::X * 0.1, Vec3::X * 0.1],
            &[[30, 0, 0]],
        );

        assert!(rigid_candidates_clear_for_step(
            &scene,
            0.01,
            DVec3::ZERO,
            &[],
        ));
    }

    #[test]
    fn rigid_candidates_reject_impending_fixed_floor_contact() {
        let scene = clearance_scene(&[[0, 0, 10]], &[Vec3::NEG_Z], &[[0, 0, 0]]);

        assert!(!rigid_candidates_clear_for_step(
            &scene,
            0.081,
            DVec3::ZERO,
            &[],
        ));
    }

    #[test]
    fn rigid_candidates_reject_approaching_aggregate_bodies() {
        let scene = clearance_scene(&[[0, 0, 0], [12, 0, 0]], &[Vec3::X, Vec3::NEG_X], &[]);

        assert!(!rigid_candidates_clear_for_step(
            &scene,
            0.04,
            DVec3::ZERO,
            &[],
        ));
    }

    #[test]
    fn rigid_candidates_reject_rotating_long_body_near_floor() {
        let oak = crate::material::Material::component_id("oak").unwrap();
        let omega = 25.0_f32;
        let mut spawns = Vec::with_capacity(49);
        for z in 5..7 {
            for y in 0..2 {
                for x in 0..12 {
                    let arm_x = (x as f32 - 5.5) * CELL;
                    let arm_z = (z as f32 - 5.5) * CELL;
                    spawns.push(VoxelSpawn::at_cell([x, y, z], oak).with_velocity(Vec3::new(
                        omega * arm_z,
                        0.0,
                        -omega * arm_x,
                    )));
                }
            }
        }
        spawns.push(VoxelSpawn::at_cell([0, 0, 0], oak).fixed());
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        scene
            .register_rigid_aggregate_candidate(&(0..48).collect::<Vec<_>>())
            .unwrap();

        assert!(!rigid_candidates_clear_for_step(
            &scene,
            0.04,
            DVec3::ZERO,
            &[],
        ));
    }

    fn two_separated_aggregate_blocks() -> Scene {
        let oak = crate::material::Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(129);
        for x_offset in [0, 6] {
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        spawns.push(VoxelSpawn::at_cell([x + x_offset, y, z], oak));
                    }
                }
            }
        }
        spawns.push(VoxelSpawn::at_cell([30, 0, 0], oak).fixed());
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        scene
            .register_rigid_aggregate_candidate(&(0..64).collect::<Vec<_>>())
            .unwrap();
        scene
            .register_rigid_aggregate_candidate(&(64..128).collect::<Vec<_>>())
            .unwrap();
        scene
    }

    #[test]
    fn separated_coarse_bodies_skip_rev_broadphase() {
        let mut scene = two_separated_aggregate_blocks();

        assert!(project(&mut scene, &[]).is_empty());
        assert_eq!(scene.rigid_aggregates.iter().flatten().count(), 2);
        assert!(!scene.collision_scratch.candidate_pairs_valid);
        assert!(scene.collision_scratch.pairs.is_empty());
    }

    #[test]
    fn swept_coarse_bounds_wake_bodies_before_rev_contact_check() {
        let mut scene = two_separated_aggregate_blocks();
        // Move the second rigid body into the first during this step. Its swept
        // aggregate bounds overlap, so the exact 1 cm collision path must run.
        for position in &mut scene.positions[64..] {
            position.x -= DCELL * 2.1;
        }

        let contacts = project(&mut scene, &[]);

        assert!(scene.aggregate_of[..64].iter().all(Option::is_none));
        assert!(scene.aggregate_of[64..].iter().all(Option::is_none));
        assert!(contacts
            .iter()
            .any(|contact| (contact.i, contact.j) == (3, 64)));
    }

    #[test]
    fn swept_coarse_bounds_wake_for_fixed_rev_contact() {
        let oak = crate::material::Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(65);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], oak));
                }
            }
        }
        // Leave a gap so this fixed obstacle is not part of the oak bond
        // component; the body's swept bound reaches it during the step.
        spawns.push(VoxelSpawn::at_cell([5, 0, 0], oak).fixed());
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        scene
            .register_rigid_aggregate_candidate(&(0..64).collect::<Vec<_>>())
            .unwrap();
        for position in &mut scene.positions[..64] {
            position.x += DCELL * 1.1;
        }

        let contacts = project(&mut scene, &[]);

        assert!(scene.aggregate_of[..64].iter().all(Option::is_none));
        assert!(contacts
            .iter()
            .any(|contact| (contact.i, contact.j) == (3, 64)));
    }

    #[test]
    fn swept_broadphase_emits_each_dynamic_pair_once() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [3, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        let mut scene = Scene::new(config);
        scene.bonds[0]
            .iter_mut()
            .for_each(|bond| bond.neighbor = EMPTY);
        scene.previous.clone_from(&scene.positions);
        scene.positions[2] += DVec3::X * (DCELL * 4.0);
        let starts = scene.previous.clone();
        let mut scratch = CollisionScratch::default();

        adjacency(
            &scene,
            &mut scratch.adjacency,
            &mut scratch.adjacency_topology,
            &mut scratch.adjacency_live_id,
        );
        candidate_pairs(&scene, &starts, &mut scratch, true);

        assert!(scratch.pairs.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(scratch.pairs.contains(&(0, 1)));
        assert!(scratch.pairs.contains(&(1, 2)));
        candidate_pairs(&scene, &starts, &mut scratch, false);
        assert!(scratch.pairs.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(scratch.pairs.contains(&(0, 1)));
        assert!(scratch.pairs.contains(&(1, 2)));
        scene.positions[2].x += DCELL * 0.5;
        candidate_pairs(&scene, &starts, &mut scratch, false);
    }

    #[test]
    fn swept_broadphase_emits_dynamic_static_pair_once() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.set_voxel_kinematic(1, true).unwrap();
        scene.previous.clone_from(&scene.positions);
        scene.positions[0].x += 3.0 * DCELL;
        let starts = scene.previous.clone();
        let mut scratch = CollisionScratch::default();

        adjacency(
            &scene,
            &mut scratch.adjacency,
            &mut scratch.adjacency_topology,
            &mut scratch.adjacency_live_id,
        );
        candidate_pairs(&scene, &starts, &mut scratch, true);

        assert_eq!(scratch.pairs, vec![(0, 1)]);
    }

    #[test]
    fn oversized_dynamic_sweep_uses_sparse_broadphase_without_losing_contact_pair() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.previous.clone_from(&scene.positions);
        // This finite displacement exceeds the i32 bin-coordinate range, so
        // conversion saturates and the checked volume guard must still route
        // to the sparse path before any bin loop starts.
        scene.positions[1].x += DCELL * 10_000_000_000.0;
        let starts = scene.previous.clone();
        let bounds = bin_bounds(
            starts[1].min(scene.positions[1]) / DCELL,
            starts[1].max(scene.positions[1]) / DCELL,
        );
        assert!(bin_volume_exceeds_limit(bounds.0, bounds.1));
        let mut scratch = CollisionScratch::default();
        adjacency(
            &scene,
            &mut scratch.adjacency,
            &mut scratch.adjacency_topology,
            &mut scratch.adjacency_live_id,
        );

        candidate_pairs(&scene, &starts, &mut scratch, true);

        assert_eq!(scratch.pairs, vec![(0, 1)]);
    }

    #[test]
    fn huge_sweep_candidates_match_exhaustive_swept_aabb_overlaps() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [4, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        let mut scene = Scene::new(config);
        for bonds in &mut scene.bonds {
            bonds.iter_mut().for_each(|bond| bond.neighbor = EMPTY);
        }
        scene.previous.clone_from(&scene.positions);
        scene.positions[1].x += DCELL * 100_000.0;
        let starts = scene.previous.clone();
        let mut scratch = CollisionScratch::default();
        adjacency(
            &scene,
            &mut scratch.adjacency,
            &mut scratch.adjacency_topology,
            &mut scratch.adjacency_live_id,
        );

        candidate_pairs(&scene, &starts, &mut scratch, true);

        let swept_bounds: Vec<_> = (0..scene.voxels.len())
            .map(|index| {
                bin_bounds(
                    starts[index].min(scene.positions[index]) / DCELL,
                    starts[index].max(scene.positions[index]) / DCELL,
                )
            })
            .collect();
        let mut exhaustive = Vec::new();
        for i in 0..swept_bounds.len() {
            for j in i + 1..swept_bounds.len() {
                let (lo_i, hi_i) = swept_bounds[i];
                let (lo_j, hi_j) = swept_bounds[j];
                if (0..3).all(|axis| lo_i[axis] <= hi_j[axis] && lo_j[axis] <= hi_i[axis]) {
                    exhaustive.push((i, j));
                }
            }
        }

        assert!(
            scratch.bins.is_empty(),
            "oversized bounds must bypass bin insertion"
        );
        assert_eq!(scratch.pairs, exhaustive);
        assert!(scratch.pairs.len() <= 2 * scene.voxels.len());
    }

    #[test]
    fn oversized_fixed_sweep_uses_sparse_broadphase_without_losing_contact_pair() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        let mut scene = Scene::new(config);
        scene.bonds[0][0].neighbor = EMPTY;
        scene.set_voxel_kinematic(1, true).unwrap();
        scene.previous.clone_from(&scene.positions);
        scene.positions[1].x += DCELL * 100_000.0;
        let starts = scene.previous.clone();
        let mut scratch = CollisionScratch::default();
        adjacency(
            &scene,
            &mut scratch.adjacency,
            &mut scratch.adjacency_topology,
            &mut scratch.adjacency_live_id,
        );

        candidate_pairs(&scene, &starts, &mut scratch, true);

        assert!(scratch.static_requires_sparse);
        assert_eq!(scratch.pairs, vec![(0, 1)]);
    }

    #[test]
    fn half_cell_bins_cover_overlapping_swept_bounds() {
        let motions = [
            (-1.25_f32, -1.25_f32),
            (-0.51, -0.49),
            (-0.25, 0.25),
            (-0.001, 0.001),
            (0.0, 0.0),
            (0.49, 0.51),
            (0.75, 1.25),
            (1.49, 1.51),
            (-2.0, 2.0),
        ];
        for &(a0, a1) in &motions {
            for &(b0, b1) in &motions {
                let a_min = a0.min(a1);
                let a_max = a0.max(a1);
                let b_min = b0.min(b1);
                let b_max = b0.max(b1);
                let geometrically_overlaps = a_min - 0.5 - 2e-4 <= b_max + 0.5 + 2e-4
                    && b_min - 0.5 - 2e-4 <= a_max + 0.5 + 2e-4;
                let (a_lo, a_hi) =
                    bin_bounds(DVec3::splat(a_min as f64), DVec3::splat(a_max as f64));
                let (b_lo, b_hi) =
                    bin_bounds(DVec3::splat(b_min as f64), DVec3::splat(b_max as f64));
                let bins_overlap =
                    (0..3).all(|axis| a_lo[axis] <= b_hi[axis] && b_lo[axis] <= a_hi[axis]);
                assert!(
                    !geometrically_overlaps || bins_overlap,
                    "swept bounds [{a_min}, {a_max}] and [{b_min}, {b_max}] overlap but bins {a_lo:?}..={a_hi:?} and {b_lo:?}..={b_hi:?} do not"
                );
            }
        }
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
        scene.positions[0].x += DCELL * 0.2;
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
    fn oak_voxel_sliding_on_concrete_floor_loses_speed_and_heats() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [1; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [1; 3];
        config.gravity = [0.0, 0.0, -9.81];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        config.iterations = 40;
        config.fracture.friction = 0.45;
        let oak = crate::material::Material::component_id("oak").unwrap();
        let concrete = crate::material::Material::component_id("concrete").unwrap();
        let mut spawns = vec![VoxelSpawn::at_cell([0, 0, 0], oak).with_velocity(Vec3::X * 0.05)];
        for y in -1..=1 {
            for x in -1..=1 {
                spawns.push(VoxelSpawn::at_cell([x, y, -1], concrete).fixed());
            }
        }
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        let initial_speed = scene.voxels[0].velocity.x;
        let initial_kinetic =
            0.5 * f64::from(scene.voxels[0].mass) * f64::from(initial_speed).powi(2);
        let initial_internal = scene.voxels[0].total_internal_energy_j();
        let pair_mu = scene.config.fracture.friction
            * (scene.properties[0].surface_friction_coefficient
                + scene.properties[1].surface_friction_coefficient);
        assert!((pair_mu - 0.45).abs() < 1e-6);
        for _ in 0..50 {
            solver::step(&mut scene, &mut None);
        }
        let final_speed = scene.voxels[0].velocity.x;
        let final_kinetic = 0.5 * f64::from(scene.voxels[0].mass) * f64::from(final_speed).powi(2);
        assert!(
            final_speed < initial_speed * 0.95,
            "speed {initial_speed} -> {final_speed}"
        );
        assert!(scene.energy.contact_friction_heat > 0.0);
        assert!(
            (initial_kinetic - final_kinetic - scene.energy.contact_friction_heat).abs()
                < initial_kinetic * 0.02
        );
        assert!(
            (scene.voxels[0].total_internal_energy_j()
                - initial_internal
                - scene.energy.contact_friction_heat
                - scene.energy.implicit_solver_heat)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn tiled_floor_internal_faces_do_not_kick_sliding_body_but_outer_wall_does() {
        let oak = crate::material::Material::component_id("oak").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut floor_spawns = Vec::new();
        for z in [-1, 0] {
            for x in 0..2 {
                floor_spawns.push(VoxelSpawn::at_cell([x, 0, z], oak).fixed());
            }
        }
        floor_spawns.push(VoxelSpawn::at_cell([0, 0, 1], oak));
        let mut floor = Scene::from_voxels(config.clone(), floor_spawns).unwrap();
        floor.previous.clone_from(&floor.positions);
        floor.positions[4].x = 0.012;
        let contacts = project(&mut floor, &[]);

        assert!(contacts.iter().all(|contact| contact.normal.x.abs() < 0.5));
        assert!((floor.positions[4].x - 0.012).abs() < 1e-12);

        let wall_spawns = vec![
            VoxelSpawn::at_cell([0, 0, 0], oak).fixed(),
            VoxelSpawn::at_cell([0, 0, 1], oak),
        ];
        let mut wall = Scene::from_voxels(config, wall_spawns).unwrap();
        wall.positions[1] = DVec3::new(0.014, 0.005, 0.014);
        wall.previous.clone_from(&wall.positions);
        let initial_x = wall.positions[1].x;
        let contacts = project(&mut wall, &[]);

        assert!(contacts.iter().any(|contact| contact.normal.x.abs() > 0.5));
        assert!(wall.positions[1].x > initial_x);
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
        assert!((scene.positions[1] - scene.positions[0]).length() <= DCELL + 2e-6);
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
            scene.positions[1].x - scene.positions[0].x >= DCELL - 1e-5,
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
        assert!((scene.positions[1] - scene.positions[0]).length() <= DCELL + 2e-6);
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
    fn warmer_liquid_contact_uses_lower_arrhenius_viscosity() {
        let water = crate::material::Material::component_id("water").unwrap();
        let heat_for_temperature = |temperature_kelvin| {
            let mut scene = crate::scene::SceneBuilder::standard()
                .with_gravity(Vec3::ZERO)
                .add_voxel(
                    crate::scene::VoxelSpawn::at_cell([0, 0, 0], water)
                        .with_velocity(Vec3::Y * 0.01)
                        .at_temperature(temperature_kelvin),
                )
                .add_voxel(
                    crate::scene::VoxelSpawn::at_cell([1, 0, 0], water)
                        .with_velocity(Vec3::NEG_Y * 0.01)
                        .at_temperature(temperature_kelvin),
                )
                .build()
                .unwrap();
            scene
                .set_material_viscosity_temperature_model("water", 293.15, 17_000.0)
                .unwrap();
            let contact = Contact {
                i: 0,
                j: 1,
                normal: Vec3::X,
                new_impact: false,
            };
            resolve_liquid_viscosity(&mut scene, &[contact], &[[None; 3]; 2], 0.01);
            scene.energy.liquid_viscous_heat
        };

        let cool_heat = heat_for_temperature(293.15);
        let warm_heat = heat_for_temperature(333.15);
        assert!(cool_heat > 0.0);
        assert!(warm_heat > 0.0 && warm_heat < cool_heat);
    }

    #[test]
    fn liquid_viscosity_transmits_shear_to_a_solid_contact_neighbor() {
        let water = crate::material::Material::component_id("water").unwrap();
        let iron = crate::material::Material::component_id("iron").unwrap();
        let mut scene = crate::scene::SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(
                crate::scene::VoxelSpawn::at_cell([0, 0, 0], water)
                    .with_velocity(Vec3::Y * 0.01)
                    .at_temperature(293.15),
            )
            .add_voxel(crate::scene::VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(293.15))
            .build()
            .unwrap();
        scene.set_material_dynamic_viscosity("water", 1.0).unwrap();
        scene.bonds[0][0].neighbor = crate::bond::EMPTY;
        scene.workspace[0][0].rest_neighbor = crate::bond::EMPTY;
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

        resolve_liquid_viscosity(&mut scene, &[contact], &[[None; 3]; 2], 0.01);

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
        assert!(scene.voxels[0].velocity.y < 0.01);
        assert!(scene.voxels[1].velocity.y > 0.0);
        assert!((momentum_after - momentum_before).length() < 1e-10);
        assert!((energy_after - energy_before).abs() < 1e-12);
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
        assert!(scene.positions[1].x - scene.positions[0].x >= DCELL - 1e-5);
    }
}
