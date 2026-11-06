use crate::{
    bond::EMPTY,
    collision,
    scene::{sync_visited, visit, visit_thermal, CpuPreconditioner, Scene, AREA, CELL, DCELL},
    tool::Tool,
};
use glam::{DVec3, Mat3, Quat, Vec3};
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Per-thread observations include restitution correction solves as well as
/// the main mechanical solve. They never participate in physical decisions.
#[derive(Clone, Copy, Default)]
struct ContactPcgProfile {
    calls: usize,
    iterations: usize,
    operator: Duration,
    projection: Duration,
    preconditioner: Duration,
    reductions: Duration,
    updates: Duration,
    total: Duration,
    gpu_setup: Duration,
    gpu_operator_calls: usize,
    gpu_preconditioner_calls: usize,
    gpu_operator_fallbacks: usize,
    unconverged_calls: usize,
    max_true_residual: Option<f32>,
}

thread_local! {
    static CONTACT_PCG_PROFILE: std::cell::Cell<ContactPcgProfile> =
        std::cell::Cell::new(ContactPcgProfile::default());
}

fn record_contact_pcg_profile(profile: ContactPcgProfile) {
    CONTACT_PCG_PROFILE.with(|slot| {
        let mut total = slot.get();
        total.calls += profile.calls;
        total.iterations += profile.iterations;
        total.operator += profile.operator;
        total.projection += profile.projection;
        total.preconditioner += profile.preconditioner;
        total.reductions += profile.reductions;
        total.updates += profile.updates;
        total.total += profile.total;
        total.gpu_setup += profile.gpu_setup;
        total.gpu_operator_calls += profile.gpu_operator_calls;
        total.gpu_preconditioner_calls += profile.gpu_preconditioner_calls;
        total.gpu_operator_fallbacks += profile.gpu_operator_fallbacks;
        total.unconverged_calls += profile.unconverged_calls;
        if let Some(residual) = profile.max_true_residual {
            total.max_true_residual = Some(total.max_true_residual.map_or(residual, |old| old.max(residual)));
        }
        slot.set(total);
    });
}

#[cfg(test)]
mod contact_profile_tests {
    use super::*;

    #[test]
    fn gpu_retry_preserves_the_cpu_iteration_limit_policy() {
        assert!(!contact_gpu_requires_cpu_retry(true, 0.01, 1e-5, 80, 80));
        assert!(contact_gpu_requires_cpu_retry(true, 0.01, 1e-5, 2, 80));
        assert!(contact_gpu_requires_cpu_retry(true, f32::NAN, 1e-5, 80, 80));
        assert!(contact_gpu_requires_cpu_retry(true, f32::INFINITY, 1e-5, 80, 80));
        assert!(!contact_gpu_requires_cpu_retry(true, 1e-6, 1e-5, 2, 80));
        assert!(!contact_gpu_requires_cpu_retry(false, f32::NAN, 1e-5, 2, 80));
    }

    #[test]
    fn contact_profile_accumulates_without_crossing_solver_threads() {
        CONTACT_PCG_PROFILE.with(|slot| slot.set(ContactPcgProfile::default()));
        record_contact_pcg_profile(ContactPcgProfile {
            calls: 1, iterations: 7, operator: Duration::from_micros(12),
            gpu_operator_calls: 3, ..Default::default()
        });
        std::thread::spawn(|| {
            assert_eq!(CONTACT_PCG_PROFILE.with(std::cell::Cell::get).calls, 0);
            record_contact_pcg_profile(ContactPcgProfile { calls: 9, ..Default::default() });
        }).join().unwrap();
        record_contact_pcg_profile(ContactPcgProfile {
            calls: 2, iterations: 5, operator: Duration::from_micros(8),
            gpu_operator_fallbacks: 1, ..Default::default()
        });
        let profile = CONTACT_PCG_PROFILE.with(std::cell::Cell::get);
        assert_eq!(profile.calls, 3);
        assert_eq!(profile.iterations, 12);
        assert_eq!(profile.operator, Duration::from_micros(20));
        assert_eq!(profile.gpu_operator_calls, 3);
        assert_eq!(profile.gpu_operator_fallbacks, 1);
        CONTACT_PCG_PROFILE.with(|slot| slot.set(ContactPcgProfile::default()));
    }
}

// A controlled 4,203-active-REV physical capture was 7.3x faster with the
// serial path than with Rayon (same 177 states and identical solver results).
// Use the pool for larger zones; keep the threshold overridable for A/B runs.
const PARALLEL_MIN_VOXELS: usize = 8_192;
const ROTATION_PARALLEL_MIN_VOXELS: usize = 8_192;
const FRACTURE_PARALLEL_MIN_VOXELS: usize = 8_192;
const ATMOSPHERE_PARALLEL_MIN_VOXELS: usize = 262_144;
const IC0_BIPARTITE_EXPERIMENT: bool = false;
const AUTO_AGGREGATE_MIN_REVS: usize = 64;
const AUTO_AGGREGATE_SCAN_INTERVAL_STEPS: u64 = 64;

fn parallel_enabled_for(voxel_count: usize) -> bool {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let threshold = THRESHOLD.get_or_init(|| {
        std::env::var("VOXEL_PARALLEL_MIN_VOXELS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(PARALLEL_MIN_VOXELS)
    });
    voxel_count >= *threshold
}

fn parallel_assembly_precompute_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_ASSEMBLY_PRECOMPUTE_PARALLEL")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

fn fused_pcg_reductions_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_PCG_FUSED_REDUCTIONS")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

fn component_pcg_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_PCG_COMPONENTS_PARALLEL")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

fn bipartite_ic0_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_IC0_BIPARTITE")
            .map(|value| value == "1")
            .unwrap_or(IC0_BIPARTITE_EXPERIMENT)
    })
}

fn atmosphere_parallel_enabled_for(voxel_count: usize) -> bool {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let threshold = THRESHOLD.get_or_init(|| {
        std::env::var("VOXEL_ATMOSPHERE_PARALLEL_MIN_VOXELS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(ATMOSPHERE_PARALLEL_MIN_VOXELS)
    });
    voxel_count >= *threshold
}

fn fracture_parallel_enabled_for(voxel_count: usize) -> bool {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let threshold = THRESHOLD.get_or_init(|| {
        std::env::var("VOXEL_FRACTURE_PARALLEL_MIN_VOXELS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(FRACTURE_PARALLEL_MIN_VOXELS)
    });
    voxel_count >= *threshold
}

fn energy_edges_parallel_enabled_for(voxel_count: usize) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_ENERGY_EDGES_PARALLEL")
            .map(|value| value != "0")
            .unwrap_or(true)
    }) && parallel_enabled_for(voxel_count)
}

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
        .phase_state(scene.voxels[n].mass, scene.voxels[n].thermal_energy_f32())
        .solid_fraction
        .clamp(1e-4, 1.0);
    tensor_compliance_at_solid_fraction(scene, n, frame, solid)
}

fn tensor_compliance_at_solid_fraction(scene: &Scene, n: usize, frame: Mat3, solid: f32) -> Mat3 {
    let material = &scene.properties[n];
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
    let rank_one = Mat3::from_cols(
        scaled_fiber * scaled_fiber.x,
        scaled_fiber * scaled_fiber.y,
        scaled_fiber * scaled_fiber.z,
    );
    let alpha = material.shear * micro.anisotropy * 0.2;
    Mat3::from_diagonal(inverse_diagonal)
        - rank_one * (alpha / (1.0 + alpha * fiber.dot(scaled_fiber)))
}

pub fn stiffness_matrix(scene: &Scene, i: usize, j: usize, frame: Mat3) -> Mat3 {
    let a_compliance = tensor_compliance(scene, i, frame);
    let b_compliance = tensor_compliance(scene, j, frame);
    stiffness_from_compliances(a_compliance, b_compliance)
}

fn stiffness_from_compliances(a_compliance: Mat3, b_compliance: Mat3) -> Mat3 {
    if a_compliance == Mat3::ZERO || b_compliance == Mat3::ZERO {
        return Mat3::ZERO;
    }
    // Equal half-lengths in series; matrix harmonic mean, not entrywise average.
    (a_compliance + b_compliance).inverse() * (2.0 * AREA / CELL)
}

/// Apply an implicit Kelvin–Voigt dashpot along each intact bond's current
/// centerline. Equal/opposite central impulses preserve linear and angular
/// momentum; the measured loss in free nodal kinetic energy becomes heat.
/// Tangential bond damping needs rotational inertia/torques, which this REV
/// model does not currently represent, so it is intentionally not fabricated.
fn translational_mobility(scene: &Scene, targets: &[[Option<f64>; 3]], index: usize) -> Vec3 {
    let voxel = &scene.voxels[index];
    if voxel.kinematic {
        Vec3::ZERO
    } else {
        Vec3::new(
            if targets[index][0].is_none() {
                1.0 / voxel.mass
            } else {
                0.0
            },
            if targets[index][1].is_none() {
                1.0 / voxel.mass
            } else {
                0.0
            },
            if targets[index][2].is_none() {
                1.0 / voxel.mass
            } else {
                0.0
            },
        )
    }
}

fn apply_solid_viscoelastic_damping(
    scene: &mut Scene,
    targets: &[[Option<f64>; 3]],
    dt: f32,
    constraints: &[Constraint],
) -> usize {
    let mut cache_hits = 0;
    // Keep scalar REV energies current bond-by-bond (they affect later
    // temperature-dependent damping), but defer the uniform microcell update.
    // Updating and re-summing 27 thermal cells for each dissipative bond is
    // redundant: positive dashpot heat is spatially uniform within each REV.
    let mut deferred_thermal_cell_delta: rustc_hash::FxHashMap<usize, f64> = Default::default();
    // Constraints are the stable axis-major list of intact bonds with at least
    // one free translational degree of freedom. Fully fixed bonds have zero
    // damping mobility, so iterating this sparse list preserves the complete
    // active damping set while avoiding a 3 * voxel_count topology scan.
    for constraint in constraints {
        let axis = constraint.axis;
        let i = constraint.i;
        if scene.aggregate_of[i].is_some() {
            continue;
        }
        let bond = scene.bonds[axis][i];
        let cached_constraint = (bond.neighbor != EMPTY && constraint.j == bond.neighbor as usize)
            .then_some(constraint);
        if bond.neighbor == EMPTY {
            continue;
        }
        let j = bond.neighbor as usize;
        if j >= scene.voxels.len() {
            continue;
        }
        let kelvin_voigt_time = 0.5
            * (scene.properties[i].solid_kelvin_voigt_time_s
                + scene.properties[j].solid_kelvin_voigt_time_s);
        if !kelvin_voigt_time.is_finite() || kelvin_voigt_time <= 0.0 {
            continue;
        }
        let separation = scene.positions[j] - scene.positions[i];
        let normal = separation.normalize_or_zero().as_vec3();
        if normal == Vec3::ZERO {
            continue;
        }
        let mobility_i = translational_mobility(scene, targets, i);
        let mobility_j = translational_mobility(scene, targets, j);
        let inverse_effective_mass = (mobility_i + mobility_j).dot(normal * normal);
        if inverse_effective_mass <= 0.0 {
            continue;
        }
        let velocity_i = scene.voxels[i].velocity;
        let velocity_j = scene.voxels[j].velocity;
        let relative_speed = (velocity_j - velocity_i).dot(normal);
        if relative_speed.abs() <= 1e-12 {
            continue;
        }

        // A zero normal rate produces no dashpot force or heat. Test it
        // before evaluating the temperature-dependent bond stiffness.
        let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
        let material_stiffness = cached_constraint
            .filter(|constraint| {
                constraint.rotation_i == scene.rotations[i]
                    && constraint.rotation_j == scene.rotations[j]
                    && constraint.thermal_energy_i == scene.voxels[i].thermal_energy_f32()
                    && constraint.thermal_energy_j == scene.voxels[j].thermal_energy_f32()
            })
            .map(|constraint| {
                cache_hits += 1;
                constraint.material_k.x_axis.x
            })
            .unwrap_or_else(|| stiffness_matrix(scene, i, j, frame).x_axis.x);
        let stiffness = material_stiffness * (1.0 - bond.damage).clamp(0.0, 1.0);
        let dashpot = kelvin_voigt_time * stiffness;
        if !dashpot.is_finite() || dashpot <= 0.0 {
            continue;
        }

        // Backward-Euler integration keeps stiff material dashpots stable.
        let denominator = 1.0 + dt * dashpot * inverse_effective_mass;
        if !denominator.is_finite() || denominator <= 0.0 {
            continue;
        }
        let impulse_scalar = -dt * dashpot * relative_speed / denominator;
        let impulse_on_j = normal * impulse_scalar;
        let new_velocity_i = velocity_i
            - Vec3::new(
                mobility_i.x * impulse_on_j.x,
                mobility_i.y * impulse_on_j.y,
                mobility_i.z * impulse_on_j.z,
            );
        let new_velocity_j = velocity_j
            + Vec3::new(
                mobility_j.x * impulse_on_j.x,
                mobility_j.y * impulse_on_j.y,
                mobility_j.z * impulse_on_j.z,
            );

        // Compute the exact kinetic-energy change of the free translational
        // degrees of freedom from the applied impulse. Prescribed axes do
        // not contribute kinetic energy, even if their target velocity is
        // nonzero. This is algebraically equivalent to measuring both
        // kinetic energies before and after the impulse, with fewer vector
        // squares and mass multiplications.
        let free_velocity_i = Vec3::new(
            if mobility_i.x > 0.0 {
                velocity_i.x
            } else {
                0.0
            },
            if mobility_i.y > 0.0 {
                velocity_i.y
            } else {
                0.0
            },
            if mobility_i.z > 0.0 {
                velocity_i.z
            } else {
                0.0
            },
        );
        let free_velocity_j = Vec3::new(
            if mobility_j.x > 0.0 {
                velocity_j.x
            } else {
                0.0
            },
            if mobility_j.y > 0.0 {
                velocity_j.y
            } else {
                0.0
            },
            if mobility_j.z > 0.0 {
                velocity_j.z
            } else {
                0.0
            },
        );
        let kinetic_change = impulse_scalar * (free_velocity_j - free_velocity_i).dot(normal)
            + 0.5 * impulse_scalar * impulse_scalar * inverse_effective_mass;
        scene.voxels[i].velocity = new_velocity_i;
        scene.voxels[j].velocity = new_velocity_j;

        let heat = (-kinetic_change).max(0.0) as f64;
        if heat > 0.0 {
            scene.energy.solid_viscoelastic_heat += heat;
            *deferred_thermal_cell_delta.entry(i).or_default() +=
                scene.voxels[i].add_internal_energy_j_deferred_thermal_cells(heat * 0.5);
            *deferred_thermal_cell_delta.entry(j).or_default() +=
                scene.voxels[j].add_internal_energy_j_deferred_thermal_cells(heat * 0.5);
        }
    }
    for (index, delta) in deferred_thermal_cell_delta {
        scene.voxels[index].distribute_deferred_thermal_energy_delta(delta);
    }
    cache_hits
}

/// Free linear thermal displacement of a bond relative to each REV's
/// stress-free temperature at scene creation. Each REV contributes half of
/// its anisotropic eigenstrain tensor projected along the bond direction.
fn bond_thermal_rest(scene: &Scene, i: usize, j: usize, frame: Mat3) -> DVec3 {
    let temperatures = [i, j].map(|n| {
        scene.properties[n]
            .phase_state(scene.voxels[n].mass, scene.voxels[n].thermal_energy_f32())
            .temperature_kelvin
    });
    bond_thermal_rest_at_temperatures(scene, i, j, frame, temperatures)
}

fn bond_thermal_rest_at_temperatures(
    scene: &Scene,
    i: usize,
    j: usize,
    frame: Mat3,
    temperatures: [f32; 2],
) -> DVec3 {
    let direction = frame.x_axis.as_dvec3().normalize_or_zero();
    let endpoint_expansion = |n: usize, temperature: f32| {
        let material = &scene.properties[n];
        let fiber = (scene.rotations[n] * scene.voxels[n].microstructure.fibers())
            .as_dvec3()
            .normalize_or_zero();
        let alpha_parallel = material.thermal_expansion_parallel as f64;
        let alpha_perpendicular = material.thermal_expansion_perpendicular as f64;
        let alpha_direction = direction * alpha_perpendicular
            + fiber * ((alpha_parallel - alpha_perpendicular) * fiber.dot(direction));
        alpha_direction * (temperature as f64 - scene.reference_temperatures_kelvin[n] as f64)
    };
    frame.as_dmat3().transpose()
        * (direction * DCELL
            + (endpoint_expansion(i, temperatures[0]) + endpoint_expansion(j, temperatures[1]))
                * (0.5 * DCELL))
}

/// Initial local-frame rest offset of a new bond at its current contact pose.
/// This lets adhesion/healing establish a stress-free connection without an
/// instantaneous spring impulse, including at a hot or plastically displaced interface.
pub(crate) fn bond_formation_offset(scene: &Scene, i: usize, j: usize, axis: usize) -> [f64; 3] {
    let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
    let current = frame.as_dmat3().transpose() * (scene.positions[j] - scene.positions[i]);
    (current - bond_thermal_rest(scene, i, j, frame)).to_array()
}

fn bond_elastic_displacement_at(
    scene: &Scene,
    positions: &[DVec3],
    i: usize,
    j: usize,
    axis: usize,
    frame: Mat3,
) -> Vec3 {
    (frame.as_dmat3().transpose() * (positions[j] - positions[i])
        - bond_thermal_rest(scene, i, j, frame)
        - DVec3::from_array(scene.bonds[axis][i].formation_offset.map(f64::from)))
    .as_vec3()
}

fn bond_elastic_displacement(scene: &Scene, i: usize, j: usize, axis: usize, frame: Mat3) -> Vec3 {
    bond_elastic_displacement_at(scene, &scene.positions, i, j, axis, frame)
}

/// Recover the current stored elastic energy of an intact bond (joules).
/// This is also used when a topology operation removes the bond before its
/// strain energy can be dissipated through fracture or melting.
pub(crate) fn bond_stored_elastic_energy_j(scene: &Scene, i: usize, j: usize, axis: usize) -> f64 {
    bond_stored_elastic_energy_at_positions_j(scene, &scene.positions, i, j, axis)
}

pub(crate) fn bond_stored_elastic_energy_at_positions_j(
    scene: &Scene,
    positions: &[DVec3],
    i: usize,
    j: usize,
    axis: usize,
) -> f64 {
    let bond = scene.bonds[axis][i];
    let frame = basis(axis, scene.rotations[i].slerp(scene.rotations[j], 0.5));
    let stiffness = stiffness_matrix(scene, i, j, frame);
    let precise = scene.cell_dofs && scene.precise_rotations.len() == scene.rotations.len();
    let displacement = if precise {
        let temperatures = [i, j].map(|n| {
            scene.properties[n]
                .phase_state(scene.voxels[n].mass, scene.voxels[n].thermal_energy_f32())
                .temperature_kelvin
        });
        let mid = scene.precise_rotations[i].slerp(scene.precise_rotations[j], 0.5);
        (basis_precise(axis, mid).transpose()
            * (positions[j] - positions[i] - face_rest(scene, i, j, axis, temperatures)))
        .as_vec3()
    } else {
        bond_elastic_displacement_at(scene, positions, i, j, axis, frame)
            - Vec3::from_array(bond.plastic_strain) * CELL
    };
    let mut energy =
        (0.5 * (1.0 - bond.damage) * displacement.dot(stiffness * displacement)).max(0.0) as f64;
    if precise {
        let bent = face_bend(scene, i, j);
        if bent != DVec3::ZERO {
            let direction = (positions[j] - positions[i]).normalize_or_zero();
            let stiff = face_rotation_stiffness(scene, i, j, direction, bond.damage);
            energy += 0.5 * bent.dot(stiff * bent);
        }
    }
    energy
}

/// Capture mechanical energy before changing material frames or positions.
/// Half of each bond's stored energy belongs to each endpoint.
fn capture_nodal_mechanical_energy(
    scene: &Scene,
    energy: &mut Vec<f64>,
    edge_energy: &mut Vec<f64>,
    parallel: bool,
) -> f64 {
    let live = scene.step_live.clone();
    let voxel_count = scene.voxels.len();
    let kinetic = |v: &crate::voxel::Voxel| {
        if v.kinematic { 0.0 } else {
            0.5 * f64::from(v.mass) * v.velocity.as_dvec3().length_squared()
        }
    };
    if live.is_some() && energy.len() == voxel_count {
        // A sleeper's entry is its (zero) kinematic value from the first full build.
        for i in visit(&live, voxel_count) {
            energy[i] = kinetic(&scene.voxels[i]);
        }
    } else {
        energy.clear();
        energy.extend(scene.voxels.iter().map(kinetic));
    }
    for i in visit(&live, voxel_count) {
        energy[i] += if scene.voxels[i].kinematic { 0.0 } else { cell_spin_energy_j(scene, i) };
    }
    edge_energy.resize(3 * voxel_count, 0.0);
    let compute_edge = |edge: usize| {
        let axis = edge / voxel_count;
        let i = edge % voxel_count;
        let bond = scene.bonds[axis][i];
        if bond.neighbor == EMPTY
            || (scene.voxels[i].kinematic && scene.voxels[bond.neighbor as usize].kinematic)
        {
            0.0
        } else {
            bond_stored_elastic_energy_j(scene, i, bond.neighbor as usize, axis)
        }
    };
    if live.is_some() {
        for axis in 0..3 {
            for i in visit(&live, voxel_count) {
                edge_energy[axis * voxel_count + i] = compute_edge(axis * voxel_count + i);
            }
        }
    } else if parallel && voxel_count > 0 {
        edge_energy.par_iter_mut().enumerate().for_each(|(edge, value)| {
            *value = compute_edge(edge);
        });
    } else {
        for (edge, value) in edge_energy.iter_mut().enumerate() {
            *value = compute_edge(edge);
        }
    }
    let mut elastic = 0.0;
    for axis in 0..3 {
        for i in visit(&live, voxel_count) {
            let bond = &scene.bonds[axis][i];
            if bond.neighbor == EMPTY { continue; }
            let j = bond.neighbor as usize;
            let stored = edge_energy[axis * voxel_count + i];
            elastic += stored;
            energy[i] += stored * 0.5;
            energy[j] += stored * 0.5;
        }
    }
    elastic
}

#[derive(Clone, Copy, Default)]
struct FreeImplicitEnergyBudget {
    before: f64,
    after: f64,
    correction: f64,
    heat: f64,
    excluded: bool,
}

/// Backward Euler damps a free elastic wave even after its contact ends.
/// For fixed linear stiffness its exact loss is
/// 1/2 sum(m |dv|²) + 1/2 sum(duᵀ K du). Return only a measured mechanical
/// deficit, capped by that local positive quadratic: no energy compensation.
/// Contacted, driven, constrained and coarse components use their own work
/// accounting and are excluded. In particular, an unrelated impact cannot
/// receive the heat of a freely vibrating component elsewhere in the scene.
fn free_implicit_dissipation(
    scene: &Scene,
    scratch: &SolverScratch,
    before: &[f64],
    start_positions: &[DVec3],
    start_velocities: &[Vec3],
    solved: &[DVec3],
    predicted: &[DVec3],
    drift: &[Vec3],
    tool_delta: &[Vec3],
    loads: &[Vec3],
    targets: &[[Option<f64>; 3]],
    contacts: &[collision::Contact],
    gravity: Vec3,
    h: f32,
    edge_energy: &mut Vec<f64>,
    parallel_edges: bool,
    budgets: &mut Vec<FreeImplicitEnergyBudget>,
    components: &mut Vec<usize>,
    nodal_heat: &mut Vec<f64>,
) -> (f64, f64) {
    // Cluster labels are rebuilt at a fixed-step boundary. A bond may already
    // have broken in an earlier substep, so derive the current material graph.
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    let live = scene.step_live.clone();
    let voxel_count = scene.voxels.len();
    if live.is_some() && components.len() == voxel_count && budgets.len() == voxel_count {
        for i in visit(&live, voxel_count) {
            components[i] = i;
            budgets[i] = FreeImplicitEnergyBudget::default();
        }
    } else {
        components.clear();
        components.extend(0..voxel_count);
        budgets.clear();
        budgets.resize(voxel_count, FreeImplicitEnergyBudget::default());
        // A sleeper is a fixed REV: its own, excluded budget, never revisited.
        for (budget, voxel) in budgets.iter_mut().zip(&scene.voxels) {
            budget.excluded = voxel.kinematic;
        }
    }
    for constraint in &scratch.constraints {
        let i = root(components, constraint.i);
        let j = root(components, constraint.j);
        components[j] = i;
    }
    for i in visit(&live, voxel_count) { components[i] = root(components, i); }
    fill_visited(nodal_heat, voxel_count, 0.0, &live);
    for i in visit(&live, voxel_count) {
        let voxel = &scene.voxels[i];
        let budget = &mut budgets[components[i]];
        budget.before += before[i];
        budget.excluded |= voxel.kinematic || scene.aggregate_of[i].is_some()
            || loads[i] != Vec3::ZERO || tool_delta[i] != Vec3::ZERO
            || targets[i].iter().any(Option::is_some);
        let correction = ((solved[i] - predicted[i]) / f64::from(h)).as_vec3();
        let velocity = drift[i] + tool_delta[i] + correction;
        budget.after += 0.5 * f64::from(voxel.mass) * velocity.as_dvec3().length_squared();
        budget.before += f64::from((gravity * voxel.mass).dot(
            (start_velocities[i] + velocity) * (0.5 * h)));
        nodal_heat[i] = 0.5 * f64::from(voxel.mass) * correction.as_dvec3().length_squared();
    }
    for contact in contacts {
        budgets[components[contact.i]].excluded = true;
        budgets[components[contact.j]].excluded = true;
    }
    for pair in scratch.pair_contacts.iter().filter(|pair| pair.active) {
        budgets[components[pair.i]].excluded = true;
        budgets[components[pair.j]].excluded = true;
    }
    // Keep the buffer's length: the next capture reuses it at three entries per REV.
    if edge_energy.len() < scratch.constraints.len() {
        edge_energy.resize(scratch.constraints.len(), 0.0);
    }
    let edge_energy = &mut edge_energy[..scratch.constraints.len()];
    let stored_energy = |edge: usize| {
        let constraint = &scratch.constraints[edge];
        let cluster = components[constraint.i];
        if budgets[cluster].excluded {
            0.0
        } else {
            bond_stored_elastic_energy_j(scene, constraint.i, constraint.j, constraint.axis)
        }
    };
    if parallel_edges {
        edge_energy.par_iter_mut().enumerate().for_each(|(edge, value)| {
            *value = stored_energy(edge);
        });
    } else {
        for (edge, value) in edge_energy.iter_mut().enumerate() {
            *value = stored_energy(edge);
        }
    }
    for (edge, constraint) in scratch.constraints.iter().enumerate() {
        let (i, j) = (constraint.i, constraint.j);
        let cluster = components[i];
        if budgets[cluster].excluded { continue; }
        budgets[cluster].after += edge_energy[edge];
        let displacement = ((solved[j] - start_positions[j])
            - (solved[i] - start_positions[i])).as_vec3();
        let loss = 0.5 * displacement.as_dvec3().dot(
            constraint.k.as_dmat3() * displacement.as_dvec3());
        nodal_heat[i] += loss.max(0.0) * 0.5;
        nodal_heat[j] += loss.max(0.0) * 0.5;
    }
    for i in visit(&live, voxel_count) {
        budgets[components[i]].correction += nodal_heat[i];
    }
    for i in visit(&live, voxel_count) {
        let budget = &mut budgets[i];
        if !budget.excluded {
            budget.heat = (budget.before - budget.after).max(0.0).min(budget.correction);
        }
    }
    let mut total = 0.0;
    for i in visit(&live, voxel_count) {
        let heat = &mut nodal_heat[i];
        let budget = budgets[components[i]];
        *heat = if budget.correction > 0.0 && !budget.excluded {
            *heat * (budget.heat / budget.correction)
        } else { 0.0 };
        total += *heat;
    }
    let free_deficit = visit(&live, voxel_count).map(|i| &budgets[i]).filter(|budget| !budget.excluded)
        .map(|budget| budget.before - budget.after).sum();
    (total, free_deficit)
}

#[cfg(test)]
mod energy_edge_parallel_tests {
    use super::*;
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    fn scene_with_energy_edges() -> Scene {
        let oak = Material::component_id("oak").unwrap();
        let spawns = [0, 1, 3, 4, 6, 7].map(|x| {
            VoxelSpawn::at_cell([x, 0, 8], oak).at_temperature(if x == 7 { 675.0 } else { 293.15 })
        });
        let mut scene = Scene::from_voxels(SceneConfig::default(), spawns.to_vec()).unwrap();
        scene.voxels[2].kinematic = true;
        for i in 0..scene.voxels.len() {
            scene.rotations[i] = Quat::from_rotation_y(0.07 * (i as f32 + 1.0));
            scene.voxels[i].microstructure.orientation =
                Quat::from_rotation_x(0.03 * (i as f32 + 1.0));
            scene.voxels[i].microstructure.anisotropy = 0.65;
            scene.voxels[i].velocity = Vec3::new(i as f32 * 0.013, -0.02, 0.004);
        }
        // A removed edge must remain absent from the capture pass.
        scene.bonds[0][2] = Default::default();
        scene
    }

    fn constraints_for(scene: &Scene) -> Vec<Constraint> {
        let mut constraints = Vec::new();
        for axis in 0..3 {
            for (i, bond) in scene.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY { continue; }
                let j = bond.neighbor as usize;
                let phases = [i, j].map(|n| scene.properties[n].phase_state(
                    scene.voxels[n].mass,
                    scene.voxels[n].thermal_energy_f32(),
                ));
                constraints.push(make_constraint_at_phases(scene, i, axis, j, phases, [0; 2], None));
            }
        }
        constraints
    }

    #[test]
    fn capture_edge_parallel_is_bitwise_equal_to_serial_reference() {
        let scene = scene_with_energy_edges();
        let mut reference = Vec::new();
        reference.extend(scene.voxels.iter().map(|v| if v.kinematic { 0.0 } else {
            0.5 * f64::from(v.mass) * v.velocity.as_dvec3().length_squared()
        }));
        let mut expected_elastic = 0.0;
        for axis in 0..3 {
            for (i, bond) in scene.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY { continue; }
                let stored = bond_stored_elastic_energy_j(&scene, i, bond.neighbor as usize, axis);
                expected_elastic += stored;
                reference[i] += stored * 0.5;
                reference[bond.neighbor as usize] += stored * 0.5;
            }
        }
        let mut serial = Vec::new();
        let mut serial_edges = Vec::new();
        let serial_elastic = capture_nodal_mechanical_energy(&scene, &mut serial, &mut serial_edges, false);
        let mut parallel = Vec::new();
        let mut parallel_edges = Vec::new();
        let parallel_elastic = capture_nodal_mechanical_energy(&scene, &mut parallel, &mut parallel_edges, true);
        assert_eq!(serial, reference);
        assert_eq!(parallel, reference);
        assert_eq!(serial_elastic.to_bits(), expected_elastic.to_bits());
        assert_eq!(parallel_elastic.to_bits(), expected_elastic.to_bits());
        assert_eq!(parallel_edges, serial_edges);
    }

    #[test]
    fn free_implicit_edge_parallel_is_bitwise_equal_to_serial() {
        let scene = scene_with_energy_edges();
        let mut scratch = SolverScratch::default();
        scratch.constraints = constraints_for(&scene);
        let count = scene.voxels.len();
        let before = vec![0.125; count];
        let start_positions = scene.positions.clone();
        let start_velocities: Vec<_> = scene.voxels.iter().map(|v| v.velocity).collect();
        let solved: Vec<_> = scene.positions.iter().enumerate().map(|(i, p)| *p + DVec3::splat(i as f64 * 1e-5)).collect();
        let predicted = scene.positions.clone();
        let drift = vec![Vec3::ZERO; count];
        let tool_delta = vec![Vec3::ZERO; count];
        let loads = vec![Vec3::ZERO; count];
        let targets = vec![[None; 3]; count];
        let contacts = Vec::new();

        let run = |parallel: bool| {
            // Simulate reuse after capture; every active constraint entry must
            // replace its stale edge value when this shorter buffer is resized.
            let mut edge_values = vec![f64::NAN; 3 * count];
            let mut budgets = Vec::new();
            let mut components = Vec::new();
            let mut heat = Vec::new();
            let total = free_implicit_dissipation(
                &scene, &scratch, &before, &start_positions, &start_velocities,
                &solved, &predicted, &drift, &tool_delta, &loads, &targets,
                &contacts, Vec3::ZERO, 0.01, &mut edge_values, parallel,
                &mut budgets, &mut components, &mut heat,
            );
            (total, budgets, components, heat, edge_values)
        };
        let serial = run(false);
        let parallel = run(true);
        assert_eq!(serial.0.0.to_bits(), parallel.0.0.to_bits());
        assert_eq!(serial.0.1.to_bits(), parallel.0.1.to_bits());
        assert_eq!(serial.1.len(), parallel.1.len());
        for (a, b) in serial.1.iter().zip(&parallel.1) {
            assert_eq!(a.before.to_bits(), b.before.to_bits());
            assert_eq!(a.after.to_bits(), b.after.to_bits());
            assert_eq!(a.correction.to_bits(), b.correction.to_bits());
            assert_eq!(a.heat.to_bits(), b.heat.to_bits());
            assert_eq!(a.excluded, b.excluded);
        }
        assert_eq!(serial.2, parallel.2);
        assert_eq!(serial.3, parallel.3);
        // Only the entries of this call's constraints are written; the rest of the
        // reused buffer keeps its stale (here NaN) values for the next capture.
        let edges = scratch.constraints.len();
        let bits = |values: &[f64]| values[..edges].iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&serial.4), bits(&parallel.4));
        assert_eq!(serial.4.len(), parallel.4.len());
    }
}

fn phase_strength(scene: &Scene, n: usize) -> (f32, f32) {
    let material = &scene.properties[n];
    let phase = material.phase_state(scene.voxels[n].mass, scene.voxels[n].thermal_energy_f32());
    (
        phase.solid_fraction.clamp(0.0, 1.0),
        material.thermal_strength_scale(phase.temperature_kelvin),
    )
}

fn reuse_assembly_phase_strengths_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_REUSE_PHASE_STRENGTHS")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

fn bond_limits_with_phase_strengths(
    scene: &Scene,
    i: usize,
    j: usize,
    direction: Vec3,
    phase_strengths: [(f32, f32); 2],
) -> crate::fracture::Limits {
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
    let (solid, thermal_strength) = phase_strengths
        .into_iter()
        .fold((1.0f32, 1.0f32), |(solid, softening), next| {
            (solid.min(next.0), softening.min(next.1))
        });
    let strength_scale = solid * solid * thermal_strength;
    limits.tension *= strength_scale;
    limits.compression *= strength_scale;
    limits.shear *= strength_scale;
    limits.yield_stress *= strength_scale;
    limits.hardening *= strength_scale;
    limits.fracture_energy *= strength_scale;
    limits
}

pub fn bond_limits(scene: &Scene, i: usize, j: usize, direction: Vec3) -> crate::fracture::Limits {
    bond_limits_with_phase_strengths(
        scene,
        i,
        j,
        direction,
        [phase_strength(scene, i), phase_strength(scene, j)],
    )
}

pub fn fracture_strength(scene: &Scene, i: usize, j: usize, direction: Vec3) -> f32 {
    bond_limits(scene, i, j, direction).tension
}

/// Set a world-space force that remains active until changed or cleared.
pub fn set_force(scene: &mut Scene, voxel: usize, force: Vec3) -> Result<(), &'static str> {
    scene.wake_voxels(&[voxel]);
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
    scene.wake_voxels(&[voxel]);
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
    scene.wake_voxels(&[voxel]);
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
    scene.wake_voxels(&[voxel]);
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
    coordinates: [Option<f64>; 3],
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
    scene.mark_render_changed();
    Ok(())
}
/// Fix all axes at a world coordinate.
pub fn fix_voxel_at(
    scene: &mut Scene,
    voxel: usize,
    coordinate: impl Into<DVec3>,
) -> Result<(), &'static str> {
    fix_voxel_axes(scene, voxel, coordinate.into().to_array().map(Some))
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
    coordinate: Option<f64>,
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
/// Cells solve their rotations directly. Pair contacts still use the translation-only
/// solve, which needs fitted frames; it then returns the change of that fit.
fn update_frames(scene: &mut Scene, scratch: &mut SolverScratch) -> f32 {
    if scene.cell_dofs {
        // Rotations are solved; report a change so the second pass still runs.
        // Skipping it when theta^2 is small gave 172/230 ms (actor on/off) against
        // 233/95 ms, and the cause of the slower resting case is not understood.
        return 1.0;
    }
    fit_rotations(scene, scratch)
}

/// Returns the largest quaternion angle changed by this fit.
fn fit_rotations(scene: &mut Scene, scratch: &mut SolverScratch) -> f32 {
    let n = scene.voxels.len();
    let live = scene.step_live.clone();
    if !visit(&live, n).any(|i| !scene.voxels[i].kinematic && scene.aggregate_of[i].is_none())
    {
        // Coarse rigid bodies carry their exact pose in RigidAggregate. Their
        // per-REV polar fits are only needed after a contact or load wakes them.
        return 0.0;
    }
    fill_visited(&mut scratch.rotation_gradients, n, Mat3::ZERO, &live);
    fill_visited(&mut scratch.rotation_counts, n, Vec3::ZERO, &live);
    for axis in 0..3 {
        for i in visit(&live, n) {
            let j = scene.bonds[axis][i].neighbor;
            if j == EMPTY {
                continue;
            }
            let j = j as usize;
            if scene.voxels[i].kinematic && scene.voxels[j].kinematic {
                continue;
            }
            let d = ((scene.positions[j] - scene.positions[i]) / DCELL).as_vec3();
            let mut m = Mat3::ZERO;
            match axis {
                0 => m.x_axis = d,
                1 => m.y_axis = d,
                _ => m.z_axis = d,
            };
            if !scene.voxels[i].kinematic && scene.aggregate_of[i].is_none() {
                scratch.rotation_gradients[i] += m;
                scratch.rotation_counts[i][axis] += 1.0;
            }
            if !scene.voxels[j].kinematic && scene.aggregate_of[j].is_none() {
                scratch.rotation_gradients[j] += m;
                scratch.rotation_counts[j][axis] += 1.0;
            }
        }
    }
    let gradients = &scratch.rotation_gradients;
    let counts = &scratch.rotation_counts;
    let fixed = &scene.voxels;
    let update_rotation = |(i, q): (usize, &mut Quat)| {
        if fixed[i].kinematic || scene.aggregate_of[i].is_some() {
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
    if live.is_some() {
        // Only awake REVs rotate; a sleeper returns 0 from `update_rotation`.
        let mut largest = 0.0_f32;
        for i in visit(&live, n) {
            largest = largest.max(update_rotation((i, &mut scene.rotations[i])));
        }
        largest
    } else if !parallel_enabled_for(n) && n < ROTATION_PARALLEL_MIN_VOXELS {
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

/// First-order ambient-air coupling for the solid REV surface. Intact bonds
/// hide their shared face; a broken or external face sees air. Pressure drag
/// is integrated facewise so internal REV faces do not create artificial area.
fn atmosphere_loads(scene: &Scene, exposed_faces: &mut Vec<[u8; 3]>, loads: &mut Vec<Vec3>) {
    let air = scene.config.atmosphere;
    let density = air.density_kg_m3();
    let voxel_count = scene.voxels.len();
    let live = scene.step_live.clone();
    if density <= 0.0 {
        fill_visited(loads, voxel_count, Vec3::ZERO, &live);
        return;
    }
    // A sleeper is kinematic: its load is zero, whatever its faces see.
    if live.is_none() || loads.len() != voxel_count {
        loads.clear();
        loads.resize(voxel_count, Vec3::ZERO);
    }

    fill_visited(exposed_faces, voxel_count, [2u8; 3], &live);
    for axis in 0..3 {
        for i in visit(&live, voxel_count) {
            let bond = &scene.bonds[axis][i];
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if j >= scene.voxels.len() || i == j {
                continue;
            }
            if live.as_ref().is_some_and(|live| !live.in_list[j]) {
                continue;
            }
            exposed_faces[i][axis] = exposed_faces[i][axis].saturating_sub(1);
            exposed_faces[j][axis] = exposed_faces[j][axis].saturating_sub(1);
        }
    }

    let wind = Vec3::from_array(air.wind_velocity_m_s);
    let local_axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let voxels = &scene.voxels;
    let properties = &scene.properties;
    let rotations = &scene.rotations;
    let evaluate_load = |i: usize, load: &mut Vec3| {
        let voxel = &voxels[i];
        if voxel.kinematic {
            *load = Vec3::ZERO;
            return;
        }
        let mut force = Vec3::ZERO;
        let material_density = properties[i].density;
        if material_density > 0.0 {
            let displaced_air_mass = density * voxel.mass / material_density;
            force -= Vec3::from_array(scene.config.gravity) * displaced_air_mass;
        }

        let relative_velocity = voxel.velocity - wind;
        for axis in 0..3 {
            let faces = exposed_faces[i][axis] as f32;
            if faces == 0.0 {
                continue;
            }
            let normal = (rotations[i] * local_axes[axis]).normalize_or_zero();
            let normal_speed = relative_velocity.dot(normal);
            // Opposing faces both contribute to pressure drag. Splitting
            // Cd evenly between them gives D = Cd * q * projected_area.
            let face_force = -normal
                * (0.25
                    * density
                    * air.drag_coefficient
                    * AREA
                    * normal_speed
                    * normal_speed.abs());
            force += face_force * faces;
        }
        *load = force;
    };
    if live.is_some() {
        for i in visit(&live, voxel_count) {
            evaluate_load(i, &mut loads[i]);
        }
    } else if atmosphere_parallel_enabled_for(voxel_count) {
        loads
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, load)| evaluate_load(i, load));
    } else {
        loads
            .iter_mut()
            .enumerate()
            .for_each(|(i, load)| evaluate_load(i, load));
    }
}

/// A worst-case chain bound rejects soft/slender bodies before coarse motion
/// can hide internal displacement (0.5% REV) or stress (0.1% of strength).
fn small_rigid_loading(scene: &Scene, members: &[usize], loads: &[Vec3]) -> bool {
    if members.is_empty() {
        return false;
    }
    let mut mass = 0.0_f64;
    let mut force = DVec3::ZERO;
    let mut stiffness = f64::INFINITY;
    let mut strength = f64::INFINITY;
    for &i in members {
        let v = &scene.voxels[i];
        let m = &scene.properties[i];
        if v.mass <= 0.0
            || !v.mass.is_finite()
            || scene.external_forces[i] != Vec3::ZERO
            || scene.external_accelerations[i] != Vec3::ZERO
        {
            return false;
        }
        mass += f64::from(v.mass);
        force += loads.get(i).copied().unwrap_or(Vec3::ZERO).as_dvec3();
        stiffness = stiffness.min(f64::from(
            m.young.min(m.young_perp).min(m.shear) * AREA / CELL,
        ));
        strength = strength.min(f64::from(
            m.tensile_strength
                .min(m.tensile_strength_perp)
                .min(m.compressive_strength)
                .min(m.compressive_strength_perp)
                .min(m.shear_strength)
                .min(m.shear_strength_perp),
        ));
    }
    if mass <= 0.0 || stiffness <= 0.0 || strength <= 0.0 || !force.is_finite() {
        return false;
    }
    let acceleration = force / mass;
    let mut differential = 0.0_f64;
    for &i in members {
        let load = loads.get(i).copied().unwrap_or(Vec3::ZERO).as_dvec3();
        differential =
            differential.max((load - acceleration * f64::from(scene.voxels[i].mass)).length());
    }
    let n = members.len() as f64;
    differential * n * n / stiffness <= DCELL * 0.005
        && differential * n / f64::from(AREA) <= strength * 0.001
}

fn quiescent_supported(
    scene: &Scene,
    scratch: &SolverScratch,
    gravity: Vec3,
    loads: &[Vec3],
) -> bool {
    if scratch.adaptive_gravity != Some(gravity)
        || scratch.adaptive_energies.len() != scene.voxels.len()
        || scratch.adaptive_loads.len() != loads.len()
        || scratch.adaptive_loads.iter().zip(loads).enumerate().any(|(i, (old, now))| {
            let scale = old.length().max(now.length()).max(scene.voxels[i].mass * gravity.length());
            (*old - *now).length() > 8.0 * f32::EPSILON * scale
        })
        || scratch.adaptive_positions != scene.positions
        || scratch.adaptive_revision != Some(scene.render_revision)
        || bincode::serialize(&scene.config).ok().as_ref() != Some(&scratch.adaptive_config)
        || scratch.contacts.is_empty()
        || !scene.releases.is_empty()
        || !scene.solver_residual.is_finite()
        || scene.solver_residual > scene.config.solver_relative_tolerance
    {
        return false;
    }
    let mut supported = vec![false; scene.cluster_count];
    for (i, v) in scene.voxels.iter().enumerate() {
        if v.kinematic || scene.axis_targets[i].iter().any(Option::is_some) {
            if let Some(supported) = supported.get_mut(scene.clusters[i] as usize) {
                *supported = true;
            }
        }
    }
    for c in &scratch.contacts {
        if c.i >= scene.voxels.len() || c.j >= scene.voxels.len() {
            return false;
        }
        if scene.voxels[c.i].kinematic {
            if let Some(supported) = supported.get_mut(scene.clusters[c.j] as usize) {
                *supported = true;
            }
        }
        if scene.voxels[c.j].kinematic {
            if let Some(supported) = supported.get_mut(scene.clusters[c.i] as usize) {
                *supported = true;
            }
        }
    }
    for (i, v) in scene.voxels.iter().enumerate() {
        if v.kinematic {
            continue;
        }
        let m = &scene.properties[i];
        if !supported
            .get(scene.clusters[i] as usize)
            .copied()
            .unwrap_or(false)
            || !v.velocity.is_finite()
            || v.velocity.length_squared() > 1e-10
            || (v.thermal_energy_f32() - scratch.adaptive_energies[i]).abs()
                > v.mass * m.heat_capacity * 0.001
            || m.phase_state(v.mass, v.thermal_energy_f32()).solid_fraction < 1.0
        {
            return false;
        }
    }
    scene.bonds.iter().enumerate().all(|(axis, bonds)| {
        bonds.iter().enumerate().all(|(i, b)| {
            if b.neighbor == EMPTY {
                return true;
            }
            let m = &scene.properties[i];
            let strength = m
                .tensile_strength
                .min(m.tensile_strength_perp)
                .min(m.compressive_strength)
                .min(m.compressive_strength_perp)
                .min(m.shear_strength)
                .min(m.shear_strength_perp);
            b.damage == 0.0
                && b.equivalent_plastic_strain == 0.0
                && scene.workspace[axis][i].last_force.length() <= strength * AREA * 0.01
        })
    })
}

fn wake_candidates_with_differential_loading(scene: &mut Scene, gravity: Vec3, loads: &[Vec3]) {
    if !scene.has_registered_rigid_candidates() {
        return;
    }
    let mut wake_voxels = Vec::new();
    for aggregate in scene.rigid_aggregates.iter().flatten() {
        if scene.adaptive_substeps {
            let members = aggregate.member_indices().collect::<Vec<_>>();
            if small_rigid_loading(scene, &members, loads) {
                continue;
            }
        }
        let mut reference_acceleration: Option<Vec3> = None;
        let mut nonuniform = false;
        for member in aggregate.member_poses() {
            let mass = member.mass_kg as f32;
            if !mass.is_finite() || mass <= 0.0 {
                nonuniform = true;
                break;
            }
            let nodal_load = loads.get(member.index).copied().unwrap_or(Vec3::ZERO);
            let acceleration = gravity + nodal_load / mass;
            if let Some(reference) = reference_acceleration {
                let scale = reference.length().max(acceleration.length()).max(1.0);
                let roundoff = 8.0 * f32::EPSILON * scale;
                if (acceleration - reference).length() > roundoff {
                    nonuniform = true;
                    break;
                }
            } else {
                reference_acceleration = Some(acceleration);
            }
        }
        if nonuniform {
            wake_voxels.extend(aggregate.member_indices());
        }
    }
    scene.wake_rigid_candidates_for_voxels(&wake_voxels);
}

/// Coarsen only complete components compatible with free rigid motion: intact
/// bonds carry no force above the configured solver error scale, the material
/// is at its reference thermal energy and fully solid, and there are no
/// constraints or applied nodal loads. Aggregate wake paths refine the
/// retained REV state before contact, nonuniform loading, thermal changes, or
/// boundary removal. This is a broadphase/DOF reduction for inert bodies, not
/// a rule that makes stressed material rigid.
fn auto_register_quiescent_components(
    scene: &mut Scene,
    targets: &[[Option<f64>; 3]],
    applied_loads: &[Vec3],
    contacts: &[collision::Contact],
) -> usize {
    let cluster_count = scene.cluster_count;
    if cluster_count == 0 || scene.clusters.len() != scene.voxels.len() {
        return 0;
    }

    // Build one compact component-member table. Avoid a Vec allocation for
    // every singleton component in large sparse scenes.
    let mut sizes = vec![0usize; cluster_count];
    let mut member_count = 0usize;
    for &cluster in &scene.clusters {
        if let Some(size) = sizes.get_mut(cluster as usize) {
            *size += 1;
            member_count += 1;
        }
    }
    if member_count == 0 {
        return 0;
    }
    let mut offsets = vec![0usize; cluster_count + 1];
    for cluster in 0..cluster_count {
        offsets[cluster + 1] = offsets[cluster] + sizes[cluster];
    }
    let mut next = offsets[..cluster_count].to_vec();
    let mut members = vec![0usize; member_count];
    for (index, &cluster) in scene.clusters.iter().enumerate() {
        let cluster = cluster as usize;
        if let Some(cursor) = next.get_mut(cluster) {
            members[*cursor] = index;
            *cursor += 1;
        }
    }
    let mut contacted = vec![false; scene.voxels.len()];
    for contact in contacts {
        contacted[contact.i] = true;
        contacted[contact.j] = true;
    }

    let mut registered_revs = 0usize;
    let gravity = Vec3::from_array(scene.config.gravity);
    let gravity_magnitude = gravity.length();
    for cluster in 0..cluster_count {
        if sizes[cluster] < AUTO_AGGREGATE_MIN_REVS {
            continue;
        }
        let component = &members[offsets[cluster]..offsets[cluster + 1]];
        if component
            .iter()
            .any(|&index| scene.aggregate_of[index].is_some() || contacted[index])
        {
            continue;
        }

        let weak_field =
            scene.adaptive_substeps && small_rigid_loading(scene, component, applied_loads);
        let unconstrained_and_unloaded = component.iter().all(|&index| {
            let voxel = &scene.voxels[index];
            let reference_energy = scene.properties[index]
                .energy_at_temperature(voxel.mass, scene.reference_temperatures_kelvin[index]);
            !voxel.kinematic
                && targets[index].iter().all(Option::is_none)
                && scene.external_forces[index] == Vec3::ZERO
                && scene.external_accelerations[index] == Vec3::ZERO
                && (applied_loads[index] == Vec3::ZERO || weak_field)
                && voxel.thermal_energy_f32().to_bits() == reference_energy.to_bits()
                && scene.properties[index]
                    .phase_state(voxel.mass, voxel.thermal_energy_f32())
                    .solid_fraction
                    == 1.0
        });
        if !unconstrained_and_unloaded {
            continue;
        }

        // A zero nodal resultant alone is insufficient: stored bond forces
        // represent elastic modes that this rigid tier cannot evolve. Ignore
        // only residuals bounded by the linear solve tolerance, scaled by the
        // component's applied and inertial load; retain real self-stress.
        let reference_load_n = component.iter().fold(0.0_f32, |maximum, &index| {
            let inertial_load = scene.voxels[index].mass * gravity_magnitude;
            maximum.max(applied_loads[index].length() + inertial_load)
        });
        if !reference_load_n.is_finite() {
            continue;
        }
        let force_tolerance_n = scene.config.solver_relative_tolerance * reference_load_n;
        let force_free = (0..3).all(|axis| {
            component.iter().all(|&index| {
                let bond = scene.bonds[axis][index];
                if bond.neighbor == EMPTY {
                    return true;
                }
                if bond.damage != 0.0
                    || bond.equivalent_plastic_strain != 0.0
                    || scene.workspace[axis][index].last_force.length() > force_tolerance_n
                {
                    return false;
                }
                if scene.substep_index == 0 {
                    let j = bond.neighbor as usize;
                    let frame = basis(axis, scene.rotations[index].slerp(scene.rotations[j], 0.5));
                    // Last-force buffers have not been evaluated at startup.
                    // Inspect retained strain rather than treating zero buffers
                    // as proof that an authored/prestressed body is unloaded.
                    let displacement = bond_elastic_displacement(scene, index, j, axis, frame)
                        - Vec3::from_array(bond.plastic_strain) * CELL;
                    displacement.length() <= CELL * 1e-7
                } else {
                    true
                }
            })
        });
        if !force_free {
            continue;
        }

        if scene
            .register_rigid_aggregate_candidate_in_cluster(
                component,
                cluster as u32,
                sizes[cluster],
            )
            .is_ok()
        {
            registered_revs += component.len();
        }
    }
    registered_revs
}

pub fn step(scene: &mut Scene, tool: &mut Option<Tool>) {
    step_with_targets(scene, tool, &[]);
}

/// `dst[i] = value` for the visited REVs; a destination of the wrong length is rebuilt
/// whole, so entries of sleepers outside the visit list hold `value` from the start.
fn fill_visited<T: Clone>(dst: &mut Vec<T>, n: usize, value: T, live: &Option<std::sync::Arc<crate::scene::Live>>) {
    match live {
        Some(live) if dst.len() == n => {
            for &i in &live.list {
                dst[i].clone_from(&value);
            }
        }
        _ => {
            dst.clear();
            dst.resize(n, value);
        }
    }
}
pub fn step_with_targets(scene: &mut Scene, tool: &mut Option<Tool>, targets: &[Option<DVec3>]) {
    let axes: Vec<_> = targets
        .iter()
        .map(|p| p.map(|v| v.to_array().map(Some)).unwrap_or([None; 3]))
        .collect();
    step_with_axis_targets(scene, tool, &axes);
}
pub fn step_with_axis_targets(
    scene: &mut Scene,
    tool: &mut Option<Tool>,
    targets: &[[Option<f64>; 3]],
) {
    step_with_axis_targets_and_loads(scene, tool, targets, &[]);
}

/// Advance with optional per-step world-space nodal forces in newtons, in
/// addition to persistent inputs stored on `Scene`. All enter the same coupled
/// solve as bonds and are shared by every substep.
pub fn step_with_axis_targets_and_loads(
    scene: &mut Scene,
    tool: &mut Option<Tool>,
    targets: &[[Option<f64>; 3]],
    loads: &[Vec3],
) {
    let dt = f64::from(scene.config.fixed_dt);
    if !crate::live_tests::enabled() {
        advance_step_with_axis_targets_and_loads(scene, tool, targets, loads);
        scene.set_step_live(None);
        scene.simulated_seconds += dt;
        return;
    }
    let started = Instant::now();
    let thread = std::thread::current();
    let test_name = thread.name().filter(|name| *name != "main").unwrap_or("");
    if scene.simulated_seconds == 0.0 {
        crate::live_tests::observe_scene(scene, test_name, 0.0, 0.0);
    }
    advance_step_with_axis_targets_and_loads(scene, tool, targets, loads);
    scene.set_step_live(None);
    scene.simulated_seconds += dt;
    crate::live_tests::observe_scene(
        scene, test_name, scene.simulated_seconds, started.elapsed().as_secs_f64() * 1000.0,
    );
}

/// One fixed step of a scene in which nothing mechanical is awake (every REV fixed or
/// asleep): heat conduction and chemistry only. Positions, velocities, strain and the
/// mechanical energy ledger are untouched because nothing can change them; a sleeper
/// whose heat or composition changes wakes at the start of the next step.
fn dormant_step(scene: &mut Scene) {
    let substeps = scene.config.substeps.max(1);
    let h = scene.config.fixed_dt / substeps as f32;
    scene.last_step_diagnostics = None;
    for _ in 0..substeps {
        match scene.conduct_heat(h) {
            Ok(used) if used > 0 => scene.mark_render_changed(),
            Ok(_) => {}
            Err(reason) => eprintln!("thermal transport skipped: {reason}"),
        }
        scene.substep_index += 1;
    }
    if let Some((i, j)) = scene.take_pending_fusion() {
        match scene.fuse_adjacent_voxels(i, j) {
            Ok(_) => rebuild_clusters(scene),
            Err(reason) => eprintln!("automatic thermal fusion cancelled for REV {i}/{j}: {reason}"),
        }
    }
    if scene.chemistry_enabled {
        let _ = scene.advance_chemistry(f64::from(scene.config.fixed_dt));
    }
}

fn advance_step_with_axis_targets_and_loads(
    scene: &mut Scene,
    tool: &mut Option<Tool>,
    targets: &[[Option<f64>; 3]],
    loads: &[Vec3],
) {
    let step_entered = Instant::now();
    CONTACT_PCG_PROFILE.with(|slot| slot.set(ContactPcgProfile::default()));
    assert!(targets.is_empty() || targets.len() == scene.voxels.len());
    assert!(loads.is_empty() || loads.len() == scene.voxels.len());
    scene.solver_residual_audit = None;
    scene.last_step_diagnostics = None;
    scene.run_sleep();
    let sleep_done = Instant::now();
    // A scene with sleepers visits only the awake REVs (and the sleepers bonded to them)
    // in the passes below; a sleeper is frozen, so skipping it changes nothing. Anything
    // that can reach into a sleeper from outside the fixed-step model keeps the full pass.
    let plain_step = tool.is_none()
        && targets.is_empty()
        && loads.is_empty()
        && !scene.adaptive_substeps
        && scene.skeletal_actuators.is_empty()
        && scene.tier_governor.is_none()
        && scene.thermal_bonding_rule().is_none()
        && scene.thermal_fusion_rule().is_none()
        && !scene.has_registered_rigid_candidates()
        && scene.beam_regions.iter().all(Option::is_none);
    let live = if plain_step { scene.sparse_live_for_step() } else { None };
    scene.set_step_live(live.clone());
    if tool.is_none() && targets.is_empty() && loads.is_empty() && scene.mechanically_dormant() {
        dormant_step(scene);
        scene.set_step_live(None);
        return;
    }
    scene.run_tier_governor();
    scene.advance_beam_regions(f64::from(scene.config.fixed_dt));
    scene.cell_dofs = decide_cell_dofs(scene);
    scene.solver_max_audited_residual = None;
    // Keep the large per-step and per-substep work arrays alive across calls.
    let mut solver_scratch = std::mem::take(&mut scene.solver_scratch);
    let mut melted = std::mem::take(&mut solver_scratch.melted);
    let mut topology_changed = break_fully_melted_bonds(scene, &mut melted);
    if topology_changed {
        scene.wake_all_rigid_candidates();
    }
    scene.wake_rigid_candidates_with_thermal_changes();
    // Explicit per-step coordinates override persistent constraints on that axis.
    let voxel_count = scene.voxels.len();
    let mut merged_targets = std::mem::take(&mut solver_scratch.merged_targets);
    if live.is_some() && merged_targets.len() == voxel_count {
        // No per-step targets here: the merge is the persistent constraint itself, and a
        // sleeper's constraint cannot change without waking it.
        for i in visit(&live, voxel_count) {
            merged_targets[i] = scene.axis_targets[i];
        }
    } else {
        merged_targets.clear();
        merged_targets.reserve(voxel_count);
        merged_targets.extend(
            scene.axis_targets.iter().enumerate().map(|(i, fixed)| {
                std::array::from_fn(|a| targets.get(i).and_then(|t| t[a]).or(fixed[a]))
            }),
        );
    }
    assert!(visit(&live, voxel_count)
        .all(|i| merged_targets[i].iter().flatten().all(|p| p.is_finite())));
    let targets = merged_targets.as_slice();
    if scene.has_registered_rigid_candidates() {
        let constrained: Vec<_> = targets
            .iter()
            .enumerate()
            .filter_map(|(voxel, target)| {
                (scene.voxels[voxel].kinematic || target.iter().any(Option::is_some))
                    .then_some(voxel)
            })
            .collect();
        scene.wake_rigid_candidates_for_voxels(&constrained);
    }
    let mut atmosphere_exposed_faces = std::mem::take(&mut solver_scratch.atmosphere_exposed_faces);
    let mut ambient_loads = std::mem::take(&mut solver_scratch.ambient_loads);
    atmosphere_loads(scene, &mut atmosphere_exposed_faces, &mut ambient_loads);
    let mut applied_loads = std::mem::take(&mut solver_scratch.applied_loads);
    applied_loads.resize(voxel_count, Vec3::ZERO);
    let has_skeletal_actuators = !scene.skeletal_actuators.is_empty();
    if has_skeletal_actuators {
        applied_loads.fill(Vec3::ZERO);
        for actuator in &mut scene.skeletal_actuators {
            let _ = actuator.accumulate_forces(
                &scene.positions,
                &scene.voxels,
                &scene.rotations,
                &scene.properties,
                f64::from(scene.config.fixed_dt),
                &mut applied_loads,
            );
        }
    }
    for i in visit(&live, voxel_count) {
        let voxel = &scene.voxels[i];
        applied_loads[i] = if !voxel.kinematic {
            scene.external_forces[i]
                + scene.external_accelerations[i] * voxel.mass
                + loads.get(i).copied().unwrap_or(Vec3::ZERO)
                + ambient_loads[i]
                + if has_skeletal_actuators {
                    applied_loads[i]
                } else {
                    Vec3::ZERO
                }
        } else {
            Vec3::ZERO
        };
    }
    let gravity = Vec3::from_array(scene.config.gravity);
    wake_candidates_with_differential_loading(scene, gravity, &applied_loads);
    let may_adapt = scene.adaptive_substeps
        && tool.is_none()
        && scene.skeletal_actuators.is_empty()
        && loads.is_empty()
        && scene.releases.is_empty()
        && !topology_changed
        && scene.thermal_bonding_rule().is_none()
        && scene.thermal_fusion_rule().is_none()
        && scene.external_forces.iter().all(|f| *f == Vec3::ZERO)
        && scene
            .external_accelerations
            .iter()
            .all(|a| *a == Vec3::ZERO)
        && targets.iter().zip(&scene.positions).all(|(target, p)| {
            target
                .iter()
                .enumerate()
                .all(|(a, value)| value.map(|v| v == p[a]).unwrap_or(true))
        });
    if may_adapt && quiescent_supported(scene, &solver_scratch, gravity, &applied_loads) {
        // A supported equilibrium under identical loads has no mechanical work.
        // Keep its complete REV/bond state; heat transport still advances by dt.
        // The certificate stays anchored to the last actual mechanical solve,
        // so successive small thermal changes eventually wake the material.
        scene.last_mechanical_substeps = 0;
        scene.previous.clone_from(&scene.positions);
        for voxel in &mut scene.voxels {
            if !voxel.kinematic {
                voxel.velocity = Vec3::ZERO;
            }
        }
        if let Err(reason) = scene.conduct_heat(scene.config.fixed_dt) {
            eprintln!("thermal transport skipped: {reason}");
        }
        scene.substep_index += 1;
        scene.mark_render_changed();
        solver_scratch.adaptive_revision = Some(scene.render_revision);
        solver_scratch.merged_targets = merged_targets;
        solver_scratch.atmosphere_exposed_faces = atmosphere_exposed_faces;
        solver_scratch.ambient_loads = ambient_loads;
        solver_scratch.applied_loads = applied_loads;
        solver_scratch.melted = melted;
        scene.solver_scratch = solver_scratch;
        scene.set_step_live(None);
        return;
    }
    if may_adapt && scene.substep_index == 0 {
        auto_register_quiescent_components(scene, targets, &applied_loads, &[]);
    }
    let coarse_clear = may_adapt
        && collision::rigid_candidates_clear_for_step(
            scene,
            f64::from(scene.config.fixed_dt),
            gravity.as_dvec3(),
            &applied_loads,
        );
    if scene.adaptive_substeps && !coarse_clear && scene.has_registered_rigid_candidates() {
        scene.wake_all_rigid_candidates();
    }
    let effective_substeps = if may_adapt && coarse_clear {
        1
    } else {
        scene.config.substeps
    };
    scene.last_mechanical_substeps = effective_substeps;
    let h = scene.config.fixed_dt / effective_substeps as f32;
    // Scenes restored from an editor snapshot carry no cell spin.
    scene.angular_velocities.resize(scene.voxels.len(), DVec3::ZERO);
    sync_precise_rotations(scene);
    let log_profile_step = std::env::var_os("VOXEL_PROFILE_STEP").is_some();
    let profile_step = log_profile_step || crate::runtime_metrics::is_active();
    let mut contacts_peak = 0usize;
    let mut new_impact_observations = 0usize;
    let mut stage_time = [Duration::ZERO; 5];
    let mut total_pcg_iterations = 0usize;
    // A later substep or corotation pass overwrites `solver_residual`, so a
    // failed solve has to be remembered for the whole fixed step.
    let mut solve_failed_this_step = false;
    let mut damping_cache_hits = 0usize;
    let mut solve_detail_time = [Duration::ZERO; 3];
    let mut contact_finish_time = Duration::ZERO;
    let mut tool_recheck_time = Duration::ZERO;
    let mut release_time = Duration::ZERO;
    let mut damping_time = Duration::ZERO;
    let mut thermal_time = Duration::ZERO;
    let mut aggregate_scan_time = Duration::ZERO;
    let mut auto_aggregated_revs = 0usize;
    let mut rotation_fit_time = Duration::ZERO;
    let mut pre_solve_detail_time = [Duration::ZERO; 4];
    let mut energy_capture_time = Duration::ZERO;
    let mut contact_energy_time = Duration::ZERO;
    let mut free_energy_time = Duration::ZERO;
    // Reuse large solver buffers across substeps and corotation corrections.
    // A failed device solve is unlikely to recover on another corotation
    // iteration/substep of this same fixed step (same device and scene state).
    // Fall back once, then avoid repeating the costly CUDA setup until the next
    // fixed step, where the matrix and residual have changed.
    let mut cuda_enabled_for_step = true;
    // Reuse a prior substep solution as an initial guess only while the bond
    // graph remains unchanged. The current matrix residual is checked against
    // the configured tolerance inside `solve_coupled` before accepting it.
    let mut reusable_cuda_solution = false;
    let voxel_count = scene.voxels.len();
    let mut substep_start_positions = std::mem::take(&mut solver_scratch.substep_start_positions);
    let mut substep_start_velocities = std::mem::take(&mut solver_scratch.substep_start_velocities);
    let mut load_integrated_velocities =
        std::mem::take(&mut solver_scratch.load_integrated_velocities);
    load_integrated_velocities.resize(voxel_count, Vec3::ZERO);
    let mut contact_work_mode = std::mem::take(&mut solver_scratch.contact_work_mode);
    contact_work_mode.resize(voxel_count, 0);
    let mut tool_contact_velocity = std::mem::take(&mut solver_scratch.tool_contact_velocity);
    let mut drift_velocity = std::mem::take(&mut solver_scratch.drift_velocity);
    let mut predicted = std::mem::take(&mut solver_scratch.predicted);
    let mut solved_positions = std::mem::take(&mut solver_scratch.solved_positions);
    let mut pre_contact_velocities = std::mem::take(&mut solver_scratch.pre_contact_velocities);
    let mut contacts = std::mem::take(&mut solver_scratch.contacts);
    let mut impact_energies_j = std::mem::take(&mut solver_scratch.impact_energies_j);
    let mut normal_impulses_ns = std::mem::take(&mut solver_scratch.normal_impulses_ns);
    let mut implicit_friction_contacts = std::mem::take(&mut solver_scratch.implicit_friction_contacts);
    let mut support_friction_modes = std::mem::take(&mut solver_scratch.support_friction_modes);
    let mut support_patch_normal = std::mem::take(&mut solver_scratch.support_patch_normal);
    let mut solve_loads = std::mem::take(&mut solver_scratch.solve_loads);
    let mut friction_used_ns = std::mem::take(&mut solver_scratch.friction_used_ns);
    let mut contact_physical_velocities =
        std::mem::take(&mut solver_scratch.contact_physical_velocities);
    let mut contact_targets = std::mem::take(&mut solver_scratch.contact_targets);
    let mut substep_mechanical_energy = std::mem::take(&mut solver_scratch.substep_mechanical_energy);
    let mut energy_edge_values = std::mem::take(&mut solver_scratch.energy_edge_values);
    let mut free_implicit_budgets = std::mem::take(&mut solver_scratch.free_implicit_budgets);
    let mut free_implicit_components = std::mem::take(&mut solver_scratch.free_implicit_components);
    let mut free_implicit_heat = std::mem::take(&mut solver_scratch.free_implicit_heat);
    let loop_entered = Instant::now();
    for _ in 0..effective_substeps {
        let mut phase_start = Instant::now();
        let mut detail_start = profile_step.then(Instant::now);
        sync_visited(&mut substep_start_positions, &scene.positions, &live);
        if substep_start_velocities.len() != voxel_count {
            substep_start_velocities.clear();
            substep_start_velocities.extend(scene.voxels.iter().map(|voxel| voxel.velocity));
        } else {
            for i in visit(&live, voxel_count) {
                substep_start_velocities[i] = scene.voxels[i].velocity;
            }
        }
        // The energy budget starts before drift. A prediction made from an
        // impact's nonuniform velocities can already contain strain; treating
        // that trial strain as incoming energy counts it a second time.
        let energy_capture_start = profile_step.then(Instant::now);
        let substep_elastic_energy_before = if coarse_clear {
            0.0
        } else {
            capture_nodal_mechanical_energy(
                scene,
                &mut substep_mechanical_energy,
                &mut energy_edge_values,
                energy_edges_parallel_enabled_for(scene.voxels.len()),
            )
        };
        if let Some(start) = energy_capture_start { energy_capture_time += start.elapsed(); }
        match &live {
            // A sleeper's `previous` was set equal to its position when it fell asleep.
            Some(live) => {
                for &i in &live.list {
                    scene.previous[i] = scene.positions[i];
                }
            }
            None => scene.previous.clone_from(&scene.positions),
        }
        if let Some(start) = detail_start {
            pre_solve_detail_time[0] += start.elapsed();
            detail_start = Some(Instant::now());
        }
        drift_velocity.resize(scene.voxels.len(), Vec3::ZERO);
        if scene.has_registered_rigid_candidates() {
            let gravity_d = DVec3::new(gravity.x as f64, gravity.y as f64, gravity.z as f64);
            if scene
                .advance_registered_rigid_candidates(
                    h as f64,
                    gravity_d,
                    &applied_loads,
                    coarse_clear.then_some(f64::from(
                        scene.config.fixed_dt / scene.config.substeps as f32,
                    )),
                )
                .is_err()
            {
                scene.wake_all_rigid_candidates();
            }
        }
        let mut all_dynamic_coarse = scene.has_registered_rigid_candidates();
        let mut target_motion = false;
        for i in visit(&live, voxel_count) {
            let voxel = &scene.voxels[i];
            if !voxel.kinematic && scene.aggregate_of[i].is_none() {
                all_dynamic_coarse = false;
            }
            if let Some(target) = targets.get(i) {
                for axis in 0..3 {
                    if let Some(position) = target[axis] {
                        if !voxel.kinematic || scene.positions[i][axis] != position {
                            target_motion = true;
                        }
                    }
                }
            }
        }
        let coarse_prediction = tool.is_none() && all_dynamic_coarse && !target_motion;
        if !coarse_prediction {
            let aggregate_membership = &scene.aggregate_of;
            for i in visit(&live, voxel_count) {
                let (p, v) = (&mut scene.positions[i], &mut scene.voxels[i]);
                if v.kinematic {
                    drift_velocity[i] = Vec3::ZERO;
                    continue;
                }
                if aggregate_membership[i].is_some() {
                    drift_velocity[i] = v.velocity;
                    continue;
                }
                v.velocity += gravity * h;
                for a in 0..3 {
                    if targets[i][a].is_some() {
                        v.velocity[a] = 0.0;
                    }
                }
                drift_velocity[i] = v.velocity;
                *p += v.velocity.as_dvec3() * h as f64;
            }
        }
        if target_motion {
            for (i, target) in targets.iter().enumerate() {
                for axis in 0..3 {
                    if let Some(p) = target[axis] {
                        scene.positions[i][axis] = p;
                    }
                }
            }
        }
        if !coarse_prediction {
            for work in &mut scene.workspace {
                for i in visit(&live, voxel_count) {
                    work[i].lambda = Vec3::ZERO;
                }
            }
        }
        if let Some(start) = detail_start {
            pre_solve_detail_time[1] += start.elapsed();
            detail_start = Some(Instant::now());
        }
        if let Some(t) = tool {
            t.advance_into(scene, h, targets, &mut tool_contact_velocity);
        } else {
            if !coarse_prediction {
                fill_visited(&mut tool_contact_velocity, voxel_count, Vec3::ZERO, &live);
            } else {
                tool_contact_velocity.resize(scene.positions.len(), Vec3::ZERO);
            }
        }
        if let Some(t) = tool.as_ref() {
            if scene.has_registered_rigid_candidates() && !t.contacted_voxels.is_empty() {
                scene.wake_rigid_candidates_for_voxels(&t.contacted_voxels);
            }
        }
        if let Some(start) = detail_start {
            pre_solve_detail_time[2] += start.elapsed();
            detail_start = Some(Instant::now());
        }
        // Treat a contact impulse as acting near the middle of this substep.
        // Without this half-step drift, the new velocity is only integrated
        // on the next substep, after the implicit bond solve has already
        // damped the impact. That delayed stress made a sharp tool stop at the
        // surface without loading bonds enough to fail.
        if tool.is_some() {
            for (p, &delta_v) in scene.positions.iter_mut().zip(&tool_contact_velocity) {
                *p += delta_v.as_dvec3() * (0.5 * h as f64);
            }
        }
        sync_visited(&mut predicted, &scene.positions, &live);
        // Fit material frames from the physical prediction. Geometric contact
        // projection can temporarily flatten multiple swept layers onto one
        // plane; that infeasible intermediate state must not rotate the rest
        // frames or create apparent strain energy.
        let rotation_start = profile_step.then(Instant::now);
        if !coarse_clear {
            update_frames(scene, &mut solver_scratch);
        }
        if let Some(start) = rotation_start {
            rotation_fit_time += start.elapsed();
        }
        // A fixed surface is a unilateral position constraint in the SAME
        // inertial/elastic minimization. Projecting only surface REVs after
        // that solve injected unresolved compression into bonds, whose huge
        // apparent strain energy was then spent on heat and fracture.
        // The inertial reference remains the unprojected prediction.
        sync_visited(&mut contact_targets, targets, &live);
        if live.is_some() && solver_scratch.contact_axis_normals.len() == voxel_count {
            sync_visited(
                &mut solver_scratch.previous_contact_axis_normals,
                &solver_scratch.contact_axis_normals,
                &live,
            );
        } else {
            solver_scratch
                .previous_contact_axis_normals
                .clone_from(&solver_scratch.contact_axis_normals);
        }
        fill_visited(&mut solver_scratch.contact_axis_normals, voxel_count, [0.0; 3], &live);
        fill_visited(&mut solver_scratch.contact_lower, voxel_count, [None; 3], &live);
        fill_visited(&mut solver_scratch.contact_upper, voxel_count, [None; 3], &live);
        if coarse_clear {
            // The whole-step conservative sweep already certified every body
            // against every other body and fixed REV. A looser collision-bin
            // query can report a false positive and wake a body after its
            // coarse drift, which would run fine elasticity at the coarse h.
            // Retain the certified rigid motion until the next step's test.
            contacts.clear();
        } else {
            collision::project_into(scene, targets, &mut contacts);
        }
        if coarse_prediction
            && scene
                .voxels
                .iter()
                .enumerate()
                .any(|(i, voxel)| !voxel.kinematic && scene.aggregate_of[i].is_none())
        {
            // A swept coarse bound may wake a body inside collision projection.
            // Restore the fine-step buffers before the ordinary solve consumes
            // them; aggregate-only steps never need these per-REV values.
            drift_velocity.fill(Vec3::ZERO);
            tool_contact_velocity.fill(Vec3::ZERO);
            for (i, voxel) in scene.voxels.iter().enumerate() {
                if !voxel.kinematic && scene.aggregate_of[i].is_none() {
                    drift_velocity[i] = voxel.velocity;
                }
            }
            for work in &mut scene.workspace {
                for w in work {
                    w.lambda = Vec3::ZERO;
                }
            }
        }
        if scene.has_registered_rigid_candidates() {
            let mut contacted_candidates = Vec::new();
            for contact in &contacts {
                for voxel in [contact.i, contact.j] {
                    if scene.aggregate_for_voxel(voxel).is_some() {
                        contacted_candidates.push(voxel);
                    }
                }
            }
            if !contacted_candidates.is_empty() {
                for &voxel in &contacted_candidates {
                    scene.positions[voxel] = predicted[voxel];
                }
                if scene.wake_rigid_candidates_for_voxels(&contacted_candidates) > 0 {
                    scene.positions.clone_from_slice(&predicted);
                    collision::project_into(scene, targets, &mut contacts);
                }
            }
        }
        solver_scratch.pair_contacts.clear();
        for contact in &contacts {
            let axis = if contact.normal.x.abs() > 0.5 {
                0
            } else if contact.normal.y.abs() > 0.5 {
                1
            } else {
                2
            };
            let fixed = |i: usize| scene.voxels[i].kinematic || targets[i][axis].is_some();
            if !fixed(contact.i) && !fixed(contact.j) {
                solver_scratch.pair_contacts.push(crate::contact_constraint::PairContact {
                    i: contact.i, j: contact.j, axis,
                    sign: f64::from(contact.normal[axis]), active: true,
                });
                continue;
            }
            let (dynamic, support, sign) = if fixed(contact.i) && !fixed(contact.j) {
                (contact.j, contact.i, contact.normal[axis])
            } else if fixed(contact.j) && !fixed(contact.i) {
                (contact.i, contact.j, -contact.normal[axis])
            } else {
                continue;
            };
            let gap =
                (predicted[dynamic][axis] - scene.positions[support][axis]) * sign as f64 - DCELL;
            if gap > 1e-6 {
                continue;
            }
            let plane = scene.positions[support][axis] + sign as f64 * DCELL;
            if (substep_start_positions[dynamic][axis] - scene.positions[support][axis])
                * (sign as f64)
                < 0.0
            {
                continue;
            }
            let plane = match contact_targets[dynamic][axis] {
                Some(previous) if solver_scratch.contact_axis_normals[dynamic][axis] == sign => {
                    if sign > 0.0 {
                        plane.max(previous)
                    } else {
                        plane.min(previous)
                    }
                }
                _ => plane,
            };
            contact_targets[dynamic][axis] = Some(plane);
            solver_scratch.contact_axis_normals[dynamic][axis] = sign;
            if sign > 0.0 {
                solver_scratch.contact_lower[dynamic][axis] = Some(plane);
            } else {
                solver_scratch.contact_upper[dynamic][axis] = Some(plane);
            }
            scene.positions[dynamic][axis] = plane;
        }
        sync_visited(&mut solver_scratch.contact_feasible_positions, &scene.positions, &live);
        if solver_scratch.previous_contact_axis_normals.len() == voxel_count
            && visit(&live, voxel_count)
                .any(|i| solver_scratch.previous_contact_axis_normals[i] != [0.0; 3])
        {
            for i in visit(&live, voxel_count) {
                for axis in 0..3 {
                    let normal = solver_scratch.contact_axis_normals[i][axis];
                    if normal != 0.0
                        && solver_scratch.previous_contact_axis_normals[i][axis] != normal
                    {
                        contact_targets[i][axis] = targets[i][axis];
                        solver_scratch.contact_axis_normals[i][axis] = 0.0;
                    }
                }
            }
        }
        if let Some(start) = detail_start {
            pre_solve_detail_time[3] += start.elapsed();
        }
        stage_time[0] += phase_start.elapsed();
        phase_start = Instant::now();
        let mut contact_solver_energy_before = if solver_scratch.pair_contacts.iter().any(|p| p.active)
            || visit(&live, voxel_count).any(|i| solver_scratch.contact_axis_normals[i] != [0.0; 3]) {
            let kinetic = visit(&live, voxel_count)
                .map(|i| (i, &scene.voxels[i]))
                .filter(|(i, voxel)| !voxel.kinematic && scene.aggregate_of[*i].is_none())
                .map(|(i, voxel)| 0.5 * f64::from(voxel.mass)
                    * f64::from((substep_start_velocities[i] + tool_contact_velocity[i]).length_squared())
                    + cell_spin_energy_j(scene, i))
                .sum::<f64>();
            Some((kinetic, substep_elastic_energy_before))
        } else { None };
        // Continue the previous substep's friction state as the first guess.
        let mut support_friction_rows = support_rows(scene, &solver_scratch, &contacts);
        solve_loads.resize(voxel_count, Vec3::ZERO);
        support_friction_modes.retain(|voxel, _| {
            support_friction_rows.iter().any(|row| row.voxel == *voxel)
        });
        apply_support_friction(
            scene,
            &support_friction_rows,
            &support_friction_modes,
            &support_patch_normal,
            targets,
            &substep_start_positions,
            &mut contact_targets,
            &applied_loads,
            &mut solve_loads,
        );
        for iteration in 0..scene.config.corotation_iterations {
            let mut contact_pass = 0usize;
            loop {
                let (iterations, detail_time) = solve_coupled(
                    scene,
                    h,
                    &contact_targets,
                    &solve_loads,
                    &predicted,
                    &mut solver_scratch,
                    iteration > 0 || contact_pass > 0,
                    reusable_cuda_solution,
                    &mut cuda_enabled_for_step,
                );
                #[cfg(test)]
                inject_test_solve_failure(scene);
                solve_failed_this_step |= !scene.solver_residual.is_finite();
                reusable_cuda_solution = cuda_enabled_for_step;
                total_pcg_iterations += iterations;
                for index in 0..solve_detail_time.len() {
                    solve_detail_time[index] += detail_time[index];
                }
                // Keep the active-set search feasible: coupled elastic motion
                // can cross a previously released plane. Activate the first
                // blocking bound along the segment from the feasible iterate.
                let mut blocking = None;
                let mut alpha = 1.0_f64;
                for i in visit(&live, voxel_count) {
                    for axis in 0..3 {
                        if targets[i][axis].is_some()
                            || solver_scratch.contact_axis_normals[i][axis] != 0.0
                        {
                            continue;
                        }
                        for (plane, normal) in [
                            (solver_scratch.contact_lower[i][axis], 1.0),
                            (solver_scratch.contact_upper[i][axis], -1.0),
                        ] {
                            let Some(plane) = plane else {
                                continue;
                            };
                            let gap = (scene.positions[i][axis] - plane) * normal;
                            if gap >= -1e-8 {
                                continue;
                            }
                            let old_gap = ((solver_scratch.contact_feasible_positions[i][axis]
                                - plane)
                                * normal)
                                .max(0.0);
                            let fraction = (old_gap / (old_gap - gap)).clamp(0.0, 1.0);
                            if fraction <= alpha {
                                alpha = fraction;
                                blocking = Some((i, axis, plane, normal));
                            }
                        }
                    }
                }
                // Near-zero reactions within the solve tolerance can flip a row back and
                // forth. Keep the last feasible state instead of looping forever.
                if scene.cell_dofs && contact_pass > 16 {
                    break;
                }
                if let Some((i, axis, plane, normal)) = blocking {
                    sync_visited(&mut solved_positions, &scene.positions, &live);
                    // A sleeper sits exactly on its feasible position: `old + 0 * alpha`.
                    for i in visit(&live, voxel_count) {
                        let old = solver_scratch.contact_feasible_positions[i];
                        scene.positions[i] = old + (scene.positions[i] - old) * alpha;
                    }
                    contact_targets[i][axis] = Some(plane);
                    solver_scratch.contact_axis_normals[i][axis] = normal as f32;
                    scene.positions[i][axis] = plane;
                    // Activate simultaneous hits together (a flat contact face
                    // must not require one matrix factorization per surface REV).
                    for j in visit(&live, voxel_count) {
                        for a in 0..3 {
                            if targets[j][a].is_some()
                                || solver_scratch.contact_axis_normals[j][a] != 0.0
                            {
                                continue;
                            }
                            for (bound, sign) in [
                                (solver_scratch.contact_lower[j][a], 1.0),
                                (solver_scratch.contact_upper[j][a], -1.0),
                            ] {
                                let Some(bound) = bound else {
                                    continue;
                                };
                                let direction = (solved_positions[j][a]
                                    - solver_scratch.contact_feasible_positions[j][a])
                                    * sign;
                                if (scene.positions[j][a] - bound).abs() <= 1e-8
                                    && direction < -1e-12
                                {
                                    contact_targets[j][a] = Some(bound);
                                    solver_scratch.contact_axis_normals[j][a] = sign as f32;
                                    scene.positions[j][a] = bound;
                                }
                            }
                        }
                    }
                    sync_visited(&mut solver_scratch.contact_feasible_positions, &scene.positions, &live);
                    contact_pass += 1;
                    continue;
                }
                sync_visited(&mut solver_scratch.contact_feasible_positions, &scene.positions, &live);
                // Unilateral contact can push away from a fixed surface; it cannot
                // pull an interior voxel down onto that surface. During a large
                // sweep several layers can touch in the prediction. Release those
                // with a tensile reaction and re-solve the SAME inertial system.
                // Each pass removes at least one temporary axis constraint, so the
                // loop terminates without an arbitrary physical force clamp.
                compute_contact_reactions(scene, &mut solver_scratch, &predicted, &solve_loads, h);
                let mut released = false;
                for i in visit(&live, voxel_count) {
                    for axis in 0..3 {
                        let normal = solver_scratch.contact_axis_normals[i][axis];
                        if normal == 0.0 {
                            continue;
                        }
                        let reaction =
                            solver_scratch.contact_reactions[i][axis] * f64::from(normal);
                        let inertial_force = ((scene.positions[i][axis] - predicted[i][axis])
                            * (f64::from(scene.voxels[i].mass) / f64::from(h).powi(2)))
                        .abs();
                        // A row inside the solve's own force error cannot be told
                        // apart from one still in compression.
                        let tolerance = f64::from(scene.config.solver_relative_tolerance)
                            * inertial_force
                                .max(f64::from(applied_loads[i][axis]).abs())
                                .max(1e-6)
                            + solver_scratch.last_force_residual_n;
                        // After a few passes a row that keeps flipping is held: the state
                        // stays free of penetration, and the tensile part is solve noise.
                        if reaction < -tolerance && contact_pass < 4 {
                            contact_targets[i][axis] = targets[i][axis];
                            solver_scratch.contact_axis_normals[i][axis] = 0.0;
                            released = true;
                        }
                    }
                }
                if solver_scratch.pair_contacts.iter().any(|p| p.active) {
                    let projection = crate::contact_constraint::ContactProjection::build(
                        &solver_scratch.pair_contacts, &solver_scratch.free, &solver_scratch.active_index);
                    let reactions = projection.reactions(&solver_scratch.pair_contacts,
                        &solver_scratch.contact_reactions);
                    let scale = reactions.iter().map(|r| r.abs()).fold(1e-6_f64, f64::max);
                    for (pair, reaction) in solver_scratch.pair_contacts.iter_mut().zip(reactions) {
                        if pair.active && reaction < -f64::from(scene.config.solver_relative_tolerance) * scale {
                            pair.active = false;
                            released = true;
                        }
                    }
                }
                // Discover contacts from the coupled iterate without modifying
                // it. New blocking pairs become constraints on the SAME solve.
                // A certified coarse step has no coupled fine iterate. The
                // looser REV-bin query can otherwise refine a body after its
                // rigid drift, leaving its fine velocity buffers uninitialized.
                if !coarse_clear {
                    collision::detect_into(scene, targets, &mut contacts);
                }
                let mut blocked_pair = false;
                for c in &contacts {
                    let axis = if c.normal.x.abs() > 0.5 {0} else if c.normal.y.abs() > 0.5 {1} else {2};
                    if scene.voxels[c.i].kinematic || scene.voxels[c.j].kinematic
                        || targets[c.i][axis].is_some() || targets[c.j][axis].is_some() { continue; }
                    let sign = f64::from(c.normal[axis]);
                    if (scene.positions[c.j][axis] - scene.positions[c.i][axis]) * sign >= DCELL - 1e-8 { continue; }
                    if let Some(pair) = solver_scratch.pair_contacts.iter_mut()
                        .find(|p| p.i == c.i && p.j == c.j && p.axis == axis && p.sign == sign) {
                        if !pair.active { pair.active = true; blocked_pair = true; }
                    } else {
                        solver_scratch.pair_contacts.push(crate::contact_constraint::PairContact {
                            i:c.i, j:c.j, axis, sign, active:true,
                        });
                        blocked_pair = true;
                    }
                }
                if blocked_pair {
                    // A fast impact can cross the contact threshold during an
                    // initially unconstrained solve. Preserve the pre-solve
                    // mechanical energy so that this same-step contact solve
                    // can return its numerical dissipation as heat.
                    if contact_solver_energy_before.is_none() {
                        let kinetic_before = visit(&live, voxel_count)
                            .map(|i| (i, &substep_start_velocities[i]))
                            .filter(|(i, voxel_velocity)| {
                                !scene.voxels[*i].kinematic
                                    && scene.aggregate_of[*i].is_none()
                                    && voxel_velocity.is_finite()
                            })
                            .map(|(i, velocity)| {
                                0.5 * f64::from(scene.voxels[i].mass)
                                    * f64::from(velocity.length_squared())
                                    + cell_spin_energy_j(scene, i)
                            })
                            .sum::<f64>();
                        contact_solver_energy_before =
                            Some((kinetic_before, substep_elastic_energy_before));
                    }
                    collision::project_into(scene, targets, &mut contacts);
                    sync_visited(&mut solver_scratch.contact_feasible_positions, &scene.positions, &live);
                    contact_pass += 1;
                    continue;
                }
                if !released {
                    break;
                }
                contact_pass += 1;
            }
            if iteration + 1 == scene.config.corotation_iterations {
                break;
            }
            let rotation_start = profile_step.then(Instant::now);
            let orientation_change = if coarse_clear {
                0.0
            } else {
                update_frames(scene, &mut solver_scratch)
            };
            if let Some(start) = rotation_start {
                rotation_fit_time += start.elapsed();
            }
            if orientation_change <= scene.config.corotation_tolerance
                && scene.solver_residual <= scene.config.solver_relative_tolerance
            {
                break;
            }
        }
        // Settle friction on the contacts this solve kept. Each extra solve either
        // starts or ends a slide, so two are enough to reach a consistent state.
        support_friction_rows = support_rows(scene, &solver_scratch, &contacts);
        for _ in 0..2 {
            if support_friction_rows.is_empty() {
                support_friction_modes.clear();
                break;
            }
            compute_contact_reactions(scene, &mut solver_scratch, &predicted, &solve_loads, h);
            let changed = update_support_friction(
                scene,
                &support_friction_rows,
                &solver_scratch,
                &mut support_friction_modes,
                &mut support_patch_normal,
                &substep_start_positions,
                h,
            );
            if !changed {
                break;
            }
            apply_support_friction(
                scene,
                &support_friction_rows,
                &support_friction_modes,
                &support_patch_normal,
                targets,
                &substep_start_positions,
                &mut contact_targets,
                &applied_loads,
                &mut solve_loads,
            );
            let (iterations, detail_time) = solve_coupled(
                scene,
                h,
                &contact_targets,
                &solve_loads,
                &predicted,
                &mut solver_scratch,
                true,
                reusable_cuda_solution,
                &mut cuda_enabled_for_step,
            );
            solve_failed_this_step |= !scene.solver_residual.is_finite();
            total_pcg_iterations += iterations;
            for index in 0..solve_detail_time.len() {
                solve_detail_time[index] += detail_time[index];
            }
        }
        // Report what each fixed support carries: the reaction at a constrained row,
        // compressive part only. Sensors read it; nothing in the solve depends on it.
        scene.support_load_n.clear();
        scene.support_load_n.resize(voxel_count, 0.0);
        if !support_friction_rows.is_empty() {
            compute_contact_reactions(scene, &mut solver_scratch, &predicted, &solve_loads, h);
            for row in &support_friction_rows {
                let reaction = solver_scratch.contact_reactions[row.voxel][row.axis] * f64::from(row.normal);
                scene.support_load_n[row.voxel] = reaction.max(0.0) as f32;
            }
        }
        stage_time[1] += phase_start.elapsed();
        phase_start = Instant::now();
        // Preserve the accurately integrated drift velocity. Recover only the
        // velocity correction from the much smaller constraint displacement;
        // subtracting two large f32 positions every step otherwise accumulates
        // measurable kinetic error in simple free fall.
        sync_visited(&mut solved_positions, &scene.positions, &live);
        pre_contact_velocities.resize(voxel_count, Vec3::ZERO);
        for i in visit(&live, voxel_count) {
            pre_contact_velocities[i] = drift_velocity[i] + tool_contact_velocity[i];
        }
        if !coarse_clear { collision::detect_into(scene, targets, &mut contacts); }
        fill_visited(&mut contact_work_mode, voxel_count, 0, &live);
        for contact in &contacts {
            let mode = if contact.new_impact { 2 } else { 1 };
            contact_work_mode[contact.i] = contact_work_mode[contact.i].max(mode);
            contact_work_mode[contact.j] = contact_work_mode[contact.j].max(mode);
        }
        let contact_energy_start = profile_step.then(Instant::now);
        let implicit_contact_heat = contact_solver_energy_before.map_or(0.0, |(kinetic_before, elastic_before)| {
            let mut kinetic_after = 0.0;
            let mut applied_work = 0.0;
            for i in visit(&live, voxel_count) {
                if scene.voxels[i].kinematic || scene.aggregate_of[i].is_some() { continue; }
                let velocity = drift_velocity[i] + tool_contact_velocity[i]
                    + ((solved_positions[i] - predicted[i]) / f64::from(h)).as_vec3();
                kinetic_after += 0.5 * f64::from(scene.voxels[i].mass)
                    * f64::from(velocity.length_squared())
                    + cell_spin_energy_j(scene, i);
                let force = applied_loads[i] + gravity * scene.voxels[i].mass;
                let displacement = match contact_work_mode[i] {
                    0 | 2 => (substep_start_velocities[i] + velocity) * (0.5 * h),
                    _ => (solved_positions[i] - substep_start_positions[i]).as_vec3(),
                };
                applied_work += f64::from(force.dot(displacement));
            }
            // Same bond set as `capture_nodal_mechanical_energy`: a bond between
            // two fixed REVs cannot change within a substep.
            let elastic_after = scene.mobile_elastic_bond_energy_j();
            kinetic_before + elastic_before + applied_work
                - kinetic_after - elastic_after
        });
        if let Some(start) = contact_energy_start { contact_energy_time += start.elapsed(); }
        if coarse_clear {
            contacts.clear();
        } else {
            collision::detect_into(scene, targets, &mut contacts);
        }
        if scene.has_registered_rigid_candidates() {
            let mut contacted_candidates = Vec::new();
            for contact in &contacts {
                for voxel in [contact.i, contact.j] {
                    if scene.aggregate_for_voxel(voxel).is_some() {
                        contacted_candidates.push(voxel);
                    }
                }
            }
            if !contacted_candidates.is_empty() {
                for &voxel in &contacted_candidates {
                    scene.positions[voxel] = solved_positions[voxel];
                }
                if scene.wake_rigid_candidates_for_voxels(&contacted_candidates) > 0 {
                    scene.positions.clone_from_slice(&solved_positions);
                    collision::project_into(scene, targets, &mut contacts);
                }
            }
        }
        fill_visited(&mut contact_work_mode, voxel_count, 0, &live);
        for contact in &contacts {
            let mode = if contact.new_impact { 2 } else { 1 };
            contact_work_mode[contact.i] = contact_work_mode[contact.i].max(mode);
            contact_work_mode[contact.j] = contact_work_mode[contact.j].max(mode);
        }
        impact_energies_j.clear();
        contacts_peak = contacts_peak.max(contacts.len());
        new_impact_observations += contacts.iter().filter(|contact| contact.new_impact).count();
        let free_energy_start = profile_step.then(Instant::now);
        let (free_implicit_total, free_implicit_deficit) = if coarse_clear { (0.0, 0.0) } else {
            free_implicit_dissipation(scene, &solver_scratch, &substep_mechanical_energy,
                &substep_start_positions, &substep_start_velocities, &solved_positions,
                &predicted, &drift_velocity, &tool_contact_velocity, &applied_loads,
                targets, &contacts, gravity, h, &mut energy_edge_values,
                energy_edges_parallel_enabled_for(scene.voxels.len()),
                &mut free_implicit_budgets, &mut free_implicit_components, &mut free_implicit_heat)
        };
        if let Some(start) = free_energy_start { free_energy_time += start.elapsed(); }
        // Remove the entire independent free-body deficit from the contact
        // pool, including any portion that the local quadratic cannot explain.
        // Such a residual stays an accuracy error; it cannot fund another hit.
        let implicit_contact_heat = (implicit_contact_heat - free_implicit_deficit).max(0.0);
        impact_energies_j.reserve(contacts.len());
        for contact in &contacts {
            let impact_energy = if !contact.new_impact {
                0.0
            } else {
                let axis = if contact.normal.x.abs() > 0.5 {
                    0
                } else if contact.normal.y.abs() > 0.5 {
                    1
                } else {
                    2
                };
                let inverse_mass = |voxel: usize| {
                    if scene.voxels[voxel].kinematic || targets[voxel][axis].is_some() {
                        0.0
                    } else {
                        1.0 / scene.voxels[voxel].mass
                    }
                };
                let inverse_effective_mass = inverse_mass(contact.i) + inverse_mass(contact.j);
                let closing_speed = (scene.voxels[contact.j].velocity
                    - scene.voxels[contact.i].velocity)
                    .dot(contact.normal);
                if inverse_effective_mass <= 0.0 || closing_speed >= 0.0 {
                    0.0
                } else {
                    0.5 * closing_speed as f64 * closing_speed as f64
                        / inverse_effective_mass as f64
                }
            };
            impact_energies_j.push(impact_energy);
        }
        stage_time[2] += phase_start.elapsed();
        phase_start = Instant::now();
        // Actual converged strain defines elastic energy and constitutive stress.
        // Reading an intermediate lambda would depend on sweep order and convergence.
        // Every bond in a physical substep reads the same pre-fracture thermal
        // state. Heat is reduced afterward in stable constraint order, avoiding
        // an axis/index-order bias where one bond's heat softens the next bond
        // before the same substep has finished.
        // `solve_coupled` just evaluated the equation of state for this same
        // pre-fracture thermal state. Reuse its derived strength cache instead
        // of evaluating every REV's phase state a second time. The opt-out is
        // retained for reproducible A/B profiling.
        let phase_strengths = if reuse_assembly_phase_strengths_enabled() {
            debug_assert_eq!(solver_scratch.assembly_phase_strengths.len(), voxel_count);
            &solver_scratch.assembly_phase_strengths
        } else {
            solver_scratch
                .fracture_phase_strengths
                .resize(voxel_count, (1.0, 1.0));
            for i in visit(&live, voxel_count) {
                solver_scratch.fracture_phase_strengths[i] = phase_strength(scene, i);
            }
            &solver_scratch.fracture_phase_strengths
        };
        let constraints = &solver_scratch.constraints;
        let outcomes = &mut solver_scratch.fracture_outcomes;
        outcomes.resize(constraints.len(), None);
        let update_constraint = |constraint: &Constraint| -> Option<FractureOutcome> {
            let i = constraint.i;
            let j = constraint.j;
            let axis = constraint.axis;
            let mut bond = scene.bonds[axis][i];
            if bond.neighbor == EMPTY || scene.voxels[i].kinematic {
                return None;
            }
            let frame = constraint.frame;
            let k = constraint.material_k;
            let limits = bond_limits_with_phase_strengths(
                scene,
                i,
                j,
                frame.x_axis,
                [phase_strengths[i], phase_strengths[j]],
            );
            let u = bond_elastic_displacement(scene, i, j, axis, frame);
            let last_force = frame
                * (k * (u - Vec3::from_array(bond.plastic_strain) * CELL))
                * (1.0 - bond.damage);
            let dissipation =
                crate::fracture::update(&mut bond, u, k, limits, h, &scene.config.fracture);
            Some(FractureOutcome {
                bond,
                last_force,
                dissipation,
            })
        };
        if fracture_parallel_enabled_for(voxel_count) {
            outcomes
                .par_iter_mut()
                .zip(constraints.par_iter())
                .for_each(|(outcome, constraint)| *outcome = update_constraint(constraint));
        } else {
            outcomes
                .iter_mut()
                .zip(constraints)
                .for_each(|(outcome, constraint)| *outcome = update_constraint(constraint));
        }
        let track_aggregate_fractures = scene.has_registered_rigid_candidates();
        let mut fractured_aggregate_voxels = Vec::new();
        for (constraint, outcome) in constraints.iter().zip(outcomes.iter_mut()) {
            let Some(outcome) = outcome.take() else {
                continue;
            };
            let i = constraint.i;
            let j = constraint.j;
            let axis = constraint.axis;
            let previous_bond = scene.bonds[axis][i];
            let actor_guarded = scene.skeletal_actuators.iter_mut().any(|actuator| {
                if !actuator.matches_scene_topology(scene.voxels.len())
                    || !actuator.protects_bond(i, j)
                {
                    return false;
                }
                let would_damage = outcome.bond.neighbor == EMPTY
                    || outcome.bond.damage > previous_bond.damage
                    || outcome.bond.plastic_strain != previous_bond.plastic_strain;
                if would_damage {
                    actuator.stall();
                }
                would_damage
            });
            if actor_guarded {
                // Hold this internal REV bond at its last safe state and stop
                // the actor's pose clock when its material law predicts damage.
                continue;
            }
            scene.bonds[axis][i] = outcome.bond;
            scene.workspace[axis][i].last_force = outcome.last_force;
            let dissipation = outcome.dissipation;
            scene.energy.record(dissipation);
            let heat = (dissipation.plastic_heat + dissipation.damage_heat) * 0.5;
            scene.voxels[i].add_internal_energy_j(heat);
            scene.voxels[j].add_internal_energy_j(heat);
            if dissipation.released > 0.0 {
                scene.releases.push(crate::fracture::Release {
                    i,
                    j,
                    energy: dissipation.released,
                    due: scene.substep_index + scene.config.fracture.propagation_delay,
                });
            }
            if outcome.bond.neighbor == EMPTY {
                scene.broken += 1;
                topology_changed = true;
                reusable_cuda_solution = false;
                if track_aggregate_fractures
                    && (scene.aggregate_of[i].is_some() || scene.aggregate_of[j].is_some())
                {
                    fractured_aggregate_voxels.push(i);
                    fractured_aggregate_voxels.push(j);
                }
            }
        }
        if !fractured_aggregate_voxels.is_empty() {
            scene.wake_rigid_candidates_for_voxels(&fractured_aggregate_voxels);
        }
        stage_time[3] += phase_start.elapsed();
        phase_start = Instant::now();
        for i in visit(&live, voxel_count) {
            if !scene.voxels[i].kinematic && scene.aggregate_of[i].is_none() {
                let constraint_velocity =
                    ((solved_positions[i] - predicted[i]) / h as f64).as_vec3();
                for axis in 0..3 {
                    if targets.get(i).and_then(|target| target[axis]).is_some() {
                        scene.voxels[i].velocity[axis] =
                            ((scene.positions[i][axis] - scene.previous[i][axis]) / h as f64)
                                as f32;
                    } else {
                        scene.voxels[i].velocity[axis] = drift_velocity[i][axis]
                            + constraint_velocity[axis]
                            + tool_contact_velocity[i][axis];
                    }
                }
            }
            load_integrated_velocities[i] = scene.voxels[i].velocity;
        }
        normal_impulses_ns.resize(contacts.len(), 0.0);
        normal_impulses_ns.fill(0.0);
        if solver_scratch.pair_contacts.iter().any(|pair| pair.active) {
            let projection = crate::contact_constraint::ContactProjection::build(
                &solver_scratch.pair_contacts,
                &solver_scratch.free,
                &solver_scratch.active_index,
            );
            let reactions = projection.reactions(
                &solver_scratch.pair_contacts,
                &solver_scratch.contact_reactions,
            );
            for (pair, reaction) in solver_scratch.pair_contacts.iter().zip(reactions) {
                if !pair.active || reaction <= 0.0 {
                    continue;
                }
                if let Some((index, _)) = contacts.iter().enumerate().find(|(_, contact)| {
                    contact.i == pair.i
                        && contact.j == pair.j
                        && contact.normal[pair.axis] * pair.sign as f32 > 0.5
                }) {
                    normal_impulses_ns[index] =
                        normal_impulses_ns[index].max(reaction as f32 * h);
                }
            }
        }
        let (implicit_contact_heat, restitution_heat) = resolve_coupled_restitution(
            scene, &solver_scratch, &contacts, &pre_contact_velocities, h,
            &mut normal_impulses_ns, implicit_contact_heat,
        );
        let heat = implicit_contact_heat + restitution_heat;
        if free_implicit_total > 0.0 {
            for i in visit(&live, voxel_count) {
                let heat = free_implicit_heat[i];
                if heat > 0.0 { scene.voxels[i].add_internal_energy_j(heat); }
            }
            scene.energy.implicit_solver_heat += free_implicit_total;
        }
        if heat > 0.0 {
            let mut receivers = Vec::new();
            for pair in solver_scratch.pair_contacts.iter().filter(|pair| pair.active) {
                let closing = f64::from(pre_contact_velocities[pair.j][pair.axis]
                    - pre_contact_velocities[pair.i][pair.axis]) * pair.sign;
                if closing < -1.0e-10 { receivers.extend([pair.i, pair.j]); }
            }
            for contact in contacts.iter().filter(|contact| contact.new_impact) {
                let closing = (pre_contact_velocities[contact.j]
                    - pre_contact_velocities[contact.i]).dot(contact.normal);
                if closing < -1.0e-10 { receivers.extend([contact.i, contact.j]); }
            }
            // A slide on a fixed support dissipates into the sliding contact.
            receivers.extend(support_friction_modes.iter().filter_map(|(voxel, mode)| {
                matches!(mode, SupportFriction::Slip(_)).then_some(*voxel)
            }));
            receivers.retain(|&i| !scene.voxels[i].kinematic);
            receivers.sort_unstable();
            receivers.dedup();
            if !receivers.is_empty() {
                let heat_per_voxel = heat / receivers.len() as f64;
                for i in receivers { scene.voxels[i].add_internal_energy_j(heat_per_voxel); }
                // A sliding support dissipates through kinetic friction in the same
                // implicit budget; a simultaneous impact in this substep shares the label.
                if support_friction_modes.values().any(|mode| matches!(mode, SupportFriction::Slip(_))) {
                    scene.energy.contact_friction_heat += implicit_contact_heat;
                } else {
                    scene.energy.implicit_solver_heat += implicit_contact_heat;
                }
                scene.energy.contact_restitution_heat += restitution_heat;
            }
        }
        // Contacts settled by the implicit friction above skip the velocity pass.
        implicit_friction_contacts.clear();
        implicit_friction_contacts.extend(contacts.iter().map(|contact| {
            let voxel = if scene.voxels[contact.i].kinematic { contact.j } else { contact.i };
            support_friction_modes.contains_key(&voxel)
                && (scene.voxels[contact.i].kinematic || scene.voxels[contact.j].kinematic)
        }));
        collision::resolve_velocities_with_precontact_into(
            scene,
            &contacts,
            targets,
            &pre_contact_velocities,
            &mut normal_impulses_ns,
            &mut contact_physical_velocities,
            &implicit_friction_contacts,
            &mut friction_used_ns,
        );
        collision::resolve_liquid_viscosity(scene, &contacts, targets, h);
        let contact_finish_start = profile_step.then(Instant::now);
        let formed_bonds =
            scene.form_thermal_bonds(&contacts, &impact_energies_j, &normal_impulses_ns, h);
        // A bond formed during this contact solve takes precedence over fusion:
        // the fusion primitive only accepts unbonded pairs.
        scene.form_thermal_fusions(&contacts, &normal_impulses_ns, h, targets);
        if formed_bonds > 0 {
            topology_changed = true;
            reusable_cuda_solution = false;
            // Bond assembly changed while the local per-step workspace was out
            // of the scene; rebuild it before the next substep uses the graph.
            solver_scratch = Default::default();
        }
        if let Some(start) = contact_finish_start {
            contact_finish_time += start.elapsed();
        }
        if let Some(t) = tool {
            // Structural correction can move material back into the blade.
            // Run a zero-time geometric/impulse pass before leaving the substep.
            let tool_start = profile_step.then(Instant::now);
            t.advance_into(scene, 0.0, targets, &mut tool_contact_velocity);
            if let Some(start) = tool_start {
                tool_recheck_time += start.elapsed();
            }
        }
        let release_start = profile_step.then(Instant::now);
        apply_releases(scene, targets);
        if let Some(start) = release_start {
            release_time += start.elapsed();
        }
        let damping_start = profile_step.then(Instant::now);
        damping_cache_hits +=
            apply_solid_viscoelastic_damping(scene, targets, h, &solver_scratch.constraints);
        if let Some(start) = damping_start {
            damping_time += start.elapsed();
        }
        let thermal_start = profile_step.then(Instant::now);
        if let Err(reason) = scene.conduct_heat(h) {
            eprintln!("thermal transport skipped: {reason}");
        }
        if let Some(start) = thermal_start {
            thermal_time += start.elapsed();
        }
        for i in visit(&live, voxel_count) {
            if !scene.voxels[i].kinematic {
                let applied_force = applied_loads[i] + gravity * scene.voxels[i].mass;
                let work = match contact_work_mode[i] {
                    // Use the velocity quadrature of the force/integration
                    // phase before an impact impulse is applied.
                    0 | 2 => applied_force.dot(
                        (substep_start_velocities[i] + load_integrated_velocities[i]) * (0.5 * h),
                    ),
                    // A persistent contact constrains the actual projected
                    // displacement, so don't count its trial velocity as work.
                    _ => applied_force
                        .dot((scene.positions[i] - substep_start_positions[i]).as_vec3()),
                };
                scene.energy.external_work += work as f64;
            }
        }
        scene.substep_index += 1;
        scene.mark_render_changed();
        stage_time[4] += phase_start.elapsed();
    }
    let loop_left = Instant::now();
    if let Some((i, j)) = scene.take_pending_fusion() {
        match scene.fuse_adjacent_voxels(i, j) {
            Ok(_) => {
                topology_changed = true;
                solver_scratch = Default::default();
            }
            Err(reason) => {
                eprintln!("automatic thermal fusion cancelled for REV {i}/{j}: {reason}")
            }
        }
    }
    if topology_changed {
        rebuild_clusters(scene);
    }
    // World exits are a topology event at the fixed-step boundary. This keeps
    // every solver/contact array in one consistent index space for the step.
    let removed_at_boundary = scene.enforce_world_bounds();
    if solve_failed_this_step {
        scene.solver_residual = f32::INFINITY;
    }
    // Reactions change composition at constant internal energy, so they run once per
    // fixed step after mechanics and heat, never inside the implicit solve.
    if scene.chemistry_enabled {
        let _ = scene.advance_chemistry(scene.config.fixed_dt as f64);
    }
    let completed_fixed_steps = scene.substep_index / (scene.config.substeps.max(1) as u64);
    if completed_fixed_steps > 0
        && completed_fixed_steps % AUTO_AGGREGATE_SCAN_INTERVAL_STEPS == 0
        && removed_at_boundary == 0
        && std::env::var_os("VOXEL_DISABLE_AUTO_AGGREGATION").is_none()
    {
        let scan_start = Instant::now();
        auto_aggregated_revs =
            auto_register_quiescent_components(scene, targets, &applied_loads, &contacts);
        aggregate_scan_time += scan_start.elapsed();
    }
    solver_scratch.merged_targets = merged_targets;
    solver_scratch.contact_targets = contact_targets;
    solver_scratch.atmosphere_exposed_faces = atmosphere_exposed_faces;
    solver_scratch.ambient_loads = ambient_loads;
    solver_scratch.applied_loads = applied_loads;
    solver_scratch.substep_start_positions = substep_start_positions;
    solver_scratch.substep_start_velocities = substep_start_velocities;
    solver_scratch.implicit_friction_contacts = implicit_friction_contacts;
    solver_scratch.support_friction_modes = support_friction_modes;
    solver_scratch.support_patch_normal = support_patch_normal;
    solver_scratch.solve_loads = solve_loads;
    solver_scratch.friction_used_ns = friction_used_ns;
    solver_scratch.substep_mechanical_energy = substep_mechanical_energy;
    solver_scratch.energy_edge_values = energy_edge_values;
    solver_scratch.free_implicit_budgets = free_implicit_budgets;
    solver_scratch.free_implicit_components = free_implicit_components;
    solver_scratch.free_implicit_heat = free_implicit_heat;
    solver_scratch.load_integrated_velocities = load_integrated_velocities;
    solver_scratch.contact_work_mode = contact_work_mode;
    solver_scratch.tool_contact_velocity = tool_contact_velocity;
    solver_scratch.drift_velocity = drift_velocity;
    solver_scratch.predicted = predicted;
    solver_scratch.solved_positions = solved_positions;
    solver_scratch.pre_contact_velocities = pre_contact_velocities;
    solver_scratch.contacts = contacts;
    solver_scratch.impact_energies_j = impact_energies_j;
    solver_scratch.melted = melted;
    solver_scratch.normal_impulses_ns = normal_impulses_ns;
    solver_scratch.contact_physical_velocities = contact_physical_velocities;
    if live.is_none() {
        // Only the quiescence test reads these, and a sparse step never runs it
        // (adaptive substeps are off); a stale record fails its checks and is not trusted.
        solver_scratch.adaptive_gravity = Some(gravity);
        solver_scratch
            .adaptive_loads
            .clone_from(&solver_scratch.applied_loads);
        solver_scratch
            .adaptive_positions
            .clone_from(&scene.positions);
        solver_scratch.adaptive_config = bincode::serialize(&scene.config).unwrap_or_default();
        solver_scratch.adaptive_revision = Some(scene.render_revision);
        solver_scratch.adaptive_energies.clear();
        solver_scratch
            .adaptive_energies
            .extend(scene.voxels.iter().map(|v| v.thermal_energy_f32()));
    }
    if profile_step {
        let contact_profile = CONTACT_PCG_PROFILE.with(std::cell::Cell::get);
        scene.last_step_diagnostics = Some(crate::scene::StepDiagnostics {
            scope: "last_fixed_step",
            fixed_dt_seconds: f64::from(scene.config.fixed_dt),
            mechanical_substeps: effective_substeps,
            active_fine_revs_last_solve: solver_scratch.active.len(),
            pcg_iterations_total: total_pcg_iterations,
            contact_pcg_calls: contact_profile.calls,
            contact_pcg_iterations_including_restitution: contact_profile.iterations,
            contact_gpu_operator_calls: contact_profile.gpu_operator_calls,
            contact_gpu_preconditioner_calls: contact_profile.gpu_preconditioner_calls,
            contact_gpu_operator_fallbacks: contact_profile.gpu_operator_fallbacks,
            contact_pcg_unconverged_calls: contact_profile.unconverged_calls,
            contact_pcg_max_true_residual: contact_profile.max_true_residual,
            relative_solver_residual: scene.solver_residual,
            contacts_peak,
            new_impact_observations,
            timings_ms: [
                ("prediction_tool", stage_time[0]),
                ("mechanics_total", stage_time[1]),
                ("matrix_assembly", solve_detail_time[0]),
                ("factorization", solve_detail_time[1]),
                ("pcg", solve_detail_time[2]),
                ("rotation_fit", rotation_fit_time),
                ("collision", stage_time[2]),
                ("fracture", stage_time[3]),
                ("finish_total", stage_time[4]),
                ("thermal", thermal_time),
                ("damping", damping_time),
                ("aggregation_scan", aggregate_scan_time),
                ("mechanical_energy_capture", energy_capture_time),
                ("contact_energy_accounting", contact_energy_time),
                ("free_implicit_energy_accounting", free_energy_time),
                ("prediction_tool_without_energy_capture", stage_time[0].saturating_sub(energy_capture_time)),
                ("mechanics_without_contact_energy_accounting", stage_time[1].saturating_sub(contact_energy_time)),
                ("collision_without_free_implicit_energy_accounting", stage_time[2].saturating_sub(free_energy_time)),
                ("contact_pcg_operator", contact_profile.operator),
                ("contact_pcg_projection", contact_profile.projection),
                ("contact_pcg_preconditioner", contact_profile.preconditioner),
                ("contact_pcg_reductions", contact_profile.reductions),
                ("contact_pcg_updates", contact_profile.updates),
                ("contact_pcg_total", contact_profile.total),
                ("contact_gpu_setup", contact_profile.gpu_setup),
            ].into_iter().map(|(name, time)| (name, time.as_secs_f64()*1000.0)).collect(),
        });
    }
    scene.solver_scratch = solver_scratch;
    if log_profile_step {
        eprintln!(
            "profile outer ms: sleep={:.3} setup={:.3} tail={:.3}",
            (sleep_done - step_entered).as_secs_f64() * 1000.0,
            (loop_entered - sleep_done).as_secs_f64() * 1000.0,
            loop_left.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!("profile energy ms: capture={:.3} contact={:.3} free_implicit={:.3}",
            energy_capture_time.as_secs_f64()*1000.0,
            contact_energy_time.as_secs_f64()*1000.0,
            free_energy_time.as_secs_f64()*1000.0);
        let ms = stage_time.map(|duration| duration.as_secs_f64() * 1000.0);
        eprintln!(
            "profile step ms: predict/tool={:.2} [snapshot={:.2} drift={:.2} tool={:.2} projection={:.2}] solve={:.2} [assemble={:.2} factor={:.2} pcg={:.2}] rotation_fit={:.2} collision={:.2} fracture={:.2} finish={:.2} post_contact={:.2} tool_recheck={:.2} releases={:.2} damping={:.2} thermal={:.2} aggregation_scan={:.2} auto_aggregated_revs={auto_aggregated_revs} damping_cache_hits={damping_cache_hits} pcg_iters={total_pcg_iterations} residual={:.3e}",
            ms[0],
            pre_solve_detail_time[0].as_secs_f64() * 1000.0,
            pre_solve_detail_time[1].as_secs_f64() * 1000.0,
            pre_solve_detail_time[2].as_secs_f64() * 1000.0,
            pre_solve_detail_time[3].as_secs_f64() * 1000.0,
            ms[1],
            solve_detail_time[0].as_secs_f64() * 1000.0,
            solve_detail_time[1].as_secs_f64() * 1000.0,
            solve_detail_time[2].as_secs_f64() * 1000.0,
            rotation_fit_time.as_secs_f64() * 1000.0,
            ms[2],
            ms[3],
            ms[4],
            contact_finish_time.as_secs_f64() * 1000.0,
            tool_recheck_time.as_secs_f64() * 1000.0,
            release_time.as_secs_f64() * 1000.0,
            damping_time.as_secs_f64() * 1000.0,
            thermal_time.as_secs_f64() * 1000.0,
            aggregate_scan_time.as_secs_f64() * 1000.0,
            scene.solver_residual
        );
    }
}

/// Convert stored bond energy to heat and remove bonds that cannot exist in a
/// fully liquid REV. Called at the fixed-step boundary so all solver workspaces
/// see one consistent topology for the whole step.
fn break_fully_melted_bonds(scene: &mut Scene, melted: &mut Vec<bool>) -> bool {
    let live = scene.step_live.clone();
    let voxel_count = scene.voxels.len();
    let is_melted = |i: usize| {
        let (voxel, material) = (&scene.voxels[i], &scene.properties[i]);
        material.phase_state(voxel.mass, voxel.thermal_energy_f32()).solid_fraction <= 0.0
    };
    if live.is_some() {
        // A sleeper is never fully molten unless its cluster is on the lists (it would
        // not have been put to sleep) and its energy cannot change without waking it,
        // so only the REVs on the thermal list can melt.
        melted.clear();
        melted.resize(voxel_count, false);
        for i in visit_thermal(&live, voxel_count) {
            melted[i] = is_melted(i);
        }
    } else {
        melted.clear();
        melted.extend((0..voxel_count).map(is_melted));
    }
    if !visit_thermal(&live, voxel_count).any(|i| melted[i]) {
        return false;
    }

    let mut topology_changed = false;
    for axis in 0..3 {
        for i in visit_thermal(&live, voxel_count) {
            let bond = scene.bonds[axis][i];
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if !melted[i] && !melted[j] {
                continue;
            }

            let frame = basis(axis, scene.rotations[i]);
            let displacement = bond_elastic_displacement(scene, i, j, axis, frame)
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
            let heat = (stored as f64 + pending) * 0.5;
            scene.voxels[i].add_internal_energy_j(heat);
            scene.voxels[j].add_internal_energy_j(heat);

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
    scene.mark_render_changed();
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
    // Bonds may have changed: the sleepers that touch awake REVs may be different ones.
    scene.refresh_live();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonfinite_residual_norm_is_never_reported_as_zero() {
        assert_eq!(checked_relative_residual(0.0, 0.0), 0.0);
        assert!(checked_relative_residual(f64::NAN, 0.0).is_infinite());
        assert!(checked_relative_residual(f64::INFINITY, 1.0).is_infinite());
        assert!(checked_relative_residual(1.0, f64::NAN).is_infinite());
    }

    fn free_falling_oak_block() -> Scene {
        use crate::material::Material;
        use crate::scene::{SceneConfig, VoxelSpawn};

        let oak = Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(64);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    spawns.push(VoxelSpawn::at_cell([x, y, z + 8], oak).at_temperature(293.15));
                }
            }
        }
        let mut config = SceneConfig::default();
        config.fixed_dt = 0.01;
        config.substeps = 1;
        config.gravity = [0.0, 0.0, -9.81];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        let mut tool = None;
        for _ in 0..63 {
            step(&mut scene, &mut tool);
        }
        scene
    }

    fn cached_test_constraint(
        scene: &Scene,
        i: usize,
        axis: usize,
        j: usize,
        previous: Option<Constraint>,
    ) -> Constraint {
        let phases = [i, j].map(|n| {
            scene.properties[n].phase_state(scene.voxels[n].mass, scene.voxels[n].thermal_energy_f32())
        });
        make_constraint_at_phases(scene, i, axis, j, phases, [0; 2], previous)
    }

    fn rest_cache_test_scene() -> Scene {
        use crate::material::Material;
        use crate::scene::{SceneConfig, VoxelSpawn};

        let oak = Material::component_id("oak").unwrap();
        Scene::from_voxels(
            SceneConfig::default(),
            vec![
                VoxelSpawn::at_cell([0, 0, 8], oak).at_temperature(293.15),
                VoxelSpawn::at_cell([1, 0, 8], oak).at_temperature(293.15),
                VoxelSpawn::at_cell([0, 1, 8], oak).at_temperature(293.15),
            ],
        )
        .unwrap()
    }

    fn rest_cache_stiffness_inputs(scene: &Scene) -> [StiffnessInput; 2] {
        [0, 1].map(|n| {
            let material = &scene.properties[n];
            let micro = scene.voxels[n].microstructure;
            let phase = material.phase_state(
                scene.voxels[n].mass,
                scene.voxels[n].thermal_energy_f32(),
            );
            StiffnessInput {
                rotation: scene.rotations[n],
                world_fiber: scene.rotations[n] * micro.fibers(),
                anisotropy: micro.anisotropy,
                moduli: Vec3::new(material.young, material.young_perp, material.shear),
                solid: phase.solid_fraction.clamp(1e-4, 1.0),
            }
        })
    }

    fn assert_rest_cache_rebuilds_after(
        mut scene: Scene,
        mutate: impl FnOnce(&mut Scene),
    ) {
        let previous_inputs = rest_cache_stiffness_inputs(&scene);
        let previous = cached_test_constraint(&scene, 0, 0, 1, None);
        let mut stale = previous;
        let sentinel = DVec3::splat(42.0);
        stale.rest = sentinel;
        mutate(&mut scene);

        let current_inputs = rest_cache_stiffness_inputs(&scene);
        let revisions = std::array::from_fn(|n| {
            if previous_inputs[n] != current_inputs[n] { 1 } else { 0 }
        });
        let phases = [0, 1].map(|n| {
            scene.properties[n].phase_state(
                scene.voxels[n].mass,
                scene.voxels[n].thermal_energy_f32(),
            )
        });
        let cached = make_constraint_at_phases(
            &scene,
            0,
            0,
            1,
            phases,
            revisions,
            Some(stale),
        );
        let rebuilt =
            make_constraint_at_phases(&scene, 0, 0, 1, phases, revisions, None);
        assert_ne!(cached.rest, sentinel);
        assert_eq!(cached.rest, rebuilt.rest);
        assert_eq!(cached.k, rebuilt.k);
    }

    #[test]
    fn bond_rest_cache_reuses_only_exact_inputs_and_damage_still_updates_stiffness() {
        let scene = rest_cache_test_scene();
        let initial = cached_test_constraint(&scene, 0, 0, 1, None);

        let sentinel = DVec3::splat(42.0);
        let sentinel_k = Mat3::from_diagonal(Vec3::splat(42.0));
        let mut reusable = initial;
        reusable.rest = sentinel;
        reusable.k = sentinel_k;
        let reused = cached_test_constraint(&scene, 0, 0, 1, Some(reusable));
        assert_eq!(reused.rest, sentinel);
        assert_eq!(reused.k, sentinel_k);

        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.voxels[0].add_internal_energy_j(1.0);
        });
        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.reference_temperatures_kelvin[0] += 1.0;
        });
        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.properties[0].thermal_expansion_parallel += 1.0e-6;
        });
        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.voxels[0].microstructure.orientation = Quat::from_rotation_y(0.25);
        });
        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.bonds[0][0].formation_offset[0] = 1.0e-5;
        });
        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.bonds[0][0].plastic_strain[0] = 0.01;
        });
        assert_rest_cache_rebuilds_after(rest_cache_test_scene(), |scene| {
            scene.rotations[0] = Quat::from_rotation_y(0.1);
        });

        let mut scene = rest_cache_test_scene();
        let before_damage = cached_test_constraint(&scene, 0, 0, 1, None);
        scene.bonds[0][0].damage = 0.4;
        let cached_damage = cached_test_constraint(&scene, 0, 0, 1, Some(before_damage));
        let rebuilt_damage = cached_test_constraint(&scene, 0, 0, 1, None);
        assert_eq!(cached_damage.rest, before_damage.rest);
        assert_eq!(cached_damage.rest, rebuilt_damage.rest);
        assert_eq!(cached_damage.k, rebuilt_damage.k);
        assert_ne!(cached_damage.k, before_damage.k);

        let topology_scene = rest_cache_test_scene();
        let topology_cache = cached_test_constraint(&topology_scene, 0, 0, 1, None);
        let mismatched_topology =
            cached_test_constraint(&topology_scene, 0, 0, 2, Some(topology_cache));
        assert_eq!(
            mismatched_topology.rest,
            cached_test_constraint(&topology_scene, 0, 0, 2, None).rest
        );
    }

    #[test]
    fn rotation_fit_skips_fully_coarse_scene_without_touching_rev_rotations() {
        let mut scene = free_falling_oak_block();
        let targets = vec![[None; 3]; scene.voxels.len()];
        let loads = vec![Vec3::ZERO; scene.voxels.len()];
        assert_eq!(
            auto_register_quiescent_components(&mut scene, &targets, &loads, &[]),
            scene.voxels.len()
        );
        let rotations = scene.rotations.clone();
        let mut scratch = SolverScratch::default();

        assert_eq!(fit_rotations(&mut scene, &mut scratch), 0.0);
        assert_eq!(scene.rotations, rotations);
        assert!(scratch.rotation_gradients.is_empty());
        assert!(scratch.rotation_counts.is_empty());
    }

    #[test]
    fn coarse_solve_skips_fixed_floor_but_keeps_prescribed_deformable_bonds() {
        use crate::material::Material;
        use crate::scene::{SceneConfig, VoxelSpawn};

        for kinematic_floor in [true, false] {
            let oak = Material::component_id("oak").unwrap();
            let mut spawns = Vec::new();
            for x in 0..2 {
                let voxel = VoxelSpawn::at_cell([x, 0, 0], oak);
                spawns.push(if kinematic_floor { voxel.fixed() } else { voxel });
            }
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        spawns.push(VoxelSpawn::at_cell([x + 10, y + 10, z + 10], oak));
                    }
                }
            }
            let mut config = SceneConfig::default();
            config.gravity = [0.0; 3];
            config.atmosphere.pressure_pa = 0.0;
            let mut scene = Scene::from_voxels(config, spawns).unwrap();
            let members: Vec<_> = (2..scene.voxels.len()).collect();
            scene.register_rigid_aggregate_candidate(&members).unwrap();
            let positions = scene.positions.clone();
            let mut targets = vec![[None; 3]; scene.voxels.len()];
            if !kinematic_floor {
                for i in 0..2 {
                    targets[i] = positions[i].to_array().map(Some);
                }
            }
            let loads = vec![Vec3::ZERO; scene.voxels.len()];
            let mut scratch = SolverScratch::default();
            let mut cuda_enabled = false;
            solve_coupled(&mut scene, 0.01, &targets, &loads, &positions,
                &mut scratch, false, false, &mut cuda_enabled);
            assert_eq!(scene.positions, positions);
            assert_eq!(scene.broken, 0);
            if kinematic_floor {
                assert!(scratch.constraints.is_empty(),
                    "fixed-only floor bonds must not trigger fine assembly");
                assert!(scratch.stiffness_inputs.is_empty());
            } else {
                assert!(!scratch.constraints.is_empty(),
                    "prescribed deformable bonds still require fracture evaluation");
            }
        }
    }

    #[test]
    fn fine_solve_filters_only_kinematic_to_kinematic_bonds() {
        use crate::material::Material;
        use crate::scene::{SceneConfig, VoxelSpawn};

        let oak = Material::component_id("oak").unwrap();
        let make_scene = || {
            let mut config = SceneConfig::default();
            config.gravity = [0.0; 3];
            config.atmosphere.pressure_pa = 0.0;
            Scene::from_voxels(
                config,
                vec![
                    // Build the lattice while all REV are deformable so the
                    // support bond exists; Scene construction omits bonds
                    // between unlike kinematic states.
                    VoxelSpawn::at_cell([0, 0, 0], oak),
                    VoxelSpawn::at_cell([1, 0, 0], oak),
                    VoxelSpawn::at_cell([0, 0, 1], oak),
                ],
            )
            .unwrap()
        };

        let mut filtered = make_scene();
        let mut reference = make_scene();
        let index_at_cell = |scene: &Scene, cell| {
            let center = VoxelSpawn::at_cell(cell, oak).center;
            scene
                .positions
                .iter()
                .position(|&position| position == center)
                .expect("the requested fixture cell should exist")
        };
        let prescribed = index_at_cell(&filtered, [0, 0, 1]);
        // Preserve the bonds generated above while imposing the fixed state,
        // so the fixture exercises the fine solver's kinematic bond filter.
        for scene in [&mut filtered, &mut reference] {
            let fixed_a = index_at_cell(scene, [0, 0, 0]);
            let fixed_b = index_at_cell(scene, [1, 0, 0]);
            scene.voxels[fixed_a].kinematic = true;
            scene.voxels[fixed_b].kinematic = true;
        }
        let mut fixed_fixed = None;
        'find_bond: for axis in 0..3 {
            for i in 0..filtered.voxels.len() {
                let j = filtered.bonds[axis][i].neighbor;
                if j != EMPTY
                    && filtered.voxels[i].kinematic
                    && filtered.voxels[j as usize].kinematic
                {
                    fixed_fixed = Some((axis, i));
                    break 'find_bond;
                }
            }
        }
        let fixed_fixed = fixed_fixed
            .expect("the fixed floor should contain a bonded kinematic pair");
        reference.bonds[fixed_fixed.0][fixed_fixed.1].neighbor = EMPTY;

        let positions = filtered.positions.clone();
        let mut targets = vec![[None; 3]; filtered.voxels.len()];
        targets[prescribed] = positions[prescribed].to_array().map(Some);
        let loads = vec![Vec3::ZERO; filtered.voxels.len()];
        let mut filtered_scratch = SolverScratch::default();
        let mut reference_scratch = SolverScratch::default();
        let mut filtered_cuda = false;
        let mut reference_cuda = false;
        solve_coupled(
            &mut filtered,
            0.01,
            &targets,
            &loads,
            &positions,
            &mut filtered_scratch,
            false,
            false,
            &mut filtered_cuda,
        );
        solve_coupled(
            &mut reference,
            0.01,
            &targets,
            &loads,
            &positions,
            &mut reference_scratch,
            false,
            false,
            &mut reference_cuda,
        );

        assert!(filtered_scratch.constraints.iter().all(|constraint| {
            !(filtered.voxels[constraint.i].kinematic
                && filtered.voxels[constraint.j].kinematic)
        }));
        assert!(filtered_scratch.constraints.iter().any(|constraint| {
            filtered.voxels[constraint.i].kinematic
                != filtered.voxels[constraint.j].kinematic
        }), "kinematic-to-prescribed deformable support bonds must remain assembled");
        assert_eq!(filtered.positions, reference.positions);
        assert_eq!(filtered.voxels.iter().map(|v| v.velocity).collect::<Vec<_>>(),
            reference.voxels.iter().map(|v| v.velocity).collect::<Vec<_>>());
        assert_eq!(filtered.broken, reference.broken);
    }

    #[test]
    fn auto_aggregation_accepts_only_solver_scale_residuals() {
        let mut scene = free_falling_oak_block();
        let mut nonzero_force_bonds = 0usize;
        let mut max_force_n = 0.0_f32;
        for axis in 0..3 {
            for index in 0..scene.voxels.len() {
                if scene.bonds[axis][index].neighbor != EMPTY {
                    let force = scene.workspace[axis][index].last_force.length();
                    nonzero_force_bonds += usize::from(force != 0.0);
                    max_force_n = max_force_n.max(force);
                }
            }
        }
        let members: Vec<_> = (0..scene.voxels.len()).collect();
        let candidate = scene.rigid_aggregate_snapshot(&members).unwrap();
        let internal_motion = candidate.has_unresolved_internal_motion();
        let targets = vec![[None; 3]; scene.voxels.len()];
        let loads = vec![Vec3::ZERO; scene.voxels.len()];
        let registered = auto_register_quiescent_components(&mut scene, &targets, &loads, &[]);
        let gravity_force_n =
            scene.voxels[0].mass * Vec3::from_array(scene.config.gravity).length();
        let force_tolerance_n = scene.config.solver_relative_tolerance * gravity_force_n;
        println!(
            "free-fall aggregate: cluster_count={} component_revs={} bonds={} nonzero_force_bonds={} max_bond_force_n={:.9e} force_tolerance_n={:.9e} solver_residual={:.3e} unresolved_internal_motion={} registered_revs={}",
            scene.cluster_count,
            members.len(),
            scene
                .bonds
                .iter()
                .map(|axis| axis.iter().filter(|bond| bond.neighbor != EMPTY).count())
                .sum::<usize>(),
            nonzero_force_bonds,
            max_force_n,
            force_tolerance_n,
            scene.solver_residual,
            internal_motion,
            registered,
        );
        assert_eq!(scene.cluster_count, 1);
        assert!(!internal_motion);
        assert!(max_force_n <= force_tolerance_n);
        assert_eq!(registered, members.len());
        assert!(scene.aggregate_for_voxel(0).is_some());
    }

    #[test]
    fn auto_aggregation_rejects_resolvable_internal_stress() {
        let mut scene = free_falling_oak_block();
        scene.workspace[0][0].last_force = Vec3::X * 1.0e-3;
        let targets = vec![[None; 3]; scene.voxels.len()];
        let loads = vec![Vec3::ZERO; scene.voxels.len()];
        let registered = auto_register_quiescent_components(&mut scene, &targets, &loads, &[]);

        assert_eq!(registered, 0);
        assert!(scene.aggregate_for_voxel(0).is_none());
    }

    #[test]
    fn contact_in_component_blocks_coarse_registration() {
        let mut scene = free_falling_oak_block();
        let targets = vec![[None; 3]; scene.voxels.len()];
        let loads = vec![Vec3::ZERO; scene.voxels.len()];
        let contacts = [collision::Contact {
            i: 0,
            j: 63,
            normal: Vec3::X,
            new_impact: false,
        }];

        let registered =
            auto_register_quiescent_components(&mut scene, &targets, &loads, &contacts);

        assert_eq!(registered, 0);
        assert!(scene.aggregate_for_voxel(0).is_none());
    }

    #[test]
    fn nonuniform_force_wakes_aggregate_before_fine_solve() {
        let mut scene = free_falling_oak_block();
        let targets = vec![[None; 3]; scene.voxels.len()];
        let loads = vec![Vec3::ZERO; scene.voxels.len()];
        assert_eq!(
            auto_register_quiescent_components(&mut scene, &targets, &loads, &[]),
            64
        );

        scene.external_forces[0] = Vec3::X * 1.0e-4;
        let mut tool = None;
        step(&mut scene, &mut tool);

        assert!(scene.aggregate_of.iter().all(Option::is_none));
        assert!(scene.voxels[0].velocity.x > 0.0);
        assert_eq!(scene.broken, 0);
    }

    #[test]
    fn coarse_prediction_wake_reinitializes_fine_velocities_before_solve() {
        use crate::{material::Material, scene::VoxelSpawn};

        let oak = Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(65);
        for z in 8..12 {
            for y in 0..4 {
                for x in 0..4 {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], oak));
                }
            }
        }
        spawns.push(VoxelSpawn::at_cell([5, 0, 8], oak).fixed());
        let mut config = crate::scene::SceneConfig::default();
        config.fixed_dt = 0.01;
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        for voxel in &mut scene.voxels[..64] {
            voxel.velocity = Vec3::X * 2.0;
        }
        scene
            .register_rigid_aggregate_candidate(&(0..64).collect::<Vec<_>>())
            .unwrap();
        scene.solver_scratch.drift_velocity = vec![Vec3::splat(99.0); scene.voxels.len()];
        scene.solver_scratch.tool_contact_velocity = vec![Vec3::splat(99.0); scene.voxels.len()];
        scene.previous.clone_from(&scene.positions);
        let mut tool = None;

        step(&mut scene, &mut tool);

        assert!(scene.aggregate_of[..64].iter().all(Option::is_none));
        assert!(scene.voxels[..64]
            .iter()
            .all(|voxel| voxel.velocity.abs().max_element() < 10.0));
        assert_eq!(scene.broken, 0);
    }

    #[test]
    fn geometric_multigrid_eligibility_uses_rest_cells_and_rejects_non_face_edges() {
        let constraint = Constraint {
            i: 0,
            j: 1,
            axis: 0,
            frame: Mat3::IDENTITY,
            material_k: Mat3::IDENTITY,
            k: Mat3::IDENTITY,
            rest: DVec3::ZERO,
            rotation_i: Quat::IDENTITY,
            rotation_j: Quat::IDENTITY,
            thermal_energy_i: 0.0,
            thermal_energy_j: 0.0,
            damage: 0.0,
            stiffness_revisions: [0; 2],
            rest_input: RestInput::default(),
        };
        let incidence = [[0_u32; 6]; 2];
        let incidence_len = [1_u8; 2];
        let active = [0_usize, 1];
        let active_index = [0_u32, 1];
        let rest = [
            DVec3::splat(DCELL * 0.5),
            DVec3::splat(DCELL * 0.5) + DVec3::X * DCELL,
        ];
        let coordinates = geometric_multigrid_coordinates(
            &active,
            &active_index,
            &incidence,
            &incidence_len,
            &[constraint],
            &rest,
        )
        .unwrap();
        assert_eq!(coordinates, [[0, 0, 0], [1, 0, 0]]);

        let diagonal_neighbor = [rest[0], rest[0] + DVec3::new(DCELL, DCELL, 0.0)];
        assert!(geometric_multigrid_coordinates(
            &active,
            &active_index,
            &incidence,
            &incidence_len,
            &[constraint],
            &diagonal_neighbor,
        )
        .is_err());
    }

    #[test]
    fn component_ic0_matches_serial_apply_for_interleaved_components() {
        let constraint = |i, j| Constraint {
            i,
            j,
            axis: 0,
            frame: Mat3::IDENTITY,
            material_k: Mat3::IDENTITY,
            k: Mat3::IDENTITY,
            rest: DVec3::ZERO,
            rotation_i: Quat::IDENTITY,
            rotation_j: Quat::IDENTITY,
            thermal_energy_i: 0.0,
            thermal_energy_j: 0.0,
            damage: 0.0,
            stiffness_revisions: [0; 2],
            rest_input: RestInput::default(),
        };
        let constraints = [constraint(0, 3), constraint(1, 2)];
        let mut incidence = [[u32::MAX; 6]; 4];
        incidence[0][0] = 0;
        incidence[3][0] = 0;
        incidence[1][0] = 1;
        incidence[2][0] = 1;
        let incidence_len = [1; 4];
        let mut active = vec![0, 1, 2, 3];
        let mut active_index = vec![0, 1, 2, 3];
        let mut serial_lower = Vec::new();
        let mut serial_lower_count = Vec::new();
        let mut serial_upper = Vec::new();
        let mut serial_upper_count = Vec::new();
        build_ic0_links(
            &constraints,
            &incidence,
            &incidence_len,
            &active,
            &active_index,
            &mut serial_lower,
            &mut serial_lower_count,
            &mut serial_upper,
            &mut serial_upper_count,
        );

        let factor_diagonal_inverse = [Mat3::IDENTITY; 4];
        let factor_edges = [Mat3::IDENTITY * 0.125; 2];
        let rhs = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::splat(2.0)];
        let mut serial_work = [Vec3::ZERO; 4];
        let mut serial_out = [Vec3::ZERO; 4];
        apply_ic0(
            &factor_diagonal_inverse,
            &factor_edges,
            &active,
            &serial_lower,
            &serial_lower_count,
            &serial_upper,
            &serial_upper_count,
            &rhs,
            &mut serial_work,
            &mut serial_out,
        );

        let mut parent = Vec::new();
        let mut root_to_id = Vec::new();
        let mut sizes = Vec::new();
        let mut offsets = Vec::new();
        let mut members = Vec::new();
        let mut ranges = Vec::new();
        assert_eq!(
            order_active_by_ic0_components(
                &constraints,
                &mut active,
                &mut active_index,
                &mut parent,
                &mut root_to_id,
                &mut sizes,
                &mut offsets,
                &mut members,
                &mut ranges,
            ),
            2
        );
        assert_eq!(active, [0, 3, 1, 2]);
        assert_eq!(ranges, [(0, 2), (2, 4)]);
        let mut batches = Vec::new();
        build_ic0_component_batches(&ranges, active.len(), &mut batches);

        let mut component_lower = Vec::new();
        let mut component_lower_count = Vec::new();
        let mut component_upper = Vec::new();
        let mut component_upper_count = Vec::new();
        build_ic0_links(
            &constraints,
            &incidence,
            &incidence_len,
            &active,
            &active_index,
            &mut component_lower,
            &mut component_lower_count,
            &mut component_upper,
            &mut component_upper_count,
        );
        let mut component_work = [Vec3::ZERO; 4];
        let mut component_out = [Vec3::ZERO; 4];
        let mut component_rhs = [Vec3::ZERO; 4];
        for (slot, &voxel) in active.iter().enumerate() {
            component_rhs[slot] = rhs[voxel];
        }
        apply_ic0_component_batches(
            &factor_diagonal_inverse,
            &factor_edges,
            &active,
            &ranges,
            &batches,
            &component_lower,
            &component_lower_count,
            &component_upper,
            &component_upper_count,
            &component_rhs,
            &mut component_work,
            &mut component_out,
        );
        let mut component_out_by_voxel = [Vec3::ZERO; 4];
        for (slot, &voxel) in active.iter().enumerate() {
            component_out_by_voxel[voxel] = component_out[slot];
        }
        assert_eq!(component_out_by_voxel, serial_out);
    }

    #[test]
    fn reused_component_ic0_rebuilds_links_when_active_graph_splits() {
        let constraint = Constraint {
            i: 0,
            j: 1,
            axis: 0,
            frame: Mat3::IDENTITY,
            material_k: Mat3::IDENTITY,
            k: Mat3::IDENTITY,
            rest: DVec3::ZERO,
            rotation_i: Quat::IDENTITY,
            rotation_j: Quat::IDENTITY,
            thermal_energy_i: 0.0,
            thermal_energy_j: 0.0,
            damage: 0.0,
            stiffness_revisions: [0; 2],
            rest_input: RestInput::default(),
        };
        let old_constraints = [constraint];
        let old_incidence = [[0_u32; 6]; 2];
        let old_incidence_len = [1_u8; 2];
        let active = [0_usize, 1];
        let active_index = [0_u32, 1];
        let mut lower_links = Vec::new();
        let mut lower_count = Vec::new();
        let mut upper_links = Vec::new();
        let mut upper_count = Vec::new();
        build_ic0_links(
            &old_constraints,
            &old_incidence,
            &old_incidence_len,
            &active,
            &active_index,
            &mut lower_links,
            &mut lower_count,
            &mut upper_links,
            &mut upper_count,
        );
        assert_eq!(upper_count[0], 1);
        let stale_neighbor = upper_links[0][0][1] as usize;
        assert_eq!(stale_neighbor, 1);
        assert!(!(0..1).contains(&stale_neighbor));

        // A contact active-set change removes the edge and splits the current
        // graph into singleton components while the caller requests reuse.
        assert!(should_rebuild_cpu_preconditioner(true, false, false, false));
        let new_incidence = [[u32::MAX; 6]; 2];
        let new_incidence_len = [0_u8; 2];
        build_ic0_links(
            &[],
            &new_incidence,
            &new_incidence_len,
            &active,
            &active_index,
            &mut lower_links,
            &mut lower_count,
            &mut upper_links,
            &mut upper_count,
        );
        assert_eq!(lower_count, [0, 0]);
        assert_eq!(upper_count, [0, 0]);

        let factor_diagonal_inverse = [Mat3::IDENTITY; 2];
        let factor_edges = [Mat3::IDENTITY];
        for start in 0..2 {
            let rhs = [Vec3::X];
            let mut work = [Vec3::ZERO];
            let mut output = [Vec3::ZERO];
            apply_ic0_component(
                start,
                &active,
                &factor_diagonal_inverse,
                &factor_edges,
                &lower_links,
                &lower_count,
                &upper_links,
                &upper_count,
                &rhs,
                &mut work,
                &mut output,
            );
            assert_eq!(output, [Vec3::X]);
        }
    }

    #[test]
    fn independent_component_pcg_solves_each_block_to_the_same_residual_contract() {
        let active = [0_usize, 1];
        let diagonal = [Mat3::IDENTITY * 2.0; 2];
        let rows = [PcgRow::default(); 2];
        let edge_blocks: [[f32; 9]; 0] = [];
        let incidence_len = [0_u8; 2];
        let rhs = [Vec3::new(2.0, 4.0, 6.0), Vec3::new(4.0, -2.0, 8.0)];
        let factor_diagonal_inverse = [Mat3::IDENTITY * (1.0 / 2.0_f32.sqrt()); 2];
        let factor_edges: [Mat3; 0] = [];
        let lower_links = [[[0_u32; 2]; 6]; 2];
        let lower_link_count = [0_u8; 2];
        let upper_links = [[[0_u32; 2]; 6]; 2];
        let upper_link_count = [0_u8; 2];
        let ranges = [(0, 1), (1, 2)];
        let mut dx = vec![Vec3::ZERO; 2];
        let mut ap = vec![Vec3::ZERO; 2];
        let mut r = rhs.to_vec();
        let mut z = vec![Vec3::ZERO; 2];
        let mut p = vec![Vec3::ZERO; 2];
        let mut work = vec![Vec3::ZERO; 2];
        let chunks = split_mut_by_ranges(&mut dx, &ranges)
            .into_iter()
            .zip(split_mut_by_ranges(&mut ap, &ranges))
            .zip(split_mut_by_ranges(&mut r, &ranges))
            .zip(split_mut_by_ranges(&mut z, &ranges))
            .zip(split_mut_by_ranges(&mut p, &ranges))
            .zip(split_mut_by_ranges(&mut work, &ranges));
        let mut states = Vec::with_capacity(ranges.len());
        for (component, (((((dx, ap), r), z), p), work)) in chunks.enumerate() {
            let (start, end) = ranges[component];
            states.push(ComponentPcgState {
                start,
                end,
                dx,
                ap,
                r,
                z,
                p,
                work,
                iterations: 0,
                converged: false,
            });
        }
        states.par_iter_mut().for_each(|state| {
            solve_pcg_component(
                state,
                &active,
                &diagonal,
                &rows,
                &edge_blocks,
                &incidence_len,
                &rhs,
                &factor_diagonal_inverse,
                &factor_edges,
                &lower_links,
                &lower_link_count,
                &upper_links,
                &upper_link_count,
                1e-5,
                8,
            );
        });
        assert!(states.iter().all(|state| state.converged));
        drop(states);

        for component in 0..2 {
            let expected = rhs[component] * 0.5;
            assert!((dx[component] - expected).length() < 1e-6);
        }
    }

    #[test]
    fn component_ic0_does_not_join_through_inactive_degree_of_freedom() {
        let constraint = Constraint {
            i: 0,
            j: 2,
            axis: 0,
            frame: Mat3::IDENTITY,
            material_k: Mat3::IDENTITY,
            k: Mat3::IDENTITY,
            rest: DVec3::ZERO,
            rotation_i: Quat::IDENTITY,
            rotation_j: Quat::IDENTITY,
            thermal_energy_i: 0.0,
            thermal_energy_j: 0.0,
            damage: 0.0,
            stiffness_revisions: [0; 2],
            rest_input: RestInput::default(),
        };
        let mut active = vec![0, 1];
        let mut active_index = vec![0, 1, u32::MAX];
        let mut parent = Vec::new();
        let mut root_to_id = Vec::new();
        let mut sizes = Vec::new();
        let mut offsets = Vec::new();
        let mut members = Vec::new();
        let mut ranges = Vec::new();
        assert_eq!(
            order_active_by_ic0_components(
                &[constraint],
                &mut active,
                &mut active_index,
                &mut parent,
                &mut root_to_id,
                &mut sizes,
                &mut offsets,
                &mut members,
                &mut ranges,
            ),
            2
        );
        assert_eq!(active, [0, 1]);
        assert_eq!(ranges, [(0, 1), (1, 2)]);
    }

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
        let initial_velocity = s.voxels[0].velocity;
        set_force(&mut s, 0, Vec3::X * (3.0 * mass)).unwrap();
        set_acceleration(&mut s, 0, Vec3::Y * 2.0).unwrap();
        let mut tool = None;
        step(&mut s, &mut tool);
        assert!((s.voxels[0].velocity.x - 0.03).abs() < 1e-5);
        assert!((s.voxels[0].velocity.y - 0.02).abs() < 1e-5);
        let total_force = Vec3::X * (3.0 * mass) + Vec3::Y * (2.0 * mass);
        let expected_work = total_force
            .dot((initial_velocity + s.voxels[0].velocity) * (0.5 * s.config.fixed_dt))
            as f64;
        assert!((s.energy.external_work - expected_work).abs() < 1e-10);
        let kinetic_change = 0.5
            * mass as f64
            * (s.voxels[0].velocity.length_squared() - initial_velocity.length_squared()) as f64;
        assert!((s.energy.external_work - kinetic_change).abs() < 1e-10);
        step(&mut s, &mut tool);
        assert!((s.voxels[0].velocity.x - 0.06).abs() < 1e-5);
        assert!((s.voxels[0].velocity.y - 0.04).abs() < 1e-5);
        clear_applied_actions(&mut s, 0).unwrap();
        step(&mut s, &mut tool);
        assert!((s.voxels[0].velocity.x - 0.06).abs() < 1e-5);
        assert!((s.voxels[0].velocity.y - 0.04).abs() < 1e-5);
    }

    #[test]
    fn solver_commits_automatic_fusion_at_fixed_step_boundary() {
        let iron = crate::material::Material::component_id("iron").unwrap();
        let copper = crate::material::Material::component_id("copper").unwrap();
        let mut s = crate::scene::SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .with_timestep(0.01, 2)
            .add_voxel(
                crate::scene::VoxelSpawn::at_cell([0, 0, 0], iron)
                    .fixed()
                    .at_temperature(363.15),
            )
            .add_voxel(crate::scene::VoxelSpawn::at_cell([1, 0, 0], copper).at_temperature(363.15))
            .build()
            .unwrap();
        s.positions[1].x += 0.005;
        s.previous[1] = s.positions[1];
        s.release_voxel(0).unwrap();
        s.voxels[0].velocity = Vec3::new(0.01, 0.005, 0.0);
        s.voxels[1].velocity = Vec3::new(-0.01, -0.005, 0.0);
        s.config.fracture.restitution = 0.0;
        s.config.fracture.friction = 0.0;
        set_force(&mut s, 0, Vec3::X * 0.1).unwrap();
        set_force(&mut s, 1, Vec3::NEG_X * 0.1).unwrap();
        s.set_thermal_fusion_rule(350.0, 900.0, 0.1).unwrap();
        let total_energy = |scene: &Scene| {
            scene
                .voxels
                .iter()
                .map(|voxel| {
                    voxel.total_internal_energy_j()
                        + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
                })
                .sum::<f64>()
        };
        let initial_energy = total_energy(&s);
        let initial_external_work = s.energy.external_work;
        let mut tool = None;
        for _ in 0..20 {
            step(&mut s, &mut tool);
            if s.voxels.len() == 1 {
                break;
            }
        }
        assert_eq!(s.voxels.len(), 1);
        assert!(s.energy.fusion_heat > 0.0);
        assert!(s.energy.external_work > 0.0);
        assert_eq!(s.cluster_count, 1);
        let residual =
            total_energy(&s) - initial_energy - (s.energy.external_work - initial_external_work);
        let energy_roundoff = 2.0 * f64::from(f32::EPSILON) * initial_energy;
        assert!(
            residual <= 1.0e-12 && residual >= -energy_roundoff,
            "automatic fusion energy residual={residual:.12e} J exceeds f32 accounting precision {energy_roundoff:.12e} J"
        );
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
            s.positions[i] = q.as_dquat() * s.positions[i];
        }
        let mut scratch = SolverScratch::default();
        fit_rotations(&mut s, &mut scratch);
        let frame = basis(0, s.rotations[0]);
        let j = s.bonds[0][0].neighbor as usize;
        let c = (frame.as_dmat3().transpose() * (s.positions[j] - s.positions[0])
            - DVec3::X * DCELL)
            .as_vec3();
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
    fn hot_solid_bond_limits_follow_material_softening_curve() {
        let mut s = scene();
        let iron = crate::material::Material::component_id("iron").unwrap();
        for i in 0..s.voxels.len() {
            s.voxels[i].composition = crate::voxel::Composition::pure(iron);
            s.properties[i] =
                crate::material::Material::mixed(&s.voxels[i].composition, &s.materials);
            s.voxels[i].internal_energy =
                s.properties[i].energy_at_temperature(s.voxels[i].mass, 300.0);
        }
        let i = 0;
        let j = s.bonds[0][i].neighbor as usize;
        let ambient = bond_limits(&s, i, j, Vec3::X);
        let material = &s.properties[i];
        let warm_temperature = (material.melt_temp * material.strength_softening_start_ratio
            + material.melt_temp)
            * 0.5;
        for n in [i, j] {
            s.voxels[n].internal_energy =
                s.properties[n].energy_at_temperature(s.voxels[n].mass, warm_temperature);
        }
        let warm = bond_limits(&s, i, j, Vec3::X);
        assert!((warm.tension / ambient.tension - 0.5).abs() < 1e-5);
        assert!((warm.shear / ambient.shear - 0.5).abs() < 1e-5);
    }

    #[test]
    fn heated_bond_expands_against_a_fixed_boundary() {
        let iron = crate::material::Material::component_id("iron").unwrap();
        let mut config = crate::scene::SceneConfig::default();
        config.gravity = [0.0; 3];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        config.corotation_iterations = 1;
        let mut s = Scene::from_voxels(
            config,
            vec![
                crate::scene::VoxelSpawn::at_cell([0, 0, 0], iron),
                crate::scene::VoxelSpawn::at_cell([1, 0, 0], iron),
            ],
        )
        .unwrap();
        s.set_material_thermal_expansion("iron", 1.0e-5).unwrap();
        s.fix_voxel(0).unwrap();
        for i in 0..2 {
            s.voxels[i].internal_energy =
                s.properties[i].energy_at_temperature(s.voxels[i].mass, 393.15);
        }

        let initial_length = (s.positions[1] - s.positions[0]).x;
        let constraint = make_constraint(&s, 0, 0, 1);
        assert!((constraint.rest.x / DCELL - 1.001).abs() < 1.0e-6);
        let mut tool = None;
        step(&mut s, &mut tool);
        let expanded_length = (s.positions[1] - s.positions[0]).x;
        assert!(expanded_length > initial_length);
        assert_ne!(s.bonds[0][0].neighbor, EMPTY);
    }

    #[test]
    fn thermal_expansion_projects_parallel_and_transverse_fiber_coefficients() {
        let wood = crate::material::Material::component_id("wood").unwrap();
        let microstructure = crate::voxel::Microstructure {
            orientation: Quat::IDENTITY,
            anisotropy: 1.0,
        };
        let spawn = |cell| {
            crate::scene::VoxelSpawn::at_cell(cell, wood)
                .at_temperature(293.15)
                .with_microstructure(microstructure)
        };
        let mut config = crate::scene::SceneConfig::default();
        config.gravity = [0.0; 3];
        let mut s = Scene::from_voxels(
            config,
            vec![
                spawn([-1, 0, 0]),
                spawn([0, 0, 0]),
                spawn([1, 0, 0]),
                spawn([4, 0, 0]),
                spawn([4, 0, 1]),
                spawn([4, 0, 2]),
            ],
        )
        .unwrap();
        s.set_material_thermal_expansion_anisotropic("wood", 3.0e-6, 30.0e-6)
            .unwrap();
        for i in 0..s.voxels.len() {
            s.voxels[i].internal_energy =
                s.properties[i].energy_at_temperature(s.voxels[i].mass, 393.15);
        }
        let transverse_frame = basis(0, Quat::IDENTITY);
        let parallel_frame = basis(2, Quat::IDENTITY);
        let transverse_rest = bond_thermal_rest(&s, 1, 2, transverse_frame).x;
        let parallel_rest = bond_thermal_rest(&s, 4, 5, parallel_frame).x;
        assert!((transverse_rest / DCELL - 1.003).abs() < 1e-6);
        assert!((parallel_rest / DCELL - 1.0003).abs() < 1e-6);
        assert!(transverse_rest > parallel_rest * 1.002);
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
                targets[i] = Some(s.positions[i] + DVec3::X * 0.00002);
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

fn apply_releases(scene: &mut Scene, targets: &[[Option<f64>; 3]]) {
    let releases = std::mem::take(&mut scene.releases);
    for r in releases {
        if r.due > scene.substep_index {
            scene.releases.push(r);
            continue;
        }
        let normal = (scene.positions[r.j] - scene.positions[r.i])
            .normalize_or_zero()
            .as_vec3();
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
            scene.voxels[r.i].add_internal_energy_j(r.energy * 0.5);
            scene.voxels[r.j].add_internal_energy_j(r.energy * 0.5);
        }
    }
}

#[derive(Clone, Copy)]
struct Constraint {
    i: usize,
    j: usize,
    axis: usize,
    frame: Mat3,
    material_k: Mat3,
    k: Mat3,
    rest: DVec3,
    rotation_i: Quat,
    rotation_j: Quat,
    thermal_energy_i: f32,
    thermal_energy_j: f32,
    damage: f32,
    stiffness_revisions: [u64; 2],
    rest_input: RestInput,
}

/// Exact inputs to the bond rest vector. Thermal expansion and bond offsets
/// can change independently of the directional compliance operator.
#[derive(Clone, Copy, PartialEq)]
struct RestInput {
    fiber_orientations: [Quat; 2],
    expansion: [[f32; 2]; 2],
    temperatures: [f32; 2],
    reference_temperatures: [f32; 2],
    formation_offset: [f64; 3],
    plastic_strain: [f32; 3],
}

impl Default for RestInput {
    fn default() -> Self {
        Self {
            fiber_orientations: [Quat::IDENTITY; 2],
            expansion: [[0.0; 2]; 2],
            temperatures: [0.0; 2],
            reference_temperatures: [0.0; 2],
            formation_offset: [0.0; 3],
            plastic_strain: [0.0; 3],
        }
    }
}

/// Exact inputs to the directional compliance operator. No rounded/hash key:
/// a changed rotation, fiber, phase fraction or modulus invalidates the cache.
#[derive(Clone, Copy, PartialEq)]
struct StiffnessInput {
    rotation: Quat,
    world_fiber: Vec3,
    anisotropy: f32,
    moduli: Vec3,
    solid: f32,
}

#[derive(Clone, Copy)]
struct PcgRow {
    neighbors: [u32; 6],
    edge_ids: [u32; 6],
    // Each constitutive block is stored once per bond in pcg_edge_blocks.
    // The bit marks rows whose block is the transpose of the canonical i→j block.
    transpose_mask: u8,
}

#[derive(Clone, Copy)]
struct FractureOutcome {
    bond: crate::bond::Bond,
    last_force: Vec3,
    dissipation: crate::fracture::Dissipation,
}

impl Default for PcgRow {
    fn default() -> Self {
        Self {
            neighbors: [u32::MAX; 6],
            edge_ids: [u32::MAX; 6],
            transpose_mask: 0,
        }
    }
}

fn assemble_pcg_row(
    voxel: usize,
    incident: &[u32],
    constraints: &[Constraint],
    active_index: &[u32],
) -> PcgRow {
    let mut row = PcgRow::default();
    for (slot, &edge) in incident.iter().enumerate() {
        let constraint = &constraints[edge as usize];
        let reverse = constraint.i != voxel;
        let neighbor = if reverse { constraint.i } else { constraint.j };
        row.neighbors[slot] = active_index[neighbor];
        row.edge_ids[slot] = edge;
        if reverse {
            row.transpose_mask |= 1 << slot;
        }
    }
    row
}

#[cfg(test)]
#[test]
fn pcg_row_reassembly_clears_removed_neighbors_and_transpose_bits() {
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};
    let oak = Material::component_id("oak").unwrap();
    let scene = Scene::from_voxels(SceneConfig::default(), (0..3)
        .map(|x| VoxelSpawn::at_cell([x, 0, 0], oak)).collect()).unwrap();
    let constraints = [make_constraint(&scene, 0, 0, 1), make_constraint(&scene, 1, 0, 2)];
    let active = [0, 1, 2];
    let old = assemble_pcg_row(1, &[0, 1], &constraints, &active);
    assert_eq!(old.transpose_mask, 1);
    assert_eq!(&old.neighbors[..2], &[0, 2]);
    // Removing the first edge moves a forward edge into a formerly reversed slot.
    let rebuilt = assemble_pcg_row(1, &[1], &constraints, &active);
    assert_eq!(rebuilt.transpose_mask, 0);
    assert_eq!(rebuilt.neighbors, [2, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    assert_eq!(rebuilt.edge_ids, [1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
}

#[inline]
fn pack_pcg_block(matrix: Mat3) -> [f32; 9] {
    [
        matrix.x_axis.x,
        matrix.y_axis.x,
        matrix.z_axis.x,
        matrix.x_axis.y,
        matrix.y_axis.y,
        matrix.z_axis.y,
        matrix.x_axis.z,
        matrix.y_axis.z,
        matrix.z_axis.z,
    ]
}

#[inline]
fn apply_pcg_block(matrix: [f32; 9], value: Vec3) -> Vec3 {
    Vec3::new(
        matrix[0] * value.x + matrix[1] * value.y + matrix[2] * value.z,
        matrix[3] * value.x + matrix[4] * value.y + matrix[5] * value.z,
        matrix[6] * value.x + matrix[7] * value.y + matrix[8] * value.z,
    )
}

#[inline]
fn transpose_pcg_block(matrix: [f32; 9]) -> [f32; 9] {
    [
        matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5],
        matrix[8],
    ]
}

#[inline]
fn apply_pcg_block_transpose(matrix: [f32; 9], value: Vec3) -> Vec3 {
    Vec3::new(
        matrix[0] * value.x + matrix[3] * value.y + matrix[6] * value.z,
        matrix[1] * value.x + matrix[4] * value.y + matrix[7] * value.z,
        matrix[2] * value.x + matrix[5] * value.y + matrix[8] * value.z,
    )
}

#[inline]
fn apply_pcg_block_f64(a: [f32; 9], x: DVec3) -> DVec3 {
    DVec3::new(
        a[0] as f64 * x.x + a[1] as f64 * x.y + a[2] as f64 * x.z,
        a[3] as f64 * x.x + a[4] as f64 * x.y + a[5] as f64 * x.z,
        a[6] as f64 * x.x + a[7] as f64 * x.y + a[8] as f64 * x.z,
    )
}

#[inline]
fn apply_pcg_block_f64_oriented(a: [f32; 9], x: DVec3, transpose: bool) -> DVec3 {
    if transpose {
        DVec3::new(
            a[0] as f64 * x.x + a[3] as f64 * x.y + a[6] as f64 * x.z,
            a[1] as f64 * x.x + a[4] as f64 * x.y + a[7] as f64 * x.z,
            a[2] as f64 * x.x + a[5] as f64 * x.y + a[8] as f64 * x.z,
        )
    } else {
        apply_pcg_block_f64(a, x)
    }
}

/// Replace the recurrence residual by an independent b-Ax evaluation. The
/// returned norm uses f64 before residual components are stored for a restart.
fn recompute_pcg_residual(
    diagonal: &[Mat3],
    rows: &[PcgRow],
    edge_blocks: &[[f32; 9]],
    active: &[usize],
    incidence_len: &[u8],
    rhs: &[Vec3],
    dx: &[Vec3],
    residual: &mut [Vec3],
) -> f64 {
    let mut norm_squared = 0.0;
    let mut rhs_squared = 0.0;
    for (slot, &i) in active.iter().enumerate() {
        let mut ax = diagonal[i].as_dmat3() * dx[slot].as_dvec3();
        for edge in 0..incidence_len[i] as usize {
            let neighbor = rows[slot].neighbors[edge];
            if neighbor != u32::MAX {
                let block = edge_blocks[rows[slot].edge_ids[edge] as usize];
                ax += apply_pcg_block_f64_oriented(
                    block,
                    dx[neighbor as usize].as_dvec3(),
                    rows[slot].transpose_mask & (1 << edge) != 0,
                );
            }
        }
        let b = rhs[i].as_dvec3();
        let r = b - ax;
        residual[slot] = r.as_vec3();
        norm_squared += r.length_squared();
        rhs_squared += b.length_squared();
    }
    if rhs_squared > 0.0 {
        (norm_squared / rhs_squared).sqrt()
    } else if norm_squared == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Evaluate the original assembled operator independently of PCG's recurrence
/// and preconditioner. Also measure the actual correction retained by scene
/// coordinates. This is an opt-in diagnostic; it does not change the trajectory.
fn audit_pcg_residual(
    diagonal: &[Mat3],
    rows: &[PcgRow],
    edge_blocks: &[[f32; 9]],
    active: &[usize],
    incidence_len: &[u8],
    rhs: &[Vec3],
    dx: &[Vec3],
    positions: &[DVec3],
) -> crate::scene::SolverResidualAudit {
    let applied = |slot: usize| {
        let i = active[slot];
        // Record the correction actually retained by authoritative coordinates.
        (positions[i] + dx[slot].as_dvec3()) - positions[i]
    };
    let mut rhs_squared = 0.0;
    let mut correction_squared = 0.0;
    let mut applied_squared = 0.0;
    let mut rounding_squared = 0.0;
    for (slot, &i) in active.iter().enumerate() {
        let x = dx[slot].as_dvec3();
        let rounded = applied(slot);
        let mut ax = diagonal[i].as_dmat3() * x;
        let mut ax_rounded = diagonal[i].as_dmat3() * rounded;
        for edge in 0..incidence_len[i] as usize {
            let neighbor = rows[slot].neighbors[edge];
            if neighbor == u32::MAX {
                continue;
            }
            let block = edge_blocks[rows[slot].edge_ids[edge] as usize];
            let transpose = rows[slot].transpose_mask & (1 << edge) != 0;
            ax += apply_pcg_block_f64_oriented(block, dx[neighbor as usize].as_dvec3(), transpose);
            ax_rounded +=
                apply_pcg_block_f64_oriented(block, applied(neighbor as usize), transpose);
        }
        let b = rhs[i].as_dvec3();
        rhs_squared += b.length_squared();
        correction_squared += (b - ax).length_squared();
        applied_squared += (b - ax_rounded).length_squared();
        rounding_squared += (x - rounded).length_squared();
    }
    let relative = |squared: f64| {
        if rhs_squared > 0.0 {
            (squared / rhs_squared).sqrt()
        } else if squared == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    };
    crate::scene::SolverResidualAudit {
        correction_relative: relative(correction_squared),
        applied_relative: relative(applied_squared),
        position_rounding_l2_m: rounding_squared.sqrt(),
    }
}

fn report_residual_audit(backend: &str, recurrence: f32, audit: crate::scene::SolverResidualAudit) {
    eprintln!("PCG residual audit: backend={backend} reported={recurrence:.9e} true={:.9e} applied={:.9e} position_rounding_l2_m={:.9e}",
        audit.correction_relative, audit.applied_relative, audit.position_rounding_l2_m);
}

fn worst_audited_residual(previous: Option<f64>, audit: crate::scene::SolverResidualAudit) -> f64 {
    if !audit.correction_relative.is_finite() || !audit.applied_relative.is_finite() {
        f64::INFINITY
    } else {
        previous
            .unwrap_or(0.0)
            .max(audit.correction_relative)
            .max(audit.applied_relative)
    }
}

#[cfg(test)]
mod residual_audit_tests {
    use super::*;

    #[test]
    fn shared_symmetric_edge_block_matches_both_projected_row_orientations() {
        let raw = Mat3::from_cols(
            Vec3::new(7.0, 1.5, -0.25),
            Vec3::new(0.5, 5.0, 0.75),
            Vec3::new(-1.25, 0.25, 3.0),
        );
        let k = (raw + raw.transpose()) * 0.5;
        let pi = Mat3::from_diagonal(Vec3::new(1.0, 0.0, 1.0));
        let pj = Mat3::from_diagonal(Vec3::new(0.0, 1.0, 1.0));
        let forward = pack_pcg_block(-pi * k * pj);
        let reverse = pack_pcg_block(-pj * k * pi);
        let shared_reverse = transpose_pcg_block(forward);
        for index in 0..9 {
            assert!((reverse[index] - shared_reverse[index]).abs() <= 1e-6);
        }
        let x = Vec3::new(0.25, -1.5, 2.0);
        assert!(
            (apply_pcg_block(shared_reverse, x) - apply_pcg_block(reverse, x)).length() <= 1e-6
        );
    }

    #[test]
    fn independent_audit_distinguishes_solved_vector_from_rounded_coordinates() {
        let diagonal = [
            Mat3::from_diagonal(Vec3::new(4.0, 5.0, 6.0)),
            Mat3::from_diagonal(Vec3::new(3.0, 4.0, 5.0)),
        ];
        let mut rows = [PcgRow::default(); 2];
        let edge_blocks = [pack_pcg_block(-Mat3::IDENTITY)];
        rows[0].neighbors[0] = 1;
        rows[1].neighbors[0] = 0;
        rows[0].edge_ids[0] = 0;
        rows[1].edge_ids[0] = 0;
        let rhs = [Vec3::new(0.0, 1.0, 2.875), Vec3::new(1.375, 0.75, 0.125)];
        let dx = [Vec3::new(0.125, 0.25, 0.5), Vec3::new(0.5, 0.25, 0.125)];
        let local = audit_pcg_residual(
            &diagonal,
            &rows,
            &edge_blocks,
            &[0, 1],
            &[1, 1],
            &rhs,
            &dx,
            &[DVec3::ZERO; 2],
        );
        assert_eq!(local.correction_relative, 0.0);
        assert_eq!(local.applied_relative, 0.0);
        assert_eq!(local.position_rounding_l2_m, 0.0);
        let far = audit_pcg_residual(
            &diagonal,
            &rows,
            &edge_blocks,
            &[0, 1],
            &[1, 1],
            &rhs,
            &dx,
            &[DVec3::splat(1e18); 2],
        );
        assert_eq!(far.correction_relative, 0.0);
        assert_eq!(far.applied_relative, 1.0);
        assert!(far.position_rounding_l2_m > 0.0);
    }

    #[test]
    fn independent_audit_handles_empty_system_without_nan() {
        let audit = audit_pcg_residual(&[], &[], &[], &[], &[], &[], &[], &[]);
        assert_eq!(audit.correction_relative, 0.0);
        assert_eq!(audit.applied_relative, 0.0);
        assert_eq!(audit.position_rounding_l2_m, 0.0);
    }

    #[test]
    fn earlier_bad_solve_is_not_hidden_by_a_later_good_solve() {
        let audit = |value| crate::scene::SolverResidualAudit {
            correction_relative: value,
            applied_relative: value,
            position_rounding_l2_m: 0.0,
        };
        let worst = worst_audited_residual(None, audit(0.2));
        assert_eq!(worst_audited_residual(Some(worst), audit(1e-6)), 0.2);
        assert_eq!(
            worst_audited_residual(Some(worst), audit(f64::NAN)),
            f64::INFINITY
        );
    }

    #[test]
    fn physical_elastic_correction_smaller_than_render_ulp_survives_application() {
        use crate::{
            material::Material,
            scene::{SceneConfig, VoxelSpawn},
        };
        let iron = Material::component_id("iron").unwrap();
        let mut config = SceneConfig::default();
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        config.fixed_dt = 0.001;
        config.substeps = 1;
        config.corotation_iterations = 1;
        config.iterations = 256;
        let mut scene = Scene::from_voxels(
            config,
            vec![
                VoxelSpawn::at_cell([0, 0, 0], iron),
                VoxelSpawn::at_cell([1, 0, 0], iron),
            ],
        )
        .unwrap();
        scene.fix_voxels(&[0]).unwrap();
        assert_eq!(
            scene.bonds[0][0].neighbor, 1,
            "the elastic test needs a retained bond to its support"
        );
        let start = scene.positions[1];
        let force = 0.01_f32;
        let stiffness = stiffness_matrix(&scene, 0, 1, Mat3::IDENTITY).x_axis.x as f64;
        let expected = force as f64 / (scene.voxels[1].mass as f64 / 0.001_f64.powi(2) + stiffness);
        set_force(&mut scene, 1, Vec3::X * force).unwrap();
        step(&mut scene, &mut None);
        let displacement = scene.positions[1].x - start.x;
        assert!(displacement > 0.0);
        assert_eq!(
            scene.positions[1].as_vec3(),
            start.as_vec3(),
            "render projection may round the motion"
        );
        assert!(
            (displacement - expected).abs() / expected < 0.005,
            "actual={displacement:.12e} expected={expected:.12e}"
        );
        assert_eq!(scene.broken, 0);
    }
}

#[derive(Default)]
pub(crate) struct SolverScratch {
    pair_contacts: Vec<crate::contact_constraint::PairContact>,
    adaptive_gravity: Option<Vec3>,
    adaptive_energies: Vec<f32>,
    adaptive_loads: Vec<Vec3>,
    adaptive_positions: Vec<DVec3>,
    adaptive_config: Vec<u8>,
    adaptive_revision: Option<u64>,
    merged_targets: Vec<[Option<f64>; 3]>,
    contact_targets: Vec<[Option<f64>; 3]>,
    cell_of: Vec<u32>,
    atmosphere_exposed_faces: Vec<[u8; 3]>,
    ambient_loads: Vec<Vec3>,
    applied_loads: Vec<Vec3>,
    substep_start_positions: Vec<DVec3>,
    substep_start_velocities: Vec<Vec3>,
    substep_mechanical_energy: Vec<f64>,
    energy_edge_values: Vec<f64>,
    free_implicit_budgets: Vec<FreeImplicitEnergyBudget>,
    free_implicit_components: Vec<usize>,
    free_implicit_heat: Vec<f64>,
    load_integrated_velocities: Vec<Vec3>,
    contact_work_mode: Vec<u8>,
    tool_contact_velocity: Vec<Vec3>,
    drift_velocity: Vec<Vec3>,
    predicted: Vec<DVec3>,
    solved_positions: Vec<DVec3>,
    pre_contact_velocities: Vec<Vec3>,
    contacts: Vec<collision::Contact>,
    impact_energies_j: Vec<f64>,
    melted: Vec<bool>,
    normal_impulses_ns: Vec<f32>,
    contact_physical_velocities: Vec<Vec3>,
    free: Vec<Vec3>,
    active: Vec<usize>,
    active_index: Vec<u32>,
    banded_factor: Option<crate::banded::BandedFactor>,
    banded_factor_free: Vec<Vec3>,
    banded_factor_pool: Vec<(Vec<Vec3>, crate::banded::BandedFactor)>,
    banded_diagonal: Vec<Mat3>,
    banded_lower: Vec<(usize, usize, Mat3)>,
    banded_rhs: Vec<Vec3>,
    banded_dx: Vec<DVec3>,
    banded_residual: Vec<DVec3>,
    banded_correction: Vec<DVec3>,
    contact_axis_normals: Vec<[f32; 3]>,
    previous_contact_axis_normals: Vec<[f32; 3]>,
    contact_lower: Vec<[Option<f64>; 3]>,
    contact_upper: Vec<[Option<f64>; 3]>,
    contact_feasible_positions: Vec<DVec3>,
    contact_reactions: Vec<DVec3>,
    masses: Vec<f32>,
    incidence: Vec<[u32; 6]>,
    incidence_len: Vec<u8>,
    diagonal: Vec<Mat3>,
    factor_matrix: Vec<Mat3>,
    factor_diagonal: Vec<Mat3>,
    factor_diagonal_inverse: Vec<Mat3>,
    factor_edges: Vec<Mat3>,
    ic0_lower_links: Vec<[[u32; 2]; 6]>,
    ic0_lower_link_count: Vec<u8>,
    ic0_upper_links: Vec<[[u32; 2]; 6]>,
    ic0_upper_link_count: Vec<u8>,
    ic0_colors: Vec<u8>,
    ic0_red_slots: Vec<usize>,
    ic0_black_slots: Vec<usize>,
    ic0_color_stack: Vec<usize>,
    ic0_natural_active: Vec<usize>,
    ic0_component_parent: Vec<u32>,
    ic0_component_root_to_id: Vec<u32>,
    ic0_component_sizes: Vec<usize>,
    ic0_component_offsets: Vec<usize>,
    ic0_component_members: Vec<usize>,
    ic0_component_ranges: Vec<(usize, usize)>,
    ic0_component_batches: Vec<(usize, usize, usize, usize)>,
    cpu_preconditioner_bipartite: bool,
    cpu_preconditioner_ready: bool,
    cpu_preconditioner_components: bool,
    cpu_preconditioner_free: Vec<Vec3>,
    cpu_preconditioner_graph: Vec<(usize, usize, usize)>,
    cpu_preconditioner_kind: Option<CpuPreconditioner>,
    multigrid: Option<crate::multigrid::Multigrid>,
    preconditioner_work: Vec<Vec3>,
    rhs: Vec<Vec3>,
    constraints: Vec<Constraint>,
    assembly_phase_strengths: Vec<(f32, f32)>,
    implicit_friction_contacts: Vec<bool>,
    support_friction_modes: rustc_hash::FxHashMap<usize, SupportFriction>,
    support_patch_normal: rustc_hash::FxHashMap<usize, f64>,
    solve_loads: Vec<Vec3>,
    friction_used_ns: Vec<f32>,
    substep_theta: Vec<DVec3>,
    spin_at_substep_start: Vec<DVec3>,
    cell_factor: Option<CellFactor>,
    cell_factor_iterations: usize,
    two_level: Option<crate::cell_two_level::TwoLevel>,
    two_level_work: crate::cell_two_level::TwoLevelWork,
    last_force_residual_n: f64,
    mechanical_mask: Vec<bool>,
    mechanical_revs: Vec<usize>,
    mechanical_slots: Vec<usize>,
    fracture_phase_strengths: Vec<(f32, f32)>,
    fracture_outcomes: Vec<Option<FractureOutcome>>,
    assembly_phases: Vec<crate::material::PhaseState>,
    stiffness_inputs: Vec<StiffnessInput>,
    stiffness_revisions: Vec<u64>,
    constraint_slots: Vec<Option<Constraint>>,
    constraint_forces: Vec<Vec3>,
    pcg_rows: Vec<PcgRow>,
    pcg_edge_blocks: Vec<[f32; 9]>,
    cuda_neighbors: Vec<[u32; 6]>,
    cuda_edge_ids: Vec<[u32; 6]>,
    cuda_transpose_masks: Vec<u8>,
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
#[cfg(test)]
thread_local! {
    static TEST_FAILED_SOLVES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn inject_test_solve_failure(scene: &mut Scene) {
    TEST_FAILED_SOLVES.with(|remaining| {
        if remaining.get() > 0 {
            remaining.set(remaining.get() - 1);
            scene.solver_residual = f32::INFINITY;
        }
    });
}

/// Two free iron cubes colliding: the cell solver, with contacts between free
/// bodies handled in the 6-DOF system, must agree with the translation-only solver
/// on a collision that has no torque, and conserve momentum without creating energy.
#[cfg(test)]
#[test]
fn colliding_cubes_agree_between_cell_and_translation_solvers() {
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    let iron = Material::component_id("iron").unwrap();
    let run = |cells: bool| {
        FORCE_CELL_DOFS.with(|force| force.set(Some(cells)));
        let mut config = SceneConfig::default();
        config.fixed_dt = 0.01;
        config.substeps = 4;
        config.gravity = [0.0; 3];
        config.atmosphere.pressure_pa = 0.0;
        let mut spawns = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], iron).with_velocity(Vec3::new(0.4, 0.0, 0.0)));
                    spawns.push(VoxelSpawn::at_cell([x + 4, y, z], iron).with_velocity(Vec3::new(-0.4, 0.0, 0.0)));
                }
            }
        }
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        scene.adaptive_substeps = false;
        let momentum_before: Vec3 = scene.voxels.iter().map(|v| v.velocity * v.mass).sum();
        let mass: f32 = scene.voxels.iter().map(|v| v.mass).sum();
        for _ in 0..40 {
            step(&mut scene, &mut None);
            assert!(scene.solver_residual.is_finite());
        }
        FORCE_CELL_DOFS.with(|force| force.set(None));
        let momentum: Vec3 = scene.voxels.iter().map(|v| v.velocity * v.mass).sum();
        let kinetic: f64 = scene
            .voxels
            .iter()
            .map(|v| 0.5 * f64::from(v.mass) * f64::from(v.velocity.length_squared()))
            .sum();
        let heat: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
        let centre = scene.positions.iter().copied().sum::<DVec3>() / scene.positions.len() as f64;
        let overlap = scene
            .positions
            .iter()
            .enumerate()
            .flat_map(|(i, a)| scene.positions.iter().skip(i + 1).map(move |b| (*a - *b).length()))
            .fold(f64::INFINITY, f64::min);
        (momentum_before, momentum, mass, kinetic, heat, centre, overlap)
    };
    let (before, after_cells, mass, kinetic_cells, _, centre_cells, overlap_cells) = run(true);
    let (_, after_old, _, kinetic_old, _, centre_old, _) = run(false);
    // Momentum is conserved by both paths.
    for after in [after_cells, after_old] {
        assert!((after - before).length() < 1.0e-4 * mass * 0.4, "momentum {before:?} -> {after:?}");
    }
    // No cell centres end closer than a REV: the contact removed the overlap.
    assert!(overlap_cells > 0.0099, "cells interpenetrated: {overlap_cells}");
    // Kinetic energy does not rise, and the two solvers agree within the 10% tolerance.
    let initial: f64 = 0.5 * f64::from(mass) * 0.4 * 0.4;
    assert!(kinetic_cells <= initial * 1.0001, "cells created energy: {kinetic_cells} > {initial}");
    assert!(
        (kinetic_cells - kinetic_old).abs() <= 0.1 * initial,
        "kinetic energy differs: cells {kinetic_cells}, translation {kinetic_old}"
    );
    assert!((centre_cells - centre_old).length() < 1.0e-3, "centre {centre_cells:?} vs {centre_old:?}");
}

#[cfg(test)]
#[test]
fn failed_solve_in_one_substep_stays_visible_after_later_substeps_succeed() {
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    let oak = Material::component_id("oak").unwrap();
    let mut config = SceneConfig::default();
    config.substeps = 4;
    let spawns = vec![
        VoxelSpawn::at_cell([0, 0, 0], oak),
        VoxelSpawn::at_cell([1, 0, 0], oak),
    ];
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    scene.adaptive_substeps = false;

    step(&mut scene, &mut None);
    assert!(scene.solver_residual.is_finite(), "control step must succeed");

    TEST_FAILED_SOLVES.with(|remaining| remaining.set(1));
    step(&mut scene, &mut None);
    TEST_FAILED_SOLVES.with(|remaining| remaining.set(0));
    assert_eq!(scene.last_mechanical_substeps, 4);
    assert!(
        !scene.solver_residual.is_finite(),
        "failure masked by a later substep: residual={}",
        scene.solver_residual
    );
    // Diagnostics exist only while profiling or runtime metrics are active.
    if let Some(status) = scene.last_step_diagnostics.as_ref() {
        assert!(!status.relative_solver_residual.is_finite());
    }
}

#[cfg(test)]
#[test]
fn releasing_a_fixed_rev_restores_its_bonds_to_the_next_assembly() {
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    let oak = Material::component_id("oak").unwrap();
    let mut config = SceneConfig::default();
    config.substeps = 1;
    let spawns = vec![
        VoxelSpawn::at_cell([0, 0, 0], oak),
        VoxelSpawn::at_cell([1, 0, 0], oak),
        VoxelSpawn::at_cell([2, 0, 0], oak),
    ];
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    scene.adaptive_substeps = false;
    scene.fix_voxels(&[0, 1]).unwrap();

    step(&mut scene, &mut None);
    assert_eq!(scene.solver_scratch.constraints.len(), 1, "fixed pair has no constraint");
    assert_eq!(scene.solver_scratch.mechanical_revs, vec![1, 2]);

    scene.set_voxel_kinematic(1, false).unwrap();
    step(&mut scene, &mut None);
    assert_eq!(scene.solver_scratch.constraints.len(), 2, "released REV regains its bond");
    assert_eq!(scene.solver_scratch.mechanical_revs, vec![0, 1, 2]);
    assert!(scene.solver_residual.is_finite());
}

#[cfg(test)]
#[test]
fn mobile_elastic_energy_differs_from_full_energy_only_by_fixed_pairs() {
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    let oak = Material::component_id("oak").unwrap();
    let spawns = vec![
        VoxelSpawn::at_cell([0, 0, 0], oak),
        VoxelSpawn::at_cell([1, 0, 0], oak),
        VoxelSpawn::at_cell([2, 0, 0], oak),
    ];
    let mut scene = Scene::from_voxels(SceneConfig::default(), spawns).unwrap();
    scene.fix_voxels(&[0, 1]).unwrap();
    let fixed_pair = |scene: &Scene| bond_stored_elastic_energy_j(scene, 0, 1, 0);

    scene.positions[2].x += 1.0e-5;
    let stretched = scene.mobile_elastic_bond_energy_j();
    assert!(stretched > 0.0);
    let expected = scene.elastic_bond_energy_j() - fixed_pair(&scene);
    assert!((stretched - expected).abs() <= expected.abs() * 1.0e-12);

    let (full_before, mobile_before) =
        (scene.elastic_bond_energy_j(), scene.mobile_elastic_bond_energy_j());
    scene.positions[2].x += 1.0e-5;
    let full_change = scene.elastic_bond_energy_j() - full_before;
    let mobile_change = scene.mobile_elastic_bond_energy_j() - mobile_before;
    assert!(full_change > 0.0);
    assert!((full_change - mobile_change).abs() <= full_change * 1.0e-12);
}

#[cfg(test)]
#[test]
fn natural_ic0_handles_bonds_whose_owner_index_follows_the_neighbor() {
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    let oak = Material::component_id("oak").unwrap();
    // Deliberately reverse spatial order: the positive-axis bond is owned by
    // the larger insertion index, which is valid for saved voxel scenes.
    let mut config = SceneConfig::default();
    config.substeps = 1;
    config.gravity = [0.0; 3];
    config.atmosphere.pressure_pa = 0.0;
    let run = |spawns| {
        let mut scene = Scene::from_voxels(config.clone(), spawns).unwrap();
        for _ in 0..8 {
            step(&mut scene, &mut None);
            assert!(scene.positions.iter().all(|value| value.is_finite()));
            assert!(scene.voxels.iter().all(|value| value.velocity.is_finite()));
            assert!(scene.solver_residual.is_finite());
        }
        scene
    };
    let forward = run(vec![
        VoxelSpawn::at_cell([0, 0, 0], oak),
        VoxelSpawn::at_cell([1, 0, 0], oak),
    ]);
    let reversed = run(vec![
        VoxelSpawn::at_cell([1, 0, 0], oak),
        VoxelSpawn::at_cell([0, 0, 0], oak),
    ]);
    assert!(reversed.bonds[0].iter().enumerate().any(|(i, bond)|
        bond.neighbor != EMPTY && (bond.neighbor as usize) < i));
    for (a, b) in [(forward.positions[0], reversed.positions[1]),
                   (forward.positions[1], reversed.positions[0])] {
        assert!(a.distance(b) < 1.0e-5, "voxel permutation changed trajectory: {a:?} vs {b:?}");
    }
    for (a, b) in [(forward.voxels[0].velocity, reversed.voxels[1].velocity),
                   (forward.voxels[1].velocity, reversed.voxels[0].velocity)] {
        assert!((a - b).length() < 1.0e-5, "voxel permutation changed velocity: {a:?} vs {b:?}");
    }
}

#[cfg(test)]
fn make_constraint(scene: &Scene, i: usize, axis: usize, j: usize) -> Constraint {
    let phases = [i, j].map(|n| {
        scene.properties[n].phase_state(scene.voxels[n].mass, scene.voxels[n].thermal_energy_f32())
    });
    make_constraint_at_phases(scene, i, axis, j, phases, [0; 2], None)
}

fn make_constraint_at_phases(
    scene: &Scene,
    i: usize,
    axis: usize,
    j: usize,
    phases: [crate::material::PhaseState; 2],
    stiffness_revisions: [u64; 2],
    previous: Option<Constraint>,
) -> Constraint {
    let rotation_i = scene.rotations[i];
    let rotation_j = scene.rotations[j];
    let thermal_energy_i = scene.voxels[i].thermal_energy_f32();
    let thermal_energy_j = scene.voxels[j].thermal_energy_f32();
    let cached = previous.filter(|old| {
        old.i == i
            && old.j == j
            && old.axis == axis
            && old.stiffness_revisions == stiffness_revisions
    });
    let (frame, material_k) = if let Some(old) = cached {
        (old.frame, old.material_k)
    } else {
        let frame = basis(axis, rotation_i.slerp(rotation_j, 0.5));
        let matrix = stiffness_from_compliances(
            tensor_compliance_at_solid_fraction(
                scene,
                i,
                frame,
                phases[0].solid_fraction.clamp(1e-4, 1.0),
            ),
            tensor_compliance_at_solid_fraction(
                scene,
                j,
                frame,
                phases[1].solid_fraction.clamp(1e-4, 1.0),
            ),
        );
        (frame, matrix)
    };
    let bond = scene.bonds[axis][i];
    let rest_input = RestInput {
        fiber_orientations: [i, j].map(|n| scene.voxels[n].microstructure.orientation),
        expansion: [i, j].map(|n| {
            let material = &scene.properties[n];
            [
                material.thermal_expansion_parallel,
                material.thermal_expansion_perpendicular,
            ]
        }),
        temperatures: [phases[0].temperature_kelvin, phases[1].temperature_kelvin],
        reference_temperatures: [
            scene.reference_temperatures_kelvin[i],
            scene.reference_temperatures_kelvin[j],
        ],
        formation_offset: bond.formation_offset,
        plastic_strain: bond.plastic_strain,
    };
    let rest = if scene.cell_dofs && scene.precise_rotations.len() == scene.rotations.len() {
        face_rest(
            scene,
            i,
            j,
            axis,
            [phases[0].temperature_kelvin, phases[1].temperature_kelvin],
        )
    } else {
        cached
        .filter(|old| {
            old.rotation_i == rotation_i
                && old.rotation_j == rotation_j
                && old.thermal_energy_i == thermal_energy_i
                && old.thermal_energy_j == thermal_energy_j
                && old.rest_input == rest_input
        })
        .map(|old| old.rest)
        .unwrap_or_else(|| {
            frame.as_dmat3()
                * (bond_thermal_rest_at_temperatures(
                    scene,
                    i,
                    j,
                    frame,
                    [phases[0].temperature_kelvin, phases[1].temperature_kelvin],
                ) + DVec3::from_array(bond.formation_offset.map(f64::from))
                    + DVec3::from_array(bond.plastic_strain.map(f64::from)) * DCELL)
        })
    };
    let k = cached
        .filter(|old| old.damage == bond.damage)
        .map(|old| old.k)
        .unwrap_or_else(|| {
            let local = material_k * (1. - bond.damage).max(0.001);
            let k = frame * local * frame.transpose();
            // The constitutive tensor is symmetric by construction. Remove
            // roundoff asymmetry once so the shared edge block can represent
            // its reverse row by an exact transpose.
            (k + k.transpose()) * 0.5
        });
    Constraint {
        i,
        j,
        axis,
        frame,
        material_k,
        k,
        rest,
        rotation_i,
        rotation_j,
        thermal_energy_i,
        thermal_energy_j,
        damage: bond.damage,
        stiffness_revisions,
        rest_input,
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

fn try_cholesky_lower(m: Mat3) -> Option<Mat3> {
    let d0 = m.x_axis.x;
    if !d0.is_finite() || d0 <= 1e-20 {
        return None;
    }
    let l00 = d0.sqrt();
    let l10 = m.x_axis.y / l00;
    let l20 = m.x_axis.z / l00;
    let d1 = m.y_axis.y - l10 * l10;
    if !d1.is_finite() || d1 <= 1e-20 {
        return None;
    }
    let l11 = d1.sqrt();
    let l21 = (m.y_axis.z - l20 * l10) / l11;
    let d2 = m.z_axis.z - l20 * l20 - l21 * l21;
    if !d2.is_finite() || d2 <= 1e-20 {
        return None;
    }
    let l22 = d2.sqrt();
    let lower = Mat3::from_cols(
        Vec3::new(l00, l10, l20),
        Vec3::new(0.0, l11, l21),
        Vec3::new(0.0, 0.0, l22),
    );
    (lower.x_axis.is_finite() && lower.y_axis.is_finite() && lower.z_axis.is_finite())
        .then_some(lower)
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

fn checked_relative_residual(residual_norm: f64, rhs_norm: f64) -> f32 {
    if !residual_norm.is_finite() || !rhs_norm.is_finite() || residual_norm < 0.0 || rhs_norm < 0.0 {
        f32::INFINITY
    } else if rhs_norm == 0.0 {
        if residual_norm == 0.0 { 0.0 } else { f32::INFINITY }
    } else {
        (residual_norm / rhs_norm) as f32
    }
}

fn inverse_cholesky_lower(l: Mat3) -> Mat3 {
    Mat3::from_cols(
        solve_lower(l, Vec3::X),
        solve_lower(l, Vec3::Y),
        solve_lower(l, Vec3::Z),
    )
}

fn build_ic0_diagonal_inverses(factor_diagonal: &[Mat3], inverses: &mut Vec<Mat3>) {
    inverses.resize(factor_diagonal.len(), Mat3::IDENTITY);
    for (inverse, &factor) in inverses.iter_mut().zip(factor_diagonal) {
        *inverse = inverse_cholesky_lower(factor);
    }
}

// Cache the exact lower/upper traversal used by apply_ic0. Each axis-aligned
// voxel has at most one incoming and one outgoing edge per axis, so each half
// has at most three links. Keeping the existing incidence order preserves the
// floating-point operation order while avoiding repeated graph branches and
// active-index lookups on every PCG iteration.
fn build_ic0_links(
    constraints: &[Constraint],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    active: &[usize],
    active_index: &[u32],
    lower_links: &mut Vec<[[u32; 2]; 6]>,
    lower_link_count: &mut Vec<u8>,
    upper_links: &mut Vec<[[u32; 2]; 6]>,
    upper_link_count: &mut Vec<u8>,
) {
    lower_links.resize(active.len(), [[u32::MAX; 2]; 6]);
    upper_links.resize(active.len(), [[u32::MAX; 2]; 6]);
    lower_link_count.resize(active.len(), 0);
    upper_link_count.resize(active.len(), 0);
    lower_link_count.fill(0);
    upper_link_count.fill(0);

    for (slot, &i) in active.iter().enumerate() {
        for &edge in &incidence[i][..incidence_len[i] as usize] {
            let edge = edge as usize;
            let constraint = &constraints[edge];
            let other = if constraint.i == i { constraint.j } else { constraint.i };
            let neighbor = active_index[other];
            if neighbor != u32::MAX {
                if neighbor < slot as u32 {
                    let count = &mut lower_link_count[slot];
                    debug_assert!((*count as usize) < lower_links[slot].len());
                    lower_links[slot][*count as usize] = [edge as u32, neighbor];
                    *count += 1;
                } else {
                    let count = &mut upper_link_count[slot];
                    debug_assert!((*count as usize) < upper_links[slot].len());
                    upper_links[slot][*count as usize] = [edge as u32, neighbor];
                    *count += 1;
                }
            }
        }
    }
}

fn ic0_component_root(parent: &mut [u32], mut node: u32) -> u32 {
    while parent[node as usize] != node {
        let parent_node = parent[node as usize];
        parent[node as usize] = parent[parent_node as usize];
        node = parent[node as usize];
    }
    node
}

// Keep each active matrix component contiguous while preserving the original
// REV order inside it. Stable intra-component order leaves IC(0)'s triangular
// arithmetic unchanged; only independent components can move relative to one
// another. Constraints touching an inactive endpoint do not couple PCG rows.
fn order_active_by_ic0_components(
    constraints: &[Constraint],
    active: &mut Vec<usize>,
    active_index: &mut [u32],
    parent: &mut Vec<u32>,
    root_to_id: &mut Vec<u32>,
    sizes: &mut Vec<usize>,
    offsets: &mut Vec<usize>,
    members: &mut Vec<usize>,
    ranges: &mut Vec<(usize, usize)>,
) -> usize {
    let active_len = active.len();
    parent.resize(active_len, 0);
    for (slot, node) in parent.iter_mut().enumerate() {
        *node = slot as u32;
    }

    for constraint in constraints {
        let a = active_index[constraint.i];
        let b = active_index[constraint.j];
        if a == u32::MAX || b == u32::MAX {
            continue;
        }
        let root_a = ic0_component_root(parent, a);
        let root_b = ic0_component_root(parent, b);
        if root_a != root_b {
            parent[root_b as usize] = root_a;
        }
    }

    root_to_id.resize(active_len, u32::MAX);
    root_to_id.fill(u32::MAX);
    sizes.clear();
    for slot in 0..active_len {
        let root = ic0_component_root(parent, slot as u32) as usize;
        let mut id = root_to_id[root];
        if id == u32::MAX {
            id = sizes.len() as u32;
            root_to_id[root] = id;
            sizes.push(0);
        }
        sizes[id as usize] += 1;
    }

    let component_count = sizes.len();
    offsets.resize(component_count + 1, 0);
    for id in 0..component_count {
        offsets[id + 1] = offsets[id] + sizes[id];
    }
    ranges.clear();
    ranges.extend((0..component_count).map(|id| (offsets[id], offsets[id + 1])));

    // Reuse the component counts as stable scatter cursors after preserving
    // the final component boundaries above.
    sizes.copy_from_slice(&offsets[..component_count]);
    members.resize(active_len, 0);
    for slot in 0..active_len {
        let root = ic0_component_root(parent, slot as u32) as usize;
        let id = root_to_id[root] as usize;
        let destination = sizes[id];
        members[destination] = active[slot];
        sizes[id] += 1;
    }
    active.copy_from_slice(members);
    active_index.fill(u32::MAX);
    for (slot, &node) in active.iter().enumerate() {
        active_index[node] = slot as u32;
    }
    component_count
}

// Coalesce small components into a bounded number of contiguous worker slices.
// This avoids scheduling one Rayon task per voxel when a scene has many islands.
fn build_ic0_component_batches(
    ranges: &[(usize, usize)],
    active_len: usize,
    batches: &mut Vec<(usize, usize, usize, usize)>,
) {
    batches.clear();
    if ranges.len() < 2 {
        return;
    }
    let target_batches = rayon::current_num_threads()
        .saturating_mul(4)
        .max(1)
        .min(ranges.len());
    let target_rows = active_len.div_ceil(target_batches);
    let mut component_start = 0;
    while component_start < ranges.len() {
        let remaining_batches = target_batches - batches.len();
        let max_component_end = ranges.len() - (remaining_batches - 1);
        let row_start = ranges[component_start].0;
        let mut component_end = component_start + 1;
        while component_end < max_component_end
            && ranges[component_end - 1].1 - row_start < target_rows
        {
            component_end += 1;
        }
        let row_end = ranges[component_end - 1].1;
        batches.push((row_start, row_end, component_start, component_end));
        component_start = component_end;
    }
}

fn should_rebuild_cpu_preconditioner(
    reuse_preconditioner: bool,
    same_graph: bool,
    bipartite_graph: bool,
    cached_bipartite_graph: bool,
) -> bool {
    !reuse_preconditioner || !same_graph || bipartite_graph || cached_bipartite_graph
}

fn split_ic0_work_by_batches<'a>(
    values: &'a mut [Vec3],
    batches: &[(usize, usize, usize, usize)],
) -> Vec<&'a mut [Vec3]> {
    let total_len = values.len();
    let mut rest = values;
    let mut previous_end = 0;
    let mut chunks = Vec::with_capacity(batches.len());
    for &(start, end, _, _) in batches {
        debug_assert_eq!(start, previous_end);
        let (chunk, tail) = rest.split_at_mut(end - start);
        chunks.push(chunk);
        rest = tail;
        previous_end = end;
    }
    debug_assert_eq!(previous_end, total_len);
    chunks
}

fn split_mut_by_ranges<'a, T>(values: &'a mut [T], ranges: &[(usize, usize)]) -> Vec<&'a mut [T]> {
    let total_len = values.len();
    let mut rest = values;
    let mut previous_end = 0;
    let mut chunks = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        debug_assert_eq!(start, previous_end);
        let (chunk, tail) = rest.split_at_mut(end - start);
        chunks.push(chunk);
        rest = tail;
        previous_end = end;
    }
    debug_assert_eq!(previous_end, total_len);
    chunks
}

struct ComponentPcgState<'a> {
    start: usize,
    end: usize,
    dx: &'a mut [Vec3],
    ap: &'a mut [Vec3],
    r: &'a mut [Vec3],
    z: &'a mut [Vec3],
    p: &'a mut [Vec3],
    work: &'a mut [Vec3],
    iterations: usize,
    converged: bool,
}

#[inline]
fn component_dot(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.dot(*y) as f64).sum()
}

fn apply_pcg_component(
    start: usize,
    end: usize,
    active: &[usize],
    diagonal: &[Mat3],
    rows: &[PcgRow],
    edge_blocks: &[[f32; 9]],
    incidence_len: &[u8],
    input: &[Vec3],
    output: &mut [Vec3],
) {
    debug_assert_eq!(input.len(), end - start);
    debug_assert_eq!(output.len(), end - start);
    for (local, (value, &voxel)) in output.iter_mut().zip(&active[start..end]).enumerate() {
        let slot = start + local;
        let row = &rows[slot];
        let mut result = diagonal[voxel] * input[local];
        for edge_slot in 0..incidence_len[voxel] as usize {
            let neighbor = row.neighbors[edge_slot];
            if neighbor == u32::MAX {
                continue;
            }
            let neighbor = neighbor as usize;
            debug_assert!((start..end).contains(&neighbor));
            let block = edge_blocks[row.edge_ids[edge_slot] as usize];
            result += if row.transpose_mask & (1 << edge_slot) != 0 {
                apply_pcg_block_transpose(block, input[neighbor - start])
            } else {
                apply_pcg_block(block, input[neighbor - start])
            };
        }
        *value = result;
    }
}

fn apply_ic0_component(
    start: usize,
    active: &[usize],
    factor_diagonal_inverse: &[Mat3],
    factor_edges: &[Mat3],
    lower_links: &[[[u32; 2]; 6]],
    lower_link_count: &[u8],
    upper_links: &[[[u32; 2]; 6]],
    upper_link_count: &[u8],
    rhs: &[Vec3],
    work: &mut [Vec3],
    output: &mut [Vec3],
) {
    debug_assert_eq!(rhs.len(), work.len());
    debug_assert_eq!(rhs.len(), output.len());
    let end = start + rhs.len();
    for local in 0..rhs.len() {
        let slot = start + local;
        let mut value = rhs[local];
        for &[edge, neighbor] in &lower_links[slot][..lower_link_count[slot] as usize] {
            let neighbor = neighbor as usize;
            debug_assert!((start..end).contains(&neighbor));
            value -= factor_edges[edge as usize] * work[neighbor - start];
        }
        work[local] = factor_diagonal_inverse[active[slot]] * value;
    }
    for local in (0..rhs.len()).rev() {
        let slot = start + local;
        let mut value = work[local];
        for &[edge, neighbor] in &upper_links[slot][..upper_link_count[slot] as usize] {
            let neighbor = neighbor as usize;
            debug_assert!((start..end).contains(&neighbor));
            value -= factor_edges[edge as usize].transpose() * output[neighbor - start];
        }
        output[local] = factor_diagonal_inverse[active[slot]].transpose() * value;
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_pcg_component(
    state: &mut ComponentPcgState<'_>,
    active: &[usize],
    diagonal: &[Mat3],
    rows: &[PcgRow],
    edge_blocks: &[[f32; 9]],
    incidence_len: &[u8],
    rhs: &[Vec3],
    factor_diagonal_inverse: &[Mat3],
    factor_edges: &[Mat3],
    lower_links: &[[[u32; 2]; 6]],
    lower_link_count: &[u8],
    upper_links: &[[[u32; 2]; 6]],
    upper_link_count: &[u8],
    relative_tolerance: f64,
    max_iterations: usize,
) {
    let len = state.end - state.start;
    debug_assert_eq!(state.dx.len(), len);
    debug_assert_eq!(state.ap.len(), len);
    debug_assert_eq!(state.r.len(), len);
    debug_assert_eq!(state.z.len(), len);
    debug_assert_eq!(state.p.len(), len);
    debug_assert_eq!(state.work.len(), len);

    let rhs_norm = active[state.start..state.end]
        .iter()
        .map(|&voxel| rhs[voxel].dot(rhs[voxel]) as f64)
        .sum::<f64>()
        .sqrt();
    let mut residual_norm = state
        .r
        .iter()
        .map(|value| value.dot(*value) as f64)
        .sum::<f64>()
        .sqrt();
    if residual_norm > rhs_norm {
        state.dx.fill(Vec3::ZERO);
        for (local, &voxel) in active[state.start..state.end].iter().enumerate() {
            state.r[local] = rhs[voxel];
        }
        residual_norm = rhs_norm;
    }
    if rhs_norm <= 1e-12 || residual_norm / rhs_norm < relative_tolerance {
        state.converged = true;
        return;
    }

    apply_ic0_component(
        state.start,
        active,
        factor_diagonal_inverse,
        factor_edges,
        lower_links,
        lower_link_count,
        upper_links,
        upper_link_count,
        state.r,
        state.work,
        state.z,
    );
    state.p.copy_from_slice(state.z);
    let mut rz = component_dot(state.r, state.z);

    for _ in 0..max_iterations {
        apply_pcg_component(
            state.start,
            state.end,
            active,
            diagonal,
            rows,
            edge_blocks,
            incidence_len,
            state.p,
            state.ap,
        );
        let denominator = component_dot(state.p, state.ap);
        state.iterations += 1;
        if !denominator.is_finite() || denominator <= 1e-30 || !rz.is_finite() || rz.abs() <= 1e-300
        {
            break;
        }

        let alpha = (rz / denominator) as f32;
        for local in 0..len {
            state.dx[local] += alpha * state.p[local];
            state.r[local] -= alpha * state.ap[local];
        }
        residual_norm = state
            .r
            .iter()
            .map(|value| value.dot(*value) as f64)
            .sum::<f64>()
            .sqrt();

        if residual_norm / rhs_norm < relative_tolerance {
            apply_pcg_component(
                state.start,
                state.end,
                active,
                diagonal,
                rows,
                edge_blocks,
                incidence_len,
                state.dx,
                state.ap,
            );
            for (local, &voxel) in active[state.start..state.end].iter().enumerate() {
                state.r[local] = rhs[voxel] - state.ap[local];
            }
            residual_norm = state
                .r
                .iter()
                .map(|value| value.dot(*value) as f64)
                .sum::<f64>()
                .sqrt();
            if residual_norm / rhs_norm < relative_tolerance {
                state.converged = true;
                break;
            }

            apply_ic0_component(
                state.start,
                active,
                factor_diagonal_inverse,
                factor_edges,
                lower_links,
                lower_link_count,
                upper_links,
                upper_link_count,
                state.r,
                state.work,
                state.z,
            );
            state.p.copy_from_slice(state.z);
            rz = component_dot(state.r, state.z);
            continue;
        }

        apply_ic0_component(
            state.start,
            active,
            factor_diagonal_inverse,
            factor_edges,
            lower_links,
            lower_link_count,
            upper_links,
            upper_link_count,
            state.r,
            state.work,
            state.z,
        );
        let next_rz = component_dot(state.r, state.z);
        if !next_rz.is_finite() || !rz.is_finite() || rz.abs() <= 1e-300 {
            break;
        }
        let beta = (next_rz / rz) as f32;
        for local in 0..len {
            state.p[local] = state.z[local] + beta * state.p[local];
        }
        rz = next_rz;
    }

    if !state.converged {
        apply_pcg_component(
            state.start,
            state.end,
            active,
            diagonal,
            rows,
            edge_blocks,
            incidence_len,
            state.dx,
            state.ap,
        );
        for (local, &voxel) in active[state.start..state.end].iter().enumerate() {
            state.r[local] = rhs[voxel] - state.ap[local];
        }
        let true_residual = state
            .r
            .iter()
            .map(|value| value.dot(*value) as f64)
            .sum::<f64>()
            .sqrt();
        state.converged = true_residual / rhs_norm < relative_tolerance;
    }
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
fn build_ic0(
    scene: &Scene,
    free: &[Vec3],
    diagonal: &[Mat3],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    constraints: &[Constraint],
    factor_matrix: &mut Vec<Mat3>,
    factor_diagonal: &mut Vec<Mat3>,
    factor_edges: &mut Vec<Mat3>,
) -> bool {
    let n = scene.voxels.len();
    factor_matrix.resize(n, Mat3::ZERO);
    factor_diagonal.resize(n, Mat3::IDENTITY);
    factor_edges.resize(constraints.len(), Mat3::ZERO);
    for i in 0..n {
        let projection = Mat3::from_diagonal(free[i]);
        factor_matrix[i] = projection * diagonal[i] * projection + Mat3::IDENTITY - projection;
    }
    for row in 0..n {
        if free[row] == Vec3::ZERO {
            factor_diagonal[row] = Mat3::IDENTITY;
            continue;
        }
        let row_projection = Mat3::from_diagonal(free[row]);
        let row_edges = &incidence[row][..incidence_len[row] as usize];
        for &edge in row_edges {
            let edge = edge as usize;
            let c = &constraints[edge];
            let col = if c.i == row { c.j } else { c.i };
            if col >= row {
                continue;
            }
            let col_projection = Mat3::from_diagonal(free[col]);
            let row_to_col = if c.i == row { c.k } else { c.k.transpose() };
            let mut value = -row_projection * row_to_col * col_projection;
            for &row_shared in row_edges {
                let row_shared = row_shared as usize;
                let r = &constraints[row_shared];
                let shared = if r.i == row { r.j } else { r.i };
                if shared >= col {
                    continue;
                }
                for &col_shared in &incidence[col][..incidence_len[col] as usize] {
                    let col_shared = col_shared as usize;
                    let s = &constraints[col_shared];
                    let col_shared_node = if s.i == col { s.j } else { s.i };
                    if col_shared_node == shared {
                        value -= factor_edges[row_shared] * factor_edges[col_shared].transpose();
                        break;
                    }
                }
            }
            factor_edges[edge] = right_solve_lower_transpose(value, factor_diagonal[col]);
            factor_matrix[row] -= factor_edges[edge] * factor_edges[edge].transpose();
        }
        let Some(factor) = try_cholesky_lower(factor_matrix[row]) else {
            // IC(0) may break down even when the assembled fine operator is
            // SPD. A failed block pivot invalidates the entire triangular
            // factor; never floor it and continue with enormous coefficients.
            if std::env::var_os("VOXEL_PROFILE_STEP").is_some() {
                eprintln!("CPU IC(0) breakdown row={row} Schur_block={:?}; using block Jacobi", factor_matrix[row]);
            }
            build_block_jacobi(diagonal, factor_diagonal, &None);
            factor_edges.fill(Mat3::ZERO);
            return false;
        };
        factor_diagonal[row] = factor;
    }
    true
}

fn color_active_graph(
    active: &[usize],
    active_index: &[u32],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    constraints: &[Constraint],
    colors: &mut Vec<u8>,
    red_slots: &mut Vec<usize>,
    black_slots: &mut Vec<usize>,
    stack: &mut Vec<usize>,
) -> bool {
    const UNCOLORED: u8 = u8::MAX;
    colors.resize(active_index.len(), UNCOLORED);
    colors.fill(UNCOLORED);
    red_slots.clear();
    black_slots.clear();
    stack.clear();
    for &root in active {
        if colors[root] != UNCOLORED {
            continue;
        }
        colors[root] = 0;
        stack.push(root);
        while let Some(i) = stack.pop() {
            let opposite = 1 - colors[i];
            for &edge in &incidence[i][..incidence_len[i] as usize] {
                let constraint = &constraints[edge as usize];
                let neighbor = if constraint.i == i {
                    constraint.j
                } else {
                    constraint.i
                };
                if active_index[neighbor] == u32::MAX {
                    continue;
                }
                if colors[neighbor] == UNCOLORED {
                    colors[neighbor] = opposite;
                    stack.push(neighbor);
                } else if colors[neighbor] != opposite {
                    return false;
                }
            }
        }
    }
    for (slot, &i) in active.iter().enumerate() {
        if colors[i] == 0 {
            red_slots.push(slot);
        } else if colors[i] == 1 {
            black_slots.push(slot);
        } else {
            return false;
        }
    }
    true
}

fn order_active_by_color(
    active: &mut Vec<usize>,
    active_index: &mut [u32],
    colors: &[u8],
    red_slots: &mut Vec<usize>,
    black_slots: &mut Vec<usize>,
    natural_active: &mut Vec<usize>,
) {
    natural_active.clone_from(active);
    active.sort_by_key(|&i| colors[i]);
    active_index.fill(u32::MAX);
    let red_count = active.iter().take_while(|&&i| colors[i] == 0).count();
    for (slot, &i) in active.iter().enumerate() {
        active_index[i] = slot as u32;
    }
    red_slots.clear();
    red_slots.extend(0..red_count);
    black_slots.clear();
    black_slots.extend(red_count..active.len());
}

fn restore_natural_active_order(
    active: &mut Vec<usize>,
    active_index: &mut [u32],
    natural_active: &[usize],
    red_slots: &mut Vec<usize>,
    black_slots: &mut Vec<usize>,
) {
    active.clone_from_slice(natural_active);
    active_index.fill(u32::MAX);
    for (slot, &i) in active.iter().enumerate() {
        active_index[i] = slot as u32;
    }
    red_slots.clear();
    black_slots.clear();
}

fn build_ic0_bipartite(
    free: &[Vec3],
    diagonal: &[Mat3],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    constraints: &[Constraint],
    colors: &[u8],
    factor_matrix: &mut Vec<Mat3>,
    factor_diagonal: &mut Vec<Mat3>,
    factor_edges: &mut Vec<Mat3>,
) -> bool {
    let n = free.len();
    factor_matrix.resize(n, Mat3::ZERO);
    factor_diagonal.resize(n, Mat3::IDENTITY);
    factor_edges.resize(constraints.len(), Mat3::ZERO);
    factor_edges.fill(Mat3::ZERO);
    for i in 0..n {
        let projection = Mat3::from_diagonal(free[i]);
        factor_matrix[i] = projection * diagonal[i] * projection + Mat3::IDENTITY - projection;
    }
    // In a bipartite ordering all red rows have no lower neighbors, so their
    // block pivots are independent. Black rows then subtract their red
    // neighbors' contributions. The fine operator itself is not reordered.
    for color in 0..=1 {
        for row in 0..n {
            if colors[row] != color || free[row] == Vec3::ZERO {
                continue;
            }
            if color == 1 {
                let row_projection = Mat3::from_diagonal(free[row]);
                for &edge in &incidence[row][..incidence_len[row] as usize] {
                    let edge = edge as usize;
                    let constraint = &constraints[edge];
                    let neighbor = if constraint.i == row {
                        constraint.j
                    } else {
                        constraint.i
                    };
                    if colors[neighbor] != 0 || free[neighbor] == Vec3::ZERO {
                        continue;
                    }
                    let column_projection = Mat3::from_diagonal(free[neighbor]);
                    let value = -row_projection * constraint.k * column_projection;
                    let factor = right_solve_lower_transpose(value, factor_diagonal[neighbor]);
                    if !factor.x_axis.is_finite()
                        || !factor.y_axis.is_finite()
                        || !factor.z_axis.is_finite()
                    {
                        return false;
                    }
                    factor_edges[edge] = factor;
                    factor_matrix[row] -= factor * factor.transpose();
                }
            }
            let Some(factor) = try_cholesky_lower(factor_matrix[row]) else {
                return false;
            };
            factor_diagonal[row] = factor;
        }
    }
    true
}

fn build_ic0_bipartite_links(
    constraints: &[Constraint],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    active: &[usize],
    active_index: &[u32],
    colors: &[u8],
    lower_links: &mut Vec<[[u32; 2]; 6]>,
    lower_link_count: &mut Vec<u8>,
    upper_links: &mut Vec<[[u32; 2]; 6]>,
    upper_link_count: &mut Vec<u8>,
) {
    lower_links.resize(active.len(), [[u32::MAX; 2]; 6]);
    upper_links.resize(active.len(), [[u32::MAX; 2]; 6]);
    lower_link_count.resize(active.len(), 0);
    upper_link_count.resize(active.len(), 0);
    lower_link_count.fill(0);
    upper_link_count.fill(0);
    for (slot, &i) in active.iter().enumerate() {
        for &edge in &incidence[i][..incidence_len[i] as usize] {
            let edge = edge as usize;
            let constraint = &constraints[edge];
            let neighbor = if constraint.i == i {
                constraint.j
            } else {
                constraint.i
            };
            let neighbor_slot = active_index[neighbor];
            if neighbor_slot == u32::MAX {
                continue;
            }
            if colors[i] == 1 {
                let count = &mut lower_link_count[slot];
                debug_assert!((*count as usize) < lower_links[slot].len());
                lower_links[slot][*count as usize] = [edge as u32, neighbor_slot];
                *count += 1;
            } else {
                let count = &mut upper_link_count[slot];
                debug_assert!((*count as usize) < upper_links[slot].len());
                upper_links[slot][*count as usize] = [edge as u32, neighbor_slot];
                *count += 1;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_ic0_with_optional_bipartite(
    scene: &Scene,
    free: &[Vec3],
    diagonal: &[Mat3],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    constraints: &[Constraint],
    use_bipartite: bool,
    colors: &[u8],
    factor_matrix: &mut Vec<Mat3>,
    factor_diagonal: &mut Vec<Mat3>,
    factor_edges: &mut Vec<Mat3>,
) -> bool {
    if use_bipartite {
        let built = build_ic0_bipartite(
            free,
            diagonal,
            incidence,
            incidence_len,
            constraints,
            colors,
            factor_matrix,
            factor_diagonal,
            factor_edges,
        );
        if std::env::var_os("VOXEL_PROFILE_STEP").is_some() {
            eprintln!(
                "CPU IC(0) bipartite ordering: {}",
                if built {
                    "active"
                } else {
                    "factor rejected; using natural ordering"
                }
            );
        }
        if built {
            return true;
        }
    }
    let built = build_ic0(
        scene,
        free,
        diagonal,
        incidence,
        incidence_len,
        constraints,
        factor_matrix,
        factor_diagonal,
        factor_edges,
    );
    if !built && std::env::var_os("VOXEL_PROFILE_STEP").is_some() {
        eprintln!("CPU IC(0) natural ordering factor rejected; using block Jacobi");
    }
    false
}

fn build_block_jacobi(
    diagonal: &[Mat3],
    factor_diagonal: &mut Vec<Mat3>,
    live: &Option<std::sync::Arc<crate::scene::Live>>,
) {
    let n = diagonal.len();
    if live.is_some() && factor_diagonal.len() == n {
        // A sleeper's row is the identity, whose factor is the identity.
        for row in visit(live, n) {
            factor_diagonal[row] = cholesky_lower(diagonal[row]);
        }
        return;
    }
    factor_diagonal.clear();
    factor_diagonal.resize(n, Mat3::IDENTITY);
    for row in 0..n {
        factor_diagonal[row] = cholesky_lower(diagonal[row]);
    }
}

fn apply_ic0(
    factor_diagonal_inverse: &[Mat3],
    factor_edges: &[Mat3],
    active: &[usize],
    lower_links: &[[[u32; 2]; 6]],
    lower_link_count: &[u8],
    upper_links: &[[[u32; 2]; 6]],
    upper_link_count: &[u8],
    rhs: &[Vec3],
    work: &mut [Vec3],
    out: &mut [Vec3],
) {
    for (slot, &i) in active.iter().enumerate() {
        let mut value = rhs[slot];
        for &[edge, neighbor] in &lower_links[slot][..lower_link_count[slot] as usize] {
            value -= factor_edges[edge as usize] * work[neighbor as usize];
        }
        work[slot] = factor_diagonal_inverse[i] * value;
    }

    for slot in (0..active.len()).rev() {
        let i = active[slot];
        let mut value = work[slot];
        for &[edge, neighbor] in &upper_links[slot][..upper_link_count[slot] as usize] {
            value -= factor_edges[edge as usize].transpose() * out[neighbor as usize];
        }
        out[slot] = factor_diagonal_inverse[i].transpose() * value;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_ic0_component_batches(
    factor_diagonal_inverse: &[Mat3],
    factor_edges: &[Mat3],
    active: &[usize],
    ranges: &[(usize, usize)],
    batches: &[(usize, usize, usize, usize)],
    lower_links: &[[[u32; 2]; 6]],
    lower_link_count: &[u8],
    upper_links: &[[[u32; 2]; 6]],
    upper_link_count: &[u8],
    rhs: &[Vec3],
    work: &mut [Vec3],
    out: &mut [Vec3],
) {
    let work_chunks = split_ic0_work_by_batches(work, batches);
    let out_chunks = split_ic0_work_by_batches(out, batches);
    work_chunks
        .into_par_iter()
        .zip(out_chunks.into_par_iter())
        .zip(batches.par_iter())
        .for_each(
            |((work, out), &(batch_start, _, component_start, component_end))| {
                for component in component_start..component_end {
                    let (start, end) = ranges[component];
                    for slot in start..end {
                        let mut value = rhs[slot];
                        for &[edge, neighbor] in
                            &lower_links[slot][..lower_link_count[slot] as usize]
                        {
                            let neighbor = neighbor as usize;
                            debug_assert!((start..end).contains(&neighbor));
                            value -= factor_edges[edge as usize] * work[neighbor - batch_start];
                        }
                        work[slot - batch_start] = factor_diagonal_inverse[active[slot]] * value;
                    }
                    for slot in (start..end).rev() {
                        let mut value = work[slot - batch_start];
                        for &[edge, neighbor] in
                            &upper_links[slot][..upper_link_count[slot] as usize]
                        {
                            let neighbor = neighbor as usize;
                            debug_assert!((start..end).contains(&neighbor));
                            value -= factor_edges[edge as usize].transpose()
                                * out[neighbor - batch_start];
                        }
                        out[slot - batch_start] =
                            factor_diagonal_inverse[active[slot]].transpose() * value;
                    }
                }
            },
        );
}

fn apply_ic0_bipartite(
    factor_diagonal_inverse: &[Mat3],
    factor_edges: &[Mat3],
    active: &[usize],
    red_slots: &[usize],
    lower_links: &[[[u32; 2]; 6]],
    lower_link_count: &[u8],
    upper_links: &[[[u32; 2]; 6]],
    upper_link_count: &[u8],
    rhs: &[Vec3],
    work: &mut [Vec3],
    out: &mut [Vec3],
) {
    let red_count = red_slots.len();
    let (work_red, work_black) = work.split_at_mut(red_count);
    let (out_red, out_black) = out.split_at_mut(red_count);
    let solve_red_forward = |(slot, (value, &i)): (usize, (&mut Vec3, &usize))| {
        *value = factor_diagonal_inverse[i] * rhs[slot];
    };
    if parallel_enabled_for(active.len()) {
        work_red
            .par_iter_mut()
            .enumerate()
            .zip(active[..red_count].par_iter())
            .for_each(|((slot, value), i)| solve_red_forward((slot, (value, i))));
    } else {
        work_red
            .iter_mut()
            .enumerate()
            .zip(&active[..red_count])
            .for_each(|((slot, value), i)| solve_red_forward((slot, (value, i))));
    }
    let solve_black_forward = |(local, value): (usize, &mut Vec3)| {
        let slot = red_count + local;
        let i = active[slot];
        let mut value_out = rhs[slot];
        for &[edge, neighbor] in &lower_links[slot][..lower_link_count[slot] as usize] {
            value_out -= factor_edges[edge as usize] * work_red[neighbor as usize];
        }
        *value = factor_diagonal_inverse[i] * value_out;
    };
    if parallel_enabled_for(active.len()) {
        work_black
            .par_iter_mut()
            .enumerate()
            .for_each(solve_black_forward);
    } else {
        work_black
            .iter_mut()
            .enumerate()
            .for_each(solve_black_forward);
    }
    let solve_black_backward = |(local, value): (usize, &mut Vec3)| {
        let i = active[red_count + local];
        *value = factor_diagonal_inverse[i].transpose() * work_black[local];
    };
    if parallel_enabled_for(active.len()) {
        out_black
            .par_iter_mut()
            .enumerate()
            .for_each(solve_black_backward);
    } else {
        out_black
            .iter_mut()
            .enumerate()
            .for_each(solve_black_backward);
    }
    let solve_red_backward = |(slot, (value, &i)): (usize, (&mut Vec3, &usize))| {
        let mut value_out = work_red[slot];
        for &[edge, neighbor] in &upper_links[slot][..upper_link_count[slot] as usize] {
            value_out -=
                factor_edges[edge as usize].transpose() * out_black[neighbor as usize - red_count];
        }
        *value = factor_diagonal_inverse[i].transpose() * value_out;
    };
    if parallel_enabled_for(active.len()) {
        out_red
            .par_iter_mut()
            .enumerate()
            .zip(active[..red_count].par_iter())
            .for_each(|((slot, value), i)| solve_red_backward((slot, (value, i))));
    } else {
        out_red
            .iter_mut()
            .enumerate()
            .zip(&active[..red_count])
            .for_each(|((slot, value), i)| solve_red_backward((slot, (value, i))));
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_preconditioner(
    kind: CpuPreconditioner,
    bipartite_ic0: bool,
    multigrid: &mut Option<crate::multigrid::Multigrid>,
    factor_diagonal: &[Mat3],
    factor_diagonal_inverse: &[Mat3],
    factor_edges: &[Mat3],
    active: &[usize],
    ic0_component_ranges: &[(usize, usize)],
    ic0_component_batches: &[(usize, usize, usize, usize)],
    red_slots: &[usize],
    lower_links: &[[[u32; 2]; 6]],
    lower_link_count: &[u8],
    upper_links: &[[[u32; 2]; 6]],
    upper_link_count: &[u8],
    rhs: &[Vec3],
    work: &mut [Vec3],
    out: &mut [Vec3],
) {
    if kind == CpuPreconditioner::AggregationMultigrid {
        if let Some(hierarchy) = multigrid {
            hierarchy.apply(rhs, out);
            return;
        }
    }
    match kind {
        CpuPreconditioner::IncompleteCholesky0 => {
            if bipartite_ic0 {
                apply_ic0_bipartite(
                    factor_diagonal_inverse,
                    factor_edges,
                    active,
                    red_slots,
                    lower_links,
                    lower_link_count,
                    upper_links,
                    upper_link_count,
                    rhs,
                    work,
                    out,
                );
            } else if !ic0_component_batches.is_empty() {
                apply_ic0_component_batches(
                    factor_diagonal_inverse,
                    factor_edges,
                    active,
                    ic0_component_ranges,
                    ic0_component_batches,
                    lower_links,
                    lower_link_count,
                    upper_links,
                    upper_link_count,
                    rhs,
                    work,
                    out,
                );
            } else {
                apply_ic0(
                    factor_diagonal_inverse,
                    factor_edges,
                    active,
                    lower_links,
                    lower_link_count,
                    upper_links,
                    upper_link_count,
                    rhs,
                    work,
                    out,
                );
            }
        }
        CpuPreconditioner::BlockJacobi | CpuPreconditioner::AggregationMultigrid => {
            let apply_one = |(slot, output): (usize, &mut Vec3)| {
                let diagonal = factor_diagonal[active[slot]];
                *output = solve_lower_transpose(diagonal, solve_lower(diagonal, rhs[slot]));
            };
            if !parallel_enabled_for(active.len()) {
                out.iter_mut().enumerate().for_each(apply_one);
            } else {
                out.par_iter_mut().enumerate().for_each(apply_one);
            }
        }
    }
}

fn build_multigrid_preconditioner(
    free: &[Vec3],
    active: &[usize],
    active_index: &[u32],
    diagonal: &[Mat3],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    constraints: &[Constraint],
    rest_positions: &[DVec3],
) -> Result<crate::multigrid::Multigrid, &'static str> {
    let coarsening = std::env::var("VOXEL_MULTIGRID_COARSENING").unwrap_or_default();
    let geometric = coarsening.eq_ignore_ascii_case("geometric")
        || coarsening.eq_ignore_ascii_case("geometric_q1");
    let geometric_q1 = coarsening.eq_ignore_ascii_case("geometric_q1");
    let coordinates = if geometric {
        Some(geometric_multigrid_coordinates(
            active,
            active_index,
            incidence,
            incidence_len,
            constraints,
            rest_positions,
        )?)
    } else {
        None
    };
    let diagonal = active
        .iter()
        .map(|&i| {
            let d = diagonal[i];
            (d + d.transpose()) * 0.5
        })
        .collect();
    let masks = active.iter().map(|&i| free[i]).collect();
    let mut offsets = Vec::with_capacity(active.len() + 1);
    let mut columns = Vec::with_capacity(active.len() * 6);
    let mut blocks = Vec::with_capacity(active.len() * 6);
    offsets.push(0);
    for &i in active {
        let pi = Mat3::from_diagonal(free[i]);
        for &edge in &incidence[i][..incidence_len[i] as usize] {
            let c = &constraints[edge as usize];
            let j = if c.i == i { c.j } else { c.i };
            if active_index[j] != u32::MAX {
                let k = (c.k + c.k.transpose()) * 0.5;
                columns.push(active_index[j] as usize);
                blocks.push(-pi * k * Mat3::from_diagonal(free[j]));
            }
        }
        offsets.push(columns.len());
    }
    crate::multigrid::Multigrid::build(
        diagonal,
        masks,
        offsets,
        columns,
        blocks,
        coordinates,
        geometric_q1,
    )
}

/// Return stable cell-centred integer coordinates for active rows, rejecting
/// scenes whose reference lattice or active bond graph is not a face-neighbor
/// REV grid. This opt-in preconditioner hint never reads current/deformed
/// positions and never changes physical REV coordinates.
fn geometric_multigrid_coordinates(
    active: &[usize],
    active_index: &[u32],
    incidence: &[[u32; 6]],
    incidence_len: &[u8],
    constraints: &[Constraint],
    rest_positions: &[DVec3],
) -> Result<Vec<[i64; 3]>, &'static str> {
    if incidence.len() != incidence_len.len() || active_index.len() != rest_positions.len() {
        return Err("invalid multigrid incidence dimensions");
    }
    let mut coordinates = Vec::with_capacity(active.len());
    let mut unique = rustc_hash::FxHashSet::default();
    for &voxel in active {
        let position = *rest_positions
            .get(voxel)
            .ok_or("missing stable rest position for geometric multigrid")?;
        let mut cell = [0_i64; 3];
        for axis in 0..3 {
            let lattice = position[axis] / DCELL - 0.5;
            let nearest = lattice.round();
            if !nearest.is_finite()
                || nearest < i64::MIN as f64
                || nearest >= -(i64::MIN as f64)
                || (lattice - nearest).abs() > 1e-5
            {
                return Err("geometric multigrid requires cell-centred reference positions");
            }
            cell[axis] = nearest as i64;
        }
        if !unique.insert(cell) {
            return Err("geometric multigrid requires unique REV lattice coordinates");
        }
        coordinates.push(cell);
    }
    for (slot, &voxel) in active.iter().enumerate() {
        for &edge in &incidence[voxel][..incidence_len[voxel] as usize] {
            let constraint = &constraints[edge as usize];
            let neighbor = if constraint.i == voxel {
                constraint.j
            } else {
                constraint.i
            };
            let neighbor_slot = active_index[neighbor];
            if neighbor_slot == u32::MAX {
                continue;
            }
            let neighbor_cell = coordinates[neighbor_slot as usize];
            let manhattan: i128 = (0..3)
                .map(|axis| (coordinates[slot][axis] as i128 - neighbor_cell[axis] as i128).abs())
                .sum();
            if manhattan != 1 {
                return Err("geometric multigrid requires face-adjacent active constraints");
            }
        }
    }
    Ok(coordinates)
}

/// True residual of the current symmetric operator, evaluated in double precision.
fn banded_residual(
    diagonal: &[Mat3],
    lower: &[(usize, usize, Mat3)],
    rhs: &[Vec3],
    dx: &[DVec3],
    residual: &mut Vec<DVec3>,
) -> f64 {
    residual.clear();
    residual.extend(
        diagonal
            .iter()
            .zip(rhs)
            .zip(dx)
            .map(|((&a, &b), &x)| b.as_dvec3() - a.as_dmat3() * x),
    );
    for &(row, column, block) in lower {
        let block = block.as_dmat3();
        residual[row] -= block * dx[column];
        residual[column] -= block.transpose() * dx[row];
    }
    let rhs_squared: f64 = rhs.iter().map(|b| b.as_dvec3().length_squared()).sum();
    let residual_squared: f64 = residual.iter().map(|r| r.length_squared()).sum();
    if rhs_squared > 0.0 {
        (residual_squared / rhs_squared).sqrt()
    } else if residual_squared == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Small narrow-band zones can reuse a complete factor instead of repeating
/// thousands of incomplete-factor PCG iterations. An older factor is used only
/// to propose corrections: acceptance always uses the freshly assembled matrix.
/// If refinement fails, factor the current matrix once, then fall back to PCG.
fn try_banded_solve(
    scene: &mut Scene,
    h: f32,
    scratch: &mut SolverScratch,
) -> Option<(Duration, Duration)> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_CPU_BANDED_DIRECT")
            .map(|s| s != "0")
            .unwrap_or(true)
    });
    let n = scratch.active.len();
    if !enabled || n == 0 || n > 4_096 {
        return None;
    }
    let start = Instant::now();
    scratch.banded_diagonal.clear();
    scratch
        .banded_diagonal
        .extend(scratch.active.iter().map(|&i| scratch.diagonal[i]));
    scratch.banded_lower.clear();
    let mut bandwidth = 0usize;
    for constraint in &scratch.constraints {
        let i = scratch.active_index[constraint.i];
        let j = scratch.active_index[constraint.j];
        if i == u32::MAX || j == u32::MAX {
            continue;
        }
        let pi = Mat3::from_diagonal(scratch.free[constraint.i]);
        let pj = Mat3::from_diagonal(scratch.free[constraint.j]);
        let block = -pi * constraint.k * pj;
        let (row, column, lower) = if i > j {
            (i as usize, j as usize, block)
        } else {
            (j as usize, i as usize, block.transpose())
        };
        bandwidth = bandwidth.max(row - column);
        scratch.banded_lower.push((row, column, lower));
    }
    // Bound both factor memory and O(n * bandwidth²) construction cost. Broad
    // graphs keep the existing PCG/multigrid path rather than using dense storage.
    if bandwidth > 96 || n.saturating_mul(bandwidth + 1) > 400_000 {
        return None;
    }
    scratch.banded_rhs.clear();
    scratch
        .banded_rhs
        .extend(scratch.active.iter().map(|&i| scratch.rhs[i]));
    let mut factor_time = Duration::ZERO;
    let mut rebuilt = false;
    // Partial axis constraints retain a vector row: equal active counts do NOT
    // imply the same system. Keep factors for the exact free-axis masks so
    // contact activation/release does not rebuild the same factors every step.
    // Graph/material changes still require the current-operator residual check.
    if scratch.banded_factor_free != scratch.free {
        if let Some(previous) = scratch.banded_factor.take() {
            if scratch.banded_factor_pool.len() == 4 {
                scratch.banded_factor_pool.remove(0);
            }
            scratch
                .banded_factor_pool
                .push((std::mem::take(&mut scratch.banded_factor_free), previous));
        }
        if let Some(slot) = scratch
            .banded_factor_pool
            .iter()
            .position(|(free, _)| *free == scratch.free)
        {
            let (free, factor) = scratch.banded_factor_pool.swap_remove(slot);
            scratch.banded_factor_free = free;
            scratch.banded_factor = Some(factor);
        } else {
            scratch.banded_factor_free.clone_from(&scratch.free);
        }
    }
    let mut accepted = None;
    for _attempt in 0..2 {
        if scratch.banded_factor.is_none() {
            let factor_start = Instant::now();
            scratch.banded_factor = crate::banded::BandedFactor::factor(
                &scratch.banded_diagonal,
                &scratch.banded_lower,
                96,
            )
            .ok();
            factor_time += factor_start.elapsed();
            rebuilt = true;
        }
        let Some(factor) = scratch.banded_factor.as_ref() else {
            break;
        };
        scratch.banded_rhs.clear();
        scratch
            .banded_rhs
            .extend(scratch.active.iter().map(|&i| scratch.rhs[i]));
        if factor
            .solve(&scratch.banded_rhs, &mut scratch.banded_dx)
            .is_err()
        {
            scratch.banded_factor = None;
            break;
        }
        for correction in 0..=3 {
            let relative = banded_residual(
                &scratch.banded_diagonal,
                &scratch.banded_lower,
                &scratch.banded_rhs,
                &scratch.banded_dx,
                &mut scratch.banded_residual,
            );
            if relative.is_finite() && relative <= f64::from(scene.config.solver_relative_tolerance)
            {
                accepted = Some((relative, correction));
                break;
            }
            if correction == 3 || !relative.is_finite() {
                break;
            }
            // Solve the small residual, then add the correction in f64. Casting
            // the residual to f32 does not round the accumulated displacement.
            scratch.banded_rhs.clear();
            scratch
                .banded_rhs
                .extend(scratch.banded_residual.iter().map(|r| r.as_vec3()));
            let correction_result =
                factor.solve(&scratch.banded_rhs, &mut scratch.banded_correction);
            scratch.banded_rhs.clear();
            scratch
                .banded_rhs
                .extend(scratch.active.iter().map(|&i| scratch.rhs[i]));
            if correction_result.is_err() {
                break;
            }
            for (dx, correction) in scratch.banded_dx.iter_mut().zip(&scratch.banded_correction) {
                *dx += *correction;
            }
        }
        if accepted.is_some() || rebuilt {
            break;
        }
        scratch.banded_factor = None;
    }
    let (relative, corrections) = accepted?;
    scene.solver_residual = relative as f32;
    if std::env::var_os("VOXEL_AUDIT_PCG_RESIDUAL").is_some() {
        scratch.banded_correction.clear();
        scratch.banded_correction.extend(
            scratch
                .active
                .iter()
                .zip(&scratch.banded_dx)
                .map(|(&i, &dx)| (scene.positions[i] + dx) - scene.positions[i]),
        );
        let applied_relative = banded_residual(
            &scratch.banded_diagonal,
            &scratch.banded_lower,
            &scratch.banded_rhs,
            &scratch.banded_correction,
            &mut scratch.banded_residual,
        );
        let position_rounding_l2_m = scratch
            .banded_dx
            .iter()
            .zip(&scratch.banded_correction)
            .map(|(a, b)| (*a - *b).length_squared())
            .sum::<f64>()
            .sqrt();
        let audit = crate::scene::SolverResidualAudit {
            correction_relative: relative,
            applied_relative,
            position_rounding_l2_m,
        };
        report_residual_audit("cpu-banded-f64", scene.solver_residual, audit);
        scene.solver_max_audited_residual = Some(worst_audited_residual(
            scene.solver_max_audited_residual,
            audit,
        ));
        scene.solver_residual_audit = Some(audit);
    }
    for (&i, &dx) in scratch.active.iter().zip(&scratch.banded_dx) {
        scene.positions[i] += dx;
    }
    for c in &scratch.constraints {
        let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest).as_vec3();
        scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h * h);
    }
    let solve_time = start.elapsed().saturating_sub(factor_time);
    if std::env::var_os("VOXEL_PROFILE_PCG_SOLVE").is_some() {
        eprintln!("CPU banded f64 solve: active={n} bandwidth={bandwidth} rebuilt={rebuilt} corrections={corrections} residual={relative:.3e} elapsed={:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    Some((factor_time, solve_time))
}

/// Primal form of the coupled vector XPBD system for a frozen corotational frame.
/// Eliminates multipliers: (M/h² + Jᵀ K J) dx = -Jᵀ K C.
/// This is algebraically the same compliant solve, with global coupling retained.
/// Solve the same inertial/elastic quadratic on the contact nullspace. The
/// material matrix and its IC(0) factor retain their six-neighbor stencil;
/// contact projection couples bodies without masquerading as a breakable bond.
/// Difference form keeps rigid translation in the exact nullspace of K.
/// Forming diagonal*x + off_diagonal*x in f32 loses the much smaller inertia
/// term for stiff metal at a coarse step, producing a false net impulse.
fn apply_contact_operator(
    scene: &Scene,
    h: f32,
    scratch: &SolverScratch,
    vector: &[Vec3],
    out: &mut [Vec3],
    work: &mut [DVec3],
) {
    if parallel_enabled_for(scratch.active.len()) && contact_operator_rows_enabled() {
        apply_contact_operator_rows(scene, h, scratch, vector, out, work, true);
        return;
    }
    for (slot, &i) in scratch.active.iter().enumerate() {
        work[slot] = (vector[slot] * scratch.free[i]).as_dvec3()
            * (f64::from(scene.voxels[i].mass) / f64::from(h).powi(2));
    }
    for c in &scratch.constraints {
        let slot_i = scratch.active_index[c.i];
        let slot_j = scratch.active_index[c.j];
        let vi = if slot_i == u32::MAX { DVec3::ZERO }
            else { (vector[slot_i as usize] * scratch.free[c.i]).as_dvec3() };
        let vj = if slot_j == u32::MAX { DVec3::ZERO }
            else { (vector[slot_j as usize] * scratch.free[c.j]).as_dvec3() };
        let force = c.k.as_dmat3() * (vi - vj);
        if slot_i != u32::MAX { work[slot_i as usize] += force * scratch.free[c.i].as_dvec3(); }
        if slot_j != u32::MAX { work[slot_j as usize] -= force * scratch.free[c.j].as_dvec3(); }
    }
    for (out, value) in out.iter_mut().zip(work) { *out = value.as_vec3(); }
}

fn contact_operator_rows_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_CONTACT_OPERATOR_ROWS")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

/// Independent row gather for the difference-form contact operator. Incidence
/// rows are edge-ID ordered, matching the serial axis-major endpoint scatter.
/// Keeping the difference and accumulation in f64 preserves rigid translations.
fn apply_contact_operator_rows(
    scene: &Scene,
    h: f32,
    scratch: &SolverScratch,
    vector: &[Vec3],
    out: &mut [Vec3],
    work: &mut [DVec3],
    parallel: bool,
) {
    let row = |slot: usize| {
        let i = scratch.active[slot];
        let free_i = scratch.free[i];
        let mut value = (vector[slot] * free_i).as_dvec3()
            * (f64::from(scene.voxels[i].mass) / f64::from(h).powi(2));
        for &edge in &scratch.incidence[i][..scratch.incidence_len[i] as usize] {
            let c = &scratch.constraints[edge as usize];
            let slot_i = scratch.active_index[c.i];
            let slot_j = scratch.active_index[c.j];
            let vi = if slot_i == u32::MAX { DVec3::ZERO }
                else { (vector[slot_i as usize] * scratch.free[c.i]).as_dvec3() };
            let vj = if slot_j == u32::MAX { DVec3::ZERO }
                else { (vector[slot_j as usize] * scratch.free[c.j]).as_dvec3() };
            let force = c.k.as_dmat3() * (vi - vj);
            if c.i == i {
                value += force * free_i.as_dvec3();
            } else {
                value -= force * free_i.as_dvec3();
            }
        }
        value
    };
    if parallel {
        use rayon::prelude::*;
        work.par_iter_mut().zip(out.par_iter_mut()).enumerate()
            .for_each(|(slot, (acc, result))| {
                *acc = row(slot);
                *result = acc.as_vec3();
            });
    } else {
        for slot in 0..scratch.active.len() {
            work[slot] = row(slot);
            out[slot] = work[slot].as_vec3();
        }
    }
}

#[cfg(test)]
mod contact_operator_row_tests {
    use super::*;
    use crate::material::Material;
    use crate::scene::{SceneConfig, VoxelSpawn};

    #[test]
    fn row_gather_matches_serial_difference_operator() {
        let oak = Material::component_id("oak").unwrap();
        let scene = Scene::from_voxels(SceneConfig::default(), (0..4)
            .map(|x| VoxelSpawn::at_cell([x, 0, 8], oak).at_temperature(293.15))
            .collect()).unwrap();
        let mut scratch = SolverScratch::default();
        scratch.active = (0..4).collect();
        scratch.active_index = vec![u32::MAX; scene.voxels.len()];
        scratch.free = vec![Vec3::ONE; scene.voxels.len()];
        for (slot, &i) in scratch.active.iter().enumerate() {
            scratch.active_index[i] = slot as u32;
        }
        scratch.free[1].y = 0.0;
        scratch.free[2].z = 0.0;
        for (i, j, axis) in [(0, 1, 0), (1, 2, 1), (2, 3, 0)] {
            let edge = scratch.constraints.len() as u32;
            scratch.constraints.push(make_constraint(&scene, i, axis, j));
            for vertex in [i, j] {
                let len = scratch.incidence_len.get(vertex).copied().unwrap_or(0) as usize;
                if scratch.incidence.len() < scene.voxels.len() {
                    scratch.incidence.resize(scene.voxels.len(), [0; 6]);
                    scratch.incidence_len.resize(scene.voxels.len(), 0);
                }
                scratch.incidence[vertex][len] = edge;
                scratch.incidence_len[vertex] += 1;
            }
        }
        let h = 0.01;
        let translation = Vec3::new(2.0e5, -3.0e5, 4.0e5);
        let vectors = [
            vec![translation; 4],
            vec![Vec3::new(0.2, -0.3, 0.7), Vec3::new(-0.5, 0.9, 0.1),
                Vec3::new(0.4, 0.2, -0.8), Vec3::new(0.6, -0.1, 0.3)],
        ];
        for vector in vectors {
            let mut expected = vec![Vec3::ZERO; 4];
            let mut expected_work = vec![DVec3::ZERO; 4];
            for (slot, &i) in scratch.active.iter().enumerate() {
                expected_work[slot] = (vector[slot] * scratch.free[i]).as_dvec3()
                    * (f64::from(scene.voxels[i].mass) / f64::from(h).powi(2));
            }
            for c in &scratch.constraints {
                let si = scratch.active_index[c.i];
                let sj = scratch.active_index[c.j];
                let vi = (vector[si as usize] * scratch.free[c.i]).as_dvec3();
                let vj = (vector[sj as usize] * scratch.free[c.j]).as_dvec3();
                let force = c.k.as_dmat3() * (vi - vj);
                expected_work[si as usize] += force * scratch.free[c.i].as_dvec3();
                expected_work[sj as usize] -= force * scratch.free[c.j].as_dvec3();
            }
            for (o, w) in expected.iter_mut().zip(&expected_work) { *o = w.as_vec3(); }
            let mut serial = vec![Vec3::ZERO; 4];
            let mut serial_work = vec![DVec3::ZERO; 4];
            apply_contact_operator_rows(&scene, h, &scratch, &vector, &mut serial, &mut serial_work, false);
            assert_eq!(serial, expected);
            let mut parallel = vec![Vec3::ZERO; 4];
            let mut parallel_work = vec![DVec3::ZERO; 4];
            apply_contact_operator_rows(&scene, h, &scratch, &vector, &mut parallel, &mut parallel_work, true);
            assert_eq!(parallel, expected);
            assert_eq!(parallel, serial);
        }
    }

    #[test]
    fn large_real_grid_wrapper_matches_serial_with_inner_bond() {
        let oak = Material::component_id("oak").unwrap();
        let mut spawns = Vec::with_capacity(21 * 21 * 21);
        for z in 0..21 {
            for y in 0..21 {
                for x in 0..21 {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], oak).at_temperature(293.15));
                }
            }
        }
        let scene = Scene::from_voxels(SceneConfig::default(), spawns).unwrap();
        let n = scene.voxels.len();
        assert_eq!(n, 9_261);
        let mut scratch = SolverScratch::default();
        scratch.active = (0..n).collect();
        scratch.active_index = (0..n).map(|slot| slot as u32).collect();
        scratch.free = vec![Vec3::ONE; n];
        // Exercise a mixed mask on an otherwise regular active grid.
        scratch.free[n / 2].z = 0.0;
        scratch.incidence = vec![[0; 6]; n];
        scratch.incidence_len = vec![0; n];
        let inner = (10 * 21 + 10) * 21 + 10;
        let neighbor = inner + 1;
        let edge = scratch.constraints.len() as u32;
        scratch.constraints.push(make_constraint(&scene, inner, 0, neighbor));
        scratch.incidence[inner][0] = edge;
        scratch.incidence_len[inner] = 1;
        scratch.incidence[neighbor][0] = edge;
        scratch.incidence_len[neighbor] = 1;

        let mut vector = vec![Vec3::new(0.125, -0.25, 0.5); n];
        vector[inner] = Vec3::new(-0.75, 0.125, 0.375);
        vector[neighbor] = Vec3::new(0.25, 0.875, -0.5);
        let mut reference = vec![Vec3::ZERO; n];
        let mut reference_work = vec![DVec3::ZERO; n];
        for (slot, &i) in scratch.active.iter().enumerate() {
            reference_work[slot] = (vector[slot] * scratch.free[i]).as_dvec3()
                * (f64::from(scene.voxels[i].mass) / f64::from(0.01_f32).powi(2));
        }
        for c in &scratch.constraints {
            let si = scratch.active_index[c.i] as usize;
            let sj = scratch.active_index[c.j] as usize;
            let vi = (vector[si] * scratch.free[c.i]).as_dvec3();
            let vj = (vector[sj] * scratch.free[c.j]).as_dvec3();
            let force = c.k.as_dmat3() * (vi - vj);
            reference_work[si] += force * scratch.free[c.i].as_dvec3();
            reference_work[sj] -= force * scratch.free[c.j].as_dvec3();
        }
        for (dst, src) in reference.iter_mut().zip(&reference_work) { *dst = src.as_vec3(); }

        let mut wrapped = vec![Vec3::ZERO; n];
        let mut wrapped_work = vec![DVec3::ZERO; n];
        apply_contact_operator(&scene, 0.01, &scratch, &vector, &mut wrapped, &mut wrapped_work);
        assert_eq!(wrapped, reference);
        let mut gathered = vec![Vec3::ZERO; n];
        let mut gathered_work = vec![DVec3::ZERO; n];
        apply_contact_operator_rows(&scene, 0.01, &scratch, &vector, &mut gathered, &mut gathered_work, true);
        assert_eq!(gathered, reference);
        assert_eq!(gathered, wrapped);
    }

    #[test]
    fn empty_rows_and_fully_masked_rows_are_safe() {
        let oak = Material::component_id("oak").unwrap();
        let scene = Scene::from_voxels(SceneConfig::default(), (0..3)
            .map(|x| VoxelSpawn::at_cell([x, 0, 8], oak).at_temperature(293.15))
            .collect()).unwrap();
        let mut scratch = SolverScratch::default();
        scratch.active = vec![0, 1, 2];
        scratch.active_index = vec![u32::MAX; scene.voxels.len()];
        scratch.free = vec![Vec3::ZERO; scene.voxels.len()];
        scratch.incidence = vec![[0; 6]; scene.voxels.len()];
        scratch.incidence_len = vec![0; scene.voxels.len()];
        for (slot, &i) in scratch.active.iter().enumerate() { scratch.active_index[i] = slot as u32; }
        let vector = vec![Vec3::ONE; 3];
        let mut output = vec![Vec3::ZERO; 3];
        let mut work = vec![DVec3::ZERO; 3];
        apply_contact_operator_rows(&scene, 0.01, &scratch, &vector, &mut output, &mut work, true);
        assert_eq!(output, vec![Vec3::ZERO; 3]);
        assert_eq!(work, vec![DVec3::ZERO; 3]);
        let empty_scratch = SolverScratch::default();
        let mut empty_out = [];
        let mut empty_work = [];
        apply_contact_operator_rows(&scene, 0.01, &empty_scratch, &[], &mut empty_out, &mut empty_work, true);
    }

    #[test]
    fn inactive_constraint_endpoint_and_zero_vector_are_safe() {
        let oak = Material::component_id("oak").unwrap();
        let scene = Scene::from_voxels(SceneConfig::default(), (0..3)
            .map(|x| VoxelSpawn::at_cell([x, 0, 8], oak).at_temperature(293.15))
            .collect()).unwrap();
        let mut scratch = SolverScratch::default();
        scratch.active = vec![0];
        scratch.active_index = vec![u32::MAX; scene.voxels.len()];
        scratch.active_index[0] = 0;
        scratch.free = vec![Vec3::ONE; scene.voxels.len()];
        scratch.incidence = vec![[0; 6]; scene.voxels.len()];
        scratch.incidence_len = vec![0; scene.voxels.len()];
        scratch.constraints.push(make_constraint(&scene, 0, 0, 1));
        scratch.incidence[0][0] = 0;
        scratch.incidence_len[0] = 1;

        for vector in [vec![Vec3::ZERO], vec![Vec3::new(1.0e5, -2.0e5, 3.0e5)]] {
            let mut expected = vec![Vec3::ZERO];
            let mut expected_work = vec![(vector[0] * scratch.free[0]).as_dvec3()
                * (f64::from(scene.voxels[0].mass) / f64::from(0.01_f32).powi(2))];
            let c = &scratch.constraints[0];
            let vi = (vector[0] * scratch.free[c.i]).as_dvec3();
            let vj = DVec3::ZERO; // Endpoint 1 is outside the active set.
            let force = c.k.as_dmat3() * (vi - vj);
            expected_work[0] += force * scratch.free[0].as_dvec3();
            expected[0] = expected_work[0].as_vec3();

            let mut output = vec![Vec3::ZERO];
            let mut work = vec![DVec3::ZERO];
            apply_contact_operator_rows(&scene, 0.01, &scratch, &vector, &mut output, &mut work, true);
            assert_eq!(output, expected);
            assert!(output[0].is_finite());
            assert!(work[0].is_finite());
            if vector[0] == Vec3::ZERO {
                assert_eq!(output, vec![Vec3::ZERO]);
            }
        }
    }

}

fn prepare_contact_gpu_operator(
    scene: &Scene,
    h: f32,
    scratch: &SolverScratch,
) -> Result<crate::cuda_solver::ContactOperatorSession, String> {
    use crate::cuda_solver::{ContactOperatorEdge, ContactOperatorInput, ContactOperatorSession};
    let masses: Vec<_> = scratch.active.iter().map(|&i| {
        f64::from(scene.voxels[i].mass) / f64::from(h).powi(2)
    }).collect();
    let free: Vec<_> = scratch.active.iter().map(|&i| scratch.free[i].to_array()).collect();
    let incidence: Vec<_> = scratch.active.iter().map(|&i| scratch.incidence[i]).collect();
    let edges: Vec<_> = scratch.constraints.iter().map(|edge| ContactOperatorEdge {
        slots: [scratch.active_index[edge.i], scratch.active_index[edge.j]],
        free: [scratch.free[edge.i].to_array(), scratch.free[edge.j].to_array()],
        stiffness: edge.k.to_cols_array(),
    }).collect();
    ContactOperatorSession::new(ContactOperatorInput {
        masses_over_h2: &masses,
        free: &free,
        incidence: &incidence,
        edges: &edges,
    })
}

fn contact_gpu_operator_requested(active_count: usize) -> bool {
    active_count >= PARALLEL_MIN_VOXELS
        && std::env::var("VOXEL_CUDA_CONTACT_OPERATOR").as_deref() == Ok("1")
}

fn prepare_contact_gpu_ic0(
    session: &mut crate::cuda_solver::ContactOperatorSession,
    scratch: &SolverScratch,
) -> Result<(), String> {
    let inverse: Vec<_> = scratch.active.iter()
        .map(|&i| scratch.factor_diagonal_inverse[i].to_cols_array()).collect();
    let edges: Vec<_> = scratch.factor_edges.iter().map(Mat3::to_cols_array).collect();
    session.prepare_ic0(crate::cuda_solver::ContactIc0Input {
        inverse_diagonal: &inverse,
        edge_factors: &edges,
        lower_links: &scratch.ic0_lower_links,
        lower_counts: &scratch.ic0_lower_link_count,
        upper_links: &scratch.ic0_upper_links,
        upper_counts: &scratch.ic0_upper_link_count,
    })
}

fn apply_contact_cpu_preconditioner(
    scene: &Scene,
    scratch: &SolverScratch,
    rhs: &[Vec3],
    work: &mut [Vec3],
    out: &mut [Vec3],
) {
    // Cells never build the translation-only IC(0) factor, so the restitution
    // solve that follows uses block Jacobi on the same translation blocks.
    if scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0 && !scene.cell_dofs {
        apply_preconditioner(scene.config.cpu_preconditioner, scratch.cpu_preconditioner_bipartite,
            &mut None, &scratch.factor_diagonal, &scratch.factor_diagonal_inverse,
            &scratch.factor_edges, &scratch.active, &scratch.ic0_component_ranges,
            &scratch.ic0_component_batches, &scratch.ic0_red_slots, &scratch.ic0_lower_links,
            &scratch.ic0_lower_link_count, &scratch.ic0_upper_links,
            &scratch.ic0_upper_link_count, rhs, work, out);
    } else {
        for (slot, &i) in scratch.active.iter().enumerate() {
            let diagonal = scratch.factor_diagonal[i];
            out[slot] = solve_lower_transpose(diagonal, solve_lower(diagonal, rhs[slot]));
        }
    }
}

fn contact_gpu_requires_cpu_retry(
    used_gpu: bool,
    residual: f32,
    tolerance: f32,
    iterations: usize,
    iteration_limit: usize,
) -> bool {
    used_gpu && (!residual.is_finite()
        || (residual > tolerance && iterations < iteration_limit))
}

fn solve_contact_increment(
    scene: &Scene,
    h: f32,
    scratch: &SolverScratch,
    pairs: &[crate::contact_constraint::PairContact],
    relative_targets: &[f64],
    input_rhs: &[Vec3],
) -> (Vec<Vec3>, usize, f32) {
    let requested = contact_gpu_operator_requested(scratch.active.len());
    let (delta, iterations, residual, used_gpu) = solve_contact_increment_backend(
        scene, h, scratch, pairs, relative_targets, input_rhs, requested);
    if contact_gpu_requires_cpu_retry(used_gpu, residual,
        scene.config.solver_relative_tolerance, iterations, scene.config.iterations) {
        // No candidate state has been applied yet. Discard an unverified GPU
        // solve and rerun the same physical system entirely on the CPU.
        record_contact_pcg_profile(ContactPcgProfile {
            gpu_operator_fallbacks: 1,
            ..Default::default()
        });
        let (cpu_delta, cpu_iterations, cpu_residual, _) = solve_contact_increment_backend(
            scene, h, scratch, pairs, relative_targets, input_rhs, false);
        let max_delta_error = delta.iter().zip(&cpu_delta)
            .map(|(gpu, cpu)| (*gpu - *cpu).abs().max_element()).fold(0.0_f32, f32::max);
        crate::runtime_metrics::publish("event", serde_json::json!({
            "kind":"contact_gpu_full_cpu_retry",
            "candidate_true_residual":residual,
            "cpu_true_residual":cpu_residual,
            "tolerance":scene.config.solver_relative_tolerance,
            "candidate_iterations":iterations,
            "cpu_iterations":cpu_iterations,
            "max_delta_difference":max_delta_error,
        }));
        return (cpu_delta, cpu_iterations, cpu_residual);
    }
    // Reaching the configured iteration limit is the same explicit stopping
    // policy on CPU and GPU. It is recorded as unconverged, never as success.
    // A CPU retry would repeat the same limited correction; outer contact and
    // corotation passes still use the CPU-evaluated true residual.
    (delta, iterations, residual)
}

fn solve_contact_increment_backend(
    scene: &Scene,
    h: f32,
    scratch: &SolverScratch,
    pairs: &[crate::contact_constraint::PairContact],
    relative_targets: &[f64],
    input_rhs: &[Vec3],
    allow_gpu: bool,
) -> (Vec<Vec3>, usize, f32, bool) {
    use crate::contact_constraint::ContactProjection;
    let profiling = crate::runtime_metrics::is_active()
        || std::env::var_os("VOXEL_PROFILE_STEP").is_some();
    let total_started = profiling.then(Instant::now);
    let progress_started = Instant::now();
    let mut next_progress = progress_started + Duration::from_millis(500);
    let operator_time = std::cell::Cell::new(Duration::ZERO);
    let projection_time = std::cell::Cell::new(Duration::ZERO);
    let preconditioner_time = std::cell::Cell::new(Duration::ZERO);
    let reduction_time = std::cell::Cell::new(Duration::ZERO);
    let mut update_time = Duration::ZERO;
    let gpu_calls = std::cell::Cell::new(0usize);
    let gpu_preconditioner_calls = std::cell::Cell::new(0usize);
    let gpu_fallbacks = std::cell::Cell::new(0usize);
    let gpu_setup_started = profiling.then(Instant::now);
    let gpu = if allow_gpu {
        match prepare_contact_gpu_operator(scene, h, scratch) {
            Ok(session) => Some(session),
            Err(error) => {
                gpu_fallbacks.set(1);
                crate::runtime_metrics::publish("event", serde_json::json!({
                    "kind":"contact_gpu_unavailable", "error":error
                }));
                None
            }
        }
    } else { None };
    let gpu = std::cell::RefCell::new(gpu);
    let gpu_ic0_ready = std::cell::Cell::new(false);
    if scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0
        && std::env::var("VOXEL_CUDA_CONTACT_IC0").as_deref() == Ok("1") {
        if let Some(session) = gpu.borrow_mut().as_mut() {
            match prepare_contact_gpu_ic0(session, scratch) {
                Ok(()) => {
                    gpu_ic0_ready.set(true);
                    crate::runtime_metrics::publish("event", serde_json::json!({
                        "kind":"contact_gpu_ic0_prepared",
                        "active_fine_revs":scratch.active.len(),
                        "dependency_levels_forward_reverse":session.ic0_level_counts(),
                    }));
                }
                Err(error) => {
                    gpu_fallbacks.set(gpu_fallbacks.get() + 1);
                    crate::runtime_metrics::publish("event", serde_json::json!({
                        "kind":"contact_gpu_ic0_unavailable", "error":error,
                        "action":"cpu_preconditioner"
                    }));
                }
            }
        }
    }
    let gpu_setup_time = if allow_gpu {
        gpu_setup_started.map(|start| start.elapsed()).unwrap_or_default()
    } else { Duration::ZERO };
    let mut gpu_verified = false;
    let verify_all_gpu_vectors =
        std::env::var("VOXEL_CUDA_CONTACT_VERIFY_ALL").as_deref() == Ok("1");
    let projection = ContactProjection::build(pairs, &scratch.free, &scratch.active_index);
    let n = scratch.active.len();
    let particular = projection.particular(pairs, relative_targets, n);
    let mut particular_force = vec![Vec3::ZERO; n];
    let mut operator_work = vec![DVec3::ZERO; n];
    let mut gpu_reference = vec![Vec3::ZERO; if gpu.borrow().is_some() { n } else { 0 }];
    let operator_start = profiling.then(Instant::now);
    apply_contact_operator(scene, h, scratch, &particular, &mut particular_force, &mut operator_work);
    if let Some(start) = operator_start { operator_time.set(start.elapsed()); }
    let mut rhs: Vec<_> = input_rhs.iter().zip(&particular_force).map(|(b, a)| *b - *a).collect();
    let projection_start = profiling.then(Instant::now);
    projection.project(&mut rhs);
    if let Some(start) = projection_start { projection_time.set(start.elapsed()); }
    let dot = |a: &[Vec3], b: &[Vec3]| {
        let started = profiling.then(Instant::now);
        let result = a.iter().zip(b)
            .map(|(a,b)| a.as_dvec3().dot(b.as_dvec3())).sum::<f64>();
        if let Some(start) = started { reduction_time.set(reduction_time.get() + start.elapsed()); }
        result
    };
    let mut apply = |v: &[Vec3], out: &mut [Vec3], true_residual: bool| {
        let operator_start = profiling.then(Instant::now);
        let device_result = if true_residual { None }
            else { gpu.borrow_mut().as_mut().map(|session| session.apply(v, out)) };
        match device_result {
            Some(Ok(())) => {
                if !gpu_verified || verify_all_gpu_vectors {
                    // Validate the first actual Krylov vector on the CPU;
                    // diagnostics can request this check for every vector.
                    // Exact f32 outputs are required for this optional path.
                    apply_contact_operator(scene, h, scratch, v, &mut gpu_reference, &mut operator_work);
                    if *out != gpu_reference {
                        out.copy_from_slice(&gpu_reference);
                        *gpu.borrow_mut() = None;
                        gpu_ic0_ready.set(false);
                        gpu_fallbacks.set(gpu_fallbacks.get() + 1);
                        crate::runtime_metrics::publish("event", serde_json::json!({
                            "kind":"contact_gpu_operator_mismatch", "action":"cpu_fallback"
                        }));
                    } else {
                        gpu_verified = true;
                    }
                }
                if gpu.borrow().is_some() { gpu_calls.set(gpu_calls.get() + 1); }
            }
            Some(Err(error)) => {
                *gpu.borrow_mut() = None;
                gpu_ic0_ready.set(false);
                gpu_fallbacks.set(gpu_fallbacks.get() + 1);
                crate::runtime_metrics::publish("event", serde_json::json!({
                    "kind":"contact_gpu_operator_error", "error":error, "action":"cpu_fallback"
                }));
                apply_contact_operator(scene, h, scratch, v, out, &mut operator_work);
            }
            None => apply_contact_operator(scene, h, scratch, v, out, &mut operator_work),
        }
        if let Some(start) = operator_start { operator_time.set(operator_time.get() + start.elapsed()); }
        let projection_start = profiling.then(Instant::now);
        projection.project(out);
        if let Some(start) = projection_start { projection_time.set(projection_time.get() + start.elapsed()); }
    };
    let mut dx = vec![Vec3::ZERO; n];
    let mut r = rhs.clone();
    let mut z = vec![Vec3::ZERO; n];
    let mut work = vec![Vec3::ZERO; n];
    let mut gpu_ic0_verified = false;
    let mut gpu_ic0_reference = vec![Vec3::ZERO; if gpu_ic0_ready.get() { n } else { 0 }];
    let mut precondition = |r: &[Vec3], z: &mut [Vec3], work: &mut [Vec3]| {
        let started = profiling.then(Instant::now);
        // Projecting an SPD preconditioner on both sides is SPD on ker(C).
        // IC(0) is reused across the contact-set passes; the contact rows never
        // alter its material graph or cause independent component solves.
        let result = if gpu_ic0_ready.get() {
            gpu.borrow_mut().as_mut().map(|session| session.apply_ic0(r, z))
        } else { None };
        match result {
            Some(Ok(())) => {
                if !gpu_ic0_verified || verify_all_gpu_vectors {
                    apply_contact_cpu_preconditioner(scene, scratch, r, work, &mut gpu_ic0_reference);
                    if *z != gpu_ic0_reference {
                        z.copy_from_slice(&gpu_ic0_reference);
                        gpu_ic0_ready.set(false);
                        gpu_fallbacks.set(gpu_fallbacks.get() + 1);
                        crate::runtime_metrics::publish("event", serde_json::json!({
                            "kind":"contact_gpu_ic0_mismatch", "action":"cpu_preconditioner"
                        }));
                    } else { gpu_ic0_verified = true; }
                }
                if gpu_ic0_ready.get() {
                    gpu_preconditioner_calls.set(gpu_preconditioner_calls.get() + 1);
                }
            }
            Some(Err(error)) => {
                // Device failures may poison the shared CUDA context; stop
                // both optional paths before reusing any candidate output.
                *gpu.borrow_mut() = None;
                gpu_ic0_ready.set(false);
                gpu_fallbacks.set(gpu_fallbacks.get() + 1);
                crate::runtime_metrics::publish("event", serde_json::json!({
                    "kind":"contact_gpu_ic0_error", "error":error, "action":"cpu_fallback"
                }));
                apply_contact_cpu_preconditioner(scene, scratch, r, work, z);
            }
            None => apply_contact_cpu_preconditioner(scene, scratch, r, work, z),
        }
        if let Some(start) = started { preconditioner_time.set(preconditioner_time.get() + start.elapsed()); }
        let projection_start = profiling.then(Instant::now);
        projection.project(z);
        if let Some(start) = projection_start { projection_time.set(projection_time.get() + start.elapsed()); }
    };
    let rhs_norm = dot(&rhs, &rhs).sqrt();
    precondition(&r, &mut z, &mut work);
    let mut p = z.clone();
    let mut ap = vec![Vec3::ZERO; n];
    let mut rz = dot(&r, &z);
    let tolerance = f64::from(scene.config.solver_relative_tolerance);
    let mut iterations = 0;
    for _ in 0..scene.config.iterations {
        if rhs_norm <= 1e-20 || dot(&r, &r).sqrt() <= tolerance * rhs_norm { break; }
        apply(&p, &mut ap, false);
        let denom = dot(&p, &ap);
        if !denom.is_finite() || denom <= 0.0 { break; }
        let alpha = (rz / denom) as f32;
        let update_start = profiling.then(Instant::now);
        for i in 0..n { dx[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
        if let Some(start) = update_start { update_time += start.elapsed(); }
        iterations += 1;
        if profiling && crate::runtime_metrics::is_active() && Instant::now() >= next_progress {
            crate::runtime_metrics::publish("event", serde_json::json!({
                "kind":"contact_pcg_progress",
                "active_fine_revs":n,
                "iteration":iterations,
                "iteration_limit":scene.config.iterations,
                "elapsed_seconds":progress_started.elapsed().as_secs_f64(),
                "gpu_operator_calls":gpu_calls.get(),
                "gpu_preconditioner_calls":gpu_preconditioner_calls.get(),
            }));
            next_progress = Instant::now() + Duration::from_millis(500);
        }
        // Validate convergence against the assembled operator, not only the
        // recursive residual; stiff metal is sensitive to f32 cancellation.
        if dot(&r, &r).sqrt() <= tolerance * rhs_norm || iterations % 32 == 0 {
            projection.project(&mut dx);
            apply(&dx, &mut ap, true);
            let update_start = profiling.then(Instant::now);
            for i in 0..n { r[i] = rhs[i] - ap[i]; }
            if let Some(start) = update_start { update_time += start.elapsed(); }
            if dot(&r, &r).sqrt() <= tolerance * rhs_norm { break; }
            precondition(&r, &mut z, &mut work);
            p.clone_from(&z); rz = dot(&r, &z);
            continue;
        }
        precondition(&r, &mut z, &mut work);
        let next = dot(&r, &z);
        if !next.is_finite() || rz <= 0.0 { break; }
        let beta = (next / rz) as f32;
        let update_start = profiling.then(Instant::now);
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
        if let Some(start) = update_start { update_time += start.elapsed(); }
        rz = next;
    }
    projection.project(&mut dx);
    apply(&dx, &mut ap, true);
    for i in 0..n { r[i] = rhs[i] - ap[i]; }
    let residual = checked_relative_residual(dot(&r, &r).sqrt(), rhs_norm);
    for (value, offset) in dx.iter_mut().zip(particular) { *value += offset; }
    if input_rhs.iter().all(|value| *value == Vec3::ZERO) {
        // Solve the rigid translation modes analytically. A relative residual
        // scaled by metal stiffness does not bound the much smaller inertial
        // residual; contact impulses on a closed free system sum to zero.
        deflate_contact_translation(scene, scratch, pairs, &mut dx);
    }
    record_contact_pcg_profile(ContactPcgProfile {
        calls: 1,
        iterations,
        operator: operator_time.get(),
        projection: projection_time.get(),
        preconditioner: preconditioner_time.get(),
        reductions: reduction_time.get(),
        updates: update_time,
        total: total_started.map(|start| start.elapsed()).unwrap_or_default(),
        gpu_setup: gpu_setup_time,
        gpu_operator_calls: gpu_calls.get(),
        gpu_preconditioner_calls: gpu_preconditioner_calls.get(),
        gpu_operator_fallbacks: gpu_fallbacks.get(),
        unconverged_calls: usize::from(!residual.is_finite() || residual > scene.config.solver_relative_tolerance),
        max_true_residual: residual.is_finite().then_some(residual),
    });
    (dx, iterations, residual,
        gpu_calls.get() > 0 || gpu_preconditioner_calls.get() > 0)
}

fn deflate_contact_translation(
    scene: &Scene,
    scratch: &SolverScratch,
    pairs: &[crate::contact_constraint::PairContact],
    delta: &mut [Vec3],
) {
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i { parent[i] = parent[parent[i]]; i = parent[i]; }
        i
    }
    let mut parent: Vec<_> = (0..scene.voxels.len()).collect();
    for (i, j) in scratch.constraints.iter().map(|c| (c.i, c.j))
        .chain(pairs.iter().filter(|p| p.active).map(|p| (p.i, p.j))) {
        let a = root(&mut parent, i);
        let b = root(&mut parent, j);
        parent[b] = a;
    }
    let mut mass = vec![0.0; parent.len()];
    let mut momentum = vec![DVec3::ZERO; parent.len()];
    let mut fixed = vec![Vec3::ZERO; parent.len()];
    for (i, voxel) in scene.voxels.iter().enumerate() {
        let group = root(&mut parent, i);
        fixed[group] = fixed[group].max(Vec3::ONE - scratch.free[i]);
        let slot = scratch.active_index[i];
        if slot != u32::MAX {
            mass[group] += f64::from(voxel.mass);
            momentum[group] += delta[slot as usize].as_dvec3() * f64::from(voxel.mass);
        }
    }
    for (slot, &i) in scratch.active.iter().enumerate() {
        let group = root(&mut parent, i);
        if mass[group] > 0.0 {
            delta[slot] -= (momentum[group] / mass[group]).as_vec3()
                * (Vec3::ONE - fixed[group]);
        }
    }
}

/// Resolve an impact manifold with the same dynamic compliance as the bond
/// solve. A local REV-only impulse would stop just the surface layer and then
/// let backward Euler dissipate the bulk motion before it could rebound.
fn resolve_coupled_restitution(
    scene: &mut Scene,
    scratch: &SolverScratch,
    contacts: &[collision::Contact],
    incoming: &[Vec3],
    h: f32,
    normal_impulses: &mut [f32],
    implicit_heat: f64,
) -> (f64, f64) {
    use crate::contact_constraint::{ContactProjection, PairContact};
    let mut pairs = Vec::new();
    let mut indices = Vec::new();
    let mut incoming_normal = Vec::new();
    let mut current_normal = Vec::new();
    for (index, contact) in contacts.iter().enumerate() {
        if !contact.new_impact { continue; }
        let axis = if contact.normal.x.abs() > 0.5 { 0 }
            else if contact.normal.y.abs() > 0.5 { 1 } else { 2 };
        if scratch.free[contact.i][axis] == 0.0 || scratch.free[contact.j][axis] == 0.0 {
            continue;
        }
        let before = f64::from((incoming[contact.j] - incoming[contact.i]).dot(contact.normal));
        if before >= -1.0e-6 { continue; }
        pairs.push(PairContact { i: contact.i, j: contact.j, axis,
            sign: f64::from(contact.normal[axis]), active: true });
        indices.push(index);
        incoming_normal.push(before);
        current_normal.push(f64::from((scene.voxels[contact.j].velocity
            - scene.voxels[contact.i].velocity).dot(contact.normal)));
    }
    if pairs.is_empty() { return (implicit_heat, 0.0); }
    let zero_rhs = vec![Vec3::ZERO; scratch.active.len()];
    let targets = |restitution: f64| incoming_normal.iter().zip(&current_normal)
        .map(|(before, current)| -restitution * before - current).collect::<Vec<_>>();
    let (mut delta, _, _) = solve_contact_increment(scene, h, scratch, &pairs,
        &targets(f64::from(scene.config.fracture.restitution)), &zero_rhs);
    let kinetic = |increment: &[Vec3]| scratch.active.iter().enumerate().map(|(slot, &i)| {
        let v = scene.voxels[i].velocity.as_dvec3() + increment[slot].as_dvec3();
        0.5 * f64::from(scene.voxels[i].mass) * v.length_squared()
    }).sum::<f64>();
    let before = kinetic(&zero_rhs);
    let budget = before + implicit_heat;
    if kinetic(&delta) > budget {
        // Reserve rebound before depositing dissipation as heat. If a contact
        // target is incompatible with the available mechanical energy, reduce
        // only its restitution; no ambient thermal energy funds the rebound.
        let (base, _, _) = solve_contact_increment(scene, h, scratch, &pairs, &targets(0.0), &zero_rhs);
        if kinetic(&base) > budget {
            // The exact local mass projection in the ordinary impulse pass is
            // the passive fallback for an unresolved compliance solve.
            return (implicit_heat, 0.0);
        }
        let mut low = 0.0_f64;
        let mut high = 1.0_f64;
        let candidate = delta.clone();
        for _ in 0..40 {
            let fraction = (low + high) * 0.5;
            for i in 0..delta.len() {
                delta[i] = base[i] + (candidate[i] - base[i]) * fraction as f32;
            }
            if kinetic(&delta) <= budget { low = fraction; } else { high = fraction; }
        }
        for i in 0..delta.len() { delta[i] = base[i] + (candidate[i] - base[i]) * low as f32; }
    }
    let after = kinetic(&delta);
    let mut gradient = vec![Vec3::ZERO; scratch.active.len()];
    let mut operator_work = vec![DVec3::ZERO; scratch.active.len()];
    apply_contact_operator(scene, h, scratch, &delta, &mut gradient, &mut operator_work);
    let mut full_gradient = vec![DVec3::ZERO; scene.voxels.len()];
    for (slot, &i) in scratch.active.iter().enumerate() { full_gradient[i] = gradient[slot].as_dvec3(); }
    let projection = ContactProjection::build(&pairs, &scratch.free, &scratch.active_index);
    for (&index, reaction) in indices.iter().zip(projection.reactions(&pairs, &full_gradient)) {
        normal_impulses[index] += (reaction.max(0.0) * f64::from(h).powi(2)) as f32;
    }
    for (slot, &i) in scratch.active.iter().enumerate() { scene.voxels[i].velocity += delta[slot]; }
    let loss = before - after;
    if loss >= 0.0 { (implicit_heat, loss) }
    else { ((implicit_heat + loss).max(0.0), 0.0) }
}

fn solve_projected_contact(scene: &mut Scene, h: f32, scratch: &mut SolverScratch) -> usize {
    let trace = std::env::var_os("VOXEL_TRACE_CONTACT_SOLVE").is_some();
    let elastic_before = trace.then(|| scene.elastic_bond_energy_j());
    let rhs: Vec<_> = scratch.active.iter().map(|&i| scratch.rhs[i]).collect();
    // New CCD projections may move a REV that already belongs to another
    // active contact. Re-establish each normal gap in the same coupled solve;
    // a homogeneous increment would preserve the newly introduced overlap.
    let gap_targets: Vec<_> = scratch.pair_contacts.iter().map(|pair| {
        if pair.active {
            DCELL - (scene.positions[pair.j][pair.axis]
                - scene.positions[pair.i][pair.axis]) * pair.sign
        } else { 0.0 }
    }).collect();
    let (dx, iterations, residual) = solve_contact_increment(
        scene, h, scratch, &scratch.pair_contacts, &gap_targets, &rhs);
    if !residual.is_finite() || dx.iter().any(|value| !value.is_finite()) {
        scene.solver_residual = f32::INFINITY;
        return iterations;
    }
    scene.solver_residual = residual;
    for (slot, &i) in scratch.active.iter().enumerate() { scene.positions[i] += dx[slot].as_dvec3(); }
    if let Some(elastic_before) = elastic_before {
        eprintln!("CONTACT_SOLVE step={} h={h:.6e} active={} pairs={} iterations={iterations} residual={residual:.6e} max_dx={:.6e} elastic_before={elastic_before:.6e} elastic_after={:.6e}",
            scene.substep_index, scratch.active.len(), scratch.pair_contacts.len(),
            dx.iter().map(|d| d.length()).fold(0.0_f32, f32::max), scene.elastic_bond_energy_j());
    }
    scratch.dx = dx;
    for c in &scratch.constraints {
        let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest).as_vec3();
        scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h*h);
    }
    iterations
}

/// Solver state that carries physics between substeps, kept in a checkpoint.
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub(crate) struct SolverWarmState {
    support_friction_modes: Vec<(usize, SupportFriction)>,
    support_patch_normal: Vec<(usize, f64)>,
    contact_axis_normals: Vec<[f32; 3]>,
}

impl SolverScratch {
    pub(crate) fn warm_state(&self) -> SolverWarmState {
        SolverWarmState {
            support_friction_modes: self.support_friction_modes.iter().map(|(k, v)| (*k, *v)).collect(),
            support_patch_normal: self.support_patch_normal.iter().map(|(k, v)| (*k, *v)).collect(),
            contact_axis_normals: self.contact_axis_normals.clone(),
        }
    }

    pub(crate) fn restore_warm_state(&mut self, state: SolverWarmState) {
        self.support_friction_modes = state.support_friction_modes.into_iter().collect();
        self.support_patch_normal = state.support_patch_normal.into_iter().collect();
        self.contact_axis_normals = state.contact_axis_normals;
    }
}

/// How one contact patch on a fixed support behaved in the previous pass.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq)]
enum SupportFriction {
    /// No tangential motion: the patch keeps its substep-start tangent position.
    Stick,
    /// Kinetic friction opposing motion along this unit tangential direction.
    Slip(Vec3),
}

/// One REV pressed against a fixed support on `axis`.
struct SupportRow {
    voxel: usize,
    axis: usize,
    normal: f32,
    friction: f32,
    patch: usize,
}

/// REVs on fixed supports, grouped into patches joined by intact bonds. A foot is
/// one patch; two feet of a body are two, and can slip independently.
fn support_rows(scene: &Scene, scratch: &SolverScratch, contacts: &[collision::Contact]) -> Vec<SupportRow> {
    let mut rows = Vec::<SupportRow>::new();
    let mut index = rustc_hash::FxHashMap::<usize, usize>::default();
    for contact in contacts {
        let abs_normal = contact.normal.abs();
        let axis = if abs_normal.x > 0.5 { 0 } else if abs_normal.y > 0.5 { 1 } else { 2 };
        let voxel = if scene.voxels[contact.i].kinematic {
            contact.j
        } else if scene.voxels[contact.j].kinematic {
            contact.i
        } else {
            continue;
        };
        let Some(normal) = scratch.contact_axis_normals.get(voxel).map(|axes| axes[axis]) else {
            continue;
        };
        if normal == 0.0 || scene.voxels[voxel].kinematic || index.contains_key(&voxel) {
            continue;
        }
        index.insert(voxel, rows.len());
        rows.push(SupportRow {
            voxel,
            axis,
            normal,
            friction: scene.config.fracture.friction
                * (scene.properties[contact.i].surface_friction_coefficient
                    + scene.properties[contact.j].surface_friction_coefficient),
            patch: rows.len(),
        });
    }
    fn root(rows: &mut [SupportRow], mut i: usize) -> usize {
        while rows[i].patch != i {
            let next = rows[i].patch;
            rows[i].patch = rows[next].patch;
            i = next;
        }
        i
    }
    for row in 0..rows.len() {
        let voxel = rows[row].voxel;
        for axis in 0..3 {
            let neighbor = scene.bonds[axis][voxel].neighbor;
            if neighbor == EMPTY {
                continue;
            }
            if let Some(&other) = index.get(&(neighbor as usize)) {
                let (a, b) = (root(&mut rows, row), root(&mut rows, other));
                rows[b].patch = a;
            }
        }
    }
    for row in 0..rows.len() {
        rows[row].patch = root(&mut rows, row);
    }
    rows
}

/// Reaction a solved row needs from its constraints:
/// `R = m (x - x_pred) / h^2 - load - bond forces`.
fn compute_contact_reactions(
    scene: &Scene,
    scratch: &mut SolverScratch,
    predicted: &[DVec3],
    loads: &[Vec3],
    h: f32,
) {
    let scale = 1.0 / f64::from(h).powi(2);
    scratch.contact_reactions.resize(scene.voxels.len(), DVec3::ZERO);
    for i in visit(&scene.step_live, scene.voxels.len()) {
        scratch.contact_reactions[i] = (scene.positions[i] - predicted[i])
            * (f64::from(scene.voxels[i].mass) * scale)
            - loads[i].as_dvec3();
    }
    for c in &scratch.constraints {
        let force = c.k.as_dmat3() * (scene.positions[c.j] - scene.positions[c.i] - c.rest);
        scratch.contact_reactions[c.i] -= force;
        scratch.contact_reactions[c.j] += force;
    }
}

/// Constrain or load the tangent axes of every support row for the next solve.
fn apply_support_friction(
    scene: &mut Scene,
    rows: &[SupportRow],
    modes: &rustc_hash::FxHashMap<usize, SupportFriction>,
    patch_normal: &rustc_hash::FxHashMap<usize, f64>,
    base_targets: &[[Option<f64>; 3]],
    substep_start: &[DVec3],
    contact_targets: &mut [[Option<f64>; 3]],
    applied_loads: &[Vec3],
    solve_loads: &mut [Vec3],
) {
    match scene.step_live.as_ref() {
        // A sleeper carries no load, and neither does its (zero) entry in `solve_loads`.
        Some(live) => {
            for &i in &live.list {
                solve_loads[i] = applied_loads[i];
            }
        }
        None => solve_loads.copy_from_slice(applied_loads),
    }
    let mut patch_rows = rustc_hash::FxHashMap::<usize, usize>::default();
    for row in rows {
        *patch_rows.entry(row.patch).or_default() += 1;
    }
    for row in rows {
        for axis in 0..3 {
            if axis != row.axis {
                contact_targets[row.voxel][axis] = base_targets[row.voxel][axis];
            }
        }
        match modes.get(&row.voxel) {
            Some(SupportFriction::Stick) => {
                for axis in 0..3 {
                    if axis != row.axis && base_targets[row.voxel][axis].is_none() {
                        let start = substep_start[row.voxel][axis];
                        contact_targets[row.voxel][axis] = Some(start);
                        scene.positions[row.voxel][axis] = start;
                    }
                }
            }
            Some(SupportFriction::Slip(direction)) => {
                let normal = patch_normal.get(&row.patch).copied().unwrap_or(0.0);
                let share = normal / patch_rows[&row.patch] as f64;
                solve_loads[row.voxel] -= *direction * (f64::from(row.friction) * share) as f32;
            }
            None => {}
        }
    }
}

/// Coulomb friction for contacts on fixed supports, solved in the implicit system
/// so stiff bonds and the weight above a contact are respected and the mechanical
/// energy it removes reaches the contact heat budget.
///
/// A patch sticks while its tangential reaction stays within `mu N`. Otherwise it
/// slides under kinetic friction until the motion reverses, then sticks. N and T
/// are per patch: single rows carry solver noise from stiff bonds that can exceed
/// a REV's weight, while bond forces cancel over a body. A body with one patch
/// therefore uses the exact body totals, and several patches are bounded by them.
/// Returns true when the mode set changed and another solve is required.
fn update_support_friction(
    scene: &Scene,
    rows: &[SupportRow],
    scratch: &SolverScratch,
    modes: &mut rustc_hash::FxHashMap<usize, SupportFriction>,
    patch_normal: &mut rustc_hash::FxHashMap<usize, f64>,
    substep_start: &[DVec3],
    h: f32,
) -> bool {
    use rustc_hash::FxHashMap;
    let reactions = &scratch.contact_reactions;
    let mut body_total = FxHashMap::<u32, DVec3>::default();
    for row in rows {
        body_total.entry(scene.clusters[row.voxel]).or_default();
    }
    for i in visit(&scene.step_live, scene.voxels.len()) {
        if scene.voxels[i].kinematic {
            continue;
        }
        if let Some(total) = body_total.get_mut(&scene.clusters[i]) {
            *total += reactions[i];
        }
    }
    struct Patch {
        normal: f64,
        tangential: DVec3,
        motion: DVec3,
        mass: f64,
        count: usize,
        cluster: u32,
        axis: usize,
        direction: f64,
        friction: f64,
    }
    let mut patches = FxHashMap::<usize, Patch>::default();
    let mut body_patches = FxHashMap::<u32, std::collections::BTreeSet<usize>>::default();
    for row in rows {
        let voxel = row.voxel;
        let cluster = scene.clusters[voxel];
        body_patches.entry(cluster).or_default().insert(row.patch);
        let patch = patches.entry(row.patch).or_insert(Patch {
            normal: 0.0,
            tangential: DVec3::ZERO,
            motion: DVec3::ZERO,
            mass: 0.0,
            count: 0,
            cluster,
            axis: row.axis,
            direction: f64::from(row.normal),
            friction: f64::from(row.friction),
        });
        let mut tangent = reactions[voxel];
        tangent[row.axis] = 0.0;
        let mut motion = scene.positions[voxel] - substep_start[voxel];
        motion[row.axis] = 0.0;
        patch.normal += (reactions[voxel][row.axis] * f64::from(row.normal)).max(0.0);
        patch.tangential += tangent;
        patch.motion += motion;
        patch.count += 1;
    }
    // Bound the noisy row sums by the exact body totals.
    for (cluster, members) in &body_patches {
        let total = body_total[cluster];
        let sum: f64 = members.iter().map(|id| patches[id].normal).sum();
        let first = &patches[members.iter().next().unwrap()];
        let body_normal = (total[first.axis] * first.direction).max(0.0);
        for id in members {
            let single = members.len() == 1;
            let patch = patches.get_mut(id).unwrap();
            if sum > body_normal && sum > 0.0 {
                patch.normal *= body_normal / sum;
            }
            if single {
                let mut tangent = total;
                tangent[patch.axis] = 0.0;
                patch.tangential = tangent;
            }
        }
    }
    let mut body_mass = FxHashMap::<u32, f64>::default();
    for (i, voxel) in scene.voxels.iter().enumerate() {
        if !voxel.kinematic && body_total.contains_key(&scene.clusters[i]) {
            *body_mass.entry(scene.clusters[i]).or_default() += f64::from(voxel.mass);
        }
    }
    for patch in patches.values_mut() {
        patch.mass = body_mass.get(&patch.cluster).copied().unwrap_or(0.0);
    }
    let h = f64::from(h);
    let mut changed = false;
    let mut next_modes = FxHashMap::<usize, SupportFriction>::default();
    for row in rows {
        let patch = &patches[&row.patch];
        let limit = patch.friction * patch.normal;
        let motion = patch.motion / patch.count.max(1) as f64;
        let speed = motion.length() / h;
        // Speed kinetic friction can remove from this body within one substep.
        let stoppable = if patch.mass > 0.0 { limit / patch.mass * h } else { 0.0 };
        let current = modes.get(&row.voxel).copied();
        let next = match current {
            Some(SupportFriction::Stick) => {
                if patch.tangential.length() <= limit * (1.0 + 1.0e-6) + 1.0e-12 {
                    SupportFriction::Stick
                } else {
                    // Slides against the reaction that could not be sustained.
                    SupportFriction::Slip((-patch.tangential.normalize()).as_vec3())
                }
            }
            Some(SupportFriction::Slip(direction)) => {
                if motion.dot(direction.as_dvec3()) <= 0.0 {
                    SupportFriction::Stick
                } else {
                    SupportFriction::Slip((motion / motion.length()).as_vec3())
                }
            }
            None => {
                if speed > stoppable.max(1.0e-9) && motion.length() > 0.0 {
                    SupportFriction::Slip((motion / motion.length()).as_vec3())
                } else {
                    SupportFriction::Stick
                }
            }
        };
        // Direction changes within a slide are applied on the next substep.
        let material_change = !matches!(
            (current, next),
            (Some(SupportFriction::Stick), SupportFriction::Stick)
                | (Some(SupportFriction::Slip(_)), SupportFriction::Slip(_))
        );
        changed |= material_change;
        next_modes.insert(row.voxel, next);
    }
    patch_normal.clear();
    for (id, patch) in &patches {
        patch_normal.insert(*id, patch.normal);
    }
    *modes = next_modes;
    changed
}

/// Rigid cells with face springs: each REV has a translation and a rotation. A bond
/// measures the gap at the shared face, so a rigid rotation produces no strain; the
/// face also carries bending `E L^3 / 12` and torsion. The older translation-only
/// solve resists torque-driven rotation with its shear, so it stays locked.
///
/// The cell solver is serial CPU code: a skyline factorisation for small and thin
/// bodies, block IC(0) otherwise. Above `CELL_DOF_MAX_REVS` mechanical REVs it takes
/// seconds per step (16k REVs: 1984 ms against 179 ms), so larger scenes keep the
/// translation-only solve, and the choice is announced once. A parallel or GPU cell
/// solver removes this limit. `VOXEL_FACE_CELLS=1` forces cells, `VOXEL_BOND_ONLY=1`
/// forbids them.
const CELL_DOF_MAX_REVS: usize = 4096;

/// Lets a test force cell rotations on the current thread regardless of size.
#[cfg(test)]
thread_local! {
    pub(crate) static FORCE_CELL_DOFS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Read-only: REVs the solver counts as mechanical (not fixed, not in a rigid
/// aggregate). This is the number `decide_cell_dofs` compares with its limit.
pub fn mechanical_rev_count(scene: &Scene) -> usize {
    scene
        .voxels
        .iter()
        .enumerate()
        .filter(|(i, voxel)| !voxel.kinematic && scene.aggregate_of[*i].is_none())
        .count()
}

/// Read-only: whether the solver uses cell rotations (true) or the
/// translation-only solve (false) for this scene, as decided by
/// `decide_cell_dofs` (including the `VOXEL_FACE_CELLS` / `VOXEL_BOND_ONLY`
/// overrides). Does not step or change the scene.
pub fn uses_cell_rotations(scene: &Scene) -> bool {
    decide_cell_dofs(scene)
}

fn decide_cell_dofs(scene: &Scene) -> bool {
    #[cfg(test)]
    if let Some(forced) = FORCE_CELL_DOFS.with(std::cell::Cell::get) {
        return forced;
    }
    if std::env::var_os("VOXEL_BOND_ONLY").is_some() {
        return false;
    }
    if std::env::var_os("VOXEL_FACE_CELLS").is_some() {
        return true;
    }
    let mechanical = visit(&scene.step_live, scene.voxels.len())
        .filter(|&i| !scene.voxels[i].kinematic && scene.aggregate_of[i].is_none())
        .count();
    let allowed = mechanical <= CELL_DOF_MAX_REVS;
    if !allowed {
        static ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "note: {mechanical} mechanical REVs exceed {CELL_DOF_MAX_REVS}; this scene uses the                  translation-only solver, so torque-driven rotation of large bodies stays locked"
            );
        }
    }
    allowed
}

/// Matrix of a bond's local axes for `axis`, in double precision.
fn basis_precise(axis: usize, rotation: glam::DQuat) -> glam::DMat3 {
    let axes = [DVec3::X, DVec3::Y, DVec3::Z];
    glam::DMat3::from_cols(
        rotation * axes[axis],
        rotation * axes[(axis + 1) % 3],
        rotation * axes[(axis + 2) % 3],
    )
}

/// Adopt rotations written elsewhere (editor, merges, fitted frames). Cell rotations
/// must be double precision: a substep's turn can be far below the 1e-7 rad step of
/// an f32 quaternion, and a turn the state cannot represent stays in the residual,
/// which grows without bound at small time steps.
fn sync_precise_rotations(scene: &mut Scene) {
    let n = scene.rotations.len();
    scene.precise_rotations.resize(n, glam::DQuat::IDENTITY);
    let live = scene.step_live.clone();
    for i in visit(&live, n) {
        if scene.precise_rotations[i].as_quat() != scene.rotations[i] {
            scene.precise_rotations[i] = scene.rotations[i].as_dquat().normalize();
        }
    }
}

fn turn_cell(scene: &mut Scene, i: usize, turn: DVec3) {
    let q = (glam::DQuat::from_scaled_axis(turn) * scene.precise_rotations[i]).normalize();
    scene.precise_rotations[i] = q;
    scene.rotations[i] = q.as_quat();
}

/// Bond rest vector at the face, world axes, metres, in double precision. Same terms
/// as `bond_thermal_rest_at_temperatures` plus formation offset and plastic strain.
fn face_rest(scene: &Scene, i: usize, j: usize, axis: usize, temperatures: [f32; 2]) -> DVec3 {
    let mid = scene.precise_rotations[i].slerp(scene.precise_rotations[j], 0.5);
    let frame = basis_precise(axis, mid);
    let direction = frame.x_axis;
    let expansion = |n: usize, temperature: f32| {
        let material = &scene.properties[n];
        let alpha_parallel = f64::from(material.thermal_expansion_parallel);
        let alpha_perpendicular = f64::from(material.thermal_expansion_perpendicular);
        let fiber = (scene.precise_rotations[n] * scene.voxels[n].microstructure.fibers().as_dvec3())
            .normalize_or_zero();
        (direction * alpha_perpendicular
            + fiber * ((alpha_parallel - alpha_perpendicular) * fiber.dot(direction)))
            * (f64::from(temperature) - f64::from(scene.reference_temperatures_kelvin[n]))
    };
    let bond = &scene.bonds[axis][i];
    direction * DCELL
        + (expansion(i, temperatures[0]) + expansion(j, temperatures[1])) * (0.5 * DCELL)
        + frame
            * (DVec3::from_array(bond.formation_offset.map(f64::from))
                + DVec3::from_array(bond.plastic_strain.map(f64::from)) * DCELL)
}

/// Bending and torsion stiffness of the face between two cells, N m/rad. Uses the
/// softer material of the pair, like a joint limited by its weaker side.
fn face_rotation_stiffness(scene: &Scene, i: usize, j: usize, direction: DVec3, damage: f32) -> glam::DMat3 {
    let cell = f64::from(CELL);
    let young = f64::from(scene.properties[i].young.min(scene.properties[j].young));
    let shear = f64::from(scene.properties[i].shear.min(scene.properties[j].shear));
    let intact = f64::from(1.0 - damage);
    let bend = young * cell.powi(3) / 12.0 * intact;
    let twist = shear * 0.1406 * cell.powi(3) * intact;
    glam::DMat3::from_diagonal(DVec3::splat(bend))
        + glam::DMat3::from_cols(direction * direction.x, direction * direction.y, direction * direction.z)
            * (twist - bend)
}

/// Relative rotation across a face as a scaled axis, radians.
fn face_bend(scene: &Scene, i: usize, j: usize) -> DVec3 {
    let relative = scene.precise_rotations[j] * scene.precise_rotations[i].inverse();
    let relative = if relative.w < 0.0 { -relative } else { relative };
    relative.to_scaled_axis()
}

/// Kinetic energy of a cell's own rotation about its centre (a cube of side CELL).
fn cell_spin_energy_j(scene: &Scene, i: usize) -> f64 {
    let inertia = f64::from(scene.voxels[i].mass) * f64::from(CELL) * f64::from(CELL) / 6.0;
    0.5 * inertia * scene.angular_velocities[i].length_squared()
}

pub(crate) type Block6 = [[f64; 6]; 6];
pub(crate) type Vec6 = [f64; 6];

pub(crate) fn block6_from(tl: glam::DMat3, tr: glam::DMat3, bl: glam::DMat3, br: glam::DMat3) -> Block6 {
    let mut out = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = tl.col(col)[row];
            out[row][col + 3] = tr.col(col)[row];
            out[row + 3][col] = bl.col(col)[row];
            out[row + 3][col + 3] = br.col(col)[row];
        }
    }
    out
}

pub(crate) fn block6_add(a: &mut Block6, b: &Block6) {
    for row in 0..6 {
        for col in 0..6 {
            a[row][col] += b[row][col];
        }
    }
}

pub(crate) fn block6_mul_vec(a: &Block6, v: &Vec6) -> Vec6 {
    std::array::from_fn(|row| (0..6).map(|col| a[row][col] * v[col]).sum())
}

pub(crate) fn block6_transpose_mul_vec(a: &Block6, v: &Vec6) -> Vec6 {
    std::array::from_fn(|col| (0..6).map(|row| a[row][col] * v[row]).sum())
}

/// Lower Cholesky factor of a symmetric block, or None if it is not positive.
pub(crate) fn block6_cholesky(a: &Block6) -> Option<Block6> {
    let mut l = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..=row {
            let mut sum = a[row][col];
            for k in 0..col {
                sum -= l[row][k] * l[col][k];
            }
            if row == col {
                if !(sum > 0.0) {
                    return None;
                }
                l[row][col] = sum.sqrt();
            } else {
                l[row][col] = sum / l[col][col];
            }
        }
    }
    Some(l)
}

pub(crate) fn block6_solve_lower(l: &Block6, v: &Vec6) -> Vec6 {
    let mut out = *v;
    for row in 0..6 {
        for col in 0..row {
            out[row] -= l[row][col] * out[col];
        }
        out[row] /= l[row][row];
    }
    out
}

pub(crate) fn block6_solve_lower_transpose(l: &Block6, v: &Vec6) -> Vec6 {
    let mut out = *v;
    for row in (0..6).rev() {
        for col in row + 1..6 {
            out[row] -= l[col][row] * out[col];
        }
        out[row] /= l[row][row];
    }
    out
}

/// Complete skyline Cholesky of the 6x6 cell matrix in reverse Cuthill-McKee order.
/// It stays a valid SPD preconditioner after stiffness changes, so one factor serves
/// many solves with a few iterations, and thin bodies, whose conditioning defeats
/// IC(0), become cheap: their small bandwidth keeps the fill low.
pub(crate) struct CellFactor {
    position: Vec<u32>,
    first: Vec<u32>,
    offset: Vec<usize>,
    lower: Vec<Block6>,
    graph: Vec<(u32, u32)>,
}

impl CellFactor {
    /// Upper bound on factorisation work this path will accept per build, in
    /// floating-point multiplications. A bulk body of 1600 cells costs 3.5e9 and
    /// then 1.9 s per step; thin bodies such as a limb stay far below this.
    const MAX_WORK: f64 = 6.0e7;

    fn block_at(&self, row: usize, col: usize) -> usize {
        self.offset[row] + col - self.first[row] as usize
    }

    pub(crate) fn build(
        count: usize,
        diagonal: &[Block6],
        edges: &[(u32, u32, Block6)],
        masks: &[[f64; 6]],
    ) -> Option<Self> {
        let mut neighbours = vec![Vec::<u32>::new(); count];
        for &(row, col, _) in edges {
            neighbours[row as usize].push(col);
            neighbours[col as usize].push(row);
        }
        let mut order = Vec::with_capacity(count);
        let mut seen = vec![false; count];
        let mut starts: Vec<u32> = (0..count as u32).collect();
        starts.sort_by_key(|&cell| neighbours[cell as usize].len());
        for start in starts {
            if seen[start as usize] {
                continue;
            }
            seen[start as usize] = true;
            let mut head = order.len();
            order.push(start);
            while head < order.len() {
                let mut next: Vec<u32> = neighbours[order[head] as usize]
                    .iter()
                    .copied()
                    .filter(|&cell| !seen[cell as usize])
                    .collect();
                next.sort_by_key(|&cell| neighbours[cell as usize].len());
                for cell in next {
                    if !seen[cell as usize] {
                        seen[cell as usize] = true;
                        order.push(cell);
                    }
                }
                head += 1;
            }
        }
        order.reverse();
        let mut position = vec![0u32; count];
        for (index, &cell) in order.iter().enumerate() {
            position[cell as usize] = index as u32;
        }
        let mut first: Vec<u32> = (0..count as u32).collect();
        for &(row, col, _) in edges {
            let (a, b) = (position[row as usize], position[col as usize]);
            let (high, low) = if a > b { (a, b) } else { (b, a) };
            first[high as usize] = first[high as usize].min(low);
        }
        let mut offset = vec![0usize; count + 1];
        let mut work = 0.0f64;
        for row in 0..count {
            let width = row - first[row] as usize + 1;
            offset[row + 1] = offset[row] + width;
            work += (width * width) as f64 * 216.0;
        }
        if work > Self::MAX_WORK {
            return None;
        }
        let mut lower = vec![[[0.0; 6]; 6]; offset[count]];
        let mut factor = Self {
            position,
            first,
            offset,
            lower: Vec::new(),
            graph: edges.iter().map(|edge| (edge.0, edge.1)).collect(),
        };
        for cell in 0..count {
            let row = factor.position[cell] as usize;
            lower[factor.block_at(row, row)] = diagonal[cell];
        }
        for (row, col, block) in edges {
            let (a, b) = (
                factor.position[*row as usize] as usize,
                factor.position[*col as usize] as usize,
            );
            if a > b {
                lower[factor.block_at(a, b)] = *block;
            } else {
                let mut transposed = [[0.0; 6]; 6];
                for p in 0..6 {
                    for q in 0..6 {
                        transposed[p][q] = block[q][p];
                    }
                }
                lower[factor.block_at(b, a)] = transposed;
            }
        }
        for row in 0..count {
            let row_first = factor.first[row] as usize;
            for col in row_first..=row {
                let col_first = factor.first[col] as usize;
                let mut value = lower[factor.block_at(row, col)];
                for k in row_first.max(col_first)..col {
                    let a = lower[factor.block_at(row, k)];
                    let b = lower[factor.block_at(col, k)];
                    for p in 0..6 {
                        for q in 0..6 {
                            value[p][q] -= (0..6).map(|t| a[p][t] * b[q][t]).sum::<f64>();
                        }
                    }
                }
                if col == row {
                    lower[factor.block_at(row, row)] = block6_cholesky(&value)?;
                } else {
                    let pivot = lower[factor.block_at(col, col)];
                    let mut solved = [[0.0; 6]; 6];
                    for p in 0..6 {
                        solved[p] = block6_solve_lower(&pivot, &value[p]);
                    }
                    lower[factor.block_at(row, col)] = solved;
                }
            }
        }
        factor.lower = lower;
        Some(factor)
    }

    pub(crate) fn matches(&self, edges: &[(u32, u32, Block6)], masks: &[[f64; 6]]) -> bool {
        // A stale mask is fine: `solve` is applied between masked vectors, which keeps
        // the preconditioner SPD on the current free subspace.
        let _ = masks;
        self.graph.len() == edges.len()
            && self.graph.iter().zip(edges).all(|(a, b)| a.0 == b.0 && a.1 == b.1)
    }

    pub(crate) fn solve(&self, r: &[Vec6], z: &mut [Vec6], work: &mut Vec<Vec6>) {
        let count = r.len();
        work.resize(count, [0.0; 6]);
        for cell in 0..count {
            work[self.position[cell] as usize] = r[cell];
        }
        for row in 0..count {
            let mut value = work[row];
            for col in self.first[row] as usize..row {
                let product = block6_mul_vec(&self.lower[self.block_at(row, col)], &work[col]);
                for p in 0..6 {
                    value[p] -= product[p];
                }
            }
            work[row] = block6_solve_lower(&self.lower[self.block_at(row, row)], &value);
        }
        for row in (0..count).rev() {
            let solved = block6_solve_lower_transpose(&self.lower[self.block_at(row, row)], &work[row]);
            work[row] = solved;
            for col in self.first[row] as usize..row {
                let product =
                    block6_transpose_mul_vec(&self.lower[self.block_at(row, col)], &solved);
                for p in 0..6 {
                    work[col][p] -= product[p];
                }
            }
        }
        for cell in 0..count {
            z[cell] = work[self.position[cell] as usize];
        }
    }
}

/// The first cell system built after `armed` is set, for preconditioner tests.
#[cfg(test)]
pub(crate) struct CapturedCellSystem {
    pub(crate) diagonal: Vec<Block6>,
    pub(crate) edges: Vec<(u32, u32, Block6)>,
    pub(crate) masks: Vec<[f64; 6]>,
    pub(crate) rhs: Vec<Vec6>,
    pub(crate) positions: Vec<DVec3>,
    pub(crate) lattice: Vec<DVec3>,
    pub(crate) bodies: Vec<u32>,
}

/// Systems collected while a test has set a capture limit.
#[cfg(test)]
pub(crate) struct CellCapture {
    pub(crate) limit: usize,
    pub(crate) systems: Vec<CapturedCellSystem>,
}

#[cfg(test)]
thread_local! {
    pub(crate) static CAPTURED_CELL_SYSTEM: std::cell::RefCell<Option<CellCapture>> =
        const { std::cell::RefCell::new(None) };
}

/// Nanoseconds spent in the phases of the cell solve, and the number of solves:
/// block assembly, preconditioner build, conjugate gradients, write-back.
pub static CELL_PROFILE: [std::sync::atomic::AtomicU64; 8] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn cell_profile_add(phase: usize, started: Instant) {
    CELL_PROFILE[phase].fetch_add(
        started.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Solve one cell update: PCG on 6x6 cell blocks, preconditioned by a reused complete
/// factor when the fill is affordable, otherwise by block IC(0). Blocks of fixed degrees of freedom are eliminated by
/// zeroing their rows and columns. On a face-connected grid two neighbours share
/// no third neighbour, so IC(0) has no fill terms between lower blocks.
fn solve_face_cells_sparse(
    scene: &mut Scene,
    h: f32,
    scratch: &mut SolverScratch,
    new_substep: bool,
    predicted: &[DVec3],
    loads: &[Vec3],
) -> usize {
    use crate::contact_constraint::ContactProjection;
    let phase_started = Instant::now();
    let n = scene.voxels.len();
    let h2 = f64::from(h) * f64::from(h);
    let cell = f64::from(CELL);
    scratch.substep_theta.resize(n, DVec3::ZERO);
    sync_precise_rotations(scene);
    let live = scene.step_live.clone();
    if new_substep {
        sync_visited(&mut scratch.spin_at_substep_start, &scene.angular_velocities, &live);
        for i in visit(&live, n) {
            scratch.substep_theta[i] = DVec3::ZERO;
            if !scene.voxels[i].kinematic {
                let step = scene.angular_velocities[i] * f64::from(h);
                turn_cell(scene, i, step);
                scratch.substep_theta[i] = step;
            }
        }
    }
    // Cells that take part: any cell with a free translation or rotation.
    if scratch.spin_at_substep_start.len() != n {
        scratch.spin_at_substep_start.resize(n, DVec3::ZERO);
    }
    let mut cell_of = std::mem::take(&mut scratch.cell_of);
    fill_visited(&mut cell_of, n, u32::MAX, &live);
    let mut cells = Vec::<usize>::new();
    for i in visit(&live, n) {
        if !scene.voxels[i].kinematic && scene.aggregate_of[i].is_none() {
            cell_of[i] = cells.len() as u32;
            cells.push(i);
        }
    }
    let count = cells.len();
    if count == 0 {
        scene.solver_residual = 0.0;
        scratch.cell_of = cell_of;
        return 0;
    }
    let masks: Vec<[f64; 6]> = cells
        .iter()
        .map(|&i| {
            let free = scratch.free[i];
            [f64::from(free.x), f64::from(free.y), f64::from(free.z), 1.0, 1.0, 1.0]
        })
        .collect();
    let mut diagonal = vec![[[0.0; 6]; 6]; count];
    let mut rhs = vec![[0.0; 6]; count];
    for (slot, &i) in cells.iter().enumerate() {
        let inertia = f64::from(scene.voxels[i].mass) * cell * cell / 6.0;
        let translation = glam::DMat3::from_diagonal(DVec3::splat(f64::from(scene.voxels[i].mass) / h2));
        let rotation = glam::DMat3::from_diagonal(DVec3::splat(inertia / h2));
        diagonal[slot] = block6_from(translation, glam::DMat3::ZERO, glam::DMat3::ZERO, rotation);
        // The inertial target stays fixed for every solve of this substep.
        let target = scratch.spin_at_substep_start[i] * f64::from(h);
        let inertial = (predicted[i] - scene.positions[i]) * (f64::from(scene.voxels[i].mass) / h2)
            + loads.get(i).map_or(DVec3::ZERO, |load| load.as_dvec3());
        for axis in 0..3 {
            rhs[slot][axis] = inertial[axis];
            rhs[slot][axis + 3] = -inertia / h2 * (scratch.substep_theta[i][axis] - target[axis]);
        }
    }
    let skew = |v: DVec3| {
        glam::DMat3::from_cols(
            DVec3::new(0.0, v.z, -v.y),
            DVec3::new(-v.z, 0.0, v.x),
            DVec3::new(v.y, -v.x, 0.0),
        )
    };
    // Per-bond blocks are independent, so they are computed in parallel and then
    // added to their cells in bond order, which keeps the sums deterministic.
    struct BondBlocks {
        slot_i: u32,
        slot_j: u32,
        h_ii: Block6,
        h_jj: Block6,
        edge: Option<(u32, u32, Block6)>,
        rhs_i: Vec6,
        rhs_j: Vec6,
    }
    let scene_ref: &Scene = scene;
    let bond_blocks = |c: &Constraint| -> Option<BondBlocks> {
        let (i, j) = (c.i, c.j);
        let (slot_i, slot_j) = (cell_of[i], cell_of[j]);
        if slot_i == u32::MAX && slot_j == u32::MAX {
            return None;
        }
        let r = c.rest;
        let k = c.k.as_dmat3();
        let half = skew(r) * 0.5;
        let k_s = k * half;
        let s_k_s = half.transpose() * k_s;
        let stiff = face_rotation_stiffness(scene_ref, i, j, r.normalize_or_zero(), c.damage);
        // G_i = [-I, S], G_j = [I, S]; H = G^T k G plus face rotation stiffness.
        let h_ii = block6_from(k, -k_s, -k_s.transpose(), s_k_s + stiff);
        let h_jj = block6_from(k, k_s, k_s.transpose(), s_k_s + stiff);
        let h_ij = block6_from(-k, -k_s, k_s.transpose(), s_k_s - stiff);
        let edge = if slot_i != u32::MAX && slot_j != u32::MAX {
            if slot_j > slot_i {
                let mut transposed = [[0.0; 6]; 6];
                for row in 0..6 {
                    for col in 0..6 {
                        transposed[row][col] = h_ij[col][row];
                    }
                }
                Some((slot_j, slot_i, transposed))
            } else {
                Some((slot_i, slot_j, h_ij))
            }
        } else {
            None
        };
        // Forces and moments on both cells, all in double precision.
        let force = k * (scene_ref.positions[j] - scene_ref.positions[i] - r);
        let moment = half.transpose() * force;
        let couple = stiff * face_bend(scene_ref, i, j);
        let mut rhs_i = [0.0; 6];
        let mut rhs_j = [0.0; 6];
        for p in 0..3 {
            rhs_i[p] = force[p];
            rhs_j[p] = -force[p];
            rhs_i[p + 3] = couple[p] - moment[p];
            rhs_j[p + 3] = -(couple[p] + moment[p]);
        }
        Some(BondBlocks { slot_i, slot_j, h_ii, h_jj, edge, rhs_i, rhs_j })
    };
    let computed: Vec<Option<BondBlocks>> = if scratch.constraints.len() >= 512 {
        scratch.constraints.par_iter().map(bond_blocks).collect()
    } else {
        scratch.constraints.iter().map(bond_blocks).collect()
    };
    let mut edges = Vec::<(u32, u32, Block6)>::with_capacity(computed.len());
    for blocks in computed.into_iter().flatten() {
        if blocks.slot_i != u32::MAX {
            let slot = blocks.slot_i as usize;
            block6_add(&mut diagonal[slot], &blocks.h_ii);
            for p in 0..6 {
                rhs[slot][p] += blocks.rhs_i[p];
            }
        }
        if blocks.slot_j != u32::MAX {
            let slot = blocks.slot_j as usize;
            block6_add(&mut diagonal[slot], &blocks.h_jj);
            for p in 0..6 {
                rhs[slot][p] += blocks.rhs_j[p];
            }
        }
        if let Some(edge) = blocks.edge {
            edges.push(edge);
        }
    }
    // Eliminate fixed axes.
    for slot in 0..count {
        for p in 0..6 {
            if masks[slot][p] == 0.0 {
                rhs[slot][p] = 0.0;
                for q in 0..6 {
                    diagonal[slot][p][q] = 0.0;
                    diagonal[slot][q][p] = 0.0;
                }
                diagonal[slot][p][p] = 1.0;
            }
        }
    }
    for (row, col, block) in &mut edges {
        for p in 0..6 {
            for q in 0..6 {
                if masks[*row as usize][p] == 0.0 || masks[*col as usize][q] == 0.0 {
                    block[p][q] = 0.0;
                }
            }
        }
    }
    edges.sort_by_key(|edge| (edge.0, edge.1));
    #[cfg(test)]
    CAPTURED_CELL_SYSTEM.with(|slot| {
        if let Some(capture) = slot.borrow_mut().as_mut() {
            if capture.systems.len() < capture.limit {
                capture.systems.push(CapturedCellSystem {
                    diagonal: diagonal.clone(),
                    edges: edges.clone(),
                    masks: masks.clone(),
                    rhs: rhs.clone(),
                    positions: cells.iter().map(|&i| scene.positions[i]).collect(),
                    lattice: cells.iter().map(|&i| scene.rest_positions[i]).collect(),
                    bodies: cells.iter().map(|&i| scene.clusters[i]).collect(),
                });
            }
        }
    });
    // Reuse a complete factor while the structure holds and it keeps solves short.
    let mut cell_factor = scratch.cell_factor.take();
    let force_two_level = std::env::var_os("VOXEL_CELL_TWO_LEVEL").is_some();
    if force_two_level {
        cell_factor = None;
    } else if !cell_factor
        .as_ref()
        .is_some_and(|factor| factor.matches(&edges, &masks) && scratch.cell_factor_iterations <= 40)
    {
        cell_factor = CellFactor::build(count, &diagonal, &edges, &masks);
    }
    let mut cell_factor_work = Vec::new();
    let system = crate::cell_two_level::CellSystem::new(&diagonal, &edges);
    cell_profile_add(0, phase_started);
    let build_started = Instant::now();
    // Contacts between two free bodies equate normal translations: a projected
    // solve on the translation part, with a particular solution that closes any
    // overlap, exactly as for the translation-only system.
    let mut projection = None;
    let mut particular = vec![[0.0f64; 6]; count];
    if scratch.pair_contacts.iter().any(|pair| pair.active) {
        let mut slot_of = vec![u32::MAX; n];
        for (slot, &i) in cells.iter().enumerate() {
            slot_of[i] = slot as u32;
        }
        let built = ContactProjection::build(&scratch.pair_contacts, &scratch.free, &slot_of);
        let targets: Vec<f64> = scratch
            .pair_contacts
            .iter()
            .map(|pair| {
                if pair.active {
                    DCELL
                        - (scene.positions[pair.j][pair.axis] - scene.positions[pair.i][pair.axis])
                            * pair.sign
                } else {
                    0.0
                }
            })
            .collect();
        let start = built.particular(&scratch.pair_contacts, &targets, count);
        for (slot, value) in start.iter().enumerate() {
            particular[slot][0] = f64::from(value.x);
            particular[slot][1] = f64::from(value.y);
            particular[slot][2] = f64::from(value.z);
        }
        let mut product = vec![[0.0f64; 6]; count];
        system.apply(&particular, &mut product);
        for slot in 0..count {
            for t in 0..6 {
                rhs[slot][t] -= product[slot][t];
            }
        }
        built.project_cells(&mut rhs);
        projection = Some(built);
    }
    // A bulk body is too wide for a skyline factor: use the parallel two-level
    // preconditioner instead of serial IC(0).
    let mut two_level = scratch.two_level.take();
    if cell_factor.is_none() {
        if !two_level.as_ref().is_some_and(|built| {
            built.matches(count, edges.len(), &masks) && scratch.cell_factor_iterations <= 30
        }) {
            let positions: Vec<DVec3> = cells.iter().map(|&i| scene.positions[i]).collect();
            let lattice: Vec<DVec3> = cells.iter().map(|&i| scene.rest_positions[i]).collect();
            let bodies: Vec<u32> = cells.iter().map(|&i| scene.clusters[i]).collect();
            two_level = crate::cell_two_level::TwoLevel::build(
                &system, &edges, &positions, &lattice, &bodies, &masks, cell,
                crate::cell_two_level::DEFAULT_DAMPING,
            );
        }
    } else {
        two_level = None;
    }
    let mut two_level_work = std::mem::take(&mut scratch.two_level_work);
    // Block IC(0); falls back to block Jacobi when a pivot breaks down.
    let mut factor = vec![[[0.0; 6]; 6]; count];
    let mut lower = vec![[[0.0; 6]; 6]; edges.len()];
    let mut row_start = vec![0usize; count + 1];
    for edge in &edges {
        row_start[edge.0 as usize + 1] += 1;
    }
    for row in 0..count {
        row_start[row + 1] += row_start[row];
    }
    let mut jacobi = false;
    'factor: for row in 0..if cell_factor.is_some() || two_level.is_some() { 0 } else { count } {
        let mut pivot = diagonal[row];
        for e in row_start[row]..row_start[row + 1] {
            let col = edges[e].1 as usize;
            // L = H L_col^-T, solved row by row.
            for p in 0..6 {
                lower[e][p] = block6_solve_lower(&factor[col], &edges[e].2[p]);
            }
            for p in 0..6 {
                for q in 0..6 {
                    pivot[p][q] -= (0..6).map(|t| lower[e][p][t] * lower[e][q][t]).sum::<f64>();
                }
            }
        }
        match block6_cholesky(&pivot) {
            Some(value) => factor[row] = value,
            None => {
                jacobi = true;
                break 'factor;
            }
        }
    }
    if jacobi && two_level.is_none() {
        for row in 0..count {
            factor[row] = match block6_cholesky(&diagonal[row]) {
                Some(value) => value,
                None => {
                    scene.solver_residual = f32::INFINITY;
                    return 0;
                }
            };
        }
    }
    cell_profile_add(1, build_started);
    let cg_started = Instant::now();
    let apply = |v: &[Vec6], out: &mut [Vec6]| {
        let started = Instant::now();
        system.apply(v, out);
        if let Some(projection) = &projection {
            projection.project_cells(out);
        }
        cell_profile_add(5, started);
    };
    let mut precondition_unprojected = |r: &[Vec6], z: &mut [Vec6]| {
        if let Some(two) = &two_level {
            two.apply(&system, r, z, &mut two_level_work);
            // The coarse space may predate a contact row that has since fixed
            // a degree of freedom: project onto the current free subspace.
            for slot in 0..count {
                for p in 0..6 {
                    z[slot][p] *= masks[slot][p];
                }
            }
            return;
        }
        if let Some(complete) = &cell_factor {
            complete.solve(r, z, &mut cell_factor_work);
            for slot in 0..count {
                for p in 0..6 {
                    z[slot][p] *= masks[slot][p];
                }
            }
            return;
        }
        if jacobi {
            for slot in 0..count {
                z[slot] = block6_solve_lower_transpose(&factor[slot], &block6_solve_lower(&factor[slot], &r[slot]));
            }
            return;
        }
        for row in 0..count {
            let mut value = r[row];
            for e in row_start[row]..row_start[row + 1] {
                let product = block6_mul_vec(&lower[e], &z[edges[e].1 as usize]);
                for p in 0..6 {
                    value[p] -= product[p];
                }
            }
            z[row] = block6_solve_lower(&factor[row], &value);
        }
        for row in (0..count).rev() {
            let solved = block6_solve_lower_transpose(&factor[row], &z[row]);
            z[row] = solved;
            for e in row_start[row]..row_start[row + 1] {
                let col = edges[e].1 as usize;
                let product = block6_transpose_mul_vec(&lower[e], &solved);
                for p in 0..6 {
                    z[col][p] -= product[p];
                }
            }
        }
    };
    let mut precondition = |r: &[Vec6], z: &mut [Vec6]| {
        let started = Instant::now();
        precondition_unprojected(r, z);
        cell_profile_add(6, started);
        if let Some(projection) = &projection {
            projection.project_cells(z);
        }
    };
    let dot = |a: &[Vec6], b: &[Vec6]| crate::cell_two_level::dot(a, b);
    // The translation and rotation parts have different units. Their inner
    // product is fine for CG, but the stop test uses the force part alone so it
    // keeps the meaning of `solver_relative_tolerance`.
    let force_norm = |a: &[Vec6]| {
        a.iter()
            .map(|x| (0..3).map(|p| x[p] * x[p]).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    };
    let moment_norm = |a: &[Vec6]| {
        a.iter()
            .map(|x| (3..6).map(|p| x[p] * x[p]).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    };
    let mut x = vec![[0.0; 6]; count];
    let mut r = rhs.clone();
    let mut z = vec![[0.0; 6]; count];
    let mut ap = vec![[0.0; 6]; count];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let (force0, moment0) = (force_norm(&r), moment_norm(&r));
    let tolerance = f64::from(scene.config.solver_relative_tolerance);
    let limit = scene.config.iterations.max(1) * 10;
    let converged = |r: &[Vec6]| {
        force_norm(r) <= tolerance * force0.max(1.0e-12)
            && moment_norm(r) <= tolerance * moment0.max(1.0e-12)
    };
    let mut iterations = 0;
    while !converged(&r) && iterations < limit {
        apply(&p, &mut ap);
        let denominator = dot(&p, &ap);
        iterations += 1;
        if !(denominator > 0.0) || !denominator.is_finite() {
            break;
        }
        let alpha = rz / denominator;
        let advance = |((x, r), (p, ap)): ((&mut Vec6, &mut Vec6), (&Vec6, &Vec6))| {
            for t in 0..6 {
                x[t] += alpha * p[t];
                r[t] -= alpha * ap[t];
            }
        };
        if count >= 1024 {
            x.par_iter_mut().zip(r.par_iter_mut()).zip(p.par_iter().zip(ap.par_iter())).for_each(advance);
        } else {
            x.iter_mut().zip(r.iter_mut()).zip(p.iter().zip(ap.iter())).for_each(advance);
        }
        precondition(&r, &mut z);
        let next = dot(&r, &z);
        let beta = next / rz;
        rz = next;
        let direction = |(p, z): (&mut Vec6, &Vec6)| {
            for t in 0..6 {
                p[t] = z[t] + beta * p[t];
            }
        };
        if count >= 1024 {
            p.par_iter_mut().zip(z.par_iter()).for_each(direction);
        } else {
            p.iter_mut().zip(z.iter()).for_each(direction);
        }
    }
    cell_profile_add(2, cg_started);
    scratch.cell_of = std::mem::take(&mut cell_of);
    CELL_PROFILE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    CELL_PROFILE[3].fetch_add(iterations as u64, std::sync::atomic::Ordering::Relaxed);
    scratch.cell_factor_iterations = iterations;
    scratch.cell_factor = cell_factor;
    scratch.two_level = two_level;
    scratch.two_level_work = two_level_work;
    if x.iter().flatten().any(|value| !value.is_finite()) {
        scene.solver_residual = f32::INFINITY;
        return iterations;
    }
    scene.solver_residual = (force_norm(&r) / force0.max(1.0e-12)) as f32;
    scratch.last_force_residual_n = force_norm(&r);
    for (slot, &i) in cells.iter().enumerate() {
        for t in 0..6 {
            x[slot][t] += particular[slot][t];
        }
        scene.positions[i] += DVec3::new(x[slot][0], x[slot][1], x[slot][2]);
        let turn = DVec3::new(x[slot][3], x[slot][4], x[slot][5]);
        if turn != DVec3::ZERO {
            turn_cell(scene, i, turn);
            scratch.substep_theta[i] += turn;
        }
        scene.angular_velocities[i] = scratch.substep_theta[i] / f64::from(h);
    }
    iterations
}

/// List the REVs and bond slots that can take part in mechanics. A bond between
/// two kinematic REVs has no constraint or fracture owner, so a large fixed floor
/// must not cost per-REV constitutive work on every assembly. Called at the start
/// of a substep: bonds and kinematic flags do not change before its solves end.
/// Fixed rows stay at the identity block and a zero right-hand side.
fn refresh_mechanical_sets(scene: &Scene, scratch: &mut SolverScratch) {
    let n = scene.voxels.len();
    let live = scene.step_live.clone();
    // A sleeper is kinematic with no mechanical bond: mask false, slots empty, row at
    // identity. Those entries never change, so a sparse refresh leaves them alone.
    if live.is_some() && scratch.mechanical_mask.len() == n && scratch.constraint_slots.len() == 3 * n {
        for i in visit(&live, n) {
            scratch.mechanical_mask[i] = !scene.voxels[i].kinematic;
        }
    } else {
        scratch.mechanical_mask.clear();
        scratch
            .mechanical_mask
            .extend(scene.voxels.iter().map(|voxel| !voxel.kinematic));
    }
    scratch.constraint_slots.resize_with(3 * n, || None);
    scratch.mechanical_slots.clear();
    for axis in 0..3 {
        for i in visit(&live, n) {
            let neighbor = scene.bonds[axis][i].neighbor;
            let slot = axis * n + i;
            if neighbor == EMPTY {
                scratch.constraint_slots[slot] = None;
                continue;
            }
            let j = neighbor as usize;
            if scene.voxels[i].kinematic && scene.voxels[j].kinematic {
                scratch.constraint_slots[slot] = None;
                continue;
            }
            scratch.mechanical_mask[i] = true;
            scratch.mechanical_mask[j] = true;
            scratch.mechanical_slots.push(slot);
        }
    }
    scratch.mechanical_revs.clear();
    scratch
        .mechanical_revs
        .extend(visit(&live, n).filter(|&i| scratch.mechanical_mask[i]));
    fill_visited(&mut scratch.free, n, Vec3::ZERO, &live);
    fill_visited(&mut scratch.diagonal, n, Mat3::IDENTITY, &live);
    fill_visited(&mut scratch.rhs, n, Vec3::ZERO, &live);
}

fn solve_coupled(
    scene: &mut Scene,
    h: f32,
    targets: &[[Option<f64>; 3]],
    loads: &[Vec3],
    predicted: &[DVec3],
    scratch: &mut SolverScratch,
    reuse_preconditioner: bool,
    reuse_previous_solution: bool,
    cuda_enabled_for_step: &mut bool,
) -> (usize, [Duration; 3]) {
    let assembly_start = Instant::now();
    scene.solver_residual_audit = None;
    let n = scene.voxels.len();
    if scene.has_registered_rigid_candidates() {
        let no_fine_degrees_of_freedom = (0..n).all(|i| {
            scene.aggregate_of[i].is_some()
                || scene.voxels[i].kinematic
                || targets[i].iter().all(Option::is_some)
        });
        let bonds_are_coarse = scene.bonds.iter().all(|bonds| {
            bonds.iter().enumerate().all(|(i, bond)| {
                if bond.neighbor == EMPTY {
                    return true;
                }
                // Kinematic-only bonds have no mechanical unknown and their
                // owner is excluded from fracture updates. An unrelated fixed
                // floor must not force assembly of every coarse body's bonds.
                if scene.voxels[i].kinematic
                    && scene.voxels[bond.neighbor as usize].kinematic
                {
                    return true;
                }
                let Some(id) = scene.aggregate_of[i] else {
                    return false;
                };
                scene.aggregate_of[bond.neighbor as usize] == Some(id)
            })
        });
        if no_fine_degrees_of_freedom && bonds_are_coarse {
            // Aggregates already advanced from their conserved rigid-body
            // momentum. Their retained REV bonds must not be rebuilt and
            // re-evaluated as fine constraints every step; they are restored
            // unchanged if a later interaction wakes the component.
            scratch.constraints.clear();
            scratch.constraint_forces.clear();
            scratch.assembly_phase_strengths.resize(n, (1.0, 1.0));
            scene.solver_residual = 0.0;
            return (
                0,
                [assembly_start.elapsed(), Duration::ZERO, Duration::ZERO],
            );
        }
    }
    if !reuse_preconditioner || scratch.mechanical_mask.len() != n {
        refresh_mechanical_sets(scene, scratch);
    }
    let mechanical_revs = std::mem::take(&mut scratch.mechanical_revs);
    let mechanical_slots = std::mem::take(&mut scratch.mechanical_slots);
    for &i in &mechanical_revs {
        let v = &scene.voxels[i];
        scratch.free[i] = Vec3::from_array(std::array::from_fn(|a| {
            if v.kinematic
                || scene.aggregate_of[i].is_some()
                || targets.get(i).and_then(|t| t[a]).is_some()
            {
                0.
            } else {
                1.
            }
        }));
    }
    scratch.active.clear();
    let live = scene.step_live.clone();
    fill_visited(&mut scratch.active_index, n, u32::MAX, &live);
    for &i in &mechanical_revs {
        if scratch.free[i] != Vec3::ZERO {
            scratch.active_index[i] =
                u32::try_from(scratch.active.len()).expect("active REV count exceeds u32");
            scratch.active.push(i);
        }
    }
    let active_count = scratch.active.len();
    let backend = std::env::var("VOXEL_SOLVER_BACKEND").unwrap_or_else(|_| "auto".into());
    let cuda_iterations = std::env::var("VOXEL_CUDA_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(scene.config.cuda_iterations);
    let coupled_contact = scratch.pair_contacts.iter().any(|p| p.active);
    let try_cuda = !scene.cell_dofs && !coupled_contact && *cuda_enabled_for_step
        && match backend.as_str() {
            "cpu" => false,
            "cuda" => true,
            _ => active_count >= 2_048,
        };
    // Per-REV constitutive inputs and row initialization are independent. Keep
    // them parallel only for large active zones; bond compaction and each
    // REV's ordered force gather retain their stable serial accumulation order.
    let parallel_assembly =
        parallel_enabled_for(mechanical_revs.len()) || (try_cuda && mechanical_revs.len() >= 4_096);
    let parallel_precompute = parallel_assembly && parallel_assembly_precompute_enabled();
    scratch.masses.resize(n, 0.0);
    for &i in &mechanical_revs {
        scratch.masses[i] = scene.voxels[i].mass / (h * h);
    }
    // The bond graph is face-connected, so each voxel can touch at most six
    // constraints. Fixed slots avoid millions of tiny heap allocations on a
    // large scene, while preserving the same per-node gather in the solver.
    fill_visited(&mut scratch.incidence, n, [u32::MAX; 6], &live);
    fill_visited(&mut scratch.incidence_len, n, 0, &live);
    let mechanical_mask = &scratch.mechanical_mask;
    let initialize_row = |i: usize, diagonal: &mut Mat3, rhs: &mut Vec3| {
        let mass = scratch.masses[i];
        *diagonal = Mat3::from_diagonal(Vec3::splat(mass));
        *rhs = (predicted[i] - scene.positions[i]).as_vec3() * mass
            + loads.get(i).copied().unwrap_or(Vec3::ZERO);
    };
    if parallel_precompute {
        scratch
            .diagonal
            .par_iter_mut()
            .zip(scratch.rhs.par_iter_mut())
            .enumerate()
            .for_each(|(i, (diagonal, rhs))| {
                if mechanical_mask[i] {
                    initialize_row(i, diagonal, rhs);
                }
            });
    } else {
        for &i in &mechanical_revs {
            initialize_row(i, &mut scratch.diagonal[i], &mut scratch.rhs[i]);
        }
    }
    // Constraint constitutive matrices are independent across bonds. Build
    // them in parallel for large zones, then consume the stable axis-major
    // sequence below so accumulation and edge indexing remain deterministic.
    scratch.constraint_slots.resize_with(3 * n, || None);
    // Read each REV's equation of state once per assembly, shared by all
    // incident bonds and by stiffness/thermal expansion. Rebuilt on every
    // assembly so heat, composition and mass changes cannot leave stale data.
    // Entries outside `mechanical_revs` are never read: every constraint and
    // fracture outcome names only REVs in that set.
    scratch.assembly_phases.resize(n, Default::default());
    scratch.assembly_phase_strengths.resize(n, (1.0, 1.0));
    let update_phase = |i: usize,
                        phase: &mut crate::material::PhaseState,
                        strength: &mut (f32, f32)| {
        let voxel = &scene.voxels[i];
        let material = &scene.properties[i];
        *phase = material.phase_state(voxel.mass, voxel.thermal_energy_f32());
        *strength = (
            phase.solid_fraction.clamp(0.0, 1.0),
            material.thermal_strength_scale(phase.temperature_kelvin),
        );
    };
    if parallel_precompute {
        scratch
            .assembly_phases
            .par_iter_mut()
            .zip(scratch.assembly_phase_strengths.par_iter_mut())
            .enumerate()
            .for_each(|(i, (phase, strength))| {
                if mechanical_mask[i] {
                    update_phase(i, phase, strength);
                }
            });
    } else {
        for &i in &mechanical_revs {
            update_phase(
                i,
                &mut scratch.assembly_phases[i],
                &mut scratch.assembly_phase_strengths[i],
            );
        }
    }
    let previous_stiffness_len = scratch
        .stiffness_inputs
        .len()
        .min(scratch.stiffness_revisions.len())
        .min(n);
    scratch.stiffness_inputs.truncate(previous_stiffness_len);
    scratch.stiffness_revisions.truncate(previous_stiffness_len);
    scratch.stiffness_inputs.resize(
        n,
        StiffnessInput {
            rotation: Quat::IDENTITY,
            world_fiber: Vec3::ZERO,
            anisotropy: 0.0,
            moduli: Vec3::ZERO,
            solid: 0.0,
        },
    );
    scratch.stiffness_revisions.resize(n, 0);
    let phases = &scratch.assembly_phases;
    let update_stiffness = |i: usize, cached: &mut StiffnessInput, revision: &mut u64| {
        let material = &scene.properties[i];
        let micro = scene.voxels[i].microstructure;
        let input = StiffnessInput {
            rotation: scene.rotations[i],
            world_fiber: scene.rotations[i] * micro.fibers(),
            anisotropy: micro.anisotropy,
            moduli: Vec3::new(material.young, material.young_perp, material.shear),
            solid: phases[i].solid_fraction.clamp(1e-4, 1.0),
        };
        if i < previous_stiffness_len {
            if *cached != input {
                *cached = input;
                *revision = revision.wrapping_add(1);
            }
        } else {
            *cached = input;
            *revision = 1;
        }
    };
    if parallel_precompute {
        scratch
            .stiffness_inputs
            .par_iter_mut()
            .zip(scratch.stiffness_revisions.par_iter_mut())
            .enumerate()
            .for_each(|(i, (cached, revision))| {
                if mechanical_mask[i] {
                    update_stiffness(i, cached, revision);
                }
            });
    } else {
        for &i in &mechanical_revs {
            update_stiffness(
                i,
                &mut scratch.stiffness_inputs[i],
                &mut scratch.stiffness_revisions[i],
            );
        }
    }
    let revisions = &scratch.stiffness_revisions;
    let build_constraint = |slot: usize, previous: Option<Constraint>| {
        let axis = slot / n;
        let i = slot % n;
        let bond = scene.bonds[axis][i];
        if bond.neighbor == EMPTY {
            return None;
        }
        let j = bond.neighbor as usize;
        // Only a bond between two truly kinematic REVs has no mechanical
        // response or fracture owner. A prescribed deformable REV still
        // needs its material constraint for support/fracture bookkeeping.
        if scene.voxels[i].kinematic && scene.voxels[j].kinematic {
            return None;
        }
        Some(make_constraint_at_phases(
            scene,
            i,
            axis,
            j,
            [phases[i], phases[j]],
            [revisions[i], revisions[j]],
            previous,
        ))
    };
    // The CUDA PCG backend leaves the host free to assemble independent bond
    // rows in parallel. Keep CPU-PCG's measured serial cutoff unchanged.
    if parallel_assembly {
        scratch
            .constraint_slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(slot, output)| *output = build_constraint(slot, *output));
    } else {
        for &slot in &mechanical_slots {
            let output = &mut scratch.constraint_slots[slot];
            *output = build_constraint(slot, *output);
        }
    }
    scratch.constraints.clear();
    for constraint in mechanical_slots.iter().filter_map(|&slot| scratch.constraint_slots[slot]) {
        let i = constraint.i;
        let j = constraint.j;
        if scratch.free[i] == Vec3::ZERO && scratch.free[j] == Vec3::ZERO {
            // The solve has no free degree of freedom, but fracture still
            // needs the current undamaged stiffness and local frame.
            scratch.constraints.push(constraint);
            continue;
        }
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
    // Edge contributions are independent. Keep stable edge IDs, then gather
    // each REV's incident terms in edge order; this reproduces the former
    // axis-major endpoint-scatter sum without serial endpoint contention.
    // Constraints were compacted from ascending axis-major slots, and each
    // edge appends its ID to both endpoints. Thus every incidence row is
    // already sorted by edge ID, which is also the deterministic sum order.
    scratch
        .constraint_forces
        .resize(scratch.constraints.len(), Vec3::ZERO);
    let constraints = &scratch.constraints;
    let forces = &mut scratch.constraint_forces;
    let calculate_edge_force = |(edge, force): (usize, &mut Vec3)| {
        let constraint = &constraints[edge];
        if scratch.free[constraint.i] == Vec3::ZERO && scratch.free[constraint.j] == Vec3::ZERO {
            return;
        }
        *force = constraint.k
            * (scene.positions[constraint.j] - scene.positions[constraint.i] - constraint.rest)
                .as_vec3();
    };
    if parallel_assembly {
        forces
            .par_iter_mut()
            .enumerate()
            .for_each(calculate_edge_force);
    } else {
        forces.iter_mut().enumerate().for_each(calculate_edge_force);
    }
    let incidence = &scratch.incidence;
    let incidence_len = &scratch.incidence_len;
    let accumulate_row = |(i, (rhs, diagonal)): (usize, (&mut Vec3, &mut Mat3))| {
        let mut row_rhs = *rhs;
        let mut row_diagonal = *diagonal;
        for &edge in &incidence[i][..incidence_len[i] as usize] {
            let edge = edge as usize;
            let constraint = &constraints[edge];
            let force = forces[edge];
            if constraint.i == i {
                row_rhs += force;
            } else {
                row_rhs -= force;
            }
            row_diagonal += constraint.k;
        }
        *rhs = row_rhs;
        *diagonal = row_diagonal;
    };
    if parallel_assembly {
        scratch
            .rhs
            .par_iter_mut()
            .zip(scratch.diagonal.par_iter_mut())
            .enumerate()
            .filter(|(i, _)| mechanical_mask[*i])
            .for_each(accumulate_row);
    } else {
        for &i in &mechanical_revs {
            accumulate_row((i, (&mut scratch.rhs[i], &mut scratch.diagonal[i])));
        }
    }
    for &i in &mechanical_revs {
        scratch.rhs[i] *= scratch.free[i];
        let projection = Mat3::from_diagonal(scratch.free[i]);
        scratch.diagonal[i] =
            projection * scratch.diagonal[i] * projection + Mat3::IDENTITY - projection;
    }
    scratch.mechanical_revs = mechanical_revs;
    scratch.mechanical_slots = mechanical_slots;
    let assembly_time = assembly_start.elapsed();
    if scene.cell_dofs {
        let solve_start = Instant::now();
        build_block_jacobi(&scratch.diagonal, &mut scratch.factor_diagonal, &live);
        let iterations =
            solve_face_cells_sparse(scene, h, scratch, !reuse_preconditioner, predicted, loads);
        return (iterations, [assembly_time, Duration::ZERO, solve_start.elapsed()]);
    }
    if !coupled_contact && !try_cuda && scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0 {
        if let Some((factor_time, solve_time)) = try_banded_solve(scene, h, scratch) {
            return (0, [assembly_time, factor_time, solve_time]);
        }
    }
    let bipartite_ic0_requested = !try_cuda
        && scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0
        && bipartite_ic0_enabled();
    let bipartite_ic0_graph = bipartite_ic0_requested
        && color_active_graph(
            &scratch.active,
            &scratch.active_index,
            &scratch.incidence,
            &scratch.incidence_len,
            &scratch.constraints,
            &mut scratch.ic0_colors,
            &mut scratch.ic0_red_slots,
            &mut scratch.ic0_black_slots,
            &mut scratch.ic0_color_stack,
        );
    if bipartite_ic0_graph {
        order_active_by_color(
            &mut scratch.active,
            &mut scratch.active_index,
            &scratch.ic0_colors,
            &mut scratch.ic0_red_slots,
            &mut scratch.ic0_black_slots,
            &mut scratch.ic0_natural_active,
        );
    }
    let component_ic0_disabled = std::env::var("VOXEL_IC0_COMPONENT_PARALLEL")
        .map(|value| matches!(value.as_str(), "0" | "false" | "off"))
        .unwrap_or(false);
    let mut component_ic0 = !try_cuda
        && !bipartite_ic0_graph
        && !component_ic0_disabled
        && scene.cluster_count > 1
        && parallel_enabled_for(active_count)
        && scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0;
    if component_ic0 {
        let component_count = order_active_by_ic0_components(
            &scratch.constraints,
            &mut scratch.active,
            &mut scratch.active_index,
            &mut scratch.ic0_component_parent,
            &mut scratch.ic0_component_root_to_id,
            &mut scratch.ic0_component_sizes,
            &mut scratch.ic0_component_offsets,
            &mut scratch.ic0_component_members,
            &mut scratch.ic0_component_ranges,
        );
        component_ic0 = component_count > 1;
        if component_ic0 {
            build_ic0_component_batches(
                &scratch.ic0_component_ranges,
                active_count,
                &mut scratch.ic0_component_batches,
            );
        } else {
            scratch.ic0_component_ranges.clear();
            scratch.ic0_component_batches.clear();
        }
    } else {
        scratch.ic0_component_ranges.clear();
        scratch.ic0_component_batches.clear();
    }
    let factor_start = Instant::now();
    // A stale factorization is still a valid SPD PCG preconditioner. Reuse it
    // across physical substeps only when the exact DOF mask, ordered bond
    // graph, and preconditioner kind match; PCG always uses the newly assembled
    // operator/RHS and checks its residual against the current system.
    let same_cpu_preconditioner_graph = !try_cuda
        && scratch.cpu_preconditioner_ready
        && scratch.cpu_preconditioner_kind == Some(scene.config.cpu_preconditioner)
        && scratch.cpu_preconditioner_bipartite == bipartite_ic0_graph
        && scratch.cpu_preconditioner_components == component_ic0
        && scratch.cpu_preconditioner_free == scratch.free
        && scratch.cpu_preconditioner_graph.len() == scratch.constraints.len()
        && scratch
            .cpu_preconditioner_graph
            .iter()
            .zip(&scratch.constraints)
            .all(|(&(i, j, axis), constraint)| {
                (i, j, axis) == (constraint.i, constraint.j, constraint.axis)
            });
    // The opt-in bipartite path changes the row permutation. Never reuse a
    // factor built for a different permutation/topology: doing so pairs the
    // cached triangular factor with unrelated active slots.
    let rebuild_cpu_preconditioner = should_rebuild_cpu_preconditioner(
        reuse_preconditioner,
        same_cpu_preconditioner_graph,
        bipartite_ic0_graph,
        scratch.cpu_preconditioner_bipartite,
    );
    if rebuild_cpu_preconditioner {
        scratch.cpu_preconditioner_ready = false;
        scratch.multigrid = None;
        scratch.cpu_preconditioner_bipartite = false;
        if !try_cuda {
            match scene.config.cpu_preconditioner {
                CpuPreconditioner::IncompleteCholesky0 => {
                    scratch.cpu_preconditioner_bipartite = build_ic0_with_optional_bipartite(
                        scene,
                        &scratch.free,
                        &scratch.diagonal,
                        &scratch.incidence,
                        &scratch.incidence_len,
                        &scratch.constraints,
                        bipartite_ic0_graph,
                        &scratch.ic0_colors,
                        &mut scratch.factor_matrix,
                        &mut scratch.factor_diagonal,
                        &mut scratch.factor_edges,
                    )
                }
                CpuPreconditioner::BlockJacobi | CpuPreconditioner::AggregationMultigrid => {
                    build_block_jacobi(&scratch.diagonal, &mut scratch.factor_diagonal, &live)
                }
            }
            if bipartite_ic0_graph && !scratch.cpu_preconditioner_bipartite {
                restore_natural_active_order(
                    &mut scratch.active,
                    &mut scratch.active_index,
                    &scratch.ic0_natural_active,
                    &mut scratch.ic0_red_slots,
                    &mut scratch.ic0_black_slots,
                );
            }
            if scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0 {
                build_ic0_diagonal_inverses(
                    &scratch.factor_diagonal,
                    &mut scratch.factor_diagonal_inverse,
                );
                if scratch.cpu_preconditioner_bipartite {
                    build_ic0_bipartite_links(
                        &scratch.constraints,
                        &scratch.incidence,
                        &scratch.incidence_len,
                        &scratch.active,
                        &scratch.active_index,
                        &scratch.ic0_colors,
                        &mut scratch.ic0_lower_links,
                        &mut scratch.ic0_lower_link_count,
                        &mut scratch.ic0_upper_links,
                        &mut scratch.ic0_upper_link_count,
                    );
                } else {
                    build_ic0_links(
                        &scratch.constraints,
                        &scratch.incidence,
                        &scratch.incidence_len,
                        &scratch.active,
                        &scratch.active_index,
                        &mut scratch.ic0_lower_links,
                        &mut scratch.ic0_lower_link_count,
                        &mut scratch.ic0_upper_links,
                        &mut scratch.ic0_upper_link_count,
                    );
                }
            }
            scratch.cpu_preconditioner_ready = true;
            scratch.cpu_preconditioner_components = component_ic0;
            scratch.cpu_preconditioner_free.clone_from(&scratch.free);
            scratch.cpu_preconditioner_graph.clear();
            scratch
                .cpu_preconditioner_graph
                .extend(scratch.constraints.iter().map(|c| (c.i, c.j, c.axis)));
            scratch.cpu_preconditioner_kind = Some(scene.config.cpu_preconditioner);
        }
    }
    let mut factor_time = factor_start.elapsed();
    let free = &scratch.free;
    let active = &scratch.active;
    let active_index = &scratch.active_index;
    let incidence = &scratch.incidence;
    let incidence_len = &scratch.incidence_len;
    let constraints = &scratch.constraints;
    scratch.pcg_rows.resize(active.len(), PcgRow::default());
    scratch.pcg_edge_blocks.resize(constraints.len(), [0.0; 9]);
    for (edge, constraint) in constraints.iter().enumerate() {
        let row_projection = Mat3::from_diagonal(free[constraint.i]);
        let column_projection = Mat3::from_diagonal(free[constraint.j]);
        scratch.pcg_edge_blocks[edge] =
            pack_pcg_block(-row_projection * constraint.k * column_projection);
    }
    for (slot, &i) in active.iter().enumerate() {
        // A changed contact mask or fracture can shorten or reorder incidence.
        // Old neighbors and transpose bits do not belong to the new operator.
        scratch.pcg_rows[slot] = assemble_pcg_row(
            i, &incidence[i][..incidence_len[i] as usize], constraints, active_index,
        );
    }
    let pcg_rows = &scratch.pcg_rows;
    let pcg_edge_blocks = &scratch.pcg_edge_blocks;
    if scene.config.cpu_preconditioner == CpuPreconditioner::AggregationMultigrid
        && !try_cuda
        && (rebuild_cpu_preconditioner || scratch.multigrid.is_none())
    {
        let start = Instant::now();
        match build_multigrid_preconditioner(
            free,
            active,
            active_index,
            &scratch.diagonal,
            incidence,
            incidence_len,
            constraints,
            &scene.rest_positions,
        ) {
            Ok(hierarchy) => scratch.multigrid = Some(hierarchy),
            Err(reason) => {
                scratch.multigrid = None;
                eprintln!("multigrid unavailable; using block Jacobi: {reason}");
            }
        }
        factor_time += start.elapsed();
    }
    if coupled_contact {
        let start = Instant::now();
        let iterations = solve_projected_contact(scene, h, scratch);
        return (iterations, [assembly_time, factor_time, start.elapsed()]);
    }
    let apply_one = |v: &[Vec3], slot: usize| {
        let i = active[slot];
        let vi = v[slot];
        let row = &pcg_rows[slot];
        let mut value = scratch.diagonal[i] * vi;
        for edge_slot in 0..incidence_len[i] as usize {
            let neighbor = row.neighbors[edge_slot];
            if neighbor != u32::MAX {
                let block = pcg_edge_blocks[row.edge_ids[edge_slot] as usize];
                let transpose = row.transpose_mask & (1 << edge_slot) != 0;
                value += if transpose {
                    apply_pcg_block_transpose(block, v[neighbor as usize])
                } else {
                    apply_pcg_block(block, v[neighbor as usize])
                };
            }
        }
        value
    };
    let apply = |v: &[Vec3], out: &mut [Vec3]| {
        if !parallel_enabled_for(active.len()) {
            out.iter_mut()
                .enumerate()
                .for_each(|(slot, out)| *out = apply_one(v, slot));
        } else {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(slot, out)| *out = apply_one(v, slot));
        }
    };
    let apply_and_dot = |v: &[Vec3], out: &mut [Vec3], lhs: &[Vec3]| -> f64 {
        if !parallel_enabled_for(active.len()) {
            out.iter_mut()
                .enumerate()
                .map(|(slot, out)| {
                    *out = apply_one(v, slot);
                    lhs[slot].dot(*out) as f64
                })
                .sum()
        } else {
            out.par_iter_mut()
                .enumerate()
                .map(|(slot, out)| {
                    *out = apply_one(v, slot);
                    lhs[slot].dot(*out) as f64
                })
                .sum()
        }
    };
    let dot = |a: &[Vec3], b: &[Vec3]| -> f64 {
        if !parallel_enabled_for(a.len()) {
            a.iter().zip(b).map(|(x, y)| x.dot(*y) as f64).sum()
        } else {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| x.dot(*y) as f64)
                .sum()
        }
    };
    let update_and_norm =
        |dx: &mut [Vec3], r: &mut [Vec3], p: &[Vec3], ap: &[Vec3], alpha: f32| -> f64 {
            if !parallel_enabled_for(active_count) {
                for i in 0..active_count {
                    dx[i] += alpha * p[i];
                    r[i] -= alpha * ap[i];
                }
                r.iter().map(|value| value.dot(*value) as f64).sum()
            } else {
                dx.par_iter_mut()
                    .zip(r.par_iter_mut())
                    .zip(p.par_iter().zip(ap.par_iter()))
                    .map(|((dx, r), (p, ap))| {
                        *dx += alpha * *p;
                        *r -= alpha * *ap;
                        r.dot(*r) as f64
                    })
                    .sum()
            }
        };
    // Reuse a preceding correction on either backend. It is only an initial
    // iterate: check the NEW system residual against its original RHS and
    // discard the guess if it is worse than starting from zero.
    // Reusing the preceding correction is safe because its residual is
    // evaluated against the newly assembled fine operator below. If the guess
    // is worse than zero, the solver discards it; set the override to `0` only
    // when comparing or diagnosing cold starts.
    let cpu_warm_start = std::env::var("VOXEL_CPU_PCG_WARM_START")
        .map(|value| value != "0")
        .unwrap_or(true);
    let warm_start = (try_cuda || cpu_warm_start)
        && (reuse_preconditioner || reuse_previous_solution)
        && scratch.dx.len() == active_count;
    scratch.dx.resize(active_count, Vec3::ZERO);
    if !warm_start {
        scratch.dx.fill(Vec3::ZERO);
    } else {
        for (slot, &i) in active.iter().enumerate() {
            scratch.dx[slot] *= free[i];
        }
    }
    scratch.ap.resize(active_count, Vec3::ZERO);
    scratch.r.resize(active_count, Vec3::ZERO);
    for (slot, &i) in active.iter().enumerate() {
        scratch.r[slot] = scratch.rhs[i];
    }
    let rhs_norm = dot(&scratch.r, &scratch.r).sqrt();
    if !rhs_norm.is_finite() {
        scene.solver_residual = f32::INFINITY;
        return (0, [assembly_time, factor_time, Duration::ZERO]);
    }
    if warm_start {
        apply(&scratch.dx, &mut scratch.ap);
        for (residual, &product) in scratch.r.iter_mut().zip(&scratch.ap) {
            *residual -= product;
        }
    }
    let mut residual_norm = dot(&scratch.r, &scratch.r).sqrt();
    if !residual_norm.is_finite() {
        scene.solver_residual = f32::INFINITY;
        return (0, [assembly_time, factor_time, Duration::ZERO]);
    }
    if warm_start && residual_norm > rhs_norm {
        scratch.dx.fill(Vec3::ZERO);
        for (slot, &i) in active.iter().enumerate() {
            scratch.r[slot] = scratch.rhs[i];
        }
        residual_norm = rhs_norm;
    }
    scene.solver_residual = checked_relative_residual(residual_norm, rhs_norm);
    scratch.z.resize(active_count, Vec3::ZERO);
    scratch.p.resize(active_count, Vec3::ZERO);
    scratch.preconditioner_work.resize(active_count, Vec3::ZERO);
    let pcg_start = Instant::now();
    let profile_cuda = std::env::var_os("VOXEL_PROFILE_STEP").is_some();
    let mut cuda_solution = None;
    if try_cuda && residual_norm <= scene.config.solver_relative_tolerance as f64 * rhs_norm {
        // The previous corotation iterate already satisfies the configured
        // residual tolerance for this assembled system.
        cuda_solution = Some(0);
    } else if try_cuda {
        scratch.cuda_neighbors.resize(active_count, [u32::MAX; 6]);
        let compact_edges = crate::cuda_solver::expand_edge_blocks_on_gpu_enabled();
        if compact_edges {
            scratch.cuda_blocks.clear();
            scratch.cuda_edge_ids.resize(active_count, [u32::MAX; 6]);
            scratch.cuda_transpose_masks.resize(active_count, 0);
        } else {
            scratch.cuda_blocks.resize(active_count, [0.0; 54]);
            scratch.cuda_blocks.fill([0.0; 54]);
        }
        scratch.cuda_diagonal.resize(active_count, [0.0; 9]);
        let inverse_on_gpu = crate::cuda_solver::inverse_diagonal_on_gpu_enabled();
        if inverse_on_gpu {
            scratch.cuda_inverse.clear();
        } else {
            scratch.cuda_inverse.resize(active_count, [0.0; 9]);
        }
        for (slot, &i) in active.iter().enumerate() {
            scratch.cuda_neighbors[slot] = scratch.pcg_rows[slot].neighbors;
            if compact_edges {
                scratch.cuda_edge_ids[slot] = scratch.pcg_rows[slot].edge_ids;
                scratch.cuda_transpose_masks[slot] = scratch.pcg_rows[slot].transpose_mask;
            } else {
                for edge_slot in 0..incidence_len[i] as usize {
                    let edge = scratch.pcg_rows[slot].edge_ids[edge_slot] as usize;
                    let block = scratch.pcg_edge_blocks[edge];
                    let packed_block = if scratch.pcg_rows[slot].transpose_mask & (1 << edge_slot) != 0
                    {
                        transpose_pcg_block(block)
                    } else {
                        block
                    };
                    let packed = &mut scratch.cuda_blocks[slot][edge_slot * 9..edge_slot * 9 + 9];
                    packed.copy_from_slice(&packed_block);
                }
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
            if !inverse_on_gpu {
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
        }
        match crate::cuda_solver::solve_pcg(
            &scratch.r,
            &scratch.cuda_neighbors,
            &scratch.cuda_diagonal,
            &scratch.cuda_blocks,
            &scratch.cuda_inverse,
            cuda_iterations,
            if warm_start && residual_norm > 0.0 {
                (scene.config.solver_relative_tolerance as f64 * rhs_norm / residual_norm).min(1.0)
                    as f32
            } else {
                scene.config.solver_relative_tolerance
            },
            compact_edges.then_some(crate::cuda_solver::CompactEdgeBlocks {
                edges: &scratch.pcg_edge_blocks,
                edge_ids: &scratch.cuda_edge_ids,
                transpose_masks: &scratch.cuda_transpose_masks,
            }),
        ) {
            Ok((correction, iterations, residual)) => {
                if warm_start {
                    for (initial, delta) in scratch.dx.iter_mut().zip(correction) {
                        *initial += delta;
                    }
                    scene.solver_residual = if rhs_norm > 1e-12 {
                        (residual as f64 * residual_norm / rhs_norm) as f32
                    } else {
                        0.0
                    };
                } else {
                    scratch.dx = correction;
                    scene.solver_residual = residual;
                }
                cuda_solution = Some(iterations);
            }
            Err(error) => {
                *cuda_enabled_for_step = false;
                if warm_start {
                    scratch.dx.fill(Vec3::ZERO);
                    for (slot, &i) in active.iter().enumerate() {
                        scratch.r[slot] = scratch.rhs[i];
                    }
                    scene.solver_residual = 0.0;
                }
                if profile_cuda {
                    eprintln!(
                        "CUDA PCG fallback: active={} max_iterations={} tolerance={:.3e} elapsed={:.2} ms reason={error}",
                        active_count,
                        cuda_iterations,
                        scene.config.solver_relative_tolerance,
                        pcg_start.elapsed().as_secs_f64() * 1_000.0,
                    );
                } else {
                    static REPORTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                    REPORTED
                        .get_or_init(|| eprintln!("CUDA PCG unavailable; using CPU PCG: {error}"));
                }
            }
        }
    }
    let cuda_time = cuda_solution.map(|_| pcg_start.elapsed());
    if cuda_solution.is_some() {
        let verified = recompute_pcg_residual(
            &scratch.diagonal,
            &scratch.pcg_rows,
            &scratch.pcg_edge_blocks,
            active,
            incidence_len,
            &scratch.rhs,
            &scratch.dx,
            &mut scratch.r,
        );
        scene.solver_residual = verified as f32;
        if !verified.is_finite() || verified > scene.config.solver_relative_tolerance as f64 {
            // Keep the GPU correction as an initial iterate for CPU refinement.
            // No position has been updated yet, and the freshly evaluated
            // residual is already in scratch.r for the CPU restart below.
            cuda_solution = None;
            *cuda_enabled_for_step = false;
            eprintln!("CUDA PCG true residual {verified:.9e} exceeds tolerance; refining on CPU");
        }
    }
    if let Some(iterations) = cuda_solution {
        if std::env::var_os("VOXEL_AUDIT_PCG_RESIDUAL").is_some() {
            let audit = audit_pcg_residual(
                &scratch.diagonal,
                &scratch.pcg_rows,
                &scratch.pcg_edge_blocks,
                active,
                incidence_len,
                &scratch.rhs,
                &scratch.dx,
                &scene.positions,
            );
            report_residual_audit("cuda", scene.solver_residual, audit);
            scene.solver_max_audited_residual = Some(worst_audited_residual(
                scene.solver_max_audited_residual,
                audit,
            ));
            scene.solver_residual_audit = Some(audit);
        }
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
            scene.positions[i] += scratch.dx[slot].as_dvec3();
        }
        for c in &scratch.constraints {
            let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest).as_vec3();
            scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h * h);
        }
        return (
            iterations,
            [assembly_time, factor_time, cuda_time.unwrap_or_default()],
        );
    }
    let cpu_pcg_start = Instant::now();
    if !scratch.cpu_preconditioner_ready {
        let factor_start = Instant::now();
        scratch.cpu_preconditioner_bipartite = false;
        match scene.config.cpu_preconditioner {
            CpuPreconditioner::IncompleteCholesky0 => {
                scratch.cpu_preconditioner_bipartite = build_ic0_with_optional_bipartite(
                    scene,
                    &scratch.free,
                    &scratch.diagonal,
                    &scratch.incidence,
                    &scratch.incidence_len,
                    &scratch.constraints,
                    bipartite_ic0_graph,
                    &scratch.ic0_colors,
                    &mut scratch.factor_matrix,
                    &mut scratch.factor_diagonal,
                    &mut scratch.factor_edges,
                )
            }
            CpuPreconditioner::BlockJacobi | CpuPreconditioner::AggregationMultigrid => {
                build_block_jacobi(&scratch.diagonal, &mut scratch.factor_diagonal, &None)
            }
        }
        if scene.config.cpu_preconditioner == CpuPreconditioner::IncompleteCholesky0 {
            build_ic0_diagonal_inverses(
                &scratch.factor_diagonal,
                &mut scratch.factor_diagonal_inverse,
            );
            if scratch.cpu_preconditioner_bipartite {
                build_ic0_bipartite_links(
                    &scratch.constraints,
                    &scratch.incidence,
                    &scratch.incidence_len,
                    &scratch.active,
                    &scratch.active_index,
                    &scratch.ic0_colors,
                    &mut scratch.ic0_lower_links,
                    &mut scratch.ic0_lower_link_count,
                    &mut scratch.ic0_upper_links,
                    &mut scratch.ic0_upper_link_count,
                );
            } else {
                build_ic0_links(
                    &scratch.constraints,
                    &scratch.incidence,
                    &scratch.incidence_len,
                    &scratch.active,
                    &scratch.active_index,
                    &mut scratch.ic0_lower_links,
                    &mut scratch.ic0_lower_link_count,
                    &mut scratch.ic0_upper_links,
                    &mut scratch.ic0_upper_link_count,
                );
            }
        }
        factor_time += factor_start.elapsed();
        scratch.cpu_preconditioner_ready = true;
        scratch.cpu_preconditioner_components = false;
    }
    // CUDA can fail after the initial CPU setup was skipped. Build the
    // hierarchy only once CPU fallback is actually needed.
    if scene.config.cpu_preconditioner == CpuPreconditioner::AggregationMultigrid
        && scratch.multigrid.is_none()
        && try_cuda
    {
        let start = Instant::now();
        match build_multigrid_preconditioner(
            free,
            active,
            active_index,
            &scratch.diagonal,
            incidence,
            incidence_len,
            constraints,
            &scene.rest_positions,
        ) {
            Ok(hierarchy) => scratch.multigrid = Some(hierarchy),
            Err(reason) => eprintln!("multigrid unavailable; using block Jacobi: {reason}"),
        }
        factor_time += start.elapsed();
    }
    if scene.solver_residual < scene.config.solver_relative_tolerance {
        scene.solver_residual = recompute_pcg_residual(
            &scratch.diagonal,
            &scratch.pcg_rows,
            &scratch.pcg_edge_blocks,
            active,
            incidence_len,
            &scratch.rhs,
            &scratch.dx,
            &mut scratch.r,
        ) as f32;
    }
    let initial = rhs_norm;
    scene.solver_residual = checked_relative_residual(dot(&scratch.r, &scratch.r).sqrt(), initial);
    let mut iterations = 0;
    let mut verified_solution = false;
    // These ranges are block diagonal by construction: no active mechanical
    // constraint crosses a component. Solving each block with the unchanged
    // tolerance is algebraically equivalent; the full true residual below
    // remains authoritative and the coupled PCG is kept as a fallback.
    let component_pcg = component_pcg_enabled()
        && component_ic0
        && !scratch.cpu_preconditioner_bipartite
        && scratch.ic0_component_ranges.len() > 1;
    if component_pcg {
        let ranges = &scratch.ic0_component_ranges;
        let dx_chunks = split_mut_by_ranges(&mut scratch.dx, ranges);
        let ap_chunks = split_mut_by_ranges(&mut scratch.ap, ranges);
        let r_chunks = split_mut_by_ranges(&mut scratch.r, ranges);
        let z_chunks = split_mut_by_ranges(&mut scratch.z, ranges);
        let p_chunks = split_mut_by_ranges(&mut scratch.p, ranges);
        let work_chunks = split_mut_by_ranges(&mut scratch.preconditioner_work, ranges);
        let mut states = Vec::with_capacity(ranges.len());
        let chunks = dx_chunks
            .into_iter()
            .zip(ap_chunks)
            .zip(r_chunks)
            .zip(z_chunks)
            .zip(p_chunks)
            .zip(work_chunks);
        for (component, (((((dx, ap), r), z), p), work)) in chunks.enumerate() {
            let (start, end) = ranges[component];
            states.push(ComponentPcgState {
                start,
                end,
                dx,
                ap,
                r,
                z,
                p,
                work,
                iterations: 0,
                converged: false,
            });
        }
        states.par_iter_mut().for_each(|state| {
            solve_pcg_component(
                state,
                active,
                &scratch.diagonal,
                &scratch.pcg_rows,
                &scratch.pcg_edge_blocks,
                incidence_len,
                &scratch.rhs,
                &scratch.factor_diagonal_inverse,
                &scratch.factor_edges,
                &scratch.ic0_lower_links,
                &scratch.ic0_lower_link_count,
                &scratch.ic0_upper_links,
                &scratch.ic0_upper_link_count,
                f64::from(scene.config.solver_relative_tolerance),
                scene.config.iterations,
            );
        });
        iterations = states
            .iter()
            .map(|state| state.iterations)
            .max()
            .unwrap_or(0);
        verified_solution = states.iter().all(|state| state.converged);
        drop(states);
        scene.solver_residual = recompute_pcg_residual(
            &scratch.diagonal,
            &scratch.pcg_rows,
            &scratch.pcg_edge_blocks,
            active,
            incidence_len,
            &scratch.rhs,
            &scratch.dx,
            &mut scratch.r,
        ) as f32;
        verified_solution &= scene.solver_residual.is_finite()
            && scene.solver_residual < scene.config.solver_relative_tolerance;
    }

    if !verified_solution {
        apply_preconditioner(
            scene.config.cpu_preconditioner,
            scratch.cpu_preconditioner_bipartite,
            &mut scratch.multigrid,
            &scratch.factor_diagonal,
            &scratch.factor_diagonal_inverse,
            &scratch.factor_edges,
            active,
            &scratch.ic0_component_ranges,
            &scratch.ic0_component_batches,
            &scratch.ic0_red_slots,
            &scratch.ic0_lower_links,
            &scratch.ic0_lower_link_count,
            &scratch.ic0_upper_links,
            &scratch.ic0_upper_link_count,
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
        let mut numerical_failure = !rz.is_finite()
            || r.iter().any(|value| !value.is_finite())
            || z.iter().any(|value| !value.is_finite());
        scene.solver_residual = checked_relative_residual(dot(r, r).sqrt(), initial);
        let fuse_reductions = fused_pcg_reductions_enabled();
        for _ in 0..scene.config.iterations {
            if numerical_failure {
                break;
            }
            if initial == 0.0 || scene.solver_residual < scene.config.solver_relative_tolerance {
                verified_solution = true;
                break;
            }
            let denom = if fuse_reductions {
                apply_and_dot(p, ap, p)
            } else {
                apply(p, ap);
                dot(&p, &ap)
            };
            iterations += 1;
            if !denom.is_finite() || denom <= 1e-30 || !rz.is_finite() {
                numerical_failure = !denom.is_finite() || !rz.is_finite();
                break;
            }
            let alpha = (rz / denom) as f32;
            if !alpha.is_finite() {
                numerical_failure = true;
                break;
            }
            let residual_squared = if fuse_reductions {
                update_and_norm(dx, r, p, ap, alpha)
            } else {
                if !parallel_enabled_for(active_count) {
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
                dot(&r, &r)
            };
            scene.solver_residual = checked_relative_residual(residual_squared.sqrt(), initial);
            numerical_failure |= !scene.solver_residual.is_finite();
            let mut restart = false;
            if scene.solver_residual < scene.config.solver_relative_tolerance {
                scene.solver_residual = recompute_pcg_residual(
                    &scratch.diagonal,
                    &scratch.pcg_rows,
                    &scratch.pcg_edge_blocks,
                    active,
                    incidence_len,
                    &scratch.rhs,
                    dx,
                    r,
                ) as f32;
                if scene.solver_residual < scene.config.solver_relative_tolerance {
                    verified_solution = true;
                    break;
                }
                restart = true;
            }
            apply_preconditioner(
                scene.config.cpu_preconditioner,
                scratch.cpu_preconditioner_bipartite,
                &mut scratch.multigrid,
                &scratch.factor_diagonal,
                &scratch.factor_diagonal_inverse,
                &scratch.factor_edges,
                active,
                &scratch.ic0_component_ranges,
                &scratch.ic0_component_batches,
                &scratch.ic0_red_slots,
                &scratch.ic0_lower_links,
                &scratch.ic0_lower_link_count,
                &scratch.ic0_upper_links,
                &scratch.ic0_upper_link_count,
                r,
                &mut scratch.preconditioner_work,
                z,
            );
            let next = dot(&r, &z);
            if restart {
                p.clone_from_slice(z);
                rz = next;
                continue;
            }
            let beta = (next / rz) as f32;
            if !parallel_enabled_for(active_count) {
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
        if numerical_failure {
            scene.solver_residual = f32::INFINITY;
        } else if !verified_solution {
            scene.solver_residual = recompute_pcg_residual(
                &scratch.diagonal,
                &scratch.pcg_rows,
                &scratch.pcg_edge_blocks,
                active,
                incidence_len,
                &scratch.rhs,
                dx,
                r,
            ) as f32;
        }
    }
    let pcg_time = cpu_pcg_start.elapsed();
    if !scene.solver_residual.is_finite() || scratch.dx.iter().any(|value| !value.is_finite()) {
        // A failed solve is reported as failed and its nonfinite candidate is
        // never committed into world positions.
        scene.solver_residual = f32::INFINITY;
        return (iterations, [assembly_time, factor_time, pcg_time]);
    }
    if std::env::var_os("VOXEL_AUDIT_PCG_RESIDUAL").is_some() {
        let audit = audit_pcg_residual(
            &scratch.diagonal,
            &scratch.pcg_rows,
            &scratch.pcg_edge_blocks,
            active,
            incidence_len,
            &scratch.rhs,
            &scratch.dx,
            &scene.positions,
        );
        report_residual_audit("cpu", scene.solver_residual, audit);
        scene.solver_max_audited_residual = Some(worst_audited_residual(
            scene.solver_max_audited_residual,
            audit,
        ));
        scene.solver_residual_audit = Some(audit);
    }
    if std::env::var_os("VOXEL_PROFILE_PCG_SOLVE").is_some() {
        eprintln!(
            "CPU PCG solve: active={active_count} preconditioner={:?} iterations={iterations} residual={:.3e} elapsed={:.2}ms",
            scene.config.cpu_preconditioner,
            scene.solver_residual,
            pcg_time.as_secs_f64() * 1_000.0,
        );
    }
    for (slot, &i) in active.iter().enumerate() {
        scene.positions[i] += scratch.dx[slot].as_dvec3();
    }
    for c in constraints {
        let force = c.k * (scene.positions[c.j] - scene.positions[c.i] - c.rest).as_vec3();
        scene.workspace[c.axis][c.i].lambda = -c.frame.transpose() * force * (h * h);
    }
    (iterations, [assembly_time, factor_time, pcg_time])
}
