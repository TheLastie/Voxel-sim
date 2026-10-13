//! Quasi-static electrical conduction on connected REV bonds.
use crate::{
    bond::EMPTY,
    scene::{Scene, AREA, CELL},
};
use std::collections::VecDeque;

const VACUUM_PERMITTIVITY_F_M: f64 = 8.854_187_812_8e-12;

/// Dirichlet voltage boundary, in volts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoltageBoundary {
    pub voxel: usize,
    pub voltage_v: f64,
}

/// Results from one steady electrical solve and Joule-heating interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ElectricalStep {
    pub potentials_v: Vec<f64>,
    /// Current injected into the network by each listed voltage boundary, in A.
    pub boundary_currents_a: Vec<f64>,
    pub source_energy_j: f64,
    pub joule_heat_j: f64,
    pub iterations: usize,
}

/// Electrostatic potentials and bound charge for one imposed voltage state.
#[derive(Clone, Debug, PartialEq)]
pub struct DielectricState {
    pub potentials_v: Vec<f64>,
    /// Charge on each REV implied by the solved field, in coulombs.
    pub voxel_charge_c: Vec<f64>,
    /// Charge injected at each listed voltage boundary, in coulombs.
    pub boundary_charge_c: Vec<f64>,
    /// Stored electrostatic field energy, in joules.
    pub stored_energy_j: f64,
    pub iterations: usize,
}

struct NetworkSolution {
    potentials_v: Vec<f64>,
    boundary_flux: Vec<f64>,
    iterations: usize,
}

#[derive(Clone, Copy)]
struct Edge {
    i: usize,
    j: usize,
    weight: f64,
}

/// Solve `∇·(σ∇V)=0` on intact bonds and convert resistive loss to internal heat.
/// Conductance uses a one dimensional finite-volume bond with length `CELL`,
/// cross-section `AREA`, and the series harmonic mean of adjacent materials.
/// Disconnected components without a voltage boundary remain at zero potential.
pub fn solve_and_heat(
    scene: &mut Scene,
    boundaries: &[VoltageBoundary],
    dt_s: f64,
) -> Result<ElectricalStep, String> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err("electrical time interval must be finite and positive".into());
    }
    if scene.voxels.len() != scene.properties.len() {
        return Err("scene voxel and material-property arrays are inconsistent".into());
    }
    for (i, material) in scene.properties.iter().enumerate() {
        if !material.electrical_conductivity.is_finite() || material.electrical_conductivity < 0.0 {
            return Err(format!("REV {i} has invalid electrical conductivity"));
        }
    }

    let sigma = &scene.properties;
    let mut edges = Vec::new();
    for axis in &scene.bonds {
        for (i, bond) in axis.iter().enumerate() {
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if j >= scene.voxels.len() {
                return Err(format!("bond from REV {i} points outside the scene"));
            }
            if i >= j {
                continue;
            }
            let sigma_a = sigma[i].electrical_conductivity as f64;
            let sigma_b = sigma[j].electrical_conductivity as f64;
            if sigma_a == 0.0 || sigma_b == 0.0 {
                continue;
            }
            let effective_sigma = 2.0 / (1.0 / sigma_a + 1.0 / sigma_b);
            let conductance_s = effective_sigma * AREA as f64 / CELL as f64;
            if !conductance_s.is_finite() || conductance_s <= 0.0 {
                return Err(format!("bond {i}-{j} has invalid conductance"));
            }
            edges.push(Edge {
                i,
                j,
                weight: conductance_s,
            });
        }
    }
    let solution = solve_network(scene.voxels.len(), &edges, boundaries)?;

    let mut heat_by_voxel = vec![0.0f64; scene.voxels.len()];
    let mut joule_heat_j = 0.0;
    for edge in &edges {
        let delta_v = solution.potentials_v[edge.i] - solution.potentials_v[edge.j];
        let heat = edge.weight * delta_v * delta_v * dt_s;
        joule_heat_j += heat;
        heat_by_voxel[edge.i] += heat * 0.5;
        heat_by_voxel[edge.j] += heat * 0.5;
    }
    let source_energy_j: f64 = boundaries
        .iter()
        .zip(&solution.boundary_flux)
        .map(|(boundary, current)| boundary.voltage_v * current * dt_s)
        .sum();

    if !joule_heat_j.is_finite()
        || !source_energy_j.is_finite()
        || heat_by_voxel.iter().any(|heat| !heat.is_finite())
        || solution
            .boundary_flux
            .iter()
            .any(|current| !current.is_finite())
    {
        return Err("electrical solve produced non-finite current or heat".into());
    }

    for (voxel, heat) in scene.voxels.iter_mut().zip(heat_by_voxel) {
        voxel.add_internal_energy_j(heat);
    }
    Ok(ElectricalStep {
        potentials_v: solution.potentials_v,
        boundary_currents_a: solution.boundary_flux,
        source_energy_j,
        joule_heat_j,
        iterations: solution.iterations,
    })
}

/// Solve quasi-static dielectric voltage and bound charge on intact REV bonds.
/// Bond capacitance is `ε₀ εᵣ A/L`; the result owns the instantaneous field
/// state, including charge and stored energy, without adding heat to the scene.
pub fn solve_dielectric_state(
    scene: &Scene,
    boundaries: &[VoltageBoundary],
) -> Result<DielectricState, String> {
    if scene.voxels.len() != scene.properties.len() {
        return Err("scene voxel and material-property arrays are inconsistent".into());
    }
    for (i, material) in scene.properties.iter().enumerate() {
        if !material.relative_permittivity.is_finite() || material.relative_permittivity <= 0.0 {
            return Err(format!("REV {i} has invalid relative permittivity"));
        }
    }
    let mut edges = Vec::new();
    for axis in &scene.bonds {
        for (i, bond) in axis.iter().enumerate() {
            if bond.neighbor == EMPTY {
                continue;
            }
            let j = bond.neighbor as usize;
            if j >= scene.voxels.len() {
                return Err(format!("bond from REV {i} points outside the scene"));
            }
            if i >= j {
                continue;
            }
            let epsilon_a = scene.properties[i].relative_permittivity as f64;
            let epsilon_b = scene.properties[j].relative_permittivity as f64;
            let effective_epsilon = 2.0 / (1.0 / epsilon_a + 1.0 / epsilon_b);
            let capacitance_f =
                VACUUM_PERMITTIVITY_F_M * effective_epsilon * AREA as f64 / CELL as f64;
            if !capacitance_f.is_finite() || capacitance_f <= 0.0 {
                return Err(format!("bond {i}-{j} has invalid capacitance"));
            }
            edges.push(Edge {
                i,
                j,
                weight: capacitance_f,
            });
        }
    }
    let solution = solve_network(scene.voxels.len(), &edges, boundaries)?;
    let mut voxel_charge_c = vec![0.0f64; scene.voxels.len()];
    let mut stored_energy_j = 0.0;
    for edge in &edges {
        let delta_v = solution.potentials_v[edge.i] - solution.potentials_v[edge.j];
        let charge = edge.weight * delta_v;
        voxel_charge_c[edge.i] += charge;
        voxel_charge_c[edge.j] -= charge;
        stored_energy_j += 0.5 * edge.weight * delta_v * delta_v;
    }
    if !stored_energy_j.is_finite()
        || voxel_charge_c.iter().any(|charge| !charge.is_finite())
        || (solution.boundary_flux.iter().sum::<f64>()).abs()
            > solution
                .boundary_flux
                .iter()
                .map(|charge| charge.abs())
                .sum::<f64>()
                * 1e-9
                + 1e-30
    {
        return Err("dielectric solve produced non-finite state or unbalanced charge".into());
    }
    Ok(DielectricState {
        potentials_v: solution.potentials_v,
        voxel_charge_c,
        boundary_charge_c: solution.boundary_flux,
        stored_energy_j,
        iterations: solution.iterations,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn apply_matrix(diagonal: &[f64], off_diagonal: &[Vec<(usize, f64)>], x: &[f64]) -> Vec<f64> {
    diagonal
        .iter()
        .enumerate()
        .map(|(i, diagonal)| {
            diagonal * x[i] - off_diagonal[i].iter().map(|&(j, g)| g * x[j]).sum::<f64>()
        })
        .collect()
}

/// Solve the shared weighted graph Laplace equation for conductance (S) or
/// capacitance (F) edge weights. `boundary_flux` has units weight × volts.
fn solve_network(
    node_count: usize,
    edges: &[Edge],
    boundaries: &[VoltageBoundary],
) -> Result<NetworkSolution, String> {
    let mut fixed = vec![None; node_count];
    for boundary in boundaries {
        if boundary.voxel >= node_count {
            return Err(format!(
                "voltage boundary voxel {} is out of range",
                boundary.voxel
            ));
        }
        if !boundary.voltage_v.is_finite() {
            return Err("boundary voltage must be finite".into());
        }
        if fixed[boundary.voxel].replace(boundary.voltage_v).is_some() {
            return Err(format!(
                "duplicate voltage boundary for voxel {}",
                boundary.voxel
            ));
        }
    }
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); node_count];
    for edge in edges {
        if edge.i >= node_count || edge.j >= node_count || edge.i == edge.j {
            return Err("electrical network contains an invalid edge".into());
        }
        if !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err("electrical network contains an invalid edge weight".into());
        }
        adjacency[edge.i].push((edge.j, edge.weight));
        adjacency[edge.j].push((edge.i, edge.weight));
    }

    // Only components attached to a voltage boundary enter the solve. Floating
    // components have no imposed electric field and therefore carry no current.
    let mut active = vec![false; node_count];
    let mut queue = VecDeque::new();
    for (i, value) in fixed.iter().enumerate() {
        if value.is_some() && !active[i] {
            active[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for &(j, _) in &adjacency[i] {
            if !active[j] {
                active[j] = true;
                queue.push_back(j);
            }
        }
    }

    let mut unknown_id = vec![usize::MAX; node_count];
    let mut unknown_nodes = Vec::new();
    for i in 0..node_count {
        if active[i] && fixed[i].is_none() {
            unknown_id[i] = unknown_nodes.len();
            unknown_nodes.push(i);
        }
    }
    let n = unknown_nodes.len();
    let mut diagonal = vec![0.0f64; n];
    let mut rhs = vec![0.0f64; n];
    let mut off_diagonal = vec![Vec::<(usize, f64)>::new(); n];
    for edge in edges {
        for (i, j) in [(edge.i, edge.j), (edge.j, edge.i)] {
            let row = unknown_id[i];
            if row == usize::MAX {
                continue;
            }
            diagonal[row] += edge.weight;
            if let Some(voltage) = fixed[j] {
                rhs[row] += edge.weight * voltage;
            } else if unknown_id[j] != usize::MAX {
                off_diagonal[row].push((unknown_id[j], edge.weight));
            }
        }
    }

    let mut x = vec![0.0f64; n];
    let mut iterations = 0;
    if n > 0 {
        if diagonal
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("electrical system has an unconstrained floating REV".into());
        }
        let norm_rhs = dot(&rhs, &rhs).sqrt();
        if norm_rhs > 0.0 {
            let mut residual = rhs.clone();
            let mut z: Vec<f64> = residual.iter().zip(&diagonal).map(|(r, d)| r / d).collect();
            let mut direction = z.clone();
            let mut rz = dot(&residual, &z);
            let max_iterations = n.saturating_mul(4).max(100).min(20_000);
            for k in 0..max_iterations {
                let product = apply_matrix(&diagonal, &off_diagonal, &direction);
                let denom = dot(&direction, &product);
                if !denom.is_finite() || denom <= 0.0 {
                    return Err(
                        "electrical conjugate-gradient system lost positive definiteness".into(),
                    );
                }
                let alpha = rz / denom;
                for i in 0..n {
                    x[i] += alpha * direction[i];
                    residual[i] -= alpha * product[i];
                }
                iterations = k + 1;
                if dot(&residual, &residual).sqrt() <= norm_rhs * 1e-11 {
                    break;
                }
                z.iter_mut()
                    .zip(&residual)
                    .zip(&diagonal)
                    .for_each(|((z, r), d)| *z = r / d);
                let next_rz = dot(&residual, &z);
                let beta = next_rz / rz;
                for i in 0..n {
                    direction[i] = z[i] + beta * direction[i];
                }
                rz = next_rz;
            }
            let relative_residual = dot(&residual, &residual).sqrt() / norm_rhs;
            if !relative_residual.is_finite() || relative_residual > 1e-9 {
                return Err(format!(
                    "electrical solve did not converge (relative residual {relative_residual:.3e})"
                ));
            }
        }
    }

    let mut potentials_v = vec![0.0; node_count];
    for (i, voltage) in fixed.iter().enumerate() {
        if let Some(voltage) = voltage {
            potentials_v[i] = *voltage;
        }
    }
    for (slot, &voxel) in unknown_nodes.iter().enumerate() {
        potentials_v[voxel] = x[slot];
    }
    if potentials_v.iter().any(|value| !value.is_finite()) {
        return Err("electrical solution produced a non-finite voltage".into());
    }
    let boundary_flux: Vec<f64> = boundaries
        .iter()
        .map(|boundary| {
            adjacency[boundary.voxel]
                .iter()
                .map(|&(neighbor, weight)| {
                    weight * (potentials_v[boundary.voxel] - potentials_v[neighbor])
                })
                .sum()
        })
        .collect();
    if boundary_flux.iter().any(|flux| !flux.is_finite()) {
        return Err("electrical solution produced a non-finite boundary flux".into());
    }
    Ok(NetworkSolution {
        potentials_v,
        boundary_flux,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        material::Material,
        scene::{SceneBuilder, VoxelSpawn},
    };
    use glam::Vec3;

    #[test]
    fn copper_chain_solves_voltage_and_conserves_joule_heat() {
        let copper = Material::component_id("copper").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], copper))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], copper))
            .add_voxel(VoxelSpawn::at_cell([2, 0, 0], copper))
            .build()
            .unwrap();
        let initial_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum();
        let result = solve_and_heat(
            &mut scene,
            &[
                VoltageBoundary {
                    voxel: 0,
                    voltage_v: 0.0,
                },
                VoltageBoundary {
                    voxel: 2,
                    voltage_v: 1.0e-6,
                },
            ],
            1.0,
        )
        .unwrap();
        assert!(result.iterations > 0);
        assert!((result.potentials_v[1] - 0.5e-6).abs() < 1e-14);
        let conductance =
            scene.properties[0].electrical_conductivity as f64 * AREA as f64 / CELL as f64;
        let expected_heat = conductance * 0.5e-12;
        assert!((result.joule_heat_j - expected_heat).abs() < expected_heat * 1e-9);
        assert!((result.source_energy_j - result.joule_heat_j).abs() < 1e-18);
        let final_energy: f64 = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .sum();
        assert!(
            (final_energy - initial_energy - result.joule_heat_j).abs() < 2e-14,
            "thermal increment mismatch: initial={initial_energy:.16e}, final={final_energy:.16e}, heat={:.16e}, diff={:.16e}",
            result.joule_heat_j,
            final_energy - initial_energy - result.joule_heat_j
        );
        assert!((result.boundary_currents_a[1] - conductance * 0.5e-6).abs() < 1e-9);
    }

    #[test]
    fn glass_chain_stores_the_analytic_dielectric_charge_and_energy() {
        let glass = Material::component_id("glass").unwrap();
        let scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], glass))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], glass))
            .add_voxel(VoxelSpawn::at_cell([2, 0, 0], glass))
            .build()
            .unwrap();
        let state = solve_dielectric_state(
            &scene,
            &[
                VoltageBoundary {
                    voxel: 0,
                    voltage_v: 0.0,
                },
                VoltageBoundary {
                    voxel: 2,
                    voltage_v: 1.0,
                },
            ],
        )
        .unwrap();
        let single_bond_capacitance = VACUUM_PERMITTIVITY_F_M
            * scene.properties[0].relative_permittivity as f64
            * AREA as f64
            / CELL as f64;
        let expected_capacitance = single_bond_capacitance * 0.5;
        let expected_charge = expected_capacitance;
        let expected_energy = 0.5 * expected_capacitance;
        assert!(state.iterations > 0);
        assert!((state.potentials_v[1] - 0.5).abs() < 1e-12);
        assert!((state.boundary_charge_c[1] - expected_charge).abs() < expected_charge * 1e-10);
        assert!((state.stored_energy_j - expected_energy).abs() < expected_energy * 1e-10);
        assert!((state.voxel_charge_c[0] + expected_charge).abs() < expected_charge * 1e-10);
        assert!(state.voxel_charge_c[1].abs() < expected_charge * 1e-10);
        assert!((state.voxel_charge_c[2] - expected_charge).abs() < expected_charge * 1e-10);
    }

    #[test]
    fn open_bond_interrupts_steady_electrical_current() {
        let copper = Material::component_id("copper").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], copper))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], copper))
            .build()
            .unwrap();
        scene.bonds[0][0].neighbor = EMPTY;
        let result = solve_and_heat(
            &mut scene,
            &[
                VoltageBoundary {
                    voxel: 0,
                    voltage_v: 0.0,
                },
                VoltageBoundary {
                    voxel: 1,
                    voltage_v: 1.0,
                },
            ],
            1.0,
        )
        .unwrap();
        assert_eq!(result.joule_heat_j, 0.0);
        assert_eq!(result.boundary_currents_a, vec![0.0, 0.0]);
    }

    #[test]
    fn zero_conductivity_material_is_an_open_circuit() {
        let copper = Material::component_id("copper").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], copper))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], copper))
            .build()
            .unwrap();
        scene.properties[0].electrical_conductivity = 0.0;
        let result = solve_and_heat(
            &mut scene,
            &[
                VoltageBoundary {
                    voxel: 0,
                    voltage_v: 0.0,
                },
                VoltageBoundary {
                    voxel: 1,
                    voltage_v: 1.0,
                },
            ],
            1.0,
        )
        .unwrap();
        assert_eq!(result.joule_heat_j, 0.0);
        assert_eq!(result.boundary_currents_a, vec![0.0, 0.0]);
    }

    #[test]
    fn overflowing_voltage_rejects_step_without_mutating_heat() {
        let copper = Material::component_id("copper").unwrap();
        let mut scene = SceneBuilder::standard()
            .with_gravity(Vec3::ZERO)
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], copper))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], copper))
            .build()
            .unwrap();
        let before: Vec<_> = scene
            .voxels
            .iter()
            .map(|voxel| voxel.total_internal_energy_j())
            .collect();
        let result = solve_and_heat(
            &mut scene,
            &[
                VoltageBoundary {
                    voxel: 0,
                    voltage_v: 0.0,
                },
                VoltageBoundary {
                    voxel: 1,
                    voltage_v: f64::MAX,
                },
            ],
            1.0,
        );
        assert!(result.is_err());
        assert_eq!(
            scene
                .voxels
                .iter()
                .map(|voxel| voxel.total_internal_energy_j())
                .collect::<Vec<_>>(),
            before
        );
    }
}
