//! Beam regions: a slender part of a scene simulated as a beam instead of as cells.
//!
//! `Scene::reduce_to_beam` replaces a chain of REV slices by a `Beam` (tier T2). The
//! region's REVs keep their identity, mass, composition, heat and bonds; only their
//! motion is taken from the beam: each REV is a rigid part of its cross-section. They
//! are made kinematic so the cell solver leaves them alone, and every fixed step they
//! are placed where the beam says. `Scene::expand_beam` hands the region back to the cell
//! solver with positions, rotations and velocities taken from the beam.
//!
//! The coupling is one way: the beam feels gravity and the forces set on its REVs, but
//! contact forces on its REVs are not carried into the beam, and fracture inside the
//! region is not simulated. A region that touches something, or whose
//! `beam_region_validity` reaches 1, must be expanded. The budget governor uses exactly
//! these two conditions to keep a region at the cell tier.
use super::*;
use crate::beam::{reduce_lattice_along_x, Beam, BeamMaterial};

/// A region whose peak stress reaches this fraction of the material limit is expanded
/// into cells at the end of the step.
const BEAM_VALIDITY_LIMIT: f64 = 0.95;

pub(crate) struct BeamRegion {
    beam: Beam,
    /// Scene index of each member REV.
    members: Vec<usize>,
    /// Rest positions of the members, in member order.
    rest: Vec<DVec3>,
    /// Member positions grouped by beam node, in member indexing.
    groups: Vec<Vec<usize>>,
    /// The clamped slice stays fixed and is never moved by the beam.
    root: Vec<bool>,
    saved_kinematic: Vec<bool>,
    substeps: usize,
    /// The beam's acceleration has been made consistent with its loads.
    primed: bool,
}

impl BeamRegion {
    /// Scene indices of the member REVs.
    pub(crate) fn members(&self) -> &[usize] {
        &self.members
    }

    /// Whether member `local` was fixed before the region took it over.
    pub(crate) fn saved_kinematic(&self, local: usize) -> bool {
        self.saved_kinematic[local]
    }
}

impl Scene {
    /// Replace the REVs `cells`, a chain of slices along X whose first slice is fixed, by
    /// a beam stepped `substeps` times per fixed step. Returns the region id.
    pub fn reduce_to_beam(&mut self, cells: &[usize], substeps: usize) -> Result<usize, String> {
        if cells.is_empty() || substeps == 0 {
            return Err("a beam region needs cells and at least one substep".into());
        }
        let mut seen = std::collections::HashSet::new();
        for &i in cells {
            if i >= self.voxels.len() {
                return Err(format!("REV {i} is out of range"));
            }
            if !seen.insert(i) {
                return Err(format!("REV {i} is listed twice"));
            }
            if self.aggregate_of.get(i).copied().flatten().is_some() {
                return Err(format!("REV {i} belongs to a rigid aggregate"));
            }
            if self
                .beam_regions
                .iter()
                .flatten()
                .any(|region| region.members.contains(&i))
            {
                return Err(format!("REV {i} already belongs to a beam region"));
            }
        }
        let rest: Vec<DVec3> = cells.iter().map(|&i| self.rest_positions[i]).collect();
        let count = cells.len() as f64;
        let mean = |value: &dyn Fn(&Material) -> f64| {
            cells.iter().map(|&i| value(&self.properties[i])).sum::<f64>() / count
        };
        let material = BeamMaterial {
            young: mean(&|m| f64::from(m.young)),
            shear: mean(&|m| f64::from(m.shear)),
            density: mean(&|m| f64::from(m.density)),
            yield_stress: cells
                .iter()
                .map(|&i| f64::from(self.properties[i].tensile_strength))
                .fold(f64::INFINITY, f64::min),
        };
        let (mut beam, groups) = reduce_lattice_along_x(&rest, CELL as f64, material)
            .ok_or("the REVs do not form a chain of slices along X")?;
        let mut root = vec![false; cells.len()];
        for &local in &groups[0] {
            root[local] = true;
        }
        if groups[0].iter().any(|&local| !self.voxels[cells[local]].kinematic) {
            return Err("the first slice must be fixed: it is the beam's clamp".into());
        }
        if cells
            .iter()
            .enumerate()
            .any(|(local, &i)| !root[local] && self.voxels[i].kinematic)
        {
            return Err("only the first slice may be fixed; other REVs must be free".into());
        }
        let positions: Vec<DVec3> = cells.iter().map(|&i| self.positions[i]).collect();
        let velocities: Vec<DVec3> = cells.iter().map(|&i| self.voxels[i].velocity.as_dvec3()).collect();
        beam.set_from_cells(&groups, &rest, &positions, &velocities);
        let saved_kinematic: Vec<bool> = cells.iter().map(|&i| self.voxels[i].kinematic).collect();
        let free: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(local, _)| !root[*local])
            .map(|(_, &i)| i)
            .collect();
        self.set_voxels_kinematic(&free, true).map_err(|error| error.to_string())?;
        let id = self.beam_regions.iter().position(Option::is_none).unwrap_or(self.beam_regions.len());
        let region = BeamRegion {
            beam,
            members: cells.to_vec(),
            rest,
            groups,
            root,
            saved_kinematic,
            substeps,
            primed: false,
        };
        if id == self.beam_regions.len() {
            self.beam_regions.push(Some(region));
        } else {
            self.beam_regions[id] = Some(region);
        }
        Ok(id)
    }

    /// Hand region `id` back to the cell solver with the beam's current state.
    pub fn expand_beam(&mut self, id: usize) -> Result<(), String> {
        let region = self
            .beam_regions
            .get_mut(id)
            .and_then(Option::take)
            .ok_or_else(|| format!("no beam region {id}"))?;
        let velocities: Vec<Vec3> = region.members.iter().map(|&i| self.voxels[i].velocity).collect();
        let free: Vec<usize> = region
            .members
            .iter()
            .enumerate()
            .filter(|(local, _)| !region.saved_kinematic[*local])
            .map(|(_, &i)| i)
            .collect();
        self.set_voxels_kinematic(&free, false).map_err(|error| error.to_string())?;
        for (local, &i) in region.members.iter().enumerate() {
            self.voxels[i].velocity = velocities[local];
        }
        Ok(())
    }

    /// Expand every region, for operations that cannot run with beams present.
    pub fn expand_all_beams(&mut self) {
        for id in 0..self.beam_regions.len() {
            if self.beam_regions[id].is_some() {
                let _ = self.expand_beam(id);
            }
        }
    }

    /// Kinetic plus strain energy of region `id`, joules.
    pub fn beam_region_energy_j(&self, id: usize) -> Option<f64> {
        self.beam_regions.get(id)?.as_ref().map(|region| region.beam.energy())
    }

    /// Peak stress of region `id` as a fraction of the material limit; 1.0 or more means
    /// the beam model is no longer valid and the region must be expanded.
    pub fn beam_region_validity(&self, id: usize) -> Option<f64> {
        self.beam_regions.get(id)?.as_ref().map(|region| region.beam.validity())
    }

    /// Step every beam region by `dt` seconds and place its REVs. Called at the start of
    /// each fixed step.
    pub(crate) fn advance_beam_regions(&mut self, dt: f64) {
        if self.beam_regions.iter().all(Option::is_none) {
            return;
        }
        let gravity = DVec3::from_array(self.config.gravity.map(f64::from));
        for id in 0..self.beam_regions.len() {
            let Some(mut region) = self.beam_regions[id].take() else {
                continue;
            };
            let nodes = region.beam.node_count();
            let mut loads = vec![[0.0f64; 6]; nodes];
            for node in 0..nodes {
                let force = gravity * region.beam.node_mass(node);
                loads[node][..3].copy_from_slice(&force.to_array());
            }
            for (node, cells) in region.groups.iter().enumerate() {
                let centre = region.beam.rest_position(node);
                for &local in cells {
                    let i = region.members[local];
                    let mass = f64::from(self.voxels[i].mass);
                    let applied = self.external_forces[i].as_dvec3()
                        + self.external_accelerations[i].as_dvec3() * mass;
                    let moment = (region.rest[local] - centre).cross(applied);
                    for axis in 0..3 {
                        loads[node][axis] += applied[axis];
                        loads[node][axis + 3] += moment[axis];
                    }
                }
            }
            if !region.primed {
                region.beam.sync_acceleration(&loads);
                region.primed = true;
            }
            let before = region.beam.displacement.clone();
            let sub_dt = dt / region.substeps as f64;
            for _ in 0..region.substeps {
                region.beam.step(sub_dt, &loads);
            }
            // Work done by the loads on the beam, so the ledger balances: constant loads
            // do F . du over the step.
            let work: f64 = (1..nodes)
                .map(|node| (0..6).map(|p| loads[node][p] * (region.beam.displacement[node][p] - before[node][p])).sum::<f64>())
                .sum();
            self.energy.external_work += work;

            let positions = region.beam.expand_positions(&region.groups, &region.rest);
            let velocities = region.beam.expand_velocities(&region.groups, &region.rest);
            for (node, cells) in region.groups.iter().enumerate() {
                let d = region.beam.displacement[node];
                let w = region.beam.velocity[node];
                let theta = DVec3::new(d[3], d[4], d[5]);
                let omega = DVec3::new(w[3], w[4], w[5]);
                for &local in cells {
                    if region.root[local] {
                        continue;
                    }
                    let i = region.members[local];
                    self.previous[i] = self.positions[i];
                    self.positions[i] = positions[local];
                    self.voxels[i].velocity = velocities[local].as_vec3();
                    let q = glam::DQuat::from_scaled_axis(theta).normalize();
                    if self.precise_rotations.len() == self.rotations.len() {
                        self.precise_rotations[i] = q;
                    }
                    self.rotations[i] = q.as_quat();
                    if i < self.angular_velocities.len() {
                        self.angular_velocities[i] = omega;
                    }
                }
            }
            let valid = region.beam.validity() < BEAM_VALIDITY_LIMIT;
            self.beam_regions[id] = Some(region);
            if !valid {
                // The reduced model no longer holds (stress near the material limit):
                // give the region back to the cell solver, which can fracture.
                let _ = self.expand_beam(id);
            }
        }
        self.mark_render_changed();
    }
}
