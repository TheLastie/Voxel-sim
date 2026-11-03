use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

pub const THERMAL_CELLS_PER_AXIS: usize = 3;
pub const THERMAL_CELL_COUNT: usize = THERMAL_CELLS_PER_AXIS.pow(3);

/// A 1 cm representative elementary volume. SI units throughout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Voxel {
    /// Physical mass remains finite even when motion is constrained.
    pub mass: f32,
    /// A kinematic boundary condition, separate from physical mass.
    #[serde(default)]
    pub kinematic: bool,
    pub composition: Composition,
    pub internal_energy: f32,
    /// Signed sub-ULP correction that preserves small heat transfers which
    /// cannot be represented by the f32 display/solver cache above.
    #[serde(default)]
    pub internal_energy_remainder_j: f64,
    /// Optional energy field for the 3×3×3 thermal control volumes inside this
    /// mechanical REV. Uniform fields remain represented by `internal_energy`
    /// alone, so cold, equilibrated regions do not allocate 27 values.
    #[serde(default)]
    pub(crate) thermal_cell_energy_j: Option<Box<[f64; THERMAL_CELL_COUNT]>>,
    pub velocity: Vec3,
    pub microstructure: Microstructure,
}

impl Voxel {
    /// Exact conserved thermal energy represented by the f32 value and its
    /// compensated residual.
    pub fn total_internal_energy_j(&self) -> f64 {
        (self.internal_energy as f64 + self.internal_energy_remainder_j).max(0.0)
    }

    /// Add heat without discarding increments smaller than one f32 ULP.
    pub fn add_internal_energy_j(&mut self, joules: f64) {
        if let Some(delta) = self.update_internal_energy_scalar(joules) {
            self.distribute_thermal_energy_delta(delta);
        }
    }

    /// Update the conserved REV energy immediately but defer applying the
    /// spatially uniform increment to its 3×3×3 thermal cells. Callers may
    /// batch positive increments and distribute their sum once after the
    /// operation. The scalar energy and compensated residual remain current,
    /// so material state reads during the operation still observe each heat
    /// increment in sequence.
    pub(crate) fn add_internal_energy_j_deferred_thermal_cells(&mut self, joules: f64) -> f64 {
        self.update_internal_energy_scalar(joules).unwrap_or(0.0)
    }

    pub(crate) fn distribute_deferred_thermal_energy_delta(&mut self, delta: f64) {
        if delta != 0.0 {
            self.distribute_thermal_energy_delta(delta);
        }
    }

    fn update_internal_energy_scalar(&mut self, joules: f64) -> Option<f64> {
        if !joules.is_finite() {
            return None;
        }
        // Accumulate small increments beside (rather than into) the much
        // larger f32 energy. Adding them to the full f64 total first can lose
        // increments below that total's f64 ULP before the f32 rounding step.
        let before = self.total_internal_energy_j();
        let increment = self.internal_energy_remainder_j + joules;
        let raw_total = self.internal_energy as f64 + increment;
        if raw_total <= 0.0 {
            self.internal_energy = 0.0;
            self.internal_energy_remainder_j = 0.0;
            if let Some(cells) = self.thermal_cell_energy_j.as_mut() {
                cells.fill(0.0);
            }
            return None;
        }
        let rounded = raw_total.min(f32::MAX as f64) as f32;
        let represented_delta = rounded as f64 - self.internal_energy as f64;
        self.internal_energy = rounded;
        self.internal_energy_remainder_j = if raw_total >= f32::MAX as f64 {
            0.0
        } else {
            increment - represented_delta
        };
        let after = self.total_internal_energy_j();
        Some(after - before)
    }

    pub(crate) fn copy_thermal_cell_energies_j(&self, output: &mut [f64]) {
        assert_eq!(output.len(), THERMAL_CELL_COUNT);
        if let Some(cells) = &self.thermal_cell_energy_j {
            output.copy_from_slice(cells.as_slice());
        } else {
            let total = self.total_internal_energy_j();
            let per_cell = total / THERMAL_CELL_COUNT as f64;
            output.fill(per_cell);
            output[THERMAL_CELL_COUNT - 1] += total - per_cell * THERMAL_CELL_COUNT as f64;
        }
    }

    pub(crate) fn reconcile_thermal_cell_energy(&mut self) {
        let target = self.total_internal_energy_j();
        let Some(cells) = self.thermal_cell_energy_j.as_mut() else {
            return;
        };
        let current: f64 = cells.iter().sum();
        if current == target {
            return;
        }
        if target <= 0.0 {
            cells.fill(0.0);
        } else if current > 0.0 {
            let scale = target / current;
            for energy in cells.iter_mut() {
                *energy *= scale;
            }
        } else {
            cells.fill(target / THERMAL_CELL_COUNT as f64);
        }
        let corrected: f64 = cells[..THERMAL_CELL_COUNT - 1].iter().sum();
        cells[THERMAL_CELL_COUNT - 1] = (target - corrected).max(0.0);
    }

    pub(crate) fn set_thermal_cell_energies_j(&mut self, values: &[f64]) {
        assert_eq!(values.len(), THERMAL_CELL_COUNT);
        let mut cells = Box::new([0.0; THERMAL_CELL_COUNT]);
        cells.copy_from_slice(values);
        let total: f64 = cells.iter().sum();
        let rounded = total.min(f32::MAX as f64) as f32;
        self.internal_energy = rounded;
        self.internal_energy_remainder_j = if total >= f32::MAX as f64 {
            0.0
        } else {
            total - rounded as f64
        };
        let first = cells[0];
        let uniform_tolerance = total.abs().max(1.0) * f64::EPSILON * 32.0;
        if cells
            .iter()
            .all(|energy| (*energy - first).abs() <= uniform_tolerance)
        {
            self.thermal_cell_energy_j = None;
        } else {
            self.thermal_cell_energy_j = Some(cells);
        }
    }

    pub(crate) fn thermal_cell_energy_at(&self, index: usize) -> f64 {
        assert!(index < THERMAL_CELL_COUNT);
        if let Some(cells) = &self.thermal_cell_energy_j {
            cells[index]
        } else {
            self.total_internal_energy_j() / THERMAL_CELL_COUNT as f64
        }
    }

    fn distribute_thermal_energy_delta(&mut self, delta: f64) {
        let expected = self.total_internal_energy_j();
        let Some(cells) = self.thermal_cell_energy_j.as_mut() else {
            return;
        };
        if delta >= 0.0 {
            let increment = delta / THERMAL_CELL_COUNT as f64;
            for energy in cells.iter_mut().take(THERMAL_CELL_COUNT - 1) {
                *energy += increment;
            }
            cells[THERMAL_CELL_COUNT - 1] += delta - increment * (THERMAL_CELL_COUNT - 1) as f64;
            let actual: f64 = cells.iter().sum();
            cells[THERMAL_CELL_COUNT - 1] += expected - actual;
            debug_assert!(
                (cells.iter().sum::<f64>() - expected).abs() <= expected.abs() * 1e-12 + 1e-12
            );
        } else {
            let total: f64 = cells.iter().sum();
            if total > 0.0 {
                let scale = expected / total;
                for energy in cells.iter_mut() {
                    *energy *= scale;
                }
                let corrected: f64 = cells[..THERMAL_CELL_COUNT - 1].iter().sum();
                cells[THERMAL_CELL_COUNT - 1] = (expected - corrected).max(0.0);
            }
        }
    }

    /// f32 thermal energy passed to the material EOS. The residual is kept
    /// until enough transfers accumulate to change this representable value.
    pub fn thermal_energy_f32(&self) -> f32 {
        self.total_internal_energy_j().min(f32::MAX as f64) as f32
    }

    /// Whether this REV is fixed in world space by the scene constraints.
    pub fn is_kinematic(&self) -> bool {
        self.kinematic
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Composition {
    pub components: SmallVec<[(u16, f32); 8]>,
}
impl Composition {
    pub fn pure(component: u16) -> Self {
        Self {
            components: smallvec::smallvec![(component, 1.0)],
        }
    }
    pub fn validate(&self, count: usize) -> Result<(), String> {
        let mut sum = 0.0;
        for (n, &(id, fraction)) in self.components.iter().enumerate() {
            if id as usize >= count || !fraction.is_finite() || fraction <= 0.0 {
                return Err("Invalid component or mass fraction".into());
            }
            if self.components[..n].iter().any(|&(other, _)| other == id) {
                return Err("Duplicate component".into());
            }
            sum += fraction;
        }
        if (sum - 1.0).abs() > 1e-5 {
            return Err("Mass fractions must sum to one".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Microstructure {
    /// Local Z is the principal material direction.
    pub orientation: Quat,
    pub anisotropy: f32,
}
impl Microstructure {
    pub fn fibers(self) -> Vec3 {
        self.orientation * Vec3::Z
    }
}
