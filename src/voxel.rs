use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

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
        if !joules.is_finite() {
            return;
        }
        let total = (self.total_internal_energy_j() + joules).max(0.0);
        let rounded = total.min(f32::MAX as f64) as f32;
        self.internal_energy = rounded;
        self.internal_energy_remainder_j = total - rounded as f64;
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
