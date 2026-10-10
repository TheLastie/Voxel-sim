use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A 1 cm representative elementary volume. SI units throughout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Voxel {
    pub mass: f32,
    pub composition: Composition,
    pub internal_energy: f32,
    pub velocity: Vec3,
    pub microstructure: Microstructure,
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
