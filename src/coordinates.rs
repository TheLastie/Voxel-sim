//! Centimetre-cell coordinates with a precise local displacement.
//!
//! Physics must subtract the cell indices before converting to floating point.
//! A rounded global/render position is an output projection, never an input
//! from which to reconstruct small elastic displacements.

use glam::{DVec3, Vec3};
use std::{error::Error, fmt};

pub const CELL_M: f64 = 0.01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateError {
    NonFinite,
    CellOverflow,
    PrecisionLoss,
}

impl fmt::Display for CoordinateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "position or displacement is not finite",
            Self::CellOverflow => "position exceeds the integer cell coordinate range",
            Self::PrecisionLoss => "displacement is too large to normalize at cell precision",
        })
    }
}

impl Error for CoordinateError {}

/// A point in the same 1 cm world lattice used by seeded material cells.
/// Local offsets are normalized into [0, CELL_M) on every axis. Their f64
/// precision is independent of distance from the world's origin.
/// This spatial anchor is not a material identity: translating a REV must not
/// replace its composition or history with material generated at the new cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellPosition {
    cell: [i64; 3],
    offset_m: DVec3,
}

impl CellPosition {
    pub fn at_cell_center(cell: [i64; 3]) -> Self {
        Self {
            cell,
            offset_m: DVec3::splat(CELL_M * 0.5),
        }
    }

    pub fn from_cell_offset(cell: [i64; 3], offset_m: DVec3) -> Result<Self, CoordinateError> {
        if !offset_m.is_finite() {
            return Err(CoordinateError::NonFinite);
        }
        let mut position = Self { cell, offset_m };
        for axis in 0..3 {
            let shift = (position.offset_m[axis] / CELL_M).floor();
            // i64::MAX as f64 rounds to 2^63, which is already out of range.
            if shift < i64::MIN as f64 || shift >= -(i64::MIN as f64) {
                return Err(CoordinateError::CellOverflow);
            }
            let shift = shift as i64;
            position.cell[axis] = position.cell[axis]
                .checked_add(shift)
                .ok_or(CoordinateError::CellOverflow)?;
            position.offset_m[axis] -= shift as f64 * CELL_M;
            // Correct a possible last-bit boundary error without clamping or
            // discarding displacement. Carry it into the neighbouring cell.
            if position.offset_m[axis] >= CELL_M {
                position.cell[axis] = position.cell[axis]
                    .checked_add(1)
                    .ok_or(CoordinateError::CellOverflow)?;
                position.offset_m[axis] -= CELL_M;
            } else if position.offset_m[axis] < 0.0 {
                position.cell[axis] = position.cell[axis]
                    .checked_sub(1)
                    .ok_or(CoordinateError::CellOverflow)?;
                position.offset_m[axis] += CELL_M;
            }
            if !(0.0..CELL_M).contains(&position.offset_m[axis]) {
                return Err(CoordinateError::PrecisionLoss);
            }
        }
        Ok(position)
    }

    pub fn cell(self) -> [i64; 3] {
        self.cell
    }

    pub fn offset_m(self) -> DVec3 {
        self.offset_m
    }

    pub fn translated(self, displacement_m: DVec3) -> Result<Self, CoordinateError> {
        if !displacement_m.is_finite() {
            return Err(CoordinateError::NonFinite);
        }
        Self::from_cell_offset(self.cell, self.offset_m + displacement_m)
    }

    /// `self - origin` in metres. Adjacent cells remain adjacent even when
    /// both global indices are too large to be distinguished by f32/f64.
    pub fn relative_to(self, origin: Self) -> DVec3 {
        let delta = std::array::from_fn::<_, 3, _>(|axis| {
            (self.cell[axis] as i128 - origin.cell[axis] as i128) as f64
        });
        DVec3::from_array(delta) * CELL_M + (self.offset_m - origin.offset_m)
    }

    /// Camera/local-origin projection for rendering, not a physics position.
    pub fn render_relative_to(self, origin: Self) -> Vec3 {
        self.relative_to(origin).as_vec3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_displacement_is_retained_far_from_world_origin() {
        let origin = CellPosition::at_cell_center([100_000_000_000, -100_000_000_000, 0]);
        let movement = DVec3::new(1e-8, -2e-8, 3e-8);
        let moved = origin.translated(movement).unwrap();
        assert!((moved.relative_to(origin) - movement).length() < 2e-18);
        assert_ne!(moved, origin);
    }

    #[test]
    fn crossing_positive_and_negative_cell_boundaries_preserves_displacement() {
        let origin = CellPosition::at_cell_center([15, -15, 3]);
        let movement = DVec3::new(0.017, -0.019, 0.006);
        let moved = origin.translated(movement).unwrap();
        assert_eq!(moved.cell(), [17, -17, 4]);
        assert!((moved.relative_to(origin) - movement).length() < 1e-17);
        assert!(moved.offset_m().cmpge(DVec3::ZERO).all());
        assert!(moved.offset_m().cmplt(DVec3::splat(CELL_M)).all());
        let returned = moved.translated(-movement).unwrap();
        assert!(returned.relative_to(origin).length() < 1e-17);
    }

    #[test]
    fn relative_coordinates_subtract_integer_indices_before_float_conversion() {
        let a = CellPosition::at_cell_center([i64::MAX - 1, i64::MIN + 1, 0]);
        let b = CellPosition::at_cell_center([i64::MAX, i64::MIN, 0]);
        assert_eq!(b.relative_to(a), DVec3::new(CELL_M, -CELL_M, 0.0));
    }

    #[test]
    fn nonfinite_and_overflowing_translations_are_rejected_without_mutation() {
        let p = CellPosition::at_cell_center([i64::MAX, 0, 0]);
        assert_eq!(
            p.translated(DVec3::X * CELL_M),
            Err(CoordinateError::CellOverflow)
        );
        assert_eq!(
            p.translated(DVec3::splat(f64::NAN)),
            Err(CoordinateError::NonFinite)
        );
        assert_eq!(p.cell(), [i64::MAX, 0, 0]);
    }
}
