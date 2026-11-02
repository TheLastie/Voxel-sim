//! Conservative exterior-face detection for uniformly sized, axis-aligned REVs.
//!
//! This only identifies faces hidden by another cube. It does not change voxel
//! topology, bonds, occupancy, or physical state.

use rustc_hash::FxHashMap;

const ALL_FACES: u8 = 0b00_111111;
const MAX_RELATIVE_ROUNDOFF: f32 = 1.0e-4;

/// Return one six-bit exterior-face mask per cube center in `positions`.
///
/// Bits are ordered by axis and sign: `2 * axis` is the negative face and
/// `2 * axis + 1` is the positive face (X, then Y, then Z). Face bits are
/// cleared only when another cube's AABB meets the whole face within the
/// f32-rounding tolerance. Invalid positions or a nonpositive/nonfinite cell
/// size are handled conservatively by leaving the affected faces visible.
pub fn exterior_face_masks(positions: &[[f32; 3]], cell: f32) -> Vec<u8> {
    if positions.is_empty() {
        return Vec::new();
    }
    if !cell.is_finite() || cell <= 0.0 {
        return vec![ALL_FACES; positions.len()];
    }

    let origin = positions
        .iter()
        .copied()
        .find(|position| position.iter().all(|value| value.is_finite()));
    let Some(origin) = origin else {
        return vec![ALL_FACES; positions.len()];
    };

    let cell64 = f64::from(cell);
    let mut bins =
        FxHashMap::<[i64; 3], usize>::with_capacity_and_hasher(positions.len(), Default::default());
    let mut next = Vec::with_capacity(positions.len());
    for (index, position) in positions.iter().enumerate() {
        let key = quantize(*position, origin, cell64);
        if let Some(key) = key {
            next.push(bins.insert(key, index).unwrap_or(usize::MAX));
        } else {
            next.push(usize::MAX);
        }
    }

    let tolerance_cap = f64::from(cell) * f64::from(MAX_RELATIVE_ROUNDOFF);
    let mut masks = vec![ALL_FACES; positions.len()];
    for (index, position) in positions.iter().enumerate() {
        let Some(key) = quantize(*position, origin, cell64) else {
            continue;
        };
        for axis in 0..3 {
            for positive in [false, true] {
                let step = if positive { 1 } else { -1 };
                let Some(neighbor_axis) = key[axis].checked_add(step) else {
                    continue;
                };
                let mut neighbor_key = key;
                neighbor_key[axis] = neighbor_axis;
                let mut candidate = bins.get(&neighbor_key).copied().unwrap_or(usize::MAX);
                let mut covered = false;
                while candidate != usize::MAX {
                    if covers_face(
                        *position,
                        positions[candidate],
                        axis,
                        positive,
                        cell,
                        tolerance_cap,
                    ) {
                        covered = true;
                        break;
                    }
                    candidate = next[candidate];
                }
                if covered {
                    let sign = if positive { 1 } else { 0 };
                    let bit = 1 << (2 * axis + sign);
                    masks[index] &= !bit;
                }
            }
        }
    }
    masks
}

fn quantize(position: [f32; 3], origin: [f32; 3], cell: f64) -> Option<[i64; 3]> {
    let mut key = [0_i64; 3];
    for axis in 0..3 {
        if !position[axis].is_finite() {
            return None;
        }
        let coordinate = (f64::from(position[axis]) - f64::from(origin[axis])) / cell;
        // Leave one representable integer at either end for the neighbor +/- 1.
        if !coordinate.is_finite() || coordinate <= i64::MIN as f64 || coordinate >= i64::MAX as f64
        {
            return None;
        }
        key[axis] = coordinate.round() as i64;
    }
    Some(key)
}

fn covers_face(
    center: [f32; 3],
    neighbor: [f32; 3],
    axis: usize,
    positive: bool,
    cell: f32,
    tolerance_cap: f64,
) -> bool {
    if !neighbor.iter().all(|value| value.is_finite()) {
        return false;
    }

    let local_scale = center
        .iter()
        .chain(neighbor.iter())
        .fold(f64::from(cell), |scale, value| {
            scale.max(f64::from(value.abs()))
        });
    let tolerance = tolerance_cap.min(2.0 * f64::from(f32::EPSILON) * local_scale);
    for component in 0..3 {
        let delta = f64::from(neighbor[component]) - f64::from(center[component]);
        let expected = if component == axis {
            if positive {
                f64::from(cell)
            } else {
                -f64::from(cell)
            }
        } else {
            0.0
        };
        if (delta - expected).abs() > tolerance {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{exterior_face_masks, ALL_FACES};

    fn visible_face_count(masks: &[u8]) -> u32 {
        masks.iter().map(|mask| mask.count_ones()).sum()
    }

    #[test]
    fn single_cube_has_six_exterior_faces() {
        assert_eq!(
            exterior_face_masks(&[[0.0, 0.0, 0.0]], 0.01),
            vec![ALL_FACES]
        );
    }

    #[test]
    fn adjacent_cubes_hide_only_their_shared_faces() {
        let masks = exterior_face_masks(&[[0.008, 0.003, -0.002], [0.018, 0.003, -0.002]], 0.01);
        assert_eq!(masks[0] & (1 << 1), 0);
        assert_eq!(masks[1] & (1 << 0), 0);
        assert_eq!(visible_face_count(&masks), 10);
    }

    #[test]
    fn solid_three_by_three_by_three_lattice_has_only_outer_faces() {
        let mut positions = Vec::with_capacity(27);
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    positions.push([
                        0.003 + x as f32 * 0.01,
                        -0.007 + y as f32 * 0.01,
                        0.004 + z as f32 * 0.01,
                    ]);
                }
            }
        }
        assert_eq!(
            visible_face_count(&exterior_face_masks(&positions, 0.01)),
            54
        );
    }

    #[test]
    fn a_removed_center_cube_exposes_six_cavity_faces() {
        let mut positions = Vec::with_capacity(26);
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    if [x, y, z] != [1, 1, 1] {
                        positions.push([x as f32 * 0.01, y as f32 * 0.01, z as f32 * 0.01]);
                    }
                }
            }
        }
        assert_eq!(
            visible_face_count(&exterior_face_masks(&positions, 0.01)),
            60
        );
    }

    #[test]
    fn a_real_gap_preserves_both_opposing_faces() {
        let masks = exterior_face_masks(&[[0.0, 0.0, 0.0], [0.0101, 0.0, 0.0]], 0.01);
        assert_ne!(masks[0] & (1 << 1), 0);
        assert_ne!(masks[1] & (1 << 0), 0);
    }

    #[test]
    fn local_f32_roundoff_and_small_translation_keep_lattice_faces_hidden() {
        let first = [1.003_f32, -0.297, 0.103];
        let second = [first[0] + 0.01, first[1], first[2]];
        let masks = exterior_face_masks(&[first, second], 0.01);
        assert_eq!(masks[0] & (1 << 1), 0);
        assert_eq!(masks[1] & (1 << 0), 0);
    }

    #[test]
    fn separated_clusters_and_tangential_gaps_keep_faces_visible() {
        let separated = exterior_face_masks(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.01);
        assert_eq!(separated, vec![ALL_FACES, ALL_FACES]);

        let offset = 0.0001_f32;
        let tangent_gap = exterior_face_masks(&[[0.0, 0.0, 0.0], [0.01, offset, 0.0]], 0.01);
        assert_ne!(tangent_gap[0] & (1 << 1), 0);
        assert_ne!(tangent_gap[1] & (1 << 0), 0);
    }

    #[test]
    fn invalid_cell_or_position_is_conservative() {
        assert_eq!(
            exterior_face_masks(&[[0.0, 0.0, 0.0]], 0.0),
            vec![ALL_FACES]
        );
        assert_eq!(
            exterior_face_masks(&[[f32::NAN, 0.0, 0.0]], 0.01),
            vec![ALL_FACES]
        );
    }
}
