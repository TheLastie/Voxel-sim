//! Export authored voxel models to common mesh formats.

use crate::{coordinates::CellPosition, documents::VoxelModelDocument};
use glam::DVec3;
use std::{
    collections::HashSet,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

const OBJ_FACE_CORNERS: [([i64; 3], [[f64; 3]; 4]); 6] = [
    (
        [1, 0, 0],
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
        ],
    ),
    (
        [-1, 0, 0],
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
        ],
    ),
    (
        [0, 1, 0],
        [
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
        ],
    ),
    (
        [0, -1, 0],
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
        ],
    ),
    (
        [0, 0, 1],
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
    ),
    (
        [0, 0, -1],
        [
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ],
    ),
];

fn lattice_key(position: DVec3, cell_size_m: f64) -> Result<[i64; 3], String> {
    let position = CellPosition::from_cell_offset([0; 3], position).map_err(|error| {
        format!("model position cannot be represented on the REV lattice: {error}")
    })?;
    let mut key = position.cell();
    let offset = position.offset_m();
    let half_cell = cell_size_m * 0.5;
    let tolerance = cell_size_m * 1e-6;
    for axis in 0..3 {
        if offset[axis] > half_cell + tolerance {
            key[axis] = key[axis]
                .checked_add(1)
                .ok_or_else(|| "model position exceeds the OBJ coordinate range".to_owned())?;
        }
    }
    Ok(key)
}

fn safe_object_name(name: &str) -> String {
    let name = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if name.is_empty() {
        "voxel_model".to_owned()
    } else {
        name
    }
}

/// Writes a watertight surface mesh of exposed REV faces. OBJ coordinates are metres.
/// The native `.vxm` remains the format for physical, thermal, and material state.
pub fn export_obj(document: &VoxelModelDocument, path: &Path) -> Result<usize, String> {
    if document.voxels.is_empty() {
        return Err("cannot export an empty voxel model".into());
    }
    if !document.cell_size_m.is_finite() || document.cell_size_m <= 0.0 {
        return Err("voxel model has an invalid cell size".into());
    }

    let mut keys = Vec::with_capacity(document.voxels.len());
    let mut occupied = HashSet::with_capacity(document.voxels.len());
    for voxel in &document.voxels {
        if !voxel.position_m.is_finite() {
            return Err("voxel model contains a non-finite position".into());
        }
        let key = lattice_key(voxel.position_m, document.cell_size_m)?;
        if !occupied.insert(key) {
            return Err(
                "multiple REVs occupy the same lattice cell; resolve overlaps before OBJ export"
                    .into(),
            );
        }
        keys.push(key);
    }

    let file = File::create(path).map_err(|error| format!("could not create OBJ file: {error}"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# Voxel model export; units are metres")
        .and_then(|_| writeln!(writer, "o {}", safe_object_name(&document.name)))
        .map_err(|error| format!("could not write OBJ header: {error}"))?;

    let mut vertex_index = 1_u64;
    let mut face_count = 0;
    let half_cell = document.cell_size_m * 0.5;
    for (voxel, key) in document.voxels.iter().zip(keys) {
        for (normal, corners) in OBJ_FACE_CORNERS {
            let mut neighbor = key;
            let adjacent = match normal.iter().position(|component| *component != 0) {
                Some(axis) => neighbor[axis]
                    .checked_add(normal[axis])
                    .map(|coordinate| {
                        neighbor[axis] = coordinate;
                        occupied.contains(&neighbor)
                    })
                    .unwrap_or(false),
                None => false,
            };
            if adjacent {
                continue;
            }

            for corner in corners {
                let point = voxel.position_m + DVec3::from_array(corner) * half_cell;
                writeln!(writer, "v {:.9} {:.9} {:.9}", point.x, point.y, point.z)
                    .map_err(|error| format!("could not write OBJ vertices: {error}"))?;
            }
            writeln!(
                writer,
                "f {} {} {} {}",
                vertex_index,
                vertex_index + 1,
                vertex_index + 2,
                vertex_index + 3
            )
            .map_err(|error| format!("could not write OBJ faces: {error}"))?;
            vertex_index += 4;
            face_count += 1;
        }
    }

    writer
        .flush()
        .map_err(|error| format!("could not finish OBJ file: {error}"))?;
    Ok(face_count)
}
