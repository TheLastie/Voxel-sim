//! Ordinary editable scene documents, built from reusable voxel model assets.
use std::path::Path;

use glam::{DVec3, Quat, Vec3};

use crate::{
    coordinates::CELL_M,
    documents::{save_document, ModelInstanceDocument, SceneDocument, VoxelModelDocument},
    material::Material,
    scene::{SceneBuilder, VoxelSpawn},
};

/// Create an editable two-cube scene plus one shared `.vxm` asset.
/// The cubes use the regular SceneDocument settings and the production solver.
pub fn create_impact_scene(path: &Path, side: usize) -> Result<SceneDocument, String> {
    if side < 2 {
        return Err("a cube needs at least two cells per side".into());
    }
    let side_cells = i32::try_from(side).map_err(|_| "cube coordinates exceed i32")?;
    side.checked_pow(3)
        .and_then(|count| count.checked_mul(2))
        .ok_or("cube voxel count overflow")?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|error| format!("create scene folder: {error}"))?;
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Cubes");
    // Allocate a fresh asset directory: an existing scene may share its assets.
    let mut sequence = 0u64;
    let asset_folder = loop {
        let name = if sequence == 0 {
            format!("{stem}-assets")
        } else {
            format!("{stem}-assets-{sequence}")
        };
        match std::fs::create_dir(parent.join(&name)) {
            Ok(()) => break name,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => sequence += 1,
            Err(error) => return Err(format!("create model asset folder: {error}")),
        }
    };
    let iron = Material::component_id("iron").ok_or("iron material is unavailable")?;
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.01, 4);
    for z in 0..side_cells {
        for y in 0..side_cells {
            for x in 0..side_cells {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let source = builder.build()?;
    let model = VoxelModelDocument::from_scene(format!("Iron cube {side} cells"), &source);
    let asset = Path::new(&asset_folder).join("cube.vxm");
    save_document(&parent.join(&asset), &model)?;
    let mut config = source.config.clone();
    config.atmosphere.pressure_pa = 0.0;
    let mut document = SceneDocument::new(format!("Two iron cubes {side}³"), config)?;
    for index in 0..2 {
        document.add_model(ModelInstanceDocument {
            asset_path: asset.clone(),
            translation_m: DVec3::new(
                if index == 0 {
                    0.0
                } else {
                    (side as f64 + 8.0) * CELL_M
                },
                0.0,
                0.0,
            ),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        })?;
        let mut settings = document.object_settings(index);
        settings.initial_velocity_m_s = Some(Vec3::X * if index == 0 { 2.0 } else { -2.0 });
        document.set_object_settings(index, settings)?;
    }
    save_document(path, &document)?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_cube_size_is_rejected_before_creating_files() {
        assert!(create_impact_scene(Path::new("unused.vxscene"), 1).is_err());
        assert!(create_impact_scene(Path::new("unused.vxscene"), usize::MAX).is_err());
    }
}
