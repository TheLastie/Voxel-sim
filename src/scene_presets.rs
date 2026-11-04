//! Ordinary editable scene documents, built from reusable voxel model assets.
use std::path::Path;

use glam::{DVec3, Quat, Vec3};

use crate::{
    coordinates::CELL_M,
    documents::{
        save_document, EditorDocument, ModelInstanceDocument, SceneDocument, VoxelModelDocument,
    },
    material::Material,
    scene::{SceneBuilder, SceneConfig, VoxelSpawn},
};

/// Physical parameters for an authored, full-REV two-cube impact scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactSceneOptions {
    pub side: usize,
    pub gap_m: f64,
    pub approach_speed_m_s: f32,
    pub fixed_dt_s: f32,
    pub substeps: usize,
}

impl Default for ImpactSceneOptions {
    fn default() -> Self {
        Self {
            side: 16,
            gap_m: 0.08,
            approach_speed_m_s: 0.2,
            fixed_dt_s: 0.01,
            substeps: 4,
        }
    }
}

impl ImpactSceneOptions {
    /// Validate all options that affect file contents before any files are made.
    pub fn validate(&self) -> Result<(), String> {
        if self.side < 2 {
            return Err("a cube needs at least two cells per side".into());
        }
        let side_cells = i32::try_from(self.side).map_err(|_| "cube coordinates exceed i32")?;
        self.side
            .checked_pow(3)
            .and_then(|count| count.checked_mul(2))
            .ok_or("cube voxel count overflow")?;
        if !self.gap_m.is_finite() || self.gap_m < 0.0 {
            return Err("cube gap must be finite and nonnegative".into());
        }
        let gap_cells_f = self.gap_m / CELL_M;
        let rounded_gap_cells = gap_cells_f.round();
        if !gap_cells_f.is_finite() || (gap_cells_f - rounded_gap_cells).abs() > 1e-6 {
            return Err("cube gap must align to the 1 cm REV lattice".into());
        }
        let gap_cells = i32::try_from(rounded_gap_cells as i64)
            .map_err(|_| "cube gap exceeds the REV coordinate range")?;
        let right_start = side_cells
            .checked_add(gap_cells)
            .ok_or("cube coordinates exceed i32")?;
        right_start
            .checked_add(side_cells - 1)
            .ok_or("cube coordinates exceed i32")?;
        if !self.approach_speed_m_s.is_finite() || self.approach_speed_m_s < 0.0 {
            return Err("approach speed must be finite and nonnegative".into());
        }
        if !self.fixed_dt_s.is_finite() || self.fixed_dt_s <= 0.0 || self.substeps == 0 {
            return Err(
                "fixed timestep must be positive and finite; substeps must be positive".into(),
            );
        }
        let mut config = SceneConfig::default();
        config.fixed_dt = self.fixed_dt_s;
        config.substeps = self.substeps;
        config.gravity = Vec3::ZERO.to_array();
        config.atmosphere.pressure_pa = 0.0;
        config.validate().map_err(str::to_owned)
    }
}

/// Create an editable two-cube scene plus one shared `.vxm` asset.
/// The cubes use the regular SceneDocument settings and the production solver.
pub fn create_impact_scene(path: &Path, side: usize) -> Result<SceneDocument, String> {
    create_impact_scene_with_options(
        path,
        &ImpactSceneOptions {
            side,
            ..ImpactSceneOptions::default()
        },
    )
}

/// Create an ordinary authored scene containing two approaching, connected
/// iron cubes. The shared model asset contains actual 1 cm REVs, not proxies.
pub fn create_impact_scene_with_options(
    path: &Path,
    options: &ImpactSceneOptions,
) -> Result<SceneDocument, String> {
    // Validate all options and coordinate arithmetic before creating any files.
    options.validate()?;
    let ImpactSceneOptions {
        side,
        gap_m,
        approach_speed_m_s,
        fixed_dt_s,
        substeps,
    } = *options;
    let side_cells = i32::try_from(side).map_err(|_| "cube coordinates exceed i32")?;
    if !path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("vxscene"))
    {
        return Err("scene documents must use the .vxscene extension".into());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Cubes");
    let iron = Material::component_id("iron").ok_or("iron material is unavailable")?;
    let mut config = SceneConfig::default();
    config.fixed_dt = fixed_dt_s;
    config.substeps = substeps;
    config.gravity = Vec3::ZERO.to_array();
    config.atmosphere.pressure_pa = 0.0;
    let mut builder = SceneBuilder::new(config)
        .with_gravity(Vec3::ZERO)
        .with_timestep(fixed_dt_s, substeps);
    for z in 0..side_cells {
        for y in 0..side_cells {
            for x in 0..side_cells {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let source = builder.build()?;
    let model = VoxelModelDocument::from_scene(format!("Iron cube {side} cells"), &source);
    model.validate()?;

    let mut document =
        SceneDocument::new(format!("Two iron cubes {side}³"), source.config.clone())?;
    for index in 0..2 {
        document.add_model(ModelInstanceDocument {
            asset_path: Path::new("pending-assets").join("cube.vxm"),
            translation_m: DVec3::new(
                if index == 0 {
                    0.0
                } else {
                    side as f64 * CELL_M + gap_m
                },
                0.0,
                0.0,
            ),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        })?;
        let mut settings = document.object_settings(index);
        settings.initial_velocity_m_s = Some(
            Vec3::X
                * if index == 0 {
                    approach_speed_m_s
                } else {
                    -approach_speed_m_s
                },
        );
        document.set_object_settings(index, settings)?;
    }
    document.validate()?;

    std::fs::create_dir_all(parent).map_err(|error| format!("create scene folder: {error}"))?;
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
    let asset = Path::new(&asset_folder).join("cube.vxm");
    save_document(&parent.join(&asset), &model)?;
    for instance in &mut document.models {
        instance.asset_path = asset.clone();
    }
    save_document(path, &document)?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::{load_document, VoxelModelDocument};
    use std::{path::PathBuf, time::SystemTime};

    fn unique_test_dir(label: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        PathBuf::from(format!(
            "E:/CodexHome/tmp/voxel-sim-{label}-{}-{timestamp}",
            std::process::id()
        ))
    }

    #[test]
    fn impact_scene_roundtrips_real_voxels_and_approach_settings() {
        let root = unique_test_dir("impact-roundtrip");
        let path = root.join("impact.vxscene");
        let options = ImpactSceneOptions {
            side: 3,
            gap_m: 0.03,
            approach_speed_m_s: 0.04,
            fixed_dt_s: 0.002,
            substeps: 2,
        };

        create_impact_scene_with_options(&path, &options).unwrap();

        let saved: SceneDocument = load_document(&path).unwrap();
        assert_eq!(saved.models.len(), 2);
        assert_eq!(saved.models[0].asset_path, saved.models[1].asset_path);
        assert_eq!(saved.config.fixed_dt, options.fixed_dt_s);
        assert_eq!(saved.config.substeps, options.substeps);
        assert_eq!(
            saved.object_settings(0).initial_velocity_m_s,
            Some(Vec3::X * options.approach_speed_m_s)
        );
        assert_eq!(
            saved.object_settings(1).initial_velocity_m_s,
            Some(Vec3::NEG_X * options.approach_speed_m_s)
        );

        let model: VoxelModelDocument =
            load_document(&root.join(&saved.models[0].asset_path)).unwrap();
        assert_eq!(model.voxels.len(), options.side.pow(3));
        assert!(model.voxels.iter().all(|voxel| {
            voxel.state.composition.components.len() == 1
                && voxel.state.composition.components[0].0
                    == Material::component_id("iron").unwrap()
        }));

        let scene = saved.build_scene(&path).unwrap();
        assert_eq!(scene.voxels.len(), 2 * options.side.pow(3));
        let left_max_x = scene
            .positions
            .iter()
            .filter(|position| position.x < options.side as f64 * CELL_M)
            .map(|position| position.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let right_min_x = scene
            .positions
            .iter()
            .filter(|position| position.x >= options.side as f64 * CELL_M)
            .map(|position| position.x)
            .fold(f64::INFINITY, f64::min);
        assert!(((right_min_x - left_max_x) - CELL_M - options.gap_m).abs() < 1e-9);
        assert!(scene
            .voxels
            .iter()
            .take(options.side.pow(3))
            .all(|voxel| { voxel.velocity == Vec3::X * options.approach_speed_m_s }));
        assert!(scene
            .voxels
            .iter()
            .skip(options.side.pow(3))
            .all(|voxel| { voxel.velocity == Vec3::NEG_X * options.approach_speed_m_s }));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_impact_options_are_rejected_before_creating_files() {
        let root = unique_test_dir("impact-invalid");
        let path = root.join("impact.vxscene");
        let valid = ImpactSceneOptions::default();
        let invalid = [
            ImpactSceneOptions { side: 1, ..valid },
            ImpactSceneOptions {
                side: usize::MAX,
                ..valid
            },
            ImpactSceneOptions {
                gap_m: f64::NAN,
                ..valid
            },
            ImpactSceneOptions {
                gap_m: -CELL_M,
                ..valid
            },
            ImpactSceneOptions {
                gap_m: CELL_M * 0.5,
                ..valid
            },
            ImpactSceneOptions {
                approach_speed_m_s: f32::INFINITY,
                ..valid
            },
            ImpactSceneOptions {
                approach_speed_m_s: -0.1,
                ..valid
            },
            ImpactSceneOptions {
                fixed_dt_s: 0.0,
                ..valid
            },
            ImpactSceneOptions {
                fixed_dt_s: f32::NAN,
                ..valid
            },
            ImpactSceneOptions {
                substeps: 0,
                ..valid
            },
        ];
        for options in invalid {
            assert!(create_impact_scene_with_options(&path, &options).is_err());
            assert!(!root.exists());
        }
    }
}
