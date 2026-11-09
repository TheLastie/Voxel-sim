//! Ordinary editable scene documents, built from reusable voxel model assets.
use std::path::Path;

use glam::{DVec3, Quat, Vec3};

use crate::{
    coordinates::CELL_M,
    documents::{
        load_document, save_document, EditorDocument, ModelInstanceDocument, SceneDocument, VoxelModelDocument,
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

/// Create an ordinary physical scene with an animated saved model above a
/// fixed, flat centimetre-REV floor. The input model is copied unchanged.
pub fn create_animated_model_scene(
    path: &Path,
    model_path: &Path,
    floor_cells: usize,
) -> Result<SceneDocument, String> {
    if !(8..=512).contains(&floor_cells) {
        return Err("flat floor size must be between 8 and 512 centimetre cells".into());
    }
    if path.exists() {
        return Err("animated scene output already exists; choose a new path".into());
    }
    if path.extension().and_then(|p| p.to_str()) != Some("vxscene") {
        return Err("animated scene output must end in .vxscene".into());
    }
    let model: VoxelModelDocument = load_document(model_path)?;
    model.validate()?;
    if model.skeleton.as_ref().is_none_or(|rig| rig.clips.is_empty()) {
        return Err("animated model needs a valid skeleton and at least one clip".into());
    }
    let minimum_z = model.voxels.iter().map(|v| v.rest_position_m.z - CELL_M * 0.5)
        .fold(f64::INFINITY, f64::min);
    // Leave a centimetre of clearance, so gravity settles the model onto the
    // floor through real contacts instead of starting with interpenetration.
    let translation_z = ((-minimum_z / CELL_M).ceil() + 1.0) * CELL_M;
    let stone = Material::component_id("stone").ok_or("stone material is unavailable")?;
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    let mut builder = SceneBuilder::new(config.clone());
    let half = floor_cells as i32 / 2;
    for y in 0..floor_cells as i32 {
        for x in 0..floor_cells as i32 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x-half, y-half, -1], stone).fixed());
        }
    }
    let floor_scene = builder.build()?;
    let floor = VoxelModelDocument::from_scene("Fixed flat ground", &floor_scene);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| format!("create scene folder: {e}"))?;
    let stem = path.file_stem().and_then(|p| p.to_str()).unwrap_or("animated-model");
    let mut sequence = 0;
    let folder = loop {
        let folder = if sequence == 0 { format!("{stem}-assets") }
            else { format!("{stem}-assets-{sequence}") };
        match std::fs::create_dir(parent.join(&folder)) {
            Ok(()) => break folder,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => sequence += 1,
            Err(e) => return Err(format!("create animated model asset folder: {e}")),
        }
    };
    let floor_asset = Path::new(&folder).join("flat-ground.vxm");
    let model_asset = Path::new(&folder).join("animated-model.vxm");
    save_document(&parent.join(&floor_asset), &floor)?;
    save_document(&parent.join(&model_asset), &model)?;
    let mut document = SceneDocument::new("Animated model on flat ground", config)?;
    for (asset_path, translation_m) in [
        (floor_asset, DVec3::ZERO),
        (model_asset, DVec3::new(0.0, 0.0, translation_z)),
    ] {
        document.add_model(ModelInstanceDocument {
            asset_path, translation_m, rotation: Quat::IDENTITY, scale: Vec3::ONE,
        })?;
    }
    document.validate()?;
    save_document(path, &document)?;
    Ok(document)
}

/// A sandbox to play with in the editor: a fixed stone ground; an oak tower 6×6×24; an iron ball rolling at the tower; an oak log lying on the ground to cut with
/// the editor's tools; an oak plank clamped in a stone post (a cantilever the budget
/// governor may run as a beam); and a loose pile of iron cubes. About 1 400 free REVs, so
/// every body keeps cell rotations. A `<scene>.physics.json` beside it switches on sleep
/// and the tier governor (5 ms budget) when a run starts.
pub fn create_playground_scene(path: &Path) -> Result<SceneDocument, String> {
    if path.extension().and_then(|p| p.to_str()) != Some("vxscene") {
        return Err("playground scene output must end in .vxscene".into());
    }
    if path.exists() {
        return Err("playground scene output already exists; choose a new path".into());
    }
    let id = |name: &str| Material::component_id(name).ok_or_else(|| format!("{name} material is unavailable"));
    let (stone, oak, iron) = (id("stone")?, id("oak")?, id("iron")?);
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 2;
    let model = |name: &str, cells: Vec<([i32; 3], u16, bool)>| -> Result<VoxelModelDocument, String> {
        let mut builder = SceneBuilder::new(config.clone());
        for (cell, material, fixed) in cells {
            let spawn = VoxelSpawn::at_cell(cell, material);
            builder = builder.add_voxel(if fixed { spawn.fixed() } else { spawn });
        }
        let model = VoxelModelDocument::from_scene(name, &builder.build()?);
        model.validate()?;
        Ok(model)
    };
    let boxed = |nx: i32, ny: i32, nz: i32, material: u16| -> Vec<([i32; 3], u16, bool)> {
        let mut cells = Vec::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    cells.push(([x, y, z], material, false));
                }
            }
        }
        cells
    };
    let ground = model("Stone ground", (-40..40).flat_map(|y| (-40..40).map(move |x| ([x, y, -1], stone, true))).collect())?;
    let block = model("Oak tower 6x6x24", boxed(6, 6, 24, oak))?;
    let ball = model("Iron ball", {
        let mut cells = Vec::new();
        for z in -3..3 {
            for y in -3..3 {
                for x in -3..3 {
                    let (cx, cy, cz) = (x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
                    if cx * cx + cy * cy + cz * cz <= 9.0 {
                        cells.push(([x, y, z], iron, false));
                    }
                }
            }
        }
        cells
    })?;
    let log = model("Oak log", boxed(30, 5, 5, oak))?;
    let cantilever = model("Plank in a stone post", {
        let mut cells = Vec::new();
        for z in 0..12 {
            for y in 0..4 {
                for x in 0..3 {
                    cells.push(([x, y, z], stone, true));
                }
            }
        }
        // Plank bonded to the post's top: its first slice sits inside the post column.
        for x in 2..32 {
            for y in 0..4 {
                cells.push(([x, y, 12], oak, x == 2));
            }
        }
        cells
    })?;
    let cube = model("Iron cube 3", boxed(3, 3, 3, iron))?;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| format!("create scene folder: {e}"))?;
    let stem = path.file_stem().and_then(|p| p.to_str()).unwrap_or("playground");
    let mut sequence = 0;
    let folder = loop {
        let folder = if sequence == 0 { format!("{stem}-assets") } else { format!("{stem}-assets-{sequence}") };
        match std::fs::create_dir(parent.join(&folder)) {
            Ok(()) => break folder,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => sequence += 1,
            Err(e) => return Err(format!("create playground asset folder: {e}")),
        }
    };
    let mut assets = Vec::new();
    for (file, document) in [
        ("ground.vxm", &ground), ("oak-block.vxm", &block), ("iron-ball.vxm", &ball),
        ("oak-log.vxm", &log), ("plank-post.vxm", &cantilever), ("iron-cube.vxm", &cube),
    ] {
        let asset = Path::new(&folder).join(file);
        save_document(&parent.join(&asset), document)?;
        assets.push(asset);
    }
    let (ground_a, block_a, ball_a, log_a, post_a, cube_a) =
        (&assets[0], &assets[1], &assets[2], &assets[3], &assets[4], &assets[5]);
    let c = CELL_M;
    // (asset, translation in metres, initial velocity)
    let mut placements: Vec<(&std::path::PathBuf, DVec3, Option<Vec3>)> = vec![(ground_a, DVec3::ZERO, None)];
    // One bonded tower: separate stacked blocks meant many simultaneous body-body contacts,
    // which the cell solver handled at 7-33 s per step (8 November 2026).
    placements.push((block_a, DVec3::new(-0.20, -0.03, c), None));
    placements.push((ball_a, DVec3::new(0.10, 0.0, 0.04), Some(Vec3::new(-1.5, 0.0, 0.0))));
    placements.push((log_a, DVec3::new(-0.15, 0.15, c), None));
    placements.push((post_a, DVec3::new(0.05, -0.25, 0.0), None));
    for (x, y) in [(0.20, 0.15), (0.26, 0.15), (0.32, 0.15), (0.26, 0.21)] {
        placements.push((cube_a, DVec3::new(x, y, c), None));
    }
    let mut document = SceneDocument::new("Playground", config)?;
    for (index, (asset, translation_m, velocity)) in placements.into_iter().enumerate() {
        document.add_model(ModelInstanceDocument {
            asset_path: asset.clone(), translation_m, rotation: Quat::IDENTITY, scale: Vec3::ONE,
        })?;
        if let Some(velocity) = velocity {
            let mut settings = document.object_settings(index);
            settings.initial_velocity_m_s = Some(velocity);
            document.set_object_settings(index, settings)?;
        }
    }
    document.validate()?;
    save_document(path, &document)?;
    let mut tiers = crate::physics_tiers::TierSettings::default();
    tiers.enabled = true;
    tiers.sleep = true;
    tiers.save(path)?;
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
    fn playground_scene_builds_settles_and_sleeps() {
        let root = unique_test_dir("playground");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("playground.vxscene");
        let document = create_playground_scene(&path).unwrap();
        let mut scene = document.build_scene(&path).unwrap();
        let tiers = crate::physics_tiers::TierSettings::load(&path).unwrap();
        assert!(tiers.enabled && tiers.sleep);
        tiers.apply(&mut scene).unwrap();
        let free = scene.voxels.iter().filter(|v| !v.kinematic).count();
        let started = std::time::Instant::now();
        let mut slowest = 0.0f64;
        let steps: usize = std::env::var("PLAYGROUND_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
        for step in 0..steps {
            let t = std::time::Instant::now();
            crate::solver::step(&mut scene, &mut None);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            slowest = slowest.max(ms);
            if step % 5 == 0 {
                eprintln!("PLAYGROUND step {step} {ms:.0} ms census {:?} cells={}", scene.tier_census(), scene.cell_dofs);
            }
            assert!(scene.positions.iter().all(|p| p.is_finite()));
        }
        let census = scene.tier_census();
        eprintln!(
            "PLAYGROUND free={free} total={} 300 steps in {:.1} s, slowest {slowest:.0} ms, census {census:?}",
            scene.voxels.len(),
            started.elapsed().as_secs_f64()
        );
        assert!(free > 1000 && free <= 4096, "free REVs {free}");
        assert_eq!(census.total(), scene.voxels.len());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn animated_scene_copies_model_and_restores_a_fixed_floor_and_actor() {
        use crate::documents::{SkeletonDocument, VoxelBoneWeightDocument};
        let root = unique_test_dir("animated-flat");
        std::fs::create_dir_all(&root).unwrap();
        let source_path = root.join("input.vxm");
        let path = root.join("flat.vxscene");
        let scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], 1)).build().unwrap();
        let mut model = VoxelModelDocument::from_scene("animated fixture", &scene);
        let mut rig = SkeletonDocument::with_root_bone("body", DVec3::ZERO);
        rig.add_animation_clip("idle", 1.0).unwrap();
        rig.voxel_weights = (0..2).map(|voxel_index| VoxelBoneWeightDocument {
            voxel_index, bone_index: 0, weight: 1.0,
        }).collect();
        model.skeleton = Some(rig);
        save_document(&source_path, &model).unwrap();
        let source_bytes = std::fs::read(&source_path).unwrap();
        let authored = create_animated_model_scene(&path, &source_path, 8).unwrap();
        let physical = authored.build_scene(&path).unwrap();
        assert_eq!(physical.voxels.len(), 66);
        assert!(physical.voxels[..64].iter().all(|v| v.kinematic));
        assert!(physical.voxels[64..].iter().all(|v| !v.kinematic));
        assert_eq!(physical.skeletal_actuators.len(), 1);
        assert_eq!(physical.skeletal_actuators[0].voxel_range(), 64..66);
        assert_eq!(physical.config.gravity, [0.0, 0.0, -9.81]);
        assert_eq!(std::fs::read(&source_path).unwrap(), source_bytes);
        let output_bytes = std::fs::read(&path).unwrap();
        assert!(create_animated_model_scene(&path, &source_path, 8).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), output_bytes);

        model.skeleton.as_mut().unwrap().clips.clear();
        let no_clips = root.join("no-clips.vxm");
        save_document(&no_clips, &model).unwrap();
        assert!(create_animated_model_scene(&root.join("invalid.vxscene"), &no_clips, 8).is_err());
        assert!(!root.join("invalid.vxscene").exists());
        std::fs::remove_dir_all(root).unwrap();
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
