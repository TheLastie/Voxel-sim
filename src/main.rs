pub mod editor_settings;
mod render;
pub mod ui;
use voxel_physics_lab::documents::{
    generate_auto_bone_weights_from_rest_positions, load_document, save_document, EditorDocument,
    ModelInstanceDocument, SceneDocument, SkeletonDocument, VoxelModelDocument,
};
use voxel_physics_lab::material::Material;
use voxel_physics_lab::mesh_import::{
    import_mesh, VoxelizationMode, VoxelizationOptions, VoxelizationPreview,
};
use voxel_physics_lab::model_export::export_obj;
use voxel_physics_lab::model_history::{ModelEditHistory, ModelVoxelEdit};
use voxel_physics_lab::{scene, solver, tool};

use glam::{DVec3, IVec3, Quat, Vec3};
use render::{AuthoredPhysicsRenderSource, Camera, Renderer};
use scene::{Scene, SceneConfig, VoxelSpawn};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};
use tool::{Shape, Tool};
use ui::{EditorInfo, EditorTool, ReticleState, SceneSelectionMode, Ui, UiAction};
use voxel_physics_lab::world::{GeneratedWorld, GrowthScheduler, WorldCharacter, WORLD_SIDE_M};
use voxel_physics_lab::creature_inspector::{inspect as inspect_creature, CreatureFile, NamedCreature};
use voxel_physics_lab::run_panel::{RunPanel, RunPanelInput, RunState, SolverMode};
use voxel_physics_lab::scene_chemistry::ChemistrySettings;
use voxel_physics_lab::scene_welding::ThermalBondingSettings;
use voxel_physics_lab::world_physics::{self, GrabTarget, PhysicsFrame, PhysicsWorker};
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, WindowBuilder},
};

const RECORDING_DURATION_SECONDS: f64 = 17.6;
const PLAYBACK_FRAME_INTERVAL_SECONDS: f64 = 0.1;
const MAX_SIMULATED_SECONDS_PER_REDRAW: f64 = 0.040;
const MAX_SOLVER_WALL_TIME_PER_REDRAW: Duration = Duration::from_millis(40);
const MAX_BUILD_GESTURE_CELLS: i32 = 256;
const MAX_DRAW_PREVIEW_CELLS: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Interactive,
    Recording,
    Playback,
}

struct PlaybackFrame {
    time_s: f64,
    positions: Vec<DVec3>,
    rotations: Vec<glam::Quat>,
    bond_neighbors: Vec<u32>,
    bond_damage: Vec<f32>,
    broken: usize,
    clusters: usize,
    tool_pose: Option<(DVec3, glam::Quat)>,
}

#[derive(Clone, Copy)]
struct GrabbedCell {
    source_cell: Option<[i32; 3]>,
    scene_index: usize,
    distance_m: f32,
}

#[derive(Clone)]
enum SceneInstanceEdit {
    Translate(IVec3),
    RotateQuarterTurn {
        axis: IVec3,
        direction: i8,
    },
    Duplicate,
    Remove,
    Position(DVec3),
    Rotation(Vec3),
    Velocity(Vec3),
    Force(Vec3),
    InitialTemperature(Option<f32>),
    ThermalBonding(Option<ThermalBondingSettings>),
    Material {
        path: PathBuf,
        voxel_indices: Vec<u32>,
    },
}

struct AuthoredInstanceBounds {
    min: Vec3,
    max: Vec3,
    center: Vec3,
    sphere_radius: f32,
}

struct CameraSnapshot {
    yaw: f32,
    pitch: f32,
    radius: f32,
    target: Vec3,
    position: Vec3,
    freecam: bool,
    first_person: bool,
    free_speed_mps: f32,
    look_sensitivity: f32,
}

impl CameraSnapshot {
    fn capture(camera: &Camera) -> Self {
        Self {
            yaw: camera.yaw,
            pitch: camera.pitch,
            radius: camera.radius,
            target: camera.target,
            position: camera.position(),
            freecam: camera.is_freecam(),
            first_person: camera.is_first_person()
                || (camera.is_freecam() && camera.target.distance(camera.position()) < 1.0e-4),
            free_speed_mps: camera.free_speed_mps,
            look_sensitivity: camera.look_sensitivity,
        }
    }

    fn restore(self, camera: &mut Camera) {
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        if self.first_person {
            camera.set_first_person(self.position);
        } else if self.freecam {
            // `set_orbit` plus `enable_freecam` restores the private freecam
            // position without exposing renderer internals to the editor.
            let target = self.position - camera.forward() * self.radius;
            camera.set_orbit(target, self.radius);
        } else {
            camera.set_orbit(self.target, self.radius);
        }
        if self.freecam {
            camera.enable_freecam();
        }
        camera.free_speed_mps = self.free_speed_mps;
        camera.look_sensitivity = self.look_sensitivity;
    }
}

fn ray_cube_distance(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: glam::Quat,
) -> Option<f32> {
    ray_cube_distance_with_limit(origin, direction, center, rotation, 4.0)
}

fn ray_cube_distance_with_limit(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: glam::Quat,
    max_distance: f32,
) -> Option<f32> {
    let inverse = rotation.conjugate();
    let local_origin = inverse * (origin - center);
    let local_direction = inverse * direction;
    let mut near = 0.0f32;
    let mut far = max_distance;
    for axis in 0..3 {
        let o = local_origin[axis];
        let d = local_direction[axis];
        if d.abs() < 1.0e-6 {
            if o.abs() > scene::CELL * 0.5 {
                return None;
            }
            continue;
        }
        let a = (-scene::CELL * 0.5 - o) / d;
        let b = (scene::CELL * 0.5 - o) / d;
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if far < near {
            return None;
        }
    }
    (far >= 0.0 && near <= max_distance).then_some(near.max(0.0))
}

fn scene_model_names(document: &SceneDocument) -> Vec<String> {
    document
        .models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let name = model
                .asset_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Voxel model")
                .to_owned();
            format!("{name} #{}", index + 1)
        })
        .collect()
}

fn scene_model_inspector_info(
    document: &SceneDocument,
    index: usize,
) -> Option<ui::SceneModelInspectorInfo> {
    let instance = document.models.get(index)?;
    let (x, y, z) = instance.rotation.normalize().to_euler(glam::EulerRot::XYZ);
    let settings = document.object_settings(index);
    Some(ui::SceneModelInspectorInfo {
        name: instance
            .asset_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| format!("{name} #{}", index + 1))
            .unwrap_or_else(|| format!("Voxel model #{}", index + 1)),
        translation_m: instance.translation_m.to_array(),
        rotation_deg: [x, y, z].map(f32::to_degrees),
        scale: instance.scale.to_array(),
        velocity_m_s: settings
            .initial_velocity_m_s
            .unwrap_or(Vec3::ZERO)
            .to_array(),
        force_n: settings.force_n.to_array(),
        material_name: settings
            .material_asset
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|name| name.to_str())
            .unwrap_or("Model materials")
            .to_owned(),
    })
}

fn authored_instance_bounds(
    document: &SceneDocument,
    scene_path: &Path,
    instance_index: usize,
) -> Result<AuthoredInstanceBounds, String> {
    let instance = document
        .models
        .get(instance_index)
        .ok_or_else(|| format!("model instance index {instance_index} is out of bounds"))?;
    let model_path = if instance.asset_path.is_absolute() {
        instance.asset_path.clone()
    } else {
        scene_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&instance.asset_path)
    };
    let model: VoxelModelDocument = load_document(&model_path)?;
    if model.voxels.is_empty() {
        return Err("cannot frame an empty voxel model".into());
    }

    let rotation = instance.rotation.normalize();
    let translation = instance.translation_m.as_vec3();
    let (minimum, maximum) = model.voxels.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(minimum, maximum), voxel| {
            let position = rotation * voxel.position_m.as_vec3() + translation;
            (minimum.min(position), maximum.max(position))
        },
    );
    let half_cell = (voxel_physics_lab::coordinates::CELL_M * 0.5) as f32;
    let minimum = minimum - Vec3::splat(half_cell);
    let maximum = maximum + Vec3::splat(half_cell);
    let extent = maximum - minimum;
    if !minimum.is_finite() || !maximum.is_finite() || !extent.is_finite() {
        return Err("model instance bounds are not finite".into());
    }
    let center = minimum + extent * 0.5;
    let sphere_radius = extent.length() * 0.5;
    if !center.is_finite() || !sphere_radius.is_finite() || sphere_radius <= 0.0 {
        return Err("model instance bounds have an invalid radius".into());
    }
    Ok(AuthoredInstanceBounds {
        min: minimum,
        max: maximum,
        center,
        sphere_radius,
    })
}

fn ray_aabb_distance(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let mut near = 0.0f32;
    let mut far = f32::INFINITY;
    for axis in 0..3 {
        let component = direction[axis];
        if component.abs() < 1e-7 {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return None;
            }
            continue;
        }
        let first = (min[axis] - origin[axis]) / component;
        let second = (max[axis] - origin[axis]) / component;
        near = near.max(first.min(second));
        far = far.min(first.max(second));
        if far < near {
            return None;
        }
    }
    (far >= 0.0).then_some(near.max(0.0))
}

fn sync_scene_instance_selection(
    ui: &mut Ui,
    document: &SceneDocument,
    scene_path: &Path,
    selected: Option<usize>,
) {
    ui.set_active_scene_document(Some(document.name.clone()), scene_model_names(document));
    ui.selected_scene_model = selected;
    let info = selected.and_then(|index| scene_model_inspector_info(document, index));
    ui.set_selected_scene_model_info(info);
    ui.set_scene_simulation_inspector_info(
        selected.and_then(|index| document.simulation_settings.object_temperature_kelvin(index)),
        document.simulation_settings.thermal_bonding.clone(),
    );
    let bounds = selected
        .and_then(|index| authored_instance_bounds(document, scene_path, index).ok())
        .map(|bounds| {
            [
                bounds.min.to_array().map(f64::from),
                bounds.max.to_array().map(f64::from),
            ]
        });
    ui.set_scene_model_selection_bounds(bounds);
}

fn apply_scene_instance_edit(
    scene: &mut Scene,
    active_document: &mut Option<SceneDocument>,
    scene_path: Option<&Path>,
    ui: &mut Ui,
    live_simulated_s: f64,
    edit: SceneInstanceEdit,
) -> Result<(), String> {
    if !ui.paused
        || live_simulated_s > 1e-9
        || (scene.voxels.is_empty()
            && active_document
                .as_ref()
                .is_some_and(|document| !document.models.is_empty()))
    {
        return Err("Reset with R and pause before editing the authored scene.".into());
    }
    let scene_path = scene_path.unwrap_or_else(|| Path::new("Untitled.vxscene"));
    let mut updated = active_document
        .as_ref()
        .cloned()
        .ok_or_else(|| "no authored scene is open".to_owned())?;
    let selected_index = ui
        .selected_scene_model
        .filter(|&index| index < updated.models.len());
    let index = if matches!(&edit, SceneInstanceEdit::ThermalBonding(_)) {
        selected_index
    } else {
        Some(selected_index.ok_or_else(|| "select a model instance first".to_owned())?)
    };
    let selected = match edit {
        SceneInstanceEdit::Translate(delta_cells) => {
            let index = index.expect("object edits require a selected model");
            updated.translate_model_cells(index, delta_cells)?;
            Some(index)
        }
        SceneInstanceEdit::RotateQuarterTurn { axis, direction } => {
            let index = index.expect("object edits require a selected model");
            updated.rotate_model_quarter_turn(index, axis, direction)?;
            Some(index)
        }
        SceneInstanceEdit::Duplicate => {
            let index = index.expect("object edits require a selected model");
            let bounds = authored_instance_bounds(&updated, scene_path, index)?;
            let width_cells = ((bounds.max.x - bounds.min.x)
                / voxel_physics_lab::coordinates::CELL_M as f32)
                .ceil()
                .max(1.0) as i32;
            Some(updated.duplicate_model_cells(index, IVec3::new(width_cells + 1, 0, 0))?)
        }
        SceneInstanceEdit::Remove => {
            let index = index.expect("object edits require a selected model");
            updated.remove_model(index)?;
            (!updated.models.is_empty()).then_some(index.min(updated.models.len() - 1))
        }
        SceneInstanceEdit::Position(position) => {
            let index = index.expect("object edits require a selected model");
            updated.models[index].translation_m = position;
            Some(index)
        }
        SceneInstanceEdit::Rotation(degrees) => {
            let index = index.expect("object edits require a selected model");
            let radians = degrees * std::f32::consts::PI / 180.0;
            updated.models[index].rotation =
                Quat::from_euler(glam::EulerRot::XYZ, radians.x, radians.y, radians.z);
            Some(index)
        }
        SceneInstanceEdit::Velocity(velocity) => {
            let index = index.expect("object edits require a selected model");
            let mut settings = updated.object_settings(index);
            settings.initial_velocity_m_s = Some(velocity);
            updated.set_object_settings(index, settings)?;
            Some(index)
        }
        SceneInstanceEdit::Force(force) => {
            let index = index.expect("object edits require a selected model");
            let mut settings = updated.object_settings(index);
            settings.force_n = force;
            updated.set_object_settings(index, settings)?;
            Some(index)
        }
        SceneInstanceEdit::InitialTemperature(kelvin) => {
            let index = index.expect("object edits require a selected model");
            updated.set_object_initial_temperature(index, kelvin)?;
            Some(index)
        }
        SceneInstanceEdit::ThermalBonding(settings) => {
            updated.set_thermal_bonding(settings)?;
            selected_index
        }
        SceneInstanceEdit::Material {
            path,
            voxel_indices,
        } => {
            let index = index.expect("object edits require a selected model");
            let mut settings = updated.object_settings(index);
            if voxel_indices.is_empty() {
                settings.material_asset = Some(path);
                settings.voxel_materials.clear();
            } else {
                for voxel_index in voxel_indices {
                    settings
                        .voxel_materials
                        .retain(|item| item.voxel_index != voxel_index);
                    settings.voxel_materials.push(
                        voxel_physics_lab::documents::SceneVoxelMaterialOverride {
                            voxel_index,
                            material_asset: path.clone(),
                        },
                    );
                }
            }
            updated.set_object_settings(index, settings)?;
            Some(index)
        }
    };
    let runtime_scene = updated.build_scene(scene_path)?;
    *scene = runtime_scene;
    *active_document = Some(updated);
    let document = active_document
        .as_ref()
        .expect("document was just assigned");
    sync_scene_instance_selection(ui, document, scene_path, selected);
    ui.world_active = true;
    ui.editor_mode = true;
    ui.paused = true;
    ui.show_bonds = false;
    ui.editor_tool = EditorTool::Select;
    Ok(())
}

/// What the Chemistry controls of the inspector ask for.
#[derive(Clone, Debug, PartialEq)]
enum ChemistryEdit {
    Enabled(bool),
    Interval(f64),
    Recipe(PathBuf),
    BuiltinRecipes,
}

/// Change the chemistry of the open scene. This is the window's side of the
/// API shared with the MCP `set_scene_chemistry` tool: both use
/// `ChemistrySettings`. The settings are checked and applied to `scene` first
/// and written beside the scene only when that worked, so an error leaves the
/// saved file, the settings and the scene as they were.
fn edit_scene_chemistry(
    current: &ChemistrySettings,
    edit: ChemistryEdit,
    scene_path: Option<&Path>,
    scene: &mut Scene,
) -> Result<(ChemistrySettings, String), String> {
    let mut next = current.clone();
    match edit {
        ChemistryEdit::Enabled(enabled) => next.set_enabled(enabled),
        ChemistryEdit::Interval(seconds) => next.set_interval(seconds)?,
        ChemistryEdit::BuiltinRecipes => next.clear_recipe(),
        ChemistryEdit::Recipe(recipe) => match scene_path {
            Some(path) => next.set_recipe_file(path, &recipe, &scene.materials)?,
            None => next.set_recipe_file_absolute(&recipe, &scene.materials)?,
        },
    }
    // `apply` leaves the scene untouched when it fails; a failed write puts
    // the previous settings back on the scene.
    next.apply(scene, scene_path)?;
    let mut message = format!("Chemistry: {}", next.summary());
    match scene_path {
        Some(path) => {
            if let Err(error) = next.save(path) {
                let _ = current.apply(scene, scene_path);
                return Err(error);
            }
            message.push_str(" (saved)");
        }
        None => message.push_str(" (save the scene to keep it)"),
    }
    Ok((next, message))
}

/// The edit a Chemistry control asks for; `pick_recipe` runs the file dialog
/// of "Load recipe…" and returns `None` when it is cancelled.
fn chemistry_edit_from_action(
    action: UiAction,
    pick_recipe: impl FnOnce() -> Option<PathBuf>,
) -> Option<ChemistryEdit> {
    match action {
        UiAction::SetChemistryEnabled(enabled) => Some(ChemistryEdit::Enabled(enabled)),
        UiAction::SetChemistryInterval { seconds } => Some(ChemistryEdit::Interval(seconds)),
        UiAction::UseBuiltinChemistryRecipes => Some(ChemistryEdit::BuiltinRecipes),
        UiAction::ChooseChemistryRecipe => pick_recipe().map(ChemistryEdit::Recipe),
        _ => None,
    }
}

#[cfg(test)]
mod chemistry_window_tests {
    use super::*;
    use voxel_physics_lab::documents::{save_document, EditorDocument};

    fn project(name: &str) -> (PathBuf, PathBuf, Scene) {
        let dir = PathBuf::from(format!(
            "E:/voxel-window-chemistry-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let scene_path = dir.join("world.vxscene");
        let document = SceneDocument::new("world", SceneConfig::default()).unwrap();
        save_document(&scene_path, &document).unwrap();
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes").join("wood_rot.json"),
            dir.join("wood_rot.json"),
        )
        .unwrap();
        let scene = document.build_scene(&scene_path).unwrap();
        (dir, scene_path, scene)
    }

    #[test]
    fn every_chemistry_action_maps_to_its_edit_and_other_actions_to_none() {
        let never = || -> Option<PathBuf> { panic!("no dialog expected") };
        assert_eq!(
            chemistry_edit_from_action(UiAction::SetChemistryEnabled(true), never),
            Some(ChemistryEdit::Enabled(true))
        );
        assert_eq!(
            chemistry_edit_from_action(UiAction::SetChemistryInterval { seconds: 5.0 }, never),
            Some(ChemistryEdit::Interval(5.0))
        );
        assert_eq!(
            chemistry_edit_from_action(UiAction::UseBuiltinChemistryRecipes, never),
            Some(ChemistryEdit::BuiltinRecipes)
        );
        assert_eq!(
            chemistry_edit_from_action(UiAction::ChooseChemistryRecipe, || Some(PathBuf::from("r.json"))),
            Some(ChemistryEdit::Recipe(PathBuf::from("r.json")))
        );
        assert_eq!(chemistry_edit_from_action(UiAction::ChooseChemistryRecipe, || None), None);
        assert_eq!(chemistry_edit_from_action(UiAction::FrameSelection, never), None);
    }

    #[test]
    fn window_edits_apply_to_the_scene_and_are_saved_where_the_mcp_tool_reads_them() {
        let (dir, scene_path, mut scene) = project("edits");
        let mut settings = ChemistrySettings::default();
        let steps = [
            ChemistryEdit::Enabled(true),
            ChemistryEdit::Interval(600.0),
            ChemistryEdit::Recipe(dir.join("wood_rot.json")),
        ];
        for edit in steps {
            let (next, message) = edit_scene_chemistry(&settings, edit, Some(&scene_path), &mut scene).unwrap();
            assert!(message.contains("(saved)"), "{message}");
            settings = next;
        }
        assert!(scene.chemistry_enabled);
        let saved = ChemistrySettings::load(&scene_path).unwrap();
        assert_eq!(saved, settings);
        assert_eq!(saved.interval_s, 600.0);
        assert_eq!(saved.recipe_file.as_deref(), Some("wood_rot.json"));

        let (builtin, _) =
            edit_scene_chemistry(&settings, ChemistryEdit::BuiltinRecipes, Some(&scene_path), &mut scene).unwrap();
        assert!(builtin.recipe_file.is_none());
        let (off, _) =
            edit_scene_chemistry(&builtin, ChemistryEdit::Enabled(false), Some(&scene_path), &mut scene).unwrap();
        assert!(!scene.chemistry_enabled);
        assert_eq!(ChemistrySettings::load(&scene_path).unwrap(), off);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn failed_window_edits_show_an_error_and_change_nothing() {
        let (dir, scene_path, mut scene) = project("errors");
        let (settings, _) =
            edit_scene_chemistry(&ChemistrySettings::default(), ChemistryEdit::Enabled(true), Some(&scene_path), &mut scene).unwrap();
        let saved = std::fs::read_to_string(ChemistrySettings::sidecar_path(&scene_path)).unwrap();
        std::fs::write(dir.join("broken.json"), "{ nope").unwrap();
        for edit in [
            ChemistryEdit::Interval(-1.0),
            ChemistryEdit::Interval(f64::NAN),
            ChemistryEdit::Recipe(dir.join("missing.json")),
            ChemistryEdit::Recipe(dir.join("broken.json")),
        ] {
            let error = edit_scene_chemistry(&settings, edit.clone(), Some(&scene_path), &mut scene).unwrap_err();
            assert!(!error.is_empty(), "{edit:?}");
            assert!(scene.chemistry_enabled && scene.chemistry_interval_seconds() == 0.0, "{edit:?} touched the scene");
            assert_eq!(std::fs::read_to_string(ChemistrySettings::sidecar_path(&scene_path)).unwrap(), saved);
        }
        // A scene whose directory cannot take the settings file reports it and keeps its settings.
        let missing_dir = dir.join("no_such_dir").join("x.vxscene");
        let error = edit_scene_chemistry(&ChemistrySettings::default(), ChemistryEdit::Enabled(true), Some(&missing_dir), &mut scene).unwrap_err();
        assert!(error.contains("cannot write"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_unsaved_scene_applies_now_and_moves_its_settings_when_saved() {
        let (dir, scene_path, mut scene) = project("unsaved");
        // Relative recipe paths need a scene file; an unsaved scene stores the absolute path.
        let (settings, message) = edit_scene_chemistry(
            &ChemistrySettings::default(),
            ChemistryEdit::Recipe(dir.join("wood_rot.json")),
            None,
            &mut scene,
        )
        .unwrap();
        assert!(message.contains("save the scene"), "{message}");
        assert!(Path::new(settings.recipe_file.as_deref().unwrap()).is_absolute());
        assert!(!ChemistrySettings::sidecar_path(&scene_path).exists());
        let (settings, _) = edit_scene_chemistry(&settings, ChemistryEdit::Enabled(true), None, &mut scene).unwrap();
        assert!(scene.chemistry_enabled);

        // Saving the scene writes the settings beside it with a path relative to it.
        let moved = settings.rebased(None, &scene_path);
        assert_eq!(moved.recipe_file.as_deref(), Some("wood_rot.json"));
        moved.save(&scene_path).unwrap();
        let mut fresh = SceneDocument::new("world", SceneConfig::default()).unwrap().build_scene(&scene_path).unwrap();
        ChemistrySettings::load(&scene_path).unwrap().apply(&mut fresh, Some(&scene_path)).unwrap();
        assert!(fresh.chemistry_enabled && fresh.has_custom_chemistry_registry());
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod scene_instance_edit_tests {
    use super::*;

    fn welding_settings() -> ThermalBondingSettings {
        ThermalBondingSettings {
            material_name: "iron".into(),
            minimum_mass_fraction: 0.5,
            minimum_temperature_kelvin: 800.0,
            minimum_impact_energy_j: 0.0,
            minimum_contact_pressure_pa: 0.0,
            minimum_dwell_seconds: 0.0,
        }
    }

    fn paused_ui() -> Ui {
        let mut ui = Ui::new(1.0, Vec3::ZERO, Vec3::Z);
        ui.paused = true;
        ui
    }

    #[test]
    fn thermal_bonding_edit_succeeds_without_selection_in_empty_authored_scene() {
        let mut scene = Scene::new(SceneConfig::default());
        let mut document = Some(
            SceneDocument::new("empty", SceneConfig::default()).expect("valid empty document"),
        );
        let mut ui = paused_ui();

        let result = apply_scene_instance_edit(
            &mut scene,
            &mut document,
            Some(Path::new("empty.vxscene")),
            &mut ui,
            0.0,
            SceneInstanceEdit::ThermalBonding(Some(welding_settings())),
        );

        assert!(result.is_ok(), "global welding edit failed: {result:?}");
        assert_eq!(ui.selected_scene_model, None);
        assert_eq!(
            document.unwrap().simulation_settings.thermal_bonding,
            Some(welding_settings())
        );
    }

    #[test]
    fn object_edit_still_requires_selection() {
        let mut scene = Scene::new(SceneConfig::default());
        let mut document = Some(
            SceneDocument::new("empty", SceneConfig::default()).expect("valid empty document"),
        );
        let mut ui = paused_ui();

        let result = apply_scene_instance_edit(
            &mut scene,
            &mut document,
            Some(Path::new("empty.vxscene")),
            &mut ui,
            0.0,
            SceneInstanceEdit::Position(DVec3::X),
        );

        assert_eq!(result.unwrap_err(), "select a model instance first");
    }
}

fn run_scene_instance_edit(
    scene: &mut Scene,
    active_document: &mut Option<SceneDocument>,
    scene_path: Option<&Path>,
    ui: &mut Ui,
    live_simulated_s: &mut f64,
    live_wall_s: &mut f64,
    accumulator: &mut f64,
    last: &mut Instant,
    edit: SceneInstanceEdit,
) -> bool {
    match apply_scene_instance_edit(
        scene,
        active_document,
        scene_path,
        ui,
        *live_simulated_s,
        edit.clone(),
    ) {
        Ok(()) => {
            *accumulator = 0.0;
            *last = Instant::now();
            if matches!(edit, SceneInstanceEdit::Remove) {
                *live_simulated_s = 0.0;
                *live_wall_s = 0.0;
            }
            ui.set_status_message(match edit {
                SceneInstanceEdit::Translate(delta) => format!(
                    "Moved model by {} cm. Save the scene to keep the change.",
                    delta.abs().max_element()
                ),
                SceneInstanceEdit::RotateQuarterTurn { direction, .. } => format!(
                    "Rotated model {}90°. Save the scene to keep the change.",
                    if direction > 0 { "+" } else { "−" }
                ),
                SceneInstanceEdit::Duplicate => {
                    "Duplicated model beside the original. Save the scene to keep the change."
                        .into()
                }
                SceneInstanceEdit::Remove => {
                    "Deleted model instance. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::Position(_) => {
                    "Object position updated. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::Rotation(_) => {
                    "Object rotation updated. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::Velocity(_) => {
                    "Initial velocity updated in m/s. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::Force(_) => {
                    "Applied force updated in N. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::InitialTemperature(_) => {
                    "Initial object temperature updated. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::ThermalBonding(_) => {
                    "Thermal welding settings updated. Save the scene to keep the change.".into()
                }
                SceneInstanceEdit::Material { .. } => {
                    "Material assignment updated. Save the scene to keep the change.".into()
                }
            });
            true
        }
        Err(error) => {
            ui.set_status_message(format!("Error: {error}"));
            false
        }
    }
}

fn scene_instance_voxel_range(
    document: &SceneDocument,
    scene_path: &Path,
    instance_index: usize,
) -> Result<std::ops::Range<usize>, String> {
    let parent = scene_path.parent().unwrap_or_else(|| Path::new("."));
    let mut start = 0usize;
    for (index, instance) in document.models.iter().enumerate() {
        let path = if instance.asset_path.is_absolute() {
            instance.asset_path.clone()
        } else {
            parent.join(&instance.asset_path)
        };
        let model: VoxelModelDocument = load_document(&path)?;
        let end = start
            .checked_add(model.voxels.len())
            .ok_or("scene REV count overflow")?;
        if index == instance_index {
            return Ok(start..end);
        }
        start = end;
    }
    Err("model instance index is out of bounds".into())
}

fn scene_instance_for_voxel(
    document: &SceneDocument,
    path: &Path,
    voxel_index: usize,
) -> Option<usize> {
    (0..document.models.len()).find(|&index| {
        scene_instance_voxel_range(document, path, index)
            .ok()
            .is_some_and(|range| range.contains(&voxel_index))
    })
}

fn select_scene_voxels(ui: &mut Ui, scene: &Scene, mut indices: Vec<usize>) {
    indices.retain(|&index| index < scene.voxels.len());
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        ui.set_model_selection(None, None, None);
        return;
    }
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut mass = 0.0;
    for &index in &indices {
        min = min.min(scene.positions[index]);
        max = max.max(scene.positions[index]);
        mass += scene.voxels[index].mass;
    }
    let cell = voxel_physics_lab::coordinates::CELL_M;
    let dimensions = ((max - min) / cell).round().as_uvec3() + glam::UVec3::ONE;
    let bounds = [
        (min - DVec3::splat(cell * 0.5)).to_array(),
        (max + DVec3::splat(cell * 0.5)).to_array(),
    ];
    let position = (indices.len() == 1).then(|| scene.positions[indices[0]].to_array());
    ui.set_model_box_selection(
        indices,
        bounds,
        dimensions.to_array(),
        [false; 64],
        position,
        mass,
    );
}

/// File dialogs belong to the desktop boundary; the saved material and scene
/// references use the same library loaders as a subsequent physics run.
fn scene_property_edit(
    action: UiAction,
    scene: &Scene,
    document: &SceneDocument,
    scene_path: &Path,
    ui: &Ui,
) -> Result<Option<SceneInstanceEdit>, String> {
    let direct = match &action {
        UiAction::SetScenePosition { value } => {
            Some(SceneInstanceEdit::Position(DVec3::from_array(*value)))
        }
        UiAction::SetSceneRotation { degrees } => {
            Some(SceneInstanceEdit::Rotation(Vec3::from_array(*degrees)))
        }
        UiAction::SetSceneVelocity { value } => {
            Some(SceneInstanceEdit::Velocity(Vec3::from_array(*value)))
        }
        UiAction::SetSceneForce { value } => {
            Some(SceneInstanceEdit::Force(Vec3::from_array(*value)))
        }
        UiAction::SetSceneInitialTemperature { kelvin } => {
            Some(SceneInstanceEdit::InitialTemperature(*kelvin))
        }
        UiAction::SetSceneThermalBonding(settings) => {
            Some(SceneInstanceEdit::ThermalBonding(settings.clone()))
        }
        _ => None,
    };
    if direct.is_some() {
        return Ok(direct);
    }
    let index = ui.selected_scene_model.ok_or("select an object first")?;
    let range = scene_instance_voxel_range(document, scene_path, index)?;
    let local_indices = if ui.scene_selection_mode == SceneSelectionMode::Voxel {
        let selected: Vec<_> = ui
            .selected_voxel_indices
            .iter()
            .copied()
            .filter(|index| range.contains(index))
            .map(|index| (index - range.start) as u32)
            .collect();
        if selected.is_empty() {
            return Err("select one or more voxels before assigning their material".into());
        }
        selected
    } else {
        Vec::new()
    };
    let first = ui
        .selected_voxel_indices
        .first()
        .copied()
        .filter(|index| range.contains(index))
        .unwrap_or(range.start);
    let material = scene
        .voxels
        .get(first)
        .and_then(|voxel| {
            voxel
                .composition
                .components
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
        })
        .and_then(|&(id, _)| scene.materials.get(id as usize))
        .ok_or("the selected object has no material")?
        .clone();
    let parent = scene_path.parent().unwrap_or_else(|| Path::new("."));
    let selected_path = match action {
        UiAction::CycleSceneMaterial => {
            let profiles = Material::table();
            let next = profiles
                .iter()
                .position(|item| item.name == material.name)
                .map_or(0, |index| (index + 1) % profiles.len());
            let profile = &profiles[next];
            let directory = parent.join("Materials");
            std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
            // A user-edited profile is never overwritten by the built-in picker.
            let mut path = directory.join(format!("{}.vxmat", profile.name));
            if path.exists() {
                let saved = voxel_physics_lab::material_asset::load_material(&path)?;
                if format!("{saved:?}") != format!("{profile:?}") {
                    path = directory.join(format!("{}-builtin.vxmat", profile.name));
                }
            }
            if !path.exists() {
                voxel_physics_lab::material_asset::save_material(&path, profile)?;
            }
            Some(path)
        }
        UiAction::LoadSceneMaterial => rfd::FileDialog::new()
            .add_filter("Voxel material", &["vxmat"])
            .set_directory(parent)
            .pick_file(),
        UiAction::SaveSceneMaterial | UiAction::CreateSceneMaterial => {
            let create = matches!(action, UiAction::CreateSceneMaterial);
            let name = if create {
                "New Material.vxmat".to_owned()
            } else {
                format!("{}.vxmat", material.name)
            };
            if let Some(mut path) = rfd::FileDialog::new()
                .add_filter("Voxel material", &["vxmat"])
                .set_directory(parent)
                .set_file_name(&name)
                .save_file()
            {
                path.set_extension("vxmat");
                let mut saved = material;
                if create {
                    saved.name = path
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("New Material")
                        .to_owned();
                }
                voxel_physics_lab::material_asset::save_material(&path, &saved)?;
                if !create {
                    return Ok(None);
                }
                Some(path)
            } else {
                None
            }
        }
        _ => return Err("unsupported scene property action".into()),
    };
    let Some(path) = selected_path else {
        return Ok(None);
    };
    voxel_physics_lab::material_asset::load_material(&path)?;
    let reference = path
        .strip_prefix(parent)
        .map(Path::to_path_buf)
        .unwrap_or(path);
    Ok(Some(SceneInstanceEdit::Material {
        path: reference,
        voxel_indices: local_indices,
    }))
}

fn update_model_ui(ui: &mut Ui, document: &VoxelModelDocument, selected: Option<usize>) {
    ui.set_active_model_document(
        Some(document.name.clone()),
        document.voxels.len(),
        document.materials.len(),
    );
    ui.set_rig_authoring_state(
        document
            .skeleton
            .as_ref()
            .map(|skeleton| {
                skeleton
                    .bones
                    .iter()
                    .map(|bone| bone.name.clone())
                    .collect()
            })
            .unwrap_or_default(),
        ui.selected_bone,
        document
            .skeleton
            .as_ref()
            .map_or(0, |skeleton| skeleton.clips.len()),
    );
    update_animation_timeline_ui(ui, document);
    if let Some(index) = selected.filter(|&index| index < document.voxels.len()) {
        update_model_ui_selection(ui, document, vec![index]);
    } else {
        ui.set_model_selection(None, None, None);
    }
}

fn update_animation_timeline_ui(ui: &mut Ui, document: &VoxelModelDocument) {
    let timeline = ui
        .selected_animation_clip
        .and_then(|clip_index| document.skeleton.as_ref()?.clips.get(clip_index))
        .map(|clip| {
            let key_times = clip
                .tracks
                .iter()
                .flat_map(|track| track.keyframes.iter().map(|key| key.time_s as f32))
                .collect();
            (clip.duration_s as f32, key_times)
        });
    if let Some((duration, key_times)) = timeline {
        ui.set_animation_timeline_info(duration, 30.0, key_times);
    } else {
        ui.set_animation_timeline_info(1.0, 30.0, Vec::new());
    }
}

type AutoWeightResult = (
    u64,
    usize,
    f64,
    Result<Vec<voxel_physics_lab::documents::VoxelBoneWeightDocument>, String>,
);

fn update_model_ui_selection(ui: &mut Ui, document: &VoxelModelDocument, indices: Vec<usize>) {
    let (indices, bounds_m, dimensions_cm, diagram, mass, position) =
        summarize_voxel_selection(document, indices);
    ui.set_model_box_selection(indices, bounds_m, dimensions_cm, diagram, position, mass);
}

fn snapshot_model_preserving_authoring(
    current: &VoxelModelDocument,
    scene: &Scene,
) -> Result<VoxelModelDocument, String> {
    if current.voxels.len() != scene.voxels.len() {
        return Err("cannot save: live REV topology differs from the authored model".into());
    }
    let mut snapshot = VoxelModelDocument::from_scene(current.name.clone(), scene);
    snapshot.groups = current.groups.clone();
    snapshot.motion_constraints = current.motion_constraints.clone();
    snapshot.skeleton = current.skeleton.clone();
    snapshot.validate()?;
    Ok(snapshot)
}

/// Build the editor's static REV preview while keeping authored rig/group data
/// in the original document. This is intentionally not a physics-play loader.
fn build_editor_preview_scene(
    document: &VoxelModelDocument,
    config: SceneConfig,
) -> Result<Scene, String> {
    document.build_editor_preview_scene(config)
}

fn invalidate_skeleton_weights_after_geometry_edit(document: &mut VoxelModelDocument) -> bool {
    let Some(skeleton) = document.skeleton.as_mut() else {
        return false;
    };
    if skeleton.voxel_weights.is_empty() {
        return false;
    }
    skeleton.voxel_weights.clear();
    true
}

fn model_has_unsupported_runtime_features(document: &VoxelModelDocument) -> bool {
    !document.groups.is_empty()
        || !document.motion_constraints.is_empty()
        || document.skeleton.is_some()
}

fn editor_ray(camera: &Camera, viewport: [u32; 4], pointer: (f32, f32)) -> Option<(Vec3, Vec3)> {
    let [left, top, width, height] = viewport;
    if width == 0 || height == 0 {
        return None;
    }
    let local_x = pointer.0 - left as f32;
    let local_y = pointer.1 - top as f32;
    if !(0.0..width as f32).contains(&local_x) || !(0.0..height as f32).contains(&local_y) {
        return None;
    }
    let ndc_x = 2.0 * local_x / width as f32 - 1.0;
    let ndc_y = 1.0 - 2.0 * local_y / height as f32;
    let forward = camera.forward().normalize_or_zero();
    let right = forward.cross(Vec3::Z).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let vertical_scale = 25.0f32.to_radians().tan();
    let direction = (forward
        + right * ndc_x * vertical_scale * width as f32 / height as f32
        + up * ndc_y * vertical_scale)
        .normalize_or_zero();
    (direction.length_squared() > 0.0).then_some((camera.position(), direction))
}

fn voxel_face_normal(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: Quat,
    distance: f32,
) -> Vec3 {
    let local_hit = rotation.conjugate() * (origin + direction * distance - center);
    let axis = (0..3)
        .max_by(|&a, &b| local_hit[a].abs().total_cmp(&local_hit[b].abs()))
        .unwrap_or(2);
    let mut local_normal = Vec3::ZERO;
    local_normal[axis] = local_hit[axis].signum();
    (rotation * local_normal).normalize_or_zero()
}

#[derive(Clone, Copy)]
struct HoveredFace {
    voxel_index: usize,
    center: Vec3,
    normal: Vec3,
}

struct SculptDrag {
    tool: EditorTool,
    start: HoveredFace,
    start_ray: (Vec3, Vec3),
    normal: IVec3,
    start_cell: IVec3,
    end_cell: IVec3,
    brush_size_cells: usize,
    occupied_cells: HashSet<IVec3>,
}

#[derive(Clone, Copy)]
struct BoxSelectionDrag {
    end_index: usize,
    start_cell: IVec3,
    end_cell: IVec3,
    dragged: bool,
}

struct MovePartDrag {
    selected: Vec<usize>,
    plane_origin: Vec3,
    plane_normal: Vec3,
    start_point: Vec3,
    delta_cells: IVec3,
}

#[derive(Clone, Copy)]
struct SceneInstanceMoveDrag {
    instance_index: usize,
    plane_origin: Vec3,
    plane_normal: Vec3,
    start_point: Vec3,
    bounds: (Vec3, Vec3),
    delta_cells: IVec3,
}

struct BoneMoveDrag {
    bone_index: usize,
    clip_index: Option<usize>,
    time_s: f64,
    original_translation: DVec3,
    parent_rotation: Quat,
    plane_origin: Vec3,
    plane_normal: Vec3,
    start_point: Vec3,
    last_delta: Vec3,
}

fn selection_bounds_for_cells(start: IVec3, end: IVec3, cell_size_m: f64) -> [[f64; 3]; 2] {
    let minimum = start.min(end).as_dvec3() * cell_size_m;
    let maximum = (start.max(end) + IVec3::ONE).as_dvec3() * cell_size_m;
    [minimum.to_array(), maximum.to_array()]
}

fn selection_diagram_xz(
    document: &VoxelModelDocument,
    indices: &[usize],
    minimum: IVec3,
    maximum: IVec3,
) -> [bool; 64] {
    let mut diagram = [false; 64];
    let size_x = (maximum.x - minimum.x + 1).max(1) as i64;
    let size_z = (maximum.z - minimum.z + 1).max(1) as i64;
    for &index in indices {
        let Some(voxel) = document.voxels.get(index) else {
            continue;
        };
        let cell = voxel_lattice_key(voxel.position_m, document.cell_size_m);
        let column = ((i64::from(cell.x - minimum.x) * 8) / size_x).clamp(0, 7) as usize;
        let row = ((i64::from(cell.z - minimum.z) * 8) / size_z).clamp(0, 7) as usize;
        diagram[row * 8 + column] = true;
    }
    diagram
}

fn selected_voxels_in_box(
    document: &VoxelModelDocument,
    start: IVec3,
    end: IVec3,
) -> (
    Vec<usize>,
    [[f64; 3]; 2],
    [u32; 3],
    [bool; 64],
    f32,
    Option<[f64; 3]>,
) {
    let minimum = start.min(end);
    let maximum = start.max(end);
    let indices = document
        .voxels
        .iter()
        .enumerate()
        .filter_map(|(index, voxel)| {
            let cell = voxel_lattice_key(voxel.position_m, document.cell_size_m);
            (cell.cmpge(minimum).all() && cell.cmple(maximum).all()).then_some(index)
        })
        .collect::<Vec<_>>();
    summarize_voxel_selection(document, indices)
}

fn summarize_voxel_selection(
    document: &VoxelModelDocument,
    mut indices: Vec<usize>,
) -> (
    Vec<usize>,
    [[f64; 3]; 2],
    [u32; 3],
    [bool; 64],
    f32,
    Option<[f64; 3]>,
) {
    indices.retain(|&index| index < document.voxels.len());
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        return (indices, [[0.0; 3]; 2], [0; 3], [false; 64], 0.0, None);
    }
    let mut minimum =
        voxel_lattice_key(document.voxels[indices[0]].position_m, document.cell_size_m);
    let mut maximum = minimum;
    for &index in &indices[1..] {
        let cell = voxel_lattice_key(document.voxels[index].position_m, document.cell_size_m);
        minimum = minimum.min(cell);
        maximum = maximum.max(cell);
    }
    let dimensions_cm = [minimum.x, minimum.y, minimum.z]
        .into_iter()
        .zip([maximum.x, maximum.y, maximum.z])
        .map(|(minimum, maximum)| {
            ((i64::from(maximum) - i64::from(minimum) + 1) as f64 * document.cell_size_m * 100.0)
                .round() as u32
        })
        .collect::<Vec<_>>();
    let dimensions_cm = [dimensions_cm[0], dimensions_cm[1], dimensions_cm[2]];
    let bounds_m = selection_bounds_for_cells(minimum, maximum, document.cell_size_m);
    let diagram = selection_diagram_xz(document, &indices, minimum, maximum);
    let total_mass = indices
        .iter()
        .filter_map(|&index| document.voxels.get(index))
        .map(|voxel| voxel.state.mass)
        .sum();
    let single_position =
        (indices.len() == 1).then(|| document.voxels[indices[0]].position_m.to_array());
    (
        indices,
        bounds_m,
        dimensions_cm,
        diagram,
        total_mass,
        single_position,
    )
}

fn connected_voxel_part(document: &VoxelModelDocument, start_index: usize) -> Vec<usize> {
    if start_index >= document.voxels.len() {
        return Vec::new();
    }
    let cell_size = document.cell_size_m;
    let occupied = document
        .voxels
        .iter()
        .enumerate()
        .map(|(index, voxel)| (voxel_lattice_key(voxel.position_m, cell_size), index))
        .collect::<HashMap<_, _>>();
    let neighbors = [
        IVec3::X,
        -IVec3::X,
        IVec3::Y,
        -IVec3::Y,
        IVec3::Z,
        -IVec3::Z,
    ];
    let mut visited = vec![false; document.voxels.len()];
    let mut pending = VecDeque::from([start_index]);
    visited[start_index] = true;
    let mut part = Vec::new();
    while let Some(index) = pending.pop_front() {
        part.push(index);
        let cell = voxel_lattice_key(document.voxels[index].position_m, cell_size);
        for offset in neighbors {
            if let Some(&neighbor) = occupied.get(&(cell + offset)) {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push_back(neighbor);
                }
            }
        }
    }
    part.sort_unstable();
    part
}

fn hovered_model_face(scene: &Scene, origin: Vec3, direction: Vec3) -> Option<HoveredFace> {
    let (index, distance, rotation) = scene
        .positions
        .iter()
        .enumerate()
        .filter_map(|(index, &position)| {
            let rotation = scene
                .rotations
                .get(index)
                .copied()
                .unwrap_or(Quat::IDENTITY);
            ray_cube_distance_with_limit(origin, direction, position.as_vec3(), rotation, 10_000.0)
                .map(|distance| (index, distance, rotation))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))?;
    let center = scene.positions[index].as_vec3();
    Some(HoveredFace {
        voxel_index: index,
        center,
        normal: voxel_face_normal(origin, direction, center, rotation, distance),
    })
}

fn hovered_face_at_pointer(
    ui: &Ui,
    scene: &Scene,
    camera: &Camera,
    pointer: (f32, f32),
) -> Option<HoveredFace> {
    if !ui.editor_mode
        || ui.menu_open
        || ui.scene_input_active()
        || !(ui.model_document_available
            || (ui.scene_document_available
                && ui.scene_selection_mode == SceneSelectionMode::Voxel))
    {
        return None;
    }
    let viewport = ui.scene_viewport()?;
    let [x, y, width, height] = viewport;
    if pointer.0 < x as f32
        || pointer.1 < y as f32
        || pointer.0 >= (x + width) as f32
        || pointer.1 >= (y + height) as f32
    {
        return None;
    }
    let (origin, direction) = editor_ray(camera, viewport, pointer)?;
    hovered_model_face(scene, origin, direction)
}

fn sculpt_drag_cell_at_pointer(
    ui: &Ui,
    camera: &Camera,
    pointer: (f32, f32),
    drag: &SculptDrag,
    cell_size_m: f64,
) -> Option<IVec3> {
    let viewport = ui.scene_viewport()?;
    let (origin, direction) = editor_ray(camera, viewport, pointer)?;
    let axis = (0..3).max_by(|&a, &b| drag.normal[a].abs().cmp(&drag.normal[b].abs()))?;
    let direction_axis = direction[axis];
    // When the view ray is almost parallel to the drawing plane, a tiny mouse
    // movement projects to an enormous world-space distance. Refuse that
    // unstable projection instead of turning it into a runaway voxel stroke.
    if direction_axis.abs() < 0.15 {
        return None;
    }
    // REVs are centered in their integer cells: cell i occupies [i, i+1]
    // and its center is i+0.5 cell widths from the model origin.
    let plane = (drag.start_cell[axis] as f32 + 0.5) * cell_size_m as f32;
    let distance = (plane - origin[axis]) / direction_axis;
    if !distance.is_finite() || distance <= 0.0 || distance > 10_000.0 {
        return None;
    }
    let point = origin + direction * distance;
    if !point.is_finite() {
        return None;
    }
    let mut cell = IVec3::new(
        (point.x as f64 / cell_size_m - 0.5).round() as i32,
        (point.y as f64 / cell_size_m - 0.5).round() as i32,
        (point.z as f64 / cell_size_m - 0.5).round() as i32,
    );
    cell[axis] = drag.start_cell[axis];
    Some(cell)
}

fn ray_plane_point(
    origin: Vec3,
    direction: Vec3,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = direction.dot(plane_normal);
    if !denominator.is_finite() || denominator.abs() < 0.15 {
        return None;
    }
    let distance = (plane_origin - origin).dot(plane_normal) / denominator;
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    let point = origin + direction * distance;
    point.is_finite().then_some(point)
}

fn drag_delta_cells(
    camera: &Camera,
    viewport: [u32; 4],
    pointer: (f32, f32),
    plane_origin: Vec3,
    plane_normal: Vec3,
    start_point: Vec3,
    cell_size_m: f64,
) -> Option<IVec3> {
    let (origin, direction) = editor_ray(camera, viewport, pointer)?;
    let point = ray_plane_point(origin, direction, plane_origin, plane_normal)?;
    let delta = (point - start_point).as_dvec3() / cell_size_m;
    if !delta.is_finite() {
        return None;
    }
    Some(IVec3::new(
        delta.x.round() as i32,
        delta.y.round() as i32,
        delta.z.round() as i32,
    ))
}

fn bone_parent_rotation(
    skeleton: &SkeletonDocument,
    clip_index: Option<usize>,
    time_s: f64,
    bone_index: usize,
) -> Result<Quat, String> {
    let mut global_rotations: Vec<Quat> = Vec::with_capacity(bone_index);
    for (index, bone) in skeleton.bones.iter().take(bone_index).enumerate() {
        let local_rotation = if let Some(clip_index) = clip_index {
            skeleton
                .sample_bone_local_pose(clip_index, index, time_s)?
                .rotation
        } else {
            bone.rest_rotation
        };
        let rotation = bone
            .parent
            .and_then(|parent| global_rotations.get(parent as usize).copied())
            .map_or(local_rotation, |parent: Quat| parent * local_rotation)
            .normalize();
        global_rotations.push(rotation);
    }
    let parent = skeleton
        .bones
        .get(bone_index)
        .and_then(|bone| bone.parent)
        .map(|parent| parent as usize);
    Ok(parent
        .and_then(|parent| global_rotations.get(parent).copied())
        .unwrap_or(Quat::IDENTITY))
}

fn bone_positions_for_drag(
    skeleton: &SkeletonDocument,
    clip_index: Option<usize>,
    time_s: f64,
) -> Result<Vec<DVec3>, String> {
    if let Some(clip_index) = clip_index {
        return skeleton.bone_global_positions_at(clip_index, time_s);
    }
    let mut positions: Vec<DVec3> = Vec::with_capacity(skeleton.bones.len());
    let mut rotations: Vec<Quat> = Vec::with_capacity(skeleton.bones.len());
    for bone in &skeleton.bones {
        let parent = bone.parent.map(|parent| parent as usize);
        let (position, rotation) = parent
            .filter(|&parent| parent < positions.len())
            .map(|parent| {
                (
                    positions[parent]
                        + (rotations[parent] * bone.rest_translation_m.as_vec3()).as_dvec3(),
                    rotations[parent] * bone.rest_rotation,
                )
            })
            .unwrap_or((bone.rest_translation_m, bone.rest_rotation));
        positions.push(position);
        rotations.push(rotation.normalize());
    }
    Ok(positions)
}

fn bone_drag_hit(
    document: &VoxelModelDocument,
    clip_index: Option<usize>,
    time_s: f64,
    origin: Vec3,
    direction: Vec3,
    viewport_height: u32,
) -> Option<(usize, Vec3)> {
    let skeleton = document.skeleton.as_ref()?;
    let positions = bone_positions_for_drag(skeleton, clip_index, time_s).ok()?;
    let mut best = None;
    for (index, position) in positions.into_iter().enumerate() {
        let center = position.as_vec3();
        let along_ray = (center - origin).dot(direction);
        if along_ray <= 0.0 {
            continue;
        }
        let nearest = origin + direction * along_ray;
        let pixel_world =
            along_ray * 2.0 * 25.0f32.to_radians().tan() / viewport_height.max(1) as f32;
        let radius = (document.cell_size_m as f32 * 0.8).clamp(0.006, 0.022);
        let tolerance = (radius + pixel_world * 7.0).clamp(0.012, 0.09);
        let distance = nearest.distance(center);
        if distance <= tolerance
            && best.map_or(true, |(_, best_distance, _): (usize, f32, Vec3)| {
                distance < best_distance
            })
        {
            best = Some((index, distance, center));
        }
    }
    best.map(|(index, _, center)| (index, center))
}

fn apply_bone_drag(
    document: &mut VoxelModelDocument,
    drag: &BoneMoveDrag,
    delta_world: Vec3,
) -> Result<(), String> {
    let skeleton = document
        .skeleton
        .as_mut()
        .ok_or_else(|| "this model has no skeleton".to_owned())?;
    let delta_local = (drag.parent_rotation.conjugate() * delta_world).as_dvec3();
    if let Some(clip_index) = drag.clip_index {
        let mut pose = skeleton.sample_bone_local_pose(clip_index, drag.bone_index, drag.time_s)?;
        pose.translation_m = drag.original_translation + delta_local;
        skeleton.set_keyframe(clip_index, drag.bone_index, pose)
    } else {
        let bone = skeleton
            .bones
            .get_mut(drag.bone_index)
            .ok_or_else(|| "bone index is out of bounds".to_owned())?;
        bone.rest_translation_m = drag.original_translation + delta_local;
        Ok(())
    }
}

fn aimed_physics_cell(frame: &PhysicsFrame, camera: &Camera) -> Option<GrabbedCell> {
    let origin = camera.position();
    let direction = camera.forward().normalize_or_zero();
    let mut nearest: Option<(f32, GrabbedCell)> = None;
    for cell in &frame.cells {
        let center = cell.world_position_m.as_vec3();
        let Some(ray_distance_m) = ray_cube_distance(origin, direction, center, cell.rotation)
        else {
            continue;
        };
        if nearest.is_none_or(|(best, _)| ray_distance_m < best) {
            nearest = Some((
                ray_distance_m,
                GrabbedCell {
                    source_cell: cell.source_cell,
                    scene_index: cell.scene_index,
                    // Keep the held point on a sphere centered at the player's eye,
                    // with the radius measured to the selected REV center.
                    distance_m: origin.distance(center),
                },
            ));
        }
    }
    nearest.map(|(_, hit)| hit)
}

fn capture_frame(scene: &Scene, tool: &Option<Tool>, time_s: f64) -> PlaybackFrame {
    PlaybackFrame {
        time_s,
        positions: scene.positions.clone(),
        rotations: scene.rotations.clone(),
        bond_neighbors: scene
            .bonds
            .iter()
            .flat_map(|axis| axis.iter().map(|bond| bond.neighbor))
            .collect(),
        bond_damage: scene
            .bonds
            .iter()
            .flat_map(|axis| axis.iter().map(|bond| bond.damage))
            .collect(),
        broken: scene.broken,
        clusters: scene.cluster_count,
        tool_pose: tool.as_ref().map(|tool| (tool.position, tool.rotation)),
    }
}

fn apply_frame(scene: &mut Scene, tool: &mut Option<Tool>, frame: &PlaybackFrame) {
    scene.positions.clone_from(&frame.positions);
    scene.rotations.clone_from(&frame.rotations);
    for ((bond, &neighbor), &damage) in scene
        .bonds
        .iter_mut()
        .flatten()
        .zip(&frame.bond_neighbors)
        .zip(&frame.bond_damage)
    {
        bond.neighbor = neighbor;
        bond.damage = damage;
    }
    scene.broken = frame.broken;
    scene.cluster_count = frame.clusters;
    if let (Some(tool), Some((position, rotation))) = (tool.as_mut(), frame.tool_pose) {
        tool.position = position;
        tool.rotation = rotation;
    }
}

fn launch(shape: Shape, scene: &Scene, ui: &Ui) -> Tool {
    let mut tool = Tool::new(shape, scene, ui.start(), ui.direction(), ui.speed);
    if matches!(shape, Shape::Wedge) {
        let rear = tool.cells.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        tool.mount_base(Vec3::new(rear, 0.0, 0.0), ui.direction())
            .expect("finite guide");
    }
    tool
}

fn fix_wood_base(scene: &mut Scene) {
    let wood_ids: Vec<_> = ["wood", "oak"]
        .into_iter()
        .filter_map(voxel_physics_lab::material::Material::component_id)
        .collect();
    if wood_ids.is_empty() {
        return;
    }
    let wood: Vec<_> = scene
        .voxels
        .iter()
        .enumerate()
        .filter(|(_, voxel)| {
            voxel
                .composition
                .components
                .iter()
                .any(|&(component, fraction)| wood_ids.contains(&component) && fraction > 0.0)
        })
        .map(|(index, _)| index)
        .collect();
    let Some(lowest_z) = wood
        .iter()
        .map(|&index| scene.positions[index].z)
        .min_by(f64::total_cmp)
    else {
        return;
    };
    let base: Vec<_> = wood
        .into_iter()
        .filter(|&index| (scene.positions[index].z - lowest_z).abs() <= scene::DCELL * 0.1)
        .collect();
    scene
        .fix_voxels(&base)
        .expect("wood base indices come from this scene");
    eprintln!("fixed wood base: {} REV at Z={lowest_z:.4} m", base.len());
}

fn selected_shape(args: &[String]) -> Option<Shape> {
    args.windows(2)
        .find(|pair| pair[0] == "--tool")
        .map(|pair| match pair[1].as_str() {
            "knife" => Shape::Blade,
            "rod" => Shape::Rod,
            "stone" => Shape::Rounded,
            _ => Shape::Wedge,
        })
}

fn two_cube_collision_scene(mut config: SceneConfig) -> Result<Scene, String> {
    config.gravity = [0.0, 0.0, -9.81];
    config.fixed_dt = 0.001;
    config.substeps = 1;
    config.iterations = 40;
    config.corotation_iterations = 2;
    // The pair coefficient is the sum of oak and concrete surface values,
    // scaled by this scene setting. Tangential loss is accounted for as heat.
    config.fracture.friction = 0.45;
    let oak = voxel_physics_lab::material::Material::component_id("oak")
        .ok_or_else(|| "oak material is unavailable".to_owned())?;
    let concrete = voxel_physics_lab::material::Material::component_id("concrete")
        .ok_or_else(|| "concrete material is unavailable".to_owned())?;
    let mut spawns =
        Vec::with_capacity(16 + config.ground_dimensions[0] * config.ground_dimensions[1]);
    for (cells, velocity) in [([-3, -2], Vec3::X * 0.025), ([1, 2], Vec3::NEG_X * 0.025)] {
        for z in 120..122 {
            for y in -1..=0 {
                for x in cells {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], oak).with_velocity(velocity));
                }
            }
        }
    }
    let [floor_x, floor_y, _] = config.ground_dimensions;
    for y in 0..floor_y {
        for x in 0..floor_x {
            let cell = [
                x as i32 - floor_x as i32 / 2,
                y as i32 - floor_y as i32 / 2,
                -1,
            ];
            spawns.push(VoxelSpawn::at_cell(cell, concrete).fixed());
        }
    }
    Scene::from_voxels(config, spawns)
}

fn friction_heat_road_scene(mut config: SceneConfig) -> Result<Scene, String> {
    config.gravity = [0.0, 0.0, -9.81];
    config.fixed_dt = 0.001;
    config.substeps = 1;
    config.iterations = 48;
    config.corotation_iterations = 2;
    config.fracture.friction = 0.45;
    let oak = voxel_physics_lab::material::Material::component_id("oak")
        .ok_or_else(|| "oak material is unavailable".to_owned())?;
    let concrete = voxel_physics_lab::material::Material::component_id("concrete")
        .ok_or_else(|| "concrete material is unavailable".to_owned())?;
    let mut spawns = Vec::with_capacity(240 * 9 + 27);
    for x in -120..120 {
        for y in -4..=4 {
            spawns.push(VoxelSpawn::at_cell([x, y, -1], concrete).fixed());
        }
    }
    let center = DVec3::new(-0.555, 0.005, 0.015);
    let omega = Vec3::new(0.0, 18.0, 0.0);
    for z in 0..3 {
        for y in -1..=1 {
            for x in -57..=-55 {
                let cell = [x, y, z];
                let position = DVec3::new(
                    f64::from(x) * scene::CELL as f64 + scene::CELL as f64 * 0.5,
                    f64::from(y) * scene::CELL as f64 + scene::CELL as f64 * 0.5,
                    f64::from(z) * scene::CELL as f64 + scene::CELL as f64 * 0.5,
                );
                let velocity =
                    Vec3::new(0.3, 0.0, 0.0) + omega.cross((position - center).as_vec3());
                spawns.push(VoxelSpawn::at_cell(cell, oak).with_velocity(velocity));
            }
        }
    }
    Scene::from_voxels(config, spawns)
}

fn cube_center(scene: &Scene, range: std::ops::Range<usize>) -> DVec3 {
    let sum = range
        .map(|index| scene.positions[index])
        .fold(DVec3::ZERO, |sum, position| sum + position);
    sum / 8.0
}

fn cube_velocity(scene: &Scene, range: std::ops::Range<usize>) -> Vec3 {
    let sum = range
        .map(|index| scene.voxels[index].velocity)
        .fold(Vec3::ZERO, |sum, velocity| sum + velocity);
    sum / 8.0
}

fn total_momentum(scene: &Scene) -> DVec3 {
    scene.voxels.iter().fold(DVec3::ZERO, |sum, voxel| {
        sum + DVec3::new(
            f64::from(voxel.mass * voxel.velocity.x),
            f64::from(voxel.mass * voxel.velocity.y),
            f64::from(voxel.mass * voxel.velocity.z),
        )
    })
}

fn angular_momentum(scene: &Scene, range: std::ops::Range<usize>) -> DVec3 {
    let total_mass: f64 = range
        .clone()
        .map(|index| f64::from(scene.voxels[index].mass))
        .sum();
    if total_mass <= 0.0 {
        return DVec3::ZERO;
    }
    let center = range
        .clone()
        .map(|index| scene.positions[index] * f64::from(scene.voxels[index].mass))
        .fold(DVec3::ZERO, |sum, weighted_position| {
            sum + weighted_position
        })
        / total_mass;
    range.fold(DVec3::ZERO, |sum, index| {
        let voxel = &scene.voxels[index];
        let momentum = DVec3::new(
            f64::from(voxel.mass * voxel.velocity.x),
            f64::from(voxel.mass * voxel.velocity.y),
            f64::from(voxel.mass * voxel.velocity.z),
        );
        sum + (scene.positions[index] - center).cross(momentum)
    })
}

fn kinetic_energy(scene: &Scene) -> f64 {
    scene.voxels.iter().fold(0.0, |sum, voxel| {
        sum + 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared())
    })
}

fn start_world_physics(
    world: &mut GeneratedWorld,
    character: &mut WorldCharacter,
    camera: &mut Camera,
    config: &SceneConfig,
    adaptive_physics: bool,
) -> Result<PhysicsWorker, String> {
    let center = world.view_bounds.map_or(
        Vec3::new(character.position.x, character.position.y, 0.0),
        |(min, max)| Vec3::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5, 0.0),
    );
    world.set_view_bounds(Some((
        center - Vec3::new(10.0, 10.0, 0.0),
        center + Vec3::new(10.0, 10.0, 512.0),
    )));
    let mut region = world_physics::contact_experiment(world, center, config.clone())?;
    region.scene.set_adaptive_substeps(adaptive_physics);
    let scene = &region.scene;
    eprintln!(
        "20m physics experiment: {} 1cm REVs ({} dynamic), dt={}s, substeps={}, adaptive={adaptive_physics}",
        scene.voxels.len(),
        scene.voxels.iter().filter(|v| !v.kinematic).count(),
        scene.config.fixed_dt,
        scene.config.substeps
    );
    character.position = Vec3::new(
        center.x + 0.9,
        center.y,
        world.terrain_height_at(center.x + 0.9, center.y),
    );
    camera.disable_freecam();
    camera.yaw = 0.0;
    camera.pitch = 0.95;
    camera.set_first_person(character.eye_position());
    PhysicsWorker::at_origin(region.scene, region.origin.as_dvec3()).map_err(|e| e.to_string())
}

fn set_editor_camera(character: &WorldCharacter, camera: &mut Camera, radius: f32) {
    camera.yaw = 0.62;
    camera.pitch = 0.42;
    camera.set_orbit(character.position + Vec3::Z * 0.55, radius);
}

fn frame_camera_on_scene(scene: &Scene, camera: &mut Camera) {
    let (center, radius) = if let Some((&first, rest)) = scene.positions.split_first() {
        let (minimum, maximum) = rest
            .iter()
            .copied()
            .fold((first, first), |(min, max), p| (min.min(p), max.max(p)));
        let center = (minimum + maximum) * 0.5;
        // Positions are cell centers, so a one-REV scene has zero center
        // extent. Include the REV's physical half-diagonal when framing it.
        let half_diagonal = (3.0_f64 * scene::DCELL * scene::DCELL).sqrt() * 0.5;
        let bounds_radius = (maximum - minimum).length() * 0.5 + half_diagonal;
        // Fit the enclosing sphere inside the vertical field of view with a
        // little margin. The old one-metre minimum made centimetre-scale
        // authored scenes effectively invisible.
        let fit_distance = bounds_radius / 25.0_f64.to_radians().sin() * 1.2;
        (
            center.as_vec3(),
            (fit_distance as f32).clamp(0.04, 10_000.0),
        )
    } else {
        (Vec3::ZERO, 5.0)
    };
    camera.set_orbit(center, radius);
    camera.yaw = -0.85;
    camera.pitch = 0.38;
    camera.enable_freecam();
}

fn pick_scene_document_path(
    save: bool,
    ui: &mut Ui,
    accumulator: &mut f64,
    last: &mut Instant,
) -> Option<PathBuf> {
    let was_paused = ui.paused;
    ui.paused = true;
    *accumulator = 0.0;
    let dialog = rfd::FileDialog::new().add_filter("Voxel scene", &["vxscene"]);
    let path = if save {
        dialog.save_file()
    } else {
        dialog.pick_file()
    };
    ui.paused = was_paused;
    *accumulator = 0.0;
    *last = Instant::now();
    path
}

fn ensure_scene_extension(mut path: PathBuf) -> PathBuf {
    if path.extension().and_then(|extension| extension.to_str()) != Some("vxscene") {
        path.set_extension("vxscene");
    }
    path
}

fn pick_voxel_model_document_path(
    save: bool,
    ui: &mut Ui,
    accumulator: &mut f64,
    last: &mut Instant,
) -> Option<PathBuf> {
    let was_paused = ui.paused;
    ui.paused = true;
    *accumulator = 0.0;
    let dialog = rfd::FileDialog::new().add_filter("Voxel model", &["vxm"]);
    let path = if save {
        dialog.save_file()
    } else {
        dialog.pick_file()
    };
    ui.paused = was_paused;
    *accumulator = 0.0;
    *last = Instant::now();
    path
}

fn pick_recipe_file_path(
    ui: &mut Ui,
    accumulator: &mut f64,
    last: &mut Instant,
) -> Option<PathBuf> {
    let was_paused = ui.paused;
    ui.paused = true;
    *accumulator = 0.0;
    let path = rfd::FileDialog::new()
        .add_filter("Chemistry recipes (JSON)", &["json"])
        .pick_file();
    ui.paused = was_paused;
    *accumulator = 0.0;
    *last = Instant::now();
    path
}

fn pick_mesh_import_source_path(
    ui: &mut Ui,
    accumulator: &mut f64,
    last: &mut Instant,
) -> Option<PathBuf> {
    let was_paused = ui.paused;
    ui.paused = true;
    *accumulator = 0.0;
    let path = rfd::FileDialog::new()
        .add_filter("OBJ or binary GLB mesh", &["obj", "glb"])
        .pick_file();
    ui.paused = was_paused;
    *accumulator = 0.0;
    *last = Instant::now();
    path
}

fn pick_obj_export_path(ui: &mut Ui, accumulator: &mut f64, last: &mut Instant) -> Option<PathBuf> {
    let was_paused = ui.paused;
    ui.paused = true;
    *accumulator = 0.0;
    let path = rfd::FileDialog::new()
        .add_filter("Wavefront OBJ mesh", &["obj"])
        .save_file();
    ui.paused = was_paused;
    *accumulator = 0.0;
    *last = Instant::now();
    path
}

fn ensure_voxel_model_extension(mut path: PathBuf) -> PathBuf {
    if path.extension().and_then(|extension| extension.to_str()) != Some("vxm") {
        path.set_extension("vxm");
    }
    path
}

fn ensure_obj_extension(mut path: PathBuf) -> PathBuf {
    if !path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("obj"))
    {
        path.set_extension("obj");
    }
    path
}

fn edit_model_at_ray(
    document: &VoxelModelDocument,
    scene: &Scene,
    origin: Vec3,
    direction: Vec3,
    tool: EditorTool,
    brush_size_cells: usize,
) -> Result<(Option<usize>, Option<ModelVoxelEdit>), String> {
    let nearest = scene
        .positions
        .iter()
        .enumerate()
        .filter_map(|(index, &position)| {
            let rotation = scene
                .rotations
                .get(index)
                .copied()
                .unwrap_or(Quat::IDENTITY);
            ray_cube_distance_with_limit(origin, direction, position.as_vec3(), rotation, 10_000.0)
                .map(|distance| (index, distance, rotation))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1));

    let Some((index, distance, rotation)) = nearest else {
        return Ok((None, None));
    };
    if tool == EditorTool::Select {
        return Ok((Some(index), None));
    }
    if document.skeleton.is_some() {
        return Err(
            "Sculpting is disabled for skeleton models until new REVs receive valid bone weights."
                .into(),
        );
    }
    if tool == EditorTool::EraseVoxel
        && (!document.groups.is_empty() || !document.motion_constraints.is_empty())
    {
        return Err(
            "Erasing this model could invalidate a group or motion-constraint reference.".into(),
        );
    }

    let center = scene.positions[index].as_vec3();
    let face = voxel_face_normal(origin, direction, center, rotation, distance);
    if face.length_squared() < 0.5 {
        return Err("Could not determine the hit face.".into());
    }
    let source_position = document.voxels[index].position_m;
    let cell = document.cell_size_m;

    if tool == EditorTool::EraseVoxel {
        let radius = (brush_size_cells.saturating_sub(1) / 2) as f64;
        let radius_squared = radius * radius + 1e-9;
        let mut indices = document
            .voxels
            .iter()
            .enumerate()
            .filter_map(|(voxel_index, voxel)| {
                let offset = (voxel.position_m - source_position) / cell;
                (offset.length_squared() <= radius_squared).then_some(voxel_index)
            })
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return Err("The erase brush did not touch any REVs.".into());
        }
        if indices.len() >= document.voxels.len() {
            return Err("A voxel model must keep at least one REV.".into());
        }
        // Removing from high to low keeps every remaining vector index valid.
        indices.sort_unstable_by(|a, b| b.cmp(a));
        let edit = if indices.len() == 1 {
            ModelVoxelEdit::Remove { index: indices[0] }
        } else {
            ModelVoxelEdit::Batch {
                edits: indices
                    .into_iter()
                    .map(|index| ModelVoxelEdit::Remove { index })
                    .collect(),
            }
        };
        return Ok((None, Some(edit)));
    }

    // Face placement follows the 1 cm REV lattice even when the rendered model
    // is rotated. The hit normal is reduced to its dominant lattice axis.
    let normal_axis = (0..3)
        .max_by(|&a, &b| face[a].abs().total_cmp(&face[b].abs()))
        .unwrap_or(2);
    let sign = if face[normal_axis] < 0.0 { -1 } else { 1 };
    let mut normal = IVec3::ZERO;
    normal[normal_axis] = sign;
    let mut transverse_axes = [0usize; 2];
    let mut next_axis = 0;
    for axis in 0..3 {
        if axis != normal_axis {
            transverse_axes[next_axis] = axis;
            next_axis += 1;
        }
    }

    if tool == EditorTool::AddVoxel {
        let target = source_position + normal.as_dvec3() * cell;
        if document
            .voxels
            .iter()
            .any(|voxel| voxel.position_m.distance_squared(target) < cell * cell * 0.0625)
        {
            return Err("The adjacent REV position is already occupied.".into());
        }
        let voxel = model_voxel_from_template(&document.voxels[index], target);
        let insert_at = document.voxels.len();
        return Ok((
            Some(insert_at),
            Some(ModelVoxelEdit::Insert {
                index: insert_at,
                voxel,
            }),
        ));
    }

    let source_cell = voxel_lattice_key(source_position, cell);
    let size = brush_size_cells.clamp(1, 31) as i32;
    let radius = (size - 1) / 2;
    let mut new_cells = Vec::new();
    match tool {
        EditorTool::AddBox => {
            // The block grows away from the clicked face; its cross-section is
            // centered on the hit REV and every center remains on the REV grid.
            for depth in 1..=size {
                for u in -radius..=radius {
                    for v in -radius..=radius {
                        let mut target = source_cell + normal * depth;
                        target[transverse_axes[0]] += u;
                        target[transverse_axes[1]] += v;
                        new_cells.push(target);
                    }
                }
            }
        }
        EditorTool::AddSphere => {
            // Centering one radius beyond the face makes the sphere touch the
            // source REV without burying its center inside the existing model.
            let sphere_center = source_cell + normal * (radius + 1);
            let radius_squared = f64::from(radius * radius);
            for x in -radius..=radius {
                for y in -radius..=radius {
                    for z in -radius..=radius {
                        let offset = IVec3::new(x, y, z);
                        if f64::from(x * x + y * y + z * z) <= radius_squared {
                            new_cells.push(sphere_center + offset);
                        }
                    }
                }
            }
        }
        EditorTool::AddCylinder => {
            for depth in 1..=size {
                for u in -radius..=radius {
                    for v in -radius..=radius {
                        if u * u + v * v > radius * radius {
                            continue;
                        }
                        let mut target = source_cell + normal * depth;
                        target[transverse_axes[0]] += u;
                        target[transverse_axes[1]] += v;
                        new_cells.push(target);
                    }
                }
            }
        }
        EditorTool::Select | EditorTool::AddVoxel | EditorTool::EraseVoxel => {
            return Err("This tool does not create a volume stamp.".into());
        }
    }

    let mut occupied = document
        .voxels
        .iter()
        .map(|voxel| voxel_lattice_key(voxel.position_m, cell))
        .collect::<HashSet<_>>();
    let mut insertions = Vec::with_capacity(new_cells.len());
    for target in new_cells {
        if !occupied.insert(target) {
            continue;
        }
        let position = (target.as_dvec3() + DVec3::splat(0.5)) * cell;
        let insert_at = document.voxels.len() + insertions.len();
        insertions.push(ModelVoxelEdit::Insert {
            index: insert_at,
            voxel: model_voxel_from_template(&document.voxels[index], position),
        });
    }
    let Some(first_new_index) = insertions.first().and_then(|edit| match edit {
        ModelVoxelEdit::Insert { index, .. } => Some(*index),
        _ => None,
    }) else {
        return Err("No empty 1 cm REVs are available at the chosen shape.".into());
    };
    let edit = if insertions.len() == 1 {
        insertions.remove(0)
    } else {
        ModelVoxelEdit::Batch { edits: insertions }
    };
    Ok((Some(first_new_index), Some(edit)))
}

fn model_voxel_from_template(
    template: &voxel_physics_lab::documents::ModelVoxel,
    position: DVec3,
) -> voxel_physics_lab::documents::ModelVoxel {
    let mut state = template.state.clone();
    state.velocity = Vec3::ZERO;
    state.kinematic = false;
    state.add_internal_energy_j(-state.total_internal_energy_j());
    voxel_physics_lab::documents::ModelVoxel {
        position_m: position,
        rest_position_m: position,
        state,
    }
}

fn voxel_lattice_key(position: DVec3, cell_size_m: f64) -> IVec3 {
    IVec3::new(
        (position.x / cell_size_m - 0.5).round() as i32,
        (position.y / cell_size_m - 0.5).round() as i32,
        (position.z / cell_size_m - 0.5).round() as i32,
    )
}

fn face_lattice_normal(face: Vec3) -> IVec3 {
    let axis = (0..3)
        .max_by(|&a, &b| face[a].abs().total_cmp(&face[b].abs()))
        .unwrap_or(2);
    let mut normal = IVec3::ZERO;
    normal[axis] = if face[axis] < 0.0 { -1 } else { 1 };
    normal
}

fn connected_voxel_row(start: IVec3, end: IVec3) -> Vec<IVec3> {
    let mut cells = vec![start];
    let delta = end - start;
    let axis = (0..3)
        .max_by_key(|&axis| delta[axis].unsigned_abs())
        .unwrap_or(0);
    let steps = delta[axis].unsigned_abs() as usize;
    let direction = delta[axis].signum();
    for step in 1..=steps {
        let mut cell = start;
        cell[axis] += direction * step as i32;
        cells.push(cell);
    }
    cells
}

fn sculpt_target_cells(drag: &SculptDrag) -> Result<Vec<IVec3>, String> {
    match drag.tool {
        EditorTool::AddVoxel => {
            let delta = drag.end_cell - drag.start_cell;
            let steps = delta.x.abs().max(delta.y.abs()).max(delta.z.abs());
            if steps > MAX_BUILD_GESTURE_CELLS {
                return Err(format!(
                    "Keep one Add REV drag within {MAX_BUILD_GESTURE_CELLS} cells; continue longer rows with another drag."
                ));
            }
            Ok(connected_voxel_row(drag.start_cell, drag.end_cell))
        }
        EditorTool::AddBox => {
            let normal_axis = (0..3)
                .find(|&axis| drag.normal[axis] != 0)
                .ok_or_else(|| "Could not determine the drawing plane.".to_owned())?;
            let transverse_axes = (0..3)
                .filter(|&axis| axis != normal_axis)
                .collect::<Vec<_>>();
            let min_u = drag.start_cell[transverse_axes[0]].min(drag.end_cell[transverse_axes[0]]);
            let max_u = drag.start_cell[transverse_axes[0]].max(drag.end_cell[transverse_axes[0]]);
            let min_v = drag.start_cell[transverse_axes[1]].min(drag.end_cell[transverse_axes[1]]);
            let max_v = drag.start_cell[transverse_axes[1]].max(drag.end_cell[transverse_axes[1]]);
            let width = (max_u - min_u + 1) as usize;
            let height = (max_v - min_v + 1) as usize;
            if width > MAX_BUILD_GESTURE_CELLS as usize
                || height > MAX_BUILD_GESTURE_CELLS as usize
                || width.saturating_mul(height) > (MAX_BUILD_GESTURE_CELLS as usize).pow(2)
            {
                return Err(format!(
                    "Keep each Block drag within {MAX_BUILD_GESTURE_CELLS} cells per side; continue larger shapes with another drag."
                ));
            }
            let mut cells = Vec::with_capacity(width.saturating_mul(height));
            for u in min_u..=max_u {
                for v in min_v..=max_v {
                    let mut target = drag.start_cell;
                    target[transverse_axes[0]] = u;
                    target[transverse_axes[1]] = v;
                    cells.push(target);
                }
            }
            Ok(cells)
        }
        _ => Err("Only Add REV and Block support drag drawing.".into()),
    }
}

fn inserted_positions(edit: Option<ModelVoxelEdit>) -> Vec<DVec3> {
    fn collect(edit: ModelVoxelEdit, positions: &mut Vec<DVec3>) {
        match edit {
            ModelVoxelEdit::Insert { voxel, .. } => positions.push(voxel.position_m),
            ModelVoxelEdit::Batch { edits } => {
                for edit in edits {
                    collect(edit, positions);
                }
            }
            ModelVoxelEdit::Remove { .. } | ModelVoxelEdit::Replace { .. } => {}
        }
    }

    let mut positions = Vec::new();
    if let Some(edit) = edit {
        collect(edit, &mut positions);
    }
    positions
}

fn sculpt_preview_geometry(
    document: &VoxelModelDocument,
    scene: &Scene,
    drag: &SculptDrag,
) -> Result<(Vec<Vec3>, Option<(Vec3, Vec3)>), String> {
    let cell_size = document.cell_size_m;
    let centers = if drag.tool == EditorTool::AddBox && drag.end_cell == drag.start_cell {
        let (_, edit) = edit_model_at_ray(
            document,
            scene,
            drag.start_ray.0,
            drag.start_ray.1,
            EditorTool::AddBox,
            drag.brush_size_cells,
        )?;
        inserted_positions(edit)
    } else {
        sculpt_target_cells(drag)?
            .into_iter()
            .filter(|cell| !drag.occupied_cells.contains(cell))
            .map(|cell| (cell.as_dvec3() + DVec3::splat(0.5)) * cell_size)
            .collect()
    };

    if centers.is_empty() {
        return Ok((Vec::new(), None));
    }
    let mut minimum = centers[0];
    let mut maximum = centers[0];
    for center in &centers[1..] {
        minimum = minimum.min(*center);
        maximum = maximum.max(*center);
    }
    let half_cell = DVec3::splat(cell_size * 0.5);
    let bounds = (
        (minimum - half_cell).as_vec3(),
        (maximum + half_cell).as_vec3(),
    );
    let preview_cells = if centers.len() <= MAX_DRAW_PREVIEW_CELLS {
        centers.into_iter().map(|center| center.as_vec3()).collect()
    } else {
        Vec::new()
    };
    Ok((preview_cells, Some(bounds)))
}

fn drag_model_edit(
    document: &VoxelModelDocument,
    scene: &Scene,
    drag: SculptDrag,
) -> Result<(Option<usize>, Option<ModelVoxelEdit>, usize), String> {
    let source_index = drag.start.voxel_index;
    let source = document
        .voxels
        .get(source_index)
        .ok_or_else(|| "The starting REV no longer exists.".to_owned())?;
    let cell = document.cell_size_m;

    if drag.tool == EditorTool::AddBox && drag.end_cell == drag.start_cell {
        let (selection, edit) = edit_model_at_ray(
            document,
            scene,
            drag.start_ray.0,
            drag.start_ray.1,
            EditorTool::AddBox,
            drag.brush_size_cells,
        )?;
        let count = edit.as_ref().map_or(0, |edit| match edit {
            ModelVoxelEdit::Insert { .. } => 1,
            ModelVoxelEdit::Batch { edits } => edits.len(),
            _ => 0,
        });
        return Ok((selection, edit, count));
    }

    let target_cells = sculpt_target_cells(&drag)?;
    let mut occupied = drag.occupied_cells;
    let mut insertions = Vec::new();
    for target in target_cells {
        if !occupied.insert(target) {
            continue;
        }
        let position = (target.as_dvec3() + DVec3::splat(0.5)) * cell;
        let index = document.voxels.len() + insertions.len();
        insertions.push(ModelVoxelEdit::Insert {
            index,
            voxel: model_voxel_from_template(source, position),
        });
    }
    let count = insertions.len();
    if count == 0 {
        return Err("The drawn cells are already occupied.".into());
    }
    let first_index = document.voxels.len();
    let edit = if count == 1 {
        insertions.remove(0)
    } else {
        ModelVoxelEdit::Batch { edits: insertions }
    };
    Ok((Some(first_index), Some(edit), count))
}

fn replay_model_history(
    document: &mut VoxelModelDocument,
    history: &mut ModelEditHistory,
    config: &SceneConfig,
    redo: bool,
) -> Result<Option<Scene>, String> {
    let mut updated_document = document.clone();
    let mut updated_history = history.clone();
    let changed = if redo {
        updated_history.redo(&mut updated_document)
    } else {
        updated_history.undo(&mut updated_document)
    }
    .map_err(|error| error.to_string())?;
    if !changed {
        return Ok(None);
    }
    let rebuilt = build_editor_preview_scene(&updated_document, config.clone())?;
    *document = updated_document;
    *history = updated_history;
    Ok(Some(rebuilt))
}

fn selected_part_move_edit(
    document: &VoxelModelDocument,
    selected: &[usize],
    axis: usize,
    direction: i8,
) -> Result<ModelVoxelEdit, String> {
    if axis >= 3 || !matches!(direction, -1 | 1) || selected.is_empty() {
        return Err("select a part and a valid movement direction".into());
    }
    let mut delta_cells = IVec3::ZERO;
    delta_cells[axis] = i32::from(direction);
    selected_part_translate_edit(document, selected, delta_cells)
}

fn selected_part_translate_edit(
    document: &VoxelModelDocument,
    selected: &[usize],
    delta_cells: IVec3,
) -> Result<ModelVoxelEdit, String> {
    if selected.is_empty() || delta_cells == IVec3::ZERO {
        return Err("select a part and drag it by at least one voxel".into());
    }
    let selected = selected
        .iter()
        .copied()
        .filter(|&index| index < document.voxels.len())
        .collect::<HashSet<_>>();
    if selected.is_empty() {
        return Err("the selected part no longer exists".into());
    }
    let touches_constraint = |part: voxel_physics_lab::documents::ConstraintPartDocument| match part
    {
        voxel_physics_lab::documents::ConstraintPartDocument::Voxel(index) => {
            selected.contains(&(index as usize))
        }
        voxel_physics_lab::documents::ConstraintPartDocument::Group(group_index) => document
            .groups
            .get(group_index as usize)
            .is_some_and(|group| {
                group
                    .voxels
                    .iter()
                    .any(|&index| selected.contains(&(index as usize)))
            }),
    };
    if document
        .motion_constraints
        .iter()
        .any(|constraint| touches_constraint(constraint.a) || touches_constraint(constraint.b))
    {
        return Err(
            "This part is used by a motion constraint; moving it could change its anchors.".into(),
        );
    }

    let offset = delta_cells.as_dvec3() * document.cell_size_m;
    let mut occupied = document
        .voxels
        .iter()
        .enumerate()
        .filter(|(index, _)| !selected.contains(index))
        .map(|(_, voxel)| voxel_lattice_key(voxel.position_m, document.cell_size_m))
        .collect::<HashSet<_>>();
    let mut edits = Vec::with_capacity(selected.len());
    for &index in &selected {
        let voxel = &document.voxels[index];
        let mut target_cell = voxel_lattice_key(voxel.position_m, document.cell_size_m);
        target_cell += delta_cells;
        if !occupied.insert(target_cell) {
            return Err("The selected part would overlap another REV.".into());
        }
        let mut moved = voxel.clone();
        moved.position_m += offset;
        moved.rest_position_m += offset;
        edits.push(ModelVoxelEdit::Replace {
            index,
            voxel: moved,
        });
    }
    edits.sort_unstable_by_key(|edit| match edit {
        ModelVoxelEdit::Replace { index, .. } => *index,
        _ => usize::MAX,
    });
    Ok(if edits.len() == 1 {
        edits.remove(0)
    } else {
        ModelVoxelEdit::Batch { edits }
    })
}

fn selected_part_rotate_z_edit(
    document: &VoxelModelDocument,
    selected: &[usize],
    direction: i8,
) -> Result<ModelVoxelEdit, String> {
    if selected.is_empty() || !matches!(direction, -1 | 1) {
        return Err("select a part and a valid rotation direction".into());
    }
    let selected = selected
        .iter()
        .copied()
        .filter(|&i| i < document.voxels.len())
        .collect::<HashSet<_>>();
    let touches_constraint = |part: voxel_physics_lab::documents::ConstraintPartDocument| match part
    {
        voxel_physics_lab::documents::ConstraintPartDocument::Voxel(index) => {
            selected.contains(&(index as usize))
        }
        voxel_physics_lab::documents::ConstraintPartDocument::Group(group_index) => document
            .groups
            .get(group_index as usize)
            .is_some_and(|group| {
                group
                    .voxels
                    .iter()
                    .any(|&index| selected.contains(&(index as usize)))
            }),
    };
    if document
        .motion_constraints
        .iter()
        .any(|constraint| touches_constraint(constraint.a) || touches_constraint(constraint.b))
    {
        return Err(
            "This part is used by a motion constraint; rotating it could change its anchors."
                .into(),
        );
    }
    let pivot_index = *selected
        .iter()
        .min()
        .ok_or("the selected part no longer exists")?;
    let pivot = voxel_lattice_key(
        document.voxels[pivot_index].position_m,
        document.cell_size_m,
    );
    let occupied = document
        .voxels
        .iter()
        .enumerate()
        .filter(|(i, _)| !selected.contains(i))
        .map(|(_, v)| voxel_lattice_key(v.position_m, document.cell_size_m))
        .collect::<HashSet<_>>();
    let mut destinations = HashSet::with_capacity(selected.len());
    let mut edits = Vec::with_capacity(selected.len());
    for &index in &selected {
        let voxel = &document.voxels[index];
        let cell = voxel_lattice_key(voxel.position_m, document.cell_size_m);
        let relative = cell - pivot;
        let rotated = if direction > 0 {
            pivot + IVec3::new(-relative.y, relative.x, relative.z)
        } else {
            pivot + IVec3::new(relative.y, -relative.x, relative.z)
        };
        if occupied.contains(&rotated) || !destinations.insert(rotated) {
            return Err("The selected part would overlap another REV.".into());
        }
        let offset = (rotated - cell).as_dvec3() * document.cell_size_m;
        let mut moved = voxel.clone();
        moved.position_m += offset;
        moved.rest_position_m += offset;
        edits.push(ModelVoxelEdit::Replace {
            index,
            voxel: moved,
        });
    }
    edits.sort_unstable_by_key(|edit| match edit {
        ModelVoxelEdit::Replace { index, .. } => *index,
        _ => usize::MAX,
    });
    Ok(if edits.len() == 1 {
        edits.remove(0)
    } else {
        ModelVoxelEdit::Batch { edits }
    })
}

fn duplicate_selected_part_edit(
    document: &VoxelModelDocument,
    selected: &[usize],
) -> Result<(ModelVoxelEdit, Vec<usize>), String> {
    if selected.is_empty() {
        return Err("select a connected part to duplicate".into());
    }
    if !document.groups.is_empty()
        || !document.motion_constraints.is_empty()
        || document.skeleton.is_some()
    {
        return Err(
            "Cannot duplicate a part with authored groups, constraints, or a skeleton.".into(),
        );
    }
    let selected = selected
        .iter()
        .copied()
        .filter(|&index| index < document.voxels.len())
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("the selected part no longer exists".into());
    }
    let cells = selected
        .iter()
        .map(|&index| voxel_lattice_key(document.voxels[index].position_m, document.cell_size_m))
        .collect::<Vec<_>>();
    let minimum = cells
        .iter()
        .copied()
        .reduce(IVec3::min)
        .unwrap_or(IVec3::ZERO);
    let maximum = cells
        .iter()
        .copied()
        .reduce(IVec3::max)
        .unwrap_or(IVec3::ZERO);
    let occupied = document
        .voxels
        .iter()
        .map(|voxel| voxel_lattice_key(voxel.position_m, document.cell_size_m))
        .collect::<HashSet<_>>();
    let directions = [
        IVec3::X,
        -IVec3::X,
        IVec3::Y,
        -IVec3::Y,
        IVec3::Z,
        -IVec3::Z,
    ];
    let offset_cells = directions
        .into_iter()
        .find_map(|direction| {
            let axis = if direction.x != 0 {
                0
            } else if direction.y != 0 {
                1
            } else {
                2
            };
            let span = maximum[axis] - minimum[axis] + 1;
            let offset = direction * span;
            cells
                .iter()
                .all(|&cell| !occupied.contains(&(cell + offset)))
                .then_some(offset)
        })
        .ok_or_else(|| "No free adjacent space is available for the selected part.".to_owned())?;
    let offset = offset_cells.as_dvec3() * document.cell_size_m;
    let first_index = document.voxels.len();
    let mut indices = Vec::with_capacity(selected.len());
    let mut edits = Vec::with_capacity(selected.len());
    for (copy_index, &index) in selected.iter().enumerate() {
        let mut duplicate = document.voxels[index].clone();
        duplicate.position_m += offset;
        duplicate.rest_position_m += offset;
        let target_index = first_index + copy_index;
        indices.push(target_index);
        edits.push(ModelVoxelEdit::Insert {
            index: target_index,
            voxel: duplicate,
        });
    }
    let edit = if edits.len() == 1 {
        edits.remove(0)
    } else {
        ModelVoxelEdit::Batch { edits }
    };
    Ok((edit, indices))
}

fn delete_selected_part_edit(
    document: &VoxelModelDocument,
    selected: &[usize],
) -> Result<ModelVoxelEdit, String> {
    let mut indices = selected
        .iter()
        .copied()
        .filter(|&index| index < document.voxels.len())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        return Err("select a connected part to delete".into());
    }
    if indices.len() >= document.voxels.len() {
        return Err("A voxel model must keep at least one REV.".into());
    }
    if !document.groups.is_empty()
        || !document.motion_constraints.is_empty()
        || document.skeleton.is_some()
    {
        return Err(
            "Cannot delete a part while authored groups, constraints, or a skeleton reference it."
                .into(),
        );
    }
    indices.reverse();
    let mut edits = indices
        .into_iter()
        .map(|index| ModelVoxelEdit::Remove { index })
        .collect::<Vec<_>>();
    Ok(if edits.len() == 1 {
        edits.remove(0)
    } else {
        ModelVoxelEdit::Batch { edits }
    })
}

fn commit_model_edit(
    document: &mut VoxelModelDocument,
    history: &mut ModelEditHistory,
    config: &SceneConfig,
    edit: ModelVoxelEdit,
) -> Result<Scene, String> {
    let mut candidate_document = document.clone();
    let mut candidate_history = history.clone();
    candidate_history
        .commit(&mut candidate_document, edit)
        .map_err(|error| error.to_string())?;
    let rebuilt = build_editor_preview_scene(&candidate_document, config.clone())?;
    *document = candidate_document;
    *history = candidate_history;
    Ok(rebuilt)
}

fn sync_model_history_ui(ui: &mut Ui, history: &ModelEditHistory) {
    ui.set_model_history_state(history.can_undo(), history.can_redo());
}

fn set_game_camera(character: &WorldCharacter, camera: &mut Camera, physics_active: bool) {
    camera.yaw = if physics_active { -0.04 } else { 0.0 };
    // In the 20 m contact scene the beam is 35 cm above ground, directly ahead
    // of the player. This angle puts the center ray on its side face at spawn.
    camera.pitch = if physics_active { 0.92 } else { 0.0 };
    camera.set_first_person(character.eye_position());
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    run_with_app_kind(false)
}

pub fn run_editor() -> Result<(), Box<dyn std::error::Error>> {
    run_with_app_kind(true)
}

fn impact_scene_creation_args(
    args: &[String],
) -> Result<Option<(PathBuf, voxel_physics_lab::scene_presets::ImpactSceneOptions)>, String> {
    fn value<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>, String> {
        let mut matches = args.iter().enumerate().filter(|(_, arg)| arg.as_str() == flag);
        let Some((index, _)) = matches.next() else {
            return Ok(None);
        };
        if matches.next().is_some() {
            return Err(format!("{flag} may appear only once"));
        }
        args.get(index + 1)
            .filter(|arg| !arg.starts_with("--"))
            .map(|arg| Some(arg.as_str()))
            .ok_or_else(|| format!("{flag} requires a value"))
    }
    let Some(path) = value(args, "--create-impact-scene")? else { return Ok(None); };
    let mut options = voxel_physics_lab::scene_presets::ImpactSceneOptions::default();
    if let Some(v) = value(args, "--side")? {
        options.side = v.parse().map_err(|_| "invalid --side")?;
    }
    if let Some(v) = value(args, "--impact-gap-m")? {
        options.gap_m = v.parse().map_err(|_| "invalid --impact-gap-m")?;
    }
    if let Some(v) = value(args, "--impact-speed-m-s")? {
        options.approach_speed_m_s = v.parse().map_err(|_| "invalid --impact-speed-m-s")?;
    }
    if let Some(v) = value(args, "--impact-dt-ms")? {
        options.fixed_dt_s = v.parse::<f32>().map_err(|_| "invalid --impact-dt-ms")? * 0.001;
    }
    if let Some(v) = value(args, "--impact-substeps")? {
        options.substeps = v.parse().map_err(|_| "invalid --impact-substeps")?;
    }
    options.validate()?;
    Ok(Some((PathBuf::from(path), options)))
}

#[cfg(test)]
mod impact_cli_tests {
    use super::*;
    fn args(text: &str) -> Vec<String> { text.split_whitespace().map(str::to_owned).collect() }
    #[test]
    fn impact_creation_parameters_have_explicit_units() {
        let (_, options) = impact_scene_creation_args(&args("editor --create-impact-scene scene.vxscene --side 55 --impact-gap-m 0 --impact-speed-m-s 0.01 --impact-dt-ms 10 --impact-substeps 4")).unwrap().unwrap();
        assert_eq!(options.side.pow(3)*2,332_750);
        assert_eq!(options.gap_m,0.0);
        assert_eq!(options.approach_speed_m_s,0.01);
        assert!((options.fixed_dt_s-0.01).abs()<1e-8);
        assert_eq!(options.substeps,4);
    }
    #[test]
    fn malformed_impact_arguments_fail_before_authoring() {
        for text in ["--create-impact-scene", "--create-impact-scene scene.vxscene --side", "--create-impact-scene a.vxscene --side 2 --side 3", "--create-impact-scene a.vxscene --impact-speed-m-s NaN", "--create-impact-scene a.vxscene --impact-substeps 0"] {
            assert!(impact_scene_creation_args(&args(text)).is_err(),"{text}");
        }
        assert!(impact_scene_creation_args(&args("editor --scene existing.vxscene")).unwrap().is_none());
    }
}

fn run_with_app_kind(standalone_editor_app: bool) -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--disable-auto-aggregation") {
        std::env::set_var("VOXEL_DISABLE_AUTO_AGGREGATION", "1");
    }
    if args.iter().any(|arg| arg == "--profile-steps") {
        std::env::set_var("VOXEL_PROFILE_STEP", "1");
    }
    if args.iter().any(|arg| arg == "--cuda-contact-operator") {
        std::env::set_var("VOXEL_CUDA_CONTACT_OPERATOR", "1");
    }
    if args.iter().any(|arg| arg == "--cuda-contact-ic0") {
        std::env::set_var("VOXEL_CUDA_CONTACT_OPERATOR", "1");
        std::env::set_var("VOXEL_CUDA_CONTACT_IC0", "1");
    }
    let creation = impact_scene_creation_args(&args)?;
    let mut created_scene_path = creation.as_ref().map(|(path, _)| path.clone());
    if let Some((path, options)) = creation.as_ref() {
        voxel_physics_lab::scene_presets::create_impact_scene_with_options(path, options)?;
    }
    if let Some(index) = args.iter().position(|arg| arg == "--create-animated-scene") {
        if creation.is_some() { return Err("choose one scene creation command".into()); }
        let output = args.get(index+1).filter(|s| !s.starts_with("--"))
            .ok_or("--create-animated-scene requires an output .vxscene path")?;
        let model = args.windows(2).find(|pair| pair[0] == "--animated-model")
            .map(|pair| pair[1].as_str()).ok_or("--animated-model requires a source .vxm path")?;
        voxel_physics_lab::scene_presets::create_animated_model_scene(
            Path::new(output), Path::new(model), 96,
        )?;
        created_scene_path = Some(std::path::PathBuf::from(output));
    }
    let requested_model_path = args
        .windows(2)
        .find(|pair| pair[0] == "--model")
        .map(|pair| pair[1].as_str());
    if args.last().is_some_and(|arg| arg == "--model") {
        return Err("--model requires a .vxm path".into());
    }
    let lab_only_args = [
        "--two-cubes",
        "--friction-road",
        "--pause-on-fracture",
        "--headless",
        "--physics-world20",
        "--adaptive-physics",
        "--seed",
        "--time-scale",
        "--tool",
    ];
    if standalone_editor_app && args.iter().any(|arg| lab_only_args.contains(&arg.as_str())) {
        return Err(
            "Physics experiment arguments are available from voxel-physics-lab, not voxel-editor."
                .into(),
        );
    }
    let two_cube_demo = !standalone_editor_app && args.iter().any(|arg| arg == "--two-cubes");
    let friction_road_demo =
        !standalone_editor_app && args.iter().any(|arg| arg == "--friction-road");
    let pause_on_fracture =
        !standalone_editor_app && args.iter().any(|arg| arg == "--pause-on-fracture");
    let adaptive_physics =
        !standalone_editor_app && args.iter().any(|arg| arg == "--adaptive-physics");
    let requested_scene_path = args
        .windows(2)
        .find(|p| p[0] == "--scene")
        .map(|pair| pair[1].as_str())
        .or_else(|| created_scene_path.as_ref().and_then(|path| path.to_str()));
    let native_scene_document = requested_scene_path
        .filter(|path| {
            Path::new(path)
                .extension()
                .is_some_and(|extension| extension == "vxscene")
        })
        .map(|path| load_document::<SceneDocument>(Path::new(path)))
        .transpose()?;
    let initial_native_scene_path = native_scene_document
        .as_ref()
        .and(requested_scene_path)
        .map(PathBuf::from);
    let initial_seed = args
        .windows(2)
        .find(|pair| pair[0] == "--seed")
        .map(|pair| pair[1].as_str());
    let startup_empty_scene = standalone_editor_app
        || (!two_cube_demo
            && !friction_road_demo
            && requested_scene_path.is_none()
            && initial_seed.is_none()
            && selected_shape(&args).is_none()
            && !args.iter().any(|arg| arg == "--headless"));
    let path = requested_scene_path.unwrap_or("scenes/mounted_axe.ron");
    let mut config = if let Some(document) = native_scene_document.as_ref() {
        document.config.clone()
    } else if two_cube_demo || friction_road_demo {
        let mut config = SceneConfig::default();
        if friction_road_demo {
            config.dimensions = [240, 9, 8];
            config.ground_dimensions = [240, 9, 1];
        } else {
            config.dimensions = [18, 2, 2];
            config.ground_dimensions = [24, 12, 1];
        }
        config
    } else if startup_empty_scene {
        SceneConfig::default()
    } else {
        SceneConfig::load(path)?
    };
    let initial_model_document = if standalone_editor_app && native_scene_document.is_none() {
        if let Some(path) = requested_model_path {
            let document: VoxelModelDocument = load_document(Path::new(path))?;
            document.validate()?;
            Some(document)
        } else {
            let oak = voxel_physics_lab::material::Material::component_id("oak")
                .or_else(|| voxel_physics_lab::material::Material::component_id("wood"))
                .ok_or("oak material is unavailable")?;
            let initial =
                Scene::from_voxels(config.clone(), vec![VoxelSpawn::at_cell([0, 0, 0], oak)])?;
            Some(VoxelModelDocument::from_scene("Untitled Model", &initial))
        }
    } else {
        None
    };
    let initial_scene_document = if native_scene_document.is_some() {
        native_scene_document
    } else if startup_empty_scene && !standalone_editor_app {
        Some(SceneDocument::new("Untitled Scene", config.clone())?)
    } else {
        None
    };
    let mut scene = if let Some(document) = initial_model_document.as_ref() {
        build_editor_preview_scene(document, config.clone())?
    } else if let Some(document) = initial_scene_document.as_ref() {
        document.build_scene(
            initial_native_scene_path
                .as_deref()
                .unwrap_or_else(|| Path::new("Untitled.vxscene")),
        )?
    } else if friction_road_demo {
        friction_heat_road_scene(config.clone())?
    } else if two_cube_demo {
        two_cube_collision_scene(config.clone())?
    } else {
        let mut scene = Scene::new(config.clone());
        fix_wood_base(&mut scene);
        scene
    };
    let mut ui = Ui::new(
        config.initial_speed,
        Vec3::from_array(config.tool_start),
        Vec3::from_array(config.tool_direction),
    );
    let mut editor_settings = editor_settings::load();
    ui.settings_bindings = editor_settings.bindings.clone();
    ui.settings_mcp_enabled = editor_settings.mcp_enabled;
    ui.standalone_editor_app = standalone_editor_app && initial_scene_document.is_none();
    if let Some(document) = initial_scene_document.as_ref() {
        ui.world_active = true;
        ui.editor_mode = true;
        ui.paused = true;
        ui.show_bonds = false;
        ui.editor_tool = EditorTool::Select;
        ui.set_active_scene_document(Some(document.name.clone()), scene_model_names(document));
        ui.set_status_message(if document.models.is_empty() {
            "Empty scene — add a voxel model or import an OBJ/GLB mesh.".to_owned()
        } else {
            format!("Opened scene: {}", document.name)
        });
    } else if let Some(document) = initial_model_document.as_ref() {
        ui.world_active = true;
        ui.editor_mode = true;
        ui.paused = true;
        ui.show_bonds = false;
        ui.editor_tool = EditorTool::AddVoxel;
        update_model_ui(&mut ui, document, Some(0));
        ui.set_status_message(
            "Add REV: click one cell or drag a connected row; Block: drag a rectangular patch.",
        );
        if requested_model_path.is_some() {
            ui.editor_tool = EditorTool::Select;
            ui.set_model_selection(None, None, None);
            ui.set_status_message(format!("Opened model: {}", document.name));
            if args.iter().any(|arg| arg == "--play-animation") {
                ui.toggle_animation_playback();
            }
        }
    }
    if two_cube_demo || friction_road_demo {
        ui.simulation_time_scale = 1.0;
    }
    if args.iter().any(|arg| arg == "--run-scene") {
        if initial_native_scene_path.is_none() {
            return Err("--run-scene requires an authored .vxscene".into());
        }
        ui.paused = false;
        ui.simulation_time_scale = 1.0;
    }
    if let Some(pair) = args.windows(2).find(|pair| pair[0] == "--time-scale") {
        let scale = pair[1].parse::<f32>()?;
        if !scale.is_finite() || scale < 0.0 {
            return Err("simulation time scale must be finite and nonnegative".into());
        }
        ui.simulation_time_scale = scale;
    }
    if let Some(index) = args.iter().position(|s| s == "--headless") {
        let steps: usize = args
            .get(index + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        let shape = selected_shape(&args).unwrap_or(Shape::Wedge);
        let mut tool = if two_cube_demo || friction_road_demo || initial_native_scene_path.is_some() {
            None
        } else {
            Some(launch(shape, &scene, &ui))
        };
        let initial_momentum = total_momentum(&scene);
        let moving_mass: f64 = scene
            .voxels
            .iter()
            .filter(|voxel| !voxel.kinematic)
            .map(|voxel| f64::from(voxel.mass))
            .sum();
        let initial_kinetic_j = kinetic_energy(&scene);
        let initial_spin =
            angular_momentum(&scene, 0..8).length() + angular_momentum(&scene, 8..16).length();
        let start = Instant::now();
        for _ in 0..steps {
            solver::step(&mut scene, &mut tool);
        }
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "voxels={} steps={} elapsed_s={:.3} steps_per_s={:.2} broken={} clusters={} linear_residual={:.2e}",
            scene.voxels.len(),
            steps,
            seconds,
            steps as f64 / seconds,
            scene.broken,
            scene.cluster_count,
            scene.solver_residual
        );
        if two_cube_demo {
            let gravity_impulse = DVec3::new(
                0.0,
                0.0,
                moving_mass
                    * f64::from(config.gravity[2])
                    * f64::from(config.fixed_dt)
                    * steps as f64,
            );
            println!(
                "cube_centers_m=({:.4},{:.4},{:.4})/({:.4},{:.4},{:.4}) cube_velocity_m_s=({:.4},{:.4},{:.4})/({:.4},{:.4},{:.4}) pair_mu_oak_concrete={:.3} momentum_error_kg_m_s={:.3e} spin_L_kg_m2_s={:.3e}->{:.3e} kinetic_J={:.6}->{:.6} restitution_heat_J={:.6} friction_heat_J={:.6}",
                cube_center(&scene, 0..8).x,
                cube_center(&scene, 0..8).y,
                cube_center(&scene, 0..8).z,
                cube_center(&scene, 8..16).x,
                cube_center(&scene, 8..16).y,
                cube_center(&scene, 8..16).z,
                cube_velocity(&scene, 0..8).x,
                cube_velocity(&scene, 0..8).y,
                cube_velocity(&scene, 0..8).z,
                cube_velocity(&scene, 8..16).x,
                cube_velocity(&scene, 8..16).y,
                cube_velocity(&scene, 8..16).z,
                scene.config.fracture.friction
                    * (scene.properties[0].surface_friction_coefficient
                        + scene.properties[16].surface_friction_coefficient),
                (total_momentum(&scene) - initial_momentum - gravity_impulse).length(),
                initial_spin,
                angular_momentum(&scene, 0..8).length() + angular_momentum(&scene, 8..16).length(),
                initial_kinetic_j,
                kinetic_energy(&scene),
                scene.energy.contact_restitution_heat,
                scene.energy.contact_friction_heat,
            );
        }
        if scene.positions.iter().any(|p| !p.is_finite()) {
            return Err("Non-finite solver output".into());
        }
        return Ok(());
    }
    // API checked with cargo check; opening a real window still needs display validation.
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(if standalone_editor_app && initial_scene_document.is_none() {
                "Voxel Model Editor | Untitled Model"
            } else if friction_road_demo {
                "Voxel lab | oak cube rolling on concrete road"
            } else if two_cube_demo {
                "Voxel lab | two 2x2x2 oak cubes collide"
            } else {
                "Voxel Editor | Untitled Scene"
            })
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
            .with_maximized(standalone_editor_app)
            .with_fullscreen(args.iter().any(|arg| arg == "--fullscreen").then_some(Fullscreen::Borderless(None)))
            .build(&event_loop)?,
    );
    let mut renderer = pollster::block_on(Renderer::new(window.clone()))?;
    let mut camera = Camera::new();
    let height = config.dimensions[2] as f32 * scene::CELL;
    camera.target = if startup_empty_scene {
        Vec3::ZERO
    } else if friction_road_demo {
        Vec3::new(-0.53, 0.0, 0.02)
    } else if two_cube_demo {
        Vec3::new(0.0, 0.0, 0.62)
    } else {
        Vec3::Z * (height * 0.5)
    };
    camera.radius = if standalone_editor_app {
        0.04
    } else if startup_empty_scene {
        5.0
    } else if friction_road_demo {
        0.38
    } else if two_cube_demo {
        0.90
    } else {
        (height * 1.5).max(
            config.ground_dimensions[0].max(config.ground_dimensions[1]) as f32 * scene::CELL * 1.6,
        )
    };
    if two_cube_demo || friction_road_demo {
        camera.yaw = 0.55;
        camera.pitch = 0.35;
    }
    if requested_model_path.is_some() || initial_scene_document.is_some() {
        frame_camera_on_scene(&scene, &mut camera);
    }
    let initial_camera_radius = camera.radius;
    let initial_shape = selected_shape(&args);
    if let Some(shape) = initial_shape {
        ui.selected = shape;
    }
    let mut tool = if two_cube_demo || friction_road_demo {
        None
    } else {
        initial_shape.map(|shape| launch(shape, &scene, &ui))
    };
    // Reproduce vegetation-loading reports without waiting decades in the UI.
    let initial_age_years = args
        .windows(2)
        .find(|pair| pair[0] == "--world-age-years")
        .map(|pair| pair[1].parse::<f32>())
        .transpose()?
        .unwrap_or(0.0);
    if !initial_age_years.is_finite() || initial_age_years < 0.0 {
        return Err("initial world age must be finite and nonnegative".into());
    }
    let mut generated_world: Option<GeneratedWorld> = initial_seed.map(|seed| {
        ui.seed_text = seed.to_owned();
        ui.world_active = true;
        ui.paused = false;
        eprintln!("generating deterministic terrain and vegetation for seed {seed:?}");
        let mut world = GeneratedWorld::from_seed(seed);
        if initial_age_years > 0.0 {
            world.advance_growth(initial_age_years * (365.25 * 24.0 * 60.0 * 60.0));
        }
        world
    });
    let mut growth_scheduler = GrowthScheduler::default();
    let mut world_character = generated_world.as_ref().map(WorldCharacter::spawn);
    if args.iter().any(|arg| arg == "--near-tree") {
        if let (Some(world), Some(character)) = (&generated_world, &mut world_character) {
            if let Some(wood) = world
                .voxels
                .iter()
                .filter(|voxel| {
                    voxel.content == voxel_physics_lab::world::GeneratedCellContent::PlantWood
                })
                .min_by(|a, b| {
                    a.center
                        .length_squared()
                        .total_cmp(&b.center.length_squared())
                })
            {
                let x = wood.center.x + 2.0;
                let y = wood.center.y;
                character.position = Vec3::new(x, y, world.terrain_height_at(x, y));
            }
        }
    }
    if world_character.is_some() {
        tool = None;
        ui.editor_mode = true;
        let radius = if args.iter().any(|arg| arg == "--physics-world20") {
            2.5
        } else {
            12.0
        };
        set_editor_camera(world_character.as_ref().unwrap(), &mut camera, radius);
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
    }
    let mut world_physics = None;
    let mut active_scene_document: Option<SceneDocument> = initial_scene_document;
    let mut active_scene_document_path: Option<PathBuf> = initial_native_scene_path;
    let mut active_model_document: Option<VoxelModelDocument> = initial_model_document;
    let mut active_model_document_path: Option<PathBuf> = requested_model_path.map(PathBuf::from);
    let mut model_edit_history = ModelEditHistory::default();
    let mut mesh_import_source_path: Option<PathBuf> = None;
    let mut mesh_import_preview: Option<VoxelizationPreview> = None;
    let mut mesh_import_preview_scene: Option<Scene> = None;
    let mut mesh_import_camera_backup: Option<CameraSnapshot> = None;
    let mut mesh_import_previous_paused: Option<bool> = None;
    let mut mesh_import_job = None;
    let mut model_export_job: Option<mpsc::Receiver<Result<usize, String>>> = None;
    let mut auto_weight_job: Option<mpsc::Receiver<AutoWeightResult>> = None;
    let mut model_revision = 0u64;
    let mut skinning_weights_revision = 0u64;
    if args.iter().any(|arg| arg == "--physics-world20") {
        let world = generated_world
            .as_mut()
            .ok_or("--physics-world20 requires --seed")?;
        world_physics = Some(start_world_physics(
            world,
            world_character.as_mut().unwrap(),
            &mut camera,
            &config,
            adaptive_physics,
        )?);
        if ui.editor_mode {
            set_editor_camera(world_character.as_ref().unwrap(), &mut camera, 2.5);
        }
    }
    let mut pointer = (0.0f32, 0.0f32);
    let mut hovered_face: Option<HoveredFace> = None;
    let mut sculpt_drag: Option<SculptDrag> = None;
    let mut selection_drag: Option<BoxSelectionDrag> = None;
    let mut part_move_drag: Option<MovePartDrag> = None;
    let mut scene_move_drag: Option<SceneInstanceMoveDrag> = None;
    let mut bone_move_drag: Option<BoneMoveDrag> = None;
    let mut orbit =
        (world_character.is_some() && !ui.editor_mode) || two_cube_demo || friction_road_demo;
    let mut ignore_mouse_motion_until = Instant::now();
    let mut dragging = false;
    let mut grabbed_cell: Option<GrabbedCell> = None;
    let mut keys_down = HashSet::new();
    let mut jump_queued = false;
    let mut last = Instant::now();
    let profile_world_geometry = std::env::var_os("VOXEL_PROFILE_WORLD_GEOMETRY").is_some();
    let mut accumulator = 0.0f64;
    let mut mode = AppMode::Interactive;
    let mut playback_frames = Vec::new();
    let mut playback_time_s = 0.0;
    let mut live_simulated_s = 0.0;
    let mut live_wall_s = 0.0;
    let mut next_record_frame_s = PLAYBACK_FRAME_INTERVAL_SECONDS;
    let mut sim_ms = 0.0;
    let mut fracture_pause_triggered = false;
    let mut sculpt_preview_cells = Vec::new();
    let mut sculpt_preview_bounds = None;
    let mut pending_scene_action: Option<UiAction> = None;
    let mut scene_load_job: Option<
        mpsc::Receiver<Result<(SceneDocument, Scene, Option<PathBuf>), String>>,
    > = None;
    let mut authored_physics: Option<PhysicsWorker> = None;
    let mut run_panel = RunPanel::new();
    // Scene whose chemistry settings the UI shows: `None` until synced, then
    // `Some(None)` for an unsaved scene and `Some(Some(path))` for a saved one.
    let mut chemistry_synced_for: Option<Option<PathBuf>> = None;
    // Creatures of the open scene (from `<scene>.creatures.json`) and what they were built for.
    let mut scene_creatures: Vec<NamedCreature> = Vec::new();
    let mut scene_creatures_key: Option<(PathBuf, usize, usize)> = None;
    let mut authored_render_source: Option<AuthoredPhysicsRenderSource> = None;
    let mut reported_physics_failure: Option<u64> = None;
    let mut fps = 0.0;
    let mut frames = 0;
    let mut fps_time = Instant::now();
    let runtime_project_root = editor_settings::settings_path().parent()
        .unwrap_or_else(|| Path::new(".")).to_path_buf();
    let mut runtime_metrics_server = if editor_settings.mcp_enabled {
        match voxel_physics_lab::runtime_metrics::start_server(&runtime_project_root) {
            Ok(server) => Some(server),
            Err(error) => { eprintln!("Live MCP metrics unavailable: {error}"); None }
        }
    } else { None };
    let mut last_runtime_publish = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let mut last_authoring_metrics_key = None;
    let mut previous_runtime_simulated = 0.0;
    let mut previous_runtime_generation = None;
    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } if orbit && (camera.is_freecam() || camera.is_first_person()) => {
                if Instant::now() >= ignore_mouse_motion_until {
                    camera.rotate_freecam(delta.0, delta.1)
                }
            }
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
                WindowEvent::Focused(false) => {
                    orbit = false;
                    keys_down.clear();
                    sculpt_drag = None;
                    sculpt_preview_cells.clear();
                    sculpt_preview_bounds = None;
                    selection_drag = None;
                    part_move_drag = None;
                    scene_move_drag = None;
                    dragging = false;
                    if let Some(drag) = bone_move_drag.take() {
                        if let Some(document) = active_model_document.as_mut() {
                            if apply_bone_drag(document, &drag, Vec3::ZERO).is_ok() {
                                model_revision = model_revision.wrapping_add(1);
                            }
                        }
                    }
                    grabbed_cell = None;
                    if let Some(physics) = world_physics.as_ref() {
                        physics.set_grab_target(None);
                    }
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
                WindowEvent::Focused(true) if camera.is_first_person() && !ui.menu_open => {
                    orbit = true;
                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                    window.set_cursor_visible(false);
                    ignore_mouse_motion_until = Instant::now() + Duration::from_millis(150);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let next = (position.x as f32, position.y as f32);
                    if orbit && !camera.is_freecam() && !camera.is_first_person() {
                        camera.yaw -= (next.0 - pointer.0) * ui.look_sensitivity;
                        camera.pitch = (camera.pitch + (next.1 - pointer.1) * ui.look_sensitivity)
                            .clamp(-1.4, 1.4);
                    }
                    if dragging && next.1 >= 185.0 {
                        ui.pointer(next.0, next.1);
                    }
                    pointer = next;
                    if sculpt_drag.is_none() {
                        hovered_face = hovered_face_at_pointer(&ui, &scene, &camera, next);
                    }
                    if let (Some(drag), Some(document)) =
                        (sculpt_drag.as_mut(), active_model_document.as_ref())
                    {
                        if let Some(cell) = sculpt_drag_cell_at_pointer(
                            &ui,
                            &camera,
                            next,
                            drag,
                            document.cell_size_m,
                        ) {
                            if drag.end_cell != cell {
                                drag.end_cell = cell;
                                match sculpt_preview_geometry(document, &scene, drag) {
                                    Ok((cells, bounds)) => {
                                        sculpt_preview_cells = cells;
                                        sculpt_preview_bounds = bounds;
                                        ui.set_status_message(if bounds.is_some() {
                                            match drag.tool {
                                                EditorTool::AddVoxel => "Cyan REVs preview the row. Drag to extend; release to build.",
                                                EditorTool::AddBox => "Cyan REVs preview the block. Drag to size; release to build.",
                                                _ => "",
                                            }
                                        } else {
                                            "No new REVs in this preview."
                                        });
                                    }
                                    Err(message) => {
                                        sculpt_preview_cells.clear();
                                        sculpt_preview_bounds = None;
                                        ui.set_status_message(message);
                                    }
                                }
                            }
                        }
                    }
                    if let (Some(drag), Some(document), Some(face)) = (
                        selection_drag.as_mut(),
                        active_model_document.as_ref(),
                        hovered_face,
                    ) {
                        if let Some(voxel) = document.voxels.get(face.voxel_index) {
                            drag.end_index = face.voxel_index;
                            drag.end_cell =
                                voxel_lattice_key(voxel.position_m, document.cell_size_m);
                            drag.dragged |= drag.end_cell != drag.start_cell;
                        }
                    }
                    if let Some(drag) = part_move_drag.as_mut() {
                        if let (Some(viewport), Some(document)) =
                            (ui.scene_viewport(), active_model_document.as_ref())
                        {
                            if let Some(delta_cells) = drag_delta_cells(
                                &camera,
                                viewport,
                                next,
                                drag.plane_origin,
                                drag.plane_normal,
                                drag.start_point,
                                document.cell_size_m,
                            ) {
                                if delta_cells != drag.delta_cells {
                                    drag.delta_cells = delta_cells;
                                    let offset = delta_cells.as_dvec3()
                                        .as_vec3()
                                        * document.cell_size_m as f32;
                                    sculpt_preview_cells = drag
                                        .selected
                                        .iter()
                                        .take(MAX_DRAW_PREVIEW_CELLS)
                                        .filter_map(|&index| document.voxels.get(index))
                                        .map(|voxel| voxel.position_m.as_vec3() + offset)
                                        .collect();
                                    sculpt_preview_bounds = None;
                                    ui.set_status_message(format!(
                                        "Drag move · Δ{} {} {} cm · release to apply.",
                                        delta_cells.x,
                                        delta_cells.y,
                                        delta_cells.z
                                    ));
                                }
                            }
                        }
                    }
                    if let Some(drag) = scene_move_drag.as_mut() {
                        if let Some(viewport) = ui.scene_viewport() {
                            if let Some(delta_cells) = drag_delta_cells(
                                &camera,
                                viewport,
                                next,
                                drag.plane_origin,
                                drag.plane_normal,
                                drag.start_point,
                                voxel_physics_lab::coordinates::CELL_M,
                            ) {
                                drag.delta_cells = delta_cells;
                                ui.set_status_message(format!(
                                    "Drag object · Δ{} {} {} cm · release to apply.",
                                    delta_cells.x, delta_cells.y, delta_cells.z
                                ));
                            }
                        }
                    }
                    if let Some(drag) = bone_move_drag.as_mut() {
                        if let Some(viewport) = ui.scene_viewport() {
                            if let Some((origin, direction)) = editor_ray(&camera, viewport, next) {
                                if let Some(point) = ray_plane_point(
                                    origin,
                                    direction,
                                    drag.plane_origin,
                                    drag.plane_normal,
                                ) {
                                    let delta = point - drag.start_point;
                                    if delta.distance_squared(drag.last_delta) > 1e-10 {
                                        if let Some(document) = active_model_document.as_mut() {
                                            match apply_bone_drag(document, drag, delta) {
                                                Ok(()) => {
                                                    drag.last_delta = delta;
                                                    model_revision = model_revision.wrapping_add(1);
                                                    ui.set_status_message(format!(
                                                        "Moved {} · release to keep the pose.",
                                                        ui.rig_bone_names
                                                            .get(drag.bone_index)
                                                            .map_or("bone", String::as_str)
                                                    ));
                                                }
                                                Err(error) => {
                                                    ui.set_status_message(format!("Bone drag failed: {error}"));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Right,
                    state,
                    ..
                } => {
                    camera.look_sensitivity = ui.look_sensitivity;
                    if camera.is_first_person() && !ui.editor_mode && !ui.menu_open {
                        orbit = true;
                    } else if camera.is_freecam() && !ui.menu_open {
                        orbit = state == ElementState::Pressed;
                        if orbit {
                            let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                            window.set_cursor_visible(false);
                        } else {
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    } else if !camera.is_freecam() {
                        orbit = state == ElementState::Pressed;
                    }
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    if state == ElementState::Released {
                        ui.end_timeline_drag();
                    }
                    dragging = state == ElementState::Pressed
                        && (ui.menu_open
                            || ui.mesh_import_open
                            || (!ui.world_active && pointer.0 < 295.0))
                        && pointer.1 >= 185.0;
                    if camera.is_first_person() && !ui.menu_open {
                        match state {
                            ElementState::Pressed => {
                                grabbed_cell = world_physics.as_ref().and_then(|physics| {
                                    aimed_physics_cell(&physics.frame, &camera)
                                });
                            }
                            ElementState::Released => grabbed_cell = None,
                        }
                        if grabbed_cell.is_none() || state == ElementState::Released {
                            if let Some(physics) = world_physics.as_ref() {
                                physics.set_grab_target(None);
                            }
                        }
                    }
                    let viewport_edit = state == ElementState::Pressed
                        && ui.world_active
                        && ui.editor_mode
                        && !ui.menu_open
                        && !ui.mesh_import_open
                        && !ui.help_open
                        && !ui.scene_input_active()
                        && active_model_document.is_some()
                        && ui.scene_viewport().is_some_and(|[x, y, width, height]| {
                            pointer.0 >= x as f32
                                && pointer.1 >= y as f32
                                && pointer.0 < (x + width) as f32
                                && pointer.1 < (y + height) as f32
                        });
                    let viewport_scene_select = state == ElementState::Pressed
                        && ui.world_active
                        && ui.editor_mode
                        && !ui.menu_open
                        && !ui.mesh_import_open
                        && !ui.help_open
                        && !ui.scene_input_active()
                        && active_scene_document.is_some()
                        && ui.scene_viewport().is_some_and(|[x, y, width, height]| {
                            pointer.0 >= x as f32
                                && pointer.1 >= y as f32
                                && pointer.0 < (x + width) as f32
                                && pointer.1 < (y + height) as f32
                        });
                    if viewport_scene_select && authored_physics.is_some() {
                        ui.set_status_message("Press R to restore the authored scene before editing.");
                    } else if viewport_scene_select && ui.scene_selection_mode == SceneSelectionMode::Voxel {
                        let path = active_scene_document_path.as_deref().unwrap_or_else(|| Path::new("Untitled.vxscene"));
                        if let (Some(document), Some(face)) = (active_scene_document.as_ref(), hovered_face_at_pointer(&ui, &scene, &camera, pointer)) {
                            let parent = scene_instance_for_voxel(document, path, face.voxel_index);
                            let add = keys_down.contains(&KeyCode::ShiftLeft) || keys_down.contains(&KeyCode::ShiftRight);
                            let mut selected = if add && parent == ui.selected_scene_model { ui.selected_voxel_indices.clone() } else { Vec::new() };
                            if let Some(position) = selected.iter().position(|&index| index == face.voxel_index) { selected.remove(position); } else { selected.push(face.voxel_index); }
                            sync_scene_instance_selection(&mut ui, document, path, parent);
                            ui.set_scene_model_selection_bounds(None);
                            select_scene_voxels(&mut ui, &scene, selected);
                            ui.set_status_message(format!("Selected {} voxels. Shift-click adds or removes a voxel; material actions affect this selection.", ui.selected_voxel_indices.len()));
                        } else { ui.set_model_selection(None, None, None); }
                    } else if viewport_scene_select {
                        let scene_path = active_scene_document_path
                            .as_deref()
                            .unwrap_or_else(|| Path::new("Untitled.vxscene"));
                        scene_move_drag = None;
                        let selected = ui.scene_viewport().and_then(|viewport| {
                            let (origin, direction) = editor_ray(&camera, viewport, pointer)?;
                            active_scene_document
                                .as_ref()?
                                .models
                                .iter()
                                .enumerate()
                                .filter_map(|(index, _)| {
                                    let bounds = authored_instance_bounds(
                                        active_scene_document.as_ref()?,
                                        scene_path,
                                        index,
                                    )
                                    .ok()?;
                                    ray_aabb_distance(origin, direction, bounds.min, bounds.max)
                                        .map(|distance| (index, distance))
                                })
                                .min_by(|left, right| left.1.total_cmp(&right.1))
                                .map(|(index, _)| index)
                        });
                        if selected.is_some() && selected == ui.selected_scene_model {
                            if !ui.paused || live_simulated_s > 1e-9 {
                                ui.set_status_message("Pause and reset before moving a scene object.");
                            } else if let (Some(viewport), Some(index)) =
                                (ui.scene_viewport(), selected)
                            {
                                let bounds = active_scene_document
                                    .as_ref()
                                    .and_then(|document| {
                                        authored_instance_bounds(document, scene_path, index).ok()
                                    });
                                if let (Some(bounds), Some((origin, direction))) = (
                                    bounds,
                                    editor_ray(&camera, viewport, pointer),
                                ) {
                                    if let Some(start_point) = ray_plane_point(
                                        origin,
                                        direction,
                                        bounds.center,
                                        camera.forward(),
                                    ) {
                                        scene_move_drag = Some(SceneInstanceMoveDrag {
                                            instance_index: index,
                                            plane_origin: bounds.center,
                                            plane_normal: camera.forward(),
                                            start_point,
                                            bounds: (bounds.min, bounds.max),
                                            delta_cells: IVec3::ZERO,
                                        });
                                        ui.set_status_message("Drag the selected model to move it; release to apply.");
                                    }
                                }
                            }
                        } else if let Some(document) = active_scene_document.as_ref() {
                            ui.set_model_selection(None, None, None);
                            sync_scene_instance_selection(&mut ui, document, scene_path, selected);
                            if selected.is_some() {
                                ui.set_status_message("Selected model. Drag it to move; use the Inspector for precise transforms.");
                            }
                        }
                        if selected.is_none() {
                            ui.selected_scene_model = None;
                            ui.set_selected_scene_model_info(None);
                            ui.set_scene_model_selection_bounds(None);
                            ui.set_status_message("No model instance under the pointer.");
                        }
                    }
                    if viewport_edit
                        && matches!(ui.editor_tool, EditorTool::AddVoxel | EditorTool::AddBox)
                    {
                        if !ui.paused || live_simulated_s > 1e-9 {
                            ui.set_status_message(
                                "Pause and reset the model before changing voxel geometry.",
                            );
                        } else if let Some(viewport) = ui.scene_viewport() {
                            if let Some((origin, direction)) = editor_ray(&camera, viewport, pointer)
                            {
                                if let Some(face) = hovered_model_face(&scene, origin, direction) {
                                    if let Some(document) = active_model_document.as_ref() {
                                        let normal = face_lattice_normal(face.normal);
                                        let source_cell = voxel_lattice_key(
                                            document.voxels[face.voxel_index].position_m,
                                            document.cell_size_m,
                                        );
                                        let start_cell = source_cell + normal;
                                        let drag = SculptDrag {
                                            tool: ui.editor_tool,
                                            start: face,
                                            start_ray: (origin, direction),
                                            normal,
                                            start_cell,
                                            end_cell: start_cell,
                                            brush_size_cells: ui.brush_size_cells,
                                            occupied_cells: document
                                                .voxels
                                                .iter()
                                                .map(|voxel| {
                                                    voxel_lattice_key(
                                                        voxel.position_m,
                                                        document.cell_size_m,
                                                    )
                                                })
                                                .collect(),
                                        };
                                        match sculpt_preview_geometry(document, &scene, &drag) {
                                            Ok((cells, bounds)) => {
                                                sculpt_preview_cells = cells;
                                                sculpt_preview_bounds = bounds;
                                            }
                                            Err(message) => {
                                                sculpt_preview_cells.clear();
                                                sculpt_preview_bounds = None;
                                                ui.set_status_message(message);
                                            }
                                        }
                                        sculpt_drag = Some(drag);
                                        ui.set_status_message(if sculpt_preview_bounds.is_some() {
                                            match ui.editor_tool {
                                                EditorTool::AddVoxel => "Cyan REVs preview the row. Drag to extend; release to build.",
                                                EditorTool::AddBox => "Cyan REVs preview the block. Drag to size; release to build.",
                                                _ => "",
                                            }
                                        } else {
                                            "No new REVs in this preview."
                                        });
                                    }
                                } else {
                                    ui.set_status_message("Aim at a REV face, then drag to draw.");
                                }
                            }
                        }
                    } else if viewport_edit {
                        if let Some(viewport) = ui.scene_viewport() {
                            if let Some((origin, direction)) = editor_ray(&camera, viewport, pointer) {
                                if ui.editor_tool == EditorTool::Select {
                                    if let (Some(document), Some(face)) =
                                        (active_model_document.as_ref(), hovered_model_face(&scene, origin, direction))
                                    {
                                        if !ui.paused || live_simulated_s > 1e-9 {
                                            let cell = voxel_lattice_key(
                                                document.voxels[face.voxel_index].position_m,
                                                document.cell_size_m,
                                            );
                                            selection_drag = Some(BoxSelectionDrag {
                                                end_index: face.voxel_index,
                                                start_cell: cell,
                                                end_cell: cell,
                                                dragged: false,
                                            });
                                            ui.set_status_message("Pause and reset before moving model parts.");
                                        } else {
                                            let clip_index = ui.selected_animation_clip.filter(|&clip| {
                                                document.skeleton.as_ref().is_some_and(|skeleton| {
                                                    clip < skeleton.clips.len()
                                                })
                                            });
                                            let time_s = f64::from(ui.selected_animation_time_seconds);
                                            let bone_hit = if ui.show_rig_overlay && ui.rig_edit_mode == ui::RigEditMode::Move {
                                                bone_drag_hit(
                                                    document,
                                                    clip_index,
                                                    time_s,
                                                    origin,
                                                    direction,
                                                    viewport[3],
                                                )
                                            } else {
                                                None
                                            }
                                            .filter(|(_, center)| {
                                                (*center - origin).dot(direction)
                                                    <= (face.center - origin).dot(direction)
                                                        + document.cell_size_m as f32 * 2.0
                                            });
                                            if let Some((bone_index, center)) = bone_hit {
                                                if let Some(skeleton) = document.skeleton.as_ref() {
                                                    let original_translation = if let Some(clip_index) = clip_index {
                                                        skeleton.sample_bone_local_pose(
                                                            clip_index,
                                                            bone_index,
                                                            time_s,
                                                        ).map(|pose| pose.translation_m)
                                                    } else {
                                                        skeleton.bones.get(bone_index)
                                                            .map(|bone| bone.rest_translation_m)
                                                            .ok_or_else(|| "bone index is out of bounds".to_owned())
                                                    };
                                                    let parent_rotation = bone_parent_rotation(
                                                        skeleton,
                                                        clip_index,
                                                        time_s,
                                                        bone_index,
                                                    );
                                                    if let (Ok(original_translation), Ok(parent_rotation)) =
                                                        (original_translation, parent_rotation)
                                                    {
                                                        if let Some(start_point) = ray_plane_point(
                                                            origin,
                                                            direction,
                                                            center,
                                                            camera.forward(),
                                                        ) {
                                                            let bone_names = skeleton.bones.iter()
                                                                .map(|bone| bone.name.clone())
                                                                .collect();
                                                            ui.set_rig_authoring_state(
                                                                bone_names,
                                                                Some(bone_index),
                                                                skeleton.clips.len(),
                                                            );
                                                            bone_move_drag = Some(BoneMoveDrag {
                                                                bone_index,
                                                                clip_index,
                                                                time_s,
                                                                original_translation,
                                                                parent_rotation,
                                                                plane_origin: center,
                                                                plane_normal: camera.forward(),
                                                                start_point,
                                                                last_delta: Vec3::ZERO,
                                                            });
                                                            ui.set_status_message(format!(
                                                                "Drag bone {} to pose it; release to keep the pose.",
                                                                ui.rig_bone_names.get(bone_index)
                                                                    .map_or("bone", String::as_str)
                                                            ));
                                                        }
                                                    }
                                                }
                                            } else if ui.selected_voxel_indices.contains(&face.voxel_index) {
                                                let start_point = ray_plane_point(
                                                    origin,
                                                    direction,
                                                    face.center,
                                                    camera.forward(),
                                                );
                                                if let Some(start_point) = start_point {
                                                    part_move_drag = Some(MovePartDrag {
                                                        selected: ui.selected_voxel_indices.clone(),
                                                        plane_origin: face.center,
                                                        plane_normal: camera.forward(),
                                                        start_point,
                                                        delta_cells: IVec3::ZERO,
                                                    });
                                                    ui.set_status_message("Drag the selected REV or connected part to move it; release to apply.");
                                                }
                                            } else {
                                                let cell = voxel_lattice_key(
                                                    document.voxels[face.voxel_index].position_m,
                                                    document.cell_size_m,
                                                );
                                                selection_drag = Some(BoxSelectionDrag {
                                                    end_index: face.voxel_index,
                                                    start_cell: cell,
                                                    end_cell: cell,
                                                    dragged: false,
                                                });
                                                ui.set_status_message(
                                                    "Click a REV to select its connected part; drag across empty space or other REVs for a 3D box.",
                                                );
                                            }
                                        }
                                    } else {
                                        ui.set_model_selection(None, None, None);
                                        ui.set_status_message("Click or drag across REVs to select them.");
                                    }
                                } else if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message(
                                        "Pause and reset the model before changing voxel geometry.",
                                    );
                                } else {
                                    accumulator = 0.0;
                                    let result = (|| {
                                        let document = active_model_document
                                            .as_mut()
                                            .ok_or_else(|| "no voxel model is open".to_owned())?;
                                        let (selection, edit) = edit_model_at_ray(
                                            document,
                                            &scene,
                                            origin,
                                            direction,
                                            ui.editor_tool,
                                            ui.brush_size_cells,
                                        )?;
                                        let Some(edit) = edit else {
                                            return Ok(None);
                                        };
                                        let rebuilt = commit_model_edit(
                                            document,
                                            &mut model_edit_history,
                                            &config,
                                            edit,
                                        )?;
                                        Ok::<_, String>(Some((rebuilt, selection)))
                                    })();
                                    match result {
                                        Ok(Some((rebuilt, selection))) => {
                                            scene = rebuilt;
                                            model_revision = model_revision.wrapping_add(1);
                                            let weights_invalidated = active_model_document
                                                .as_mut()
                                                .is_some_and(invalidate_skeleton_weights_after_geometry_edit);
                                            if let Some(document) = active_model_document.as_ref() {
                                                update_model_ui(&mut ui, document, selection);
                                            }
                                            sync_model_history_ui(&mut ui, &model_edit_history);
                                            ui.set_status_message(match ui.editor_tool {
                                                EditorTool::AddVoxel => "Added one adjacent REV. Ctrl+Z undoes this edit.",
                                                EditorTool::EraseVoxel => "Carved the brush area. Ctrl+Z restores it.",
                                                EditorTool::AddBox => "Stamped a block of REVs. Ctrl+Z undoes the stamp.",
                                                EditorTool::AddSphere => "Stamped a sphere of REVs. Ctrl+Z undoes the stamp.",
                                                EditorTool::AddCylinder => "Stamped a cylinder of REVs. Ctrl+Z undoes the stamp.",
                                                EditorTool::Select => "Selected REV.",
                                            });
                                            if weights_invalidated {
                                                ui.set_status_message("Model edited. Bone weights were cleared because voxel geometry changed; regenerate automatic weights.");
                                            }
                                            hovered_face = hovered_face_at_pointer(
                                                &ui, &scene, &camera, pointer,
                                            );
                                        }
                                        Ok(None) => ui.set_status_message("Aim at a REV to edit the model."),
                                        Err(error) => ui.set_status_message(format!("Error: {error}")),
                                        }
                                }
                            }
                        }
                    }
                    if state == ElementState::Released {
                        bone_move_drag = None;
                        if let Some(drag) = part_move_drag.take() {
                            sculpt_preview_cells.clear();
                            sculpt_preview_bounds = None;
                            if drag.delta_cells != IVec3::ZERO {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before moving voxel parts.");
                                } else if let Some(document) = active_model_document.as_mut() {
                                    let result = selected_part_translate_edit(
                                        document,
                                        &drag.selected,
                                        drag.delta_cells,
                                    )
                                    .and_then(|edit| {
                                        commit_model_edit(
                                            document,
                                            &mut model_edit_history,
                                            &config,
                                            edit,
                                        )
                                    });
                                    match result {
                                        Ok(rebuilt) => {
                                            scene = rebuilt;
                                            model_revision = model_revision.wrapping_add(1);
                                            let weights_invalidated = active_model_document
                                                .as_mut()
                                                .is_some_and(invalidate_skeleton_weights_after_geometry_edit);
                                            if let Some(document) = active_model_document.as_ref() {
                                                update_model_ui_selection(
                                                    &mut ui,
                                                    document,
                                                    drag.selected,
                                                );
                                            }
                                            sync_model_history_ui(&mut ui, &model_edit_history);
                                            ui.set_status_message("Moved selected part. Ctrl+Z undoes this edit.");
                                            if weights_invalidated {
                                                ui.set_status_message("Moved selected part. Bone weights were cleared because voxel positions changed; regenerate automatic weights.");
                                            }
                                            accumulator = 0.0;
                                            last = Instant::now();
                                        }
                                        Err(error) => ui.set_status_message(format!("Could not move selection: {error}")),
                                    }
                                }
                            }
                        }
                        if let Some(drag) = scene_move_drag.take() {
                            if drag.delta_cells != IVec3::ZERO
                                && ui.selected_scene_model == Some(drag.instance_index)
                            {
                                run_scene_instance_edit(
                                    &mut scene,
                                    &mut active_scene_document,
                                    active_scene_document_path.as_deref(),
                                    &mut ui,
                                    &mut live_simulated_s,
                                    &mut live_wall_s,
                                    &mut accumulator,
                                    &mut last,
                                    SceneInstanceEdit::Translate(drag.delta_cells),
                                );
                            }
                        }
                        if let Some(mut drag) = selection_drag.take() {
                            if let Some(viewport) = ui.scene_viewport() {
                                if let Some((origin, direction)) = editor_ray(&camera, viewport, pointer) {
                                    if let Some(face) = hovered_model_face(&scene, origin, direction) {
                                        if let Some(document) = active_model_document.as_ref() {
                                            if let Some(voxel) = document.voxels.get(face.voxel_index) {
                                                drag.end_index = face.voxel_index;
                                                drag.end_cell = voxel_lattice_key(
                                                    voxel.position_m,
                                                    document.cell_size_m,
                                                );
                                                drag.dragged |= drag.end_cell != drag.start_cell;
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(document) = active_model_document.as_ref() {
                                let (indices, bounds_m, dimensions_cm, diagram, mass, position) =
                                    if drag.dragged {
                                        selected_voxels_in_box(
                                            document,
                                            drag.start_cell,
                                            drag.end_cell,
                                        )
                                    } else {
                                        summarize_voxel_selection(
                                            document,
                                            connected_voxel_part(document, drag.end_index),
                                        )
                                    };
                                let count = indices.len();
                                ui.set_model_box_selection(
                                    indices,
                                    bounds_m,
                                    dimensions_cm,
                                    diagram,
                                    position,
                                    mass,
                                );
                                ui.set_status_message(if drag.dragged {
                                    format!(
                                        "Selected {count} REV{} in a {} × {} × {} cm box.",
                                        if count == 1 { "" } else { "s" },
                                        dimensions_cm[0],
                                        dimensions_cm[1],
                                        dimensions_cm[2],
                                    )
                                } else {
                                    format!(
                                        "Selected connected part: {count} REV{}.",
                                        if count == 1 { "" } else { "s" }
                                    )
                                });
                            }
                        }
                        if let Some(drag) = sculpt_drag.take() {
                            sculpt_preview_cells.clear();
                            sculpt_preview_bounds = None;
                            let sculpt_tool = drag.tool;
                            if !ui.paused || live_simulated_s > 1e-9 {
                                ui.set_status_message(
                                    "Pause and reset the model before changing voxel geometry.",
                                );
                            } else {
                                accumulator = 0.0;
                                let result = (|| {
                                    let document = active_model_document
                                        .as_mut()
                                        .ok_or_else(|| "no voxel model is open".to_owned())?;
                                    let (selection, edit, count) =
                                        drag_model_edit(document, &scene, drag)?;
                                    let Some(edit) = edit else {
                                        return Ok(None);
                                    };
                                    let rebuilt = commit_model_edit(
                                        document,
                                        &mut model_edit_history,
                                        &config,
                                        edit,
                                    )?;
                                    Ok::<_, String>(Some((rebuilt, selection, count)))
                                })();
                                match result {
                                    Ok(Some((rebuilt, selection, count))) => {
                                        scene = rebuilt;
                                        model_revision = model_revision.wrapping_add(1);
                                        let weights_invalidated = active_model_document
                                            .as_mut()
                                            .is_some_and(invalidate_skeleton_weights_after_geometry_edit);
                                        if let Some(document) = active_model_document.as_ref() {
                                            update_model_ui(&mut ui, document, selection);
                                        }
                                        sync_model_history_ui(&mut ui, &model_edit_history);
                                        ui.set_status_message(match sculpt_tool {
                                            EditorTool::AddVoxel => format!(
                                                "Added {count} connected REVs. Ctrl+Z undoes the whole row."
                                            ),
                                            EditorTool::AddBox => format!(
                                                "Created a {count} REV rectangle. Ctrl+Z undoes the whole patch."
                                            ),
                                            _ => format!("Added {count} REVs."),
                                        });
                                        if weights_invalidated {
                                            ui.set_status_message("Model edited. Bone weights were cleared because voxel geometry changed; regenerate automatic weights.");
                                        }
                                        hovered_face = hovered_face_at_pointer(
                                            &ui, &scene, &camera, pointer,
                                        );
                                    }
                                    Ok(None) => {}
                                    Err(error) => {
                                        ui.set_status_message(format!("Error: {error}"));
                                    }
                                }
                            }
                        }
                    }
                    if state == ElementState::Pressed && !viewport_edit && !viewport_scene_select {
                        let click_action = ui.click(pointer.0, pointer.1);
                        if ui.timeline_is_dragging() {
                            dragging = true;
                        }
                        match click_action {
                            Some(action @ (UiAction::SetScenePhysics { .. }
                                | UiAction::SetSceneSelectionMode(_)
                                | UiAction::SetScenePosition { .. } | UiAction::SetSceneRotation { .. }
                                | UiAction::SetSceneVelocity { .. } | UiAction::SetSceneForce { .. }
                                | UiAction::SetSceneInitialTemperature { .. }
                                | UiAction::SetSceneThermalBonding(_)
                                | UiAction::CycleSceneMaterial | UiAction::LoadSceneMaterial
                                | UiAction::SaveSceneMaterial | UiAction::CreateSceneMaterial
                                | UiAction::SetChemistryEnabled(_) | UiAction::SetChemistryInterval { .. }
                                | UiAction::ChooseChemistryRecipe | UiAction::UseBuiltinChemistryRecipes
                                | UiAction::CreateImpactScene { .. })) => pending_scene_action = Some(action),
                            Some(UiAction::Shape(shape)) => {
                                if mode == AppMode::Interactive && generated_world.is_none() {
                                    tool = Some(launch(shape, &scene, &ui));
                                }
                            }
                            Some(UiAction::NewScene) => {
                                scene_load_job = None;
                                authored_physics = None;
                                if let Some(source) = authored_render_source.take() { renderer.set_authored_physics_frame(&source, None); }
                                ui.standalone_editor_app = false;
                                let result = SceneDocument::new("Untitled Scene", config.clone())
                                    .and_then(|document| {
                                        document
                                            .build_scene(Path::new("Untitled.vxscene"))
                                            .map(|runtime_scene| (document, runtime_scene))
                                    });
                                match result {
                                    Ok((document, runtime_scene)) => {
                                        grabbed_cell = None;
                                        world_physics = None;
                                        generated_world = None;
                                        world_character = None;
                                        growth_scheduler = GrowthScheduler::default();
                                        scene = runtime_scene;
                                        active_scene_document = Some(document);
                                        active_scene_document_path = None;
                                        active_model_document = None;
                                        model_revision = model_revision.wrapping_add(1);
                                        active_model_document_path = None;
                                        let document = active_scene_document.as_ref().unwrap();
                                        ui.set_active_scene_document(
                                            Some(document.name.clone()),
                                            scene_model_names(document),
                                        );
                                        ui.set_status_message(
                                            "New empty scene. Save to create a .vxscene file.",
                                        );
                                        ui.world_active = true;
                                        ui.editor_mode = true;
                                        ui.paused = true;
                                        ui.show_bonds = false;
                                        tool = None;
                                        mode = AppMode::Interactive;
                                        playback_frames.clear();
                                        playback_time_s = 0.0;
                                        live_simulated_s = 0.0;
                                        live_wall_s = 0.0;
                                        accumulator = 0.0;
                                        frame_camera_on_scene(&scene, &mut camera);
                                        orbit = false;
                                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                                        window.set_cursor_visible(true);
                                        last = Instant::now();
                                    }
                                    Err(error) => {
                                        ui.set_status_message(format!("Error: {error}"));
                                        eprintln!("new scene could not be created: {error}");
                                    }
                                }
                            }
                            Some(UiAction::OpenScene) => {
                                if let Some(path) = pick_scene_document_path(false, &mut ui, &mut accumulator, &mut last) {
                                    ui.set_status_message("Loading scene assets in the background…");
                                    let (sender, receiver) = mpsc::channel();
                                    scene_load_job = Some(receiver);
                                    std::thread::spawn(move || {
                                        let result = load_document::<SceneDocument>(&path).and_then(|document|
                                            document.build_scene(&path).map(|runtime| (document, runtime, Some(path))));
                                        let _ = sender.send(result);
                                    });
                                }
                            }
                            Some(UiAction::SaveScene) => {
                                if let Some(document) = active_scene_document.as_ref() {
                                    let selected_path = active_scene_document_path.clone().or_else(|| {
                                        pick_scene_document_path(
                                            true,
                                            &mut ui,
                                            &mut accumulator,
                                            &mut last,
                                        )
                                    });
                                    if let Some(path) = selected_path.map(ensure_scene_extension) {
                                        let old_path = active_scene_document_path.as_deref().unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                        let result = document.rebased_paths(old_path, &path).and_then(|saved| {
                                            save_document(&path, &saved).map(|()| saved)
                                        });
                                        match result {
                                            Ok(saved) => {
                                                active_scene_document = Some(saved);
                                                let previous_scene_path = active_scene_document_path.clone();
                                                active_scene_document_path = Some(path.clone());
                                                ui.set_status_message(format!(
                                                    "Saved authored scene; live physics is not a checkpoint: {}",
                                                    path.file_name()
                                                        .and_then(|name| name.to_str())
                                                        .unwrap_or(".vxscene"),
                                                ));
                                                // The chemistry settings live beside the scene file, so a
                                                // scene saved under a new name takes them along.
                                                let moved = ui
                                                    .scene_chemistry
                                                    .rebased(previous_scene_path.as_deref(), &path);
                                                if moved != ChemistrySettings::default()
                                                    || ChemistrySettings::sidecar_path(&path).exists()
                                                {
                                                    if let Err(error) = moved.save(&path) {
                                                        eprintln!("chemistry settings: {error}");
                                                        ui.set_status_message(format!(
                                                            "Error: scene saved, but its chemistry settings were not: {error}"
                                                        ));
                                                    }
                                                }
                                                ui.scene_chemistry = moved;
                                                chemistry_synced_for = Some(Some(path.clone()));
                                            }
                                            Err(error) => {
                                                ui.set_status_message(format!("Error: {error}"));
                                                eprintln!(
                                                    "scene {} could not be saved: {error}",
                                                    path.display()
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            Some(action @ (UiAction::SavePhysicalState | UiAction::LoadPhysicalState)) => {
                                let saving = matches!(action, UiAction::SavePhysicalState);
                                let state_path = active_scene_document_path
                                    .as_deref()
                                    .map(|path| path.with_extension("vxstate"));
                                let result: Result<String, String> = match state_path {
                                    None => Err("Save the scene first; its state is stored beside it.".into()),
                                    Some(_) if authored_physics.is_some() || !ui.paused => {
                                        Err("Pause the simulation first; Stop keeps the full state.".into())
                                    }
                                    Some(path) if saving => scene
                                        .save_checkpoint(&path)
                                        .map(|()| format!("Saved physical state at {:.3} s", scene.simulated_seconds)),
                                    Some(path) => Scene::load_checkpoint(&path).map(|loaded| {
                                        let seconds = loaded.simulated_seconds;
                                        scene = loaded;
                                        model_revision = model_revision.wrapping_add(1);
                                        format!("Loaded physical state at {seconds:.3} s; Run continues it, R resets.")
                                    }),
                                };
                                match result {
                                    Ok(message) => ui.set_status_message(message),
                                    Err(error) => {
                                        eprintln!("physical state: {error}");
                                        ui.set_status_message(format!("Error: {error}"));
                                    }
                                }
                            }
                            Some(UiAction::NewModel) => {
                                scene_load_job = None;
                                let result = (|| {
                                    let oak = voxel_physics_lab::material::Material::component_id("oak")
                                        .or_else(|| voxel_physics_lab::material::Material::component_id("wood"))
                                        .ok_or_else(|| "oak material is unavailable".to_owned())?;
                                    let initial = Scene::from_voxels(
                                        config.clone(),
                                        vec![VoxelSpawn::at_cell([0, 0, 0], oak)],
                                    )?;
                                    let document = VoxelModelDocument::from_scene("Untitled Model", &initial);
                                    let runtime_scene = document.build_scene(config.clone())?;
                                    Ok::<_, String>((document, runtime_scene))
                                })();
                                match result {
                                    Ok((document, runtime_scene)) => {
                                        grabbed_cell = None;
                                        world_physics = None;
                                        generated_world = None;
                                        world_character = None;
                                        growth_scheduler = GrowthScheduler::default();
                                        active_scene_document = None;
                                        active_scene_document_path = None;
                                        active_model_document_path = None;
                                        scene = runtime_scene;
                                        update_model_ui(&mut ui, &document, Some(0));
                                        model_edit_history.clear();
                                        sync_model_history_ui(&mut ui, &model_edit_history);
                                        active_model_document = Some(document);
                                        model_revision = model_revision.wrapping_add(1);
                                        skinning_weights_revision = skinning_weights_revision.wrapping_add(1);
                                        ui.set_status_message(
                                            "Add REV: click one cell or drag a connected row; Block: drag a rectangular patch.",
                                        );
                                        ui.world_active = true;
                                        ui.editor_mode = true;
                                        ui.paused = true;
                                        ui.show_bonds = false;
                                        ui.editor_tool = EditorTool::AddVoxel;
                                        tool = None;
                                        mode = AppMode::Interactive;
                                        playback_frames.clear();
                                        playback_time_s = 0.0;
                                        live_simulated_s = 0.0;
                                        live_wall_s = 0.0;
                                        accumulator = 0.0;
                                        frame_camera_on_scene(&scene, &mut camera);
                                        orbit = false;
                                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                                        window.set_cursor_visible(true);
                                        last = Instant::now();
                                    }
                                    Err(error) => {
                                        ui.set_status_message(format!("Error: {error}"));
                                        eprintln!("new voxel model could not be created: {error}");
                                    }
                                }
                            }
                            Some(UiAction::OpenModel) => {
                                scene_load_job = None;
                                if let Some(path) = pick_voxel_model_document_path(
                                    false,
                                    &mut ui,
                                    &mut accumulator,
                                    &mut last,
                                ) {
                                    let result = load_document::<VoxelModelDocument>(&path)
                                        .and_then(|document| {
                                            build_editor_preview_scene(&document, config.clone())
                                                .map(|preview_scene| (document, preview_scene))
                                        });
                                    match result {
                                        Ok((document, runtime_scene)) => {
                                            grabbed_cell = None;
                                            world_physics = None;
                                            generated_world = None;
                                            world_character = None;
                                            growth_scheduler = GrowthScheduler::default();
                                            active_scene_document = None;
                                            active_scene_document_path = None;
                                            active_model_document_path = Some(path.clone());
                                            scene = runtime_scene;
                                            update_model_ui(&mut ui, &document, Some(0));
                                            model_edit_history.clear();
                                            sync_model_history_ui(&mut ui, &model_edit_history);
                                            active_model_document = Some(document);
                                            model_revision = model_revision.wrapping_add(1);
                                            skinning_weights_revision = skinning_weights_revision.wrapping_add(1);
                                            ui.set_status_message(format!(
                                                "Opened voxel model: {}",
                                                path.file_name().and_then(|name| name.to_str()).unwrap_or(".vxm"),
                                            ));
                                            ui.world_active = true;
                                            ui.editor_mode = true;
                                            ui.paused = true;
                                            ui.show_bonds = false;
                                            ui.editor_tool = EditorTool::AddVoxel;
                                            tool = None;
                                            mode = AppMode::Interactive;
                                            playback_frames.clear();
                                            playback_time_s = 0.0;
                                            live_simulated_s = 0.0;
                                            live_wall_s = 0.0;
                                            accumulator = 0.0;
                                            frame_camera_on_scene(&scene, &mut camera);
                                            orbit = false;
                                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                                            window.set_cursor_visible(true);
                                            last = Instant::now();
                                        }
                                        Err(error) => {
                                            ui.set_status_message(format!("Error: {error}"));
                                            eprintln!("model {} could not be opened: {error}", path.display());
                                        }
                                    }
                                }
                            }
                            Some(action @ (UiAction::SaveModel | UiAction::SaveModelAs)) => {
                                let save_as = matches!(action, UiAction::SaveModelAs);
                                if let Some(current) = active_model_document.as_ref() {
                                    let was_paused = ui.paused;
                                    ui.paused = true;
                                    accumulator = 0.0;
                                    match snapshot_model_preserving_authoring(current, &scene) {
                                        Ok(snapshot) => {
                                            let path = if save_as {
                                                pick_voxel_model_document_path(
                                                    true,
                                                    &mut ui,
                                                    &mut accumulator,
                                                    &mut last,
                                                )
                                            } else {
                                                active_model_document_path.clone().or_else(|| {
                                                    pick_voxel_model_document_path(
                                                        true,
                                                        &mut ui,
                                                        &mut accumulator,
                                                        &mut last,
                                                    )
                                                })
                                            };
                                            if let Some(path) = path.map(ensure_voxel_model_extension) {
                                                match save_document(&path, &snapshot) {
                                                    Ok(()) => {
                                                        let selected = ui.selected_voxel_indices.clone();
                                                        update_model_ui_selection(&mut ui, &snapshot, selected);
                                                        sync_model_history_ui(&mut ui, &model_edit_history);
                                                        active_model_document = Some(snapshot);
                                                        skinning_weights_revision = skinning_weights_revision.wrapping_add(1);
                                                        active_model_document_path = Some(path.clone());
                                                        ui.set_status_message(format!(
                                                            "Saved voxel model: {}",
                                                            path.file_name().and_then(|name| name.to_str()).unwrap_or(".vxm"),
                                                        ));
                                                    }
                                                    Err(error) => {
                                                        ui.set_status_message(format!("Error: {error}"));
                                                        eprintln!("model {} could not be saved: {error}", path.display());
                                                    }
                                                }
                                            }
                                        }
                                        Err(error) => ui.set_status_message(format!("Model save skipped: {error}")),
                                    }
                                    ui.paused = was_paused;
                                    accumulator = 0.0;
                                    last = Instant::now();
                                }
                            }
                            Some(UiAction::ExportModelObj) => {
                                if model_export_job.is_some() {
                                    ui.set_status_message("OBJ export is already running.");
                                } else if let Some(current) = active_model_document.as_ref() {
                                    if let Some(path) = pick_obj_export_path(
                                        &mut ui,
                                        &mut accumulator,
                                        &mut last,
                                    )
                                    .map(ensure_obj_extension)
                                    {
                                        let was_paused = ui.paused;
                                        ui.paused = true;
                                        accumulator = 0.0;
                                        let snapshot = VoxelModelDocument::from_scene(
                                            current.name.clone(),
                                            &scene,
                                        );
                                        let (sender, receiver) = mpsc::channel();
                                        let worker_path = path.clone();
                                        match std::thread::Builder::new()
                                            .name("voxel-model-obj-export".into())
                                            .spawn(move || {
                                                let result = export_obj(&snapshot, &worker_path);
                                                let _ = sender.send(result);
                                            })
                                        {
                                            Ok(_) => {
                                                model_export_job = Some(receiver);
                                                ui.set_status_message(format!(
                                                    "Exporting OBJ: {}",
                                                    path.file_name()
                                                        .and_then(|name| name.to_str())
                                                        .unwrap_or("model.obj"),
                                                ));
                                            }
                                            Err(error) => ui.set_status_message(format!(
                                                "Could not start OBJ export: {error}"
                                            )),
                                        }
                                        ui.paused = was_paused;
                                        accumulator = 0.0;
                                        last = Instant::now();
                                    }
                                }
                            }
                            Some(UiAction::AddModelToScene) => {
                                if authored_physics.is_some() || !ui.paused {
                                    ui.set_status_message("Pause and press R before adding a model.");
                                } else if let Some(mut updated) = active_scene_document.clone() {
                                    if let Some(path) = pick_voxel_model_document_path(false, &mut ui, &mut accumulator, &mut last) {
                                        let scene_path = active_scene_document_path.clone();
                                        let max_x = scene.positions.iter().map(|position| position.x).max_by(f64::total_cmp);
                                        let (sender, receiver) = mpsc::channel();
                                        scene_load_job = Some(receiver);
                                        ui.set_status_message("Loading voxel model and rebuilding scene in the background…");
                                        std::thread::spawn(move || {
                                            let result = (|| {
                                                let model: VoxelModelDocument = load_document(&path)?;
                                                let min_x = model.voxels.iter().map(|voxel| voxel.position_m.x).min_by(f64::total_cmp)
                                                    .ok_or("voxel model contains no REVs")?;
                                                let x = max_x.map_or(0.0, |value| ((value + scene::DCELL * 10.0 - min_x)/scene::DCELL).ceil() * scene::DCELL);
                                                updated.add_model(ModelInstanceDocument { asset_path: path, translation_m: DVec3::new(x,0.0,0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE })?;
                                                let runtime = updated.build_scene(scene_path.as_deref().unwrap_or_else(|| Path::new("Untitled.vxscene")))?;
                                                Ok((updated, runtime, scene_path))
                                            })();
                                            let _ = sender.send(result);
                                        });
                                    }
                                }
                            }
                            Some(UiAction::ImportMesh) => {
                                if generated_world.is_some()
                                    || (active_scene_document.is_none()
                                        && active_model_document.is_none())
                                {
                                    ui.set_status_message(
                                        "Mesh import requires an authored scene or voxel model.",
                                    );
                                } else if let Some(path) = pick_mesh_import_source_path(
                                    &mut ui,
                                    &mut accumulator,
                                    &mut last,
                                ) {
                                    mesh_import_previous_paused = Some(ui.paused);
                                    ui.paused = true;
                                    mesh_import_source_path = Some(path.clone());
                                    mesh_import_preview = None;
                                    mesh_import_preview_scene = None;
                                    mesh_import_camera_backup = None;
                                    mesh_import_job = None;
                                    ui.begin_mesh_import(
                                        path.file_name()
                                            .and_then(|name| name.to_str())
                                            .unwrap_or("mesh"),
                                    );
                                    ui.set_status_message("Choose scale and voxelization mode, then build a preview.");
                                }
                            }
                            Some(UiAction::BuildMeshPreview { scale, filled }) => {
                                'build_mesh_preview: {
                                let Some(source_path) = mesh_import_source_path.clone() else {
                                    ui.set_mesh_import_error("Select an OBJ or binary GLB file first.");
                                    break 'build_mesh_preview;
                                };
                                let Some(oak_id) = Material::component_id("oak")
                                    .or_else(|| Material::component_id("wood"))
                                else {
                                    ui.set_mesh_import_error("Oak material is unavailable.");
                                    break 'build_mesh_preview;
                                };
                                let options = VoxelizationOptions {
                                    scale: DVec3::splat(f64::from(scale)),
                                    mode: if filled {
                                        VoxelizationMode::Filled
                                    } else {
                                        VoxelizationMode::Shell
                                    },
                                    ..VoxelizationOptions::default()
                                };
                                let worker_config = config.clone();
                                let (sender, receiver) = mpsc::channel();
                                match std::thread::Builder::new()
                                    .name("mesh-voxelization".into())
                                    .spawn(move || {
                                        let result = import_mesh(&source_path, options).and_then(
                                            |preview| {
                                                preview
                                                    .build_scene(worker_config, oak_id)
                                                    .map(|scene| (preview, scene))
                                            },
                                        );
                                        let _ = sender.send(result);
                                    })
                                {
                                    Ok(_worker) => {
                                        mesh_import_preview = None;
                                        mesh_import_preview_scene = None;
                                        mesh_import_job = Some(receiver);
                                        ui.set_mesh_import_busy(true);
                                        ui.set_status_message("Voxelizing mesh on a background worker...");
                                    }
                                    Err(error) => ui.set_mesh_import_error(format!(
                                        "Could not start mesh voxelization: {error}"
                                    )),
                                }
                                }
                            }
                            Some(UiAction::CommitMeshImport { add_to_scene }) => {
                                'commit_mesh_import: {
                                let Some(preview) = mesh_import_preview.as_ref() else {
                                    ui.set_mesh_import_error("Build a mesh preview before saving.");
                                    break 'commit_mesh_import;
                                };
                                if add_to_scene && active_scene_document.is_none() {
                                    ui.set_mesh_import_error(
                                        "Adding an imported model requires an authored scene.",
                                    );
                                    break 'commit_mesh_import;
                                }
                                let Some(path) = pick_voxel_model_document_path(
                                    true,
                                    &mut ui,
                                    &mut accumulator,
                                    &mut last,
                                ) else {
                                    break 'commit_mesh_import;
                                };
                                let path = ensure_voxel_model_extension(path);
                                let material = Material::table()
                                    .into_iter()
                                    .find(|material| material.name.eq_ignore_ascii_case("oak"))
                                    .or_else(|| {
                                        Material::table()
                                            .into_iter()
                                            .find(|material| material.name.eq_ignore_ascii_case("wood"))
                                    });
                                let Some(material) = material else {
                                    ui.set_mesh_import_error("Oak material is unavailable.");
                                    break 'commit_mesh_import;
                                };
                                let model_name = path
                                    .file_stem()
                                    .and_then(|name| name.to_str())
                                    .filter(|name| !name.trim().is_empty())
                                    .unwrap_or("Imported Model")
                                    .to_owned();
                                let document = match preview.to_model_document(model_name, material) {
                                    Ok(document) => document,
                                    Err(error) => {
                                        ui.set_mesh_import_error(error);
                                            break 'commit_mesh_import;
                                    }
                                };
                                if !add_to_scene {
                                    let result = document
                                        .build_scene(config.clone())
                                        .and_then(|runtime_scene| {
                                            save_document(&path, &document)
                                                .map(|()| runtime_scene)
                                        });
                                    match result {
                                        Ok(runtime_scene) => {
                                            grabbed_cell = None;
                                            world_physics = None;
                                            generated_world = None;
                                            world_character = None;
                                            growth_scheduler = GrowthScheduler::default();
                                            active_scene_document = None;
                                            active_scene_document_path = None;
                                            active_model_document_path = Some(path.clone());
                                            scene = runtime_scene;
                                            update_model_ui(&mut ui, &document, None);
                                            model_edit_history.clear();
                                            sync_model_history_ui(&mut ui, &model_edit_history);
                                            active_model_document = Some(document);
                                            model_revision = model_revision.wrapping_add(1);
                                            skinning_weights_revision = skinning_weights_revision.wrapping_add(1);
                                            ui.set_status_message(format!(
                                                "Saved and opened voxel model: {}",
                                                path.file_name()
                                                    .and_then(|name| name.to_str())
                                                    .unwrap_or(".vxm"),
                                            ));
                                            ui.world_active = true;
                                            ui.editor_mode = true;
                                            ui.paused = true;
                                            ui.show_bonds = false;
                                            ui.editor_tool = EditorTool::Select;
                                            tool = None;
                                            mode = AppMode::Interactive;
                                            playback_frames.clear();
                                            playback_time_s = 0.0;
                                            live_simulated_s = 0.0;
                                            live_wall_s = 0.0;
                                            accumulator = 0.0;
                                            frame_camera_on_scene(&scene, &mut camera);
                                            orbit = false;
                                            mesh_import_camera_backup = None;
                                            mesh_import_preview = None;
                                            mesh_import_preview_scene = None;
                                            mesh_import_source_path = None;
                                            mesh_import_previous_paused = None;
                                            ui.finish_mesh_import();
                                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                                            window.set_cursor_visible(true);
                                            last = Instant::now();
                                        }
                                        Err(error) => ui.set_mesh_import_error(error),
                                    }
                                } else {
                                    let result = (|| {
                                        save_document(&path, &document)?;
                                        let model_min_x = document
                                            .voxels
                                            .iter()
                                            .map(|voxel| voxel.position_m.x)
                                            .min_by(f64::total_cmp)
                                            .ok_or_else(|| "voxel model contains no REVs".to_owned())?;
                                        let mut updated = active_scene_document
                                            .as_ref()
                                            .cloned()
                                            .ok_or_else(|| "no authored scene is open".to_owned())?;
                                        let translation_x = scene
                                            .positions
                                            .iter()
                                            .map(|position| position.x)
                                            .max_by(f64::total_cmp)
                                            .map_or(0.0, |max_x| {
                                                ((max_x + scene::DCELL * 10.0 - model_min_x)
                                                    / scene::DCELL)
                                                    .ceil()
                                                    * scene::DCELL
                                            });
                                        updated.add_model(ModelInstanceDocument {
                                            asset_path: path.clone(),
                                            translation_m: DVec3::new(translation_x, 0.0, 0.0),
                                            rotation: Quat::IDENTITY,
                                            scale: Vec3::ONE,
                                        })?;
                                        let scene_path = active_scene_document_path
                                            .as_deref()
                                            .unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                        let runtime_scene = updated.build_scene(scene_path)?;
                                        Ok::<_, String>((updated, runtime_scene))
                                    })();
                                    match result {
                                        Ok((updated, runtime_scene)) => {
                                            active_scene_document = Some(updated);
                                            active_model_document = None;
                                            model_revision = model_revision.wrapping_add(1);
                                            active_model_document_path = None;
                                            let imported_index = active_scene_document
                                                .as_ref()
                                                .map_or(0, |document| document.models.len().saturating_sub(1));
                                            ui.set_active_scene_document(
                                                active_scene_document.as_ref().map(|doc| doc.name.clone()),
                                                active_scene_document
                                                    .as_ref()
                                                    .map(scene_model_names)
                                                    .unwrap_or_default(),
                                            );
                                            ui.selected_scene_model = Some(imported_index);
                                            ui.set_selected_scene_model_info(
                                                active_scene_document.as_ref().and_then(|document| {
                                                    scene_model_inspector_info(document, imported_index)
                                                }),
                                            );
                                            scene = runtime_scene;
                                            ui.world_active = true;
                                            ui.editor_mode = true;
                                            ui.paused = true;
                                            ui.show_bonds = false;
                                            ui.editor_tool = EditorTool::Select;
                                            tool = None;
                                            frame_camera_on_scene(&scene, &mut camera);
                                            ui.set_status_message(format!(
                                                "Saved model and added it to the scene: {}. Save the scene to keep the reference.",
                                                path.file_name()
                                                    .and_then(|name| name.to_str())
                                                    .unwrap_or(".vxm"),
                                            ));
                                            mesh_import_camera_backup = None;
                                            mesh_import_preview = None;
                                            mesh_import_preview_scene = None;
                                            mesh_import_source_path = None;
                                            mesh_import_previous_paused = None;
                                            ui.finish_mesh_import();
                                        }
                                        Err(error) => ui.set_mesh_import_error(error),
                                    }
                                }
                                }
                            }
                            Some(UiAction::CancelMeshImport) => {
                                if mesh_import_job.is_some() {
                                    // The current voxelizer has no interruption checkpoints.
                                    // Keep its receiver alive and make the limitation visible
                                    // instead of silently detaching a CPU-heavy worker.
                                    ui.mesh_import_open = true;
                                    ui.mesh_import_busy = true;
                                    ui.mesh_import_error = Some(
                                        "Voxelization cannot be interrupted yet. Wait for the preview, then cancel safely.".into(),
                                    );
                                    ui.set_status_message(
                                        "The mesh import is still running; cancellation is unavailable until voxelization finishes.",
                                    );
                                } else {
                                    mesh_import_preview = None;
                                    mesh_import_preview_scene = None;
                                    mesh_import_source_path = None;
                                    if let Some(snapshot) = mesh_import_camera_backup.take() {
                                        snapshot.restore(&mut camera);
                                    }
                                    if let Some(was_paused) = mesh_import_previous_paused.take() {
                                        ui.paused = was_paused;
                                    }
                                    ui.finish_mesh_import();
                                    ui.set_status_message("Mesh import cancelled; the authored scene was not changed.");
                                    accumulator = 0.0;
                                    last = Instant::now();
                                }
                            }
                            Some(UiAction::SelectSceneModel(index)) => {
                                let scene_path = active_scene_document_path
                                    .as_deref()
                                    .unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                if let Some(document) = active_scene_document.as_ref() {
                                    sync_scene_instance_selection(
                                        &mut ui,
                                        document,
                                        scene_path,
                                        Some(index),
                                    );
                                }
                                if let Some(instance) = active_scene_document.as_ref().and_then(|doc| doc.models.get(index)) {
                                    ui.set_status_message(format!(
                                        "Selected model instance: {}",
                                        instance
                                            .asset_path
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .unwrap_or("Voxel model"),
                                    ));
                                }
                            }
                            Some(UiAction::SetSceneMoveStep(step)) => {
                                ui.set_status_message(format!(
                                    "Move step: {} cm.",
                                    step
                                ));
                            }
                            Some(UiAction::MoveSelectedSceneModel { axis, direction }) => {
                                if let Some(axis) = [IVec3::X, IVec3::Y, IVec3::Z].get(axis).copied() {
                                    let delta = axis * i32::from(direction) * ui.scene_move_step_cells;
                                    run_scene_instance_edit(
                                        &mut scene,
                                        &mut active_scene_document,
                                        active_scene_document_path.as_deref(),
                                        &mut ui,
                                        &mut live_simulated_s,
                                        &mut live_wall_s,
                                        &mut accumulator,
                                        &mut last,
                                        SceneInstanceEdit::Translate(delta),
                                    );
                                }
                            }
                            Some(UiAction::RotateSelectedSceneModel { axis, direction }) => {
                                if let Some(axis) = [IVec3::X, IVec3::Y, IVec3::Z].get(axis).copied() {
                                    run_scene_instance_edit(
                                        &mut scene,
                                        &mut active_scene_document,
                                        active_scene_document_path.as_deref(),
                                        &mut ui,
                                        &mut live_simulated_s,
                                        &mut live_wall_s,
                                        &mut accumulator,
                                        &mut last,
                                        SceneInstanceEdit::RotateQuarterTurn { axis, direction },
                                    );
                                }
                            }
                            Some(UiAction::DuplicateSelectedSceneModel) => {
                                run_scene_instance_edit(
                                    &mut scene,
                                    &mut active_scene_document,
                                    active_scene_document_path.as_deref(),
                                    &mut ui,
                                    &mut live_simulated_s,
                                    &mut live_wall_s,
                                    &mut accumulator,
                                    &mut last,
                                    SceneInstanceEdit::Duplicate,
                                );
                            }
                            Some(UiAction::RemoveSelectedSceneModel) => {
                                if run_scene_instance_edit(
                                    &mut scene,
                                    &mut active_scene_document,
                                    active_scene_document_path.as_deref(),
                                    &mut ui,
                                    &mut live_simulated_s,
                                    &mut live_wall_s,
                                    &mut accumulator,
                                    &mut last,
                                    SceneInstanceEdit::Remove,
                                ) {
                                    active_model_document = None;
                                    model_revision = model_revision.wrapping_add(1);
                                    active_model_document_path = None;
                                    tool = None;
                                }
                            }
                            Some(UiAction::SetEditorTool(editor_tool)) => {
                                ui.editor_tool = editor_tool;
                            }
                            Some(UiAction::SelectBone(index)) => {
                                if let Some(document) = active_model_document.as_ref() {
                                    let selected = document
                                        .skeleton
                                        .as_ref()
                                        .filter(|skeleton| index < skeleton.bones.len())
                                        .map(|_| index);
                                    ui.set_rig_authoring_state(
                                        document.skeleton.as_ref().map_or_else(Vec::new, |s| {
                                            s.bones.iter().map(|bone| bone.name.clone()).collect()
                                        }),
                                        selected,
                                        document.skeleton.as_ref().map_or(0, |s| s.clips.len()),
                                    );
                                }
                            }
                            Some(UiAction::AdjustSelectedVoxelBoneWeight { bone_index, delta }) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before editing bone weights.");
                                } else if ui.selected_voxel_indices.len() != 1 {
                                    ui.set_status_message("Select exactly one REV to paint its bone weight.");
                                } else if let Some(document) = active_model_document.as_mut() {
                                    let voxel_index = ui.selected_voxel_indices[0];
                                    let result = document
                                        .skeleton
                                        .as_mut()
                                        .ok_or_else(|| "create a skeleton before editing weights".to_owned())
                                        .and_then(|skeleton| {
                                            skeleton.adjust_voxel_bone_weight(
                                                document.voxels.len(),
                                                voxel_index,
                                                bone_index,
                                                delta,
                                            )
                                        });
                                    match result {
                                        Ok(weight) => {
                                            model_revision = model_revision.wrapping_add(1);
                                            skinning_weights_revision = skinning_weights_revision.wrapping_add(1);
                                            ui.set_status_message(format!(
                                                "REV {} · {} weight {:.2}",
                                                voxel_index + 1,
                                                ui.rig_bone_names
                                                    .get(bone_index)
                                                    .map_or("bone", String::as_str),
                                                weight,
                                            ));
                                        }
                                        Err(error) => ui.set_status_message(format!(
                                            "Could not edit bone weight: {error}"
                                        )),
                                    }
                                }
                            }
                            Some(UiAction::CreateRootBone) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before editing its skeleton.");
                                } else if let Some(document) = active_model_document.as_mut() {
                                    if document.skeleton.is_some() {
                                        ui.set_status_message("This model already has a root bone.");
                                    } else if document.voxels.is_empty() {
                                        ui.set_status_message("Add at least one REV before creating a skeleton.");
                                    } else {
                                        let mut min = DVec3::splat(f64::INFINITY);
                                        let mut max = DVec3::splat(f64::NEG_INFINITY);
                                        for voxel in &document.voxels {
                                            min = min.min(voxel.rest_position_m);
                                            max = max.max(voxel.rest_position_m);
                                        }
                                        document.skeleton = Some(SkeletonDocument::with_root_bone(
                                            "Root",
                                            DVec3::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5, min.z),
                                        ));
                                        model_revision = model_revision.wrapping_add(1);
                                        ui.selected_bone = Some(0);
                                        update_model_ui(&mut ui, document, None);
                                        ui.set_status_message("Created root bone at the model's lower centre.");
                                    }
                                }
                            }
                            Some(UiAction::CreateChildBone { parent_bone }) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before editing its skeleton.");
                                } else if let Some(document) = active_model_document.as_mut() {
                                    let result = document.skeleton.as_mut().ok_or_else(|| "create a root bone first".to_owned()).and_then(|skeleton| {
                                        if parent_bone >= skeleton.bones.len() {
                                            return Err("selected parent bone no longer exists".to_owned());
                                        }
                                        let rest_position = DVec3::new(0.0, 0.0, (document.cell_size_m * 5.0).max(0.05));
                                        skeleton.add_bone(format!("Bone {}", skeleton.bones.len() + 1), Some(parent_bone), rest_position)
                                    });
                                    match result {
                                        Ok(index) => {
                                            model_revision = model_revision.wrapping_add(1);
                                            ui.selected_bone = Some(index);
                                            update_model_ui(&mut ui, document, None);
                                            ui.set_status_message("Created child bone 5 cm above its parent in parent-local space.");
                                        }
                                        Err(error) => ui.set_status_message(format!("Could not create bone: {error}")),
                                    }
                                }
                            }
                            Some(UiAction::GenerateAutomaticWeights) => {
                                if auto_weight_job.is_some() {
                                    ui.set_status_message("Automatic weights are already being calculated.");
                                } else if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before binding bone weights.");
                                } else if let Some(document) = active_model_document.as_ref() {
                                    if let Some(skeleton) = document.skeleton.clone() {
                                        let revision = model_revision;
                                        let voxel_count = document.voxels.len();
                                        let cell_size_m = document.cell_size_m;
                                        // Only the compact rest-position array and the small rig schema
                                        // cross to the worker; REV state/materials are never cloned.
                                        let rest_positions = document.voxels.iter().map(|voxel| voxel.rest_position_m).collect::<Vec<_>>();
                                        let (sender, receiver) = mpsc::channel();
                                        let spawn = std::thread::Builder::new()
                                            .name("voxel-auto-weights".into())
                                            .spawn(move || {
                                                // Keep this Rayon-backed calculation off the
                                                // global pool used by interactive rendering work.
                                                let result = rayon::ThreadPoolBuilder::new()
                                                    .num_threads(1)
                                                    .thread_name(|_| "voxel-weight-cpu".into())
                                                    .build()
                                                    .map_err(|error| error.to_string())
                                                    .and_then(|pool| {
                                                        pool.install(|| {
                                                            generate_auto_bone_weights_from_rest_positions(
                                                                &rest_positions,
                                                                cell_size_m,
                                                                &skeleton,
                                                                4,
                                                            )
                                                        })
                                                    });
                                                let _ = sender.send((revision, voxel_count, cell_size_m, result));
                                            });
                                        match spawn {
                                            Ok(_) => {
                                                auto_weight_job = Some(receiver);
                                                ui.set_status_message(format!("Calculating automatic weights for {voxel_count} REVs in background…"));
                                            }
                                            Err(error) => ui.set_status_message(format!("Could not start weight worker: {error}")),
                                        }
                                    } else {
                                        ui.set_status_message("Create at least one bone before generating weights.");
                                    }
                                }
                            }
                            Some(UiAction::CreateAnimationClip) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before editing its animation.");
                                } else if let Some(document) = active_model_document.as_mut() {
                                    let result = document.skeleton.as_mut().ok_or_else(|| "create bones before adding an animation clip".to_owned()).and_then(|skeleton| {
                                        let name = format!("Animation {}", skeleton.clips.len() + 1);
                                        skeleton.add_animation_clip(name, 1.0)
                                    });
                                    match result {
                                        Ok(index) => {
                                            model_revision = model_revision.wrapping_add(1);
                                            update_model_ui(&mut ui, document, None);
                                            ui.set_animation_timeline_state(
                                                Some(index),
                                                0.0,
                                                ui.animation_time_step_seconds,
                                            );
                                            update_animation_timeline_ui(&mut ui, document);
                                            ui.set_status_message(format!("Created 1-second animation clip {}.", index + 1));
                                        }
                                        Err(error) => ui.set_status_message(format!("Could not create clip: {error}")),
                                    }
                                }
                            }
                            Some(UiAction::SelectAnimationClip(index)) => {
                                if let Some(document) = active_model_document.as_ref() {
                                    let valid = document
                                        .skeleton
                                        .as_ref()
                                        .is_some_and(|skeleton| index < skeleton.clips.len());
                                    if valid {
                                        ui.set_animation_timeline_state(
                                            Some(index),
                                            0.0,
                                            ui.animation_time_step_seconds,
                                        );
                                        if let Some(document) = active_model_document.as_ref() {
                                            update_animation_timeline_ui(&mut ui, document);
                                        }
                                        ui.set_status_message(format!("Selected animation clip {} at 0.00 s.", index + 1));
                                    } else {
                                        ui.set_status_message("That animation clip no longer exists.");
                                    }
                                }
                            }
                            Some(UiAction::ToggleAnimationPlayback) => {
                                let playing = ui.toggle_animation_playback();
                                ui.set_status_message(if playing {
                                    "Animation playback started; click Pause to stop."
                                } else {
                                    "Animation playback paused."
                                });
                            }
                            Some(UiAction::SetAnimationFrame(frame)) => {
                                if let (Some(document), Some(clip_index)) = (
                                    active_model_document.as_ref(),
                                    ui.selected_animation_clip,
                                ) {
                                    if let Some(duration_s) = document
                                        .skeleton
                                        .as_ref()
                                        .and_then(|skeleton| skeleton.clips.get(clip_index))
                                        .map(|clip| clip.duration_s)
                                    {
                                        let time_s = (frame as f32
                                            / ui.animation_frame_rate.max(1.0))
                                            .min(duration_s as f32);
                                        ui.set_animation_timeline_state(
                                            Some(clip_index),
                                            time_s,
                                            ui.animation_time_step_seconds,
                                        );
                                    }
                                }
                            }
                            Some(UiAction::StepAnimationTime { .. }) => {
                                if let (Some(document), Some(clip_index)) = (
                                    active_model_document.as_ref(),
                                    ui.selected_animation_clip,
                                ) {
                                    if let Some(duration_s) = document
                                        .skeleton
                                        .as_ref()
                                        .and_then(|skeleton| skeleton.clips.get(clip_index))
                                        .map(|clip| clip.duration_s)
                                    {
                                        let time_s = f64::from(ui.selected_animation_time_seconds)
                                            .clamp(0.0, duration_s);
                                        ui.set_animation_timeline_state(
                                            Some(clip_index),
                                            time_s as f32,
                                            ui.animation_time_step_seconds,
                                        );
                                    }
                                }
                            }
                            Some(UiAction::SetRigEditMode(_)) => {}
                            Some(UiAction::AdjustSelectedBone {
                                bone_index,
                                axis,
                                delta,
                                rotation,
                            }) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before editing a bone pose.");
                                } else if let (Some(document), Some(clip_index)) = (
                                    active_model_document.as_mut(),
                                    ui.selected_animation_clip,
                                ) {
                                    let time_s = f64::from(ui.selected_animation_time_seconds);
                                    let result = document
                                        .skeleton
                                        .as_mut()
                                        .ok_or_else(|| "this model has no skeleton".to_owned())
                                        .and_then(|skeleton| {
                                            let mut pose = skeleton.sample_bone_local_pose(
                                                clip_index,
                                                bone_index,
                                                time_s,
                                            )?;
                                            if axis >= 3 || !delta.is_finite() {
                                                return Err("bone edit has an invalid axis or amount".into());
                                            }
                                            if rotation {
                                                let axis = match axis {
                                                    0 => Vec3::X,
                                                    1 => Vec3::Y,
                                                    _ => Vec3::Z,
                                                };
                                                pose.rotation = (pose.rotation
                                                    * Quat::from_axis_angle(axis, delta.to_radians()))
                                                .normalize();
                                            } else {
                                                let mut translation = pose.translation_m.to_array();
                                                translation[axis] += f64::from(delta);
                                                pose.translation_m = DVec3::from_array(translation);
                                            }
                                            skeleton.set_keyframe(clip_index, bone_index, pose)?;
                                            Ok(pose)
                                        });
                                    match result {
                                        Ok(_) => {
                                            model_revision = model_revision.wrapping_add(1);
                                            ui.set_status_message(format!(
                                                "Updated {} pose at {:.2} s; keyframe stored.",
                                                ui.rig_bone_names.get(bone_index).map_or("bone", String::as_str),
                                                ui.selected_animation_time_seconds,
                                            ));
                                        }
                                        Err(error) => ui.set_status_message(format!("Could not edit bone pose: {error}")),
                                    }
                                }
                            }
                            Some(UiAction::InsertBoneKeyframe { bone_index }) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message("Pause and reset the model before editing its animation.");
                                } else if let (Some(document), Some(clip_index)) = (
                                    active_model_document.as_mut(),
                                    ui.selected_animation_clip,
                                ) {
                                    let time_s = f64::from(ui.selected_animation_time_seconds);
                                    let result = document
                                        .skeleton
                                        .as_mut()
                                        .ok_or_else(|| "this model has no skeleton".to_owned())
                                        .and_then(|skeleton| {
                                            let pose = skeleton.sample_bone_local_pose(
                                                clip_index,
                                                bone_index,
                                                time_s,
                                            )?;
                                            skeleton.set_keyframe(clip_index, bone_index, pose)?;
                                            Ok(pose)
                                        });
                                    match result {
                                        Ok(_) => {
                                            model_revision = model_revision.wrapping_add(1);
                                            ui.set_status_message(format!(
                                                "Keyed {} at {:.2} s.",
                                                ui.rig_bone_names.get(bone_index).map_or("bone", String::as_str),
                                                ui.selected_animation_time_seconds,
                                            ));
                                        }
                                        Err(error) => ui.set_status_message(format!("Could not insert keyframe: {error}")),
                                    }
                                }
                            }
                            Some(UiAction::AdjustBrushSize(change)) => {
                                let next = (ui.brush_size_cells as i16 + i16::from(change))
                                    .clamp(1, 31);
                                ui.brush_size_cells = if next % 2 == 0 {
                                    (next + i16::from(change.signum())).clamp(1, 31) as usize
                                } else {
                                    next as usize
                                };
                            }
                            Some(action @ UiAction::MoveSelectedPart { .. })
                            | Some(action @ UiAction::RotateSelectedPart { .. })
                            | Some(action @ UiAction::DuplicateSelectedPart)
                            | Some(action @ UiAction::DeleteSelectedPart) => {
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message(
                                        "Pause and reset the model before changing voxel geometry.",
                                    );
                                } else if let Some(document) = active_model_document.as_mut() {
                                    let selected = ui.selected_voxel_indices.clone();
                                    let result = (|| {
                                        let (edit, next_selection, message) = match action {
                                            UiAction::MoveSelectedPart { axis, direction } => (
                                                selected_part_move_edit(document, &selected, axis, direction)?,
                                                selected.clone(),
                                                "Moved the selected part by 1 cm. Ctrl+Z undoes this edit.",
                                            ),
                                            UiAction::RotateSelectedPart { direction } => (
                                                selected_part_rotate_z_edit(document, &selected, direction)?,
                                                selected.clone(),
                                                "Rotated the selected part 90° around Z. Ctrl+Z undoes this edit.",
                                            ),
                                            UiAction::DuplicateSelectedPart => {
                                                let (edit, indices) = duplicate_selected_part_edit(document, &selected)?;
                                                (edit, indices, "Duplicated the selected part. Ctrl+Z undoes this edit.")
                                            }
                                            UiAction::DeleteSelectedPart => (
                                                delete_selected_part_edit(document, &selected)?,
                                                Vec::new(),
                                                "Deleted the selected part. Ctrl+Z restores it.",
                                            ),
                                            _ => unreachable!(),
                                        };
                                        let rebuilt = commit_model_edit(
                                            document,
                                            &mut model_edit_history,
                                            &config,
                                            edit,
                                        )?;
                                        Ok::<_, String>((rebuilt, next_selection, message))
                                    })();
                                    match result {
                                        Ok((rebuilt, selection, message)) => {
                                            scene = rebuilt;
                                            model_revision = model_revision.wrapping_add(1);
                                            let weights_invalidated = active_model_document
                                                .as_mut()
                                                .is_some_and(invalidate_skeleton_weights_after_geometry_edit);
                                            if let Some(document) = active_model_document.as_ref() {
                                                update_model_ui_selection(&mut ui, document, selection);
                                            }
                                            sync_model_history_ui(&mut ui, &model_edit_history);
                                            ui.set_status_message(message);
                                            if weights_invalidated {
                                                ui.set_status_message("Model edited. Bone weights were cleared because voxel geometry changed; regenerate automatic weights.");
                                            }
                                            accumulator = 0.0;
                                            last = Instant::now();
                                        }
                                        Err(error) => {
                                            ui.set_status_message(format!("Error: {error}"));
                                        }
                                    }
                                }
                            }
                            Some(action @ UiAction::UndoVoxelModel)
                            | Some(action @ UiAction::RedoVoxelModel) => {
                                let redo = matches!(action, UiAction::RedoVoxelModel);
                                if !ui.paused || live_simulated_s > 1e-9 {
                                    ui.set_status_message(
                                        "Pause and reset the model before undoing or redoing edits.",
                                    );
                                } else if let Some(document) = active_model_document.as_mut() {
                                    match replay_model_history(
                                        document,
                                        &mut model_edit_history,
                                        &config,
                                        redo,
                                    ) {
                                        Ok(Some(rebuilt)) => {
                                            scene = rebuilt;
                                            model_revision = model_revision.wrapping_add(1);
                                            let weights_invalidated =
                                                invalidate_skeleton_weights_after_geometry_edit(document);
                                            update_model_ui(&mut ui, document, None);
                                            sync_model_history_ui(&mut ui, &model_edit_history);
                                            ui.set_status_message(if redo {
                                                "Voxel edit reapplied."
                                            } else {
                                                "Voxel edit undone."
                                            });
                                            if weights_invalidated {
                                                ui.set_status_message("Model history applied. Bone weights were cleared because voxel geometry changed; regenerate automatic weights.");
                                            }
                                            accumulator = 0.0;
                                            last = Instant::now();
                                        }
                                        Ok(None) => ui.set_status_message(if redo {
                                            "Nothing to redo."
                                        } else {
                                            "Nothing to undo."
                                        }),
                                        Err(error) => {
                                            ui.set_status_message(format!("Error: {error}"));
                                        }
                                    }
                                }
                            }
                            Some(UiAction::FrameSelection) => {
                                if let (Some(document), Some(index)) =
                                    (active_scene_document.as_ref(), ui.selected_scene_model)
                                {
                                    let scene_path = active_scene_document_path
                                        .as_deref()
                                        .unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                    match authored_instance_bounds(document, scene_path, index) {
                                        Ok(bounds) => {
                                            let aspect = ui
                                                .scene_viewport()
                                                .map(|[_, _, width, height]| {
                                                    width as f32 / height.max(1) as f32
                                                })
                                                .unwrap_or(1.0);
                                            camera.set_orbit(
                                                bounds.center,
                                                Camera::fit_orbit_distance_for_sphere(
                                                    bounds.sphere_radius,
                                                    aspect,
                                                ),
                                            );
                                            camera.yaw = -0.85;
                                            camera.pitch = 0.38;
                                        }
                                        Err(error) => {
                                            ui.set_status_message(format!("Error: {error}"));
                                        }
                                    }
                                } else if active_model_document.is_some() {
                                    if let Some(bounds) = ui.selected_bounds_m {
                                    let minimum = Vec3::from_array(bounds[0].map(|value| value as f32));
                                    let maximum = Vec3::from_array(bounds[1].map(|value| value as f32));
                                    let center = (minimum + maximum) * 0.5;
                                    let radius = (maximum - minimum).length().max(0.02) * 1.4;
                                    camera.set_orbit(center, radius);
                                    camera.yaw = -0.85;
                                    camera.pitch = 0.38;
                                    } else {
                                        frame_camera_on_scene(&scene, &mut camera);
                                    }
                                } else {
                                    frame_camera_on_scene(&scene, &mut camera);
                                }
                            }
                            Some(UiAction::GenerateWorld(seed)) => {
                                grabbed_cell = None;
                                world_physics = None;
                                let generated = GeneratedWorld::from_seed(&seed);
                                eprintln!(
                                    "generated deterministic world: seed={seed:?}, area=10 km2, voxels={}, {}, simulation speed=1x",
                                    generated.voxels.len(),
                                    generated.vegetation_summary(),
                                );
                                generated_world = Some(generated);
                                active_scene_document = None;
                                active_scene_document_path = None;
                                active_model_document = None;
                                model_revision = model_revision.wrapping_add(1);
                                active_model_document_path = None;
                                ui.set_active_scene_document(None, Vec::new());
                                growth_scheduler = GrowthScheduler::default();
                                world_character = generated_world.as_ref().map(WorldCharacter::spawn);
                                ui.world_active = true;
                                ui.editor_mode = true;
                                ui.paused = false;
                                ui.show_bonds = false;
                                tool = None;
                                mode = AppMode::Interactive;
                                playback_frames.clear();
                                playback_time_s = 0.0;
                                accumulator = 0.0;
                                camera.disable_freecam();
                                if let Some(character) = world_character.as_ref() {
                                    set_editor_camera(character, &mut camera, 12.0);
                                    orbit = false;
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                }
                                last = Instant::now();
                            }
                            Some(UiAction::SelectSettingsTab(tab)) => {
                                ui.settings_tab = tab;
                            }
                            Some(UiAction::BeginRebind(action)) => {
                                ui.settings_rebind_action = Some(action);
                                ui.set_status_message("Press a key combination to assign it; Esc cancels.");
                            }
                            Some(UiAction::ToggleMcp(enabled)) => {
                                let previous = editor_settings.mcp_enabled;
                                editor_settings.mcp_enabled = enabled;
                                match editor_settings::save(&editor_settings) {
                                    Ok(()) => {
                                        runtime_metrics_server = None;
                                        if enabled {
                                            match voxel_physics_lab::runtime_metrics::start_server(&runtime_project_root) {
                                                Ok(server) => {
                                                    runtime_metrics_server = Some(server);
                                                    last_authoring_metrics_key = None;
                                                    ui.set_status_message("MCP enabled; live metrics are available immediately.");
                                                }
                                                Err(error) => ui.set_status_message(format!("Saved-file MCP enabled; live metrics could not start: {error}")),
                                            }
                                        } else {
                                            ui.set_status_message("MCP disabled; the live metrics endpoint is closed.");
                                        }
                                    }
                                    Err(error) => {
                                        editor_settings.mcp_enabled = previous;
                                        ui.settings_mcp_enabled = previous;
                                        ui.set_status_message(format!("Could not save settings: {error}"));
                                    }
                                }
                            }
                            Some(UiAction::Resume | UiAction::Back | UiAction::Settings) => {
                                accumulator = 0.0;
                                last = Instant::now();
                                if !ui.menu_open && camera.is_first_person() {
                                    orbit = true;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                    ignore_mouse_motion_until = Instant::now() + Duration::from_millis(150);
                                }
                            }
                            Some(UiAction::EnterPlay) => {
                                if active_model_document
                                    .as_ref()
                                    .is_some_and(model_has_unsupported_runtime_features)
                                {
                                    ui.set_status_message("Play is unavailable for this model: skeletons, voxel groups, and motion constraints are authored but their runtime solver is not implemented yet. Static model preview and saving remain available.");
                                } else {
                                grabbed_cell = None;
                                if let Some(physics) = world_physics.as_ref() {
                                    physics.set_grab_target(None);
                                }
                                if let Some(character) = world_character.as_ref() {
                                    set_game_camera(character, &mut camera, world_physics.is_some());
                                    orbit = true;
                                    ui.paused = false;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                    ignore_mouse_motion_until = Instant::now() + Duration::from_millis(150);
                                    last = Instant::now();
                                } else {
                                    camera.enable_freecam();
                                    orbit = false;
                                    ui.paused = false;
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                    last = Instant::now();
                                }
                                }
                            }
                            Some(UiAction::ReturnToEditor) => {
                                if active_scene_document.is_some() { ui.paused = true; }
                                grabbed_cell = None;
                                if let Some(physics) = world_physics.as_ref() {
                                    physics.set_grab_target(None);
                                }
                                ui.close_menu();
                                ui.editor_mode = true;
                                if let Some(character) = world_character.as_ref() {
                                    set_editor_camera(character, &mut camera, 8.0);
                                }
                                orbit = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                                last = Instant::now();
                            }
                            Some(UiAction::ToggleSimulationPause) => {
                                if ui.paused
                                    && active_model_document
                                        .as_ref()
                                        .is_some_and(model_has_unsupported_runtime_features)
                                {
                                    ui.set_status_message("Simulation is unavailable for this model: skeletons, voxel groups, and motion constraints do not have a runtime solver yet.");
                                } else {
                                    ui.paused = !ui.paused;
                                    accumulator = 0.0;
                                    last = Instant::now();
                                }
                            }
                            Some(UiAction::ToggleBonds) => {
                                ui.show_bonds = !ui.show_bonds;
                            }
                            Some(UiAction::Exit) => target.exit(),
                            None => {}
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                    };
                    if !ui.scroll_scene_hierarchy(amount, pointer.0, pointer.1) {
                        if camera.is_freecam() {
                            camera.adjust_free_speed(amount);
                        } else if let Some(grabbed) = grabbed_cell.as_mut() {
                            grabbed.distance_m = (grabbed.distance_m + amount * 0.25)
                                .clamp(0.25, 4.0);
                        } else if !camera.is_first_person() {
                            let min_radius = if ui.standalone_editor_app { 0.015 } else { 0.3 };
                            camera.radius = (camera.radius * (-amount * 0.03).exp())
                                .clamp(min_radius, WORLD_SIDE_M * 1.5);
                        }
                    }
                    hovered_face = hovered_face_at_pointer(&ui, &scene, &camera, pointer);
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat
                        && event.physical_key == PhysicalKey::Code(KeyCode::F11) =>
                {
                    window.set_fullscreen(if window.fullscreen().is_some() { None } else {
                        Some(Fullscreen::Borderless(window.current_monitor()))
                    });
                }
                WindowEvent::KeyboardInput { event, .. }
                    if ui.scene_input_active() =>
                {
                    keys_down.clear();
                    if event.state == ElementState::Pressed {
                        if let PhysicalKey::Code(key) = event.physical_key {
                            if let Some(action) = ui.scene_input_key(key) { pending_scene_action = Some(action); }
                            if !matches!(key, KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Escape | KeyCode::Tab | KeyCode::Backspace) {
                                if let Some(text) = event.text.as_ref() { for character in text.chars() { ui.scene_input_char(character); } }
                            }
                        }
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Released =>
                {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        keys_down.remove(&key);
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed
                        && !event.repeat
                        && ui.settings_rebind_action.is_some()
                        && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
                {
                    ui.settings_rebind_action = None;
                    ui.set_status_message("Key binding cancelled.");
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed
                        && !event.repeat
                        && ui.help_open
                        && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
                {
                    ui.help_open = false;
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed
                        && !event.repeat
                        && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
                {
                    ui.toggle_pause_menu();
                    keys_down.clear();
                    jump_queued = false;
                    grabbed_cell = None;
                    if let Some(physics) = world_physics.as_ref() {
                        physics.set_grab_target(None);
                    }
                    orbit = false;
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                    if !ui.menu_open && camera.is_first_person() {
                        orbit = true;
                        let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                        window.set_cursor_visible(false);
                    }
                    accumulator = 0.0;
                    last = Instant::now();
                }
                WindowEvent::KeyboardInput { event, .. }
                    if ui.menu_open && event.state == ElementState::Pressed =>
                {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        if matches!(key, KeyCode::ControlLeft | KeyCode::ControlRight) {
                            keys_down.insert(key);
                        } else if let Some(action) = ui.settings_rebind_action.take() {
                            let control_down = keys_down.contains(&KeyCode::ControlLeft)
                                || keys_down.contains(&KeyCode::ControlRight);
                            let key_name = match format!("{key:?}").strip_prefix("Key") {
                                Some(letter) => letter.to_owned(),
                                None => format!("{key:?}"),
                            };
                            let binding = if control_down { format!("Ctrl+{key_name}") } else { key_name };
                            match editor_settings::set_binding(&mut editor_settings, &action, binding.clone()) {
                                Ok(()) => match editor_settings::save(&editor_settings) {
                                    Ok(()) => {
                                        ui.settings_bindings = editor_settings.bindings.clone();
                                        ui.set_status_message(format!("{} is now bound to {binding}.", editor_settings::label(&action)));
                                    }
                                    Err(error) => {
                                        ui.settings_rebind_action = Some(action);
                                        ui.set_status_message(format!("Could not save settings: {error}"));
                                    }
                                },
                                Err(error) => {
                                    ui.settings_rebind_action = Some(action);
                                    ui.set_status_message(error);
                                }
                            }
                        }
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat =>
                {
                    if ui.standalone_editor_app
                        && matches!(
                            event.physical_key,
                            PhysicalKey::Code(
                                KeyCode::Space
                                    | KeyCode::KeyP
                                    | KeyCode::KeyB
                                    | KeyCode::F6
                            )
                        )
                    {
                        return;
                    }
                    if let PhysicalKey::Code(key) = event.physical_key {
                        keys_down.insert(key);
                        if ui.world_active
                            && ui.editor_mode
                            && !ui.menu_open
                            && !ui.seed_editing
                            && !keys_down.contains(&KeyCode::ControlLeft)
                            && !keys_down.contains(&KeyCode::ControlRight)
                            && !camera.is_freecam()
                            && !camera.is_first_person()
                            && matches!(
                                key,
                                KeyCode::KeyW
                                    | KeyCode::KeyA
                                    | KeyCode::KeyS
                                    | KeyCode::KeyD
                            )
                        {
                            camera.enable_freecam();
                            orbit = false;
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    }
                    let editing_seed = ui.seed_editing;
                    if editing_seed {
                        if event.physical_key == PhysicalKey::Code(KeyCode::Backspace) {
                            ui.erase_seed_character();
                        } else if let Some(text) = event.text.as_ref() {
                            ui.append_seed_text(text.as_str());
                        }
                    }
                    let control_down = keys_down.contains(&KeyCode::ControlLeft)
                        || keys_down.contains(&KeyCode::ControlRight);
                    let event_key_debug = match event.physical_key {
                        PhysicalKey::Code(key) => format!("{key:?}"),
                        _ => String::new(),
                    };
                    let binding_pressed = |action| editor_settings::is_pressed(
                        &editor_settings, action, &event_key_debug, control_down,
                    );
                    let model_history_redo = if binding_pressed("redo") { Some(true) }
                        else if binding_pressed("undo") { Some(false) } else { None }
                    .filter(|_| {
                        ui.editor_mode
                            && !ui.menu_open
                            && !editing_seed
                            && active_model_document.is_some()
                    });
                    if let Some(redo) = model_history_redo {
                        if !ui.paused || live_simulated_s > 1e-9 {
                            ui.set_status_message(
                                "Pause and reset the model before undoing or redoing edits.",
                            );
                        } else if let Some(document) = active_model_document.as_mut() {
                            match replay_model_history(
                                document,
                                &mut model_edit_history,
                                &config,
                                redo,
                            ) {
                                Ok(Some(rebuilt)) => {
                                    scene = rebuilt;
                                    model_revision = model_revision.wrapping_add(1);
                                    let weights_invalidated =
                                        invalidate_skeleton_weights_after_geometry_edit(document);
                                    update_model_ui(&mut ui, document, None);
                                    sync_model_history_ui(&mut ui, &model_edit_history);
                                    ui.set_status_message(if redo {
                                        "Voxel edit reapplied."
                                    } else {
                                        "Voxel edit undone."
                                    });
                                    if weights_invalidated {
                                        ui.set_status_message("Model history applied. Bone weights were cleared because voxel geometry changed; regenerate automatic weights.");
                                    }
                                    accumulator = 0.0;
                                    last = Instant::now();
                                }
                                Ok(None) => ui.set_status_message(if redo {
                                    "Nothing to redo."
                                } else {
                                    "Nothing to undo."
                                }),
                                Err(error) => {
                                    ui.set_status_message(format!("Error: {error}"));
                                }
                            }
                        }
                    }
                    let instance_edit = if ui.editor_mode && !ui.menu_open && !editing_seed {
                        let step = ui.scene_move_step_cells;
                        if binding_pressed("move_x_neg") { Some(SceneInstanceEdit::Translate(IVec3::new(-step, 0, 0))) }
                        else if binding_pressed("move_x_pos") { Some(SceneInstanceEdit::Translate(IVec3::new(step, 0, 0))) }
                        else if binding_pressed("move_y_pos") { Some(SceneInstanceEdit::Translate(IVec3::new(0, step, 0))) }
                        else if binding_pressed("move_y_neg") { Some(SceneInstanceEdit::Translate(IVec3::new(0, -step, 0))) }
                        else if binding_pressed("move_z_pos") { Some(SceneInstanceEdit::Translate(IVec3::new(0, 0, step))) }
                        else if binding_pressed("move_z_neg") { Some(SceneInstanceEdit::Translate(IVec3::new(0, 0, -step))) }
                        else if binding_pressed("rotate_z_pos") { Some(SceneInstanceEdit::RotateQuarterTurn { axis: IVec3::Z, direction: 1 }) }
                        else if binding_pressed("rotate_z_neg") { Some(SceneInstanceEdit::RotateQuarterTurn { axis: IVec3::Z, direction: -1 }) }
                        else if binding_pressed("delete") { Some(SceneInstanceEdit::Remove) }
                        else if binding_pressed("duplicate") { Some(SceneInstanceEdit::Duplicate) }
                        else { None }
                    } else {
                        None
                    };
                    if active_scene_document.is_some() {
                        if let Some(edit) = instance_edit {
                            run_scene_instance_edit(
                                &mut scene,
                                &mut active_scene_document,
                                active_scene_document_path.as_deref(),
                                &mut ui,
                                &mut live_simulated_s,
                                &mut live_wall_s,
                                &mut accumulator,
                                &mut last,
                                edit,
                            );
                        }
                    }
                    let model_part_action = if ui.editor_mode && !ui.menu_open && !editing_seed
                        && active_model_document.is_some() && !ui.selected_voxel_indices.is_empty()
                    {
                        if binding_pressed("move_x_neg") { Some(UiAction::MoveSelectedPart { axis: 0, direction: -1 }) }
                        else if binding_pressed("move_x_pos") { Some(UiAction::MoveSelectedPart { axis: 0, direction: 1 }) }
                        else if binding_pressed("move_y_neg") { Some(UiAction::MoveSelectedPart { axis: 1, direction: -1 }) }
                        else if binding_pressed("move_y_pos") { Some(UiAction::MoveSelectedPart { axis: 1, direction: 1 }) }
                        else if binding_pressed("move_z_neg") { Some(UiAction::MoveSelectedPart { axis: 2, direction: -1 }) }
                        else if binding_pressed("move_z_pos") { Some(UiAction::MoveSelectedPart { axis: 2, direction: 1 }) }
                        else if binding_pressed("rotate_z_neg") { Some(UiAction::RotateSelectedPart { direction: -1 }) }
                        else if binding_pressed("rotate_z_pos") { Some(UiAction::RotateSelectedPart { direction: 1 }) }
                        else if binding_pressed("duplicate") { Some(UiAction::DuplicateSelectedPart) }
                        else if binding_pressed("delete") { Some(UiAction::DeleteSelectedPart) }
                        else { None }
                    } else { None };
                    if let Some(action) = model_part_action {
                        if !ui.paused || live_simulated_s > 1e-9 {
                            ui.set_status_message("Pause and reset the model before changing voxel geometry.");
                        } else if let Some(document) = active_model_document.as_mut() {
                            let selected = ui.selected_voxel_indices.clone();
                            let result = (|| {
                                let (edit, next_selection, message) = match action {
                                    UiAction::MoveSelectedPart { axis, direction } => (selected_part_move_edit(document, &selected, axis, direction)?, selected.clone(), "Moved selected part. Ctrl+Z undoes this edit."),
                                    UiAction::RotateSelectedPart { direction } => (selected_part_rotate_z_edit(document, &selected, direction)?, selected.clone(), "Rotated selected part around Z. Ctrl+Z undoes this edit."),
                                    UiAction::DuplicateSelectedPart => { let (edit, indices) = duplicate_selected_part_edit(document, &selected)?; (edit, indices, "Duplicated selected part. Ctrl+Z undoes this edit.") },
                                    UiAction::DeleteSelectedPart => (delete_selected_part_edit(document, &selected)?, Vec::new(), "Deleted selected part. Ctrl+Z restores it."),
                                    _ => unreachable!(),
                                };
                                let rebuilt = commit_model_edit(document, &mut model_edit_history, &config, edit)?;
                                Ok::<_, String>((rebuilt, next_selection, message))
                            })();
                            match result {
                                Ok((rebuilt, selection, message)) => {
                                    scene = rebuilt;
                                    model_revision = model_revision.wrapping_add(1);
                                    let weights_invalidated =
                                        invalidate_skeleton_weights_after_geometry_edit(document);
                                    update_model_ui_selection(&mut ui, document, selection);
                                    sync_model_history_ui(&mut ui, &model_edit_history);
                                    ui.set_status_message(message);
                                    if weights_invalidated {
                                        ui.set_status_message("Model edited. Bone weights were cleared because voxel geometry changed; regenerate automatic weights.");
                                    }
                                    accumulator = 0.0;
                                    last = Instant::now();
                                }
                                Err(error) => ui.set_status_message(format!("Error: {error}")),
                            }
                        }
                    }
                    let shape = if editing_seed {
                        None
                    } else {
                        match event.physical_key {
                        PhysicalKey::Code(KeyCode::F1) => {
                            ui.toggle_help();
                            None
                        }
                        PhysicalKey::Code(KeyCode::F2) => {
                            grabbed_cell = None;
                            if let Some(physics) = world_physics.as_ref() {
                                physics.set_grab_target(None);
                            }
                            if ui.editor_mode {
                                ui.editor_mode = false;
                                if let Some(character) = world_character.as_ref() {
                                    set_game_camera(character, &mut camera, world_physics.is_some());
                                    orbit = true;
                                    ui.paused = false;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                    ignore_mouse_motion_until = Instant::now() + Duration::from_millis(150);
                                    last = Instant::now();
                                }
                            } else if let Some(character) = world_character.as_ref() {
                                ui.editor_mode = true;
                                set_editor_camera(character, &mut camera, 8.0);
                                orbit = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                            } else if active_scene_document.is_some()
                                || active_model_document.is_some()
                            {
                                ui.editor_mode = true;
                                camera.enable_freecam();
                                orbit = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                            }
                            None
                        }
                        PhysicalKey::Code(KeyCode::F6) => {
                            grabbed_cell = None;
                            if let Some(physics) = world_physics.as_ref() {
                                physics.set_grab_target(None);
                            }
                            if let (Some(world),Some(character)) =
                                (generated_world.as_mut(),world_character.as_mut()) {
                                match start_world_physics(world,character,&mut camera,&config,adaptive_physics) {
                                    Ok(worker) => {
                                        world_physics = Some(worker);
                                        ui.paused = false;
                                        if ui.editor_mode {
                                            set_editor_camera(character, &mut camera, 2.5);
                                            orbit = false;
                                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                                            window.set_cursor_visible(true);
                                        } else {
                                            orbit = true;
                                            let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                            window.set_cursor_visible(false);
                                        }
                                    }
                                    Err(error) => eprintln!("world physics could not start: {error}"),
                                }
                            }
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyF) => {
                            grabbed_cell = None;
                            if let Some(physics) = world_physics.as_ref() {
                                physics.set_grab_target(None);
                            }
                            if camera.is_freecam() {
                                camera.disable_freecam();
                                if let Some(character) = world_character.as_ref() {
                                    camera.set_first_person(character.eye_position());
                                    orbit = true;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                } else {
                                    orbit = false;
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                }
                            } else {
                                camera.enable_freecam();
                                orbit = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                            }
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.x = (camera.target.x - pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.x = (camera.target.x + pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.y = (camera.target.y + pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.y = (camera.target.y - pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::Digit1) => Some(Shape::Wedge),
                        PhysicalKey::Code(KeyCode::Digit2) => Some(Shape::Blade),
                        PhysicalKey::Code(KeyCode::Digit3) => Some(Shape::Rod),
                        PhysicalKey::Code(KeyCode::Digit4) => Some(Shape::Rounded),
                        PhysicalKey::Code(KeyCode::Space) => {
                            if world_character.is_some() && !camera.is_freecam() {
                                jump_queued = true;
                            } else if ui.paused
                                && active_model_document
                                    .as_ref()
                                    .is_some_and(model_has_unsupported_runtime_features)
                            {
                                ui.set_status_message("Simulation is unavailable for this model: skeletons, voxel groups, and motion constraints do not have a runtime solver yet.");
                            } else {
                                ui.paused = !ui.paused;
                                if mode != AppMode::Playback {
                                    accumulator = 0.0;
                                }
                            }
                            last = Instant::now();
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyB) => {
                            ui.show_bonds = !ui.show_bonds;
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            authored_physics = None;
                            if let Some(source) = authored_render_source.take() { renderer.set_authored_physics_frame(&source, None); }
                            mode = AppMode::Interactive;
                            world_physics = None;
                            generated_world = None;
                            world_character = None;
                            ui.world_active = false;
                            ui.editor_mode = false;
                            if let Some(document) = active_model_document.as_ref() {
                                match build_editor_preview_scene(document, config.clone()) {
                                    Ok(restored_scene) => {
                                        scene = restored_scene;
                                        frame_camera_on_scene(&scene, &mut camera);
                                        let selected = ui.selected_voxel_indices.clone();
                                        update_model_ui_selection(&mut ui, document, selected);
                                        ui.world_active = true;
                                        ui.editor_mode = true;
                                        ui.set_status_message("Model reset to its last authored state.");
                                    }
                                    Err(error) => {
                                        ui.set_status_message(format!("Error: {error}"));
                                        eprintln!("model reset failed: {error}");
                                    }
                                }
                            } else if let Some(document) = active_scene_document.as_ref() {
                                let path = active_scene_document_path
                                    .as_deref()
                                    .unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                match document.build_scene(path) {
                                    Ok(restored_scene) => {
                                        scene = restored_scene;
                                        frame_camera_on_scene(&scene, &mut camera);
                                        ui.world_active = true;
                                        ui.editor_mode = true;
                                        ui.set_status_message("Scene reset to its saved definition.");
                                    }
                                    Err(error) => {
                                        ui.set_status_message(format!("Error: {error}"));
                                        eprintln!("scene reset failed: {error}");
                                    }
                                }
                                ui.set_active_scene_document(
                                    Some(document.name.clone()),
                                    scene_model_names(document),
                                );
                            } else if friction_road_demo {
                                active_scene_document_path = None;
                                ui.set_scene_document_available(false);
                                scene = friction_heat_road_scene(config.clone())
                                    .expect("the built-in friction road scene is valid");
                                camera.set_orbit(Vec3::new(-0.53, 0.0, 0.015), 0.38);
                                camera.yaw = 0.55;
                                camera.pitch = 0.35;
                            } else if two_cube_demo {
                                active_scene_document_path = None;
                                ui.set_scene_document_available(false);
                                scene = two_cube_collision_scene(config.clone())
                                    .expect("the built-in cube collision scene is valid");
                                camera.set_orbit(Vec3::new(0.0, 0.0, 0.62), 0.90);
                                camera.yaw = 0.55;
                                camera.pitch = 0.35;
                            } else if active_scene_document.is_none() {
                                active_scene_document_path = None;
                                ui.set_scene_document_available(false);
                                scene = Scene::new(config.clone());
                                fix_wood_base(&mut scene);
                                camera.set_orbit(Vec3::Z * (height * 0.5), initial_camera_radius);
                            }
                            tool = None;
                            playback_frames.clear();
                            playback_time_s = 0.0;
                            live_simulated_s = 0.0;
                            live_wall_s = 0.0;
                            next_record_frame_s = PLAYBACK_FRAME_INTERVAL_SECONDS;
                            accumulator = 0.0;
                            ui.paused = active_scene_document.is_some()
                                || active_model_document.is_some();
                            fracture_pause_triggered = false;
                            last = Instant::now();
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyP) => {
                            if world_character.is_some() || active_scene_document.is_some() {
                                ui.paused = !ui.paused;
                                accumulator = 0.0;
                            } else { match mode {
                                AppMode::Interactive => {
                                    playback_frames.clear();
                                    playback_frames.push(capture_frame(&scene, &tool, 0.0));
                                    playback_time_s = 0.0;
                                    live_simulated_s = 0.0;
                                    live_wall_s = 0.0;
                                    next_record_frame_s = PLAYBACK_FRAME_INTERVAL_SECONDS;
                                    accumulator = 0.0;
                                    ui.paused = false;
                                    fracture_pause_triggered = false;
                                    mode = AppMode::Recording;
                                    last = Instant::now();
                                }
                                AppMode::Recording => {
                                    if let Some(frame) = playback_frames.last() {
                                        if live_simulated_s > frame.time_s + 1e-9 {
                                            playback_frames.push(capture_frame(
                                                &scene,
                                                &tool,
                                                live_simulated_s,
                                            ));
                                        }
                                    }
                                    mode = AppMode::Playback;
                                    playback_time_s = 0.0;
                                    accumulator = 0.0;
                                    ui.paused = false;
                                    if let Some(frame) = playback_frames.first() {
                                        apply_frame(&mut scene, &mut tool, frame);
                                    }
                                    last = Instant::now();
                                }
                                AppMode::Playback => {
                                    playback_time_s = 0.0;
                                    ui.paused = false;
                                    if let Some(frame) = playback_frames.first() {
                                        apply_frame(&mut scene, &mut tool, frame);
                                    }
                                    last = Instant::now();
                                }
                            }}
                            None
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            None
                        }
                        _ => None,
                    }};
                    if let Some(shape) = shape {
                        ui.selected = shape;
                        if mode == AppMode::Interactive {
                            tool = Some(launch(shape, &scene, &ui));
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    if let Some(action) = pending_scene_action.take() {
                        match action {
                            UiAction::SetScenePhysics {
                                fixed_dt_seconds,
                                substeps,
                                gravity,
                                restitution,
                                friction_scale,
                                solver_iterations,
                                solver_tolerance,
                            } => {
                                if !ui.paused {
                                    ui.set_status_message("Pause the scene before changing physics settings.");
                                } else if let Some(document) = active_scene_document.as_ref() {
                                    let path = active_scene_document_path.as_deref()
                                        .unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                    let mut updated = document.clone();
                                    updated.config.fixed_dt = fixed_dt_seconds;
                                    updated.config.substeps = substeps as usize;
                                    updated.config.gravity = gravity;
                                    updated.config.fracture.restitution = restitution;
                                    updated.config.fracture.friction = friction_scale;
                                    updated.config.iterations = solver_iterations;
                                    updated.config.solver_relative_tolerance = solver_tolerance;
                                    let result = updated.config.validate()
                                        .map_err(|error| error.to_string())
                                        .and_then(|()| updated.build_scene(path));
                                    match result {
                                        Ok(rebuilt) => {
                                            // Dropping the worker discards its paused runtime state; the
                                            // authored document remains the source for all scene edits.
                                            authored_physics = None;
                                            if let Some(source) = authored_render_source.take() {
                                                renderer.set_authored_physics_frame(&source, None);
                                            }
                                            scene = rebuilt;
                                            config = updated.config.clone();
                                            active_scene_document = Some(updated);
                                            if let Some(document) = active_scene_document.as_ref() {
                                                ui.set_scene_physics_config(&document.config);
                                            }
                                            live_simulated_s = 0.0;
                                            live_wall_s = 0.0;
                                            accumulator = 0.0;
                                            last = Instant::now();
                                            model_revision = model_revision.wrapping_add(1);
                                            ui.set_status_message("Scene physics settings updated. Save the scene to keep the change.");
                                        }
                                        Err(error) => ui.set_status_message(format!("Error: {error}")),
                                    }
                                } else {
                                    ui.set_status_message("Open a scene first.");
                                }
                            }
                            UiAction::SetSceneSelectionMode(selection_mode) => {
                                ui.scene_selection_mode = selection_mode;
                                ui.set_model_selection(None, None, None);
                                if let Some(document) = active_scene_document.as_ref() {
                                    let path = active_scene_document_path.as_deref().unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                    let selected = ui.selected_scene_model;
                                    sync_scene_instance_selection(&mut ui, document, path, selected);
                                    if selection_mode == SceneSelectionMode::Voxel { ui.set_scene_model_selection_bounds(None); }
                                }
                            }
                            UiAction::CreateImpactScene { side } => {
                                if let Some(path) = pick_scene_document_path(true, &mut ui, &mut accumulator, &mut last) {
                                    let path = ensure_scene_extension(path);
                                    ui.set_status_message(format!("Creating an editable {side}³ cube scene…"));
                                    let (sender, receiver) = mpsc::channel();
                                    scene_load_job = Some(receiver);
                                    std::thread::spawn(move || {
                                        let result = voxel_physics_lab::scene_presets::create_impact_scene(&path, side)
                                            .and_then(|document| document.build_scene(&path).map(|runtime| (document, runtime, Some(path))));
                                        let _ = sender.send(result);
                                    });
                                }
                            }
                            action @ (UiAction::SetChemistryEnabled(_)
                                | UiAction::SetChemistryInterval { .. }
                                | UiAction::ChooseChemistryRecipe
                                | UiAction::UseBuiltinChemistryRecipes) => {
                                let edit = chemistry_edit_from_action(action, || {
                                    pick_recipe_file_path(&mut ui, &mut accumulator, &mut last)
                                });
                                if let Some(edit) = edit {
                                    let result = if authored_physics.is_some() {
                                        Err("Press R to restore the authored scene before changing chemistry.".to_owned())
                                    } else if active_scene_document.is_none() {
                                        Err("Open a scene first.".to_owned())
                                    } else {
                                        edit_scene_chemistry(
                                            &ui.scene_chemistry,
                                            edit,
                                            active_scene_document_path.as_deref(),
                                            &mut scene,
                                        )
                                    };
                                    match result {
                                        Ok((settings, message)) => {
                                            ui.scene_chemistry = settings;
                                            ui.set_status_message(message);
                                        }
                                        Err(error) => {
                                            eprintln!("chemistry: {error}");
                                            ui.set_status_message(format!("Error: {error}"));
                                        }
                                    }
                                }
                            }
                            action => {
                                let result = if authored_physics.is_some() || !ui.paused {
                                    Err("Pause and press R before editing the authored scene.".into())
                                } else if let Some(document) = active_scene_document.as_ref() {
                                    let path = active_scene_document_path.as_deref().unwrap_or_else(|| Path::new("Untitled.vxscene"));
                                    scene_property_edit(action, &scene, document, path, &ui)
                                } else { Err("Open a scene first.".into()) };
                                match result {
                                    Ok(Some(edit)) => {
                                        let selection = ui.selected_voxel_indices.clone();
                                        if run_scene_instance_edit(&mut scene, &mut active_scene_document, active_scene_document_path.as_deref(), &mut ui,
                                            &mut live_simulated_s, &mut live_wall_s, &mut accumulator, &mut last, edit) {
                                            model_revision = model_revision.wrapping_add(1);
                                            if ui.scene_selection_mode == SceneSelectionMode::Voxel {
                                                ui.set_scene_model_selection_bounds(None);
                                                select_scene_voxels(&mut ui, &scene, selection);
                                            }
                                        }
                                    }
                                    Ok(None) => {},
                                    Err(error) => ui.set_status_message(format!("Error: {error}")),
                                }
                            }
                        }
                    }
                    let loaded_scene = scene_load_job.as_ref().and_then(|receiver| match receiver.try_recv() {
                        Ok(result) => Some(result),
                        Err(mpsc::TryRecvError::Disconnected) => Some(Err("Scene loader stopped unexpectedly.".into())),
                        Err(mpsc::TryRecvError::Empty) => None,
                    });
                    if let Some(result) = loaded_scene {
                        scene_load_job = None;
                        match result {
                            Ok((document, runtime, path)) => {
                                authored_physics = None;
                                if let Some(source) = authored_render_source.take() { renderer.set_authored_physics_frame(&source, None); }
                                world_physics = None;
                                generated_world = None;
                                world_character = None;
                                active_model_document = None;
                                active_model_document_path = None;
                                scene = runtime;
                                ui.standalone_editor_app = false;
                                ui.world_active = true;
                                ui.editor_mode = true;
                                ui.paused = true;
                                ui.show_bonds = false;
                                ui.set_model_selection(None, None, None);
                                let selected = (!document.models.is_empty()).then_some(0);
                                sync_scene_instance_selection(&mut ui, &document, path.as_deref().unwrap_or_else(|| Path::new("Untitled.vxscene")), selected);
                                ui.set_status_message("Editable scene ready. Select an object or voxels; Play runs the scene.");
                                active_scene_document = Some(document);
                                active_scene_document_path = path;
                                tool = None;
                                mode = AppMode::Interactive;
                                playback_frames.clear();
                                live_simulated_s = 0.0;
                                live_wall_s = 0.0;
                                accumulator = 0.0;
                                model_revision = model_revision.wrapping_add(1);
                                frame_camera_on_scene(&scene, &mut camera);
                                last = Instant::now();
                            }
                            Err(error) => ui.set_status_message(format!("Scene loading failed: {error}")),
                        }
                    }
                    let completed_auto_weights = auto_weight_job.as_ref().and_then(|receiver| {
                        match receiver.try_recv() {
                            Ok(result) => Some(result),
                            Err(mpsc::TryRecvError::Disconnected) => Some((
                                u64::MAX,
                                0,
                                0.0,
                                Err("automatic weight worker stopped before returning results".into()),
                            )),
                            Err(mpsc::TryRecvError::Empty) => None,
                        }
                    });
                    if let Some((revision, voxel_count, cell_size_m, result)) = completed_auto_weights {
                        auto_weight_job = None;
                        let is_current = revision == model_revision
                            && active_model_document.as_ref().is_some_and(|document| {
                                document.voxels.len() == voxel_count
                                    && document.cell_size_m == cell_size_m
                                    && document.skeleton.is_some()
                            });
                        if is_current {
                            match result {
                                Ok(weights) => {
                                    if let Some(document) = active_model_document.as_mut() {
                                        if let Some(skeleton) = document.skeleton.as_mut() {
                                            skeleton.voxel_weights = weights;
                                        }
                                        model_revision = model_revision.wrapping_add(1);
                                        skinning_weights_revision = skinning_weights_revision.wrapping_add(1);
                                        update_model_ui(&mut ui, document, None);
                                        ui.set_status_message(format!(
                                            "Automatic weights assigned to {voxel_count} REVs (up to 4 influences each)."
                                        ));
                                    }
                                }
                                Err(error) => ui.set_status_message(format!("Automatic weights failed: {error}")),
                            }
                        } else {
                            ui.set_status_message("Discarded automatic weights because the model or skeleton changed during calculation.");
                        }
                    }
                    let completed_import = mesh_import_job.as_ref().and_then(|receiver| {
                        match receiver.try_recv() {
                            Ok(result) => Some(result),
                            Err(mpsc::TryRecvError::Disconnected) => Some(Err(
                                "mesh import worker stopped before returning a preview".to_owned(),
                            )),
                            Err(mpsc::TryRecvError::Empty) => None,
                        }
                    });
                    if let Some(result) = completed_import {
                        mesh_import_job = None;
                        ui.set_mesh_import_busy(false);
                        match result {
                            Ok((preview, preview_scene)) => {
                                if mesh_import_camera_backup.is_none() {
                                    mesh_import_camera_backup = Some(CameraSnapshot::capture(&camera));
                                }
                                frame_camera_on_scene(&preview_scene, &mut camera);
                                let dimensions = preview.source_bounds_max_m
                                    - preview.source_bounds_min_m;
                                ui.set_mesh_import_preview(ui::MeshImportPreviewInfo {
                                    watertight: preview.is_watertight,
                                    dimensions_m: dimensions.as_vec3().to_array(),
                                    cell_count: preview.cells.len(),
                                });
                                ui.set_status_message(if !preview.is_watertight
                                    && preview.mode == VoxelizationMode::Filled
                                {
                                    format!(
                                        "Mesh preview ready: {} REVs; interior filled at 1 cm grid resolution.",
                                        preview.cells.len()
                                    )
                                } else {
                                    format!("Mesh preview ready: {} REVs.", preview.cells.len())
                                });
                                mesh_import_preview = Some(preview);
                                mesh_import_preview_scene = Some(preview_scene);
                            }
                            Err(error) => {
                                ui.set_status_message(format!("Mesh preview failed: {error}"));
                                ui.set_mesh_import_error(error);
                            }
                        }
                    }
                    let completed_model_export = model_export_job.as_ref().and_then(|receiver| {
                        match receiver.try_recv() {
                            Ok(result) => Some(result),
                            Err(mpsc::TryRecvError::Disconnected) => Some(Err(
                                "OBJ export worker stopped before finishing".to_owned(),
                            )),
                            Err(mpsc::TryRecvError::Empty) => None,
                        }
                    });
                    if let Some(result) = completed_model_export {
                        model_export_job = None;
                        match result {
                            Ok(faces) => ui.set_status_message(format!(
                                "OBJ exported: {faces} faces. Save .vxm to keep voxel physics and materials."
                            )),
                            Err(error) => ui.set_status_message(format!("OBJ export failed: {error}")),
                        }
                    }
                    let now = Instant::now();
                    let elapsed = now.duration_since(last).as_secs_f32();
                    last = now;
                    if ui.world_active && ui.editor_mode && !ui.menu_open && !ui.mesh_import_open {
                        ui.advance_animation(elapsed);
                    }
                    if profile_world_geometry && generated_world.is_some() && elapsed > 0.05 {
                        eprintln!("world frame gap_ms={:.2}", elapsed * 1_000.0);
                    }

                    if camera.is_freecam() && !ui.scene_input_active() && !ui.menu_open && !ui.mesh_import_open {
                        let pressed = |key| keys_down.contains(&key);
                        let right = (pressed(KeyCode::KeyD) || pressed(KeyCode::ArrowRight))
                            as u8 as f32
                            - (pressed(KeyCode::KeyA) || pressed(KeyCode::ArrowLeft)) as u8 as f32;
                        let forward = (pressed(KeyCode::KeyW) || pressed(KeyCode::ArrowUp)) as u8
                            as f32
                            - (pressed(KeyCode::KeyS) || pressed(KeyCode::ArrowDown)) as u8 as f32;
                        let up = f32::from(pressed(KeyCode::Space))
                            - f32::from(pressed(KeyCode::ControlLeft))
                            - f32::from(pressed(KeyCode::ControlRight));
                        let fast = pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight);
                        camera.move_freecam(right, forward, up, elapsed.min(0.1), fast);
                    }

                    // Keep the Chemistry section in step with the open scene: a different
                    // scene brings its own saved settings (`<scene>.chemistry.json`).
                    let chemistry_scene = active_scene_document
                        .as_ref()
                        .map(|_| active_scene_document_path.clone());
                    if chemistry_synced_for != chemistry_scene {
                        ui.scene_chemistry = match active_scene_document_path.as_deref() {
                            Some(path) => ChemistrySettings::load(path).unwrap_or_else(|error| {
                                ui.set_status_message(format!("Error: {error}"));
                                ChemistrySettings::default()
                            }),
                            None => ChemistrySettings::default(),
                        };
                        chemistry_synced_for = chemistry_scene;
                    }
                    if active_scene_document.is_none() {
                        authored_physics = None;
                        ui.set_physics_cost_ms_per_10ms(None);
                        if let Some(source) = authored_render_source.take() { renderer.set_authored_physics_frame(&source, None); }
                    } else if !ui.paused
                        && authored_physics.is_none()
                        && {
                            // The Chemistry section is the source of the scene's reaction
                            // settings; give them to the scene the worker is about to own
                            // (a restored or reloaded scene starts without them).
                            let applied = ui
                                .scene_chemistry
                                .apply(&mut scene, active_scene_document_path.as_deref());
                            if let Err(error) = &applied {
                                ui.paused = true;
                                ui.set_status_message(format!(
                                    "Error: scene not started, chemistry settings could not be applied: {error}"
                                ));
                            }
                            applied.is_ok()
                        }
                    {
                        let source = AuthoredPhysicsRenderSource::from_scene(&scene);
                        let empty = Scene::from_voxels(scene.config.clone(), Vec::new()).expect("empty scene is valid");
                        let runtime = std::mem::replace(&mut scene, empty);
                        match PhysicsWorker::new(runtime) {
                            Ok(worker) => {
                                authored_render_source = Some(source);
                                authored_physics = Some(worker);
                                ui.set_model_selection(None, None, None);
                                ui.set_scene_model_selection_bounds(None);
                                hovered_face = None;
                                ui.set_status_message("Scene running — Space pauses — R restores the authored state.");
                            }
                            Err(error) => {
                                ui.paused = true;
                                ui.set_status_message(format!("Could not start scene physics: {error}. Press R to restore."));
                            }
                        }
                    }
                    if let Some(physics) = authored_physics.as_mut() {
                        physics.advance(f64::from(elapsed) * f64::from(ui.simulation_time_scale), ui.paused);
                        run_panel.record(physics.frame.generation, physics.frame.revision, physics.frame.full_worker_ms_per_10ms_simulated);
                        ui.set_physics_cost_ms_per_10ms(
                            physics.frame.full_worker_ms_per_10ms_simulated,
                        );
                        ui.set_physics_worker_status(Some((
                            physics.backlog_seconds(),
                            physics.failure_message().is_some(),
                        )));
                        if let Some(error) = physics.failure_message() {
                            ui.paused = true;
                            if reported_physics_failure != Some(physics.frame.generation) {
                                ui.set_status_message(format!("Physics stopped: {error}. Press R to restore the scene."));
                                reported_physics_failure = Some(physics.frame.generation);
                            }
                        }
                        live_simulated_s = physics.frame.simulated_seconds;
                        sim_ms = physics.frame.step_ms as f32;
                        if !ui.paused { live_wall_s += f64::from(elapsed); }
                        if let Some(source) = authored_render_source.as_ref() {
                            renderer.set_authored_physics_frame(source, Some(&physics.frame));
                        }
                    } else {
                        // No scene worker: clear any stale failure or backlog. The
                        // world worker, if present, republishes its own status below.
                        ui.set_physics_worker_status(None);
                    }
                    if let Some(world) = generated_world.as_mut() {
                        growth_scheduler.advance(world, if ui.paused {
                            0.0
                        } else {
                            elapsed * ui.simulation_time_scale
                        });
                    }
                    if !ui.paused {
                        if let Some(world) = generated_world.as_mut() {
                            if !camera.is_freecam() && !ui.editor_mode {
                                if let Some(character) = world_character.as_mut() {
                                    let pressed = |key| keys_down.contains(&key);
                                    let right = f32::from(pressed(KeyCode::KeyD) || pressed(KeyCode::ArrowRight))
                                        - f32::from(pressed(KeyCode::KeyA) || pressed(KeyCode::ArrowLeft));
                                    let forward = f32::from(pressed(KeyCode::KeyW) || pressed(KeyCode::ArrowUp))
                                        - f32::from(pressed(KeyCode::KeyS) || pressed(KeyCode::ArrowDown));
                                    let sprint = pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight);
                                    character.set_crouching(world,
                                        pressed(KeyCode::ControlLeft) || pressed(KeyCode::ControlRight));
                                    character.advance(
                                        world,
                                        Vec3::new(right, forward, 0.0),
                                        camera.yaw,
                                        ui.walk_speed_mps,
                                        sprint,
                                        jump_queued,
                                        elapsed,
                                    );
                                    camera.set_first_person(character.eye_position());
                                }
                            }
                        }
                    }
                    jump_queued = false;
                    if let Some(physics) = world_physics.as_mut() {
                        let grab_target = grabbed_cell.map(|grabbed| GrabTarget {
                            source_cell: grabbed.source_cell,
                            scene_index: grabbed.scene_index,
                            target_world_m: camera.position().as_dvec3()
                                + camera.forward().as_dvec3() * f64::from(grabbed.distance_m),
                        });
                        physics.set_grab_target(grab_target);
                        physics.advance(f64::from(elapsed)*f64::from(ui.simulation_time_scale),ui.paused);
                        run_panel.record(physics.frame.generation, physics.frame.revision, physics.frame.full_worker_ms_per_10ms_simulated);
                        ui.set_physics_cost_ms_per_10ms(
                            physics.frame.full_worker_ms_per_10ms_simulated,
                        );
                        ui.set_physics_worker_status(Some((
                            physics.backlog_seconds(),
                            physics.failure_message().is_some(),
                        )));
                        if let Some(error) = physics.failure_message() {
                            ui.paused = true;
                            if reported_physics_failure != Some(physics.frame.generation) {
                                ui.set_status_message(format!("Physics stopped: {error}. Reset the scene to continue."));
                                reported_physics_failure = Some(physics.frame.generation);
                            }
                        }
                    }
                    if authored_physics.is_none() && world_physics.is_none() {
                        run_panel.clear();
                    }

                    if generated_world.is_none()
                        && active_scene_document.is_none()
                        && matches!(mode, AppMode::Interactive | AppMode::Recording)
                        && !ui.paused
                    {
                        live_wall_s += f64::from(elapsed);
                        accumulator += f64::from(elapsed) * f64::from(ui.simulation_time_scale);
                        // Preserve elapsed time. Limit simulation catch-up to 40 ms per redraw
                        // and retain any backlog instead of dropping simulated time.
                        let start = Instant::now();
                        let mut steps = 0;
                        let fixed_dt = f64::from(config.fixed_dt);
                        let max_steps =
                            (MAX_SIMULATED_SECONDS_PER_REDRAW / fixed_dt).floor() as usize;
                        while accumulator >= fixed_dt
                            && steps < max_steps.max(1)
                            && (steps == 0 || start.elapsed() < MAX_SOLVER_WALL_TIME_PER_REDRAW)
                        {
                            if let Some(active_tool) = tool.as_mut() {
                                active_tool
                                    .set_external_force(ui.direction() * ui.force_newtons)
                                    .expect("UI force is finite");
                            }
                            solver::step(&mut scene, &mut tool);
                            accumulator -= fixed_dt;
                            live_simulated_s += fixed_dt;
                            steps += 1;
                            if mode == AppMode::Recording {
                                while live_simulated_s + fixed_dt * 0.5
                                    >= next_record_frame_s
                                {
                                    playback_frames.push(capture_frame(
                                        &scene,
                                        &tool,
                                        live_simulated_s,
                                    ));
                                    next_record_frame_s += PLAYBACK_FRAME_INTERVAL_SECONDS;
                                }
                                if live_simulated_s + fixed_dt * 0.5
                                    >= RECORDING_DURATION_SECONDS
                                {
                                    if playback_frames
                                        .last()
                                        .is_some_and(|frame| {
                                            frame.time_s + fixed_dt * 0.5
                                                < live_simulated_s
                                        })
                                    {
                                        playback_frames.push(capture_frame(
                                            &scene,
                                            &tool,
                                            live_simulated_s,
                                        ));
                                    }
                                    mode = AppMode::Playback;
                                    playback_time_s = 0.0;
                                    accumulator = 0.0;
                                    if let Some(frame) = playback_frames.first() {
                                        apply_frame(&mut scene, &mut tool, frame);
                                    }
                                    break;
                                }
                            }
                        }
                        if pause_on_fracture && !fracture_pause_triggered && scene.broken > 0 {
                            ui.paused = true;
                            ui.show_bonds = true;
                            fracture_pause_triggered = true;
                        }
                        if steps > 0 {
                            sim_ms = start.elapsed().as_secs_f32() * 1000.0 / steps as f32;
                        }
                        if scene.positions.iter().any(|p| !p.is_finite()) {
                            eprintln!("Solver became non-finite; paused.");
                            ui.paused = true;
                        }
                    } else if mode == AppMode::Playback && !ui.paused {
                        let duration = playback_frames
                            .last()
                            .map_or(0.0, |frame| frame.time_s);
                        // Keep the last saved state visible for one sample interval, then
                        // wrap to the beginning. Playback is a view of captured solver states;
                        // it must not stop the physical run or synthesize intermediate states.
                        let loop_duration = duration + PLAYBACK_FRAME_INTERVAL_SECONDS;
                        playback_time_s = (playback_time_s + f64::from(elapsed)) % loop_duration;
                        let sample_time_s = playback_time_s.min(duration);
                        let frame_index = playback_frames
                            .partition_point(|frame| frame.time_s <= sample_time_s)
                            .saturating_sub(1);
                        if let Some(frame) = playback_frames.get(frame_index) {
                            apply_frame(&mut scene, &mut tool, frame);
                        }
                    }
                    let reticle = if camera.is_first_person() && !ui.menu_open {
                        Some(if grabbed_cell.is_some() {
                            ReticleState::Holding
                        } else if world_physics.is_none() {
                            ReticleState::PhysicsOff
                        } else if world_physics.as_ref().is_some_and(|physics| {
                            aimed_physics_cell(&physics.frame, &camera).is_some()
                        }) {
                            ReticleState::Target
                        } else {
                            ReticleState::Idle
                        })
                    } else {
                        None
                    };
                    let viewport = window.inner_size();
                    let vegetation_summary = generated_world
                        .as_ref()
                        .map_or_else(String::new, GeneratedWorld::vegetation_summary);
                    let editor_seed = ui.seed_text.clone();
                    let editor_info = generated_world.as_ref().map(|world| EditorInfo {
                        physics_cells: world_physics.as_ref().map_or(0, |physics| physics.frame.cells.len()),
                        physics_enabled: world_physics.is_some(),
                        world_age_years: world.age_years,
                        seed: &editor_seed,
                        vegetation_summary: &vegetation_summary,
                    });
                    let visible_scene = mesh_import_preview_scene.as_ref().unwrap_or(&scene);
                    let import_preview_active = mesh_import_preview_scene.is_some();
                    if let Some(document) = active_scene_document.as_ref() {
                        ui.set_scene_physics_config(&document.config);
                    }
                    let forward = camera.forward().normalize_or_zero();
                    let mut view_right = forward.cross(Vec3::Z).normalize_or_zero();
                    if view_right.length_squared() < 0.5 { view_right = Vec3::X; }
                    ui.set_view_basis(view_right, view_right.cross(forward).normalize_or_zero());
                    let rects = ui.draw(
                        visible_scene.broken,
                        visible_scene.cluster_count,
                        fps,
                        sim_ms,
                        generated_world.as_ref().map(|world| world.age_years),
                        reticle,
                        viewport.width as f32,
                        viewport.height as f32,
                        editor_info.as_ref(),
                    );
                    let scene_viewport = ui.scene_viewport();
                    let selection_bounds = ui.selected_bounds_m.map(|bounds| {
                        (
                            Vec3::from_array(bounds[0].map(|value| value as f32)),
                            Vec3::from_array(bounds[1].map(|value| value as f32)),
                        )
                    });
                    let selection_preview = selection_drag
                        .and_then(|drag| {
                            active_model_document.as_ref().map(|document| {
                                let bounds = selection_bounds_for_cells(
                                    drag.start_cell,
                                    drag.end_cell,
                                    document.cell_size_m,
                                );
                                (
                                    Vec3::from_array(bounds[0].map(|value| value as f32)),
                                    Vec3::from_array(bounds[1].map(|value| value as f32)),
                                )
                            })
                        })
                        .or_else(|| {
                            scene_move_drag.map(|drag| {
                                let offset = drag.delta_cells.as_vec3()
                                    * voxel_physics_lab::coordinates::CELL_M as f32;
                                (drag.bounds.0 + offset, drag.bounds.1 + offset)
                            })
                        });
                    match renderer.draw(
                        visible_scene,
                        if import_preview_active { &None } else { &tool },
                        if import_preview_active { None } else { generated_world.as_ref() },
                        if import_preview_active {
                            None
                        } else {
                            world_physics.as_ref().map(|physics| &physics.frame)
                        },
                        &camera,
                        &rects,
                        scene_viewport,
                        if import_preview_active {
                            None
                        } else {
                            hovered_face.map(|face| (face.center, face.normal))
                        },
                        if import_preview_active {
                            &[]
                        } else {
                            &ui.selected_voxel_indices
                        },
                        if import_preview_active {
                            None
                        } else {
                            selection_bounds
                        },
                        if import_preview_active {
                            None
                        } else {
                            selection_preview
                        },
                        if import_preview_active {
                            &[]
                        } else {
                            &sculpt_preview_cells
                        },
                        if import_preview_active {
                            None
                        } else {
                            sculpt_preview_bounds
                        },
                        active_model_document
                            .as_ref()
                            .map_or(0.01, |document| document.cell_size_m as f32),
                        ui.show_bonds,
                        if import_preview_active {
                            None
                        } else {
                            active_model_document.as_ref().and_then(|document| {
                                document.skeleton.as_ref().map(|_| {
                                    (
                                        document,
                                        ui.selected_bone,
                                        document.cell_size_m as f32,
                                        ui.selected_animation_clip,
                                        f64::from(ui.selected_animation_time_seconds),
                                    )
                                })
                            })
                        },
                        ui.show_rig_overlay,
                        skinning_weights_revision,
                        model_revision,
                    ) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let s = window.inner_size();
                            renderer.resize(s.width, s.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            target.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {}
                    }
                    frames += 1;
                    if last_runtime_publish.elapsed() >= Duration::from_millis(250) {
                        let physics = authored_physics.as_ref().or(world_physics.as_ref());
                        let simulated = physics.map_or(scene.simulated_seconds, |worker| worker.frame.simulated_seconds);
                        let elapsed_wall = last_runtime_publish.elapsed().as_secs_f64();
                        let generation = physics.map(|worker| worker.frame.generation);
                        let rate_valid = previous_runtime_generation == generation && simulated >= previous_runtime_simulated;
                        let speed = if rate_valid {
                            (simulated-previous_runtime_simulated)/elapsed_wall.max(1.0e-9)
                        } else { 0.0 };
                        ui.set_measured_simulation_rate(
                            (physics.is_some() && rate_valid).then_some(speed),
                        );
                        // Run panel: the same values the MCP `get_run_panel` tool serves.
                        let (mechanical_revs, cell_rotations) = match physics {
                            Some(worker) => (worker.frame.mechanical_revs, worker.frame.cell_rotations),
                            None => (
                                solver::mechanical_rev_count(&scene),
                                solver::uses_cell_rotations(&scene),
                            ),
                        };
                        let run_panel_snapshot = run_panel.snapshot(RunPanelInput {
                            state: RunState::from_flags(ui.paused, ui.physics_failed),
                            requested_speed: ui.simulation_time_scale,
                            measured_speed: ui.measured_simulation_rate,
                            backlog_s: ui.physics_backlog_seconds,
                            active_mechanical_revs: Some(mechanical_revs),
                            solver_mode: Some(SolverMode::from_cell_rotations(cell_rotations)),
                        });
                        if runtime_metrics_server.is_some() {
                            voxel_physics_lab::runtime_metrics::publish(
                                "run_panel",
                                run_panel_snapshot.to_json(),
                            );
                        }
                        ui.set_run_panel(Some(run_panel_snapshot));
                        // Creature inspector: read-only, and only while the scene is not
                        // owned by the physics worker.
                        if authored_physics.is_some() {
                            ui.set_creature_stale(true);
                        } else {
                            let creature_scene = active_scene_document_path
                                .as_ref()
                                .filter(|_| active_scene_document.is_some() && !scene.skeletal_actuators.is_empty());
                            match creature_scene {
                                Some(path) => {
                                    let key = (path.clone(), scene.positions.as_ptr() as usize, scene.render_revision as usize);
                                    if scene_creatures_key.as_ref() != Some(&key) {
                                        scene_creatures = match CreatureFile::load(path)
                                            .and_then(|file| file.map_or(Ok(Vec::new()), |file| file.build(&scene)))
                                        {
                                            Ok(list) => list,
                                            Err(error) => {
                                                ui.set_status_message(format!("Error: creatures: {error}"));
                                                Vec::new()
                                            }
                                        };
                                        scene_creatures_key = Some(key);
                                    }
                                    let inspections: Result<Vec<_>, String> = scene_creatures
                                        .iter()
                                        .map(|named| inspect_creature(&named.name, &named.creature, &scene, None))
                                        .collect();
                                    match inspections {
                                        Ok(list) => ui.set_creature_inspections(list),
                                        Err(error) => {
                                            ui.set_status_message(format!("Error: creature inspector: {error}"));
                                            ui.set_creature_inspections(Vec::new());
                                        }
                                    }
                                }
                                None => {
                                    scene_creatures.clear();
                                    scene_creatures_key = None;
                                    ui.set_creature_inspections(Vec::new());
                                }
                            }
                            ui.set_creature_stale(false);
                        }
                        if runtime_metrics_server.is_some() {
                        voxel_physics_lab::runtime_metrics::publish("app", serde_json::json!({
                            "scene_name":active_scene_document.as_ref().map(|doc| &doc.name),
                            "scene_path":active_scene_document_path,
                            "fps":fps,"fps_sample_valid":fps>0.0,"paused":ui.paused,
                            "requested_time_scale":ui.simulation_time_scale,
                            "simulated_seconds":simulated,"measured_simulated_seconds_per_wall_second":speed,
                            "simulation_rate_sample_valid":rate_valid,"generation":generation,
                            "backlog_seconds":physics.map_or(0.0,PhysicsWorker::backlog_seconds),
                            "backlog_scope":"pending_plus_unacknowledged_requested_time",
                            "worker_failure":physics.and_then(PhysicsWorker::failure_message),
                            "camera_position_m":camera.position().to_array(),
                            "physics_mode":if physics.is_some(){"worker"}else{"authoring"},
                            "automatic_aggregation_disabled":std::env::var_os("VOXEL_DISABLE_AUTO_AGGREGATION").is_some(),
                            "cuda_contact_operator_requested":std::env::var("VOXEL_CUDA_CONTACT_OPERATOR").as_deref()==Ok("1"),
                            "cuda_contact_ic0_requested":std::env::var("VOXEL_CUDA_CONTACT_IC0").as_deref()==Ok("1"),
                        }));
                        if physics.is_none() {
                            let key=(scene.positions.as_ptr() as usize,scene.render_revision);
                            if last_authoring_metrics_key!=Some(key) {
                                world_physics::publish_scene_diagnostics(&scene,serde_json::json!({"source":"authoring"}));
                                last_authoring_metrics_key=Some(key);
                            }
                        } else { last_authoring_metrics_key=None; }
                        }
                        previous_runtime_simulated=simulated;
                        previous_runtime_generation=generation;
                        last_runtime_publish=Instant::now();
                    }
                    if fps_time.elapsed() >= Duration::from_secs(1) {
                        fps = frames as f32 / fps_time.elapsed().as_secs_f32();
                        frames = 0;
                        fps_time = Instant::now();
                        let tool_axis_position = tool
                            .as_ref()
                            .map_or(f32::NAN, |t| t.position.dot(ui.direction().as_dvec3()) as f32);
                        let tool_speed = tool.as_ref().map_or(0.0, |t| t.velocity.length());
                        let live_rate = if live_wall_s > 0.0 {
                            live_simulated_s / live_wall_s
                        } else {
                            0.0
                        };
                        let mode_title = match mode {
                            AppMode::Interactive => {
                                if ui.paused {
                                    "LIVE PAUSED".to_owned()
                                } else {
                                    format!("LIVE {:.3}x", live_rate)
                                }
                            }
                            AppMode::Recording => format!(
                                "RECORD {} {:.1}/{:.1}s | LIVE {:.3}x",
                                if ui.paused { "PAUSED" } else { "LIVE" },
                                live_simulated_s,
                                RECORDING_DURATION_SECONDS,
                                live_rate
                            ),
                            AppMode::Playback => {
                                let state = if ui.paused { "PAUSED" } else { "1.00x" };
                                let frame_count = playback_frames.len();
                                let loop_duration = playback_frames
                                    .last()
                                    .map_or(PLAYBACK_FRAME_INTERVAL_SECONDS, |frame| {
                                        frame.time_s + PLAYBACK_FRAME_INTERVAL_SECONDS
                                    });
                                format!(
                                    "PLAYBACK {state} {:.1}/{:.1}s | {frame_count} frames",
                                    playback_time_s.min(loop_duration),
                                    loop_duration,
                                )
                            }
                        };
                        if let Some(physics) = world_physics.as_ref() {
                            window.set_title(&format!("Voxel lab | 20 x 20 m | 1cm physics | {:.2}s simulated | backlog {:.2}s | {:.1}ms/step | broken {} | {:.1} FPS | Ctrl crouch | F6 restart",
                                physics.frame.simulated_seconds,physics.backlog_seconds(),physics.frame.step_ms,
                                physics.frame.broken,fps));
                        } else if generated_world.is_some() {
                            let world = generated_world.as_ref().unwrap();
                            window.set_title(&format!(
                                "Voxel lab | 10 km2 map | seed {} | 1cm REV / overview LOD | age {:.0} y | {} | {:.1} FPS",
                                ui.seed_text,
                                world.age_years,
                                world.vegetation_summary(),
                                fps,
                            ));
                        } else if let Some(document) = active_model_document.as_ref() {
                            let state = if ui.paused { "PAUSED" } else { "RUNNING" };
                            window.set_title(&format!(
                                "{} | Model {} | {} | {} REVs | {:.1} FPS",
                                if standalone_editor_app { "Voxel Model Editor" } else { "Voxel Editor" },
                                document.name,
                                state,
                                scene.voxels.len(),
                                fps,
                            ));
                        } else if let Some(document) = active_scene_document.as_ref() {
                            let state = if ui.paused { "PAUSED" } else { "RUNNING" };
                            let count = authored_physics.as_ref().map_or(scene.voxels.len(), |physics| physics.frame.cells.len());
                            let backlog = authored_physics.as_ref().map_or(0.0, PhysicsWorker::backlog_seconds);
                            window.set_title(&format!(
                                "Voxel Editor | {} | {} | {} REVs | {:.3}s simulated | {:.1} ms/step | backlog {:.2}s | {:.1} FPS",
                                document.name, state, count, live_simulated_s, sim_ms, backlog, fps,
                            ));
                        } else if two_cube_demo {
                            window.set_title(&format!(
                                "Voxel lab | 2x2x2 oak cubes | {mode_title} | {:.2}s simulated | {:.1} FPS | broken {}",
                                live_simulated_s, fps, scene.broken
                            ));
                        } else if friction_road_demo {
                            window.set_title(&format!(
                                "Voxel lab | oak 3x3x3 rolling on concrete | {mode_title} | {:.2}s simulated | friction heat {:.3} mJ | {:.1} FPS",
                                live_simulated_s, scene.energy.contact_friction_heat * 1000.0, fps
                            ));
                        } else {
                            window.set_title(&format!(
                                "Voxel lab | {mode_title} | {:.1} FPS | {:.1} ms/step | tool {:.2} m/s axis {:.3} | broken {}",
                                fps, sim_ms, tool_speed, tool_axis_position, scene.broken
                            ));
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                if !window.is_minimized().unwrap_or(false) {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    })?;
    Ok(())
}
