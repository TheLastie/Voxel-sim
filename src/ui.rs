use crate::scene::SceneConfig;
use crate::tool::Shape;
use glam::Vec3;
use std::collections::BTreeMap;
use voxel_physics_lab::scene_welding::ThermalBondingSettings;
use winit::keyboard::KeyCode;
const MAX_PUSH_FORCE_N: f32 = 15_000.0;
const MAX_SIMULATION_TIME_SCALE: f32 = 31_557_600.0;
const RIG_HIERARCHY_TOP: f32 = 282.0;
const RIG_HIERARCHY_ROW_HEIGHT: f32 = 22.0;
const RIG_ACTION_ROW_HEIGHT: f32 = 27.0;
const ANIMATION_FRAME_RATE: f32 = 30.0;
const PROJECT_DOCK_HEIGHT: f32 = 200.0;
const ANIMATION_TIMELINE_COLLAPSED_HEIGHT: f32 = 30.0;
const ANIMATION_TIMELINE_EXPANDED_HEIGHT: f32 = 128.0;

fn animation_slider_geometry(inspector_left: f32) -> (f32, f32) {
    (18.0, (inspector_left - 36.0).max(1.0))
}
pub use voxel_physics_lab::ui_text::Rect;
pub struct Ui {
    /// Runs as the standalone authoring app instead of the physics lab.
    pub standalone_editor_app: bool,
    pub speed: f32,
    pub start: Vec3,
    pub direction: Vec3,
    pub force_newtons: f32,
    pub paused: bool,
    pub show_bonds: bool,
    pub selected: Shape,
    pub seed_text: String,
    pub seed_editing: bool,
    pub world_active: bool,
    pub menu_open: bool,
    pub settings_open: bool,
    pub settings_tab: u8,
    settings_from_toolbar: bool,
    pub settings_bindings: BTreeMap<String, String>,
    pub settings_rebind_action: Option<String>,
    pub settings_mcp_enabled: bool,
    /// Shows the scene-editor workspace; Play switches back to first-person.
    pub editor_mode: bool,
    pub selected_hierarchy: usize,
    pub project_category: usize,
    pub file_menu_open: bool,
    pub mesh_import_open: bool,
    pub mesh_import_filename: Option<String>,
    pub mesh_import_scale: f32,
    pub mesh_import_filled: bool,
    pub mesh_import_busy: bool,
    pub mesh_import_preview: Option<MeshImportPreviewInfo>,
    pub mesh_import_error: Option<String>,
    pub scene_document_available: bool,
    pub model_document_available: bool,
    pub can_undo: bool,
    pub can_redo: bool,
    pub active_scene_name: Option<String>,
    pub scene_model_names: Vec<String>,
    pub scene_hierarchy_scroll: usize,
    scene_inspector_scroll: f32,
    inspector_scroll_dragging: bool,
    scene_physics: ScenePhysicsValues,
    scene_inspector_collapsed: [bool; 5],
    view_right: Vec3,
    view_up: Vec3,
    pub selected_scene_model: Option<usize>,
    pub selected_scene_model_info: Option<SceneModelInspectorInfo>,
    pub selected_scene_initial_temperature_kelvin: Option<f32>,
    pub scene_thermal_bonding: Option<ThermalBondingSettings>,
    pub scene_selection_mode: SceneSelectionMode,
    scene_input: Option<SceneInputState>,
    welding_input: Option<WeldingInputState>,
    scene_physics_input: Option<ScenePhysicsInputState>,
    /// Scene-object translation increment in 1 cm REV cells.
    pub scene_move_step_cells: i32,
    pub active_model_name: Option<String>,
    pub model_voxel_count: usize,
    pub model_material_count: usize,
    pub rig_bone_names: Vec<String>,
    pub selected_bone: Option<usize>,
    pub animation_clip_count: usize,
    pub selected_animation_clip: Option<usize>,
    pub selected_animation_time_seconds: f32,
    pub animation_time_step_seconds: f32,
    pub animation_duration_seconds: f32,
    pub animation_frame_rate: f32,
    pub animation_keyframe_times: Vec<f32>,
    pub animation_timeline_expanded: bool,
    pub animation_playing: bool,
    /// Playback rate multiplier, independent of simulation time scaling.
    pub animation_playback_speed: f32,
    pub show_rig_overlay: bool,
    timeline_dragging: bool,
    pub rig_edit_mode: RigEditMode,
    pub rig_hierarchy_scroll: usize,
    pub selected_voxel: Option<usize>,
    pub selected_voxel_position: Option<[f64; 3]>,
    pub selected_voxel_mass_kg: Option<f32>,
    pub selected_voxel_indices: Vec<usize>,
    pub selected_bounds_m: Option<[[f64; 3]; 2]>,
    pub selected_dimensions_cm: Option<[u32; 3]>,
    pub selection_diagram_xz: [bool; 64],
    pub editor_tool: EditorTool,
    /// Odd edge length of the sculpt brush, measured in 1 cm REVs.
    pub brush_size_cells: usize,
    pub status_message: Option<String>,
    /// Requested simulation seconds per wall-clock second. 1.0 requests real time.
    pub simulation_time_scale: f32,
    /// Observed physical clock rate; distinct from the requested speed.
    pub measured_simulation_rate: Option<f64>,
    /// Latest completed worker cost for 10 ms of simulated physical time.
    pub physics_cost_ms_per_10ms: Option<f64>,
    pub walk_speed_mps: f32,
    pub look_sensitivity: f32,
    paused_before_menu: bool,
    canvas_scale: f32,
    canvas_offset: [f32; 2],
    canvas_width: f32,
    canvas_height: f32,
}

pub struct EditorInfo<'a> {
    pub physics_cells: usize,
    pub physics_enabled: bool,
    pub world_age_years: f32,
    pub seed: &'a str,
    pub vegetation_summary: &'a str,
}

pub enum UiAction {
    Shape(Shape),
    GenerateWorld(String),
    Resume,
    Settings,
    SelectSettingsTab(u8),
    BeginRebind(String),
    ToggleMcp(bool),
    Back,
    Exit,
    EnterPlay,
    ReturnToEditor,
    ToggleSimulationPause,
    ToggleBonds,
    NewScene,
    OpenScene,
    SaveScene,
    /// Write the paused physical state beside the scene as `<name>.vxstate`.
    SavePhysicalState,
    /// Continue from the `<name>.vxstate` saved beside the scene.
    LoadPhysicalState,
    NewModel,
    OpenModel,
    SaveModel,
    SaveModelAs,
    ExportModelObj,
    AddModelToScene,
    SetEditorTool(EditorTool),
    AdjustBrushSize(i8),
    FrameSelection,
    UndoVoxelModel,
    RedoVoxelModel,
    MoveSelectedPart {
        axis: usize,
        direction: i8,
    },
    RotateSelectedPart {
        direction: i8,
    },
    DuplicateSelectedPart,
    DeleteSelectedPart,
    SelectSceneModel(usize),
    SetSceneMoveStep(i32),
    CreateRootBone,
    CreateChildBone {
        parent_bone: usize,
    },
    GenerateAutomaticWeights,
    CreateAnimationClip,
    InsertBoneKeyframe {
        bone_index: usize,
    },
    SelectBone(usize),
    AdjustSelectedVoxelBoneWeight {
        bone_index: usize,
        delta: f32,
    },
    SelectAnimationClip(usize),
    SetAnimationFrame(usize),
    ToggleAnimationPlayback,
    StepAnimationTime {
        direction: i8,
        step_seconds: f32,
    },
    SetRigEditMode(RigEditMode),
    AdjustSelectedBone {
        bone_index: usize,
        axis: usize,
        delta: f32,
        rotation: bool,
    },
    MoveSelectedSceneModel {
        axis: usize,
        direction: i8,
    },
    RotateSelectedSceneModel {
        axis: usize,
        direction: i8,
    },
    DuplicateSelectedSceneModel,
    RemoveSelectedSceneModel,
    ImportMesh,
    BuildMeshPreview {
        scale: f32,
        filled: bool,
    },
    CommitMeshImport {
        add_to_scene: bool,
    },
    CancelMeshImport,
    SetSceneSelectionMode(SceneSelectionMode),
    SetScenePosition {
        value: [f64; 3],
    },
    SetSceneRotation {
        degrees: [f32; 3],
    },
    SetSceneVelocity {
        value: [f32; 3],
    },
    SetSceneForce {
        value: [f32; 3],
    },
    SetSceneInitialTemperature {
        kelvin: Option<f32>,
    },
    SetSceneThermalBonding(Option<ThermalBondingSettings>),
    SetScenePhysics {
        fixed_dt_seconds: f32,
        substeps: u32,
        gravity: [f32; 3],
        restitution: f32,
        friction_scale: f32,
        solver_iterations: usize,
        solver_tolerance: f32,
    },
    CycleSceneMaterial,
    LoadSceneMaterial,
    SaveSceneMaterial,
    CreateSceneMaterial,
    CreateImpactScene {
        side: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshImportPreviewInfo {
    pub watertight: bool,
    /// Dimensions after applying the selected scale, in metres.
    pub dimensions_m: [f32; 3],
    pub cell_count: usize,
}

/// Authored transform data for the currently selected scene model instance.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneModelInspectorInfo {
    pub name: String,
    pub translation_m: [f64; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
    pub velocity_m_s: [f32; 3],
    pub force_n: [f32; 3],
    pub material_name: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneSelectionMode {
    #[default]
    Object,
    Voxel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneInputTarget {
    Position,
    Rotation,
    Velocity,
    Force,
    InitialTemperature,
}

#[derive(Clone, Debug)]
struct WeldingInputState {
    values: [String; 6],
    field: usize,
    editing: bool,
}

#[derive(Clone, Debug)]
struct SceneInputState {
    target: SceneInputTarget,
    values: [String; 3],
    axis: usize,
    editing: bool,
}

#[derive(Clone, Debug)]
struct ScenePhysicsInputState {
    values: [String; 9],
    field: usize,
    editing: bool,
}

#[derive(Clone, Copy, Debug)]
struct ScenePhysicsValues {
    fixed_dt: f32,
    substeps: usize,
    gravity: [f32; 3],
    restitution: f32,
    friction_scale: f32,
    iterations: usize,
    tolerance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneInspectorSection {
    Transform,
    Dynamics,
    Material,
    Welding,
    Simulation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneInspectorItem {
    Actions,
    Section(SceneInspectorSection),
    Position,
    Rotation,
    Bounds,
    Velocity,
    Force,
    Temperature,
    MaterialCycle,
    MaterialActions,
    WeldingToggle,
    WeldingConfigure,
    PhysicsField(usize),
    PhysicsApply,
    Hint,
    InputHint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SceneInspectorRow {
    item: SceneInspectorItem,
    top: f32,
    height: f32,
}

impl Default for ScenePhysicsValues {
    fn default() -> Self {
        let config = SceneConfig::default();
        Self {
            fixed_dt: config.fixed_dt,
            substeps: config.substeps,
            gravity: config.gravity,
            restitution: config.fracture.restitution,
            friction_scale: config.fracture.friction,
            iterations: config.iterations,
            tolerance: config.solver_relative_tolerance,
        }
    }
}

fn scene_physics_action(values: &[String; 9]) -> Result<UiAction, &'static str> {
    let parse = |index: usize| {
        values[index]
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite())
    };
    let fixed_dt_seconds = parse(0)
        .filter(|value| *value > 0.0)
        .ok_or("Fixed dt must be finite and greater than zero.")?;
    let substeps = values[1]
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or("Substeps must be a positive whole number.")?;
    let gravity = [parse(2), parse(3), parse(4)];
    if gravity.iter().any(Option::is_none) {
        return Err("Gravity components must be finite numbers.");
    }
    let gravity = gravity.map(Option::unwrap);
    let restitution = parse(5)
        .filter(|value| (0.0..=1.0).contains(value))
        .ok_or("Restitution must be between 0 and 1.")?;
    let friction_scale = parse(6)
        .filter(|value| *value >= 0.0)
        .ok_or("Friction scale must be finite and nonnegative.")?;
    let solver_iterations = values[7]
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or("Solver iterations must be a positive whole number.")?;
    let solver_tolerance = parse(8)
        .filter(|value| *value > 0.0)
        .ok_or("Solver tolerance must be finite and greater than zero.")?;
    Ok(UiAction::SetScenePhysics {
        fixed_dt_seconds,
        substeps,
        gravity,
        restitution,
        friction_scale,
        solver_iterations,
        solver_tolerance,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EditorTool {
    #[default]
    Select,
    AddVoxel,
    EraseVoxel,
    AddBox,
    AddSphere,
    AddCylinder,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RigEditMode {
    #[default]
    Move,
    Rotate,
    Weights,
}

#[derive(Clone, Copy)]
pub enum ReticleState {
    PhysicsOff,
    Idle,
    Target,
    Holding,
}

impl Ui {
    pub fn set_measured_simulation_rate(&mut self, rate: Option<f64>) {
        self.measured_simulation_rate = rate.filter(|value| value.is_finite() && *value >= 0.0);
    }

    pub fn set_physics_cost_ms_per_10ms(&mut self, cost: Option<f64>) {
        self.physics_cost_ms_per_10ms = cost.filter(|value| value.is_finite() && *value >= 0.0);
    }

    pub fn new(speed: f32, start: Vec3, direction: Vec3) -> Self {
        Self {
            standalone_editor_app: false,
            speed,
            start,
            direction: direction.normalize(),
            force_newtons: 0.0,
            paused: false,
            show_bonds: false,
            selected: Shape::Wedge,
            seed_text: "1001".to_owned(),
            seed_editing: false,
            world_active: false,
            menu_open: false,
            settings_open: false,
            settings_tab: 0,
            settings_from_toolbar: false,
            settings_bindings: crate::app::editor_settings::default_bindings(),
            settings_rebind_action: None,
            settings_mcp_enabled: true,
            editor_mode: false,
            selected_hierarchy: 0,
            project_category: 0,
            file_menu_open: false,
            mesh_import_open: false,
            mesh_import_filename: None,
            mesh_import_scale: 1.0,
            mesh_import_filled: false,
            mesh_import_busy: false,
            mesh_import_preview: None,
            mesh_import_error: None,
            scene_document_available: false,
            model_document_available: false,
            can_undo: false,
            can_redo: false,
            active_scene_name: None,
            scene_model_names: Vec::new(),
            scene_hierarchy_scroll: 0,
            scene_inspector_scroll: 0.0,
            inspector_scroll_dragging: false,
            scene_physics: ScenePhysicsValues::default(),
            scene_inspector_collapsed: [false, false, false, true, true],
            view_right: Vec3::X,
            view_up: Vec3::Z,
            selected_scene_model: None,
            selected_scene_model_info: None,
            selected_scene_initial_temperature_kelvin: None,
            scene_thermal_bonding: None,
            scene_selection_mode: SceneSelectionMode::Object,
            scene_input: None,
            welding_input: None,
            scene_physics_input: None,
            scene_move_step_cells: 10,
            active_model_name: None,
            model_voxel_count: 0,
            model_material_count: 0,
            rig_bone_names: Vec::new(),
            selected_bone: None,
            animation_clip_count: 0,
            selected_animation_clip: None,
            selected_animation_time_seconds: 0.0,
            animation_time_step_seconds: 0.1,
            animation_duration_seconds: 1.0,
            animation_frame_rate: 30.0,
            animation_keyframe_times: Vec::new(),
            animation_timeline_expanded: false,
            animation_playing: false,
            animation_playback_speed: 1.0,
            show_rig_overlay: false,
            timeline_dragging: false,
            rig_edit_mode: RigEditMode::Move,
            rig_hierarchy_scroll: 0,
            selected_voxel: None,
            selected_voxel_position: None,
            selected_voxel_mass_kg: None,
            selected_voxel_indices: Vec::new(),
            selected_bounds_m: None,
            selected_dimensions_cm: None,
            selection_diagram_xz: [false; 64],
            editor_tool: EditorTool::Select,
            brush_size_cells: 3,
            status_message: None,
            simulation_time_scale: 1.0,
            physics_cost_ms_per_10ms: None,
            measured_simulation_rate: None,
            walk_speed_mps: 4.5,
            look_sensitivity: 0.005,
            paused_before_menu: false,
            canvas_scale: 1.0,
            canvas_offset: [0.0; 2],
            canvas_width: 1280.0,
            canvas_height: 720.0,
        }
    }
    pub fn direction(&self) -> Vec3 {
        self.direction
    }
    pub fn start(&self) -> Vec3 {
        self.start
    }
    pub fn click(&mut self, x: f32, y: f32) -> Option<UiAction> {
        let (x, mut y) = self.to_canvas(x, y);
        if let Some(input) = self.welding_input.as_mut() {
            let left = (self.canvas_width - 500.0) * 0.5;
            let top = (self.canvas_height - 430.0) * 0.5;
            if (left..=left + 500.0).contains(&x) && (top..=top + 430.0).contains(&y) {
                for field in 0..6 {
                    let row_top = top + 88.0 + field as f32 * 42.0;
                    if (left + 20.0..=left + 480.0).contains(&x)
                        && (row_top..=row_top + 34.0).contains(&y)
                    {
                        input.field = field;
                        input.editing = false;
                        return None;
                    }
                }
                if (left + 20.0..=left + 480.0).contains(&x)
                    && (top + 362.0..=top + 408.0).contains(&y)
                {
                    let parsed = input.values[1..]
                        .iter()
                        .map(|value| value.parse::<f64>().ok().filter(|v| v.is_finite()))
                        .collect::<Vec<_>>();
                    if parsed.iter().any(Option::is_none) || input.values[0].trim().is_empty() {
                        self.status_message =
                            Some("Enter a material name and five finite thresholds.".into());
                        return None;
                    }
                    let settings = ThermalBondingSettings {
                        material_name: input.values[0].trim().to_owned(),
                        minimum_mass_fraction: parsed[0].unwrap() as f32,
                        minimum_temperature_kelvin: parsed[1].unwrap() as f32,
                        minimum_impact_energy_j: parsed[2].unwrap(),
                        minimum_contact_pressure_pa: parsed[3].unwrap() as f32,
                        minimum_dwell_seconds: parsed[4].unwrap() as f32,
                    };
                    self.welding_input = None;
                    return Some(UiAction::SetSceneThermalBonding(Some(settings)));
                }
                return None;
            }
            self.welding_input = None;
            return None;
        }
        let dock_top = self.editor_dock_top();
        if self.world_active && self.editor_mode && !self.menu_open {
            if self.mesh_import_open {
                let (left, top) = mesh_import_modal_origin(self.canvas_width, self.canvas_height);
                if (left..=left + 560.0).contains(&x) && (top..=top + 430.0).contains(&y) {
                    if (left + 24.0..=left + 286.0).contains(&x)
                        && (top + 121.0..=top + 151.0).contains(&y)
                    {
                        let fraction = ((x - (left + 24.0)) / 262.0).clamp(0.0, 1.0);
                        self.mesh_import_scale = 0.1 + fraction * 9.9;
                        self.mesh_import_preview = None;
                        self.mesh_import_error = None;
                        return None;
                    }
                    if (left + 310.0..=left + 418.0).contains(&x)
                        && (top + 83.0..=top + 115.0).contains(&y)
                    {
                        self.mesh_import_filled = false;
                        self.mesh_import_preview = None;
                        self.mesh_import_error = None;
                        return None;
                    }
                    if (left + 422.0..=left + 536.0).contains(&x)
                        && (top + 83.0..=top + 115.0).contains(&y)
                    {
                        self.mesh_import_filled = true;
                        self.mesh_import_preview = None;
                        self.mesh_import_error = None;
                        return None;
                    }
                    if (left + 24.0..=left + 536.0).contains(&x)
                        && (top + 164.0..=top + 198.0).contains(&y)
                        && !self.mesh_import_busy
                    {
                        self.mesh_import_preview = None;
                        self.mesh_import_error = None;
                        return Some(UiAction::BuildMeshPreview {
                            scale: self.mesh_import_scale,
                            filled: self.mesh_import_filled,
                        });
                    }
                    if !self.mesh_import_busy && self.mesh_import_preview.is_some() {
                        if (left + 24.0..=left + 248.0).contains(&x)
                            && (top + 381.0..=top + 415.0).contains(&y)
                        {
                            return Some(UiAction::CommitMeshImport {
                                add_to_scene: false,
                            });
                        }
                        if self.scene_document_available
                            && (left + 260.0..=left + 438.0).contains(&x)
                            && (top + 381.0..=top + 415.0).contains(&y)
                        {
                            return Some(UiAction::CommitMeshImport { add_to_scene: true });
                        }
                    }
                    if (left + 448.0..=left + 536.0).contains(&x)
                        && (top + 381.0..=top + 415.0).contains(&y)
                    {
                        self.mesh_import_open = false;
                        self.mesh_import_busy = false;
                        return Some(UiAction::CancelMeshImport);
                    }
                }
                // The modal owns the entire pointer surface; clicking outside it
                // must never activate editor controls underneath.
                return None;
            }
            if self.file_menu_open {
                if (208.0..=292.0).contains(&x) && (0.0..=28.0).contains(&y) {
                    self.open_settings_from_toolbar();
                    return Some(UiAction::Settings);
                }
                self.file_menu_open = false;
                if (116.0..=356.0).contains(&x) {
                    let row = ((y - 30.0) / 29.0).floor() as i32;
                    return match row {
                        0 if (30.0..=58.0).contains(&y) => Some(UiAction::NewScene),
                        1 if (59.0..=87.0).contains(&y) => Some(UiAction::OpenScene),
                        2 if (88.0..=116.0).contains(&y) => Some(UiAction::NewModel),
                        3 if (117.0..=145.0).contains(&y) => Some(UiAction::OpenModel),
                        4 if (146.0..=174.0).contains(&y)
                            && (self.scene_document_available || self.model_document_available) =>
                        {
                            Some(UiAction::ImportMesh)
                        }
                        5 if (175.0..=203.0).contains(&y) && self.scene_document_available => {
                            Some(UiAction::AddModelToScene)
                        }
                        6 if (204.0..=232.0).contains(&y) && self.scene_document_available => {
                            Some(UiAction::SaveScene)
                        }
                        7 if (233.0..=261.0).contains(&y) && self.model_document_available => {
                            Some(UiAction::SaveModel)
                        }
                        8 if (262.0..=290.0).contains(&y) && self.model_document_available => {
                            Some(UiAction::SaveModelAs)
                        }
                        9 if (291.0..=319.0).contains(&y) && self.model_document_available => {
                            Some(UiAction::ExportModelObj)
                        }
                        10 if (320.0..=348.0).contains(&y) && self.scene_document_available => {
                            Some(UiAction::CreateImpactScene { side: 16 })
                        }
                        11 if (349.0..=377.0).contains(&y) && self.scene_document_available => {
                            Some(UiAction::CreateImpactScene { side: 74 })
                        }
                        12 if (378.0..=406.0).contains(&y) && self.scene_document_available => {
                            Some(UiAction::SavePhysicalState)
                        }
                        13 if (407.0..=435.0).contains(&y) && self.scene_document_available => {
                            Some(UiAction::LoadPhysicalState)
                        }
                        _ => None,
                    };
                }
                if (160.0..=202.0).contains(&x) && (0.0..=28.0).contains(&y) {
                    self.file_menu_open = true;
                }
                return None;
            }
            if (160.0..=202.0).contains(&x) && (0.0..=28.0).contains(&y) {
                self.file_menu_open = true;
                return None;
            }
            if (208.0..=292.0).contains(&x) && (0.0..=28.0).contains(&y) {
                self.open_settings_from_toolbar();
                return Some(UiAction::Settings);
            }
            if self.scene_document_available
                && (370.0..=434.0).contains(&x)
                && (0.0..=28.0).contains(&y)
            {
                self.scene_selection_mode = SceneSelectionMode::Object;
                return Some(UiAction::SetSceneSelectionMode(SceneSelectionMode::Object));
            }
            if self.scene_document_available
                && (438.0..=502.0).contains(&x)
                && (0.0..=28.0).contains(&y)
            {
                self.scene_selection_mode = SceneSelectionMode::Voxel;
                return Some(UiAction::SetSceneSelectionMode(SceneSelectionMode::Voxel));
            }
            if (12.0..=68.0).contains(&x) && (34.0..=61.0).contains(&y) {
                return self
                    .model_document_available
                    .then_some(UiAction::SetEditorTool(EditorTool::Select));
            }
            if (74.0..=130.0).contains(&x) && (34.0..=61.0).contains(&y) {
                if self.model_document_available {
                    return Some(UiAction::SetEditorTool(EditorTool::AddVoxel));
                }
                return None;
            }
            if (136.0..=192.0).contains(&x) && (34.0..=61.0).contains(&y) {
                if self.model_document_available {
                    return Some(UiAction::SetEditorTool(EditorTool::EraseVoxel));
                }
                return None;
            }
            if (198.0..=254.0).contains(&x) && (34.0..=61.0).contains(&y) {
                return self
                    .model_document_available
                    .then_some(UiAction::SetEditorTool(EditorTool::AddBox));
            }
            if (260.0..=316.0).contains(&x) && (34.0..=61.0).contains(&y) {
                return self
                    .model_document_available
                    .then_some(UiAction::SetEditorTool(EditorTool::AddSphere));
            }
            if (322.0..=378.0).contains(&x) && (34.0..=61.0).contains(&y) {
                return self
                    .model_document_available
                    .then_some(UiAction::SetEditorTool(EditorTool::AddCylinder));
            }
            if (384.0..=440.0).contains(&x) && (34.0..=61.0).contains(&y) {
                if self.model_document_available
                    || (self.scene_document_available && !self.scene_model_names.is_empty())
                {
                    return Some(UiAction::FrameSelection);
                }
                return None;
            }
            if (446.0..=502.0).contains(&x) && (34.0..=61.0).contains(&y) {
                return self.can_undo.then_some(UiAction::UndoVoxelModel);
            }
            if (508.0..=564.0).contains(&x) && (34.0..=61.0).contains(&y) {
                return self.can_redo.then_some(UiAction::RedoVoxelModel);
            }
            if self.standalone_editor_app && self.model_document_available {
                if (570.0..=610.0).contains(&x) && (34.0..=61.0).contains(&y) {
                    return Some(UiAction::AdjustBrushSize(-2));
                }
                if (744.0..=784.0).contains(&x) && (34.0..=61.0).contains(&y) {
                    return Some(UiAction::AdjustBrushSize(2));
                }
            }
            if self.model_document_available {
                let timeline_top = self.animation_timeline_top();
                let inspector_left = self.canvas_width - 250.0;
                if (timeline_top..=self.canvas_height).contains(&y)
                    && (0.0..inspector_left).contains(&x)
                {
                    if (timeline_top..=timeline_top + 30.0).contains(&y) {
                        if (92.0..=158.0).contains(&x) && self.selected_animation_clip.is_some() {
                            return Some(UiAction::ToggleAnimationPlayback);
                        }
                        if self.selected_animation_clip.is_some() && (480.0..=516.0).contains(&x) {
                            self.animation_playback_speed =
                                (self.animation_playback_speed * 0.5).max(0.25);
                            return None;
                        }
                        if self.selected_animation_clip.is_some() && (624.0..=660.0).contains(&x) {
                            self.animation_playback_speed =
                                (self.animation_playback_speed * 2.0).min(8.0);
                            return None;
                        }
                        if (inspector_left - 118.0..=inspector_left - 48.0).contains(&x) {
                            self.show_rig_overlay = !self.show_rig_overlay;
                            return None;
                        }
                        if x <= 88.0 || x >= inspector_left - 42.0 {
                            self.animation_timeline_expanded = !self.animation_timeline_expanded;
                        }
                        return None;
                    }
                    if self.animation_timeline_expanded
                        && self.selected_animation_clip.is_some()
                        && (timeline_top + 58.0..=timeline_top + 98.0).contains(&y)
                    {
                        let (left, width) = animation_slider_geometry(inspector_left);
                        self.animation_playing = false;
                        self.timeline_dragging = true;
                        self.scrub_animation_time(x, left, width);
                        return Some(UiAction::SetAnimationFrame(self.animation_current_frame()));
                    }
                    return None;
                }
            }
            if self.model_document_available
                && (0.0..=235.0).contains(&x)
                && (RIG_HIERARCHY_TOP..dock_top).contains(&y)
            {
                let row = self.rig_hierarchy_scroll
                    + ((y - RIG_HIERARCHY_TOP) / RIG_HIERARCHY_ROW_HEIGHT).floor() as usize;
                if self.rig_bone_names.get(row).is_some() {
                    self.selected_bone = Some(row);
                    self.show_rig_overlay = true;
                    return Some(UiAction::SelectBone(row));
                }
                return None;
            }
            if self.model_document_available {
                let inspector_left = self.canvas_width - 250.0;
                let columns = [
                    inspector_left + 15.0,
                    inspector_left + 87.0,
                    inspector_left + 159.0,
                ];
                let child_bone = self
                    .selected_bone
                    .filter(|&index| index < self.rig_bone_names.len());
                let can_adjust_weight =
                    self.selected_voxel_indices.len() == 1 && child_bone.is_some();
                let can_edit_clip = self
                    .selected_animation_clip
                    .is_some_and(|clip| clip < self.animation_clip_count);
                let can_edit_pose = child_bone.is_some() && can_edit_clip;
                if (708.0..=735.0).contains(&y) {
                    if (columns[0]..=columns[0] + 66.0).contains(&x) {
                        let next_mode = match self.rig_edit_mode {
                            RigEditMode::Move => RigEditMode::Rotate,
                            RigEditMode::Rotate => RigEditMode::Weights,
                            RigEditMode::Weights => RigEditMode::Move,
                        };
                        let can_switch_mode = child_bone.is_some()
                            && (self.rig_edit_mode != RigEditMode::Rotate || can_adjust_weight);
                        if can_switch_mode {
                            self.rig_edit_mode = next_mode;
                            return Some(UiAction::SetRigEditMode(next_mode));
                        }
                        return None;
                    }
                    return None;
                }
                if (739.0..=766.0).contains(&y) {
                    let compact_columns = [
                        inspector_left + 15.0,
                        inspector_left + 70.0,
                        inspector_left + 125.0,
                        inspector_left + 180.0,
                    ];
                    if self.rig_edit_mode == RigEditMode::Weights {
                        if can_adjust_weight {
                            if (compact_columns[0]..=compact_columns[0] + 52.0).contains(&x) {
                                return child_bone.map(|bone_index| {
                                    UiAction::AdjustSelectedVoxelBoneWeight {
                                        bone_index,
                                        delta: -0.05,
                                    }
                                });
                            }
                            if (compact_columns[1]..=compact_columns[1] + 52.0).contains(&x) {
                                return child_bone.map(|bone_index| {
                                    UiAction::AdjustSelectedVoxelBoneWeight {
                                        bone_index,
                                        delta: 0.05,
                                    }
                                });
                            }
                        }
                    } else if can_edit_pose {
                        for axis in 0..3 {
                            let left = compact_columns[axis];
                            if (left..=left + 52.0).contains(&x) {
                                let direction = if x < left + 26.0 { -1.0 } else { 1.0 };
                                let rotation = self.rig_edit_mode == RigEditMode::Rotate;
                                let delta = direction * if rotation { 1.0 } else { 0.01 };
                                return child_bone.map(|bone_index| UiAction::AdjustSelectedBone {
                                    bone_index,
                                    axis,
                                    delta,
                                    rotation,
                                });
                            }
                        }
                    }
                    if (compact_columns[3]..=compact_columns[3] + 52.0).contains(&x)
                        && can_edit_pose
                    {
                        return child_bone
                            .map(|bone_index| UiAction::InsertBoneKeyframe { bone_index });
                    }
                    return None;
                }
                let actions = [
                    (
                        columns[0],
                        646.0,
                        Some(UiAction::CreateRootBone),
                        self.rig_bone_names.is_empty(),
                    ),
                    (
                        columns[1],
                        646.0,
                        child_bone.map(|parent_bone| UiAction::CreateChildBone { parent_bone }),
                        child_bone.is_some(),
                    ),
                    (
                        columns[2],
                        646.0,
                        Some(UiAction::GenerateAutomaticWeights),
                        self.model_voxel_count > 0 && !self.rig_bone_names.is_empty(),
                    ),
                    (
                        columns[0],
                        677.0,
                        Some(UiAction::CreateAnimationClip),
                        !self.rig_bone_names.is_empty(),
                    ),
                    (
                        columns[1],
                        677.0,
                        self.selected_animation_clip
                            .map(|clip| UiAction::SelectAnimationClip(clip.saturating_sub(1))),
                        self.animation_clip_count > 0
                            && self.selected_animation_clip.unwrap_or(0) > 0,
                    ),
                    (
                        columns[2],
                        677.0,
                        self.selected_animation_clip.map(|clip| {
                            UiAction::SelectAnimationClip(
                                (clip + 1).min(self.animation_clip_count.saturating_sub(1)),
                            )
                        }),
                        self.animation_clip_count > 0
                            && self
                                .selected_animation_clip
                                .is_some_and(|clip| clip + 1 < self.animation_clip_count),
                    ),
                ];
                for (left, top, action, enabled) in actions {
                    if (left..=left + 66.0).contains(&x)
                        && (top..=top + RIG_ACTION_ROW_HEIGHT).contains(&y)
                    {
                        match action {
                            Some(UiAction::SelectAnimationClip(clip)) => {
                                if enabled {
                                    self.selected_animation_clip = Some(clip);
                                    self.selected_animation_time_seconds = 0.0;
                                    return Some(UiAction::SelectAnimationClip(clip));
                                }
                                return None;
                            }
                            action => return enabled.then_some(action).flatten(),
                        }
                    }
                }
            }
            if self.model_document_available && !self.selected_voxel_indices.is_empty() {
                let inspector_left = self.canvas_width - 250.0;
                let buttons = [
                    (0, -1, inspector_left + 15.0, 405.0),
                    (0, 1, inspector_left + 87.0, 405.0),
                    (1, -1, inspector_left + 159.0, 405.0),
                    (1, 1, inspector_left + 15.0, 437.0),
                    (2, -1, inspector_left + 87.0, 437.0),
                    (2, 1, inspector_left + 159.0, 437.0),
                ];
                for (axis, direction, left, top) in buttons {
                    if (left..=left + 66.0).contains(&x) && (top..=top + 27.0).contains(&y) {
                        return Some(UiAction::MoveSelectedPart { axis, direction });
                    }
                }
                let rotate_buttons = [(-1, inspector_left + 15.0), (1, inspector_left + 129.0)];
                for (direction, left) in rotate_buttons {
                    if (left..=left + 106.0).contains(&x) && (469.0..=497.0).contains(&y) {
                        return Some(UiAction::RotateSelectedPart { direction });
                    }
                }
                if (inspector_left + 15.0..=inspector_left + 235.0).contains(&x)
                    && (501.0..=529.0).contains(&y)
                {
                    return Some(UiAction::DuplicateSelectedPart);
                }
                if (inspector_left + 15.0..=inspector_left + 235.0).contains(&x)
                    && (533.0..=561.0).contains(&y)
                {
                    return Some(UiAction::DeleteSelectedPart);
                }
            }
            let simulation_target = !self.standalone_editor_app
                && (!self.scene_document_available || !self.scene_model_names.is_empty());
            if simulation_target && (570.0..=626.0).contains(&x) && (30.0..=66.0).contains(&y) {
                self.editor_mode = false;
                return Some(UiAction::EnterPlay);
            }
            if simulation_target && (630.0..=686.0).contains(&x) && (30.0..=66.0).contains(&y) {
                return Some(UiAction::ToggleSimulationPause);
            }
            if simulation_target && (690.0..=740.0).contains(&x) && (30.0..=66.0).contains(&y) {
                return Some(UiAction::ToggleBonds);
            }
            if self.scene_document_available
                && (0.0..=235.0).contains(&x)
                && (167.0..=167.0 + ((dock_top - 167.0) / 30.0).floor().max(1.0) * 30.0)
                    .contains(&y)
            {
                let visible_row = ((y - 167.0) / 30.0).floor() as usize;
                let row = self.scene_hierarchy_scroll + visible_row;
                if row < self.scene_model_names.len() {
                    self.selected_scene_model = Some(row);
                    self.selected_scene_model_info = None;
                    return Some(UiAction::SelectSceneModel(row));
                }
            }
            if self.scene_document_available || self.model_document_available {
                let inspector_left = self.canvas_width - 250.0;
                let [track_top, track_height, thumb_top, thumb_height, max_scroll] =
                    inspector_scroll_geometry(self);
                if max_scroll > 0.0
                    && x >= self.canvas_width - 14.0
                    && (track_top..=track_top + track_height).contains(&y)
                {
                    self.inspector_scroll_dragging = true;
                    self.timeline_dragging = true;
                    if !(thumb_top..=thumb_top + thumb_height).contains(&y) {
                        self.set_inspector_scroll_from_thumb(
                            y - thumb_height * 0.5,
                            track_top,
                            track_height,
                            thumb_height,
                            max_scroll,
                        );
                    }
                    return None;
                }
                if self.inspector_scroll_dragging {
                    return None;
                }
                if x >= inspector_left && y >= 144.0 {
                    y += self.scene_inspector_scroll;
                }
            }
            if self.scene_document_available && x >= self.canvas_width - 250.0 && y >= 144.0 {
                return self.click_scene_inspector(x, y);
            }
            if (8.0..=112.0).contains(&x) && (dock_top + 36.0..=dock_top + 62.0).contains(&y) {
                self.project_category = 0;
                return None;
            }
            if (116.0..=220.0).contains(&x) && (dock_top + 36.0..=dock_top + 62.0).contains(&y) {
                self.project_category = 1;
                return None;
            }
            return None;
        }
        if self.world_active && !self.menu_open {
            return None;
        }
        if self.menu_open {
            if self.settings_open {
                if (self.canvas_width - 124.0..=self.canvas_width - 24.0).contains(&x)
                    && (14.0..=54.0).contains(&y)
                {
                    self.leave_settings();
                    return Some(UiAction::Back);
                }
                for (tab, top) in [(0, 124.0), (1, 172.0), (2, 220.0)] {
                    if (16.0..=216.0).contains(&x) && (top..=top + 42.0).contains(&y) {
                        self.settings_tab = tab;
                        self.settings_rebind_action = None;
                        return Some(UiAction::SelectSettingsTab(tab));
                    }
                }
                if self.settings_tab == 0 {
                    for row in 0..3 {
                        let [left, top, width, _] = settings_slider_rect(self.canvas_width, row);
                        if (left - 12.0..=left + width + 12.0).contains(&x)
                            && (top - 18.0..=top + 28.0).contains(&y)
                        {
                            self.set_settings_slider(row, x, left, width);
                            return None;
                        }
                    }
                }
                if self.settings_tab == 1 {
                    let binding_left = self.canvas_width - 184.0;
                    for row in 0..crate::app::editor_settings::REBINDABLE_ACTIONS.len() {
                        let top = settings_control_row_top(row);
                        if (binding_left..=binding_left + 144.0).contains(&x)
                            && (top..=top + 30.0).contains(&y)
                        {
                            return Some(UiAction::BeginRebind(
                                crate::app::editor_settings::REBINDABLE_ACTIONS[row].to_owned(),
                            ));
                        }
                    }
                }
                if self.settings_tab == 2
                    && (self.canvas_width - 220.0..=self.canvas_width - 48.0).contains(&x)
                    && (210.0..=254.0).contains(&y)
                {
                    self.settings_mcp_enabled = !self.settings_mcp_enabled;
                    return Some(UiAction::ToggleMcp(self.settings_mcp_enabled));
                }
                return None;
            }
            let (menu_left, menu_top) = self.menu_origin();
            let local_x = x - menu_left;
            let local_y = y - menu_top;
            if (40.0..=380.0).contains(&local_x) {
                if (193.0..=245.0).contains(&local_y) {
                    self.close_menu();
                    return Some(UiAction::Resume);
                }
                if (257.0..=309.0).contains(&local_y) {
                    self.settings_open = true;
                    return Some(UiAction::Settings);
                }
                if (321.0..=373.0).contains(&local_y) {
                    return Some(UiAction::ReturnToEditor);
                }
                if (385.0..=437.0).contains(&local_y) {
                    return Some(UiAction::Exit);
                }
            }
            return None;
        }
        if !self.world_active {
            let open_scene_y = self.canvas_height - 190.0;
            if (20.0..=280.0).contains(&x) && (open_scene_y..=open_scene_y + 24.0).contains(&y) {
                self.seed_editing = false;
                return Some(UiAction::OpenScene);
            }
            let new_scene_y = self.canvas_height - 162.0;
            if (20.0..=280.0).contains(&x) && (new_scene_y..=new_scene_y + 24.0).contains(&y) {
                self.seed_editing = false;
                return Some(UiAction::NewScene);
            }
        }
        let seed_y = self.canvas_height - 74.0;
        let generate_y = self.canvas_height - 50.0;
        if (20.0..=280.0).contains(&x) && (seed_y..=seed_y + 20.0).contains(&y) {
            self.seed_editing = true;
            return None;
        }
        if (20.0..=280.0).contains(&x) && (generate_y..=generate_y + 24.0).contains(&y) {
            self.seed_editing = false;
            return Some(UiAction::GenerateWorld(self.seed_text.clone()));
        }
        self.seed_editing = false;
        self.pointer(x, y).map(UiAction::Shape)
    }
    pub fn append_seed_text(&mut self, text: &str) {
        if !self.seed_editing {
            return;
        }
        for c in text.chars() {
            if self.seed_text.chars().count() >= 24 {
                break;
            }
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                self.seed_text.push(c);
            }
        }
    }
    pub fn erase_seed_character(&mut self) {
        if self.seed_editing {
            self.seed_text.pop();
        }
    }
    pub fn toggle_pause_menu(&mut self) {
        self.file_menu_open = false;
        if self.menu_open {
            if self.settings_open {
                self.leave_settings();
            } else {
                self.close_menu();
            }
        } else {
            self.seed_editing = false;
            self.paused_before_menu = self.paused;
            self.paused = true;
            self.menu_open = true;
        }
    }
    pub fn close_menu(&mut self) {
        self.menu_open = false;
        self.settings_open = false;
        self.settings_from_toolbar = false;
        self.paused = self.paused_before_menu;
    }
    fn leave_settings(&mut self) {
        self.settings_rebind_action = None;
        if self.settings_from_toolbar {
            self.close_menu();
        } else {
            self.settings_open = false;
        }
    }
    fn set_settings_slider(&mut self, row: usize, x: f32, left: f32, width: f32) {
        let t = ((x - left) / width.max(1.0)).clamp(0.0, 1.0);
        match row {
            0 => self.simulation_time_scale = MAX_SIMULATION_TIME_SCALE.powf(t),
            1 => self.walk_speed_mps = 1.0 + 9.0 * t,
            2 => self.look_sensitivity = 0.001 + 0.011 * t,
            _ => {}
        }
    }
    fn open_settings_from_toolbar(&mut self) {
        self.file_menu_open = false;
        self.seed_editing = false;
        self.paused_before_menu = self.paused;
        self.paused = true;
        self.menu_open = true;
        self.settings_open = true;
        self.settings_tab = 0;
        self.settings_from_toolbar = true;
    }
    pub fn set_scene_document_available(&mut self, available: bool) {
        self.scene_document_available = available;
        if !available {
            self.file_menu_open = false;
        }
    }
    pub fn set_active_scene_document(&mut self, name: Option<String>, models: Vec<String>) {
        if self.active_scene_name != name {
            self.scene_hierarchy_scroll = 0;
            self.scene_inspector_scroll = 0.0;
        }
        self.scene_document_available = name.is_some();
        self.active_scene_name = name;
        self.scene_model_names = models;
        let visible_rows = ((self.editor_dock_top() - 167.0) / 30.0).floor().max(1.0) as usize;
        self.scene_hierarchy_scroll = self
            .scene_hierarchy_scroll
            .min(self.scene_model_names.len().saturating_sub(visible_rows));
        self.selected_scene_model = None;
        self.selected_scene_model_info = None;
        self.project_category = 0;
        self.selected_hierarchy = 0;
        self.active_model_name = None;
        self.clear_rig_authoring_state();
        self.model_document_available = false;
        self.set_model_history_state(false, false);
        self.model_voxel_count = 0;
        self.model_material_count = 0;
        self.set_model_selection(None, None, None);
    }
    pub fn set_active_model_document(
        &mut self,
        name: Option<String>,
        voxel_count: usize,
        material_count: usize,
    ) {
        if self.active_model_name != name {
            self.clear_rig_authoring_state();
        }
        self.model_document_available = name.is_some();
        self.set_model_history_state(false, false);
        self.active_model_name = name;
        self.model_voxel_count = voxel_count;
        self.model_material_count = material_count;
        self.project_category = 1;
        self.selected_hierarchy = 0;
        self.scene_document_available = false;
        self.active_scene_name = None;
        self.scene_model_names.clear();
        self.selected_scene_model = None;
        self.selected_scene_model_info = None;
        self.set_model_selection(None, None, None);
    }
    /// Refresh the rig controls from the active voxel model document.
    pub fn set_rig_authoring_state(
        &mut self,
        bone_names: Vec<String>,
        selected_bone: Option<usize>,
        animation_clip_count: usize,
    ) {
        let had_clips = self.animation_clip_count > 0;
        if selected_bone.is_some() && selected_bone != self.selected_bone {
            self.show_rig_overlay = true;
        }
        self.rig_bone_names = bone_names;
        self.selected_bone = selected_bone.filter(|&index| index < self.rig_bone_names.len());
        self.animation_clip_count = animation_clip_count;
        self.selected_animation_clip = if animation_clip_count == 0 {
            None
        } else {
            Some(
                self.selected_animation_clip
                    .unwrap_or(0)
                    .min(animation_clip_count - 1),
            )
        };
        if !had_clips && animation_clip_count > 0 {
            self.animation_timeline_expanded = true;
        } else if animation_clip_count == 0 {
            self.animation_playing = false;
            self.animation_timeline_expanded = false;
        }
        let visible_rows = self.rig_hierarchy_visible_rows();
        let max_scroll = self.rig_bone_names.len().saturating_sub(visible_rows);
        self.rig_hierarchy_scroll = self.rig_hierarchy_scroll.min(max_scroll);
        if let Some(selected) = self.selected_bone {
            if selected < self.rig_hierarchy_scroll {
                self.rig_hierarchy_scroll = selected;
            } else if selected >= self.rig_hierarchy_scroll + visible_rows {
                self.rig_hierarchy_scroll = (selected + 1 - visible_rows).min(max_scroll);
            }
        }
    }
    pub fn set_animation_timeline_state(
        &mut self,
        selected_clip: Option<usize>,
        time_seconds: f32,
        time_step_seconds: f32,
    ) {
        self.animation_playing = false;
        self.selected_animation_clip =
            selected_clip.filter(|&clip| clip < self.animation_clip_count);
        self.selected_animation_time_seconds = if time_seconds.is_finite() {
            time_seconds.clamp(0.0, self.animation_duration_seconds.max(0.0))
        } else {
            0.0
        };
        self.animation_time_step_seconds = if time_step_seconds.is_finite() {
            time_step_seconds.abs().max(0.001)
        } else {
            0.1
        };
    }
    pub fn set_animation_timeline_info(
        &mut self,
        duration_seconds: f32,
        frame_rate: f32,
        keyframe_times: Vec<f32>,
    ) {
        self.animation_duration_seconds = if duration_seconds.is_finite() {
            duration_seconds.max(1.0 / 120.0)
        } else {
            1.0
        };
        self.animation_frame_rate = if frame_rate.is_finite() {
            frame_rate.clamp(1.0, 240.0)
        } else {
            30.0
        };
        self.animation_keyframe_times = keyframe_times
            .into_iter()
            .filter(|time| time.is_finite() && *time >= 0.0)
            .map(|time| time.min(self.animation_duration_seconds))
            .collect();
        self.animation_keyframe_times
            .sort_by(|left, right| left.total_cmp(right));
        self.animation_keyframe_times
            .dedup_by(|left, right| (*left - *right).abs() < 1e-5);
        self.selected_animation_time_seconds = self
            .selected_animation_time_seconds
            .clamp(0.0, self.animation_duration_seconds);
    }
    pub fn end_timeline_drag(&mut self) {
        self.timeline_dragging = false;
        self.inspector_scroll_dragging = false;
    }
    pub fn timeline_is_dragging(&self) -> bool {
        self.timeline_dragging
    }
    fn animation_last_frame(&self) -> usize {
        (self.animation_duration_seconds.max(0.0) * self.animation_frame_rate.max(1.0)).floor()
            as usize
    }
    fn animation_current_frame(&self) -> usize {
        (self.selected_animation_time_seconds.max(0.0) * self.animation_frame_rate.max(1.0)).round()
            as usize
    }
    fn scrub_animation_time(&mut self, x: f32, left: f32, width: f32) {
        if width <= 0.0 {
            return;
        }
        let frame = (((x - left) / width).clamp(0.0, 1.0) * self.animation_last_frame() as f32)
            .round() as usize;
        self.selected_animation_time_seconds =
            (frame as f32 / self.animation_frame_rate).min(self.animation_duration_seconds);
    }
    fn set_inspector_scroll_from_thumb(
        &mut self,
        thumb_top: f32,
        track_top: f32,
        track_height: f32,
        thumb_height: f32,
        max_scroll: f32,
    ) {
        let travel = (track_height - thumb_height).max(1.0);
        self.scene_inspector_scroll =
            ((thumb_top - track_top) / travel).clamp(0.0, 1.0) * max_scroll;
    }
    fn clear_rig_authoring_state(&mut self) {
        self.rig_bone_names.clear();
        self.selected_bone = None;
        self.animation_clip_count = 0;
        self.selected_animation_clip = None;
        self.selected_animation_time_seconds = 0.0;
        self.animation_duration_seconds = 1.0;
        self.animation_frame_rate = ANIMATION_FRAME_RATE;
        self.animation_keyframe_times.clear();
        self.animation_timeline_expanded = false;
        self.animation_playing = false;
        self.timeline_dragging = false;
        self.show_rig_overlay = false;
        self.rig_edit_mode = RigEditMode::Move;
        self.rig_hierarchy_scroll = 0;
    }

    pub fn toggle_animation_playback(&mut self) -> bool {
        if self.selected_animation_clip.is_none() {
            self.animation_playing = false;
            return false;
        }
        if !self.animation_playing
            && self.selected_animation_time_seconds >= self.animation_duration_seconds
        {
            self.selected_animation_time_seconds = 0.0;
        }
        self.animation_playing = !self.animation_playing;
        self.animation_playing
    }

    pub fn advance_animation(&mut self, elapsed_seconds: f32) {
        if !self.animation_playing || self.selected_animation_clip.is_none() {
            return;
        }
        let duration = self
            .animation_duration_seconds
            .max(1.0 / self.animation_frame_rate);
        let elapsed = if elapsed_seconds.is_finite() {
            elapsed_seconds.clamp(0.0, 0.1)
        } else {
            0.0
        };
        let playback_delta = elapsed * self.animation_playback_speed.clamp(0.25, 8.0);
        self.selected_animation_time_seconds =
            (self.selected_animation_time_seconds + playback_delta).rem_euclid(duration);
    }

    fn animation_timeline_height(&self) -> f32 {
        if !self.model_document_available {
            0.0
        } else if self.animation_timeline_expanded {
            ANIMATION_TIMELINE_EXPANDED_HEIGHT
        } else {
            ANIMATION_TIMELINE_COLLAPSED_HEIGHT
        }
    }

    fn animation_timeline_top(&self) -> f32 {
        self.canvas_height - self.animation_timeline_height()
    }

    fn editor_dock_top(&self) -> f32 {
        self.animation_timeline_top() - PROJECT_DOCK_HEIGHT
    }

    fn rig_hierarchy_visible_rows(&self) -> usize {
        ((self.editor_dock_top() - RIG_HIERARCHY_TOP) / RIG_HIERARCHY_ROW_HEIGHT)
            .floor()
            .max(1.0) as usize
    }
    pub fn set_selected_scene_model_info(&mut self, info: Option<SceneModelInspectorInfo>) {
        self.selected_scene_model_info = info.filter(|info| {
            info.translation_m.iter().all(|value| value.is_finite())
                && info.rotation_deg.iter().all(|value| value.is_finite())
                && info.scale.iter().all(|value| value.is_finite())
                && info.velocity_m_s.iter().all(|value| value.is_finite())
                && info.force_n.iter().all(|value| value.is_finite())
        });
    }

    pub fn set_scene_physics_config(&mut self, config: &SceneConfig) {
        self.scene_physics = ScenePhysicsValues {
            fixed_dt: config.fixed_dt,
            substeps: config.substeps,
            gravity: config.gravity,
            restitution: config.fracture.restitution,
            friction_scale: config.fracture.friction,
            iterations: config.iterations,
            tolerance: config.solver_relative_tolerance,
        };
    }
    pub fn set_scene_simulation_inspector_info(
        &mut self,
        temperature_kelvin: Option<f32>,
        thermal_bonding: Option<ThermalBondingSettings>,
    ) {
        self.selected_scene_initial_temperature_kelvin =
            temperature_kelvin.filter(|value| value.is_finite() && *value > 0.0);
        self.scene_thermal_bonding = thermal_bonding;
    }
    pub fn scene_input_active(&self) -> bool {
        self.scene_input.is_some()
            || self.welding_input.is_some()
            || self.scene_physics_input.is_some()
    }
    pub fn scene_input_char(&mut self, character: char) {
        if let Some(input) = self.scene_physics_input.as_mut() {
            let value = &mut input.values[input.field];
            let integer_field = matches!(input.field, 1 | 7);
            let accepted = character.is_ascii_digit()
                || (!integer_field && character == '.' && (!input.editing || !value.contains('.')))
                || (!integer_field
                    && matches!(character, 'e' | 'E')
                    && input.editing
                    && !value.contains(['e', 'E']))
                || (!integer_field
                    && matches!(character, '+' | '-')
                    && value.ends_with(['e', 'E']))
                || (!integer_field && character == '-' && (!input.editing || value.is_empty()));
            if accepted {
                if !input.editing {
                    value.clear();
                    input.editing = true;
                }
                if value.len() < 24 {
                    value.push(character);
                }
            }
            return;
        }
        if let Some(input) = self.welding_input.as_mut() {
            let value = &mut input.values[input.field];
            let accepted = if input.field == 0 {
                character.is_ascii_alphanumeric()
                    || character == ' '
                    || character == '_'
                    || character == '-'
            } else {
                character.is_ascii_digit()
                    || (character == '.' && (!input.editing || !value.contains('.')))
            };
            if accepted {
                if !input.editing {
                    value.clear();
                    input.editing = true;
                }
                if value.len() < 32 {
                    value.push(character);
                }
            }
            return;
        }
        let Some(input) = self.scene_input.as_mut() else {
            return;
        };
        let value = &mut input.values[input.axis];
        if character.is_ascii_digit()
            || (character == '.' && (!input.editing || !value.contains('.')))
            || (character == '-' && (!input.editing || value.is_empty()))
        {
            if !input.editing {
                value.clear();
                input.editing = true;
            }
            if value.len() < 18 {
                value.push(character);
            }
        }
    }
    pub fn scene_input_key(&mut self, key: KeyCode) -> Option<UiAction> {
        if let Some(input) = self.scene_physics_input.as_mut() {
            match key {
                KeyCode::Escape => {
                    self.scene_physics_input = None;
                    return None;
                }
                KeyCode::Tab => {
                    input.field = (input.field + 1) % input.values.len();
                    input.editing = false;
                    return None;
                }
                KeyCode::Backspace => {
                    if input.editing {
                        input.values[input.field].pop();
                    } else {
                        input.values[input.field].clear();
                        input.editing = true;
                    }
                    return None;
                }
                KeyCode::Enter | KeyCode::NumpadEnter => {
                    match scene_physics_action(&input.values) {
                        Ok(action) => {
                            if let UiAction::SetScenePhysics {
                                fixed_dt_seconds,
                                substeps,
                                gravity,
                                restitution,
                                friction_scale,
                                solver_iterations,
                                solver_tolerance,
                            } = &action
                            {
                                self.scene_physics = ScenePhysicsValues {
                                    fixed_dt: *fixed_dt_seconds,
                                    substeps: *substeps as usize,
                                    gravity: *gravity,
                                    restitution: *restitution,
                                    friction_scale: *friction_scale,
                                    iterations: *solver_iterations,
                                    tolerance: *solver_tolerance,
                                };
                            }
                            self.scene_physics_input = None;
                            return Some(action);
                        }
                        Err(message) => self.status_message = Some(message.to_owned()),
                    }
                    return None;
                }
                _ => return None,
            }
        }
        if let Some(input) = self.welding_input.as_mut() {
            match key {
                KeyCode::Escape => {
                    self.welding_input = None;
                    return None;
                }
                KeyCode::Tab => {
                    input.field = (input.field + 1) % 6;
                    input.editing = false;
                    return None;
                }
                KeyCode::Backspace => {
                    if input.editing {
                        input.values[input.field].pop();
                    } else {
                        input.values[input.field].clear();
                        input.editing = true;
                    }
                    return None;
                }
                KeyCode::Enter | KeyCode::NumpadEnter => {
                    let values = input.values.clone();
                    let parsed = values[1..]
                        .iter()
                        .map(|value| value.parse::<f64>().ok().filter(|v| v.is_finite()))
                        .collect::<Vec<_>>();
                    if parsed.iter().any(Option::is_none) || values[0].trim().is_empty() {
                        self.status_message =
                            Some("Enter a material name and five finite thresholds.".into());
                        return None;
                    }
                    let settings = ThermalBondingSettings {
                        material_name: values[0].trim().to_owned(),
                        minimum_mass_fraction: parsed[0].unwrap() as f32,
                        minimum_temperature_kelvin: parsed[1].unwrap() as f32,
                        minimum_impact_energy_j: parsed[2].unwrap(),
                        minimum_contact_pressure_pa: parsed[3].unwrap() as f32,
                        minimum_dwell_seconds: parsed[4].unwrap() as f32,
                    };
                    if let Err(error) = settings.validate() {
                        self.status_message = Some(error);
                        return None;
                    }
                    self.welding_input = None;
                    return Some(UiAction::SetSceneThermalBonding(Some(settings)));
                }
                _ => return None,
            }
        }
        let input = self.scene_input.as_mut()?;
        match key {
            KeyCode::Escape => {
                self.scene_input = None;
                None
            }
            KeyCode::Tab => {
                input.axis = (input.axis + 1) % 3;
                input.editing = false;
                None
            }
            KeyCode::Backspace => {
                if input.editing {
                    input.values[input.axis].pop();
                } else {
                    input.values[input.axis].clear();
                    input.editing = true;
                }
                None
            }
            KeyCode::Enter => {
                let parsed = input
                    .values
                    .each_ref()
                    .map(|value| value.parse::<f64>().ok().filter(|v| v.is_finite()));
                if parsed.iter().any(Option::is_none) {
                    self.status_message = Some("Enter a finite number for X, Y, and Z.".to_owned());
                    return None;
                }
                let target = input.target;
                let [x, y, z] = parsed.map(Option::unwrap);
                if target == SceneInputTarget::Rotation
                    && [x, y, z]
                        .iter()
                        .any(|value| (value / 90.0 - (value / 90.0).round()).abs() > 1e-6)
                {
                    self.status_message =
                        Some("Rotation currently supports 90° increments.".to_owned());
                    return None;
                }
                let action = match target {
                    SceneInputTarget::Position => UiAction::SetScenePosition { value: [x, y, z] },
                    SceneInputTarget::Rotation => {
                        let degrees = [x as f32, y as f32, z as f32];
                        if !degrees.iter().all(|value| value.is_finite()) {
                            self.status_message =
                                Some("Rotation is outside the supported range.".to_owned());
                            return None;
                        }
                        UiAction::SetSceneRotation { degrees }
                    }
                    SceneInputTarget::Velocity => {
                        let value = [x as f32, y as f32, z as f32];
                        if !value.iter().all(|value| value.is_finite()) {
                            self.status_message =
                                Some("Velocity is outside the supported range.".to_owned());
                            return None;
                        }
                        UiAction::SetSceneVelocity { value }
                    }
                    SceneInputTarget::Force => {
                        let value = [x as f32, y as f32, z as f32];
                        if !value.iter().all(|value| value.is_finite()) {
                            self.status_message =
                                Some("Force is outside the supported range.".to_owned());
                            return None;
                        }
                        UiAction::SetSceneForce { value }
                    }
                    SceneInputTarget::InitialTemperature => {
                        if x <= 0.0 || x > f64::from(f32::MAX) {
                            self.status_message =
                                Some("Temperature must be above absolute zero.".into());
                            return None;
                        }
                        UiAction::SetSceneInitialTemperature {
                            kelvin: Some(x as f32),
                        }
                    }
                };
                self.scene_input = None;
                Some(action)
            }
            _ => None,
        }
    }
    pub fn set_scene_model_selection_bounds(&mut self, bounds: Option<[[f64; 3]; 2]>) {
        self.selected_bounds_m = bounds;
    }
    pub fn scroll_scene_hierarchy(&mut self, amount: f32, x: f32, y: f32) -> bool {
        if !self.editor_mode || self.menu_open {
            return false;
        }
        let (x, y) = self.to_canvas(x, y);
        let inspector_left = self.canvas_width - 250.0;
        if (self.scene_document_available || self.model_document_available)
            && (inspector_left..=self.canvas_width).contains(&x)
            && (144.0..self.canvas_height - 8.0).contains(&y)
        {
            let max_scroll = inspector_scroll_geometry(self)[4];
            if max_scroll > 0.0 {
                self.scene_inspector_scroll = if amount > 0.0 {
                    (self.scene_inspector_scroll - 36.0).max(0.0)
                } else if amount < 0.0 {
                    (self.scene_inspector_scroll + 36.0).min(max_scroll)
                } else {
                    self.scene_inspector_scroll
                };
            }
            return true;
        }
        if !(0.0..=235.0).contains(&x) {
            return false;
        }
        if self.model_document_available && (RIG_HIERARCHY_TOP..self.editor_dock_top()).contains(&y)
        {
            let visible_rows = self.rig_hierarchy_visible_rows();
            let max_scroll = self.rig_bone_names.len().saturating_sub(visible_rows);
            if amount > 0.0 {
                self.rig_hierarchy_scroll = self.rig_hierarchy_scroll.saturating_sub(1);
            } else if amount < 0.0 {
                self.rig_hierarchy_scroll = (self.rig_hierarchy_scroll + 1).min(max_scroll);
            }
            return true;
        }
        if !self.scene_document_available || !(137.0..self.editor_dock_top()).contains(&y) {
            return false;
        }
        let visible_rows = ((self.editor_dock_top() - 167.0) / 30.0).floor().max(1.0) as usize;
        let max_scroll = self.scene_model_names.len().saturating_sub(visible_rows);
        if amount > 0.0 {
            self.scene_hierarchy_scroll = self.scene_hierarchy_scroll.saturating_sub(1);
        } else if amount < 0.0 {
            self.scene_hierarchy_scroll = (self.scene_hierarchy_scroll + 1).min(max_scroll);
        }
        true
    }

    fn click_scene_inspector(&mut self, x: f32, y: f32) -> Option<UiAction> {
        let inspector_left = self.canvas_width - 250.0;
        let rows = scene_inspector_rows(self);
        let Some(row) = rows
            .into_iter()
            .find(|row| (row.top..=row.top + row.height).contains(&y))
        else {
            return None;
        };
        match row.item {
            SceneInspectorItem::Section(section) => {
                let index = scene_section_index(section);
                self.scene_inspector_collapsed[index] = !self.scene_inspector_collapsed[index];
            }
            SceneInspectorItem::Actions => {
                let column = ((x - (inspector_left + 10.0)) / 73.0).floor() as usize;
                return match column {
                    0 => Some(UiAction::DuplicateSelectedSceneModel),
                    1 => Some(UiAction::RemoveSelectedSceneModel),
                    2 => Some(UiAction::FrameSelection),
                    _ => None,
                };
            }
            SceneInspectorItem::Position
            | SceneInspectorItem::Rotation
            | SceneInspectorItem::Velocity
            | SceneInspectorItem::Force => {
                let target = match row.item {
                    SceneInspectorItem::Position => SceneInputTarget::Position,
                    SceneInspectorItem::Rotation => SceneInputTarget::Rotation,
                    SceneInspectorItem::Velocity => SceneInputTarget::Velocity,
                    _ => SceneInputTarget::Force,
                };
                let axis = scene_vector_axis(x, inspector_left)?;
                if let Some(input) = self
                    .scene_input
                    .as_mut()
                    .filter(|input| input.target == target)
                {
                    input.axis = axis;
                    input.editing = false;
                    return None;
                }
                let pending_action = self.commit_inline_editor_before_focus_change()?;
                if self.scene_input_active() {
                    return pending_action;
                }
                let Some(instance) = self.selected_scene_model_info.as_ref() else {
                    return None;
                };
                let values = match target {
                    SceneInputTarget::Position => {
                        instance.translation_m.map(|value| format!("{value:.3}"))
                    }
                    SceneInputTarget::Rotation => {
                        instance.rotation_deg.map(|value| format!("{value:.0}"))
                    }
                    SceneInputTarget::Velocity => {
                        instance.velocity_m_s.map(|value| format!("{value:.3}"))
                    }
                    SceneInputTarget::Force => instance.force_n.map(|value| format!("{value:.2}")),
                    SceneInputTarget::InitialTemperature => unreachable!(),
                };
                self.scene_input = Some(SceneInputState {
                    target,
                    values,
                    axis,
                    editing: false,
                });
                return pending_action;
            }
            SceneInspectorItem::Temperature => {
                let pending_action = self.commit_inline_editor_before_focus_change()?;
                if self.scene_input_active() {
                    return pending_action;
                }
                let value = self
                    .selected_scene_initial_temperature_kelvin
                    .unwrap_or(293.15);
                self.scene_input = Some(SceneInputState {
                    target: SceneInputTarget::InitialTemperature,
                    values: [format!("{value:.2}"), "0".into(), "0".into()],
                    axis: 0,
                    editing: false,
                });
                return pending_action;
            }
            SceneInspectorItem::MaterialCycle => return Some(UiAction::CycleSceneMaterial),
            SceneInspectorItem::MaterialActions => {
                let column = ((x - (inspector_left + 10.0)) / 73.0).floor() as usize;
                return match column {
                    0 => Some(UiAction::LoadSceneMaterial),
                    1 => Some(UiAction::SaveSceneMaterial),
                    2 => Some(UiAction::CreateSceneMaterial),
                    _ => None,
                };
            }
            SceneInspectorItem::WeldingToggle => {
                if self.scene_thermal_bonding.is_some() {
                    return Some(UiAction::SetSceneThermalBonding(None));
                }
                self.welding_input = Some(default_welding_input());
            }
            SceneInspectorItem::WeldingConfigure => {
                if let Some(settings) = self.scene_thermal_bonding.as_ref() {
                    self.welding_input = Some(welding_input_from_settings(settings));
                }
            }
            SceneInspectorItem::PhysicsField(field) => {
                if let Some(input) = self.scene_physics_input.as_mut() {
                    input.field = field;
                    input.editing = false;
                    return None;
                }
                let pending_action = self.commit_inline_editor_before_focus_change()?;
                if self.scene_input_active() {
                    return pending_action;
                }
                let input =
                    self.scene_physics_input
                        .get_or_insert_with(|| ScenePhysicsInputState {
                            values: scene_physics_strings(self.scene_physics),
                            field,
                            editing: false,
                        });
                input.field = field;
                input.editing = false;
                return pending_action;
            }
            SceneInspectorItem::PhysicsApply => {
                if self.scene_physics_input.is_none() {
                    self.scene_physics_input = Some(ScenePhysicsInputState {
                        values: scene_physics_strings(self.scene_physics),
                        field: 0,
                        editing: false,
                    });
                }
                return self.scene_input_key(KeyCode::Enter);
            }
            SceneInspectorItem::Bounds
            | SceneInspectorItem::Hint
            | SceneInspectorItem::InputHint => {}
        }
        None
    }

    fn commit_inline_editor_before_focus_change(&mut self) -> Option<Option<UiAction>> {
        if !self.scene_input_active() {
            return Some(None);
        }
        let action = self.scene_input_key(KeyCode::Enter);
        if self.scene_input_active() {
            return None;
        }
        Some(action)
    }
    pub fn set_model_history_state(&mut self, can_undo: bool, can_redo: bool) {
        self.can_undo = self.model_document_available && can_undo;
        self.can_redo = self.model_document_available && can_redo;
    }
    pub fn set_model_selection(
        &mut self,
        index: Option<usize>,
        position: Option<[f64; 3]>,
        mass_kg: Option<f32>,
    ) {
        self.selected_voxel = index;
        self.selected_voxel_position = position;
        self.selected_voxel_mass_kg = mass_kg;
        self.selected_voxel_indices = index.into_iter().collect();
        self.selected_bounds_m = position.map(|position| {
            let half_cell = 0.005;
            [
                position.map(|value| value - half_cell),
                position.map(|value| value + half_cell),
            ]
        });
        self.selected_dimensions_cm = index.map(|_| [1; 3]);
        self.selection_diagram_xz = [false; 64];
        if index.is_some() {
            self.selection_diagram_xz[3 * 8 + 3] = true;
        }
        if self.model_document_available {
            self.selected_hierarchy = if index.is_some() { 2 } else { 0 };
        }
    }
    pub fn set_model_box_selection(
        &mut self,
        indices: Vec<usize>,
        bounds_m: [[f64; 3]; 2],
        dimensions_cm: [u32; 3],
        diagram_xz: [bool; 64],
        single_position: Option<[f64; 3]>,
        total_mass_kg: f32,
    ) {
        let single_index = (indices.len() == 1).then(|| indices[0]);
        self.selected_voxel = single_index;
        self.selected_voxel_position = single_position;
        self.selected_voxel_mass_kg = (!indices.is_empty()).then_some(total_mass_kg);
        self.selected_voxel_indices = indices;
        self.selected_bounds_m = (!self.selected_voxel_indices.is_empty()).then_some(bounds_m);
        self.selected_dimensions_cm =
            (!self.selected_voxel_indices.is_empty()).then_some(dimensions_cm);
        self.selection_diagram_xz = diagram_xz;
        if self.model_document_available {
            self.selected_hierarchy = if self.selected_voxel_indices.is_empty() {
                0
            } else {
                2
            };
        }
    }
    pub fn set_status_message(&mut self, message: impl Into<String>) {
        self.status_message = Some(message.into());
    }
    pub fn begin_mesh_import(&mut self, filename: impl Into<String>) {
        self.mesh_import_open = true;
        self.mesh_import_filename = Some(filename.into());
        self.mesh_import_scale = 1.0;
        self.mesh_import_filled = false;
        self.mesh_import_busy = false;
        self.mesh_import_preview = None;
        self.mesh_import_error = None;
        self.file_menu_open = false;
    }
    pub fn set_mesh_import_busy(&mut self, busy: bool) {
        self.mesh_import_busy = busy;
        if busy {
            self.mesh_import_preview = None;
            self.mesh_import_error = None;
        }
    }
    pub fn set_mesh_import_preview(&mut self, preview: MeshImportPreviewInfo) {
        self.mesh_import_busy = false;
        self.mesh_import_error = None;
        self.mesh_import_preview = Some(preview);
    }
    pub fn set_mesh_import_error(&mut self, message: impl Into<String>) {
        self.mesh_import_busy = false;
        self.mesh_import_preview = None;
        self.mesh_import_error = Some(message.into());
    }
    pub fn finish_mesh_import(&mut self) {
        self.mesh_import_open = false;
        self.mesh_import_busy = false;
    }
    pub fn pointer(&mut self, x: f32, y: f32) -> Option<Shape> {
        let (x, y) = self.to_canvas(x, y);
        self.pointer_canvas(x, y)
    }
    fn pointer_canvas(&mut self, x: f32, y: f32) -> Option<Shape> {
        if self.inspector_scroll_dragging {
            let [track_top, track_height, _, thumb_height, max_scroll] =
                inspector_scroll_geometry(self);
            self.set_inspector_scroll_from_thumb(
                y - thumb_height * 0.5,
                track_top,
                track_height,
                thumb_height,
                max_scroll,
            );
            return None;
        }
        if self.timeline_dragging {
            let inspector_left = self.canvas_width - 250.0;
            let (left, width) = animation_slider_geometry(inspector_left);
            self.scrub_animation_time(x, left, width);
            return None;
        }
        if self.mesh_import_open {
            let (left, top) = mesh_import_modal_origin(self.canvas_width, self.canvas_height);
            if (left + 24.0..=left + 286.0).contains(&x) && (top + 121.0..=top + 151.0).contains(&y)
            {
                let fraction = ((x - (left + 24.0)) / 262.0).clamp(0.0, 1.0);
                self.mesh_import_scale = 0.1 + fraction * 9.9;
                self.mesh_import_preview = None;
                self.mesh_import_error = None;
            }
            return None;
        }
        if self.menu_open {
            if self.settings_open {
                for row in 0..3 {
                    let [left, top, width, _] = settings_slider_rect(self.canvas_width, row);
                    if (left - 12.0..=left + width + 12.0).contains(&x)
                        && (top - 18.0..=top + 28.0).contains(&y)
                    {
                        self.set_settings_slider(row, x, left, width);
                        break;
                    }
                }
                return None;
            }
            return None;
        }
        if !(20.0..=280.0).contains(&x) {
            return None;
        }
        for (n, shape) in [Shape::Wedge, Shape::Blade, Shape::Rod, Shape::Rounded]
            .into_iter()
            .enumerate()
        {
            let top = 50.0 + n as f32 * 32.0;
            if (top..top + 26.0).contains(&y) {
                self.selected = shape;
                return Some(shape);
            }
        }
        if (246.0..278.0).contains(&y) {
            let t = ((x - 25.0) / 245.0).clamp(0.0, 1.0);
            self.force_newtons = t * t * MAX_PUSH_FORCE_N;
        }
        None
    }
    pub fn draw(
        &mut self,
        broken: usize,
        clusters: usize,
        fps: f32,
        sim_ms: f32,
        world_age_years: Option<f32>,
        reticle: Option<ReticleState>,
        viewport_width: f32,
        viewport_height: f32,
        editor: Option<&EditorInfo<'_>>,
    ) -> Vec<Rect> {
        let viewport_width = viewport_width.max(1.0);
        let viewport_height = viewport_height.max(1.0);
        // The inspector's rig and animation controls reach y=766 in the
        // logical canvas. Scale against that full height so they remain
        // inside the client area at common resolutions such as 1920×1080.
        // Keep one uniform scale for drawing, hit testing, and the scene
        // viewport. The editor's tallest controls reach y=766, so short
        // windows need to scale below 0.55 to keep the full logical canvas
        // visible (for example, 1280×400).
        self.canvas_scale = ((viewport_height / 780.0).min(viewport_width / 960.0)).min(2.5);
        self.canvas_width = viewport_width / self.canvas_scale;
        self.canvas_height = viewport_height / self.canvas_scale;
        self.canvas_offset = [0.0; 2];
        let inspector_content_bottom = if self.model_document_available {
            766.0
        } else if self.scene_document_available {
            scene_inspector_content_bottom(self)
        } else {
            455.0
        };
        self.scene_inspector_scroll = self.scene_inspector_scroll.clamp(
            0.0,
            (inspector_content_bottom - (self.canvas_height - 8.0)).max(0.0),
        );
        let mut r = Vec::new();
        if self.world_active && self.editor_mode && !self.menu_open {
            draw_editor(&mut r, self, editor, self.canvas_width, self.canvas_height);
            if let Some(input) = self.welding_input.as_ref() {
                draw_welding_input_modal(&mut r, input, self.canvas_width, self.canvas_height);
            }
            if self.mesh_import_open {
                draw_mesh_import_modal(&mut r, self, self.canvas_width, self.canvas_height);
            }
        } else if !self.world_active {
            rect(
                &mut r,
                [10.0, 10.0, 285.0, (self.canvas_height - 20.0).max(1.0)],
                [0.025, 0.04, 0.06, 0.94],
            );
            text(&mut r, 24., 24., "VOXEL MECHANICS LAB", 1.5);
            for (i, name) in [
                "1 УДАРИТЬ ТОПОРОМ",
                "2 УДАРИТЬ НОЖОМ",
                "3 УДАРИТЬ ПАЛКОЙ",
                "4 УДАРИТЬ КАМНЕМ",
            ]
            .iter()
            .enumerate()
            {
                let y = 50. + i as f32 * 32.;
                rect(&mut r, [20., y, 260., 26.], [0.14, 0.23, 0.29, 1.]);
                text(&mut r, 30., y + 7., name, 1.5);
            }
            let force_fraction = (self.force_newtons / MAX_PUSH_FORCE_N)
                .sqrt()
                .clamp(0.0, 1.0);
            text(&mut r, 24., 218., "CONSTANT PUSH FORCE", 1.5);
            text(
                &mut r,
                24.,
                237.,
                &format!("{:.0} N", self.force_newtons),
                1.6,
            );
            rect(&mut r, [25., 260., 245., 10.], [0.2, 0.26, 0.3, 1.]);
            rect(
                &mut r,
                [25., 260., 245. * force_fraction, 10.],
                [0.18, 0.65, 0.56, 1.],
            );
            rect(
                &mut r,
                [25. + force_fraction * 245. - 4., 256., 8., 18.],
                [0.3, 0.85, 0.65, 1.],
            );
            text(&mut r, 25., 280., "0 N", 1.1);
            text(&mut r, 216., 280., "15000 N", 1.1);
            text(&mut r, 24., 300., "0 N CLEARS FORCE; TOOL COASTS", 0.9);
            let (axis, sign) = if self.direction.x.abs() >= self.direction.y.abs() {
                ("X", if self.direction.x < 0.0 { "-" } else { "+" })
            } else {
                ("Y", if self.direction.y < 0.0 { "-" } else { "+" })
            };
            text(&mut r, 24., 322., &format!("PUSH AXIS {sign}{axis}"), 1.2);
            let stats_y = self.canvas_height - 250.0;
            text(
                &mut r,
                24.,
                stats_y,
                &format!("BROKEN {} CLUSTERS {}", broken, clusters),
                1.2,
            );
            text(
                &mut r,
                24.,
                stats_y + 20.0,
                &format!("FPS {:.1} STEP {:.1} MS", fps, sim_ms),
                1.2,
            );
            if self.world_active {
                text(
                    &mut r,
                    24.,
                    self.canvas_height - 190.0,
                    "ESC MENU  P PAUSE  R RESET",
                    1.1,
                );
                if self.editor_mode {
                    text(
                        &mut r,
                        24.,
                        self.canvas_height - 172.0,
                        "WASD MOVE CAMERA  SHIFT FAST",
                        0.8,
                    );
                    text(
                        &mut r,
                        24.,
                        self.canvas_height - 154.0,
                        "F TOGGLE FREECAM  WHEEL SPEED",
                        0.8,
                    );
                } else {
                    text(
                        &mut r,
                        24.,
                        self.canvas_height - 172.0,
                        "WASD WALK  SPACE JUMP  SHIFT RUN",
                        0.85,
                    );
                    text(
                        &mut r,
                        24.,
                        self.canvas_height - 154.0,
                        "MOUSE LOOK  F FREE CAMERA",
                        0.9,
                    );
                }
            } else {
                text(
                    &mut r,
                    24.,
                    self.canvas_height - 212.0,
                    "SPACE PAUSE   R RESET",
                    1.2,
                );
                text(
                    &mut r,
                    24.,
                    self.canvas_height - 194.0,
                    "WASD FLY  SPACE/CTRL UP/DOWN  RMB LOOK",
                    0.8,
                );
            }
            text(
                &mut r,
                24.,
                self.canvas_height - 130.0,
                "P START / STOP RECORDING",
                1.,
            );
            text(
                &mut r,
                24.,
                self.canvas_height - 112.0,
                "REPLAY LOOPS AT 1X",
                1.,
            );
            text(
                &mut r,
                24.,
                self.canvas_height - 94.0,
                "MAP 10 KM2 | REV 1 CM | LOAD 1 KM",
                1.1,
            );
            if !self.world_active {
                let open_scene_y = self.canvas_height - 162.0;
                rect(
                    &mut r,
                    [20.0, open_scene_y - 28.0, 260.0, 24.0],
                    [0.10, 0.25, 0.29, 1.0],
                );
                text(&mut r, 68.0, open_scene_y - 22.0, "OPEN .VXSCENE", 1.3);
                rect(
                    &mut r,
                    [20.0, open_scene_y, 260.0, 24.0],
                    [0.10, 0.25, 0.29, 1.0],
                );
                text(&mut r, 89.0, open_scene_y + 6.0, "NEW SCENE", 1.3);
            }
            let seed_y = self.canvas_height - 74.0;
            rect(
                &mut r,
                [20., seed_y, 260., 20.],
                if self.seed_editing {
                    [0.13, 0.22, 0.25, 1.]
                } else {
                    [0.09, 0.14, 0.18, 1.]
                },
            );
            let seed_display = self.seed_text.clone();
            text(&mut r, 28., seed_y + 6.0, &seed_display, 1.3);
            if self.seed_editing {
                let cursor_x = (28. + seed_display.len() as f32 * 9.0).min(270.0);
                rect(
                    &mut r,
                    [cursor_x, seed_y + 4.0, 2., 15.],
                    [0.3, 0.85, 0.65, 1.],
                );
            }
            let generate_y = self.canvas_height - 50.0;
            rect(
                &mut r,
                [20., generate_y, 260., 24.],
                if self.world_active {
                    [0.12, 0.38, 0.33, 1.]
                } else {
                    [0.14, 0.23, 0.29, 1.]
                },
            );
            text(
                &mut r,
                50.,
                generate_y + 6.0,
                if self.world_active {
                    "REGENERATE WORLD"
                } else {
                    "GENERATE WORLD"
                },
                1.4,
            );
            if let Some(age) = world_age_years {
                text(
                    &mut r,
                    24.,
                    self.canvas_height - 21.0,
                    &format!("AGE {age:.0} Y | SIM {:.2}X", self.simulation_time_scale),
                    1.0,
                );
            }
        }
        if self.menu_open {
            if self.settings_open {
                draw_settings_workspace(&mut r, self);
            } else {
                let (menu_x, menu_y) = self.menu_origin();
                rect(
                    &mut r,
                    [0., 0., self.canvas_width, self.canvas_height],
                    [0.015, 0.025, 0.04, 0.78],
                );
                rect(
                    &mut r,
                    [menu_x, menu_y, 420., 520.],
                    [0.055, 0.085, 0.105, 0.99],
                );
                rect(
                    &mut r,
                    [menu_x, menu_y + 20., 420., 4.],
                    [0.22, 0.68, 0.57, 1.],
                );
                text(&mut r, menu_x + 52., menu_y + 77., "Paused", 2.4);
                text(
                    &mut r,
                    menu_x + 52.,
                    menu_y + 121.,
                    "World state is preserved",
                    1.1,
                );
                menu_button(
                    &mut r,
                    menu_x + 40.,
                    menu_y + 193.,
                    340.,
                    52.,
                    "Resume",
                    true,
                );
                menu_button(
                    &mut r,
                    menu_x + 40.,
                    menu_y + 257.,
                    340.,
                    52.,
                    "Settings",
                    false,
                );
                menu_button(
                    &mut r,
                    menu_x + 40.,
                    menu_y + 321.,
                    340.,
                    52.,
                    "Return to editor",
                    false,
                );
                menu_button(
                    &mut r,
                    menu_x + 40.,
                    menu_y + 385.,
                    340.,
                    52.,
                    "Exit",
                    false,
                );
                text(
                    &mut r,
                    menu_x + 52.,
                    menu_y + 449.,
                    "F2 switches workspaces",
                    1.0,
                );
                text(
                    &mut r,
                    menu_x + 52.,
                    menu_y + 471.,
                    "Ctrl crouch | F6 physics",
                    0.9,
                );
            }
        }
        if !self.menu_open && !self.editor_mode && viewport_width > 0.0 && viewport_height > 0.0 {
            if let Some(state) = reticle {
                let (cx, cy) = (self.canvas_width * 0.5, self.canvas_height * 0.5);
                let color = match state {
                    ReticleState::PhysicsOff => [0.72, 0.77, 0.8, 0.9],
                    ReticleState::Idle => [0.88, 0.93, 0.94, 0.9],
                    ReticleState::Target => [0.26, 0.9, 0.63, 0.98],
                    ReticleState::Holding => [0.33, 0.78, 1.0, 1.0],
                };
                rect(
                    &mut r,
                    [cx - 1.5, cy - 10.0, 3.0, 20.0],
                    [0.01, 0.02, 0.025, 0.8],
                );
                rect(
                    &mut r,
                    [cx - 10.0, cy - 1.5, 20.0, 3.0],
                    [0.01, 0.02, 0.025, 0.8],
                );
                rect(&mut r, [cx - 0.5, cy - 7.0, 1.0, 14.0], color);
                rect(&mut r, [cx - 7.0, cy - 0.5, 14.0, 1.0], color);
                let hint = match state {
                    ReticleState::PhysicsOff => Some("F6 ENABLES PHYSICS IN THIS AREA"),
                    ReticleState::Idle => Some("AIM AT A LOOSE OBJECT"),
                    ReticleState::Target => Some("HOLD LEFT MOUSE TO GRAB"),
                    ReticleState::Holding => Some("LOOK TO MOVE | SCROLL DISTANCE | RELEASE"),
                };
                if let Some(hint) = hint {
                    let scale = 1.2;
                    let width = hint.chars().count() as f32 * 6.0 * scale;
                    text(
                        &mut r,
                        cx - width * 0.5,
                        self.canvas_height - 34.0,
                        hint,
                        scale,
                    );
                }
            }
        }
        for item in &mut r {
            let x0 = (item.rect[0] * self.canvas_scale + self.canvas_offset[0]).round();
            let y0 = (item.rect[1] * self.canvas_scale + self.canvas_offset[1]).round();
            let x1 =
                ((item.rect[0] + item.rect[2]) * self.canvas_scale + self.canvas_offset[0]).round();
            let y1 =
                ((item.rect[1] + item.rect[3]) * self.canvas_scale + self.canvas_offset[1]).round();
            item.rect = [x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0)];
        }
        r
    }
    fn to_canvas(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.canvas_offset[0]) / self.canvas_scale,
            (y - self.canvas_offset[1]) / self.canvas_scale,
        )
    }

    fn menu_origin(&self) -> (f32, f32) {
        (
            (self.canvas_width - 420.0) * 0.5,
            ((self.canvas_height - 520.0) * 0.5 + 25.0).max(8.0),
        )
    }

    pub fn set_view_basis(&mut self, right: Vec3, up: Vec3) {
        if right.is_finite()
            && up.is_finite()
            && right.length_squared() > 0.5
            && up.length_squared() > 0.5
        {
            self.view_right = right.normalize();
            self.view_up = up.normalize();
        }
    }

    pub fn scene_viewport(&self) -> Option<[u32; 4]> {
        if !(self.world_active && self.editor_mode && !self.menu_open) {
            return None;
        }
        let left = (235.0 * self.canvas_scale + self.canvas_offset[0]).round();
        let top = (98.0 * self.canvas_scale + self.canvas_offset[1]).round();
        let right =
            ((self.canvas_width - 250.0) * self.canvas_scale + self.canvas_offset[0]).round();
        let bottom = (self.editor_dock_top() * self.canvas_scale + self.canvas_offset[1]).round();
        Some([
            left.max(0.0) as u32,
            top.max(0.0) as u32,
            (right - left).max(1.0) as u32,
            (bottom - top).max(1.0) as u32,
        ])
    }
}

fn draw_editor(
    out: &mut Vec<Rect>,
    ui: &Ui,
    info: Option<&EditorInfo<'_>>,
    canvas_width: f32,
    canvas_height: f32,
) {
    let panel = [0.09, 0.105, 0.12, 0.98];
    let header = [0.12, 0.14, 0.16, 1.0];
    let selected = [0.16, 0.31, 0.38, 1.0];
    let inspector_left = canvas_width - 250.0;
    let dock_top = ui.editor_dock_top();
    let scene_width = (inspector_left - 235.0).max(1.0);
    rect(
        out,
        [0.0, 0.0, canvas_width, 28.0],
        [0.075, 0.085, 0.095, 1.0],
    );
    text(
        out,
        12.0,
        7.0,
        if ui.standalone_editor_app {
            "Voxel Model Editor"
        } else {
            "Voxel Editor"
        },
        1.1,
    );
    text(out, 168.0, 7.0, "File", 0.9);
    rect(
        out,
        [208.0, 0.0, 84.0, 27.0],
        if ui.settings_open { selected } else { header },
    );
    text(out, 216.0, 7.0, "Settings", 0.9);
    if ui.scene_document_available {
        for (x, label, mode) in [
            (370.0, "Object", SceneSelectionMode::Object),
            (438.0, "Voxel", SceneSelectionMode::Voxel),
        ] {
            rect(
                out,
                [x, 0.0, 70.0, 27.0],
                if ui.scene_selection_mode == mode {
                    selected
                } else {
                    header
                },
            );
            text(out, x + 10.0, 7.0, label, 0.82);
        }
    }

    rect(out, [0.0, 28.0, canvas_width, 39.0], header);
    let can_frame = ui.model_document_available
        || (ui.scene_document_available && !ui.scene_model_names.is_empty());
    if ui.model_document_available {
        for (i, (tool, label)) in [
            (EditorTool::Select, "Select"),
            (EditorTool::AddVoxel, "Add REV"),
            (EditorTool::EraseVoxel, "Erase"),
            (EditorTool::AddBox, "Block"),
            (EditorTool::AddSphere, "Sphere"),
            (EditorTool::AddCylinder, "Cylinder"),
        ]
        .iter()
        .enumerate()
        {
            let x = 12.0 + i as f32 * 62.0;
            rect(
                out,
                [x, 34.0, 56.0, 27.0],
                if ui.editor_tool == *tool {
                    selected
                } else {
                    panel
                },
            );
            text(out, x + 7.0, 41.0, label, 0.72);
        }
    }
    if can_frame {
        rect(out, [384.0, 34.0, 56.0, 27.0], panel);
        text(out, 391.0, 41.0, "Frame", 0.72);
    }
    if ui.model_document_available {
        for (x, label, enabled) in [(446.0, "Undo", ui.can_undo), (508.0, "Redo", ui.can_redo)] {
            rect(
                out,
                [x, 34.0, 56.0, 27.0],
                if enabled {
                    panel
                } else {
                    [0.075, 0.085, 0.095, 1.0]
                },
            );
            if enabled {
                text(out, x + 8.0, 41.0, label, 0.72);
            } else {
                let text_start = out.len();
                text(out, x + 8.0, 41.0, label, 0.72);
                for glyph in &mut out[text_start..] {
                    glyph.color[3] *= 0.38;
                }
            }
        }
    }
    if ui.standalone_editor_app && ui.model_document_available {
        rect(out, [570.0, 34.0, 40.0, 27.0], panel);
        text(out, 582.0, 41.0, "-", 0.85);
        text(
            out,
            620.0,
            41.0,
            &format!("Brush {} cm", ui.brush_size_cells),
            0.72,
        );
        rect(out, [744.0, 34.0, 40.0, 27.0], panel);
        text(out, 756.0, 41.0, "+", 0.85);
    }
    let simulation_target = !ui.standalone_editor_app
        && (!ui.scene_document_available || !ui.scene_model_names.is_empty());
    if simulation_target {
        rect(out, [570.0, 32.0, 56.0, 31.0], [0.16, 0.42, 0.29, 1.0]);
        text(out, 580.0, 40.0, "Play", 0.7);
        rect(out, [630.0, 32.0, 56.0, 31.0], panel);
        text(
            out,
            635.0,
            40.0,
            if ui.paused { "Run" } else { "Pause" },
            0.55,
        );
        rect(
            out,
            [690.0, 32.0, 50.0, 31.0],
            if ui.show_bonds { selected } else { panel },
        );
        text(out, 700.0, 40.0, "Bonds", 0.7);
        text(
            out,
            770.0,
            40.0,
            &physics_toolbar_label(
                ui.physics_cost_ms_per_10ms,
                ui.measured_simulation_rate,
                ui.simulation_time_scale,
                ui.canvas_width < 1200.0,
            ),
            0.68,
        );
    }

    rect(out, [0.0, 67.0, 235.0, (dock_top - 67.0).max(1.0)], panel);
    rect(out, [0.0, 67.0, 235.0, 31.0], header);
    text(out, 13.0, 75.0, "Hierarchy", 0.9);
    let rows = if ui.model_document_available {
        vec![
            format!(
                "Model: {}",
                ui.active_model_name.as_deref().unwrap_or("Untitled")
            ),
            format!("REVs ({})", ui.model_voxel_count),
            match ui.selected_voxel_indices.as_slice() {
                [] => "No REV selected".to_owned(),
                [index] => format!("Selected REV {}", index + 1),
                indices => format!("Selected {} REVs", indices.len()),
            },
            format!("Materials ({})", ui.model_material_count),
        ]
    } else if ui.scene_document_available {
        let mut rows = vec![format!(
            "Scene: {}",
            ui.active_scene_name.as_deref().unwrap_or("Untitled")
        )];
        let visible_rows = ((ui.editor_dock_top() - 167.0) / 30.0).floor().max(1.0) as usize;
        rows.extend(
            ui.scene_model_names
                .iter()
                .skip(ui.scene_hierarchy_scroll)
                .take(visible_rows)
                .map(|name| format!("Model: {name}")),
        );
        if ui.scene_model_names.is_empty() {
            rows.push("No models in scene".to_owned());
        }
        rows
    } else {
        vec![if info.is_some() {
            "Live generated world".to_owned()
        } else {
            "Live simulation".to_owned()
        }]
    };
    for (i, name) in rows.iter().enumerate() {
        let y = 137.0 + i as f32 * 30.0;
        let selected_model_row = ui.model_document_available && i == ui.selected_hierarchy;
        let selected_scene_row = ui.scene_document_available
            && i > 0
            && ui.selected_scene_model == Some(ui.scene_hierarchy_scroll + i - 1);
        if selected_model_row || selected_scene_row {
            rect(out, [0.0, y - 4.0, 235.0, 28.0], selected);
        }
        text(
            out,
            14.0 + if i > 0 { 13.0 } else { 0.0 },
            y + 1.0,
            name,
            0.85,
        );
    }
    if ui.model_document_available {
        text(
            out,
            13.0,
            260.0,
            &format!("Bones ({})", ui.rig_bone_names.len()),
            0.72,
        );
        let visible_rows = ui.rig_hierarchy_visible_rows();
        if ui.rig_bone_names.is_empty() {
            text(out, 17.0, RIG_HIERARCHY_TOP + 5.0, "No bones", 0.66);
        } else {
            for (visible_index, bone_index) in ui
                .rig_bone_names
                .iter()
                .enumerate()
                .skip(ui.rig_hierarchy_scroll)
                .take(visible_rows)
            {
                let row = visible_index - ui.rig_hierarchy_scroll;
                let y = RIG_HIERARCHY_TOP + row as f32 * RIG_HIERARCHY_ROW_HEIGHT;
                if ui.selected_bone == Some(visible_index) {
                    rect(out, [6.0, y, 223.0, 20.0], selected);
                }
                let visible_name = bone_index.chars().take(23).collect::<String>();
                text(
                    out,
                    16.0,
                    y + 5.0,
                    &format!("{}  {visible_name}", visible_index + 1),
                    0.66,
                );
            }
        }
    }
    if let Some(info) = info {
        let summary = info.vegetation_summary.chars().take(30).collect::<String>();
        text(out, 12.0, 342.0, &summary, 0.62);
    }

    rect(
        out,
        [inspector_left, 67.0, 250.0, canvas_height - 67.0],
        panel,
    );
    rect(out, [inspector_left, 67.0, 250.0, 31.0], header);
    text(out, inspector_left + 13.0, 75.0, "Inspector", 0.9);
    let selected_name = ui
        .selected_scene_model_info
        .as_ref()
        .filter(|_| ui.scene_document_available)
        .map(|info| info.name.chars().take(25).collect::<String>());
    let selected_name = selected_name.as_deref().unwrap_or_else(|| {
        rows.get(ui.selected_hierarchy)
            .map(String::as_str)
            .unwrap_or("World")
    });
    rect(out, [inspector_left + 10.0, 108.0, 230.0, 30.0], selected);
    text(out, inspector_left + 20.0, 116.0, selected_name, 0.9);
    if !ui.scene_document_available {
        inspector_section(out, "Details", 148.0, inspector_left + 10.0);
    }
    let inspector_text_x = inspector_left + 15.0;
    if ui.model_document_available {
        if !ui.selected_voxel_indices.is_empty() {
            let count = ui.selected_voxel_indices.len();
            text(
                out,
                inspector_text_x,
                178.0,
                &format!("Selected {count} REV{}", if count == 1 { "" } else { "s" }),
                0.68,
            );
            if let Some([x, y, z]) = ui.selected_dimensions_cm {
                text(
                    out,
                    inspector_text_x,
                    197.0,
                    &format!("Size {x} × {y} × {z} cm"),
                    0.64,
                );
            }
            if let Some(position) = ui.selected_voxel_position {
                text(
                    out,
                    inspector_text_x,
                    264.0,
                    &format!(
                        "Centre {:.2}, {:.2}, {:.2} m",
                        position[0], position[1], position[2]
                    ),
                    0.56,
                );
            }
            text(out, inspector_left + 157.0, 178.0, "X/Z view", 0.6);
            for row in 0..8 {
                for column in 0..8 {
                    let occupied = ui.selection_diagram_xz[row * 8 + column];
                    rect(
                        out,
                        [
                            inspector_left + 157.0 + column as f32 * 8.0,
                            195.0 + row as f32 * 8.0,
                            7.0,
                            7.0,
                        ],
                        if occupied {
                            selected
                        } else {
                            [0.10, 0.12, 0.14, 1.0]
                        },
                    );
                }
            }
        } else {
            text(
                out,
                inspector_text_x,
                196.0,
                "Click for connected part; drag for box",
                0.68,
            );
        }
    } else if ui.scene_document_available {
        draw_scene_inspector_contents(out, ui, inspector_left, header, selected);
    } else {
        text(
            out,
            inspector_text_x,
            196.0,
            "Live world status below",
            0.66,
        );
    }
    if ui.model_document_available {
        inspector_section(out, "Voxel", 296.0, inspector_left + 10.0);
        if let Some(mass) = ui.selected_voxel_mass_kg {
            text(
                out,
                inspector_text_x,
                326.0,
                &format!("Selection mass {:.4} kg", mass),
                0.68,
            );
        }
        text(
            out,
            inspector_text_x,
            345.0,
            &format!("REV count {}", ui.model_voxel_count),
            0.68,
        );
        text(
            out,
            inspector_text_x,
            364.0,
            &format!("Selected REVs {}", ui.selected_voxel_indices.len()),
            0.66,
        );
        text(
            out,
            inspector_text_x,
            382.0,
            "Move selected part - 1 cm",
            0.62,
        );
        for (left, top, label) in [
            (inspector_left + 15.0, 405.0, "X-"),
            (inspector_left + 87.0, 405.0, "X+"),
            (inspector_left + 159.0, 405.0, "Y-"),
            (inspector_left + 15.0, 437.0, "Y+"),
            (inspector_left + 87.0, 437.0, "Z-"),
            (inspector_left + 159.0, 437.0, "Z+"),
        ] {
            let enabled = !ui.selected_voxel_indices.is_empty();
            rect(
                out,
                [left, top, 66.0, 27.0],
                if enabled { header } else { panel },
            );
            text(out, left + 24.0, top + 7.0, label, 0.68);
        }
        for (left, label) in [
            (inspector_left + 15.0, "Rotate Z -90°"),
            (inspector_left + 129.0, "Rotate Z +90°"),
        ] {
            let enabled = !ui.selected_voxel_indices.is_empty();
            rect(
                out,
                [left, 469.0, 106.0, 28.0],
                if enabled { selected } else { panel },
            );
            text(out, left + 9.0, 477.0, label, 0.63);
        }
        rect(
            out,
            [inspector_left + 15.0, 501.0, 220.0, 28.0],
            if !ui.selected_voxel_indices.is_empty() {
                selected
            } else {
                panel
            },
        );
        text(
            out,
            inspector_left + 22.0,
            509.0,
            "Duplicate selected part",
            0.68,
        );
        rect(
            out,
            [inspector_left + 15.0, 533.0, 220.0, 28.0],
            if !ui.selected_voxel_indices.is_empty() {
                selected
            } else {
                panel
            },
        );
        text(
            out,
            inspector_left + 22.0,
            541.0,
            "Delete selected part",
            0.68,
        );
        text(
            out,
            inspector_text_x,
            574.0,
            "Save Voxel Model in File",
            0.62,
        );
        let keys = |action: &str| {
            ui.settings_bindings
                .get(action)
                .map(String::as_str)
                .unwrap_or("-")
        };
        text(
            out,
            inspector_text_x,
            594.0,
            &format!(
                "Move {} / {} | Rotate {} / {}",
                keys("move_x_neg"),
                keys("move_x_pos"),
                keys("rotate_z_neg"),
                keys("rotate_z_pos")
            ),
            0.52,
        );
        text(
            out,
            inspector_text_x,
            610.0,
            &format!(
                "Copy {} | Delete {} | Undo {}",
                keys("duplicate"),
                keys("delete"),
                keys("undo")
            ),
            0.52,
        );
        let child_bone = ui
            .selected_bone
            .filter(|&index| index < ui.rig_bone_names.len());
        let selected_clip = ui
            .selected_animation_clip
            .filter(|&clip| clip < ui.animation_clip_count);
        let can_edit_pose = child_bone.is_some() && selected_clip.is_some();
        let current_frame = ui.animation_current_frame();
        let last_frame = ui.animation_last_frame();
        text(
            out,
            inspector_text_x,
            628.0,
            &format!(
                "Rig {} · clip {}/{} · F {}/{} @{}fps",
                ui.rig_bone_names.len(),
                selected_clip.map_or_else(|| "—".to_owned(), |clip| (clip + 1).to_string()),
                ui.animation_clip_count,
                current_frame,
                last_frame,
                ui.animation_frame_rate as u32
            ),
            0.54,
        );
        let rig_buttons = [
            (
                inspector_left + 15.0,
                646.0,
                "Root",
                ui.rig_bone_names.is_empty(),
            ),
            (inspector_left + 87.0, 646.0, "Child", child_bone.is_some()),
            (
                inspector_left + 159.0,
                646.0,
                "Auto wt",
                ui.model_voxel_count > 0 && !ui.rig_bone_names.is_empty(),
            ),
            (
                inspector_left + 15.0,
                677.0,
                "New clip",
                !ui.rig_bone_names.is_empty(),
            ),
            (
                inspector_left + 87.0,
                677.0,
                "Clip -",
                selected_clip.is_some_and(|clip| clip > 0),
            ),
            (
                inspector_left + 159.0,
                677.0,
                "Clip +",
                selected_clip.is_some_and(|clip| clip + 1 < ui.animation_clip_count),
            ),
        ];
        for (left, top, label, enabled) in rig_buttons {
            rect(
                out,
                [left, top, 66.0, RIG_ACTION_ROW_HEIGHT],
                if enabled {
                    header
                } else {
                    [0.075, 0.085, 0.095, 1.0]
                },
            );
            let first_text = out.len();
            text(out, left + 6.0, top + 8.0, label, 0.58);
            if !enabled {
                for glyph in &mut out[first_text..] {
                    glyph.color[3] *= 0.38;
                }
            }
        }
        let can_adjust_weight = ui.selected_voxel_indices.len() == 1 && child_bone.is_some();
        let can_choose_mode =
            child_bone.is_some() && (ui.rig_edit_mode != RigEditMode::Rotate || can_adjust_weight);
        let mode_label = match ui.rig_edit_mode {
            RigEditMode::Move => "Move",
            RigEditMode::Rotate => "Rotate",
            RigEditMode::Weights => "Weights",
        };
        let mode_left = inspector_left + 15.0;
        rect(
            out,
            [mode_left, 708.0, 66.0, 27.0],
            if can_choose_mode {
                header
            } else {
                [0.075, 0.085, 0.095, 1.0]
            },
        );
        let first_text = out.len();
        text(out, mode_left + 5.0, 716.0, mode_label, 0.5);
        if !can_choose_mode {
            for glyph in &mut out[first_text..] {
                glyph.color[3] *= 0.38;
            }
        }

        let compact_columns = [
            inspector_left + 15.0,
            inspector_left + 70.0,
            inspector_left + 125.0,
            inspector_left + 180.0,
        ];
        let action_labels = if ui.rig_edit_mode == RigEditMode::Weights {
            ["Wt -", "Wt +", "", "Key"]
        } else {
            ["X -/+", "Y -/+", "Z -/+", "Key"]
        };
        for (index, (left, label)) in compact_columns.into_iter().zip(action_labels).enumerate() {
            let enabled = if index == 3 {
                can_edit_pose
            } else if ui.rig_edit_mode == RigEditMode::Weights {
                can_adjust_weight && index < 2
            } else {
                can_edit_pose
            };
            rect(
                out,
                [left, 739.0, 52.0, 27.0],
                if enabled {
                    header
                } else {
                    [0.075, 0.085, 0.095, 1.0]
                },
            );
            let first_text = out.len();
            text(out, left + 3.0, 747.0, label, 0.46);
            if !enabled {
                for glyph in &mut out[first_text..] {
                    glyph.color[3] *= 0.38;
                }
            }
        }
    } else if !ui.scene_document_available {
        inspector_section(out, "Live world", 296.0, inspector_left + 10.0);
    }
    if let Some(info) =
        info.filter(|_| !ui.model_document_available && !ui.scene_document_available)
    {
        text(
            out,
            inspector_text_x,
            326.0,
            &format!("Seed {}", info.seed),
            0.68,
        );
        text(
            out,
            inspector_text_x,
            345.0,
            &format!("Age {:.0} y", info.world_age_years),
            0.72,
        );
        text(
            out,
            inspector_text_x,
            364.0,
            if info.physics_enabled {
                "Physics active"
            } else {
                "Physics off | F6 to start"
            },
            0.64,
        );
        text(
            out,
            inspector_text_x,
            383.0,
            &format!("REVs {}", info.physics_cells),
            0.72,
        );
        text(
            out,
            inspector_text_x,
            402.0,
            &format!(
                "Plants {}",
                info.vegetation_summary.chars().take(18).collect::<String>()
            ),
            0.58,
        );
    }
    if !ui.standalone_editor_app && !ui.scene_document_available {
        let simulation_target = !ui.model_document_available;
        if simulation_target {
            text(out, inspector_text_x, 430.0, "Play starts the world", 0.62);
        }
        text(out, inspector_text_x, 448.0, "F2 switches workspaces", 0.62);
    }

    rect(out, [235.0, 67.0, scene_width, 31.0], header);
    text(
        out,
        250.0,
        75.0,
        if ui.model_document_available {
            "Voxel Model View"
        } else {
            "Scene View"
        },
        0.85,
    );
    if let Some(message) = ui.status_message.as_deref() {
        let message = message.chars().take(62).collect::<String>();
        let error = message.starts_with("Error:");
        rect(
            out,
            [245.0, 103.0, scene_width.min(540.0), 25.0],
            if error {
                [0.32, 0.12, 0.12, 0.96]
            } else {
                [0.08, 0.22, 0.19, 0.96]
            },
        );
        text(out, 252.0, 110.0, &message, 0.72);
    }
    if ui.file_menu_open {
        let mesh_import_available = ui.scene_document_available || ui.model_document_available;
        let menu_rows = [
            ("New Scene", true),
            ("Open Scene...", true),
            ("New Voxel Model", true),
            ("Open Voxel Model...", true),
            ("Import OBJ/GLB...", mesh_import_available),
            ("Add Voxel Model to Scene...", ui.scene_document_available),
            ("Save Scene", ui.scene_document_available),
            ("Save Voxel Model", ui.model_document_available),
            ("Save Model As...", ui.model_document_available),
            ("Export Model as OBJ...", ui.model_document_available),
            ("Create Impact Scene (16³)", ui.scene_document_available),
            ("Create Impact Scene (74³)", ui.scene_document_available),
            ("Save Physical State (paused)", ui.scene_document_available),
            ("Load Physical State", ui.scene_document_available),
        ];
        rect(
            out,
            [116.0, 27.0, 240.0, menu_rows.len() as f32 * 29.0 + 1.0],
            [0.055, 0.075, 0.09, 0.99],
        );
        for (row, (label, enabled)) in menu_rows.into_iter().enumerate() {
            let y = 30.0 + row as f32 * 29.0;
            rect(
                out,
                [119.0, y, 234.0, 27.0],
                if enabled {
                    [0.11, 0.15, 0.17, 1.0]
                } else {
                    [0.075, 0.09, 0.10, 1.0]
                },
            );
            text(out, 128.0, y + 8.0, label, 0.72);
        }
    }

    rect(out, [0.0, dock_top, inspector_left, 200.0], panel);
    rect(out, [0.0, dock_top, inspector_left, 32.0], header);
    text(out, 18.0, dock_top + 8.0, "Project", 0.8);
    for (index, label) in ["Scenes", "Voxel Models"].iter().enumerate() {
        let x = 8.0 + index as f32 * 108.0;
        rect(
            out,
            [x, dock_top + 36.0, 104.0, 26.0],
            if ui.project_category == index {
                selected
            } else {
                header
            },
        );
        text(out, x + 9.0, dock_top + 43.0, label, 0.7);
    }
    rect(
        out,
        [8.0, dock_top + 67.0, inspector_left - 16.0, 122.0],
        [0.065, 0.075, 0.085, 1.0],
    );
    let mut lines = if ui.project_category == 0 {
        if ui.scene_document_available {
            vec![
                format!(
                    "Scene: {}",
                    ui.active_scene_name.as_deref().unwrap_or("Untitled")
                ),
                format!("Model instances: {}", ui.scene_model_names.len()),
                "File menu: Save Scene / Add Voxel Model".to_owned(),
                "Click a model to select it; drag the selected model to move.".to_owned(),
            ]
        } else {
            vec![
                "No authored scene open".to_owned(),
                "Use File > New Scene or Open Scene".to_owned(),
            ]
        }
    } else if ui.model_document_available {
        vec![
            format!(
                "Model: {}",
                ui.active_model_name.as_deref().unwrap_or("Untitled")
            ),
            format!(
                "{} REVs | {} materials",
                ui.model_voxel_count, ui.model_material_count
            ),
            "Select: click a part, drag it to move; drag elsewhere for a box | Add REV: drag a row | Block: drag a rectangle".to_owned(),
        ]
    } else {
        vec![
            "No voxel model open".to_owned(),
            "Use File > New or Open Voxel Model".to_owned(),
        ]
    };
    if ui.project_category == 0 && ui.model_document_available && !ui.rig_bone_names.is_empty() {
        lines.push("Bones: drag a joint to pose it.".to_owned());
    }
    for (index, line) in lines.iter().enumerate() {
        text(out, 20.0, dock_top + 81.0 + index as f32 * 22.0, line, 0.72);
    }
    draw_coordinate_grid(out, ui);
    draw_animation_timeline(out, ui, inspector_left, canvas_height);
    clip_and_scroll_scene_inspector(out, ui);
    draw_scene_inspector_scrollbar(out, ui);
}

fn physics_cost_label(cost_ms_per_10ms: Option<f64>) -> String {
    match cost_ms_per_10ms {
        Some(cost) if cost.is_finite() && cost >= 0.0 => {
            format!("10 ms physics: {cost:.2} ms / 5 ms")
        }
        _ => "10 ms physics: — / 5 ms".to_owned(),
    }
}

fn simulation_rate_label(measured: Option<f64>, requested: f32) -> String {
    let actual = match measured {
        Some(rate) if rate.is_finite() && rate >= 0.0 => format!("{rate:.2}×"),
        _ => "—".to_owned(),
    };
    format!("Real {actual} / target {requested:.2}×")
}

fn physics_toolbar_label(cost: Option<f64>, measured: Option<f64>, requested: f32, compact: bool) -> String {
    if compact {
        let cost = cost.filter(|value| value.is_finite() && *value >= 0.0)
            .map_or_else(|| "—".to_owned(), |value| format!("{value:.1}"));
        let rate = measured.filter(|value| value.is_finite() && *value >= 0.0)
            .map_or_else(|| "—".to_owned(), |value| format!("{value:.2}×"));
        format!("{cost} ms · Real {rate}")
    } else {
        format!("{} · {}", physics_cost_label(cost), simulation_rate_label(measured, requested))
    }
}

fn coordinate_grid_bounds(ui: &Ui) -> [f32; 4] {
    [247.0, ui.editor_dock_top() - 116.0, 104.0, 104.0]
}

fn coordinate_projection(direction: Vec3, right: Vec3, up: Vec3) -> [f32; 2] {
    [direction.dot(right), -direction.dot(up)]
}

fn coordinate_line(
    out: &mut Vec<Rect>,
    from: [f32; 2],
    to: [f32; 2],
    thickness: f32,
    color: [f32; 4],
) {
    let delta = [to[0] - from[0], to[1] - from[1]];
    let steps = delta[0].abs().max(delta[1].abs()).ceil().max(1.0) as usize;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        rect(
            out,
            [
                from[0] + delta[0] * t - thickness * 0.5,
                from[1] + delta[1] * t - thickness * 0.5,
                thickness,
                thickness,
            ],
            color,
        );
    }
}

/// Camera-relative orientation grid. Its spacing is schematic, not world distance.
fn draw_coordinate_grid(out: &mut Vec<Rect>, ui: &Ui) {
    let [left, top, width, height] = coordinate_grid_bounds(ui);
    rect(out, [left, top, width, height], [0.035, 0.045, 0.055, 0.80]);
    let origin = [left + width * 0.5, top + height * 0.5];
    let project = |world: Vec3| {
        let p = coordinate_projection(world, ui.view_right, ui.view_up);
        [origin[0] + p[0], origin[1] + p[1]]
    };
    // The world uses Z up: the small ground grid lies in XY.
    for line in -2..=2 {
        let offset = line as f32 * 10.0;
        let color = [0.24, 0.29, 0.33, 0.52];
        coordinate_line(
            out,
            project(Vec3::new(-20.0, offset, 0.0)),
            project(Vec3::new(20.0, offset, 0.0)),
            1.0,
            color,
        );
        coordinate_line(
            out,
            project(Vec3::new(offset, -20.0, 0.0)),
            project(Vec3::new(offset, 20.0, 0.0)),
            1.0,
            color,
        );
    }
    let axes = [
        (Vec3::X, "X", [0.92, 0.29, 0.26, 1.0]),
        (Vec3::Y, "Y", [0.36, 0.80, 0.35, 1.0]),
        (Vec3::Z, "Z", [0.34, 0.62, 1.0, 1.0]),
    ];
    for (axis, label, color) in axes {
        let mut dim = color;
        dim[3] = 0.30;
        coordinate_line(out, origin, project(-axis * 26.0), 1.0, dim);
        let end = project(axis * 32.0);
        coordinate_line(out, origin, end, 2.0, color);
        rect(out, [end[0] - 2.0, end[1] - 2.0, 4.0, 4.0], color);
        let mut label_position = project(axis * 41.0);
        if (end[0] - origin[0]).hypot(end[1] - origin[1]) < 3.0 {
            label_position = [origin[0] + 10.0, origin[1] + 10.0];
        }
        let start = out.len();
        text(
            out,
            label_position[0] - 3.0,
            label_position[1] - 5.0,
            label,
            0.72,
        );
        for glyph in &mut out[start..] {
            glyph.color[0] = color[0];
            glyph.color[1] = color[1];
            glyph.color[2] = color[2];
        }
    }
    rect(
        out,
        [origin[0] - 2.0, origin[1] - 2.0, 4.0, 4.0],
        [0.83, 0.91, 0.93, 1.0],
    );
}

fn draw_animation_timeline(out: &mut Vec<Rect>, ui: &Ui, inspector_left: f32, canvas_height: f32) {
    let height = ui.animation_timeline_height();
    if height <= 0.0 {
        return;
    }
    let top = canvas_height - height;
    let panel = [0.065, 0.075, 0.085, 1.0];
    let header = [0.12, 0.14, 0.16, 1.0];
    let accent = [0.45, 0.92, 0.79, 1.0];
    let enabled = ui.selected_animation_clip.is_some();
    rect(out, [0.0, top, inspector_left, height], panel);
    rect(out, [0.0, top, inspector_left, 30.0], header);
    text(out, 16.0, top + 8.0, "Timeline", 0.72);

    rect(
        out,
        [92.0, top + 3.0, 66.0, 24.0],
        if enabled {
            [0.16, 0.31, 0.38, 1.0]
        } else {
            [0.075, 0.085, 0.095, 1.0]
        },
    );
    let play_label = if ui.animation_playing {
        "Pause"
    } else {
        "Play"
    };
    let play_text_start = out.len();
    text(out, 108.0, top + 10.0, play_label, 0.62);
    if !enabled {
        for glyph in &mut out[play_text_start..] {
            glyph.color[3] *= 0.38;
        }
    }

    let current_frame = ui.animation_current_frame();
    let last_frame = ui.animation_last_frame();
    let clip_index = ui.selected_animation_clip.map_or(0, |clip| clip + 1);
    text(
        out,
        170.0,
        top + 10.0,
        &format!(
            "Clip {}/{}   F {}/{}   {:.2} / {:.2} s   {} fps",
            clip_index,
            ui.animation_clip_count,
            current_frame,
            last_frame,
            ui.selected_animation_time_seconds,
            ui.animation_duration_seconds,
            ui.animation_frame_rate as u32
        ),
        0.62,
    );
    rect(out, [480.0, top + 3.0, 36.0, 24.0], [0.16, 0.19, 0.21, 1.0]);
    text(out, 493.0, top + 10.0, "-", 0.72);
    text(
        out,
        525.0,
        top + 10.0,
        &format!("Speed {:.2}x", ui.animation_playback_speed),
        0.62,
    );
    rect(out, [624.0, top + 3.0, 36.0, 24.0], [0.16, 0.19, 0.21, 1.0]);
    text(out, 637.0, top + 10.0, "+", 0.72);
    let collapse_x = inspector_left - 38.0;
    rect(
        out,
        [inspector_left - 118.0, top + 3.0, 70.0, 24.0],
        if ui.show_rig_overlay {
            [0.16, 0.31, 0.38, 1.0]
        } else {
            [0.075, 0.085, 0.095, 1.0]
        },
    );
    text(out, inspector_left - 106.0, top + 10.0, "Bones", 0.62);
    rect(
        out,
        [collapse_x, top + 3.0, 30.0, 24.0],
        [0.16, 0.19, 0.21, 1.0],
    );
    text(
        out,
        collapse_x + 11.0,
        top + 9.0,
        if ui.animation_timeline_expanded {
            "-"
        } else {
            "+"
        },
        0.72,
    );

    if !ui.animation_timeline_expanded {
        return;
    }
    if !enabled {
        text(
            out,
            18.0,
            top + 57.0,
            "Add an animation clip and keyframes to preview motion here.",
            0.68,
        );
        return;
    }

    let (slider_left, slider_width) = animation_slider_geometry(inspector_left);
    let track_y = top + 77.0;
    let frame_fraction = if last_frame == 0 {
        0.0
    } else {
        current_frame as f32 / last_frame as f32
    }
    .clamp(0.0, 1.0);
    rect(
        out,
        [slider_left, track_y, slider_width, 5.0],
        [0.15, 0.18, 0.20, 1.0],
    );
    rect(
        out,
        [slider_left, track_y, slider_width * frame_fraction, 5.0],
        [0.20, 0.38, 0.39, 1.0],
    );
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let x = slider_left + slider_width * fraction;
        rect(out, [x, track_y - 4.0, 1.0, 14.0], [0.42, 0.47, 0.49, 1.0]);
        let frame = (last_frame as f32 * fraction).round() as usize;
        text(out, x - 3.0, track_y + 20.0, &frame.to_string(), 0.5);
    }
    for key_time in &ui.animation_keyframe_times {
        let key_frame = (*key_time * ui.animation_frame_rate).round() as usize;
        let key_fraction = if last_frame == 0 {
            0.0
        } else {
            key_frame as f32 / last_frame as f32
        }
        .clamp(0.0, 1.0);
        let x = slider_left + slider_width * key_fraction;
        rect(
            out,
            [x - 1.0, track_y - 7.0, 3.0, 18.0],
            [0.95, 0.78, 0.35, 1.0],
        );
    }
    let thumb_x = slider_left + slider_width * frame_fraction;
    rect(out, [thumb_x - 3.0, track_y - 8.0, 7.0, 21.0], accent);
}

fn mesh_import_modal_origin(canvas_width: f32, canvas_height: f32) -> (f32, f32) {
    (
        (canvas_width - 560.0) * 0.5,
        ((canvas_height - 430.0) * 0.5).max(8.0),
    )
}

fn draw_mesh_import_modal(out: &mut Vec<Rect>, ui: &Ui, canvas_width: f32, canvas_height: f32) {
    let (left, top) = mesh_import_modal_origin(canvas_width, canvas_height);
    rect(
        out,
        [0.0, 0.0, canvas_width, canvas_height],
        [0.0, 0.0, 0.0, 0.64],
    );
    rect(out, [left, top, 560.0, 430.0], [0.085, 0.10, 0.115, 1.0]);
    rect(out, [left, top, 560.0, 38.0], [0.13, 0.16, 0.18, 1.0]);
    text(out, left + 20.0, top + 12.0, "Import 3D Mesh", 0.95);
    text(out, left + 24.0, top + 51.0, "Source file", 0.7);
    rect(
        out,
        [left + 24.0, top + 66.0, 512.0, 28.0],
        [0.055, 0.07, 0.08, 1.0],
    );
    let filename = ui
        .mesh_import_filename
        .as_deref()
        .unwrap_or("No file selected");
    let visible_filename = filename.chars().take(66).collect::<String>();
    text(out, left + 32.0, top + 76.0, &visible_filename, 0.7);

    text(
        out,
        left + 24.0,
        top + 105.0,
        &format!("Uniform scale  {:.2}×", ui.mesh_import_scale),
        0.72,
    );
    rect(
        out,
        [left + 24.0, top + 130.0, 262.0, 8.0],
        [0.18, 0.26, 0.29, 1.0],
    );
    let scale_fraction = ((ui.mesh_import_scale - 0.1) / 9.9).clamp(0.0, 1.0);
    rect(
        out,
        [left + 24.0, top + 130.0, 262.0 * scale_fraction, 8.0],
        [0.18, 0.65, 0.56, 1.0],
    );
    rect(
        out,
        [left + 20.0 + 262.0 * scale_fraction, top + 125.0, 8.0, 18.0],
        [0.5, 0.9, 0.75, 1.0],
    );
    text(out, left + 24.0, top + 145.0, "0.1×", 0.62);
    text(out, left + 254.0, top + 145.0, "10×", 0.62);
    text(out, left + 310.0, top + 54.0, "Voxelization", 0.7);
    for (filled, label, x, width) in [
        (false, "Shell", left + 310.0, 108.0),
        (true, "Filled", left + 422.0, 114.0),
    ] {
        rect(
            out,
            [x, top + 83.0, width, 32.0],
            if ui.mesh_import_filled == filled {
                [0.16, 0.38, 0.38, 1.0]
            } else {
                [0.12, 0.15, 0.17, 1.0]
            },
        );
        text(out, x + 27.0, top + 94.0, label, 0.72);
    }

    let build_label = if ui.mesh_import_busy {
        "Building Preview..."
    } else {
        "Build Preview"
    };
    rect(
        out,
        [left + 24.0, top + 164.0, 512.0, 34.0],
        if ui.mesh_import_busy {
            [0.12, 0.18, 0.19, 1.0]
        } else {
            [0.15, 0.48, 0.41, 1.0]
        },
    );
    text(out, left + 220.0, top + 176.0, build_label, 0.75);
    rect(
        out,
        [left + 24.0, top + 210.0, 512.0, 142.0],
        [0.055, 0.07, 0.08, 1.0],
    );
    if let Some(error) = ui.mesh_import_error.as_deref() {
        text(out, left + 38.0, top + 229.0, "Preview failed", 0.75);
        let visible_error = error.chars().take(70).collect::<String>();
        text(out, left + 38.0, top + 250.0, &visible_error, 0.62);
    } else if ui.mesh_import_busy {
        text(
            out,
            left + 38.0,
            top + 235.0,
            "Preparing voxel preview...",
            0.75,
        );
    } else if let Some(preview) = ui.mesh_import_preview {
        text(out, left + 38.0, top + 224.0, "Preview ready", 0.75);
        text(
            out,
            left + 38.0,
            top + 247.0,
            if preview.watertight {
                "Mesh is watertight"
            } else if ui.mesh_import_filled {
                "Filled on voxel grid; source mesh has openings"
            } else {
                "Mesh is not watertight"
            },
            0.68,
        );
        text(
            out,
            left + 38.0,
            top + 269.0,
            &format!(
                "Dimensions {:.2} × {:.2} × {:.2} m",
                preview.dimensions_m[0], preview.dimensions_m[1], preview.dimensions_m[2]
            ),
            0.68,
        );
        text(
            out,
            left + 38.0,
            top + 291.0,
            &format!("Voxel cells {}", preview.cell_count),
            0.68,
        );
    } else {
        text(
            out,
            left + 38.0,
            top + 235.0,
            "Choose scale and fill mode, then build a preview.",
            0.7,
        );
    }

    let can_commit = !ui.mesh_import_busy && ui.mesh_import_preview.is_some();
    rect(
        out,
        [left + 24.0, top + 381.0, 224.0, 34.0],
        if can_commit {
            [0.12, 0.38, 0.33, 1.0]
        } else {
            [0.09, 0.12, 0.13, 1.0]
        },
    );
    text(out, left + 43.0, top + 393.0, "Create / Open Model", 0.68);
    if ui.scene_document_available {
        rect(
            out,
            [left + 260.0, top + 381.0, 178.0, 34.0],
            if can_commit {
                [0.12, 0.38, 0.33, 1.0]
            } else {
                [0.09, 0.12, 0.13, 1.0]
            },
        );
        text(out, left + 279.0, top + 393.0, "Add to Scene", 0.68);
    }
    rect(
        out,
        [left + 448.0, top + 381.0, 88.0, 34.0],
        [0.12, 0.15, 0.17, 1.0],
    );
    text(out, left + 469.0, top + 393.0, "Cancel", 0.68);
}

fn inspector_section(out: &mut Vec<Rect>, title: &str, y: f32, x: f32) {
    rect(out, [x, y, 230.0, 24.0], [0.12, 0.14, 0.16, 1.0]);
    text(out, x + 8.0, y + 6.0, title, 0.72);
}

/// Clip numeric text to its own field, keeping the insertion end visible while typing.
fn clipped_inspector_text(
    out: &mut Vec<Rect>,
    area: [f32; 4],
    value: &str,
    scale: f32,
    editing: bool,
) {
    let start = out.len();
    text(out, area[0], area[1], value, scale);
    let right = out[start..]
        .iter()
        .map(|r| r.rect[0] + r.rect[2])
        .fold(area[0], f32::max);
    let shift = if editing {
        (right - area[0] - area[2]).max(0.0)
    } else {
        0.0
    };
    for glyph in &mut out[start..] {
        let x = glyph.rect[0] - shift;
        let y = glyph.rect[1];
        let clipped_x = x.max(area[0]);
        let clipped_y = y.max(area[1]);
        glyph.rect = [
            clipped_x,
            clipped_y,
            ((x + glyph.rect[2]).min(area[0] + area[2]) - clipped_x).max(0.0),
            ((y + glyph.rect[3]).min(area[1] + area[3]) - clipped_y).max(0.0),
        ];
    }
    let mut write = start;
    for read in start..out.len() {
        if out[read].rect[2] > 0.0 && out[read].rect[3] > 0.0 {
            out[write] = out[read];
            write += 1;
        }
    }
    out.truncate(write);
}

fn draw_scene_inspector_contents(
    out: &mut Vec<Rect>,
    ui: &Ui,
    left: f32,
    header: [f32; 4],
    selected: [f32; 4],
) {
    let rows = scene_inspector_rows(ui);
    let has_selection = ui
        .selected_scene_model
        .is_some_and(|index| index < ui.scene_model_names.len());
    let axis_labels = ["X", "Y", "Z"];
    for row in rows {
        let y = row.top;
        match row.item {
            SceneInspectorItem::Section(section) => {
                let index = scene_section_index(section);
                let title = match section {
                    SceneInspectorSection::Transform => "Transform",
                    SceneInspectorSection::Dynamics => "Dynamics",
                    SceneInspectorSection::Material => "Material",
                    SceneInspectorSection::Welding => "Thermal welding",
                    SceneInspectorSection::Simulation => "Simulation / physics",
                };
                let marker = if ui.scene_inspector_collapsed[index] {
                    ">"
                } else {
                    "v"
                };
                rect(out, [left + 10.0, y, 230.0, row.height], header);
                text(out, left + 19.0, y + 7.0, marker, 0.62);
                text(out, left + 34.0, y + 7.0, title, 0.68);
            }
            SceneInspectorItem::Actions => {
                for (column, label) in ["Duplicate", "Delete", "Frame"].into_iter().enumerate() {
                    let x = left + 10.0 + column as f32 * 73.0;
                    rect(
                        out,
                        [x, y, 69.0, row.height],
                        if column == 1 {
                            [0.30, 0.13, 0.13, 1.0]
                        } else {
                            header
                        },
                    );
                    text(out, x + 7.0, y + 8.0, label, 0.56);
                }
            }
            SceneInspectorItem::Position
            | SceneInspectorItem::Rotation
            | SceneInspectorItem::Velocity
            | SceneInspectorItem::Force => {
                let (label, target, values) = match row.item {
                    SceneInspectorItem::Position => (
                        "Position (m)",
                        SceneInputTarget::Position,
                        ui.selected_scene_model_info
                            .as_ref()
                            .map(|info| info.translation_m.map(|v| format!("{v:.3}"))),
                    ),
                    SceneInspectorItem::Rotation => (
                        "Rotation (90° steps)",
                        SceneInputTarget::Rotation,
                        ui.selected_scene_model_info
                            .as_ref()
                            .map(|info| info.rotation_deg.map(|v| format!("{v:.0}"))),
                    ),
                    SceneInspectorItem::Velocity => (
                        "Velocity (m/s)",
                        SceneInputTarget::Velocity,
                        ui.selected_scene_model_info
                            .as_ref()
                            .map(|info| info.velocity_m_s.map(|v| format!("{v:.3}"))),
                    ),
                    _ => (
                        "Force (N)",
                        SceneInputTarget::Force,
                        ui.selected_scene_model_info
                            .as_ref()
                            .map(|info| info.force_n.map(|v| format!("{v:.2}"))),
                    ),
                };
                text(out, left + 14.0, y + 3.0, label, 0.64);
                let active = ui
                    .scene_input
                    .as_ref()
                    .filter(|input| input.target == target);
                let original =
                    values.unwrap_or_else(|| [String::new(), String::new(), String::new()]);
                for axis in 0..3 {
                    let x = left + 12.0 + axis as f32 * 76.0;
                    let focused = active.is_some_and(|input| input.axis == axis);
                    rect(
                        out,
                        [x, y + 21.0, 71.0, 24.0],
                        if focused {
                            [0.12, 0.32, 0.31, 1.0]
                        } else {
                            [0.035, 0.05, 0.06, 1.0]
                        },
                    );
                    let value =
                        active.map_or(original[axis].as_str(), |input| input.values[axis].as_str());
                    let cursor = focused.then_some("|").unwrap_or("");
                    text(out, x + 5.0, y + 27.0, axis_labels[axis], 0.58);
                    clipped_inspector_text(
                        out,
                        [x + 17.0, y + 26.0, 50.0, 17.0],
                        &format!("{value}{cursor}"),
                        0.62,
                        focused,
                    );
                }
            }
            SceneInspectorItem::Temperature => {
                text(out, left + 14.0, y + 8.0, "Initial temperature (K)", 0.55);
                let active = ui
                    .scene_input
                    .as_ref()
                    .filter(|input| input.target == SceneInputTarget::InitialTemperature);
                let value = active.map_or_else(
                    || {
                        ui.selected_scene_initial_temperature_kelvin
                            .map_or_else(|| "Default".to_owned(), |v| format!("{v:.1}"))
                    },
                    |input| input.values[0].clone(),
                );
                let focused = active.is_some();
                rect(
                    out,
                    [left + 154.0, y + 3.0, 81.0, row.height - 6.0],
                    if focused {
                        [0.12, 0.32, 0.31, 1.0]
                    } else {
                        [0.035, 0.05, 0.06, 1.0]
                    },
                );
                text(
                    out,
                    left + 158.0,
                    y + 9.0,
                    &format!(
                        "{}{}",
                        value.chars().take(11).collect::<String>(),
                        if focused { "|" } else { "" }
                    ),
                    0.49,
                );
            }
            SceneInspectorItem::Bounds => {
                if let Some(bounds) = ui.selected_bounds_m {
                    let size =
                        std::array::from_fn::<_, 3, _>(|axis| bounds[1][axis] - bounds[0][axis]);
                    let scale = if size.iter().copied().fold(0.0f64, f64::max) < 1.0 {
                        100.0
                    } else {
                        1.0
                    };
                    let unit = if scale == 100.0 { "cm" } else { "m" };
                    text(
                        out,
                        left + 14.0,
                        y + 6.0,
                        &format!(
                            "Size {:.1} × {:.1} × {:.1} {unit}",
                            size[0] * scale,
                            size[1] * scale,
                            size[2] * scale
                        ),
                        0.54,
                    );
                }
            }
            SceneInspectorItem::MaterialCycle => {
                let material = ui
                    .selected_scene_model_info
                    .as_ref()
                    .map_or("Unknown", |info| info.material_name.as_str());
                rect(
                    out,
                    [left + 10.0, y + 2.0, 230.0, row.height - 4.0],
                    selected,
                );
                text(
                    out,
                    left + 18.0,
                    y + 9.0,
                    &format!(
                        "Cycle material · {}",
                        material.chars().take(22).collect::<String>()
                    ),
                    0.56,
                );
            }
            SceneInspectorItem::MaterialActions => {
                for (column, label) in ["Load", "Save", "New"].into_iter().enumerate() {
                    let x = left + 10.0 + column as f32 * 73.0;
                    rect(out, [x, y + 1.0, 69.0, row.height - 2.0], header);
                    text(out, x + 20.0, y + 8.0, label, 0.58);
                }
            }
            SceneInspectorItem::WeldingToggle => {
                let label = if ui.scene_thermal_bonding.is_some() {
                    "Enabled · click to disable"
                } else {
                    "Disabled · configure to enable"
                };
                rect(
                    out,
                    [left + 10.0, y + 1.0, 230.0, row.height - 2.0],
                    if ui.scene_thermal_bonding.is_some() {
                        selected
                    } else {
                        header
                    },
                );
                text(out, left + 18.0, y + 8.0, label, 0.56);
            }
            SceneInspectorItem::WeldingConfigure => {
                rect(out, [left + 10.0, y + 1.0, 230.0, row.height - 2.0], header);
                text(out, left + 18.0, y + 8.0, "Edit welding thresholds…", 0.56);
            }
            SceneInspectorItem::PhysicsField(field) => {
                let labels = [
                    "Time step (s)",
                    "Substeps",
                    "Gravity X (m/s²)",
                    "Gravity Y (m/s²)",
                    "Gravity Z (m/s²)",
                    "Restitution",
                    "Friction scale",
                    "Solver iterations",
                    "Solver tolerance",
                ];
                let values = ui
                    .scene_physics_input
                    .as_ref()
                    .map(|input| &input.values)
                    .cloned()
                    .unwrap_or_else(|| scene_physics_strings(ui.scene_physics));
                let focused = ui
                    .scene_physics_input
                    .as_ref()
                    .is_some_and(|input| input.field == field);
                text(out, left + 14.0, y + 7.0, labels[field], 0.52);
                rect(
                    out,
                    [left + 147.0, y + 2.0, 88.0, row.height - 4.0],
                    if focused {
                        [0.12, 0.32, 0.31, 1.0]
                    } else {
                        [0.035, 0.05, 0.06, 1.0]
                    },
                );
                text(
                    out,
                    left + 152.0,
                    y + 7.0,
                    &format!(
                        "{}{}",
                        values[field].chars().take(13).collect::<String>(),
                        if focused { "|" } else { "" }
                    ),
                    0.48,
                );
            }
            SceneInspectorItem::PhysicsApply => {
                rect(
                    out,
                    [left + 10.0, y + 1.0, 230.0, row.height - 2.0],
                    selected,
                );
                text(out, left + 29.0, y + 8.0, "Apply simulation settings", 0.65);
            }
            SceneInspectorItem::Hint => {
                text(
                    out,
                    left + 14.0,
                    y + 8.0,
                    if has_selection {
                        "Inspector data is loading"
                    } else {
                        "Select a model instance to edit"
                    },
                    0.55,
                );
            }
            SceneInspectorItem::InputHint => {
                text(
                    out,
                    left + 14.0,
                    y + 5.0,
                    "Click value · type · Enter applies · Esc cancels",
                    0.48,
                );
            }
        }
    }
}

fn scene_section_index(section: SceneInspectorSection) -> usize {
    match section {
        SceneInspectorSection::Transform => 0,
        SceneInspectorSection::Dynamics => 1,
        SceneInspectorSection::Material => 2,
        SceneInspectorSection::Welding => 3,
        SceneInspectorSection::Simulation => 4,
    }
}

fn push_scene_row(
    rows: &mut Vec<SceneInspectorRow>,
    y: &mut f32,
    item: SceneInspectorItem,
    height: f32,
) {
    rows.push(SceneInspectorRow {
        item,
        top: *y,
        height,
    });
    *y += height;
}

fn scene_inspector_rows(ui: &Ui) -> Vec<SceneInspectorRow> {
    let selected = ui
        .selected_scene_model
        .is_some_and(|index| index < ui.scene_model_names.len());
    let has_inspector_data = selected && ui.selected_scene_model_info.is_some();
    let mut rows = Vec::new();
    let mut y = 148.0;
    if selected {
        push_scene_row(&mut rows, &mut y, SceneInspectorItem::Actions, 28.0);
        y += 4.0;
    }

    let sections = [
        SceneInspectorSection::Transform,
        SceneInspectorSection::Dynamics,
        SceneInspectorSection::Material,
        SceneInspectorSection::Welding,
        SceneInspectorSection::Simulation,
    ];
    if !selected {
        push_scene_row(&mut rows, &mut y, SceneInspectorItem::Hint, 30.0);
    }
    for section in sections {
        if !selected
            && matches!(
                section,
                SceneInspectorSection::Transform
                    | SceneInspectorSection::Dynamics
                    | SceneInspectorSection::Material
            )
        {
            continue;
        }
        push_scene_row(
            &mut rows,
            &mut y,
            SceneInspectorItem::Section(section),
            24.0,
        );
        if ui.scene_inspector_collapsed[scene_section_index(section)] {
            y += 4.0;
            continue;
        }
        match section {
            SceneInspectorSection::Transform if has_inspector_data => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Position, 48.0);
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Rotation, 48.0);
                if ui.selected_bounds_m.is_some() {
                    push_scene_row(&mut rows, &mut y, SceneInspectorItem::Bounds, 22.0);
                }
            }
            SceneInspectorSection::Transform => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Hint, 26.0);
            }
            SceneInspectorSection::Dynamics if has_inspector_data => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Velocity, 48.0);
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Force, 48.0);
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Temperature, 28.0);
            }
            SceneInspectorSection::Dynamics => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Hint, 26.0);
            }
            SceneInspectorSection::Material if selected => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::MaterialCycle, 28.0);
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::MaterialActions, 28.0);
            }
            SceneInspectorSection::Material => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::Hint, 26.0);
            }
            SceneInspectorSection::Welding => {
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::WeldingToggle, 28.0);
                if ui.scene_thermal_bonding.is_some() {
                    push_scene_row(
                        &mut rows,
                        &mut y,
                        SceneInspectorItem::WeldingConfigure,
                        28.0,
                    );
                }
            }
            SceneInspectorSection::Simulation => {
                for field in 0..9 {
                    push_scene_row(
                        &mut rows,
                        &mut y,
                        SceneInspectorItem::PhysicsField(field),
                        23.0,
                    );
                }
                push_scene_row(&mut rows, &mut y, SceneInspectorItem::PhysicsApply, 28.0);
            }
        }
        y += 5.0;
    }
    if ui.scene_input.is_some() || ui.scene_physics_input.is_some() {
        push_scene_row(&mut rows, &mut y, SceneInspectorItem::InputHint, 20.0);
    }
    rows
}

fn scene_inspector_content_bottom(ui: &Ui) -> f32 {
    scene_inspector_rows(ui)
        .last()
        .map_or(170.0, |row| row.top + row.height + 8.0)
}

fn scene_physics_strings(values: ScenePhysicsValues) -> [String; 9] {
    [
        format!("{:.5}", values.fixed_dt),
        values.substeps.to_string(),
        format!("{:.3}", values.gravity[0]),
        format!("{:.3}", values.gravity[1]),
        format!("{:.3}", values.gravity[2]),
        format!("{:.3}", values.restitution),
        format!("{:.3}", values.friction_scale),
        values.iterations.to_string(),
        format!("{:.2e}", values.tolerance),
    ]
}

fn scene_vector_axis(x: f32, inspector_left: f32) -> Option<usize> {
    let left = inspector_left + 12.0;
    (0..3).find(|axis| {
        let column_left = left + *axis as f32 * 76.0;
        (column_left..=column_left + 71.0).contains(&x)
    })
}

fn default_welding_input() -> WeldingInputState {
    let settings = ThermalBondingSettings {
        material_name: "iron".into(),
        minimum_mass_fraction: 0.5,
        minimum_temperature_kelvin: 1000.0,
        minimum_impact_energy_j: 0.0,
        minimum_contact_pressure_pa: 1.0e6,
        minimum_dwell_seconds: 0.02,
    };
    welding_input_from_settings(&settings)
}

fn welding_input_from_settings(settings: &ThermalBondingSettings) -> WeldingInputState {
    WeldingInputState {
        values: [
            settings.material_name.clone(),
            settings.minimum_mass_fraction.to_string(),
            settings.minimum_temperature_kelvin.to_string(),
            settings.minimum_impact_energy_j.to_string(),
            settings.minimum_contact_pressure_pa.to_string(),
            settings.minimum_dwell_seconds.to_string(),
        ],
        field: 0,
        editing: false,
    }
}

fn draw_welding_input_modal(
    out: &mut Vec<Rect>,
    input: &WeldingInputState,
    width: f32,
    height: f32,
) {
    let left = (width - 500.0) * 0.5;
    let top = (height - 430.0) * 0.5;
    rect(out, [0.0, 0.0, width, height], [0.0, 0.0, 0.0, 0.58]);
    rect(out, [left, top, 500.0, 430.0], [0.07, 0.085, 0.095, 1.0]);
    text(
        out,
        left + 20.0,
        top + 18.0,
        "Thermal welding thresholds",
        1.0,
    );
    text(
        out,
        left + 20.0,
        top + 48.0,
        "Tab selects a field; Enter applies; Esc cancels",
        0.62,
    );
    let labels = [
        "Material component",
        "Minimum mass fraction (0–1)",
        "Minimum temperature (K)",
        "Minimum impact energy (J; 0 disables)",
        "Minimum contact pressure (Pa)",
        "Minimum dwell time (s)",
    ];
    for (index, label) in labels.into_iter().enumerate() {
        let y = top + 88.0 + index as f32 * 42.0;
        text(out, left + 22.0, y + 1.0, label, 0.62);
        rect(
            out,
            [left + 258.0, y - 5.0, 220.0, 30.0],
            if input.field == index {
                [0.12, 0.32, 0.31, 1.0]
            } else {
                [0.035, 0.05, 0.06, 1.0]
            },
        );
        text(
            out,
            left + 268.0,
            y + 3.0,
            &input.values[index].chars().take(27).collect::<String>(),
            0.62,
        );
    }
    rect(
        out,
        [left + 20.0, top + 362.0, 460.0, 42.0],
        [0.15, 0.52, 0.44, 1.0],
    );
    text(out, left + 205.0, top + 374.0, "Apply settings", 0.75);
}

fn inspector_scroll_geometry(ui: &Ui) -> [f32; 5] {
    let track_top = 144.0;
    let track_bottom = (ui.canvas_height - 8.0).max(track_top + 1.0);
    let track_height = track_bottom - track_top;
    let content_bottom = if ui.model_document_available {
        766.0
    } else if ui.scene_document_available {
        scene_inspector_content_bottom(ui)
    } else {
        455.0
    };
    let content_height = (content_bottom - track_top).max(track_height);
    let max_scroll = (content_bottom - track_bottom).max(0.0);
    let thumb_height = (track_height * track_height / content_height).clamp(24.0, track_height);
    let scroll = ui.scene_inspector_scroll.clamp(0.0, max_scroll);
    let thumb_top = track_top + (scroll / max_scroll.max(1.0)) * (track_height - thumb_height);
    [track_top, track_height, thumb_top, thumb_height, max_scroll]
}

fn clip_and_scroll_scene_inspector(out: &mut [Rect], ui: &Ui) {
    let [track_top, track_height, _, _, max_scroll] = inspector_scroll_geometry(ui);
    let scroll = ui.scene_inspector_scroll.clamp(0.0, max_scroll);
    let left = ui.canvas_width - 250.0;
    let right = ui.canvas_width - 12.0;
    let bottom = track_top + track_height;
    for item in out {
        if item.rect[0] < left || item.rect[0] >= ui.canvas_width || item.rect[1] < track_top {
            continue;
        }
        item.rect[1] -= scroll;
        let x0 = item.rect[0].max(left).min(right);
        let x1 = (item.rect[0] + item.rect[2]).min(right);
        let y0 = item.rect[1].max(track_top).min(bottom);
        let y1 = (item.rect[1] + item.rect[3]).min(bottom);
        if x1 <= x0 || y1 <= y0 {
            item.rect = [x0, y0, 0.0, 0.0];
            item.color[3] = 0.0;
        } else {
            item.rect = [x0, y0, x1 - x0, y1 - y0];
        }
    }
}

fn draw_scene_inspector_scrollbar(out: &mut Vec<Rect>, ui: &Ui) {
    if !ui.scene_document_available && !ui.model_document_available {
        return;
    }
    let [track_top, track_height, thumb_top, thumb_height, max_scroll] =
        inspector_scroll_geometry(ui);
    if max_scroll <= 0.0 {
        return;
    }
    let x = ui.canvas_width - 9.0;
    rect(
        out,
        [x, track_top, 4.0, track_height],
        [0.035, 0.05, 0.06, 1.0],
    );
    rect(
        out,
        [x - 2.0, thumb_top, 8.0, thumb_height],
        [0.30, 0.58, 0.54, 1.0],
    );
}

fn menu_button(out: &mut Vec<Rect>, x: f32, y: f32, w: f32, h: f32, label: &str, primary: bool) {
    rect(
        out,
        [x, y, w, h],
        if primary {
            [0.15, 0.52, 0.44, 1.]
        } else {
            [0.12, 0.19, 0.23, 1.]
        },
    );
    text(out, x + 24., y + 16., label, 1.5);
}
fn settings_slider_rect(canvas_width: f32, row: usize) -> [f32; 4] {
    let y = [246.0, 352.0, 458.0].get(row).copied().unwrap_or(458.0);
    [292.0, y, (canvas_width - 352.0).max(260.0), 8.0]
}

fn settings_control_row_top(row: usize) -> f32 {
    252.0 + row as f32 * 34.0
}

fn draw_settings_workspace(out: &mut Vec<Rect>, ui: &Ui) {
    let width = ui.canvas_width;
    let height = ui.canvas_height;
    let content_left = 264.0;
    let content_width = (width - 296.0).max(360.0);
    let content_right = content_left + content_width;
    let bg = [0.025, 0.032, 0.042, 1.0];
    let topbar = [0.05, 0.063, 0.072, 1.0];
    let sidebar = [0.035, 0.045, 0.055, 1.0];
    let panel = [0.065, 0.08, 0.09, 1.0];
    let row = [0.08, 0.10, 0.11, 1.0];
    let accent = [0.18, 0.48, 0.42, 1.0];

    rect(out, [0.0, 0.0, width, height], bg);
    rect(out, [0.0, 0.0, width, 68.0], topbar);
    rect(out, [0.0, 66.0, width, 2.0], accent);
    text(out, 28.0, 22.0, "Settings", 1.65);
    text(out, 150.0, 27.0, "Voxel Editor", 0.88);
    settings_button(out, width - 116.0, 14.0, 92.0, 40.0, "Done", true);

    rect(out, [0.0, 68.0, 232.0, (height - 68.0).max(0.0)], sidebar);
    rect(
        out,
        [231.0, 68.0, 1.0, (height - 68.0).max(0.0)],
        [0.26, 0.31, 0.33, 1.0],
    );
    text(out, 24.0, 96.0, "PREFERENCES", 0.78);
    for (tab, top, label) in [
        (0, 124.0, "General"),
        (1, 172.0, "Controls"),
        (2, 220.0, "Plugins"),
    ] {
        settings_nav_item(out, 16.0, top, 200.0, 42.0, label, ui.settings_tab == tab);
    }

    let (title, subtitle) = match ui.settings_tab {
        1 => (
            "Controls",
            "Change the editor shortcuts used to manipulate voxel models.",
        ),
        2 => (
            "Plugins",
            "Manage local integrations available to the editor.",
        ),
        _ => ("General", "Adjust simulation and first-person movement."),
    };
    text(out, content_left, 98.0, title, 1.8);
    text(out, content_left, 132.0, subtitle, 0.9);

    match ui.settings_tab {
        1 => {
            let card_height = (height - 218.0).max(438.0);
            rect(
                out,
                [content_left, 176.0, content_width, card_height],
                panel,
            );
            text(out, content_left + 24.0, 194.0, "Model editing", 1.08);
            text(
                out,
                content_left + 24.0,
                218.0,
                "Player movement remains on WASD.",
                0.82,
            );
            for (index, action) in crate::app::editor_settings::REBINDABLE_ACTIONS
                .iter()
                .enumerate()
            {
                let y = settings_control_row_top(index);
                rect(
                    out,
                    [content_left + 16.0, y, content_width - 32.0, 30.0],
                    row,
                );
                text(
                    out,
                    content_left + 30.0,
                    y + 8.0,
                    crate::app::editor_settings::label(action),
                    0.92,
                );
                let binding = ui
                    .settings_bindings
                    .get(*action)
                    .map(String::as_str)
                    .unwrap_or("—");
                let is_capturing = ui.settings_rebind_action.as_deref() == Some(*action);
                settings_button(
                    out,
                    width - 184.0,
                    y + 2.0,
                    144.0,
                    26.0,
                    if is_capturing {
                        "Press a key..."
                    } else {
                        binding
                    },
                    is_capturing,
                );
            }
        }
        2 => {
            rect(out, [content_left, 176.0, content_width, 182.0], panel);
            text(out, content_left + 24.0, 198.0, "Voxel Editor MCP", 1.18);
            text(
                out,
                content_left + 24.0,
                230.0,
                "Local tools for connected MCP clients",
                0.9,
            );
            text(
                out,
                content_left + 24.0,
                259.0,
                "The companion process reads this setting on startup.",
                0.82,
            );
            settings_button(
                out,
                content_right - 192.0,
                210.0,
                164.0,
                42.0,
                if ui.settings_mcp_enabled {
                    "Enabled"
                } else {
                    "Disabled"
                },
                ui.settings_mcp_enabled,
            );
        }
        _ => {
            rect(out, [content_left, 176.0, content_width, 402.0], panel);
            let sliders = [
                (
                    "Simulation speed",
                    format!("{:.2}×", ui.simulation_time_scale),
                    "1× is real time; maximum is 1 year per second.",
                ),
                (
                    "Walking speed",
                    format!("{:.1} m/s", ui.walk_speed_mps),
                    "Player movement speed in the world.",
                ),
                (
                    "Look sensitivity",
                    format!("{:.3}", ui.look_sensitivity),
                    "Mouse sensitivity while looking around.",
                ),
            ];
            for (index, (label, value, hint)) in sliders.into_iter().enumerate() {
                let top = 204.0 + index as f32 * 106.0;
                let [slider_x, slider_y, slider_width, _] = settings_slider_rect(width, index);
                text(out, content_left + 28.0, top, label, 1.0);
                text(out, content_right - 150.0, top, &value, 0.96);
                let fraction = match index {
                    0 => (ui.simulation_time_scale.ln() / MAX_SIMULATION_TIME_SCALE.ln())
                        .clamp(0.0, 1.0),
                    1 => ((ui.walk_speed_mps - 1.0) / 9.0).clamp(0.0, 1.0),
                    _ => ((ui.look_sensitivity - 0.001) / 0.011).clamp(0.0, 1.0),
                };
                settings_slider(out, slider_x, slider_y, slider_width, fraction);
                text(out, content_left + 28.0, slider_y + 17.0, hint, 0.78);
            }
        }
    }

    rect(
        out,
        [content_left, height - 44.0, content_width, 1.0],
        [0.26, 0.31, 0.33, 1.0],
    );
    text(
        out,
        content_left,
        height - 32.0,
        "Changes apply immediately",
        0.78,
    );
}

fn settings_nav_item(
    out: &mut Vec<Rect>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: &str,
    selected: bool,
) {
    let fill = if selected {
        [0.075, 0.15, 0.15, 1.0]
    } else {
        [0.035, 0.045, 0.055, 1.0]
    };
    rect(out, [x, y, w, h], fill);
    if selected {
        rect(out, [x, y, 3.0, h], [0.18, 0.48, 0.42, 1.0]);
    }
    text(out, x + 16.0, y + 15.0, label, 0.98);
}

fn settings_button(
    out: &mut Vec<Rect>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: &str,
    primary: bool,
) {
    let fill = if primary {
        [0.14, 0.4, 0.34, 1.0]
    } else {
        [0.075, 0.105, 0.12, 1.0]
    };
    rect(out, [x, y, w, h], fill);
    text(out, x + 14.0, y + h * 0.5 + 4.0, label, 0.86);
}

fn settings_slider(out: &mut Vec<Rect>, x: f32, y: f32, width: f32, fraction: f32) {
    let t = fraction.clamp(0.0, 1.0);
    rect(out, [x, y, width, 8.0], [0.06, 0.085, 0.09, 1.0]);
    rect(out, [x, y, width * t, 8.0], [0.18, 0.55, 0.47, 1.0]);
    rect(
        out,
        [x + width * t - 4.0, y - 5.0, 8.0, 18.0],
        [0.5, 0.9, 0.75, 1.0],
    );
}
fn rect(out: &mut Vec<Rect>, rect: [f32; 4], color: [f32; 4]) {
    out.push(Rect { rect, color });
}
fn text(out: &mut Vec<Rect>, x: f32, y: f32, s: &str, scale: f32) {
    voxel_physics_lab::ui_text::draw_text(out, x, y, s, scale);
}

#[cfg(test)]
mod scene_inspector_tests {
    use super::*;

    #[test]
    fn physics_cost_label_distinguishes_unmeasured_from_zero_and_shows_budget() {
        assert_eq!(physics_cost_label(None), "10 ms physics: — / 5 ms");
        assert_eq!(physics_cost_label(Some(0.0)), "10 ms physics: 0.00 ms / 5 ms");
        assert_eq!(physics_cost_label(Some(2.5)), "10 ms physics: 2.50 ms / 5 ms");
        assert_eq!(physics_cost_label(Some(f64::NAN)), "10 ms physics: — / 5 ms");
    }

    #[test]
    fn observed_simulation_rate_is_distinct_from_requested_speed() {
        let mut ui = Ui::new(1.0, Vec3::ZERO, Vec3::Z);
        assert_eq!(ui.measured_simulation_rate, None);
        ui.simulation_time_scale = 1.0;
        ui.set_measured_simulation_rate(Some(0.08));
        assert_eq!(simulation_rate_label(ui.measured_simulation_rate, ui.simulation_time_scale),
                   "Real 0.08× / target 1.00×");
        assert_eq!(physics_toolbar_label(Some(95.0), Some(0.08), 1.0, true),
                   "95.0 ms · Real 0.08×");
        assert_eq!(simulation_rate_label(None, 1.0), "Real — / target 1.00×");
        ui.set_measured_simulation_rate(Some(0.0));
        assert_eq!(ui.measured_simulation_rate, Some(0.0));
        for invalid in [f64::NAN, f64::INFINITY, -0.01] {
            ui.set_measured_simulation_rate(Some(invalid));
            assert_eq!(ui.measured_simulation_rate, None);
        }
    }

    #[test]
    fn physical_state_menu_rows_need_an_open_scene() {
        let mut ui = Ui::new(1.0, Vec3::ZERO, Vec3::Z);
        ui.world_active = true;
        ui.editor_mode = true;
        ui.file_menu_open = true;
        assert!(ui.click(130.0, 392.0).is_none());
        ui.file_menu_open = true;
        assert!(ui.click(130.0, 421.0).is_none());

        ui.set_scene_document_available(true);
        ui.file_menu_open = true;
        assert!(matches!(ui.click(130.0, 392.0), Some(UiAction::SavePhysicalState)));
        ui.file_menu_open = true;
        assert!(matches!(ui.click(130.0, 421.0), Some(UiAction::LoadPhysicalState)));
        // The row above is still the 74-cube impact scene, not a state action.
        ui.file_menu_open = true;
        assert!(matches!(
            ui.click(130.0, 363.0),
            Some(UiAction::CreateImpactScene { side: 74 })
        ));
    }

    fn scene_ui(width: f32, height: f32) -> Ui {
        let mut ui = Ui::new(1.0, Vec3::ZERO, Vec3::Z);
        ui.world_active = true;
        ui.editor_mode = true;
        ui.set_active_scene_document(Some("Scene".to_owned()), vec!["Impact object".to_owned()]);
        ui.selected_scene_model = Some(0);
        ui.set_selected_scene_model_info(Some(SceneModelInspectorInfo {
            name: "Impact object".to_owned(),
            translation_m: [0.01, 0.02, 0.03],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [1.0; 3],
            velocity_m_s: [0.0; 3],
            force_n: [0.0; 3],
            material_name: "Steel".to_owned(),
        }));
        ui.draw(0, 0, 60.0, 0.0, None, None, width, height, None);
        ui
    }

    fn click_canvas(ui: &mut Ui, x: f32, y: f32) -> Option<UiAction> {
        ui.click(x * ui.canvas_scale, y * ui.canvas_scale)
    }

    fn click_scene_row(ui: &mut Ui, item: SceneInspectorItem, x_offset: f32) -> Option<UiAction> {
        let row = scene_inspector_rows(ui)
            .into_iter()
            .find(|row| row.item == item)
            .unwrap();
        let left = ui.canvas_width - 250.0;
        click_canvas(
            ui,
            left + x_offset,
            row.top - ui.scene_inspector_scroll + row.height * 0.5,
        )
    }

    fn has_text_pixel(ui: &Ui, rects: &[Rect], logical_area: [f32; 4]) -> bool {
        let area = [
            logical_area[0] * ui.canvas_scale,
            logical_area[1] * ui.canvas_scale,
            logical_area[2] * ui.canvas_scale,
            logical_area[3] * ui.canvas_scale,
        ];
        rects.iter().any(|item| {
            item.color[0] > 0.8
                && item.color[1] > 0.8
                && item.color[2] > 0.8
                && item.rect[0] < area[0] + area[2]
                && item.rect[0] + item.rect[2] > area[0]
                && item.rect[1] < area[1] + area[3]
                && item.rect[1] + item.rect[3] > area[1]
        })
    }

    fn enter_vector(ui: &mut Ui, item: SceneInspectorItem, values: [&str; 3]) -> Option<UiAction> {
        click_scene_row(ui, item, 30.0);
        assert!(ui.scene_input_active());
        for (axis, value) in values.into_iter().enumerate() {
            for character in value.chars() {
                ui.scene_input_char(character);
            }
            if axis < 2 {
                ui.scene_input_key(KeyCode::Tab);
            }
        }
        ui.scene_input_key(KeyCode::Enter)
    }

    #[test]
    fn scene_input_enter_emits_typed_transform_and_dynamics_actions() {
        let mut ui = scene_ui(1280.0, 720.0);
        assert!(
            matches!(enter_vector(&mut ui, SceneInspectorItem::Position, ["1.25", "-2.5", "3.75"]),
            Some(UiAction::SetScenePosition { value }) if value == [1.25, -2.5, 3.75])
        );
        assert!(
            matches!(enter_vector(&mut ui, SceneInspectorItem::Rotation, ["90", "-180", "270"]),
            Some(UiAction::SetSceneRotation { degrees }) if degrees == [90.0, -180.0, 270.0])
        );
        assert!(
            matches!(enter_vector(&mut ui, SceneInspectorItem::Velocity, ["0.5", "-1.25", "2"]),
            Some(UiAction::SetSceneVelocity { value }) if value == [0.5, -1.25, 2.0])
        );
        assert!(
            matches!(enter_vector(&mut ui, SceneInspectorItem::Force, ["10", "-20", "30"]),
            Some(UiAction::SetSceneForce { value }) if value == [10.0, -20.0, 30.0])
        );
        assert!(!ui.scene_input_active());
    }

    #[test]
    fn scene_input_rejects_invalid_numbers_and_non_lattice_rotation() {
        let mut ui = scene_ui(1280.0, 720.0);
        click_scene_row(&mut ui, SceneInspectorItem::Position, 100.0);
        ui.scene_input_char('-');
        assert!(ui.scene_input_key(KeyCode::Enter).is_none());
        assert!(ui.scene_input_active());
        assert!(ui.scene_input_key(KeyCode::Escape).is_none());

        click_scene_row(&mut ui, SceneInspectorItem::Rotation, 100.0);
        for (axis, value) in ["45", "0", "0"].into_iter().enumerate() {
            for character in value.chars() {
                ui.scene_input_char(character);
            }
            if axis < 2 {
                ui.scene_input_key(KeyCode::Tab);
            }
        }
        assert!(ui.scene_input_key(KeyCode::Enter).is_none());
        assert!(ui.scene_input_active());
    }

    #[test]
    fn inline_axis_clicks_preserve_pending_values_and_escape_cancels() {
        let mut ui = scene_ui(1280.0, 720.0);
        click_scene_row(&mut ui, SceneInspectorItem::Position, 30.0);
        for c in "1.25".chars() {
            ui.scene_input_char(c);
        }
        assert!(click_scene_row(&mut ui, SceneInspectorItem::Position, 110.0).is_none());
        for c in "-2.5".chars() {
            ui.scene_input_char(c);
        }
        assert!(click_scene_row(&mut ui, SceneInspectorItem::Position, 185.0).is_none());
        for c in "3.75".chars() {
            ui.scene_input_char(c);
        }
        assert!(matches!(ui.scene_input_key(KeyCode::Enter),
            Some(UiAction::SetScenePosition { value }) if value == [1.25, -2.5, 3.75]));
        click_scene_row(&mut ui, SceneInspectorItem::Velocity, 30.0);
        ui.scene_input_char('9');
        assert!(ui.scene_input_key(KeyCode::Escape).is_none());
        assert!(!ui.scene_input_active());
        assert_eq!(
            ui.selected_scene_model_info.as_ref().unwrap().velocity_m_s,
            [0.0; 3]
        );
    }

    #[test]
    fn foldouts_reflow_rows_and_empty_scene_omits_object_properties() {
        let mut ui = scene_ui(1920.0, 1080.0);
        assert!(scene_inspector_content_bottom(&ui) < ui.canvas_height - 8.0);
        assert!(!scene_inspector_rows(&ui)
            .iter()
            .any(|r| matches!(r.item, SceneInspectorItem::PhysicsField(_))));
        click_scene_row(
            &mut ui,
            SceneInspectorItem::Section(SceneInspectorSection::Transform),
            30.0,
        );
        assert!(!scene_inspector_rows(&ui)
            .iter()
            .any(|r| r.item == SceneInspectorItem::Position));
        assert!(matches!(
            click_scene_row(&mut ui, SceneInspectorItem::MaterialCycle, 25.0),
            Some(UiAction::CycleSceneMaterial)
        ));
        ui.selected_scene_model = None;
        let rows = scene_inspector_rows(&ui);
        assert_eq!(
            rows.iter()
                .filter(|r| r.item == SceneInspectorItem::Hint)
                .count(),
            1
        );
        assert!(!rows
            .iter()
            .any(|r| r.item == SceneInspectorItem::Section(SceneInspectorSection::Transform)));
    }

    #[test]
    fn physics_field_clicks_preserve_buffer_and_accept_scientific_notation() {
        let mut ui = scene_ui(1280.0, 720.0);
        ui.scene_inspector_collapsed[4] = false;
        ui.scene_inspector_scroll = 400.0;
        click_scene_row(&mut ui, SceneInspectorItem::PhysicsField(0), 170.0);
        for c in "0.002".chars() {
            ui.scene_input_char(c);
        }
        assert!(click_scene_row(&mut ui, SceneInspectorItem::PhysicsField(8), 170.0).is_none());
        for c in "1e-5".chars() {
            ui.scene_input_char(c);
        }
        assert!(
            matches!(ui.scene_input_key(KeyCode::Enter), Some(UiAction::SetScenePhysics {
            fixed_dt_seconds, solver_tolerance, ..
        }) if fixed_dt_seconds == 0.002 && solver_tolerance == 1e-5)
        );
    }

    #[test]
    fn long_numeric_values_are_clipped_to_the_field() {
        let mut rects = Vec::new();
        let area = [12.0, 20.0, 50.0, 17.0];
        for editing in [false, true] {
            rects.clear();
            clipped_inspector_text(
                &mut rects,
                area,
                "-12345678901234567890.12345|",
                0.62,
                editing,
            );
            assert!(!rects.is_empty());
            assert!(rects.iter().all(|r| r.rect[0] >= area[0]
                && r.rect[1] >= area[1]
                && r.rect[0] + r.rect[2] <= area[0] + area[2] + 0.001
                && r.rect[1] + r.rect[3] <= area[1] + area[3] + 0.001));
        }
    }

    #[test]
    fn coordinate_grid_stays_inside_scene_viewport_at_common_resolutions() {
        for (width, height) in [
            (1920.0, 1080.0),
            (1280.0, 720.0),
            (640.0, 480.0),
            (1280.0, 400.0),
            (800.0, 1100.0),
        ] {
            let ui = scene_ui(width, height);
            let [x, y, w, h] = coordinate_grid_bounds(&ui);
            assert!(x >= 235.0 && x + w <= ui.canvas_width - 250.0);
            assert!(y >= 98.0 && y + h <= ui.editor_dock_top());
            let mut rects = Vec::new();
            draw_coordinate_grid(&mut rects, &ui);
            assert!(rects.iter().all(|r| r.rect[0] >= x
                && r.rect[1] >= y
                && r.rect[0] + r.rect[2] <= x + w
                && r.rect[1] + r.rect[3] <= y + h));
            assert!(rects.iter().any(|r| r.color == [0.92, 0.29, 0.26, 1.0]));
            assert!(rects.iter().any(|r| r.color == [0.36, 0.80, 0.35, 1.0]));
            assert!(rects.iter().any(|r| r.color == [0.34, 0.62, 1.0, 1.0]));
        }
    }

    #[test]
    fn coordinate_grid_projection_tracks_camera_rotation_and_keeps_z_up() {
        let mut ui = scene_ui(1280.0, 720.0);
        ui.set_view_basis(Vec3::X, Vec3::Z);
        assert_eq!(
            coordinate_projection(Vec3::X, ui.view_right, ui.view_up),
            [1.0, 0.0]
        );
        assert_eq!(
            coordinate_projection(Vec3::Z, ui.view_right, ui.view_up),
            [0.0, -1.0]
        );
        ui.set_view_basis(Vec3::Y, Vec3::Z);
        assert_eq!(
            coordinate_projection(Vec3::X, ui.view_right, ui.view_up),
            [0.0, 0.0]
        );
        assert_eq!(
            coordinate_projection(Vec3::Y, ui.view_right, ui.view_up),
            [1.0, 0.0]
        );
        ui.set_view_basis(Vec3::NAN, Vec3::ZERO);
        assert_eq!(ui.view_right, Vec3::Y);
        assert_eq!(ui.view_up, Vec3::Z);
    }

    #[test]
    fn scene_selection_mode_buttons_emit_object_and_voxel_actions() {
        let mut ui = scene_ui(1280.0, 720.0);
        assert!(matches!(
            click_canvas(&mut ui, 375.0, 12.0),
            Some(UiAction::SetSceneSelectionMode(SceneSelectionMode::Object))
        ));
        assert!(matches!(
            click_canvas(&mut ui, 443.0, 12.0),
            Some(UiAction::SetSceneSelectionMode(SceneSelectionMode::Voxel))
        ));
        assert_eq!(ui.scene_selection_mode, SceneSelectionMode::Voxel);
    }

    #[test]
    fn scene_inspector_wheel_and_thumb_scroll_keep_hit_testing_aligned() {
        let mut ui = scene_ui(640.0, 480.0);
        let scale = ui.canvas_scale;
        let inspector_left = ui.canvas_width - 250.0;
        ui.scene_inspector_collapsed = [false; 5];
        let start = ui.scene_inspector_scroll;
        assert!(ui.scroll_scene_hierarchy(-1.0, (inspector_left + 120.0) * scale, 220.0 * scale));
        assert!(ui.scene_inspector_scroll > start);
        assert!(matches!(
            click_scene_row(&mut ui, SceneInspectorItem::MaterialCycle, 25.0),
            Some(UiAction::CycleSceneMaterial)
        ));

        let [track_top, track_height, thumb_top, thumb_height, max_scroll] =
            inspector_scroll_geometry(&ui);
        assert!(max_scroll > 0.0);
        assert!(thumb_height < track_height);
        assert!(ui
            .draw(0, 0, 60.0, 0.0, None, None, 640.0, 480.0, None)
            .iter()
            .any(|item| item.color == [0.30, 0.58, 0.54, 1.0]
                && item.rect[0] > (ui.canvas_width - 12.0) * scale));
        let thumb_center = thumb_top + thumb_height * 0.5;
        assert!(ui
            .click((ui.canvas_width - 8.0) * scale, thumb_center * scale)
            .is_none());
        assert!(ui.timeline_is_dragging());
        ui.pointer(
            (ui.canvas_width - 8.0) * scale,
            (track_top + track_height) * scale,
        );
        assert!((ui.scene_inspector_scroll - max_scroll).abs() < 0.1);
        ui.end_timeline_drag();
        assert!(!ui.timeline_is_dragging());
    }

    #[test]
    fn scene_physics_dialog_emits_validated_runtime_settings() {
        let mut ui = scene_ui(1280.0, 720.0);
        click_scene_row(
            &mut ui,
            SceneInspectorItem::Section(SceneInspectorSection::Simulation),
            30.0,
        );
        let row = scene_inspector_rows(&ui)
            .into_iter()
            .find(|row| row.item == SceneInspectorItem::PhysicsField(0))
            .unwrap();
        ui.scene_inspector_scroll = row.top - 160.0;
        click_scene_row(&mut ui, SceneInspectorItem::PhysicsField(0), 170.0);
        assert!(ui.scene_input_active());
        let input = ui.scene_physics_input.as_mut().unwrap();
        input.values = [
            "0.002".into(),
            "3".into(),
            "0".into(),
            "1".into(),
            "-9.81".into(),
            "0.5".into(),
            "1.25".into(),
            "64".into(),
            "0.00001".into(),
        ];
        assert!(
            matches!(ui.scene_input_key(KeyCode::Enter), Some(UiAction::SetScenePhysics {
            fixed_dt_seconds, substeps, gravity, restitution, friction_scale,
            solver_iterations, solver_tolerance,
        }) if fixed_dt_seconds == 0.002 && substeps == 3 && gravity == [0.0, 1.0, -9.81]
            && restitution == 0.5 && friction_scale == 1.25 && solver_iterations == 64
            && solver_tolerance == 0.00001)
        );
        assert!(!ui.scene_input_active());
        assert!(scene_physics_action(&[
            "0".into(),
            "3".into(),
            "0".into(),
            "0".into(),
            "-9.81".into(),
            "0.5".into(),
            "1.0".into(),
            "64".into(),
            "0.00001".into(),
        ])
        .is_err());
        assert!(scene_physics_action(&[
            "0.001".into(),
            "3".into(),
            "0".into(),
            "0".into(),
            "-9.81".into(),
            "1.5".into(),
            "1.0".into(),
            "64".into(),
            "0.00001".into(),
        ])
        .is_err());
    }

    #[test]
    fn scene_inspector_material_and_coordinate_controls_fit_compact_windows() {
        let sizes = [
            (1920.0, 1080.0),
            (2560.0, 1440.0),
            (1280.0, 720.0),
            (960.0, 600.0),
            (800.0, 600.0),
            (640.0, 480.0),
            (800.0, 1100.0),
            (1280.0, 400.0),
        ];
        for (width, height) in sizes {
            let mut ui = scene_ui(width, height);
            let rects = ui.draw(0, 0, 60.0, 0.0, None, None, width, height, None);
            assert!(
                rects.iter().all(|item| {
                    item.rect[0] >= 0.0
                        && item.rect[1] >= 0.0
                        && item.rect[0] + item.rect[2] <= width + 1.0
                        && item.rect[1] + item.rect[3] <= height + 1.0
                }),
                "out-of-canvas inspector rect at {width}×{height}: {:?}",
                rects
                    .iter()
                    .find(|item| {
                        item.rect[0] < 0.0
                            || item.rect[1] < 0.0
                            || item.rect[0] + item.rect[2] > width + 1.0
                            || item.rect[1] + item.rect[3] > height + 1.0
                    })
                    .map(|item| item.rect)
            );
            let [vx, vy, vw, vh] = ui.scene_viewport().expect("scene viewport available");
            assert!(vw > 0 && vh > 0, "empty viewport at {width}×{height}");
            assert!(
                vx + vw <= width as u32 && vy + vh <= height as u32,
                "viewport outside {width}×{height}: {vx},{vy} {vw}×{vh}"
            );
            assert!(matches!(
                click_scene_row(&mut ui, SceneInspectorItem::MaterialCycle, 25.0),
                Some(UiAction::CycleSceneMaterial)
            ));
            assert!(matches!(
                click_scene_row(&mut ui, SceneInspectorItem::MaterialActions, 35.0),
                Some(UiAction::LoadSceneMaterial)
            ));
            assert!(matches!(
                click_scene_row(&mut ui, SceneInspectorItem::MaterialActions, 105.0),
                Some(UiAction::SaveSceneMaterial)
            ));
            assert!(matches!(
                click_scene_row(&mut ui, SceneInspectorItem::MaterialActions, 180.0),
                Some(UiAction::CreateSceneMaterial)
            ));
            assert!(matches!(
                enter_vector(&mut ui, SceneInspectorItem::Position, ["1", "2", "3"]),
                Some(UiAction::SetScenePosition { .. })
            ));
            assert!(matches!(
                click_canvas(&mut ui, 375.0, 12.0),
                Some(UiAction::SetSceneSelectionMode(SceneSelectionMode::Object))
            ));
            assert!(matches!(
                click_canvas(&mut ui, 443.0, 12.0),
                Some(UiAction::SetSceneSelectionMode(SceneSelectionMode::Voxel))
            ));
        }
    }

    #[test]
    fn model_editor_controls_fit_short_and_minimum_windows() {
        for (width, height) in [(1280.0, 400.0), (640.0, 480.0)] {
            let mut ui = Ui::new(1.0, Vec3::ZERO, Vec3::Z);
            ui.world_active = true;
            ui.editor_mode = true;
            ui.set_active_model_document(Some("Model".to_owned()), 64, 1);
            ui.set_rig_authoring_state(vec!["root".to_owned()], Some(0), 0);
            let rects = ui.draw(0, 0, 60.0, 0.0, None, None, width, height, None);
            assert!(
                rects.iter().all(|item| {
                    item.rect[0] >= 0.0
                        && item.rect[1] >= 0.0
                        && item.rect[0] + item.rect[2] <= width + 1.0
                        && item.rect[1] + item.rect[3] <= height + 1.0
                }),
                "model editor clipped at {width}×{height}"
            );
            let [vx, vy, vw, vh] = ui.scene_viewport().expect("model viewport available");
            assert!(vw > 0 && vh > 0);
            assert!(vx + vw <= width as u32 && vy + vh <= height as u32);
        }
    }

    #[test]
    fn scene_inspector_rows_are_not_painted_over_or_obscured_by_legacy_hints() {
        for (width, height) in [(1920.0, 1080.0), (640.0, 480.0)] {
            let mut ui = scene_ui(width, height);
            let rects = ui.draw(0, 0, 60.0, 0.0, None, None, width, height, None);
            let left = ui.canvas_width - 250.0;
            assert!(
                !rects.iter().any(|item| {
                    item.color == [0.085, 0.11, 0.12, 1.0]
                        && item.rect[0] < (left + 240.0) * ui.canvas_scale
                        && item.rect[0] + item.rect[2] > (left + 10.0) * ui.canvas_scale
                        && item.rect[1] < 275.0 * ui.canvas_scale
                        && item.rect[1] + item.rect[3] > 185.0 * ui.canvas_scale
                }),
                "opaque row fill overlays inspector values at {width}×{height}"
            );
            let rows = scene_inspector_rows(&ui);
            for row in rows.iter().filter(|r| {
                matches!(
                    r.item,
                    SceneInspectorItem::Position
                        | SceneInspectorItem::Rotation
                        | SceneInspectorItem::Velocity
                        | SceneInspectorItem::Force
                )
            }) {
                assert!(
                    has_text_pixel(&ui, &rects, [left + 15.0, row.top, 190.0, row.height]),
                    "inspector row {:?} missing at {width}x{height}",
                    row.item
                );
            }
            assert!(rows.windows(2).all(|w| w[0].top + w[0].height <= w[1].top));
        }
    }

    #[test]
    fn scene_inspector_draws_exactly_one_layout_without_legacy_controls() {
        let mut ui = scene_ui(1280.0, 720.0);
        let full = ui.draw(0, 0, 60.0, 0.0, None, None, 1280.0, 720.0, None);
        let left = ui.canvas_width - 250.0;
        let mut expected = Vec::new();
        draw_scene_inspector_contents(
            &mut expected,
            &ui,
            left,
            [0.12, 0.14, 0.16, 1.0],
            [0.16, 0.31, 0.38, 1.0],
        );
        clip_and_scroll_scene_inspector(&mut expected, &ui);
        let scale = ui.canvas_scale;
        for r in &mut expected {
            let x = (r.rect[0] * scale).round();
            let y = (r.rect[1] * scale).round();
            let right = ((r.rect[0] + r.rect[2]) * scale).round();
            let bottom = ((r.rect[1] + r.rect[3]) * scale).round();
            r.rect = [x, y, (right - x).max(1.0), (bottom - y).max(1.0)];
        }
        let body = |r: &&Rect| {
            r.rect[0] >= left * scale
                && r.rect[1] >= 144.0 * scale
                && r.rect[2] > 0.0
                && r.rect[3] > 0.0
        };
        let actual: Vec<_> = full
            .iter()
            .filter(body)
            .map(|r| (r.rect, r.color))
            .collect();
        let expected: Vec<_> = expected
            .iter()
            .filter(body)
            .map(|r| (r.rect, r.color))
            .collect();
        assert!(actual == expected,
            "another panel is painted over the inspector: actual {} quads, expected {}; first mismatch {:?}",
            actual.len(), expected.len(), actual.iter().zip(&expected).position(|(a, b)| a != b));
    }
}
