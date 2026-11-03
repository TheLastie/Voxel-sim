use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::PathBuf};

pub const SETTINGS_FILE: &str = ".voxel_editor_settings.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditorSettings {
    #[serde(default = "settings_version")]
    pub version: u32,
    #[serde(default = "default_mcp_enabled")]
    pub mcp_enabled: bool,
    #[serde(default = "default_bindings")]
    pub bindings: BTreeMap<String, String>,
}

fn settings_version() -> u32 {
    1
}
fn default_mcp_enabled() -> bool {
    true
}

pub fn default_bindings() -> BTreeMap<String, String> {
    [
        ("move_x_neg", "Numpad4"),
        ("move_x_pos", "Numpad6"),
        ("move_y_pos", "Numpad8"),
        ("move_y_neg", "Numpad2"),
        ("move_z_pos", "Numpad9"),
        ("move_z_neg", "Numpad3"),
        ("rotate_z_pos", "E"),
        ("rotate_z_neg", "Q"),
        ("duplicate", "Ctrl+D"),
        ("delete", "Delete"),
        ("undo", "Ctrl+Z"),
        ("redo", "Ctrl+Y"),
    ]
    .into_iter()
    .map(|(action, key)| (action.to_owned(), key.to_owned()))
    .collect()
}

impl Default for EditorSettings {
    fn default() -> Self {
        Self {
            version: 1,
            mcp_enabled: true,
            bindings: default_bindings(),
        }
    }
}

pub fn settings_path() -> PathBuf {
    let mut dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    // Debug/release binaries live under target/<profile>; keep the file at the
    // project root so the editor and its MCP companion share one stable path.
    if dir
        .file_name()
        .is_some_and(|n| n == "debug" || n == "release")
        && dir
            .parent()
            .is_some_and(|p| p.file_name().is_some_and(|n| n == "target"))
    {
        dir = dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(&dir)
            .to_path_buf();
    }
    dir.join(SETTINGS_FILE)
}

pub fn load() -> EditorSettings {
    let path = settings_path();
    let Ok(bytes) = fs::read(path) else {
        return EditorSettings::default();
    };
    let Ok(mut settings) = serde_json::from_slice::<EditorSettings>(&bytes) else {
        return EditorSettings::default();
    };
    let defaults = default_bindings();
    for (key, value) in defaults {
        settings.bindings.entry(key).or_insert(value);
    }
    settings.version = 1;
    settings
}

pub fn save(settings: &EditorSettings) -> Result<(), String> {
    let path = settings_path();
    let bytes = serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(path, bytes).map_err(|e| e.to_string())
}

pub fn binding_matches(binding: &str, key_debug: &str, control_down: bool) -> bool {
    let key = match key_debug.strip_prefix("Key") {
        Some(letter) => letter.to_owned(),
        None => key_debug.to_owned(),
    };
    let actual = if control_down {
        format!("Ctrl+{key}")
    } else {
        key
    };
    binding.eq_ignore_ascii_case(&actual)
}

pub fn set_binding(settings: &mut EditorSettings, action: &str, key: String) -> Result<(), String> {
    if settings
        .bindings
        .iter()
        .any(|(other, value)| other != action && value.eq_ignore_ascii_case(&key))
    {
        return Err(format!("{key} is already assigned to another action."));
    }
    settings.bindings.insert(action.to_owned(), key);
    Ok(())
}

pub fn is_pressed(
    settings: &EditorSettings,
    action: &str,
    key_debug: &str,
    control_down: bool,
) -> bool {
    settings
        .bindings
        .get(action)
        .is_some_and(|binding| binding_matches(binding, key_debug, control_down))
}

pub fn label(action: &str) -> &'static str {
    match action {
        "move_x_neg" => "Move selected part X-",
        "move_x_pos" => "Move selected part X+",
        "move_y_pos" => "Move selected part Y+",
        "move_y_neg" => "Move selected part Y-",
        "move_z_pos" => "Move selected part Z+",
        "move_z_neg" => "Move selected part Z-",
        "rotate_z_pos" => "Rotate selected part Z+",
        "rotate_z_neg" => "Rotate selected part Z-",
        "duplicate" => "Duplicate selected part",
        "delete" => "Delete selected part",
        "undo" => "Undo voxel edit",
        "redo" => "Redo voxel edit",
        _ => "Unknown action",
    }
}

pub const REBINDABLE_ACTIONS: [&str; 12] = [
    "move_x_neg",
    "move_x_pos",
    "move_y_pos",
    "move_y_neg",
    "move_z_pos",
    "move_z_neg",
    "rotate_z_pos",
    "rotate_z_neg",
    "duplicate",
    "delete",
    "undo",
    "redo",
];
