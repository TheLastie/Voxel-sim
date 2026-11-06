//! Authored chemistry settings of a scene: reactions on or off, the time
//! between reaction updates and an optional recipe file.
//!
//! This is the one API behind the editor window and the MCP server. The
//! settings live beside the scene as `<scene>.chemistry.json`, so the binary
//! `.vxscene` schema does not change and a scene without the file behaves
//! exactly as before (reactions off). [`ChemistrySettings::apply`] hands them
//! to the engine through [`Scene::set_chemistry_registry`],
//! [`Scene::set_chemistry_interval`] and [`Scene::chemistry_enabled`]; it adds
//! no reaction rule of its own.
//!
//! A recipe file (see `recipe_data.rs`, `recipes/wood_rot.json`) replaces the
//! built-in recipes for the scene; it does not extend them.

use crate::{material::Material, recipes::RecipeRegistry, scene::Scene};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

/// Schema version of the settings file.
pub const CHEMISTRY_FILE_VERSION: u32 = 1;
/// Longest interval accepted (one simulated year), seconds.
pub const MAX_CHEMISTRY_INTERVAL_S: f64 = 31_557_600.0;
const MAX_SETTINGS_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChemistrySettings {
    pub version: u32,
    /// Run the fixed-step reaction hook (`Scene::chemistry_enabled`).
    pub enabled: bool,
    /// Simulated seconds between reaction updates; `0` reacts at every step.
    pub interval_s: f64,
    /// Recipe file, relative to the scene's directory when possible. `None`
    /// uses the built-in recipes.
    #[serde(default)]
    pub recipe_file: Option<String>,
}

impl Default for ChemistrySettings {
    fn default() -> Self {
        Self {
            version: CHEMISTRY_FILE_VERSION,
            enabled: false,
            interval_s: 0.0,
            recipe_file: None,
        }
    }
}

impl ChemistrySettings {
    /// `scenes/a.vxscene` -> `scenes/a.chemistry.json`.
    pub fn sidecar_path(scene_path: &Path) -> PathBuf {
        scene_path.with_extension("chemistry.json")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != CHEMISTRY_FILE_VERSION {
            return Err(format!(
                "unsupported chemistry settings version {} (expected {CHEMISTRY_FILE_VERSION})",
                self.version
            ));
        }
        check_interval(self.interval_s)?;
        if self.recipe_file.as_deref().is_some_and(|path| path.trim().is_empty()) {
            return Err("recipe file path is empty".into());
        }
        Ok(())
    }

    /// Settings saved beside `scene_path`; the defaults when there is no file.
    /// A damaged or unsupported file is an error, never silently replaced.
    pub fn load(scene_path: &Path) -> Result<Self, String> {
        let path = Self::sidecar_path(scene_path);
        match fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
            Ok(meta) if meta.len() > MAX_SETTINGS_BYTES => {
                return Err(format!("{} is larger than {MAX_SETTINGS_BYTES} bytes", path.display()));
            }
            Ok(_) => {}
        }
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let settings: Self = serde_json::from_str(&text)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        settings.validate().map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(settings)
    }

    /// Write the settings beside `scene_path` (temporary file, then rename).
    pub fn save(&self, scene_path: &Path) -> Result<PathBuf, String> {
        self.validate()?;
        let path = Self::sidecar_path(scene_path);
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| format!("cannot encode chemistry settings: {error}"))?;
        let temporary = path.with_extension("json.tmp");
        let write = || -> std::io::Result<()> {
            let mut file = fs::File::create(&temporary)?;
            file.write_all(text.as_bytes())?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &path)
        };
        write().map_err(|error| {
            let _ = fs::remove_file(&temporary);
            format!("cannot write {}: {error}", path.display())
        })?;
        Ok(path)
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_interval(&mut self, interval_s: f64) -> Result<(), String> {
        check_interval(interval_s)?;
        self.interval_s = interval_s;
        Ok(())
    }

    /// Use the built-in recipes again.
    pub fn clear_recipe(&mut self) {
        self.recipe_file = None;
    }

    /// Choose `recipe` after proving it parses and resolves against
    /// `materials`. Nothing changes when it does not. The stored path is
    /// relative to the scene's directory when the two share a root.
    pub fn set_recipe_file(
        &mut self,
        scene_path: &Path,
        recipe: &Path,
        materials: &[Material],
    ) -> Result<(), String> {
        RecipeRegistry::from_file(materials, recipe)?;
        self.recipe_file = Some(stored_recipe_path(scene_path, recipe));
        Ok(())
    }

    /// As [`set_recipe_file`](Self::set_recipe_file) for a scene that has no
    /// path yet: the absolute `recipe` path is stored as given.
    pub fn set_recipe_file_absolute(
        &mut self,
        recipe: &Path,
        materials: &[Material],
    ) -> Result<(), String> {
        if !recipe.is_absolute() {
            return Err("an unsaved scene needs an absolute recipe path".into());
        }
        RecipeRegistry::from_file(materials, recipe)?;
        self.recipe_file = Some(recipe.to_string_lossy().into_owned());
        Ok(())
    }

    /// Absolute location of the recipe file, if one is chosen.
    pub fn resolve_recipe(&self, scene_path: Option<&Path>) -> Option<PathBuf> {
        let recipe = Path::new(self.recipe_file.as_deref()?);
        if recipe.is_absolute() {
            return Some(recipe.to_path_buf());
        }
        let base = scene_path.and_then(Path::parent).unwrap_or_else(|| Path::new("."));
        Some(base.join(recipe))
    }

    /// The same settings for a scene saved under a new path: a relative recipe
    /// path keeps pointing at the same file. `old_scene_path` is where the
    /// recipe path was relative to (`None` for an unsaved scene, whose recipe
    /// path is absolute).
    pub fn rebased(&self, old_scene_path: Option<&Path>, new_scene_path: &Path) -> Self {
        let mut moved = self.clone();
        if let Some(recipe) = self.resolve_recipe(old_scene_path) {
            moved.recipe_file = Some(stored_recipe_path(new_scene_path, &recipe));
        }
        moved
    }

    /// Check that the recipe file (if any) still parses and resolves against
    /// `materials`. Returns the number of recipes it defines.
    pub fn check_recipe(
        &self,
        scene_path: Option<&Path>,
        materials: &[Material],
    ) -> Result<Option<usize>, String> {
        match self.resolve_recipe(scene_path) {
            None => Ok(None),
            Some(path) => RecipeRegistry::from_file(materials, &path)
                .map(|registry| Some(registry.recipes().len())),
        }
    }

    /// Configure `scene` exactly as these settings say. The scene is left
    /// unchanged when the recipe file cannot be read or does not fit the
    /// scene's material table.
    pub fn apply(&self, scene: &mut Scene, scene_path: Option<&Path>) -> Result<(), String> {
        self.validate()?;
        let registry = match self.resolve_recipe(scene_path) {
            Some(path) => Some(RecipeRegistry::from_file(&scene.materials, &path)?),
            None => None,
        };
        scene
            .set_chemistry_interval(self.interval_s)
            .map_err(str::to_owned)?;
        match registry {
            Some(registry) => scene.set_chemistry_registry(registry).map_err(str::to_owned)?,
            None => scene.chemistry_registry = None,
        }
        scene.chemistry_enabled = self.enabled;
        Ok(())
    }

    /// One line for the status bar and the MCP reply.
    pub fn summary(&self) -> String {
        format!(
            "reactions {}, interval {}, recipes {}",
            if self.enabled { "on" } else { "off" },
            interval_text(self.interval_s),
            self.recipe_file.as_deref().map_or("built-in".to_owned(), recipe_name),
        )
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "version": self.version,
            "enabled": self.enabled,
            "interval_s": self.interval_s,
            "recipe_file": self.recipe_file,
        })
    }
}

fn check_interval(interval_s: f64) -> Result<(), String> {
    if !interval_s.is_finite() || !(0.0..=MAX_CHEMISTRY_INTERVAL_S).contains(&interval_s) {
        return Err(format!(
            "chemistry interval must be between 0 and {MAX_CHEMISTRY_INTERVAL_S} s"
        ));
    }
    Ok(())
}

/// `0` is "every step"; otherwise seconds.
pub fn interval_text(interval_s: f64) -> String {
    if interval_s == 0.0 {
        "every step".to_owned()
    } else if interval_s >= 1.0e4 || interval_s < 1.0e-3 {
        format!("{interval_s:.3e} s")
    } else {
        format!("{interval_s} s")
    }
}

/// File name of a stored recipe path, for display.
pub fn recipe_name(stored: &str) -> String {
    Path::new(stored)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(stored)
        .to_owned()
}

/// `recipe` relative to the scene's directory when both are on one root,
/// otherwise the path as given. Always written with `/`.
fn stored_recipe_path(scene_path: &Path, recipe: &Path) -> String {
    let relative = || -> Option<PathBuf> {
        let base = scene_path.parent().filter(|dir| !dir.as_os_str().is_empty());
        let base = fs::canonicalize(base.unwrap_or_else(|| Path::new("."))).ok()?;
        let target = fs::canonicalize(recipe).ok()?;
        let base: Vec<Component> = base.components().collect();
        let target: Vec<Component> = target.components().collect();
        let common = base.iter().zip(&target).take_while(|(a, b)| a == b).count();
        // Different roots (drives) have no relative form.
        if common == 0 {
            return None;
        }
        let mut path = PathBuf::new();
        for _ in common..base.len() {
            path.push("..");
        }
        for component in &target[common..] {
            path.push(component.as_os_str());
        }
        Some(path)
    };
    relative()
        .unwrap_or_else(|| recipe.to_path_buf())
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{SceneBuilder, VoxelSpawn};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!(
            "E:/voxel-chemistry-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn scene() -> Scene {
        SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1))
            .build()
            .unwrap()
    }

    fn wood_rot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes").join("wood_rot.json")
    }

    #[test]
    fn missing_file_means_defaults_and_defaults_change_nothing() {
        let dir = temp_dir("defaults");
        let scene_path = dir.join("a.vxscene");
        let settings = ChemistrySettings::load(&scene_path).unwrap();
        assert_eq!(settings, ChemistrySettings::default());
        let mut scene = scene();
        settings.apply(&mut scene, Some(&scene_path)).unwrap();
        assert!(!scene.chemistry_enabled);
        assert_eq!(scene.chemistry_interval_s, 0.0);
        assert!(scene.chemistry_registry.is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn settings_round_trip_beside_the_scene_and_apply_to_the_scene() {
        let dir = temp_dir("roundtrip");
        let scene_path = dir.join("a.vxscene");
        let mut settings = ChemistrySettings::default();
        settings.set_enabled(true);
        settings.set_interval(3600.0).unwrap();
        let mut scene = scene();
        settings
            .set_recipe_file(&scene_path, &wood_rot(), &scene.materials)
            .unwrap();
        let written = settings.save(&scene_path).unwrap();
        assert_eq!(written, dir.join("a.chemistry.json"));
        assert!(!dir.join("a.chemistry.json.tmp").exists());

        let loaded = ChemistrySettings::load(&scene_path).unwrap();
        assert_eq!(loaded, settings);
        loaded.apply(&mut scene, Some(&scene_path)).unwrap();
        assert!(scene.chemistry_enabled);
        assert_eq!(scene.chemistry_interval_s, 3600.0);
        let registry = scene.chemistry_registry.as_ref().expect("recipe registry");
        assert_eq!(registry.recipes().len(), 1);
        assert_eq!(registry.recipes()[0].name, "decay of wood");

        // Disabling and going back to the built-in recipes undoes all of it.
        let mut off = loaded.clone();
        off.set_enabled(false);
        off.clear_recipe();
        off.set_interval(0.0).unwrap();
        off.apply(&mut scene, Some(&scene_path)).unwrap();
        assert!(!scene.chemistry_enabled);
        assert!(scene.chemistry_registry.is_none());
        assert_eq!(scene.chemistry_interval_s, 0.0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn invalid_values_are_errors_and_leave_the_settings_and_scene_unchanged() {
        let mut settings = ChemistrySettings::default();
        for bad in [-1.0, f64::NAN, f64::INFINITY, MAX_CHEMISTRY_INTERVAL_S * 2.0] {
            assert!(settings.set_interval(bad).is_err(), "{bad}");
        }
        assert_eq!(settings, ChemistrySettings::default());

        let dir = temp_dir("invalid");
        let scene_path = dir.join("a.vxscene");
        let scene = scene();
        let missing = dir.join("nope.json");
        assert!(settings
            .set_recipe_file(&scene_path, &missing, &scene.materials)
            .unwrap_err()
            .contains("cannot read recipe file"));
        let broken = dir.join("broken.json");
        fs::write(&broken, "{ not json").unwrap();
        assert!(settings.set_recipe_file(&scene_path, &broken, &scene.materials).is_err());
        let unknown = dir.join("unknown.json");
        fs::write(
            &unknown,
            r#"{"version":1,"recipes":[{"name":"x","key":"x","terms":[{"component":"no_such_stuff","nu":-1.0},{"component":"water","nu":1.0}],"rate":{"reference_rate_per_s":1.0,"reference_temperature_k":300.0,"activation_energy_j_mol":1000.0,"minimum_temperature_k":250.0,"maximum_temperature_k":400.0}}]}"#,
        )
        .unwrap();
        let error = settings
            .set_recipe_file(&scene_path, &unknown, &scene.materials)
            .unwrap_err();
        assert!(error.contains("no_such_stuff"), "{error}");
        assert_eq!(settings, ChemistrySettings::default());

        // A stored recipe that has since disappeared: apply fails, scene untouched.
        let mut stale = ChemistrySettings::default();
        stale.enabled = true;
        stale.interval_s = 5.0;
        stale.recipe_file = Some("gone.json".to_owned());
        let mut target = scene;
        let error = stale.apply(&mut target, Some(&scene_path)).unwrap_err();
        assert!(error.contains("gone.json"), "{error}");
        assert!(!target.chemistry_enabled && target.chemistry_interval_s == 0.0);

        // Damaged and unsupported settings files are reported, not replaced.
        fs::write(ChemistrySettings::sidecar_path(&scene_path), "garbage").unwrap();
        assert!(ChemistrySettings::load(&scene_path).is_err());
        fs::write(
            ChemistrySettings::sidecar_path(&scene_path),
            r#"{"version":9,"enabled":true,"interval_s":0.0}"#,
        )
        .unwrap();
        assert!(ChemistrySettings::load(&scene_path).unwrap_err().contains("version 9"));
        fs::write(
            ChemistrySettings::sidecar_path(&scene_path),
            r#"{"version":1,"enabled":true,"interval_s":-4.0}"#,
        )
        .unwrap();
        assert!(ChemistrySettings::load(&scene_path).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn recipe_paths_are_stored_relative_to_the_scene_directory() {
        let dir = temp_dir("relative");
        fs::create_dir_all(dir.join("scenes")).unwrap();
        fs::create_dir_all(dir.join("recipes")).unwrap();
        let recipe = dir.join("recipes").join("wood_rot.json");
        fs::copy(wood_rot(), &recipe).unwrap();
        let scene_path = dir.join("scenes").join("a.vxscene");
        let scene = scene();
        let mut settings = ChemistrySettings::default();
        settings.set_recipe_file(&scene_path, &recipe, &scene.materials).unwrap();
        assert_eq!(settings.recipe_file.as_deref(), Some("../recipes/wood_rot.json"));
        assert_eq!(
            settings.resolve_recipe(Some(&scene_path)).unwrap(),
            dir.join("scenes").join("../recipes/wood_rot.json")
        );
        assert_eq!(settings.check_recipe(Some(&scene_path), &scene.materials).unwrap(), Some(1));
        assert!(settings.summary().contains("wood_rot.json"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn enabled_reactions_really_change_the_scene_through_the_engine_hook() {
        // Wet wood decays through the recipe file only when the hook is on.
        let dir = temp_dir("react");
        let scene_path = dir.join("a.vxscene");
        let mut settings = ChemistrySettings::default();
        settings.set_enabled(true);
        settings.set_interval(10.0).unwrap();
        let mut scene = scene();
        settings.set_recipe_file(&scene_path, &wood_rot(), &scene.materials).unwrap();
        settings.apply(&mut scene, Some(&scene_path)).unwrap();
        assert!(scene.advance_chemistry(5.0).unwrap().is_none(), "accumulates below the interval");
        assert!(scene.advance_chemistry(5.0).unwrap().is_some(), "reacts once the interval is reached");
        let _ = fs::remove_dir_all(dir);
    }
}
