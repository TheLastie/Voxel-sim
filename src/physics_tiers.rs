//! Authored settings of the physics tier governor of a scene, and what the run
//! panel reports about it.
//!
//! One API behind the editor window ("Physics tiers" in the inspector) and the
//! MCP tools `get_scene_physics_tiers` / `set_scene_physics_tiers`. The
//! settings live beside the scene as `<scene>.physics.json`; a scene without
//! the file has the governor off, exactly as before.
//! [`TierSettings::apply`] hands them to the engine through
//! [`Scene::enable_tier_governor`] / [`Scene::disable_tier_governor`]; the
//! governor itself (`scene/tier_governor.rs`) is unchanged.
//!
//! [`GovernorReading`] is the read-only summary the physics worker publishes
//! with each frame, so the window and `get_run_panel` never touch a scene that
//! another thread owns.

use crate::scene::{Scene, TierGovernorConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs, io::Write, path::{Path, PathBuf}};

pub const TIER_FILE_VERSION: u32 = 1;
const MAX_SETTINGS_BYTES: u64 = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierField {
    BudgetMs,
    CellMsPerRev,
    Tolerance,
    IntervalSteps,
    BeamSubsteps,
}

impl TierField {
    pub const ALL: [TierField; 5] = [
        TierField::BudgetMs,
        TierField::CellMsPerRev,
        TierField::Tolerance,
        TierField::IntervalSteps,
        TierField::BeamSubsteps,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TierField::BudgetMs => "Budget (ms/step)",
            TierField::CellMsPerRev => "Cell cost (ms/REV)",
            TierField::Tolerance => "Tolerance",
            TierField::IntervalSteps => "Interval (steps)",
            TierField::BeamSubsteps => "Beam substeps",
        }
    }

    pub fn is_integer(self) -> bool {
        matches!(self, TierField::IntervalSteps | TierField::BeamSubsteps)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierSettings {
    pub version: u32,
    pub enabled: bool,
    /// Physics time budget of the mechanical solve, milliseconds per fixed step.
    pub budget_ms: f64,
    /// Estimated fine-solver cost of one mechanical REV per fixed step, ms.
    pub cell_ms_per_rev: f64,
    /// Error the governor tolerates, as a fraction (0.10 is the paradigm's 10%).
    pub tolerance: f64,
    /// Fixed steps between governor rounds.
    pub interval_steps: u32,
    /// Beam integrator substeps per fixed step.
    pub beam_substeps: usize,
    /// Let settled bodies sleep (`Scene::enable_sleep`). Absent in older files: off.
    #[serde(default)]
    pub sleep: bool,
}

impl Default for TierSettings {
    fn default() -> Self {
        let config = TierGovernorConfig::default();
        Self {
            version: TIER_FILE_VERSION,
            enabled: false,
            budget_ms: config.budget_ms,
            cell_ms_per_rev: config.cell_ms_per_rev,
            tolerance: config.tolerance,
            interval_steps: config.interval_steps,
            beam_substeps: config.beam_substeps,
            sleep: false,
        }
    }
}

/// A change to the settings; `None` fields are kept.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TierPatch {
    pub enabled: Option<bool>,
    pub budget_ms: Option<f64>,
    pub cell_ms_per_rev: Option<f64>,
    pub tolerance: Option<f64>,
    pub interval_steps: Option<u32>,
    pub beam_substeps: Option<usize>,
}

impl TierPatch {
    pub fn enabled(enabled: bool) -> Self {
        Self { enabled: Some(enabled), ..Self::default() }
    }

    /// One numeric field from a typed number. Integer fields must be whole.
    pub fn field(field: TierField, value: f64) -> Result<Self, String> {
        if !value.is_finite() {
            return Err(format!("{} must be a finite number", field.label()));
        }
        if field.is_integer() && (value.fract() != 0.0 || value < 1.0 || value > f64::from(u32::MAX)) {
            return Err(format!("{} must be a whole number of at least 1", field.label()));
        }
        let mut patch = Self::default();
        match field {
            TierField::BudgetMs => patch.budget_ms = Some(value),
            TierField::CellMsPerRev => patch.cell_ms_per_rev = Some(value),
            TierField::Tolerance => patch.tolerance = Some(value),
            TierField::IntervalSteps => patch.interval_steps = Some(value as u32),
            TierField::BeamSubsteps => patch.beam_substeps = Some(value as usize),
        }
        Ok(patch)
    }

    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl TierSettings {
    /// `scenes/a.vxscene` -> `scenes/a.physics.json`.
    pub fn sidecar_path(scene_path: &Path) -> PathBuf {
        scene_path.with_extension("physics.json")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != TIER_FILE_VERSION {
            return Err(format!(
                "unsupported physics tier settings version {} (expected {TIER_FILE_VERSION})",
                self.version
            ));
        }
        let in_range = |value: f64, low: f64, high: f64| value.is_finite() && value > low && value <= high;
        if !in_range(self.budget_ms, 0.0, 1000.0) {
            return Err("budget must be above 0 and at most 1000 ms per step".into());
        }
        if !in_range(self.cell_ms_per_rev, 0.0, 100.0) {
            return Err("cell cost must be above 0 and at most 100 ms per REV".into());
        }
        if !in_range(self.tolerance, 0.0, 1.0) {
            return Err("tolerance must be above 0 and at most 1 (a fraction)".into());
        }
        if !(1..=1_000_000).contains(&self.interval_steps) {
            return Err("interval must be between 1 and 1000000 steps".into());
        }
        if !(1..=10_000).contains(&self.beam_substeps) {
            return Err("beam substeps must be between 1 and 10000".into());
        }
        Ok(())
    }

    /// The settings after `patch`; `self` is untouched and an invalid result is an error.
    pub fn updated(&self, patch: TierPatch) -> Result<Self, String> {
        let mut next = self.clone();
        if let Some(value) = patch.enabled {
            next.enabled = value;
        }
        if let Some(value) = patch.budget_ms {
            next.budget_ms = value;
        }
        if let Some(value) = patch.cell_ms_per_rev {
            next.cell_ms_per_rev = value;
        }
        if let Some(value) = patch.tolerance {
            next.tolerance = value;
        }
        if let Some(value) = patch.interval_steps {
            next.interval_steps = value;
        }
        if let Some(value) = patch.beam_substeps {
            next.beam_substeps = value;
        }
        next.validate()?;
        Ok(next)
    }

    pub fn field_value(&self, field: TierField) -> f64 {
        match field {
            TierField::BudgetMs => self.budget_ms,
            TierField::CellMsPerRev => self.cell_ms_per_rev,
            TierField::Tolerance => self.tolerance,
            TierField::IntervalSteps => f64::from(self.interval_steps),
            TierField::BeamSubsteps => self.beam_substeps as f64,
        }
    }

    pub fn load(scene_path: &Path) -> Result<Self, String> {
        let path = Self::sidecar_path(scene_path);
        match fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
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

    pub fn save(&self, scene_path: &Path) -> Result<PathBuf, String> {
        self.validate()?;
        let path = Self::sidecar_path(scene_path);
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| format!("cannot encode physics tier settings: {error}"))?;
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

    /// The engine configuration these settings stand for.
    pub fn to_config(&self) -> TierGovernorConfig {
        TierGovernorConfig {
            budget_ms: self.budget_ms,
            cell_ms_per_rev: self.cell_ms_per_rev,
            tolerance: self.tolerance,
            interval_steps: self.interval_steps,
            beam_substeps: self.beam_substeps,
            ..TierGovernorConfig::default()
        }
    }

    /// Configure `scene`: enable the governor with these settings, or disable
    /// it (regions it created go back to cells). Unchanged on error.
    pub fn apply(&self, scene: &mut Scene) -> Result<(), String> {
        self.validate()?;
        if self.sleep {
            scene.enable_sleep(crate::scene::SleepConfig::default())?;
        } else {
            scene.disable_sleep();
        }
        if self.enabled {
            scene.enable_tier_governor(self.to_config())
        } else {
            scene.disable_tier_governor();
            Ok(())
        }
    }

    pub fn summary(&self) -> String {
        if !self.enabled {
            return "physics tiers off".to_owned();
        }
        format!(
            "physics tiers on: budget {} ms, {} ms per REV, tolerance {}, every {} steps, {} beam substeps",
            self.budget_ms, self.cell_ms_per_rev, self.tolerance, self.interval_steps, self.beam_substeps
        )
    }

    pub fn to_json(&self) -> Value {
        json!({
            "version": self.version,
            "enabled": self.enabled,
            "budget_ms": self.budget_ms,
            "cell_ms_per_rev": self.cell_ms_per_rev,
            "tolerance": self.tolerance,
            "interval_steps": self.interval_steps,
            "beam_substeps": self.beam_substeps,
            "sleep": self.sleep,
        })
    }
}

/// The governor's latest plan, as published.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanReading {
    pub total_cost_ms: f64,
    pub budget_ms: f64,
    pub overloaded: bool,
    pub over_tolerance_regions: usize,
}

/// What the physics worker publishes about an enabled governor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GovernorReading {
    /// Candidate beam regions currently on the beam tier.
    pub governed_beams: usize,
    /// `None` until the governor has made its first plan.
    pub plan: Option<PlanReading>,
}

impl GovernorReading {
    /// `None` when the scene has no governor.
    pub fn from_scene(scene: &Scene) -> Option<Self> {
        scene.tier_governor.as_ref()?;
        Some(Self {
            governed_beams: scene.governed_beam_count(),
            plan: scene.tier_plan().map(|plan| PlanReading {
                total_cost_ms: plan.total_cost_ms,
                budget_ms: plan.budget_ms,
                overloaded: plan.overloaded,
                over_tolerance_regions: plan.over_tolerance.len(),
            }),
        })
    }

    pub fn line(reading: Option<&Self>) -> String {
        let Some(reading) = reading else {
            return "Tier governor: off".to_owned();
        };
        let beams = format!("{} governed beam{}", reading.governed_beams, if reading.governed_beams == 1 { "" } else { "s" });
        match &reading.plan {
            None => format!("Tier governor: on, {beams}, no plan yet"),
            Some(plan) => format!(
                "Tier governor: {beams}, plan {:.2} / {:.2} ms{}{}",
                plan.total_cost_ms,
                plan.budget_ms,
                if plan.overloaded { ", OVERLOADED" } else { "" },
                if plan.over_tolerance_regions > 0 {
                    format!(", {} over tolerance", plan.over_tolerance_regions)
                } else {
                    String::new()
                },
            ),
        }
    }

    pub fn to_json(reading: Option<&Self>) -> Value {
        match reading {
            None => json!({"enabled": false}),
            Some(reading) => json!({
                "enabled": true,
                "governed_beam_count": reading.governed_beams,
                "plan": reading.plan.as_ref().map(|plan| json!({
                    "total_cost_ms": plan.total_cost_ms,
                    "budget_ms": plan.budget_ms,
                    "overloaded": plan.overloaded,
                    "over_tolerance_regions": plan.over_tolerance_regions,
                })),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{SceneBuilder, VoxelSpawn};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!(
            "E:/voxel-tiers-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn scene() -> Scene {
        SceneBuilder::standard().add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1)).build().unwrap()
    }

    /// A clamped bar along X: a fixed first slice and free slices behind it.
    fn bar() -> Scene {
        let mut builder = SceneBuilder::standard();
        for x in 0..40 {
            let spawn = VoxelSpawn::at_cell([x, 0, 0], 1);
            builder = builder.add_voxel(if x == 0 { spawn.fixed() } else { spawn });
        }
        builder.build().unwrap()
    }

    #[test]
    fn defaults_are_the_governors_defaults_and_off() {
        let settings = TierSettings::default();
        let config = TierGovernorConfig::default();
        assert!(!settings.enabled);
        assert_eq!(settings.to_config().budget_ms, config.budget_ms);
        assert_eq!(settings.to_config().beam_error, config.beam_error);
        settings.validate().unwrap();
        let dir = temp_dir("defaults");
        assert_eq!(TierSettings::load(&dir.join("a.vxscene")).unwrap(), settings);
        let mut scene = scene();
        settings.apply(&mut scene).unwrap();
        assert!(scene.tier_governor.is_none());
        assert!(GovernorReading::from_scene(&scene).is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn invalid_values_change_nothing() {
        let base = TierSettings::default();
        let bad: [TierPatch; 12] = [
            TierPatch { budget_ms: Some(0.0), ..Default::default() },
            TierPatch { budget_ms: Some(-1.0), ..Default::default() },
            TierPatch { budget_ms: Some(f64::NAN), ..Default::default() },
            TierPatch { budget_ms: Some(1e9), ..Default::default() },
            TierPatch { cell_ms_per_rev: Some(0.0), ..Default::default() },
            TierPatch { cell_ms_per_rev: Some(f64::INFINITY), ..Default::default() },
            TierPatch { tolerance: Some(0.0), ..Default::default() },
            TierPatch { tolerance: Some(1.5), ..Default::default() },
            TierPatch { interval_steps: Some(0), ..Default::default() },
            TierPatch { interval_steps: Some(2_000_000), ..Default::default() },
            TierPatch { beam_substeps: Some(0), ..Default::default() },
            TierPatch { beam_substeps: Some(100_000), ..Default::default() },
        ];
        for patch in bad {
            assert!(base.updated(patch).is_err(), "{patch:?}");
        }
        assert!(TierPatch::field(TierField::IntervalSteps, 2.5).is_err());
        assert!(TierPatch::field(TierField::BeamSubsteps, 0.0).is_err());
        assert!(TierPatch::field(TierField::BudgetMs, f64::NAN).is_err());
        assert!(TierPatch::field(TierField::IntervalSteps, 4.0).unwrap().interval_steps == Some(4));
        assert_eq!(base.updated(TierPatch::field(TierField::Tolerance, 0.2).unwrap()).unwrap().tolerance, 0.2);
        assert_eq!(base, TierSettings::default());
        assert!(TierPatch::default().is_empty());
        for field in TierField::ALL {
            assert_eq!(base.field_value(field) > 0.0, true);
        }
    }

    #[test]
    fn settings_round_trip_beside_the_scene_and_damaged_files_are_errors() {
        let dir = temp_dir("roundtrip");
        let scene_path = dir.join("a.vxscene");
        let settings = TierSettings::default()
            .updated(TierPatch { enabled: Some(true), budget_ms: Some(2.0), interval_steps: Some(5), ..Default::default() })
            .unwrap();
        assert_eq!(settings.save(&scene_path).unwrap(), dir.join("a.physics.json"));
        assert!(!dir.join("a.physics.json.tmp").exists());
        assert_eq!(TierSettings::load(&scene_path).unwrap(), settings);
        for broken in [
            "garbage".to_owned(),
            r#"{"version":7,"enabled":true,"budget_ms":1.0,"cell_ms_per_rev":0.1,"tolerance":0.1,"interval_steps":1,"beam_substeps":1}"#.to_owned(),
            r#"{"version":1,"enabled":true,"budget_ms":-1.0,"cell_ms_per_rev":0.1,"tolerance":0.1,"interval_steps":1,"beam_substeps":1}"#.to_owned(),
            r#"{"version":1,"enabled":true,"extra":1}"#.to_owned(),
        ] {
            fs::write(TierSettings::sidecar_path(&scene_path), &broken).unwrap();
            assert!(TierSettings::load(&scene_path).is_err(), "{broken}");
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn apply_enables_and_disables_the_engine_governor_and_the_reading_follows() {
        let mut scene = bar();
        let on = TierSettings::default()
            .updated(TierPatch { enabled: Some(true), budget_ms: Some(0.5), interval_steps: Some(1), ..Default::default() })
            .unwrap();
        on.apply(&mut scene).unwrap();
        let reading = GovernorReading::from_scene(&scene).expect("governor enabled");
        assert_eq!((reading.governed_beams, reading.plan.is_none()), (0, true));
        assert_eq!(GovernorReading::line(Some(&reading)), "Tier governor: on, 0 governed beams, no plan yet");

        // After a step the governor has made a plan with the configured budget.
        crate::solver::step(&mut scene, &mut None);
        let reading = GovernorReading::from_scene(&scene).unwrap();
        let plan = reading.plan.as_ref().expect("plan after a governor round");
        assert_eq!(plan.budget_ms, 0.5);
        assert!(plan.total_cost_ms.is_finite());
        let json = GovernorReading::to_json(Some(&reading));
        assert_eq!(json["enabled"], true);
        assert_eq!(json["governed_beam_count"], reading.governed_beams);
        assert_eq!(json["plan"]["budget_ms"], 0.5);

        TierSettings::default().apply(&mut scene).unwrap();
        assert!(GovernorReading::from_scene(&scene).is_none());
        assert_eq!(GovernorReading::to_json(None), json!({"enabled": false}));
        assert_eq!(GovernorReading::line(None), "Tier governor: off");
    }


    #[test]
    fn the_worker_publishes_the_governor_reading_with_its_frames() {
        let mut scene = bar();
        TierSettings::default()
            .updated(TierPatch { enabled: Some(true), interval_steps: Some(1), ..Default::default() })
            .unwrap()
            .apply(&mut scene)
            .unwrap();
        let mut worker = crate::world_physics::PhysicsWorker::new(scene).unwrap();
        assert_eq!(worker.frame.tier_governor, Some(GovernorReading { governed_beams: 0, plan: None }));
        let started = std::time::Instant::now();
        while worker.frame.completed_steps == 0 {
            assert!(started.elapsed().as_secs() < 60, "no frame from the worker");
            worker.advance(0.02, false);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let reading = worker.frame.tier_governor.expect("governor reading in the frame");
        assert!(reading.plan.is_some(), "the worker's frame carries the governor's plan");

        // A scene without a governor publishes none.
        let plain = crate::world_physics::PhysicsWorker::new(self::scene()).unwrap();
        assert!(plain.frame.tier_governor.is_none());
    }
    #[test]
    fn plan_line_reports_overload_and_tolerance() {
        let reading = GovernorReading {
            governed_beams: 1,
            plan: Some(PlanReading { total_cost_ms: 7.5, budget_ms: 5.0, overloaded: true, over_tolerance_regions: 2 }),
        };
        let line = GovernorReading::line(Some(&reading));
        assert_eq!(line, "Tier governor: 1 governed beam, plan 7.50 / 5.00 ms, OVERLOADED, 2 over tolerance");
        let json = GovernorReading::to_json(Some(&reading));
        assert_eq!(json["plan"]["overloaded"], true);
        assert_eq!(json["plan"]["over_tolerance_regions"], 2);
    }
}
