//! The run panel: one measured status of the physics run, shared by the editor
//! window and the MCP `get_run_panel` tool.
//!
//! Everything shown is read from the physics worker or the scene; nothing is
//! estimated. The worker cost is the full cost of a batch per 10 ms of
//! simulated time (`PhysicsFrame::full_worker_ms_per_10ms_simulated`); the
//! panel keeps the last [`COST_WINDOW`] distinct batches in a
//! [`CostWindow`] and reports p50, p95 and the maximum next to the latest
//! value, because a mean hides the tail a frame-by-frame game feels.

use crate::budget::CostWindow;
use serde_json::{json, Value};

/// Batches kept for p50, p95 and max.
pub const COST_WINDOW: usize = 128;
/// Upper limit of the worker cost per 10 ms of physics (INTEGRATION_PLAN).
pub const COST_BUDGET_MS: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    Run,
    Pause,
    Failed,
}

impl RunState {
    pub fn from_flags(paused: bool, failed: bool) -> Self {
        if failed {
            RunState::Failed
        } else if paused {
            RunState::Pause
        } else {
            RunState::Run
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            RunState::Run => "Run",
            RunState::Pause => "Pause",
            RunState::Failed => "Failed",
        }
    }
}

/// Which solve the solver uses for this scene (`solver::uses_cell_rotations`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverMode {
    CellRotations,
    TranslationOnly,
}

impl SolverMode {
    pub fn from_cell_rotations(cell_rotations: bool) -> Self {
        if cell_rotations {
            SolverMode::CellRotations
        } else {
            SolverMode::TranslationOnly
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SolverMode::CellRotations => "cell rotations",
            SolverMode::TranslationOnly => "translation only",
        }
    }

    fn key(self) -> &'static str {
        match self {
            SolverMode::CellRotations => "cell_rotations",
            SolverMode::TranslationOnly => "translation_only",
        }
    }
}

/// Collects worker costs between redraws. Call [`RunPanel::record`] on every
/// redraw with the worker's current frame; a cost enters the window once per
/// published batch, not once per redraw.
#[derive(Clone, Debug)]
pub struct RunPanel {
    window: CostWindow,
    latest: Option<f64>,
    last_key: Option<(u64, u64)>,
}

impl Default for RunPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl RunPanel {
    pub fn new() -> Self {
        Self { window: CostWindow::new(COST_WINDOW), latest: None, last_key: None }
    }

    /// Forget all samples (no worker, or a different scene).
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// `generation` identifies a worker run and `revision` its published
    /// state, as in `PhysicsFrame`. A new generation starts a new window.
    pub fn record(&mut self, generation: u64, revision: u64, cost_ms: Option<f64>) {
        if self.last_key.is_some_and(|(last, _)| last != generation) {
            self.clear();
        }
        let key = (generation, revision);
        if self.last_key == Some(key) {
            return;
        }
        self.last_key = Some(key);
        let Some(cost) = cost_ms.filter(|value| value.is_finite() && *value >= 0.0) else {
            return;
        };
        self.latest = Some(cost);
        self.window.push(cost);
    }

    pub fn snapshot(&self, input: RunPanelInput) -> RunPanelSnapshot {
        RunPanelSnapshot {
            state: input.state,
            requested_speed: input.requested_speed,
            measured_speed: input.measured_speed.filter(|v| v.is_finite() && *v >= 0.0),
            backlog_s: input.backlog_s.filter(|v| v.is_finite() && *v >= 0.0),
            cost_latest_ms: self.latest,
            cost_p50_ms: self.window.p50(),
            cost_p95_ms: self.window.p95(),
            cost_max_ms: self.window.max(),
            cost_samples: self.window.len(),
            active_mechanical_revs: input.active_mechanical_revs,
            solver_mode: input.solver_mode,
        }
    }
}

/// What the caller measures; the panel adds the cost statistics.
#[derive(Clone, Copy, Debug)]
pub struct RunPanelInput {
    pub state: RunState,
    pub requested_speed: f32,
    pub measured_speed: Option<f64>,
    pub backlog_s: Option<f64>,
    pub active_mechanical_revs: Option<usize>,
    pub solver_mode: Option<SolverMode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunPanelSnapshot {
    pub state: RunState,
    pub requested_speed: f32,
    pub measured_speed: Option<f64>,
    pub backlog_s: Option<f64>,
    pub cost_latest_ms: Option<f64>,
    pub cost_p50_ms: Option<f64>,
    pub cost_p95_ms: Option<f64>,
    pub cost_max_ms: Option<f64>,
    pub cost_samples: usize,
    pub active_mechanical_revs: Option<usize>,
    pub solver_mode: Option<SolverMode>,
}

fn ms(value: Option<f64>) -> String {
    value.map_or_else(|| "—".to_owned(), |v| format!("{v:.2} ms"))
}

impl RunPanelSnapshot {
    /// The rows shown in the window, top to bottom.
    pub fn lines(&self) -> Vec<String> {
        let speed = self
            .measured_speed
            .map_or_else(|| "—".to_owned(), |v| format!("{v:.2}×"));
        let backlog = self
            .backlog_s
            .map_or_else(|| "—".to_owned(), |v| format!("{v:.2} s"));
        let revs = self
            .active_mechanical_revs
            .map_or_else(|| "—".to_owned(), |v| v.to_string());
        let mode = self.solver_mode.map_or("—", SolverMode::name);
        vec![
            format!(
                "State {}   World speed {} (target {:.2}×)   Backlog {}",
                self.state.name(),
                speed,
                self.requested_speed,
                backlog
            ),
            format!(
                "Worker cost per 10 ms of physics (budget {COST_BUDGET_MS} ms):   latest {}",
                ms(self.cost_latest_ms)
            ),
            format!(
                "p50 {}   p95 {}   max {}   (last {} batches)",
                ms(self.cost_p50_ms),
                ms(self.cost_p95_ms),
                ms(self.cost_max_ms),
                self.cost_samples
            ),
            format!("Active mechanical REVs {revs}   Solver: {mode}"),
        ]
    }

    pub fn to_json(&self) -> Value {
        json!({
            "state": self.state.name(),
            "requested_speed": self.requested_speed,
            "measured_world_speed": self.measured_speed,
            "backlog_seconds": self.backlog_s,
            "worker_cost_ms_per_10ms": {
                "latest": self.cost_latest_ms,
                "p50": self.cost_p50_ms,
                "p95": self.cost_p95_ms,
                "max": self.cost_max_ms,
                "samples": self.cost_samples,
                "window_capacity": COST_WINDOW,
                "budget_ms": COST_BUDGET_MS,
                "scope": "full worker batch normalized to 10 ms of simulated time",
            },
            "active_mechanical_revs": self.active_mechanical_revs,
            "solver_mode": self.solver_mode.map(SolverMode::key),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> RunPanelInput {
        RunPanelInput {
            state: RunState::Run,
            requested_speed: 1.0,
            measured_speed: Some(0.97),
            backlog_s: Some(0.25),
            active_mechanical_revs: Some(512),
            solver_mode: Some(SolverMode::CellRotations),
        }
    }

    #[test]
    fn a_cost_enters_the_window_once_per_published_batch() {
        let mut panel = RunPanel::new();
        for cost in [1.0, 2.0, 3.0, 4.0, 10.0] {
            // The same batch is seen on several redraws.
            let revision = cost as u64;
            panel.record(1, revision, Some(cost));
            panel.record(1, revision, Some(cost));
            panel.record(1, revision, Some(cost));
        }
        let snapshot = panel.snapshot(input());
        assert_eq!(snapshot.cost_samples, 5);
        assert_eq!(snapshot.cost_latest_ms, Some(10.0));
        assert_eq!(snapshot.cost_p50_ms, Some(3.0));
        assert_eq!(snapshot.cost_p95_ms, Some(10.0));
        assert_eq!(snapshot.cost_max_ms, Some(10.0));
    }

    #[test]
    fn percentiles_come_from_the_cost_window_and_see_the_tail() {
        let mut panel = RunPanel::new();
        for index in 0..100_u64 {
            panel.record(1, index + 1, Some(if index < 95 { 1.0 } else { 9.0 }));
        }
        let snapshot = panel.snapshot(input());
        assert_eq!(snapshot.cost_p50_ms, Some(1.0));
        assert_eq!(snapshot.cost_p95_ms, Some(1.0));
        assert_eq!(snapshot.cost_max_ms, Some(9.0), "a mean of 1.4 would hide this");
        // Old batches fall out of the window.
        for index in 0..COST_WINDOW as u64 {
            panel.record(1, 1000 + index, Some(2.0));
        }
        let snapshot = panel.snapshot(input());
        assert_eq!(snapshot.cost_samples, COST_WINDOW);
        assert_eq!(snapshot.cost_max_ms, Some(2.0));
    }

    #[test]
    fn a_new_worker_generation_starts_a_new_window_and_bad_costs_are_ignored() {
        let mut panel = RunPanel::new();
        panel.record(1, 1, Some(7.0));
        panel.record(2, 1, Some(1.0));
        assert_eq!(panel.snapshot(input()).cost_max_ms, Some(1.0));
        panel.record(2, 2, Some(f64::NAN));
        panel.record(2, 3, Some(-1.0));
        panel.record(2, 4, None);
        let snapshot = panel.snapshot(input());
        assert_eq!(snapshot.cost_samples, 1);
        panel.clear();
        let empty = panel.snapshot(input());
        assert_eq!((empty.cost_latest_ms, empty.cost_p50_ms, empty.cost_max_ms), (None, None, None));
        assert_eq!(empty.cost_samples, 0);
    }

    #[test]
    fn state_follows_failed_then_paused_then_running() {
        assert_eq!(RunState::from_flags(false, false), RunState::Run);
        assert_eq!(RunState::from_flags(true, false), RunState::Pause);
        assert_eq!(RunState::from_flags(false, true), RunState::Failed);
        assert_eq!(RunState::from_flags(true, true), RunState::Failed);
    }

    #[test]
    fn lines_and_json_report_the_same_measured_values() {
        let mut panel = RunPanel::new();
        panel.record(1, 1, Some(2.5));
        let snapshot = panel.snapshot(input());
        let lines = snapshot.lines().join("\n");
        assert!(lines.contains("State Run"), "{lines}");
        assert!(lines.contains("World speed 0.97×"), "{lines}");
        assert!(lines.contains("Backlog 0.25 s"), "{lines}");
        assert!(lines.contains("latest 2.50 ms"), "{lines}");
        assert!(lines.contains("p50 2.50 ms   p95 2.50 ms   max 2.50 ms"), "{lines}");
        assert!(lines.contains("Active mechanical REVs 512"), "{lines}");
        assert!(lines.contains("Solver: cell rotations"), "{lines}");
        let json = snapshot.to_json();
        assert_eq!(json["state"], "Run");
        assert_eq!(json["measured_world_speed"], 0.97);
        assert_eq!(json["worker_cost_ms_per_10ms"]["p95"], 2.5);
        assert_eq!(json["worker_cost_ms_per_10ms"]["budget_ms"], 5.0);
        assert_eq!(json["active_mechanical_revs"], 512);
        assert_eq!(json["solver_mode"], "cell_rotations");
    }

    #[test]
    fn unmeasured_values_are_dashes_and_nulls_never_zeros() {
        let snapshot = RunPanel::new().snapshot(RunPanelInput {
            state: RunState::Pause,
            requested_speed: 1.0,
            measured_speed: None,
            backlog_s: Some(f64::NAN),
            active_mechanical_revs: None,
            solver_mode: None,
        });
        let lines = snapshot.lines().join("\n");
        assert!(lines.contains("State Pause   World speed —"), "{lines}");
        assert!(lines.contains("latest —"), "{lines}");
        assert!(lines.contains("p50 —   p95 —   max —"), "{lines}");
        assert!(lines.contains("Active mechanical REVs —   Solver: —"), "{lines}");
        let json = snapshot.to_json();
        assert!(json["measured_world_speed"].is_null());
        assert!(json["backlog_seconds"].is_null());
        assert!(json["worker_cost_ms_per_10ms"]["p50"].is_null());
        assert!(json["solver_mode"].is_null());
    }

    #[test]
    fn solver_mode_matches_the_solver_decision_for_a_real_scene() {
        use crate::scene::{SceneBuilder, VoxelSpawn};
        let mut builder = SceneBuilder::standard();
        for x in 0..3 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, 0, 0], 1));
        }
        builder = builder.add_voxel(VoxelSpawn::at_cell([0, 0, -1], 1).fixed());
        let scene = builder.build().unwrap();
        // Three free REVs; the fixed one is not mechanical.
        assert_eq!(crate::solver::mechanical_rev_count(&scene), 3);
        assert!(crate::solver::uses_cell_rotations(&scene), "small scenes use cell rotations");
        let frame = crate::world_physics::PhysicsFrame::capture(
            &scene,
            glam::DVec3::ZERO,
            0.0,
            0.0,
            1,
            1,
        )
        .unwrap();
        assert_eq!(frame.mechanical_revs, 3);
        assert!(frame.cell_rotations);
        assert_eq!(SolverMode::from_cell_rotations(frame.cell_rotations), SolverMode::CellRotations);
        assert_eq!(SolverMode::from_cell_rotations(false).name(), "translation only");
    }
}
