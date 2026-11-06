//! Automatic physics tiers for clamped slender parts of a scene.
//!
//! The budget governor (`crate::budget`) decides which tier each region runs at; this
//! module finds the regions that can run as beams, feeds the governor with their cost and
//! error, and applies its plan with `reduce_to_beam` / `expand_beam`.
//!
//! A candidate is a face-connected set of REVs that is a chain of slices along X, whose
//! first slice is entirely fixed and holds all the fixed REVs, with a length of at least
//! `MIN_ASPECT` cross-section widths. Anything else (a body resting on the floor, a part
//! with a free end sitting in a larger structure, a rigid aggregate) is left to the cell
//! solver.
//!
//! A candidate must stay at the cell tier, whatever the budget, while another REV is within
//! `CONTACT_MARGIN` of it, while any bond inside it is damaged, or after the beam model
//! reached its stress limit (for `COOLDOWN_ROUNDS` rounds). Those are exactly the cases the
//! one-way coupling of `beam_region.rs` cannot represent.
use super::*;
use crate::budget::{Governor, Plan, RegionEstimate, Tier};
use std::collections::HashMap;

/// Shortest allowed length, in widths of the larger cross-section dimension.
const MIN_ASPECT: f64 = 6.0;
/// Another REV closer than this, in cells, to a member means contact.
const CONTACT_MARGIN: f64 = 1.6;
/// Governor rounds a region stays at cells after the beam model failed.
const COOLDOWN_ROUNDS: u32 = 30;
/// Peak stress fraction above which a beam region is considered at risk.
const RISK_VALIDITY: f64 = 0.8;

#[derive(Clone, Debug)]
pub struct TierGovernorConfig {
    /// Physics time budget for the whole mechanical solve, milliseconds per fixed step.
    pub budget_ms: f64,
    /// Estimated fine-solver cost of one mechanical REV per fixed step, milliseconds.
    /// 0.13 is the measured value for bulk iron on the i5-14450HX (VALIDATION 6 Nov).
    pub cell_ms_per_rev: f64,
    /// Relative error of the beam against the fine solver for a valid region.
    pub beam_error: f64,
    /// Error tolerated by the governor (the paradigm's 10%).
    pub tolerance: f64,
    /// Beam integrator substeps per fixed step.
    pub beam_substeps: usize,
    /// Fixed steps between governor rounds.
    pub interval_steps: u32,
}

impl Default for TierGovernorConfig {
    fn default() -> Self {
        Self {
            budget_ms: 5.0,
            cell_ms_per_rev: 0.13,
            beam_error: 0.05,
            tolerance: 0.10,
            beam_substeps: 40,
            interval_steps: 10,
        }
    }
}

struct Candidate {
    id: u64,
    /// Scene indices, ascending.
    cells: Vec<usize>,
    slices: usize,
    /// Members that are not fixed: the mechanical REVs the region would cost as cells.
    free: usize,
    region: Option<usize>,
    cooldown: u32,
    /// `reduce_to_beam` refused this chain; do not try again until the scene changes.
    unusable: bool,
}

pub(crate) struct TierGovernor {
    config: TierGovernorConfig,
    governor: Governor,
    candidates: Vec<Candidate>,
    signature: Option<(usize, usize)>,
    since: u32,
    plan: Option<Plan>,
}

impl Scene {
    /// Let the budget governor put clamped slender parts of the scene on the beam tier
    /// when the fine solver would exceed `config.budget_ms`, or when the beam is within the
    /// tolerance and cheaper. Off by default.
    pub fn enable_tier_governor(&mut self, config: TierGovernorConfig) -> Result<(), String> {
        if !(config.budget_ms > 0.0) || config.interval_steps == 0 || config.beam_substeps == 0 {
            return Err("the tier governor needs a positive budget, interval and substep count".into());
        }
        let governor = Governor::new(config.tolerance, 0.5, 3);
        self.tier_governor = Some(Box::new(TierGovernor {
            config,
            governor,
            candidates: Vec::new(),
            signature: None,
            since: u32::MAX,
            plan: None,
        }));
        Ok(())
    }

    /// Stop governing; regions it created are expanded back into cells.
    pub fn disable_tier_governor(&mut self) {
        if let Some(state) = self.tier_governor.take() {
            for candidate in &state.candidates {
                if let Some(id) = candidate.region {
                    let _ = self.expand_beam(id);
                }
            }
        }
    }

    /// The last plan the governor made, for the run panel and tests.
    pub fn tier_plan(&self) -> Option<&Plan> {
        self.tier_governor.as_ref()?.plan.as_ref()
    }

    /// Number of candidate beam regions currently on the beam tier.
    pub fn governed_beam_count(&self) -> usize {
        self.tier_governor.as_ref().map_or(0, |state| {
            state
                .candidates
                .iter()
                .filter(|c| c.region.is_some_and(|id| self.beam_regions.get(id).is_some_and(Option::is_some)))
                .count()
        })
    }

    /// Called at the start of every fixed step.
    pub(crate) fn run_tier_governor(&mut self) {
        let Some(mut state) = self.tier_governor.take() else {
            return;
        };
        state.since = state.since.saturating_add(1);
        if state.since >= state.config.interval_steps {
            state.since = 0;
            self.tier_governor_round(&mut state);
        }
        self.tier_governor = Some(state);
    }

    fn tier_governor_round(&mut self, state: &mut TierGovernor) {
        // Which scene REVs are members of a beam region, and what their own fixed flag is.
        let mut saved: HashMap<usize, bool> = HashMap::new();
        for region in self.beam_regions.iter().flatten() {
            for (local, &i) in region.members().iter().enumerate() {
                saved.insert(i, region.saved_kinematic(local));
            }
        }
        let fixed = |scene: &Scene, i: usize| saved.get(&i).copied().unwrap_or(scene.voxels[i].kinematic);
        let fixed_count = (0..self.voxels.len()).filter(|&i| fixed(self, i)).count();
        let signature = (self.voxels.len(), fixed_count);
        if state.signature != Some(signature) {
            state.signature = Some(signature);
            let found = self.find_beam_candidates(&fixed);
            let old = std::mem::take(&mut state.candidates);
            state.candidates = found
                .into_iter()
                .map(|(cells, slices, free)| {
                    let id = cells[0] as u64;
                    let previous = old.iter().find(|c| c.cells == cells);
                    Candidate {
                        id,
                        slices,
                        free,
                        region: previous.and_then(|c| c.region),
                        cooldown: previous.map_or(0, |c| c.cooldown),
                        unusable: false,
                        cells,
                    }
                })
                .collect();
            // Regions of candidates that no longer exist go back to cells.
            for candidate in &old {
                if let Some(id) = candidate.region {
                    if !state.candidates.iter().any(|c| c.region == Some(id)) {
                        let _ = self.expand_beam(id);
                    }
                }
            }
        }

        // A region that disappeared on its own was expanded by the stress limit.
        for candidate in &mut state.candidates {
            if let Some(id) = candidate.region {
                if self.beam_regions.get(id).map_or(true, Option::is_none) {
                    candidate.region = None;
                    candidate.cooldown = COOLDOWN_ROUNDS;
                }
            }
            candidate.cooldown = candidate.cooldown.saturating_sub(1);
        }

        let touching = self.touching_candidates(&state.candidates);
        let cfg = state.config.clone();
        let mechanical = (0..self.voxels.len())
            .filter(|&i| !self.voxels[i].kinematic && self.aggregate_of[i].is_none())
            .count()
            + saved.iter().filter(|(_, &was_fixed)| !was_fixed).count();
        let governed: usize = state.candidates.iter().filter(|c| !c.unusable).map(|c| c.free).sum();
        let rest_ms = mechanical.saturating_sub(governed) as f64 * cfg.cell_ms_per_rev;

        let mut estimates = Vec::new();
        for (slot, candidate) in state.candidates.iter().enumerate() {
            if candidate.unusable {
                continue;
            }
            let validity = candidate
                .region
                .and_then(|id| self.beam_region_validity(id))
                .unwrap_or(0.0);
            let damaged = self.has_damaged_bond(&candidate.cells);
            let must_be_cells = touching[slot] || damaged || validity >= RISK_VALIDITY || candidate.cooldown > 0;
            let beam_ms = candidate.slices as f64 * 0.002 * cfg.beam_substeps as f64 / 40.0;
            let cells_ms = candidate.free as f64 * cfg.cell_ms_per_rev;
            estimates.push(RegionEstimate {
                id: candidate.id,
                importance: 1.0,
                min_tier: if must_be_cells { Tier::Cells } else { Tier::Beam },
                max_tier: Tier::Cells,
                cost_ms: [beam_ms, beam_ms, beam_ms, cells_ms],
                error: [1.0, 1.0, cfg.beam_error, 0.0],
            });
        }
        let plan = state.governor.decide(&estimates, (cfg.budget_ms - rest_ms).max(0.0));
        for decision in &plan.decisions {
            let Some(candidate) = state.candidates.iter_mut().find(|c| c.id == decision.id) else {
                continue;
            };
            match (decision.tier, candidate.region) {
                (Tier::Beam, None) => match self.reduce_to_beam(&candidate.cells, cfg.beam_substeps) {
                    Ok(id) => candidate.region = Some(id),
                    Err(_) => candidate.unusable = true,
                },
                (Tier::Cells, Some(id)) => {
                    let _ = self.expand_beam(id);
                    candidate.region = None;
                }
                _ => {}
            }
        }
        state.plan = Some(plan);
    }

    /// Chains of slices along X clamped by a fully fixed first slice. `fixed` tells
    /// whether a REV is fixed in its own right (REVs the governor made kinematic count
    /// as free). Each result is (ascending member indices, slice count).
    fn find_beam_candidates(&self, fixed: &dyn Fn(&Scene, usize) -> bool) -> Vec<(Vec<usize>, usize, usize)> {
        let cell = f64::from(CELL);
        let coordinate = |p: DVec3| [(p.x / cell).round() as i32, (p.y / cell).round() as i32, (p.z / cell).round() as i32];
        let n = self.voxels.len();
        let coords: Vec<[i32; 3]> = self.rest_positions.iter().map(|&p| coordinate(p)).collect();
        let mut at: HashMap<[i32; 3], usize> = HashMap::with_capacity(n);
        for (i, c) in coords.iter().enumerate() {
            at.insert(*c, i);
        }
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for (i, c) in coords.iter().enumerate() {
            for axis in 0..3 {
                let mut next = *c;
                next[axis] += 1;
                if let Some(&j) = at.get(&next) {
                    let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                    if a != b {
                        parent[a.max(b)] = a.min(b);
                    }
                }
            }
        }
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let r = root(&mut parent, i);
            components.entry(r).or_default().push(i);
        }
        let mut found = Vec::new();
        for (_, members) in components {
            if members.len() < 12 || members.iter().any(|&i| self.aggregate_of[i].is_some()) {
                continue;
            }
            let min_x = members.iter().map(|&i| coords[i][0]).min().unwrap();
            let max_x = members.iter().map(|&i| coords[i][0]).max().unwrap();
            let clamp_ok = members.iter().all(|&i| fixed(self, i) == (coords[i][0] == min_x));
            if !clamp_ok {
                continue;
            }
            let extent = |axis: usize| {
                let lo = members.iter().map(|&i| coords[i][axis]).min().unwrap();
                let hi = members.iter().map(|&i| coords[i][axis]).max().unwrap();
                f64::from(hi - lo + 1)
            };
            let length = f64::from(max_x - min_x + 1);
            if length < MIN_ASPECT * extent(1).max(extent(2)) {
                continue;
            }
            let free = members.iter().filter(|&&i| !fixed(self, i)).count();
            let mut cells = members;
            cells.sort_unstable();
            found.push((cells, (max_x - min_x + 1) as usize, free));
        }
        found.sort_by_key(|(cells, _, _)| cells[0]);
        found
    }

    /// For each candidate, whether a REV that is not a member is within the contact margin
    /// of a member.
    fn touching_candidates(&self, candidates: &[Candidate]) -> Vec<bool> {
        let margin = CONTACT_MARGIN * f64::from(CELL);
        let mut owner = vec![usize::MAX; self.voxels.len()];
        for (slot, candidate) in candidates.iter().enumerate() {
            for &i in &candidate.cells {
                owner[i] = slot;
            }
        }
        let key = |p: DVec3| [(p.x / margin).floor() as i32, (p.y / margin).floor() as i32, (p.z / margin).floor() as i32];
        let mut grid: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
        for (i, &p) in self.positions.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i);
        }
        let mut touching = vec![false; candidates.len()];
        for (slot, candidate) in candidates.iter().enumerate() {
            'members: for &i in &candidate.cells {
                let k = key(self.positions[i]);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let Some(bucket) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                                continue;
                            };
                            for &j in bucket {
                                if owner[j] != slot && (self.positions[j] - self.positions[i]).length() < margin {
                                    touching[slot] = true;
                                    break 'members;
                                }
                            }
                        }
                    }
                }
            }
        }
        touching
    }

    fn has_damaged_bond(&self, cells: &[usize]) -> bool {
        cells.iter().any(|&i| (0..3).any(|axis| self.bonds[axis].get(i).is_some_and(|bond| bond.damage > 1.0e-6)))
    }
}
