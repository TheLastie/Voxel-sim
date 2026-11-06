//! Physics budget governor.
//!
//! The world is simulated at four physics tiers (INTEGRATION_PLAN_2026-11-05.md section 7):
//! sleeping, rigid body, reduced beam and fine cells. Every region of the world can be
//! represented at any tier its content allows, at a different cost and with a different
//! error against the fine solver. The governor picks one tier per region each step so that
//! the summed cost fits the budget.
//!
//! Rules it enforces:
//! * Time is never slowed or skipped. When the budget cannot be met, regions are demoted
//!   to coarser tiers, lowest importance first, and the plan says so; it does not return
//!   fewer steps.
//! * A region is never demoted below its `min_tier` (a live contact or crack needs cells),
//!   and a region whose importance is infinite is never demoted at all. If those alone
//!   exceed the budget the plan is marked `overloaded` and still returned.
//! * The tier is the cheapest one whose estimated error is within the tolerance (10% by
//!   default, section 10 of the paradigm). Leftover budget may buy extra fidelity for
//!   important regions, but never at the price of thrashing between tiers.
//! * The same inputs give the same plan, whatever order the regions arrive in.
//!
//! This module only decides; the tiers themselves live elsewhere.
use std::collections::{BTreeMap, VecDeque};

/// Physics tiers, coarsest first. `Cells` is the fine REV solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Sleeping = 0,
    Rigid = 1,
    Beam = 2,
    Cells = 3,
}

pub const TIERS: [Tier; 4] = [Tier::Sleeping, Tier::Rigid, Tier::Beam, Tier::Cells];

impl Tier {
    pub fn index(self) -> usize {
        self as usize
    }

    fn below(self) -> Option<Tier> {
        match self {
            Tier::Sleeping => None,
            Tier::Rigid => Some(Tier::Sleeping),
            Tier::Beam => Some(Tier::Rigid),
            Tier::Cells => Some(Tier::Beam),
        }
    }

    fn above(self) -> Option<Tier> {
        match self {
            Tier::Sleeping => Some(Tier::Rigid),
            Tier::Rigid => Some(Tier::Beam),
            Tier::Beam => Some(Tier::Cells),
            Tier::Cells => None,
        }
    }
}

/// What the caller knows about one region this step. Costs are milliseconds of wall time
/// for one 10 ms physics step; errors are relative to the fine solver on the observable
/// the region cares about.
#[derive(Clone, Debug)]
pub struct RegionEstimate {
    pub id: u64,
    /// Larger is more important to keep accurate. `f64::INFINITY` pins the region: it is
    /// never demoted (the player's hands, a live crack).
    pub importance: f64,
    pub min_tier: Tier,
    pub max_tier: Tier,
    pub cost_ms: [f64; 4],
    pub error: [f64; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The cheapest tier within the tolerance.
    Tolerance,
    /// Kept at a finer tier for a few steps to avoid thrashing.
    Held,
    /// Moved to a coarser tier because the budget was exceeded.
    Demoted,
    /// Moved to a finer tier with leftover budget.
    Promoted,
    /// Raised because the tolerance, or the region's minimum tier, requires it.
    Required,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub id: u64,
    pub tier: Tier,
    pub reason: Reason,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub decisions: Vec<Decision>,
    pub total_cost_ms: f64,
    pub budget_ms: f64,
    /// The pinned and minimum-tier regions alone cost more than the budget. The plan is
    /// still the cheapest allowed; the world is not slowed down.
    pub overloaded: bool,
    /// Regions whose chosen tier has an estimated error above the tolerance.
    pub over_tolerance: Vec<u64>,
}

#[derive(Clone, Debug)]
struct RegionState {
    tier: Tier,
    /// Consecutive steps a coarser tier has been enough.
    coarser_enough: u32,
    /// Steps during which surplus promotion is not allowed after a demotion.
    cooldown: u32,
}

#[derive(Clone, Debug)]
pub struct Governor {
    pub tolerance: f64,
    /// Share of the leftover budget that may buy extra fidelity.
    pub surplus_fraction: f64,
    /// Steps a coarser tier must stay sufficient before a region is demoted.
    pub hold_steps: u32,
    state: BTreeMap<u64, RegionState>,
}

impl Default for Governor {
    fn default() -> Self {
        Self { tolerance: 0.10, surplus_fraction: 0.5, hold_steps: 5, state: BTreeMap::new() }
    }
}

fn cheapest_sufficient(region: &RegionEstimate, tolerance: f64) -> (Tier, bool) {
    for tier in TIERS {
        if tier < region.min_tier || tier > region.max_tier {
            continue;
        }
        if region.error[tier.index()] <= tolerance {
            return (tier, false);
        }
    }
    // No allowed tier is accurate enough: take the most accurate one.
    let mut best = region.min_tier;
    for tier in TIERS {
        if tier >= region.min_tier
            && tier <= region.max_tier
            && region.error[tier.index()] < region.error[best.index()]
        {
            best = tier;
        }
    }
    (best, true)
}

impl Governor {
    pub fn new(tolerance: f64, surplus_fraction: f64, hold_steps: u32) -> Self {
        Self { tolerance, surplus_fraction, hold_steps, state: BTreeMap::new() }
    }

    /// Forget a region (it left the loaded area or was merged).
    pub fn forget(&mut self, id: u64) {
        self.state.remove(&id);
    }

    pub fn tier_of(&self, id: u64) -> Option<Tier> {
        self.state.get(&id).map(|state| state.tier)
    }

    /// Choose a tier for every region so that the summed cost fits `budget_ms`.
    pub fn decide(&mut self, regions: &[RegionEstimate], budget_ms: f64) -> Plan {
        let mut sorted: Vec<&RegionEstimate> = regions.iter().collect();
        sorted.sort_by_key(|region| region.id);
        self.state.retain(|id, _| sorted.iter().any(|region| region.id == *id));

        let mut chosen: Vec<(Tier, Reason)> = Vec::with_capacity(sorted.len());
        for region in &sorted {
            let (wanted, _) = cheapest_sufficient(region, self.tolerance);
            let entry = self.state.entry(region.id).or_insert(RegionState {
                tier: wanted,
                coarser_enough: 0,
                cooldown: 0,
            });
            entry.cooldown = entry.cooldown.saturating_sub(1);
            let mut tier = entry.tier.clamp(region.min_tier, region.max_tier);
            let mut reason = Reason::Tolerance;
            if tier != entry.tier {
                reason = Reason::Required;
            }
            if wanted > tier {
                tier = wanted;
                reason = Reason::Required;
                entry.coarser_enough = 0;
            } else if wanted < tier {
                entry.coarser_enough += 1;
                if entry.coarser_enough >= self.hold_steps {
                    tier = wanted;
                    entry.coarser_enough = 0;
                    entry.cooldown = self.hold_steps * 4;
                } else {
                    reason = Reason::Held;
                }
            } else {
                entry.coarser_enough = 0;
            }
            chosen.push((tier, reason));
        }

        let cost = |chosen: &[(Tier, Reason)]| -> f64 {
            sorted.iter().zip(chosen).map(|(region, (tier, _))| region.cost_ms[tier.index()]).sum()
        };
        let mut total = cost(&chosen);

        // Over budget: demote the least important first, never below min_tier and never
        // a pinned region. One tier at a time, so the order is stable and explainable.
        while total > budget_ms {
            let mut victim: Option<usize> = None;
            for (slot, region) in sorted.iter().enumerate() {
                let (tier, _) = chosen[slot];
                if region.importance.is_infinite() || tier <= region.min_tier || tier.below().is_none() {
                    continue;
                }
                let better = match victim {
                    None => true,
                    Some(current) => {
                        let other = sorted[current];
                        region.importance < other.importance
                            || (region.importance == other.importance && region.id < other.id)
                    }
                };
                if better {
                    victim = Some(slot);
                }
            }
            let Some(slot) = victim else { break };
            let region = sorted[slot];
            let tier = chosen[slot].0.below().unwrap().max(region.min_tier);
            chosen[slot] = (tier, Reason::Demoted);
            if let Some(entry) = self.state.get_mut(&region.id) {
                entry.cooldown = self.hold_steps * 4;
            }
            total = cost(&chosen);
        }
        let overloaded = total > budget_ms;

        // Leftover budget buys fidelity for important regions, best gain per millisecond
        // first, only for regions that have not been demoted recently.
        if !overloaded {
            let mut surplus = (budget_ms - total) * self.surplus_fraction;
            loop {
                let mut best: Option<(usize, f64, f64)> = None;
                for (slot, region) in sorted.iter().enumerate() {
                    let (tier, _) = chosen[slot];
                    let Some(next) = tier.above() else { continue };
                    if next > region.max_tier || region.importance.is_infinite() {
                        continue;
                    }
                    if self.state.get(&region.id).map_or(false, |entry| entry.cooldown > 0) {
                        continue;
                    }
                    let gain = region.importance
                        * (region.error[tier.index()] - region.error[next.index()]).max(0.0);
                    if gain <= 0.0 {
                        continue;
                    }
                    let extra = (region.cost_ms[next.index()] - region.cost_ms[tier.index()]).max(0.0);
                    if extra > surplus {
                        continue;
                    }
                    let ratio = if extra <= 1.0e-12 { f64::INFINITY } else { gain / extra };
                    let better = match best {
                        None => true,
                        Some((_, best_ratio, _)) => ratio > best_ratio,
                    };
                    if better {
                        best = Some((slot, ratio, extra));
                    }
                }
                let Some((slot, _, extra)) = best else { break };
                chosen[slot] = (chosen[slot].0.above().unwrap(), Reason::Promoted);
                surplus -= extra;
            }
            total = cost(&chosen);
        }

        let mut decisions = Vec::with_capacity(sorted.len());
        let mut over_tolerance = Vec::new();
        for (region, (tier, reason)) in sorted.iter().zip(&chosen) {
            if let Some(entry) = self.state.get_mut(&region.id) {
                entry.tier = *tier;
            }
            if region.error[tier.index()] > self.tolerance {
                over_tolerance.push(region.id);
            }
            decisions.push(Decision { id: region.id, tier: *tier, reason: *reason });
        }
        Plan { decisions, total_cost_ms: total, budget_ms, overloaded, over_tolerance }
    }
}

/// Rolling window of measured step costs, for p50, p95 and maximum. A mean hides the
/// tail that a frame-by-frame game actually feels.
#[derive(Clone, Debug)]
pub struct CostWindow {
    samples: VecDeque<f64>,
    capacity: usize,
}

impl CostWindow {
    pub fn new(capacity: usize) -> Self {
        Self { samples: VecDeque::with_capacity(capacity), capacity: capacity.max(1) }
    }

    pub fn push(&mut self, milliseconds: f64) {
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return;
        }
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(milliseconds);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Nearest-rank percentile, `fraction` in 0..=1.
    pub fn percentile(&self, fraction: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let rank = ((sorted.len() as f64 * fraction.clamp(0.0, 1.0)).ceil() as usize).clamp(1, sorted.len());
        Some(sorted[rank - 1])
    }

    pub fn p50(&self) -> Option<f64> {
        self.percentile(0.5)
    }

    pub fn p95(&self) -> Option<f64> {
        self.percentile(0.95)
    }

    pub fn max(&self) -> Option<f64> {
        self.samples.iter().copied().reduce(f64::max)
    }
}

/// How the step budget is split between the systems, summing to 1. Starting values from
/// the plan; the real split is measured and adjusted.
pub const DEFAULT_SHARES: [(&str, f64); 4] = [
    ("mechanics", 0.5),
    ("heat_and_chemistry", 0.2),
    ("life", 0.15),
    ("bookkeeping", 0.15),
];

/// Milliseconds each system may use out of `budget_ms`.
pub fn split_budget(budget_ms: f64) -> Vec<(&'static str, f64)> {
    DEFAULT_SHARES.iter().map(|(name, share)| (*name, budget_ms * share)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(id: u64, importance: f64, cost: [f64; 4], error: [f64; 4]) -> RegionEstimate {
        RegionEstimate { id, importance, min_tier: Tier::Sleeping, max_tier: Tier::Cells, cost_ms: cost, error }
    }

    /// Typical content: sleeping is free but wrong when anything happens, cells exact but dear.
    fn typical(id: u64, importance: f64) -> RegionEstimate {
        region(id, importance, [0.0, 0.002, 0.03, 0.5], [0.9, 0.2, 0.05, 0.0])
    }

    #[test]
    fn picks_the_cheapest_tier_within_the_tolerance() {
        let mut governor = Governor::new(0.10, 0.0, 5);
        let plan = governor.decide(&[typical(1, 1.0)], 2.0);
        assert_eq!(plan.decisions[0].tier, Tier::Beam, "beam is the cheapest with error <= 10%");
        assert!(plan.over_tolerance.is_empty());
        assert!(!plan.overloaded);
    }

    #[test]
    fn overload_demotes_the_least_important_first_and_fits_the_budget() {
        let mut governor = Governor::new(0.10, 0.0, 5);
        // Three regions that each want cells (beam error too high), 0.5 ms each.
        let wants_cells = |id, importance| region(id, importance, [0.0, 0.002, 0.03, 0.5], [0.9, 0.5, 0.3, 0.0]);
        let plan = governor.decide(
            &[wants_cells(1, 5.0), wants_cells(2, 1.0), wants_cells(3, 3.0)],
            1.1,
        );
        let tier = |id| plan.decisions.iter().find(|d| d.id == id).unwrap().tier;
        assert_eq!(tier(1), Tier::Cells, "the most important keeps its cells");
        assert!(tier(2) < Tier::Cells, "the least important is demoted");
        assert!(plan.total_cost_ms <= 1.1 + 1e-12, "cost {} over the budget", plan.total_cost_ms);
        assert!(plan.over_tolerance.contains(&2), "a demoted region reports its shortfall");
        assert!(!plan.overloaded);
    }

    #[test]
    fn pinned_and_minimum_tier_regions_are_never_demoted_and_time_is_not_skipped() {
        let mut governor = Governor::default();
        let mut crack = typical(1, f64::INFINITY);
        crack.min_tier = Tier::Cells;
        let mut contact = typical(2, 1.0);
        contact.min_tier = Tier::Cells;
        // Two live cracks cost 1.0 ms against a 0.4 ms budget: overloaded, not dropped.
        let plan = governor.decide(&[crack, contact], 0.4);
        assert!(plan.overloaded);
        assert_eq!(plan.decisions.len(), 2, "every region still gets a decision");
        assert!(plan.decisions.iter().all(|d| d.tier == Tier::Cells));
    }

    #[test]
    fn leftover_budget_buys_fidelity_for_the_important_region() {
        let mut governor = Governor::new(0.10, 1.0, 5);
        let plan = governor.decide(&[typical(1, 100.0), typical(2, 0.1)], 0.6);
        let tier = |id| plan.decisions.iter().find(|d| d.id == id).unwrap().tier;
        assert_eq!(tier(1), Tier::Cells, "important region promoted with the spare 0.5 ms");
        assert_eq!(tier(2), Tier::Beam);
        assert!(plan.total_cost_ms <= 0.6 + 1e-12);
        assert_eq!(plan.decisions.iter().find(|d| d.id == 1).unwrap().reason, Reason::Promoted);
    }

    #[test]
    fn a_region_that_needs_more_detail_is_raised_at_once_but_lowered_only_after_a_hold() {
        let mut governor = Governor::new(0.10, 0.0, 3);
        let calm = region(1, 1.0, [0.0, 0.002, 0.03, 0.5], [0.9, 0.05, 0.02, 0.0]);
        let stressed = region(1, 1.0, [0.0, 0.002, 0.03, 0.5], [0.9, 0.5, 0.3, 0.0]);
        assert_eq!(governor.decide(&[calm.clone()], 2.0).decisions[0].tier, Tier::Rigid);
        assert_eq!(governor.decide(&[stressed.clone()], 2.0).decisions[0].tier, Tier::Cells, "raised at once");
        for step in 1..=3 {
            let tier = governor.decide(&[calm.clone()], 2.0).decisions[0].tier;
            if step < 3 {
                assert_eq!(tier, Tier::Cells, "held on step {step}");
            } else {
                assert_eq!(tier, Tier::Rigid, "lowered after the hold");
            }
        }
    }

    #[test]
    fn alternating_estimates_do_not_make_the_tier_thrash() {
        let mut governor = Governor::new(0.10, 1.0, 5);
        let calm = region(1, 10.0, [0.0, 0.002, 0.03, 0.5], [0.9, 0.05, 0.02, 0.0]);
        let stressed = region(1, 10.0, [0.0, 0.002, 0.03, 0.5], [0.9, 0.5, 0.3, 0.0]);
        let mut changes = 0;
        let mut last = None;
        for step in 0..60 {
            let estimate = if step % 2 == 0 { &calm } else { &stressed };
            let tier = governor.decide(&[estimate.clone()], 2.0).decisions[0].tier;
            if last.is_some_and(|previous| previous != tier) {
                changes += 1;
            }
            last = Some(tier);
        }
        assert!(changes <= 2, "tier changed {changes} times in 60 alternating steps");
    }

    #[test]
    fn the_plan_does_not_depend_on_the_order_of_the_regions() {
        let regions: Vec<RegionEstimate> = (1..=6).map(|id| typical(id, f64::from(id as u32))).collect();
        let mut reversed = regions.clone();
        reversed.reverse();
        let a = Governor::default().decide(&regions, 0.2);
        let b = Governor::default().decide(&reversed, 0.2);
        let key = |plan: &Plan| plan.decisions.iter().map(|d| (d.id, d.tier)).collect::<Vec<_>>();
        assert_eq!(key(&a), key(&b));
    }

    #[test]
    fn random_scenes_fit_the_budget_whenever_the_minimum_tiers_do() {
        let mut state = 12345u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64
        };
        for _ in 0..200 {
            let count = 1 + (random() * 20.0) as usize;
            let regions: Vec<RegionEstimate> = (0..count as u64)
                .map(|id| {
                    let mut r = typical(id, random() * 10.0);
                    r.cost_ms[3] = 0.1 + random();
                    r
                })
                .collect();
            let budget = 0.02 + random() * 2.0;
            let minimum: f64 = regions.iter().map(|r| r.cost_ms[r.min_tier.index()]).sum();
            let plan = Governor::default().decide(&regions, budget);
            if minimum <= budget {
                assert!(plan.total_cost_ms <= budget + 1e-9, "cost {} over {budget}", plan.total_cost_ms);
                assert!(!plan.overloaded);
            }
            assert_eq!(plan.decisions.len(), regions.len());
        }
    }

    #[test]
    fn cost_window_reports_the_tail_not_the_mean() {
        let mut window = CostWindow::new(100);
        for _ in 0..95 {
            window.push(0.8);
        }
        for _ in 0..5 {
            window.push(4.0);
        }
        assert_eq!(window.p50(), Some(0.8));
        assert_eq!(window.p95(), Some(0.8), "nearest rank puts p95 at the 95th of 100 samples");
        assert_eq!(window.max(), Some(4.0));
        window.push(f64::NAN);
        window.push(-1.0);
        assert_eq!(window.len(), 100, "non-finite and negative samples are ignored");
    }

    #[test]
    fn the_default_split_adds_up_to_the_budget() {
        let parts = split_budget(2.0);
        let sum: f64 = parts.iter().map(|(_, ms)| ms).sum();
        assert!((sum - 2.0).abs() < 1e-12);
        assert_eq!(parts[0].0, "mechanics");
    }
}
