//! Sleeping tier (T0): bodies at rest cost nothing until something disturbs them.
//!
//! A bonded cluster whose free REVs have stayed at rest for several consecutive checks,
//! carry no applied force and hold only modest bond stress, is put to sleep: its free REVs
//! are made kinematic (positions, strain and energy stay exactly as they are, so nothing is
//! lost or invented) and the solver skips them. They wake, all together, when
//!
//! * an applied force or acceleration appears on a member,
//! * an awake REV (or a moving fixed one) comes within the wake margin, which grows with
//!   its speed so a falling body wakes the sleeper before it touches it,
//! * gravity changes, or the REV count changes (removal, spawn, compaction).
//!
//! Limits, kept visible on purpose: a sleeping body does not feel a *resting* neighbour
//! that is added or changed without moving; temperature keeps evolving but its thermal
//! stress is not re-solved while asleep; a creature that actuates through a channel other
//! than `external_forces` must not be left to sleep (keep sleep disabled for such scenes).
use super::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Edge of the coarse grid used to find sleepers near a moving REV, in cells.
const GRID_CELLS: f64 = 4.0;

static NEXT_LIVE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The REVs a sparse step has to visit while clusters sleep. A sleeper is kinematic
/// and frozen, and so is a fixed REV that nobody touches and that does not move: their
/// position, velocity and every solver entry derived from them stay as they are, so a
/// pass that would only copy or zero those entries can skip them without changing any
/// result. What a step must still visit is
///
/// * `list` (mechanics): every free REV, every fixed or sleeping REV that shares a bond
///   with a free one or with the other side of a sleep boundary (the "rim", whose bonds
///   the solver reads), every REV of a thermally unsettled sleeper, and every fixed REV
///   that moved since the last step;
/// * `thermal_list` (heat, melting, bounds): additionally every fixed REV, whose
///   energy can change by conduction or external heating without any mechanical sign.
pub(crate) struct Live {
    /// Distinguishes lists: caches built for one list are not trusted for another.
    pub(crate) id: u64,
    /// Ascending REV indices for the mechanical passes.
    pub(crate) list: Vec<usize>,
    /// Ascending REV indices for the thermal passes (a superset of `list`).
    pub(crate) thermal_list: Vec<usize>,
    /// Fixed, unsleeping REVs outside `list`: checked each step for movement.
    pub(crate) fixed: Vec<usize>,
    /// REV `i` is asleep (it may still be on the lists as rim).
    pub(crate) asleep: Vec<bool>,
    /// REV `i` is in `list`.
    pub(crate) in_list: Vec<bool>,
    /// REV `i` is in `thermal_list`.
    pub(crate) thermal_in_list: Vec<bool>,
    /// REV `i` sleeps in a region whose heat is settled: nothing flows through it, so
    /// the thermal pass need not look inside, and heat arriving at it ends the shortcut.
    pub(crate) cold: Vec<bool>,
}

/// Indices a pass visits: all REVs, or only the live ones during a sparse step.
pub(crate) enum Visit<'a> {
    All(std::ops::Range<usize>),
    Some(std::iter::Copied<std::slice::Iter<'a, usize>>),
}

impl Iterator for Visit<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            Visit::All(range) => range.next(),
            Visit::Some(iter) => iter.next(),
        }
    }
}

/// Walk `0..n`, or the live list when a sparse step is running.
#[inline]
pub(crate) fn visit<'a>(live: &'a Option<Arc<Live>>, n: usize) -> Visit<'a> {
    match live {
        Some(live) => Visit::Some(live.list.iter().copied()),
        None => Visit::All(0..n),
    }
}

/// Like `visit`, over the thermal list.
#[inline]
pub(crate) fn visit_thermal<'a>(live: &'a Option<Arc<Live>>, n: usize) -> Visit<'a> {
    match live {
        Some(live) => Visit::Some(live.thermal_list.iter().copied()),
        None => Visit::All(0..n),
    }
}

impl Live {
    /// The same lists with fixed REVs that moved this step added to the mechanical one.
    fn with_moving(&self, moving: &[usize]) -> Live {
        let mut in_list = self.in_list.clone();
        for &i in moving {
            in_list[i] = true;
        }
        let list = (0..in_list.len()).filter(|&i| in_list[i]).collect();
        Live {
            id: NEXT_LIVE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            list,
            thermal_list: self.thermal_list.clone(),
            fixed: self.fixed.iter().copied().filter(|i| !moving.contains(i)).collect(),
            asleep: self.asleep.clone(),
            in_list,
            thermal_in_list: self.thermal_in_list.clone(),
            cold: self.cold.clone(),
        }
    }
}

/// `dst[i] = src[i]` for every visited index. A destination of the wrong length is
/// rebuilt whole, so entries of REVs outside the live list always mirror the source.
pub(crate) fn sync_visited<T: Clone>(dst: &mut Vec<T>, src: &[T], live: &Option<Arc<Live>>) {
    match live {
        Some(live) if dst.len() == src.len() && src.len() == live.asleep.len() => {
            for &i in &live.list {
                dst[i].clone_from(&src[i]);
            }
        }
        _ => {
            dst.clear();
            dst.extend_from_slice(src);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SleepConfig {
    /// A REV is at rest below this speed, metres per second.
    pub rest_speed_m_s: f64,
    /// ... and below this angular speed, radians per second.
    pub rest_spin_rad_s: f64,
    /// Consecutive checks a cluster must be at rest before it sleeps.
    pub rest_rounds: u32,
    /// Fixed steps between checks.
    pub interval_steps: u32,
    /// Distance, in cells, at which an awake REV wakes a sleeper (plus twice the distance
    /// it travels in one step).
    pub wake_margin_cells: f64,
    /// Bonds loaded above this fraction of their limit keep their cluster awake.
    pub stress_ratio_limit: f32,
    /// Awake REVs slower than this do not wake neighbours.
    pub mover_speed_m_s: f64,
    /// Let a step that has sleepers visit only the REVs that are awake (plus the bonded
    /// rim of the sleepers), instead of every REV. Results do not change; turn it off to
    /// compare against the full pass.
    pub sparse_passes: bool,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            rest_speed_m_s: 1.0e-4,
            rest_spin_rad_s: 1.0e-2,
            rest_rounds: 3,
            interval_steps: 10,
            wake_margin_cells: 1.5,
            stress_ratio_limit: 0.3,
            mover_speed_m_s: 1.0e-3,
            sparse_passes: true,
        }
    }
}

struct SleepRegion {
    members: Vec<usize>,
    /// Heat was still moving in or around the cluster when it fell asleep (an uneven
    /// field, a gradient across one of its bonds, a molten member). Its REVs stay on
    /// the visit list, so conduction and melting go on as they do for awake REVs; the
    /// change of energy then wakes it at the next step, as always.
    unsettled: bool,
    /// Internal energy and mass of each member when it fell asleep, as bits. A change
    /// means heat, a reaction or a composition change reached the sleeper.
    signature: Vec<(u64, u32)>,
}

pub(crate) struct SleepState {
    config: SleepConfig,
    regions: Vec<Option<SleepRegion>>,
    /// Consecutive quiet checks per cluster, keyed by (lowest member, size).
    quiet: HashMap<(usize, usize), u32>,
    /// Coarse grid cell -> regions with a member in it.
    grid: HashMap<[i32; 3], Vec<usize>>,
    grid_dirty: bool,
    since: u32,
    voxel_count: usize,
    gravity: [f32; 3],
    /// Visit list for sparse passes; `None` while nothing sleeps.
    live: Option<Arc<Live>>,
}

fn grid_key(p: DVec3) -> [i32; 3] {
    let g = GRID_CELLS * f64::from(CELL);
    [(p.x / g).floor() as i32, (p.y / g).floor() as i32, (p.z / g).floor() as i32]
}

impl Scene {
    /// Let settled clusters sleep. Off by default.
    pub fn enable_sleep(&mut self, config: SleepConfig) -> Result<(), String> {
        if !(config.rest_speed_m_s > 0.0) || config.interval_steps == 0 || config.rest_rounds == 0 {
            return Err("sleep needs a positive rest speed, interval and round count".into());
        }
        self.wake_all();
        self.sleep = Some(Box::new(SleepState {
            config,
            regions: Vec::new(),
            quiet: HashMap::new(),
            grid: HashMap::new(),
            grid_dirty: false,
            since: 0,
            voxel_count: self.voxels.len(),
            gravity: self.config.gravity,
            live: None,
        }));
        Ok(())
    }

    /// Stop sleeping; every sleeper wakes.
    pub fn disable_sleep(&mut self) {
        self.wake_all();
        self.sleep = None;
    }

    /// Wake every sleeping cluster.
    pub fn wake_all(&mut self) {
        let Some(mut state) = self.sleep.take() else {
            return;
        };
        let all: Vec<usize> = (0..state.regions.len()).collect();
        self.wake_regions(&mut state, &all);
        self.sleep = Some(state);
    }

    /// Wake the sleeping clusters that contain any of `voxels`. Public mutators call this
    /// before touching a REV, so a sleeper never looks fixed to them.
    pub fn wake_voxels(&mut self, voxels: &[usize]) {
        let Some(mut state) = self.sleep.take() else {
            return;
        };
        let targets: std::collections::HashSet<usize> = voxels.iter().copied().collect();
        let ids: Vec<usize> = state
            .regions
            .iter()
            .enumerate()
            .filter_map(|(id, region)| {
                region.as_ref().filter(|r| r.members.iter().any(|m| targets.contains(m))).map(|_| id)
            })
            .collect();
        if !ids.is_empty() {
            self.wake_regions(&mut state, &ids);
        }
        self.sleep = Some(state);
    }

    /// True when nothing mechanical is awake: every REV is fixed or asleep, no rigid
    /// aggregate or beam region exists. Such a step only conducts heat and runs chemistry.
    pub(crate) fn mechanically_dormant(&self) -> bool {
        self.sleep.as_ref().is_some_and(|state| state.regions.iter().any(Option::is_some))
            && !self.has_registered_rigid_candidates()
            && self.beam_regions.iter().all(Option::is_none)
            && match self.sparse_live() {
                // Sleepers are kinematic by construction; only the rest needs a look.
                Some(live) => live.list.iter().all(|&i| self.voxels[i].kinematic),
                None => self.voxels.iter().all(|v| v.kinematic),
            }
    }

    /// The visit list for sparse passes, when sleepers exist and sparse passes are on.
    pub(crate) fn sparse_live(&self) -> Option<Arc<Live>> {
        let state = self.sleep.as_ref()?;
        if !state.config.sparse_passes {
            return None;
        }
        state.live.as_ref().filter(|live| live.asleep.len() == self.voxels.len()).cloned()
    }

    /// The visit list for this step: the standing one plus the fixed REVs that moved
    /// since the last step (a platform, a hand), which touch whatever they now overlap.
    pub(crate) fn sparse_live_for_step(&self) -> Option<Arc<Live>> {
        let live = self.sparse_live()?;
        let moving: Vec<usize> = live
            .fixed
            .iter()
            .copied()
            .filter(|&i| self.positions[i] != self.previous[i])
            .collect();
        Some(if moving.is_empty() { live } else { Arc::new(live.with_moving(&moving)) })
    }

    /// Recompute the visit list after the bond graph changed.
    pub(crate) fn refresh_live(&mut self) {
        if let Some(mut state) = self.sleep.take() {
            state.live = self.compute_live(&state.regions);
            self.sleep = Some(state);
        }
    }

    fn compute_live(&self, regions: &[Option<SleepRegion>]) -> Option<Arc<Live>> {
        if !regions.iter().any(Option::is_some) {
            return None;
        }
        let n = self.voxels.len();
        let mut asleep = vec![false; n];
        for region in regions.iter().flatten() {
            for &i in &region.members {
                if i < n {
                    asleep[i] = true;
                }
            }
        }
        let mut cold = asleep.clone();
        let mut touch = vec![false; n];
        for region in regions.iter().flatten().filter(|region| region.unsettled) {
            for &i in &region.members {
                if i < n {
                    touch[i] = true;
                    cold[i] = false;
                }
            }
        }
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY {
                    continue;
                }
                let j = bond.neighbor as usize;
                if j < n
                    && (asleep[i] != asleep[j] || self.voxels[i].kinematic != self.voxels[j].kinematic)
                {
                    touch[i] = true;
                    touch[j] = true;
                }
            }
        }
        let in_list: Vec<bool> = (0..n).map(|i| !self.voxels[i].kinematic || touch[i]).collect();
        let thermal_in_list: Vec<bool> = (0..n).map(|i| !asleep[i] || touch[i]).collect();
        let list = (0..n).filter(|&i| in_list[i]).collect();
        let thermal_list = (0..n).filter(|&i| thermal_in_list[i]).collect();
        let fixed = (0..n).filter(|&i| thermal_in_list[i] && !in_list[i]).collect();
        Some(Arc::new(Live {
            id: NEXT_LIVE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            list,
            thermal_list,
            fixed,
            asleep,
            in_list,
            thermal_in_list,
            cold,
        }))
    }

    /// Free REVs currently asleep.
    pub fn sleeping_rev_count(&self) -> usize {
        self.sleep.as_ref().map_or(0, |state| state.regions.iter().flatten().map(|r| r.members.len()).sum())
    }

    /// Per-REV sleeping flags, for the tier census and for checkpoints.
    pub(crate) fn sleeping_flags(&self) -> Vec<bool> {
        let mut flags = vec![false; self.voxels.len()];
        if let Some(state) = &self.sleep {
            for region in state.regions.iter().flatten() {
                for &i in &region.members {
                    if i < flags.len() {
                        flags[i] = true;
                    }
                }
            }
        }
        flags
    }

    fn wake_regions(&mut self, state: &mut SleepState, ids: &[usize]) {
        let mut members = Vec::new();
        for &id in ids {
            if let Some(region) = state.regions.get_mut(id).and_then(Option::take) {
                members.extend(region.members);
            }
        }
        if members.is_empty() {
            return;
        }
        members.retain(|&i| i < self.voxels.len());
        let _ = self.set_voxels_kinematic_raw(&members, false);
        state.grid_dirty = true;
        state.quiet.clear();
        state.live = self.compute_live(&state.regions);
    }

    /// Called at the start of every fixed step.
    pub(crate) fn run_sleep(&mut self) {
        let Some(mut state) = self.sleep.take() else {
            return;
        };
        if self.voxels.len() != state.voxel_count || self.config.gravity != state.gravity {
            let all: Vec<usize> = (0..state.regions.len()).collect();
            self.wake_regions(&mut state, &all);
            state.voxel_count = self.voxels.len();
            state.gravity = self.config.gravity;
        }
        if state.regions.iter().any(Option::is_some) && state.live.is_none() {
            state.live = self.compute_live(&state.regions);
        }
        if state.regions.iter().any(Option::is_some) {
            self.wake_disturbed(&mut state);
        }
        state.since += 1;
        if state.since >= state.config.interval_steps {
            state.since = 0;
            self.put_quiet_clusters_to_sleep(&mut state);
        }
        if state.grid_dirty {
            state.grid.clear();
            for (id, region) in state.regions.iter().enumerate() {
                let Some(region) = region else { continue };
                for &i in &region.members {
                    let entry = state.grid.entry(grid_key(self.positions[i])).or_default();
                    if entry.last() != Some(&id) {
                        entry.push(id);
                    }
                }
            }
            state.grid_dirty = false;
        }
        self.sleep = Some(state);
    }

    fn wake_disturbed(&mut self, state: &mut SleepState) {
        let mut wake: Vec<usize> = Vec::new();
        // Applied forces and accelerations on a sleeper.
        for (id, region) in state.regions.iter().enumerate() {
            let Some(region) = region else { continue };
            let changed = region.members.iter().zip(&region.signature).any(|(&i, &(energy, mass))| {
                self.voxels
                    .get(i)
                    .map_or(true, |v| v.total_internal_energy_j().to_bits() != energy || v.mass.to_bits() != mass)
            });
            if changed {
                wake.push(id);
                continue;
            }
            let loaded = region.members.iter().any(|&i| {
                self.external_forces.get(i).is_some_and(|f| *f != Vec3::ZERO)
                    || self.external_accelerations.get(i).is_some_and(|a| *a != Vec3::ZERO)
            });
            if loaded {
                wake.push(id);
            }
        }
        // Movers: awake free REVs, and fixed REVs that moved since the last step.
        let dt = f64::from(self.config.fixed_dt);
        let cell = f64::from(CELL);
        let live = state.live.clone();
        let asleep = if live.is_some() { Vec::new() } else { self.sleeping_flags() };
        for i in visit_thermal(&live, self.voxels.len()) {
            if live.as_ref().map_or_else(|| asleep[i], |live| live.asleep[i]) {
                continue;
            }
            let speed = if self.voxels[i].kinematic {
                (self.positions[i] - self.previous[i]).length() / dt.max(1.0e-12)
            } else {
                f64::from(self.voxels[i].velocity.length())
            };
            if speed < state.config.mover_speed_m_s {
                continue;
            }
            let margin = state.config.wake_margin_cells * cell + 2.0 * speed * dt + cell;
            let p = self.positions[i];
            let (lo, hi) = (grid_key(p - DVec3::splat(margin)), grid_key(p + DVec3::splat(margin)));
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        if let Some(ids) = state.grid.get(&[x, y, z]) {
                            wake.extend_from_slice(ids);
                        }
                    }
                }
            }
        }
        if !wake.is_empty() {
            wake.sort_unstable();
            wake.dedup();
            self.wake_regions(state, &wake);
        }
    }

    fn put_quiet_clusters_to_sleep(&mut self, state: &mut SleepState) {
        if self.clusters.len() != self.voxels.len() || !(self.solver_residual < 1.0e-3) {
            state.quiet.clear();
            return;
        }
        let mut in_beam = vec![false; self.voxels.len()];
        for region in self.beam_regions.iter().flatten() {
            for &i in region.members() {
                in_beam[i] = true;
            }
        }
        let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
        for i in visit(&state.live, self.voxels.len()) {
            if self.voxels[i].kinematic || in_beam[i] || self.aggregate_of[i].is_some() {
                continue;
            }
            groups.entry(self.clusters[i]).or_default().push(i);
        }
        let cfg = state.config.clone();
        let mut seen: HashMap<(usize, usize), u32> = HashMap::new();
        let mut sleep_now: Vec<Vec<usize>> = Vec::new();
        for (_, members) in groups {
            let quiet = members.iter().all(|&i| {
                f64::from(self.voxels[i].velocity.length()) < cfg.rest_speed_m_s
                    && self.angular_velocities.get(i).map_or(true, |w| w.length() < cfg.rest_spin_rad_s)
                    && self.external_forces.get(i).map_or(true, |f| *f == Vec3::ZERO)
                    && self.external_accelerations.get(i).map_or(true, |a| *a == Vec3::ZERO)
                    && (0..3).all(|axis| {
                        self.bonds[axis].get(i).map_or(true, |bond| {
                            bond.previous_stress_ratio.abs() < cfg.stress_ratio_limit && bond.damage < 1.0e-6
                        })
                    })
            });
            let key = (*members.iter().min().unwrap(), members.len());
            if !quiet {
                continue;
            }
            let rounds = state.quiet.get(&key).copied().unwrap_or(0) + 1;
            if rounds >= cfg.rest_rounds {
                sleep_now.push(members);
            } else {
                seen.insert(key, rounds);
            }
        }
        state.quiet = seen;
        if sleep_now.is_empty() {
            return;
        }
        let unsettled = self.thermally_unsettled(&sleep_now);
        let mut all = Vec::new();
        for (members, unsettled) in sleep_now.into_iter().zip(unsettled) {
            all.extend_from_slice(&members);
            let slot = state.regions.iter().position(Option::is_none).unwrap_or(state.regions.len());
            let signature = members
                .iter()
                .map(|&i| (self.voxels[i].total_internal_energy_j().to_bits(), self.voxels[i].mass.to_bits()))
                .collect();
            let region = SleepRegion { members, unsettled, signature };
            if slot == state.regions.len() {
                state.regions.push(Some(region));
            } else {
                state.regions[slot] = Some(region);
            }
        }
        let _ = self.set_voxels_kinematic_raw(&all, true);
        state.grid_dirty = true;
        state.live = self.compute_live(&state.regions);
    }

    /// Per cluster: is heat still moving in or around it? True when a member is fully
    /// molten, keeps an uneven field inside the REV, or has a bond to a REV at another
    /// temperature (the test that seeds conduction). The sparse thermal pass only looks
    /// at the bonds around REVs on the visit list, so such a cluster stays on it.
    fn thermally_unsettled(&self, groups: &[Vec<usize>]) -> Vec<bool> {
        let n = self.voxels.len();
        let mut group_of = vec![usize::MAX; n];
        let mut unsettled = vec![false; groups.len()];
        for (g, members) in groups.iter().enumerate() {
            for &i in members {
                group_of[i] = g;
                let voxel = &self.voxels[i];
                if voxel.thermal_cell_energy_j.is_some()
                    || self.properties[i].phase_state(voxel.mass, voxel.thermal_energy_f32()).solid_fraction <= 0.0
                {
                    unsettled[g] = true;
                }
            }
        }
        let temperature = |i: usize| {
            self.properties[i].temperature(self.voxels[i].mass, self.voxels[i].thermal_energy_f32())
        };
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                if bond.neighbor == EMPTY {
                    continue;
                }
                let j = bond.neighbor as usize;
                let (a, b) = (group_of[i], group_of[j]);
                if a == usize::MAX && b == usize::MAX {
                    continue;
                }
                if bond.damage >= 1.0
                    || self.properties[i].thermal_cond <= 0.0
                    || self.properties[j].thermal_cond <= 0.0
                {
                    continue;
                }
                if temperature(i) != temperature(j) {
                    for g in [a, b] {
                        if g != usize::MAX {
                            unsettled[g] = true;
                        }
                    }
                }
            }
        }
        unsettled
    }

    /// Voxels as a checkpoint must store them: sleepers are free, not fixed.
    pub(crate) fn voxels_with_sleepers_awake(&self) -> Vec<Voxel> {
        let mut voxels = self.voxels.clone();
        if let Some(state) = &self.sleep {
            for region in state.regions.iter().flatten() {
                for &i in &region.members {
                    if i < voxels.len() {
                        voxels[i].kinematic = false;
                    }
                }
            }
        }
        voxels
    }
}
