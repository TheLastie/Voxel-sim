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

/// Edge of the coarse grid used to find sleepers near a moving REV, in cells.
const GRID_CELLS: f64 = 4.0;

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
        }
    }
}

struct SleepRegion {
    members: Vec<usize>,
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
            && self.voxels.iter().all(|v| v.kinematic)
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
        let asleep = self.sleeping_flags();
        for i in 0..self.voxels.len() {
            if asleep[i] {
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
        for i in 0..self.voxels.len() {
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
        let mut all = Vec::new();
        for members in sleep_now {
            all.extend_from_slice(&members);
            let slot = state.regions.iter().position(Option::is_none).unwrap_or(state.regions.len());
            let signature = members
                .iter()
                .map(|&i| (self.voxels[i].total_internal_energy_j().to_bits(), self.voxels[i].mass.to_bits()))
                .collect();
            let region = SleepRegion { members, signature };
            if slot == state.regions.len() {
                state.regions.push(Some(region));
            } else {
                state.regions[slot] = Some(region);
            }
        }
        let _ = self.set_voxels_kinematic_raw(&all, true);
        state.grid_dirty = true;
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
