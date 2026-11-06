//! Functional organ graph of a creature (`WORLD_PARADIGM.md` 2.4.1 and 2.4.2).
//!
//! Nodes (brain, digestion, respiration, stores, muscles, sense organs) are
//! *views* of existing material REVs: a node owns a set of stable REV ids
//! (`Scene::rev_id`), not copies of mass or energy. Mass and internal energy
//! are always read from the scene, and [`OrganGraph::owned_mass_kg`] counts a
//! REV once even when several nodes share it. Typed channels (signal, nutrient,
//! mechanical) connect nodes. A channel carries a capacity, a delay per link and
//! an availability that is *recomputed from the tissue*: the best route through
//! the bonded REVs of the body between the two nodes, where each bond has a
//! health `1 - damage` (zero when broken or when a REV is gone) and the route
//! is as healthy as its weakest bond (widest path, fewest links on ties).
//! A logical channel never adds stiffness; it only reads the bonds.
//!
//! Damage rules implemented here:
//! - a cut or damaged path lowers the channel's throughput (a detour through
//!   intact tissue is used when one exists, and lengthens the delay);
//! - brain damage scales the controller's command ([`OrganGraph::gate_command`]);
//! - sense-organ damage degrades the sensor inputs ([`OrganGraph::degrade_sensors`]);
//! - damage to a muscle or digestive organ limits its output
//!   ([`OrganGraph::strength_cap`], [`OrganGraph::digestion_throughput`]).
//! - losing the brain removes control, never the body from the scene.
//!
//! The coefficients here (the sensor detection threshold, default capacities)
//! are modelling choices that are not calibrated against any real animal.

use super::{ControlCommand, SensorReading};
use crate::{bond::EMPTY, scene::Scene};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
};

/// Stable id of an organ node; never reused inside one graph.
pub type OrganId = u32;
/// Stable id of a channel; never reused inside one graph.
pub type ChannelId = u32;

/// A sense organ reports contact only while its signal strength is at least this.
pub const SENSE_DETECTION_THRESHOLD: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganRole {
    Brain,
    Digestion,
    Respiration,
    Store,
    Muscle,
    SenseOrgan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelKind {
    /// Nerve signal. Capacity unit: messages per second.
    Signal,
    /// Transport of nutrients between tissues. Capacity unit: chemical watts.
    Nutrient,
    /// Force transmission through tissue. Capacity unit: newtons.
    Mechanical,
}

/// What a sense organ senses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SenseScope {
    /// Contact and load at these feet (indices into the sensor layout's feet).
    Feet(Vec<usize>),
    /// Orientation, joint angles, velocities and pose of the whole body.
    Body,
}

/// Description of a node to add.
#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub role: OrganRole,
    pub label: String,
    /// Current scene indices of the REVs that host this organ. Converted to
    /// stable ids; the indices are not kept.
    pub revs: Vec<usize>,
    /// For [`OrganRole::Muscle`]: slot of the creature actuator it moves.
    pub actuator_slot: Option<usize>,
    /// For [`OrganRole::SenseOrgan`]: what it senses.
    pub scope: Option<SenseScope>,
}

impl NodeSpec {
    pub fn new(role: OrganRole, label: &str, revs: Vec<usize>) -> Self {
        Self { role, label: label.to_owned(), revs, actuator_slot: None, scope: None }
    }
    pub fn muscle(label: &str, revs: Vec<usize>, actuator_slot: usize) -> Self {
        Self { actuator_slot: Some(actuator_slot), ..Self::new(OrganRole::Muscle, label, revs) }
    }
    pub fn sense(label: &str, revs: Vec<usize>, scope: SenseScope) -> Self {
        Self { scope: Some(scope), ..Self::new(OrganRole::SenseOrgan, label, revs) }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrganNode {
    pub id: OrganId,
    pub role: OrganRole,
    pub label: String,
    /// Stable REV ids hosting the organ, ascending. A removed REV stays listed
    /// (it is the organ's missing tissue) and reads as absent.
    tissue: Vec<u64>,
    /// Mass of the tissue when the node was added (kg).
    reference_mass_kg: f64,
    pub actuator_slot: Option<usize>,
    pub scope: Option<SenseScope>,
    /// Functional integrity in `0..=1` at the last refresh.
    integrity: f64,
}

impl OrganNode {
    pub fn tissue_ids(&self) -> &[u64] {
        &self.tissue
    }
    pub fn reference_mass_kg(&self) -> f64 {
        self.reference_mass_kg
    }
    /// Functional integrity at the last [`OrganGraph::refresh`]: surviving mass
    /// over the original mass, times the mean health of the bonds inside the
    /// tissue (of all its bonds for a one-REV organ).
    pub fn integrity(&self) -> f64 {
        self.integrity
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    pub id: ChannelId,
    pub kind: ChannelKind,
    pub from: OrganId,
    pub to: OrganId,
    /// Capacity of the intact path, in the unit of its kind.
    pub capacity: f64,
    /// Delay added by each link (REV to REV) of the route, in seconds.
    pub delay_per_link_s: f64,
    availability: f64,
    /// Stable REV ids of the current best route (empty when cut).
    route: Vec<u64>,
}

impl Channel {
    /// Health of the best route in `0..=1` (0 when cut).
    pub fn availability(&self) -> f64 {
        self.availability
    }
    /// What the channel can carry now, in the unit of its kind.
    pub fn throughput(&self) -> f64 {
        self.capacity * self.availability
    }
    /// Delay of the current route (s); infinite when the channel is cut.
    pub fn delay_s(&self) -> f64 {
        if self.availability > 0.0 {
            self.delay_per_link_s * self.route.len().saturating_sub(1) as f64
        } else {
            f64::INFINITY
        }
    }
    pub fn route_ids(&self) -> &[u64] {
        &self.route
    }
}

/// Work done by graph refreshes, for the creature's cost report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrganCost {
    pub refreshes: u64,
    pub link_reads: u64,
    pub search_visits: u64,
}

/// Where the organs of a standard creature live. All lists are current scene
/// indices (they are turned into stable ids at build time).
#[derive(Clone, Debug, Default)]
pub struct OrganPlan {
    pub brain: Vec<usize>,
    pub stomach: Vec<usize>,
    pub store: Vec<usize>,
    /// Sense tissue of the feet (contact and load). Also its `foot` indices.
    pub foot_sense: Vec<usize>,
    pub foot_indices: Vec<usize>,
    /// Tissue that senses body posture and motion.
    pub body_sense: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrganGraph {
    /// Stable ids of every REV the channels may route through, ascending.
    body: Vec<u64>,
    nodes: Vec<OrganNode>,
    channels: Vec<Channel>,
    next_node: OrganId,
    next_channel: ChannelId,
    /// Scene REV count at the last refresh (a change forces the next refresh).
    #[serde(skip)]
    refreshed_count: Option<usize>,
    #[serde(skip)]
    cost: OrganCost,
}

struct Links {
    /// Scene index of each local REV.
    index: Vec<usize>,
    local_of_id: HashMap<u64, u32>,
    adj: Vec<Vec<(u32, f32)>>,
}

impl OrganGraph {
    /// A graph whose channels route through `body_revs` (current scene indices).
    pub fn new(scene: &Scene, body_revs: impl IntoIterator<Item = usize>) -> Result<Self, String> {
        let mut body = Vec::new();
        for i in body_revs {
            body.push(scene.rev_id(i).ok_or("body REV index outside the scene")?);
        }
        body.sort_unstable();
        body.dedup();
        if body.is_empty() {
            return Err("a body needs at least one REV".into());
        }
        Ok(Self {
            body,
            nodes: Vec::new(),
            channels: Vec::new(),
            next_node: 0,
            next_channel: 0,
            refreshed_count: None,
            cost: OrganCost::default(),
        })
    }

    pub fn nodes(&self) -> &[OrganNode] {
        &self.nodes
    }
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }
    pub fn node(&self, id: OrganId) -> Option<&OrganNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
    pub fn channel(&self, id: ChannelId) -> Option<&Channel> {
        self.channels.iter().find(|c| c.id == id)
    }
    pub fn body_ids(&self) -> &[u64] {
        &self.body
    }
    pub fn cost(&self) -> OrganCost {
        self.cost
    }
    pub fn first_node(&self, role: OrganRole) -> Option<&OrganNode> {
        self.nodes.iter().find(|n| n.role == role)
    }

    /// Add a node. Its tissue must be inside the body, present now, and not
    /// empty. Tissue may be shared with another node (a REV can host two
    /// functions); mass and energy are still counted once by
    /// [`owned_mass_kg`](Self::owned_mass_kg).
    pub fn add_node(&mut self, scene: &Scene, spec: NodeSpec) -> Result<OrganId, String> {
        if spec.revs.is_empty() {
            return Err("an organ needs at least one REV".into());
        }
        let mut tissue = Vec::with_capacity(spec.revs.len());
        for &i in &spec.revs {
            let id = scene.rev_id(i).ok_or("organ REV index outside the scene")?;
            if self.body.binary_search(&id).is_err() {
                return Err("organ tissue must belong to the body".into());
            }
            tissue.push(id);
        }
        tissue.sort_unstable();
        tissue.dedup();
        let mass: f64 = tissue
            .iter()
            .filter_map(|&id| scene.rev_index(id))
            .map(|i| f64::from(scene.voxels[i].mass))
            .sum();
        if !(mass.is_finite() && mass > 0.0) {
            return Err("organ tissue has no mass".into());
        }
        let id = self.next_node;
        self.next_node += 1;
        self.nodes.push(OrganNode {
            id,
            role: spec.role,
            label: spec.label,
            tissue,
            reference_mass_kg: mass,
            actuator_slot: spec.actuator_slot,
            scope: spec.scope,
            integrity: 1.0,
        });
        self.refreshed_count = None;
        Ok(id)
    }

    pub fn add_channel(
        &mut self,
        kind: ChannelKind,
        from: OrganId,
        to: OrganId,
        capacity: f64,
        delay_per_link_s: f64,
    ) -> Result<ChannelId, String> {
        if self.node(from).is_none() || self.node(to).is_none() {
            return Err("channel endpoint is not a node of this graph".into());
        }
        if !(capacity.is_finite() && capacity > 0.0) || !(delay_per_link_s.is_finite() && delay_per_link_s >= 0.0) {
            return Err("channel capacity must be positive and delay nonnegative".into());
        }
        let id = self.next_channel;
        self.next_channel += 1;
        self.channels.push(Channel {
            id,
            kind,
            from,
            to,
            capacity,
            delay_per_link_s,
            availability: 1.0,
            route: Vec::new(),
        });
        self.refreshed_count = None;
        Ok(id)
    }

    /// Standard anatomy: brain, stomach, store, sense organs and one muscle per
    /// actuator (`muscle_revs[k]` is actuator `k`'s dynamic tissue), with signal
    /// channels sense -> brain -> muscle, nutrient channels stomach -> store ->
    /// muscle, and a mechanical channel from each muscle to the foot sense tissue.
    /// Capacities are illustrative defaults.
    pub fn standard(
        scene: &Scene,
        body_revs: impl IntoIterator<Item = usize>,
        plan: &OrganPlan,
        muscle_revs: &[Vec<usize>],
        nutrient_capacity_w: f64,
        mechanical_capacity_n: f64,
    ) -> Result<Self, String> {
        let mut g = Self::new(scene, body_revs)?;
        let brain = g.add_node(scene, NodeSpec::new(OrganRole::Brain, "brain", plan.brain.clone()))?;
        let stomach = g.add_node(scene, NodeSpec::new(OrganRole::Digestion, "stomach", plan.stomach.clone()))?;
        let store = g.add_node(scene, NodeSpec::new(OrganRole::Store, "store", plan.store.clone()))?;
        let feet = if plan.foot_sense.is_empty() {
            None
        } else {
            Some(g.add_node(scene, NodeSpec::sense("feet", plan.foot_sense.clone(), SenseScope::Feet(plan.foot_indices.clone())))?)
        };
        let body = if plan.body_sense.is_empty() {
            None
        } else {
            Some(g.add_node(scene, NodeSpec::sense("balance", plan.body_sense.clone(), SenseScope::Body))?)
        };
        const SIGNAL_CAPACITY_HZ: f64 = 1000.0;
        const LINK_DELAY_S: f64 = 1.0e-4;
        for sense in [feet, body].into_iter().flatten() {
            g.add_channel(ChannelKind::Signal, sense, brain, SIGNAL_CAPACITY_HZ, LINK_DELAY_S)?;
        }
        g.add_channel(ChannelKind::Nutrient, stomach, store, nutrient_capacity_w, 0.0)?;
        for (slot, revs) in muscle_revs.iter().enumerate() {
            let m = g.add_node(scene, NodeSpec::muscle(&format!("muscle {slot}"), revs.clone(), slot))?;
            g.add_channel(ChannelKind::Signal, brain, m, SIGNAL_CAPACITY_HZ, LINK_DELAY_S)?;
            g.add_channel(ChannelKind::Nutrient, store, m, nutrient_capacity_w, 0.0)?;
            if let Some(f) = feet {
                g.add_channel(ChannelKind::Mechanical, m, f, mechanical_capacity_n, 0.0)?;
            }
        }
        g.refresh(scene);
        Ok(g)
    }

    // ----- state from the scene ---------------------------------------------

    fn build_links(&mut self, scene: &Scene) -> Links {
        let n = scene.voxels.len();
        let mut local = vec![u32::MAX; n];
        let mut index = Vec::with_capacity(self.body.len());
        let mut local_of_id = HashMap::with_capacity(self.body.len());
        for &id in &self.body {
            if let Some(i) = scene.rev_index(id) {
                local[i] = index.len() as u32;
                local_of_id.insert(id, index.len() as u32);
                index.push(i);
            }
        }
        let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); index.len()];
        for (a, &i) in index.iter().enumerate() {
            for axis in 0..3 {
                let bond = &scene.bonds[axis][i];
                let rest = scene.workspace[axis][i].rest_neighbor;
                let (j, health) = if bond.neighbor != EMPTY {
                    (bond.neighbor, (1.0 - bond.damage).clamp(0.0, 1.0))
                } else if rest != EMPTY {
                    (rest, 0.0)
                } else {
                    continue;
                };
                self.cost.link_reads += 1;
                let b = local[j as usize];
                if b == u32::MAX {
                    continue;
                }
                let health = if health.is_finite() { health } else { 0.0 };
                adj[a].push((b, health));
                adj[b as usize].push((a as u32, health));
            }
        }
        Links { index, local_of_id, adj }
    }

    /// Recompute node integrity and channel availability from the scene. Cost
    /// grows with the body size and the number of channels; call it at the rate
    /// the game needs, not necessarily every step (see
    /// [`needs_refresh`](Self::needs_refresh)).
    pub fn refresh(&mut self, scene: &Scene) {
        self.cost.refreshes += 1;
        let links = self.build_links(scene);
        for node in &mut self.nodes {
            let present: Vec<u32> = node
                .tissue
                .iter()
                .filter_map(|id| links.local_of_id.get(id).copied())
                .collect();
            let mass: f64 = present
                .iter()
                .map(|&l| f64::from(scene.voxels[links.index[l as usize]].mass))
                .sum();
            let fraction = (mass / node.reference_mass_kg).clamp(0.0, 1.0);
            let set: HashSet<u32> = present.iter().copied().collect();
            let (mut internal_sum, mut internal_n) = (0.0f64, 0u64);
            let (mut all_sum, mut all_n) = (0.0f64, 0u64);
            for &l in &present {
                for &(nb, h) in &links.adj[l as usize] {
                    all_sum += f64::from(h);
                    all_n += 1;
                    if set.contains(&nb) {
                        internal_sum += f64::from(h);
                        internal_n += 1;
                    }
                }
            }
            let health = if internal_n > 0 {
                internal_sum / internal_n as f64
            } else if all_n > 0 {
                all_sum / all_n as f64
            } else {
                1.0
            };
            node.integrity = if present.is_empty() { 0.0 } else { fraction * health };
        }
        let tissues: HashMap<OrganId, Vec<u32>> = self
            .nodes
            .iter()
            .map(|n| {
                (n.id, n.tissue.iter().filter_map(|id| links.local_of_id.get(id).copied()).collect())
            })
            .collect();
        let ids: Vec<u64> = links.index.iter().map(|&i| scene.rev_ids[i]).collect();
        let mut visits = 0u64;
        for ch in &mut self.channels {
            let (sources, targets) = (&tissues[&ch.from], &tissues[&ch.to]);
            match widest_path(&links, sources, targets, &mut visits) {
                Some((bottleneck, path)) => {
                    ch.availability = f64::from(bottleneck);
                    ch.route = path.iter().map(|&l| ids[l as usize]).collect();
                }
                None => {
                    ch.availability = 0.0;
                    ch.route.clear();
                }
            }
        }
        self.cost.search_visits += visits;
        self.refreshed_count = Some(scene.voxels.len());
    }

    /// True when the scene's REV count differs from the last refresh (REVs were
    /// removed or fused) or nothing was refreshed yet.
    pub fn needs_refresh(&self, scene: &Scene) -> bool {
        self.refreshed_count != Some(scene.voxels.len())
    }

    /// Mass (kg) of the union of all organ tissue, each REV counted once.
    pub fn owned_mass_kg(&self, scene: &Scene) -> f64 {
        self.owned_union(scene)
            .into_iter()
            .map(|i| f64::from(scene.voxels[i].mass))
            .sum()
    }

    /// Internal energy (J) of the union of all organ tissue, each REV counted once.
    pub fn owned_internal_energy_j(&self, scene: &Scene) -> f64 {
        self.owned_union(scene)
            .into_iter()
            .map(|i| scene.voxels[i].total_internal_energy_j())
            .sum()
    }

    fn owned_union(&self, scene: &Scene) -> Vec<usize> {
        let mut ids: Vec<u64> = self.nodes.iter().flat_map(|n| n.tissue.iter().copied()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter().filter_map(|id| scene.rev_index(id)).collect()
    }

    // ----- effects of the state ------------------------------------------------

    fn availability_between(&self, kind: ChannelKind, from: OrganId, to: OrganId) -> Option<f64> {
        self.channels
            .iter()
            .filter(|c| c.kind == kind && c.from == from && c.to == to)
            .map(|c| c.availability)
            .reduce(f64::max)
    }

    /// Integrity of the first brain node, `1.0` when the graph has none (no
    /// brain organ is modelled) and `0.0` when its tissue is gone.
    pub fn brain_integrity(&self) -> f64 {
        self.first_node(OrganRole::Brain).map_or(1.0, |b| b.integrity)
    }

    /// Fraction of the controller's command that reaches the muscle of
    /// `actuator_slot`: the smaller of the brain's integrity and the signal availability from
    /// the brain to that muscle (the weakest link limits, damage next to the
    /// brain is not counted twice).
    pub fn command_gain(&self, actuator_slot: usize) -> f64 {
        let Some(brain) = self.first_node(OrganRole::Brain) else { return 1.0 };
        let Some(muscle) = self.nodes.iter().find(|n| n.actuator_slot == Some(actuator_slot)) else {
            return 1.0;
        };
        let path = self.availability_between(ChannelKind::Signal, brain.id, muscle.id).unwrap_or(1.0);
        brain.integrity.min(path).clamp(0.0, 1.0)
    }

    /// Largest strength factor the muscle of `actuator_slot` can deliver:
    /// the smaller of its integrity and the nutrient availability from the store.
    pub fn strength_cap(&self, actuator_slot: usize) -> f64 {
        let Some(muscle) = self.nodes.iter().find(|n| n.actuator_slot == Some(actuator_slot)) else {
            return 1.0;
        };
        let supply = self
            .first_node(OrganRole::Store)
            .and_then(|s| self.availability_between(ChannelKind::Nutrient, s.id, muscle.id))
            .unwrap_or(1.0);
        muscle.integrity.min(supply).clamp(0.0, 1.0)
    }

    /// Share of the nominal digestion rate that works now: the smaller of the
    /// digestive organ's integrity and the nutrient availability to the store.
    pub fn digestion_throughput(&self) -> f64 {
        let Some(stomach) = self.first_node(OrganRole::Digestion) else { return 1.0 };
        let transport = self
            .first_node(OrganRole::Store)
            .and_then(|s| self.availability_between(ChannelKind::Nutrient, stomach.id, s.id))
            .unwrap_or(1.0);
        stomach.integrity.min(transport).clamp(0.0, 1.0)
    }

    /// Scale the controller's command by what reaches each muscle. With the
    /// brain gone every command is zero: the muscles go slack, the body stays.
    pub fn gate_command(&self, command: &mut ControlCommand) {
        for (slot, s) in command.actuator_strength.iter_mut().enumerate() {
            *s *= self.command_gain(slot) as f32;
        }
    }

    /// Strength of the signal a sense organ delivers to the brain: the
    /// smaller of its integrity and the signal availability to the brain.
    pub fn sense_signal(&self, organ: &OrganNode) -> f64 {
        let path = self
            .first_node(OrganRole::Brain)
            .and_then(|b| self.availability_between(ChannelKind::Signal, organ.id, b.id))
            .unwrap_or(1.0);
        organ.integrity.min(path).clamp(0.0, 1.0)
    }

    /// Degrade a sensor reading by the state of the sense organs: a foot organ
    /// scales the loads of its feet by the signal strength and reports contact
    /// only while the strength is at least [`SENSE_DETECTION_THRESHOLD`]; a body
    /// organ below the threshold blanks orientation, joints, velocities and
    /// pose. Feet and quantities without an organ are untouched. The total
    /// support load is a body-wide quantity and is left as measured.
    pub fn degrade_sensors(&self, reading: &mut SensorReading) {
        for organ in self.nodes.iter().filter(|n| n.role == OrganRole::SenseOrgan) {
            let f = self.sense_signal(organ);
            match &organ.scope {
                Some(SenseScope::Feet(feet)) => {
                    for &k in feet {
                        if let Some(c) = reading.foot_contact.get_mut(k) {
                            *c &= f >= SENSE_DETECTION_THRESHOLD;
                        }
                        if let Some(l) = reading.foot_normal_load_n.get_mut(k) {
                            *l *= f as f32;
                        }
                    }
                }
                Some(SenseScope::Body) if f < SENSE_DETECTION_THRESHOLD => {
                    reading.body_orientation = None;
                    reading.up_in_body = None;
                    reading.tilt_rad = None;
                    reading.com_height_m = None;
                    reading.body_height_m = None;
                    reading.pose_error_rms_m = None;
                    reading.angular_velocity_rad_s = glam::Vec3::ZERO;
                    reading.com_velocity_m_s = glam::Vec3::ZERO;
                    for j in &mut reading.joints {
                        j.rotation_vector_rad = None;
                    }
                }
                _ => {}
            }
        }
    }
}

/// Widest path from any source to any target over links with positive health;
/// ties go to the route with fewer links. Returns the bottleneck and the local
/// REV indices of the route from source to target.
fn widest_path(links: &Links, sources: &[u32], targets: &[u32], visits: &mut u64) -> Option<(f32, Vec<u32>)> {
    if sources.is_empty() || targets.is_empty() {
        return None;
    }
    let n = links.adj.len();
    let mut is_target = vec![false; n];
    for &t in targets {
        is_target[t as usize] = true;
    }
    // (bottleneck bits, hops); bits of non-negative f32 order like the values.
    let mut best: Vec<(u32, u32)> = vec![(0, u32::MAX); n];
    let mut prev = vec![u32::MAX; n];
    let mut done = vec![false; n];
    let mut heap = BinaryHeap::new();
    for &s in sources {
        best[s as usize] = (1.0f32.to_bits(), 0);
        heap.push((1.0f32.to_bits(), Reverse(0u32), s));
    }
    while let Some((bits, Reverse(hops), at)) = heap.pop() {
        let a = at as usize;
        if done[a] {
            continue;
        }
        done[a] = true;
        *visits += 1;
        if is_target[a] {
            let mut path = vec![at];
            let mut cur = a;
            while prev[cur] != u32::MAX {
                cur = prev[cur] as usize;
                path.push(cur as u32);
            }
            path.reverse();
            return Some((f32::from_bits(bits), path));
        }
        for &(nb, health) in &links.adj[a] {
            if health <= 0.0 || done[nb as usize] {
                continue;
            }
            let b = bits.min(health.to_bits());
            let h = hops + 1;
            let cur = best[nb as usize];
            if b > cur.0 || (b == cur.0 && h < cur.1) {
                best[nb as usize] = (b, h);
                prev[nb as usize] = at;
                heap.push((b, Reverse(h), nb));
            }
        }
    }
    None
}
