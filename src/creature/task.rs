//! A deliberately simple learnable task: raise one chosen muscle channel's
//! activation while a condition on the real sensor readings holds, and keep it
//! low otherwise.
//!
//! Only the creature's [`LearnedState`] changes. The task needs no walking and
//! learning it says nothing about learning to walk.

use super::{HillClimber, LearnedState, NetworkLayout, SensorReading, SpikingNetwork};

/// A yes/no statement about one sensor reading.
#[derive(Clone, Debug, PartialEq)]
pub enum SensorCondition {
    /// The foot (index into the layout's feet) carries at least this load (N).
    FootLoadAtLeast { foot: usize, newtons: f32 },
    /// The foot carries at least this fraction of the creature's weight.
    FootLoadFractionAtLeast { foot: usize, fraction: f32 },
    /// The body is tilted by at least this angle from its original vertical (rad).
    TiltAtLeast { rad: f32 },
}

impl SensorCondition {
    /// False when the reading cannot answer (missing foot or tilt).
    pub fn holds(&self, reading: &SensorReading) -> bool {
        match *self {
            Self::FootLoadAtLeast { foot, newtons } => reading
                .foot_normal_load_n
                .get(foot)
                .is_some_and(|&load| load >= newtons),
            Self::FootLoadFractionAtLeast { foot, fraction } => reading
                .foot_normal_load_n
                .get(foot)
                .is_some_and(|&load| load >= fraction * reading.body_weight_n),
            Self::TiltAtLeast { rad } => reading.tilt_rad.is_some_and(|tilt| tilt >= rad),
        }
    }
}

/// What a learning run did.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskReport {
    pub reward_before: f64,
    pub reward_after: f64,
    pub trials: usize,
    pub accepted: usize,
}

/// Raise `muscle_channel` (an output neuron's decoded activation) when
/// `condition` holds on the sensor frame, lower it otherwise.
///
/// Reward is minus the class-balanced mean absolute error of the activation
/// against the target (1 while the condition holds, 0 while it does not),
/// measured on frames after the first `settle_frames`. Balancing the classes
/// makes the best constant activation score exactly -0.5, so a better reward
/// can only come from using the sensors.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivationTask {
    pub muscle_channel: usize,
    pub condition: SensorCondition,
    pub settle_frames: usize,
    /// Initial step size of the hill climber as a fraction of the limits.
    pub sigma: f32,
}

impl ActivationTask {
    pub fn new(muscle_channel: usize, condition: SensorCondition) -> Self {
        Self {
            muscle_channel,
            condition,
            settle_frames: 3,
            sigma: 0.1,
        }
    }

    /// Reward of `learned` on `frames`, each held for `frame_dt_s` seconds.
    /// Needs frames of both classes after the settling time.
    pub fn reward(
        &self,
        layout: &NetworkLayout,
        learned: &LearnedState,
        frames: &[SensorReading],
        frame_dt_s: f64,
    ) -> Result<f64, String> {
        if self.muscle_channel >= layout.output_neurons.len() {
            return Err("muscle channel outside the network outputs".into());
        }
        let mut network = SpikingNetwork::new(layout.clone(), learned)?;
        let mut inputs = Vec::new();
        let mut error = [0.0f64; 2];
        let mut count = [0usize; 2];
        for (k, frame) in frames.iter().enumerate() {
            frame.encode(&mut inputs);
            let activation = network.step(frame_dt_s, &inputs)?[self.muscle_channel];
            if k < self.settle_frames {
                continue;
            }
            let class = usize::from(self.condition.holds(frame));
            error[class] += (f64::from(activation) - class as f64).abs();
            count[class] += 1;
        }
        if count.contains(&0) {
            return Err("the frames must contain the condition both holding and not holding".into());
        }
        Ok(-0.5 * (error[0] / count[0] as f64 + error[1] / count[1] as f64))
    }

    /// Reward-gated hill climb over `learned` (see [`HillClimber`]): a change is
    /// kept only when it raises the reward, so the reward never decreases.
    /// Deterministic for a given seed, layout, starting state and frames.
    pub fn learn(
        &self,
        layout: &NetworkLayout,
        learned: &mut LearnedState,
        frames: &[SensorReading],
        frame_dt_s: f64,
        seed: u64,
        trials: usize,
    ) -> Result<TaskReport, String> {
        learned.validate_against(layout)?;
        let before = self.reward(layout, learned, frames, frame_dt_s)?;
        let mut climber = HillClimber::new(seed, self.sigma);
        climber.set_baseline(before);
        let accepted_before = learned.accepted_updates;
        for _ in 0..trials {
            let candidate = climber.propose(learned, layout);
            let reward = self.reward(layout, &candidate, frames, frame_dt_s)?;
            climber.judge(learned, candidate, reward);
        }
        Ok(TaskReport {
            reward_before: before,
            reward_after: self.reward(layout, learned, frames, frame_dt_s)?,
            trials,
            accepted: (learned.accepted_updates - accepted_before) as usize,
        })
    }
}
