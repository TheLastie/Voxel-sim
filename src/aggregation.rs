//! Conservative conversion between detailed REV motion and a rigid far-field
//! aggregate. Material, bond, and thermal-cell state remains owned by the
//! original members; this module compresses only their mechanical motion.

use glam::{DMat3, DQuat, DVec3};
use std::{error::Error, fmt};

/// One detailed mechanical REV at the instant it enters an aggregate.
#[derive(Clone, Debug)]
pub struct ConstituentState {
    /// Stable index used to reconnect this state to its material and bond data.
    pub index: usize,
    pub mass_kg: f64,
    pub position_m: DVec3,
    /// Stress-free lattice position, transformed with the body during coarse
    /// motion so refinement does not create artificial elastic strain.
    pub rest_position_m: DVec3,
    /// Position at the preceding scene step, retained for consistent wake-up.
    pub previous_position_m: DVec3,
    pub velocity_m_s: DVec3,
    /// Current world orientation used by the material microstructure.
    pub orientation: DQuat,
    /// Last world-space force for each positive-axis bond owned by this REV.
    pub bond_last_force_n: [DVec3; 3],
    /// Conserved thermal energy for this REV, in joules.
    pub thermal_energy_j: f64,
    /// Optional nonuniform 3×3×3 thermal state. `None` means the REV's
    /// existing uniform-energy representation and avoids a 27-cell payload.
    pub thermal_cell_energy_j: Option<Box<[f64; crate::voxel::THERMAL_CELL_COUNT]>>,
}

/// A member state recovered when an aggregate is refined back into REVs.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinedConstituent {
    pub index: usize,
    pub mass_kg: f64,
    pub position_m: DVec3,
    pub rest_position_m: DVec3,
    pub previous_position_m: DVec3,
    pub velocity_m_s: DVec3,
    pub orientation: DQuat,
    pub bond_last_force_n: [DVec3; 3],
    pub thermal_energy_j: f64,
    pub thermal_cell_energy_j: Option<Box<[f64; crate::voxel::THERMAL_CELL_COUNT]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationError {
    Empty,
    InvalidMass(usize),
    InvalidState(usize),
    InvalidThermalEnergy(usize),
    InvalidThermalCells(usize),
    InvalidAcceleration,
    DuplicateIndex(usize),
    IndexOutOfBounds(usize),
    ConstrainedMember(usize),
    CrossingBond,
    DisconnectedSet,
    SingularInertia,
    InvalidTimeStep,
    InvalidImpulse,
    InvalidWrench,
    UnresolvedInternalMotion,
    InactiveAggregate(usize),
    AlreadyAggregated(usize),
}

impl fmt::Display for AggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "an aggregate needs at least one REV"),
            Self::InvalidMass(i) => write!(f, "REV {i} has non-finite or non-positive mass"),
            Self::InvalidState(i) => write!(f, "REV {i} has non-finite position or velocity"),
            Self::InvalidThermalEnergy(i) => {
                write!(f, "REV {i} has non-finite or negative thermal energy")
            }
            Self::InvalidThermalCells(i) => {
                write!(f, "REV {i} has invalid 3×3×3 thermal-cell energies")
            }
            Self::InvalidAcceleration => write!(f, "aggregate acceleration must be finite"),
            Self::DuplicateIndex(i) => write!(f, "REV index {i} occurs more than once"),
            Self::IndexOutOfBounds(i) => write!(f, "REV index {i} is outside the scene"),
            Self::ConstrainedMember(i) => {
                write!(f, "REV {i} is fixed or has a persistent coordinate target")
            }
            Self::CrossingBond => write!(f, "an intact bond crosses the aggregate boundary"),
            Self::DisconnectedSet => write!(f, "aggregate members are not one connected component"),
            Self::SingularInertia => write!(f, "aggregate inertia tensor is not invertible"),
            Self::InvalidTimeStep => write!(f, "time step must be finite and non-negative"),
            Self::InvalidImpulse => write!(f, "impulse and application point must be finite"),
            Self::InvalidWrench => write!(f, "force and torque must be finite"),
            Self::UnresolvedInternalMotion => write!(
                f,
                "component has relative motion that the rigid aggregate cannot evolve"
            ),
            Self::InactiveAggregate(i) => write!(f, "aggregate {i} is not active"),
            Self::AlreadyAggregated(i) => {
                write!(f, "REV {i} already belongs to an active aggregate")
            }
        }
    }
}

impl Error for AggregationError {}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct AggregateMember {
    index: usize,
    mass_kg: f64,
    local_position_m: DVec3,
    local_rest_position_m: DVec3,
    local_previous_position_m: DVec3,
    local_orientation: DQuat,
    local_bond_last_force_n: [DVec3; 3],
    /// Unresolved velocity modes, expressed in the aggregate's body frame.
    residual_velocity_m_s: DVec3,
    thermal_energy_j: f64,
    thermal_cell_energy_j: Option<Box<[f64; crate::voxel::THERMAL_CELL_COUNT]>>,
}

/// Allocation-free mechanical state used while advancing a coarse body.
/// Thermal-cell arrays stay on their retained REV members and are cloned only
/// when an explicit full refinement snapshot is requested.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RigidMemberPose {
    pub index: usize,
    pub mass_kg: f64,
    pub position_m: DVec3,
    pub rest_position_m: DVec3,
    pub previous_position_m: DVec3,
    pub velocity_m_s: DVec3,
    pub orientation: DQuat,
    pub bond_last_force_n: [DVec3; 3],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RigidMemberTranslationPose {
    pub index: usize,
    pub position_m: DVec3,
    pub rest_position_m: DVec3,
    pub velocity_m_s: DVec3,
}

/// A spatially coarse mechanical body with the original REV states retained
/// as body-frame residuals for conservative refinement. The 1 cm REVs and
/// their material/bond data are not replaced by invented material states.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct RigidAggregate {
    mass_kg: f64,
    center_of_mass_m: DVec3,
    linear_momentum_kg_m_s: DVec3,
    angular_momentum_kg_m2_s: DVec3,
    orientation: DQuat,
    inertia_body_kg_m2: DMat3,
    /// Spherical bound around the center of mass, including each REV's cube
    /// extent. Rotation cannot move a member outside this radius.
    bounding_radius_m: f64,
    residual_kinetic_energy_j: f64,
    thermal_energy_j: f64,
    members: Vec<AggregateMember>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AggregateInvariants {
    pub mass_kg: f64,
    pub linear_momentum_kg_m_s: DVec3,
    /// Angular momentum about the aggregate center of mass.
    pub angular_momentum_kg_m2_s: DVec3,
    pub kinetic_energy_j: f64,
    pub residual_kinetic_energy_j: f64,
    pub thermal_energy_j: f64,
}

impl RigidAggregate {
    /// Coarsen a closed connected component while retaining exact mass, linear and
    /// angular momentum, thermal energy, and unresolved relative kinetic
    /// modes. The inertia uses the translational REV degrees of freedom that
    /// the current fine solver actually carries; it does not invent an
    /// untracked spin degree of freedom for each REV. The caller must ensure
    /// that no active bond or interaction crosses the selected component's
    /// boundary and must refine it before local contact or fracture can occur.
    pub fn from_constituents(constituents: &[ConstituentState]) -> Result<Self, AggregationError> {
        if constituents.is_empty() {
            return Err(AggregationError::Empty);
        }

        let mut seen = std::collections::HashSet::with_capacity(constituents.len());
        let mut mass_kg = 0.0;
        let mut weighted_position = DVec3::ZERO;
        let mut linear_momentum = DVec3::ZERO;
        let mut thermal_energy_j = 0.0;
        for c in constituents {
            if !seen.insert(c.index) {
                return Err(AggregationError::DuplicateIndex(c.index));
            }
            if !c.mass_kg.is_finite() || c.mass_kg <= 0.0 {
                return Err(AggregationError::InvalidMass(c.index));
            }
            if !c.position_m.is_finite()
                || !c.rest_position_m.is_finite()
                || !c.previous_position_m.is_finite()
                || !c.velocity_m_s.is_finite()
                || !c.orientation.is_finite()
                || c.orientation.length_squared() <= 1e-24
                || c.bond_last_force_n.iter().any(|force| !force.is_finite())
            {
                return Err(AggregationError::InvalidState(c.index));
            }
            if !c.thermal_energy_j.is_finite() || c.thermal_energy_j < 0.0 {
                return Err(AggregationError::InvalidThermalEnergy(c.index));
            }
            if c.thermal_cell_energy_j.as_ref().is_some_and(|cells| {
                cells
                    .iter()
                    .any(|energy| !energy.is_finite() || *energy < 0.0)
            }) {
                return Err(AggregationError::InvalidThermalCells(c.index));
            }
            mass_kg += c.mass_kg;
            weighted_position += c.position_m * c.mass_kg;
            linear_momentum += c.velocity_m_s * c.mass_kg;
            thermal_energy_j += c.thermal_energy_j;
        }
        if !mass_kg.is_finite()
            || !weighted_position.is_finite()
            || !linear_momentum.is_finite()
            || !thermal_energy_j.is_finite()
        {
            return Err(AggregationError::InvalidState(0));
        }

        let center_of_mass_m = weighted_position / mass_kg;
        let linear_velocity_m_s = linear_momentum / mass_kg;
        let rev_radius_m = 0.5 * crate::scene::CELL as f64 * 3.0_f64.sqrt();
        let bounding_radius_m = constituents
            .iter()
            .map(|c| (c.position_m - center_of_mass_m).length() + rev_radius_m)
            .fold(0.0_f64, f64::max);
        let mut inertia_body = DMat3::ZERO;
        let mut angular_momentum = DVec3::ZERO;
        for c in constituents {
            let r = c.position_m - center_of_mass_m;
            let rr = DMat3::from_cols(r * r.x, r * r.y, r * r.z);
            inertia_body += (DMat3::IDENTITY * r.length_squared() - rr) * c.mass_kg;
            angular_momentum += r.cross((c.velocity_m_s - linear_velocity_m_s) * c.mass_kg);
        }
        let determinant = inertia_body.determinant();
        if !determinant.is_finite() || determinant <= 0.0 {
            return Err(AggregationError::SingularInertia);
        }
        let angular_velocity = inertia_body.inverse() * angular_momentum;
        if !angular_velocity.is_finite() {
            return Err(AggregationError::SingularInertia);
        }

        let mut residual_kinetic_energy_j = 0.0;
        let members = constituents
            .iter()
            .map(|c| {
                let r = c.position_m - center_of_mass_m;
                let residual_velocity =
                    c.velocity_m_s - linear_velocity_m_s - angular_velocity.cross(r);
                residual_kinetic_energy_j += 0.5 * c.mass_kg * residual_velocity.length_squared();
                AggregateMember {
                    index: c.index,
                    mass_kg: c.mass_kg,
                    local_position_m: r,
                    local_rest_position_m: c.rest_position_m - center_of_mass_m,
                    local_previous_position_m: c.previous_position_m - center_of_mass_m,
                    local_orientation: c.orientation.normalize(),
                    local_bond_last_force_n: c.bond_last_force_n,
                    residual_velocity_m_s: residual_velocity,
                    thermal_energy_j: c.thermal_energy_j,
                    thermal_cell_energy_j: c.thermal_cell_energy_j.clone(),
                }
            })
            .collect();
        if !residual_kinetic_energy_j.is_finite() {
            return Err(AggregationError::InvalidState(0));
        }

        Ok(Self {
            mass_kg,
            center_of_mass_m,
            linear_momentum_kg_m_s: linear_momentum,
            angular_momentum_kg_m2_s: angular_momentum,
            orientation: DQuat::IDENTITY,
            inertia_body_kg_m2: inertia_body,
            bounding_radius_m,
            residual_kinetic_energy_j,
            thermal_energy_j,
            members,
        })
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    pub fn member_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.members.iter().map(|member| member.index)
    }

    pub fn center_of_mass_m(&self) -> DVec3 {
        self.center_of_mass_m
    }

    /// Conservative radius covering every constituent REV's full cubic
    /// volume, independent of aggregate orientation.
    pub fn bounding_radius_m(&self) -> f64 {
        self.bounding_radius_m
    }

    /// Axis-aligned bounds swept by the aggregate during `dt_s` under a
    /// constant center-of-mass acceleration. This is suitable for waking and
    /// refining before an event query; the caller must include the relative
    /// motion of the queried object in its own swept bounds.
    pub fn swept_bounds_under_acceleration(
        &self,
        dt_s: f64,
        acceleration_m_s2: DVec3,
    ) -> Result<(DVec3, DVec3), AggregationError> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(AggregationError::InvalidTimeStep);
        }
        if !acceleration_m_s2.is_finite() {
            return Err(AggregationError::InvalidAcceleration);
        }

        let position = self.center_of_mass_m.to_array();
        let velocity = self.linear_velocity_m_s().to_array();
        let acceleration = acceleration_m_s2.to_array();
        let mut min = [0.0; 3];
        let mut max = [0.0; 3];
        for axis in 0..3 {
            let coordinate = |time: f64| {
                position[axis] + velocity[axis] * time + 0.5 * acceleration[axis] * time * time
            };
            let start = coordinate(0.0);
            let end = coordinate(dt_s);
            let mut lo = start.min(end);
            let mut hi = start.max(end);
            if acceleration[axis] != 0.0 {
                let extremum_time = -velocity[axis] / acceleration[axis];
                if extremum_time > 0.0 && extremum_time < dt_s {
                    let extremum = coordinate(extremum_time);
                    lo = lo.min(extremum);
                    hi = hi.max(extremum);
                }
            }
            min[axis] = lo - self.bounding_radius_m;
            max[axis] = hi + self.bounding_radius_m;
        }
        Ok((DVec3::from_array(min), DVec3::from_array(max)))
    }

    pub fn linear_velocity_m_s(&self) -> DVec3 {
        self.linear_momentum_kg_m_s / self.mass_kg
    }

    pub(crate) fn mass_kg(&self) -> f64 {
        self.mass_kg
    }

    pub fn angular_velocity_rad_s(&self) -> DVec3 {
        if self.angular_momentum_kg_m2_s == DVec3::ZERO {
            return DVec3::ZERO;
        }
        let rotation = DMat3::from_quat(self.orientation);
        let inertia_world = rotation * self.inertia_body_kg_m2 * rotation.transpose();
        inertia_world.inverse() * self.angular_momentum_kg_m2_s
    }

    pub(crate) fn angular_momentum_kg_m2_s(&self) -> DVec3 {
        self.angular_momentum_kg_m2_s
    }

    /// Frobenius norm of the inverse inertia tensor, invariant under the
    /// aggregate's world rotation and an upper bound on its spectral norm.
    pub(crate) fn inverse_inertia_frobenius_norm(&self) -> f64 {
        let inverse = self.inertia_body_kg_m2.inverse();
        (inverse.x_axis.length_squared()
            + inverse.y_axis.length_squared()
            + inverse.z_axis.length_squared())
        .sqrt()
    }

    pub fn orientation(&self) -> DQuat {
        self.orientation
    }

    pub(crate) fn has_zero_angular_momentum(&self) -> bool {
        self.angular_momentum_kg_m2_s == DVec3::ZERO
    }

    /// True when relative member motion exceeds roundoff in the resolved rigid
    /// translation and rotation. The coarse tier cannot evolve these internal
    /// modes, so such a component must remain at REV scale.
    pub fn has_unresolved_internal_motion(&self) -> bool {
        let total_kinetic = self.invariants().kinetic_energy_j.max(1.0e-30);
        self.residual_kinetic_energy_j > 128.0 * f64::EPSILON * total_kinetic
    }

    /// Iterate the transformed mechanical pose without cloning the retained
    /// thermal state. Fixed-step coarse integration calls this for every
    /// substep; full `refine()` remains the explicit detailed-state transfer.
    pub(crate) fn member_poses(&self) -> impl Iterator<Item = RigidMemberPose> + '_ {
        let linear_velocity = self.linear_velocity_m_s();
        let angular_velocity = self.angular_velocity_rad_s();
        self.members.iter().map(move |member| {
            let offset = self.orientation * member.local_position_m;
            RigidMemberPose {
                index: member.index,
                mass_kg: member.mass_kg,
                position_m: self.center_of_mass_m + offset,
                rest_position_m: self.center_of_mass_m
                    + self.orientation * member.local_rest_position_m,
                previous_position_m: self.center_of_mass_m
                    + self.orientation * member.local_previous_position_m,
                velocity_m_s: linear_velocity
                    + angular_velocity.cross(offset)
                    + self.orientation * member.residual_velocity_m_s,
                orientation: (self.orientation * member.local_orientation).normalize(),
                bond_last_force_n: member
                    .local_bond_last_force_n
                    .map(|force| self.orientation * force),
            }
        })
    }

    /// Return only the REV state that changes during pure translation. The
    /// material orientation and internal bond force are unchanged in this
    /// case, so the scene can retain their already synchronized arrays.
    pub(crate) fn translation_member_poses(
        &self,
    ) -> impl Iterator<Item = RigidMemberTranslationPose> + '_ {
        let linear_velocity = self.linear_velocity_m_s();
        self.members.iter().map(move |member| {
            let position_offset = self.orientation * member.local_position_m;
            let rest_offset = self.orientation * member.local_rest_position_m;
            RigidMemberTranslationPose {
                index: member.index,
                position_m: self.center_of_mass_m + position_offset,
                rest_position_m: self.center_of_mass_m + rest_offset,
                velocity_m_s: linear_velocity + self.orientation * member.residual_velocity_m_s,
            }
        })
    }

    pub fn invariants(&self) -> AggregateInvariants {
        let angular_velocity = self.angular_velocity_rad_s();
        let kinetic_energy = 0.5 * self.linear_momentum_kg_m_s.length_squared() / self.mass_kg
            + 0.5 * self.angular_momentum_kg_m2_s.dot(angular_velocity)
            + self.residual_kinetic_energy_j;
        AggregateInvariants {
            mass_kg: self.mass_kg,
            linear_momentum_kg_m_s: self.linear_momentum_kg_m_s,
            angular_momentum_kg_m2_s: self.angular_momentum_kg_m2_s,
            kinetic_energy_j: kinetic_energy,
            residual_kinetic_energy_j: self.residual_kinetic_energy_j,
            thermal_energy_j: self.thermal_energy_j,
        }
    }

    /// Apply an external impulse at a world-space point. Its work must be
    /// accounted by the scene's energy ledger at the interaction site.
    pub fn apply_impulse(
        &mut self,
        impulse_kg_m_s: DVec3,
        point_m: DVec3,
    ) -> Result<(), AggregationError> {
        if !impulse_kg_m_s.is_finite() || !point_m.is_finite() {
            return Err(AggregationError::InvalidImpulse);
        }
        self.linear_momentum_kg_m_s += impulse_kg_m_s;
        self.angular_momentum_kg_m2_s += (point_m - self.center_of_mass_m).cross(impulse_kg_m_s);
        Ok(())
    }

    /// Advance under a constant world-space force and torque for one interval.
    /// The symmetric half-kick/drift/half-kick update preserves the exact
    /// constant-force center-of-mass trajectory and applies external angular
    /// impulse without converting the body's physical mass into a kinematic
    /// constraint. Work and any dissipated energy remain the scene's
    /// responsibility.
    pub fn advance_under_wrench(
        &mut self,
        force_n: DVec3,
        torque_nm: DVec3,
        dt_s: f64,
    ) -> Result<(), AggregationError> {
        if !force_n.is_finite() || !torque_nm.is_finite() {
            return Err(AggregationError::InvalidWrench);
        }
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(AggregationError::InvalidTimeStep);
        }
        if dt_s == 0.0 {
            return Ok(());
        }

        let half_dt = 0.5 * dt_s;
        self.linear_momentum_kg_m_s += force_n * half_dt;
        self.angular_momentum_kg_m2_s += torque_nm * half_dt;
        self.drift_free(dt_s)?;
        self.linear_momentum_kg_m_s += force_n * half_dt;
        self.angular_momentum_kg_m2_s += torque_nm * half_dt;
        Ok(())
    }

    /// Match the translational drift of repeated semi-implicit inertial steps
    /// without repeating the rigid solve. For constant acceleration their sum
    /// is v0*dt + a*(dt*dt + dt*h)/2. The h-dependent term is a discretization
    /// error of the detailed integrator, not thermal energy or external work.
    pub(crate) fn advance_under_wrench_matching_inertial_step(
        &mut self,
        force_n: DVec3,
        torque_nm: DVec3,
        dt_s: f64,
        reference_step_s: f64,
    ) -> Result<(), AggregationError> {
        if !reference_step_s.is_finite()
            || reference_step_s <= 0.0
            || !dt_s.is_finite()
            || reference_step_s > dt_s
        {
            return Err(AggregationError::InvalidTimeStep);
        }
        let drift_bias = force_n / self.mass_kg * (0.5 * dt_s * reference_step_s);
        if !drift_bias.is_finite() {
            return Err(AggregationError::InvalidWrench);
        }
        self.advance_under_wrench(force_n, torque_nm, dt_s)?;
        self.center_of_mass_m += drift_bias;
        Ok(())
    }

    /// Drift the isolated aggregate using a second-order midpoint update for
    /// its torque-free rotation. Linear and angular momentum remain constant.
    pub fn drift_free(&mut self, dt_s: f64) -> Result<(), AggregationError> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(AggregationError::InvalidTimeStep);
        }
        if dt_s == 0.0 {
            return Ok(());
        }
        self.center_of_mass_m += self.linear_velocity_m_s() * dt_s;
        let omega_start = self.angular_velocity_rad_s();
        let speed_start = omega_start.length();
        if speed_start > 0.0 {
            let midpoint_delta =
                DQuat::from_axis_angle(omega_start / speed_start, 0.5 * speed_start * dt_s);
            let midpoint_orientation = (midpoint_delta * self.orientation).normalize();
            let rotation = DMat3::from_quat(midpoint_orientation);
            let inertia_midpoint = rotation * self.inertia_body_kg_m2 * rotation.transpose();
            let omega_midpoint = inertia_midpoint.inverse() * self.angular_momentum_kg_m2_s;
            let speed_midpoint = omega_midpoint.length();
            if speed_midpoint > 0.0 {
                let full_delta =
                    DQuat::from_axis_angle(omega_midpoint / speed_midpoint, speed_midpoint * dt_s);
                self.orientation = (full_delta * self.orientation).normalize();
            }
        }
        Ok(())
    }

    /// Recover fine REV states by applying the current rigid pose to the
    /// retained body-frame geometry and unresolved velocity modes.
    pub fn refine(&self) -> Vec<RefinedConstituent> {
        self.members
            .iter()
            .zip(self.member_poses())
            .map(|(member, pose)| RefinedConstituent {
                index: pose.index,
                mass_kg: pose.mass_kg,
                position_m: pose.position_m,
                rest_position_m: pose.rest_position_m,
                previous_position_m: pose.previous_position_m,
                velocity_m_s: pose.velocity_m_s,
                orientation: pose.orientation,
                bond_last_force_n: pose.bond_last_force_n,
                thermal_energy_j: member.thermal_energy_j,
                thermal_cell_energy_j: member.thermal_cell_energy_j.clone(),
            })
            .collect()
    }
}
