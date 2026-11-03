use crate::{
    material::Material,
    scene::{Scene, AREA, CELL, DCELL},
    voxel::Composition,
};
use glam::{DVec3, Quat, Vec3};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
pub enum Shape {
    Wedge,
    Blade,
    Rod,
    Rounded,
}
/// A fixed-orientation support for a local base point. A unit direction permits
/// sliding along one world axis; zero direction fixes the base completely.
#[derive(Clone, Copy, Debug)]
pub struct GuideMount {
    pub local_base: Vec3,
    pub world_base: DVec3,
    pub direction: Vec3,
    pub rotation: Quat,
}
/// A rigid voxel object. The collision law never branches on Shape.
pub struct Tool {
    pub cells: Vec<Vec3>,
    collision_radius: f32,
    /// Local exposed REV faces: (face centre, outward normal).
    surface_faces: Vec<(Vec3, Vec3)>,
    spatial_bins: HashMap<[i32; 3], SmallVec<[usize; 2]>>,
    spatial_keys: Vec<[i32; 3]>,
    spatial_occupancy: Vec<u64>,
    spatial_occupancy_min: [i32; 3],
    spatial_occupancy_dims: [usize; 3],
    spatial_occupancy_enabled: bool,
    probe_bins: Vec<[i32; 3]>,
    probe_cells: Vec<Vec3>,
    probe_position: DVec3,
    probe_rotation: Quat,
    probe_cache_valid: bool,
    center_cache: Vec<DVec3>,
    /// REV nodes touched during the most recent advance, including resting
    /// contacts that produce no impulse. Used to wake coarse bodies before the
    /// mechanical solve consumes their interaction.
    pub(crate) contacted_voxels: Vec<usize>,
    pub composition: Composition,
    surface_friction_coefficient: f32,
    pub mass: f32,
    pub position: DVec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Persistent world-space force and torque, analogous to voxel inputs.
    pub external_force: Vec3,
    pub external_torque: Vec3,
    pub inverse_inertia: Vec3,
    pub color: [f32; 4],
    pub mount: Option<GuideMount>,
}
impl Tool {
    pub fn new(
        shape: Shape,
        scene: &Scene,
        position: impl Into<DVec3>,
        direction: Vec3,
        speed: f32,
    ) -> Self {
        let position = position.into();
        let d = scene.config.tool_dimensions;
        let (dims, component, color) = match shape {
            Shape::Wedge => (d, 2, [0.65, 0.68, 0.72, 1.0]),
            // Both tools share a bevel that tapers the 5 REV spine to a
            // one-REV cutting edge at the leading (+X) face.
            Shape::Blade => ([d[0], d[1].max(5), d[2]], 2, [0.8, 0.8, 0.85, 1.0]),
            Shape::Rod => ([d[0], 4, 4], 1, [0.5, 0.28, 0.1, 1.0]),
            Shape::Rounded => ([14, 14, 14], 3, [0.4, 0.42, 0.45, 1.0]),
        };
        let mut cells = Vec::new();
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
                        - Vec3::new(dims[0] as f32, dims[1] as f32, dims[2] as f32) * 0.5;
                    let include = match shape {
                        Shape::Wedge | Shape::Blade => {
                            let thickness = 0.5
                                + (dims[0] - 1 - x) as f32 / (dims[0] - 1) as f32
                                    * (dims[1] as f32 * 0.5 - 0.5);
                            p.y.abs() <= thickness
                        }
                        Shape::Rod => p.y * p.y + p.z * p.z <= 4.0,
                        Shape::Rounded => p.length_squared() <= 49.0,
                    };
                    if include {
                        cells.push(p * CELL);
                    }
                }
            }
        }
        let composition = Composition::pure(component);
        let material = Material::mixed(&composition, &scene.materials);
        let cell_mass = material.density * CELL.powi(3);
        let mass = cells.len() as f32 * cell_mass;
        let rotation = Quat::from_rotation_arc(Vec3::X, direction.normalize());
        // Dynamics are expressed about the occupied volume's center of mass.
        // Shift the reference position too, preserving every world-space cell.
        let centroid = cells.iter().copied().sum::<Vec3>() / cells.len() as f32;
        for p in &mut cells {
            *p -= centroid;
        }
        let collision_radius = cells.iter().map(|p| p.length()).fold(0.0, f32::max);
        let cell_key = |p: Vec3| (p / CELL).round().as_ivec3().to_array();
        let occupied: HashSet<_> = cells.iter().copied().map(cell_key).collect();
        let mut surface_faces = Vec::new();
        for &cell in &cells {
            let key = cell_key(cell);
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                for sign in [-1.0f32, 1.0] {
                    let normal = axis * sign;
                    let offset = (normal * 1.01).round().as_ivec3().to_array();
                    let neighbor = [key[0] + offset[0], key[1] + offset[1], key[2] + offset[2]];
                    if !occupied.contains(&neighbor) {
                        surface_faces.push((cell + normal * (CELL * 0.5), normal));
                    }
                }
            }
        }
        let mut inertia = Vec3::ZERO;
        for &p in &cells {
            inertia += (Vec3::splat(p.length_squared() + CELL * CELL / 6.0) - p * p) * cell_mass;
        }
        Self {
            cells,
            collision_radius,
            surface_faces,
            spatial_bins: HashMap::new(),
            spatial_keys: Vec::new(),
            spatial_occupancy: Vec::new(),
            spatial_occupancy_min: [0; 3],
            spatial_occupancy_dims: [0; 3],
            spatial_occupancy_enabled: false,
            probe_bins: Vec::new(),
            probe_cells: Vec::new(),
            probe_position: DVec3::ZERO,
            probe_rotation: Quat::IDENTITY,
            probe_cache_valid: false,
            center_cache: Vec::new(),
            contacted_voxels: Vec::new(),
            composition,
            surface_friction_coefficient: material.surface_friction_coefficient,
            mass,
            position: position + (rotation * centroid).as_dvec3(),
            rotation,
            velocity: direction.normalize() * speed,
            angular_velocity: Vec3::ZERO,
            external_force: Vec3::ZERO,
            external_torque: Vec3::ZERO,
            inverse_inertia: inertia.recip(),
            color,
            mount: None,
        }
    }
    pub fn set_external_force(&mut self, force: Vec3) -> Result<(), &'static str> {
        if !force.is_finite() {
            return Err("force must be finite");
        }
        self.external_force = force;
        Ok(())
    }
    pub fn set_external_torque(&mut self, torque: Vec3) -> Result<(), &'static str> {
        if !torque.is_finite() {
            return Err("torque must be finite");
        }
        self.external_torque = torque;
        Ok(())
    }
    pub fn clear_external_wrench(&mut self) {
        self.external_force = Vec3::ZERO;
        self.external_torque = Vec3::ZERO;
    }
    /// Anchor a base to a fixed-orientation guide. Zero direction fixes all motion.
    pub fn mount_base(&mut self, local_base: Vec3, direction: Vec3) -> Result<(), &'static str> {
        if !local_base.is_finite() || !direction.is_finite() {
            return Err("mount coordinates must be finite");
        }
        let direction = direction.try_normalize().unwrap_or(Vec3::ZERO);
        self.mount = Some(GuideMount {
            local_base,
            world_base: self.position + (self.rotation * local_base).as_dvec3(),
            direction,
            rotation: self.rotation,
        });
        self.velocity = self.project_translation(self.velocity);
        self.angular_velocity = Vec3::ZERO;
        Ok(())
    }
    pub fn unmount(&mut self) {
        self.mount = None;
    }
    fn project_translation(&self, vector: Vec3) -> Vec3 {
        self.mount
            .map_or(vector, |m| m.direction * m.direction.dot(vector))
    }
    fn linear_response(&self, impulse: Vec3) -> Vec3 {
        self.project_translation(impulse) / self.mass
    }
    fn constrain_pose(&mut self) {
        if let Some(m) = self.mount {
            self.rotation = m.rotation;
            let base = self.position + (self.rotation * m.local_base).as_dvec3();
            self.position += m.world_base
                + m.direction.as_dvec3() * (base - m.world_base).dot(m.direction.as_dvec3())
                - base;
            self.velocity = self.project_translation(self.velocity);
            self.angular_velocity = Vec3::ZERO;
        }
    }
    fn contact_correction(&self, normal: Vec3, penetration: f32) -> Vec3 {
        let projected = self.project_translation(normal);
        let mobility = normal.dot(projected);
        if mobility > 1e-6 {
            -projected * penetration / mobility
        } else {
            Vec3::ZERO
        }
    }
    pub fn world_cells(&self) -> impl Iterator<Item = DVec3> + '_ {
        self.cells
            .iter()
            .map(|&p| self.position + (self.rotation * p).as_dvec3())
    }
    fn angular_response(&self, torque: Vec3) -> Vec3 {
        if self.mount.is_some() {
            Vec3::ZERO
        } else {
            self.rotation * (self.inverse_inertia * (self.rotation.conjugate() * torque))
        }
    }

    fn atmospheric_wrench(&self, scene: &Scene) -> (Vec3, Vec3) {
        let air = scene.config.atmosphere;
        let density = air.density_kg_m3();
        if density <= 0.0 {
            return (Vec3::ZERO, Vec3::ZERO);
        }

        let gravity = Vec3::from_array(scene.config.gravity);
        let displaced_air_mass = density * self.cells.len() as f32 * CELL.powi(3);
        let mut force = -gravity * displaced_air_mass;
        let mut torque = Vec3::ZERO;
        let wind = Vec3::from_array(air.wind_velocity_m_s);
        for &(local_center, local_normal) in &self.surface_faces {
            let arm = self.rotation * local_center;
            let normal = self.rotation * local_normal;
            let surface_velocity = self.velocity + self.angular_velocity.cross(arm);
            let normal_speed = (surface_velocity - wind).dot(normal);
            let face_force = -normal
                * (0.25
                    * density
                    * air.drag_coefficient
                    * AREA
                    * normal_speed
                    * normal_speed.abs());
            force += face_force;
            torque += arm.cross(face_force);
        }
        (force, torque)
    }
    /// Finite rigid mass, friction and angular impulse, with equal/opposite reaction.
    /// Microsteps prevent tunnelling through a 1 cm REV at typical strike speeds.
    pub fn advance(
        &mut self,
        scene: &mut Scene,
        dt: f32,
        targets: &[[Option<f64>; 3]],
    ) -> Vec<Vec3> {
        let mut contact_velocity = Vec::with_capacity(scene.positions.len());
        self.advance_into(scene, dt, targets, &mut contact_velocity);
        contact_velocity
    }

    pub(crate) fn advance_into(
        &mut self,
        scene: &mut Scene,
        dt: f32,
        targets: &[[Option<f64>; 3]],
        contact_velocity: &mut Vec<Vec3>,
    ) {
        self.constrain_pose();
        let track_contact_voxels = scene.has_registered_rigid_candidates();
        let mut contacted_voxels = std::mem::take(&mut self.contacted_voxels);
        contacted_voxels.clear();
        contact_velocity.resize(scene.positions.len(), Vec3::ZERO);
        contact_velocity.fill(Vec3::ZERO);
        let tool_radius = self.collision_radius;
        let gravity = Vec3::from_array(scene.config.gravity);
        // Reuse the initial air wrench for the first motion substep below.
        // Tool and scene state are unchanged between the swept-step estimate
        // and the first integration pass, so evaluating the same scene-wide
        // drag sum twice would be redundant. A zero-duration recheck cannot
        // apply aerodynamic work or impulse.
        let (initial_air_force, initial_air_torque) = if dt == 0.0 {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            self.atmospheric_wrench(scene)
        };
        let linear_acceleration = self
            .project_translation(gravity + (self.external_force + initial_air_force) / self.mass);
        let angular_acceleration = self.angular_response(self.external_torque + initial_air_torque);
        // The integrator updates velocity before position (semi-implicit Euler),
        // so estimate travel with the same end-of-step velocity it will use.
        let linear_distance = (self.velocity + linear_acceleration * dt).length() * dt;
        let angular_distance =
            (self.angular_velocity + angular_acceleration * dt).length() * dt * tool_radius;
        let linear_steps = linear_distance / (CELL * 0.4);
        let angular_steps = angular_distance / (CELL * 0.4);
        let steps = (linear_steps.max(angular_steps).ceil() as usize).max(1);
        let h = dt / steps as f32;
        let mut spatial = std::mem::take(&mut self.spatial_bins);
        let mut spatial_keys = std::mem::take(&mut self.spatial_keys);
        let mut spatial_occupancy = std::mem::take(&mut self.spatial_occupancy);
        let mut spatial_occupancy_min = self.spatial_occupancy_min;
        let mut spatial_occupancy_dims = self.spatial_occupancy_dims;
        let mut spatial_occupancy_enabled = self.spatial_occupancy_enabled;
        let mut probe_bins = std::mem::take(&mut self.probe_bins);
        let mut probe_cells = std::mem::take(&mut self.probe_cells);
        let mut probe_position = self.probe_position;
        let mut probe_rotation = self.probe_rotation;
        let mut probe_cache_valid = self.probe_cache_valid;
        let mut centers_cache = std::mem::take(&mut self.center_cache);
        for step_index in 0..steps {
            let (air_force, air_torque) = if step_index == 0 {
                (initial_air_force, initial_air_torque)
            } else {
                self.atmospheric_wrench(scene)
            };
            let linear_acceleration =
                self.project_translation(gravity + (self.external_force + air_force) / self.mass);
            let angular_acceleration = self.angular_response(self.external_torque + air_torque);
            let old_position = self.position;
            let old_angular_velocity = self.angular_velocity;
            self.velocity += linear_acceleration * h;
            self.angular_velocity += angular_acceleration * h;
            self.position += self.velocity.as_dvec3() * h as f64;
            let angle = self.angular_velocity.length() * h;
            if angle > 1e-8 {
                self.rotation = (Quat::from_axis_angle(self.angular_velocity.normalize(), angle)
                    * self.rotation)
                    .normalize();
            }
            scene.energy.external_work += ((air_force + self.external_force)
                .dot((self.position - old_position).as_vec3())
                + (air_torque + self.external_torque)
                    .dot((old_angular_velocity + self.angular_velocity) * (0.5 * h)))
                as f64;
            refresh_spatial_bins(&scene.positions, &mut spatial, &mut spatial_keys);
            let axes = [
                self.rotation * Vec3::X,
                self.rotation * Vec3::Y,
                self.rotation * Vec3::Z,
            ];
            centers_cache.clear();
            centers_cache.extend(self.world_cells());
            spatial_occupancy_enabled = refresh_spatial_occupancy(
                &spatial_keys,
                &mut spatial_occupancy,
                &mut spatial_occupancy_min,
                &mut spatial_occupancy_dims,
            );
            if !probe_cache_valid
                || probe_position != self.position
                || probe_rotation != self.rotation
                || probe_cells != self.cells
            {
                probe_bins.clear();
                for &center in &centers_cache {
                    let (lo, hi) = obb_bin_bounds(center, axes);
                    for z in lo[2]..=hi[2] {
                        for y in lo[1]..=hi[1] {
                            for x in lo[0]..=hi[0] {
                                probe_bins.push([x, y, z]);
                            }
                        }
                    }
                }
                probe_bins.sort_unstable();
                probe_bins.dedup();
                probe_cells.clone_from(&self.cells);
                probe_position = self.position;
                probe_rotation = self.rotation;
                probe_cache_valid = true;
            }
            let has_scene_candidate = if spatial_occupancy_enabled {
                probe_bins.iter().any(|&cell| {
                    spatial_occupancy_contains(
                        cell,
                        &spatial_occupancy,
                        spatial_occupancy_min,
                        spatial_occupancy_dims,
                    )
                })
            } else {
                probe_bins.iter().any(|cell| spatial.contains_key(cell))
            };
            if !has_scene_candidate {
                continue;
            }
            // Several adjacent tool cells can overlap the same REV during one
            // microstep. Accumulate only the deepest correction; applying the
            // full depenetration once per pair would push the whole tool back
            // by the number of contacts and prevent it from loading the wood.
            let mut correction = Vec3::ZERO;
            let mut max_penetration = 0.0f32;
            for &center in &centers_cache {
                let (lo, hi) = obb_bin_bounds(center, axes);
                let query_bounds = if spatial_occupancy_enabled {
                    let Some(bounds) =
                        clip_spatial_bounds(lo, hi, spatial_occupancy_min, spatial_occupancy_dims)
                    else {
                        continue;
                    };
                    bounds
                } else {
                    (lo, hi)
                };
                let x_stride = spatial_occupancy_dims[1] * spatial_occupancy_dims[2];
                for z in query_bounds.0[2]..=query_bounds.1[2] {
                    for y in query_bounds.0[1]..=query_bounds.1[1] {
                        let row_offset = if spatial_occupancy_enabled {
                            ((y - spatial_occupancy_min[1]) as usize * spatial_occupancy_dims[2])
                                + (z - spatial_occupancy_min[2]) as usize
                        } else {
                            0
                        };
                        let mut occupancy_index = if spatial_occupancy_enabled {
                            (query_bounds.0[0] - spatial_occupancy_min[0]) as usize * x_stride
                                + row_offset
                        } else {
                            0
                        };
                        for x in query_bounds.0[0]..=query_bounds.1[0] {
                            let cell = [x, y, z];
                            if spatial_occupancy_enabled {
                                let occupied = spatial_occupancy
                                    [occupancy_index / u64::BITS as usize]
                                    & (1u64 << (occupancy_index % u64::BITS as usize))
                                    != 0;
                                occupancy_index += x_stride;
                                if !occupied {
                                    continue;
                                }
                            }
                            let Some(indices) = spatial.get(&cell) else {
                                continue;
                            };
                            for &i in indices {
                                let p = scene.positions[i];
                                let Some((n, penetration, _contact_point)) =
                                    obb_aabb_contact(center, axes, p)
                                else {
                                    continue;
                                };
                                if track_contact_voxels {
                                    contacted_voxels.push(i);
                                }
                                // The scene REV carries linear momentum at its node.
                                // Use that same point for the rigid reaction, otherwise
                                // the equal/opposite pair creates angular momentum.
                                let arm = (p - self.position).as_vec3();
                                let rel = self.velocity + self.angular_velocity.cross(arm)
                                    - scene.voxels[i].velocity;
                                let closing = rel.dot(n);
                                let free = Vec3::new(
                                    if !scene.voxels[i].kinematic
                                        && targets.get(i).and_then(|t| t[0]).is_none()
                                    {
                                        1.0
                                    } else {
                                        0.0
                                    },
                                    if !scene.voxels[i].kinematic
                                        && targets.get(i).and_then(|t| t[1]).is_none()
                                    {
                                        1.0
                                    } else {
                                        0.0
                                    },
                                    if !scene.voxels[i].kinematic
                                        && targets.get(i).and_then(|t| t[2]).is_none()
                                    {
                                        1.0
                                    } else {
                                        0.0
                                    },
                                );
                                let wi = if !scene.voxels[i].kinematic {
                                    (n * free).dot(n) / scene.voxels[i].mass
                                } else {
                                    0.0
                                };
                                let denom = wi
                                    + n.dot(self.linear_response(n))
                                    + n.dot(self.angular_response(arm.cross(n)).cross(arm));
                                if closing > 0.0 {
                                    // This is a persistent, multi-point manifold. Applying
                                    // restitution at every cell contact repeatedly adds
                                    // energy as the active normal changes, so the normal
                                    // solve is perfectly inelastic. Its lost relative kinetic
                                    // energy must be thermalized on the receiving REV.
                                    let normal_impulse = closing / denom.max(1e-9);
                                    let normal_heat =
                                        (0.5 * normal_impulse as f64 * closing as f64).max(0.0);
                                    scene.voxels[i].add_internal_energy_j(normal_heat);
                                    scene.energy.contact_restitution_heat += normal_heat;
                                    let normal_vector = normal_impulse * n;
                                    let normal_voxel_delta =
                                        normal_vector * free / scene.voxels[i].mass;
                                    scene.voxels[i].velocity += normal_voxel_delta;
                                    contact_velocity[i] += normal_voxel_delta;
                                    self.velocity -= self.linear_response(normal_vector);
                                    self.angular_velocity -=
                                        self.angular_response(arm.cross(normal_vector));

                                    // Recompute relative motion after the normal impulse. This
                                    // includes normal-to-tangent coupling from an off-centre hit.
                                    let relative_after_normal = self.velocity
                                        + self.angular_velocity.cross(arm)
                                        - scene.voxels[i].velocity;
                                    let tangent =
                                        relative_after_normal - relative_after_normal.dot(n) * n;
                                    let tangent_speed = tangent.length();
                                    let tangent_dir = tangent.try_normalize().unwrap_or(Vec3::ZERO);
                                    let tangent_free = Vec3::new(
                                        if !scene.voxels[i].kinematic
                                            && targets.get(i).and_then(|t| t[0]).is_none()
                                        {
                                            1.0
                                        } else {
                                            0.0
                                        },
                                        if !scene.voxels[i].kinematic
                                            && targets.get(i).and_then(|t| t[1]).is_none()
                                        {
                                            1.0
                                        } else {
                                            0.0
                                        },
                                        if !scene.voxels[i].kinematic
                                            && targets.get(i).and_then(|t| t[2]).is_none()
                                        {
                                            1.0
                                        } else {
                                            0.0
                                        },
                                    );
                                    let tangent_voxel_w = if !scene.voxels[i].kinematic {
                                        (tangent_dir * tangent_free).dot(tangent_dir)
                                            / scene.voxels[i].mass
                                    } else {
                                        0.0
                                    };
                                    let tangent_denom = tangent_voxel_w
                                        + tangent_dir.dot(self.linear_response(tangent_dir))
                                        + tangent_dir.dot(
                                            self.angular_response(arm.cross(tangent_dir))
                                                .cross(arm),
                                        );
                                    if tangent_speed > 1e-6 && tangent_denom > 1e-9 {
                                        let pair_friction = scene.config.fracture.friction
                                            * (scene.properties[i].surface_friction_coefficient
                                                + self.surface_friction_coefficient);
                                        let friction_magnitude = (tangent_speed / tangent_denom)
                                            .min(pair_friction * normal_impulse);
                                        let friction = tangent_dir * friction_magnitude;
                                        let friction_voxel_delta =
                                            friction * free / scene.voxels[i].mass;
                                        scene.voxels[i].velocity += friction_voxel_delta;
                                        contact_velocity[i] += friction_voxel_delta;
                                        self.velocity -= self.linear_response(friction);
                                        self.angular_velocity -=
                                            self.angular_response(arm.cross(friction));

                                        let friction_heat = (friction_magnitude as f64
                                            * tangent_speed as f64
                                            - 0.5
                                                * (friction_magnitude as f64).powi(2)
                                                * tangent_denom as f64)
                                            .max(0.0);
                                        scene.voxels[i].add_internal_energy_j(friction_heat);
                                        scene.energy.contact_friction_heat += friction_heat;
                                    }
                                }
                                // Split positional correction from velocity. It enforces
                                // non-penetration without turning depenetration into energy.
                                if penetration > max_penetration {
                                    max_penetration = penetration;
                                    correction = self.contact_correction(n, penetration);
                                }
                            }
                        }
                    }
                }
            }
            // Keep projection separate from momentum. The contact impulses
            // carry the load into the bonded material; this only removes overlap.
            self.position += correction.as_dvec3();
            self.constrain_pose();
            // Multiple nearby surface contacts can constrain different axes.
            // Reproject the rigid shape as a whole until every sampled OBB/AABB
            // pair is clear, rather than leaving a corner embedded after one pass.
            for _ in 0..6 {
                let mut correction = Vec3::ZERO;
                let mut max_penetration = 1e-6f32;
                for center in self.world_cells() {
                    let (lo, hi) = obb_bin_bounds(center, axes);
                    for z in lo[2]..=hi[2] {
                        for y in lo[1]..=hi[1] {
                            for x in lo[0]..=hi[0] {
                                let Some(indices) = spatial.get(&[x, y, z]) else {
                                    continue;
                                };
                                for &i in indices {
                                    let Some((normal, penetration, _)) =
                                        obb_aabb_contact(center, axes, scene.positions[i])
                                    else {
                                        continue;
                                    };
                                    if track_contact_voxels {
                                        contacted_voxels.push(i);
                                    }
                                    if penetration > max_penetration {
                                        max_penetration = penetration;
                                        correction = self.contact_correction(normal, penetration);
                                    }
                                }
                            }
                        }
                    }
                }
                if max_penetration <= 1e-6 {
                    break;
                }
                self.position += correction.as_dvec3();
                self.constrain_pose();
            }
        }
        self.spatial_bins = spatial;
        self.spatial_keys = spatial_keys;
        self.spatial_occupancy = spatial_occupancy;
        self.spatial_occupancy_min = spatial_occupancy_min;
        self.spatial_occupancy_dims = spatial_occupancy_dims;
        self.spatial_occupancy_enabled = spatial_occupancy_enabled;
        self.probe_bins = probe_bins;
        self.probe_cells = probe_cells;
        self.probe_position = probe_position;
        self.probe_rotation = probe_rotation;
        self.probe_cache_valid = probe_cache_valid;
        self.center_cache = centers_cache;
        if track_contact_voxels {
            contacted_voxels.sort_unstable();
            contacted_voxels.dedup();
        } else {
            contacted_voxels.clear();
        }
        self.contacted_voxels = contacted_voxels;
    }
}
fn key(p: impl Into<DVec3>) -> [i32; 3] {
    let p = p.into();
    (p / DCELL).floor().as_ivec3().to_array()
}

const MAX_TOOL_OCCUPANCY_CELLS: usize = 1_000_000;

/// Rebuild a compact exact occupancy mask when the scene's bounded grid is
/// small enough. A sparse or unusually large scene keeps using the hash map.
fn refresh_spatial_occupancy(
    keys: &[[i32; 3]],
    occupancy: &mut Vec<u64>,
    minimum: &mut [i32; 3],
    dimensions: &mut [usize; 3],
) -> bool {
    if keys.is_empty() {
        occupancy.clear();
        *minimum = [0; 3];
        *dimensions = [0; 3];
        return true;
    }

    let mut low = keys[0];
    let mut high = keys[0];
    for key in &keys[1..] {
        for axis in 0..3 {
            low[axis] = low[axis].min(key[axis]);
            high[axis] = high[axis].max(key[axis]);
        }
    }
    let mut next_dimensions = [0usize; 3];
    for axis in 0..3 {
        let span = i64::from(high[axis]) - i64::from(low[axis]) + 1;
        let Ok(dimension) = usize::try_from(span) else {
            occupancy.clear();
            *minimum = [0; 3];
            *dimensions = [0; 3];
            return false;
        };
        next_dimensions[axis] = dimension;
    }
    let Some(cell_count) = next_dimensions
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
    else {
        occupancy.clear();
        *minimum = [0; 3];
        *dimensions = [0; 3];
        return false;
    };
    if cell_count > MAX_TOOL_OCCUPANCY_CELLS {
        occupancy.clear();
        *minimum = [0; 3];
        *dimensions = [0; 3];
        return false;
    }

    *minimum = low;
    *dimensions = next_dimensions;
    let word_count = cell_count.div_ceil(u64::BITS as usize);
    occupancy.resize(word_count, 0);
    occupancy.fill(0);
    for key in keys {
        let x = (key[0] - low[0]) as usize;
        let y = (key[1] - low[1]) as usize;
        let z = (key[2] - low[2]) as usize;
        let index = (x * next_dimensions[1] + y) * next_dimensions[2] + z;
        occupancy[index / u64::BITS as usize] |= 1u64 << (index % u64::BITS as usize);
    }
    true
}

#[inline]
fn clip_spatial_bounds(
    lo: [i32; 3],
    hi: [i32; 3],
    minimum: [i32; 3],
    dimensions: [usize; 3],
) -> Option<([i32; 3], [i32; 3])> {
    let mut clipped_lo = [0i32; 3];
    let mut clipped_hi = [0i32; 3];
    for axis in 0..3 {
        if dimensions[axis] == 0 {
            return None;
        }
        let maximum = minimum[axis] + dimensions[axis] as i32 - 1;
        if hi[axis] < minimum[axis] || lo[axis] > maximum {
            return None;
        }
        clipped_lo[axis] = lo[axis].max(minimum[axis]);
        clipped_hi[axis] = hi[axis].min(maximum);
    }
    Some((clipped_lo, clipped_hi))
}

#[inline]
fn spatial_occupancy_contains(
    key: [i32; 3],
    occupancy: &[u64],
    minimum: [i32; 3],
    dimensions: [usize; 3],
) -> bool {
    let mut local = [0usize; 3];
    for axis in 0..3 {
        if key[axis] < minimum[axis] {
            return false;
        }
        local[axis] = (key[axis] - minimum[axis]) as usize;
        if local[axis] >= dimensions[axis] {
            return false;
        }
    }
    let index = (local[0] * dimensions[1] + local[1]) * dimensions[2] + local[2];
    occupancy[index / u64::BITS as usize] & (1u64 << (index % u64::BITS as usize)) != 0
}

/// Keep the tool's broadphase bins identical to a full scene-position rebuild,
/// while moving only REV indices whose cell key changed. Sorted bucket indices
/// preserve the original deterministic contact traversal order.
fn refresh_spatial_bins(
    positions: &[DVec3],
    bins: &mut HashMap<[i32; 3], SmallVec<[usize; 2]>>,
    keys: &mut Vec<[i32; 3]>,
) {
    if keys.len() != positions.len() {
        bins.clear();
        keys.clear();
        keys.reserve(positions.len());
        for (i, &position) in positions.iter().enumerate() {
            let cell = key(position);
            bins.entry(cell).or_default().push(i);
            keys.push(cell);
        }
        return;
    }

    for (i, &position) in positions.iter().enumerate() {
        let next = key(position);
        let previous = keys[i];
        if next == previous {
            continue;
        }

        let mut remove_previous_bin = false;
        if let Some(indices) = bins.get_mut(&previous) {
            if let Ok(slot) = indices.binary_search(&i) {
                indices.remove(slot);
            }
            remove_previous_bin = indices.is_empty();
        }
        if remove_previous_bin {
            bins.remove(&previous);
        }

        let indices = bins.entry(next).or_default();
        if let Err(slot) = indices.binary_search(&i) {
            indices.insert(slot, i);
        }
        keys[i] = next;
    }
}

/// Grid bins intersecting the tool REV's conservative world-space AABB,
/// expanded by the scene REV half-width and contact tolerance.
fn obb_bin_bounds(center: DVec3, axes: [Vec3; 3]) -> ([i32; 3], [i32; 3]) {
    let half = CELL * 0.5;
    let extent = (axes[0].abs() + axes[1].abs() + axes[2].abs()) * half + Vec3::splat(half + 1e-6);
    (
        key(center - extent.as_dvec3()),
        key(center + extent.as_dvec3()),
    )
}

/// Separating axis test for a rotated tool REV against an axis-aligned scene
/// REV. The 15 SAT axes give a tight contact shell without rounding every tool
/// cell into an over-sized sphere (which blocks wedge entry).
fn obb_aabb_contact(center: DVec3, axes: [Vec3; 3], voxel: DVec3) -> Option<(Vec3, f32, DVec3)> {
    let half = CELL * 0.5;
    let world_axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let delta = (voxel - center).as_vec3();
    let mut best_overlap = f32::INFINITY;
    let mut best_normal = Vec3::ZERO;
    for mut axis in axes.into_iter().chain(world_axes).chain(
        axes.into_iter()
            .flat_map(|a| world_axes.map(move |b| a.cross(b))),
    ) {
        let length = axis.length();
        if length < 1e-6 {
            continue;
        }
        axis /= length;
        let radius_tool = half * axes.iter().map(|u| axis.dot(*u).abs()).sum::<f32>();
        let radius_voxel = half * axis.abs().element_sum();
        let signed_distance = delta.dot(axis);
        let overlap = radius_tool + radius_voxel - signed_distance.abs();
        // Numerical tolerance is 1 µm; without it, a face already projected
        // to contact can chatter due to sub-ulp overlap.
        if overlap <= 1e-6 {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            best_normal = axis * signed_distance.signum();
        }
    }
    if best_normal == Vec3::ZERO {
        return None;
    }
    let closest = axes.into_iter().fold(center, |point, axis| {
        point + (axis * delta.dot(axis).clamp(-half, half)).as_dvec3()
    });
    Some((best_normal, best_overlap, closest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scene::SceneConfig, solver};

    #[test]
    fn tapered_blade_uses_center_of_mass_and_preserves_geometry() {
        let config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        let scene = Scene::new(config);
        let origin = Vec3::new(0.4, 0.2, 0.3);
        let tool = Tool::new(Shape::Blade, &scene, origin, Vec3::Y, 0.0);
        let centroid = tool.cells.iter().copied().sum::<Vec3>() / tool.cells.len() as f32;
        assert!(centroid.length() < 1e-6);
        let dims = scene.config.tool_dimensions;
        let expected_min_x = -(dims[0] as f32 - 1.0) * CELL * 0.5;
        let min_x = tool
            .world_cells()
            .map(|p| (tool.rotation.conjugate() * (p - origin.as_dvec3()).as_vec3()).x)
            .fold(f32::INFINITY, f32::min);
        assert!((min_x - expected_min_x).abs() < 1e-6);
        let cell_mass = tool.mass / tool.cells.len() as f32;
        let inertia = tool.cells.iter().fold(Vec3::ZERO, |sum, &p| {
            sum + (Vec3::splat(p.length_squared() + CELL * CELL / 6.0) - p * p) * cell_mass
        });
        assert!((inertia * tool.inverse_inertia - Vec3::ONE).length() < 1e-5);
    }

    #[test]
    fn external_force_accelerates_and_persists_on_rigid_voxel_group() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [1; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.tool_dimensions = [2; 3];
        let mut scene = Scene::new(config);
        let mut tool = Tool::new(Shape::Blade, &scene, Vec3::splat(1.0), Vec3::X, 0.0);
        tool.set_external_force(Vec3::X * (tool.mass * 3.0))
            .unwrap();
        tool.advance(&mut scene, 0.01, &[]);
        assert!((tool.velocity.x - 0.03).abs() < 1e-6);
        tool.advance(&mut scene, 0.01, &[]);
        assert!((tool.velocity.x - 0.06).abs() < 1e-6);
        tool.clear_external_wrench();
        tool.advance(&mut scene, 0.01, &[]);
        assert!((tool.velocity.x - 0.06).abs() < 1e-6);
    }

    fn kinetic_energy(scene: &Scene, tool: &Tool) -> f64 {
        let wood: f64 = scene
            .voxels
            .iter()
            .filter(|v| !v.kinematic)
            .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
            .sum();
        let rotational_inertia = tool.inverse_inertia.recip();
        let angular_energy = 0.5
            * tool.angular_velocity.dot(
                tool.rotation
                    * (rotational_inertia * (tool.rotation.conjugate() * tool.angular_velocity)),
            );
        wood + 0.5 * tool.mass as f64 * tool.velocity.length_squared() as f64
            + angular_energy as f64
    }

    #[test]
    fn off_center_contact_preserves_linear_and_angular_momentum() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [1; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.tool_dimensions = [2; 3];
        config.fracture.friction = 0.0;
        let mut scene = Scene::new(config);
        let mut tool = Tool::new(
            Shape::Blade,
            &scene,
            Vec3::new(0.0199, 0.006, 0.005),
            -Vec3::X,
            0.1,
        );
        let momentum = |s: &Scene, t: &Tool| {
            let p = t.velocity * t.mass + s.voxels[0].velocity * s.voxels[0].mass;
            let spin = t.rotation
                * (t.inverse_inertia.recip() * (t.rotation.conjugate() * t.angular_velocity));
            // Use the original COM: position projection is a geometric split,
            // not a physical translation carrying angular momentum.
            (p, spin)
        };
        let com = tool.position;
        let before = momentum(&scene, &tool);
        let e0 = kinetic_energy(&scene, &tool);
        tool.advance(&mut scene, 0.0, &[]);
        let after = momentum(&scene, &tool);
        let orbital = (scene.positions[0] - com)
            .cross((scene.voxels[0].velocity * scene.voxels[0].mass).as_dvec3())
            .as_vec3();
        assert!((after.0 - before.0).length() < 1e-7);
        assert!((after.1 + orbital - before.1).length() < 1e-8);
        assert!(scene.voxels[0].velocity.length() > 0.0);
        assert!(kinetic_energy(&scene, &tool) <= e0 * (1.0 + 1e-5));
    }

    #[test]
    fn tool_friction_converts_contact_energy_to_voxel_heat() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [1; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.tool_dimensions = [2; 3];
        config.fracture.friction = 0.35;
        let mut scene = Scene::new(config);
        let mut tool = Tool::new(
            Shape::Blade,
            &scene,
            Vec3::new(0.0199, 0.006, 0.005),
            -Vec3::X,
            0.1,
        );
        tool.angular_velocity = Vec3::Z * 10.0;
        let energy_before = kinetic_energy(&scene, &tool);
        let internal_before = scene.voxels[0].internal_energy as f64;
        tool.advance(&mut scene, 0.0, &[]);

        let friction_heat = scene.energy.contact_friction_heat;
        let internal_gain = scene.voxels[0].internal_energy as f64 - internal_before;
        let thermal_rounding = f32::EPSILON as f64 * internal_before.max(1.0);
        assert!(
            friction_heat > 0.0,
            "off-center sliding generated no friction"
        );
        assert!(kinetic_energy(&scene, &tool) <= energy_before * (1.0 + 1e-5));
        assert!(
            (internal_gain - friction_heat).abs() <= thermal_rounding,
            "internal gain={internal_gain:.12e}, friction heat={friction_heat:.12e}, tolerance={thermal_rounding:.12e}"
        );
    }

    #[test]
    fn one_cube_contact_does_not_turn_overlap_correction_into_energy() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [1; 3];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        config.gravity = [0.0; 3];
        config.fixed_dt = 0.001;
        config.substeps = 1;
        config.corotation_iterations = 1;
        config.tool_dimensions = [2; 3];
        config.fracture.friction = 0.0;
        let mut scene = Scene::new(config);
        let mut tool = Some(Tool::new(
            Shape::Blade,
            &scene,
            Vec3::new(0.0199, 0.005, 0.005),
            -Vec3::X,
            0.1,
        ));
        let initial_energy = kinetic_energy(&scene, tool.as_ref().unwrap());
        solver::step(&mut scene, &mut tool);
        let final_energy = kinetic_energy(&scene, tool.as_ref().unwrap());
        assert!(
            final_energy <= initial_energy * (1.0 + 1e-4),
            "contact added kinetic energy: before={initial_energy} J after={final_energy} J"
        );
        let voxel = scene.positions[0];
        let axes = [
            tool.as_ref().unwrap().rotation * Vec3::X,
            tool.as_ref().unwrap().rotation * Vec3::Y,
            tool.as_ref().unwrap().rotation * Vec3::Z,
        ];
        for center in tool.as_ref().unwrap().world_cells() {
            assert!(
                obb_aabb_contact(center, axes, voxel).is_none(),
                "tool REV penetrated voxel: center={center:?}, voxel={voxel:?}"
            );
        }
    }
}
