use crate::{
    material::Material,
    scene::{Scene, CELL},
    voxel::Composition,
};
use glam::{Quat, Vec3};
use smallvec::SmallVec;
use std::collections::HashMap;

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
    pub world_base: Vec3,
    pub direction: Vec3,
    pub rotation: Quat,
}
/// A rigid voxel object. The collision law never branches on Shape.
pub struct Tool {
    pub cells: Vec<Vec3>,
    spatial_bins: HashMap<[i32; 3], SmallVec<[usize; 2]>>,
    center_cache: Vec<Vec3>,
    pub composition: Composition,
    pub mass: f32,
    pub position: Vec3,
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
    pub fn new(shape: Shape, scene: &Scene, position: Vec3, direction: Vec3, speed: f32) -> Self {
        let d = scene.config.tool_dimensions;
        let (dims, component, color) = match shape {
            Shape::Wedge => (d, 2, [0.65, 0.68, 0.72, 1.0]),
            // Preserve a 5 REV spine and taper it to one REV at the edge.
            // A single-row slab lost the bevel entirely and behaved like a
            // flat plate that stalled against the first wood layer.
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
                            let bluntness = if matches!(shape, Shape::Wedge) {
                                0.5
                            } else {
                                0.0
                            };
                            p.y.abs() <= thickness + bluntness
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
        let mut inertia = Vec3::ZERO;
        for &p in &cells {
            inertia += (Vec3::splat(p.length_squared() + CELL * CELL / 6.0) - p * p) * cell_mass;
        }
        Self {
            cells,
            spatial_bins: HashMap::new(),
            center_cache: Vec::new(),
            composition,
            mass,
            position: position + rotation * centroid,
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
            world_base: self.position + self.rotation * local_base,
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
            let base = self.position + self.rotation * m.local_base;
            self.position +=
                m.world_base + m.direction * (base - m.world_base).dot(m.direction) - base;
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
    pub fn world_cells(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.cells
            .iter()
            .map(|&p| self.position + self.rotation * p)
    }
    fn angular_response(&self, torque: Vec3) -> Vec3 {
        if self.mount.is_some() {
            Vec3::ZERO
        } else {
            self.rotation * (self.inverse_inertia * (self.rotation.conjugate() * torque))
        }
    }
    /// Finite rigid mass, friction and angular impulse, with equal/opposite reaction.
    /// Microsteps prevent tunnelling through a 1 cm REV at typical strike speeds.
    pub fn advance(
        &mut self,
        scene: &mut Scene,
        dt: f32,
        targets: &[[Option<f32>; 3]],
    ) -> Vec<Vec3> {
        self.constrain_pose();
        let mut contact_velocity = vec![Vec3::ZERO; scene.positions.len()];
        let tool_radius = self.cells.iter().map(|p| p.length()).fold(0.0, f32::max);
        let gravity = Vec3::from_array(scene.config.gravity);
        let linear_acceleration =
            self.project_translation(gravity + self.external_force / self.mass);
        let angular_acceleration = self.angular_response(self.external_torque);
        let linear_distance =
            self.velocity.length() * dt + 0.5 * linear_acceleration.length() * dt * dt;
        let angular_distance = (self.angular_velocity.length() * dt
            + 0.5 * angular_acceleration.length() * dt * dt)
            * tool_radius;
        let linear_steps = linear_distance / (CELL * 0.4);
        let angular_steps = angular_distance / (CELL * 0.4);
        let steps = (linear_steps.max(angular_steps).ceil() as usize).max(1);
        let h = dt / steps as f32;
        let mut spatial = std::mem::take(&mut self.spatial_bins);
        let mut centers_cache = std::mem::take(&mut self.center_cache);
        for _ in 0..steps {
            self.velocity += linear_acceleration * h;
            self.angular_velocity += self.angular_response(self.external_torque) * h;
            self.position += self.velocity * h;
            let angle = self.angular_velocity.length() * h;
            if angle > 1e-8 {
                self.rotation = (Quat::from_axis_angle(self.angular_velocity.normalize(), angle)
                    * self.rotation)
                    .normalize();
            }
            spatial.clear();
            for (i, &p) in scene.positions.iter().enumerate() {
                spatial.entry(key(p)).or_default().push(i);
            }
            let axes = [
                self.rotation * Vec3::X,
                self.rotation * Vec3::Y,
                self.rotation * Vec3::Z,
            ];
            centers_cache.clear();
            centers_cache.extend(self.world_cells());
            // Several adjacent tool cells can overlap the same REV during one
            // microstep. Accumulate only the deepest correction; applying the
            // full depenetration once per pair would push the whole tool back
            // by the number of contacts and prevent it from loading the wood.
            let mut correction = Vec3::ZERO;
            let mut max_penetration = 0.0f32;
            for &center in &centers_cache {
                let (lo, hi) = obb_bin_bounds(center, axes);
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            let Some(indices) = spatial.get(&[x, y, z]) else {
                                continue;
                            };
                            for &i in indices {
                                let p = scene.positions[i];
                                let Some((n, penetration, _contact_point)) =
                                    obb_aabb_contact(center, axes, p)
                                else {
                                    continue;
                                };
                                // The scene REV carries linear momentum at its node.
                                // Use that same point for the rigid reaction, otherwise
                                // the equal/opposite pair creates angular momentum.
                                let arm = p - self.position;
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
                                    // energy as the active normal changes. Use an inelastic
                                    // normal solve here; friction and fracture dissipate it.
                                    let normal_impulse = closing / denom.max(1e-9);
                                    let tangent = rel - closing * n;
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
                                    let friction = tangent_dir
                                        * (tangent.length() / tangent_denom.max(1e-9))
                                            .min(scene.config.fracture.friction * normal_impulse);
                                    let impulse = normal_impulse * n + friction;
                                    let voxel_delta_v = impulse * free / scene.voxels[i].mass;
                                    scene.voxels[i].velocity += voxel_delta_v;
                                    contact_velocity[i] += voxel_delta_v;
                                    self.velocity -= self.linear_response(impulse);
                                    self.angular_velocity -=
                                        self.angular_response(arm.cross(impulse));
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
            self.position += correction;
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
                self.position += correction;
                self.constrain_pose();
            }
        }
        self.spatial_bins = spatial;
        self.center_cache = centers_cache;
        contact_velocity
    }
}
fn key(p: Vec3) -> [i32; 3] {
    (p / CELL).floor().as_ivec3().to_array()
}

/// Grid bins intersecting the tool REV's conservative world-space AABB,
/// expanded by the scene REV half-width and contact tolerance.
fn obb_bin_bounds(center: Vec3, axes: [Vec3; 3]) -> ([i32; 3], [i32; 3]) {
    let half = CELL * 0.5;
    let extent = (axes[0].abs() + axes[1].abs() + axes[2].abs()) * half + Vec3::splat(half + 1e-6);
    (key(center - extent), key(center + extent))
}

/// Separating axis test for a rotated tool REV against an axis-aligned scene
/// REV. The 15 SAT axes give a tight contact shell without rounding every tool
/// cell into an over-sized sphere (which blocks wedge entry).
fn obb_aabb_contact(center: Vec3, axes: [Vec3; 3], voxel: Vec3) -> Option<(Vec3, f32, Vec3)> {
    let half = CELL * 0.5;
    let world_axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let delta = voxel - center;
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
        point + axis * delta.dot(axis).clamp(-half, half)
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
            .map(|p| (tool.rotation.conjugate() * (p - origin)).x)
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
        let orbital =
            (scene.positions[0] - com).cross(scene.voxels[0].velocity * scene.voxels[0].mass);
        assert!((after.0 - before.0).length() < 1e-7);
        assert!((after.1 + orbital - before.1).length() < 1e-8);
        assert!(scene.voxels[0].velocity.length() > 0.0);
        assert!(kinetic_energy(&scene, &tool) <= e0 * (1.0 + 1e-5));
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
