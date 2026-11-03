//! Deterministic conversion of static OBJ/GLB triangle meshes to mechanical REVs.
//! The voxelizer uses triangle/cell SAT for the shell and a 6-neighbour flood fill
//! for the interior of a watertight mesh. It never substitutes a bounding box for
//! mesh geometry.

use glam::{DVec3, Mat4, Quat, Vec3};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::Path,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelizationMode {
    Shell,
    Filled,
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelizationOptions {
    /// The mechanics grid is currently fixed at 1 cm by the engine.
    pub cell_size_m: f64,
    /// User-editable, positive scale applied after source units are converted.
    pub scale: DVec3,
    pub mode: VoxelizationMode,
    pub max_voxels: usize,
    pub max_grid_cells: usize,
    pub max_triangle_cell_tests: u64,
}

impl Default for VoxelizationOptions {
    fn default() -> Self {
        Self {
            cell_size_m: crate::coordinates::CELL_M,
            scale: DVec3::ONE,
            mode: VoxelizationMode::Shell,
            max_voxels: 4_000_000,
            max_grid_cells: 16_000_000,
            max_triangle_cell_tests: 40_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoxelizationPreview {
    /// Coordinates are local to `grid_origin_m`; order is deterministic Z/Y/X.
    pub cells: Vec<[u32; 3]>,
    pub grid_origin_m: DVec3,
    pub cell_size_m: f64,
    pub source_bounds_min_m: DVec3,
    pub source_bounds_max_m: DVec3,
    pub grid_dimensions: [u32; 3],
    pub estimated_grid_cells: u64,
    pub is_watertight: bool,
    pub mode: VoxelizationMode,
}

impl VoxelizationPreview {
    /// Convert this preview into a reusable native `.vxm` model document.
    /// Each preview cell becomes one REV centered on the 1 cm lattice, with
    /// the supplied material stored as the model's sole composition component.
    pub fn to_model_document(
        &self,
        name: impl Into<String>,
        material: crate::material::Material,
    ) -> Result<crate::documents::VoxelModelDocument, String> {
        use crate::{
            documents::{EditorDocument, ModelVoxel, VoxelModelDocument},
            voxel::{Composition, Microstructure, Voxel},
        };

        let name = name.into();
        if name.trim().is_empty() || name.len() > 4096 {
            return Err("model name must contain 1..=4096 UTF-8 bytes".into());
        }
        let cell_size_m = crate::coordinates::CELL_M;
        if !self.cell_size_m.is_finite() || (self.cell_size_m - cell_size_m).abs() > 1e-12 {
            return Err("mesh preview must use the engine's 1 cm REV grid".into());
        }
        if self.cells.is_empty() {
            return Err("mesh preview contains no voxel cells".into());
        }
        if self.grid_dimensions.contains(&0)
            || !self.grid_origin_m.is_finite()
            || !self.source_bounds_min_m.is_finite()
            || !self.source_bounds_max_m.is_finite()
            || self
                .source_bounds_min_m
                .cmpgt(self.source_bounds_max_m)
                .any()
        {
            return Err("mesh preview has invalid grid bounds or dimensions".into());
        }
        let lattice_origin = self.grid_origin_m / cell_size_m;
        if (lattice_origin - lattice_origin.round())
            .abs()
            .max_element()
            > 1e-9
        {
            return Err("mesh preview grid origin is not aligned to the 1 cm REV lattice".into());
        }
        if material.name.trim().is_empty() {
            return Err("model material must have a non-empty name".into());
        }
        if !material.density.is_finite() || material.density <= 0.0 {
            return Err("model material density must be finite and positive".into());
        }

        let mut seen = HashSet::with_capacity(self.cells.len());
        let mass = material.density * (cell_size_m as f32).powi(3);
        if !mass.is_finite() || mass <= 0.0 {
            return Err("model material produces an invalid REV mass".into());
        }
        let composition = Composition::pure(0);
        let internal_energy = material.energy_at_temperature(mass, 293.15);
        if !internal_energy.is_finite() || internal_energy < 0.0 {
            return Err("model material produces an invalid initial thermal state".into());
        }
        let mut voxels = Vec::with_capacity(self.cells.len());
        for cell in &self.cells {
            if cell
                .iter()
                .zip(self.grid_dimensions)
                .any(|(&coordinate, dimension)| coordinate >= dimension)
            {
                return Err(format!(
                    "mesh preview cell {cell:?} is outside its grid dimensions"
                ));
            }
            if !seen.insert(*cell) {
                return Err(format!("mesh preview contains duplicate cell {cell:?}"));
            }
            let position_m = self.grid_origin_m
                + DVec3::new(
                    cell[0] as f64 + 0.5,
                    cell[1] as f64 + 0.5,
                    cell[2] as f64 + 0.5,
                ) * cell_size_m;
            let state = Voxel {
                mass,
                kinematic: false,
                composition: composition.clone(),
                internal_energy,
                internal_energy_remainder_j: 0.0,
                thermal_cell_energy_j: None,
                velocity: Vec3::ZERO,
                microstructure: Microstructure {
                    orientation: Quat::IDENTITY,
                    anisotropy: 0.0,
                },
            };
            voxels.push(ModelVoxel {
                position_m,
                rest_position_m: position_m,
                state,
            });
        }

        let document = VoxelModelDocument {
            name,
            cell_size_m,
            materials: vec![material],
            voxels,
            groups: Vec::new(),
            motion_constraints: Vec::new(),
            skeleton: None,
        };
        document.validate()?;
        Ok(document)
    }

    /// Convert preview cells to the engine's 1 cm REV authoring API.
    pub fn to_spawns(&self, material_component: u16) -> Vec<crate::scene::VoxelSpawn> {
        self.cells
            .iter()
            .map(|cell| {
                let center = self.grid_origin_m
                    + DVec3::new(
                        cell[0] as f64 + 0.5,
                        cell[1] as f64 + 0.5,
                        cell[2] as f64 + 0.5,
                    ) * self.cell_size_m;
                crate::scene::VoxelSpawn::new_precise(center, material_component)
            })
            .collect()
    }

    /// Open the converted geometry as a separate physical scene.
    pub fn build_scene(
        &self,
        config: crate::scene::SceneConfig,
        material_component: u16,
    ) -> Result<crate::scene::Scene, String> {
        crate::scene::Scene::from_voxels(config, self.to_spawns(material_component))
    }
}

#[derive(Clone, Debug)]
struct TriangleMesh {
    triangles: Vec<[DVec3; 3]>,
}

#[derive(Clone, Copy, Debug)]
struct Grid {
    min_cell: [i64; 3],
    dimensions: [usize; 3],
    origin_m: DVec3,
    cell_size_m: f64,
    volume: u64,
}

pub fn import_obj(
    path: &Path,
    options: VoxelizationOptions,
) -> Result<VoxelizationPreview, String> {
    let mesh = load_obj(path, options.scale)?;
    voxelize(mesh, options)
}

pub fn import_glb(
    path: &Path,
    options: VoxelizationOptions,
) -> Result<VoxelizationPreview, String> {
    let mesh = load_glb(path, options.scale)?;
    voxelize(mesh, options)
}

pub fn import_mesh(
    path: &Path,
    options: VoxelizationOptions,
) -> Result<VoxelizationPreview, String> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("obj") => import_obj(path, options),
        Some(extension) if extension.eq_ignore_ascii_case("glb") => import_glb(path, options),
        _ => Err("supported mesh formats are OBJ and binary GLB".into()),
    }
}

fn load_obj(path: &Path, scale: DVec3) -> Result<TriangleMesh, String> {
    validate_scale_and_cell_size(scale, crate::coordinates::CELL_M)?;
    let options = tobj::LoadOptions {
        triangulate: false,
        ignore_points: true,
        ignore_lines: true,
        ..Default::default()
    };
    let (models, _materials) = tobj::load_obj(path, &options)
        .map_err(|error| format!("could not read OBJ mesh: {error}"))?;
    let mut triangles = Vec::new();
    for model in models {
        let mesh = model.mesh;
        if mesh.positions.len() % 3 != 0 {
            return Err("OBJ contains an incomplete vertex position".into());
        }
        let positions = mesh
            .positions
            .chunks_exact(3)
            .map(|v| DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64) * scale)
            .collect::<Vec<_>>();
        if mesh.indices.is_empty() {
            continue;
        }
        let arities = if mesh.face_arities.is_empty() {
            vec![3_u32; mesh.indices.len() / 3]
        } else {
            mesh.face_arities.clone()
        };
        let mut offset = 0_usize;
        for arity in arities {
            let arity = arity as usize;
            let end = offset
                .checked_add(arity)
                .ok_or_else(|| "OBJ face index overflow".to_string())?;
            let face_indices = mesh
                .indices
                .get(offset..end)
                .ok_or_else(|| "OBJ face index list is truncated".to_string())?;
            offset = end;
            if arity < 3 {
                continue;
            }
            let polygon = face_indices
                .iter()
                .map(|&index| {
                    positions
                        .get(index as usize)
                        .copied()
                        .ok_or_else(|| "OBJ face references a missing vertex".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            triangulate_polygon(&polygon, &mut triangles)?;
        }
        if offset != mesh.indices.len() {
            return Err("OBJ face arities do not match the index list".into());
        }
    }
    TriangleMesh::new(triangles)
}

fn load_glb(path: &Path, scale: DVec3) -> Result<TriangleMesh, String> {
    validate_scale_and_cell_size(scale, crate::coordinates::CELL_M)?;
    let (document, buffers, _images) =
        gltf::import(path).map_err(|error| format!("could not read GLB buffers: {error}"))?;
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or_else(|| "GLB has no scene to import".to_string())?;
    let mut triangles = Vec::new();
    for node in scene.nodes() {
        visit_gltf_node(node, Mat4::IDENTITY, &buffers, scale, &mut triangles)?;
    }
    TriangleMesh::new(triangles)
}

fn visit_gltf_node(
    node: gltf::Node<'_>,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    scale: DVec3,
    triangles: &mut Vec<[DVec3; 3]>,
) -> Result<(), String> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let transform = parent_transform * local;
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(format!(
                    "GLB mesh {} has a non-triangle primitive",
                    mesh.index()
                ));
            }
            let reader = primitive
                .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
            let positions = reader
                .read_positions()
                .ok_or_else(|| "GLB triangle primitive has no POSITION attribute".to_string())?
                .map(|position| {
                    let transformed = transform.transform_point3(Vec3::from_array(position));
                    DVec3::new(
                        transformed.x as f64 * scale.x,
                        transformed.y as f64 * scale.y,
                        transformed.z as f64 * scale.z,
                    )
                })
                .collect::<Vec<_>>();
            let indices = reader
                .read_indices()
                .map(|indices| indices.into_u32().collect::<Vec<_>>())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            if indices.len() % 3 != 0 {
                return Err("GLB triangle index count is not divisible by three".into());
            }
            for face in indices.chunks_exact(3) {
                let a = *positions
                    .get(face[0] as usize)
                    .ok_or_else(|| "GLB index references a missing position".to_string())?;
                let b = *positions
                    .get(face[1] as usize)
                    .ok_or_else(|| "GLB index references a missing position".to_string())?;
                let c = *positions
                    .get(face[2] as usize)
                    .ok_or_else(|| "GLB index references a missing position".to_string())?;
                triangles.push([a, b, c]);
            }
        }
    }
    for child in node.children() {
        visit_gltf_node(child, transform, buffers, scale, triangles)?;
    }
    Ok(())
}

impl TriangleMesh {
    fn new(triangles: Vec<[DVec3; 3]>) -> Result<Self, String> {
        if triangles.is_empty() {
            return Err("mesh contains no triangle geometry".into());
        }
        for triangle in &triangles {
            if triangle.iter().any(|point| !point.is_finite()) {
                return Err("mesh contains a non-finite vertex".into());
            }
            let ab = triangle[1] - triangle[0];
            let ac = triangle[2] - triangle[0];
            let area_twice = ab.cross(ac).length();
            if !area_twice.is_finite() || area_twice <= 1e-18 {
                return Err("mesh contains a degenerate triangle".into());
            }
        }
        Ok(Self { triangles })
    }

    fn bounds(&self) -> (DVec3, DVec3) {
        self.triangles.iter().flatten().fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(min, max), point| (min.min(*point), max.max(*point)),
        )
    }
}

fn voxelize(
    mesh: TriangleMesh,
    options: VoxelizationOptions,
) -> Result<VoxelizationPreview, String> {
    validate_options(options)?;
    let (bounds_min, bounds_max) = mesh.bounds();
    let watertight = is_watertight(&mesh.triangles, bounds_min, bounds_max);
    let grid = Grid::from_bounds(bounds_min, bounds_max, options.cell_size_m)?;
    if options.mode == VoxelizationMode::Filled && grid.volume > options.max_grid_cells as u64 {
        return Err(format!(
            "filled voxel bounds require {} grid cells; limit is {}",
            grid.volume, options.max_grid_cells
        ));
    }

    let mut tests = 0_u64;
    let cells = match options.mode {
        VoxelizationMode::Shell => {
            let mut surface = HashSet::new();
            for triangle in &mesh.triangles {
                rasterize_triangle(
                    *triangle,
                    grid,
                    options.max_triangle_cell_tests,
                    &mut tests,
                    |index| {
                        surface.insert(index);
                        if surface.len() > options.max_voxels {
                            return Err("voxel shell exceeds the configured REV limit".into());
                        }
                        Ok(())
                    },
                )?;
            }
            surface.into_iter().collect::<Vec<_>>()
        }
        VoxelizationMode::Filled => {
            let mut occupancy = vec![0_u8; grid.volume as usize];
            for triangle in &mesh.triangles {
                rasterize_triangle(
                    *triangle,
                    grid,
                    options.max_triangle_cell_tests,
                    &mut tests,
                    |index| {
                        occupancy[linear_index(grid.dimensions, index)] = 1;
                        Ok(())
                    },
                )?;
            }
            let enclosed_cells = fill_interior(grid.dimensions, &mut occupancy);
            if !watertight && enclosed_cells == 0 {
                return Err(
                    "no enclosed voxel volume at this cell size; use Shell or repair the mesh"
                        .into(),
                );
            }
            let filled_count = occupancy.iter().filter(|&&value| value == 1).count();
            if filled_count > options.max_voxels {
                return Err(format!(
                    "filled model contains {filled_count} REVs; limit is {}",
                    options.max_voxels
                ));
            }
            occupancy
                .iter()
                .enumerate()
                .filter_map(|(linear, &value)| {
                    (value == 1).then(|| decode_index(grid.dimensions, linear))
                })
                .collect()
        }
    };

    if cells.is_empty() {
        return Err("mesh did not cover any mechanical REV".into());
    }
    let mut cells = cells;
    cells.sort_unstable_by_key(|cell| (cell[2], cell[1], cell[0]));
    cells.dedup();
    if cells.len() > options.max_voxels {
        return Err("voxel model exceeds the configured REV limit".into());
    }
    Ok(VoxelizationPreview {
        cells,
        grid_origin_m: grid.origin_m,
        cell_size_m: grid.cell_size_m,
        source_bounds_min_m: bounds_min,
        source_bounds_max_m: bounds_max,
        grid_dimensions: [
            grid.dimensions[0] as u32,
            grid.dimensions[1] as u32,
            grid.dimensions[2] as u32,
        ],
        estimated_grid_cells: grid.volume,
        is_watertight: watertight,
        mode: options.mode,
    })
}

impl Grid {
    fn from_bounds(min: DVec3, max: DVec3, cell_size_m: f64) -> Result<Self, String> {
        let mut min_cell = [0_i64; 3];
        let mut max_cell = [0_i64; 3];
        for axis in 0..3 {
            let low = (min[axis] / cell_size_m).floor();
            let high = (max[axis] / cell_size_m).floor();
            if !low.is_finite() || !high.is_finite() || low.abs() > i64::MAX as f64 / 4.0 {
                return Err("mesh coordinates exceed the supported world range".into());
            }
            min_cell[axis] = low as i64 - 1;
            max_cell[axis] = high as i64 + 1;
        }
        let mut dimensions = [0_usize; 3];
        for axis in 0..3 {
            let length = max_cell[axis]
                .checked_sub(min_cell[axis])
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| "mesh voxel bounds overflow".to_string())?;
            dimensions[axis] = usize::try_from(length)
                .map_err(|_| "mesh voxel dimension exceeds addressable memory".to_string())?;
            if dimensions[axis] > u32::MAX as usize {
                return Err("mesh voxel dimension exceeds the supported REV index".into());
            }
        }
        let volume = (dimensions[0] as u64)
            .checked_mul(dimensions[1] as u64)
            .and_then(|value| value.checked_mul(dimensions[2] as u64))
            .ok_or_else(|| "mesh voxel volume overflows".to_string())?;
        Ok(Self {
            min_cell,
            dimensions,
            origin_m: DVec3::new(
                min_cell[0] as f64 * cell_size_m,
                min_cell[1] as f64 * cell_size_m,
                min_cell[2] as f64 * cell_size_m,
            ),
            cell_size_m,
            volume,
        })
    }

    fn relative(self, cell: [i64; 3]) -> Option<[u32; 3]> {
        let mut index = [0_u32; 3];
        for axis in 0..3 {
            let relative = cell[axis].checked_sub(self.min_cell[axis])?;
            if relative < 0 || relative as u64 >= self.dimensions[axis] as u64 {
                return None;
            }
            index[axis] = relative as u32;
        }
        Some(index)
    }
}

fn rasterize_triangle(
    triangle: [DVec3; 3],
    grid: Grid,
    max_tests: u64,
    tests: &mut u64,
    mut visit: impl FnMut([u32; 3]) -> Result<(), String>,
) -> Result<(), String> {
    let triangle_min = triangle[0].min(triangle[1]).min(triangle[2]);
    let triangle_max = triangle[0].max(triangle[1]).max(triangle[2]);
    let mut low = [0_i64; 3];
    let mut high = [0_i64; 3];
    for axis in 0..3 {
        low[axis] =
            ((triangle_min[axis] / grid.cell_size_m).floor() as i64 - 1).max(grid.min_cell[axis]);
        high[axis] = ((triangle_max[axis] / grid.cell_size_m).floor() as i64)
            .min(grid.min_cell[axis] + grid.dimensions[axis] as i64 - 1);
    }
    if (0..3).any(|axis| low[axis] > high[axis]) {
        return Ok(());
    }
    let edge_a = triangle[1] - triangle[0];
    let edge_b = triangle[2] - triangle[1];
    let edge_c = triangle[0] - triangle[2];
    let normal = edge_a.cross(triangle[2] - triangle[0]);
    let half = DVec3::splat(grid.cell_size_m * 0.5);
    let basis = [DVec3::X, DVec3::Y, DVec3::Z];
    for z in low[2]..=high[2] {
        for y in low[1]..=high[1] {
            for x in low[0]..=high[0] {
                *tests = tests.saturating_add(1);
                if *tests > max_tests {
                    return Err("mesh voxelization exceeded its triangle-cell work limit".into());
                }
                let index = grid
                    .relative([x, y, z])
                    .expect("candidate cell lies in grid");
                let center = grid.origin_m
                    + DVec3::new(
                        index[0] as f64 + 0.5,
                        index[1] as f64 + 0.5,
                        index[2] as f64 + 0.5,
                    ) * grid.cell_size_m;
                let local = [
                    triangle[0] - center,
                    triangle[1] - center,
                    triangle[2] - center,
                ];
                let mut overlaps = true;
                for axis in basis {
                    if separated_on_axis(axis, local, half) {
                        overlaps = false;
                        break;
                    }
                }
                if overlaps && !separated_on_axis(normal, local, half) {
                    for edge in [edge_a, edge_b, edge_c] {
                        for box_axis in basis {
                            if separated_on_axis(edge.cross(box_axis), local, half) {
                                overlaps = false;
                                break;
                            }
                        }
                        if !overlaps {
                            break;
                        }
                    }
                } else {
                    overlaps = false;
                }
                if overlaps {
                    visit(index)?;
                }
            }
        }
    }
    Ok(())
}

fn separated_on_axis(axis: DVec3, triangle: [DVec3; 3], half: DVec3) -> bool {
    if axis.length_squared() <= 1e-28 {
        return false;
    }
    let p0 = axis.dot(triangle[0]);
    let p1 = axis.dot(triangle[1]);
    let p2 = axis.dot(triangle[2]);
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    let radius = half.dot(axis.abs());
    let tolerance = 1e-12 * (radius.abs() + min.abs().max(max.abs())) + 1e-15;
    min > radius + tolerance || max < -radius - tolerance
}

fn fill_interior(dimensions: [usize; 3], occupancy: &mut [u8]) -> usize {
    let mut outside = VecDeque::new();
    let mut add_outside = |x: usize, y: usize, z: usize| {
        let index = (z * dimensions[1] + y) * dimensions[0] + x;
        if occupancy[index] == 0 {
            occupancy[index] = 2;
            outside.push_back(index);
        }
    };
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            add_outside(0, y, z);
            add_outside(dimensions[0] - 1, y, z);
        }
    }
    for z in 0..dimensions[2] {
        for x in 0..dimensions[0] {
            add_outside(x, 0, z);
            add_outside(x, dimensions[1] - 1, z);
        }
    }
    for y in 0..dimensions[1] {
        for x in 0..dimensions[0] {
            add_outside(x, y, 0);
            add_outside(x, y, dimensions[2] - 1);
        }
    }
    drop(add_outside);

    while let Some(index) = outside.pop_front() {
        let x = index % dimensions[0];
        let yz = index / dimensions[0];
        let y = yz % dimensions[1];
        let z = yz / dimensions[1];
        if x > 0 {
            mark_outside(index - 1, occupancy, &mut outside);
        }
        if x + 1 < dimensions[0] {
            mark_outside(index + 1, occupancy, &mut outside);
        }
        if y > 0 {
            mark_outside(index - dimensions[0], occupancy, &mut outside);
        }
        if y + 1 < dimensions[1] {
            mark_outside(index + dimensions[0], occupancy, &mut outside);
        }
        let plane = dimensions[0] * dimensions[1];
        if z > 0 {
            mark_outside(index - plane, occupancy, &mut outside);
        }
        if z + 1 < dimensions[2] {
            mark_outside(index + plane, occupancy, &mut outside);
        }
    }
    let mut enclosed_cells = 0;
    for cell in occupancy {
        if *cell == 0 {
            *cell = 1;
            enclosed_cells += 1;
        }
    }
    enclosed_cells
}

fn mark_outside(index: usize, occupancy: &mut [u8], queue: &mut VecDeque<usize>) {
    if occupancy[index] == 0 {
        occupancy[index] = 2;
        queue.push_back(index);
    }
}

fn linear_index(dimensions: [usize; 3], cell: [u32; 3]) -> usize {
    (cell[2] as usize * dimensions[1] + cell[1] as usize) * dimensions[0] + cell[0] as usize
}

fn decode_index(dimensions: [usize; 3], linear: usize) -> [u32; 3] {
    let x = linear % dimensions[0];
    let yz = linear / dimensions[0];
    let y = yz % dimensions[1];
    let z = yz / dimensions[1];
    [x as u32, y as u32, z as u32]
}

fn is_watertight(triangles: &[[DVec3; 3]], bounds_min: DVec3, bounds_max: DVec3) -> bool {
    let epsilon = ((bounds_max - bounds_min).max_element() * 1e-9).max(1e-12);
    let mut edges: HashMap<([i64; 3], [i64; 3]), u8> = HashMap::new();
    for triangle in triangles {
        let points = triangle.map(|point| quantize_point(point, epsilon));
        if points[0] == points[1] || points[1] == points[2] || points[2] == points[0] {
            return false;
        }
        for (a, b) in [
            (points[0], points[1]),
            (points[1], points[2]),
            (points[2], points[0]),
        ] {
            let edge = if a <= b { (a, b) } else { (b, a) };
            let count = edges.entry(edge).or_default();
            *count = count.saturating_add(1);
            if *count > 2 {
                return false;
            }
        }
    }
    !edges.is_empty() && edges.values().all(|&count| count == 2)
}

fn quantize_point(point: DVec3, epsilon: f64) -> [i64; 3] {
    [
        (point.x / epsilon).round() as i64,
        (point.y / epsilon).round() as i64,
        (point.z / epsilon).round() as i64,
    ]
}

fn triangulate_polygon(polygon: &[DVec3], triangles: &mut Vec<[DVec3; 3]>) -> Result<(), String> {
    if polygon.len() < 3 {
        return Ok(());
    }
    if polygon.iter().any(|point| !point.is_finite()) {
        return Err("OBJ polygon contains a non-finite vertex".into());
    }
    if polygon.len() == 3 {
        triangles.push([polygon[0], polygon[1], polygon[2]]);
        return Ok(());
    }
    let normal = polygon
        .iter()
        .enumerate()
        .fold(DVec3::ZERO, |sum, (i, point)| {
            let next = polygon[(i + 1) % polygon.len()];
            sum + DVec3::new(
                (point.y - next.y) * (point.z + next.z),
                (point.z - next.z) * (point.x + next.x),
                (point.x - next.x) * (point.y + next.y),
            )
        });
    let dominant = if normal.x.abs() >= normal.y.abs() && normal.x.abs() >= normal.z.abs() {
        0
    } else if normal.y.abs() >= normal.z.abs() {
        1
    } else {
        2
    };
    if normal.length_squared() <= 1e-24 {
        return Err("OBJ polygon is degenerate".into());
    }
    let project = |point: DVec3| match dominant {
        0 => [point.y, point.z],
        1 => [point.x, point.z],
        _ => [point.x, point.y],
    };
    let projected = polygon.iter().copied().map(project).collect::<Vec<_>>();
    let signed_area = (0..projected.len())
        .map(|i| {
            let a = projected[i];
            let b = projected[(i + 1) % projected.len()];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        * 0.5;
    if signed_area.abs() <= 1e-18 {
        return Err("OBJ polygon has zero projected area".into());
    }
    let orientation = signed_area.signum();
    let mut remaining = (0..polygon.len()).collect::<Vec<_>>();
    let mut attempts = 0;
    while remaining.len() > 3 {
        let mut ear = None;
        for position in 0..remaining.len() {
            let prev = remaining[(position + remaining.len() - 1) % remaining.len()];
            let curr = remaining[position];
            let next = remaining[(position + 1) % remaining.len()];
            if cross2(projected[prev], projected[curr], projected[next]) * orientation <= 1e-14 {
                continue;
            }
            let contains_vertex = remaining.iter().copied().any(|other| {
                other != prev
                    && other != curr
                    && other != next
                    && point_in_triangle_2d(
                        projected[other],
                        projected[prev],
                        projected[curr],
                        projected[next],
                        orientation,
                    )
            });
            if !contains_vertex {
                ear = Some((position, prev, curr, next));
                break;
            }
        }
        let Some((position, prev, curr, next)) = ear else {
            return Err("OBJ polygon is self-intersecting or cannot be triangulated".into());
        };
        triangles.push([polygon[prev], polygon[curr], polygon[next]]);
        remaining.remove(position);
        attempts += 1;
        if attempts > polygon.len() * polygon.len() {
            return Err("OBJ polygon triangulation exceeded its work limit".into());
        }
    }
    triangles.push([
        polygon[remaining[0]],
        polygon[remaining[1]],
        polygon[remaining[2]],
    ]);
    Ok(())
}

fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn point_in_triangle_2d(
    point: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    orientation: f64,
) -> bool {
    cross2(a, b, point) * orientation >= -1e-14
        && cross2(b, c, point) * orientation >= -1e-14
        && cross2(c, a, point) * orientation >= -1e-14
}

fn validate_scale_and_cell_size(scale: DVec3, cell_size_m: f64) -> Result<(), String> {
    if !scale.is_finite() || scale.min_element() <= 0.0 {
        return Err("model scale must be finite and greater than zero on every axis".into());
    }
    if (cell_size_m - crate::coordinates::CELL_M).abs() > 1e-12 {
        return Err("the mechanics grid currently uses fixed 1 cm REVs".into());
    }
    Ok(())
}

fn validate_options(options: VoxelizationOptions) -> Result<(), String> {
    validate_scale_and_cell_size(options.scale, options.cell_size_m)?;
    if options.max_voxels == 0
        || options.max_grid_cells == 0
        || options.max_triangle_cell_tests == 0
    {
        return Err("voxelization limits must be greater than zero".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tetrahedron() -> TriangleMesh {
        TriangleMesh::new(vec![
            [
                DVec3::new(0.02, 0.02, 0.02),
                DVec3::new(0.32, 0.02, 0.02),
                DVec3::new(0.02, 0.32, 0.02),
            ],
            [
                DVec3::new(0.02, 0.02, 0.02),
                DVec3::new(0.02, 0.32, 0.02),
                DVec3::new(0.02, 0.02, 0.32),
            ],
            [
                DVec3::new(0.02, 0.02, 0.02),
                DVec3::new(0.02, 0.02, 0.32),
                DVec3::new(0.32, 0.02, 0.02),
            ],
            [
                DVec3::new(0.32, 0.02, 0.02),
                DVec3::new(0.02, 0.02, 0.32),
                DVec3::new(0.02, 0.32, 0.02),
            ],
        ])
        .unwrap()
    }

    #[test]
    fn shell_voxelization_is_deterministic_and_covers_triangle_faces() {
        let mesh = tetrahedron();
        let options = VoxelizationOptions::default();
        let first = voxelize(mesh.clone(), options).unwrap();
        let second = voxelize(mesh, options).unwrap();
        assert!(first.is_watertight);
        assert_eq!(first.cells, second.cells);
        assert!(!first.cells.is_empty());
        assert!(first.cells.windows(2).all(|pair| {
            (pair[0][2], pair[0][1], pair[0][0]) < (pair[1][2], pair[1][1], pair[1][0])
        }));
    }

    #[test]
    fn closed_tetrahedron_fills_inside_but_open_plane_is_rejected() {
        let mut options = VoxelizationOptions::default();
        let shell = voxelize(tetrahedron(), options).unwrap();
        options.mode = VoxelizationMode::Filled;
        let filled = voxelize(tetrahedron(), options).unwrap();
        assert!(filled.cells.len() > shell.cells.len());
        assert!(filled.is_watertight);

        let open = TriangleMesh::new(vec![[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.1, 0.0, 0.0),
            DVec3::new(0.0, 0.1, 0.0),
        ]])
        .unwrap();
        assert!(voxelize(open.clone(), VoxelizationOptions::default()).is_ok());
        assert!(voxelize(open, options).unwrap_err().contains("watertight"));
    }

    #[test]
    fn ear_clipping_handles_a_concave_obj_polygon() {
        let polygon = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.1, 0.0, 0.0),
            DVec3::new(0.1, 0.1, 0.0),
            DVec3::new(0.05, 0.04, 0.0),
            DVec3::new(0.0, 0.1, 0.0),
        ];
        let mut triangles = Vec::new();
        triangulate_polygon(&polygon, &mut triangles).unwrap();
        assert_eq!(triangles.len(), polygon.len() - 2);
        assert!(triangles.iter().all(|triangle| {
            (triangle[1] - triangle[0])
                .cross(triangle[2] - triangle[0])
                .length_squared()
                > 0.0
        }));
    }

    #[test]
    fn obj_import_uses_meters_and_accepts_concave_faces() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("target/import-test-{}.obj", std::process::id()));
        fs::write(
            &path,
            "v 0 0 0\nv 0.1 0 0\nv 0.1 0.1 0\nv 0.05 0.04 0\nv 0 0.1 0\nf 1 2 3 4 5\n",
        )
        .unwrap();
        let preview = import_obj(&path, VoxelizationOptions::default()).unwrap();
        assert!((preview.source_bounds_max_m.x - 0.1).abs() < 1e-7);
        assert!(!preview.cells.is_empty());
        let scene = preview
            .build_scene(crate::scene::SceneConfig::default(), 1)
            .unwrap();
        assert_eq!(scene.voxels.len(), preview.cells.len());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn glb_import_applies_node_transforms_and_user_scale() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("target/import-test-{}.glb", std::process::id()));
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"translation":[2,3,4],"scale":[0.5,0.5,0.5]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"buffers":[{"byteLength":42}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[0.06,0.06,0]},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}]}"#;
        let mut json_bytes = json.as_bytes().to_vec();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let mut binary = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.06, 0.0] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0_u16, 1, 2] {
            binary.extend_from_slice(&index.to_le_bytes());
        }
        while binary.len() % 4 != 0 {
            binary.push(0);
        }
        let total_len = 12 + 8 + json_bytes.len() + 8 + binary.len();
        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4e4f_534a_u32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004e_4942_u32.to_le_bytes());
        glb.extend_from_slice(&binary);
        fs::write(&path, glb).unwrap();

        let mut options = VoxelizationOptions::default();
        options.scale = DVec3::splat(2.0);
        let preview = import_glb(&path, options).unwrap();
        assert!((preview.source_bounds_min_m.x - 4.0).abs() < 1e-5);
        assert!((preview.source_bounds_min_m.y - 6.0).abs() < 1e-5);
        assert!((preview.source_bounds_min_m.z - 8.0).abs() < 1e-5);
        assert!((preview.source_bounds_max_m.x - 4.06).abs() < 1e-5);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn resource_limits_prevent_large_grid_allocation() {
        let mesh = tetrahedron();
        let mut options = VoxelizationOptions::default();
        options.mode = VoxelizationMode::Filled;
        options.max_grid_cells = 10;
        assert!(voxelize(mesh, options).unwrap_err().contains("grid cells"));
    }
}
