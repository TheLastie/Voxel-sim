//! Resident wgpu conjugate gradients for the 6x6 cell system, preconditioned by
//! the same multilevel V-cycle as `cell_two_level::TwoLevel`.
//!
//! This is the portable GPU path (wgpu: Vulkan, DX12 or Metal) of step C of
//! INTEGRATION_PLAN_2026-11-05.md. Design:
//!
//! * Storage is f32. Matrices keep the CPU layout: a 6x6 diagonal block per cell,
//!   the lower blocks once, and per row a list of (neighbour, block, transposed).
//! * Every level of the CPU hierarchy is copied once: the Galerkin coarse
//!   operators, the masked damped block-Jacobi smoother (`DEFAULT_DAMPING` comes
//!   with the hierarchy), the aggregate map with the rigid-body offsets. The fine
//!   operator, the masks and the right-hand side are uploaded for every solve.
//! * The bottom level is solved on the GPU by a dense matrix: the CPU skyline
//!   factor applied to unit vectors, masked, stored in f32. `z = B r` is one
//!   kernel with a workgroup per row.
//! * One CG iteration is about twenty dispatches: residual norm and stop test,
//!   `A p`, `p . Ap`, the vector updates, the V-cycle (five kernels per level and
//!   the bottom), the masked `r . z`, the new direction. The scalars (`alpha`,
//!   `beta`, `r.z`, the stop flag, the iteration count) live in a GPU buffer and
//!   are computed there, so no vector leaves the GPU during the solve. The CPU
//!   submits `check_every` iterations at a time and reads back one small state
//!   record; iterations after the stop flag is set do nothing, so the iteration
//!   count is the same as with a check every iteration.
//! * The solution is read back once and the true residual `b - A x` is computed
//!   in f64 on the CPU. If f32 rounding left it above the tolerance, the residual
//!   is solved again as a correction (iterative refinement); the iteration count
//!   includes those passes.
//!
//! Errors (no adapter, validation error, device loss, a bottom too large for a
//! dense matrix) are returned, never hidden; the solver falls back to the CPU.
use crate::cell_two_level::{CellBond, CellSystem, TwoLevel};
use crate::solver::{Block6, Vec6};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use wgpu::util::DeviceExt;

/// Largest bottom level solved as a dense matrix (degrees of freedom).
const MAX_BOTTOM_DOFS: usize = 1536;
/// Partial sums of a reduction: one per workgroup of 256.
const REDUCE_GROUPS: usize = 256;
/// Refinement passes after the first solve.
const MAX_REFINEMENTS: usize = 3;

const COMMON: &str = r#"
struct Params { n: u32, mode: u32, groups: u32, pad: u32 };
struct State {
    rz: f32, alpha: f32, beta: f32, den: f32,
    force_limit: f32, moment_limit: f32, force: f32, moment: f32,
    done: u32, iterations: u32, limit: u32, pad: u32,
};
"#;

/// `z = S r` (mode 0) or `z += S r` (mode 1); `S` is the masked smoother block.
const SMOOTH: &str = r#"
@group(0) @binding(0) var<storage, read> smoother: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.n * 6u) { return; }
    let row = index / 6u;
    let base = row * 36u + (index % 6u) * 6u;
    var acc = 0.0;
    for (var q = 0u; q < 6u; q = q + 1u) {
        acc = acc + smoother[base + q] * r[row * 6u + q];
    }
    if (params.mode == 1u) { z[index] = z[index] + acc; } else { z[index] = acc; }
}
"#;

/// `out = r - A x` (mode 0) or `out = A x` (mode 1).
const RESIDUAL: &str = r#"
@group(0) @binding(0) var<storage, read> diag: array<f32>;
@group(0) @binding(1) var<storage, read> row_start: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<u32>;
@group(0) @binding(3) var<storage, read> blocks: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read> r: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.n * 6u) { return; }
    let row = index / 6u;
    let p = index % 6u;
    var acc = 0.0;
    for (var q = 0u; q < 6u; q = q + 1u) {
        acc = acc + diag[row * 36u + p * 6u + q] * x[row * 6u + q];
    }
    for (var e = row_start[row]; e < row_start[row + 1u]; e = e + 1u) {
        let col = entries[2u * e];
        let packed = entries[2u * e + 1u];
        let base = (packed & 0x7fffffffu) * 36u;
        if ((packed >> 31u) == 0u) {
            for (var q = 0u; q < 6u; q = q + 1u) {
                acc = acc + blocks[base + p * 6u + q] * x[col * 6u + q];
            }
        } else {
            for (var q = 0u; q < 6u; q = q + 1u) {
                acc = acc + blocks[base + q * 6u + p] * x[col * 6u + q];
            }
        }
    }
    if (params.mode == 0u) { out[index] = r[index] - acc; } else { out[index] = acc; }
}
"#;

/// Fine operator in bond form, one thread per cell: `out = r - A x` (mode 0) or
/// `out = A x` (mode 1). `A x = R x + sum over bonds`, `R` the f64 remainder of
/// the diagonal (inertia, one-sided bonds, identity rows of fixed axes). A bond
/// has 21 values: `k` row-major, the rest offset, the face stiffness row-major.
/// `graph` holds the cell starts, then `bond | side << 31` per incidence, then
/// the cell pair of each bond from `params.groups` on.
const BOND_RESIDUAL: &str = r#"
@group(0) @binding(0) var<storage, read> remainder: array<f32>;
@group(0) @binding(1) var<storage, read> graph: array<u32>;
@group(0) @binding(2) var<storage, read> bonds: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read> r: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
fn load3(base: u32) -> vec3<f32> {
    return vec3<f32>(x[base], x[base + 1u], x[base + 2u]);
}
fn row(base: u32, v: vec3<f32>) -> f32 {
    return bonds[base] * v.x + bonds[base + 1u] * v.y + bonds[base + 2u] * v.z;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    var y = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var p = 0u; p < 6u; p = p + 1u) {
        var acc = 0.0;
        for (var q = 0u; q < 6u; q = q + 1u) {
            acc = acc + remainder[i * 36u + p * 6u + q] * x[6u * i + q];
        }
        y[p] = acc;
    }
    var force = vec3<f32>(0.0, 0.0, 0.0);
    var moment = vec3<f32>(0.0, 0.0, 0.0);
    let pairs = params.groups;
    for (var e = graph[i]; e < graph[i + 1u]; e = e + 1u) {
        let packed = graph[e];
        let b = packed & 0x7fffffffu;
        let first = (packed >> 31u) == 0u;
        let a = graph[pairs + 2u * b];
        let c = graph[pairs + 2u * b + 1u];
        let pa = load3(6u * a + 3u);
        let pc = load3(6u * c + 3u);
        let base = b * 21u;
        let rest = vec3<f32>(bonds[base + 9u], bonds[base + 10u], bonds[base + 11u]);
        let strain = (load3(6u * c) - load3(6u * a)) + 0.5 * cross(rest, pa + pc);
        let f = vec3<f32>(row(base, strain), row(base + 3u, strain), row(base + 6u, strain));
        let turn = select(pc - pa, pa - pc, first);
        let couple = vec3<f32>(row(base + 12u, turn), row(base + 15u, turn), row(base + 18u, turn));
        if (first) { force = force - f; } else { force = force + f; }
        moment = moment + 0.5 * cross(f, rest) + couple;
    }
    y[0] = y[0] + force.x * mask[6u * i];
    y[1] = y[1] + force.y * mask[6u * i + 1u];
    y[2] = y[2] + force.z * mask[6u * i + 2u];
    y[3] = y[3] + moment.x * mask[6u * i + 3u];
    y[4] = y[4] + moment.y * mask[6u * i + 4u];
    y[5] = y[5] + moment.z * mask[6u * i + 5u];
    for (var p = 0u; p < 6u; p = p + 1u) {
        if (params.mode == 0u) { out[6u * i + p] = r[6u * i + p] - y[p]; } else { out[6u * i + p] = y[p]; }
    }
}
"#;

/// Restriction, one thread per aggregate: `F = sum f`, `M = sum (r x f + m)`.
const RESTRICT: &str = r#"
@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> offset: array<f32>;
@group(0) @binding(3) var<storage, read> member_start: array<u32>;
@group(0) @binding(4) var<storage, read> members: array<u32>;
@group(0) @binding(5) var<storage, read> coarse_mask: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if (a >= params.n) { return; }
    var force = vec3<f32>(0.0, 0.0, 0.0);
    var moment = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = member_start[a]; k < member_start[a + 1u]; k = k + 1u) {
        let i = members[k];
        let masked = vec3<f32>(
            residual[6u * i] * mask[6u * i],
            residual[6u * i + 1u] * mask[6u * i + 1u],
            residual[6u * i + 2u] * mask[6u * i + 2u],
        );
        force = force + masked;
        let r = vec3<f32>(offset[3u * i], offset[3u * i + 1u], offset[3u * i + 2u]);
        moment = moment + cross(r, masked)
            + vec3<f32>(residual[6u * i + 3u], residual[6u * i + 4u], residual[6u * i + 5u]);
    }
    out[6u * a] = force.x * coarse_mask[6u * a];
    out[6u * a + 1u] = force.y * coarse_mask[6u * a + 1u];
    out[6u * a + 2u] = force.z * coarse_mask[6u * a + 2u];
    out[6u * a + 3u] = moment.x * coarse_mask[6u * a + 3u];
    out[6u * a + 4u] = moment.y * coarse_mask[6u * a + 4u];
    out[6u * a + 5u] = moment.z * coarse_mask[6u * a + 5u];
}
"#;

/// Prolongation, one thread per fine cell: `u += U + theta x r`, `phi += theta`.
const PROLONG: &str = r#"
@group(0) @binding(0) var<storage, read> coarse: array<f32>;
@group(0) @binding(1) var<storage, read> aggregate: array<u32>;
@group(0) @binding(2) var<storage, read> offset: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> z: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let a = aggregate[i];
    let theta = vec3<f32>(coarse[6u * a + 3u], coarse[6u * a + 4u], coarse[6u * a + 5u]);
    let r = vec3<f32>(offset[3u * i], offset[3u * i + 1u], offset[3u * i + 2u]);
    let shift = cross(theta, r);
    z[6u * i] = z[6u * i] + (coarse[6u * a] + shift.x) * mask[6u * i];
    z[6u * i + 1u] = z[6u * i + 1u] + (coarse[6u * a + 1u] + shift.y) * mask[6u * i + 1u];
    z[6u * i + 2u] = z[6u * i + 2u] + (coarse[6u * a + 2u] + shift.z) * mask[6u * i + 2u];
    z[6u * i + 3u] = z[6u * i + 3u] + theta.x;
    z[6u * i + 4u] = z[6u * i + 4u] + theta.y;
    z[6u * i + 5u] = z[6u * i + 5u] + theta.z;
}
"#;

/// Dense bottom solve `z = B r`, one workgroup per row.
const DENSE: &str = r#"
@group(0) @binding(0) var<storage, read> inverse: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let n = params.n;
    var acc = 0.0;
    for (var c = lid; c < n; c = c + 64u) {
        acc = acc + inverse[row * n + c] * r[c];
    }
    sums[lid] = acc;
    workgroupBarrier();
    for (var width = 32u; width > 0u; width = width / 2u) {
        if (lid < width) { sums[lid] = sums[lid] + sums[lid + width]; }
        workgroupBarrier();
    }
    if (lid == 0u) { z[row] = sums[0]; }
}
"#;

/// Partial sums per workgroup. Mode 0: `a . b`. Mode 1: `|a_force|^2`, `|a_moment|^2`.
/// Mode 2: `b *= mask`, then `a . b`.
const REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> sx: array<f32, 256>;
var<workgroup> sy: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let total = params.n * 6u;
    let stride = params.groups * 256u;
    var x = 0.0;
    var y = 0.0;
    for (var i = wid.x * 256u + lid; i < total; i = i + stride) {
        if (params.mode == 0u) {
            x = x + a[i] * b[i];
        } else if (params.mode == 1u) {
            let v = a[i] * a[i];
            if (i % 6u < 3u) { x = x + v; } else { y = y + v; }
        } else {
            let m = b[i] * mask[i];
            b[i] = m;
            x = x + a[i] * m;
        }
    }
    sx[lid] = x;
    sy[lid] = y;
    workgroupBarrier();
    for (var width = 128u; width > 0u; width = width / 2u) {
        if (lid < width) {
            sx[lid] = sx[lid] + sx[lid + width];
            sy[lid] = sy[lid] + sy[lid + width];
        }
        workgroupBarrier();
    }
    if (lid == 0u) { partial[wid.x] = vec2<f32>(sx[0], sy[0]); }
}
"#;

/// Sums the partials and runs the scalar step of CG. Mode 0: stop test.
/// Mode 1: initial `r.z`. Mode 2: `p.Ap`, `alpha`, iteration count. Mode 3: `beta`.
const SCALAR: &str = r#"
@group(0) @binding(0) var<storage, read> partial: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> state: State;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> sx: array<f32, 256>;
var<workgroup> sy: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    var v = vec2<f32>(0.0, 0.0);
    if (lid < params.groups) { v = partial[lid]; }
    sx[lid] = v.x;
    sy[lid] = v.y;
    workgroupBarrier();
    for (var width = 128u; width > 0u; width = width / 2u) {
        if (lid < width) {
            sx[lid] = sx[lid] + sx[lid + width];
            sy[lid] = sy[lid] + sy[lid + width];
        }
        workgroupBarrier();
    }
    if (lid != 0u) { return; }
    let sum_x = sx[0];
    let sum_y = sy[0];
    if (params.mode == 0u) {
        let f = sqrt(max(sum_x, 0.0));
        let m = sqrt(max(sum_y, 0.0));
        state.force = f;
        state.moment = m;
        if (state.done == 0u) {
            if ((f <= state.force_limit && m <= state.moment_limit) || state.iterations >= state.limit) {
                state.done = 1u;
            }
        }
    } else if (params.mode == 1u) {
        state.rz = sum_x;
        state.beta = 0.0;
    } else if (params.mode == 2u) {
        if (state.done == 0u) {
            state.den = sum_x;
            state.iterations = state.iterations + 1u;
            if (sum_x > 0.0 && sum_x <= 3.0e38) {
                state.alpha = state.rz / sum_x;
            } else {
                state.alpha = 0.0;
                state.done = 2u;
            }
        }
    } else {
        if (state.done == 0u) {
            state.beta = sum_x / state.rz;
            state.rz = sum_x;
        }
    }
}
"#;

/// `x += alpha p`, `r -= alpha Ap` while running.
const AXPY: &str = r#"
@group(0) @binding(0) var<storage, read> state: State;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> ap: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> r: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n * 6u || state.done != 0u) { return; }
    let alpha = state.alpha;
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * ap[i];
}
"#;

/// `p = z + beta p` while running.
const DIRECTION: &str = r#"
@group(0) @binding(0) var<storage, read> state: State;
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n * 6u || state.done != 0u) { return; }
    p[i] = z[i] + state.beta * p[i];
}
"#;

#[derive(Clone, Copy)]
enum Kernel {
    Smooth = 0,
    Residual,
    Restrict,
    Prolong,
    Dense,
    Reduce,
    Scalar,
    Axpy,
    Direction,
    BondResidual,
}

const SOURCES: [(&str, &str); 10] = [
    ("cell smooth", SMOOTH),
    ("cell residual", RESIDUAL),
    ("cell restrict", RESTRICT),
    ("cell prolong", PROLONG),
    ("cell dense bottom", DENSE),
    ("cell reduce", REDUCE),
    ("cell scalar", SCALAR),
    ("cell axpy", AXPY),
    ("cell direction", DIRECTION),
    ("cell bond residual", BOND_RESIDUAL),
];

/// Scalar modes of `SCALAR`.
const CHECK: u32 = 0;
const INIT_RZ: u32 = 1;
const DENOMINATOR: u32 = 2;
const BETA: u32 = 3;

/// Device, queue and compiled kernels, shared by every solver on the process.
pub(crate) struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    pipelines: Vec<wgpu::ComputePipeline>,
    error: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    /// Adapter selection: `VOXEL_GPU_ADAPTER=<substring of the name>` picks one of
    /// the enumerated adapters, otherwise the high-performance adapter.
    fn create() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..Default::default()
        });
        let adapter = match std::env::var("VOXEL_GPU_ADAPTER") {
            Ok(wanted) if !wanted.is_empty() => {
                let wanted = wanted.to_lowercase();
                instance
                    .enumerate_adapters(wgpu::Backends::PRIMARY)
                    .into_iter()
                    .find(|adapter| adapter.get_info().name.to_lowercase().contains(&wanted))
                    .ok_or_else(|| format!("no wgpu adapter matches VOXEL_GPU_ADAPTER={wanted}"))?
            }
            _ => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok_or("no wgpu adapter")?,
        };
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("only a software adapter is available: {}", info.name));
        }
        let supported = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cell pcg"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ))
        .map_err(|error| format!("request_device: {error}"))?;
        let error = Arc::new(Mutex::new(None::<String>));
        let slot = error.clone();
        device.on_uncaptured_error(Box::new(move |e| {
            let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if slot.is_none() {
                *slot = Some(e.to_string());
            }
        }));
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = SOURCES
            .iter()
            .map(|(label, source)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!("{COMMON}{source}"))),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &module,
                    entry_point: "main",
                })
            })
            .collect();
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(format!("cell kernels: {e}"));
        }
        Ok(Self { device, queue, info, pipelines, error })
    }

    /// The process-wide context, created on first use.
    pub(crate) fn shared() -> Result<Arc<Self>, String> {
        static CONTEXT: OnceLock<Result<Arc<GpuContext>, String>> = OnceLock::new();
        CONTEXT.get_or_init(|| GpuContext::create().map(Arc::new)).clone()
    }

    /// Adapter name, type, backend and driver.
    pub(crate) fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}, driver {} {})",
            self.info.name, self.info.device_type, self.info.backend, self.info.driver, self.info.driver_info
        )
    }

    fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
    }

    fn storage(&self, label: &str, bytes: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
        let mut padded;
        let contents = if bytes.len() < 16 {
            padded = bytes.to_vec();
            padded.resize(16, 0);
            &padded[..]
        } else {
            bytes
        };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        })
    }

    fn zeroed(&self, label: &str, bytes: usize, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    }

    fn op(&self, kernel: Kernel, buffers: &[&wgpu::Buffer], params: [u32; 4], groups: usize) -> Op {
        let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: uniform.as_entire_binding(),
        });
        let layout = self.pipelines[kernel as usize].get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(SOURCES[kernel as usize].0),
            layout: &layout,
            entries: &entries,
        });
        Op { kernel: kernel as usize, bind, groups: groups.max(1) as u32, _uniform: uniform }
    }
}

struct Op {
    kernel: usize,
    bind: wgpu::BindGroup,
    groups: u32,
    _uniform: wgpu::Buffer,
}

/// Buffers of one level. Level 0 is the system being solved; `r` and `z` of the
/// coarser levels are their restricted right-hand side and their correction.
struct LevelBuffers {
    count: usize,
    edges: usize,
    entries: usize,
    diag: wgpu::Buffer,
    row_start: wgpu::Buffer,
    entry_list: wgpu::Buffer,
    blocks: wgpu::Buffer,
    smoother: wgpu::Buffer,
    mask: wgpu::Buffer,
    r: wgpu::Buffer,
    z: wgpu::Buffer,
    residual: wgpu::Buffer,
    transfer: Option<TransferBuffers>,
    /// The fine level's operator in bond form; coarse levels use the blocks.
    bond: Option<BondBuffers>,
}

struct BondBuffers {
    bonds: usize,
    remainder: wgpu::Buffer,
    graph: wgpu::Buffer,
    data: wgpu::Buffer,
    pairs_offset: u32,
}

struct TransferBuffers {
    aggregate: wgpu::Buffer,
    offset: wgpu::Buffer,
    member_start: wgpu::Buffer,
    members: wgpu::Buffer,
}

/// The hierarchy and the CG vectors, resident on the GPU.
struct Resident {
    hierarchy: u64,
    levels: Vec<LevelBuffers>,
    /// Buffers only the command lists refer to.
    _keep: Vec<wgpu::Buffer>,
    x: wgpu::Buffer,
    p: wgpu::Buffer,
    state: wgpu::Buffer,
    staging: wgpu::Buffer,
    init: Vec<Op>,
    iteration: Vec<Op>,
    check: Vec<Op>,
    dispatches_per_iteration: usize,
}

/// Per-solve figures.
#[derive(Clone, Debug, Default)]
pub(crate) struct GpuSolveStats {
    /// CG iterations, all refinement passes together.
    pub(crate) iterations: usize,
    /// Solves run: 1 plus refinement passes.
    pub(crate) passes: usize,
    /// Small state readbacks during the solve.
    pub(crate) readbacks: usize,
    /// True `|b - A x|` of the force and moment parts, relative to `|b|`, in f64.
    pub(crate) force_residual: f64,
    pub(crate) moment_residual: f64,
    pub(crate) converged: bool,
    /// Conversion and upload of the fine system and right-hand side, ms.
    pub(crate) upload_ms: f64,
    /// From the first submit to the solution on the CPU, ms.
    pub(crate) gpu_ms: f64,
    /// True residual checks on the CPU, ms.
    pub(crate) check_ms: f64,
    pub(crate) dispatches_per_iteration: usize,
}

/// A GPU solver for one stream of cell systems; keeps its hierarchy resident
/// while the CPU hierarchy it was copied from stays in use.
pub(crate) struct GpuCellSolver {
    context: Arc<GpuContext>,
    resident: Option<Resident>,
    /// Iterations submitted between state readbacks after the first batch.
    pub(crate) check_every: usize,
    /// The first batch has the iteration count of the previous solve, since
    /// successive systems of one step converge alike: usually one readback.
    last_iterations: usize,
    /// The GPU stops at this fraction of the tolerance, so f32 rounding of the
    /// recurrence residual rarely needs a refinement pass.
    pub(crate) inner_factor: f64,
}

fn to_f32_blocks(blocks: &[Block6]) -> Vec<f32> {
    let mut out = vec![0.0f32; blocks.len() * 36];
    out.par_chunks_mut(36).zip(blocks.par_iter()).for_each(|(target, block)| {
        for p in 0..6 {
            for q in 0..6 {
                target[p * 6 + q] = block[p][q] as f32;
            }
        }
    });
    out
}

fn to_f32_vectors(values: &[[f64; 6]]) -> Vec<f32> {
    values.iter().flat_map(|v| v.iter().map(|x| *x as f32)).collect()
}

fn graph_arrays(system: &CellSystem) -> (Vec<u32>, Vec<u32>) {
    let row_start = system.row_start.iter().map(|&v| v as u32).collect();
    let mut entries = Vec::with_capacity(system.entries.len() * 2);
    for &(col, block, transposed) in &system.entries {
        entries.push(col);
        entries.push(block | if transposed { 1 << 31 } else { 0 });
    }
    (row_start, entries)
}

fn masked_smoother(smoother: &[Block6], masks: &[[f64; 6]]) -> Vec<Block6> {
    smoother
        .iter()
        .zip(masks)
        .map(|(block, mask)| std::array::from_fn(|p| std::array::from_fn(|q| block[p][q] * mask[p])))
        .collect()
}

/// The fine operator in the layout of `BOND_RESIDUAL`.
struct FineArrays {
    remainder: Vec<f32>,
    graph: Vec<u32>,
    data: Vec<f32>,
    pairs_offset: u32,
}

/// The remainder `R_c = D_c - sum H_cc(bond)` is formed in f64, so the large
/// bond stiffness cancels before rounding and `R` keeps the inertia exactly.
fn fine_arrays(fine: &CellSystem, bonds: &[CellBond], masks: &[[f64; 6]]) -> Result<FineArrays, String> {
    let n = fine.count;
    if bonds.len() != fine.blocks.len() || masks.len() != n {
        return Err(format!(
            "bond form does not match the system: {} bonds for {} blocks",
            bonds.len(),
            fine.blocks.len()
        ));
    }
    let head = n + 1;
    let mut start = vec![0usize; n + 1];
    for bond in bonds {
        let (i, j) = (bond.i as usize, bond.j as usize);
        if i >= n || j >= n || i == j {
            return Err(format!("bond ({i}, {j}) outside {n} cells"));
        }
        start[i + 1] += 1;
        start[j + 1] += 1;
    }
    for c in 0..n {
        start[c + 1] += start[c];
    }
    let pairs_offset = head + 2 * bonds.len();
    let mut graph = vec![0u32; pairs_offset + 2 * bonds.len()];
    for c in 0..=n {
        graph[c] = (head + start[c]) as u32;
    }
    let mut fill = start.clone();
    for (b, bond) in bonds.iter().enumerate() {
        graph[head + fill[bond.i as usize]] = b as u32;
        fill[bond.i as usize] += 1;
        graph[head + fill[bond.j as usize]] = b as u32 | 1 << 31;
        fill[bond.j as usize] += 1;
        graph[pairs_offset + 2 * b] = bond.i;
        graph[pairs_offset + 2 * b + 1] = bond.j;
    }
    let incidence = &graph[head..pairs_offset];
    let mut remainder = vec![0.0f32; n * 36];
    remainder.par_chunks_mut(36).enumerate().for_each(|(c, target)| {
        let mut block = fine.diagonal[c];
        for &packed in &incidence[start[c]..start[c + 1]] {
            let (ii, jj) = bonds[(packed & 0x7fff_ffff) as usize].diagonal_blocks();
            let own = if packed >> 31 == 0 { ii } else { jj };
            for p in 0..6 {
                for q in 0..6 {
                    block[p][q] -= own[p][q];
                }
            }
        }
        for p in 0..6 {
            if masks[c][p] == 0.0 {
                for q in 0..6 {
                    block[p][q] = 0.0;
                    block[q][p] = 0.0;
                }
                block[p][p] = 1.0;
            }
        }
        for p in 0..6 {
            for q in 0..6 {
                target[p * 6 + q] = block[p][q] as f32;
            }
        }
    });
    let mut data = vec![0.0f32; bonds.len() * 21];
    data.par_chunks_mut(21).zip(bonds.par_iter()).for_each(|(target, bond)| {
        for row in 0..3 {
            for col in 0..3 {
                target[row * 3 + col] = bond.k.col(col)[row] as f32;
                target[12 + row * 3 + col] = bond.stiff.col(col)[row] as f32;
            }
            target[9 + row] = bond.rest[row] as f32;
        }
    });
    Ok(FineArrays { remainder, graph, data, pairs_offset: pairs_offset as u32 })
}

fn norms(v: &[Vec6]) -> (f64, f64) {
    let mut force = 0.0;
    let mut moment = 0.0;
    for x in v {
        force += x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        moment += x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
    }
    (force.sqrt(), moment.sqrt())
}

impl GpuCellSolver {
    pub(crate) fn new() -> Result<Self, String> {
        Ok(Self { context: GpuContext::shared()?, resident: None, check_every: 2, last_iterations: 0, inner_factor: 0.5 })
    }

    pub(crate) fn describe(&self) -> String {
        self.context.describe()
    }

    /// Whether the resident hierarchy is the copy of `two`.
    pub(crate) fn holds(&self, two: &TwoLevel) -> bool {
        self.resident.as_ref().is_some_and(|resident| resident.hierarchy == two.id())
    }

    /// Copy the hierarchy `two` built for `fine` to the GPU and record the
    /// command lists of one solve.
    pub(crate) fn load(&mut self, fine: &CellSystem, bonds: &[CellBond], two: &TwoLevel) -> Result<(), String> {
        self.resident = None;
        let bottom_dofs = two.bottom_dofs();
        if bottom_dofs > MAX_BOTTOM_DOFS {
            return Err(format!("bottom level of {bottom_dofs} dofs is too large for a dense GPU solve"));
        }
        let ctx = &*self.context;
        ctx.take_error();
        let f32_bytes = |v: &[f32]| bytemuck::cast_slice::<f32, u8>(v).to_vec();
        let mut levels = Vec::<LevelBuffers>::new();
        for (index, data) in two.levels.iter().enumerate() {
            let system = if index == 0 { fine } else { &two.systems[index - 1] };
            let count = system.count;
            if count * 6 > 65_535 * 64 {
                return Err(format!("{count} cells exceed one dispatch dimension"));
            }
            let usage = wgpu::BufferUsages::empty();
            // The fine level is applied in bond form: its blocks are not needed.
            let (row_start, entries, diagonal, lower) = if index == 0 {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            } else {
                let (row_start, entries) = graph_arrays(system);
                (row_start, entries, to_f32_blocks(&system.diagonal), to_f32_blocks(&system.blocks))
            };
            let diag = ctx.storage("diag", &f32_bytes(&diagonal), usage);
            let row_start_buffer = ctx.storage("row_start", bytemuck::cast_slice(&row_start), usage);
            let entry_list = ctx.storage("entries", bytemuck::cast_slice(&entries), usage);
            let blocks = ctx.storage("blocks", &f32_bytes(&lower), usage);
            let bond = if index == 0 {
                let arrays = fine_arrays(fine, bonds, &data.masks)?;
                Some(BondBuffers {
                    bonds: bonds.len(),
                    remainder: ctx.storage("remainder", &f32_bytes(&arrays.remainder), usage),
                    graph: ctx.storage("bond graph", bytemuck::cast_slice(&arrays.graph), usage),
                    data: ctx.storage("bonds", &f32_bytes(&arrays.data), usage),
                    pairs_offset: arrays.pairs_offset,
                })
            } else {
                None
            };
            let smoother = ctx.storage(
                "smoother",
                &f32_bytes(&to_f32_blocks(&masked_smoother(&data.smoother, &data.masks))),
                usage,
            );
            let mask = ctx.storage("mask", &f32_bytes(&to_f32_vectors(&data.masks)), usage);
            let vector = count * 6 * 4;
            let r = ctx.zeroed("r", vector, wgpu::BufferUsages::empty());
            let z = ctx.zeroed("z", vector, wgpu::BufferUsages::empty());
            let residual = ctx.zeroed("residual", vector, wgpu::BufferUsages::empty());
            let transfer = data.transfer.as_ref().map(|t| {
                let offset: Vec<f32> =
                    t.offset.iter().flat_map(|o| [o.x as f32, o.y as f32, o.z as f32]).collect();
                let member_start: Vec<u32> = t.member_start.iter().map(|&v| v as u32).collect();
                TransferBuffers {
                    aggregate: ctx.storage("aggregate", bytemuck::cast_slice(&t.aggregate), usage),
                    offset: ctx.storage("offset", &f32_bytes(&offset), usage),
                    member_start: ctx.storage("member_start", bytemuck::cast_slice(&member_start), usage),
                    members: ctx.storage("members", bytemuck::cast_slice(&t.members), usage),
                }
            });
            levels.push(LevelBuffers {
                count,
                edges: system.blocks.len(),
                entries: system.entries.len(),
                diag,
                row_start: row_start_buffer,
                entry_list,
                blocks,
                smoother,
                mask,
                r,
                z,
                residual,
                transfer,
                bond,
            });
        }
        let (dofs, dense) = two.bottom_dense();
        let dense: Vec<f32> = dense.iter().map(|v| *v as f32).collect();
        let bottom = ctx.storage("bottom", &f32_bytes(&dense), wgpu::BufferUsages::empty());
        let n = levels[0].count;
        let vector = n * 6 * 4;
        let x = ctx.zeroed("x", vector, wgpu::BufferUsages::COPY_SRC);
        let p = ctx.zeroed("p", vector, wgpu::BufferUsages::empty());
        let ap = ctx.zeroed("ap", vector, wgpu::BufferUsages::empty());
        let partial = ctx.zeroed("partial", REDUCE_GROUPS * 8, wgpu::BufferUsages::empty());
        let state = ctx.zeroed("state", 48, wgpu::BufferUsages::COPY_SRC);
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (64 + vector) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let per_dof = |count: usize| (count * 6).div_ceil(64);
        let groups = (n * 6).div_ceil(256).min(REDUCE_GROUPS);
        let n32 = n as u32;
        // `out = r - A x` (mode 0) or `A x` (mode 1) on one level.
        fn residual_op(ctx: &GpuContext, level: &LevelBuffers, x: &wgpu::Buffer, r: &wgpu::Buffer, out: &wgpu::Buffer, mode: u32) -> Op {
            let count = level.count;
            match &level.bond {
                Some(b) => ctx.op(
                    Kernel::BondResidual,
                    &[&b.remainder, &b.graph, &b.data, &level.mask, x, r, out],
                    [count as u32, mode, b.pairs_offset, 0],
                    count.div_ceil(64),
                ),
                None => ctx.op(
                    Kernel::Residual,
                    &[&level.diag, &level.row_start, &level.entry_list, &level.blocks, x, r, out],
                    [count as u32, mode, 0, 0],
                    (count * 6).div_ceil(64),
                ),
            }
        }
        // V-cycle on level `l`, from `levels[l].r` into `levels[l].z`.
        fn cycle(ctx: &GpuContext, levels: &[LevelBuffers], l: usize, bottom: &wgpu::Buffer, dofs: usize, ops: &mut Vec<Op>) {
            let level = &levels[l];
            let count = level.count;
            let c32 = count as u32;
            let per_dof = (count * 6).div_ceil(64);
            let Some(t) = &level.transfer else {
                ops.push(ctx.op(Kernel::Dense, &[bottom, &level.r, &level.z], [dofs as u32, 0, 0, 0], dofs));
                return;
            };
            let coarse = &levels[l + 1];
            ops.push(ctx.op(Kernel::Smooth, &[&level.smoother, &level.r, &level.z], [c32, 0, 0, 0], per_dof));
            ops.push(residual_op(ctx, level, &level.z, &level.r, &level.residual, 0));
            ops.push(ctx.op(
                Kernel::Restrict,
                &[&level.residual, &level.mask, &t.offset, &t.member_start, &t.members, &coarse.mask, &coarse.r],
                [coarse.count as u32, 0, 0, 0],
                coarse.count.div_ceil(64),
            ));
            cycle(ctx, levels, l + 1, bottom, dofs, ops);
            ops.push(ctx.op(
                Kernel::Prolong,
                &[&coarse.z, &t.aggregate, &t.offset, &level.mask, &level.z],
                [c32, 0, 0, 0],
                count.div_ceil(64),
            ));
            ops.push(residual_op(ctx, level, &level.z, &level.r, &level.residual, 0));
            ops.push(ctx.op(Kernel::Smooth, &[&level.smoother, &level.residual, &level.z], [c32, 1, 0, 0], per_dof));
        }
        let fine_level = &levels[0];
        let (r, z) = (&fine_level.r, &fine_level.z);
        let reduce = |a: &wgpu::Buffer, b: &wgpu::Buffer, mode: u32| {
            ctx.op(Kernel::Reduce, &[a, b, &fine_level.mask, &partial], [n32, mode, groups as u32, 0], groups)
        };
        let scalar = |mode: u32| ctx.op(Kernel::Scalar, &[&partial, &state], [0, mode, groups as u32, 0], 1);
        let mut init = Vec::new();
        cycle(ctx, &levels, 0, &bottom, dofs, &mut init);
        init.push(reduce(r, z, 2));
        init.push(scalar(INIT_RZ));
        init.push(ctx.op(Kernel::Direction, &[&state, z, &p], [n32, 0, 0, 0], per_dof(n)));
        let check = vec![reduce(r, z, 1), scalar(CHECK)];
        let mut iteration = Vec::new();
        iteration.push(reduce(r, z, 1));
        iteration.push(scalar(CHECK));
        iteration.push(residual_op(ctx, fine_level, &p, r, &ap, 1));
        iteration.push(reduce(&p, &ap, 0));
        iteration.push(scalar(DENOMINATOR));
        iteration.push(ctx.op(Kernel::Axpy, &[&state, &p, &ap, &x, r], [n32, 0, 0, 0], per_dof(n)));
        cycle(ctx, &levels, 0, &bottom, dofs, &mut iteration);
        iteration.push(reduce(r, z, 2));
        iteration.push(scalar(BETA));
        iteration.push(ctx.op(Kernel::Direction, &[&state, z, &p], [n32, 0, 0, 0], per_dof(n)));
        let dispatches_per_iteration = iteration.len();
        ctx.device.poll(wgpu::Maintain::Wait);
        if let Some(error) = ctx.take_error() {
            return Err(format!("loading the hierarchy: {error}"));
        }
        self.resident = Some(Resident {
            hierarchy: two.id(),
            levels,
            _keep: vec![bottom, ap, partial],
            x,
            p,
            state,
            staging,
            init,
            iteration,
            check,
            dispatches_per_iteration,
        });
        Ok(())
    }

    /// Upload the current fine operator; its graph must have the sizes of the
    /// one the hierarchy was loaded with.
    fn upload_fine(&self, fine: &CellSystem, bonds: &[CellBond], masks: &[[f64; 6]]) -> Result<(), String> {
        let resident = self.resident.as_ref().ok_or("no hierarchy loaded")?;
        let level = &resident.levels[0];
        let bond = level.bond.as_ref().ok_or("fine level has no bond form")?;
        if fine.count != level.count || fine.blocks.len() != level.edges || bonds.len() != bond.bonds {
            return Err("fine graph differs from the loaded hierarchy".into());
        }
        let arrays = fine_arrays(fine, bonds, masks)?;
        let queue = &self.context.queue;
        queue.write_buffer(&bond.remainder, 0, bytemuck::cast_slice(&arrays.remainder));
        queue.write_buffer(&bond.graph, 0, bytemuck::cast_slice(&arrays.graph));
        if !bonds.is_empty() {
            queue.write_buffer(&bond.data, 0, bytemuck::cast_slice(&arrays.data));
        }
        queue.write_buffer(&level.mask, 0, bytemuck::cast_slice(&to_f32_vectors(masks)));
        Ok(())
    }

    /// One CG solve on the GPU from `x = 0`; returns `x` and the iterations.
    fn run(&self, rhs: &[Vec6], limits: (f64, f64), limit: usize, first_batch: usize, readbacks: &mut usize) -> Result<(Vec<Vec6>, usize, u32), String> {
        let ctx = &*self.context;
        let resident = self.resident.as_ref().ok_or("no hierarchy loaded")?;
        let n = resident.levels[0].count;
        ctx.queue.write_buffer(&resident.levels[0].r, 0, bytemuck::cast_slice(&to_f32_vectors(rhs)));
        let mut state = [0u32; 12];
        state[4] = (limits.0 as f32).to_bits();
        state[5] = (limits.1 as f32).to_bits();
        state[10] = limit as u32;
        ctx.queue.write_buffer(&resident.state, 0, bytemuck::cast_slice(&state));
        let vector = (n * 6 * 4) as u64;
        let mut first = true;
        loop {
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cell pcg") });
            if first {
                encoder.clear_buffer(&resident.x, 0, None);
                encoder.clear_buffer(&resident.p, 0, None);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                let init: &[Op] = if first { &resident.init } else { &[] };
                let ops = init
                    .iter()
                    .chain((0..(if first { first_batch } else { self.check_every }).max(1)).flat_map(|_| resident.iteration.iter()))
                    .chain(resident.check.iter());
                for op in ops {
                    pass.set_pipeline(&ctx.pipelines[op.kernel]);
                    pass.set_bind_group(0, &op.bind, &[]);
                    pass.dispatch_workgroups(op.groups, 1, 1);
                }
            }
            first = false;
            encoder.copy_buffer_to_buffer(&resident.state, 0, &resident.staging, 0, 48);
            encoder.copy_buffer_to_buffer(&resident.x, 0, &resident.staging, 64, vector);
            ctx.queue.submit([encoder.finish()]);
            let slice = resident.staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            *readbacks += 1;
            if let Some(error) = ctx.take_error() {
                return Err(format!("GPU solve: {error}"));
            }
            receiver
                .recv()
                .map_err(|_| "map callback dropped".to_string())?
                .map_err(|error| format!("map: {error}"))?;
            let mapped = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&mapped[..48]);
            let (done, iterations) = (words[8], words[9] as usize);
            let result = if done != 0 {
                let values: &[f32] = bytemuck::cast_slice(&mapped[64..64 + vector as usize]);
                let x: Vec<Vec6> =
                    values.chunks_exact(6).map(|c| std::array::from_fn(|t| f64::from(c[t]))).collect();
                Some((x, iterations, done))
            } else {
                None
            };
            drop(mapped);
            resident.staging.unmap();
            if let Some(result) = result {
                return Ok(result);
            }
        }
    }

    /// Solve `A x = b` with the resident hierarchy (`load` first). `tolerance` is
    /// relative to the force and moment norms of `rhs`, as on the CPU; `limit`
    /// caps the iterations of all passes together.
    pub(crate) fn solve(
        &mut self,
        fine: &CellSystem,
        bonds: &[CellBond],
        masks: &[[f64; 6]],
        rhs: &[Vec6],
        tolerance: f64,
        limit: usize,
    ) -> Result<(Vec<Vec6>, GpuSolveStats), String> {
        let mut stats = GpuSolveStats::default();
        let started = Instant::now();
        self.upload_fine(fine, bonds, masks)?;
        stats.upload_ms = started.elapsed().as_secs_f64() * 1000.0;
        stats.dispatches_per_iteration = self.resident.as_ref().map_or(0, |r| r.dispatches_per_iteration);
        let (force0, moment0) = norms(rhs);
        let limits = (tolerance * force0.max(1.0e-12), tolerance * moment0.max(1.0e-12));
        let inner = (limits.0 * self.inner_factor, limits.1 * self.inner_factor);
        let count = rhs.len();
        let mut x = vec![[0.0f64; 6]; count];
        let mut pass_rhs = rhs.to_vec();
        let mut product = vec![[0.0f64; 6]; count];
        let mut gpu = 0.0;
        loop {
            let pass_started = Instant::now();
            let first_batch = if stats.passes == 0 { self.last_iterations.max(self.check_every) } else { self.check_every };
            let (dx, iterations, done) = self.run(
                &pass_rhs,
                inner,
                limit.saturating_sub(stats.iterations).max(1),
                first_batch,
                &mut stats.readbacks,
            )?;
            gpu += pass_started.elapsed().as_secs_f64();
            stats.iterations += iterations;
            stats.passes += 1;
            let check_started = Instant::now();
            for (target, delta) in x.iter_mut().zip(&dx) {
                for t in 0..6 {
                    target[t] += delta[t];
                }
            }
            fine.apply(&x, &mut product);
            for slot in 0..count {
                for t in 0..6 {
                    pass_rhs[slot][t] = rhs[slot][t] - product[slot][t];
                }
            }
            let (force, moment) = norms(&pass_rhs);
            stats.check_ms += check_started.elapsed().as_secs_f64() * 1000.0;
            stats.force_residual = force / force0.max(1.0e-12);
            stats.moment_residual = moment / moment0.max(1.0e-12);
            stats.converged = force <= limits.0 && moment <= limits.1;
            if stats.converged
                || done == 2
                || stats.iterations >= limit
                || stats.passes > MAX_REFINEMENTS
                || x.iter().flatten().any(|v| !v.is_finite())
            {
                break;
            }
        }
        stats.gpu_ms = gpu * 1000.0;
        self.last_iterations = stats.iterations;
        if x.iter().flatten().any(|v| !v.is_finite()) {
            return Err("GPU solution is not finite".into());
        }
        Ok((x, stats))
    }
}

/// `VOXEL_CELL_GPU=1`: solve bulk cell systems with this path.
pub(crate) fn opt_in() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("VOXEL_CELL_GPU").is_ok_and(|value| value == "1"))
}

/// The solver's entry: solve on the GPU with `two` resident, or return None so
/// the caller runs the CPU loop. Any error, panic or unconverged result drops the
/// GPU state for this solve and is reported once.
pub(crate) fn solve_with_fallback(
    slot: &mut Option<GpuCellSolver>,
    fine: &CellSystem,
    bonds: &[CellBond],
    two: &TwoLevel,
    masks: &[[f64; 6]],
    rhs: &[Vec6],
    tolerance: f64,
    limit: usize,
) -> Option<(Vec<Vec6>, usize)> {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let report = |message: &str| {
        if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("VOXEL_CELL_GPU: falling back to the CPU: {message}");
        }
    };
    if slot.is_none() {
        match GpuCellSolver::new() {
            Ok(solver) => *slot = Some(solver),
            Err(error) => {
                report(&error);
                return None;
            }
        }
    }
    let solver = slot.as_mut()?;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !solver.holds(two) {
            solver.load(fine, bonds, two)?;
        }
        solver.solve(fine, bonds, masks, rhs, tolerance, limit)
    }));
    match outcome {
        Ok(Ok((x, stats))) if stats.converged => Some((x, stats.iterations)),
        Ok(Ok((_, stats))) => {
            report(&format!(
                "not converged after {} iterations (force residual {:.2e})",
                stats.iterations, stats.force_residual
            ));
            None
        }
        Ok(Err(error)) => {
            report(&error);
            *slot = None;
            None
        }
        Err(_) => {
            report("panic in the GPU solve");
            *slot = None;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell_two_level::tests::captured_bulk;
    use crate::cell_two_level::{dot, TwoLevelWork, DEFAULT_DAMPING};
    use glam::DVec3;

    /// The CPU loop of `solve_face_cells_sparse`: PCG with the V-cycle, `z` masked,
    /// stop on the force and moment norms of the recurrence residual.
    fn cpu_pcg(
        system: &CellSystem,
        two: &TwoLevel,
        rhs: &[Vec6],
        masks: &[[f64; 6]],
        tolerance: f64,
        limit: usize,
    ) -> (Vec<Vec6>, usize) {
        let count = rhs.len();
        let mut work = TwoLevelWork::default();
        let mut precondition = |r: &[Vec6], z: &mut [Vec6]| {
            two.apply(system, r, z, &mut work);
            for slot in 0..count {
                for p in 0..6 {
                    z[slot][p] *= masks[slot][p];
                }
            }
        };
        let (force0, moment0) = norms(rhs);
        let mut x = vec![[0.0; 6]; count];
        let mut r = rhs.to_vec();
        let mut z = vec![[0.0; 6]; count];
        let mut ap = vec![[0.0; 6]; count];
        precondition(&r, &mut z);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut iterations = 0;
        loop {
            let (f, m) = norms(&r);
            if (f <= tolerance * force0.max(1e-12) && m <= tolerance * moment0.max(1e-12)) || iterations >= limit {
                return (x, iterations);
            }
            system.apply(&p, &mut ap);
            let denominator = dot(&p, &ap);
            iterations += 1;
            if !(denominator > 0.0) {
                return (x, iterations);
            }
            let alpha = rz / denominator;
            for i in 0..count {
                for t in 0..6 {
                    x[i][t] += alpha * p[i][t];
                    r[i][t] -= alpha * ap[i][t];
                }
            }
            precondition(&r, &mut z);
            let next = dot(&r, &z);
            let beta = next / rz;
            rz = next;
            for i in 0..count {
                for t in 0..6 {
                    p[i][t] = z[i][t] + beta * p[i][t];
                }
            }
        }
    }

    fn true_residual(system: &CellSystem, rhs: &[Vec6], x: &[Vec6]) -> (f64, f64) {
        let mut product = vec![[0.0; 6]; x.len()];
        system.apply(x, &mut product);
        let r: Vec<Vec6> = rhs
            .iter()
            .zip(&product)
            .map(|(b, ax)| std::array::from_fn(|t| b[t] - ax[t]))
            .collect();
        let (f0, m0) = norms(rhs);
        let (f, m) = norms(&r);
        (f / f0.max(1e-12), m / m0.max(1e-12))
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[values.len() / 2]
    }

    /// GPU and CPU PCG on the systems of a real bulk scene: same tolerance (the
    /// solver's 1e-5 relative force and moment residual), comparable iterations.
    fn parity(sizes: &[(i32, i32, usize)]) {
        let mut solver = match GpuCellSolver::new() {
            Ok(solver) => solver,
            Err(error) => {
                eprintln!("GPU PCG parity skipped: {error}");
                return;
            }
        };
        if let Some(k) = std::env::var("VOXEL_GPU_CHECK_EVERY").ok().and_then(|v| v.parse().ok()) {
            solver.check_every = k;
        }
        if let Some(f) = std::env::var("VOXEL_GPU_INNER").ok().and_then(|v| v.parse().ok()) {
            solver.inner_factor = f;
        }
        let tolerance = 1.0e-5;
        let limit = 800;
        eprintln!(
            "GPU adapter: {} check_every={} inner_factor={}",
            solver.describe(),
            solver.check_every,
            solver.inner_factor
        );
        for &(width, height, warmup) in sizes {
            let systems = captured_bulk(width, height, warmup);
            assert!(!systems.is_empty());
            let mut rows = Vec::new();
            let (mut cpu_times, mut gpu_times, mut loads, mut uploads) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut levels = 0;
            let mut bottom = 0;
            for captured in &systems {
                let system = CellSystem::new(&captured.diagonal, &captured.edges);
                let masks = &captured.masks;
                let two = TwoLevel::build(
                    &system, &captured.edges, &captured.positions, &captured.lattice,
                    &captured.bodies, masks, 0.01, DEFAULT_DAMPING,
                )
                .expect("hierarchy");
                levels = two.levels.len();
                bottom = two.bottom_dofs();
                let started = Instant::now();
                let (x_cpu, cpu_iterations) = cpu_pcg(&system, &two, &captured.rhs, masks, tolerance, limit);
                cpu_times.push(started.elapsed().as_secs_f64() * 1000.0);
                let cpu_residual = true_residual(&system, &captured.rhs, &x_cpu);
                // Near-exact reference, to show how far each 1e-5 solution is from it.
                let (x_ref, _) = cpu_pcg(&system, &two, &captured.rhs, masks, 1.0e-11, 4000);
                let error_of = |x: &[Vec6]| {
                    x.iter()
                        .zip(&x_ref)
                        .map(|(a, b)| (0..3).map(|t| (a[t] - b[t]).powi(2)).sum::<f64>())
                        .sum::<f64>()
                        .sqrt()
                        / x_ref.iter().map(|a| (0..3).map(|t| a[t] * a[t]).sum::<f64>()).sum::<f64>().sqrt().max(1e-300)
                };
                let cpu_error = error_of(&x_cpu);
                let started = Instant::now();
                solver.load(&system, &captured.bonds, &two).expect("load hierarchy");
                loads.push(started.elapsed().as_secs_f64() * 1000.0);
                let (x_gpu, stats) = solver
                    .solve(&system, &captured.bonds, masks, &captured.rhs, tolerance, limit)
                    .expect("GPU solve");
                // Warm repeats: the hierarchy stays resident, the fine system and
                // right-hand side are uploaded again as in a solver step.
                let mut warm = Vec::new();
                for _ in 0..3 {
                    let started = Instant::now();
                    let (_, again) = solver.solve(&system, &captured.bonds, masks, &captured.rhs, tolerance, limit).unwrap();
                    warm.push(started.elapsed().as_secs_f64() * 1000.0);
                    uploads.push(again.upload_ms);
                    assert_eq!(again.iterations, stats.iterations, "GPU solve is not repeatable");
                }
                gpu_times.push(median(&mut warm));
                let gpu_residual = true_residual(&system, &captured.rhs, &x_gpu);
                let gpu_error = error_of(&x_gpu);
                let difference = x_gpu
                    .iter()
                    .zip(&x_cpu)
                    .map(|(a, b)| (0..3).map(|t| (a[t] - b[t]).powi(2)).sum::<f64>())
                    .sum::<f64>()
                    .sqrt()
                    / x_cpu.iter().map(|a| (0..3).map(|t| a[t] * a[t]).sum::<f64>()).sum::<f64>().sqrt().max(1e-300);
                rows.push((cpu_iterations, stats.iterations, stats.passes, cpu_residual, gpu_residual, difference, stats.readbacks, stats.dispatches_per_iteration, cpu_error, gpu_error));
                assert!(
                    gpu_residual.0 <= tolerance && gpu_residual.1 <= tolerance,
                    "{width}x{height}: GPU true residual {gpu_residual:?} above {tolerance}"
                );
                // f32 with the bond-form operator: 1e-5..2e-3 from the near-exact
                // solution; assembled f32 blocks gave 1e-2 and more (rigid modes).
                assert!(gpu_error <= 5.0e-3, "{width}x{height}: GPU solution {gpu_error:e} from the reference");
                let slack = (cpu_iterations / 4).max(3);
                assert!(
                    stats.iterations <= cpu_iterations + slack,
                    "{width}x{height}: GPU {} iterations against CPU {cpu_iterations}",
                    stats.iterations
                );
            }
            let cells = systems[0].diagonal.len();
            eprintln!(
                "GPU PCG {width}x{width}x{height}: cells={cells} solves={} levels={levels} bottom_dofs={bottom}",
                systems.len()
            );
            for (cpu_it, gpu_it, passes, cpu_res, gpu_res, diff, readbacks, dispatches, cpu_err, gpu_err) in &rows {
                eprintln!(
                    "  iters cpu={cpu_it:3} gpu={gpu_it:3} passes={passes} readbacks={readbacks} dispatches/iter={dispatches} \
                     true force/moment residual cpu={:.2e}/{:.2e} gpu={:.2e}/{:.2e} |x_gpu-x_cpu|/|x_cpu|={diff:.2e} \
                     error vs 1e-11 reference cpu={cpu_err:.2e} gpu={gpu_err:.2e}",
                    cpu_res.0, cpu_res.1, gpu_res.0, gpu_res.1
                );
            }
            let cpu_iters: f64 = rows.iter().map(|r| r.0 as f64).sum::<f64>() / rows.len() as f64;
            let gpu_iters: f64 = rows.iter().map(|r| r.1 as f64).sum::<f64>() / rows.len() as f64;
            let (cpu_ms, gpu_ms) = (median(&mut cpu_times), median(&mut gpu_times));
            eprintln!(
                "  TIMING per solve (median): cpu {cpu_ms:.2} ms ({:.3} ms/iter at {cpu_iters:.1} iters), \
                 gpu {gpu_ms:.2} ms ({:.3} ms/iter at {gpu_iters:.1} iters), incl. upload {:.2} ms; hierarchy load {:.1} ms",
                cpu_ms / cpu_iters,
                gpu_ms / gpu_iters,
                median(&mut uploads),
                median(&mut loads),
            );
        }
    }

    /// `BOND_RESIDUAL` evaluated in f64 on the CPU must equal the assembled matrix.
    #[test]
    fn bond_form_reproduces_the_assembled_operator() {
        let systems = captured_bulk(6, 10, 6);
        for captured in systems.iter().take(3) {
            let system = CellSystem::new(&captured.diagonal, &captured.edges);
            let arrays = fine_arrays(&system, &captured.bonds, &captured.masks).expect("bond form");
            let n = system.count;
            let x: Vec<Vec6> = (0..n)
                .map(|i| std::array::from_fn(|p| ((i * 7 + p * 3) as f64).sin() * captured.masks[i][p]))
                .collect();
            let mut expected = vec![[0.0; 6]; n];
            system.apply(&x, &mut expected);
            // Remainder in f64 (the f32 copy only checks the layout).
            let mut worst = 0.0f64;
            let mut scale = 0.0f64;
            for i in 0..n {
                let mut y = [0.0f64; 6];
                let mut block = captured.diagonal[i];
                for bond in &captured.bonds {
                    let (ii, jj) = bond.diagonal_blocks();
                    let own = if bond.i as usize == i { Some(ii) } else if bond.j as usize == i { Some(jj) } else { None };
                    if let Some(own) = own {
                        for p in 0..6 {
                            for q in 0..6 {
                                block[p][q] -= own[p][q];
                            }
                        }
                    }
                }
                for p in 0..6 {
                    for q in 0..6 {
                        let masked = captured.masks[i][p] == 0.0 || captured.masks[i][q] == 0.0;
                        let value = if masked { if p == q && captured.masks[i][p] == 0.0 { 1.0 } else { 0.0 } } else { block[p][q] };
                        y[p] += value * x[i][q];
                        assert!(
                            masked || (f64::from(arrays.remainder[i * 36 + p * 6 + q]) - value).abs() <= 1e-6 * value.abs().max(1.0),
                        );
                    }
                }
                let (mut force, mut moment) = (DVec3::ZERO, DVec3::ZERO);
                for bond in &captured.bonds {
                    let (a, c) = (bond.i as usize, bond.j as usize);
                    if a != i && c != i {
                        continue;
                    }
                    let u = |k: usize| DVec3::new(x[k][0], x[k][1], x[k][2]);
                    let phi = |k: usize| DVec3::new(x[k][3], x[k][4], x[k][5]);
                    let strain = u(c) - u(a) + 0.5 * bond.rest.cross(phi(a) + phi(c));
                    let f = bond.k * strain;
                    let turn = if a == i { phi(a) - phi(c) } else { phi(c) - phi(a) };
                    force += if a == i { -f } else { f };
                    moment += 0.5 * f.cross(bond.rest) + bond.stiff * turn;
                }
                for p in 0..3 {
                    y[p] += force[p] * captured.masks[i][p];
                    y[p + 3] += moment[p] * captured.masks[i][p + 3];
                }
                for p in 0..6 {
                    worst = worst.max((y[p] - expected[i][p]).abs());
                    scale = scale.max(expected[i][p].abs());
                }
            }
            assert!(worst <= 1e-9 * scale, "bond form differs: {worst:e} of {scale:e}");
        }
    }

    #[test]
    fn gpu_pcg_matches_cpu_on_small_bulk_systems() {
        parity(&[(6, 10, 6)]);
    }

    #[test]
    #[ignore = "GPU parity and timing on 5 488 and 16 000 cells, about a minute"]
    fn gpu_pcg_matches_cpu_on_large_bulk_systems() {
        parity(&[(14, 28, 4), (20, 40, 4)]);
    }
}