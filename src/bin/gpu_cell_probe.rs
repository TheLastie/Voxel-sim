//! Probe: sparse 6x6-block matrix-vector product of the cell system on the GPU.
//!
//! This is a synthetic kernel benchmark, not a simulation benchmark. It builds a cubic
//! lattice with the same sparsity as the cell system (six face neighbours per REV,
//! one dense 6x6 f32 block per directed bond), runs `y = A x` on the wgpu adapter,
//! compares the result with an f64 CPU product of the same data, and reports time per
//! product and effective memory bandwidth. It is the measurement behind step C of
//! INTEGRATION_PLAN_2026-11-05.md: a resident GPU V-cycle is bounded by this
//! throughput, not by launch overhead.
//!
//! Usage: gpu_cell_probe [side=25] [repetitions=200]
use std::{borrow::Cow, time::Instant};
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Params { rows: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> diag: array<f32>;
@group(0) @binding(1) var<storage, read> row_start: array<u32>;
@group(0) @binding(2) var<storage, read> cols: array<u32>;
@group(0) @binding(3) var<storage, read> blocks: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.rows * 6u) { return; }
    let row = index / 6u;
    let component = index % 6u;
    var acc = 0.0;
    for (var q = 0u; q < 6u; q = q + 1u) {
        acc = acc + diag[row * 36u + component * 6u + q] * x[row * 6u + q];
    }
    for (var e = row_start[row]; e < row_start[row + 1u]; e = e + 1u) {
        let col = cols[e];
        let base = e * 36u + component * 6u;
        for (var q = 0u; q < 6u; q = q + 1u) {
            acc = acc + blocks[base + q] * x[col * 6u + q];
        }
    }
    y[index] = acc;
}
"#;

struct Lattice {
    rows: usize,
    diag: Vec<f32>,
    row_start: Vec<u32>,
    cols: Vec<u32>,
    blocks: Vec<f32>,
}

/// Deterministic pseudo-random numbers in [-1, 1).
fn random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn lattice(side: usize) -> Lattice {
    let rows = side * side * side;
    let index = |x: usize, y: usize, z: usize| (x * side + y) * side + z;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut diag = vec![0.0f32; rows * 36];
    let mut row_start = vec![0u32; rows + 1];
    let mut cols = Vec::new();
    let mut blocks = Vec::new();
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let row = index(x, y, z);
                for q in 0..36 {
                    diag[row * 36 + q] = random(&mut state) * 0.1;
                }
                for c in 0..6 {
                    diag[row * 36 + c * 6 + c] += 6.0;
                }
                let mut neighbours = Vec::new();
                if x > 0 { neighbours.push(index(x - 1, y, z)); }
                if x + 1 < side { neighbours.push(index(x + 1, y, z)); }
                if y > 0 { neighbours.push(index(x, y - 1, z)); }
                if y + 1 < side { neighbours.push(index(x, y + 1, z)); }
                if z > 0 { neighbours.push(index(x, y, z - 1)); }
                if z + 1 < side { neighbours.push(index(x, y, z + 1)); }
                for col in neighbours {
                    cols.push(col as u32);
                    for _ in 0..36 {
                        blocks.push(random(&mut state) * 0.1 - 0.0);
                    }
                }
                row_start[row + 1] = cols.len() as u32;
            }
        }
    }
    Lattice { rows, diag, row_start, cols, blocks }
}

fn cpu_product(l: &Lattice, x: &[f32]) -> Vec<f64> {
    let mut y = vec![0.0f64; l.rows * 6];
    for row in 0..l.rows {
        for c in 0..6 {
            let mut acc = 0.0f64;
            for q in 0..6 {
                acc += f64::from(l.diag[row * 36 + c * 6 + q]) * f64::from(x[row * 6 + q]);
            }
            for e in l.row_start[row] as usize..l.row_start[row + 1] as usize {
                let col = l.cols[e] as usize;
                for q in 0..6 {
                    acc += f64::from(l.blocks[e * 36 + c * 6 + q]) * f64::from(x[col * 6 + q]);
                }
            }
            y[row * 6 + c] = acc;
        }
    }
    y
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let side: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(25);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(200);
    let l = lattice(side);
    let mut state = 12345u64;
    let x: Vec<f32> = (0..l.rows * 6).map(|_| random(&mut state)).collect();
    let expected = cpu_product(&l, &x);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no wgpu adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cell probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("device");

    let storage = |label: &str, data: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let diag = storage("diag", bytemuck::cast_slice(&l.diag));
    let row_start = storage("row_start", bytemuck::cast_slice(&l.row_start));
    let cols = storage("cols", bytemuck::cast_slice(&l.cols));
    let blocks = storage("blocks", bytemuck::cast_slice(&l.blocks));
    let x_buffer = storage("x", bytemuck::cast_slice(&x));
    let y_size = (l.rows * 6 * 4) as u64;
    let y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y"),
        size: y_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: y_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&[l.rows as u32, 0, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cell spmv"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cell spmv"),
        layout: None,
        module: &module,
        entry_point: "main",
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cell spmv"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: diag.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: row_start.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cols.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: blocks.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: x_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: y_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: params.as_entire_binding() },
        ],
    });
    let groups = ((l.rows * 6 + 63) / 64) as u32;
    let dispatch = |count: usize, copy: bool| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        if copy {
            encoder.copy_buffer_to_buffer(&y_buffer, 0, &staging, 0, y_size);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);
    };

    // Warm-up, then parity.
    dispatch(5, true);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().expect("map");
    let mapped = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&mapped);
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for (a, b) in got.iter().zip(&expected) {
        worst = worst.max((f64::from(*a) - b).abs());
        scale = scale.max(b.abs());
    }
    drop(mapped);
    staging.unmap();

    let started = Instant::now();
    dispatch(reps, false);
    let seconds = started.elapsed().as_secs_f64() / reps as f64;
    let nnz = l.cols.len() as f64;
    let bytes = nnz * 144.0 + l.rows as f64 * 144.0 + l.rows as f64 * 24.0 * 2.0;
    println!(
        "cells={} directed bonds={} matrix={:.1} MB",
        l.rows,
        l.cols.len(),
        (nnz + l.rows as f64) * 144.0 / 1.0e6
    );
    println!("parity: max abs error {worst:.3e} relative {:.3e}", worst / scale.max(1e-30));
    println!(
        "spmv: {:.3} ms per product, {:.1} GB/s effective (minimum traffic, f32 blocks)",
        seconds * 1000.0,
        bytes / seconds / 1.0e9
    );
}
