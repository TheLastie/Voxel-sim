//! Optional CUDA PCG backend loaded through the CUDA Driver API.
//!
//! The sparse matrix is assembled by the engine on the host; the iterative
//! PCG solve (matvec, block-Jacobi preconditioner, reductions, and vector
//! updates) runs on the CUDA device. A missing driver/device is reported to
//! the caller so the engine can use its CPU solver.

use glam::Vec3;
use libloading::Library;
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

type Status = i32;
type DevicePtr = u64;
type Handle = usize;
type Init = unsafe extern "C" fn(u32) -> Status;
type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> Status;
type ContextCreate = unsafe extern "C" fn(*mut Handle, u32, i32) -> Status;
type ContextSetCurrent = unsafe extern "C" fn(Handle) -> Status;
type ModuleLoadData = unsafe extern "C" fn(*mut Handle, *const c_void) -> Status;
type ModuleLoadDataEx = unsafe extern "C" fn(*mut Handle, *const c_void, u32, *mut i32, *mut *mut c_void) -> Status;
type ModuleGetFunction = unsafe extern "C" fn(*mut Handle, Handle, *const i8) -> Status;
type MemAlloc = unsafe extern "C" fn(*mut DevicePtr, usize) -> Status;
type MemFree = unsafe extern "C" fn(DevicePtr) -> Status;
type MemsetD8 = unsafe extern "C" fn(DevicePtr, u8, usize) -> Status;
type CopyHtoD = unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> Status;
type CopyDtoH = unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> Status;
type StreamCreate = unsafe extern "C" fn(*mut Handle, u32) -> Status;
type StreamDestroy = unsafe extern "C" fn(Handle) -> Status;
type StreamSynchronize = unsafe extern "C" fn(Handle) -> Status;
type StreamBeginCapture = unsafe extern "C" fn(Handle, i32) -> Status;
type StreamEndCapture = unsafe extern "C" fn(Handle, *mut Handle) -> Status;
type GraphInstantiateWithFlags = unsafe extern "C" fn(*mut Handle, Handle, u64) -> Status;
type GraphLaunch = unsafe extern "C" fn(Handle, Handle) -> Status;
type GraphExecDestroy = unsafe extern "C" fn(Handle) -> Status;
type GraphDestroy = unsafe extern "C" fn(Handle) -> Status;
type LaunchKernel = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> Status;

// RTX 5060 Laptop measurements: single-CTA wins through 8,192 REV;
// multi-CTA wins starting at 16,384 REV.
const SINGLE_CTA_REDUCTION_MAX_REVS: u32 = 8_192;

/// Optional canonical per-edge blocks with their active-row slot mapping.
/// `edge_ids` entries use `u32::MAX` for empty slots; transpose bits select
/// the reverse orientation for the corresponding slot.
pub(crate) struct CompactEdgeBlocks<'a> {
    pub edges: &'a [[f32; 9]],
    pub edge_ids: &'a [[u32; 6]],
    pub transpose_masks: &'a [u8],
}

pub(crate) fn expand_edge_blocks_on_gpu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_CUDA_EXPAND_BLOCKS_ON_GPU")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

struct Driver {
    _library: Library,
    init: Init,
    device_get: DeviceGet,
    context_create: ContextCreate,
    context_set_current: ContextSetCurrent,
    module_load_data: ModuleLoadData,
    module_load_data_ex: ModuleLoadDataEx,
    module_get_function: ModuleGetFunction,
    mem_alloc: MemAlloc,
    mem_free: MemFree,
    memset_d8: MemsetD8,
    copy_htod: CopyHtoD,
    copy_dtoh: CopyDtoH,
    stream_create: StreamCreate,
    stream_destroy: StreamDestroy,
    stream_synchronize: StreamSynchronize,
    stream_begin_capture: Option<StreamBeginCapture>,
    stream_end_capture: Option<StreamEndCapture>,
    graph_instantiate_with_flags: Option<GraphInstantiateWithFlags>,
    graph_launch: Option<GraphLaunch>,
    graph_exec_destroy: Option<GraphExecDestroy>,
    graph_destroy: Option<GraphDestroy>,
    launch_kernel: LaunchKernel,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PcgGraphKey {
    count: u32,
    iterations: u32,
    diagonal: DevicePtr,
    p: DevicePtr,
    neighbors: DevicePtr,
    blocks: DevicePtr,
    colors: DevicePtr,
    ap: DevicePtr,
    inverse: DevicePtr,
    dx: DevicePtr,
    r: DevicePtr,
    z: DevicePtr,
    alpha: DevicePtr,
    rho_a: DevicePtr,
    rho_b: DevicePtr,
}

struct CudaSolver {
    driver: Driver,
    context: Handle,
    module: Handle,
    kernels: [Handle; 15],
    inverse_diagonal_kernel: Option<Handle>,
    expand_edge_blocks_kernel: Option<Handle>,
    buffers: HashMap<&'static str, (DevicePtr, usize)>,
    cached_neighbors: Vec<[u32; 6]>,
    cached_colors: Option<Vec<u8>>,
    cached_edge_ids: Vec<[u32; 6]>,
    cached_transpose_masks: Vec<u8>,
    host_rhs: Vec<f32>,
    stream: Handle,
    pcg_graph: Option<(PcgGraphKey, Handle)>,
    graphs_available: bool,
    contact_kernel: Option<Handle>,
    ic0_module: Option<Handle>,
    ic0_kernel: Option<Handle>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ContactOperatorEdge {
    pub slots: [u32; 2],
    /// Endpoint free masks, in xyz order.
    pub free: [[f32; 3]; 2],
    /// Constraint stiffness in glam Mat3 column-major order.
    pub stiffness: [f32; 9],
}

pub(crate) struct ContactOperatorInput<'a> {
    pub masses_over_h2: &'a [f64],
    pub free: &'a [[f32; 3]],
    /// Six incident edge IDs per active row; unused slots are `u32::MAX`.
    pub incidence: &'a [[u32; 6]],
    pub edges: &'a [ContactOperatorEdge],
}

pub(crate) struct ContactIc0Input<'a> {
    pub inverse_diagonal: &'a [[f32; 9]],
    pub edge_factors: &'a [[f32; 9]],
    pub lower_links: &'a [[[u32; 2]; 6]],
    pub lower_counts: &'a [u8],
    pub upper_links: &'a [[[u32; 2]; 6]],
    pub upper_counts: &'a [u8],
}

/// Owns the shared CUDA mutex for the whole contact solve, so no other CUDA
/// solver can replace these persistent topology buffers between matvecs.
pub(crate) struct ContactOperatorSession {
    cuda: std::sync::MutexGuard<'static, CudaSolver>,
    count: usize,
    mass: DevicePtr,
    row_free: DevicePtr,
    incidence: DevicePtr,
    edge_slots: DevicePtr,
    edge_free: DevicePtr,
    stiffness: DevicePtr,
    vector: DevicePtr,
    output: DevicePtr,
    host_vector: Vec<f32>,
    host_output: Vec<f32>,
    ic0: Option<cuda_contact_preconditioner::PreparedIc0>,
}

#[path = "cuda_contact_preconditioner.rs"]
mod cuda_contact_preconditioner;

// Kept in this Rust module so the optional inverse path can be A/B tested
// without changing the checked-in PTX artifact. Host CUDA matrices are packed
// row-major. Reading that same byte sequence as column-major gives A^T; the
// resulting inverse bytes are therefore the row-major packing of inverse(A).
const INVERSE_DIAGONAL_PTX: &str = r#"

.visible .entry block_inverse_diagonal(
    .param .u64 diagonal_ptr,
    .param .u64 inverse_ptr,
    .param .u32 count
) {
    .reg .pred %p<8>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<25>;
    ld.param.u64 %rd1, [diagonal_ptr];
    ld.param.u64 %rd2, [inverse_ptr];
    ld.param.u32 %r4, [count];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r2, %r3;
    setp.ge.u32 %p0, %r0, %r4;
    @%p0 bra INV_DONE;

    mul.wide.u32 %rd3, %r0, 36;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd4+4];
    ld.global.f32 %f2, [%rd4+8];
    ld.global.f32 %f3, [%rd4+12];
    ld.global.f32 %f4, [%rd4+16];
    ld.global.f32 %f5, [%rd4+20];
    ld.global.f32 %f6, [%rd4+24];
    ld.global.f32 %f7, [%rd4+28];
    ld.global.f32 %f8, [%rd4+32];

    // C = cof(A^T); the output bytes are row-major inverse(A) after scaling.
    mul.f32 %f9, %f4, %f8;
    neg.f32 %f23, %f7;
    fma.rn.f32 %f9, %f23, %f5, %f9;
    mul.f32 %f10, %f7, %f2;
    neg.f32 %f23, %f1;
    fma.rn.f32 %f10, %f23, %f8, %f10;
    mul.f32 %f11, %f1, %f5;
    neg.f32 %f23, %f4;
    fma.rn.f32 %f11, %f23, %f2, %f11;
    mul.f32 %f12, %f6, %f5;
    neg.f32 %f23, %f3;
    fma.rn.f32 %f12, %f23, %f8, %f12;
    mul.f32 %f13, %f0, %f8;
    neg.f32 %f23, %f6;
    fma.rn.f32 %f13, %f23, %f2, %f13;
    mul.f32 %f14, %f3, %f2;
    neg.f32 %f23, %f0;
    fma.rn.f32 %f14, %f23, %f5, %f14;
    mul.f32 %f15, %f3, %f7;
    neg.f32 %f23, %f6;
    fma.rn.f32 %f15, %f23, %f4, %f15;
    mul.f32 %f16, %f6, %f1;
    neg.f32 %f23, %f0;
    fma.rn.f32 %f16, %f23, %f7, %f16;
    mul.f32 %f17, %f0, %f4;
    neg.f32 %f23, %f3;
    fma.rn.f32 %f17, %f23, %f1, %f17;

    // Match glam's determinant expansion using z dot (x cross y).
    mul.f32 %f18, %f6, %f11;
    fma.rn.f32 %f18, %f7, %f14, %f18;
    fma.rn.f32 %f18, %f8, %f17, %f18;
    setp.eq.f32 %p1, %f18, 0f00000000;
    setp.ne.f32 %p2, %f18, %f18;
    abs.f32 %f19, %f18;
    setp.gtu.f32 %p3, %f19, 0f7f7fffff;
    or.pred %p4, %p1, %p2;
    or.pred %p4, %p4, %p3;
    @%p4 bra INV_INVALID;

    div.rn.f32 %f20, 0f3f800000, %f18;
    mul.wide.u32 %rd5, %r0, 36;
    add.u64 %rd5, %rd2, %rd5;
    mul.f32 %f21, %f9, %f20;
    st.global.f32 [%rd5], %f21;
    mul.f32 %f21, %f10, %f20;
    st.global.f32 [%rd5+4], %f21;
    mul.f32 %f21, %f11, %f20;
    st.global.f32 [%rd5+8], %f21;
    mul.f32 %f21, %f12, %f20;
    st.global.f32 [%rd5+12], %f21;
    mul.f32 %f21, %f13, %f20;
    st.global.f32 [%rd5+16], %f21;
    mul.f32 %f21, %f14, %f20;
    st.global.f32 [%rd5+20], %f21;
    mul.f32 %f21, %f15, %f20;
    st.global.f32 [%rd5+24], %f21;
    mul.f32 %f21, %f16, %f20;
    st.global.f32 [%rd5+28], %f21;
    mul.f32 %f21, %f17, %f20;
    st.global.f32 [%rd5+32], %f21;
    bra INV_DONE;

INV_INVALID:
    mul.wide.u32 %rd5, %r0, 36;
    add.u64 %rd5, %rd2, %rd5;
    mov.f32 %f22, 0f7fc00000;
    st.global.f32 [%rd5], %f22;
    st.global.f32 [%rd5+4], %f22;
    st.global.f32 [%rd5+8], %f22;
    st.global.f32 [%rd5+12], %f22;
    st.global.f32 [%rd5+16], %f22;
    st.global.f32 [%rd5+20], %f22;
    st.global.f32 [%rd5+24], %f22;
    st.global.f32 [%rd5+28], %f22;
    st.global.f32 [%rd5+32], %f22;
INV_DONE:
    ret;
}
"#;

const CONTACT_OPERATOR_PTX: &str = r#"
.visible .entry contact_difference_operator(
    .param .u64 mass_ptr, .param .u64 row_free_ptr, .param .u64 incidence_ptr,
    .param .u64 edge_slots_ptr, .param .u64 edge_free_ptr,
    .param .u64 stiffness_ptr, .param .u64 vector_ptr,
    .param .u64 output_ptr, .param .u32 count
) {
    .reg .pred %p<8>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<8>;
    .reg .f64 %d<12>;
    .local .align 8 .b8 delta[24];
    .local .align 8 .b8 result[24];
    ld.param.u64 %rd1, [mass_ptr]; ld.param.u64 %rd2, [row_free_ptr];
    ld.param.u64 %rd3, [incidence_ptr]; ld.param.u64 %rd4, [edge_slots_ptr];
    ld.param.u64 %rd5, [edge_free_ptr]; ld.param.u64 %rd6, [stiffness_ptr];
    ld.param.u64 %rd7, [vector_ptr]; ld.param.u64 %rd8, [output_ptr];
    ld.param.u32 %r1, [count];
    mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
    mad.lo.u32 %r0, %r2, %r3, %r4;
    setp.ge.u32 %p0, %r0, %r1; @%p0 bra CONTACT_DONE;
    mul.wide.u32 %rd9, %r0, 24; add.u64 %rd10, %rd3, %rd9;
    mul.wide.u32 %rd11, %r0, 12; add.u64 %rd12, %rd7, %rd11;
    add.u64 %rd13, %rd2, %rd11;
    mul.wide.u32 %rd14, %r0, 8; add.u64 %rd14, %rd1, %rd14;
    ld.global.f64 %d0, [%rd14];
    // Initial inertial term, evaluated in f64 after f32 free-mask multiplication.
    mov.u32 %r5, 0;
CONTACT_INIT:
    setp.ge.u32 %p1, %r5, 3; @%p1 bra CONTACT_EDGES_START;
    mul.wide.u32 %rd14, %r5, 4; add.u64 %rd15, %rd12, %rd14;
    ld.global.f32 %f0, [%rd15]; add.u64 %rd16, %rd13, %rd14;
    ld.global.f32 %f1, [%rd16]; mul.rn.f32 %f2, %f0, %f1;
    cvt.f64.f32 %d1, %f2; mul.rn.f64 %d2, %d1, %d0;
    mul.wide.u32 %rd16, %r5, 8; mov.u64 %rd15, result;
    add.u64 %rd15, %rd15, %rd16; st.local.f64 [%rd15], %d2;
    cvt.f64.f32 %d3, %f2; mov.u64 %rd15, delta;
    add.u64 %rd15, %rd15, %rd16; st.local.f64 [%rd15], %d3;
    add.u32 %r5, %r5, 1; bra CONTACT_INIT;
CONTACT_EDGES_START:
    mov.u32 %r6, 0;
CONTACT_EDGE_LOOP:
    setp.ge.u32 %p2, %r6, 6; @%p2 bra CONTACT_STORE_START;
    mul.wide.u32 %rd14, %r6, 4; add.u64 %rd15, %rd10, %rd14;
    ld.global.u32 %r7, [%rd15]; setp.eq.u32 %p3, %r7, 0xffffffff; @%p3 bra CONTACT_NEXT_EDGE;
    mul.wide.u32 %rd14, %r7, 8; add.u64 %rd15, %rd4, %rd14;
    ld.global.u32 %r8, [%rd15]; ld.global.u32 %r9, [%rd15+4];
    // Mask each endpoint in f32, promote both, then subtract in f64.
    mov.u32 %r5, 0;
CONTACT_DIFF:
    setp.ge.u32 %p4, %r5, 3; @%p4 bra CONTACT_FORCE_START;
    mul.wide.u32 %rd16, %r5, 4; mul.wide.u32 %rd17, %r8, 12;
    add.u64 %rd18, %rd7, %rd17; add.u64 %rd18, %rd18, %rd16;
    setp.eq.u32 %p7, %r8, 0xffffffff; @%p7 mov.f32 %f0, 0f00000000; @!%p7 ld.global.f32 %f0, [%rd18]; mul.wide.u32 %rd17, %r9, 12;
    add.u64 %rd19, %rd7, %rd17; add.u64 %rd19, %rd19, %rd16;
    setp.eq.u32 %p7, %r9, 0xffffffff; @%p7 mov.f32 %f1, 0f00000000; @!%p7 ld.global.f32 %f1, [%rd19]; mul.wide.u32 %rd17, %r7, 24;
    add.u64 %rd18, %rd5, %rd17; add.u64 %rd18, %rd18, %rd16;
    ld.global.f32 %f2, [%rd18]; add.u64 %rd19, %rd18, 12;
    ld.global.f32 %f3, [%rd19]; mul.rn.f32 %f4, %f0, %f2;
    mul.rn.f32 %f5, %f1, %f3; cvt.f64.f32 %d8, %f4; cvt.f64.f32 %d9, %f5;
    sub.rn.f64 %d3, %d8, %d9;
    mul.wide.u32 %rd16, %r5, 8; mov.u64 %rd15, delta;
    add.u64 %rd15, %rd15, %rd16; st.local.f64 [%rd15], %d3;
    add.u32 %r5, %r5, 1; bra CONTACT_DIFF;
CONTACT_FORCE_START:
    mov.u32 %r10, 0;
CONTACT_FORCE_AXIS:
    setp.ge.u32 %p5, %r10, 3; @%p5 bra CONTACT_NEXT_EDGE;
    mov.f64 %d4, 0d0000000000000000; mov.u32 %r11, 0;
CONTACT_FORCE_COL:
    setp.ge.u32 %p6, %r11, 3; @%p6 bra CONTACT_FORCE_DONE;
    // stiffness is column-major: element(column,row) = column*3 + row.
    mul.wide.u32 %rd16, %r7, 36; mul.wide.u32 %rd17, %r10, 4;
    mul.wide.u32 %rd18, %r11, 12; add.u64 %rd16, %rd16, %rd17;
    add.u64 %rd16, %rd16, %rd18; add.u64 %rd16, %rd6, %rd16;
    ld.global.f32 %f0, [%rd16]; cvt.f64.f32 %d5, %f0;
    mul.wide.u32 %rd19, %r11, 8; mov.u64 %rd18, delta;
    add.u64 %rd18, %rd18, %rd19; ld.local.f64 %d6, [%rd18]; mul.rn.f64 %d7, %d5, %d6;
    add.rn.f64 %d4, %d4, %d7; add.u32 %r11, %r11, 1; bra CONTACT_FORCE_COL;
CONTACT_FORCE_DONE:
    // Orientation follows the CPU endpoint test: c.i == row means add.
    setp.eq.u32 %p7, %r0, %r8; @%p7 bra CONTACT_ADD_FORCE;
    neg.f64 %d4, %d4;
CONTACT_ADD_FORCE:
    mul.wide.u32 %rd16, %r10, 8;
    // Multiply by the row endpoint's free mask in f64.
    mul.wide.u32 %rd17, %r10, 4; mul.wide.u32 %rd18, %r0, 12;
    add.u64 %rd18, %rd2, %rd18; add.u64 %rd18, %rd18, %rd17;
    ld.global.f32 %f0, [%rd18]; cvt.f64.f32 %d6, %f0;
    mul.rn.f64 %d4, %d4, %d6; mov.u64 %rd15, result;
    add.u64 %rd15, %rd15, %rd16; ld.local.f64 %d5, [%rd15];
    add.rn.f64 %d5, %d5, %d4; mov.u64 %rd15, result;
    add.u64 %rd15, %rd15, %rd16; st.local.f64 [%rd15], %d5;
    add.u32 %r10, %r10, 1; bra CONTACT_FORCE_AXIS;
CONTACT_NEXT_EDGE:
    add.u32 %r6, %r6, 1; bra CONTACT_EDGE_LOOP;
CONTACT_STORE_START:
    mov.u32 %r5, 0;
CONTACT_STORE:
    setp.ge.u32 %p7, %r5, 3; @%p7 bra CONTACT_DONE;
    mul.wide.u32 %rd14, %r5, 8; mov.u64 %rd15, result;
    add.u64 %rd15, %rd15, %rd14; ld.local.f64 %d0, [%rd15]; cvt.rn.f32.f64 %f0, %d0;
    mul.wide.u32 %rd14, %r5, 4; add.u64 %rd15, %rd8, %rd11;
    add.u64 %rd15, %rd15, %rd14; st.global.f32 [%rd15], %f0;
    add.u32 %r5, %r5, 1; bra CONTACT_STORE;
CONTACT_DONE:
    ret;
}
"#;

const EXPAND_EDGE_BLOCKS_PTX: &str = r#"
.visible .entry expand_edge_blocks(
    .param .u64 edge_blocks_ptr,
    .param .u64 edge_ids_ptr,
    .param .u64 transpose_masks_ptr,
    .param .u64 output_blocks_ptr,
    .param .u32 count
) {
    .reg .pred %p<5>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [edge_blocks_ptr];
    ld.param.u64 %rd2, [edge_ids_ptr];
    ld.param.u64 %rd3, [transpose_masks_ptr];
    ld.param.u64 %rd4, [output_blocks_ptr];
    ld.param.u32 %r10, [count];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    setp.ge.u32 %p1, %r3, %r10;
    @%p1 bra EXP_DONE;

    mul.wide.u32 %rd5, %r3, 24;
    add.u64 %rd6, %rd2, %rd5;
    mul.wide.u32 %rd7, %r3, 216;
    add.u64 %rd8, %rd4, %rd7;
    mul.wide.u32 %rd9, %r3, 1;
    add.u64 %rd9, %rd3, %rd9;
    ld.global.u8 %r11, [%rd9];
    mov.u32 %r4, 0;
EXP_SLOT:
    setp.ge.u32 %p1, %r4, 6;
    @%p1 bra EXP_DONE;
    mul.wide.u32 %rd5, %r4, 4;
    add.u64 %rd5, %rd6, %rd5;
    ld.global.u32 %r5, [%rd5];
    mul.wide.u32 %rd6, %r4, 36;
    add.u64 %rd6, %rd8, %rd6;
    shr.u32 %r12, %r11, %r4;
    and.b32 %r12, %r12, 1;
    setp.ne.u32 %p2, %r12, 0;
    mov.u32 %r6, 0;
EXP_ELEM:
    setp.ge.u32 %p3, %r6, 9;
    @%p3 bra EXP_NEXT_SLOT;
    setp.eq.u32 %p4, %r5, 0xffffffff;
    @%p4 bra EXP_ZERO;
    div.u32 %r13, %r6, 3;
    rem.u32 %r14, %r6, 3;
    mul.lo.u32 %r14, %r14, 3;
    add.u32 %r14, %r14, %r13;
    selp.u32 %r14, %r14, %r6, %p2;
    mul.wide.u32 %rd5, %r5, 36;
    add.u64 %rd5, %rd1, %rd5;
    mul.wide.u32 %rd7, %r14, 4;
    add.u64 %rd5, %rd5, %rd7;
    ld.global.u32 %r15, [%rd5];
    bra EXP_STORE;
EXP_ZERO:
    mov.u32 %r15, 0;
EXP_STORE:
    mul.wide.u32 %rd5, %r6, 4;
    add.u64 %rd5, %rd6, %rd5;
    st.global.u32 [%rd5], %r15;
    add.u32 %r6, %r6, 1;
    bra EXP_ELEM;
EXP_NEXT_SLOT:
    // Restore the row metadata bases after the per-element scratch use.
    mul.wide.u32 %rd6, %r3, 24;
    add.u64 %rd6, %rd2, %rd6;
    mul.wide.u32 %rd7, %r3, 216;
    add.u64 %rd8, %rd4, %rd7;
    add.u32 %r4, %r4, 1;
    bra EXP_SLOT;
EXP_DONE:
    ret;
}
"#;

static CUDA: OnceLock<Result<Mutex<CudaSolver>, String>> = OnceLock::new();

impl ContactOperatorSession {
    pub(crate) fn new(input: ContactOperatorInput<'_>) -> Result<Self, String> {
        let count = input.masses_over_h2.len();
        if count == 0 || count > u32::MAX as usize {
            return Err(format!("CUDA contact operator requires 1..=u32::MAX rows, got {count}"));
        }
        if input.free.len() != count || input.incidence.len() != count {
            return Err("CUDA contact operator row arrays have inconsistent lengths".into());
        }
        for (row, mask) in input.free.iter().enumerate() {
            if !mask.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)) {
                return Err(format!("CUDA contact row {row} has an invalid free mask"));
            }
            if !input.masses_over_h2[row].is_finite() || input.masses_over_h2[row] < 0.0 {
                return Err(format!("CUDA contact row {row} has invalid mass/h^2"));
            }
        }
        for (edge_id, edge) in input.edges.iter().enumerate() {
            if edge.slots.iter().any(|&slot| slot != u32::MAX && slot as usize >= count)
                || !edge.free.iter().flatten().all(|x| x.is_finite() && (0.0..=1.0).contains(x))
                || !edge.stiffness.iter().all(|x| x.is_finite())
            {
                return Err(format!("CUDA contact edge {edge_id} has invalid endpoints or coefficients"));
            }
        }
        for (row, edges) in input.incidence.iter().enumerate() {
            let mut padding = false;
            let mut degree = 0;
            for &edge_id in edges {
                if edge_id == u32::MAX { padding = true; continue; }
                if padding || edge_id as usize >= input.edges.len() {
                    return Err(format!("CUDA contact row {row} has invalid incidence padding or edge ID"));
                }
                let edge = input.edges[edge_id as usize];
                if edge.slots[0] != row as u32 && edge.slots[1] != row as u32 {
                    return Err(format!("CUDA contact row {row} references a nonincident edge"));
                }
                degree += 1;
            }
            if degree > 6 { return Err(format!("CUDA contact row {row} exceeds six incident edges")); }
        }

        let store = CUDA.get_or_init(|| CudaSolver::initialize().map(Mutex::new));
        let mut cuda = match store {
            Ok(cuda) => cuda.lock().map_err(|_| "CUDA solver lock poisoned".to_string())?,
            Err(error) => return Err(error.clone()),
        };
        check(unsafe { (cuda.driver.context_set_current)(cuda.context) }, "cuCtxSetCurrent(contact operator)")?;
        cuda.ensure_contact_kernel()?;
        let slots: Vec<[u32; 2]> = input.edges.iter().map(|edge| edge.slots).collect();
        let edge_free: Vec<[f32; 6]> = input.edges.iter().map(|edge| [
            edge.free[0][0], edge.free[0][1], edge.free[0][2],
            edge.free[1][0], edge.free[1][1], edge.free[1][2],
        ]).collect();
        let stiffness: Vec<[f32; 9]> = input.edges.iter().map(|edge| edge.stiffness).collect();
        let mass = cuda.upload("contact_mass", input.masses_over_h2)?;
        let row_free = cuda.upload("contact_row_free", input.free)?;
        let incidence = cuda.upload("contact_incidence", input.incidence)?;
        let edge_slots = cuda.upload("contact_edge_slots", &slots)?;
        let edge_free = cuda.upload("contact_edge_free", &edge_free)?;
        let stiffness = cuda.upload("contact_stiffness", &stiffness)?;
        let vector = cuda.buffer("contact_vector", count * 12)?;
        let output = cuda.buffer("contact_output", count * 12)?;
        Ok(Self {
            cuda, count, mass, row_free, incidence, edge_slots, edge_free, stiffness,
            vector, output, host_vector: vec![0.0; count * 3], host_output: vec![0.0; count * 3],
            ic0: None,
        })
    }

    pub(crate) fn prepare_ic0(&mut self, input: ContactIc0Input<'_>) -> Result<(), String> {
        self.ic0 = None;
        let prepared = cuda_contact_preconditioner::prepare(&mut self.cuda, self.count, input)?;
        self.ic0 = Some(prepared);
        Ok(())
    }

    pub(crate) fn apply_ic0(&mut self, rhs: &[Vec3], out: &mut [Vec3]) -> Result<(), String> {
        let prepared = self.ic0.as_mut().ok_or_else(|| "CUDA IC0 has not been prepared".to_string())?;
        cuda_contact_preconditioner::apply(&mut self.cuda, self.count, prepared, rhs, out)
    }

    pub(crate) fn ic0_level_counts(&self) -> Option<[usize; 2]> {
        self.ic0.as_ref().map(cuda_contact_preconditioner::level_counts)
    }

    pub(crate) fn apply(&mut self, vector: &[Vec3], output: &mut [Vec3]) -> Result<(), String> {
        if vector.len() != self.count || output.len() != self.count {
            return Err(format!("CUDA contact matvec expected {} rows, got {} and {}", self.count, vector.len(), output.len()));
        }
        self.host_vector.clear();
        self.host_vector.extend(vector.iter().flat_map(|v| [v.x, v.y, v.z]));
        if !self.host_vector.iter().all(|x| x.is_finite()) {
            return Err("CUDA contact matvec input contains non-finite values".into());
        }
        let count = u32::try_from(self.count).map_err(|_| "CUDA contact row count exceeds u32::MAX".to_string())?;
        self.cuda.upload("contact_vector", &self.host_vector)?;
        self.cuda.launch_contact_operator(self.mass, self.row_free, self.incidence,
            self.edge_slots, self.edge_free, self.stiffness, self.vector, self.output, count)?;
        self.cuda.download(self.host_output.as_mut_ptr().cast(), self.output, self.host_output.len() * 4)?;
        if !self.host_output.iter().all(|x| x.is_finite()) {
            return Err("CUDA contact matvec produced non-finite output".into());
        }
        for (dst, xyz) in output.iter_mut().zip(self.host_output.chunks_exact(3)) {
            *dst = Vec3::new(xyz[0], xyz[1], xyz[2]);
        }
        Ok(())
    }
}

fn bipartite_colors(neighbors: &[[u32; 6]]) -> Option<Vec<u8>> {
    let mut colors = vec![u8::MAX; neighbors.len()];
    let mut queue = Vec::with_capacity(neighbors.len());
    let mut has_edge = false;
    for root in 0..neighbors.len() {
        if colors[root] != u8::MAX {
            continue;
        }
        colors[root] = 0;
        queue.clear();
        queue.push(root);
        let mut head = 0;
        while head < queue.len() {
            let row = queue[head];
            head += 1;
            for &neighbor in &neighbors[row] {
                if neighbor == u32::MAX {
                    continue;
                }
                has_edge = true;
                let neighbor = neighbor as usize;
                if neighbor >= neighbors.len() || neighbor == row {
                    return None;
                }
                let expected = colors[row] ^ 1;
                if colors[neighbor] == u8::MAX {
                    colors[neighbor] = expected;
                    queue.push(neighbor);
                } else if colors[neighbor] != expected {
                    return None;
                }
            }
        }
    }
    has_edge.then_some(colors)
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    // CUDA's public C ABI uses stable function pointers for these symbols.
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol| *symbol)
            .map_err(|e| e.to_string())
    }
}

fn optional_symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
}

impl CudaSolver {
    fn initialize() -> Result<Self, String> {
        Self::initialize_with_kernels(
            inverse_diagonal_on_gpu_enabled(),
            expand_edge_blocks_on_gpu_enabled(),
        )
    }

    #[cfg(test)]
    fn initialize_with_inverse_diagonal(
        load_inverse_diagonal_kernel: bool,
    ) -> Result<Self, String> {
        Self::initialize_with_kernels(load_inverse_diagonal_kernel, false)
    }

    fn initialize_with_kernels(
        load_inverse_diagonal_kernel: bool,
        load_expand_edge_blocks_kernel: bool,
    ) -> Result<Self, String> {
        #[cfg(target_os = "linux")]
        let library = unsafe { Library::new("libcuda.so.1") }.map_err(|e| e.to_string())?;
        #[cfg(target_os = "windows")]
        let library = unsafe { Library::new("nvcuda.dll") }.map_err(|e| e.to_string())?;
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        return Err("CUDA Driver API is unsupported on this platform".into());

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let driver = Driver {
                init: symbol(&library, b"cuInit\0")?,
                device_get: symbol(&library, b"cuDeviceGet\0")?,
                context_create: symbol(&library, b"cuCtxCreate_v2\0")?,
                context_set_current: symbol(&library, b"cuCtxSetCurrent\0")?,
                module_load_data: symbol(&library, b"cuModuleLoadData\0")?,
                module_load_data_ex: symbol(&library, b"cuModuleLoadDataEx\0")?,
                module_get_function: symbol(&library, b"cuModuleGetFunction\0")?,
                mem_alloc: symbol(&library, b"cuMemAlloc_v2\0")?,
                mem_free: symbol(&library, b"cuMemFree_v2\0")?,
                memset_d8: symbol(&library, b"cuMemsetD8_v2\0")?,
                copy_htod: symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                copy_dtoh: symbol(&library, b"cuMemcpyDtoH_v2\0")?,
                stream_create: symbol(&library, b"cuStreamCreate\0")?,
                stream_destroy: symbol(&library, b"cuStreamDestroy_v2\0")?,
                stream_synchronize: symbol(&library, b"cuStreamSynchronize\0")?,
                stream_begin_capture: optional_symbol(&library, b"cuStreamBeginCapture\0"),
                stream_end_capture: optional_symbol(&library, b"cuStreamEndCapture\0"),
                graph_instantiate_with_flags: optional_symbol(
                    &library,
                    b"cuGraphInstantiateWithFlags\0",
                ),
                graph_launch: optional_symbol(&library, b"cuGraphLaunch\0"),
                graph_exec_destroy: optional_symbol(&library, b"cuGraphExecDestroy\0"),
                graph_destroy: optional_symbol(&library, b"cuGraphDestroy\0"),
                launch_kernel: symbol(&library, b"cuLaunchKernel\0")?,
                _library: library,
            };
            check(unsafe { (driver.init)(0) }, "cuInit")?;
            let mut device = 0;
            check(
                unsafe { (driver.device_get)(&mut device, 0) },
                "cuDeviceGet(0)",
            )?;
            let mut context = 0;
            check(
                unsafe { (driver.context_create)(&mut context, 0, device) },
                "cuCtxCreate",
            )?;
            check(
                unsafe { (driver.context_set_current)(context) },
                "cuCtxSetCurrent",
            )?;
            let mut stream = 0;
            check(
                // Pageable synchronous HtoD copies may return after staging,
                // before DMA finishes. A stream created with the nonblocking
                // flag skips legacy-stream ordering, so subsequent kernels can
                // read a large upload too early. Preserve the driver's default
                // legacy ordering without synchronizing the host per upload.
                unsafe { (driver.stream_create)(&mut stream, 0) },
                "cuStreamCreate(legacy-ordered)",
            )?;
            let ptx = include_bytes!("cuda/pcg.ptx");
            let mut ptx_z = ptx.to_vec();
            ptx_z.push(0);
            let mut module = 0;
            let status = unsafe { (driver.module_load_data)(&mut module, ptx_z.as_ptr().cast()) };
            if status != 0 {
                return Err(format!(
                    "cuModuleLoadData(PCG PTX) failed with CUDA status {status}"
                ));
            }
            let mut kernels = [0; 15];
            for (slot, name) in [
                b"block_sparse_matvec\0".as_slice(),
                b"block_jacobi\0".as_slice(),
                b"pcg_update\0".as_slice(),
                b"pcg_update_direction\0".as_slice(),
                b"dot_product\0".as_slice(),
                b"dot_reduce\0".as_slice(),
                b"dot_reduce_ratio\0".as_slice(),
                b"scalar_ratio\0".as_slice(),
                b"pcg_update_precondition\0".as_slice(),
                b"dot_product_single\0".as_slice(),
                b"dot_reduce_ratio_single\0".as_slice(),
                b"pcg_update_direction_ratio\0".as_slice(),
                b"block_color_correction\0".as_slice(),
                b"pcg_residual_replace\0".as_slice(),
                b"block_sparse_matvec_dot\0".as_slice(),
            ]
            .into_iter()
            .enumerate()
            {
                check(
                    unsafe {
                        (driver.module_get_function)(
                            &mut kernels[slot],
                            module,
                            name.as_ptr().cast(),
                        )
                    },
                    "cuModuleGetFunction",
                )?;
            }
            let inverse_diagonal_kernel = if load_inverse_diagonal_kernel {
                let mut inverse_ptx = b".version 7.0\n.target sm_75\n.address_size 64\n".to_vec();
                inverse_ptx.extend_from_slice(INVERSE_DIAGONAL_PTX.as_bytes());
                inverse_ptx.push(0);
                let mut inverse_module = 0;
                check(
                    unsafe {
                        (driver.module_load_data)(&mut inverse_module, inverse_ptx.as_ptr().cast())
                    },
                    "cuModuleLoadData(inverse-diagonal PTX)",
                )?;
                let mut kernel = 0;
                check(
                    unsafe {
                        (driver.module_get_function)(
                            &mut kernel,
                            inverse_module,
                            b"block_inverse_diagonal\0".as_ptr().cast(),
                        )
                    },
                    "cuModuleGetFunction(block_inverse_diagonal)",
                )?;
                Some(kernel)
            } else {
                None
            };
            let expand_edge_blocks_kernel = if load_expand_edge_blocks_kernel {
                let mut expand_ptx = b".version 7.0\n.target sm_75\n.address_size 64\n".to_vec();
                expand_ptx.extend_from_slice(EXPAND_EDGE_BLOCKS_PTX.as_bytes());
                expand_ptx.push(0);
                let mut expand_module = 0;
                check(
                    unsafe {
                        (driver.module_load_data)(
                            &mut expand_module,
                            expand_ptx.as_ptr().cast(),
                        )
                    },
                    "cuModuleLoadData(edge-block expansion PTX)",
                )?;
                let mut kernel = 0;
                check(
                    unsafe {
                        (driver.module_get_function)(
                            &mut kernel,
                            expand_module,
                            b"expand_edge_blocks\0".as_ptr().cast(),
                        )
                    },
                    "cuModuleGetFunction(expand_edge_blocks)",
                )?;
                Some(kernel)
            } else {
                None
            };
            let graphs_available = driver.stream_begin_capture.is_some()
                && driver.stream_end_capture.is_some()
                && driver.graph_instantiate_with_flags.is_some()
                && driver.graph_launch.is_some()
                && driver.graph_exec_destroy.is_some()
                && driver.graph_destroy.is_some();
            Ok(Self {
                driver,
                context,
                module,
                kernels,
                inverse_diagonal_kernel,
                expand_edge_blocks_kernel,
                buffers: HashMap::new(),
                cached_neighbors: Vec::new(),
                cached_colors: None,
                cached_edge_ids: Vec::new(),
                cached_transpose_masks: Vec::new(),
                host_rhs: Vec::new(),
                stream,
                pcg_graph: None,
                graphs_available,
                contact_kernel: None,
                ic0_module: None,
                ic0_kernel: None,
            })
        }
    }

    fn buffer(&mut self, name: &'static str, bytes: usize) -> Result<DevicePtr, String> {
        let bytes = bytes.max(4);
        if let Some((pointer, capacity)) = self.buffers.get(name).copied() {
            if capacity >= bytes {
                return Ok(pointer);
            }
            check(
                unsafe { (self.driver.mem_free)(pointer) },
                "cuMemFree(old buffer)",
            )?;
            self.buffers.remove(name);
        }
        let mut pointer = 0;
        check(
            unsafe { (self.driver.mem_alloc)(&mut pointer, bytes) },
            "cuMemAlloc",
        )?;
        self.buffers.insert(name, (pointer, bytes));
        Ok(pointer)
    }

    fn ensure_contact_kernel(&mut self) -> Result<Handle, String> {
        if let Some(kernel) = self.contact_kernel { return Ok(kernel); }
        let mut ptx = b".version 7.0\n.target sm_75\n.address_size 64\n".to_vec();
        ptx.extend_from_slice(CONTACT_OPERATOR_PTX.as_bytes());
        ptx.push(0);
        let mut module = 0;
        let mut jit_log = vec![0i8; 8192];
        let mut log_capacity = jit_log.len() as u32;
        let mut options = [5i32, 6i32];
        let mut values = [jit_log.as_mut_ptr().cast::<c_void>(), (&mut log_capacity as *mut u32).cast()];
        let status = unsafe { (self.driver.module_load_data_ex)(
            &mut module, ptx.as_ptr().cast(), options.len() as u32,
            options.as_mut_ptr(), values.as_mut_ptr(),
        ) };
        if status != 0 {
            let diagnostic = unsafe { std::ffi::CStr::from_ptr(jit_log.as_ptr()) }.to_string_lossy();
            return Err(format!("cuModuleLoadData(contact operator PTX) failed with CUDA status {status}: {diagnostic}"));
        }
        let mut kernel = 0;
        check(unsafe { (self.driver.module_get_function)(
            &mut kernel, module, b"contact_difference_operator\0".as_ptr().cast(),
        ) }, "cuModuleGetFunction(contact_difference_operator)")?;
        self.contact_kernel = Some(kernel);
        Ok(kernel)
    }

    fn upload<T>(&mut self, name: &'static str, values: &[T]) -> Result<DevicePtr, String> {
        let bytes = std::mem::size_of_val(values);
        let pointer = self.buffer(name, bytes)?;
        if bytes != 0 {
            check(
                unsafe { (self.driver.copy_htod)(pointer, values.as_ptr().cast(), bytes) },
                "cuMemcpyHtoD",
            )?;
        }
        Ok(pointer)
    }

    fn upload_compact_edge_metadata(
        &mut self,
        edge_ids: &[[u32; 6]],
        transpose_masks: &[u8],
    ) -> Result<(DevicePtr, DevicePtr), String> {
        if self.cached_edge_ids == edge_ids && self.cached_transpose_masks == transpose_masks {
            if let (Some((edge_ids_ptr, _)), Some((transpose_masks_ptr, _))) = (
                self.buffers.get("edge_ids"),
                self.buffers.get("edge_transpose_masks"),
            ) {
                return Ok((*edge_ids_ptr, *transpose_masks_ptr));
            }
        }
        let edge_ids_ptr = self.upload("edge_ids", edge_ids)?;
        let transpose_masks_ptr = self.upload("edge_transpose_masks", transpose_masks)?;
        self.cached_edge_ids.clear();
        self.cached_edge_ids.extend_from_slice(edge_ids);
        self.cached_transpose_masks.clear();
        self.cached_transpose_masks
            .extend_from_slice(transpose_masks);
        Ok((edge_ids_ptr, transpose_masks_ptr))
    }

    fn launch(&self, kernel: usize, grid: u32, params: &mut [*mut c_void]) -> Result<(), String> {
        check(
            unsafe {
                (self.driver.launch_kernel)(
                    self.kernels[kernel],
                    grid,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    fn launch_inverse_diagonal(
        &self,
        diagonal: DevicePtr,
        inverse: DevicePtr,
        count: u32,
    ) -> Result<(), String> {
        let kernel = self.inverse_diagonal_kernel.ok_or_else(|| {
            "CUDA inverse-diagonal kernel was not enabled at initialization".to_string()
        })?;
        let mut diagonal_arg = diagonal;
        let mut inverse_arg = inverse;
        let mut count_arg = count;
        let mut params = [
            (&mut diagonal_arg as *mut DevicePtr).cast(),
            (&mut inverse_arg as *mut DevicePtr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        check(
            unsafe {
                (self.driver.launch_kernel)(
                    kernel,
                    count.div_ceil(256),
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel(block_inverse_diagonal)",
        )
    }

    fn launch_expand_edge_blocks(
        &self,
        edge_blocks: DevicePtr,
        edge_ids: DevicePtr,
        transpose_masks: DevicePtr,
        output_blocks: DevicePtr,
        count: u32,
    ) -> Result<(), String> {
        let kernel = self.expand_edge_blocks_kernel.ok_or_else(|| {
            "CUDA edge-block expansion kernel was not enabled at initialization".to_string()
        })?;
        let mut edge_blocks_arg = edge_blocks;
        let mut edge_ids_arg = edge_ids;
        let mut transpose_masks_arg = transpose_masks;
        let mut output_blocks_arg = output_blocks;
        let mut count_arg = count;
        let mut params = [
            (&mut edge_blocks_arg as *mut DevicePtr).cast(),
            (&mut edge_ids_arg as *mut DevicePtr).cast(),
            (&mut transpose_masks_arg as *mut DevicePtr).cast(),
            (&mut output_blocks_arg as *mut DevicePtr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        check(
            unsafe {
                (self.driver.launch_kernel)(
                    kernel,
                    count.div_ceil(256),
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel(expand_edge_blocks)",
        )
    }

    fn pcg_graph_exec(
        &mut self,
        key: PcgGraphKey,
        grid: u32,
        vector_grid: u32,
    ) -> Result<Option<Handle>, String> {
        if !self.graphs_available {
            return Ok(None);
        }
        if let Some((cached_key, executable)) = self.pcg_graph {
            if cached_key == key {
                return Ok(Some(executable));
            }
        }
        if let Some((_, executable)) = self.pcg_graph.take() {
            let destroy = self
                .driver
                .graph_exec_destroy
                .expect("graph support checked during CUDA initialization");
            check(unsafe { destroy(executable) }, "cuGraphExecDestroy")?;
        }
        if key.count > SINGLE_CTA_REDUCTION_MAX_REVS {
            let partial_count = key.count.div_ceil(256);
            self.buffer("dot_partials", partial_count as usize * 8)?;
        }
        let begin_capture = self
            .driver
            .stream_begin_capture
            .expect("graph support checked during CUDA initialization");
        let end_capture = self
            .driver
            .stream_end_capture
            .expect("graph support checked during CUDA initialization");
        let graph_destroy = self
            .driver
            .graph_destroy
            .expect("graph support checked during CUDA initialization");
        let graph_instantiate = self
            .driver
            .graph_instantiate_with_flags
            .expect("graph support checked during CUDA initialization");
        let begin_status = unsafe { begin_capture(self.stream, 0) };
        if begin_status != 0 {
            eprintln!(
                "CUDA PCG graph capture unavailable: status={begin_status}; using direct launches"
            );
            self.graphs_available = false;
            return Ok(None);
        }

        let capture_result = (|| {
            let mut rho = key.rho_a;
            let mut rho_next = key.rho_b;
            for _ in 0..key.iterations {
                launch_pcg_iteration(
                    self,
                    key.count,
                    grid,
                    vector_grid,
                    key.diagonal,
                    key.p,
                    key.neighbors,
                    key.blocks,
                    key.colors,
                    key.ap,
                    key.inverse,
                    key.dx,
                    key.r,
                    key.z,
                    key.alpha,
                    rho,
                    rho_next,
                )?;
                std::mem::swap(&mut rho, &mut rho_next);
            }
            Ok::<(), String>(())
        })();
        let mut graph = 0;
        let end_status = unsafe { end_capture(self.stream, &mut graph) };
        if let Err(error) = capture_result {
            self.graphs_available = false;
            if graph != 0 {
                let _ = unsafe { graph_destroy(graph) };
            }
            return Err(error);
        }
        if end_status != 0 {
            eprintln!("CUDA PCG graph capture failed to end: status={end_status}");
            self.graphs_available = false;
            return Err(format!(
                "cuStreamEndCapture failed with CUDA status {end_status}"
            ));
        }
        if graph == 0 {
            eprintln!("CUDA PCG graph capture returned an empty graph; using direct launches");
            self.graphs_available = false;
            return Err("cuStreamEndCapture returned an empty CUDA graph".into());
        }
        let mut executable = 0;
        let instantiate_status = unsafe { graph_instantiate(&mut executable, graph, 0) };
        let destroy_status = unsafe { graph_destroy(graph) };
        check(destroy_status, "cuGraphDestroy")?;
        if instantiate_status != 0 || executable == 0 {
            eprintln!("CUDA PCG graph instantiation unavailable: status={instantiate_status}; using direct launches");
            self.graphs_available = false;
            return Ok(None);
        }
        eprintln!(
            "CUDA PCG graph captured: active={} iterations={}",
            key.count, key.iterations
        );
        self.pcg_graph = Some((key, executable));
        Ok(Some(executable))
    }

    fn launch_graph(&self, executable: Handle) -> Result<(), String> {
        check(
            unsafe {
                (self
                    .driver
                    .graph_launch
                    .expect("graph support checked during CUDA initialization"))(
                    executable,
                    self.stream,
                )
            },
            "cuGraphLaunch",
        )
    }

    fn download(&self, output: *mut c_void, input: DevicePtr, bytes: usize) -> Result<(), String> {
        check(
            unsafe { (self.driver.stream_synchronize)(self.stream) },
            "cuStreamSynchronize",
        )?;
        check(
            unsafe { (self.driver.copy_dtoh)(output, input, bytes) },
            "cuMemcpyDtoH",
        )
    }

    fn launch_contact_operator(
        &self,
        mass: DevicePtr,
        row_free: DevicePtr,
        incidence: DevicePtr,
        edge_slots: DevicePtr,
        edge_free: DevicePtr,
        stiffness: DevicePtr,
        vector: DevicePtr,
        output: DevicePtr,
        count: u32,
    ) -> Result<(), String> {
        let kernel = self.contact_kernel.ok_or_else(|| "CUDA contact operator kernel is not loaded".to_string())?;
        let mut args = [mass, row_free, incidence, edge_slots, edge_free, stiffness, vector, output];
        let mut count_arg = count;
        let mut params: Vec<*mut c_void> = args.iter_mut()
            .map(|arg| (arg as *mut DevicePtr).cast())
            .collect();
        params.push((&mut count_arg as *mut u32).cast());
        check(unsafe {
            (self.driver.launch_kernel)(kernel, count.div_ceil(256), 1, 1, 256, 1, 1, 0,
                self.stream, params.as_mut_ptr(), std::ptr::null_mut())
        }, "cuLaunchKernel(contact_difference_operator)")
    }

    fn launch_dot(
        &mut self,
        a: DevicePtr,
        b: DevicePtr,
        count: u32,
        output: DevicePtr,
    ) -> Result<(), String> {
        if count > SINGLE_CTA_REDUCTION_MAX_REVS {
            let partial_count = count.div_ceil(256);
            let partials = self.buffer("dot_partials", partial_count as usize * 8)?;
            let mut a = a;
            let mut b = b;
            let mut partials_arg = partials;
            let mut count_arg = count;
            let mut partial_params = [
                (&mut a as *mut DevicePtr).cast(),
                (&mut b as *mut DevicePtr).cast(),
                (&mut partials_arg as *mut DevicePtr).cast(),
                (&mut count_arg as *mut u32).cast(),
            ];
            self.launch(4, partial_count, &mut partial_params)?;
            let mut output = output;
            let mut partial_count_arg = partial_count;
            let mut reduce_params = [
                (&mut partials_arg as *mut DevicePtr).cast(),
                (&mut output as *mut DevicePtr).cast(),
                (&mut partial_count_arg as *mut u32).cast(),
            ];
            return self.launch(5, 1, &mut reduce_params);
        }
        let mut a = a;
        let mut b = b;
        let mut output = output;
        let mut count = count;
        let mut params = [
            (&mut a as *mut DevicePtr).cast(),
            (&mut b as *mut DevicePtr).cast(),
            (&mut output as *mut DevicePtr).cast(),
            (&mut count as *mut u32).cast(),
        ];
        self.launch(9, 1, &mut params)
    }

    fn launch_dot_ratio(
        &mut self,
        a: DevicePtr,
        b: DevicePtr,
        count: u32,
        numerator: DevicePtr,
        output: DevicePtr,
    ) -> Result<(), String> {
        if count > SINGLE_CTA_REDUCTION_MAX_REVS {
            let partial_count = count.div_ceil(256);
            let partials = self.buffer("dot_partials", partial_count as usize * 8)?;
            let mut a = a;
            let mut b = b;
            let mut partials_arg = partials;
            let mut count_arg = count;
            let mut partial_params = [
                (&mut a as *mut DevicePtr).cast(),
                (&mut b as *mut DevicePtr).cast(),
                (&mut partials_arg as *mut DevicePtr).cast(),
                (&mut count_arg as *mut u32).cast(),
            ];
            self.launch(4, partial_count, &mut partial_params)?;
            let mut numerator = numerator;
            let mut output = output;
            let mut partial_count_arg = partial_count;
            let mut reduce_params = [
                (&mut partials_arg as *mut DevicePtr).cast(),
                (&mut numerator as *mut DevicePtr).cast(),
                (&mut output as *mut DevicePtr).cast(),
                (&mut partial_count_arg as *mut u32).cast(),
            ];
            return self.launch(6, 1, &mut reduce_params);
        }
        let mut a = a;
        let mut b = b;
        let mut numerator = numerator;
        let mut output = output;
        let mut count_arg = count;
        let mut params = [
            (&mut a as *mut DevicePtr).cast(),
            (&mut b as *mut DevicePtr).cast(),
            (&mut numerator as *mut DevicePtr).cast(),
            (&mut output as *mut DevicePtr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        self.launch(10, 1, &mut params)
    }

    /// Compute the fine sparse matvec and its PCG denominator partials in one
    /// pass. The final reduction still accumulates in f64 and produces alpha
    /// against the current rho.
    fn launch_matvec_dot_ratio(
        &mut self,
        diagonal: DevicePtr,
        x: DevicePtr,
        neighbors: DevicePtr,
        blocks: DevicePtr,
        applied: DevicePtr,
        count: u32,
        rho: DevicePtr,
        alpha: DevicePtr,
    ) -> Result<(), String> {
        let partial_count = count.div_ceil(256);
        let partials = self.buffer("dot_partials", partial_count as usize * 8)?;
        let mut diagonal = diagonal;
        let mut x = x;
        let mut neighbors = neighbors;
        let mut blocks = blocks;
        let mut applied = applied;
        let mut partials_arg = partials;
        let mut count_arg = count;
        let mut params = [
            (&mut diagonal as *mut DevicePtr).cast(),
            (&mut x as *mut DevicePtr).cast(),
            (&mut neighbors as *mut DevicePtr).cast(),
            (&mut blocks as *mut DevicePtr).cast(),
            (&mut applied as *mut DevicePtr).cast(),
            (&mut partials_arg as *mut DevicePtr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        self.launch(14, partial_count, &mut params)?;

        let mut partials_arg = partials;
        let mut rho = rho;
        let mut alpha = alpha;
        let mut partial_count_arg = partial_count;
        let mut reduce_params = [
            (&mut partials_arg as *mut DevicePtr).cast(),
            (&mut rho as *mut DevicePtr).cast(),
            (&mut alpha as *mut DevicePtr).cast(),
            (&mut partial_count_arg as *mut u32).cast(),
        ];
        self.launch(6, 1, &mut reduce_params)
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_pcg_iteration(
    cuda: &mut CudaSolver,
    count: u32,
    grid: u32,
    vector_grid: u32,
    diagonal: DevicePtr,
    p: DevicePtr,
    neighbors: DevicePtr,
    blocks: DevicePtr,
    colors: DevicePtr,
    ap: DevicePtr,
    inverse: DevicePtr,
    dx: DevicePtr,
    r: DevicePtr,
    z: DevicePtr,
    alpha: DevicePtr,
    rho: DevicePtr,
    rho_next: DevicePtr,
) -> Result<(), String> {
    let mut x_arg = p;
    let mut neighbors_arg = neighbors;
    let mut blocks_arg = blocks;
    let mut y_arg = ap;
    let mut diagonal_arg = diagonal;
    let mut count_arg = count;
    let mut matvec_params = [
        (&mut diagonal_arg as *mut DevicePtr).cast(),
        (&mut x_arg as *mut DevicePtr).cast(),
        (&mut neighbors_arg as *mut DevicePtr).cast(),
        (&mut blocks_arg as *mut DevicePtr).cast(),
        (&mut y_arg as *mut DevicePtr).cast(),
        (&mut count_arg as *mut u32).cast(),
    ];
    if fused_matvec_dot_enabled() && count > SINGLE_CTA_REDUCTION_MAX_REVS {
        cuda.launch_matvec_dot_ratio(diagonal, p, neighbors, blocks, ap, count, rho, alpha)?;
    } else {
        cuda.launch(0, grid, &mut matvec_params)?;
        cuda.launch_dot_ratio(p, ap, count, rho, alpha)?;
    }

    let mut inverse_arg = inverse;
    let mut dx_arg = dx;
    let mut residual_arg = r;
    let mut p_arg = p;
    let mut ap_arg = ap;
    let mut alpha_arg = alpha;
    let mut preconditioned_arg = z;
    let mut count_arg = count;
    let mut update_precondition_params = [
        (&mut inverse_arg as *mut DevicePtr).cast(),
        (&mut dx_arg as *mut DevicePtr).cast(),
        (&mut residual_arg as *mut DevicePtr).cast(),
        (&mut p_arg as *mut DevicePtr).cast(),
        (&mut ap_arg as *mut DevicePtr).cast(),
        (&mut alpha_arg as *mut DevicePtr).cast(),
        (&mut preconditioned_arg as *mut DevicePtr).cast(),
        (&mut count_arg as *mut u32).cast(),
    ];
    cuda.launch(8, grid, &mut update_precondition_params)?;
    if colors != 0 {
        launch_color_correction(
            cuda, count, grid, inverse, r, z, neighbors, blocks, colors, 1,
        )?;
        launch_color_correction(
            cuda, count, grid, inverse, r, z, neighbors, blocks, colors, 0,
        )?;
    }
    cuda.launch_dot(r, z, count, rho_next)?;

    let mut p_arg = p;
    let mut z_arg = z;
    let mut rho_next_arg = rho_next;
    let mut rho_arg = rho;
    let mut count_arg = count;
    let mut direction_params = [
        (&mut p_arg as *mut DevicePtr).cast(),
        (&mut z_arg as *mut DevicePtr).cast(),
        (&mut rho_next_arg as *mut DevicePtr).cast(),
        (&mut rho_arg as *mut DevicePtr).cast(),
        (&mut count_arg as *mut u32).cast(),
    ];
    cuda.launch(11, vector_grid, &mut direction_params)
}

#[allow(clippy::too_many_arguments)]
fn launch_color_correction(
    cuda: &mut CudaSolver,
    count: u32,
    grid: u32,
    inverse: DevicePtr,
    residual: DevicePtr,
    output: DevicePtr,
    neighbors: DevicePtr,
    blocks: DevicePtr,
    colors: DevicePtr,
    target_color: u32,
) -> Result<(), String> {
    let mut inverse_arg = inverse;
    let mut residual_arg = residual;
    let mut output_arg = output;
    let mut neighbors_arg = neighbors;
    let mut blocks_arg = blocks;
    let mut colors_arg = colors;
    let mut count_arg = count;
    let mut target_color_arg = target_color;
    let mut params = [
        (&mut inverse_arg as *mut DevicePtr).cast(),
        (&mut residual_arg as *mut DevicePtr).cast(),
        (&mut output_arg as *mut DevicePtr).cast(),
        (&mut neighbors_arg as *mut DevicePtr).cast(),
        (&mut blocks_arg as *mut DevicePtr).cast(),
        (&mut colors_arg as *mut DevicePtr).cast(),
        (&mut count_arg as *mut u32).cast(),
        (&mut target_color_arg as *mut u32).cast(),
    ];
    cuda.launch(12, grid, &mut params)
}

impl Drop for CudaSolver {
    fn drop(&mut self) {
        if let Some((_, executable)) = self.pcg_graph.take() {
            if let Some(graph_exec_destroy) = self.driver.graph_exec_destroy {
                let _ = unsafe { graph_exec_destroy(executable) };
            }
        }
        let _ = unsafe { (self.driver.stream_destroy)(self.stream) };
        for (_, (pointer, _)) in self.buffers.drain() {
            unsafe {
                (self.driver.mem_free)(pointer);
            }
        }
        // Context destruction reclaims the loaded module and all allocations.
        let _ = self.module;
        let _ = self.context;
    }
}

fn check(status: Status, operation: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed with CUDA status {status}"))
    }
}

fn residual_replacement_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_CUDA_RESIDUAL_REPLACEMENT")
            .map(|value| value == "1")
            .unwrap_or(true)
    })
}

fn fused_matvec_dot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_CUDA_FUSED_MATVEC_DOT")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

pub(crate) fn inverse_diagonal_on_gpu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOXEL_CUDA_INVERSE_DIAGONAL_ON_GPU")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

/// Solve the engine's assembled block sparse system with CUDA PCG and a
/// block-Jacobi preconditioner. Matrix and RHS layouts are supplied by the
/// engine; output is a correction vector in meters.
pub(crate) fn solve_pcg(
    rhs: &[Vec3],
    neighbors: &[[u32; 6]],
    diagonal: &[[f32; 9]],
    blocks: &[[f32; 54]],
    inverse_diagonal: &[[f32; 9]],
    max_iterations: usize,
    tolerance: f32,
    compact: Option<CompactEdgeBlocks<'_>>,
) -> Result<(Vec<Vec3>, usize, f32), String> {
    let n = rhs.len();
    if n == 0 {
        return Ok((Vec::new(), 0, 0.0));
    }
    if let Some(compact) = compact.as_ref() {
        if !blocks.is_empty() {
            return Err("CUDA compact edge blocks require an empty expanded block slice".into());
        }
        if compact.edge_ids.len() != n || compact.transpose_masks.len() != n {
            return Err(format!(
                "CUDA compact edge mapping has edge_ids={} masks={} for {n} active REVs",
                compact.edge_ids.len(),
                compact.transpose_masks.len(),
            ));
        }
        if compact.edges.len() > u32::MAX as usize {
            return Err("CUDA compact edge count exceeds u32::MAX".into());
        }
        for (row, (&edge_ids, &transpose_mask)) in compact
            .edge_ids
            .iter()
            .zip(compact.transpose_masks)
            .enumerate()
        {
            if transpose_mask & !0x3f != 0 {
                return Err(format!(
                    "CUDA compact edge row {row} has transpose bits outside six slots"
                ));
            }
            for (slot, edge_id) in edge_ids.into_iter().enumerate() {
                if edge_id == u32::MAX {
                    if transpose_mask & (1 << slot) != 0 {
                        return Err(format!(
                            "CUDA compact edge row {row} transposes missing slot {slot}"
                        ));
                    }
                } else if edge_id as usize >= compact.edges.len() {
                    return Err(format!(
                        "CUDA compact edge row {row} slot {slot} references edge {edge_id} of {}",
                        compact.edges.len()
                    ));
                }
            }
        }
    } else if blocks.len() != n {
        return Err(format!(
            "CUDA PCG has {} expanded block rows for {n} active REVs",
            blocks.len()
        ));
    }
    let inverse_on_gpu = inverse_diagonal_on_gpu_enabled();
    if inverse_diagonal.len() != n && !(inverse_on_gpu && inverse_diagonal.is_empty()) {
        return Err(format!(
            "CUDA PCG inverse diagonal has {} blocks for {n} active REVs",
            inverse_diagonal.len()
        ));
    }
    let count = u32::try_from(n)
        .map_err(|_| "CUDA PCG supports at most u32::MAX active REV".to_string())?;
    let initial_squared = rhs
        .iter()
        .flat_map(|v| [v.x, v.y, v.z])
        .map(|x| (x as f64) * (x as f64))
        .sum::<f64>();
    let initial = initial_squared.sqrt();
    if initial <= 1e-12 {
        return Ok((vec![Vec3::ZERO; n], 0, 0.0));
    }

    let store = CUDA.get_or_init(|| CudaSolver::initialize().map(Mutex::new));
    let mut cuda = match store {
        Ok(cuda) => cuda
            .lock()
            .map_err(|_| "CUDA solver lock poisoned".to_string())?,
        Err(error) => return Err(error.clone()),
    };
    check(
        unsafe { (cuda.driver.context_set_current)(cuda.context) },
        "cuCtxSetCurrent",
    )?;
    // PCG is called repeatedly for every physical substep and corotation
    // correction. Retain the packed host RHS so this transfer does not allocate
    // a fresh Vec on each solve.
    cuda.host_rhs.clear();
    cuda.host_rhs
        .extend(rhs.iter().flat_map(|v| [v.x, v.y, v.z]));
    let rhs_len = cuda.host_rhs.len();
    // The packed adjacency is a pure function of the current topology and is
    // shared by all corotational PCG solves until a bond breaks or a REV is
    // removed. Compare the complete packed table so topology changes always
    // refresh the device copy without relying on a lossy hash or an external
    // invalidation flag.
    let (neighbor_ptr, color_ptr) = if cuda.cached_neighbors == neighbors {
        let neighbor_ptr = cuda
            .buffers
            .get("neighbors")
            .map(|(pointer, _)| *pointer)
            .ok_or_else(|| "cached CUDA neighbors have no device buffer".to_string())?;
        let color_ptr = if cuda.cached_colors.is_some() {
            cuda.buffers
                .get("colors")
                .map(|(pointer, _)| *pointer)
                .ok_or_else(|| "cached CUDA colors have no device buffer".to_string())?
        } else {
            0
        };
        (neighbor_ptr, color_ptr)
    } else {
        let neighbor_ptr = cuda.upload("neighbors", neighbors)?;
        cuda.cached_neighbors.clear();
        cuda.cached_neighbors.extend_from_slice(neighbors);
        let color_ptr = if let Some(colors) = bipartite_colors(neighbors) {
            let color_ptr = cuda.upload("colors", &colors)?;
            eprintln!(
                "CUDA PCG preconditioner: symmetric red-black, active={}",
                colors.len()
            );
            cuda.cached_colors = Some(colors);
            color_ptr
        } else {
            eprintln!("CUDA PCG preconditioner: graph is not bipartite; using block-Jacobi");
            cuda.cached_colors = None;
            0
        };
        (neighbor_ptr, color_ptr)
    };
    let diagonal_ptr = cuda.upload("diagonal", diagonal)?;
    let block_ptr = if let Some(compact) = compact {
        let edge_blocks_ptr = cuda.upload("canonical_edge_blocks", compact.edges)?;
        let (edge_ids_ptr, transpose_masks_ptr) =
            cuda.upload_compact_edge_metadata(compact.edge_ids, compact.transpose_masks)?;
        let expanded_blocks =
            cuda.buffer("blocks", n * std::mem::size_of::<[f32; 54]>())?;
        cuda.launch_expand_edge_blocks(
            edge_blocks_ptr,
            edge_ids_ptr,
            transpose_masks_ptr,
            expanded_blocks,
            count,
        )?;
        expanded_blocks
    } else {
        cuda.upload("blocks", blocks)?
    };
    let inverse_ptr = if inverse_on_gpu {
        let inverse_ptr = cuda.buffer("inverse", n * std::mem::size_of::<[f32; 9]>())?;
        cuda.launch_inverse_diagonal(diagonal_ptr, inverse_ptr, count)?;
        inverse_ptr
    } else {
        cuda.upload("inverse", inverse_diagonal)?
    };
    let residual_replacement = residual_replacement_enabled();
    let rhs_exact = if residual_replacement {
        let pointer = cuda.buffer("rhs_exact", rhs_len * 4)?;
        check(
            unsafe { (cuda.driver.copy_htod)(pointer, cuda.host_rhs.as_ptr().cast(), rhs_len * 4) },
            "upload exact PCG RHS",
        )?;
        Some(pointer)
    } else {
        None
    };
    let dx = cuda.buffer("dx", rhs_len * 4)?;
    let r = cuda.buffer("r", rhs_len * 4)?;
    let z = cuda.buffer("z", rhs_len * 4)?;
    let p = cuda.buffer("p", rhs_len * 4)?;
    let ap = cuda.buffer("ap", rhs_len * 4)?;
    let rho_a = cuda.buffer("rho_a", 8)?;
    let rho_b = cuda.buffer("rho_b", 8)?;
    let residual_squared = cuda.buffer("residual_squared", 8)?;
    let alpha = cuda.buffer("alpha", 4)?;
    let beta_zero = cuda.buffer("beta_zero", 4)?;
    check(
        unsafe { (cuda.driver.copy_htod)(beta_zero, (&0.0f32 as *const f32).cast(), 4) },
        "initialize zero beta",
    )?;
    check(
        unsafe { (cuda.driver.memset_d8)(dx, 0, rhs_len * 4) },
        "initialize dx",
    )?;
    check(
        unsafe { (cuda.driver.copy_htod)(r, cuda.host_rhs.as_ptr().cast(), rhs_len * 4) },
        "initialize residual",
    )?;

    let grid = count.div_ceil(256);
    // The vector update kernels index the packed xyz components, so they need
    // three work items per REV. Matrix and block-Jacobi kernels remain one
    // work item per REV.
    let vector_grid = count.saturating_mul(3).div_ceil(256);
    let mut inverse = inverse_ptr;
    let mut residual = r;
    let mut preconditioned = z;
    let mut n_arg = count;
    let mut params = [
        (&mut inverse as *mut DevicePtr).cast(),
        (&mut residual as *mut DevicePtr).cast(),
        (&mut preconditioned as *mut DevicePtr).cast(),
        (&mut n_arg as *mut u32).cast(),
    ];
    cuda.launch(1, grid, &mut params)?;
    if color_ptr != 0 {
        launch_color_correction(
            &mut cuda,
            count,
            grid,
            inverse_ptr,
            r,
            z,
            neighbor_ptr,
            block_ptr,
            color_ptr,
            1,
        )?;
        launch_color_correction(
            &mut cuda,
            count,
            grid,
            inverse_ptr,
            r,
            z,
            neighbor_ptr,
            block_ptr,
            color_ptr,
            0,
        )?;
    }
    cuda.launch_dot(r, z, count, rho_a)?;
    let mut p_arg = p;
    let mut z_arg = z;
    let mut beta_arg = beta_zero;
    let mut count_arg = count;
    let mut init_params = [
        (&mut p_arg as *mut DevicePtr).cast(),
        (&mut z_arg as *mut DevicePtr).cast(),
        (&mut beta_arg as *mut DevicePtr).cast(),
        (&mut count_arg as *mut u32).cast(),
    ];
    cuda.launch(3, vector_grid, &mut init_params)?;
    // Vectors and recurrence scalars stay on the device. Check the residual in
    // coarse batches so converged systems do not spend the remaining iteration
    // budget in no-op PCG work; the host transfers one scalar per checkpoint.
    let mut rho = rho_a;
    let mut rho_next = rho_b;
    let mut iterations = 0;
    let mut residual_squared_host = f64::INFINITY;
    while iterations < max_iterations {
        let batch_iterations = (max_iterations - iterations).min(48);
        let graph_key = PcgGraphKey {
            count,
            iterations: batch_iterations as u32,
            diagonal: diagonal_ptr,
            p,
            neighbors: neighbor_ptr,
            blocks: block_ptr,
            colors: color_ptr,
            ap,
            inverse: inverse_ptr,
            dx,
            r,
            z,
            alpha,
            rho_a: rho,
            rho_b: rho_next,
        };
        if let Some(executable) = cuda.pcg_graph_exec(graph_key, grid, vector_grid)? {
            cuda.launch_graph(executable)?;
            if batch_iterations % 2 == 1 {
                std::mem::swap(&mut rho, &mut rho_next);
            }
        } else {
            for _ in 0..batch_iterations {
                launch_pcg_iteration(
                    &mut cuda,
                    count,
                    grid,
                    vector_grid,
                    diagonal_ptr,
                    p,
                    neighbor_ptr,
                    block_ptr,
                    color_ptr,
                    ap,
                    inverse_ptr,
                    dx,
                    r,
                    z,
                    alpha,
                    rho,
                    rho_next,
                )?;
                std::mem::swap(&mut rho, &mut rho_next);
            }
        }
        iterations += batch_iterations;
        if let Some(rhs_exact) = rhs_exact {
            // Rebuild r=b-Ax from the current solution, then restart the Krylov
            // direction. This controls the finite-precision residual gap that
            // can otherwise make the recursive GPU residual overstate accuracy.
            let mut diagonal_arg = diagonal_ptr;
            let mut x_arg = dx;
            let mut neighbors_arg = neighbor_ptr;
            let mut blocks_arg = block_ptr;
            let mut ap_arg = ap;
            let mut count_arg = count;
            let mut matvec_params = [
                (&mut diagonal_arg as *mut DevicePtr).cast(),
                (&mut x_arg as *mut DevicePtr).cast(),
                (&mut neighbors_arg as *mut DevicePtr).cast(),
                (&mut blocks_arg as *mut DevicePtr).cast(),
                (&mut ap_arg as *mut DevicePtr).cast(),
                (&mut count_arg as *mut u32).cast(),
            ];
            cuda.launch(0, grid, &mut matvec_params)?;

            let mut rhs_arg = rhs_exact;
            let mut applied_arg = ap;
            let mut residual_arg = r;
            let mut count_arg = count;
            let mut residual_params = [
                (&mut rhs_arg as *mut DevicePtr).cast(),
                (&mut applied_arg as *mut DevicePtr).cast(),
                (&mut residual_arg as *mut DevicePtr).cast(),
                (&mut count_arg as *mut u32).cast(),
            ];
            cuda.launch(13, vector_grid, &mut residual_params)?;

            let mut inverse_arg = inverse_ptr;
            let mut residual_arg = r;
            let mut preconditioned_arg = z;
            let mut count_arg = count;
            let mut precondition_params = [
                (&mut inverse_arg as *mut DevicePtr).cast(),
                (&mut residual_arg as *mut DevicePtr).cast(),
                (&mut preconditioned_arg as *mut DevicePtr).cast(),
                (&mut count_arg as *mut u32).cast(),
            ];
            cuda.launch(1, grid, &mut precondition_params)?;
            if color_ptr != 0 {
                launch_color_correction(
                    &mut cuda,
                    count,
                    grid,
                    inverse_ptr,
                    r,
                    z,
                    neighbor_ptr,
                    block_ptr,
                    color_ptr,
                    1,
                )?;
                launch_color_correction(
                    &mut cuda,
                    count,
                    grid,
                    inverse_ptr,
                    r,
                    z,
                    neighbor_ptr,
                    block_ptr,
                    color_ptr,
                    0,
                )?;
            }
            cuda.launch_dot(r, z, count, rho)?;

            let mut p_arg = p;
            let mut z_arg = z;
            let mut beta_arg = beta_zero;
            let mut count_arg = count;
            let mut restart_params = [
                (&mut p_arg as *mut DevicePtr).cast(),
                (&mut z_arg as *mut DevicePtr).cast(),
                (&mut beta_arg as *mut DevicePtr).cast(),
                (&mut count_arg as *mut u32).cast(),
            ];
            cuda.launch(3, vector_grid, &mut restart_params)?;
        }
        if iterations % 16 == 0 || iterations == max_iterations {
            cuda.launch_dot(r, r, count, residual_squared)?;
            cuda.download(
                (&mut residual_squared_host as *mut f64).cast(),
                residual_squared,
                8,
            )?;
            if !residual_squared_host.is_finite() || residual_squared_host < 0.0 {
                return Err("CUDA PCG produced a non-finite residual norm".into());
            }
            let relative = (residual_squared_host.sqrt() / initial) as f32;
            if relative <= tolerance {
                break;
            }
        }
    }
    let mut output = vec![0.0f32; rhs_len];
    cuda.download(output.as_mut_ptr().cast(), dx, output.len() * 4)?;
    if !residual_squared_host.is_finite() || residual_squared_host < 0.0 {
        return Err("CUDA PCG produced a non-finite residual norm".into());
    }
    let relative = (residual_squared_host.sqrt() / initial) as f32;
    if !relative.is_finite() || relative > tolerance {
        return Err(format!(
            "CUDA PCG did not converge: iterations={iterations}, relative residual={relative:.3e}, tolerance={tolerance:.3e}"
        ));
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("CUDA PCG produced a non-finite correction".into());
    }
    let correction = output
        .chunks_exact(3)
        .map(|v| Vec3::new(v[0], v[1], v[2]))
        .collect();
    Ok((correction, iterations, relative))
}

#[cfg(test)]
mod tests {
    use super::{ContactIc0Input, ContactOperatorEdge, ContactOperatorInput, ContactOperatorSession, CudaSolver};
    use glam::{Mat3, Vec3};

    #[test]
    fn cuda_contact_difference_operator_matches_f64_cpu_reference() {
        if std::env::var_os("VOXEL_REQUIRE_CUDA_TEST").is_none() {
            return;
        }
        let free = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let edges = [
            ContactOperatorEdge {
                slots: [0, 1], free: [free[0], free[1]],
                stiffness: [3.0, 0.5, -0.25, 0.5, 2.0, 0.125, -0.25, 0.125, 1.0],
            },
            ContactOperatorEdge {
                slots: [1, 2], free: [free[1], free[2]],
                stiffness: [1.5, 0.25, 0.0, 0.25, 4.0, 0.5, 0.0, 0.5, 2.5],
            },
        ];
        let incidence = [[0, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            [0, 1, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            [1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]];
        let masses = [2.0e6, 3.0e6, 4.0e6];
        let vectors = [Vec3::new(0.25, -0.5, 0.75), Vec3::new(-0.125, 0.4, 0.2), Vec3::new(0.8, -0.2, 0.3)];
        let no_edges: [ContactOperatorEdge; 0] = [];
        let no_incidence = [[u32::MAX; 6]; 3];
        let mut inertial = ContactOperatorSession::new(ContactOperatorInput {
            masses_over_h2: &masses, free: &free, incidence: &no_incidence, edges: &no_edges,
        }).expect("CUDA contact inertial-only session");
        let mut inertial_result = vec![Vec3::ZERO; 3];
        inertial.apply(&vectors, &mut inertial_result).expect("CUDA inertial-only matvec");
        for row in 0..3 {
            let expected = ((vectors[row] * Vec3::from_array(free[row])).as_dvec3()
                * masses[row]).as_vec3();
            assert_eq!(inertial_result[row], expected);
        }
        drop(inertial);
        let mut session = ContactOperatorSession::new(ContactOperatorInput {
            masses_over_h2: &masses, free: &free, incidence: &incidence, edges: &edges,
        }).expect("CUDA contact operator session");
        let mut actual = vec![Vec3::ZERO; 3];
        session.apply(&vectors, &mut actual).expect("CUDA contact matvec");
        let mut expected = vec![glam::DVec3::ZERO; 3];
        for row in 0..3 {
            let masked = vectors[row] * Vec3::from_array(free[row]);
            expected[row] = masked.as_dvec3() * masses[row];
            for &edge_id in incidence[row].iter().take_while(|&&id| id != u32::MAX) {
                let edge = edges[edge_id as usize];
                let vi = if edge.slots[0] == u32::MAX { glam::DVec3::ZERO } else {
                    (vectors[edge.slots[0] as usize] * Vec3::from_array(edge.free[0])).as_dvec3()
                };
                let vj = if edge.slots[1] == u32::MAX { glam::DVec3::ZERO } else {
                    (vectors[edge.slots[1] as usize] * Vec3::from_array(edge.free[1])).as_dvec3()
                };
                let force = Mat3::from_cols_array(&edge.stiffness).as_dmat3() * (vi - vj);
                if edge.slots[0] == row as u32 {
                    expected[row] += force * glam::DVec3::from_array(free[row].map(f64::from));
                } else {
                    expected[row] -= force * glam::DVec3::from_array(free[row].map(f64::from));
                }
            }
        }
        for (a, e) in actual.iter().zip(expected) {
            let expected = e.as_vec3();
            assert!((*a - expected).abs().max_element() <= 2.0e-6 * expected.abs().max_element().max(1.0),
                "actual={a:?}, expected={expected:?}");
        }
        drop(session);

        // Subtraction must happen after promotion. A f32 subtraction would
        // erase the small endpoint and lose the force of this rank-one bond.
        let free = [[1.0; 3]; 2];
        let stiffness = [1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let edges = [ContactOperatorEdge { slots: [0, 1], free, stiffness }];
        let incidence = [[0, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]; 2];
        let mut session = ContactOperatorSession::new(ContactOperatorInput {
            masses_over_h2: &[0.0; 2], free: &free, incidence: &incidence, edges: &edges,
        }).unwrap();
        let vectors = [Vec3::new(1.0e9, 1.0e9, 0.0), Vec3::new(0.125, 0.25, 0.0)];
        let force = (Mat3::from_cols_array(&stiffness).as_dmat3()
            * (vectors[0].as_dvec3() - vectors[1].as_dvec3())).as_vec3();
        let mut actual = [Vec3::ZERO; 2];
        session.apply(&vectors, &mut actual).unwrap();
        assert_eq!(force, Vec3::new(0.125, -0.125, 0.0));
        assert_eq!(actual, [force, -force]);
        drop(session);

        // An absent endpoint must never be loaded. A fractional mask also
        // distinguishes masking each force from masking the running sum.
        let free = [[0.5, 0.0, 1.0]];
        let edges = [ContactOperatorEdge {
            slots: [0, u32::MAX], free: [free[0], [0.0; 3]], stiffness,
        }];
        let incidence = [[0, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]];
        let mut session = ContactOperatorSession::new(ContactOperatorInput {
            masses_over_h2: &[2.0], free: &free, incidence: &incidence, edges: &edges,
        }).unwrap();
        let vector = Vec3::new(2.0e5, -3.0e5, 4.0e5);
        let masked = (vector * Vec3::from_array(free[0])).as_dvec3();
        let expected = (masked * 2.0 + Mat3::from_cols_array(&stiffness).as_dmat3()
            * masked * glam::DVec3::new(0.5, 0.0, 1.0)).as_vec3();
        let mut actual = [Vec3::ZERO];
        session.apply(&[vector], &mut actual).unwrap();
        assert_eq!(actual[0], expected);
    }

    #[test]
    fn cuda_contact_ic0_matches_ordered_cpu_apply_and_rejects_bad_topology() {
        if std::env::var_os("VOXEL_REQUIRE_CUDA_TEST").is_none() { return; }
        let matrices = [
            Mat3::from_cols(Vec3::new(0.7, 0.1, -0.05), Vec3::new(0.2, 0.9, 0.03), Vec3::new(-0.1, 0.04, 0.8)),
            Mat3::from_cols(Vec3::new(0.8, -0.04, 0.07), Vec3::new(0.12, 0.75, 0.02), Vec3::new(0.03, -0.08, 0.95)),
            Mat3::from_cols(Vec3::new(0.6, 0.02, 0.11), Vec3::new(-0.07, 0.85, 0.06), Vec3::new(0.09, 0.01, 0.72)),
            Mat3::from_cols(Vec3::new(0.91, 0.05, 0.02), Vec3::new(-0.02, 0.68, 0.08), Vec3::new(0.04, 0.03, 0.77)),
            Mat3::from_cols(Vec3::new(0.73, -0.03, 0.06), Vec3::new(0.09, 0.88, -0.02), Vec3::new(-0.05, 0.07, 0.82)),
            Mat3::from_cols(Vec3::new(0.82, 0.02, -0.04), Vec3::new(0.06, 0.79, 0.03), Vec3::new(0.01, -0.05, 0.9)),
        ];
        let inverse: Vec<[f32; 9]> = matrices.iter().map(Mat3::to_cols_array).collect();
        let factors = [
            Mat3::from_cols(Vec3::new(0.11, -0.03, 0.02), Vec3::new(0.04, 0.08, -0.01), Vec3::new(-0.02, 0.05, 0.07)).to_cols_array(),
            Mat3::from_cols(Vec3::new(-0.06, 0.02, 0.04), Vec3::new(0.03, 0.09, 0.01), Vec3::new(0.07, -0.02, 0.05)).to_cols_array(),
            Mat3::from_cols(Vec3::new(0.05, 0.01, -0.03), Vec3::new(-0.04, 0.06, 0.02), Vec3::new(0.03, 0.08, 0.04)).to_cols_array(),
        ];
        let mut lower = [[[u32::MAX; 2]; 6]; 6];
        let mut lower_counts = [0u8; 6];
        let mut upper = [[[u32::MAX; 2]; 6]; 6];
        let mut upper_counts = [0u8; 6];
        // A chain, an isolated row, then a disconnected two-row component.
        lower[1][0] = [0, 0]; lower_counts[1] = 1;
        lower[2][0] = [1, 1]; lower_counts[2] = 1;
        lower[5][0] = [2, 4]; lower_counts[5] = 1;
        upper[0][0] = [0, 1]; upper_counts[0] = 1;
        upper[1][0] = [1, 2]; upper_counts[1] = 1;
        upper[4][0] = [2, 5]; upper_counts[4] = 1;
        let rhs = [
            Vec3::new(0.75, -0.5, 0.125), Vec3::new(-0.25, 0.875, 0.375),
            Vec3::new(0.625, 0.25, -0.75), Vec3::new(-0.125, 0.5, 0.875),
            Vec3::new(0.375, -0.625, 0.25), Vec3::new(0.5, 0.125, -0.375),
        ];
        let mut session = ContactOperatorSession::new(ContactOperatorInput {
            masses_over_h2: &[1.0; 6], free: &[[1.0; 3]; 6],
            incidence: &[[u32::MAX; 6]; 6], edges: &[],
        }).expect("forced CUDA contact session");
        session.prepare_ic0(ContactIc0Input {
            inverse_diagonal: &inverse, edge_factors: &factors,
            lower_links: &lower, lower_counts: &lower_counts,
            upper_links: &upper, upper_counts: &upper_counts,
        }).expect("prepare CUDA IC0");
        assert_eq!(session.ic0_level_counts(), Some([3, 3]));
        let mut work = [Vec3::ZERO; 6];
        let mut expected = [Vec3::ZERO; 6];
        for row in 0..6 {
            let mut value = rhs[row];
            for &[edge, neighbor] in &lower[row][..lower_counts[row] as usize] {
                value -= Mat3::from_cols_array(&factors[edge as usize]) * work[neighbor as usize];
            }
            work[row] = Mat3::from_cols_array(&inverse[row]) * value;
        }
        for row in (0..6).rev() {
            let mut value = work[row];
            for &[edge, neighbor] in &upper[row][..upper_counts[row] as usize] {
                value -= Mat3::from_cols_array(&factors[edge as usize]).transpose() * expected[neighbor as usize];
            }
            expected[row] = Mat3::from_cols_array(&inverse[row]).transpose() * value;
        }
        let mut actual = [Vec3::ZERO; 6];
        session.apply_ic0(&rhs, &mut actual).expect("CUDA IC0 apply");
        assert_eq!(actual, expected, "CUDA IC0 must preserve glam f32 arithmetic");

        let mut cyclic = lower;
        cyclic[0][0] = [0, 1];
        lower_counts[0] = 1;
        assert!(session.prepare_ic0(ContactIc0Input {
            inverse_diagonal: &inverse, edge_factors: &factors,
            lower_links: &cyclic, lower_counts: &lower_counts,
            upper_links: &upper, upper_counts: &upper_counts,
        }).unwrap_err().contains("cycle"));
        assert!(session.prepare_ic0(ContactIc0Input {
            inverse_diagonal: &inverse, edge_factors: &factors,
            lower_links: &lower, lower_counts: &[7, 0, 0, 0, 0, 0],
            upper_links: &upper, upper_counts: &upper_counts,
        }).is_err());
        assert!(session.prepare_ic0(ContactIc0Input {
            inverse_diagonal: &inverse, edge_factors: &factors,
            lower_links: &lower, lower_counts: &[1, 1, 1, 0, 0, 1],
            upper_links: &upper, upper_counts: &upper_counts,
        }).is_err());
        assert!(session.apply_ic0(&rhs, &mut actual).unwrap_err().contains("not been prepared"));
        let mut nonfinite = inverse.clone();
        nonfinite[0][0] = f32::NAN;
        assert!(session.prepare_ic0(ContactIc0Input {
            inverse_diagonal: &nonfinite, edge_factors: &factors,
            lower_links: &lower, lower_counts: &[0; 6],
            upper_links: &upper, upper_counts: &upper_counts,
        }).unwrap_err().contains("non-finite"));
        assert!(session.prepare_ic0(ContactIc0Input {
            inverse_diagonal: &inverse, edge_factors: &factors,
            lower_links: &lower[..5], lower_counts: &lower_counts,
            upper_links: &upper, upper_counts: &upper_counts,
        }).unwrap_err().contains("inconsistent"));
    }

    #[test]
    fn cuda_adaptive_dot_and_ratio_match_cpu_when_explicitly_requested() {
        if std::env::var_os("VOXEL_REQUIRE_CUDA_TEST").is_none() {
            return;
        }
        let mut cuda = CudaSolver::initialize().expect("forced CUDA PTX module initialization");
        super::check(
            unsafe { (cuda.driver.context_set_current)(cuda.context) },
            "cuCtxSetCurrent(test)",
        )
        .unwrap();
        for count in [600u32, 8_192, 8_193, 16_385, 65_537, 400_000] {
            let left: Vec<f32> = (0..count * 3)
                .map(|i| (i % 97 + 1) as f32 * 0.03125)
                .collect();
            let right: Vec<f32> = (0..count * 3)
                .map(|i| (i % 53 + 1) as f32 * 0.015625)
                .collect();
            let expected: f64 = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let left_ptr = cuda.upload("test_left", &left).unwrap();
            let right_ptr = cuda.upload("test_right", &right).unwrap();
            let dot_ptr = cuda.buffer("test_dot", 8).unwrap();
            cuda.launch_dot(left_ptr, right_ptr, count, dot_ptr)
                .unwrap();
            let mut dot = f64::NAN;
            cuda.download((&mut dot as *mut f64).cast(), dot_ptr, 8)
                .unwrap();
            assert!(
                (dot - expected).abs() <= expected.abs().max(1.0) * 1e-12,
                "CUDA dot mismatch for {count} REV: actual={dot:.17e}, expected={expected:.17e}, error={:.3e}",
                (dot - expected).abs(),
            );

            let rho = cuda.upload("test_rho", &[expected * 0.75]).unwrap();
            let alpha_ptr = cuda.buffer("test_alpha", 4).unwrap();
            cuda.launch_dot_ratio(left_ptr, right_ptr, count, rho, alpha_ptr)
                .unwrap();
            let mut alpha = f32::NAN;
            cuda.download((&mut alpha as *mut f32).cast(), alpha_ptr, 4)
                .unwrap();
            assert!(
                (alpha - 0.75).abs() <= 1e-6,
                "CUDA dot ratio mismatch for {count} REV: actual={alpha:.9e}, expected=7.5e-1",
            );
        }
    }

    #[test]
    fn cuda_inverse_diagonal_matches_glam_and_marks_invalid_blocks() {
        if std::env::var_os("VOXEL_REQUIRE_CUDA_TEST").is_none() {
            return;
        }
        let mut cuda = CudaSolver::initialize_with_inverse_diagonal(true)
            .expect("forced CUDA inverse-diagonal PTX module initialization");
        super::check(
            unsafe { (cuda.driver.context_set_current)(cuda.context) },
            "cuCtxSetCurrent(test)",
        )
        .unwrap();

        let valid = Mat3::from_cols(
            Vec3::new(4.0, 1.0, 0.25),
            Vec3::new(0.5, 3.0, 0.75),
            Vec3::new(0.2, 0.4, 2.0),
        );
        let singular = Mat3::from_cols(Vec3::X, Vec3::X * 2.0, Vec3::Z);
        let nonfinite = Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::new(0.0, 0.0, f32::INFINITY));
        let row_packed = |matrix: Mat3| {
            let [x, y, z] = matrix.to_cols_array_2d();
            [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
        };
        let diagonal = [
            row_packed(valid),
            row_packed(singular),
            row_packed(nonfinite),
        ];
        let diagonal_ptr = cuda.upload("test_inverse_diagonal", &diagonal).unwrap();
        let inverse_ptr = cuda
            .buffer(
                "test_inverse_diagonal_output",
                std::mem::size_of_val(&diagonal),
            )
            .unwrap();
        let count_arg = diagonal.len() as u32;
        cuda.launch_inverse_diagonal(diagonal_ptr, inverse_ptr, count_arg)
            .unwrap();

        let mut actual = [[0.0f32; 9]; 3];
        cuda.download(
            actual.as_mut_ptr().cast(),
            inverse_ptr,
            std::mem::size_of_val(&actual),
        )
        .unwrap();
        let expected = row_packed(valid.inverse());
        for (actual, expected) in actual[0].iter().zip(expected) {
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 2e-6);
        }
        assert!(actual[1].iter().all(|value| value.is_nan()));
        assert!(actual[2].iter().all(|value| value.is_nan()));
    }

    #[test]
    fn cuda_expands_compact_edge_blocks_bitwise() {
        if std::env::var_os("VOXEL_REQUIRE_CUDA_TEST").is_none() {
            return;
        }
        let mut cuda = CudaSolver::initialize_with_kernels(false, true)
            .expect("forced CUDA edge-block expansion PTX initialization");
        super::check(
            unsafe { (cuda.driver.context_set_current)(cuda.context) },
            "cuCtxSetCurrent(test)",
        )
        .unwrap();

        let edge_blocks = [
            [
                1.0,
                -0.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                f32::from_bits(0x7fc0_1234),
            ],
            [-1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0, 9.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ];
        let edge_ids = [
            [0, 1, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            [2, 0, 1, u32::MAX, u32::MAX, u32::MAX],
        ];
        let transpose_masks = [0b000010, 0b000101];
        let edge_blocks_ptr = cuda
            .upload("test_canonical_edge_blocks", &edge_blocks)
            .unwrap();
        let (edge_ids_ptr, transpose_masks_ptr) = cuda
            .upload_compact_edge_metadata(&edge_ids, &transpose_masks)
            .unwrap();
        let cached_metadata = cuda
            .upload_compact_edge_metadata(&edge_ids, &transpose_masks)
            .unwrap();
        assert_eq!(cached_metadata, (edge_ids_ptr, transpose_masks_ptr));

        let output_ptr = cuda
            .buffer("test_expanded_edge_blocks", 2 * std::mem::size_of::<[f32; 54]>())
            .unwrap();
        cuda.launch_expand_edge_blocks(
            edge_blocks_ptr,
            edge_ids_ptr,
            transpose_masks_ptr,
            output_ptr,
            2,
        )
        .unwrap();
        let mut actual = [[0.0f32; 54]; 2];
        cuda.download(
            actual.as_mut_ptr().cast(),
            output_ptr,
            std::mem::size_of_val(&actual),
        )
        .unwrap();

        let mut expected = [[0.0f32; 54]; 2];
        for row in 0..edge_ids.len() {
            for slot in 0..6 {
                let edge_id = edge_ids[row][slot];
                if edge_id == u32::MAX {
                    continue;
                }
                for element in 0..9 {
                    let source_element = if transpose_masks[row] & (1 << slot) != 0 {
                        (element % 3) * 3 + element / 3
                    } else {
                        element
                    };
                    expected[row][slot * 9 + element] =
                        edge_blocks[edge_id as usize][source_element];
                }
            }
        }
        let actual_bytes: &[u8] = bytemuck::cast_slice(actual.as_flattened());
        let expected_bytes: &[u8] = bytemuck::cast_slice(expected.as_flattened());
        assert_eq!(actual_bytes, expected_bytes);
    }

}
