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
type ModuleGetFunction = unsafe extern "C" fn(*mut Handle, Handle, *const i8) -> Status;
type MemAlloc = unsafe extern "C" fn(*mut DevicePtr, usize) -> Status;
type MemFree = unsafe extern "C" fn(DevicePtr) -> Status;
type CopyHtoD = unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> Status;
type CopyDtoH = unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> Status;
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
type ContextSynchronize = unsafe extern "C" fn() -> Status;

struct Driver {
    _library: Library,
    init: Init,
    device_get: DeviceGet,
    context_create: ContextCreate,
    context_set_current: ContextSetCurrent,
    module_load_data: ModuleLoadData,
    module_get_function: ModuleGetFunction,
    mem_alloc: MemAlloc,
    mem_free: MemFree,
    copy_htod: CopyHtoD,
    copy_dtoh: CopyDtoH,
    launch_kernel: LaunchKernel,
    context_synchronize: ContextSynchronize,
}

struct CudaSolver {
    driver: Driver,
    context: Handle,
    module: Handle,
    kernels: [Handle; 5],
    buffers: HashMap<&'static str, (DevicePtr, usize)>,
    dot_partials: Vec<f64>,
}

static CUDA: OnceLock<Result<Mutex<CudaSolver>, String>> = OnceLock::new();

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    // CUDA's public C ABI uses stable function pointers for these symbols.
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol| *symbol)
            .map_err(|e| e.to_string())
    }
}

impl CudaSolver {
    fn initialize() -> Result<Self, String> {
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
                module_get_function: symbol(&library, b"cuModuleGetFunction\0")?,
                mem_alloc: symbol(&library, b"cuMemAlloc_v2\0")?,
                mem_free: symbol(&library, b"cuMemFree_v2\0")?,
                copy_htod: symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                copy_dtoh: symbol(&library, b"cuMemcpyDtoH_v2\0")?,
                launch_kernel: symbol(&library, b"cuLaunchKernel\0")?,
                context_synchronize: symbol(&library, b"cuCtxSynchronize\0")?,
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
            let mut kernels = [0; 5];
            for (slot, name) in [
                b"block_sparse_matvec\0".as_slice(),
                b"block_jacobi\0".as_slice(),
                b"pcg_update\0".as_slice(),
                b"pcg_update_direction\0".as_slice(),
                b"dot_product\0".as_slice(),
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
            Ok(Self {
                driver,
                context,
                module,
                kernels,
                buffers: HashMap::new(),
                dot_partials: Vec::new(),
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
                    0,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    fn dot(&mut self, a: DevicePtr, b: DevicePtr, count: u32) -> Result<f64, String> {
        let result = self.buffer("dot_partial", count as usize * 8)?;
        let mut a = a;
        let mut b = b;
        let mut result = result;
        let mut count = count;
        let mut params = [
            (&mut a as *mut DevicePtr).cast(),
            (&mut b as *mut DevicePtr).cast(),
            (&mut result as *mut DevicePtr).cast(),
            (&mut count as *mut u32).cast(),
        ];
        self.launch(4, count.div_ceil(256), &mut params)?;
        self.dot_partials.resize(count as usize, 0.0);
        check(
            unsafe {
                (self.driver.copy_dtoh)(
                    self.dot_partials.as_mut_ptr().cast(),
                    result,
                    self.dot_partials.len() * 8,
                )
            },
            "cuMemcpyDtoH(dot partials)",
        )?;
        Ok(self.dot_partials.iter().sum())
    }
}

impl Drop for CudaSolver {
    fn drop(&mut self) {
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
) -> Result<(Vec<Vec3>, usize, f32), String> {
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
    let n = rhs.len();
    if n == 0 {
        return Ok((Vec::new(), 0, 0.0));
    }
    let count = u32::try_from(n)
        .map_err(|_| "CUDA PCG supports at most u32::MAX active REV".to_string())?;
    let flat_rhs: Vec<f32> = rhs.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
    let neighbor_ptr = cuda.upload("neighbors", neighbors)?;
    let diagonal_ptr = cuda.upload("diagonal", diagonal)?;
    let block_ptr = cuda.upload("blocks", blocks)?;
    let inverse_ptr = cuda.upload("inverse", inverse_diagonal)?;
    let dx = cuda.buffer("dx", flat_rhs.len() * 4)?;
    let r = cuda.buffer("r", flat_rhs.len() * 4)?;
    let z = cuda.buffer("z", flat_rhs.len() * 4)?;
    let p = cuda.buffer("p", flat_rhs.len() * 4)?;
    let ap = cuda.buffer("ap", flat_rhs.len() * 4)?;
    let zeroes = vec![0.0f32; flat_rhs.len()];
    check(
        unsafe { (cuda.driver.copy_htod)(dx, zeroes.as_ptr().cast(), zeroes.len() * 4) },
        "initialize dx",
    )?;
    check(
        unsafe { (cuda.driver.copy_htod)(r, flat_rhs.as_ptr().cast(), flat_rhs.len() * 4) },
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
    check(
        unsafe { (cuda.driver.copy_htod)(p, zeroes.as_ptr().cast(), zeroes.len() * 4) },
        "initialize direction",
    )?;
    let zero = 0.0f32;
    let mut beta_arg = zero;
    let mut p_arg = p;
    let mut z_arg = z;
    let mut count_arg = count;
    let mut init_params = [
        (&mut p_arg as *mut DevicePtr).cast(),
        (&mut z_arg as *mut DevicePtr).cast(),
        (&mut beta_arg as *mut f32).cast(),
        (&mut count_arg as *mut u32).cast(),
    ];
    cuda.launch(3, vector_grid, &mut init_params)?;
    let mut p_host = vec![0.0f32; flat_rhs.len()];
    check(
        unsafe { (cuda.driver.copy_dtoh)(p_host.as_mut_ptr().cast(), p, p_host.len() * 4) },
        "read initial direction",
    )?;
    let r_host = &flat_rhs;
    let mut rz: f64 = r_host
        .iter()
        .zip(&p_host)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let initial = r_host
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if initial <= 1e-12 {
        return Ok((vec![Vec3::ZERO; n], 0, 0.0));
    }
    // Replace the initial direction with the Jacobi-preconditioned residual;
    // the first direction update above uses beta=0 and performs that copy.
    let mut residual_norm = initial;
    let mut relative = 1.0f32;
    let mut iterations = 0;
    for _ in 0..max_iterations {
        let mut x_arg = p;
        let mut neighbors_arg = neighbor_ptr;
        let mut blocks_arg = block_ptr;
        let mut y_arg = ap;
        let mut diagonal_arg = diagonal_ptr;
        let mut count_arg = count;
        let mut matvec_params = [
            (&mut diagonal_arg as *mut DevicePtr).cast(),
            (&mut x_arg as *mut DevicePtr).cast(),
            (&mut neighbors_arg as *mut DevicePtr).cast(),
            (&mut blocks_arg as *mut DevicePtr).cast(),
            (&mut y_arg as *mut DevicePtr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        cuda.launch(0, grid, &mut matvec_params)?;
        let denom = cuda.dot(p, ap, count)?;
        if denom <= 1e-30 {
            break;
        }
        let alpha = (rz / denom) as f32;
        let mut dx_arg = dx;
        let mut r_arg = r;
        let mut p_arg = p;
        let mut ap_arg = ap;
        let mut alpha_arg = alpha;
        let mut count_arg = count;
        let mut update_params = [
            (&mut dx_arg as *mut DevicePtr).cast(),
            (&mut r_arg as *mut DevicePtr).cast(),
            (&mut p_arg as *mut DevicePtr).cast(),
            (&mut ap_arg as *mut DevicePtr).cast(),
            (&mut alpha_arg as *mut f32).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        cuda.launch(2, vector_grid, &mut update_params)?;
        let rr = cuda.dot(r, r, count)?;
        iterations += 1;
        residual_norm = rr.sqrt();
        relative = (residual_norm / initial) as f32;
        if relative < tolerance {
            break;
        }
        let mut inverse = inverse_ptr;
        let mut residual = r;
        let mut preconditioned = z;
        let mut count_arg = count;
        let mut precondition_params = [
            (&mut inverse as *mut DevicePtr).cast(),
            (&mut residual as *mut DevicePtr).cast(),
            (&mut preconditioned as *mut DevicePtr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        cuda.launch(1, grid, &mut precondition_params)?;
        let next_rz = cuda.dot(r, z, count)?;
        let beta = (next_rz / rz) as f32;
        let mut p_arg = p;
        let mut z_arg = z;
        let mut beta_arg = beta;
        let mut count_arg = count;
        let mut direction_params = [
            (&mut p_arg as *mut DevicePtr).cast(),
            (&mut z_arg as *mut DevicePtr).cast(),
            (&mut beta_arg as *mut f32).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        cuda.launch(3, vector_grid, &mut direction_params)?;
        rz = next_rz;
    }
    let mut output = vec![0.0f32; flat_rhs.len()];
    check(
        unsafe { (cuda.driver.copy_dtoh)(output.as_mut_ptr().cast(), dx, output.len() * 4) },
        "read PCG correction",
    )?;
    check(
        unsafe { (cuda.driver.context_synchronize)() },
        "cuCtxSynchronize",
    )?;
    let correction = output
        .chunks_exact(3)
        .map(|v| Vec3::new(v[0], v[1], v[2]))
        .collect();
    let _ = residual_norm;
    Ok((correction, iterations, relative))
}
