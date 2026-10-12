//! Report the graphics and compute adapters visible to wgpu.

#[cfg(target_os = "linux")]
const CUDA_DRIVER_LIB: &str = "libcuda.so.1";
#[cfg(target_os = "windows")]
const CUDA_DRIVER_LIB: &str = "nvcuda.dll";

fn main() {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        println!("No wgpu adapters found.");
    }

    for (index, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let limits = adapter.limits();
        println!(
            "[{index}] {} ({:?}, {:?}) — {:?}; max compute workgroups/dimension: {}",
            info.name,
            info.device_type,
            info.backend,
            info.driver,
            limits.max_compute_workgroups_per_dimension,
        );
    }
    report_cuda_driver();
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn report_cuda_driver() {
    use libloading::Library;
    use std::{ffi::CStr, os::raw::c_char};

    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, i32, i32) -> i32;
    type CuDriverGetVersion = unsafe extern "C" fn(*mut i32) -> i32;

    let library = match unsafe { Library::new(CUDA_DRIVER_LIB) } {
        Ok(library) => library,
        Err(error) => {
            println!("CUDA Driver API unavailable: {error}");
            return;
        }
    };
    let (Ok(init), Ok(get_count), Ok(get_device), Ok(get_name), Ok(get_version)) = (unsafe {
        (
            library.get::<CuInit>(b"cuInit\0"),
            library.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0"),
            library.get::<CuDeviceGet>(b"cuDeviceGet\0"),
            library.get::<CuDeviceGetName>(b"cuDeviceGetName\0"),
            library.get::<CuDriverGetVersion>(b"cuDriverGetVersion\0"),
        )
    }) else {
        println!("CUDA Driver API library is missing required symbols.");
        return;
    };
    let (init, get_count, get_device, get_name, get_version) =
        (*init, *get_count, *get_device, *get_name, *get_version);

    let status = unsafe { init(0) };
    if status != 0 {
        println!("CUDA Driver API initialization failed: {status}");
        return;
    }
    let mut count = 0;
    let status = unsafe { get_count(&mut count) };
    if status != 0 {
        println!("CUDA device enumeration failed: {status}");
        return;
    }
    let mut driver_version = 0;
    let _ = unsafe { get_version(&mut driver_version) };
    println!("CUDA Driver API: driver version {driver_version}, {count} device(s)");
    for ordinal in 0..count {
        let mut device = 0;
        let mut name = [0 as c_char; 256];
        let device_status = unsafe { get_device(&mut device, ordinal) };
        let name_status = unsafe { get_name(name.as_mut_ptr(), name.len() as i32, device) };
        if device_status == 0 && name_status == 0 {
            let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
            println!("CUDA device {ordinal}: {name}");
        } else {
            println!("CUDA device {ordinal}: query failed ({device_status}, {name_status})");
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn report_cuda_driver() {
    println!("CUDA Driver API probe is unsupported on this platform.");
}
