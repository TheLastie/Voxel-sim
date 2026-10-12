#!/usr/bin/env python3
"""Check and benchmark the voxel solver's 3×3 block sparse matvec on CUDA."""

from array import array
import ctypes
import time
from pathlib import Path


CUDA = ctypes.CDLL("libcuda.so.1")
PTR = ctypes.c_void_p
U32 = ctypes.c_uint
U64 = ctypes.c_uint64
SIZE = ctypes.c_size_t
INT = ctypes.c_int


def api(name, result, *arguments):
    function = getattr(CUDA, name)
    function.restype = result
    function.argtypes = arguments
    return function


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


PTX = (Path(__file__).resolve().parents[1] / "src/cuda/pcg.ptx").read_bytes()


def main():
    nx, ny, nz = 20, 25, 20
    count = nx * ny * nz
    invalid = 0xFFFFFFFF
    mass = array("f", [2.5] * count)
    x = array("f", (((i % 19) - 9) * 0.03125 for i in range(count * 3)))
    neighbors = array("I", [invalid]) * (count * 6)
    block = (700.0, 31.0, -12.0, 31.0, 520.0, 8.0, -12.0, 8.0, 430.0)
    blocks = array("f", block) * (count * 6)
    cpu = array("f", [0.0]) * (count * 3)
    gpu = array("f", [0.0]) * (count * 3)
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    for z in range(nz):
        for y in range(ny):
            for x0 in range(nx):
                i = (z * ny + y) * nx + x0
                for edge, (dx, dy, dz) in enumerate(offsets):
                    a, b, c = x0 + dx, y + dy, z + dz
                    if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                        neighbors[i * 6 + edge] = (c * ny + b) * nx + a

    for i in range(count):
        vx, vy, vz = x[i * 3 : i * 3 + 3]
        out = [mass[i] * vx, mass[i] * vy, mass[i] * vz]
        for edge in range(6):
            j = neighbors[i * 6 + edge]
            if j == invalid:
                continue
            delta = [vx - x[j * 3], vy - x[j * 3 + 1], vz - x[j * 3 + 2]]
            base = (i * 6 + edge) * 9
            for row in range(3):
                out[row] += sum(blocks[base + row * 3 + col] * delta[col] for col in range(3))
        cpu[i * 3 : i * 3 + 3] = array("f", out)

    check(api("cuInit", INT, U32)(0), "cuInit")
    device = INT()
    check(api("cuDeviceGet", INT, ctypes.POINTER(INT), INT)(ctypes.byref(device), 0), "cuDeviceGet")
    context = PTR()
    check(api("cuCtxCreate_v2", INT, ctypes.POINTER(PTR), U32, INT)(ctypes.byref(context), 0, device.value), "cuCtxCreate")
    module = PTR()
    ptx_buffer = ctypes.create_string_buffer(PTX)
    load_module = api("cuModuleLoadDataEx", INT, ctypes.POINTER(PTR), PTR, U32, ctypes.POINTER(INT), ctypes.POINTER(PTR))
    log = ctypes.create_string_buffer(8192)
    log_size = SIZE(len(log))
    options = (INT * 2)(3, 4)  # CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
    option_values = (PTR * 2)(ctypes.cast(log, PTR), ctypes.cast(ctypes.byref(log_size), PTR))
    status = load_module(ctypes.byref(module), ctypes.cast(ptx_buffer, PTR), 2, options, option_values)
    if status:
        raise RuntimeError(f"cuModuleLoadDataEx failed with CUDA status {status}: {log.value.decode(errors='replace')}")
    kernel = PTR()
    check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(kernel), module, b"block_sparse_matvec"), "cuModuleGetFunction")

    arrays = (mass, x, neighbors, blocks, gpu)
    device_ptrs = [U64() for _ in arrays]
    allocate = api("cuMemAlloc_v2", INT, ctypes.POINTER(U64), SIZE)
    release = api("cuMemFree_v2", INT, U64)
    copy_to_device = api("cuMemcpyHtoD_v2", INT, U64, PTR, SIZE)
    copy_to_host = api("cuMemcpyDtoH_v2", INT, PTR, U64, SIZE)
    for host, device_ptr in zip(arrays, device_ptrs):
        byte_count = len(host) * host.itemsize
        check(allocate(ctypes.byref(device_ptr), byte_count), "cuMemAlloc")
        if host is not gpu:
            check(copy_to_device(device_ptr, PTR(host.buffer_info()[0]), byte_count), "copy input")

    n = U32(count)
    parameters = (PTR * 6)(
        *[ctypes.cast(ctypes.byref(pointer), PTR) for pointer in device_ptrs[:4]],
        ctypes.cast(ctypes.byref(device_ptrs[4]), PTR),
        ctypes.cast(ctypes.byref(n), PTR),
    )
    launch = api("cuLaunchKernel", INT, PTR, U32, U32, U32, U32, U32, U32, U32, PTR, ctypes.POINTER(PTR), PTR)
    synchronize = api("cuCtxSynchronize", INT)
    grid_x = (count + 255) // 256
    check(launch(kernel, grid_x, 1, 1, 256, 1, 1, 0, None, parameters, None), "sparse matvec launch")
    check(synchronize(), "sparse matvec synchronize")
    check(copy_to_host(PTR(gpu.buffer_info()[0]), device_ptrs[4], len(gpu) * gpu.itemsize), "copy output")
    max_error = max(abs(a - b) for a, b in zip(cpu, gpu))
    if max_error > 2e-3:
        raise RuntimeError(f"sparse matvec mismatch: max absolute error {max_error}")
    print(f"CUDA block sparse matvec: nodes={count}, max_abs_error={max_error:.3e}")

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        check(launch(kernel, grid_x, 1, 1, 256, 1, 1, 0, None, parameters, None), "sparse matvec launch")
    check(synchronize(), "benchmark synchronize")
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{iterations} GPU sparse matvecs (dispatch + sync): {elapsed_ms:.2f} ms; {elapsed_ms / iterations:.3f} ms/operator")

    for pointer in device_ptrs:
        check(release(pointer), "cuMemFree")
    check(api("cuModuleUnload", INT, PTR)(module), "cuModuleUnload")
    check(api("cuCtxDestroy_v2", INT, PTR)(context), "cuCtxDestroy")


if __name__ == "__main__":
    main()
