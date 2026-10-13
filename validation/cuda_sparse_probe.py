#!/usr/bin/env python3
"""Check and benchmark the voxel solver's 3×3 block sparse matvec on CUDA."""

from array import array
import ctypes
import sys
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
    nx, ny, nz = (map(int, sys.argv[1:]) if len(sys.argv) == 4 else (20, 25, 20))
    count = nx * ny * nz
    invalid = 0xFFFFFFFF
    diagonal_block = (2.5, 0.1, -0.05, 0.1, 3.0, 0.08, -0.05, 0.08, 4.0)
    diagonal = array("f", diagonal_block) * count
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
        out = [
            diagonal_block[row * 3] * vx
            + diagonal_block[row * 3 + 1] * vy
            + diagonal_block[row * 3 + 2] * vz
            for row in range(3)
        ]
        for edge in range(6):
            j = neighbors[i * 6 + edge]
            if j == invalid:
                continue
            neighbor_x = [x[j * 3], x[j * 3 + 1], x[j * 3 + 2]]
            base = (i * 6 + edge) * 9
            for row in range(3):
                out[row] += sum(blocks[base + row * 3 + col] * neighbor_x[col] for col in range(3))
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

    arrays = (diagonal, x, neighbors, blocks, gpu)
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

    jacobi_kernel = PTR()
    check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(jacobi_kernel), module, b"block_jacobi"), "cuModuleGetFunction(block_jacobi)")
    jacobi_parameters = (PTR * 4)(
        ctypes.cast(ctypes.byref(device_ptrs[0]), PTR),
        ctypes.cast(ctypes.byref(device_ptrs[1]), PTR),
        ctypes.cast(ctypes.byref(device_ptrs[4]), PTR),
        ctypes.cast(ctypes.byref(n), PTR),
    )
    check(launch(jacobi_kernel, grid_x, 1, 1, 256, 1, 1, 0, None, jacobi_parameters, None), "block Jacobi launch")
    check(synchronize(), "block Jacobi synchronize")
    check(copy_to_host(PTR(gpu.buffer_info()[0]), device_ptrs[4], len(gpu) * gpu.itemsize), "copy Jacobi output")
    jacobi_reference = array("f")
    for i in range(count):
        values = x[i * 3 : i * 3 + 3]
        jacobi_reference.extend(
            sum(diagonal_block[row * 3 + col] * values[col] for col in range(3))
            for row in range(3)
        )
    jacobi_error = max(abs(a - b) for a, b in zip(jacobi_reference, gpu))
    if jacobi_error > 2e-3:
        raise RuntimeError(f"block Jacobi mismatch: max absolute error {jacobi_error}")
    print(f"CUDA block Jacobi: nodes={count}, max_abs_error={jacobi_error:.3e}")

    direction_kernel = PTR()
    check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(direction_kernel), module, b"pcg_update_direction"), "cuModuleGetFunction(pcg_update_direction)")
    zeros = array("f", [0.0]) * len(gpu)
    check(copy_to_device(device_ptrs[4], PTR(zeros.buffer_info()[0]), len(zeros) * zeros.itemsize), "initialize direction output")
    beta = array("f", [0.0])
    beta_device = U64()
    check(allocate(ctypes.byref(beta_device), beta.itemsize), "beta allocation")
    check(copy_to_device(beta_device, PTR(beta.buffer_info()[0]), beta.itemsize), "copy beta")
    vector_grid = (count * 3 + 255) // 256
    direction_parameters = (PTR * 4)(
        ctypes.cast(ctypes.byref(device_ptrs[4]), PTR),
        ctypes.cast(ctypes.byref(device_ptrs[1]), PTR),
        ctypes.cast(ctypes.byref(beta_device), PTR),
        ctypes.cast(ctypes.byref(n), PTR),
    )
    check(launch(direction_kernel, vector_grid, 1, 1, 256, 1, 1, 0, None, direction_parameters, None), "PCG direction launch")
    check(synchronize(), "PCG direction synchronize")
    check(copy_to_host(PTR(gpu.buffer_info()[0]), device_ptrs[4], len(gpu) * gpu.itemsize), "copy direction output")
    direction_error = max(abs(a - b) for a, b in zip(x, gpu))
    if direction_error > 1e-6:
        raise RuntimeError(f"PCG direction mismatch: max absolute error {direction_error}")
    print(f"CUDA PCG direction update: nodes={count}, max_abs_error={direction_error:.3e}")
    check(release(beta_device), "beta free")

    dot_kernel = PTR()
    check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(dot_kernel), module, b"dot_product"), "dot product kernel")
    partial_count = (count + 255) // 256
    partials = array("d", [0.0]) * partial_count
    partial_device = U64()
    check(allocate(ctypes.byref(partial_device), len(partials) * partials.itemsize), "dot partial allocation")
    dot_count = U32(count)
    dot_parameters = (PTR * 4)(
        ctypes.cast(ctypes.byref(device_ptrs[1]), PTR),
        ctypes.cast(ctypes.byref(device_ptrs[1]), PTR),
        ctypes.cast(ctypes.byref(partial_device), PTR),
        ctypes.cast(ctypes.byref(dot_count), PTR),
    )
    check(launch(dot_kernel, partial_count, 1, 1, 256, 1, 1, 0, None, dot_parameters, None), "dot product launch")
    check(synchronize(), "dot product synchronize")
    check(copy_to_host(PTR(partials.buffer_info()[0]), partial_device, len(partials) * partials.itemsize), "copy dot partials")
    actual_dot = sum(partials)
    expected_dot = sum(value * value for value in x)
    relative_error = abs(actual_dot - expected_dot) / max(abs(expected_dot), 1e-30)
    if relative_error > 1e-12:
        raise RuntimeError(f"CUDA dot product mismatch: relative error {relative_error:.3e}")
    print(f"CUDA block-reduced dot: nodes={count}, blocks={partial_count}, relative_error={relative_error:.3e}")

    ratio_kernel = PTR()
    check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(ratio_kernel), module, b"scalar_ratio"), "scalar ratio kernel")
    scalar_inputs = array("d", [4.5, 1.5])
    check(copy_to_device(partial_device, PTR(scalar_inputs.buffer_info()[0]), len(scalar_inputs) * scalar_inputs.itemsize), "copy scalar inputs")
    scalar_output = array("f", [0.0])
    scalar_output_device = U64()
    check(allocate(ctypes.byref(scalar_output_device), scalar_output.itemsize), "scalar output allocation")
    numerator_ptr = U64(partial_device.value)
    denominator_ptr = U64(partial_device.value + scalar_inputs.itemsize)
    ratio_parameters = (PTR * 3)(
        ctypes.cast(ctypes.byref(numerator_ptr), PTR),
        ctypes.cast(ctypes.byref(denominator_ptr), PTR),
        ctypes.cast(ctypes.byref(scalar_output_device), PTR),
    )
    check(launch(ratio_kernel, 1, 1, 1, 256, 1, 1, 0, None, ratio_parameters, None), "scalar ratio launch")
    check(synchronize(), "scalar ratio synchronize")
    check(copy_to_host(PTR(scalar_output.buffer_info()[0]), scalar_output_device, scalar_output.itemsize), "copy scalar result")
    if abs(scalar_output[0] - 3.0) > 1e-6:
        raise RuntimeError(f"CUDA scalar ratio mismatch: got {scalar_output[0]}")
    print(f"CUDA scalar ratio: {scalar_inputs[0]} / {scalar_inputs[1]} = {scalar_output[0]}")

    # Exercise the same asynchronous device-side PCG sequence as the engine on
    # a 2-node SPD block system with an analytic solution.
    functions = {}
    for name in (b"block_jacobi", b"pcg_update", b"pcg_update_direction", b"dot_reduce"):
        function = PTR()
        check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(function), module, name), f"{name.decode()} kernel")
        functions[name] = function

    def device_buffer(host):
        pointer = U64()
        check(allocate(ctypes.byref(pointer), len(host) * host.itemsize), "PCG buffer allocation")
        if len(host):
            check(copy_to_device(pointer, PTR(host.buffer_info()[0]), len(host) * host.itemsize), "PCG buffer upload")
        return pointer

    def launch_args(function, *values):
        params = (PTR * len(values))(*[ctypes.cast(ctypes.byref(value), PTR) for value in values])
        check(launch(function, 1, 1, 1, 256, 1, 1, 0, None, params, None), "PCG kernel launch")

    # Production captures at 4,203 active REV use 17 first-stage dot partials.
    # Check a nonzero residual explicitly; an exactly solved 2-node system can
    # hide a duplicated-partial multiplier behind its near-zero residual.
    reduction_input = array("d", range(1, 18))
    reduction_output = array("d", [0.0])
    reduction_input_d = device_buffer(reduction_input)
    reduction_output_d = device_buffer(reduction_output)
    reduction_count = U32(len(reduction_input))
    launch_args(functions[b"dot_reduce"], reduction_input_d, reduction_output_d, reduction_count)
    check(synchronize(), "17-partial reduction synchronize")
    check(copy_to_host(PTR(reduction_output.buffer_info()[0]), reduction_output_d, reduction_output.itemsize), "17-partial reduction download")
    if abs(reduction_output[0] - sum(reduction_input)) > 1e-12:
        raise RuntimeError(f"CUDA 17-partial reduction mismatch: got {reduction_output[0]}, expected {sum(reduction_input)}")
    print(f"CUDA dot reduction: partials=17, sum={reduction_output[0]:.1f}")

    ratio_reduce = PTR()
    check(api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(ctypes.byref(ratio_reduce), module, b"dot_reduce_ratio"), "dot reduce ratio kernel")
    numerator = array("d", [153.0])
    ratio_output = array("f", [0.0])
    numerator_d = device_buffer(numerator)
    ratio_output_d = device_buffer(ratio_output)
    launch_args(ratio_reduce, reduction_input_d, numerator_d, ratio_output_d, reduction_count)
    check(synchronize(), "fused dot ratio synchronize")
    check(copy_to_host(PTR(ratio_output.buffer_info()[0]), ratio_output_d, ratio_output.itemsize), "fused dot ratio download")
    if abs(ratio_output[0] - 1.0) > 1e-7:
        raise RuntimeError(f"CUDA fused dot ratio mismatch: got {ratio_output[0]}, expected 1.0")
    zero_partials = array("d", [0.0])
    zero_partials_d = device_buffer(zero_partials)
    one_partial = U32(1)
    launch_args(ratio_reduce, zero_partials_d, numerator_d, ratio_output_d, one_partial)
    check(synchronize(), "zero denominator fused ratio synchronize")
    check(copy_to_host(PTR(ratio_output.buffer_info()[0]), ratio_output_d, ratio_output.itemsize), "zero denominator fused ratio download")
    if ratio_output[0] != 0.0:
        raise RuntimeError(f"CUDA fused dot ratio must be zero for a zero denominator, got {ratio_output[0]}")
    print("CUDA fused dot ratio: quotient=1.0, zero denominator guarded")

    rhs_host = array("f", [1.0, 2.0, 3.0, 4.0, -1.0, 0.5])
    zero_host = array("f", [0.0]) * 6
    diagonal_host = array("f", [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0] * 2)
    inverse_host = array("f", [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5] * 2)
    invalid = 0xFFFFFFFF
    neighbors_host = array("I", [1, invalid, invalid, invalid, invalid, invalid,
                                  invalid, 0, invalid, invalid, invalid, invalid])
    blocks_host = array("f", [0.0]) * 108
    for node, edge in ((0, 0), (1, 1)):
        base = node * 54 + edge * 9
        for axis in range(3):
            blocks_host[base + axis * 3 + axis] = -0.2

    diagonal_d = device_buffer(diagonal_host)
    inverse_d = device_buffer(inverse_host)
    neighbors_d = device_buffer(neighbors_host)
    blocks_d = device_buffer(blocks_host)
    rhs_d = device_buffer(rhs_host)
    dx_d = device_buffer(zero_host)
    p_d = device_buffer(zero_host)
    ap_d = device_buffer(zero_host)
    z_d = device_buffer(zero_host)
    dot_partials_d = device_buffer(array("d", [0.0]))
    rho_a_d = device_buffer(array("d", [0.0]))
    rho_b_d = device_buffer(array("d", [0.0]))
    denominator_d = device_buffer(array("d", [0.0]))
    residual_d = device_buffer(array("d", [0.0]))
    alpha_d = device_buffer(array("f", [0.0]))
    beta_d = device_buffer(array("f", [0.0]))
    beta_zero_d = device_buffer(array("f", [0.0]))
    n_arg = U32(2)

    launch_args(functions[b"block_jacobi"], inverse_d, rhs_d, z_d, n_arg)

    def pcg_dot(left, right, target):
        launch_args(dot_kernel, left, right, dot_partials_d, n_arg)
        one_block = U32(1)
        launch_args(functions[b"dot_reduce"], dot_partials_d, target, one_block)

    def ratio(numerator, denominator, target):
        launch_args(ratio_kernel, numerator, denominator, target)

    pcg_dot(rhs_d, z_d, rho_a_d)
    launch_args(functions[b"pcg_update_direction"], p_d, z_d, beta_zero_d, n_arg)
    rho_d, rho_next_d = rho_a_d, rho_b_d
    # Match the engine's fixed configured PCG budget. The analytic system
    # converges in two iterations; the remaining iterations must not corrupt
    # an already-converged correction.
    for _ in range(80):
        launch_args(kernel, diagonal_d, p_d, neighbors_d, blocks_d, ap_d, n_arg)
        pcg_dot(p_d, ap_d, denominator_d)
        ratio(rho_d, denominator_d, alpha_d)
        launch_args(functions[b"pcg_update"], dx_d, rhs_d, p_d, ap_d, alpha_d, n_arg)
        launch_args(functions[b"block_jacobi"], inverse_d, rhs_d, z_d, n_arg)
        pcg_dot(rhs_d, z_d, rho_next_d)
        ratio(rho_next_d, rho_d, beta_d)
        launch_args(functions[b"pcg_update_direction"], p_d, z_d, beta_d, n_arg)
        rho_d, rho_next_d = rho_next_d, rho_d
    pcg_dot(rhs_d, rhs_d, residual_d)
    check(synchronize(), "PCG sequence synchronize")

    correction_host = array("f", [0.0]) * 6
    residual_host = array("d", [0.0])
    check(copy_to_host(PTR(correction_host.buffer_info()[0]), dx_d, len(correction_host) * correction_host.itemsize), "PCG result download")
    check(copy_to_host(PTR(residual_host.buffer_info()[0]), residual_d, residual_host.itemsize), "PCG residual download")
    expected = [0.0] * 6
    for component in range(3):
        b0, b1 = rhs_host[component], rhs_host[3 + component]
        expected[component] = (2.0 * b0 + 0.2 * b1) / 3.96
        expected[3 + component] = (0.2 * b0 + 2.0 * b1) / 3.96
    max_correction_error = max(abs(a - b) for a, b in zip(correction_host, expected))
    initial_norm = sum(value * value for value in rhs_host) ** 0.5
    relative_residual = residual_host[0] ** 0.5 / initial_norm
    if relative_residual > 1e-5 or max_correction_error > 1e-5:
        raise RuntimeError(f"CUDA PCG mismatch: relative residual={relative_residual:.3e}, max correction error={max_correction_error:.3e}")
    print(f"CUDA device-resident PCG: nodes=2, iterations=80, relative_residual={relative_residual:.3e}, max_correction_error={max_correction_error:.3e}")

    for pointer in (diagonal_d, inverse_d, neighbors_d, blocks_d, rhs_d, dx_d, p_d, ap_d, z_d,
                    dot_partials_d, rho_a_d, rho_b_d, denominator_d, residual_d,
                    alpha_d, beta_d, beta_zero_d):
        check(release(pointer), "PCG buffer free")

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        check(launch(kernel, grid_x, 1, 1, 256, 1, 1, 0, None, parameters, None), "sparse matvec launch")
    check(synchronize(), "benchmark synchronize")
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{iterations} GPU sparse matvecs (dispatch + sync): {elapsed_ms:.2f} ms; {elapsed_ms / iterations:.3f} ms/operator")

    for pointer in device_ptrs:
        check(release(pointer), "cuMemFree")
    check(release(partial_device), "dot partial free")
    check(release(scalar_output_device), "scalar output free")
    check(api("cuModuleUnload", INT, PTR)(module), "cuModuleUnload")
    check(api("cuCtxDestroy_v2", INT, PTR)(context), "cuCtxDestroy")


if __name__ == "__main__":
    main()
