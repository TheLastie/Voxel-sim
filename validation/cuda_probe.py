#!/usr/bin/env python3
"""Run a CUDA Driver API/PTX smoke benchmark without the CUDA toolkit."""

from array import array
import ctypes
import time


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


PTX = b"""
.version 7.0
.target sm_75
.address_size 64
.visible .entry vector_add(
    .param .u64 input_a,
    .param .u64 input_b,
    .param .u64 output_c,
    .param .u32 length
) {
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [input_a];
    ld.param.u64 %rd2, [input_b];
    ld.param.u64 %rd3, [output_c];
    ld.param.u32 %r4, [length];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r1, %r2, %r3, %r5;
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r1, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;
DONE:
    ret;
}
"""


def main():
    check(api("cuInit", INT, U32)(0), "cuInit")
    device = INT()
    check(api("cuDeviceGet", INT, ctypes.POINTER(INT), INT)(ctypes.byref(device), 0), "cuDeviceGet")
    context = PTR()
    check(
        api("cuCtxCreate_v2", INT, ctypes.POINTER(PTR), U32, INT)(
            ctypes.byref(context), 0, device.value
        ),
        "cuCtxCreate",
    )
    module = PTR()
    ptx_buffer = ctypes.create_string_buffer(PTX)
    check(
        api("cuModuleLoadData", INT, ctypes.POINTER(PTR), PTR)(
            ctypes.byref(module), ctypes.cast(ptx_buffer, PTR)
        ),
        "cuModuleLoadData",
    )
    kernel = PTR()
    check(
        api("cuModuleGetFunction", INT, ctypes.POINTER(PTR), PTR, ctypes.c_char_p)(
            ctypes.byref(kernel), module, b"vector_add"
        ),
        "cuModuleGetFunction",
    )

    count = 1 << 20
    left = array("f", ((i % 97) * 0.25 for i in range(count)))
    right = array("f", ((i % 43) * -0.5 for i in range(count)))
    output = array("f", [0.0]) * count
    device_left, device_right, device_output = U64(), U64(), U64()
    allocate = api("cuMemAlloc_v2", INT, ctypes.POINTER(U64), SIZE)
    release = api("cuMemFree_v2", INT, U64)
    for pointer in (ctypes.byref(device_left), ctypes.byref(device_right), ctypes.byref(device_output)):
        check(allocate(pointer, count * 4), "cuMemAlloc")
    copy_to_device = api("cuMemcpyHtoD_v2", INT, U64, PTR, SIZE)
    copy_to_host = api("cuMemcpyDtoH_v2", INT, PTR, U64, SIZE)
    check(copy_to_device(device_left, PTR(left.buffer_info()[0]), count * 4), "copy input A")
    check(copy_to_device(device_right, PTR(right.buffer_info()[0]), count * 4), "copy input B")
    length = U32(count)
    parameters = (PTR * 4)(
        ctypes.cast(ctypes.byref(device_left), PTR),
        ctypes.cast(ctypes.byref(device_right), PTR),
        ctypes.cast(ctypes.byref(device_output), PTR),
        ctypes.cast(ctypes.byref(length), PTR),
    )
    launch = api(
        "cuLaunchKernel", INT, PTR, U32, U32, U32, U32, U32, U32, U32, PTR,
        ctypes.POINTER(PTR), PTR
    )
    synchronize = api("cuCtxSynchronize", INT)
    grid_x = (count + 255) // 256
    check(launch(kernel, grid_x, 1, 1, 256, 1, 1, 0, None, parameters, None), "kernel launch")
    check(synchronize(), "kernel synchronize")
    check(copy_to_host(PTR(output.buffer_info()[0]), device_output, count * 4), "copy output")
    mismatches = sum(output[i] != left[i] + right[i] for i in range(count))
    print(f"CUDA PTX vector_add: device={device.value}, elements={count}, mismatches={mismatches}")
    if mismatches:
        raise RuntimeError(f"CUDA kernel returned {mismatches} incorrect values")

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        check(launch(kernel, grid_x, 1, 1, 256, 1, 1, 0, None, parameters, None), "kernel launch")
    check(synchronize(), "benchmark synchronize")
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{iterations} launches including dispatch and sync: {elapsed_ms:.2f} ms")

    for pointer in (device_left, device_right, device_output):
        check(release(pointer), "cuMemFree")
    check(api("cuModuleUnload", INT, PTR)(module), "cuModuleUnload")
    check(api("cuCtxDestroy_v2", INT, PTR)(context), "cuCtxDestroy")


if __name__ == "__main__":
    main()
