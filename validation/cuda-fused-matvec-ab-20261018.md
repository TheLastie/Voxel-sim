# CUDA fused matvec + PCG dot: A/B, 2026-10-18

## Change

Experimental path `VOXEL_CUDA_FUSED_MATVEC_DOT=1` fuses the block sparse
`A·p` operation with generation of f64 partial sums for `p·(A·p)`. The existing
f64 final reduction and PCG alpha calculation remain. Matrix assembly,
preconditioner, convergence tolerance, residual replacement cadence, and
independent true-residual audit are unchanged. The default is disabled; systems
with at most 8,192 REV rows use the original kernels.

## Measurement

RTX 5060 Laptop, 1,000,000 REV rows / 990,000 active, 4 mechanics substeps,
1 ms fixed timestep, IC(0), 48 PCG iterations per residual replacement. The
same release binary ran alternating A/B captures; A has fusion disabled and B
enabled.

| Run | Wall time (ms) |
| --- | ---: |
| A1 | 22,683.993 |
| B1 | 22,253.737 |
| B2 | 22,231.910 |
| A2 | 25,125.680 |
| A3 | 22,831.560 |
| B3 | 22,382.686 |

Control mean: 23,547.078 ms. Fused mean: 22,289.444 ms. Mean speedup: 1.056×
(5.34% less elapsed time). The second control is a clear slow outlier. The first
and third paired comparisons each improve by roughly 1.9%, so the likely
steady-state gain is modest and may be within device/load variation. This is
not a large-scale optimization by itself.

## Numerical and physical checks

- Independent f64 true residual maximum: `9.217903422e-6`, below the unchanged
  `1e-5` tolerance.
- All captures reported CUDA; no CPU fallback.
- CSV output matched across runs: mass `7.869998924e3 kg`, kinetic energy
  `6.821475434e-6 J`, broken bonds `0`.
- A 32³ smoke scene loaded the PTX kernel and passed its independent residual
  audit (four true residuals between `2.64e-6` and `3.20e-6`).

The scale fixture does not provide a complete energy ledger and does not cover
long contact, fracture, or heating evolution. Therefore this path remains
opt-in and this result does not establish long-horizon energy conservation or
full-game performance. The change only removes a small amount of repeated GPU
memory traffic and a kernel launch during each PCG iteration.

## Raw captures

- `cuda-fused-matvec-ab-A1-20261018.log` through `A3`
- `cuda-fused-matvec-ab-B1-20261018.log` through `B3`
- `cuda-fused-matvec-smoke-20261018.log`
- `cuda-fused-matvec-build-20261018.log`
