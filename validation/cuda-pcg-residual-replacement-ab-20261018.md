# CUDA PCG residual replacement at 1M REVs — 2026-10-18

## Change under review

An opt-in CUDA PCG mode recomputes the residual as `r = b - A x` every 48
iterations, re-applies the existing preconditioner, and restarts the Krylov
direction. This addresses finite-precision drift between the recursive PCG
residual and the true residual. Enable with `VOXEL_CUDA_RESIDUAL_REPLACEMENT=1`.
`VOXEL_CUDA_ITERATIONS` controls the maximum iteration budget for this capture;
without it, the project's existing configured budget is unchanged. The feature
remains opt-in pending physical impact/thermal energy-ledger validation.

## Reproduction

- Release `mechanics_scale_capture`, RTX 5060 Laptop; CPU reference i5-14450HX.
- `mechanics_scale_capture 100 1 ic0`: 1,000,000 REVs, 990,000 active,
  `dt=1 ms`, four substeps, IC(0), relative true-residual target `1e-5`.
- Candidate: CUDA backend, residual replacement on, iteration cap 336.
- Alternating order: CPU A1, CUDA B1, CUDA B2, CPU A2. Each full capture
  independently audits all four fine-operator residuals in f64.

| Run | Backend | Wall ms | Max true residual | Broken | Mass kg | Kinetic J |
|---|---|---:|---:|---:|---:|---:|
| A1 | CPU | 20,572.357 | 8.572593e-6 | 0 | 7.869998924e3 | 6.772421827e-6 |
| B1 | CUDA + replacement | 17,464.977 | 9.217903e-6 | 0 | 7.869998924e3 | 6.821475434e-6 |
| B2 | CUDA + replacement | 17,617.483 | 9.217903e-6 | 0 | 7.869998924e3 | 6.821475434e-6 |
| A2 | CPU | 20,431.928 | 8.572593e-6 | 0 | 7.869998924e3 | 6.772421827e-6 |

CPU mean: 20,502.143 ms. CUDA mean: 17,541.230 ms. The candidate is 1.169x
faster, or 14.44% less elapsed time on this fixture. Maximum audited CUDA true
residual is `9.218e-6`, below the unchanged `1e-5` tolerance. Mass and broken
count match exactly at output precision. Kinetic energy differs by +0.724%.
The applied-coordinate residual matches the true residual to printed
precision, and position-rounding L2 is `3.41e-14 m`.

The earlier 352-iteration paired run also passed, at 1.141x speedup. A 320 cap
without residual replacement missed tolerance and fell back to CPU. The 336
run therefore preserves the tolerance while avoiding that fallback in this
million-REV fixture.

## Limits

This is an isolated synthetic mechanics capture, not a full game frame or a
long physical run. It has no complete energy ledger, so matching mass and
remaining below the 10% observed kinetic-energy deviation bound does not prove
that total energy is never created over time. Residual replacement is opt-in;
keep the CPU refinement/fallback and independent true-residual acceptance.
Before making it a default, validate tool impact, fracture, frictional heating,
and long coasting with a complete energy ledger and broader scene sizes.
This result does not meet real-time performance or 50x speedup goals.

Raw logs: `scale1m-cuda-residual-replace-336-20261018.log`,
`cuda-replace-336-ab-A1-20261018.log`, `cuda-replace-336-ab-B1-20261018.log`,
`cuda-replace-336-ab-B2-20261018.log`, and
`cuda-replace-336-ab-A2-20261018.log`.
