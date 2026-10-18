# Million-REV atmosphere-load A/B — 2026-10-18

## Setup

- Intel Core i5-14450HX; 16 logical processors reported by Windows.
- Release CPU backend; 1,000,000 total / 990,000 dynamic REVs.
- `mechanics_scale_capture 100 1 ic0`; 1 ms fixed step, 4 substeps, PCG tolerance `1e-5`.
- Atmospheric force law and PCG are unchanged. Only the independent per-REV
  atmosphere-load loop is run serially or with Rayon. A/B threshold is set by
  `VOXEL_ATMOSPHERE_PARALLEL_MIN_VOXELS`; unrelated solver parallel thresholds
  remain unchanged.
- Three fresh one-step runs per path, in alternating batches, with step profiling
  enabled. Timings include scene setup separately and exclude compilation.

## Results

| Path | Step wall time, three runs (s) | Median (s) | Median predict/tool (s) | Median PCG (s) |
| --- | ---: | ---: | ---: | ---: |
| Serial atmosphere | 45.464, 44.542, 39.907 | 44.542 | 17.477 | 16.110 |
| Parallel atmosphere | 39.645, 39.147, 38.337 | 39.147 | 16.320 | 14.540 |

The median full-step throughput improved by `1.138×` (12.1% less wall time).
Solver profile times are noisy across cold runs; this modest result is an
incremental gain, not the required 1.6–1.8× checkpoint and nowhere near 50×.

Every run reported the same 259 PCG iterations, solver residual `7.137991361e-6`,
zero broken bonds, mass `7869.998924 kg`, and kinetic energy
`6.772421827e-6 J`. No independent true-residual audit was enabled for these A/B
runs. The per-REV force arithmetic and summation order are unchanged, so the
parallel path has no cross-REV reduction or nondeterministic accumulation.

At 32,768 REVs the parallel atmosphere path did not help and added overhead.
The default atmosphere threshold is therefore `262,144` REVs; smaller scenes
keep the serial path. An environment override remains available for A/B work.

## Remaining work

This optimization touches only a small part of the million-REV step. PCG remains
the largest measured phase, and the main goal still requires a conservative
deformable hierarchy with local refinement. The separate oak-on-concrete heat
experiment also has an unresolved energy residual of about `-0.501 mJ`; do not
use it as proof that the cube physically heats or burns until its stepwise
mechanical energy loss is accounted for.
