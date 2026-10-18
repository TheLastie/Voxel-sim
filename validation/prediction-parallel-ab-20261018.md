# Prediction-loop parallel A/B — 2026-10-18

## Result

Rejected as a performance optimization. Parallelizing independent per-REV applied-load, drift/target, constraint-reset, and half-contact-drift loops did not improve the million-REV workload. The code change was removed; only the temporary measurement logs remain.

## Method

- Same release binary, alternated serial and parallel modes in three pairs.
- Fixture: 1,000,000 total / 990,000 dynamic REVs, CPU IC(0), 1 ms update, four substeps, tolerance `1e-5`.
- The A/B switch changed only the four added prediction loops. Existing PCG and atmosphere parallel settings were identical.
- Each run recorded PCG residual audit and physical summary.

| Mode | App wall median | `predict/tool` median | PCG iterations | Worst true residual |
|---|---:|---:|---:|---:|
| Serial | 48.620 s | 14.226 s | 289 | `7.3774e-6` |
| Parallel | 48.642 s | 14.383 s | 289 | `7.3774e-6` |

Parallel was 0.05% slower by total wall time and about 1.1% slower in `predict/tool`; this is effectively no speedup. All six runs matched exactly in reported final residual (`7.038592685e-6`), true residual sequence, mass (7,869.998924 kg), kinetic energy (`6.773904106e-6` J), and zero broken bonds. The candidate therefore preserved the checked physics but did not meet the performance goal and was removed.

The previous three initial candidate-only runs in `predict-parallel-candidate-*` are exploratory and excluded from the paired result.
