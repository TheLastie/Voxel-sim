# Million-REV V-cycle smoother and transfer A/B — 2026-10-18

## Setup

- Release `mechanics_scale_capture`, 1,000,000 total / 990,000 active REVs.
- CPU, physical step 1 ms, four 0.25 ms substeps, PCG tolerance `1e-5`.
- Independent `b-Ax` residual audit and step profiling enabled in every run.
- The GUI was closed for the benchmark. IC(0) control and Jacobi candidate used the same executable; candidate was selected by `VOXEL_MULTIGRID_SMOOTHER=jacobi`.
- Two alternating IC(0)/Jacobi pairs; an additional Jacobi-only profile explored disabling first-transfer smoothing.

## Paired results

| Path | Whole-step wall times | Median | Solve median | Factor median | PCG median | PCG iterations |
|---|---:|---:|---:|---:|---:|---:|
| CPU IC(0) control | 32.756, 32.619 s | 32.688 s | 16.385 s | 0.462 s | 11.751 s | 259 |
| Jacobi-smoothed multigrid | 40.277, 40.106 s | 40.191 s | 23.942 s | 4.343 s | 15.524 s | 114 |

Jacobi multigrid is 23.0% slower on the cold step. It cuts PCG iterations by
56%, but its coarsening/setup and each V-cycle cost more than natural IC(0).
This is not accepted as a performance optimization. The symmetry/positive-form
unit check passed with the Jacobi setting enabled. Every solve's independent
true residual remained below `1e-5`; maximum observed was `9.141e-6`.

Mass and broken-bond count matched between the paired modes. Candidate final
kinetic energy was `6.770102047e-6 J`, versus `6.772421827e-6 J` for IC(0), a
decrease of 0.034%. This fixture does not report a complete energy ledger, so
the kinetic comparison alone is not a claim that total energy is conserved.

## First-transfer variant

With both Jacobi smoothing and `VOXEL_MG_SMOOTH_FIRST_TRANSFER=0`, the first
coarse level had 1,559,082 off-diagonal blocks instead of 4,582,644. The full
step nevertheless rose to `42.880 s`; PCG took `18.985 s` and 204 iterations.
Its true residual stayed below `6.02e-6`. Reducing coarse fill alone did not
make the V-cycle cheaper enough to beat the smoothed-transfer candidate.

## Artifacts and status

Raw logs: `multigrid-gs-current-ab-20261018.log`,
`multigrid-jacobi-current-ab-20261018.log`,
`ic0-jacobi-pair{2,3}-{control,candidate}.log`, and
`multigrid-jacobi-unsmoothed-ab-20261018.log` in this directory.

Both switches remain opt-in. Neither multigrid variant replaces IC(0), and the
full physical energy balance, contact/fracture trajectory equivalence, warm
multi-step throughput, and 1.6–1.8× commit gate remain unverified/unmet.
