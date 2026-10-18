# Preliminary physical substep refinement — 2026-10-16

## Setup

- Scene: `scenes/mounted_axe_impact.ron`; 12×12×30 wood, 80×80×1 ground, mounted wedge.
- Outer physical step: 10 ms. Simulated duration: 0.2 s (20 steps).
- Applied tool force: 150 N for all 20 steps; same material, solver iteration caps, tolerances, and initial state.
- Only internal substep count changed. Each run used the release `axe_decay_capture` binary on the RTX 5060 Laptop.

## Results

| Substeps | Internal dt | Wall time | Broken bonds at 0.2 s | Tool speed | Peak wood speed | Damage heat | External work |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.25 ms | not timed in the capture script | 760 | 2.402 m/s | 2.407 m/s | 14.98 J | 50.72 J |
| 20 | 0.5 ms | 23.62 s | 1,391 | 3.136 m/s | 3.143 m/s | 44.37 J | 57.35 J |
| 10 | 1 ms | 11.49 s | 3,719 | 3.719 m/s | 3.729 m/s | 519.83 J | 62.25 J |
| 5 | 2 ms | 11.21 s | 11,688 | 3.910 m/s | 178.348 m/s | 785,208 J | −1,089.77 J |

All levels first reported bond failure on outer step 9. The 20-substep result already differs materially from the 40-substep baseline. At 10 and 5 substeps, damage heat greatly exceeds the supplied work; 5 substeps also produce extreme wood velocity and reversed external work. Those are unstable, nonphysical outcomes, not viable performance settings.

## Decision

Keep the 10 ms outer step and 40 × 0.25 ms internal steps for this impact scene. This short fracture run is not a formal convergence-order/GCI result: bond failure is a discontinuous quantity, the coarser runs diverge strongly, and the 40-step run has not itself been refined. Any further speed work must preserve this temporal resolution and solver tolerances while optimizing the implementation.

## Physics-preserving implementation change

The viscoelastic damping heat is now computed from the kinetic-energy change induced by its applied impulse, including the free-axis mask for prescribed degrees of freedom. This is algebraically the same energy difference as the prior before/after kinetic-energy evaluation. The 0.2 s, 40-substep capture retained the same break step, 760 broken bonds, tool and wood velocities, damage heat, and external work at the reported precision.

The CUDA PCG direction update now computes the scalar `rho_next / rho` once per block and shares it with that block's vector lanes, removing a separate scalar-ratio kernel launch per PCG iteration. On warmed first-three-step profiles, median PCG time changed from 284.34–288.01 ms to 249.23–254.36 ms; residual tolerances and iteration counts remained satisfied. The full 0.2 s capture took 46.14 s, versus 47.01 s before this fusion (about 1.9% faster end-to-end for this short run). This is useful but far below the required real-time target.

## Deterministic CUDA-path assembly parallelism — 2026-10-16

When CUDA PCG is selected for at least 4,096 scene voxels, the independent constraint-build, force-build, and per-voxel matrix-row assembly passes use Rayon. Each work item writes its own fixed slot or row; the subsequent constraint gather remains in stable axis-major order. The CPU PCG cutoff and path are unchanged.

On the same warmed three-step capture, assembly fell from 280–286 ms to 89–91 ms per 10 ms step. PCG iteration counts (3,648 and 3,696 in the two warmed steps) and reported residuals (`7.850e-6`, `7.781e-6`) matched the serial CUDA-assembly run. The full 0.2 s capture completed in 41.17 s versus 45.63 s immediately before this change (9.8% lower wall time; one capture per version, so treat as an initial measurement). It retained first fracture at outer step 9, 760 broken bonds, tool-axis position `-0.184896 m`, tool speed `2.402173 m/s`, damage heat `14.97714747 J`, and external work `50.71601266 J`.

The threshold for committing the performance branch has not been met. Further optimization must keep the 0.25 ms physical substep and configured PCG tolerance, and each larger change needs a same-scene physical replay.

## Independent rotation-fit updates — 2026-10-16

The per-REV quaternion update in `fit_rotations` is independent after the stable serial gradient gather. It now uses Rayon at 8,192 scene REVs, while the existing 32,768 cutoff still governs other CPU parallel paths. On warmed three-step profiles with 16 Rayon workers, `predict/tool` decreased from 133–135 ms to 123–124 ms. A full 0.2 s replay took 40.85 s versus 41.17 s with serial rotation updates; the physics endpoint remained exactly the same at the reported precision. This small end-to-end change is near capture noise and is not a major performance gain.

An exact dependency-layered parallel damping prototype was measured and removed. It preserved the per-REV dependency order but required 51 layers for 12,096 candidate bonds; warmed damping rose from about 202 ms to 238 ms. It did not justify the extra scheduling complexity.
