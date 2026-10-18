# Physical million-REV scaling investigation, 2026-10-17

The target is a world containing millions of authoritative 1 cm REVs, with
physical interactions and reversible refinement. Neither this investigation
nor its experimental preconditioner establishes real-time operation or the
requested 50x improvement. IC(0) remains the default CPU method.

## Workload and hardware

CPU: Intel Core i5-14450HX, Windows, reported physical RAM 16,868,958,208 bytes.
CPU backend forced with `VOXEL_SOLVER_BACKEND=cpu`; release build. No browser
or web server is used. The new `mechanics_scale_capture` binary constructs a
solid iron cube from the general `Scene::from_voxels` API. The bottom face is
kinematic, gravity is 9.81 m/s2, and the standard atmosphere remains enabled.
There are no prescribed trajectories or special physical rules in the capture.
The fixed/dynamic interface has no bonds in this builder, so the dynamic cube
is supported by contact with its kinematic bottom layer, not elastically clamped.

All scale runs below measure **one cold 1 ms physical update**, containing four
0.25 ms substeps. The CPU cap is 1,024 iterations per solve, with relative
tolerance 1e-5. The million-REV cube is 1 m on each side: 1,000,000 total REVs,
990,000 dynamic REVs and 10,000 base REVs. Scene preparation is excluded from
the per-update clock. These measurements are exploratory single runs, not a
repeated performance acceptance result or a steady-state throughput estimate.

## Implementation

`src/multigrid.rs` implements an opt-in aggregation V-cycle. Connected graph
aggregates define the initial transfer space. Coarse operators are Galerkin
projections `A_c = P^T A P`; setup accumulation uses f64. Restriction is the
transpose of prolongation. Forward and backward block Gauss-Seidel smoothing
and a small dense Cholesky bottom solve form a symmetric preconditioner.
Partially fixed coordinates are projected out of transfer operators.

This changes the numerical preconditioner only. The outer PCG still solves
the original fine matrix; contact, damage, thermal history and physical
resolution remain fine. It is not the deformable physical coarse tier needed
to avoid advancing every REV in a large world.

The first implementation used unsmoothed transfer throughout. Smoothing on
every level lowered iterations further but caused expensive coarse stencil
growth. The current implementation smooths the finest transfer once and uses
connected unsmoothed transfer on subsequent levels. This limits setup growth.
For the million-REV cube the current row counts are:

`990000 -> 129949 -> 16342 -> 2092 -> 294 -> 61 -> 30 -> 23`.

## Measured cold update times

| Million-REV cube | Default IC(0) | All-level smoothed prototype | Current bounded prototype |
| --- | ---: | ---: | ---: |
| Whole physical update | 36,925.713 ms | 90,037.289 ms | 44,175.401 ms |
| Coupled solve | 20,864.41 ms | 74,150.63 ms | 27,956.64 ms |
| Matrix assembly, nested in solve | 1,823.29 ms | 1,758.98 ms | 1,744.77 ms |
| Preconditioner setup, nested in solve | 437.78 ms | 60,129.81 ms | 4,550.89 ms |
| PCG, nested in solve | 17,353.19 ms | 11,047.21 ms | 20,393.24 ms |
| Prediction/tool phase, includes first collision search and rotation fit | 14,091.43 ms | 14,030.03 ms | 14,341.75 ms |
| Accumulated PCG iterations | 384 | 38 | 100 |
| Final reported residual | 9.517e-6 | 8.202e-6 | 6.262e-6 |

Limiting coarse fill reduced prototype setup by about 13.2x, and the prototype
cold update by about 2.0x relative to its all-level-smoothed version. The
current prototype is nevertheless about 19.6% slower than the default on this
cold workload, so it is not accepted as an application performance improvement.
The low iteration count alone did not translate into a proportional wall-time
gain: each V-cycle traverses several matrices and transfer operators.

At 32,768 total / 31,744 active REVs, the initial all-level-smoothed version
took 971.037 ms versus 746.240 ms for IC(0). PCG iterations were 32 versus 175;
PCG itself took 247.34 versus 357.56 ms, but setup took 362.17 versus 13.56 ms.

## Physical diagnostics and limits

The million-REV runs all reported zero broken bonds and mass 7,869.998924 kg.
Final kinetic energies were respectively 3.334952596e-5 J, 4.112414581e-5 J
and 4.073898710e-5 J. The last difference is small in absolute energy but
about 22% relative to the IC(0) result. Physical equivalence is **not proven**.
Reported runtime residuals are the existing PCG recurrence residual, rather
than an independent high-precision residual of every solved runtime system.
Tighter/refined comparisons and a true residual audit remain necessary.

The mounted-axe check used the unchanged 10 ms / 40 substep / 150 N workload.
The initial unsmoothed prototype took 14.014 s for 10 updates versus 12.922 s
for IC(0). The all-level-smoothed version took 14.173 s. Final broken-bond
counts were 241, 228 and 233 respectively. These are further reasons to keep
the method experimental. They are historical prototype runs, not measurements
of the final bounded implementation on that impact.

Three numerical checks pass for the current module: Galerkin quadratic energy
with partial constraints, symmetric positive V-cycle action with fixed axes
kept at zero, and a PCG solve checked against the original fine operator's
explicit residual. They cover a 512-row anisotropic lattice, not a proof of
accuracy or performance for the full evolving world.

At the multigrid milestone, `cargo check --release --all-targets` passed; the
release library suite passed all 91 checks after correcting the boundary-ledger
fixture to compare exit state after ambient drag with an unbounded reference
step. No physical force model was disabled for that ledger check.

## Independent residual audit follow-up

`VOXEL_AUDIT_PCG_RESIDUAL=1` now independently computes f64 `b-Ax` for the
returned correction vector and for the correction actually retained after
adding it to f32 scene positions. The check uses the original assembled fine
diagonal/off-diagonal blocks and RHS, independently of PCG's residual recurrence
and of its preconditioner. It does not change the trajectory. When enabled,
`mechanics_scale_capture` fails if any independent or applied residual in the
fixed step exceeds the existing tolerance; failed captures retain the measured
state. The worst value is tracked over all substeps/corotation corrections.

| Workload, last frozen solve | Recurrence residual | Independent correction residual | Applied-coordinate residual |
| --- | ---: | ---: | ---: |
| 32,768 REVs, IC(0) | 9.333198250e-6 | 9.842746359e-6 | 0.2023379729 |
| 32,768 REVs, bounded multigrid | 9.170083104e-6 | 9.379298314e-6 | 0.2023443261 |
| 1,000,000 REVs, IC(0) | 9.517246326e-6 | 1.070006328e-5 | 0.4365206060 |

All audited captures fail the new accuracy gate, as they should. On the small
IC(0) run a preceding solve also has true correction residual 1.028951802e-5,
slightly above tolerance despite its smaller recurrence residual. More
seriously, f32 position updates lose enough of the solved elastic correction
to produce a 0.20–0.47 applied residual across the audited operators. The
reported position-loss metric is a whole-vector L2 norm, not a per-voxel error:
about 1.52e-6 m at 32,768 REVs and 2.22e-5 m at one million for the last solve.

The prior timing tables remain execution-cost observations, **not validation
of accurate million-REV dynamics**. The audit distinguishes the error sources:
preconditioner acceleration alone cannot repair rounded mechanical positions.
It does not establish the sole cause of earlier tree explosions.

`src/coordinates.rs` provides a checked centimetre-cell-plus-f64-offset type,
relative subtraction before float conversion, and camera-relative rendering
projection. This is a migration primitive; scene mechanics still uses f32
positions. Next work must migrate all position mutation and force/contact
geometry consistently, then repeat the audit and conservation checks. Never
reconstruct fine state from its rounded rendering projection.

Audit artifacts: `scale32-residual-audit-{ic0,mg}-20261017.*` and
`scale1m-residual-audit-ic0-20261017.*`. Timing with the audit enabled includes
extra operator evaluations and is not directly comparable with disabled timing.
The million-REV audited update took 37.164 s and had the same recorded mass,
breakage and kinetic energy as the unaudited IC(0) update.

Primary references: [PETSc true residual monitoring](https://petsc.org/release/manual/ksp/)
and [PhysX coordinate origin handling](https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/Simulation.html).

Latest verification after the audit and coordinate primitive: all 98 release
library checks and release all-target compilation pass. The final 32,768-REV
physical audit capture correctly exits 1, records last applied residual
0.2023379729 and whole-step worst residual 0.2028634928. Thus the diagnostic
and rejection gate pass their checks while the existing physical coordinates
still fail the numerical accuracy requirement. The canonical coordinate type
is not yet wired into scene integration. No performance acceptance or physical
world completion is claimed.

## Automatic rigid aggregation reference

The original free-fall capture accidentally retained ambient drag while
assuming isolated uniform gravity. The fixture now explicitly uses zero air
pressure as a labelled reference case; the interactive world retains air.
Progress lines and configurable body counts make the capture observable.
A 64-REV body over 68 updates produced zero automatic registrations. Strict
zero-force eligibility is too narrow for the existing finite-precision fine
trajectory. Its conditions have not been relaxed to force a speedup, and no
automatic-aggregation gain is claimed.

## Next architectural work

**Subsequent state:** active-scene coordinates have since migrated to f64 and
successful PCG convergence now checks the true fine residual. Both 32,768 and
1,000,000-REV audit runs now pass. See
`precise-scene-coordinates-2026-10-17.md` for the current measurements. The failed
f32 cases above remain historical evidence and must not be treated as current
accuracy results. The physical world and performance goal remain incomplete.

The dominant requirement is a deformable physical hierarchy with local fine
activation. Fine material and bond history must remain recoverable, while
coarse mass, momentum, strain response and internal energy evolve under the
same laws. Refine conservatively before contacts, fracture and unresolved
wave/thermal gradients. Connected sleeping/rigid components are a special case.

Also audit the independent true mechanical residual and the many contacts in
the nominally intact cube before accepting physical equivalence. Profile warm
updates separately from initial collision indexing and hierarchy construction.
Do not count reduced iterations, a cold setup improvement, or render FPS as
evidence that the world now supports millions of active REVs in real time.

Primary algorithm reference: [PETSc KSP/GAMG manual](https://petsc.org/release/manual/ksp/).
For elasticity, near-null-space modes and transfer quality affect robustness;
the prototype is not a replacement for a full elasticity-aware AMG library.

## Reproduction

Build `mechanics_scale_capture` in release. Set `VOXEL_SOLVER_BACKEND=cpu`,
`VOXEL_PROFILE_STEP=1`, and optionally `VOXEL_PROFILE_PCG_SOLVE=1`,
`VOXEL_PROFILE_MULTIGRID_SETUP=1`, `VOXEL_PROFILE_COLLISION=1`.
Run `mechanics_scale_capture.exe 100 1 ic0` and
`mechanics_scale_capture.exe 100 1 multigrid` sequentially.
The current code reproduces the bounded candidate variant. Historical
all-level-smoothed and unsmoothed results are retained in the named CSV/logs.

Artifacts: `scale1m-ic0-20261017.*`, `scale1m-multigrid-20261017.*`,
`scale1m-bounded-multigrid-20261017.*`, `scale32-*-20261017.*`,
`axe-multigrid-*-20261017.*`, `axe-smoothed-multigrid-*-20261017.*`.
