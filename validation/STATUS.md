# Acceptance status

## 2026-10-11: local contacts and derived damage colors

| Check | Result |
|---|---|
| Physics library | PASS, 30 unit tests |
| Contact/motion smoke tests | PASS, 5 checks |
| Rigidly rotated intact 3³ sample | PASS, no internal AABB projection |
| Crack and nonlocal self-contact exclusions | PASS, broken face and remote pairs remain eligible for contact |
| Damage display | PASS, intact outside faces stay intact; both endpoints show partial damage and broken bonds |
| Desktop build | PASS, `cargo check --offline`; GPU/window runtime not measured |
| Tension | PASS, parallel/perpendicular force ratio 39.458× |
| Compression | PASS, parallel/perpendicular force ratio 25.877× |
| Bending | PASS, 9/9 bonds cut in one cross-section at 3.9 kN; residual 8.178×10⁻⁶ |
| 4 m/s pilot, 0.5 s after contact | 812 broken bonds, two pieces of 4318 and 2 REV |
| 4 m/s, 6 s capture, 12 N/s to 60 N | 894 broken bonds; two pieces of 4319 and 1 REV; maximum two components throughout the saved trace |
| Encoded capture | PASS, 17.10 s MP4 at 1440×900 and 17.08 s GIF; all frames decoded |
| Complete blade severing | NOT VERIFIED; a band of damaged bonds is not a complete cut |
| Energy balance | NOT VALIDATED; contact dissipation and projection/integration work need a complete ledger |

The force-ramp recording and final diagnostics are in `knife-local-run.txt`
and `knife-local-measurements.json`. There is no fragment-count clamp, material
strength reduction, or prescribed fracture plane. The capture keeps the
material inputs and fracture parameters used by the earlier recordings.

## Earlier checkpoint

Current checkpoint after fixing knife contact loading and mixed wood fracture.

| Check | Result |
|---|---|
| Physics library | PASS, 21 unit tests |
| Contact/motion smoke tests | PASS, 5 tests |
| Constant force on a free REV | PASS, 10 m/s² for 100 ms gives 1.000 m/s (error 3.2×10⁻⁵ m/s) |
| Velocity-set head-on collision | PASS, 20 m/s inputs; no overlap and momentum conserved |
| Tension, 10³ wood sample | PASS, measured parallel/perpendicular force ratio 39.458× |
| Compression, 10³ wood sample | PASS, measured parallel/perpendicular force ratio 32.539× |
| Knife strike, corrected initial clearance | 1,328 bonds broken, 1 connected piece, and 3.975 m/s tool speed after 60 ms at 6 m/s |
| Knife onset regression with adaptive corotation | PASS, 1,105 bonds broken and 1 connected wood piece after 30 ms |
| Complete knife severing | NOT VERIFIED; the corrected strike still produces a partial cut only |
| Directional wood fibers | PASS, 100/2.5 MPa tension (40×), 50/2 MPa compression (25×), and 12/3 MPa shear (4×) |
| Pitched axe strike | NOT PASSING; current 10° contact response ejects the tool within 5 ms |
| Full mechanical suite | INCOMPLETE; process exited 137 during bending, so wedge and bending were not rechecked |
| Window/GPU runtime | NOT MEASURED |
| Full-scene real-time target | NOT PASSING; release headless step on 299,440 voxels took 16.064 s and 16.698 s in two runs, with PCG residual 3.36×10⁻¹ |

The knife uses a voxel bevel that tapers from a five-REV spine to a one-REV
edge. Contact impulse displacement is integrated in the same substep as bond
stress. Mixed-mode wood bonds now accumulate damage when any strength limit is
exceeded, even when the plastic yield stress is higher than the shear limit.

Each wood REV stores its fiber axis in `Microstructure.orientation`; there are
no separate fiber voxels. The six-neighbor bond derives axial stiffness and
tensile, compressive and shear limits from the projection of that axis into the
bond direction. The regression test checks actual vertical and horizontal
neighboring bonds, including their normal and shear limits.

`knife_capture <csv> <speed> <frames> <stride> <angle-degrees>` accepts a
pitched strike, but nonzero pitch is not validated. In the current 10° test,
the contact response ejects the tool by frame 5; this needs a separate collision
fix before angled strikes can be presented as a result. Longer fracture
propagation also needs calibration.

Per-voxel force, acceleration, impulse, and velocity inputs now use the general
solver APIs and have mass-response and momentum tests. Rigid voxel groups also
accept applied force and torque. Thermally activated
adhesion, welding, and alloy formation are not implemented; a metal collision
does not fuse voxels in this checkpoint.

Corotational solves can stop before their configured maximum only when both
the orientation change and the PCG residual meet their independent tolerances.
The solver now reuses its large work buffers across substeps and corotation
corrections. Two full-scene runs took 16.1–16.7 s, down from the 20.374 s
baseline, with the same 4,816 broken bonds, 156 clusters, and 3.36×10⁻¹ PCG
residual. This reduces allocation churn without changing the computed result.
The linear solver remains far from its residual target after 40 iterations;
multilevel preconditioning is still the major performance and convergence
task. The 10 FPS target has not been met.
