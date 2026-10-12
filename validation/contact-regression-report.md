# Contact regressions — 2026-10-11

Base: `1c6e5b6d4c543935fc6d8a0880a7d93a1940a9e6`. Branch: `fix/contact-regressions-2026-10-11`.
Exact modified source hashes: `contact-regression-source-hashes.json`.

## Changes

- Rigid tapered geometry now uses its occupied-volume center of mass. Local cells and reference position move together, preserving initial world geometry. Inertia is recomputed about that center.
- Tool impulse and its angular reaction use the same scene node. Closest OBB point replaces discontinuous corner support for returned contact geometry.
- Collision broadphase includes fixed/free pairs; only fixed/fixed pairs are excluded.
- Removed the invisible infinite ground plane. Ground support uses explicit fixed REV contacts.
- Restitution uses pre-projection start geometry and only the first velocity iteration; a resting pair receives no repeated restitution.
- Fracture energy uses the minimal-magnitude quadratic root continuous at zero energy, including approaching pairs. Release impulses are projected onto free axes; fully constrained budgets go to internal energy and damage heat.

No material parameters, fracture thresholds, or tool-specific cutting rules were added or changed.

## Verification

Rust 1.99.0, release CPU builds, headless. Local Rust toolchain was reconstructed because rustup requires unavailable `/proc/self/exe`. Environment:

```sh
CARGO_HOME="$PWD/../.toolchain/cargo" \
RUSTC="$PWD/../.toolchain/direct/bin/rustc" \
LD_LIBRARY_PATH="$PWD/../.toolchain/direct/lib" \
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' \
TMPDIR="$PWD/../.toolchain/tmp" \
../.toolchain/direct/bin/cargo test --no-default-features --lib
```

27 library tests passed. Regression coverage includes fixed-wall CCD, unbounded free fall, center of mass and unchanged geometry, off-center impulse conservation at the fixed projection reference, small approaching-pair releases, and fully constrained heat allocation. This angular-momentum test covers velocity impulses; geometric split correction is still an approximation.

```sh
RAYON_NUM_THREADS=1 target/release/physics_smoke_tests
RAYON_NUM_THREADS=1 target/release/mechanical_tests all
RAYON_NUM_THREADS=1 target/release/knife_capture validation/contact-regression-capture.csv 6 20 1 0
```

All five smoke tests pass. Mechanical tension, compression, and bending pass; wedge fails with 264 broken bonds and one wood component. Knife acceptance is not run. Mechanical wall time is 70.864 s. See corresponding raw logs. Mechanical acceptance stops at the first failed stage; do not infer that knife acceptance ran from this command.

## Short knife diagnostic — NOT acceptance footage

4320 wood REV, 840 rigid blade REV, dt 1 ms, 4 substeps, 120 PCG iterations, 2 corotation passes, speed 6 m/s, pitch 0, four bottom-corner constraints. The existing capture config uses its original fracture parameters and gravity; these were not recalibrated. It records 20 ms, with no one-second approach or follow-through, so no demonstration video is presented.

Initial tool KE: 118.9943876 J. Final wood KE: 0.2096278 J; final tool KE: 84.4973778 J. 1097 broken bonds, one component. Maximum node speed 5.95652 m/s. Surface work 20.59595 J, damage heat 41.41915 J, plastic heat 2.30742 J, stored plastic 0.25638 J. Their sum with KE is about 149.286 J; this is not a closed energy balance. Kinetic release is a transfer, not another independently stored energy term, and must not be added again. Elastic energy, gravitational potential, constraint reactions, and numerical losses need a complete ledger before attributing the discrepancy.

Wall time: 5.924 s including trace writing for 0.020 s simulated, about 296.2 times slower than real time in this capture. This is not an isolated solver benchmark, and does not meet the realtime goal.

## Remaining work

Corotation can inject energy into the strain model. Restore and verify implicit displacement before rotation fitting, then implement energy-consistent orthotropic frame updates. Couple contact to continuum compliance/effective mass instead of treating the impacted REV as an independent point mass. Verify the complete energy ledger and support reactions before any cutting claim. Fixed cubes and disconnected nodes still have the existing axis-aligned contact approximation. Six face bonds alone do not establish a converged continuum discretization; shear, bending, mesh/time convergence, and objectivity require separate validation. Thermodynamics, chemistry, deformable tools, liquids, adaptive aggregation, and the broader world specification are not implemented by this patch.
