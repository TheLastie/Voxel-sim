# Voxel Physics Lab

An experimental Rust voxel mechanics sandbox. Matter is represented as 1 cm
representative volumes joined by directional bonds. The project explores how
contact, material orientation, deformation, and bond failure can produce cutting
and fracture without tool-specific damage scripts. It is a prototype, not a
validated continuum simulator.

## Current results

- Wood REVs carry a local grain orientation. For vertical grain (`Z`), the
  example material table sets bond limits to 100 MPa tension, 50 MPa compression
  and 12 MPa shear along the grain; across the grain the limits are 2.5 MPa,
  2 MPa and 3 MPa. The corresponding directional stiffness uses the same grain
  axis. These are illustrative calibration values.
- Displacement-controlled 10³-REV tests measure parallel/transverse force
  ratios of 39.458 in tension and 32.539 in compression. The bond regression
  checks real vertical and horizontal neighbors, including shear limits.
- The voxelized blade has a beveled one-REV edge. Its collision impulse and
  friction load the same bond stress law used by other materials.
- An experimental strike-pitch parameter is available in the capture tool;
  nonzero-pitch collision stability still needs validation.

### Demos

![Wood strike preview](validation/knife-cut-demo.gif)

- [4.54-second strike video](validation/knife-cut-demo.mp4), with a 1-second
  lead-in and 1-second outro. It shows the dense impact sequence and 0.25 seconds
  of simulated post-impact motion; the last state is held for the rest of the outro.
- [Cut-onset still](validation/knife-cut-final-frame.png)
- Detailed results: [validation status](validation/STATUS.md) and
  [knife diagnosis](validation/knife-diagnosis.md).

## Build and run

Requires Rust 2021 and Cargo. The desktop frontend uses optional `wgpu` and
`winit` dependencies.

```sh
cargo test --no-default-features --lib
cargo test --no-default-features --bin physics_smoke_tests
cargo run --release --no-default-features --bin mechanical_tests -- tension
cargo run --release --no-default-features --bin mechanical_tests -- compression
cargo run --release --no-default-features --bin mechanical_tests -- knife
cargo run --release --bin voxel-physics-lab
```

The full mechanical sequence includes a bending case that can exceed memory in
constrained environments; run fixtures individually to inspect each result.

## Repository map

- `src/scene.rs`, `src/voxel.rs`, `src/material.rs`: REV state, composition,
  microstructure orientation, and material table.
- `src/bond.rs`, `src/fracture.rs`: six-neighbor bond state and mixed-mode
  failure, plasticity, fatigue, and energy accounting.
- `src/solver.rs`, `src/collision.rs`, `src/tool.rs`: coupled mechanics,
  non-penetration contacts, and voxelized rigid tools.
- `src/experiments.rs`, `src/bin/`: reproducible mechanical and contact tests.
- `scenes/wood_chop.ron`: dimensions, grain axis, gravity, and solver settings.
- `validation/`: test notes, captures, and review media.

## Limits

The solver simulates a reduced active region. The material model uses a small
illustrative table; wood species, moisture, fracture-energy calibration,
independent fragment collisions, and large-scene real-time performance still
need validation. Long strikes can propagate damage too broadly. See
`validation/STATUS.md` for the latest verified checkpoint.
