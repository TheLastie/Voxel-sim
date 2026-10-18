# Same-substep phase cache reuse: A/B, 2026-10-18

## Change

`solve_coupled` evaluates each REV's thermodynamic `PhaseState` while assembling
the mechanics operator. The fracture pass in the same physical substep used to
evaluate that state again solely to derive solid fraction and temperature-based
strength scale. The solver now derives and stores both strength inputs with the
existing assembly phase cache and reuses them for fracture. Thermal energy is
still updated after fracture/contact/damping and the next substep recomputes
the phase state. Bond order, constitutive equations, matrix, PCG settings, and
residual audit are unchanged.

The exact prior path is available for A/B reproduction with
`VOXEL_REUSE_PHASE_STRENGTHS=0`; default is enabled.

## Measurement

Same release binary, CPU `i5-14450HX`, `mechanics_scale_capture 100 1 ic0`:
1,000,000 total REVs, 990,000 active, 1 ms fixed timestep, four substeps,
IC(0), relative residual tolerance `1e-5`, independent f64 true-residual audit.
Runs were alternating, `A = cache disabled`, `B = cache enabled`.

| Pair | A total (ms) | B total (ms) | A fracture (ms) | B fracture (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20,413.344 | 20,372.786 | 1,152.23 | 1,015.08 |
| 2 | 21,014.639 | 20,765.595 | 1,067.24 | 1,035.28 |
| 3 | 20,755.280 | 20,423.799 | 1,116.05 | 1,068.76 |

Mean total time: 20,727.754 → 20,520.727 ms (1.0101×, 1.00% less time).
Mean fracture phase: 1,111.84 → 1,039.71 ms (6.49% less time). This is a
small exact optimization; the total benefit is close to run-to-run timing
variation and does not materially close the 50× performance goal.

All four per-substep independent f64 audits matched: maximum true/applied
residual `8.572592852e-6`, below `1e-5`; 259 PCG iterations; mass
`7.869998924e3 kg`; kinetic energy `6.772421827e-6 J`; zero broken bonds.

## Physical regressions

Both modes passed `cargo test --release --offline --test contact_energy
--test fatigue_topology`: 1 contact-energy test and 3 fatigue/release/fracture
tests. These cover momentum and mass plus conversion of restitution energy into
internal energy, and fracture release accounting. They are narrow regression
checks, not a full long-horizon energy ledger for the million-REV scale fixture.

## Raw captures

- `phase-cache-A1-20261018.log`, `phase-cache-B1-20261018.log`
- `phase-cache-A2-20261018.log`, `phase-cache-B2-20261018.log`
- `phase-cache-A3-20261018.log`, `phase-cache-B3-20261018.log`
