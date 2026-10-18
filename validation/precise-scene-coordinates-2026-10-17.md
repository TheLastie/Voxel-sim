# Precise physical coordinates and verified solves — 2026-10-17

## Result and scope

Active scene mechanics now retains positions, reference positions, previous
positions, fixed-axis targets, tool poses, guide anchors and world bounds in
double precision. Lattice spawns are constructed directly from integer cell
indices and the double-precision 0.01 m spacing. Fine spring rest geometry and
bond-formation offsets also retain double precision. Bond deformation is
subtracted before converting the small relative result to constitutive f32
vectors. Rendering explicitly projects coordinates to f32; playback snapshots
retain double coordinates and do not restore rounded render data into physics.

This is a migration of the active scene to local f64 coordinates. It is not a
claim that the entire world now uses the prepared integer-cell `CellPosition`
representation. That type remains available for a future global cell/origin
layer. World generation, global activation, physical coarsening and continuous
streaming remain unfinished. Force/material/velocity arrays and PCG vectors
still use f32; independent residuals are accumulated in f64.

## Verified convergence

Successful CPU PCG convergence now requires an independent `b-Ax` evaluation
against the assembled fine operator. If the recurrence residual is below the
configured tolerance but the actual residual is not, the residual is replaced
and the search direction restarted. Iteration-cap exhaustion reports the
actual residual rather than a falsely converged recurrence value.

CUDA results are checked against the same original host operator before
position updates. A rejected GPU correction becomes the initial iterate for
CPU refinement; it is not applied twice. This path compiles but a new GPU
physical benchmark has not been run in this change. A finite iteration cap can
still leave a solve unconverged; captures with the audit enabled reject such
results. This is not a claim that every possible application scene converges.

The optional audit additionally checks the correction actually retained after
updating scene coordinates and tracks the worst result over all substeps and
corotation corrections. The configured tolerance remains 1e-5 in the scale
captures; it has not been relaxed.

## Physical measurements

Recorded host: Intel Core i5-14450HX, 16,868,958,208 bytes RAM, Windows,
release CPU backend. The scale fixture is an iron cube under ordinary gravity
and atmosphere. Its bottom layer is kinematic. `Scene::from_voxels` does not
create bonds across the fixed/dynamic interface, so this is a **contact-supported
cube**, not an elastically clamped cube. This was also true in the earlier
timing runs. Each run below contains one cold 1 ms update with four 0.25 ms
substeps. Reported times include the enabled independent audit.

| Measured quantity | 32,768 REVs, earlier f32 | 32,768 REVs, current | 1,000,000 REVs, earlier f32 | 1,000,000 REVs, current |
| --- | ---: | ---: | ---: | ---: |
| Final applied-coordinate residual | 0.2023379729 | 9.644359916e-6 | 0.4365206060 | 7.019753266e-6 |
| Whole-step worst audited residual | 0.2028634928 | 9.644359944e-6 | not aggregated in that artifact | 7.645732160e-6 |
| Lost displacement, whole-vector L2 (m) | 1.518277776e-6 | 1.515939572e-15 | 2.221775971e-5 | 3.412890145e-14 |
| Cold update wall time | 801.802 ms | 854.972 ms | 37,164.423 ms | 39,870.545 ms |
| Broken bonds | 0 | 0 | 0 | 0 |
| Independent accuracy gate | failed | passed | failed | passed |

The last-million-REV correction residual is 7.019752872e-6; applying it no
longer produces the previous 0.44 residual. Mass remains 7,869.998924 kg.
The current kinetic energy is 6.773401544e-6 J, versus 3.334952596e-5 J in the
earlier inaccurate-coordinate run. This difference is not proof of full
physical equivalence or long-run energy conservation. Coordinate precision,
rest geometry and first-contact classification changed; accuracy and physical
convergence must be assessed rather than demanding identical old trajectories.

This change does **not** demonstrate an execution speedup. The current cold
million-REV update is about 7.3% slower than the earlier audited run. A 1 ms
step taking 39.9 seconds is far from real time. The 50x target and prior
performance commit gate are not met. Cold setup must not be confused with
steady-state throughput, and these are single runs rather than repeated A/B
statistics.

The 150 N mounted-axe run used the existing 10 ms / 40 substep configuration
for 10 updates (0.1 s physical time). It took 15.849 s and ended with 239 broken
bonds, one cluster, about 2.34514 J damage heat and 16.8530 J external work.
Earlier IC(0) data had 228 broken bonds. Long-term post-impact stability and
equivalence are not established by this short capture.

## Regression repairs and verification

The coordinate migration exposed a first-contact classification error:
entering the contact shell from a genuine preceding gap was always marked as
resting contact by `touching_normal`'s caller. The caller now uses preceding
separation consistently with swept/overlapping contacts. Existing hot-impact
bonding and persistent-contact heat checks pass; the material law does not
branch on tool or experiment identity.

Formation offsets use f64 so a new hot adhesive bond can retain its actual
stress-free contact pose rather than add an impulse through rounded geometry.
The existing exact hot-connection position/energy check passes.

All 99 release library checks pass. A new two-REV elastic fixture creates a
real bond, pins one endpoint with the engine API, applies 0.01 N, and checks
the displacement against `F / (M/h^2 + K)`. The retained physical motion is
positive and within 0.5% of that analytic value while its f32 render projection
remains unchanged. All-target release compilation and capture builds pass.
The native release application also builds successfully; the completed log is
`precise-native-build-20261017.stderr.log` (70 seconds for the successful build).
The updated executable is `target/release/voxel-physics-lab.exe`. No native
window was running during the final physical captures. This build result is
not a visual runtime or gameplay acceptance test.

The isolated rigid-aggregation diagnostic still produces zero automatic
registrations. Its strict force-free criteria have not been relaxed; no
aggregation speedup is claimed.

## Reproduction and next work

Build the release `mechanics_scale_capture` target. Enable
`VOXEL_SOLVER_BACKEND=cpu`, `VOXEL_AUDIT_PCG_RESIDUAL=1`, and
`VOXEL_PROFILE_STEP=1`. Run `mechanics_scale_capture.exe 32 1 ic0` and
`mechanics_scale_capture.exe 100 1 ic0` sequentially. Current verified artifacts:

- `scale32-precise-verified-20261017.csv` / `.log`
- `scale1m-precise-verified-20261017.csv` / `.log`
- `axe-precise-verified-20261017.csv` / `.log`
- `aggregate-precise-coordinates-20261017.log` / `.stderr.log`

The earlier `scale32-precise-coordinates` run predates true-residual restart:
it removed position loss but correctly failed because a preceding solve had
true residual 1.05248e-5. Retain it as evidence of why the convergence guard
was needed.

Next: reduce active fine mechanical DOFs through a deformable physical
hierarchy, with conservative transfer of mass, linear/angular momentum and
energy, retained REV history, and refinement before contact, fracture and
unresolved waves. Use the corrected fine model as reference. Separately verify
space/time convergence, longer impact/coast energy balance, GPU behavior and
warm-update performance before accepting an acceleration claim.

## Million-REV CPU/GPU throughput check

The user clarified that the target must handle millions of centimeter REVs.
The paradigm's explicit million-REV target is 20–30 FPS on CPU; merely loading
one million REVs into the solver is not sufficient. On the Intel Core
i5-14450HX, the audited CPU cold step remains 39,870.545 ms for 1 ms of
simulation (four 0.25 ms substeps, 990,000 dynamic REVs, 1,000,000 total).
The release GPU trials used the RTX 5060 Laptop and the same physical fixture:

| Backend / experiment | Cold step wall | Verified outcome |
| --- | ---: | --- |
| CPU IC(0), reference | 39,870.545 ms | whole-step worst residual 7.646e-6; passed 1e-5 gate |
| CUDA, configured tolerance | 52,099.341 ms | first GPU solve true residual 1.083e-5; CPU refined it; 30.7% slower than CPU |
| CUDA, half internal tolerance | 49,898.907 ms | first GPU solve passed at 9.872e-6; later corotation solve had true residual 4.539e-4 and fell back; 25.1% slower than CPU |
| CUDA retained after CPU refinement | 53,921.708 ms | later GPU warm-start solve diverged (relative residual 1.384e3); CPU fallback; 35,800.79 ms solve stage |

The two tolerance/backend-retry changes were reverted because they did not
accelerate the complete step and exposed further warm-start convergence
problems. These are single cold-step experiments, not steady-state results.
No tolerance was relaxed and the accepted CPU accuracy gate was preserved.
Current evidence therefore confirms that the project can represent and solve
a million-REV fixture accurately, but it cannot yet process that scale at the
paradigm's intended rate. The high-level next step remains reducing active
fine mechanical degrees of freedom with deformable, conservative levels while
retaining every constituent REV's state, then refining before contact,
fracture, heat gradients, or unresolved waves.

## IC(0) dependency-level parallel trial — 2026-10-17

To test whether CPU PCG triangular substitutions could use all cores without
changing the matrix or tolerance, lower/upper rows were scheduled by dependency
depth and independent rows within a level were evaluated in parallel. The same
release million-REV fixture (1,000,000 total, 990,000 dynamic; 1 ms simulated,
four 0.25 ms substeps) took **89,313.582 ms** on CPU IC(0), versus the prior
serial result **39,870.545 ms**. PCG consumed 67,624.91 ms over 380 iterations.
The maximum independently audited residual remained 7.645732e-6 (inside 1e-5)
and no bonds broke, but throughput regressed by 2.24x. Frequent synchronization
between narrow dependency levels outweighed within-level parallel work. The
level scheduler was removed; the release capture binary rebuilt and
`git diff --check` passed. Raw capture: `scale1m-ic0-level-scheduled-20261017.csv`
and `.log`.

This rejects level-scheduled IC(0) for the million-cell case. The next strong
solver experiment is a structured geometric multigrid preconditioner on the
GPU, used only to accelerate convergence of PCG against the original fine
matrix. Every REV and its full material, thermal, and damage history stays
resident; the independently audited fine-system residual remains the acceptance
gate. Coarsening solver error modes must not erase physical cells or state.
