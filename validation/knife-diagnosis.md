# Knife contact and wood fracture diagnosis

## Root causes

The blade was a flat two-cell slab. Its leading face spread the impact over a
wide band, so local wood bonds did not reach their failure limit. Making the
whole blade one cell thick fixed the width but removed the bevel and left a
flat plate. The blade is now a voxel wedge: a five-REV spine tapers to a
one-REV edge.

Tool contact impulses were also applied after the bond solve. The new velocity
did not create strain until the following substep, where the implicit elastic
solve damped most of the short impact. The impulse now contributes its
half-substep displacement before the structural solve, while its velocity is
still kept separate from geometric depenetration.

Finally, mixed-mode bonds only accumulated damage after plastic strain had
started. Exceeding any strength limit now starts damage accumulation for mixed
materials; yielding still controls plastic deformation. Wood's shear limit is
directional as well as its tension and compression limits.

## Verified results

The table uses strong longitudinal and weaker transverse wood properties. The
reduced knife fixture is calibrated separately at 6 m/s and damage rate 15;
the interactive scene still uses its scene-level starting speed and damage rate.

| Fixture | Result |
|---|---|
| Mixed wood shear below yield | PASS; damage accumulates without plastic strain |
| Knife strike, 12×12×30 REVs, first 30 ms | PASS; 1,298 broken bonds, 1 connected piece in acceptance test |
| Tension, 10³ wood sample | PASS; parallel/perpendicular force ratio 39.458× |
| Compression, 10³ wood sample | PASS; parallel/perpendicular force ratio 32.539× |
| Physics smoke tests | PASS; 4/4 |

The latest saved capture has 1,438 broken bonds at 30 ms and 1,461 at 60 ms,
with one connected piece at both points. The video contains 61 consecutive
solver states (0–60 ms); the 30 ms still shows the developing cut.

## Fibers and strike angle

Wood grain is a per-REV microstructure direction, not a second kind of voxel.
`Microstructure.orientation` stores a local quaternion whose +Z axis is the
grain direction. For each bond, the solver transforms that axis to world space
and projects it onto the bond axis. The directional tensile limit interpolates
between 100 MPa parallel to the grain and 2.5 MPa across it. Compression is
50/2 MPa and shear is 12/3 MPa. The solver also uses the grain direction for
stiffness. A unit test finds actual vertical and horizontal neighbors in the
REV graph and checks all three strength modes plus axial stiffness.

The capture executable accepts an attack pitch in degrees after the stride
argument. A 10° pitched test currently ejects the tool from the specimen by
5 ms; its measured bond damage is therefore not a valid comparison with a
straight strike. Keep attack angle separate from bevel geometry and fix the
rotated contact manifold before using this test to tune chopping.

The current image and videos use saved solver states and draw each blade REV
individually:

- `knife-cut-final-frame.png`
- `knife-cut-demo.mp4`
- `knife-cut-demo.gif`
