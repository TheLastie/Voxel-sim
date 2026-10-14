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

An earlier saved capture had 1,438 broken bonds at 30 ms and 1,461 at 60 ms,
with one connected piece at both points. The existing video contains 61
consecutive solver states (0–60 ms); the 30 ms still shows the developing cut.

The capture fixture initially placed the blade's leading-cell center on the
outer wood-cell center, so contact depenetration occurred before the strike.
The fixture now places those centers one REV apart. At 6 m/s the 60 ms capture
shows 1,328 broken bonds and a partial notch, but still one connected wood
component. This is a geometry correction, not evidence of complete severing;
the collision and fracture response still need work before a full-cut video is
presented.

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

## Guided severing regression — 2026-10-14

The old `knife_cut` acceptance check stopped at 30 ms and explicitly passed
with one connected wood component. It has been replaced by a stricter force-
driven test using the public rigid-tool guide and external-force APIs. A 20 cm
beveled iron blade advances along X under 200 N through a 12×12×30 REV wood
coupon; the four bottom corner REVs are fixed, fibers point along Z, gravity is
off, `dt=1 ms`, four substeps, 120 solver iterations and damage rate 15/s. The
test stops at the first state with two fragments each at least 10% of the
coupon, or after 150 ms.

The original 6 m/s setting produced 5,373 broken bonds, 260 clusters and
largest fragments of 2,142 and 1,669 REV at 139 ms. Reducing initial speed to
5.8 m/s produced a through-cut at 136 ms with 4,239 broken bonds, 82 clusters
and largest pieces of 2,582 and 1,444 REV. The two largest pieces contain
4,026/4,320 REV (93.2%); the remaining 294 REV are debris. This improves the
first cut state, but does not establish a stable post-cut result.
The guided, force-driven cutter is not a free axe strike, and the result does
not validate the contact/fracture energy budget or the post-cut fall. A full
CSV/GIF/MP4 capture remains outstanding.

### Dynamic follow-through pilot — 2026-10-14

A 0.5 s, 500-step pilot used the same 6 m/s, 200 N guided fixture and applied
an opposing 2 kN actuator force once the two-large-fragment gate passed at
0.162 s. The 141 saved solver states include the unfiltered post-cut response.
At severing it had 2,938 broken bonds and 233 clusters (largest pieces 2,189
and 1,885 REV). The brake reduced blade speed from 5.592 m/s to 0.126 m/s by
frame 180, but that frame already had 8,066 broken bonds and 622 clusters. At
0.5 s the sample had 11,462 broken bonds and 3,686 clusters; peak REV speed was
16.531 m/s. The brake therefore limits tool follow-through but does not cure
the excessive fracture cascade. The raw diagnostic is
`validation/guided-knife-braked-pilot.csv`; it is not final cutting footage.

### Tested guided-drive threshold

The same coupon was tested with fixed feed force as well as the initial speed.
Results below are at 150 ms or at the first two-large-fragment gate. They show
a sharp threshold, so the faster case is retained as an explicit diagnostic,
not described as a calibrated axe strike.

| Initial speed | Fracture energy fraction | Result |
|---:|---:|---|
| 0 m/s | 0.005 | 0 breaks; 1 component |
| 3 m/s | 0.005 | 0 breaks; 1 component |
| 4.5 m/s | 0.005, 200 N | 1,173 breaks; largest component 4,289 REV |
| 4.5 m/s | 0.005, 300 N | 3,629 breaks; largest component 4,118 REV |
| 5.5 m/s | 0.005, 200 N | 1,971 breaks; largest component 4,260 REV |
| 5.5 m/s | 0.05, 200 N | 1,950 breaks; largest component 4,221 REV |
| 6 m/s | 0.005, 200 N | Gate passes at 139 ms; major components 2,142 and 1,669 REV |
| 5.8 m/s | 0.005, 200 N | Gate passes at 136 ms; major components 2,582 and 1,444 REV |

This sweep does not establish a material calibration. The successful setting
still creates an excessive fracture cascade after the first complete split.

### Long guided capture status — 2026-10-14

The diagnostic run reached frame 6,800 (6.8 s) before it was stopped after
post-cut instability was established. By frame 6,400 (6.4 s), the actuator
brake had stopped feed motion, but damage had cascaded to 11,706 broken bonds
and 3,930 clusters; peak REV speed was 16.53 m/s. The partial CSV remains
served by the dedicated knife viewer on port 8850. This is a physical failure
of post-cut stability; the requested complete, stable tree chop and its final
GIF/MP4 are not validated.

### Reduced-speed post-cut pilot — 2026-10-14

A 5.8 m/s, 200 N pilot confirmed the acceptance cut at frame 136 (0.136 s),
with 4,239 broken bonds and 82 clusters. The actuator brake then brought the
tool almost to rest, but at frame 370 (0.370 s) the coupon had 11,019 broken
bonds, 3,243 clusters and a 21.26 m/s peak REV speed. The run was stopped at
that point because post-cut collapse was already clear. The partial real-state
CSV is `validation/guided-knife-5_8-pilot.csv`; the dedicated live viewer on
port 8850 currently serves it. This lower initial speed improves the cut
threshold but does not fix post-cut stability.
