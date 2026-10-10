# Acceptance status

Current checkpoint after fixing knife contact loading and mixed wood fracture.

| Check | Result |
|---|---|
| Physics library | PASS, 17 unit tests |
| Contact/motion smoke tests | PASS, 4 tests |
| Tension, 10³ wood sample | PASS, measured parallel/perpendicular force ratio 39.458× |
| Compression, 10³ wood sample | PASS, measured parallel/perpendicular force ratio 32.539× |
| Knife strike onset, 12×12×30 REVs | PASS, 1,298 bonds broken and 1 connected piece after 30 ms at 6 m/s; fixture damage rate 15 |
| Directional wood fibers | PASS, 100/2.5 MPa tension (40×), 50/2 MPa compression (25×), and 12/3 MPa shear (4×) |
| Pitched axe strike | NOT PASSING; current 10° contact response ejects the tool within 5 ms |
| Full mechanical suite | INCOMPLETE; process exited 137 during bending, so wedge and bending were not rechecked |
| Window/GPU runtime | NOT MEASURED |

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
