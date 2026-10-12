# Longer knife diagnostic — 2026-10-11

Command: `RAYON_NUM_THREADS=1 target/release/knife_capture validation/knife-long.csv 6 2100 20 0 1 0`.

4320 wood REV, 840 rigid blade REV, vertical fibers; dt=0.001 s, four substeps, 120 PCG iterations, two corotation passes, zero gravity, four fixed bottom corners. Existing damage rate 15/s and release fraction 0.005. The long isolated-contact fixture disables gravity; it must not be treated as an extension of the earlier gravity-enabled 20 ms run with identical parameters. One second of fully computed approach and 1.1 seconds of computed reaction; no repeated static snapshots used to extend physical time.

Physical duration: 2.100 s. Wall duration including sparse trace writing: 457.073 s, approximately 217.7 times slower than real time. 173 full snapshots; every snapshot contains 4320 wood nodes, exact blade quaternion, 840 exact local blade centers, explicit broken bond endpoints and elastic energy. Source and executable hashes are in `knife-long-source-hashes.json`. Raw trace SHA256: `3fbf54b1ecd4ea0a77944476dbae3e52923681c5895126b45743ea185156fa66`.

Result: 1263 broken bonds and 16 components. Their sizes are [4296, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]. Thus 4296/4320 REV remain in the large trunk, and 24 REV form small fragments; this is not a completed trunk cut. The maximum recorded node speed is 6.127361 m/s; end tool speed is 0.239644 m/s.

Energy diagnosis: initial tool KE 118.994388 J. End tool KE 0.209511 J, wood KE 0.000283 J, elastic energy 0.186282 J. KE + elastic + damage/plastic heat + surface work + stored plastic + pending release sums to 73.364573 J at the end; 45.629825 J are not accounted for by this budget. Its peak is 185.193711 J, above the initial 118.994398 J. Gravity is off and constraints are stationary; the observed budget therefore shows both temporary energy creation and later unresolved loss. Contact friction is not yet included as a complete thermal channel. Do not call this result physically validated.

Rendering: exact saved tool poses rotate both centers and cube vertices. Red separations use actual saved broken neighbor IDs; occluded links are hidden under the blade silhouette. Scene REV remain the solver's translated axis-aligned volumes. The close view has a fixed camera; an overview strip shows the full metre-scale X approach. Contact replay is slowed 20 times over the physical interval 0.96–1.24 s, explicitly labelled. The remainder plays at the physical clock. GIF uses rounded durations; MP4 timestamps are verified after encoding. Playback and calculation times are separate from physical time.

The only code changes for this run concern capture and rendering, not the mechanical solver or material strength. The 27/5 checks refer to the prior verified mechanics version; the capture binary was rebuilt for the new CLI and trace format.
