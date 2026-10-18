# Million-REV collision-bin A/B — 2026-10-18

## Change

The swept collision broadphase bin scale defaults to `1.05` instead of `2.0`. This changes only the spatial hash bin width used to generate broadphase candidates; swept bounds, narrow-phase contact tests, and the mechanical solve remain the same. `VOXEL_COLLISION_BIN_SCALE` still allows controlled experiments in the range 0.25–8.

The million-REV fixture produced the same 88,804 broadphase candidates and the same contact counts with both settings. The selected scale reduced the cold first broadphase pass substantially. Scale 1.0 was rejected because it generated about 47 million candidate pairs and made projection much slower.

## Clean, unprofiled measurements

Same release binary and fixture: 1,000,000 total REVs, 990,000 active, 1 ms physical step, four substeps, CPU IC(0), tolerance 1e-5. Profiling environment variables were removed for these runs.

| Pair/order | Bin scale | Step wall time |
| --- | ---: | ---: |
| A1 | 2.00 | 32,332.013 ms |
| B1 | 1.05 | 20,099.780 ms |
| B2 | 1.05 | 20,046.721 ms |
| A2 | 2.00 | 32,300.566 ms |

Pair means: control 32,316.290 ms, candidate 20,073.251 ms; speed ratio 1.610x (37.9% less elapsed time). The two settings reported the same PCG residual (`7.137991361e-6`), zero broken REVs, mass (`7.869998924e3 kg`), and final kinetic energy (`6.772421827e-6 J`). The profile runs also had the same 259 PCG iterations and broadphase candidate/contact counts.

All 18 `collision::` library tests pass with the selected scale. This scale fixture does not expose a complete total-energy ledger, so the measured equality of physical outputs and unchanged narrow phase are evidence of equivalence for this fixture, not a substitute for the full energy audit on an impact/thermal scene.

Raw clean runs: `scale1m-collision-ab-A1-20261018.log`, `scale1m-collision-ab-B1-20261018.log`, `scale1m-collision-ab-B2-20261018.log`, `scale1m-collision-ab-A2-20261018.log`. Profile sweeps: `scale1m-collision-bins1_05-profile-20261018.log`, `scale1m-collision-bins1_1-profile-20261018.log`, `scale1m-collision-bins1_25-profile-20261018.log`, and `scale1m-collision-bins1_5-profile-20261018.log`.
