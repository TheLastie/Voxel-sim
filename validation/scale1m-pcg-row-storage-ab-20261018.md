# PCG row storage and fused reductions A/B — 2026-10-18

## Scope

Compared the committed full-row PCG layout against a candidate that stores each
3×3 constitutive block once per physical bond, keeps compact neighbor/edge IDs
per active row, and fuses the sparse matvec with `p·Ap` plus the residual update
with its norm. The fused path is enabled by default; setting
`VOXEL_PCG_FUSED_REDUCTIONS=0` restores the unfused reduction sequence.

The bond tensor is explicitly symmetrized at construction to remove floating
point roundoff asymmetry before the reverse row uses the transpose of the
stored block. The 1 cm REV state and 3×3×3 thermal state are unchanged.

## Reproduction and settings

- Release build, CPU backend, Intel Core i5-14450HX.
- `mechanics_scale_capture 100 1 ic0`.
- 1,000,000 REV total; 990,000 active; `dt=1 ms`; 4 substeps.
- IC(0), relative residual tolerance `1e-5`.
- A/B order: control A1, candidate B1, candidate B2, control A2.
- Wall time is the complete synthetic mechanics step, not FPS and not a full
  game-world measurement.

| Run | Layout | Fusion | Wall ms | Residual | Broken | Mass kg | Kinetic J |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | full rows | off | 20,378.616 | 7.137991361e-6 | 0 | 7.869998924e3 | 6.772421827e-6 |
| B1 | shared edge blocks | on | 19,736.321 | 7.137991361e-6 | 0 | 7.869998924e3 | 6.772421827e-6 |
| B2 | shared edge blocks | on | 19,974.456 | 7.137991361e-6 | 0 | 7.869998924e3 | 6.772421827e-6 |
| A2 | full rows | off | 20,443.948 | 7.137991361e-6 | 0 | 7.869998924e3 | 6.772421827e-6 |

Control mean: 20,411.282 ms. Candidate mean: 19,855.388 ms, a 1.0280×
speedup (2.72% less elapsed time). Logged physical outputs match at CSV
precision. The candidate by itself without fused reductions was only about
1.4% faster in two pairs and the stage profiles were indistinguishable within
run noise; most repeatable measured gain is from reduction fusion.

An independent f64 audit of the final candidate measured true residuals
7.0111e-6, 8.5726e-6, 7.3975e-6, and 7.1380e-6 across the four PCG solves.
The largest applied-coordinate residual was 8.5726e-6; position rounding L2
was 3.42e-14 m. This verifies the linear solve against the configured
tolerance, not long-run energy conservation. The synthetic fixture has no
complete energy ledger.

## Memory estimate and limits

At one million active REVs, `PcgRow` shrinks from 240 bytes to 52 bytes per
row. Storing one 36-byte 3×3 f32 block per undirected bond instead of two
blocks per bond is estimated to reduce the CPU sparse-operator arrays by about
80 MB for this grid. This is a layout estimate, not a process RSS measurement.
The CUDA upload still expands the blocks to the existing per-row GPU layout.

The full step still takes about 19.9 seconds for 1 ms of simulated time. This
does not meet real-time goals or the requested 50× improvement. The candidate
remains local and uncommitted pending more substantial end-to-end gains.

Raw logs: `sharededge-A1/A2`, `sharededge-B1/B2`, and
`fusededge-A1/A2/B1/B2` under this directory.
