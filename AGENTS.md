# Project instructions for agents

## Product goal

Build the general-purpose voxel world described in [`WORLD_PARADIGM.md`](WORLD_PARADIGM.md). Read that document before changing engine behavior. Treat it as the product contract, keep its unfinished requirements visible, and do not call the whole paradigm complete while major systems remain absent.

The long-term performance targets are 30 FPS for the intended CPU workload and 60 FPS for the intended GPU workload. Label measurements by hardware, scene size, solver settings, and whether they measure the physical simulation, rendering, or a synthetic kernel. A CUDA operator microbenchmark is not a GPU simulation benchmark.

## Engine design

- Implement reusable physical behavior in the engine (`src/`), with clear scene-level APIs and documented units. Demonstrations and capture binaries should configure and exercise engine APIs; they must not contain one-off physics rules that belong in the engine.
- Prefer discoverable, composable scene and world APIs so a user can create and configure a scene without editing solver internals or copying an experiment. Keep defaults sensible and validate invalid input at API boundaries.
- Keep SI units explicit. Physical mass, composition, and internal energy are state; temperature, phase fractions, and display values are derived from that state. Constraints such as kinematic fixation must remain distinct from mass.
- World bounds are engine behavior. A REV crossing configured bounds is removed at the fixed-step boundary; all indexed arrays, bonds, pending events, spatial caches, and connectivity data must be updated together. Record removed mass, momentum, and energy in the boundary ledger.
- Engine mutations that change topology or constraints must invalidate/rebuild dependent caches and cluster labels. Avoid exposing mutable implementation flags as the only public API.
- Keep the engine independent of any one demonstration scene. Add reusable APIs first, then use those APIs in scenes, probes, renderers, and capture tools.

## Physical runs and visual output

- Before starting a model run, confirm filesystem access and start or verify the live viewer. Stream the run's CSV to `http://127.0.0.1:8765` so the user can see it while it runs.
- Save a GIF for every physical capture and expose it from the viewer. Keep the CSV and MP4 where useful for diagnostics and seeking. The browser page should show the GIF and the live solver frame during a run.
- Keep the run's actual physics results in the recording. Do not synthesize trajectories or hide unstable behavior in a rendered result; report breakage, drift, and other failures beside the visual.
- If the user gives a delivery deadline, check local time (`Asia/Yekaterinburg`) during the work and prioritize a runnable, inspectable result before that time. Do not imply a target has been achieved without a measurement.

## Long-running work

- Do not wait idle for a long build, capture, render, or benchmark. Start it in a session, then work on independent engine, documentation, or review tasks while it runs. Poll progress periodically and keep the user informed during ongoing work.
- Keep live captures usable while independent edits proceed. Avoid starting competing heavy CPU work that would distort a performance run.

## Git and project reporting

- Check repository read/write access before editing. Preserve the user's existing work; inspect status before staging and never discard unrelated changes.
- Commit coherent milestones and push to `origin/main` when the user has asked for pushes. Verify the remote ref actually moved, then give the commit/link. If command-line push lacks credentials, try the available authorized GitHub route; never describe a local commit as pushed. Clearly report a permission/authentication blocker if no write route succeeds.
- Update [`PROGRESS_2026-10-12.md`](PROGRESS_2026-10-12.md) with measured results, exact settings, known limitations, and the next incomplete requirements. Keep the report factual and distinguish old results from new code.
