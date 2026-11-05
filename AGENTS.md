# Project instructions for agents

## Product goal

Build the general-purpose voxel world described in [`WORLD_PARADIGM.md`](WORLD_PARADIGM.md). Read that document before changing engine behavior. Treat it as the product contract, keep its unfinished requirements visible, and do not call the whole paradigm complete while major systems remain absent.

At the start of a task, inspect repository status and confirm the relevant project files are readable and the workspace is writable. Read the relevant Markdown project notes before implementation. Preserve existing user changes.

The current targets, explicitly revised by the user on 2026-11-05, are 60 FPS in the native application, 50 thousand active mechanical REVs with 10 ms of physical time computed in at most 5 ms, and 20–40 billion compressed inactive logical REVs. The user's current i5-14450HX / RTX 5060 Laptop / 16 GB system is the initial minimum performance reference. Windows must support NVIDIA, AMD and Intel; CUDA is an optional NVIDIA acceleration path, not a mandatory engine dependency. No production simulation slowdown or skipped time is accepted; approximation stays within the paradigm's 10% limit. Preserve the accuracy and energy rules in WORLD_PARADIGM.md. Label measurements by hardware, scene size, solver settings, and whether they measure the physical simulation, rendering, or a synthetic kernel. A CUDA operator microbenchmark is not a GPU simulation benchmark. The earlier 300–400k target and benchmarks remain historical evidence, not the current acceptance threshold.

## Engine design

- Implement reusable physical behavior in the engine (`src/`), with clear scene-level APIs and documented units. Demonstrations and capture binaries should configure and exercise engine APIs; they must not contain one-off physics rules that belong in the engine.
- Prefer discoverable, composable scene and world APIs so a user can create and configure a scene without editing solver internals or copying an experiment. Keep defaults sensible and validate invalid input at API boundaries.
- Keep SI units explicit. Physical mass, composition, and internal energy are state; temperature, phase fractions, and display values are derived from that state. Constraints such as kinematic fixation must remain distinct from mass.
- World bounds are engine behavior. A REV crossing configured bounds is removed at the fixed-step boundary; all indexed arrays, bonds, pending events, spatial caches, and connectivity data must be updated together. Record removed mass, momentum, and energy in the boundary ledger.
- Engine mutations that change topology or constraints must invalidate/rebuild dependent caches and cluster labels. Avoid exposing mutable implementation flags as the only public API.
- Keep the engine independent of any one demonstration scene. Add reusable APIs first, then use those APIs in scenes, probes, renderers, and capture tools.
- When a feature first appears in an experiment (for example, pinning a voxel), move its behavior into the engine and expose it through a clear public API. Do not leave the experiment with a private physics shortcut.
- Build world boundaries as a general scene feature: remove escaped REVs consistently and account for their mass, momentum, and energy. Keep this behavior active in normal fixed-step simulation, not only in capture code.
- Prefer complete engine capabilities over experiment-specific partial implementations. Update examples to demonstrate the public API so users can configure scenes without editing engine internals.

## Physical runs and visual output

- The primary native visual path for a physical scene is the ordinary editor: open its authored `.vxscene` document with `voxel-editor --scene <path>`. Do not turn experiments into a special test UI or make `voxel-test-viewer` part of the product flow. The viewer and `validation/run-visible-test.ps1` may be used as secondary developer diagnostics for a specific test/probe when they add useful evidence; label observer overhead and never invent or interpolate solver states. Do not start a browser or HTTP viewer for this native workflow.
- Save a GIF for physical captures requested as recordings. Keep CSV and MP4 where useful for diagnostics and seeking. Live images must come from actual solver states; distinguish snapshot frequency and observer overhead from physical computation and rendering FPS.
- Maintain one reproducible physical experiment and recording for each engine mechanic in [`MECHANICS_EXPERIMENTS.md`](MECHANICS_EXPERIMENTS.md). Use at least 176 real saved states and a 17.6-second GIF at 10 FPS, twice the earlier 88-frame / 8.8-second recording. Extend the actual physical run to obtain more states; never duplicate or interpolate frames to meet the duration.
- Keep the run's actual physics results in the recording. Do not synthesize trajectories or hide unstable behavior in a rendered result; report breakage, drift, and other failures beside the visual.
- If the user gives a delivery deadline, check local time (`Asia/Yekaterinburg`) during the work and prioritize a runnable, inspectable result before that time. Do not imply a target has been achieved without a measurement.
- When a developer explicitly uses the live viewer for diagnostics, expose only the run's actual scene states and results; checks without a physical scene must not be given an invented one. Verify receipt of the current run before describing that diagnostic view as visible. The ordinary authored scene workflow remains in the native editor.

## Long-running work

- Do not wait idle for a long build, capture, render, or benchmark. Start it in a session, then work on independent engine, documentation, or review tasks while it runs. Poll progress periodically and keep the user informed during ongoing work.
- Keep live captures usable while independent edits proceed. Avoid starting competing heavy CPU work that would distort a performance run.
- Give concise progress updates during extended work. When a new user message arrives, treat it as guidance on the active project unless it clearly replaces or cancels that work.

## Git and project reporting

- Check repository read/write access before editing. Preserve the user's existing work; inspect status before staging and never discard unrelated changes.
- Commit coherent milestones and push to `origin/main` when the user has asked for pushes. Verify the remote ref actually moved, then give the commit/link. If command-line push lacks credentials, try the available authorized GitHub route; never describe a local commit as pushed. Clearly report a permission/authentication blocker if no write route succeeds.
- Update [`PROGRESS_2026-10-12.md`](PROGRESS_2026-10-12.md) with measured results, exact settings, known limitations, and the next incomplete requirements. Keep the report factual and distinguish old results from new code.
