# Voxel Editor MCP

`voxel-editor-mcp` is the editor's local, **stdio-only** MCP server. It opens no
MCP network listener. File tools edit saved project documents; separate
read-only telemetry tools query the running native engine and never change its
state. The server supports MCP
`2026-07-28` and the backward-compatible `2025-11-25` initialize handshake.

## Build and launch

From the repository root:

```powershell
cargo build --release --bin voxel-editor-mcp
```

Configure the MCP client with the executable and arguments in
[`host-config.example.json`](host-config.example.json), replacing the example
paths with the local checkout and project directory. The equivalent direct
launch is:

```powershell
target\release\voxel-editor-mcp.exe --project-root E:\path\to\voxel-project
```

The project root defaults to the process working directory when
`--project-root` is omitted. Files written by the voxelizer must target an
existing directory beneath that root. Parent directories are never created.

## Tools

- `list_documents` — find saved `.vxm` and `.vxscene` assets (bounded to 20,000
  files and 16 levels; skips symlinks, `.git`, `target`, and `.omx`).
- `get_runtime_status` — read runtime availability, session identity, and
  telemetry freshness.
- `read_runtime_metrics` — read a bounded sample batch after a sequence cursor.
  `after_sequence` defaults to `0`, `max_samples` to `16` (range 1–64), and
  `wait_ms` to `0` (range 0–10,000). Optional `topic` filters to `app` or
  `physics`. Responses identify the runtime session and mark sequence gaps.
- `get_runtime_objects` — read a snapshot of objects currently known to the
  native runtime.
- `inspect_document` — validate a model or scene and report its geometry,
  materials, groups, constraints, skeleton, weights, clips, keyframes, and scene
  references.
- `create_model`, `create_scene` — make valid native starter assets.
- `voxelize_mesh` — import OBJ / GLB as a shell or filled `.vxm` using the
  editor's mesh voxelizer.
- `export_model_obj` — export the exposed REV faces of a `.vxm` as a watertight
  OBJ mesh in metres.
- `edit_voxel_model` — add, move, duplicate, remove, or recolor a REV; stamp a
  block, sphere, cylinder, or frame. Shape bounds are inclusive and one action
  is limited to 100,000 candidate cells. Recolors preserve the REV's mass and
  internal energy.
- `edit_scene` — add, move, duplicate, or remove model instances in `.vxscene`.
- `edit_model_skeleton` — create a root/child bone hierarchy, clips and
  keyframes, generate automatic weights, or adjust an individual REV weight.
- `batch_actions` — run 1–64 existing MCP tools sequentially in one call.
  Every action name and argument schema is checked before any action runs;
  optional IDs must be unique. Results include per-action status and counts.
  `stop_on_error` defaults to `true`; set it to `false` to continue after an
  action error. Telemetry long polls share a 10-second total wait limit. A
  batch is not atomic: successful file writes remain if a later action fails.
  Batching does not add controls for changing the live editor or physics state;
  it only combines the saved-file tools and read-only telemetry queries listed
  here.
- `author_animation` — create a root, ordered child-bone hierarchy, named clip,
  quaternion keyframes, and automatic REV weights in one call. The source model
  stays unchanged and the result is written as a new validated `.vxm`.
- `author_procedural_gait` — infer a raised body and 2–16 leg contacts from a
  connected model, then bake an in-place IK walk in one call. The body stays
  rigid; each upper/lower leg has one coherent bone influence. Fixed bone
  lengths, a horizontal stance stroke, a smooth lifted return stroke, and
  staggered phases produce a closed wave gait. The response reports contact,
  interpolation, length and loop errors measured on the generated clip.
- `sample_animation` — evaluate interpolated local bone poses and global joint
  positions at a time in seconds.

Example MCP call for the supplied spider model. The real model has four legs
and one connected body; the vertical axis is detected automatically:

```json
{
  "name": "author_procedural_gait",
  "arguments": {
    "source_path": "mcp/examples/spider.vxm",
    "output_path": "mcp/examples/spider_walk_ik.vxm"
  }
}
```

Create a model, make a second copy with an edit, and inspect both results in one
sequential batch. The new model is written first; the edit reads that file and
writes to a separate path, preserving the existing no-overwrite rule:

```json
{
  "name": "batch_actions",
  "arguments": {
    "actions": [
      {"id": "starter", "name": "create_model", "arguments": {"output_path": "mcp/examples/batch.vxm"}},
      {"id": "edited", "name": "edit_voxel_model", "arguments": {"source_path": "mcp/examples/batch.vxm", "output_path": "mcp/examples/batch-edited.vxm", "operation": "stamp_shape", "shape": "block", "min_cell": [0, 1, 0], "max_cell": [1, 1, 0]}},
      {"id": "check", "name": "inspect_document", "arguments": {"path": "mcp/examples/batch-edited.vxm"}}
    ]
  }
}
```

Defaults: 1.8 s cycle, 2.5 cm stride, 1.4 cm lift, 80% stance and 120 baked
poses per second. Change `duration_s`, `stride_m`, `lift_m`, `stance_fraction`,
`forward_direction` or `up_axis` to tune the walk. This inference currently
requires a raised body and separate shin contacts on a common ground plane;
it reports an error for anatomy it cannot infer. It does not run mechanical
physics, add geometry, or change any voxel's saved physical state. In-place
stance feet travel backward relative to the body; world locomotion should
advance the body by stride / (cycle duration × stance fraction).

Open and play a saved clip directly in the desktop editor:

```powershell
target\release\voxel-editor.exe --model mcp/examples/spider_walk_ik.vxm --play-animation
```

The bottom timeline has Play/Pause, frame scrubbing, collapse/expand and a
`Bones` toggle. Rig overlays start hidden for a clear animation preview and
can be enabled for editing; selecting a bone in the hierarchy also shows them.

The frame timeline in the native editor scrubs the selected clip at 30 frames
per second; key positions appear as markers. MCP animation times are stored in
seconds and are independent of the editor's display frame rate.

All input and output paths are project-relative. Existing symlinks are
canonicalized and rejected if they resolve outside the root. Mutating tools
validate the full result before writing, create a new output file, and refuse
overwrite; output directories must already exist. Mesh import retains the
voxelizer's built-in resource limits (up to 4 million voxels and its grid /
triangle-work limits). Filled import requires a watertight mesh. OBJ export
leaves the `.vxm` untouched.

## Settings

The server reads `.voxel_editor_settings.json` from the project root. When the
file is absent MCP is enabled. When present, it must contain a supported
version and `mcp_enabled: true` to expose tools:

```json
{
  "version": 1,
  "mcp_enabled": true,
  "bindings": {}
}
```

If MCP is disabled, `tools/list` returns no tools and `tools/call` returns an
MCP tool error. Invalid or unsupported settings fail closed the same way. The
server never creates or edits this settings file. Standard output contains
protocol messages only; diagnostics go to standard error. Closing stdin exits
the process cleanly.

## Protocol

Supports current `2026-07-28` per-request metadata and `server/discover`, plus
the `2025-11-25` `initialize` compatibility path. Both use newline-delimited
UTF-8 JSON-RPC on stdin/stdout. No HTTP or website/server-site component is
used.

## Live telemetry examples

Check whether a native editor runtime is publishing telemetry, then request
successive batches by advancing the cursor to the last returned sequence:

```json
{"name":"get_runtime_status","arguments":{}}
{"name":"read_runtime_metrics","arguments":{"after_sequence":0,"max_samples":16,"wait_ms":1000,"topic":"physics"}}
{"name":"read_runtime_metrics","arguments":{"after_sequence":128,"max_samples":16,"wait_ms":0}}
```

The app stream reports published state such as frame rate, pause, time scale,
and scene path. The physics stream contains only fields the engine publishes,
such as step timings, substeps, solver diagnostics, energy, mass, momentum,
and body state. Use `get_runtime_objects` for a live object snapshot. Compare
session identity and freshness before combining batches; sequence gaps indicate
samples skipped between reads. An absent or stale runtime is reported clearly,
never replaced with fabricated telemetry. These tools cannot edit live engine
state; file-authoring tools remain separate and create validated project files.

For scenes above 2,048 REVs, runtime state inspection is bounded to 128
deterministic samples. Unknown whole-scene sums are null; sampled component
rows describe sampled members, not whole bodies. Solver counters, the energy
ledger and total REV count remain direct engine values. The rolling budget
window includes observer overhead and never extrapolates to a larger workload.
See `LIVE_MCP_DIAGNOSTICS.md` for scopes and current physics limitations.

### Continuous JSONL capture

`watch_runtime.py` runs the configured MCP executable over stdio, retries while
the native editor is closed, and starts a fresh sequence cursor when the
runtime session changes. It writes samples, session changes, and gap markers to
rotating JSONL files (four files, 8 MiB each). Console output is limited to
connection changes and periodic counts; the child process's stderr is not
buffered into an unread pipe.

```powershell
python mcp/watch_runtime.py `
  --mcp-exe E:\Projects\gpt\voxel-sim\Voxel-sim-guided-knife-validation\target\release\voxel-editor-mcp.exe `
  --project-root E:\Projects\gpt\voxel-sim\Voxel-sim-guided-knife-validation `
  --output-dir E:\Projects\gpt\voxel-sim\Voxel-sim-guided-knife-validation\validation\runtime-metrics `
  --duration-seconds 30
```

All three paths are required and must resolve to `E:`. Omit
`--duration-seconds` for a continuous capture; press Ctrl+C to close the MCP
child cleanly. The optional duration is capped at 24 hours.
