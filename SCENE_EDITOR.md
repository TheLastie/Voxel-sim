# Authored scenes in the native editor

Impact examples and other visible physical scenes are ordinary authored scene
documents. Open one in the native editor with `--scene`:

```powershell
voxel-editor --scene scenes/impact.vxscene
```

This is the main visible workflow. The separate `voxel-test-viewer` is only a
secondary developer diagnostic; see [LIVE_TESTS.md](LIVE_TESTS.md).

## Build a scene

1. Open or create a scene in the editor.
2. Use **File > Add Model** to add a reusable `.vxm` asset as a scene object.
3. Choose **Object** to select a scene instance, or **Voxel** to select REVs
   within the scene. The native viewport and camera remain the scene view.
4. Select an instance in the view or hierarchy. Its Inspector shows position,
   rotation, scale, velocity, force, and the active material. Click the XYZ
   position, rotation, velocity, or force row to edit all three values in the
   popup. Enter applies; Escape cancels; Tab moves between X, Y, and Z.
5. Use the material controls to cycle built-ins or load, save, and create a
   `.vxmat` asset. Per-voxel material overrides are saved with the scene.
6. Save the authored scene as `.vxscene` from the File menu.

The File menu also provides impact scene presets for 16³ and 74³ REVs. The
native editor accepts a preset output path and side length, for example:

```powershell
voxel-editor --create-impact-scene scenes/impact-16.vxscene --side 16
voxel-editor --create-impact-scene scenes/impact-74.vxscene --side 74
```

These presets create ordinary `.vxm` and `.vxscene` assets that can be opened
and edited in the editor. They are not test-mode launches. A command or
configuration alone is not evidence that a scene was run or visually checked.

## Window size and fullscreen

The interface uses one uniform scale based on both client width and height;
the 3D viewport uses its own actual rectangle for camera projection and picking.
F11 switches between the window and borderless fullscreen. `--fullscreen` opens
the editor in fullscreen directly.

Layout, viewport bounds, and control clicks have regression checks at
1920×1080, 2560×1440, 1280×720, 960×600, 800×600, 640×480, 800×1100,
and 1280×400. Short windows reduce the interface scale to keep controls inside
the client area. Native-scene controls do not share their space with old lab
keyboard hints.

## What the scene stores

The version 5 `.vxscene` document stores authored scene objects plus optional
`SceneObjectSettings` per object: initial velocity in m/s, force in N, a
`.vxmat` asset reference, and per-voxel material overrides. `.vxmat` material
assets currently use schema version 1. These settings are part of the saved
scene; they are not a separate experiment-only format.

Authored object transforms currently preserve the 1 cm REV lattice: translation
is aligned to 1 cm, rotation is limited to 90-degree increments, and scale is
1. Runtime physics may still move and rotate objects continuously. Do not
describe authored transforms as arbitrary continuous transforms until the
scene geometry can preserve the REV lattice under them.

Scene editing and physics features do not establish a performance target or a
solver-energy result. Claims such as million-object scale, 60 FPS, or corrected
energy behavior require separate measurements with hardware, scene size,
solver settings, and physical versus rendering time clearly identified.

## Физическое состояние: Stop, сохранение, продолжение

Pause (Space) останавливает время и оставляет полное состояние запуска в редактируемой
сцене. Файл `.vxscene` описывает, как сцена начинается; файл `.vxstate` описывает, где
находится запуск сейчас: позиции, скорости, вращения ячеек, внутреннюю энергию и 27
тепловых ячеек, связи с повреждением и пластической историей, состояние контактов и
трения, журналы энергии, часы приводов и агрегаты.

- File → Save Physical State (paused) записывает `<имя сцены>.vxstate` рядом со сценой.
  Работает только на паузе; сцену надо сначала сохранить.
- File → Load Physical State читает этот файл; Run продолжает расчёт с того же места.
- R возвращает сцену к сохранённому определению (Reset — отдельная команда).

Файл версионируется, защищён контрольной суммой и пишется атомарно (временный файл,
затем переименование); повреждённый или чужой файл отклоняется до применения. Рабочие
буферы и факторизации не сохраняются и строятся заново, поэтому продолженный расчёт
совпадает с непрерывным в пределах допуска решателя (в тесте — до 1e-6 м за 20 шагов),
а не побитово. Из GUI проверены только тест меню и тест файла; запуск в окне редактора
не выполнялся.

## Chemistry, run panel and creature inspector

These controls call the same engine API as the MCP tools of the same name
(`mcp/README.md`).

- **Inspector > Chemistry** (scene open; collapsed by default): *Reactions on/off*,
  *Interval (s)* (simulated seconds between reaction updates, `0` = every step),
  *Load recipe…* (pick a recipe JSON such as `recipes/wood_rot.json`; it replaces
  the built-in recipes for this scene) and *Built-in* (back to the built-in
  recipes). Settings are saved at once in `<scene>.chemistry.json` beside the
  scene (an unsaved scene keeps them in memory until it is saved) and handed to
  the scene when the simulation starts. Changes need the authored scene, so press
  R first if the simulation has run. Errors, such as a recipe that does not parse
  or names a missing material, appear in the status bar and change nothing.
- **Project dock > Run** shows state (Run, Pause, Failed), measured world speed,
  backlog, the worker cost per 10 ms of physics as latest, p50, p95 and max over
  the last 128 batches, the active mechanical REV count and the solver mode (cell
  rotations or translation only, from `solver::uses_cell_rotations`).
- **Project dock > Creature** exists only when `<scene>.creatures.json` lists a
  creature that fits the scene. It is read-only: resource ledger, sensor channels
  with units (paged), sensor and network cost counters. The editor does not drive
  creatures, so the ledger shows the birth state and the network is not running;
  values are read while the scene is not owned by the physics worker and marked
  `(stale)` otherwise.
- **F1** lists every key the event loop handles; a test fails if a key is added
  to `src/main.rs` without a help row (or an explained exception).

Layout of all of these is tested at 1920×1009 and 1280×720. The real window was
not opened or looked at for this work.
