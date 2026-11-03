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
