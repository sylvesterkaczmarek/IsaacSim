# `CameraCalibrationManager` API reference

Module: `isaacsim.sensors.rtx.calibration.camera_calibration.camera_calibration_manager`
Re-exported from the package root: `from isaacsim.sensors.rtx.calibration import CameraCalibrationManager`.

Verified against the `isaacsim.sensors.rtx.calibration` build pinned by `isaacsim.exp.action_and_event_data_generation.base.kit`. The extension is authored in
the `metrosim` repo, not this one — see `SKILL.md` § Limitations.

`CameraCalibrationManager` is a singleton. Always obtain it with
`CameraCalibrationManager.get_instance()`.

The package exports exactly three names:

```python
__all__ = ["GeneralSetting", "CameraCalibrationSettings", "CameraCalibrationManager"]
```

---

## Pipeline methods, in call order

| Stage | Method | Sync/async | Produces |
|---|---|---|---|
| 1 | `create_top_view_camera(root_prim_path=None, clipping_height=None, top_view_resolution=None)` | sync | Orthographic camera; sets `top_view_camera_path` |
| 2 | `generate_calibration_dot_prim_async()` | **async** | Dots under `calibration_prim_path` |
| 3 | `generate_calibration_async()` | **async** | `calibration.json` |
| 4 | `capture_and_process_camera_view_image()` | **async** | `Top.png`, `imageMetadata.json` |

Each stage has a sync sibling — `generate_calibration_dot_prim()`,
`generate_calibration()`, `capture_camera_images()` — that the UI buttons call.
**Those are fire-and-forget**: they wrap `asyncio.ensure_future(...)` and return
before the work completes. Scripts must await the `_async` forms, or stage 3 will
run against dots that do not exist yet.

### `create_top_view_camera` arguments

| Parameter | Default when `None` | Notes |
|---|---|---|
| `root_prim_path` | `CameraCalibrationSettings.scene_bounding_box_path` | Prim whose world bbox frames the orthographic view. Defaults to `""`, which is not a valid prim — pass it explicitly. |
| `clipping_height` | `GeneralSetting.customized_ceiling_height` (`-1` = auto) | Near-clip height for the top camera. |
| `top_view_resolution` | `CameraCalibrationHelper.VIEWPORT_RESOLUTION` | `(width, height)` of the top-view render. |

The created camera lands at
`{top_camera_parent_prim_path}/{top_view_camera_prim_name}`, i.e.
`/World/Top_Camera/Calibration_Top_Camera` by default.

### Readiness and inspection

| Method | Notes |
|---|---|
| `is_ready_to_calibrate()` | **Returns `None` on success**, `False` on failure. See below. |
| `refresh_fov_info(store_fov_to_file=False)` | Recompute FOV contours; called internally by stage 3. |
| `draw_fov_polygon()` | Draw FOV polygons in the viewport (needs `show_fov_polygon_enabled`). |
| `calculate_conversion_factors()` | World↔image scale; prefers the scene bbox, falls back to the dots. |

---

## Non-obvious behavior

**`is_ready_to_calibrate()` never returns `True`.** The body has exactly three
`return False` branches — no cameras under the parent prim, missing calibration
dots, non-orthographic top camera — and then falls off the end, yielding `None`:

```text
def is_ready_to_calibrate(self):
    if not camera_prim_list:            return False
    if not is_valid_prim(...dots...):   return False
    if not is_valid_top_view_camera_type(): return False
    # <- no return True; falls through to None
```

`None` is falsy, so the natural-looking guard is always wrong:

```python
# WRONG - the body never executes, even on a perfectly valid stage.
if manager.is_ready_to_calibrate():
    await manager.generate_calibration_async()

# RIGHT
if manager.is_ready_to_calibrate() is False:
    raise RuntimeError("not ready; see the carb log")
await manager.generate_calibration_async()
```

Worth fixing upstream in `metrosim` by adding the missing `return True`.

**One uncovered camera aborts everything.** `generate_calibration_async()` loops
over cameras and bails on the first one whose pose does not project into the top
view, logging `camera <path> is not covered by top view! Fail to generate
calibration file.` and returning — no partial file. Make `scene_root` bound every
camera.

**`scene_bounding_box_path` is advisory for calibration, load-bearing for the top
view.** The UI treats an invalid scene root as non-fatal for
calibration and falls back to dot-based conversion factors. But
`create_top_view_camera` defaults its framing prim to the same setting, so an
empty value still yields a useless top view. Earlier versions failed hard here
with a `UsdGeomBBoxCache::ComputeWorldBound` null-prim error.

**`top_view_camera_path` is non-persistent.** Stage 1 must run every session even
if `/World/Top_Camera/Calibration_Top_Camera` still exists on the stage, because
the setting that later stages read is reset on app start.

---

## Settings reference

Keys live under `/persistent/exts/isaacsim.sensors.rtx.calibration/`. Note the
persistence column — the non-persistent ones reset on every app start.

### `GeneralSetting`

Same class name and defaults as the placement extension's `GeneralSetting`, and
the two share the `camera_parent_prim_path` value.

| Attribute | Default | Persistent | Meaning |
|---|---|---|---|
| `camera_parent_prim_path` | `/World/Cameras` | yes | Parent of the cameras to calibrate |
| `customized_floor_height` | `0` | yes | Floor height |
| `customized_ceiling_height` | `-1` | yes | Ceiling height; `-1` = auto |
| `need_navmesh_check` | `True` | yes | Shared with placement; unused here |

### `CameraCalibrationSettings`

| Attribute | Default | Persistent | Meaning |
|---|---|---|---|
| `camera_calibration_output_folder_path` | `""` | yes | Local output folder |
| `place_info` | `""` | yes | `key=value` pairs joined by `/` |
| `calibration_prim_path` | `/World/Calibration_Dots` | yes | Parent of the dots |
| `top_camera_parent_prim_path` | `/World/Top_Camera` | yes | Parent of the top-view camera |
| `raycast_seed` | `100` | yes | Dot-placement seed |
| `top_view_camera_path` | `""` | **no** | Set by `create_top_view_camera()` |
| `scene_bounding_box_path` | `""` | **no** | Framing prim for the top view |
| `calibration_prim_num` | `6` | **no** | Dots per camera |
| `show_fov_polygon_enabled` | `False` | **no** | Draw FOV polygons |
| `capture_camera_view_images_enabled` | `False` | **no** | Also capture per-camera views |
| `fov_contour_simplification_threshold` | `0` | **no** | FOV contour simplification |
| `fov_area_filter_threshold` | `0` | **no** | Drop FOV regions below this area |

Class constants (not settings): `top_view_camera_prim_name =
"Calibration_Top_Camera"`, `top_image_name = "Top.png"`,
`default_calibration_file_name = "calibration.json"`, `image_metadata_name =
"imageMetadata.json"`, `debug_data_folder_name = "Debug"`.

Read and write settings as class attributes:

```python
from isaacsim.sensors.rtx.calibration import CameraCalibrationSettings, GeneralSetting

CameraCalibrationSettings.camera_calibration_output_folder_path = "/tmp/calib"
CameraCalibrationSettings.place_info = "room=warehouse/floor=1"
print(GeneralSetting.camera_parent_prim_path)
```

---

## `place_info` format

Slash-separated `key=value` pairs. Validation passes if **at least one** pair has
a non-empty key and value, neither equal to `"none"`:

| Value | Valid | Why |
|---|---|---|
| `room=warehouse` | yes | one good pair |
| `room=warehouse/floor=1` | yes | two good pairs |
| `room=warehouse/garbage` | yes | one good pair is enough |
| `warehouse` | no | no `=` |
| `room=none` | no | value is `none` |
| `""` | no | empty |

The string is copied verbatim into each sensor's `place` field in
`calibration.json` and into `imageMetadata.json`.
