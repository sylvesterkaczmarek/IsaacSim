# `CameraPlacementManager` API reference

Module: `isaacsim.sensors.rtx.placement.camera_placement.cover_scope.camera_placement_manager`
Re-exported from the package root: `from isaacsim.sensors.rtx.placement import CameraPlacementManager`.

Verified against the `isaacsim.sensors.rtx.placement` build pinned by `isaacsim.exp.action_and_event_data_generation.base.kit`. The extension is authored
in the `metrosim` repo, not this one — see `SKILL.md` § Limitations.

`CameraPlacementManager` is a singleton. Always obtain it with
`CameraPlacementManager.get_instance()`; do not construct one.

---

## `place_camera_in_target_scope_explicit(...)`

The scripting entry point. `place_camera_in_target_scope()` (no `_explicit`) is
the GUI path — it reads every parameter from persistent carb settings and
forwards to this method.

Returns `list[dict]` with `"translate"` and `"rotation"` (a `Gf.Quatf`) keys when
`output_camera_data=True`, `None` otherwise.

### Scope and budget

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `target_scope` | `list[tuple[float, float]]` | `None` | `[(x_min, x_max), (y_min, y_max)]`. Exactly two ordered pairs — see `validate_scope` below. `None` falls back to the navmesh. |
| `camera_num` | `int \| None` | `-1` | Camera budget. `-1` means place until `target_coverage_ratio` is met. |
| `target_coverage_ratio` | `float \| None` | `0.9` | Fraction of accessible patches that must be covered. Saved and restored around the call. |
| `consider_pre_exist_camera` | `bool \| None` | `True` | Count cameras already on the stage toward coverage. |

### Camera envelope

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `camera_height_range` | `tuple[float, float] \| None` | `(2.5, 5.0)` | World-space camera height. |
| `camera_distance_range` | `tuple[float, float] \| None` | `(8, 16.0)` | Camera-to-patch distance the solver aims to satisfy. |
| `camera_look_down_angle_range` | `tuple[float, float] \| None` | `(20, 60)` | Degrees. `0` = horizontal, `90` = straight down. |
| `restrict_camera_scope` | `bool \| None` | `True` | Maps to `limit_fov_by_distance`: clip the estimated FOV by `camera_distance_range`. |

### Coverage grid

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `target_section_distance` | `float \| None` | `1` | Patch size. **Auto-shrunk** — see below. |
| `required_camera_per_patch` | `int \| None` | `1` | Cameras that must see each patch. `None` sets it to `camera_num`, which is rarely intended. |
| `focus_platform_height` | `float \| None` | `0.5` | Height of the focus plane coverage is measured on. |
| `random_seed` | `int \| None` | `0` | Same seed + same stage = same layout. |

### Output

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `spawn_camera` | `bool \| None` | `True` | Author `Camera` prims. Note this defaults **on**, unlike the aim-at API. |
| `output_camera_data` | `bool \| None` | `False` | Return the pose list. When `False` the call returns `None` even on success. |
| `camera_parent_prim_path` | `str \| None` | `None` → `GeneralSetting.camera_parent_prim_path` (`/World/Cameras`) | Saved and restored around the call. |

---

## Non-obvious behavior

**Explicit arguments are written into persistent carb settings.** Every named
parameter is assigned onto `CameraPlacementSettings` before the solver runs.
Only `camera_parent_prim_path` and `target_coverage_ratio` are saved and
restored afterwards — `patch_size`, `min/max_camera_height`,
`min/max_camera_distance`, `min/max_camera_look_down_angle`,
`total_camera_number`, `required_camera_per_patch`, `limit_fov_by_distance`, and
`random_seed` are all **left mutated** and persist across app restarts. A later
GUI run, or a later call that omits those parameters, inherits them.

**`target_section_distance` is silently shrunk.** If it exceeds
`min(x_span, y_span) / 10`, it is replaced by that value and a warning is logged,
so the grid always has at least ten cells per side. Passing `patch_size=5` on a
20 m scope quietly becomes `2`.

**Exceptions are swallowed.** The body is wrapped in a `try/except` that logs
through `carb.log_error`, restores the two saved settings, and falls through to
`return camera_info_cache` — a list that may be empty or partially filled. A
successful-looking return does not mean placement succeeded. Count `Camera`
prims on the stage instead; `scripts/place_camera_max_coverage.py` does exactly
this.

**`output_camera_data=False` returns `None` on success.** The early return sits
before the pose-collection loop. This is easy to misread as failure.

**`required_camera_per_patch=None` is a trap.** It sets coverage-per-patch equal
to `camera_num`, i.e. every patch must be seen by every camera. Pass an explicit
integer.

**`spawn_camera` defaults to `True` here** but to `False` in
`CircularCameraPlacement.circular_camera_placement`. The two APIs disagree; be
explicit in both.

---

## Scope validation

`CameraPlacementHelper.validate_scope(scope)` returns `False` unless the scope is
exactly two pairs with `min < max` in each:

```python
from isaacsim.sensors.rtx.placement.camera_placement.cover_scope.camera_placement_helper import (
    CameraPlacementHelper,
)

assert CameraPlacementHelper.validate_scope([(-9.5, 9.5), (-19.0, 0.0)])
assert not CameraPlacementHelper.validate_scope([(-9.5, 9.5)])          # one pair
assert not CameraPlacementHelper.validate_scope([(9.5, -9.5), (0.0, 1.0)])  # inverted
```

An invalid scope is not an error — `place_camera_in_target_scope` logs a warning
and falls back to the navmesh.

---

## Settings reference

All keys live under `/persistent/exts/isaacsim.sensors.rtx.placement/` and are
persistent across restarts.

### `GeneralSetting` — `general_setting/`

| Attribute | Default | Meaning |
|---|---|---|
| `camera_parent_prim_path` | `/World/Cameras` | Parent prim for created cameras |
| `customized_floor_height` | `0` | Floor height override |
| `customized_ceiling_height` | `-1` | Ceiling height override; `-1` = auto |
| `need_navmesh_check` | `True` | Require a navmesh at all |

### `CameraPlacementSettings` — `camera_placement_settings/`

| Attribute | Default | Meaning |
|---|---|---|
| `camera_placement_output_folder_path` | `""` | Where `camera_info_payload.json` is written |
| `min_camera_height` / `max_camera_height` | `2.0` / `4.0` | Camera height range |
| `min_camera_distance` / `max_camera_distance` | `6.5` / `14` | Camera-to-patch distance range |
| `min_camera_look_down_angle` / `max_camera_look_down_angle` | `0.0` / `60.0` | Look-down angle range, degrees |
| `focus_height` | `0.5` | Focus plane height |
| `limit_fov_by_distance` | `False` | Clip estimated FOV by distance |
| `target_coverage_ratio` | `0.9` | Coverage target |
| `raycast_density` | `250` | Raycast density |
| `estimated_agent_radius` | `0.7` | Agent radius used for accessibility |
| `camera_distance_step_size` | `0.3` | Distance sampling step |
| `camera_on_navmesh` | `True` | Restrict camera positions to the navmesh |
| `min_coverage_increase` | `2` | Stop when a new camera adds fewer patches than this |
| `total_camera_number` | `-1` | Camera budget; `-1` = auto |
| `border_checking_index` | `0` | How close to the boundary cameras may sit |
| `patch_size` | `0.5` | Coverage grid cell size |
| `min_view_distance` | `1` | Stop if a new camera only sees closer than this |
| `required_camera_per_patch` | `1` | Cameras required per patch |
| `random_seed` | `0` | Placement seed |

Read and write them as class attributes:

```python
from isaacsim.sensors.rtx.placement import CameraPlacementSettings, GeneralSetting

GeneralSetting.camera_parent_prim_path = "/World/Cameras/Coverage"
CameraPlacementSettings.camera_on_navmesh = False
print(CameraPlacementSettings.target_coverage_ratio)
```

Or through carb, which is what a launch script or UI preset does:

```python
import carb

settings = carb.settings.get_settings()
settings.set(
    "/persistent/exts/isaacsim.sensors.rtx.placement/camera_placement_settings/camera_on_navmesh",
    False,
)
```

Note that `camera_on_navmesh` and `need_navmesh_check` are **settings only** —
`place_camera_in_target_scope_explicit` has no keyword for either. Set them
before the call when working on a stage without a navmesh.

---

## Other useful methods

| Method | Purpose |
|---|---|
| `cache_camera_data_as_json()` | Write `camera_info_payload.json` to `camera_placement_output_folder_path`. Returns `False` and logs when no folder is set. |
| `check_fully_coverage_ratio()` | Whether the coverage target was reached. |
| `estimate_camera_fov(camera_position, camera_dir)` | Ground-plane polygon of one camera's frustum. |
| `calculate_existing_camera_coverage(camera_path)` | Patches covered by an existing camera prim. |
| `get_coverage_area_scope()` / `get_section_size()` | Inspect the resolved grid. |
