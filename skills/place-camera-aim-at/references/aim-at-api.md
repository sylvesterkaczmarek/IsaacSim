# `CircularCameraPlacement` API reference

Module: `isaacsim.sensors.rtx.placement.camera_placement.aim_at_object.circular_camera_placement`
Re-exported from the package root: `from isaacsim.sensors.rtx.placement import CircularCameraPlacement`.

Verified against the `isaacsim.sensors.rtx.placement` build pinned by `isaacsim.exp.action_and_event_data_generation.base.kit`. The extension is authored
in the `metrosim` repo, not this one — see `SKILL.md` § Limitations.

---

## `circular_camera_placement(...)`

Static method. Returns `list[CameraPose]` on success and `None` on validation
failure (the reason goes to `carb.log_error`), so callers must null-check — but
see the zero-direction crash below, which raises instead.

### Target selection

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `target_object_path` | `str \| None` | `None` | Prim path to aim at. Required for occlusion filtering. |
| `target_point` | `np.ndarray \| None` | `None` | World point to aim at. Used when `target_object_path` is absent. |
| `target_object_radius` | `float \| None` | `0.5` | Self-occlusion radius. **Overridden to `0.25` whenever `target_object_path` is set** — the target is hidden during the sweep, so the term is moot. |

At least one of `target_object_path` / `target_point` must be given. When only
the path is given, the target point becomes the centre of the padded bounding
box.

### Camera envelope

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `camera_height_range` | `tuple[float, float]` | `(1.5, 5)` | World-space camera height. See the auto-clamp below. |
| `camera_distance_range` | `tuple[float, float]` | `(6, 18.0)` | Camera-to-target distance. Rejected if `min > max` or `min < 0`. |
| `look_down_angle_range` | `tuple[float, float]` | `(0, 60.0)` | Degrees. `0` = horizontal, `90` = straight down. Rejected if `min > max`. |
| `yaw_range_list` | `list[tuple[float, float]] \| None` | `None` → `[(0.0, 360.0)]` | Azimuth arcs to sample. Each pair must be ordered and within `[0, 360]`. |
| `scope` | `list[tuple[float, float]] \| None` | `None` | `[(x_min, x_max), (y_min, y_max)]` bound on camera *positions*. |
| `on_navmesh` | `bool` | `True` | Keep camera positions on the navmesh. |

### Sampling and filtering

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `number_of_camera_to_select` | `int` | `2` | Directions kept. Fewer surviving directions than this is a **hard failure**. |
| `raycast_density` | `float` | `200.0` | Density of the direction sweep. Higher is slower and more thorough. |
| `camera_distance_step_size` | `float` | `0.5` | Step used to walk outward along each direction. |
| `occlusion_threshold` | `float` | `0.4` | Poses with occlusion ratio **≥** this are dropped. `-1` disables filtering. |
| `occlusion_check_density` | `int` | `200` | Sample density of the occlusion test. Dominates runtime. |
| `padding` | `float` | `0.5` | World-unit padding applied to the target bbox before the occlusion test. |
| `random_seed` | `int` | `42` | Seeds both direction selection and per-direction pose choice. Same seed + same stage = same result. |

### Output

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `spawn_camera` | `bool` | `False` | Author `Camera` prims. When `False` the poses are returned but nothing is written. |
| `camera_parent_prim_path` | `str \| None` | `None` → `GeneralSetting.camera_parent_prim_path` (`/World/Cameras`) | Parent for spawned cameras. Created if missing. |
| `visualize_camera_direction` | `bool` | `False` | Debug-draw all swept directions plus the selected ones. |
| `refresh_vis` | `bool` | `True` | Clear prior debug draw first. |
| `color` / `selected_color` | `tuple[float, float, float, float]` | `(1,0,1,1)` / `(0,1,1,1)` | Debug-draw colours. |
| `width` | `float` | `5` | Debug-draw line width; selected lines use `width * 3`. |

### Return value

```python
from dataclasses import dataclass

@dataclass
class CameraPose:
    camera_position: "np.ndarray"   # 3D world position
    camera_direction: "np.ndarray"  # 3D unit vector
    focus_point: "np.ndarray"       # 3D world point the camera looks at
```

---

## Non-obvious behavior

These are the things that cost debugging time. All observed in
`circular_camera_placement` and its `validate_target_info` helper.

**The target is hidden during the sweep.** Its `visibility` attribute is set to
`invisible` before raycasting and restored afterwards. A target that is *already*
invisible fails validation with "target object is invisible" — make it visible
first.

**Occlusion filtering silently switches itself off.** `occlusion_threshold` is
forced to `-1` when `target_object_path` is `None`, or when the target has no
`Mesh` descendants. You get cameras, but nothing checked whether they can see
anything. The skill's `action=preflight` reports the mesh count for this reason.

**`camera_height_range` is auto-clamped.** If the target's Z sits above
`camera_height_range[0]`, the low bound is raised to the target's Z and a warning
is logged. If the target is above `camera_height_range[1]`, placement fails
outright with "camera height should not be lower than object".

**Validation failure is a `None` return, not an exception.** Bad ranges, a
hidden target, or a target above the height range log through `carb.log_error`
and return `None`. Anything reading the return value directly will hit a
`TypeError` on the next line instead of a useful message — check first, and read
the Isaac Sim console for the actual cause.

**Zero surviving directions raises an opaque `ValueError` — upstream defect.**
Verified by live run and by source inspection of the pinned build.
When the sweep yields no usable directions, the call reaches

```text
math_util.select_sparse_vectors(X=direction_array, ...)
  -> n, d = X.shape
ValueError: not enough values to unpack (expected 2, got 1)
```

because `np.array([])` is 1-D, not `(0, 3)`. The function's own guard —

```text
if len(list(direction_camera_pos_dict.keys())) < number_of_camera_to_select:
    carb.log_error("Fail to place N cameras from different directions, ...")
    return None
```

— sits at line 212, *after* the `select_sparse_vectors` call at line 202, so it
is unreachable whenever the dict is empty. It only fires when the count is
non-zero but below `number_of_camera_to_select`.

So there are two distinct too-few-directions outcomes:

| Surviving directions | Behavior |
|---|---|
| `0` | Raises `ValueError: not enough values to unpack` |
| `1 .. number_of_camera_to_select - 1` | Logs the "Fail to place N cameras" error, returns `None` |

Either way it does not fall back to placing fewer cameras.
`scripts/place_camera_aim_at.py` catches the `ValueError` and re-raises it with
the remedies (widen the ranges, relax `occlusion_threshold`, disable
`on_navmesh`). Worth fixing upstream in `metrosim` by moving the guard above the
`select_sparse_vectors` call.

**`spawn_camera=False` is the default.** The bare API call returns poses without
authoring anything. `scripts/place_camera_aim_at.py` passes `spawn_camera=True`
for `action=place`.

---

## Supporting static methods

```python
from isaacsim.sensors.rtx.placement import CircularCameraPlacement

# Drop the debug-draw overlay from a previous visualize_camera_direction run.
CircularCameraPlacement.clean_vis_debug_draw()

# DESTRUCTIVE: deletes the entire camera parent prim subtree.
CircularCameraPlacement.refresh_camera_prim()
```

`filter_camera_poses_by_occlusion(direction_camera_pos_dict, occlusion_threshold,
target_object_path, occlusion_check_density=300, padding=0.5)` and
`validate_target_info(target_info_dict) -> bool` are also public, but
`circular_camera_placement` calls both internally. Reach for them only when
building a custom sweep.
