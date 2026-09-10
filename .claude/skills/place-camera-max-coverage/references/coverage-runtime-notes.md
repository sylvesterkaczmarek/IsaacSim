# Coverage placement: runtime notes

Scope resolution, coverage visualisation, the JSON payload, BEV clustering, and
cost tuning for `place-camera-max-coverage`. Parameter semantics live in
[`references/coverage-api.md`](coverage-api.md).

---

## Resolving the target scope

Three sources, in the order you should prefer them.

### 1. The navmesh (default)

```python
from omni.metropolis.pipeline.simulation_util import get_navmesh_scope

bounds = get_navmesh_scope()
target_scope = [bounds[0], bounds[1]]  # drop the Z pair
```

**`get_navmesh_scope()` returns three pairs — `[(x_min, x_max), (y_min, y_max),
(z_min, z_max)]` — but `validate_scope` accepts exactly two.** Passing the raw
return straight through fails validation and silently falls back. Always slice.

It also fails soft: with no navmesh interface, or no navmesh, it logs a warning
and returns `[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]`. A zero-sized scope is the
signature of "no navmesh", not "an empty facility". The skill's
`action=preflight` and `action=scope` both treat it as `none` for that reason.

When no navmesh exists, the call attempts `start_navmesh_baking_and_wait()`
before giving up — which is why the first `action=scope` on a large stage can
take a while.

### 2. Explicit bounds

The docs mark explicit `Stage Scope` as an escape hatch, not the normal path:

> This parameter is not recommended for normal use. Only use it in edge cases
> when a valid navmesh cannot be built for the stage.

Use it when the navmesh leaks outside the building, when the facility has no
walkable surface, or when you deliberately want to cover a sub-area:

```python
target_scope = [(-9.5, 9.5), (-19.0, 0.0)]
```

With an explicit scope on a stage that has no navmesh, also turn off the navmesh
requirement — neither is reachable as a keyword argument:

```python
from isaacsim.sensors.rtx.placement import CameraPlacementSettings, GeneralSetting

GeneralSetting.need_navmesh_check = False
CameraPlacementSettings.camera_on_navmesh = False
```

### 3. An occupancy map

When neither is available, derive bounds from an occupancy grid
(`origin + resolution × image dimensions`) via the `occupancy-map` skill, and
feed the result in as an explicit scope.

---

## Visualising coverage

```python
from isaacsim.sensors.rtx.placement.camera_placement.visualize.visualize_camera_placement import (
    clean_the_stage,
    show_all_selected_camera_coverage,
)

show_all_selected_camera_coverage(
    camera_prim_path_list=["/World/Cameras/Camera", "/World/Cameras/Camera_01"],
    draw_camera_focus_point=True,
    target_scope=[(-9.5, 9.5), (-19.0, 0.0)],
)
clean_the_stage()  # clear the overlay
```

Points are debug-drawn and coloured by how many cameras see each patch, capped
at `required_camera_per_patch`; green points mark camera focus points. Two ratios
go to the console via `carb.log_warn`:

- **coverage ratio** — accessible patches seen by at least one camera.
- **full coverage ratio** — accessible patches seen by at least
  `required_camera_per_patch` cameras.

With `required_camera_per_patch=1` the two are identical. If you set it higher
for redundancy, the second number is the one that matters.

`show_all_selected_camera_coverage` calls `initialize_section_status` internally,
so it rebuilds the grid from the current settings. Coverage numbers only compare
across runs when `patch_size` and `focus_height` match. `action=coverage` in
`scripts/place_camera_max_coverage.py` wraps this call.

**Always pass `target_scope`.** With `target_scope=None` the function rebuilds
the grid over a degenerate scope and reports **`0.0`** for both ratios no matter
how many cameras are placed — it reads as "the cameras see nothing" rather than
as a missing argument. Verified on a warehouse stage with 5 placed cameras:

| `target_scope` | Reported coverage |
|---|---|
| `None` | `0.0` |
| navmesh bounds | `0.27` |

`action=coverage` now falls back to the navmesh scope for this reason, and
errors out when neither a navmesh nor an explicit scope is available rather than
printing a misleading zero.

---

## The camera info payload

`cache_camera_data_as_json()` writes `camera_info_payload.json` into
`CameraPlacementSettings.camera_placement_output_folder_path`, creating the
folder if needed. With no folder set it logs an error and returns `False`.

```python
from isaacsim.sensors.rtx.placement import CameraPlacementManager, CameraPlacementSettings

CameraPlacementSettings.camera_placement_output_folder_path = "/tmp/isp"
CameraPlacementManager.get_instance().cache_camera_data_as_json()
```

Cameras are grouped by the cardinal direction they face:

```json
{
  "X_Positive": [
    {
      "camera_path": "/World/Cameras/Camera",
      "camera_position": [-25.489889, -14.219901, 3.3197348],
      "focus_point": [-20.801025, -18.900035, 0.5]
    }
  ]
}
```

The path must be local. The output folder is not a Nucleus or cloud URI.

---

## Cost model

Runtime is dominated by the patch grid, which is quadratic in resolution:

*cells ≈ (x_span / `patch_size`) × (y_span / `patch_size`)*

Halving `patch_size` quadruples the work. On a 40 × 60 m facility, `patch_size`
of `1.0` gives 2400 cells; `0.5` gives 9600. Expect 30–60 s+ on large stages, and
budget more when `consider_pre_exist_camera=True` forces an initial coverage pass
over cameras already on the stage.

Tuning order when placement is slow:

1. Raise `patch_size`. Remember it is auto-shrunk to at most
   `min(x_span, y_span) / 10`, so on a small scope you may already be at the cap.
2. Narrow `target_scope` to the area you actually care about.
3. Set `consider_pre_exist_camera=False` on a stage with many existing cameras.
4. Leave `restrict_camera_scope` off (the default) while iterating — FOV clipping
   by distance adds work per candidate.

---

## Stopping conditions

Placement ends for one of four reasons, and the distinction matters when the
camera count surprises you:

| Condition | Setting | Symptom |
|---|---|---|
| Coverage target reached | `target_coverage_ratio` (0.9) | Fewer cameras than the budget |
| Budget spent | `camera_num` / `total_camera_number` | Coverage below target |
| Marginal gain too small | `min_coverage_increase` (2 patches) | Stops early despite budget and target |
| New camera only sees nearby | `min_view_distance` (1 m) | Stops early in cluttered scenes |

`min_coverage_increase` and `min_view_distance` are settings-only — there is no
keyword argument for either. When placement stops well short of both budget and
target, they are the usual cause:

```python
from isaacsim.sensors.rtx.placement import CameraPlacementSettings

CameraPlacementSettings.min_coverage_increase = 1
```

---

## BEV clustering

Downstream Metropolis workflows group placed cameras into Bird's-Eye-View
clusters:

```python
from isaacsim.sensors.rtx.placement.camera_placement.visualize.visualize_camera_placement import (
    get_camera_cluster,
)

clusters = get_camera_cluster(n=3, start_index=0, max_iteration=1000, target_scope=None)
```

The clustering is **not deterministic** — successive runs on the same stage
produce different groupings. Downstream payload configuration commonly supports
one active BEV group per simulation, so `n=1` is the usual choice for a
MetroSensor-style bridge. Grouping metadata itself is written by the consuming
pipeline, not by this extension.

---

## Reading the console

The solver logs progress and its failure reasons through `carb`. Raise the
channel before a debugging run:

```python
import carb

carb.settings.get_settings().set(
    "/log/channels/isaacsim.sensors.rtx.placement",
    "info",
)
```

Most solver progress goes through `carb.log_warn`, so it is visible by default:
the resolved target scope, the effective height and distance ranges, the
auto-shrunk patch size, and the final increased-camera count. The swallowed
exception, when one occurs, appears as a single `Failed to place camera in
target scope: …` error line — that is the only trace, since the call still
returns normally.
