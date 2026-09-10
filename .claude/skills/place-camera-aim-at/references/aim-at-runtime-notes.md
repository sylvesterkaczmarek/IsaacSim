# Aim-at placement: runtime notes

Occlusion internals, settings fallbacks, destructive operations, and cost
tuning for `place-camera-aim-at`. Parameter semantics live in
[`references/aim-at-api.md`](aim-at-api.md).

---

## How occlusion is measured

`OcclusionEstimate.estimate_object_occlusion(camera_position, target_object_path,
raycast_density=200, padding=0.5)` returns a ratio in `[0, 1]`, or `-1` when no
valid sample point exists.

The pipeline is bounding-box based, not per-triangle:

1. `get_object_bbox_info(target_object_path, padding)` computes the world-space
   AABB over the `default` and `render` purposes, expands every axis by
   `padding`, and returns the 8 corners plus a radius (half the XY diagonal).
2. `visible_faces(bbox_vertices, camera_position)` picks the AABB faces facing
   the camera.
3. `CameraPlacementUtils.sample_rectangle(face, raycast_density)` scatters sample
   points across each visible face.
4. `calculate_occlusion_ratio(...)` casts a ray from the camera to each sample
   point and returns `(valid - visible) / valid`.

**Consequence:** a concave or thin target can report a clear view that a render
disagrees with, because the test sees the box, not the geometry. Treat the
result as a filter, not a guarantee — render one frame before trusting a
placement.

**`padding` cuts both ways.** It pushes sample points off the surface so rays do
not self-intersect the target, but too much padding starts sampling empty space
next to the object and under-reports occlusion. `0.5` is the default; scale it
with the object, not the scene.

---

## Cost model

Two densities dominate runtime, and they multiply:

| Knob | Default | Drives |
|---|---|---|
| `raycast_density` | `200.0` | How many candidate directions the sweep produces |
| `occlusion_check_density` | `200` | Sample points per visible face, per candidate pose |

Total occlusion work is roughly *(surviving poses) × (visible faces) ×
`occlusion_check_density`*. Doubling `occlusion_check_density` roughly doubles
placement time.

Tuning order when placement is slow:

1. Narrow `camera_distance_range` and `yaw_range_list` — fewer candidates beats
   cheaper candidates.
2. Raise `camera_distance_step_size` (default `0.5`) to sample fewer positions
   per direction.
3. Drop `occlusion_check_density` to `100` for a rough pass, restore it for the
   final run.
4. Set `occlusion_threshold=-1` to skip occlusion entirely while iterating on
   the geometric envelope, then turn it back on.

---

## Settings fallbacks

Parameters left as `None` fall back to persistent carb settings under
`/persistent/exts/isaacsim.sensors.rtx.placement/`. The one that matters here:

| Setting | Key | Default |
|---|---|---|
| `GeneralSetting.camera_parent_prim_path` | `general_setting/camera_parent_prim_path` | `/World/Cameras` |

Read or write it from Python:

```python
from isaacsim.sensors.rtx.placement import GeneralSetting

print(GeneralSetting.camera_parent_prim_path)
GeneralSetting.camera_parent_prim_path = "/World/Cameras/AimAt"
```

Or through carb directly, which is what a launch script or a UI preset does:

```python
import carb

settings = carb.settings.get_settings()
settings.set(
    "/persistent/exts/isaacsim.sensors.rtx.placement/general_setting/camera_parent_prim_path",
    "/World/Cameras/AimAt",
)
```

These settings are **persistent** — they survive app restarts. Passing
`camera_parent_prim_path` explicitly to `circular_camera_placement` is safer
than mutating the setting, because the API restores the previous value only in
the coverage path, not here.

`GeneralSetting.need_navmesh_check` (default `True`) gates whether a navmesh is
required at all. Turn it off only when supplying an explicit `scope`.

---

## Destructive operations

`CircularCameraPlacement.refresh_camera_prim()` runs
`DeletePrimsCommand([camera_parent_prim_path])` — it deletes the **entire**
camera parent subtree, not just cameras this skill authored. If
`camera_parent_prim_path` is the default `/World/Cameras`, that includes every
other camera on the stage.

`scripts/place_camera_aim_at.py` keeps this behind an explicit `refresh=true`
arg for that reason. Before using it, either point
`camera_parent_prim_path` at a dedicated subtree (e.g.
`/World/Cameras/AimAt`), or delete individual prims yourself.

`clean_vis_debug_draw()` is safe — it only clears the debug-draw overlay
produced by `visualize_camera_direction=True`.

---

## Placing around several targets

There is no batch entry point. Loop, and give each target its own parent so a
later `refresh` cannot wipe an earlier result:

```python
from isaacsim.sensors.rtx.placement import CircularCameraPlacement

targets = ["/World/Pallet_A", "/World/Pallet_B", "/World/Shelf_01"]
results = {}
for index, target in enumerate(targets):
    poses = CircularCameraPlacement.circular_camera_placement(
        target_object_path=target,
        number_of_camera_to_select=2,
        random_seed=42 + index,
        camera_parent_prim_path=f"/World/Cameras/{target.rsplit('/', 1)[-1]}",
        spawn_camera=True,
    )
    results[target] = poses or []
    if not poses:
        print(f"placement failed for {target}; see the carb log")
```

Vary `random_seed` per target. Reusing one seed across targets with similar
geometry tends to produce the same relative arrangement every time, which is
usually not what an SDG run wants.

---

## Reading the console

The API is chatty through `carb`, and the useful detail is at levels a default
app filters out. Raise the channel before a debugging run:

```python
import carb

carb.settings.get_settings().set(
    "/log/channels/isaacsim.sensors.rtx.placement",
    "verbose",
)
```

At `verbose` the call logs its full argument list and every selected direction.
At the default level, only `carb.log_error` / `carb.log_warn` lines appear —
which is where the real failure reason lands whenever the API returns `None`.
