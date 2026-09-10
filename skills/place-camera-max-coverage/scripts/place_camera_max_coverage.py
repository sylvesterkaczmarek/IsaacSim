# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cover a floor area with cameras using isaacsim.sensors.rtx.placement.

Wraps ``CameraPlacementManager.place_camera_in_target_scope_explicit()``: divides
the target scope into patches, then adds cameras greedily until the coverage
target is met or the camera budget runs out.

Send this file to a running Isaac Sim through the python_server socket (see the
``isaac-sim-remote`` skill). It needs a live stage and cannot run under plain
Python.

Injected args (via isaacsim_send.py --arg):
    action: str - "preflight" (default), "scope", "place", or "coverage".
    scope: str - "xmin,xmax,ymin,ymax" placement bounds; omitted means use the navmesh.
    num_cameras: int - camera budget; -1 (default) places until target_coverage_ratio is met.
    target_coverage_ratio: float - fraction of accessible patches to cover (default 0.9).
    height_range: str - "min,max" camera height (default "2.5,5").
    distance_range: str - "min,max" camera-to-patch distance (default "8,16").
    look_down_angle_range: str - "min,max" degrees, 0 = horizontal (default "20,60").
    patch_size: float - coverage grid cell size in metres (default 1).
    required_camera_per_patch: int - cameras that must see each patch (default 1).
    focus_height: float - height of the focus plane (default 0.5).
    random_seed: int - makes placement deterministic (default 0).
    consider_pre_exist_camera: bool - count cameras already on the stage (default true).
    limit_fov_by_distance: bool - clip estimated FOV by distance_range (default false).
    camera_parent_prim_path: str - parent for spawned cameras (default: extension setting).
    output_dir: str - when set, also write camera_info_payload.json there.

Examples:
    isaacsim_send.py --file place_camera_max_coverage.py
    isaacsim_send.py --file place_camera_max_coverage.py --arg action=scope
    isaacsim_send.py --file place_camera_max_coverage.py --arg action=place --arg num_cameras=5
    isaacsim_send.py --file place_camera_max_coverage.py --arg action=place \
        --arg "scope=-9.5,9.5,-19,0" --arg target_coverage_ratio=0.95 --arg output_dir=/tmp/isp
"""

import math

import omni.kit.app
import omni.usd
from pxr import UsdGeom

EXTENSION_NAME = "isaacsim.sensors.rtx.placement"

if "action" not in dir():
    action = "preflight"
if "scope" not in dir():
    scope = None
if "num_cameras" not in dir():
    num_cameras = -1
if "target_coverage_ratio" not in dir():
    target_coverage_ratio = 0.9
if "height_range" not in dir():
    height_range = "2.5,5"
if "distance_range" not in dir():
    distance_range = "8,16"
if "look_down_angle_range" not in dir():
    look_down_angle_range = "20,60"
if "patch_size" not in dir():
    patch_size = 1.0
if "required_camera_per_patch" not in dir():
    required_camera_per_patch = 1
if "focus_height" not in dir():
    focus_height = 0.5
if "random_seed" not in dir():
    random_seed = 0
if "consider_pre_exist_camera" not in dir():
    consider_pre_exist_camera = True
if "limit_fov_by_distance" not in dir():
    limit_fov_by_distance = False
if "camera_parent_prim_path" not in dir():
    camera_parent_prim_path = None
if "output_dir" not in dir():
    output_dir = None


def _as_bool(value, default=False):
    """Parse an injected arg that may arrive as a real bool or as a string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_floats(value):
    """Parse a comma-separated string or a sequence into a list of floats."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _as_range(name, value):
    """Parse "min,max" and reject non-finite or inverted bounds."""
    values = _as_floats(value)
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"{name} must be 'min,max'; got {value!r}.")
    low, high = values
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    if low > high:
        raise ValueError(f"{name} must be ordered min,max; got {low} > {high}.")
    return (low, high)


def _as_scope(value):
    """Parse "xmin,xmax,ymin,ymax" into [(xmin, xmax), (ymin, ymax)].

    Mirrors CameraPlacementHelper.validate_scope, which rejects anything that is
    not exactly two ordered pairs.
    """
    values = _as_floats(value)
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError(f"scope must be 'xmin,xmax,ymin,ymax'; got {value!r}.")
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"scope must be finite; got {value!r}.")
    if values[0] >= values[1] or values[2] >= values[3]:
        raise ValueError(f"scope bounds must have min < max; got {values}.")
    return [(values[0], values[1]), (values[2], values[3])]


def _enable_extension(name):
    """Enable an extension without raising. Returns (ok, detail)."""
    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.is_extension_enabled(name):
        return True, "already-enabled"
    try:
        manager.set_extension_enabled_immediate(name, True)
        omni.kit.app.get_app().update()
    except Exception as exc:  # extension missing from this app's extscache
        return False, f"{type(exc).__name__}: {exc}"
    if manager.is_extension_enabled(name):
        return True, "enabled"
    return False, "not resolvable in this app"


def _navmesh_scope():
    """Return the navmesh XY bounds as [(x_min, x_max), (y_min, y_max)], or None.

    get_navmesh_scope() returns three pairs (X, Y, Z) but validate_scope accepts
    exactly two, so the Z pair is dropped here. A zero-sized result means no
    navmesh was reachable.
    """
    try:
        from omni.metropolis.pipeline.simulation_util import get_navmesh_scope
    except Exception:
        return None
    try:
        bounds = get_navmesh_scope()
    except Exception:
        return None
    if not bounds or len(bounds) < 2:
        return None
    xy = [(float(bounds[0][0]), float(bounds[0][1])), (float(bounds[1][0]), float(bounds[1][1]))]
    if xy[0][0] >= xy[0][1] or xy[1][0] >= xy[1][1]:
        return None
    return xy


def _camera_paths(stage, parent_path):
    """Camera prim paths directly under parent_path."""
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return []
    return [str(child.GetPath()) for child in parent.GetChildren() if child.IsA(UsdGeom.Camera)]


def _resolve_parent_path():
    """The parent prim path cameras land under: the arg, else the extension setting."""
    if camera_parent_prim_path:
        return camera_parent_prim_path
    from isaacsim.sensors.rtx.placement import GeneralSetting

    return GeneralSetting.camera_parent_prim_path


ext_ok, ext_detail = _enable_extension(EXTENSION_NAME)
stage = omni.usd.get_context().get_stage()
explicit_scope = _as_scope(scope)

if action == "preflight":
    resolved = explicit_scope if explicit_scope else (_navmesh_scope() if ext_ok and stage is not None else None)
    print(
        f"MAX_COVERAGE_PREFLIGHT: ext={'enabled' if ext_ok else 'unavailable'} "
        f"stage={'ok' if stage is not None else 'none'} "
        f"scope={'explicit' if explicit_scope else ('navmesh' if resolved else 'none')}"
    )
    print(f"  extension: {EXTENSION_NAME} ({ext_detail})")
    if resolved:
        print(f"  bounds: x={resolved[0]} y={resolved[1]}")
    if not ext_ok:
        print("  remedy: launch an app that pins the extension, e.g.")
        print("          isaacsim.exp.action_and_event_data_generation.base.sh")
    if stage is None:
        print("  remedy: open a stage before placing cameras")
    if resolved is None and stage is not None and ext_ok:
        print("  remedy: bake a navmesh, or pass scope=xmin,xmax,ymin,ymax")

elif action == "scope":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}).")
    resolved = _navmesh_scope()
    if resolved is None:
        raise RuntimeError("No navmesh scope is available. Bake a navmesh, or pass scope=xmin,xmax,ymin,ymax.")
    print(f"MAX_COVERAGE_SCOPE: x={resolved[0]} y={resolved[1]}")

elif action == "coverage":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}).")
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    from isaacsim.sensors.rtx.placement.camera_placement.visualize.visualize_camera_placement import (
        show_all_selected_camera_coverage,
    )

    parent_path = _resolve_parent_path()
    cameras = _camera_paths(stage, parent_path)
    if not cameras:
        raise RuntimeError(f"No Camera prims under {parent_path}; nothing to visualize.")

    # Fall back to the navmesh exactly as action=place does. Passing target_scope=None
    # makes show_all_selected_camera_coverage rebuild the grid over a degenerate scope
    # and silently report a 0.0 coverage ratio, which reads as "the cameras see nothing".
    target_scope = explicit_scope if explicit_scope else _navmesh_scope()
    if target_scope is None:
        raise RuntimeError(
            "No scope to measure coverage over. Bake a navmesh, or pass scope=xmin,xmax,ymin,ymax. "
            "Measuring without a scope reports 0.0 regardless of how many cameras are placed."
        )
    show_all_selected_camera_coverage(camera_prim_path_list=cameras, target_scope=target_scope)
    print(f"MAX_COVERAGE_VISUALIZED: cameras={len(cameras)} parent={parent_path}")
    print("  coverage and full-coverage ratios are logged to the Isaac Sim console")

elif action == "place":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot place cameras.")
    if stage is None:
        raise RuntimeError("No USD stage is open; cannot place cameras.")

    target_scope = explicit_scope if explicit_scope else _navmesh_scope()
    if target_scope is None:
        raise RuntimeError("No placement scope. Bake a navmesh, or pass scope=xmin,xmax,ymin,ymax.")

    camera_budget = int(num_cameras)
    if camera_budget == 0 or camera_budget < -1:
        raise ValueError(f"num_cameras must be a positive integer or -1 for auto; got {num_cameras!r}.")

    ratio = float(target_coverage_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"target_coverage_ratio must be in (0, 1]; got {ratio}.")

    from isaacsim.sensors.rtx.placement import CameraPlacementManager, CameraPlacementSettings

    parent_path = _resolve_parent_path()
    before = set(_camera_paths(stage, parent_path))

    if output_dir:
        CameraPlacementSettings.camera_placement_output_folder_path = str(output_dir)

    manager = CameraPlacementManager.get_instance()
    manager.place_camera_in_target_scope_explicit(
        target_scope=target_scope,
        camera_num=camera_budget,
        target_coverage_ratio=ratio,
        camera_height_range=_as_range("height_range", height_range),
        camera_distance_range=_as_range("distance_range", distance_range),
        camera_look_down_angle_range=_as_range("look_down_angle_range", look_down_angle_range),
        target_section_distance=float(patch_size),
        required_camera_per_patch=int(required_camera_per_patch),
        focus_platform_height=float(focus_height),
        random_seed=int(random_seed),
        consider_pre_exist_camera=_as_bool(consider_pre_exist_camera, True),
        restrict_camera_scope=_as_bool(limit_fov_by_distance, False),
        camera_parent_prim_path=camera_parent_prim_path or None,
        spawn_camera=True,
        output_camera_data=True,
    )

    # The API swallows exceptions into carb.log_error and can still return a
    # partially filled cache, so trust the stage rather than the return value.
    after = _camera_paths(stage, parent_path)
    new_cameras = [path for path in after if path not in before]
    placed = len(new_cameras)
    if placed <= 0:
        raise RuntimeError(
            f"No cameras were authored under {parent_path}. Check the Isaac Sim "
            f"console for the carb error; common causes are a scope that misses "
            f"the walkable floor, a patch_size larger than the scope allows, or "
            f"no navmesh with camera_on_navmesh enabled."
        )

    if output_dir:
        manager.cache_camera_data_as_json()

    print(
        f"MAX_COVERAGE_RESULT: placed={placed} total={len(after)} "
        f"scope=x{target_scope[0]}y{target_scope[1]} ratio={ratio} parent={parent_path}"
    )
    for path in new_cameras:
        print(f"  camera {path}")
    if output_dir:
        print(f"  payload: {output_dir}/camera_info_payload.json")

else:
    raise ValueError(f"Unknown action {action!r}; expected 'preflight', 'scope', 'place', or 'coverage'.")
