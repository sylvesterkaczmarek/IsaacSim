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

"""Place cameras aimed at a target object using isaacsim.sensors.rtx.placement.

Wraps ``CircularCameraPlacement.circular_camera_placement()``: sweeps observation
directions around a target, discards poses whose view of the target is occluded,
then keeps the ``num_cameras`` most widely separated directions.

Send this file to a running Isaac Sim through the python_server socket (see the
``isaac-sim-remote`` skill). It needs a live stage and cannot run under plain
Python.

Injected args (via isaacsim_send.py --arg):
    action: str - "preflight" (default) reports readiness only; "place" authors cameras.
    target_prim: str - prim path of the object to aim at.
    target_point: str - "x,y,z" world point; alternative to target_prim.
    num_cameras: int - cameras to keep (default 2).
    height_range: str - "min,max" camera height (default "1.5,5").
    distance_range: str - "min,max" camera-to-target distance (default "6,18").
    look_down_angle_range: str - "min,max" degrees, 0 = horizontal (default "0,60").
    yaw_ranges: str - "0,180;270,360" degrees; default is the full circle.
    occlusion_threshold: float - reject poses at or above this ratio; -1 disables (default 0.4).
    raycast_density: float - direction sweep density (default 200).
    occlusion_check_density: int - occlusion sample density (default 200).
    camera_distance_step_size: float - distance sampling step (default 0.5).
    padding: float - target bbox padding used by the occlusion test (default 0.5).
    random_seed: int - makes selection deterministic (default 42).
    on_navmesh: bool - restrict camera positions to the navmesh (default true).
    scope: str - "xmin,xmax,ymin,ymax" bounds for camera positions.
    camera_parent_prim_path: str - parent for spawned cameras (default: extension setting).
    refresh: bool - DESTRUCTIVE, deletes camera_parent_prim_path first (default false).

Examples:
    isaacsim_send.py --file place_camera_aim_at.py
    isaacsim_send.py --file place_camera_aim_at.py --arg action=place --arg target_prim=/World/Pallet
    isaacsim_send.py --file place_camera_aim_at.py --arg action=place --arg "target_point=3,4,0.5" \
        --arg num_cameras=3 --arg occlusion_threshold=-1
"""

import math

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Usd

EXTENSION_NAME = "isaacsim.sensors.rtx.placement"

if "action" not in dir():
    action = "preflight"
if "target_prim" not in dir():
    target_prim = None
if "target_point" not in dir():
    target_point = None
if "num_cameras" not in dir():
    num_cameras = 2
if "height_range" not in dir():
    height_range = "1.5,5"
if "distance_range" not in dir():
    distance_range = "6,18"
if "look_down_angle_range" not in dir():
    look_down_angle_range = "0,60"
if "yaw_ranges" not in dir():
    yaw_ranges = None
if "occlusion_threshold" not in dir():
    occlusion_threshold = 0.4
if "raycast_density" not in dir():
    raycast_density = 200.0
if "occlusion_check_density" not in dir():
    occlusion_check_density = 200
if "camera_distance_step_size" not in dir():
    camera_distance_step_size = 0.5
if "padding" not in dir():
    padding = 0.5
if "random_seed" not in dir():
    random_seed = 42
if "on_navmesh" not in dir():
    on_navmesh = True
if "scope" not in dir():
    scope = None
if "camera_parent_prim_path" not in dir():
    camera_parent_prim_path = None
if "refresh" not in dir():
    refresh = False


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


def _as_yaw_ranges(value):
    """Parse "0,180;270,360" into [(0.0, 180.0), (270.0, 360.0)]."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return [tuple(float(v) for v in pair) for pair in value]
    ranges = []
    for chunk in str(value).split(";"):
        if chunk.strip():
            ranges.append(_as_range("yaw_ranges", chunk))
    return ranges or None


def _as_scope(value):
    """Parse "xmin,xmax,ymin,ymax" into [(xmin, xmax), (ymin, ymax)]."""
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


def _navmesh_available():
    """Report whether a baked navmesh is reachable, without baking one."""
    try:
        import omni.anim.navigation.core as nav

        interface = nav.acquire_interface()
        if interface is None:
            return False
        return interface.get_navmesh() is not None
    except Exception:
        return False


def _mesh_count(stage, prim_path):
    """Count Mesh prims at or under prim_path, or -1 when the prim does not resolve.

    Occlusion filtering needs at least one Mesh; the extension silently disables
    it otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return -1
    return sum(1 for child in Usd.PrimRange(prim) if child.GetTypeName() == "Mesh")


ext_ok, ext_detail = _enable_extension(EXTENSION_NAME)
stage = omni.usd.get_context().get_stage()

if action == "preflight":
    if stage is None or not target_prim:
        target_state = "n/a"
        mesh_state = "n/a"
    else:
        meshes = _mesh_count(stage, target_prim)
        target_state = "missing" if meshes < 0 else "ok"
        mesh_state = "none" if meshes == 0 else str(max(meshes, 0))

    print(
        f"AIM_AT_PREFLIGHT: ext={'enabled' if ext_ok else 'unavailable'} "
        f"stage={'ok' if stage is not None else 'none'} "
        f"target={target_state} meshes={mesh_state} "
        f"navmesh={'ok' if _navmesh_available() else 'none'}"
    )
    print(f"  extension: {EXTENSION_NAME} ({ext_detail})")
    if not ext_ok:
        print("  remedy: launch an app that pins the extension, e.g.")
        print("          isaacsim.exp.action_and_event_data_generation.base.sh")
    if stage is None:
        print("  remedy: open a stage before placing cameras")
    if target_prim and target_state == "missing":
        print(f"  remedy: {target_prim} does not resolve on this stage")

elif action == "place":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot place cameras.")
    if stage is None:
        raise RuntimeError("No USD stage is open; cannot place cameras.")
    if not target_prim and target_point is None:
        raise ValueError("Provide target_prim or target_point.")

    parsed_point = None
    if target_point is not None:
        coords = _as_floats(target_point)
        if coords is None or len(coords) != 3:
            raise ValueError(f"target_point must be 'x,y,z'; got {target_point!r}.")
        if any(not math.isfinite(v) for v in coords):
            raise ValueError(f"target_point must be finite; got {target_point!r}.")
        parsed_point = np.array(coords, dtype=float)

    if target_prim:
        meshes = _mesh_count(stage, target_prim)
        if meshes < 0:
            raise ValueError(f"target_prim {target_prim} does not resolve on this stage.")
        prim = stage.GetPrimAtPath(target_prim)
        visibility = prim.GetAttribute("visibility")
        if visibility and visibility.Get() == "invisible":
            raise ValueError(f"target_prim {target_prim} is invisible; the placement API rejects hidden targets.")
        if meshes == 0 and float(occlusion_threshold) != -1:
            print(f"  note: no Mesh under {target_prim}; the extension will disable occlusion filtering")

    camera_count = int(num_cameras)
    if camera_count <= 0:
        raise ValueError(f"num_cameras must be a positive integer; got {num_cameras!r}.")

    from isaacsim.sensors.rtx.placement import CircularCameraPlacement

    if _as_bool(refresh):
        # Deletes the whole camera parent subtree. Guarded behind an explicit arg.
        CircularCameraPlacement.refresh_camera_prim()

    no_direction_remedy = (
        "The direction sweep around "
        f"{target_prim or parsed_point} produced no usable camera directions.\n"
        "  Widen distance_range or height_range, widen/remove yaw_ranges, raise\n"
        "  occlusion_threshold (or set it to -1 to disable occlusion filtering),\n"
        "  or set on_navmesh=false when the stage has no navmesh. Targets wedged\n"
        "  against walls or with little clearance are the common cause."
    )

    try:
        poses = CircularCameraPlacement.circular_camera_placement(
            target_object_path=target_prim or None,
            target_point=parsed_point,
            number_of_camera_to_select=camera_count,
            camera_height_range=_as_range("height_range", height_range),
            camera_distance_range=_as_range("distance_range", distance_range),
            look_down_angle_range=_as_range("look_down_angle_range", look_down_angle_range),
            yaw_range_list=_as_yaw_ranges(yaw_ranges),
            raycast_density=float(raycast_density),
            occlusion_threshold=float(occlusion_threshold),
            occlusion_check_density=int(occlusion_check_density),
            camera_distance_step_size=float(camera_distance_step_size),
            padding=float(padding),
            random_seed=int(random_seed),
            on_navmesh=_as_bool(on_navmesh, True),
            scope=_as_scope(scope),
            camera_parent_prim_path=camera_parent_prim_path or None,
            spawn_camera=True,
        )
    except ValueError as exc:
        # Upstream defect (verified against the pinned build): when zero
        # directions survive, circular_camera_placement reaches
        # math_util.select_sparse_vectors with a 1-D empty array and dies on
        # `n, d = X.shape`. Its own "not enough directions" guard sits *after*
        # that call, so it never runs. Translate the opaque unpack error.
        if "not enough values to unpack" in str(exc):
            raise RuntimeError(no_direction_remedy) from exc
        raise

    # Validation failures (bad ranges, hidden target) do return None, logging via carb.
    if not poses:
        raise RuntimeError(
            f"circular_camera_placement returned no poses for "
            f"{target_prim or parsed_point}. Check the Isaac Sim console for the "
            f"carb error; common causes are a camera_height_range below the "
            f"target, an invisible target prim, or inverted min/max ranges."
        )

    print(f"AIM_AT_RESULT: placed={len(poses)} target={target_prim or parsed_point.tolist()}")
    for index, pose in enumerate(poses):
        position = [round(float(v), 4) for v in pose.camera_position]
        focus = [round(float(v), 4) for v in pose.focus_point]
        print(f"  camera[{index}] position={position} focus={focus}")

else:
    raise ValueError(f"Unknown action {action!r}; expected 'preflight' or 'place'.")
