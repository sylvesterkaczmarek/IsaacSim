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

"""Extract intrinsics, extrinsics, and FOV polygons for placed cameras.

Wraps ``CameraCalibrationManager`` from ``isaacsim.sensors.rtx.calibration``. Runs
the four-stage pipeline in order — orthographic top-view camera, per-camera
calibration dots, calibration file, top-view image — awaiting each stage so the
next one sees finished work.

Send this file to a running Isaac Sim through the python_server socket (see the
``isaac-sim-remote`` skill). It needs a live stage with cameras already placed;
use ``place-camera-max-coverage`` or ``place-camera-aim-at`` first.

Injected args (via isaacsim_send.py --arg):
    action: str - "preflight" (default) reports readiness; "calibrate" runs the
        full pipeline; "topview" only creates the top-view camera and image.
    output_dir: str - LOCAL folder for calibration.json / Top.png. Required to calibrate.
    place_info: str - "key=value" pairs joined by "/" (default "room=warehouse").
    scene_root: str - prim bounding the area of interest (default "/World").
    camera_parent_prim_path: str - parent of the cameras to calibrate (default: extension setting).
    dot_count: int - calibration dots per camera (default 6).
    raycast_seed: int - dot-placement seed (default 100).
    ceiling_height: float - clipping height for the top-view camera; -1 = auto (default -1).
    floor_height: float - floor height used by the placement/calibration pair (default 0).
    top_view_width / top_view_height: int - top-view render resolution (default 1920x1080).

Examples:
    isaacsim_send.py --file calibrate_camera.py
    isaacsim_send.py --file calibrate_camera.py --arg action=calibrate --arg output_dir=/tmp/calib
    isaacsim_send.py --file calibrate_camera.py --arg action=calibrate --arg output_dir=/tmp/calib \
        --arg "place_info=room=warehouse/floor=1" --arg scene_root=/World/warehouse
"""

import os

import omni.kit.app
import omni.usd
from pxr import UsdGeom

EXTENSION_NAME = "isaacsim.sensors.rtx.calibration"

if "action" not in dir():
    action = "preflight"
if "output_dir" not in dir():
    output_dir = None
if "place_info" not in dir():
    place_info = "room=warehouse"
if "scene_root" not in dir():
    scene_root = "/World"
if "camera_parent_prim_path" not in dir():
    camera_parent_prim_path = None
if "dot_count" not in dir():
    dot_count = 6
if "raycast_seed" not in dir():
    raycast_seed = 100
if "ceiling_height" not in dir():
    ceiling_height = -1
if "floor_height" not in dir():
    floor_height = 0
if "top_view_width" not in dir():
    top_view_width = 1920
if "top_view_height" not in dir():
    top_view_height = 1080


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


def _valid_place_info(text):
    """Mirror CalibrationDataProcessUtils.check_place_info: one good key=value wins."""
    for item in str(text or "").split("/"):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key.lower() != "none" and value.lower() != "none":
            return True
    return False


def _camera_paths(stage, parent_path):
    """Camera prim paths directly under parent_path."""
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return []
    return [str(c.GetPath()) for c in parent.GetChildren() if c.IsA(UsdGeom.Camera)]


ext_ok, ext_detail = _enable_extension(EXTENSION_NAME)
stage = omni.usd.get_context().get_stage()

if ext_ok:
    from isaacsim.sensors.rtx.calibration import CameraCalibrationManager, CameraCalibrationSettings, GeneralSetting

    parent_path = camera_parent_prim_path or GeneralSetting.camera_parent_prim_path
else:
    parent_path = camera_parent_prim_path or "/World/Cameras"

cameras = _camera_paths(stage, parent_path) if stage is not None else []
root_ok = bool(stage is not None and scene_root and stage.GetPrimAtPath(scene_root).IsValid())

if action == "preflight":
    print(
        f"CALIBRATE_PREFLIGHT: ext={'enabled' if ext_ok else 'unavailable'} "
        f"stage={'ok' if stage is not None else 'none'} "
        f"cameras={len(cameras)} "
        f"place_info={'ok' if _valid_place_info(place_info) else 'invalid'} "
        f"scene_root={'ok' if root_ok else 'missing'} "
        f"output_dir={'set' if output_dir else 'unset'}"
    )
    print(f"  extension: {EXTENSION_NAME} ({ext_detail})")
    print(f"  camera parent: {parent_path}")
    if not ext_ok:
        print("  remedy: launch an app that pins the extension, e.g.")
        print("          isaacsim.exp.action_and_event_data_generation.base.sh")
    if stage is None:
        print("  remedy: open a stage before calibrating")
    if not cameras:
        print(f"  remedy: no cameras under {parent_path}; run place-camera-max-coverage or place-camera-aim-at first")
    if not _valid_place_info(place_info):
        print("  remedy: place_info needs at least one key=value pair, e.g. 'room=warehouse'")
    if not root_ok:
        print(f"  remedy: scene_root {scene_root!r} does not resolve; the top-view camera is framed from it")
    if not output_dir:
        print("  remedy: pass output_dir=<local folder> to write calibration.json and Top.png")

elif action in ("calibrate", "topview"):
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot calibrate.")
    if stage is None:
        raise RuntimeError("No USD stage is open; cannot calibrate.")
    if not cameras:
        raise RuntimeError(
            f"No Camera prims under {parent_path}. Place cameras first with "
            f"place-camera-max-coverage or place-camera-aim-at, or pass camera_parent_prim_path."
        )
    if not output_dir:
        raise ValueError("output_dir is required; calibration writes calibration.json and Top.png into it.")
    if "://" in str(output_dir):
        raise ValueError(f"output_dir must be a local filesystem path, not a URI; got {output_dir!r}.")
    if not _valid_place_info(place_info):
        raise ValueError(f"place_info needs at least one key=value pair; got {place_info!r}.")
    if not root_ok:
        raise ValueError(
            f"scene_root {scene_root!r} does not resolve to a prim. The top-view camera is framed "
            f"from this prim's bounding box, so an invalid path yields an unusable top view."
        )

    os.makedirs(str(output_dir), exist_ok=True)

    CameraCalibrationSettings.camera_calibration_output_folder_path = str(output_dir)
    CameraCalibrationSettings.place_info = str(place_info)
    # Framing prim for the orthographic top view. Also the fallback source of the
    # world->image conversion factors when the calibration dots are unusable.
    CameraCalibrationSettings.scene_bounding_box_path = str(scene_root)
    CameraCalibrationSettings.calibration_prim_num = int(dot_count)
    CameraCalibrationSettings.raycast_seed = int(raycast_seed)
    GeneralSetting.customized_ceiling_height = float(ceiling_height)
    GeneralSetting.customized_floor_height = float(floor_height)
    if camera_parent_prim_path:
        GeneralSetting.camera_parent_prim_path = str(camera_parent_prim_path)

    manager = CameraCalibrationManager.get_instance()

    # Stage 1: orthographic top-view camera. Sets top_view_camera_path, which every
    # later stage checks; that setting is non-persistent, so this must run each session.
    manager.create_top_view_camera(
        root_prim_path=str(scene_root),
        top_view_resolution=(int(top_view_width), int(top_view_height)),
    )
    top_camera = CameraCalibrationSettings.top_view_camera_path
    if not top_camera or not stage.GetPrimAtPath(top_camera).IsValid():
        raise RuntimeError(
            f"create_top_view_camera did not produce a valid camera (top_view_camera_path={top_camera!r}). "
            f"Check that scene_root {scene_root!r} has renderable geometry."
        )

    if action == "calibrate":
        # Stage 2: per-camera calibration dots. Await the async form -- the sync
        # wrapper fires a task and returns immediately, so stage 3 would race it.
        await manager.generate_calibration_dot_prim_async()

        # is_ready_to_calibrate() returns None (falsy) on success and False on
        # failure, so compare against False rather than testing truthiness.
        if manager.is_ready_to_calibrate() is False:
            raise RuntimeError(
                "is_ready_to_calibrate() rejected the stage. Check the Isaac Sim console: "
                "usually no cameras under the parent prim, missing calibration dots, or a "
                "top-view camera that is not orthographic."
            )

        # Stage 3: writes calibration.json.
        await manager.generate_calibration_async()

    # Stage 4: writes Top.png (and imageMetadata.json).
    await manager.capture_and_process_camera_view_image()

    produced = []
    for name in ("calibration.json", "Top.png", "imageMetadata.json"):
        path = os.path.join(str(output_dir), name)
        if os.path.exists(path):
            produced.append(f"{name} ({os.path.getsize(path)} bytes)")

    if action == "calibrate" and not any(p.startswith("calibration.json") for p in produced):
        raise RuntimeError(
            f"calibration.json was not written to {output_dir}. Check the Isaac Sim console; a "
            f"camera not covered by the top view aborts generate_calibration_async() with "
            f"'camera ... is not covered by top view!'."
        )

    print(f"CALIBRATE_RESULT: action={action} cameras={len(cameras)} top_camera={top_camera} output={output_dir}")
    for item in produced:
        print(f"  wrote {item}")
    if not produced:
        print("  WARNING: no output files found; see the Isaac Sim console")

else:
    raise ValueError(f"Unknown action {action!r}; expected 'preflight', 'calibrate', or 'topview'.")
