#!/usr/bin/env python3
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

"""Overnight PoseWriter capture from a fixed camera grid over a shelf scene.

Sets up an NxM grid of cameras at fixed positions looking at a shelf, tags
objects with semantic labels, and runs a headless capture loop writing 6-DoF
pose annotations (JSON + RGB + optional debug overlays) to disk.

Run with:
    $ISAAC_SIM_DIR/python.sh shelf_pose_grid_capture.py \
        --num-frames 5000 --output-dir /data/pose_overnight
"""

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

_DLSS_EXEC_MODE_QUALITY = 2
_RENDER_RESOLUTION = (1280, 720)
_STATIC_DELTA_TIME = 0.0
_DEFAULT_OUTPUT_DIR = str(Path(tempfile.gettempdir()) / "pose_shelf_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoseWriter grid capture over a shelf scene.")
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR, help="Root output directory for captured frames."
    )
    parser.add_argument(
        "--num-frames", type=int, default=5000, help="Number of capture steps (each step writes from all cameras)."
    )
    parser.add_argument(
        "--rt-subframes", type=int, default=32, help="RT subframes per capture for material convergence."
    )
    parser.add_argument("--grid-rows", type=int, default=3, help="Number of camera rows in the grid.")
    parser.add_argument("--grid-cols", type=int, default=4, help="Number of camera columns in the grid.")
    parser.add_argument(
        "--write-debug-images", action="store_true", help="Write overlay debug images with projected cuboids."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible object placement.")
    parser.add_argument(
        "--scene-usd",
        type=str,
        default=None,
        help="Custom shelf scene USD path (Nucleus or local). " "Defaults to Isaac assets Simple_Warehouse shelf.",
    )
    parser.add_argument(
        "--visibility-threshold", type=float, default=0.05, help="Skip objects with visibility below this ratio."
    )
    return parser.parse_args()


def build_camera_grid(grid_rows: int, grid_cols: int, look_at: tuple):
    """Compute fixed camera positions for an NxM grid above/around the shelf.

    Cameras are arranged in a hemisphere arc facing the shelf center, spanning
    azimuth [-60, 60] degrees and elevation [15, 55] degrees.
    """
    import numpy as np

    radius = 3.0
    azimuth_range = (-60, 60)
    elevation_range = (15, 55)

    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], grid_cols)
    elevations = np.linspace(elevation_range[0], elevation_range[1], grid_rows)

    positions = []
    for elev, azim in itertools.product(elevations, azimuths):
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        x = look_at[0] + radius * np.cos(elev_rad) * np.sin(azim_rad)
        y = look_at[1] + radius * np.cos(elev_rad) * np.cos(azim_rad)
        z = look_at[2] + radius * np.sin(elev_rad)
        positions.append((float(x), float(y), float(z)))
    return positions


def run_shelf_pose_capture(args: argparse.Namespace) -> None:
    import carb.settings
    import isaacsim.core.experimental.utils.stage as stage_utils
    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.experimental.utils.semantics import add_labels
    from isaacsim.storage.native import get_assets_root_path

    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", _DLSS_EXEC_MODE_QUALITY)
    rep.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Could not resolve Isaac Sim assets root.")

    # --- Scene ---
    if args.scene_usd:
        stage_utils.open_stage(args.scene_usd)
    else:
        stage_utils.open_stage(assets_root + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")

    # --- Objects on the shelf ---
    shelf_objects = [
        ("/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd", "cracker_box"),
        ("/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd", "sugar_box"),
        ("/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd", "tomato_soup_can"),
        ("/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd", "mustard_bottle"),
        ("/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd", "pudding_box"),
        ("/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd", "potted_meat_can"),
    ]

    shelf_center = (0.0, 0.0, 1.0)
    stage = stage_utils.get_current_stage()

    for i, (asset_path, label) in enumerate(shelf_objects):
        prim_path = f"/World/ShelfObjects/obj_{i:02d}"
        stage_utils.add_reference_to_stage(
            usd_path=assets_root + asset_path,
            path=prim_path,
        )
        prim = stage.GetPrimAtPath(prim_path)
        add_labels(prim, labels=[label], taxonomy="class")
        x_offset = (i % 3 - 1) * 0.3
        z_offset = (i // 3) * 0.25
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set((shelf_center[0] + x_offset, shelf_center[1], shelf_center[2] + z_offset))

    # --- Camera grid ---
    camera_positions = build_camera_grid(args.grid_rows, args.grid_cols, look_at=shelf_center)
    render_products = []
    cameras = []

    for idx, pos in enumerate(camera_positions):
        cam = rep.functional.create.camera(
            position=pos,
            look_at=shelf_center,
            name=f"GridCam_{idx:02d}",
        )
        rp = rep.create.render_product(cam, _RENDER_RESOLUTION, name=f"grid_view_{idx:02d}")
        cameras.append(cam)
        render_products.append(rp)

    # --- PoseWriter ---
    pose_writer = rep.WriterRegistry.get("PoseWriter")
    pose_writer.initialize(
        output_dir=args.output_dir,
        use_subfolders=True,
        write_debug_images=args.write_debug_images,
        visibility_threshold=args.visibility_threshold,
        skip_empty_frames=False,
    )
    pose_writer.attach(render_products)

    # --- Capture loop ---
    print(
        f"[shelf_pose_grid] Starting capture: {args.num_frames} frames, "
        f"{len(render_products)} cameras, output={args.output_dir}"
    )
    try:
        for frame_idx in range(args.num_frames):
            rep.orchestrator.step(delta_time=_STATIC_DELTA_TIME, rt_subframes=args.rt_subframes)
            if (frame_idx + 1) % 100 == 0:
                print(f"[shelf_pose_grid] Captured frame {frame_idx + 1}/{args.num_frames}")

            # Per-frame object pose randomization (small jitter to simulate shelf disturbance)
            for i, (_, _) in enumerate(shelf_objects):
                prim_path = f"/World/ShelfObjects/obj_{i:02d}"
                prim = stage.GetPrimAtPath(prim_path)
                xformable = UsdGeom.Xformable(prim)
                ops = xformable.GetOrderedXformOps()
                if ops:
                    base_x = shelf_center[0] + ((i % 3) - 1) * 0.3
                    base_z = shelf_center[2] + (i // 3) * 0.25
                    jitter_x = float(rng.uniform(-0.02, 0.02))
                    jitter_y = float(rng.uniform(-0.02, 0.02))
                    ops[0].Set((base_x + jitter_x, shelf_center[1] + jitter_y, base_z))

        rep.orchestrator.wait_until_complete()
    finally:
        pose_writer.detach()
        for rp in render_products:
            rp.destroy()

    print(f"[shelf_pose_grid] Done. Output at: {args.output_dir}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"renderer": "RealTimePathTracing", "headless": True})

    parsed_args = parse_args()
    try:
        run_shelf_pose_capture(parsed_args)
    finally:
        simulation_app.close()
