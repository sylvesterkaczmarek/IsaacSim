# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Run sequential scene and camera randomizations for synthetic data generation."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import itertools
import os
import time
from typing import Any

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.bounds as bounds_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import isaacsim.core.experimental.utils.xform as xform_utils
import numpy as np
import omni.replicator.core as rep
from isaacsim.storage.native import get_assets_root_path

NUM_FRAMES = 90
WRITE_DATA = True
DELAY = 0.2
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="Number of randomization frames to run.")
parser.add_argument(
    "--write-data",
    action=argparse.BooleanOptionalAction,
    default=WRITE_DATA,
    help="Write captured output to disk.",
)
parser.add_argument("--delay", type=float, default=DELAY, help="Delay in seconds between frames (0 to disable).")
args, _ = parser.parse_known_args()


# Fibonacci sphere algorithm: https://arxiv.org/pdf/0912.4540
def next_point_on_sphere(
    idx: int,
    num_points: int,
    radius: float = 1,
    origin: tuple[float, float, float] = (0, 0, 0),
) -> list[float]:
    """Compute the next evenly distributed point on a sphere.

    Args:
        idx: Index of the point to compute.
        num_points: Total number of points distributed over the sphere.
        radius: Sphere radius.
        origin: Sphere center.

    Returns:
        The computed point in Cartesian coordinates.
    """
    offset = 2.0 / num_points
    inc = np.pi * (3.0 - np.sqrt(5.0))
    z = ((idx * offset) - 1) + (offset / 2)
    phi = ((idx + 1) % num_points) * inc
    r = np.sqrt(1 - pow(z, 2))
    y = np.cos(phi) * r
    x = np.sin(phi) * r
    return [(x * radius) + origin[0], (y * radius) + origin[1], (z * radius) + origin[2]]


FORKLIFT_PATH = "/Isaac/Props/Forklift/forklift.usd"
PALLET_PATH = "/Isaac/Props/Pallet/pallet.usd"
BIN_PATH = "/Isaac/Props/KLT_Bin/small_KLT_visual.usd"
DOME_TEXTURES = [
    "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
    "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
    "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
    "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
]


def run_randomizations(
    num_frames: int,
    forklift_path: str,
    pallet_path: str,
    bin_path: str,
    dome_textures: list[str],
    write_data: bool = True,
    delay: float | None = None,
    rng: Any = None,
) -> None:
    """Run sequential asset, lighting, and camera randomizations.

    Args:
        num_frames: Number of randomization frames to run.
        forklift_path: Forklift asset path relative to the assets root.
        pallet_path: Pallet asset path relative to the assets root.
        bin_path: Bin asset path relative to the assets root.
        dome_textures: Dome texture paths relative to the assets root.
        write_data: Whether to capture and write RGB images.
        delay: Delay between frames, or None to run without a delay.
        rng: Replicator random-number generator, or None to create one from the example seed.
    """
    if rng is None:
        rng = rep.rng.ReplicatorRNG(seed=SEED)
    gen = rng.generator
    assets_root_path = get_assets_root_path()

    stage_utils.create_new_stage()
    rep.functional.create.xform(name="World")
    rep.functional.create.scope(name="Lights", parent="/World")
    dome_light = rep.functional.create.dome_light(intensity=1000, parent="/World/Lights")

    rep.functional.create.reference(
        usd_path=assets_root_path + forklift_path,
        parent="/World",
        name="Forklift",
        position=(-4.5, -4.5, 0),
    )
    pallet = rep.functional.create.reference(
        usd_path=assets_root_path + pallet_path,
        parent="/World",
        name="Pallet",
    )
    bin_prim = rep.functional.create.reference(
        usd_path=assets_root_path + bin_path,
        parent="/World",
        name="Bin",
    )
    view_cam = rep.functional.create.camera(parent="/World", name="Camera")

    dome_textures_full = [assets_root_path + tex for tex in dome_textures]
    textures_cycle = itertools.cycle(dome_textures_full)

    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_rand_sphere_scan")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        persp_cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="PerspCamera")
        rp_persp = rep.create.render_product(persp_cam, (512, 512), name="PerspView")
        rp_view = rep.create.render_product(view_cam, (512, 512), name="SphereView")
        writer.attach([rp_view, rp_persp])

    bb_cache = bounds_utils.create_bbox_cache()
    pallet_aabb = bounds_utils.compute_aabb(pallet, bbox_cache=bb_cache)
    bin_aabb = bounds_utils.compute_aabb(bin_prim, bbox_cache=bb_cache)
    pallet_size = pallet_aabb[3:6] - pallet_aabb[0:3]
    bin_size = bin_aabb[3:6] - bin_aabb[0:3]
    pallet_length = float(np.linalg.norm(pallet_size))

    for i in range(num_frames):
        if i % 5 == 0:
            rep.functional.modify.attribute(dome_light, "inputs:texture:file", next(textures_cycle))
            app_utils.update_app()

        rand_z_rot = gen.uniform(-90, 90)
        rep.functional.modify.pose(
            pallet,
            position_value=(gen.uniform(-1.5, 1.5), gen.uniform(-1.5, 1.5), 0),
            rotation_value=(0, 0, rand_z_rot),
        )
        pallet_pos, pallet_quat = xform_utils.get_world_pose(pallet)
        pallet_pos = tuple(map(float, pallet_pos.numpy()))
        pallet_euler = transform_utils.quaternion_to_euler_angles(pallet_quat.numpy(), degrees=True).numpy().flatten()

        rand_transl_x = gen.uniform(-pallet_size[0] / 2 + bin_size[0] / 2, pallet_size[0] / 2 - bin_size[0] / 2)
        rand_transl_y = gen.uniform(-pallet_size[1] / 2 + bin_size[1] / 2, pallet_size[1] / 2 - bin_size[1] / 2)

        rand_z_rot_rad = np.deg2rad(rand_z_rot)
        rot_adjusted_transl_x = rand_transl_x * np.cos(rand_z_rot_rad) - rand_transl_y * np.sin(rand_z_rot_rad)
        rot_adjusted_transl_y = rand_transl_x * np.sin(rand_z_rot_rad) + rand_transl_y * np.cos(rand_z_rot_rad)
        rep.functional.modify.pose(
            bin_prim,
            position_value=(
                float(pallet_pos[0] + rot_adjusted_transl_x),
                float(pallet_pos[1] + rot_adjusted_transl_y),
                float(pallet_pos[2] + pallet_size[2] + bin_size[2] / 2),
            ),
            rotation_value=tuple(map(float, pallet_euler)),
        )

        rand_radius = gen.normal(3, 0.5) * pallet_length
        bin_pos, _ = xform_utils.get_world_pose(bin_prim)
        bin_pos = tuple(map(float, bin_pos.numpy()))
        cam_pos = tuple(map(float, next_point_on_sphere(i, num_points=num_frames, radius=rand_radius, origin=bin_pos)))
        rep.functional.modify.pose(view_cam, position_value=cam_pos, look_at_value=bin_pos, look_at_up_axis=(0, 0, 1))

        if write_data:
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
        else:
            app_utils.update_app()
        if delay is not None and delay > 0:
            time.sleep(delay)

    if write_data:
        rep.orchestrator.wait_until_complete()
        writer.detach()
        rp_persp.destroy()
        rp_view.destroy()


def run_example(num_frames: int, write_data: bool, delay: float | None = None, rng: Any = None) -> None:
    """Run the sequential-randomization example.

    Args:
        num_frames: Number of randomization frames to run.
        write_data: Whether to capture and write RGB images.
        delay: Delay between frames, or None to run without a delay.
        rng: Replicator random-number generator, or None to create one from the example seed.
    """
    if rng is None:
        rep.set_global_seed(SEED)
        rng = rep.rng.ReplicatorRNG(seed=SEED)
    run_randomizations(num_frames, FORKLIFT_PATH, PALLET_PATH, BIN_PATH, DOME_TEXTURES, write_data, delay, rng=rng)


run_example(args.num_frames, args.write_data, args.delay)

# <start-sequential-randomizations-test>
test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against expected counts and exit.",
)
test_args, _ = test_parser.parse_known_args()

if test_args.test:
    import sys

    import omni.kit.app
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.test.utils")
    enable_extension("isaacsim.replicator.examples")
    from isaacsim.test.utils.file_validation import validate_folder_contents
    from isaacsim.test.utils.image_comparison import compare_images_in_directories

    if not args.write_data:
        print("[SDG][Test][SKIP] Output validation skipped because --no-write-data was set")
    else:
        rgb_mean_diff_tolerance = 5
        replicator_examples_ext_path = (
            omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("isaacsim.replicator.examples")
        )
        golden_dir = os.path.join(
            replicator_examples_ext_path,
            "isaacsim",
            "replicator",
            "examples",
            "tests",
            "data",
            "golden",
            "_out_rand_sphere_scan",
        )
        out_dir = os.path.join(os.getcwd(), "_out_rand_sphere_scan")
        expected_png_per_view = args.num_frames
        expected_png = expected_png_per_view * 2
        ok = validate_folder_contents(
            path=out_dir,
            recursive=True,
            expected_counts={"png": expected_png},
            fail_on_empty_files=True,
        )
        if not ok:
            print(f"[SDG][Test][FAIL] Output validation failed for {out_dir}")
            sys.exit(1)

        all_passed = True
        for view in ("PerspView", "SphereView"):
            view_golden_dir = os.path.join(golden_dir, view, "rgb")
            view_out_dir = os.path.join(out_dir, view, "rgb")
            rgb_result = compare_images_in_directories(
                golden_dir=os.path.join(golden_dir, view, "rgb"),
                test_dir=os.path.join(out_dir, view, "rgb"),
                path_pattern=r"^rgb_.*\.png$",
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=rgb_mean_diff_tolerance,
                print_all_stats=False,
            )
            if not rgb_result["all_passed"]:
                all_passed = False
                print(
                    f"[SDG][Test][FAIL] RGB image comparison failed for {view} "
                    f"(tol={rgb_mean_diff_tolerance}). "
                    f"Golden dir: {os.path.join(golden_dir, view, 'rgb')}, "
                    f"output dir: {os.path.join(out_dir, view, 'rgb')}"
                )
        if not all_passed:
            sys.exit(1)
        print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-sequential-randomizations-test>

simulation_app.close()
