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

"""Randomize material textures and capture synthetic images."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
import time
from typing import Any

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.replicator.core as rep
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdShade

NUM_FRAMES = 10
NUM_CUBES = 10
WRITE_DATA = True
DELAY = 0.2
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="Number of randomization frames to run.")
parser.add_argument("--num-cubes", type=int, default=NUM_CUBES, help="Number of cubes to create.")
parser.add_argument(
    "--write-data",
    action=argparse.BooleanOptionalAction,
    default=WRITE_DATA,
    help="Write captured output to disk.",
)
parser.add_argument("--delay", type=float, default=DELAY, help="Delay in seconds between frames (0 to disable).")
args, _ = parser.parse_known_args()


def run_randomizations(
    num_frames: int,
    shapes: list[Any],
    materials: list[Any],
    write_data: bool = True,
    delay: float | None = None,
    rng: Any = None,
) -> None:
    """Randomize bound material textures over multiple frames.

    Args:
        num_frames: Number of randomization frames to run.
        shapes: Shape prims whose materials are randomized.
        materials: Materials to bind and randomize.
        write_data: Whether to capture and write RGB images.
        delay: Delay between frames, or None to run without a delay.
        rng: Replicator random-number generator, or None to create one from the example seed.
    """
    if rng is None:
        rng = rep.rng.ReplicatorRNG(seed=SEED)
    gen = rng.generator
    assets_root_path = get_assets_root_path()
    textures = [
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
    ]

    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_rand_textures")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="Camera")
        rp = rep.create.render_product(cam, resolution=(512, 512))
        writer.attach(rp)

    initial_materials = {}
    for i, shape in enumerate(shapes):
        cur_mat, _ = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial()
        initial_materials[shape] = cur_mat
        rep.functional.modify.material(shape, materials[i])

    for _ in range(num_frames):
        for mat in materials:
            rep.functional.modify.attribute(mat, "inputs:diffuse_texture", gen.choice(textures))
            project_uvw = gen.choice([True, False], p=[0.9, 0.1])
            rep.functional.modify.attribute(mat, "inputs:project_uvw", bool(project_uvw))
            texture_scale = gen.uniform(0.1, 1)
            rep.functional.modify.attribute(mat, "inputs:texture_scale", (texture_scale, texture_scale))
            rep.functional.modify.attribute(mat, "inputs:texture_rotate", gen.uniform(0, 45))

        if write_data:
            rep.orchestrator.step(rt_subframes=16)
        else:
            app_utils.update_app()

        if delay is not None and delay > 0:
            time.sleep(delay)

    if write_data:
        rep.orchestrator.wait_until_complete()
        writer.detach()
        rp.destroy()

    for shape, mat in initial_materials.items():
        if mat:
            rep.functional.modify.material(shape, mat.GetPrim())
        else:
            UsdShade.MaterialBindingAPI(shape).UnbindAllBindings()


def run_example(
    num_frames: int,
    num_cubes: int,
    write_data: bool,
    delay: float | None = None,
    rng: Any = None,
) -> None:
    """Build the texture-randomization scene and run the capture loop.

    Args:
        num_frames: Number of randomization frames to run.
        num_cubes: Number of cubes to create.
        write_data: Whether to capture and write RGB images.
        delay: Delay between frames, or None to run without a delay.
        rng: Replicator random-number generator, or None to create one from the example seed.
    """
    if rng is None:
        rep.set_global_seed(SEED)
        rng = rep.rng.ReplicatorRNG(seed=SEED)
    gen = rng.generator
    stage_utils.create_new_stage()
    rep.functional.create.xform(name="World")
    rep.functional.create.scope(name="Looks", parent="/World")
    rep.functional.create.dome_light(intensity=1000, parent="/World")

    sphere = rep.functional.create.sphere(
        parent="/World", name="Sphere", position=(0.0, 0.0, 1.0), semantics={"class": "sphere"}
    )
    cubes = rep.functional.create_batch.cube(count=num_cubes, parent="/World", semantics={"class": "cube"})
    for cube in cubes:
        scale_rand = gen.uniform(0.25, 0.5)
        rep.functional.modify.pose(
            cube,
            position_value=(gen.uniform(-3.5, 3.5), gen.uniform(-3.5, 3.5), 1),
            scale_value=scale_rand,
        )
    rep.functional.create.plane(parent="/World", name="Plane", scale=(10, 10, 1))

    shapes = [sphere] + list(cubes)
    materials = rep.functional.create_batch.material(mdl="OmniPBR.mdl", count=len(shapes), parent="/World/Looks")
    run_randomizations(num_frames, shapes, materials, write_data, delay, rng=rng)


run_example(
    num_frames=args.num_frames,
    num_cubes=args.num_cubes,
    write_data=args.write_data,
    delay=args.delay,
)

# <start-randomizing-textures-test>
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
            "_out_rand_textures",
        )
        out_dir = os.path.join(os.getcwd(), "_out_rand_textures")
        ok = validate_folder_contents(
            path=out_dir,
            recursive=True,
            expected_counts={"png": args.num_frames},
            fail_on_empty_files=True,
        )
        if not ok:
            print(f"[SDG][Test][FAIL] Output validation failed for {out_dir}")
            sys.exit(1)

        rgb_result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=r"^rgb_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=rgb_mean_diff_tolerance,
            print_all_stats=False,
        )
        if not rgb_result["all_passed"]:
            print(
                f"[SDG][Test][FAIL] RGB image comparison failed (tol={rgb_mean_diff_tolerance}). "
                f"Golden dir: {golden_dir}, output dir: {out_dir}"
            )
            sys.exit(1)
        print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-randomizing-textures-test>

simulation_app.close()
