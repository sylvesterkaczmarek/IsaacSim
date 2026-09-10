# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Demonstrate synthetic data generation using the CosmosWriter."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
from typing import Any

import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.replicator.core as rep

SEGMENTATION_MAPPING = {
    "plane": [0, 0, 255, 255],
    "cube": [255, 0, 0, 255],
    "sphere": [0, 255, 0, 255],
}
NUM_FRAMES = 60

parser = argparse.ArgumentParser()
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="Number of simulation frames to capture.")
args, _ = parser.parse_known_args()


def run_cosmos_example(num_frames: int, segmentation_mapping: dict[str, Any] | None = None) -> None:
    """Run a CosmosWriter example capturing physics simulation frames.

    Args:
        num_frames: Number of simulation frames to capture.
        segmentation_mapping: RGBA output colors keyed by semantic class, or None to use writer defaults.
    """
    # Create a new stage
    stage_utils.create_new_stage()

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # CosmosWriter requires script nodes to be enabled
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set the stage properties
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    rep.functional.create.dome_light(intensity=500)

    # Create the scenario with a ground plane and a falling sphere and cube.
    plane = rep.functional.create.plane(position=(0, 0, 0), scale=(10, 10, 1), semantics={"class": "plane"})
    rep.functional.physics.apply_collider(plane)

    sphere = rep.functional.create.sphere(position=(0, 0, 3), semantics={"class": "sphere"})
    rep.functional.physics.apply_collider(sphere)
    rep.functional.physics.apply_rigid_body(sphere)

    cube = rep.functional.create.cube(position=(1, 1, 2), scale=0.5, semantics={"class": "cube"})
    rep.functional.physics.apply_collider(cube)
    rep.functional.physics.apply_rigid_body(cube)

    # Set up the writer
    camera = rep.functional.create.camera(position=(5, 5, 3), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, (1280, 720))
    out_dir = os.path.join(os.getcwd(), "_out_cosmos_simple")
    print(f"Output directory: {out_dir}")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out_dir)
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    cosmos_writer.initialize(backend=backend, segmentation_mapping=segmentation_mapping)
    cosmos_writer.attach(rp)

    app_utils.play()

    # Capture a frame every app update
    for i in range(num_frames):
        print(f"Frame {i+1}/{num_frames}")
        app_utils.update_app()
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
    app_utils.pause()

    # Wait for all data to be written
    rep.orchestrator.wait_until_complete()
    print("Data generation complete!")
    app_utils.update_app(steps=3)
    cosmos_writer.detach()
    rp.destroy()


run_cosmos_example(num_frames=args.num_frames, segmentation_mapping=SEGMENTATION_MAPPING)

# <start-cosmos-writer-simple-test>
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
    from isaacsim.test.utils.file_validation import get_folder_file_summary, validate_folder_contents
    from isaacsim.test.utils.image_comparison import compare_images_in_directories

    image_mean_diff_tolerance = 5
    num_modalities = 5
    cosmos_modalities = ("rgb", "shaded_seg", "segmentation", "depth", "edges")
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
        "_out_cosmos_simple",
    )
    out_dir = os.path.join(os.getcwd(), "_out_cosmos_simple")
    ok = validate_folder_contents(
        path=out_dir,
        recursive=True,
        expected_counts={"png": args.num_frames * num_modalities, "mp4": num_modalities},
        fail_on_empty_extensions={"mp4"},
    )
    if not ok:
        summary = get_folder_file_summary(out_dir, recursive=True)
        print(
            f"[SDG][Test][FAIL] Output validation failed for {out_dir}: "
            f"expected png={args.num_frames * num_modalities}, mp4={num_modalities}, found {summary}"
        )
        sys.exit(1)

    clip_golden_dir = os.path.join(golden_dir, "clip_0000")
    clip_out_dir = os.path.join(out_dir, "clip_0000")
    for modality in cosmos_modalities:
        modality_golden_dir = os.path.join(clip_golden_dir, modality)
        modality_out_dir = os.path.join(clip_out_dir, modality)
        compare_result = compare_images_in_directories(
            golden_dir=modality_golden_dir,
            test_dir=modality_out_dir,
            path_pattern=rf"^{modality}_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=image_mean_diff_tolerance,
            print_all_stats=False,
        )
        if not compare_result["all_passed"]:
            print(
                f"[SDG][Test][FAIL] {modality} image comparison failed (tol={image_mean_diff_tolerance}). "
                f"Golden dir: {modality_golden_dir}, output dir: {modality_out_dir}"
            )
            sys.exit(1)
    print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-cosmos-writer-simple-test>

simulation_app.close()
