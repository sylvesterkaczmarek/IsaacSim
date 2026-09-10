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

"""Demonstrate motion blur capture with the RTX Interactive (Path Tracing) render mode."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import carb.settings
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import PhysicsScene
from isaacsim.storage.native import get_assets_root_path

# Paths to the animated and physics-ready assets
PHYSICS_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
ANIM_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"

# -z velocities and start locations of the animated (left side) and physics (right side) assets (stage units/s)
ASSET_VELOCITIES = [0, 5, 10]
ASSET_X_MIRRORED_LOCATIONS = [(0.5, 0, 0.3), (0.3, 0, 0.3), (0.1, 0, 0.3)]

# Used to calculate how many frames to animate the assets to maintain the same velocity as the physics assets
ANIMATION_DURATION = 10

# Number of frames to capture for each scenario
NUM_FRAMES = 3

DEFAULT_DELTA_TIMES = [None, 1 / 30, 1 / 60, 1 / 240]
DEFAULT_SAMPLES_PER_PIXEL = [32, 128]
DEFAULT_MOTION_BLUR_SUBSAMPLES = [4, 16]


def parse_delta_time(value: str) -> float | None:
    """Parse a capture time step from a command-line value.

    Args:
        value: Numeric time step, or ``None``, ``-1``, or ``0`` to select the stage time step.

    Returns:
        Parsed time step, or None when the stage time step should be used.

    Raises:
        ValueError: If ``value`` is neither a number nor ``None``.
    """
    if value.lower() == "none":
        return None
    float_value = float(value)
    return None if float_value in (-1, 0) else float_value


parser = argparse.ArgumentParser()
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="Number of frames to capture per scenario.")
parser.add_argument(
    "--delta_times",
    nargs="*",
    type=parse_delta_time,
    default=DEFAULT_DELTA_TIMES,
    help="List of delta times (seconds per frame) to use for motion blur captures. Use 'None' for default stage time.",
)
parser.add_argument(
    "--samples_per_pixel",
    nargs="*",
    type=int,
    default=DEFAULT_SAMPLES_PER_PIXEL,
    help="List of samples per pixel (spp) values for path tracing",
)
parser.add_argument(
    "--motion_blur_subsamples",
    nargs="*",
    type=int,
    default=DEFAULT_MOTION_BLUR_SUBSAMPLES,
    help="List of motion blur subsample values for path tracing",
)
args, _ = parser.parse_known_args()
delta_times = args.delta_times
samples_per_pixel = args.samples_per_pixel
motion_blur_subsamples = args.motion_blur_subsamples
num_frames = args.num_frames


def setup_stage() -> None:
    """Create a new USD stage with animated and physics-enabled assets with synchronized motion."""
    stage_utils.create_new_stage()
    settings = carb.settings.get_settings()
    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    settings.set("rtx/post/dlss/execMode", 2)

    # Capture data only on request
    rep.orchestrator.set_capture_on_play(False)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(ANIMATION_DURATION)
    timeline.commit()

    # Create lights
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=100, parent="/World", name="DomeLight")
    rep.functional.create.distant_light(intensity=2500, rotation=(315, 0, 0), parent="/World", name="DistantLight")

    # Setup the physics assets with gravity disabled and the requested velocity
    assets_root_path = get_assets_root_path()
    physics_asset_url = assets_root_path + PHYSICS_ASSET_URL
    for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
        prim = rep.functional.create.reference(
            usd_path=physics_asset_url, parent="/World", name=f"physics_asset_{int(abs(velocity))}", position=location
        )
        rigid_prim = RigidPrim(str(prim.GetPrimPath()))
        rigid_prim.set_enabled_gravities([False])
        rigid_prim.set_velocities(linear_velocities=[(0, 0, -velocity)], angular_velocities=[(0, 0, 0)])

    # Setup animated assets maintaining the same velocity as the physics assets
    anim_asset_url = assets_root_path + ANIM_ASSET_URL
    for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
        start_location = (-location[0], location[1], location[2])
        prim = rep.functional.create.reference(
            usd_path=anim_asset_url, parent="/World", name=f"anim_asset_{int(abs(velocity))}", position=start_location
        )
        animation_distance = velocity * ANIMATION_DURATION
        end_location = (start_location[0], start_location[1], start_location[2] - animation_distance)
        end_keyframe_time = timeline.get_time_codes_per_seconds() * ANIMATION_DURATION
        # Timesampled keyframe (animated) translation
        prim.GetAttribute("xformOp:translate").Set(start_location, time=0)
        prim.GetAttribute("xformOp:translate").Set(end_location, time=end_keyframe_time)


def run_motion_blur_example_pt(
    num_frames: int,
    delta_time: float | None = None,
    motion_blur_subsamples: int = 8,
    samples_per_pixel: int = 64,
) -> None:
    """Capture motion blur frames with the given delta time step using Path Tracing rendering.

    Args:
        num_frames: Number of motion-blurred frames to capture.
        delta_time: Simulation time advanced per capture, or None to use the stage time step.
        motion_blur_subsamples: Number of temporal subframes rendered for each captured frame.
        samples_per_pixel: Path-tracing sample count for each pixel and frame.
    """
    setup_stage()
    stage = omni.usd.get_context().get_stage()
    settings = carb.settings.get_settings()

    # Enable motion blur capture
    settings.set("/omni/replicator/captureMotionBlur", True)

    # Set Path Tracing motion blur settings
    print("[MotionBlur] Setting PathTracing render mode motion blur settings")
    settings.set("/rtx/rendermode", "PathTracing")
    # (int): Total number of samples for each rendered pixel, per frame.
    settings.set("/rtx/pathtracing/spp", samples_per_pixel)
    # (int): Maximum number of samples to accumulate per pixel. When this count is reached the rendering stops until a scene or setting change is detected, restarting the rendering process. Set to 0 to remove this limit.
    settings.set("/rtx/pathtracing/totalSpp", samples_per_pixel)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)
    # Number of sub samples to render if in PathTracing render mode and motion blur is enabled.
    settings.set("/omni/replicator/pathTracedMotionBlurSubSamples", motion_blur_subsamples)

    # Setup backend
    delta_time_str = "None" if delta_time is None else f"{delta_time:.4f}"
    mode_str = f"pt_subsamples_{motion_blur_subsamples}_spp_{samples_per_pixel}"
    output_directory = os.path.join(os.getcwd(), f"_out_motion_blur_func_dt_{delta_time_str}_{mode_str}")
    print(f"[MotionBlur] Output directory: {output_directory}")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_directory)

    # Setup writer and render product
    camera = rep.functional.create.camera(
        position=(0, 1.5, 0), look_at=(0, 0, 0), parent="/World", name="MotionBlurCam"
    )
    render_product = rep.create.render_product(camera, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True)
    writer.attach(render_product)

    # Run a few updates to make sure all materials are fully loaded for capture
    for _ in range(5):
        simulation_app.update()

    rep.functional.physics.create_physics_scene(path="/PhysicsScene")
    physics_scene = PhysicsScene("/PhysicsScene")

    # Path tracing renders multiple subframes per frame, so the physics FPS is scaled accordingly
    stage_time_codes_per_second = stage.GetTimeCodesPerSecond()
    target_physics_fps = stage_time_codes_per_second if delta_time is None else 1 / delta_time
    target_physics_fps *= motion_blur_subsamples

    target_physics_dt = 1.0 / target_physics_fps
    original_physics_dt = physics_scene.get_dt()

    if target_physics_dt < original_physics_dt:
        print(f"[MotionBlur] Changing physics FPS from {1.0 / original_physics_dt:.0f} to {target_physics_fps:.0f}")
        physics_scene.set_dt(target_physics_dt)

    # Start the timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    timeline.commit()

    for i in range(num_frames):
        print(f"[MotionBlur] \tCapturing frame {i}")
        rep.orchestrator.step(delta_time=delta_time)

    if target_physics_dt < original_physics_dt:
        print(f"[MotionBlur] Restoring physics FPS from {target_physics_fps:.0f} to {1.0 / original_physics_dt:.0f}")
        physics_scene.set_dt(original_physics_dt)

    # Switch back to the raytracing render mode
    print("[MotionBlur] Restoring render mode to RealTimePathTracing")
    settings.set("/rtx/rendermode", "RealTimePathTracing")

    # Wait until the data is fully written
    rep.orchestrator.wait_until_complete()

    # Stop the timeline
    timeline.stop()
    timeline.commit()

    # Cleanup
    writer.detach()
    render_product.destroy()


def run_motion_blur_examples(
    num_frames: int,
    delta_times: list[float | None],
    samples_per_pixel: list[int],
    motion_blur_subsamples: list[int],
) -> None:
    """Run Path Tracing motion blur examples across all delta time, subsample, and spp combinations.

    Args:
        num_frames: Number of frames to capture for each configuration.
        delta_times: Simulation time steps to evaluate; None entries use the stage time step.
        samples_per_pixel: Path-tracing sample counts to evaluate.
        motion_blur_subsamples: Temporal subframe counts to evaluate.
    """
    print(
        f"[MotionBlur] Running with delta_times={delta_times}, samples_per_pixel={samples_per_pixel}, motion_blur_subsamples={motion_blur_subsamples}"
    )
    for delta_time in delta_times:
        for motion_blur_subsample in motion_blur_subsamples:
            for samples_per_pixel_value in samples_per_pixel:
                run_motion_blur_example_pt(
                    num_frames=num_frames,
                    delta_time=delta_time,
                    motion_blur_subsamples=motion_blur_subsample,
                    samples_per_pixel=samples_per_pixel_value,
                )


run_motion_blur_examples(
    num_frames=num_frames,
    delta_times=delta_times,
    samples_per_pixel=samples_per_pixel,
    motion_blur_subsamples=motion_blur_subsamples,
)

# <start-motion-blur-test>
test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against expected counts and exit.",
)
test_args, _ = test_parser.parse_known_args()

if test_args.test:
    import sys

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.test.utils")
    from isaacsim.test.utils.file_validation import validate_folder_contents

    # Each run_motion_blur_example_pt call produces one output dir with NUM_FRAMES rgb pngs.
    # Mirror the iteration in run_motion_blur_examples to reconstruct the expected dir names.
    expected_out_dirs = []
    for dt in delta_times:
        dt_str = "None" if dt is None else f"{dt:.4f}"
        for sub in motion_blur_subsamples:
            for spp in samples_per_pixel:
                expected_out_dirs.append(
                    os.path.join(os.getcwd(), f"_out_motion_blur_func_dt_{dt_str}_pt_subsamples_{sub}_spp_{spp}")
                )

    for out_dir in expected_out_dirs:
        if not validate_folder_contents(
            path=out_dir,
            recursive=True,
            expected_counts={"png": num_frames},
            fail_on_empty_files=True,
        ):
            print(f"[SDG][Test][FAIL] Output validation failed for {out_dir}")
            sys.exit(1)
        print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-motion-blur-test>

simulation_app.close()
