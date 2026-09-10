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

"""Demonstrate synthetic data generation with a 3D object-reconstructed real-world asset."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
import random

import carb
import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.replicator.core as rep
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.storage.native import get_assets_root_path

# Asset produced by the NVIDIA 3D Object Reconstruction workflow (https://github.com/NVIDIA/3DObjectReconstruction)
TOASTER_PASTRY_USD = "/Isaac/Samples/Replicator/3DObjectReconstruction/toaster_pastry.usd"
NUM_ASSETS = 24  # Total instances spawned per scenario; any remainder is released in the final wave
NUM_WAVES = 2  # Number of sequential waves per scenario, each settled wave is captured before the next is released
NUM_SCENARIOS = 2  # Number of times the wave sequence is reset and repeated; total captures = NUM_SCENARIOS * NUM_WAVES
BASE_DROP_HEIGHT = 1.0  # Spawn height for the first wave of a scenario (m)
HEIGHT_STEP_PER_WAVE = 0.3  # Extra spawn height per wave, giving clearance above the growing pile (m)
WITHIN_WAVE_HEIGHT_JITTER = 0.05  # Per-asset height variance within a wave to avoid exact overlaps (m)
SPAWN_XY_JITTER = 0.35  # Random horizontal offset +/- (m)
FALL_SPEED_THRESHOLD = 0.3  # Linear speed above which a wave is considered actively falling (m/s)
SETTLE_SPEED_THRESHOLD = 0.02  # Linear speed below which a wave is considered settled (m/s)
RNG_SEED = 17  # Reproducible randomization seed
MAX_STEPS_PER_WAVE = 50  # Maximum simulation steps to wait for a wave to settle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-assets", type=int, default=NUM_ASSETS, help="Total number of asset instances to spawn per scenario."
)
parser.add_argument("--num-waves", type=int, default=NUM_WAVES, help="Number of sequential waves per scenario.")
parser.add_argument(
    "--num-scenarios", type=int, default=NUM_SCENARIOS, help="Number of times the wave sequence is reset and repeated."
)
args, _ = parser.parse_known_args()


def run_example(num_assets: int, num_waves: int, num_scenarios: int) -> None:
    """Drop batches of a 3D-reconstructed asset onto a growing pile and capture a frame per settled wave.

    Args:
        num_assets: Total number of asset instances to spawn per scenario; any remainder is released in the final wave.
        num_waves: Number of sequential waves per scenario. Each wave is released only after the previous one settled.
        num_scenarios: Number of times the wave sequence is reset (assets re-spawned) and repeated.
    """
    if num_waves <= 0:
        raise ValueError("num_waves must be positive.")
    if num_assets < num_waves:
        raise ValueError("num_assets must be at least num_waves.")

    assets_per_wave, remainder = divmod(num_assets, num_waves)
    if remainder:
        print(
            f"[SDG] Asset count is not divisible by wave count; last wave takes the remainder of {remainder} "
            f"(size {assets_per_wave + remainder})."
        )

    stage_utils.create_new_stage()
    assets_root_path = get_assets_root_path()
    rng = random.Random(RNG_SEED)

    # Disable capture on play, frames will be captured manually
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto))
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    rep.functional.create.scope(name="Looks", parent="/World")

    # Using a cube to avoid collision tunneling: a zero-thickness plane can let fast-falling assets pass through.
    floor = rep.functional.create.cube(parent="/World", name="Floor", scale=(5, 5, 0.05))
    rep.functional.physics.apply_collider(floor)
    floor_material = rep.functional.create.material(
        mdl="OmniPBR.mdl",
        diffuse_color_constant=(0.6, 0.6, 0.6),
        bind_prims=floor,
        parent="/World/Looks",
        name="FloorMaterial",
    )

    # Batch-create every instance in a single call instead of looping over individual create.reference calls.
    all_prims = rep.functional.create_batch.reference(
        usd_path=assets_root_path + TOASTER_PASTRY_USD,
        semantics={"class": "toaster_pastry"},
        parent="/World",
        name="ToasterPastry",
        count=num_assets,
    )
    # Every asset starts out kinematic: it ignores gravity and can be freely repositioned until its wave releases
    # it by disabling kinematic mode. A single persistent view covers every asset across all waves and scenarios.
    rep.functional.physics.apply_rigid_body(all_prims, approximation="convexHull", kinematicEnabled=True)
    all_rigid_prims = RigidPrim([prim.GetPath().pathString for prim in all_prims])

    # Replicator setup, render product is disabled by default and enabled only at capture time
    camera = rep.functional.create.camera(position=(1.5, 1.5, 1.5), look_at=(0, 0, 0), parent="/World")
    render_product = rep.create.render_product(camera, (640, 480))
    # Enable render product updates only at capture time
    render_product.hydra_texture.set_updates_enabled(False)
    output_dir = os.path.join(os.getcwd(), "_out_object_reconstruction_assets_drop")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_dir)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        distance_to_camera=True,
        colorize_depth=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
    )
    writer.attach(render_product)

    # Start the simulation
    print("[SDG] Starting simulation")
    app_utils.play()

    for scenario_idx in range(num_scenarios):
        # Reset every asset for a new scenario: freeze it back to kinematic (no-op the first time) and stagger
        # each wave's spawn height for clearance above the pile the previous scenario left behind.
        rep.functional.modify.attribute(all_prims, "physics:kinematicEnabled", True)
        positions, rotations = [], []
        for i in range(num_assets):
            wave_idx = min(i // assets_per_wave, num_waves - 1)
            x = rng.uniform(-SPAWN_XY_JITTER, SPAWN_XY_JITTER)
            y = rng.uniform(-SPAWN_XY_JITTER, SPAWN_XY_JITTER)
            z = BASE_DROP_HEIGHT + wave_idx * HEIGHT_STEP_PER_WAVE + rng.uniform(0, WITHIN_WAVE_HEIGHT_JITTER)
            positions.append((x, y, z))
            rotations.append((rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360)))
        rep.functional.modify.pose(all_prims, position_value=positions, rotation_value=rotations)
        # Let the kinematic targets sync to their new poses before releasing any wave, otherwise the released
        # wave inherits a large implied velocity from the pose jump instead of starting at rest.
        app_utils.update_app()

        # Drop one wave at a time: release it, wait for it to settle onto the (growing) pile, then capture
        for wave_idx in range(num_waves):
            wave_end = num_assets if wave_idx == num_waves - 1 else (wave_idx + 1) * assets_per_wave
            wave_slice = slice(wave_idx * assets_per_wave, wave_end)
            wave_prims = all_prims[wave_slice]
            wave_size = wave_slice.stop - wave_slice.start

            # Release this wave from kinematic mode so gravity takes over
            rep.functional.modify.attribute(wave_prims, "physics:kinematicEnabled", False)
            # Zero out the implied teleport velocity only after disabling kinematic mode, since PhysX
            # rejects velocity writes on kinematic bodies.
            all_rigid_prims.set_velocities(
                linear_velocities=np.zeros((wave_size, 3)),
                angular_velocities=np.zeros((wave_size, 3)),
                indices=list(range(wave_slice.start, wave_slice.stop)),
            )

            # Step the simulation until the wave has fallen and settled
            has_fallen = False
            for _ in range(MAX_STEPS_PER_WAVE):
                app_utils.update_app()
                linear_velocities, _ = all_rigid_prims.get_velocities()
                max_speed = float(np.linalg.norm(linear_velocities.numpy()[wave_slice], axis=1).max())
                if not has_fallen:
                    # Wait until the wave has started moving before checking for a settled state
                    if max_speed > FALL_SPEED_THRESHOLD:
                        has_fallen = True
                    continue
                if max_speed < SETTLE_SPEED_THRESHOLD:
                    break

            # Randomize the floor color and capture a frame of the settled wave
            color = (rng.random(), rng.random(), rng.random())
            rep.functional.modify.attribute(floor_material, "inputs:diffuse_color_constant", color)
            print(
                f"[SDG]  Scenario {scenario_idx + 1}/{num_scenarios} - Wave {wave_idx + 1}/{num_waves} settled, "
                f"floor color -> {tuple(round(c, 2) for c in color)}"
            )
            render_product.hydra_texture.set_updates_enabled(True)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=16)
            render_product.hydra_texture.set_updates_enabled(False)

    # Pause the simulation and clean up resources
    total_captures = num_scenarios * num_waves
    print(f"[SDG] Simulation complete. {total_captures} frames saved to {output_dir}")
    app_utils.pause()
    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()


run_example(args.num_assets, args.num_waves, args.num_scenarios)

# <start-object-reconstruction-assets-sdg-test>
test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against golden data and exit.",
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

    out_dir = os.path.join(os.getcwd(), "_out_object_reconstruction_assets_drop")
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
        "_out_object_reconstruction_assets_drop",
    )
    total_captures = args.num_scenarios * args.num_waves
    expected_pngs = total_captures * 3  # rgb + colorized depth + colorized semantic segmentation per capture
    expected_json = total_captures  # semantic segmentation label json per capture
    ok = validate_folder_contents(
        path=out_dir,
        recursive=True,
        expected_counts={"png": expected_pngs, "json": expected_json},
        fail_on_empty_extensions={"png", "json"},
    )
    if not ok:
        summary = get_folder_file_summary(out_dir, recursive=True)
        carb.log_error(
            f"[SDG][Test][FAIL] Output validation failed for {out_dir}: "
            f"expected png={expected_pngs}, json={expected_json}, found {summary}"
        )
        sys.exit(1)

    # Compare every output image type against golden data, each with its own tolerance.
    comparisons = (
        ("RGB", r"^rgb_.*\.png$", 7.5),
        ("depth", r"^distance_to_camera_.*\.png$", 1.5),
        ("semantic segmentation", r"^semantic_segmentation_.*\.png$", 4.5),
    )
    for label, path_pattern, mean_diff_tolerance in comparisons:
        result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=path_pattern,
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=mean_diff_tolerance,
            print_all_stats=False,
        )
        if not result["all_passed"]:
            carb.log_error(
                f"[SDG][Test][FAIL] {label} image comparison failed (tol={mean_diff_tolerance}). "
                f"Golden dir: {golden_dir}, output dir: {out_dir}"
            )
            sys.exit(1)

    print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-object-reconstruction-assets-sdg-test>

simulation_app.close()
