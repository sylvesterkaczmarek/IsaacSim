# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""UR10 bin palletizing standalone launcher."""

from __future__ import annotations

import argparse
import random
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed for random bin spawn positions/orientations and the flip coin flip. "
    "Pass an integer to make the spawn sequence deterministic across runs.",
)
parser.add_argument("--max-bins", type=int, default=None, help="Limit palletizing cycles for smoke tests.")
parser.add_argument(
    "--expected-path",
    choices=("direct", "flip"),
    default=None,
    help="Require the deterministic smoke test to exercise the selected palletizing path.",
)
args, _ = parser.parse_known_args()

launch_config = {"headless": args.headless, "hide_ui": False}
if args.test:
    launch_config.update({"renderer": "MinimalRendering", "disable_viewport_updates": True})
simulation_app = SimulationApp(launch_config)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app

ext_mgr = omni.kit.app.get_app().get_extension_manager()
ext_mgr.set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)

from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.examples.manipulation.ur10_palletizing import (
    BinSpawner,
    BinStackingContext,
    PalletizerController,
    Ur10,
    Ur10Assets,
    build_scene,
    configure_conveyor,
)

if args.headless and not args.test:
    from isaacsim.core.experimental.utils.app import enable_extension

    simulation_app.set_setting("/app/window/drawMouse", True)
    enable_extension("omni.kit.livestream.app")


def run_loop(
    robot: Ur10,
    spawner: BinSpawner,
    context: BinStackingContext,
    controller: PalletizerController,
) -> None:
    """Read estimates, run the controller, then apply its output."""
    initialized = False
    needs_reset = True
    frame_count = 0
    completed = False
    visited_states = set()
    attachment_count = 0
    was_attached = False
    while simulation_app.is_running():
        simulation_app.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            if not initialized:
                robot.initialize()
                initialized = True
            if needs_reset:
                robot.reset()
                spawner.reset()
                context.reset()
                controller.reset()
                needs_reset = False
            spawner.step()
            observation = context.read(SimulationManager.get_simulation_time())
            if observation.active_bin_attached and not was_attached:
                attachment_count += 1
            was_attached = observation.active_bin_attached
            command = controller.step(observation)
            context.apply(command, SimulationManager.get_physics_dt())
            frame_count += 1
            visited_states.add(controller.state)
            if controller.state == "done":
                took_flip_path = "flipping" in visited_states
                if args.expected_path == "flip" and not took_flip_path:
                    raise RuntimeError("Palletizing completed without exercising the required flip path.")
                if args.expected_path == "direct" and took_flip_path:
                    raise RuntimeError("Palletizing exercised the flip path instead of the required direct path.")
                expected_attachments = 2 if took_flip_path else 1
                if attachment_count < expected_attachments:
                    raise RuntimeError(
                        f"Palletizing completed with {attachment_count} physical attachment(s); "
                        f"expected at least {expected_attachments}."
                    )
                print("<palletizing complete>")
                completed = True
                break
            if args.test and frame_count >= 2400:
                raise RuntimeError(f"Timed out before palletizing completed; state={controller.state}")
        elif app_utils.is_stopped():
            needs_reset = True
    if args.test and not completed:
        raise RuntimeError(f"Stopped before palletizing completed; state={controller.state}")


def main() -> None:
    """Build and run the UR10 palletizing example."""
    if args.seed is not None:
        random.seed(args.seed)

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    env_path = "/World/Ur10Table"
    ur10_assets = Ur10Assets()

    robot = build_scene(env_path, ur10_assets)
    while stage_utils.is_stage_loading():
        simulation_app.update()
    configure_conveyor(env_path)
    robot.setup()
    simulation_app.update()
    spawner = BinSpawner(env_path, ur10_assets)
    context = BinStackingContext(robot, spawner, max_bins=args.max_bins)
    controller = PalletizerController()

    if args.test:
        app_utils.play()

    run_loop(robot, spawner, context, controller)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
