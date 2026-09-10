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

"""UR10 demo: builds an OnPlaybackTick action graph for ROS 2 clock and
ROS2ControlManager, then presses Play. The in-process CM publishes /robot_description and serves
FollowJointTrajectory; run alongside `ros2 launch isaac_ros2_control_demo
ur10_in_process.launch.py` from the workspace."""

import argparse
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--hide-ui", action="store_true")
args, _ = parser.parse_known_args()
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": args.headless, "hide_ui": args.hide_ui})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path_async

UR10_PRIM_PATH = "/World/UR10"
UR10_ASSET_REL = "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
BACKGROUND_PRIM_PATH = "/background"
BACKGROUND_ASSET_REL = "/Isaac/Environments/Simple_Room/simple_room.usd"

CONTROLLER_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ur10_controllers.yaml")


def main() -> int:
    app_utils.enable_extension("isaacsim.ros2.core")
    app_utils.enable_extension("isaacsim.ros2.bridge")
    app_utils.enable_extension("isaacsim.ros2.control")
    simulation_app.update()

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)

    try:
        assets_root = simulation_app.run_coroutine(get_assets_root_path_async())
    except RuntimeError as exc:
        carb.log_error(f"Could not find Isaac Sim assets folder: {exc}")
        simulation_app.close()
        return 1

    stage_utils.add_reference_to_stage(
        usd_path=assets_root + BACKGROUND_ASSET_REL,
        path=BACKGROUND_PRIM_PATH,
    )
    stage_utils.add_reference_to_stage(
        usd_path=assets_root + UR10_ASSET_REL,
        path=UR10_PRIM_PATH,
    )
    simulation_app.update()
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[2.5, 2.5, 1.8], target=[0.0, 0.0, 0.7])

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2ControlGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("ROS2ControlManager", "isaacsim.ros2.control.ROS2ControlManager"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "ROS2ControlManager.inputs:execIn"),
            ],
            keys.SET_VALUES: [
                ("ROS2ControlManager.inputs:targetPrim", [usdrt.Sdf.Path(UR10_PRIM_PATH)]),
                ("ROS2ControlManager.inputs:controllerConfig", CONTROLLER_YAML),
                ("ROS2ControlManager.inputs:namespace", ""),
                ("ROS2ControlManager.inputs:publishRobotDescription", True),
            ],
        },
    )

    # Start simulation.
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    app_utils.play()
    carb.log_info("UR10 + ros2_control demo running. Open a second terminal and run:")
    carb.log_info("  ros2 launch isaac_ros2_control_demo ur10_in_process.launch.py")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    app_utils.stop()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
