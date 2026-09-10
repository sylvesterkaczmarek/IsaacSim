# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Drive a Jetbot around using DifferentialDriveController.

Usage:
    ./python.sh standalone_examples/api/isaacsim.robot_motion.controllers/differential_drive.py

Sliders in the UI window let you adjust linear speed and angular rate at runtime.
"""

import argparse

# Parse args before SimulationApp so --test is available for the loop guard.
_parser = argparse.ArgumentParser(description="Differential drive example")
_parser.add_argument(
    "--test",
    default=False,
    action="store_true",
    help="Run in test mode (a few frames, no window interaction).",
)
args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.controllers as ctrl
import isaacsim.robot_motion.experimental.motion_generation as mg
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

if not args.test:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
    from isaacsim.gui.components.ui_utils import combo_floatfield_slider_builder

# ---------------------------------------------------------------------------
# Robot configuration
# Geometry from jetbot.usda (root xformOp:scale = 0.01; all values in metres).
# ---------------------------------------------------------------------------

ASSET_SUBPATH = "/Isaac/Robots_Multiphysics/NVIDIA/Jetbot/jetbot.usda"
PRIM_PATH = "/World/Jetbot"

LEFT_WHEEL_JOINT = "left_wheel_joint"
RIGHT_WHEEL_JOINT = "right_wheel_joint"
WHEEL_RADIUS = 0.03  # 30 mm
WHEEL_BASE = 0.1125  # 112.5 mm centre-to-centre

DEFAULT_SPEED = 0.05  # forward speed [m/s]
DEFAULT_YAW_RATE = 1.0  # yaw rate [rad/s]
MAX_SPEED = 3.0 * DEFAULT_SPEED
MAX_YAW_RATE = 3.0 * DEFAULT_YAW_RATE


async def _load_stage(usd_path: str, prim_path: str) -> Articulation:
    """Create a fresh stage, load the robot USD, return an Articulation handle."""
    await stage_utils.create_new_stage_async(template="default stage")
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)
    await app_utils.update_app_async()
    robot = Articulation(prim_path)
    await app_utils.update_app_async()
    SimulationManager.setup_simulation(dt=1.0 / 60.0)
    return robot


def main():
    assets_root_path = get_assets_root_path()
    robot = simulation_app.run_coroutine(_load_stage(assets_root_path + ASSET_SUBPATH, PRIM_PATH))

    print(f"Joint space: {robot.dof_names}")

    # <start-controller-snippet>
    controller = ctrl.DifferentialDriveController(
        robot_joint_space=robot.dof_names,  # every joint, not just the wheels
        left_wheel_joint=LEFT_WHEEL_JOINT,
        right_wheel_joint=RIGHT_WHEEL_JOINT,
        wheel_radius=WHEEL_RADIUS,  # metres
        wheel_base=WHEEL_BASE,  # metres, centre-to-centre
        max_linear_speed=MAX_SPEED,  # optional clamp [m/s]
        max_angular_speed=MAX_YAW_RATE,  # optional clamp [rad/s]
    )
    # <end-controller-snippet>

    if not args.test:
        # Two windows, stacked down the left of the viewport. Explicit positions
        # keep them from opening on top of each other at the default location.
        speed_window = ScrollingWindow(
            title="Speed (m/s)",
            width=420,
            height=100,
            position_x=40,
            position_y=120,
        )
        with speed_window.frame:
            speed_model, _ = combo_floatfield_slider_builder(
                label="Speed (m/s)",
                default_val=DEFAULT_SPEED,
                min=-MAX_SPEED,
                max=MAX_SPEED,
                step=0.01,
            )

        omega_window = ScrollingWindow(
            title="Angular rate (rad/s)",
            width=420,
            height=100,
            position_x=40,
            position_y=240,
        )
        with omega_window.frame:
            omega_model, _ = combo_floatfield_slider_builder(
                label="Angular rate (rad/s)",
                default_val=DEFAULT_YAW_RATE,
                min=-MAX_YAW_RATE,
                max=MAX_YAW_RATE,
                step=0.01,
            )

    app_utils.play()
    simulation_app.update()
    robot.set_world_poses(positions=wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32))
    simulation_app.update()

    ViewportManager.set_camera_view(
        ViewportManager.get_camera(),
        eye=[0.7, 0.7, 0.45],
        target=[0.0, 0.0, 0.05],
    )

    # <start-reset-snippet>
    controller.reset(mg.RobotState(), mg.RobotState(), 0.0)
    # <end-reset-snippet>

    frame_count = 0
    while simulation_app.is_running():
        simulation_app.update()

        if args.test:
            v = DEFAULT_SPEED
            omega = DEFAULT_YAW_RATE
        else:
            v = speed_model.get_value_as_float()
            omega = omega_model.get_value_as_float()

        # <start-setpoint-snippet>
        # Both linear and angular velocities are required on the control point site.
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[v, 0.0, 0.0]], dtype=wp.float32)),
                angular_velocities=(["control_point"], wp.array([[0.0, 0.0, omega]], dtype=wp.float32)),
            )
        )
        # <end-setpoint-snippet>

        # <start-apply-snippet>
        desired_state = controller.forward(mg.RobotState(), setpoint, 0.0)

        if desired_state is not None and desired_state.joints is not None:
            if desired_state.joints.velocities is not None:
                robot.set_dof_velocity_targets(
                    desired_state.joints.velocities,
                    dof_indices=robot.get_dof_indices(desired_state.joints.velocity_names),
                )
        # <end-apply-snippet>

        frame_count += 1
        if args.test and frame_count >= 10:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
