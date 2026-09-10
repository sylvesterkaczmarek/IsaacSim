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

"""Drive a Kaya holonomic robot with interactive sliders.

Three sliders set the body twist: forward speed, strafe speed, and yaw rate. A holonomic
base can do all three at once, so any combination is valid.

Run with:  ./python.sh standalone_examples/api/isaacsim.robot_motion.controllers/holonomic.py

Pass --front-control-point to move the command site well ahead of the wheel base; the yaw
slider then pivots the robot about that point instead of about its centre. A small red
sphere marks the command site in either case.
"""

import argparse

# Parse args before SimulationApp so --test is available for the loop guard.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--front-control-point",
    default=False,
    action="store_true",
    help="Move the command site to the front of the wheel base, so yaw pivots about the front.",
)
parser.add_argument(
    "--test", default=False, action="store_true", help="Run in test mode (a few frames, no window interaction)"
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.controllers as ctrl
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import DomeLight, Sphere
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.robots import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

if not args.test:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
    from isaacsim.gui.components.ui_utils import combo_floatfield_slider_builder

WHEEL_JOINTS = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]

DEFAULT_FORWARD_SPEED = 0.2  # m/s along +x
DEFAULT_STRAFE_SPEED = 0.0  # m/s along +y
DEFAULT_YAW_RATE = 1.0  # rad/s about +z
MAX_LINEAR_SPEED = 0.6  # m/s
MAX_YAW_RATE = 3.0  # rad/s

# --front-control-point places the command site this many wheel-base radii ahead of the
# robot. >1 puts it clear of the chassis, so the pivot is unmistakable.
FRONT_CONTROL_POINT_SCALE = 2.5
CONTROL_POINT_MARKER_RADIUS = 0.025  # m

# ---------------------------------------------------------------------------- Scene ----
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

stage_utils.set_stage_up_axis("Z")
stage_utils.set_stage_units(meters_per_unit=1.0)
stage_utils.add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
    path="/World/ground",
)
DomeLight("/World/DomeLight").set_intensities(500)

stage_utils.add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/NVIDIA/Kaya/kaya.usda",
    path="/World/Kaya",
)
my_kaya = Articulation(
    "/World/Kaya", positions=[0.0, 0.0, 0.02], orientations=[1.0, 0.0, 0.0, 0.0], reset_xform_op_properties=True
)

# <start-usd-setup-snippet>
# Read the authored wheel geometry from the asset.
kaya_setup = HolonomicRobotUsdSetup(
    robot_prim_path="/World/Kaya",
    com_prim_path="/World/Kaya/base_link/control_offset",
)
wheel_radius, wheel_positions, wheel_orientations, mecanum_angles, wheel_axis, up_axis = (
    kaya_setup.get_holonomic_controller_params()
)
# <end-usd-setup-snippet>

# Where the twist command is interpreted. The wheel geometry above is measured in the
# control_offset prim's frame, so the site pose is given in that same frame: the default
# of zero puts the site on that prim, at the centre of the wheel base.
# <start-command-site-snippet>
command_site_position = np.zeros(3)
if args.front_control_point:
    wheel_base_radius = float(np.max(np.linalg.norm(wheel_positions, axis=1)))
    command_site_position = np.array([FRONT_CONTROL_POINT_SCALE * wheel_base_radius, 0.0, 0.0])
# <end-command-site-snippet>
print(f"Command site (control_offset frame): {np.round(command_site_position, 4).tolist()} m")

# Visual marker for the command site, parented under control_offset so it rides with the
# robot. Purely visual — no collision or physics API is applied, so it does not affect
# the articulation's dynamics.
Sphere(
    "/World/Kaya/base_link/control_offset/command_site_marker",
    radii=CONTROL_POINT_MARKER_RADIUS,
    colors=(1.0, 0.0, 0.0),
    translations=command_site_position,
)

# Start physics, then build the controller: it needs the articulation's dof_names.
SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
app_utils.play()
app_utils.update_app(steps=10)

# The Kaya is only ~0.3 m across, so the default camera leaves it a speck. Frame it
# closely enough to see the wheels, with room to drive around before it leaves view.
ViewportManager.set_camera_view(
    ViewportManager.get_camera(),
    eye=[0.7, 0.7, 0.5],
    target=[0.0, 0.0, 0.05],
)

# <start-controller-snippet>
controller = ctrl.HolonomicController(
    robot_joint_space=list(my_kaya.dof_names),
    wheel_joint_names=WHEEL_JOINTS,
    wheel_radius=wheel_radius,
    wheel_positions=wheel_positions,
    wheel_orientations=wheel_orientations,
    mecanum_angles=mecanum_angles,  # authored per wheel; 90 on the Kaya, i.e. plain omni wheels
    wheel_axis=wheel_axis,
    rotation_direction=up_axis,
    command_site_position=command_site_position,
)
# <end-controller-snippet>

# <start-reset-snippet>
controller.reset(mg.RobotState(), mg.RobotState(), 0.0)
# <end-reset-snippet>

# -------------------------------------------------------------------------- Sliders ----
# One window per slider, stacked down the left of the viewport. Explicit positions keep
# them from opening on top of each other at the default location.
if not args.test:
    forward_window = ScrollingWindow(title="Forward speed (m/s)", width=420, height=100, position_x=40, position_y=120)
    with forward_window.frame:
        forward_model, _ = combo_floatfield_slider_builder(
            label="Forward (+x)", default_val=DEFAULT_FORWARD_SPEED, min=-MAX_LINEAR_SPEED, max=MAX_LINEAR_SPEED
        )

    strafe_window = ScrollingWindow(title="Strafe speed (m/s)", width=420, height=100, position_x=40, position_y=240)
    with strafe_window.frame:
        strafe_model, _ = combo_floatfield_slider_builder(
            label="Strafe (+y)", default_val=DEFAULT_STRAFE_SPEED, min=-MAX_LINEAR_SPEED, max=MAX_LINEAR_SPEED
        )

    yaw_window = ScrollingWindow(title="Yaw rate (rad/s)", width=420, height=100, position_x=40, position_y=360)
    with yaw_window.frame:
        yaw_model, _ = combo_floatfield_slider_builder(
            label="Yaw (+z)", default_val=DEFAULT_YAW_RATE, min=-MAX_YAW_RATE, max=MAX_YAW_RATE
        )

# ----------------------------------------------------------------------------- Loop ----
frame_count = 0
while simulation_app.is_running():
    simulation_app.update()
    if app_utils.is_playing():
        if args.test:
            command = np.array([DEFAULT_FORWARD_SPEED, DEFAULT_STRAFE_SPEED, DEFAULT_YAW_RATE])
        else:
            command = np.array(
                [
                    forward_model.get_value_as_float(),
                    strafe_model.get_value_as_float(),
                    yaw_model.get_value_as_float(),
                ]
            )

        # <start-setpoint-snippet>
        site = "control_point"
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=[site],
                linear_velocities=([site], wp.array([[command[0], command[1], 0.0]], dtype=wp.float32)),
                angular_velocities=([site], wp.array([[0.0, 0.0, command[2]]], dtype=wp.float32)),
            )
        )
        # <end-setpoint-snippet>
        # <start-apply-snippet>
        desired = controller.forward(None, setpoint, 0.0)
        if desired is not None and desired.joints is not None:
            my_kaya.set_dof_velocity_targets(
                desired.joints.velocities,
                dof_indices=my_kaya.get_dof_indices(desired.joints.velocity_names),
            )
        # <end-apply-snippet>
    frame_count += 1
    if args.test and frame_count >= 10:
        break

app_utils.stop()
simulation_app.close()
