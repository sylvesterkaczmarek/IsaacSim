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

"""Drive a Franka arm to a draggable target with the cuMotion RMPflow controller.

Select the red target cube in the viewport and move it with the translate gizmo; the arm
reaches for it continuously and bends around the grey obstacle on the way.

Run with:
    ./python.sh standalone_examples/api/isaacsim.robot_motion.cumotion/rmpflow_follow_target.py
"""

import argparse

# Parse args before SimulationApp so --test is available for the loop guard.
_parser = argparse.ArgumentParser(description="cuMotion RMPflow follow-target example")
_parser.add_argument("--test", default=False, action="store_true", help="Run a few frames, then exit.")
args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim
from isaacsim.core.experimental.utils import transform as transform_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.experimental.motion_generation import (
    ObstacleConfiguration,
    ObstacleStrategy,
    SceneQuery,
    TrackableApi,
    WorldBinding,
)
from isaacsim.storage.native import get_assets_root_path

ROBOT_PRIM_PATH = "/World/Franka"
TARGET_PRIM_PATH = "/World/target"
OBSTACLE_PRIM_PATH = "/World/obstacle"
PHYSICS_DT = 1.0 / 60.0

# ---------------------------------------------------------------------------- Scene ----

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

stage_utils.create_new_stage(template="default stage")
stage_utils.set_stage_up_axis("Z")
stage_utils.set_stage_units(meters_per_unit=1.0)

stage_utils.add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
    path=ROBOT_PRIM_PATH,
)
robot = Articulation(ROBOT_PRIM_PATH)

# <start-target-snippet>
# The goal is an ordinary visual cube — drag it in the viewport to move the goal.
target = Cube(paths=TARGET_PRIM_PATH, sizes=0.04, positions=[0.5, 0.0, 0.7], colors=(1.0, 0.0, 0.0))
# Point the end effector down at the target rather than at it side-on.
target.set_world_poses(orientations=np.array([transform_utils.euler_angles_to_quaternion([0, np.pi, 0]).numpy()]))
# <end-target-snippet>

# <start-obstacle-snippet>
# Also an ordinary cube. What makes it an obstacle is the collision API, which is
# what the scene search below looks for.
Cube(OBSTACLE_PRIM_PATH, sizes=0.05, positions=[np.array([0.4, 0.0, 0.45])], colors=(0.4, 0.4, 0.4))
GeomPrim(OBSTACLE_PRIM_PATH, apply_collision_apis=True)
# <end-obstacle-snippet>

SimulationManager.setup_simulation(dt=PHYSICS_DT)
app_utils.play()
app_utils.update_app(steps=10)

ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=[1.8, 1.4, 1.4], target=[0.3, 0.0, 0.4])

# ------------------------------------------------------------------- Planning world ----

# <start-robot-config-snippet>
# cuMotion ships tuned configurations for supported robots, so you do not have to
# describe the arm's kinematics or collision spheres yourself.
robot_config = load_cumotion_supported_robot("franka")
# <end-robot-config-snippet>

# <start-scene-query-snippet>
# SceneQuery finds prims on the stage that the planner should treat as obstacles.
# Here: everything with a collision API in a 20 m box around the robot, minus the robot.
robot_positions, robot_orientations = robot.get_world_poses()
obstacles = SceneQuery().get_prims_in_aabb(
    search_box_origin=robot_positions.numpy()[0],
    search_box_minimum=[-10.0, -10.0, -10.0],
    search_box_maximum=[10.0, 10.0, 10.0],
    tracked_api=TrackableApi.PHYSICS_COLLISION,
    exclude_prim_paths=[ROBOT_PRIM_PATH],
)
# <end-scene-query-snippet>

# <start-obstacle-strategy-snippet>
# ObstacleStrategy says how each kind of shape is represented for planning, and how
# much clearance to keep around it. Our obstacle is a Cube, so configure that: keep
# its cube shape, and hold the arm 2 cm clear of it.
obstacle_strategy = ObstacleStrategy()
obstacle_strategy.set_default_configuration(Cube, ObstacleConfiguration("cube", 0.02))
# <end-obstacle-strategy-snippet>

# <start-world-binding-snippet>
# WorldBinding is the live link between the USD stage and the planner's collision
# world: it builds that world once, then keeps it in step as prims move.
world_binding = WorldBinding(
    world_interface=CumotionWorldInterface(device="cpu"),
    obstacle_strategy=obstacle_strategy,
    tracked_prims=obstacles,
    tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
)

# populate the cuMotion planning world:
world_binding.initialize()

# update the robot position in the planning world:
world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_positions, robot_orientations))

# Run this whenever you want to synchronize
# the poses of obstacles in the planning world
# to the USD scene.
world_binding.synchronize_transforms()
# <end-world-binding-snippet>

# ------------------------------------------------------------------------ Controller ----

# <start-controller-snippet>
# Sites are the named frames you can command. The robot configuration lists the ones
# cuMotion knows about; the first is the arm's tool frame (its end effector).
robot_joint_space = robot.dof_names
robot_site_space = robot_config.robot_description.tool_frame_names()
tool_frame = robot_site_space[0]

controller = RmpFlowController(
    cumotion_robot=robot_config,
    cumotion_world_interface=world_binding.get_world_interface(),
    robot_joint_space=robot_joint_space,
    robot_site_space=robot_site_space,
    tool_frame=tool_frame,
)
# <end-controller-snippet>


# <start-setpoint-snippet>
def read_target_setpoint() -> mg.RobotState:
    """Wrap the target cube's current pose as a setpoint on the tool frame."""
    positions, orientations = target.get_world_poses()
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=robot_site_space,
            positions=([tool_frame], positions),
            orientations=([tool_frame], orientations),
        )
    )


# <end-setpoint-snippet>

# <start-reset-snippet>
# reset() seeds the controller with where the robot actually is. It is the only place
# the measured joint state is needed — forward() runs open-loop from there.
measured_state = mg.RobotState(
    joints=mg.JointState.from_name(
        robot_joint_space=robot_joint_space,
        positions=(robot_joint_space, robot.get_dof_positions()),
        velocities=(robot_joint_space, robot.get_dof_velocities()),
    )
)
if not controller.reset(measured_state, read_target_setpoint(), 0.0):
    raise RuntimeError("RmpFlowController failed to reset.")
# <end-reset-snippet>

# ------------------------------------------------------------------------------ Loop ----

# <start-loop-snippet>
sim_time = 0.0
frame_count = 0

while simulation_app.is_running():
    simulation_app.update()

    # Push any prim movement into the planning world, then solve for this step.
    world_binding.synchronize_transforms()
    desired_state = controller.forward(None, read_target_setpoint(), sim_time)

    if desired_state is not None and desired_state.joints is not None and desired_state.joints.positions is not None:
        robot.set_dof_position_targets(
            positions=desired_state.joints.positions,
            dof_indices=robot.get_dof_indices(desired_state.joints.position_names),
        )

    sim_time += PHYSICS_DT
    frame_count += 1
    if args.test and frame_count >= 10:
        break
# <end-loop-snippet>

simulation_app.close()
