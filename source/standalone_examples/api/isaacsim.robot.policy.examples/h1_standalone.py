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

"""Demonstrate H1 humanoid robot simulation with policy control.

Each robot deploys through the generic :class:`RobotPolicyRunner` (bundled H1 spec, derived
binding, policy runtime) instead of the legacy ``H1FlatTerrainPolicy`` class.
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Define the number of robots and select simulation settings.")
parser.add_argument("--num-robots", type=int, default=1, help="Number of robots (default: 1)")
parser.add_argument(
    "--env-url",
    default="/Isaac/Environments/Grid/default_environment.usd",
    required=False,
    help="Path to the environment url",
)
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Simulation device")
parser.add_argument("--engine", type=str, choices=["physx", "newton"], default="physx", help="Physics engine")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

args, unknown = parser.parse_known_args()
extra_args = [f"--/exts/isaacsim.core.simulation_manager/default_engine={args.engine}"]
if args.engine == "newton":
    extra_args.extend(["--enable", "isaacsim.physics.newton", "--enable", "isaacsim.physics.newton.tensors"])
simulation_app = SimulationApp({"headless": False, "extra_args": extra_args})

import carb
import numpy as np
import omni.timeline
from command_path import TraveledPath, author_command_path, phase_boundary_frames, report_tracking
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import RobotPolicyRunner, get_h1_spec
from isaacsim.storage.native import get_assets_root_path

print(f"Number of robots: {args.num_robots}")
print(f"Using engine: {args.engine}")
print(f"Using device: {args.device}")

first_step = True
reset_needed = False
policy_failed = False
robots = []


# initialize robot on first step, run robot advance
def on_physics_step(step_size: float, context: object) -> None:
    """Initialize, reset, or advance the H1 policy runners after a physics step.

    Args:
        step_size: Duration of the completed physics step.
        context: User context supplied when the callback was registered.
    """
    global first_step, reset_needed, policy_failed
    if policy_failed:
        return
    try:
        if first_step:
            for robot in robots:
                robot.restart_from_default_state(base_command)
            first_step = False
        elif reset_needed:
            reset_needed = False
            first_step = True
        else:
            for robot in robots:
                robot.step(step_size, base_command)
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        policy_failed = True
        carb.log_error(f"h1_standalone: policy deployment failed, stopping: {error}")


assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + args.env_url
prim.GetReferences().AddReference(asset_path)

# spawn physics scene
# TODO: physics scene should be created by simulation manager
define_prim("/World/PhysicsScene", "PhysicsScene")

# set rendering manager
frame_dt = 8.0 / 200.0
RenderingManager.set_dt(frame_dt)

# spawn simulation manager
SimulationManager.set_physics_sim_device(args.device)
SimulationManager.set_physics_dt(1.0 / 200.0)

# scripted command loop: (app frames, [vx, vy, yaw_rate]), one command held per frame. The last
# three phases turn toward the spawn point, walk back to it, and restore the spawn heading, so the
# commanded course closes on itself and the robot patrols the same circuit every lap.
# H1's policy is trained on forward and turning commands only, so this course has no strafe leg.
COMMAND_PHASES = [
    (32, [0.5, 0.0, 0.0]),  # forward
    (54, [0.45, 0.0, 0.7]),  # arc left
    (28, [0.5, 0.0, 0.0]),  # forward
    (54, [0.45, 0.0, 0.7]),  # arc left
    (28, [0.5, 0.0, 0.0]),  # forward
    (54, [0.45, 0.0, 0.7]),  # arc left
    (22, [0.0, 0.0, 0.6]),  # turn toward the spawn point
    (74, [0.5, 0.0, 0.0]),  # return leg
    (51, [0.0, 0.0, 0.6]),  # turn back to the spawn heading
]
frame_commands = np.concatenate([np.tile(np.asarray(c, dtype=np.float32), (n, 1)) for n, c in COMMAND_PHASES])
phase_boundaries = phase_boundary_frames(COMMAND_PHASES)
commanded_ends = []
traveled_paths = []
# robot command
base_command = np.zeros(3, dtype=np.float32)

# spawn robots through the generic policy runner (bundled H1 spec)
for i in range(0, args.num_robots):
    h1 = RobotPolicyRunner(
        get_h1_spec(),
        prim_path="/World/H1_" + str(i),
        position=[0.0, float(i), 1.05],
    )
    h1.spawn()
    # green: this robot's commanded course from its spawn row; red: where it actually went
    commanded_ends.append(
        author_command_path(
            COMMAND_PHASES,
            start_position=(0.0, float(i)),
            frame_dt=frame_dt,
            prim_path=f"/World/CommandedCourse_{i}",
        )
    )
    traveled_paths.append(TraveledPath(prim_path=f"/World/TraveledPath_{i}"))

    robots.append(h1)

_physics_callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()

i = 0
loop_index = 0
while simulation_app.is_running():
    simulation_app.update()

    if SimulationManager.is_simulating():
        if i == len(frame_commands):
            i = 0
            loop_index += 1
            if args.test:
                report_tracking(commanded_ends[0], traveled_paths[0])
                break
        base_command = frame_commands[i]
        # the commanded course describes the first circuit, so only mark its phase boundaries
        for robot, traveled in zip(robots, traveled_paths):
            traveled.record(robot.articulation, mark=(loop_index == 0 and i in phase_boundaries))
        i += 1
    else:
        reset_needed = True
timeline.stop()
SimulationManager.deregister_callback(_physics_callback_id)
for robot in robots:
    robot.close()
simulation_app.close()
