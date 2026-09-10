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

"""Demonstrate Go2 robot simulation with policy control.

The robot deploys through the generic :class:`RobotPolicyRunner` (bundled Go2 spec, derived
binding, policy runtime) and follows a scripted velocity trajectory. For keyboard control,
use the interactive Go2 example in the examples browser.
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Select simulation engine and device.")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Simulation device")
parser.add_argument("--engine", type=str, choices=["physx", "newton"], default="physx", help="Physics engine")

args, unknown = parser.parse_known_args()
extra_args = [f"--/exts/isaacsim.core.simulation_manager/default_engine={args.engine}"]
if args.engine == "newton":
    extra_args.extend(["--enable", "isaacsim.physics.newton", "--enable", "isaacsim.physics.newton.tensors"])
simulation_app = SimulationApp({"headless": False, "extra_args": extra_args})

import carb
import numpy as np
import omni.timeline
from command_path import TraveledPath, author_command_path, phase_boundary_frames, report_tracking
from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import PolicyEnvConfig, RobotPolicyRunner, get_go2_spec
from isaacsim.storage.native import get_assets_root_path

first_step = True
reset_needed = False
policy_failed = False

print(f"Using engine: {args.engine}")
print(f"Using device: {args.device}")


# initialize robot on first step, run robot advance
def on_physics_step(step_size: float, context: object) -> None:
    """Initialize, reset, or advance the Go2 policy runner after a physics step.

    Args:
        step_size: Duration of the completed physics step.
        context: User context supplied when the callback was registered.
    """
    global first_step, reset_needed, policy_failed
    if policy_failed:
        return
    try:
        if first_step:
            go2.restart_from_default_state(base_command)
            first_step = False
        elif reset_needed:
            reset_needed = False
            first_step = True
        else:
            go2.step(step_size, base_command)
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        policy_failed = True
        carb.log_error(f"go2_standalone: policy deployment failed, stopping: {error}")


# spawn world
set_stage_up_axis("Z")
set_stage_units(meters_per_unit=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn ground scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
prim.GetReferences().AddReference(asset_path)

# spawn physics scene
# TODO: physics scene should be created by simulation manager
define_prim("/World/PhysicsScene", "PhysicsScene")

# select the simulation device before constructing the policy articulation
SimulationManager.set_physics_sim_device(args.device)

# spawn robot through the generic policy runner (bundled Go2 spec)
spec = get_go2_spec()
go2 = RobotPolicyRunner(spec, prim_path="/World/Go2", position=[0.0, 0.0, 0.5])
go2.spawn()

# timing from the deployed artifact's env config (render cadence comes from render_interval)
timing = PolicyEnvConfig.from_file(spec.engines[args.engine].env_config_path).timing
frame_dt = timing.render_interval * timing.physics_dt
RenderingManager.set_dt(frame_dt)
SimulationManager.set_physics_dt(go2.physics_dt)

# scripted command loop: (app frames, [vx, vy, yaw_rate]), one command held per frame. The last
# three phases turn toward the spawn point, walk back to it, and restore the spawn heading, so the
# commanded course closes on itself and the robot patrols the same circuit every lap.
COMMAND_PHASES = [
    (55, [1.1, 0.0, 0.0]),  # forward
    (32, [0.0, 0.7, 0.0]),  # strafe left
    (75, [0.9, 0.0, 1.2]),  # arc left
    (42, [1.0, 0.0, 0.0]),  # forward
    (75, [0.9, 0.0, 1.2]),  # arc left
    (26, [0.0, -0.6, 0.0]),  # strafe right
    (42, [1.0, 0.0, 0.0]),  # forward
    (75, [0.9, 0.0, 1.2]),  # arc left
    (18, [0.0, 0.0, -1.1]),  # turn toward the spawn point
    (69, [1.1, 0.0, 0.0]),  # return leg
    (58, [0.0, 0.0, 1.1]),  # turn back to the spawn heading
]
frame_commands = np.concatenate([np.tile(np.asarray(c, dtype=np.float32), (n, 1)) for n, c in COMMAND_PHASES])
phase_boundaries = phase_boundary_frames(COMMAND_PHASES)
# green: the commanded course, with a waypoint arrow per phase; red: where the robot actually went
commanded_end = author_command_path(COMMAND_PHASES, start_position=(0.0, 0.0), frame_dt=frame_dt)
traveled = TraveledPath()
# robot command
base_command = np.zeros(3, dtype=np.float32)

# register physics callback
_physics_callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

# play simulation
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
            if args.test is True:
                report_tracking(commanded_end, traveled)
                break
        base_command = frame_commands[i]
        # the commanded course describes the first circuit, so only mark its phase boundaries
        traveled.record(go2.articulation, mark=(loop_index == 0 and i in phase_boundaries))
        i += 1
    else:
        reset_needed = True
timeline.stop()
SimulationManager.deregister_callback(_physics_callback_id)
go2.close()
simulation_app.close()
