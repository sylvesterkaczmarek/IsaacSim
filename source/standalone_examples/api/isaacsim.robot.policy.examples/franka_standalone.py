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

"""Demonstrate the Franka open-drawer manipulation policy.

The robot deploys through the generic :class:`RobotPolicyRunner` (bundled Franka spec, explicit
binding, policy runtime) against a caller-owned Sektion cabinet articulation whose task state
feeds the policy through the bundled task-state provider. The policy observes no command
channel — the runner is stepped without a command and the policy opens the top drawer, then the
scene auto-resets at the exported episode duration. Each physics engine selects its corresponding
policy artifact and exported environment configuration.
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
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.stage import (
    add_reference_to_stage,
    define_prim,
    set_stage_units,
    set_stage_up_axis,
)
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import (
    PolicyEnvConfig,
    RobotPolicyRunner,
    get_franka_spec,
    make_franka_task_state_provider,
)
from isaacsim.storage.native import get_assets_root_path

first_step = True
reset_needed = False
policy_failed = False

print(f"Using engine: {args.engine}")
print(f"Using device: {args.device}")


# initialize robot on first step, run robot advance
def on_physics_step(step_size: float, context: object) -> None:
    """Initialize, reset, or advance the Franka policy runner before a physics step.

    Args:
        step_size: Duration of the completed physics step.
        context: User context supplied when the callback was registered.
    """
    global first_step, reset_needed, policy_failed
    if policy_failed:
        return
    try:
        if first_step:
            franka.initialize()
            # PhysX stabilization/sleep would freeze the near-still arm mid-task; the drawer
            # example disables both, as the 6.x class did.
            franka.articulation.set_stabilization_thresholds([0.0])
            franka.articulation.set_sleep_thresholds([0.0])
            first_step = False
        elif reset_needed:
            franka.reset()
            reset_needed = False
        else:
            # The drawer policy observes no command channel: step without a command.
            franka.step(step_size)
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        policy_failed = True
        carb.log_error(f"franka_standalone: policy deployment failed, stopping: {error}")


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

# Cabinet articulation, owned by the caller and consumed by the task state provider.
spec = get_franka_spec()
env_config = PolicyEnvConfig.from_file(spec.engines[args.engine].env_config_path)
cabinet_prim_path = "/World/cabinet"
cabinet_usd_path = env_config.scene_entity_usd_path("cabinet")
if cabinet_usd_path is None:
    raise ValueError("Franka policy env config does not define scene.cabinet.spawn.usd_path.")
add_reference_to_stage(cabinet_usd_path, cabinet_prim_path)
cabinet = Articulation(paths=cabinet_prim_path, reset_xform_op_properties=True)
cabinet_position, cabinet_orientation = env_config.scene_entity_root_pose("cabinet")
if cabinet_position is None or cabinet_orientation is None:
    raise ValueError("Franka policy env config does not define the cabinet initial root pose.")
cabinet.set_world_poses([cabinet_position], [cabinet_orientation])

# spawn robot through the generic policy runner (bundled Franka spec + cabinet task state)
franka = RobotPolicyRunner(
    spec,
    prim_path="/World/franka",
    task_state_provider=make_franka_task_state_provider(cabinet, env_config),
)
franka.spawn()
applied_materials = franka.apply_scene_properties({"cabinet": cabinet})
if applied_materials != {"robot", "cabinet"}:
    raise ValueError("The exported policy config is missing the robot or drawer-handle startup material event.")

# timing from the deployed artifact's env config (render cadence comes from render_interval)
timing = env_config.timing
RenderingManager.set_dt(timing.render_interval * timing.physics_dt)
SimulationManager.set_physics_dt(franka.physics_dt)

# register physics callback
_physics_callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.PRE_PHYSICS_STEP)

# play simulation
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()

drawer_index = None
max_drawer_opening = 0.0
i = 0
while simulation_app.is_running():
    simulation_app.update()
    if SimulationManager.is_simulating():
        if drawer_index is None:
            drawer_index = cabinet.get_dof_indices("drawer_top_joint")
        drawer_opening = float(np.asarray(cabinet.get_dof_positions(dof_indices=drawer_index).numpy()).reshape(-1)[0])
        max_drawer_opening = max(max_drawer_opening, drawer_opening)
        if i >= round(env_config.episode_length_s / timing.physics_dt):
            i = 0
            if args.test is True:
                print("Max drawer opening: ", max_drawer_opening)
                break
            reset_needed = True
            max_drawer_opening = 0.0
        i += 1
    else:
        reset_needed = True
timeline.stop()
SimulationManager.deregister_callback(_physics_callback_id)
franka.close()
simulation_app.close()
