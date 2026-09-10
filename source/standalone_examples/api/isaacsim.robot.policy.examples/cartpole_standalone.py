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

"""Demonstrate Cartpole balancing with policy control.

The cartpole deploys through the generic :class:`RobotPolicyRunner` (bundled Cartpole spec, derived
binding, policy runtime). The policy observes no command channel — the runner is stepped
without a command and the effort-mode policy balances the pole indefinitely.
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
from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import PolicyEnvConfig, RobotPolicyRunner, get_cartpole_spec
from isaacsim.storage.native import get_assets_root_path

first_step = True
reset_needed = False
policy_failed = False

print(f"Using engine: {args.engine}")
print(f"Using device: {args.device}")


# Starting disturbance, drawn fresh on every reset. From its default state the cartpole is
# already balanced and the policy barely has to act, so the demo looks static. A leaning,
# tipping pole with the cart offset the other way makes the recovery obvious while staying
# inside what the policy handles: it drives the cart under the pole, catches it, and re-centers.
# The lean direction is random but the seed is fixed, so a run is still reproducible.
POLE_ANGLE_RANGE = (0.15, 0.30)  # rad on cart_to_pole, the passive joint
POLE_RATE_RANGE = (0.3, 0.8)  # rad/s, tipping further over so the correction has to be decisive
CART_POSITION_RANGE = (0.2, 0.5)  # m on slider_to_cart, the effort-commanded joint
RESET_PERIOD_SECONDS = 8.0  # replay the recovery on a loop
_disturbance_rng = np.random.default_rng(0)


def apply_initial_disturbance() -> None:
    """Tip the pole and offset the cart so the balancing policy has visible work to do."""
    lean = float(_disturbance_rng.choice((-1.0, 1.0)))
    dof_names = list(cartpole.articulation.dof_names)
    positions = [0.0] * len(dof_names)
    velocities = [0.0] * len(dof_names)
    positions[dof_names.index("cart_to_pole")] = lean * float(_disturbance_rng.uniform(*POLE_ANGLE_RANGE))
    velocities[dof_names.index("cart_to_pole")] = lean * float(_disturbance_rng.uniform(*POLE_RATE_RANGE))
    # Offset the cart away from the lean so it has to travel to get back under the pole.
    positions[dof_names.index("slider_to_cart")] = -lean * float(_disturbance_rng.uniform(*CART_POSITION_RANGE))
    cartpole.articulation.set_dof_positions([positions])
    cartpole.articulation.set_dof_velocities([velocities])


# initialize robot on first step, run robot advance
def on_physics_step(step_size: float, context: object) -> None:
    """Initialize, reset, or advance the Cartpole policy runner after a physics step.

    Args:
        step_size: Duration of the completed physics step.
        context: User context supplied when the callback was registered.
    """
    global first_step, reset_needed, policy_failed
    if policy_failed:
        return
    try:
        if first_step:
            # Build once so exported defaults are configured before applying the custom state.
            cartpole.initialize()
            cartpole.articulation.reset_to_default_state()
            apply_initial_disturbance()
            cartpole.step(step_size)
            first_step = False
        elif reset_needed:
            reset_needed = False
            first_step = True
        else:
            # The cartpole policy observes no command channel: step without a command.
            cartpole.step(step_size)
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        policy_failed = True
        carb.log_error(f"cartpole_standalone: policy deployment failed, stopping: {error}")


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

# spawn robot through the generic policy runner (bundled Cartpole spec)
spec = get_cartpole_spec()
cartpole = RobotPolicyRunner(spec, prim_path="/World/Cartpole")
cartpole.spawn()

# timing from the deployed artifact's env config (render cadence comes from render_interval)
timing = PolicyEnvConfig.from_file(spec.engines[args.engine].env_config_path).timing
frame_dt = timing.render_interval * timing.physics_dt
RenderingManager.set_dt(frame_dt)
reset_period_frames = max(1, round(RESET_PERIOD_SECONDS / frame_dt))
SimulationManager.set_physics_dt(cartpole.physics_dt)

# register physics callback
_physics_callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

# play simulation
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()

i = 0
while simulation_app.is_running():
    simulation_app.update()
    if SimulationManager.is_simulating():
        if i == reset_period_frames:
            i = 0
            if args.test is True:
                print("Cart pose: ", cartpole.articulation.get_world_poses()[0])
                break
            # Replay the recovery: the timeline stop restores the authored stage state and the
            # next physics step reinitializes the runner and draws a fresh disturbance.
            timeline.stop()
            timeline.play()
        i += 1
    else:
        reset_needed = True
timeline.stop()
SimulationManager.deregister_callback(_physics_callback_id)
cartpole.close()
simulation_app.close()
