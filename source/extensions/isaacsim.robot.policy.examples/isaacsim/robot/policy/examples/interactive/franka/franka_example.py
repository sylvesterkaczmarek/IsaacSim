# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Example demonstration of a Franka robot performing an open drawer task using a policy-based approach."""

from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from isaacsim.robot.policy.examples import (
    PolicyEnvConfig,
    RobotPolicyRunner,
    get_franka_spec,
    make_franka_task_state_provider,
)
from isaacsim.robot.policy.examples.interactive.example_base import PolicySampleBase
from isaacsim.storage.native import get_assets_root_path


class FrankaExample(PolicySampleBase):
    """Franka open-drawer policy example over the bundled spec and cabinet task state.

    Creates a ground plane, a Sektion cabinet, and a Franka deployed through the generic
    ``RobotPolicyRunner`` (explicit binding + task state provider + controller) with the
    caller-owned cabinet articulation. The robot initializes on the first physics step
    after play and the scene auto-resets at the exported episode duration. Each physics
    engine selects its corresponding policy artifact and exported environment configuration.
    """

    physics_callback_event = IsaacEvents.PRE_PHYSICS_STEP

    def __init__(self) -> None:
        super().__init__()
        engine = (SimulationManager.get_active_physics_engine() or "").lower()
        self._spec = get_franka_spec()
        self._env_config = PolicyEnvConfig.from_file(self._spec.engines[engine].env_config_path)
        timing = self._env_config.timing
        self._world_settings["physics_dt"] = timing.physics_dt
        self._world_settings["rendering_dt"] = timing.render_interval * timing.physics_dt

        self.franka = None
        self.cabinet = None
        self._time_elapsed = 0.0

    def setup_scene(self) -> None:
        """Set up the scene with robot, cabinet, and environment."""
        add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/defaultGroundPlane",
        )

        # Cabinet articulation, owned by the caller and consumed by the task state provider.
        cabinet_prim_path = "/World/cabinet"
        cabinet_usd_path = self._env_config.scene_entity_usd_path("cabinet")
        if cabinet_usd_path is None:
            raise ValueError("Franka policy env config does not define scene.cabinet.spawn.usd_path.")
        add_reference_to_stage(cabinet_usd_path, cabinet_prim_path)
        self.cabinet = Articulation(paths=cabinet_prim_path, reset_xform_op_properties=True)
        cabinet_position, cabinet_orientation = self._env_config.scene_entity_root_pose("cabinet")
        if cabinet_position is None or cabinet_orientation is None:
            raise ValueError("Franka policy env config does not define the cabinet initial root pose.")
        self.cabinet.set_world_poses([cabinet_position], [cabinet_orientation])

        # Author the Franka through the runner; policy initialization happens on the first physics step.
        self.franka = RobotPolicyRunner(
            self._spec,
            prim_path="/World/franka",
            task_state_provider=make_franka_task_state_provider(self.cabinet, self._env_config),
        )
        self.franka.spawn()
        applied_materials = self.franka.apply_scene_properties({"cabinet": self.cabinet})
        if applied_materials != {"robot", "cabinet"}:
            raise ValueError("The exported policy config is missing the robot or drawer-handle startup material event.")
        self._episode_length_s = self._env_config.episode_length_s
        print("Scene setup complete with Franka robot and cabinet")

    async def setup_post_load(self) -> None:
        """Set up the physics callback after initial load."""
        self._physics_ready = False
        self._register_physics_callback()
        print("Franka open drawer scene loaded successfully")

    async def setup_pre_reset(self) -> None:
        """Reset the physics-ready flag and elapsed time before a world reset."""
        await super().setup_pre_reset()
        self._time_elapsed = 0.0

    async def setup_post_reset(self) -> None:
        """Reset flags after a reset; the runner's replay reset clears the previous action."""
        await super().setup_post_reset()
        self._time_elapsed = 0.0

    def on_physics_step(self, dt: float, context: object) -> None:
        """Initialize the runner on the first step, then step the policy and reset in place.

        Args:
            dt: Time delta for the physics step.
            context: Physics step context information.
        """
        if self.franka is None or self.franka.articulation is None:
            return

        if self._physics_ready:
            self._time_elapsed += dt
            if self._time_elapsed >= self._episode_length_s:
                self._time_elapsed = 0.0
                self.franka.reset()
                print(f"Simulation reset at {self._episode_length_s:g} seconds")
                return

        # If the physics tensors were invalidated, reinitialize on this step.
        if not self.franka.articulation.is_physics_tensor_entity_valid():
            self._physics_ready = False

        if self._physics_ready:
            self.franka.step(dt)
        else:
            # First physics step after play - initialize the robot and (re)start the runtime.
            self._physics_ready = True
            self.franka.initialize()
            # PhysX stabilization/sleep would freeze the near-still arm mid-task; the drawer
            # example disables both, as the 6.x class did.
            self.franka.articulation.set_stabilization_thresholds([0.0])
            self.franka.articulation.set_sleep_thresholds([0.0])

    def physics_cleanup(self) -> None:
        """Clean up physics resources."""
        self._deregister_physics_callback()
        if self.franka is not None:
            self.franka.close()
        self.franka = None
        self.cabinet = None
        self._physics_ready = False
        self._time_elapsed = 0.0
