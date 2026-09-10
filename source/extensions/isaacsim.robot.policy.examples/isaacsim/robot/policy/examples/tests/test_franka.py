# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Behavioral tests for the bundled Franka open-drawer policy on the runner deployment path.

These are the retained Franka drawer suites of the refactor's verification matrix: the same
scene construction and drawer-opening criterion that drove the removed
``FrankaOpenDrawerPolicy`` class now drive the generic ``RobotPolicyRunner`` with the bundled Franka
spec and its explicit-reference task state provider.
"""

import asyncio

import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.timeline
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples.bundled.franka import get_franka_spec, make_franka_task_state_provider
from isaacsim.robot.policy.examples.env_config import PolicyEnvConfig
from isaacsim.robot.policy.examples.runtime import RobotPolicyRunner
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics


class TestFrankaExampleExtension(omni.kit.test.AsyncTestCase):
    """Test suite for the Franka open-drawer runner deployment.

    This test class validates the Franka drawer-opening policy by setting up a simulation
    environment with a Franka robot and cabinet, then verifying the robot can successfully open
    a drawer. The tests ensure proper robot initialization, cabinet interaction, and physics
    simulation behavior on the runner runtime path.

    The test suite creates a physics scene with a ground plane and Sektion cabinet, deploys a
    ``RobotPolicyRunner`` over the bundled Franka spec against the caller-owned cabinet articulation,
    and validates that the robot can open the cabinet drawer to a specified threshold. It uses
    each policy artifact's exported physics rate.
    """

    def get_device(self) -> str:
        """Return the simulation device. Override in subclasses.

        Returns:
            The simulation device string.
        """
        return "cpu"

    async def setUp(self) -> None:
        """Set up the test environment with physics scene, ground plane, and cabinet."""
        engine = (SimulationManager.get_active_physics_engine() or "").lower()
        self._engine = engine
        await stage_utils.create_new_stage_async()
        # This needs to be set so that kit updates match physics updates
        self._franka = None
        self._physics_callback_id = None

        device_str = self.get_device()
        print(f"Setting up test with device: {device_str}")

        self._spec = get_franka_spec()
        self._env_config = PolicyEnvConfig.from_file(self._spec.engines[engine].env_config_path)
        self._physics_dt = self._env_config.timing.physics_dt
        stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")

        # spawn simulation manager
        SimulationManager.set_physics_sim_device(device_str)
        SimulationManager.set_physics_dt(self._physics_dt)

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        cabinet_prim_path = "/World/cabinet"
        cabinet_usd_path = self._env_config.scene_entity_usd_path("cabinet")
        if cabinet_usd_path is None:
            raise ValueError("Franka policy env config does not define scene.cabinet.spawn.usd_path.")

        stage_utils.add_reference_to_stage(cabinet_usd_path, cabinet_prim_path)

        self.cabinet = Articulation(cabinet_prim_path, reset_xform_op_properties=True)
        cabinet_position, cabinet_orientation = self._env_config.scene_entity_root_pose("cabinet")
        if cabinet_position is None or cabinet_orientation is None:
            raise ValueError("Franka policy env config does not define the cabinet initial root pose.")
        self.cabinet.set_world_poses([cabinet_position], [cabinet_orientation])

        self._timeline = omni.timeline.get_timeline_interface()
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Clean up test environment by stopping timeline and deregistering callbacks."""
        await omni.kit.app.get_app().next_update_async()
        self._timeline.stop()
        if self._franka is not None:
            self._franka.close()
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    async def test_franka_add(self) -> None:
        """Test adding a Franka robot to the scene and verify its properties."""
        await self.spawn_franka()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._franka.articulation.num_dofs, 9)

        # Verify robot prim exists
        robot_prim = stage_utils.get_current_stage().GetPrimAtPath("/World/franka")
        self.assertIsNotNone(robot_prim, "Robot prim should exist in stage at /World/franka")
        self.assertTrue(robot_prim.IsValid(), "Robot prim should be valid")
        self.assertTrue(
            prim_utils.has_api(robot_prim, UsdPhysics.ArticulationRootAPI),
            "Robot base prim should have ArticulationRootAPI",
        )

        # Verify cabinet prim exists
        cabinet_prim = stage_utils.get_current_stage().GetPrimAtPath("/World/cabinet")
        self.assertIsNotNone(cabinet_prim, "Cabinet prim should exist in stage at /World/cabinet")
        self.assertTrue(cabinet_prim.IsValid(), "Cabinet prim should be valid")

    async def test_franka_open_drawer(self) -> None:
        """Test the Franka robot's ability to open a drawer through the runner deployment."""
        await self.spawn_franka()
        await omni.kit.app.get_app().next_update_async()
        drawer_link_idx = self.cabinet.get_dof_indices("drawer_top_joint")
        min_drawer_opening = 0.30
        drawer_openings = []
        for _ in range(480):
            await omni.kit.app.get_app().next_update_async()
            position = self.cabinet.get_dof_positions(indices=[0], dof_indices=drawer_link_idx)
            drawer_openings.append(float(position.numpy()[0][0]))

        first_open_step = next(
            (index for index, opening in enumerate(drawer_openings) if opening > min_drawer_opening), None
        )
        self.assertIsNotNone(first_open_step, f"Expected drawer to open past {min_drawer_opening}")
        if self._engine == "physx":
            self.assertGreater(
                min(drawer_openings[first_open_step:]),
                min_drawer_opening,
                f"Expected drawer to remain open past {min_drawer_opening}",
            )
        hand_index = self._franka.articulation.link_names.index("panda_hand")
        hand = RigidPrim(self._franka.articulation.link_paths[0][hand_index])
        hand_position, _ = hand.get_world_poses()
        self.assertGreater(float(hand_position.numpy()[0][2]), 0.4)

    async def spawn_franka(self, name: str = "franka", add_physics_callback: bool = True) -> None:
        """Spawn a Franka robot with the drawer-opening runner in the simulation.

        Args:
            name: Name for the robot prim in the stage.
            add_physics_callback: Whether to register a physics step callback for the robot.
        """
        self._prim_path = "/World/" + name

        self._franka = RobotPolicyRunner(
            self._spec,
            prim_path=self._prim_path,
            position=[0, 0, 0],
            task_state_provider=make_franka_task_state_provider(self.cabinet, self._env_config),
        )
        self._franka.spawn()
        applied_materials = self._franka.apply_scene_properties({"cabinet": self.cabinet})
        self.assertEqual(applied_materials, {"robot", "cabinet"})
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        self._franka.initialize()
        # PhysX stabilization/sleep would freeze the near-still arm mid-task; disable both,
        # as the 6.x class did.
        self._franka.articulation.set_stabilization_thresholds([0.0])
        self._franka.articulation.set_sleep_thresholds([0.0])
        await omni.kit.app.get_app().next_update_async()

        if add_physics_callback:
            self._physics_callback_id = SimulationManager.register_callback(
                self.on_physics_step, IsaacEvents.PRE_PHYSICS_STEP
            )
        await omni.kit.app.get_app().next_update_async()

    def on_physics_step(self, step_size: float, context: object) -> None:
        """Physics step callback that advances the Franka runner by one physics tick.

        Args:
            step_size: The physics time step size.
            context: The simulation context.
        """
        if self._franka:
            self._franka.step(step_size)


class TestFrankaGPU(TestFrankaExampleExtension):
    """GPU-accelerated test class for the Franka drawer-opening runner deployment.

    This class extends the base Franka test functionality to run on the CUDA simulation
    device while preserving the selected policy artifact's exported physics cadence.
    """

    def get_device(self) -> str:
        """Return the simulation device.

        Returns:
            The CUDA simulation device string.
        """
        return "cuda"
