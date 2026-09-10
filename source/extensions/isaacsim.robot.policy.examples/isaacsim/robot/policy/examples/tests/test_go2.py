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

"""Behavioral tests for the bundled Go2 locomotion policy on the runner deployment path.

These are the retained Go2 movement/stability suites of the refactor's verification matrix:
the same spawn/standing/forward/turn criteria that drove the removed ``Go2FlatTerrainPolicy``
class now drive the generic ``RobotPolicyRunner`` with the bundled Go2 spec; the observation and
action interface derives from the artifact's exported IO descriptor at bind time.

The configured PhysX and Newton extension test suites both run this module.
"""

import asyncio

import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np
import omni.kit.test
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples.bundled.go2 import get_go2_spec
from isaacsim.robot.policy.examples.runtime import RobotPolicyRunner
from pxr import UsdPhysics


class TestGo2CPU(omni.kit.test.AsyncTestCase):
    """Test the Go2 runner deployment on the CPU simulation device."""

    def get_device(self) -> str:
        """Return the simulation device. Override in subclasses.

        Returns:
            The simulation device string.
        """
        return "cpu"

    async def setUp(self) -> None:
        """Set up test environment with physics scene and ground plane."""
        self._go2 = None
        self._physics_callback_id = None
        await stage_utils.create_new_stage_async()
        self._physics_rate = 200

        device_str = self.get_device()
        print(f"Setting up test with device: {device_str}")

        self._physics_dt = 1 / self._physics_rate
        stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")

        SimulationManager.set_physics_sim_device(device_str)
        SimulationManager.set_physics_dt(self._physics_dt)

        # Add ground plane
        from isaacsim.storage.native import get_assets_root_path

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        self._base_command = np.zeros(3, dtype=np.float32)
        self._stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Tear down test environment and deregister physics callback."""
        await omni.kit.app.get_app().next_update_async()
        self._timeline.stop()
        if self._go2 is not None:
            self._go2.close()
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    async def test_go2_add(self) -> None:
        """Test spawning a Go2 robot and verify its configuration.

        Verifies that the robot has the expected 12 degrees of freedom and that the robot prim
        exists in the stage with proper ArticulationRootAPI.
        """
        await self.spawn_go2()
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(self._go2.articulation.num_dofs, 12)

        # Verify root prim exists at spawn path
        root_prim = stage_utils.get_current_stage().GetPrimAtPath(self._prim_path)
        self.assertIsNotNone(root_prim, f"Robot root prim should exist at {self._prim_path}")
        self.assertTrue(root_prim.IsValid(), "Robot root prim should be valid")

        # Verify articulation root (may be nested under root for some USD assets) has ArticulationRootAPI
        articulation_root_path = self._go2.articulation.paths[0]
        articulation_prim = stage_utils.get_current_stage().GetPrimAtPath(articulation_root_path)
        self.assertTrue(
            prim_utils.has_api(articulation_prim, UsdPhysics.ArticulationRootAPI),
            f"Articulation root prim at {articulation_root_path} should have ArticulationRootAPI",
        )

    async def test_robot_standing(self) -> None:
        """Test the robot remains upright with a zero command (standing in place)."""
        await self.spawn_go2()
        await omni.kit.app.get_app().next_update_async()

        start_positions_wp, _ = self._go2.articulation.get_world_poses()
        start_pos = start_positions_wp.numpy()[0]

        self._base_command = np.zeros(3, dtype=np.float32)

        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()

        current_positions_wp, _ = self._go2.articulation.get_world_poses()
        current_pos = current_positions_wp.numpy()[0]

        self.assertGreater(current_pos[2], 0.2, f"Robot should remain upright. base z={current_pos[2]:.4f}m")
        horizontal_drift = float(np.linalg.norm(current_pos[:2] - start_pos[:2]))
        self.assertLess(horizontal_drift, 0.5, f"Robot should not drift horizontally. drift={horizontal_drift:.4f}m")

    async def test_robot_move_forward_command(self) -> None:
        """Test the robot moves forward in response to a forward command."""
        await self.spawn_go2()
        await omni.kit.app.get_app().next_update_async()

        # Get current poses
        start_positions_wp, _ = self._go2.articulation.get_world_poses()
        self.start_pos = start_positions_wp.numpy()[0]

        self._base_command = np.array([1, 0, 0], dtype=np.float32)

        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()

        # Get current poses
        current_positions_wp, _ = self._go2.articulation.get_world_poses()
        self.current_pos = current_positions_wp.numpy()[0]

        delta = abs(self.current_pos[0] - self.start_pos[0])
        self.assertGreater(delta, 0.45)
        self.assertLess(delta, 2.0)

    async def test_robot_turn_command(self) -> None:
        """Test the robot turns in response to a yaw command."""
        await self.spawn_go2()
        await omni.kit.app.get_app().next_update_async()

        # Get current poses
        _, start_orientations_wp = self._go2.articulation.get_world_poses()
        self.start_orientation = start_orientations_wp.numpy()[0]

        self._base_command = np.array([0, 0, 1], dtype=np.float32)

        for _ in range(80):
            await omni.kit.app.get_app().next_update_async()

        _, current_orientations_wp = self._go2.articulation.get_world_poses()
        self.current_orientation = current_orientations_wp.numpy()[0]

        start_rot = transform_utils.quaternion_to_rotation_matrix(self.start_orientation).numpy()
        current_rot = transform_utils.quaternion_to_rotation_matrix(self.current_orientation).numpy()

        start_yaw = np.arctan2(start_rot[1, 0], start_rot[0, 0])
        current_yaw = np.arctan2(current_rot[1, 0], current_rot[0, 0])

        heading_delta = abs(current_yaw - start_yaw)
        self.assertGreater(heading_delta, 0.35)

    async def spawn_go2(self, name: str = "go2") -> None:
        """Spawn a Go2 robot and register the physics callback.

        Args:
            name: Name for the robot prim.
        """
        self._prim_path = "/World/" + name

        self._go2 = RobotPolicyRunner(
            get_go2_spec(),
            prim_path=self._prim_path,
            position=[0, 0, 0.47],
        )
        self._go2.spawn()
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        self._go2.restart_from_default_state(self._base_command)

        self._physics_callback_id = SimulationManager.register_callback(
            self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP
        )
        await omni.kit.app.get_app().next_update_async()

    def on_physics_step(self, step_size: float, context: object) -> None:
        """Execute one policy step on physics update.

        Args:
            step_size: Physics step duration.
            context: Physics step context.
        """
        if self._go2:
            self._go2.step(step_size, self._base_command)


class TestGo2GPU(TestGo2CPU):
    """Test the Go2 runner deployment on the CUDA simulation device."""

    def get_device(self) -> str:
        """Return the simulation device.

        Returns:
            The CUDA simulation device string.
        """
        return "cuda"
