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

"""Behavioral tests for the bundled Spot flat-terrain policy on the runner deployment path.

These are the retained Spot movement/stability suites of the refactor's verification matrix:
the same spawn/standing/forward/turn criteria that drove the removed ``SpotFlatTerrainPolicy``
class now drive the generic ``RobotPolicyRunner`` with the bundled Spot spec; the action scale and
joint order derive from the selected engine's staged IO descriptor, and the spawn pose comes
from the hosted env config.
"""

import asyncio

import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples.bundled.spot import get_spot_spec
from isaacsim.robot.policy.examples.runtime import RobotPolicyRunner
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics


class TestSpotCPU(omni.kit.test.AsyncTestCase):
    """Test suite for the Spot runner deployment on the CPU simulation device.

    This test class validates the bundled Spot flat-terrain policy by running comprehensive
    tests for robot spawning, movement commands, and physics simulation on CPU. It inherits
    from AsyncTestCase to support asynchronous test execution.

    The test suite includes:
    - Spawning and validating the Spot robot in the simulation environment
    - Verifying the robot remains standing under a zero command
    - Testing forward movement commands and verifying displacement
    - Testing turning commands and verifying rotational changes
    - Physics simulation integration with proper setup and teardown

    Each test creates a new stage with physics scene, deploys the Spot runner, and runs physics
    simulation to verify expected behaviors through measured robot pose changes.
    """

    def get_device(self) -> str:
        """Return the simulation device. Override in subclasses.

        Returns:
            The simulation device string.
        """
        return "cpu"

    async def setUp(self) -> None:
        """Set up the test environment with physics scene and ground plane.

        Initializes a new USD stage, configures physics simulation parameters, spawns a ground
        plane, and sets up the simulation manager with the selected device configuration.
        """
        self._physics_callback_id = None
        self._spot = None
        await stage_utils.create_new_stage_async()
        stage_utils.set_stage_up_axis("Z")
        stage_utils.set_stage_units(meters_per_unit=1.0)
        # This needs to be set so that kit updates match physics updates
        self._physics_rate = 500

        device_str = self.get_device()
        print(f"Setting up test with device: {device_str}")

        self._physics_dt = 1 / self._physics_rate
        self._physics_time = 0.0
        stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")

        # spawn simulation manager
        SimulationManager.set_physics_sim_device(device_str)
        SimulationManager.set_physics_dt(self._physics_dt)

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        self._base_command = np.zeros(3, dtype=np.float32)
        self._stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()

    async def tearDown(self) -> None:
        """Clean up the test environment.

        Stops the timeline, deregisters physics callbacks, and waits for all assets to finish
        loading before completing teardown.
        """
        await omni.kit.app.get_app().next_update_async()
        self._timeline.stop()
        if self._spot is not None:
            self._spot.close()
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        await omni.usd.get_context().close_stage_async()

    async def test_spot_add(self) -> None:
        """Test spawning a Spot robot and verify its configuration.

        Verifies that the robot has the expected 12 degrees of freedom and that the robot prim
        exists in the stage with proper ArticulationRootAPI.
        """
        await self.spawn_spot()
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._spot.articulation.num_dofs, 12)

        # Verify root prim exists at spawn path
        root_prim = stage_utils.get_current_stage().GetPrimAtPath(self._prim_path)
        self.assertIsNotNone(root_prim, f"Robot root prim should exist at {self._prim_path}")
        self.assertTrue(root_prim.IsValid(), "Robot root prim should be valid")

        # Verify articulation root (may be nested under root for some USD assets) has ArticulationRootAPI
        articulation_root_path = self._spot.articulation.paths[0]
        articulation_prim = stage_utils.get_current_stage().GetPrimAtPath(articulation_root_path)
        self.assertTrue(
            prim_utils.has_api(articulation_prim, UsdPhysics.ArticulationRootAPI),
            f"Articulation root prim at {articulation_root_path} should have ArticulationRootAPI",
        )

    async def test_robot_standing(self) -> None:
        """Test the robot remains upright with a zero command (standing in place)."""
        await self.spawn_spot()
        await omni.kit.app.get_app().next_update_async()

        start_positions_wp, _ = self._spot.articulation.get_world_poses()
        start_pos = start_positions_wp.numpy()[0]

        end_time = self._physics_time + 1.0
        while self._physics_time < end_time:
            await omni.kit.app.get_app().next_update_async()

        current_positions_wp, _ = self._spot.articulation.get_world_poses()
        current_pos = current_positions_wp.numpy()[0]

        self.assertGreater(current_pos[2], 0.2, f"Robot should remain upright. base z={current_pos[2]:.4f}m")
        horizontal_drift = float(np.linalg.norm(current_pos[:2] - start_pos[:2]))
        self.assertLess(horizontal_drift, 0.6, f"Robot should not drift horizontally. drift={horizontal_drift:.4f}m")

    async def test_robot_move_forward_command(self) -> None:
        """Test robot forward movement command.

        Commands the robot to move forward and verifies that it moves the expected distance
        within reasonable bounds after 1 second of simulation.
        """
        self._base_command = np.array([2, 0, 0], dtype=np.float32)
        await self.spawn_spot()
        await omni.kit.app.get_app().next_update_async()

        # Get current poses and convert to numpy arrays for efficient operations
        start_positions_wp, _ = self._spot.articulation.get_world_poses()

        self.start_pos = start_positions_wp.numpy()[0]

        end_time = self._physics_time + 1.0
        while self._physics_time < end_time:
            await omni.kit.app.get_app().next_update_async()

        current_positions_wp, _ = self._spot.articulation.get_world_poses()

        self.current_pos = current_positions_wp.numpy()[0]

        delta = self.current_pos[0] - self.start_pos[0]

        print(f"Spot forward displacement: {delta:.3f} m")
        self.assertGreater(delta, 0.5, f"Spot moved only {delta:.3f} m")
        self.assertLess(delta, 2.0)

    async def test_robot_turn_command(self) -> None:
        """Test robot turn command.

        Commands the robot to turn and verifies that it rotates at least 90 degrees
        after 2 seconds of simulation.
        """
        self._base_command = np.array([0, 0, 1], dtype=np.float32)
        await self.spawn_spot()
        await omni.kit.app.get_app().next_update_async()

        # Get current poses and convert to numpy arrays for efficient operations
        _, start_orientations_wp = self._spot.articulation.get_world_poses()

        self.start_orientation = start_orientations_wp.numpy()[0]

        end_time = self._physics_time + 2.0
        while self._physics_time < end_time:
            await omni.kit.app.get_app().next_update_async()

        _, current_orientations_wp = self._spot.articulation.get_world_poses()

        self.current_orientation = current_orientations_wp.numpy()[0]

        # Convert quaternions to rotation matrices and extract yaw angles
        start_rot_matrix = transform_utils.quaternion_to_rotation_matrix(self.start_orientation)
        current_rot_matrix = transform_utils.quaternion_to_rotation_matrix(self.current_orientation)

        # Convert Warp arrays to numpy arrays for indexing
        start_rot_matrix_np = start_rot_matrix.numpy()
        current_rot_matrix_np = current_rot_matrix.numpy()

        # Extract yaw angle from rotation matrix (element [1,0] / [0,0] gives tan(yaw))
        start_yaw = np.arctan2(start_rot_matrix_np[1, 0], start_rot_matrix_np[0, 0])
        current_yaw = np.arctan2(current_rot_matrix_np[1, 0], current_rot_matrix_np[0, 0])

        yaw_delta = current_yaw - start_yaw
        heading_delta = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))

        # should have turned at least 90 deg
        self.assertGreater(heading_delta, 1.5)

    async def spawn_spot(self, name: str = "spot") -> None:
        """Spawn a Spot robot in the simulation.

        Deploys a ``RobotPolicyRunner`` with the bundled Spot spec (derived binding, env-config spawn
        pose), starts the timeline, initializes the policy runtime, and registers a physics
        callback for robot control.

        One uncontrolled physics frame elapses between ``play()`` and ``initialize()``, so the
        robot settles (or, on Newton, collapses) before control starts. The runner's product half
        of the recovery authors the resolved spawn pose (here the env config's
        ``scene.robot.init_state``) as the articulation's default root state during
        ``initialize()`` but deliberately never teleports; the test half restores that pose with
        ``reset_to_default_state()`` before control starts — the same semantics as the removed
        class's ``initialize(reset_to_default_state=True)`` idiom.

        Args:
            name: The name for the robot prim in the stage.
        """
        self._prim_path = "/World/" + name

        self._spot = RobotPolicyRunner(get_spot_spec(), prim_path=self._prim_path)
        self._spot.spawn()
        await omni.kit.app.get_app().next_update_async()
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        self._spot.initialize()
        # Restore the spawn pose after the uncontrolled pre-initialize window; initialize()
        # already authored it as the default root state (and initialize_articulation set the
        # default velocities and DOF state), so the reset teleports back to the env-config pose.
        self._spot.articulation.reset_to_default_state()
        await omni.kit.app.get_app().next_update_async()

        self._physics_callback_id = SimulationManager.register_callback(
            self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP
        )

    def on_physics_step(self, step_size: float, context: object) -> None:
        """Physics step callback to control the Spot robot.

        Called on each physics simulation step to send movement commands to the robot.

        Args:
            step_size: The physics simulation step size.
            context: The simulation context.
        """
        self._physics_time += step_size
        if self._spot:
            self._spot.step(step_size, self._base_command)


class TestSpotGPU(TestSpotCPU):
    """GPU-based test suite for the Spot quadruped runner deployment.

    This test class extends TestSpotCPU to run all Spot policy tests on the CUDA simulation
    device. It inherits comprehensive test coverage for robot spawning, standing stability,
    movement commands, and turning behaviors while leveraging GPU compute.
    """

    def get_device(self) -> str:
        """Return the simulation device.

        Returns:
            The CUDA simulation device string.
        """
        return "cuda"
