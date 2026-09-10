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

"""RoboFactory Interactive Example.

This interactive example demonstrates multiple robots performing stacking tasks
in the same scene using experimental APIs.
"""

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.robot_motion.examples.manipulation import PickPlaceTask


class RoboFactory(BaseSample):
    """Interactive sample for multiple robots performing stacking tasks."""

    def __init__(self) -> None:
        super().__init__()
        self._stackings: list[PickPlaceTask] = []
        self._num_of_tasks = 4
        self._physics_callback_id = None
        self._is_executing = False

    def setup_scene(self) -> None:
        """Set up the scene with multiple robots and cubes."""
        # Create multiple stackings with different offsets
        for i in range(self._num_of_tasks):
            offset = (0.0, float((i * 2) - 3), 0.0)
            robot_path = f"/World/robot_{i}"
            stacking = PickPlaceTask(
                robot_path=robot_path,
                cube_path=f"/World/Cube_{i}",
                offset=offset,
                cube_positions=[(0.3, 0.3, 0.0258), (0.6, -0.25, 0.0258)],
                place_position=(0.5, 0.5, 0.0258),
            )
            stacking.setup_scene()
            self._stackings.append(stacking)

    async def setup_post_load(self) -> None:
        """Called after the scene is loaded."""
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()
        await app_utils.update_app_async()

        if any(
            not stacking.scenario.articulation.valid
            or not stacking.scenario.articulation.is_physics_tensor_entity_valid()
            for stacking in self._stackings
        ):
            SimulationManager.invalidate_physics()
            SimulationManager.initialize_physics()
            for stacking in self._stackings:
                stacking.scenario.refresh_robot_prims()
                articulation = stacking.scenario.articulation
                if not articulation.valid or not articulation.is_physics_tensor_entity_valid():
                    raise RuntimeError(f"Failed to initialize robot at {stacking.scenario.robot_prim_path}.")

        for stacking in self._stackings:
            other_tasks = (other for other in self._stackings if other is not stacking)
            exclude_paths = [
                path for other in other_tasks for path in (other.scenario.robot_prim_path, *other.cube_paths)
            ]
            stacking.initialize(exclude_paths)
        # Set camera view
        ViewportManager.set_camera_view(eye=[10.0, 0.0, 5.0], target=[0.0, 0.0, 0.0], camera="/OmniverseKit_Persp")
        print(f"Scene loaded with {self._num_of_tasks} robots")

    async def setup_pre_reset(self) -> None:
        """Called before world reset."""
        # Stop any ongoing execution and remove callbacks
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

        # Reset all stackings
        for stacking in self._stackings:
            stacking.reset()

        self._is_executing = False

    async def setup_post_reset(self) -> None:
        """Called after world reset."""
        # Reset all robots to default poses
        for stacking in self._stackings:
            stacking.reset_robot()

    async def setup_post_clear(self) -> None:
        """Called after clearing the scene."""
        # Stop any ongoing execution and remove callbacks
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

        for stacking in self._stackings:
            stacking.cleanup()
        self._stackings = []
        self._is_executing = False

    def physics_cleanup(self) -> None:
        """Clean up world resources."""
        # Stop any ongoing execution and remove callbacks
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

        for stacking in self._stackings:
            stacking.cleanup()
        self._stackings = []
        self._is_executing = False

    def _stacking_physics_callback(self, dt: float, context: object) -> None:
        """Physics callback to execute stacking operations step by step.

        Args:
            dt: Time delta for the physics step.
            context: Physics simulation context.
        """
        if not self._is_executing:
            return

        # Execute one step for each stacking
        for stacking in self._stackings:
            if not stacking.is_done and not stacking.failed:
                stacking.step(dt)

        if all(stacking.is_done or stacking.failed for stacking in self._stackings):
            if any(stacking.failed for stacking in self._stackings):
                failures = [
                    f"robot {index}: {stacking.status()['failure_reason']}"
                    for index, stacking in enumerate(self._stackings)
                    if stacking.failed
                ]
                print(f"Stacking failed: {'; '.join(failures)}")
            else:
                print("All robots finished stacking!")
            self._is_executing = False
            if self._physics_callback_id is not None:
                SimulationManager.deregister_callback(self._physics_callback_id)
                self._physics_callback_id = None

    async def _on_start_stacking_event_async(self) -> None:
        """Start the stacking execution."""
        if self._is_executing:
            print("Stacking already in progress...")
            return

        print("Starting stacking execution...")
        for stacking in self._stackings:
            stacking.reset()
        self._is_executing = True

        # Register physics callback using SimulationManager
        self._physics_callback_id = SimulationManager.register_callback(
            self._stacking_physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
        )

        # Start timeline playback
        app_utils.play()
        await app_utils.update_app_async()
