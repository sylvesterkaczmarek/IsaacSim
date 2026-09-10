# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Interactive UR10 bin palletizing example."""

from __future__ import annotations

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.robot_motion.examples.manipulation.ur10_palletizing import (
    BinSpawner,
    BinStackingContext,
    PalletizerController,
    Ur10,
    Ur10Assets,
    build_scene,
    configure_conveyor,
)


class Palletizing(BaseSample):
    """Interactive wrapper around the shared UR10 palletizing backend."""

    def __init__(self) -> None:
        super().__init__()
        self._env_path = "/World/Ur10Table"
        self._assets: Ur10Assets | None = None
        self._robot: Ur10 | None = None
        self._spawner: BinSpawner | None = None
        self._context: BinStackingContext | None = None
        self._controller: PalletizerController | None = None
        self._physics_callback_id: int | None = None
        self._is_executing = False

    def setup_scene(self) -> None:
        """Build the palletizing workcell from reusable asset references."""
        self._assets = Ur10Assets()
        self._robot = build_scene(self._env_path, self._assets)

    async def setup_post_load(self) -> None:
        """Initialize the backend after referenced assets finish loading."""
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        if self._assets is None or self._robot is None:
            raise RuntimeError("Palletizing scene was not created.")
        configure_conveyor(self._env_path)
        self._robot.initialize()
        self._spawner = BinSpawner(self._env_path, self._assets)
        self._context = BinStackingContext(self._robot, self._spawner)
        self._controller = PalletizerController()
        self._reset_runtime_state()
        ViewportManager.set_camera_view(
            eye=[3.0, 3.0, 1.5],
            target=[0.35, 0.2, -0.4],
            camera="/OmniverseKit_Persp",
        )

    async def setup_pre_reset(self) -> None:
        """Stop execution and remove spawned bins while physics is stopped."""
        self._remove_callback()
        self._is_executing = False
        if not self._backend_ready():
            return

        self._robot.suction_gripper.open()
        await app_utils.update_app_async()
        app_utils.stop()
        await app_utils.update_app_async()
        self._spawner.reset()
        await app_utils.update_app_async()

    async def setup_post_reset(self) -> None:
        """Reset the palletizing backend after the simulation restarts."""
        self._reset_runtime_state()

    async def setup_post_clear(self) -> None:
        """Release task resources after clearing the stage."""

        self.physics_cleanup()

    def physics_cleanup(self) -> None:
        """Release callbacks and task resources."""
        self._remove_callback()
        self._assets = None
        self._robot = None
        self._spawner = None
        self._context = None
        self._controller = None
        self._is_executing = False

    async def start_palletizing_async(self) -> bool:
        """Start the palletizing sequence.

        Returns:
            True if the sequence started, otherwise False.
        """

        if not self._backend_ready() or self._is_executing:
            return False
        self._reset_runtime_state()
        self._is_executing = True
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_step,
            event=SimulationEvent.PHYSICS_POST_STEP,
        )
        app_utils.play()
        await app_utils.update_app_async()
        return True

    def _physics_step(self, dt: float, context: object) -> None:
        if not self._is_executing or not self._backend_ready():
            return

        self._spawner.step()
        observation = self._context.read(SimulationManager.get_simulation_time())
        command = self._controller.step(observation)
        self._context.apply(command, dt)
        if self._controller.state == "done":
            self._is_executing = False
            self._remove_callback()

    def _backend_ready(self) -> bool:
        return all(component is not None for component in (self._robot, self._spawner, self._context, self._controller))

    def _reset_runtime_state(self) -> None:
        if not self._backend_ready():
            return
        self._robot.reset()
        self._context.reset()
        self._controller.reset()

    def _remove_callback(self) -> None:
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
