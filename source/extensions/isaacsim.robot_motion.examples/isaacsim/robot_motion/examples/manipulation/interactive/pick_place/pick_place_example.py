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

"""Franka pick-and-place interactive example."""

from __future__ import annotations

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample

from ...pick_place_task import PickPlaceTask


class FrankaPickPlaceInteractive(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.controller: PickPlaceTask | None = None
        self._is_executing = False
        self._physics_callback_id: int | None = None

    def setup_scene(self) -> None:
        self.controller = PickPlaceTask()
        self.controller.setup_scene()

    async def setup_post_load(self) -> None:
        if self.controller is not None:
            self.controller.initialize()

    def _remove_callback(self) -> None:
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

    async def setup_pre_reset(self) -> None:
        self._remove_callback()
        if self.controller is not None:
            self.controller.reset()
        self._is_executing = False

    async def setup_post_reset(self) -> None:
        if self.controller is not None:
            self.controller.reset_robot()

    async def setup_post_clear(self) -> None:
        self.physics_cleanup()

    def physics_cleanup(self) -> None:
        self._remove_callback()
        if self.controller is not None:
            self.controller.cleanup()
        self.controller = None
        self._is_executing = False

    def _pick_place_physics_callback(self, dt: float, context: object) -> None:
        if not self._is_executing or self.controller is None:
            return
        if self.controller.is_done or not self.controller.step(dt):
            self._is_executing = False
            self._remove_callback()

    def get_controller_status(self) -> dict[str, object]:
        return self.controller.status() if self.controller is not None else {"error": "Controller not initialized"}

    def is_executing(self) -> bool:
        return self._is_executing

    async def execute_pick_place_async(self) -> bool:
        if self.controller is None or self._is_executing:
            return False
        self._remove_callback()
        self.controller.reset()
        self._is_executing = True
        self._physics_callback_id = SimulationManager.register_callback(
            self._pick_place_physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
        )
        app_utils.play()
        await app_utils.update_app_async()
        return True
