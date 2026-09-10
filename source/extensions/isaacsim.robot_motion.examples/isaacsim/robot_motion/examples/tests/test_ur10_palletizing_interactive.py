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

"""Lifecycle tests for the interactive UR10 palletizing wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import isaacsim.core.experimental.utils.app as app_utils
import omni.kit.test
from isaacsim.robot_motion.examples.manipulation.interactive.palletizing import Palletizing


class TestUr10PalletizingInteractive(omni.kit.test.AsyncTestCase):
    async def _assert_reset_restarts_after_bin_spawn(self, stop_before_reset: bool) -> None:
        sample = Palletizing()
        try:
            await sample.load_world_async()
            sample._spawner.step()
            await app_utils.update_app_async()
            self.assertEqual(len(sample._spawner.bins), 1)
            self.assertTrue(app_utils.is_playing())

            if stop_before_reset:
                app_utils.stop()
                await app_utils.update_app_async()
                self.assertTrue(app_utils.is_stopped())
            await sample.reset_async()

            self.assertEqual(sample._spawner.bins, [])
            self.assertTrue(app_utils.is_playing())
            self.assertTrue(await sample.start_palletizing_async())
            self.assertIsNotNone(sample._robot.articulation.get_dof_positions())
        finally:
            sample._remove_callback()
            await sample.clear_async()

    async def test_reset_while_playing_restarts_after_bin_spawn(self) -> None:
        await self._assert_reset_restarts_after_bin_spawn(stop_before_reset=False)

    async def test_reset_after_stop_restarts_after_bin_spawn(self) -> None:
        await self._assert_reset_restarts_after_bin_spawn(stop_before_reset=True)

    async def test_start_step_and_completion_manage_callback(self) -> None:
        sample = Palletizing()
        sample._robot = MagicMock()
        sample._spawner = MagicMock()
        sample._context = MagicMock()
        sample._controller = MagicMock(state="idle")

        with (
            patch.object(sample, "_reset_runtime_state") as reset_runtime_state,
            patch(
                "isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing."
                "SimulationManager.register_callback",
                return_value=17,
            ) as register_callback,
            patch(
                "isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing."
                "SimulationManager.deregister_callback"
            ) as deregister_callback,
            patch("isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing.app_utils.play"),
            patch(
                "isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing."
                "app_utils.update_app_async",
                new=AsyncMock(),
            ),
        ):
            self.assertTrue(await sample.start_palletizing_async())
            reset_runtime_state.assert_called_once()
            register_callback.assert_called_once()

            sample._controller.state = "done"
            sample._physics_step(1.0 / 60.0, None)

            sample._spawner.step.assert_called_once()
            sample._context.read.assert_called_once()
            sample._controller.step.assert_called_once()
            sample._context.apply.assert_called_once()
            deregister_callback.assert_called_once_with(17)
            self.assertFalse(sample._is_executing)

    async def test_reset_and_cleanup_reset_all_backend_state(self) -> None:
        sample = Palletizing()
        sample._robot = MagicMock()
        sample._spawner = MagicMock()
        sample._context = MagicMock()
        sample._controller = MagicMock()

        with (
            patch("isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing.app_utils.stop"),
            patch(
                "isaacsim.robot_motion.examples.manipulation.interactive.palletizing.palletizing."
                "app_utils.update_app_async",
                new=AsyncMock(),
            ),
        ):
            await sample.setup_pre_reset()
        await sample.setup_post_reset()

        sample._robot.reset.assert_called_once()
        sample._spawner.reset.assert_called_once()
        sample._context.reset.assert_called_once()
        sample._controller.reset.assert_called_once()

        sample.physics_cleanup()
        self.assertFalse(sample._backend_ready())
