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

"""End-to-end tests for the interactive Franka pick/place example."""

import numpy as np
import omni.kit.test
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.robot_motion.examples.manipulation.interactive.pick_place import FrankaPickPlaceInteractive


class TestFrankaPickPlaceInteractive(omni.kit.test.AsyncTestCase):
    async def test_pick_place_completes(self) -> None:
        sample = FrankaPickPlaceInteractive()
        try:
            await sample.load_world_async()
            await sample.reset_async()
            self.assertTrue(await sample.execute_pick_place_async())

            await app_utils.update_app_async(steps=1800, callback=lambda _step, _steps: sample.is_executing())

            status = sample.get_controller_status()
            self.assertFalse(status["failed"], status)
            self.assertTrue(status["done"], status)
            controller = sample.controller
            self.assertIsNotNone(controller)
            actual = controller.cubes[0].get_world_poses()[0].numpy()[0]
            self.assertLessEqual(
                np.linalg.norm(actual - controller.place_position),
                0.03,
                (actual, controller.place_position),
            )
        finally:
            await sample.clear_async()
