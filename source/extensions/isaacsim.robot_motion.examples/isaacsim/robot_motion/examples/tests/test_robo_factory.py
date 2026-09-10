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

import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.robot_motion.examples.manipulation.interactive.robo_factory.robo_factory import RoboFactory
from isaacsim.storage.native import get_assets_root_path


def _is_at_or_below(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


class _RecomposedRoboFactory(RoboFactory):
    def __init__(self) -> None:
        super().__init__()
        self._num_of_tasks = 1

    async def setup_post_load(self) -> None:
        scenario = self._stackings[0].scenario
        stage = omni.usd.get_context().get_stage()
        stage.RemovePrim(scenario.robot_prim_path)
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
            if not scenario.articulation.valid:
                break

        asset_path = get_assets_root_path() + scenario.robot_config.asset_path
        stage_utils.add_reference_to_stage(
            asset_path,
            scenario.robot_prim_path,
            variants=list(scenario.robot_config.variants),
        )
        while stage_utils.is_stage_loading():
            await omni.kit.app.get_app().next_update_async()

        if scenario.articulation.valid:
            raise AssertionError("Removing and restoring the robot reference did not invalidate its existing wrapper")
        await super().setup_post_load()


class TestRoboFactory(omni.kit.test.AsyncTestCase):
    async def test_stacking_completes(self) -> None:
        sample = RoboFactory()
        try:
            await sample.load_world_async()
            await sample.reset_async()
            await sample._on_start_stacking_event_async()

            def continue_until_finished(_step: int, _steps: int) -> bool:
                return not all(stacking.is_done or stacking.failed for stacking in sample._stackings)

            await app_utils.update_app_async(steps=3200, callback=continue_until_finished)
            statuses = [stacking.status() for stacking in sample._stackings]
            self.assertFalse(any(stacking.failed for stacking in sample._stackings), statuses)
            self.assertTrue(all(stacking.is_done for stacking in sample._stackings), statuses)
            for stacking in sample._stackings:
                for index, cube in enumerate(stacking.cubes):
                    expected = stacking.place_position.copy()
                    expected[2] += 0.0515 * index
                    actual = cube.get_world_poses()[0].numpy()[0]
                    self.assertLessEqual(np.linalg.norm(actual - expected), 0.03, (actual, expected))
                    self.assertLessEqual(np.linalg.norm(cube.get_velocities()[0].numpy()[0]), 0.1)
        finally:
            await sample.clear_async()

    async def test_robot_wrapper_recovers_after_reference_recomposition(self) -> None:
        sample = _RecomposedRoboFactory()
        try:
            await sample.load_world_async()

            scenario = sample._stackings[0].scenario
            self.assertTrue(scenario.articulation.valid)
            self.assertTrue(scenario.articulation.is_physics_tensor_entity_valid())
            self.assertTrue(scenario._tracked_collision_paths)
            self.assertIsNotNone(sample._stackings[0].controller)
            self.assertIsNotNone(scenario.read_robot_state())
            for cube_path in sample._stackings[0].cube_paths:
                self.assertTrue(any(_is_at_or_below(path, cube_path) for path in scenario._tracked_collision_paths))
        finally:
            await sample.clear_async()

    async def test_each_robot_excludes_other_tasks(self) -> None:
        sample = RoboFactory()
        try:
            await sample.load_world_async()

            for stacking in sample._stackings:
                scenario = stacking.scenario
                tracked_paths = scenario._tracked_collision_paths
                self.assertTrue(scenario.articulation.valid)
                self.assertTrue(tracked_paths)

                for cube_path in stacking.cube_paths:
                    self.assertTrue(any(_is_at_or_below(path, cube_path) for path in tracked_paths))

                for other in sample._stackings:
                    if other is stacking:
                        continue
                    excluded_paths = [other.scenario.robot_prim_path, *other.cube_paths]
                    for excluded_path in excluded_paths:
                        self.assertFalse(any(_is_at_or_below(path, excluded_path) for path in tracked_paths))
        finally:
            await sample.clear_async()
