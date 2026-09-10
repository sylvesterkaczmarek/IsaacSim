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

"""Runtime coverage for the replay follow-target example."""

import os

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.kit.test
from isaacsim.robot_motion.examples.manipulation.interactive.replay_follow_target import ReplayFollowTarget


def _get_data_file() -> str:
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = extension_manager.get_enabled_extension_id("isaacsim.robot_motion.examples")
    return os.path.join(extension_manager.get_extension_path(extension_id), "data", "example_data_file.json")


class TestReplayFollowTarget(omni.kit.test.AsyncTestCase):
    async def test_replay_trajectory_excludes_target(self) -> None:
        sample = ReplayFollowTarget()
        try:
            await sample.load_world_async()
            stage = stage_utils.get_current_stage(backend="usd")
            self.assertFalse(stage.GetPrimAtPath(sample._target_path).IsValid())

            await sample._on_replay_scene_event_async(_get_data_file())
            self.assertTrue(stage.GetPrimAtPath(sample._target_path).IsValid())

            if app_utils.is_playing():
                app_utils.stop()
            await sample.reset_async()
            await sample._on_replay_trajectory_event_async(_get_data_file())
            self.assertFalse(stage.GetPrimAtPath(sample._target_path).IsValid())
        finally:
            if app_utils.is_playing():
                app_utils.stop()
            await sample.clear_async()

    async def test_replay_scene(self) -> None:
        sample = ReplayFollowTarget()
        try:
            await sample.load_world_async()
            stage = stage_utils.get_current_stage(backend="usd")
            self.assertFalse(stage.GetPrimAtPath(sample._target_path).IsValid())

            await sample._on_replay_scene_event_async(_get_data_file())
            self.assertTrue(stage.GetPrimAtPath(sample._target_path).IsValid())

            first_frame = sample._data_frames[0]["data"]
            self.assertAlmostEqual(first_frame["target_position"][2], 0.7)
            self.assertLessEqual(max(first_frame["applied_joint_positions"][-2:]), 0.04)

            await app_utils.update_app_async(
                steps=60, callback=lambda _step, _steps: sample._current_time_step_index == 0
            )

            self.assertGreater(sample._current_time_step_index, 0)
        finally:
            if app_utils.is_playing():
                app_utils.stop()
            await sample.clear_async()
