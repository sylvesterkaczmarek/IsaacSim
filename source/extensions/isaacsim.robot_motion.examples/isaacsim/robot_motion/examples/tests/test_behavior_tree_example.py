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

"""End-to-end test for the Franka Behavior Tree example."""

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.robot_motion.examples.manipulation._behavior_tree_franka import clear_franka_motion_adapters
from isaacsim.storage.native import get_assets_root_path

_SCENE_PATH = "/Isaac/Samples/BehaviorTree/FrankaPickPlace/franka_pick_place.usd"
_CUBE_PATHS = (
    "/World/Pyramid/CubeTop",
    "/World/Pyramid/CubeMiddleLeft",
    "/World/Pyramid/CubeMiddleRight",
    "/World/Pyramid/CubeBaseLeft",
    "/World/Pyramid/CubeBaseCenter",
    "/World/Pyramid/CubeBaseRight",
)
_EXPECTED_STACKS = {
    "left": np.asarray([[0.46, -0.28, 0.025], [0.46, -0.28, 0.075], [0.46, -0.28, 0.125]]),
    "right": np.asarray([[0.46, 0.28, 0.025], [0.46, 0.28, 0.075], [0.46, 0.28, 0.125]]),
}
_POSITION_TOLERANCE = 0.03
_SETTLED_SPEED = 0.10


def _match_stack(positions: np.ndarray, expected: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(positions[:, :2] - expected[0, :2], axis=1)
    return positions[distances <= _POSITION_TOLERANCE]


class TestBehaviorTreeExample(omni.kit.test.AsyncTestCase):
    """Verify the published Behavior Tree example completes its two stacks."""

    async def test_cubes_form_expected_stacks(self) -> None:
        """Place three settled cubes at ascending heights in each target stack."""
        usd_context = omni.usd.get_context()
        opened, error = await usd_context.open_stage_async(get_assets_root_path() + _SCENE_PATH)
        self.assertTrue(opened, error)
        await omni.kit.app.get_app().next_update_async()

        cubes = RigidPrim(_CUBE_PATHS)
        timeline = omni.timeline.get_timeline_interface()
        settled_frames = 0

        def continue_until_stacked(_step: int, _steps: int) -> bool:
            nonlocal settled_frames
            try:
                positions = cubes.get_world_poses()[0].numpy()
                speeds = np.linalg.norm(cubes.get_velocities()[0].numpy(), axis=1)
            except (IndexError, RuntimeError, ValueError):
                settled_frames = 0
                return True
            stacked = all(
                len(_match_stack(positions, expected)) == len(expected) for expected in _EXPECTED_STACKS.values()
            )
            settled_frames = settled_frames + 1 if stacked and np.all(speeds <= _SETTLED_SPEED) else 0
            return settled_frames < 15

        try:
            timeline.play()
            await app_utils.update_app_async(steps=6000, callback=continue_until_stacked)

            positions = cubes.get_world_poses()[0].numpy()
            speeds = np.linalg.norm(cubes.get_velocities()[0].numpy(), axis=1)
            for name, expected in _EXPECTED_STACKS.items():
                actual = _match_stack(positions, expected)
                self.assertEqual(len(actual), 3, f"{name} stack positions: {positions.tolist()}")
                actual = actual[np.argsort(actual[:, 2])]
                np.testing.assert_allclose(actual, expected, atol=_POSITION_TOLERANCE, rtol=0.0)
                self.assertTrue(np.all(np.diff(actual[:, 2]) > 0.02), f"{name} stack Z values: {actual[:, 2]}")
            self.assertTrue(np.all(speeds <= _SETTLED_SPEED), f"Cube speeds: {speeds}")
        finally:
            timeline.stop()
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()
            clear_franka_motion_adapters()
            await usd_context.close_stage_async()
