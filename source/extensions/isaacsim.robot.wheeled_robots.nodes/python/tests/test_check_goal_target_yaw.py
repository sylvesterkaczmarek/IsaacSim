# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for CheckGoal2D target-yaw comparisons."""

import math

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.graph.core as og
import omni.graph.core.tests as ogts


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


class TestCheckGoalTargetYaw(ogts.OmniGraphTestCase):
    """Verify orientation completion uses target yaw and wrapped angular error."""

    async def setUp(self) -> None:
        await stage_utils.create_new_stage_async()
        stage_utils.set_stage_up_axis("Z")
        stage_utils.set_stage_units(meters_per_unit=1.0)
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        app_utils.stop()
        await app_utils.update_app_async()

    async def _evaluate(self, current_yaw: float, target_yaw: float, threshold: float = 0.1) -> list[bool]:
        graph, [check_goal_node], _, _ = og.Controller.edit(
            {"graph_path": "/ActionGraph"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("CheckGoal2D", "isaacsim.robot.wheeled_robots.CheckGoal2D"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("CheckGoal2D.inputs:currentPosition", [0.0, 0.0, 0.0]),
                    ("CheckGoal2D.inputs:currentOrientation", _yaw_quaternion(current_yaw)),
                    ("CheckGoal2D.inputs:target", [0.0, 0.0, target_yaw]),
                    ("CheckGoal2D.inputs:targetChanged", True),
                    ("CheckGoal2D.inputs:thresholds", [0.1, threshold]),
                ],
            },
        )
        await og.Controller.evaluate(graph)
        return list(og.Controller(og.Controller.attribute("outputs:reachedGoal", check_goal_node)).get())

    async def test_nonzero_target_yaw_is_not_reached_by_zero_yaw(self) -> None:
        self.assertEqual(await self._evaluate(current_yaw=0.0, target_yaw=1.0), [True, False])

    async def test_yaw_comparison_wraps_across_pi_boundary(self) -> None:
        reached = await self._evaluate(
            current_yaw=-math.pi + 0.02,
            target_yaw=math.pi - 0.02,
            threshold=0.05,
        )
        self.assertEqual(reached, [True, True])
