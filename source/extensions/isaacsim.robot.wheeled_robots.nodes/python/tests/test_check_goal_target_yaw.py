# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for CheckGoal2D target-yaw comparisons."""

import math
from types import SimpleNamespace

import numpy as np
import omni.kit.test
from isaacsim.robot.wheeled_robots.nodes.OgnCheckGoal2D import OgnCheckGoal2D


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


class TestCheckGoalTargetYaw(omni.kit.test.AsyncTestCase):
    """Verify orientation completion uses target yaw and wrapped angular error."""

    @staticmethod
    def _db(current_yaw: float, target_yaw: float, threshold: float = 0.1) -> SimpleNamespace:
        return SimpleNamespace(
            per_instance_state=SimpleNamespace(target=[0.0, 0.0, 0.0]),
            inputs=SimpleNamespace(
                targetChanged=True,
                target=[0.0, 0.0, target_yaw],
                currentPosition=[0.0, 0.0, 0.0],
                currentOrientation=_yaw_quaternion(current_yaw),
                thresholds=[0.1, threshold],
            ),
            outputs=SimpleNamespace(),
        )

    async def test_nonzero_target_yaw_is_not_reached_by_zero_yaw(self) -> None:
        db = self._db(current_yaw=0.0, target_yaw=1.0)

        self.assertTrue(OgnCheckGoal2D.compute(db))
        self.assertEqual(db.outputs.reachedGoal, [True, False])

    async def test_yaw_comparison_wraps_across_pi_boundary(self) -> None:
        db = self._db(current_yaw=-math.pi + 0.02, target_yaw=math.pi - 0.02, threshold=0.05)

        self.assertTrue(OgnCheckGoal2D.compute(db))
        self.assertEqual(db.outputs.reachedGoal, [True, True])
