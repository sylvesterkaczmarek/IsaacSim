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

"""Unit tests for the Snap to Limits test mode."""

from __future__ import annotations

import math
from unittest import mock

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.snap_to_limits import _Phase

from .common import (
    DriveSubmodality,
    JointModality,
    TestGainTunerHarness,
)


class TestSnapToLimitsClassification(omni.kit.test.AsyncTestCase):
    """Hold-phase blocked vs fail classification (no articulation)."""

    def _run_hold_classification(self, hold_errors: list[float]) -> bool:
        test = gain_tuner.SnapToLimitsTest()
        test._tolerance = 0.01
        joint_metrics: dict[int, dict] = {0: {}}
        test._record_hold_metrics(
            _Phase.HOLD_LOWER,
            [0],
            np.array([-1.0]),
            np.array([1.0]),
            articulation=mock.MagicMock(),
            joint_metrics=joint_metrics,
            hold_errors={0: hold_errors},
        )
        return joint_metrics[0].get("lower_blocked", False)

    async def test_hold_blocked_stalled_at_limit(self) -> None:
        """Stable hold errors at a limit are classified as blocked."""
        errs = [0.05, 0.051, 0.049, 0.05]
        self.assertTrue(self._run_hold_classification(errs))

    async def test_hold_fail_oscillating(self) -> None:
        """Oscillating hold errors are not classified as blocked."""
        errs = [0.05 + 0.02 * math.sin(i) for i in range(20)]
        self.assertFalse(self._run_hold_classification(errs))

    async def test_hold_fail_still_approaching(self) -> None:
        """Converging hold errors are not classified as blocked."""
        errs = [0.2 - 0.01 * i for i in range(10)]
        self.assertFalse(self._run_hold_classification(errs))


class TestSnapToLimitsSmoke(TestGainTunerHarness):
    """Short registered snap-to-limits run."""

    async def test_snap_to_limits_produces_metrics(self) -> None:
        """A short snap-to-limits run produces joint metrics."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            joint_limit_revolute=(-45.0, 45.0),
        )
        await self._run_setup_and_compute_inertia(robot_path, num_physics_steps=80)
        self._gain_tuner.register_test(gain_tuner.GainsTestMode.SNAP_TO_LIMITS, gain_tuner.SnapToLimitsTest())
        try:
            self._gain_tuner.initialize_gains_test(
                {
                    "test_mode": gain_tuner.GainsTestMode.SNAP_TO_LIMITS,
                    "joint_indices": [0],
                    "hold_duration": 0.05,
                    "tolerance": 0.5,
                    "sequence": [{"joint_indices": np.array([0], dtype=np.int32)}],
                }
            )
            self._timeline.play()
            done = False
            for _ in range(600):
                done = self._gain_tuner.update_gains_test(self._physics_dt)
                await app_utils.update_app_async()
                if done:
                    break
            self._timeline.stop()
            self.assertTrue(done, "snap-to-limits test should finish within step budget")
            metrics = self._gain_tuner.get_test_result_metrics()
            self.assertIn(0, metrics)
            self.assertTrue(metrics[0])
        finally:
            self._gain_tuner.unregister_test(gain_tuner.GainsTestMode.SNAP_TO_LIMITS)
