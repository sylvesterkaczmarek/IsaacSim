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

"""Unit tests for the Stress Test mode."""

from __future__ import annotations

from unittest import mock

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.kit.test
from isaacsim.robot_setup import gain_tuner

from .common import (
    DriveSubmodality,
    JointModality,
    TestGainTunerHarness,
)


class TestStressTestUnit(omni.kit.test.AsyncTestCase):
    """Stress test setup and empty articulation path."""

    async def test_setup_reads_test_params(self) -> None:
        """Stress test setup reads its mode, duration, and seed parameters."""
        test = gain_tuner.StressTest()
        articulation = mock.MagicMock()
        test.setup(
            articulation,
            [0],
            {0: 0},
            {
                "stress_test_mode": 1,
                "duration": 3.5,
                "velocity_threshold": 50.0,
                "sigma": 0.02,
                "snap_interval": 5,
                "seed": 7,
            },
        )
        self.assertEqual(test._mode, gain_tuner.StressTestMode.ADVERSARIAL)
        self.assertAlmostEqual(test._duration, 3.5)
        self.assertEqual(test._seed, 7)

    async def test_run_empty_articulation_returns_empty_result(self) -> None:
        """Stress test run returns an empty result when articulation is missing."""
        test = gain_tuner.StressTest()
        test._articulation = None
        gen = test.run()
        with self.assertRaises(StopIteration) as ctx:
            gen.send(None)
        result = ctx.exception.value
        self.assertEqual(result.joint_position_commands.size, 0)


class TestStressTestSmoke(TestGainTunerHarness):
    """Short registered stress test run."""

    async def test_stress_test_produces_metrics(self) -> None:
        """A short stress-test run records its configured seed in metrics."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            joint_limit_revolute=(-90.0, 90.0),
        )
        await self._run_setup_and_compute_inertia(robot_path, num_physics_steps=80)
        self._gain_tuner.register_test(gain_tuner.GainsTestMode.STRESS_TEST, gain_tuner.StressTest())
        try:
            self._gain_tuner.initialize_gains_test(
                {
                    "test_mode": gain_tuner.GainsTestMode.STRESS_TEST,
                    "joint_indices": [0],
                    "duration": 0.05,
                    "seed": 99,
                    "sequence": [{"joint_indices": np.array([0], dtype=np.int32)}],
                }
            )
            self._timeline.play()
            for _ in range(30):
                done = self._gain_tuner.update_gains_test(self._physics_dt)
                await app_utils.update_app_async()
                if done:
                    break
            self._timeline.stop()
            metrics = self._gain_tuner.get_test_result_metrics()
            self.assertIn(0, metrics)
            self.assertEqual(metrics[0].get("seed"), 99)
        finally:
            self._gain_tuner.unregister_test(gain_tuner.GainsTestMode.STRESS_TEST)
