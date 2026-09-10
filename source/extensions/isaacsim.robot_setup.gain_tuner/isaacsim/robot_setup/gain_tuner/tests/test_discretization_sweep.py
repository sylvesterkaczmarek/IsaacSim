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

"""Unit tests for the Discretization Sweep test mode."""

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


class TestDiscretizationSweepLevels(omni.kit.test.AsyncTestCase):
    """dt-sweep timestep level generation (pure, no articulation)."""

    async def test_levels_run_coarse_to_fine(self) -> None:
        """Levels are logarithmically spaced from dt_max down to dt_min."""
        levels = gain_tuner.dt_sweep_levels(1.0 / 30.0, 1.0 / 600.0, 5)
        self.assertEqual(len(levels), 5)
        self.assertAlmostEqual(float(levels[0]), 1.0 / 30.0)
        self.assertAlmostEqual(float(levels[-1]), 1.0 / 600.0)
        # Monotonically decreasing (coarse -> fine).
        self.assertTrue(np.all(np.diff(levels) < 0.0))

    async def test_single_level_returns_dt_max(self) -> None:
        """A single-step sweep returns just the coarsest timestep."""
        levels = gain_tuner.dt_sweep_levels(0.02, 0.001, 1)
        self.assertEqual(len(levels), 1)
        self.assertAlmostEqual(float(levels[0]), 0.02)

    async def test_zero_levels_returns_empty(self) -> None:
        """A non-positive step count yields no levels."""
        self.assertEqual(len(gain_tuner.dt_sweep_levels(0.02, 0.001, 0)), 0)

    async def test_reversed_bounds_rejected(self) -> None:
        """Reversed bounds (dt_max < dt_min) raise before running the sweep."""
        with self.assertRaises(ValueError):
            gain_tuner.validate_dt_bounds(0.001, 0.02)
        with self.assertRaises(ValueError):
            gain_tuner.dt_sweep_levels(0.001, 0.02, 5)

    async def test_equal_bounds_rejected(self) -> None:
        """Equal bounds are degenerate (all levels identical) and are rejected."""
        with self.assertRaises(ValueError):
            gain_tuner.validate_dt_bounds(0.01, 0.01)
        with self.assertRaises(ValueError):
            gain_tuner.dt_sweep_levels(0.01, 0.01, 5)

    async def test_non_positive_bounds_rejected(self) -> None:
        """Zero or negative bounds are rejected (log-spacing is undefined)."""
        for dt_max, dt_min in ((0.0, 0.001), (0.02, 0.0), (0.02, -0.001), (-0.02, -0.03)):
            with self.assertRaises(ValueError):
                gain_tuner.validate_dt_bounds(dt_max, dt_min)
            with self.assertRaises(ValueError):
                gain_tuner.dt_sweep_levels(dt_max, dt_min, 5)

    async def test_non_finite_bounds_rejected(self) -> None:
        """NaN / inf bounds are rejected before they reach np.logspace."""
        for dt_max, dt_min in ((float("inf"), 0.001), (0.02, float("nan")), (float("nan"), float("inf"))):
            with self.assertRaises(ValueError):
                gain_tuner.validate_dt_bounds(dt_max, dt_min)
            with self.assertRaises(ValueError):
                gain_tuner.dt_sweep_levels(dt_max, dt_min, 5)


class TestSelectTargetLevelIndex(omni.kit.test.AsyncTestCase):
    """Target-level index selection used for the retained charts (pure)."""

    async def test_picks_first_level_at_or_below_target(self) -> None:
        """The coarsest level whose dt is at or below the target is chosen."""
        levels = gain_tuner.dt_sweep_levels(1.0 / 30.0, 1.0 / 600.0, 5)  # coarse -> fine
        idx = gain_tuner.select_target_level_index(levels, target_dt=1.0 / 120.0)
        self.assertTrue(0 <= idx < len(levels))
        self.assertLessEqual(float(levels[idx]), 1.0 / 120.0 + 1e-9)
        # It is the FIRST (coarsest) such level, so the one just coarser is above target.
        if idx > 0:
            self.assertGreater(float(levels[idx - 1]), 1.0 / 120.0 + 1e-9)

    async def test_falls_back_to_finest_when_target_below_all(self) -> None:
        """A target finer than every level falls back to the finest (last) level."""
        levels = gain_tuner.dt_sweep_levels(0.02, 0.005, 4)
        idx = gain_tuner.select_target_level_index(levels, target_dt=0.0001)
        self.assertEqual(idx, len(levels) - 1)

    async def test_coarsest_when_target_above_all(self) -> None:
        """A target coarser than every level selects the coarsest (first) level."""
        levels = gain_tuner.dt_sweep_levels(0.02, 0.005, 4)
        self.assertEqual(gain_tuner.select_target_level_index(levels, target_dt=1.0), 0)

    async def test_empty_levels_return_negative_one(self) -> None:
        """No levels -> sentinel -1 (no trajectory to retain)."""
        self.assertEqual(gain_tuner.select_target_level_index([], target_dt=0.01), -1)


class TestAggregateDtSweep(omni.kit.test.AsyncTestCase):
    """Per-joint dt-sweep aggregation and classification (pure)."""

    @staticmethod
    def _level(dt: float, settle: float | None, err: float, settled: bool = True, skipped: bool = False) -> dict:
        return {"dt": dt, "settling_time": settle, "steady_state_error": err, "settled": settled, "skipped": skipped}

    async def test_accurate_joint_no_degradation(self) -> None:
        """A joint whose coarse dt tracks the fine reference is classified accurate."""
        sweep = [
            self._level(0.02, 0.50, 0.001),
            self._level(0.01, 0.50, 0.001),
            self._level(0.005, 0.50, 0.001),
        ]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.01, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        self.assertEqual(out[0]["status"], "accurate")
        # Coarsest dt still qualifies as the accuracy cliff.
        self.assertAlmostEqual(out[0]["accuracy_cliff_dt"], 0.02)
        self.assertAlmostEqual(out[0]["target_dt"], 0.01)

    async def test_degraded_joint_settle_time_blows_up(self) -> None:
        """A joint whose settling time balloons at the target dt is classified degraded."""
        sweep = [
            self._level(0.02, 5.0, 0.001),
            self._level(0.01, 5.0, 0.001),
            self._level(0.005, 0.50, 0.001),
        ]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.01, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        self.assertEqual(out[0]["status"], "degraded")
        # Only the finest level is within tolerance of the reference.
        self.assertAlmostEqual(out[0]["accuracy_cliff_dt"], 0.005)
        # Settle degradation at the target level is ~900% (5.0 vs 0.5 reference).
        self.assertAlmostEqual(out[0]["target_level"]["settle_degradation"], 9.0, places=3)

    async def test_did_not_settle_when_target_level_unsettled(self) -> None:
        """A joint that never settles at the target dt is classified did_not_settle."""
        sweep = [
            self._level(0.02, None, float("nan"), settled=False),
            self._level(0.01, None, float("nan"), settled=False),
            self._level(0.005, 0.50, 0.001),
        ]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.01, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        self.assertEqual(out[0]["status"], "did_not_settle")

    async def test_all_skipped_yields_none_reference(self) -> None:
        """A joint whose every level was skipped (cancelled) has no target/cliff."""
        sweep = [self._level(0.02, None, float("nan"), settled=False, skipped=True)]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.01, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        self.assertEqual(out[0]["status"], "did_not_settle")
        self.assertIsNone(out[0]["accuracy_cliff_dt"])
        self.assertIsNone(out[0]["target_level"])

    async def test_inaccurate_finest_dt_not_reported_accurate(self) -> None:
        """An inaccurate finest (reference) dt is not auto-classified accurate.

        Every level settled but the finest dt's absolute steady-state error exceeds
        tolerance, so its (trivially zero) self-relative degradation must not make it
        an accuracy cliff or an accurate verdict.
        """
        sweep = [
            self._level(0.02, 0.50, 0.05),
            self._level(0.01, 0.50, 0.05),
            self._level(0.005, 0.50, 0.05),  # finest reference, error 0.05 > tolerance 0.01
        ]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.005, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        # No level (including the finest) is within absolute tolerance, so there is
        # no accuracy cliff and the target dt (== finest) is degraded, not accurate.
        self.assertIsNone(out[0]["accuracy_cliff_dt"])
        self.assertEqual(out[0]["status"], "degraded")

    async def test_accurate_finest_dt_reported_accurate(self) -> None:
        """A finest dt within absolute tolerance is still a valid accuracy cliff."""
        sweep = [
            self._level(0.02, 5.0, 0.05),  # coarse: settles slowly / inaccurate
            self._level(0.005, 0.50, 0.001),  # finest reference within tolerance
        ]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.005, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        self.assertAlmostEqual(out[0]["accuracy_cliff_dt"], 0.005)
        self.assertEqual(out[0]["status"], "accurate")

    async def test_non_settling_reference_is_indeterminate(self) -> None:
        """A settled target dt with a non-settling finest reference is indeterminate.

        The finest dt is the accuracy baseline; if it never settled there is nothing
        to compare against, so the joint must be indeterminate rather than degraded.
        """
        sweep = [
            self._level(0.02, 0.50, 0.001, settled=True),
            self._level(0.01, 0.50, 0.001, settled=True),
            self._level(0.005, None, float("nan"), settled=False),  # finest reference never settled
        ]
        out = gain_tuner.aggregate_dt_sweep(
            {0: sweep}, target_dt=0.01, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
        )
        self.assertEqual(out[0]["status"], "indeterminate")
        self.assertIsNone(out[0]["accuracy_cliff_dt"])

    async def test_non_finite_target_dt_rejected(self) -> None:
        """A non-finite or non-positive target dt is rejected by the aggregator."""
        sweep = [self._level(0.02, 0.50, 0.001), self._level(0.005, 0.50, 0.001)]
        for bad_target in (float("nan"), float("inf"), 0.0, -0.01):
            with self.assertRaises(ValueError):
                gain_tuner.aggregate_dt_sweep(
                    {0: sweep}, target_dt=bad_target, settle_threshold=0.2, error_threshold=0.1, tolerance=0.01
                )


class TestDiscretizationSweepUnit(omni.kit.test.AsyncTestCase):
    """DiscretizationSweepTest setup and empty articulation path."""

    async def test_setup_reads_test_params(self) -> None:
        """Setup reads the timeout, hold, tolerance, and target-fraction parameters."""
        test = gain_tuner.DiscretizationSweepTest()
        test.setup(
            mock.MagicMock(),
            [0],
            {0: 0},
            {"timeout": 4.0, "hold_duration": 0.5, "tolerance": 0.02, "target_pct": 0.4},
        )
        self.assertAlmostEqual(test._timeout, 4.0)
        self.assertAlmostEqual(test._hold_duration, 0.5)
        self.assertAlmostEqual(test._tolerance, 0.02)
        self.assertAlmostEqual(test._target_pct, 0.4)

    async def test_run_empty_articulation_returns_empty_result(self) -> None:
        """The probe returns an empty result when the articulation is missing."""
        test = gain_tuner.DiscretizationSweepTest()
        test._articulation = None
        gen = test.run()
        with self.assertRaises(StopIteration) as ctx:
            gen.send(None)
        result = ctx.exception.value
        self.assertEqual(result.joint_position_commands.size, 0)


class TestDiscretizationSweepSmoke(TestGainTunerHarness):
    """Short registered dt-sweep single-level probe run."""

    async def test_discretization_probe_produces_metrics(self) -> None:
        """A single dt-sweep probe level produces per-joint settling/error metrics."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.7,
            joint_limit_revolute=(-45.0, 45.0),
        )
        await self._run_setup_and_compute_inertia(robot_path, num_physics_steps=80)
        self._gain_tuner.register_test(gain_tuner.GainsTestMode.DISCRETIZATION, gain_tuner.DiscretizationSweepTest())
        try:
            self._gain_tuner.initialize_gains_test(
                {
                    "test_mode": gain_tuner.GainsTestMode.DISCRETIZATION,
                    "joint_indices": [0],
                    "timeout": 2.0,
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
            self.assertTrue(done, "dt-sweep probe should finish within step budget")
            metrics = self._gain_tuner.get_test_result_metrics()
            self.assertIn(0, metrics)
            self.assertEqual(metrics[0].get("test_type"), "discretization_sweep")
            self.assertIn("settling_time", metrics[0])
            self.assertIn("steady_state_error", metrics[0])
        finally:
            self._gain_tuner.unregister_test(gain_tuner.GainsTestMode.DISCRETIZATION)
