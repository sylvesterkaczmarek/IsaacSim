# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Results panels: metric rows, table rows, verdicts, chart series, and widgets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import omni.kit.test
import omni.ui as ui
from isaacsim.robot_setup.gain_tuner.ui.gains_tuner_backend import GainsTestMode
from isaacsim.robot_setup.gain_tuner.ui.global_variables import (
    GT_DT_SWEEP_VERDICT_ALL_ACCURATE,
    GT_DT_SWEEP_VERDICT_DEGRADED,
    GT_DT_SWEEP_VERDICT_DID_NOT_SETTLE,
    GT_SNAP_VERDICT_ALL_PASSED,
    GT_SNAP_VERDICT_SOME_BLOCKED,
    GT_SNAP_VERDICT_SOME_FAILED,
    GT_STRESS_VERDICT_ALL_STABLE,
    GT_STRESS_VERDICT_UNSTABLE,
    GT_TEST_INFO_ACCURACY_CLIFF,
    GT_TEST_INFO_ERROR_DEGRAD,
    GT_TEST_INFO_LOWER_MAX_ERROR,
    GT_TEST_INFO_LOWER_MEAN_ERROR,
    GT_TEST_INFO_LOWER_SETTLE,
    GT_TEST_INFO_MAX_VELOCITY,
    GT_TEST_INFO_OVERSHOOT,
    GT_TEST_INFO_PEAK_ERROR,
    GT_TEST_INFO_RESULT,
    GT_TEST_INFO_RMS_ERROR,
    GT_TEST_INFO_SETTLE_AT_TARGET,
    GT_TEST_INFO_SS_ERROR,
    GT_TEST_INFO_TARGET_DT,
    GT_TEST_INFO_TEST_TYPE,
    GT_TEST_INFO_TRIGGER_TIME,
    GT_TEST_INFO_TRIGGER_VELOCITY,
    GT_TEST_INFO_UPPER_MAX_ERROR,
    GT_TEST_INFO_UPPER_MEAN_ERROR,
    GT_TEST_INFO_UPPER_SETTLE,
)
from isaacsim.robot_setup.gain_tuner.ui.ui_builder import UIBuilder


class TestDofMetricRows(omni.kit.test.AsyncTestCase):
    """Per-DOF validation metric-row assembly (E)."""

    async def test_linear_peak_and_rms(self) -> None:
        """Linear DOF: peak/RMS use base units and the leading Test row names the run."""
        cmd = np.zeros(4)
        obs = np.full(4, 0.1)
        rows = UIBuilder._dof_metric_rows(cmd, obs, is_rotational=False, last_test_mode=GainsTestMode.STEP)
        row_map = dict(rows)
        self.assertEqual(row_map[GT_TEST_INFO_TEST_TYPE], "Step Function")
        self.assertEqual(row_map[GT_TEST_INFO_PEAK_ERROR], "0.100 m")
        self.assertEqual(row_map[GT_TEST_INFO_RMS_ERROR], "0.100 m")
        # Flat command -> not step-like -> no overshoot.
        self.assertNotIn(GT_TEST_INFO_OVERSHOOT, row_map)

    async def test_rotational_scaling_rad_to_deg(self) -> None:
        """Rotational DOF: errors scale rad -> deg (180/pi) and use the degree unit."""
        cmd = np.zeros(2)
        obs = np.full(2, np.pi / 180.0)  # 1 degree in radians
        rows = UIBuilder._dof_metric_rows(cmd, obs, is_rotational=True, last_test_mode=None)
        row_map = dict(rows)
        self.assertEqual(row_map[GT_TEST_INFO_PEAK_ERROR], "1.000 \u00b0")
        self.assertEqual(row_map[GT_TEST_INFO_RMS_ERROR], "1.000 \u00b0")

    async def test_no_test_row_when_mode_unset(self) -> None:
        """A None mode omits the leading Test row."""
        rows = UIBuilder._dof_metric_rows(np.zeros(3), np.zeros(3), is_rotational=False, last_test_mode=None)
        self.assertNotIn(GT_TEST_INFO_TEST_TYPE, dict(rows))

    async def test_overshoot_row_present_for_step(self) -> None:
        """STEP step-like response includes an Overshoot row."""
        cmd = np.concatenate([np.zeros(10), np.full(90, 10.0)])
        obs = np.concatenate([np.linspace(0.0, 12.0, 50), np.full(50, 10.0)])
        rows = UIBuilder._dof_metric_rows(cmd, obs, is_rotational=False, last_test_mode=GainsTestMode.STEP)
        row_map = dict(rows)
        self.assertIn(GT_TEST_INFO_OVERSHOOT, row_map)
        self.assertEqual(row_map[GT_TEST_INFO_OVERSHOOT], "20.0 %")

    async def test_empty_arrays_return_empty(self) -> None:
        """No samples -> empty row list."""
        self.assertEqual(
            UIBuilder._dof_metric_rows(
                np.array([]), np.array([]), is_rotational=False, last_test_mode=GainsTestMode.STEP
            ),
            [],
        )


class TestStressSnapMetricRows(omni.kit.test.AsyncTestCase):
    """Develop-parity metric sets for the Stress and Snap-to-Limits results (E2).

    Stress and Snap report the same rich per-joint metrics as the develop build,
    read from the backend's per-DOF metrics dict rather than recomputed from the
    trajectory, so both the "Current Test Results" detail summary and the Test
    Gains tab (which share ``_compute_dof_validation_summary``) match develop.
    """

    async def test_stress_metric_rows(self) -> None:
        """Stress reports Max Velocity / Trigger Time / Trigger Velocity / Result."""
        metric = {
            "max_velocity": 12.345,
            "trigger_time": 1.5,
            "trigger_velocity": 8.2,
            "status": "unstable",
        }
        rows = dict(UIBuilder._stress_metric_rows(metric))
        self.assertEqual(rows[GT_TEST_INFO_TEST_TYPE], "Stress")
        self.assertEqual(rows[GT_TEST_INFO_MAX_VELOCITY], "12.35")
        self.assertEqual(rows[GT_TEST_INFO_TRIGGER_TIME], "1.500 s")
        self.assertEqual(rows[GT_TEST_INFO_TRIGGER_VELOCITY], "8.20")
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "UNSTABLE")

    async def test_stress_stable_uses_na_for_untriggered(self) -> None:
        """A stable joint never triggered -> Trigger Time/Velocity render as N/A."""
        metric = {
            "max_velocity": 3.0,
            "trigger_time": float("nan"),
            "trigger_velocity": float("nan"),
            "status": "stable",
        }
        rows = dict(UIBuilder._stress_metric_rows(metric))
        self.assertEqual(rows[GT_TEST_INFO_TRIGGER_TIME], "N/A")
        self.assertEqual(rows[GT_TEST_INFO_TRIGGER_VELOCITY], "N/A")
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "STABLE")

    async def test_snap_metric_rows_rotational(self) -> None:
        """Snap reports per-limit mean/max error (deg-scaled) + settle + PASS/BLOCKED/FAIL."""
        metric = {
            "lower_position_error": np.pi / 180.0,  # 1 deg
            "lower_max_error": 2.0 * np.pi / 180.0,  # 2 deg
            "lower_settling_time": 0.25,
            "upper_position_error": 3.0 * np.pi / 180.0,  # 3 deg
            "upper_max_error": 4.0 * np.pi / 180.0,  # 4 deg
            "upper_settling_time": float("nan"),
            "status": "pass",
        }
        rows = dict(UIBuilder._snap_metric_rows(metric, is_rotational=True))
        self.assertEqual(rows[GT_TEST_INFO_TEST_TYPE], "Snap to Limits")
        self.assertEqual(rows[GT_TEST_INFO_LOWER_MEAN_ERROR], "1.0000 \u00b0")
        self.assertEqual(rows[GT_TEST_INFO_LOWER_MAX_ERROR], "2.0000 \u00b0")
        self.assertEqual(rows[GT_TEST_INFO_LOWER_SETTLE], "0.250 s")
        self.assertEqual(rows[GT_TEST_INFO_UPPER_MEAN_ERROR], "3.0000 \u00b0")
        self.assertEqual(rows[GT_TEST_INFO_UPPER_MAX_ERROR], "4.0000 \u00b0")
        self.assertEqual(rows[GT_TEST_INFO_UPPER_SETTLE], "N/A")  # NaN settle -> N/A
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "PASS")

    async def test_snap_metric_rows_linear_and_status(self) -> None:
        """Linear DOF uses base units; status maps through to the Result row."""
        metric = {
            "lower_position_error": 0.01,
            "lower_max_error": 0.02,
            "lower_settling_time": 0.1,
            "upper_position_error": 0.03,
            "upper_max_error": 0.04,
            "upper_settling_time": 0.2,
            "status": "blocked",
        }
        rows = dict(UIBuilder._snap_metric_rows(metric, is_rotational=False))
        self.assertEqual(rows[GT_TEST_INFO_LOWER_MEAN_ERROR], "0.0100 m")
        self.assertEqual(rows[GT_TEST_INFO_UPPER_MAX_ERROR], "0.0400 m")
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "BLOCKED")


class TestDiscretizationMetricRows(omni.kit.test.AsyncTestCase):
    """dt-sweep per-DOF result rows shared by the detail summary and Test Gains tab."""

    @staticmethod
    def _metric() -> dict:
        return {
            "test_type": "discretization_sweep",
            "status": "degraded",
            "accuracy_cliff_dt": 0.02,
            "target_dt": 1.0 / 120.0,
            "target_level": {
                "dt": 1.0 / 120.0,
                "settling_time": 0.5,
                "settle_degradation": 0.1,
                "steady_state_error": np.pi / 180.0,  # 1 deg in rad
                "error_degradation": 0.05,
                "settled": True,
            },
        }

    async def test_rows_rotational_scale_and_format(self) -> None:
        """Short summary folds settle degradation into Settle @ Target; SS error in degrees."""
        rows = dict(UIBuilder._discretization_metric_rows(self._metric(), is_rotational=True))
        self.assertEqual(rows[GT_TEST_INFO_TEST_TYPE], "dt Sweep")
        self.assertEqual(rows[GT_TEST_INFO_TARGET_DT], "0.00833 s (120 Hz)")
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "DEGRADED")
        self.assertEqual(rows[GT_TEST_INFO_SETTLE_AT_TARGET], "0.500 s (+10.0%)")
        self.assertEqual(rows[GT_TEST_INFO_SS_ERROR], "1.0000 \u00b0")
        self.assertEqual(rows[GT_TEST_INFO_ACCURACY_CLIFF], "0.02000 s (50 Hz)")
        # Error degradation is meaningfully positive here (+5%), so it is shown.
        self.assertEqual(rows[GT_TEST_INFO_ERROR_DEGRAD], "+5.0 %")
        # The standalone Settling Time row is no longer part of the short summary.
        self.assertNotIn("Settling Time (2%):", rows)

    async def test_error_degrad_row_dropped_when_zero(self) -> None:
        """The uninformative +0.0 % error-degradation row is dropped from the summary."""
        metric = self._metric()
        metric["target_level"]["error_degradation"] = 0.0
        rows = dict(UIBuilder._discretization_metric_rows(metric, is_rotational=True))
        self.assertNotIn(GT_TEST_INFO_ERROR_DEGRAD, rows)
        # The other core rows remain.
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "DEGRADED")
        self.assertEqual(rows[GT_TEST_INFO_SETTLE_AT_TARGET], "0.500 s (+10.0%)")

    async def test_rows_missing_reference_render_na(self) -> None:
        """Missing settle / SS error / cliff render as N/A and status maps through."""
        metric = {
            "status": "did_not_settle",
            "accuracy_cliff_dt": None,
            "target_dt": 1.0 / 120.0,
            "target_level": {
                "dt": 1.0 / 120.0,
                "settling_time": None,
                "settle_degradation": None,
                "steady_state_error": float("nan"),
                "error_degradation": None,
                "settled": False,
            },
        }
        rows = dict(UIBuilder._discretization_metric_rows(metric, is_rotational=False))
        self.assertEqual(rows[GT_TEST_INFO_SETTLE_AT_TARGET], "N/A")
        self.assertEqual(rows[GT_TEST_INFO_SS_ERROR], "N/A")
        self.assertEqual(rows[GT_TEST_INFO_ACCURACY_CLIFF], "N/A")
        self.assertEqual(rows[GT_TEST_INFO_RESULT], "DID NOT SETTLE")
        # No error-degradation row when the value is unavailable.
        self.assertNotIn(GT_TEST_INFO_ERROR_DEGRAD, rows)


class TestDiscretizationTableRow(omni.kit.test.AsyncTestCase):
    """All-joints dt-sweep table row construction + result color mapping (pure)."""

    @staticmethod
    def _metric(status: str = "degraded") -> dict:
        return {
            "test_type": "discretization_sweep",
            "status": status,
            "accuracy_cliff_dt": 0.02,
            "target_dt": 1.0 / 120.0,
            "target_level": {
                "dt": 1.0 / 120.0,
                "settling_time": 0.5,
                "settle_degradation": 0.1,
                "steady_state_error": np.pi / 180.0,  # 1 deg in rad
                "error_degradation": 0.05,
                "settled": True,
            },
        }

    async def test_row_fields_rotational(self) -> None:
        """Row carries joint, result, settle @ target (with delta), SS error, cliff, status."""
        row = UIBuilder._discretization_table_row("shoulder", self._metric(), is_rotational=True)
        joint, result, settle, ss_error, cliff, status = row
        self.assertEqual(joint, "shoulder")
        self.assertEqual(result, "DEGRADED")
        self.assertEqual(settle, "0.500 s (+10.0%)")
        self.assertEqual(ss_error, "1.0000 \u00b0")
        self.assertEqual(cliff, "0.02000 s (50 Hz)")
        self.assertEqual(status, "degraded")

    async def test_row_linear_units_and_na(self) -> None:
        """Linear DOF reports SS error in meters; missing cliff/settle become N/A."""
        metric = {
            "status": "did_not_settle",
            "accuracy_cliff_dt": None,
            "target_dt": 1.0 / 120.0,
            "target_level": {
                "settling_time": None,
                "settle_degradation": None,
                "steady_state_error": float("nan"),
                "settled": False,
            },
        }
        row = UIBuilder._discretization_table_row("slider", metric, is_rotational=False)
        _joint, result, settle, ss_error, cliff, status = row
        self.assertEqual(result, "DID NOT SETTLE")
        self.assertEqual(settle, "N/A")
        self.assertEqual(ss_error, "N/A")
        self.assertEqual(cliff, "N/A")
        self.assertEqual(status, "did_not_settle")


class TestDiscretizationVerdict(omni.kit.test.AsyncTestCase):
    """Scene-level verdict aggregation over per-joint classifications (pure)."""

    async def test_verdict_precedence(self) -> None:
        """Worst outcome wins (did_not_settle > degraded > accurate); empty is conservative."""
        cases = [
            (["accurate", "accurate"], "accurate", GT_DT_SWEEP_VERDICT_ALL_ACCURATE),
            (["accurate", "degraded", "accurate"], "degraded", GT_DT_SWEEP_VERDICT_DEGRADED),
            (["accurate", "degraded", "did_not_settle"], "did_not_settle", GT_DT_SWEEP_VERDICT_DID_NOT_SETTLE),
            ([], "did_not_settle", GT_DT_SWEEP_VERDICT_DID_NOT_SETTLE),
        ]
        for statuses, expected, text in cases:
            with self.subTest(statuses=statuses):
                verdict = UIBuilder._dt_sweep_verdict(statuses)
                self.assertEqual(verdict, expected)
                self.assertEqual(UIBuilder._dt_sweep_verdict_text(verdict), text)


class TestDiscretizationProgressLabel(omni.kit.test.AsyncTestCase):
    """During-run dt-level progress label formatting (pure)."""

    async def test_label_names_level_hz_and_dt(self) -> None:
        """Label names the 1-based level, its frequency, and its timestep."""
        text = UIBuilder._format_dt_level_progress(2, 10, 1.0 / 120.0)
        self.assertEqual(text, "dt level 3/10: 120 Hz (dt=0.00833 s)")

    async def test_label_clamps_index_into_range(self) -> None:
        """The 1-based level is clamped to [1, num_levels]."""
        self.assertTrue(UIBuilder._format_dt_level_progress(-5, 5, 1.0 / 60.0).startswith("dt level 1/5"))
        self.assertTrue(UIBuilder._format_dt_level_progress(99, 5, 1.0 / 60.0).startswith("dt level 5/5"))


class TestDiscretizationChartSeries(omni.kit.test.AsyncTestCase):
    """vs-dt chart series extraction (ascending dt in ms; skips/NaN filtered) (pure)."""

    @staticmethod
    def _metric() -> dict:
        # Coarse-to-fine, mirroring the sweep order; one skipped + one NaN level.
        return {
            "dt_sweep": [
                {"dt": 0.02, "settling_time": 0.9, "steady_state_error": 0.02, "skipped": False},
                {"dt": 0.01, "settling_time": None, "steady_state_error": float("nan"), "skipped": True},
                {"dt": 0.005, "settling_time": 0.5, "steady_state_error": 0.01, "skipped": False},
            ]
        }

    async def test_settle_series_sorted_ascending_ms(self) -> None:
        """dt is returned ascending in milliseconds; skipped/NaN levels are dropped."""
        xs, ys = UIBuilder._dt_sweep_chart_series(self._metric(), "settling_time", scale=1.0)
        self.assertEqual(xs, [5.0, 20.0])  # 0.005 s and 0.02 s in ms, ascending
        self.assertEqual(ys, [0.5, 0.9])

    async def test_ss_error_series_applies_scale(self) -> None:
        """The y scale (rad -> deg) is applied to the steady-state-error series."""
        xs, ys = UIBuilder._dt_sweep_chart_series(self._metric(), "steady_state_error", scale=100.0)
        self.assertEqual(xs, [5.0, 20.0])
        self.assertAlmostEqual(ys[0], 1.0)  # 0.01 * 100
        self.assertAlmostEqual(ys[1], 2.0)  # 0.02 * 100

    async def test_empty_when_all_skipped(self) -> None:
        """A fully skipped sweep yields no plottable points."""
        metric = {"dt_sweep": [{"dt": 0.02, "settling_time": 1.0, "steady_state_error": 0.0, "skipped": True}]}
        xs, ys = UIBuilder._dt_sweep_chart_series(metric, "settling_time", scale=1.0)
        self.assertEqual(xs, [])
        self.assertEqual(ys, [])


class TestSnapTableRow(omni.kit.test.AsyncTestCase):
    """All-joints Snap-to-Limits table row construction + result color mapping (pure)."""

    async def test_row_fields_rotational(self) -> None:
        """Row carries joint, per-limit mean/max (deg) + settle, result, and status."""
        metric = {
            "lower_position_error": np.pi / 180.0,  # 1 deg
            "lower_max_error": 2.0 * np.pi / 180.0,  # 2 deg
            "lower_settling_time": 0.25,
            "upper_position_error": 3.0 * np.pi / 180.0,  # 3 deg
            "upper_max_error": 4.0 * np.pi / 180.0,  # 4 deg
            "upper_settling_time": float("nan"),
            "status": "pass",
        }
        row = UIBuilder._snap_table_row("shoulder", metric, is_rotational=True)
        joint, lmean, lmax, lsettle, umean, umax, usettle, result, status = row
        self.assertEqual(joint, "shoulder")
        self.assertEqual(lmean, "1.0000 \u00b0")
        self.assertEqual(lmax, "2.0000 \u00b0")
        self.assertEqual(lsettle, "0.250 s")
        self.assertEqual(umean, "3.0000 \u00b0")
        self.assertEqual(umax, "4.0000 \u00b0")
        self.assertEqual(usettle, "N/A")  # NaN settle -> N/A
        self.assertEqual(result, "PASS")
        self.assertEqual(status, "pass")

    async def test_row_linear_units_and_status(self) -> None:
        """Linear DOF reports errors in meters; status maps through to the result cell."""
        metric = {
            "lower_position_error": 0.01,
            "lower_max_error": 0.02,
            "lower_settling_time": 0.1,
            "upper_position_error": 0.03,
            "upper_max_error": 0.04,
            "upper_settling_time": 0.2,
            "status": "blocked",
        }
        row = UIBuilder._snap_table_row("slider", metric, is_rotational=False)
        joint, lmean, _lmax, _lsettle, _umean, umax, _usettle, result, status = row
        self.assertEqual(lmean, "0.0100 m")
        self.assertEqual(umax, "0.0400 m")
        self.assertEqual(result, "BLOCKED")
        self.assertEqual(status, "blocked")


class TestSnapVerdict(omni.kit.test.AsyncTestCase):
    """Scene-level snap verdict aggregation over per-joint statuses (pure)."""

    async def test_verdict_precedence(self) -> None:
        """Worst outcome wins (fail > blocked > pass); empty defaults conservatively to fail."""
        cases = [
            (["pass", "pass"], "pass", GT_SNAP_VERDICT_ALL_PASSED),
            (["pass", "blocked", "pass"], "blocked", GT_SNAP_VERDICT_SOME_BLOCKED),
            (["pass", "blocked", "fail"], "fail", GT_SNAP_VERDICT_SOME_FAILED),
            ([], "fail", GT_SNAP_VERDICT_SOME_FAILED),
        ]
        for statuses, expected, text in cases:
            with self.subTest(statuses=statuses):
                verdict = UIBuilder._snap_verdict(statuses)
                self.assertEqual(verdict, expected)
                self.assertEqual(UIBuilder._snap_verdict_text(verdict), text)


class TestStressTableRow(omni.kit.test.AsyncTestCase):
    """All-joints Stress-test table row construction + result color mapping (pure)."""

    async def test_row_fields_unstable(self) -> None:
        """Row carries joint, max vel, trigger time/vel, result, and status."""
        metric = {
            "max_velocity": 12.345,
            "trigger_time": 1.5,
            "trigger_velocity": 8.2,
            "status": "unstable",
        }
        row = UIBuilder._stress_table_row("elbow", metric)
        joint, max_vel, trigger_time, trigger_vel, result, status = row
        self.assertEqual(joint, "elbow")
        self.assertEqual(max_vel, "12.35")
        self.assertEqual(trigger_time, "1.500 s")
        self.assertEqual(trigger_vel, "8.20")
        self.assertEqual(result, "UNSTABLE")
        self.assertEqual(status, "unstable")

    async def test_stable_uses_na_for_untriggered(self) -> None:
        """A stable joint never triggered -> trigger time/velocity render as N/A."""
        metric = {
            "max_velocity": 3.0,
            "trigger_time": float("nan"),
            "trigger_velocity": float("nan"),
            "status": "stable",
        }
        row = UIBuilder._stress_table_row("wrist", metric)
        _joint, max_vel, trigger_time, trigger_vel, result, status = row
        self.assertEqual(max_vel, "3.00")
        self.assertEqual(trigger_time, "N/A")
        self.assertEqual(trigger_vel, "N/A")
        self.assertEqual(result, "STABLE")
        self.assertEqual(status, "stable")


class TestStressVerdict(omni.kit.test.AsyncTestCase):
    """Scene-level stress verdict aggregation over per-joint statuses (pure)."""

    async def test_verdict_precedence(self) -> None:
        """Any unstable joint downgrades the scene; empty defaults to stable."""
        cases = [
            (["stable", "stable"], "stable", GT_STRESS_VERDICT_ALL_STABLE),
            (["stable", "unstable"], "unstable", GT_STRESS_VERDICT_UNSTABLE),
            ([], "stable", GT_STRESS_VERDICT_ALL_STABLE),
        ]
        for statuses, expected, text in cases:
            with self.subTest(statuses=statuses):
                verdict = UIBuilder._stress_verdict(statuses)
                self.assertEqual(verdict, expected)
                self.assertEqual(UIBuilder._stress_verdict_text(verdict), text)


class TestResultStatusColors(omni.kit.test.AsyncTestCase):
    """Per-mode result-status -> color maps for all three results tables (pure).

    Consolidates the previously duplicated per-mode ``test_status_colors_map_*``
    cases into one parametrized check.
    """

    async def test_status_color_maps(self) -> None:
        """pass/stable/accurate -> green, blocked/degraded -> amber, everything else -> red."""
        from isaacsim.robot_setup.gain_tuner.ui.style import FAIL_COLOR, PASS_COLOR, WARNING_COLOR

        cases = [
            (UIBuilder._snap_status_color, "pass", PASS_COLOR),
            (UIBuilder._snap_status_color, "blocked", WARNING_COLOR),
            (UIBuilder._snap_status_color, "fail", FAIL_COLOR),
            (UIBuilder._snap_status_color, "unknown", FAIL_COLOR),
            (UIBuilder._stress_status_color, "stable", PASS_COLOR),
            (UIBuilder._stress_status_color, "unstable", FAIL_COLOR),
            (UIBuilder._stress_status_color, "unknown", FAIL_COLOR),
            (UIBuilder._dt_sweep_status_color, "accurate", PASS_COLOR),
            (UIBuilder._dt_sweep_status_color, "degraded", WARNING_COLOR),
            (UIBuilder._dt_sweep_status_color, "did_not_settle", FAIL_COLOR),
            (UIBuilder._dt_sweep_status_color, "unknown", FAIL_COLOR),
        ]
        for color_fn, status, expected in cases:
            with self.subTest(fn=color_fn.__name__, status=status):
                self.assertEqual(color_fn(status), expected)


class TestResultsPanelWidgets(omni.kit.test.AsyncTestCase):
    """Real widget construction for the Snap / Stress / dt-Sweep results panels.

    Builds each panel inside a live ``ui.Window`` frame against fabricated metrics
    and asserts the rendered row count and scene-level verdict text, exercising the
    ``results_panels`` module extracted from ``UIBuilder`` (verdict banner + table +
    empty state). The banner/row/empty primitives are wrapped to record calls while
    still constructing the real widgets; the vs-dt charts are stubbed (covered by
    their own series test).
    """

    @staticmethod
    def _shell(metrics: dict):
        ub = UIBuilder.__new__(UIBuilder)
        ub._gains_tuner = SimpleNamespace(
            get_test_result_metrics=lambda: metrics,
            get_articulation=lambda: None,  # -> dof_names None, is_rotational False (linear formatting)
            is_data_ready=lambda: True,
        )
        return ub

    @staticmethod
    def _instrument(ub):
        rows: list = []
        banners: list = []
        empties: list = []

        def rec_row(cells):
            rows.append(cells)
            UIBuilder._build_results_table_row(ub, cells)

        def rec_banner(text, color):
            banners.append((text, color))
            UIBuilder._build_verdict_banner(ub, text, color)

        def rec_empty():
            empties.append(True)
            UIBuilder._build_test_info_empty_state(ub)

        ub._build_results_table_row = rec_row
        ub._build_verdict_banner = rec_banner
        ub._build_test_info_empty_state = rec_empty
        ub._build_dt_sweep_vs_dt_charts = lambda *a, **k: None
        return rows, banners, empties

    @staticmethod
    def _snap_metric(status: str) -> dict:
        return {
            "lower_position_error": 0.01,
            "lower_max_error": 0.02,
            "lower_settling_time": 0.1,
            "upper_position_error": 0.03,
            "upper_max_error": 0.04,
            "upper_settling_time": 0.2,
            "status": status,
        }

    @staticmethod
    def _stress_metric(status: str) -> dict:
        return {
            "max_velocity": 3.0,
            "trigger_time": 1.0,
            "trigger_velocity": 2.0,
            "status": status,
            "test_mode": "random_walk",
            "seed": 42,
        }

    @staticmethod
    def _dt_metric(status: str) -> dict:
        return {
            "test_type": "discretization_sweep",
            "status": status,
            "accuracy_cliff_dt": 0.02,
            "target_dt": 1.0 / 120.0,
            "target_level": {
                "dt": 1.0 / 120.0,
                "settling_time": 0.5,
                "settle_degradation": 0.1,
                "steady_state_error": 0.001,
                "error_degradation": 0.05,
                "settled": True,
            },
            "dt_sweep": [{"dt": 0.02, "settling_time": 0.9, "steady_state_error": 0.02, "skipped": False}],
        }

    async def test_snap_panel_rows_and_verdict(self) -> None:
        """The Snap panel builds one row per DOF and the fail verdict wins."""
        ub = self._shell({0: self._snap_metric("pass"), 1: self._snap_metric("fail")})
        rows, banners, _ = self._instrument(ub)
        window = ui.Window("gt_snap_panel_test", width=640, height=400)
        with window.frame:
            with ui.VStack():
                ub._build_snap_results()
        self.assertEqual(len(rows), 2)
        self.assertEqual(banners[0][0], GT_SNAP_VERDICT_SOME_FAILED)
        window.destroy()

    async def test_stress_panel_rows_and_verdict(self) -> None:
        """The Stress panel builds one row per DOF; any unstable joint downgrades the verdict."""
        ub = self._shell({0: self._stress_metric("stable"), 1: self._stress_metric("unstable")})
        rows, banners, _ = self._instrument(ub)
        window = ui.Window("gt_stress_panel_test", width=640, height=400)
        with window.frame:
            with ui.VStack():
                ub._build_stress_results()
        self.assertEqual(len(rows), 2)
        self.assertEqual(banners[0][0], GT_STRESS_VERDICT_UNSTABLE)
        window.destroy()

    async def test_dt_sweep_panel_rows_and_verdict(self) -> None:
        """The dt-Sweep panel builds one row per DOF; a degraded joint downgrades an otherwise-accurate scene."""
        ub = self._shell({0: self._dt_metric("accurate"), 1: self._dt_metric("degraded")})
        rows, banners, _ = self._instrument(ub)
        window = ui.Window("gt_dt_panel_test", width=640, height=400)
        with window.frame:
            with ui.VStack():
                ub._build_dt_sweep_results()
        self.assertEqual(len(rows), 2)
        self.assertEqual(banners[0][0], GT_DT_SWEEP_VERDICT_DEGRADED)
        window.destroy()

    async def test_empty_state_when_no_matching_metrics(self) -> None:
        """A panel whose mode-filter excludes every metric renders the empty state (no rows/banner)."""
        # A dt-sweep metric is filtered OUT by the Snap panel's keep predicate.
        ub = self._shell({0: self._dt_metric("accurate")})
        rows, banners, empties = self._instrument(ub)
        window = ui.Window("gt_empty_panel_test", width=640, height=400)
        with window.frame:
            with ui.VStack():
                ub._build_snap_results()
        self.assertEqual(len(rows), 0)
        self.assertEqual(len(banners), 0)
        self.assertEqual(len(empties), 1)
        window.destroy()


class TestDetailFieldRowWidgets(omni.kit.test.AsyncTestCase):
    """Real widget construction for the shared detail-panel field rows (``_field_row``)."""

    class _FakeAttr:
        """Minimal stand-in for a USD attribute the gain field writes back to."""

        def __init__(self) -> None:
            self.writes: list = []

        def IsValid(self) -> bool:
            return True

        def Set(self, value) -> None:
            self.writes.append(value)

    async def test_editable_gain_field_writes_back_to_attr(self) -> None:
        """Editing an editable gain field writes the new value to its bound attribute."""
        import omni.kit.app

        attr = self._FakeAttr()
        ub = UIBuilder.__new__(UIBuilder)
        ub._suspend_detail_writes = False
        ub._sync_selected_row_to_table = lambda: None
        window = ui.Window("gt_field_rw_test", width=320, height=200)
        with window.frame:
            with ui.VStack():
                model = ub._gain_field_row("Kp", 1.0, attr, read_only=False, unit="deg")
        model.set_value(2.5)
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(attr.writes[-1], 2.5)
        window.destroy()

    async def test_readonly_gain_field_has_no_writeback(self) -> None:
        """A read-only gain field wires no change handler, so edits never touch the attribute."""
        attr = self._FakeAttr()
        ub = UIBuilder.__new__(UIBuilder)
        ub._suspend_detail_writes = False
        ub._sync_selected_row_to_table = lambda: None
        window = ui.Window("gt_field_ro_test", width=320, height=200)
        with window.frame:
            with ui.VStack():
                model = ub._gain_field_row("Kp", 1.0, attr, read_only=True)
        model.set_value(9.0)
        self.assertEqual(attr.writes, [])
        window.destroy()

    async def test_placeholder_and_readonly_rows_build(self) -> None:
        """A None-model row and the advanced read-only placeholder build without a backing model."""
        ub = UIBuilder.__new__(UIBuilder)
        window = ui.Window("gt_field_placeholder_test", width=320, height=200)
        with window.frame:
            with ui.VStack():
                placeholder = ub._field_row("Newton param", None, width=160)
                ub._adv_readonly_field("Pending API")
        self.assertIsNone(placeholder)
        window.destroy()
