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

# ruff: noqa: ANN001, ANN202, D101, D102

"""Tests for post-solve per-parameter confidence estimation."""

from __future__ import annotations

import math

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.analysis import (
    CONFIDENCE_BOUND_LIMITED,
    CONFIDENCE_WEAKLY_CONSTRAINED,
    CONFIDENCE_WELL_CONSTRAINED,
    NOISE_FLOOR_COST_FACTOR,
    _metrics_by_key,
    assess_segmented_noise_floor,
    build_parameter_confidence_report,
    estimate_series_noise_power,
)
from isaacsim.robot_setup.sysid.optimizer_base import OptimizerConfig
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.residual_config import ResidualWeightConfig
from isaacsim.robot_setup.sysid.rollout_plot_data import SysIdRolloutPlotData
from isaacsim.robot_setup.sysid.trajectory_csv import TrajectoryDataset
from isaacsim.robot_setup.sysid.trajectory_segments import (
    ChunkValidationMetric,
    TelemetryChunkRunSpec,
    TrajectoryChunk,
)


def _trajectory(num_samples: int = 10) -> TrajectoryDataset:
    values = np.zeros((num_samples, 1), dtype=np.float64)
    return TrajectoryDataset(
        times=np.arange(num_samples, dtype=np.float64) * 0.01,
        positions=values.copy(),
        velocities=values.copy(),
        commands=values.copy(),
    )


def _chunk(trajectory: TrajectoryDataset) -> TrajectoryChunk:
    return TrajectoryChunk(
        spec=TelemetryChunkRunSpec(name="Validation", role="validation", start=0.0, end=0.09),
        trajectory=trajectory,
        sample_count=int(trajectory.times.shape[0]),
        duration_seconds=float(trajectory.times[-1] - trajectory.times[0]),
    )


class _QuadraticOptimizer:
    """Fake optimizer with cost c(theta) = c0 + sum_i a_i * (theta_i - center_i)^2.

    Args:
        c0: Constructor value for ``c0``.
        coefficients: Constructor value for ``coefficients``.
        centers: Constructor value for ``centers``.
        nominal_plot: Constructor value for ``nominal_plot``.
    """

    def __init__(
        self,
        c0: float,
        coefficients: list[float],
        centers: list[float],
        nominal_plot: SysIdRolloutPlotData | None = None,
    ) -> None:
        self._c0 = float(c0)
        self._coefficients = coefficients
        self._centers = centers
        self._nominal_plot = nominal_plot
        self._bridge = type("B", (), {"device": torch.device("cpu")})()
        self.batches: list[torch.Tensor] = []

    async def compute_costs_for_theta_batch(self, config, theta_batch, on_progress=None, include_plot_data=True):
        self.batches.append(theta_batch.detach().clone())
        costs = torch.full((theta_batch.shape[0],), self._c0, dtype=torch.float32)
        for column, (coefficient, center) in enumerate(zip(self._coefficients, self._centers)):
            costs = costs + coefficient * (theta_batch[:, column] - center) ** 2
        plot = self._nominal_plot if include_plot_data else None
        return costs, costs.reshape(-1, 1), plot


def _residual_plot(residual_pos: np.ndarray, residual_vel: np.ndarray) -> SysIdRolloutPlotData:
    """Plot data whose simulated-minus-measured series equal the given residuals.

    Args:
        residual_pos: Value supplied for ``residual_pos``.
        residual_vel: Value supplied for ``residual_vel``.

    Returns:
        Result produced by the operation.
    """
    num_samples = residual_pos.shape[0]
    times = (np.arange(num_samples) * 0.01).tolist()
    zeros = [np.zeros(num_samples).tolist()]
    return SysIdRolloutPlotData(
        times=times,
        measured_positions=zeros,
        simulated_positions=[residual_pos.tolist()],
        measured_velocities=zeros,
        simulated_velocities=[residual_vel.tolist()],
    )


def _config(theta_min: list[float], theta_max: list[float], num_params: int) -> OptimizerConfig:
    entries = [SysIdParameterEntry(SysIdParameterType.JOINT_FRICTION, index) for index in range(num_params)]
    return OptimizerConfig(
        trajectory=_trajectory(),
        param_entries=entries,
        theta_initial=torch.zeros(num_params),
        theta_min=torch.tensor(theta_min),
        theta_max=torch.tensor(theta_max),
        epsilon=1e-4,
        damping_initial=1e-2,
        max_iterations=1,
        max_rollout_steps=10,
    )


class ParameterConfidenceTests(omni.kit.test.AsyncTestCase):
    async def test_curvature_and_range_recovered_on_quadratic(self) -> None:
        c0, a = 0.5, 400.0
        theta_hat = [1.0]
        optimizer = _QuadraticOptimizer(c0, [a], theta_hat)
        config = _config([0.0], [2.0], 1)
        report = await build_parameter_confidence_report(optimizer, config, [_chunk(_trajectory())], theta_hat)
        entry = report.entries[0]
        # Second derivative of a*(d)^2 is 2a; range where cost rises 10%: sqrt(0.1*c0/a).
        self.assertAlmostEqual(entry.curvature, 2.0 * a, delta=0.05 * 2.0 * a)
        expected_range = math.sqrt(0.1 * c0 / a)
        self.assertAlmostEqual(entry.ten_percent_range, expected_range, delta=0.05 * expected_range)
        # Range (~0.011) is well under 10% of the [0, 2] span.
        self.assertEqual(entry.verdict, CONFIDENCE_WELL_CONSTRAINED)
        # All 2N+1 evaluations went through one batched call.
        self.assertEqual(len(optimizer.batches), 1)
        self.assertEqual(int(optimizer.batches[0].shape[0]), 3)

    async def test_flat_direction_is_weakly_constrained(self) -> None:
        c0 = 0.5
        theta_hat = [1.0]
        optimizer = _QuadraticOptimizer(c0, [1e-4], theta_hat)
        config = _config([0.0], [2.0], 1)
        report = await build_parameter_confidence_report(optimizer, config, [_chunk(_trajectory())], theta_hat)
        self.assertEqual(report.entries[0].verdict, CONFIDENCE_WEAKLY_CONSTRAINED)

    async def test_bound_pinned_parameter_reports_bound_limited(self) -> None:
        theta_hat = [0.0]  # sits on the lower bound -> one-sided perturbation
        optimizer = _QuadraticOptimizer(0.5, [400.0], theta_hat)
        config = _config([0.0], [2.0], 1)
        report = await build_parameter_confidence_report(optimizer, config, [_chunk(_trajectory())], theta_hat)
        self.assertEqual(report.entries[0].verdict, CONFIDENCE_BOUND_LIMITED)

    async def test_budget_degrades_to_one_sided_then_skips(self) -> None:
        theta_hat = [1.0, 1.0]
        config = _config([0.0, 0.0], [2.0, 2.0], 2)
        chunk = _chunk(_trajectory())
        centered = await build_parameter_confidence_report(
            _QuadraticOptimizer(0.5, [400.0, 400.0], theta_hat),
            config,
            [chunk],
            theta_hat,
        )
        # Full centered plan needs 5 evaluations x 10 samples = 50; cap at 30 -> one-sided (3 x 10).
        optimizer = _QuadraticOptimizer(0.5, [400.0, 400.0], theta_hat)
        report = await build_parameter_confidence_report(
            optimizer, config, [chunk], theta_hat, max_rollout_sample_budget=30
        )
        self.assertTrue(report.budget_capped)
        self.assertFalse(report.skipped_reason)
        self.assertEqual(report.rollout_samples_used, 30)
        self.assertEqual(len(report.entries), len(centered.entries))
        for centered_entry, one_sided_entry in zip(centered.entries, report.entries):
            self.assertIsNotNone(centered_entry.cost_minus)
            self.assertIsNone(one_sided_entry.cost_minus)
            self.assertEqual(one_sided_entry.verdict, centered_entry.verdict)
            self.assertAlmostEqual(
                one_sided_entry.curvature,
                centered_entry.curvature,
                delta=0.05 * abs(centered_entry.curvature),
            )
        # Cap below even the one-sided plan -> skipped entirely.
        report = await build_parameter_confidence_report(
            optimizer, config, [chunk], theta_hat, max_rollout_sample_budget=10
        )
        self.assertIn("budget", report.skipped_reason)
        self.assertEqual(report.entries, [])

    async def test_report_round_trips_and_summarizes(self) -> None:
        theta_hat = [1.0]
        optimizer = _QuadraticOptimizer(0.5, [400.0], theta_hat)
        config = _config([0.0], [2.0], 1)
        report = await build_parameter_confidence_report(optimizer, config, [_chunk(_trajectory())], theta_hat)
        payload = report.to_dict()
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(len(payload["entries"]), 1)
        summary = "\n".join(report.summary_lines())
        self.assertIn("sens. span", summary)
        self.assertNotIn("±", summary)
        # No plot data from this fake -> floor not estimated, verdicts stand un-gated.
        self.assertFalse(payload["noise_floor"]["estimated"])
        self.assertFalse(payload["noise_floor"]["at_noise_floor"])
        self.assertFalse(payload["entries"][0]["at_noise_floor"])


class NoiseFloorGateTests(omni.kit.test.AsyncTestCase):
    """The noise-floor validity gate must fire when the residual is noise-dominated."""

    def test_series_noise_power_recovers_white_noise_and_ignores_structure(self) -> None:
        rng = np.random.default_rng(1234)
        sigma = 1e-3
        noise = rng.normal(0.0, sigma, size=2000)
        estimate = estimate_series_noise_power(noise)
        self.assertAlmostEqual(estimate, sigma * sigma, delta=0.3 * sigma * sigma)
        # A smooth low-frequency sine has near-zero high-frequency power.
        smooth = 0.1 * np.sin(2.0 * math.pi * 0.5 * np.arange(2000) * 0.01)
        self.assertLess(estimate_series_noise_power(smooth), np.mean(smooth * smooth) / 100.0)
        self.assertIsNone(estimate_series_noise_power(np.zeros(4)))

    async def _report_for_residual(self, residual_pos: np.ndarray, residual_vel: np.ndarray):
        theta_hat = [1.0]
        plot = _residual_plot(residual_pos, residual_vel)
        optimizer = _QuadraticOptimizer(0.5, [400.0], theta_hat, nominal_plot=plot)
        config = _config([0.0], [2.0], 1)
        return await build_parameter_confidence_report(optimizer, config, [_chunk(_trajectory())], theta_hat)

    async def test_noise_dominated_residual_flags_all_verdicts(self) -> None:
        rng = np.random.default_rng(7)
        residual_pos = rng.normal(0.0, 1e-3, size=500)
        residual_vel = rng.normal(0.0, 2e-3, size=500)
        report = await self._report_for_residual(residual_pos, residual_vel)
        self.assertTrue(report.noise_floor.estimated)
        self.assertTrue(report.noise_floor.at_noise_floor)
        self.assertLessEqual(report.noise_floor.cost_to_floor_ratio, NOISE_FLOOR_COST_FACTOR)
        # The verdict itself stays intact (the gate annotates, it does not recompute).
        self.assertEqual(report.entries[0].verdict, CONFIDENCE_WELL_CONSTRAINED)
        self.assertTrue(all(entry.at_noise_floor for entry in report.entries))
        payload = report.to_dict()
        self.assertTrue(payload["noise_floor"]["at_noise_floor"])
        self.assertTrue(payload["entries"][0]["at_noise_floor"])
        summary = "\n".join(report.summary_lines())
        self.assertIn("at noise floor", summary)
        self.assertIn("verdicts unreliable", summary)

    async def test_structured_residual_above_floor_keeps_verdicts(self) -> None:
        rng = np.random.default_rng(7)
        times = np.arange(500) * 0.01
        structure = 0.05 * np.sin(2.0 * math.pi * 0.5 * times)
        residual_pos = structure + rng.normal(0.0, 1e-4, size=500)
        residual_vel = 2.0 * structure + rng.normal(0.0, 1e-4, size=500)
        report = await self._report_for_residual(residual_pos, residual_vel)
        self.assertTrue(report.noise_floor.estimated)
        self.assertFalse(report.noise_floor.at_noise_floor)
        self.assertGreater(report.noise_floor.cost_to_floor_ratio, NOISE_FLOOR_COST_FACTOR)
        self.assertFalse(any(entry.at_noise_floor for entry in report.entries))
        summary = "\n".join(report.summary_lines())
        self.assertNotIn("at noise floor", summary)

    async def test_flag_toggles_across_noise_ladder(self) -> None:
        rng = np.random.default_rng(11)
        times = np.arange(500) * 0.01
        structure = 0.05 * np.sin(2.0 * math.pi * 0.5 * times)
        flags = []
        for structure_scale in (1.0, 0.0):
            residual = structure * structure_scale + rng.normal(0.0, 1e-3, size=500)
            report = await self._report_for_residual(residual, residual.copy())
            flags.append(report.noise_floor.at_noise_floor)
        self.assertEqual(flags, [False, True])

    async def test_zero_residual_does_not_flag(self) -> None:
        report = await self._report_for_residual(np.zeros(500), np.zeros(500))
        self.assertTrue(report.noise_floor.estimated)
        self.assertFalse(report.noise_floor.at_noise_floor)

    async def test_multichunk_confidence_aggregates_every_nominal_residual(self) -> None:
        rng = np.random.default_rng(17)
        samples = 500
        noise = rng.normal(0.0, 1e-3, size=samples)
        times = np.arange(samples) * 0.01
        structured = 0.1 * np.sin(2.0 * math.pi * 0.5 * times) + rng.normal(0.0, 1e-4, size=samples)
        noise_plot = _residual_plot(noise, noise)
        structured_plot = _residual_plot(structured, structured)

        class ChunkPlotOptimizer(_QuadraticOptimizer):
            async def compute_costs_for_theta_batch(
                self, config, theta_batch, on_progress=None, include_plot_data=True
            ):
                costs, residuals, _plot = await super().compute_costs_for_theta_batch(
                    config,
                    theta_batch,
                    on_progress=on_progress,
                    include_plot_data=False,
                )
                marker = float(config.trajectory.positions[0, 0])
                plot = (noise_plot if marker == 0.0 else structured_plot) if include_plot_data else None
                return costs, residuals, plot

        first_trajectory = _trajectory(samples)
        second_trajectory = _trajectory(samples)
        second_trajectory.positions[:] = 1.0
        chunks = []
        for index, item in enumerate((first_trajectory, second_trajectory)):
            chunks.append(
                TrajectoryChunk(
                    spec=TelemetryChunkRunSpec(
                        name=f"validation-{index}",
                        role="validation",
                        start=0.0,
                        end=float(item.times[-1]),
                    ),
                    trajectory=item,
                    sample_count=samples,
                    duration_seconds=item.duration,
                )
            )
        optimizer = ChunkPlotOptimizer(0.5, [400.0], [1.0])
        config = _config([0.0], [2.0], 1)
        config.max_rollout_steps = samples

        report = await build_parameter_confidence_report(optimizer, config, chunks, [1.0])

        self.assertTrue(report.noise_floor.estimated)
        self.assertFalse(report.noise_floor.at_noise_floor)
        self.assertEqual(len({row["source"] for row in report.noise_floor.channels}), 2)
        self.assertEqual(len(optimizer.batches), 2)

    async def test_segmented_noise_floor_applies_sample_weights(self) -> None:
        rng = np.random.default_rng(23)
        residual = rng.normal(0.0, 1e-3, size=500)
        plot = _residual_plot(residual, residual)
        sample_weights = np.linspace(0.0, 1.0, residual.shape[0])

        assessment = assess_segmented_noise_floor(
            [(plot, sample_weights, 1.0, "weighted")],
            ResidualWeightConfig(),
            0.5,
        )

        self.assertTrue(assessment.estimated)
        self.assertTrue(all(row["sample_weighted"] for row in assessment.channels))


class ChunkMetricIdentityTests(omni.kit.test.AsyncTestCase):
    async def test_duplicate_chunk_names_keep_distinct_time_windows(self) -> None:
        rows = [
            ChunkValidationMetric(
                name="repeat",
                role="validation",
                excitation="sine",
                start=float(index),
                end=float(index + 1),
                sample_count=10,
                cost=1.0,
                normalized_cost=0.1,
                position_rmse=0.1,
                velocity_rmse=0.1,
            )
            for index in range(2)
        ]

        self.assertEqual(len(_metrics_by_key(rows)), 2)
