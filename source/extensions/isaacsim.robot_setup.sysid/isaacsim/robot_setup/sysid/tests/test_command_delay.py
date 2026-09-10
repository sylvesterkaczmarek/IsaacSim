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

# ruff: noqa: D102

"""Unit tests for common-mode command-lag estimation and load-time correction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.ingest import load_trajectory
from isaacsim.robot_setup.sysid.ingest.trajectory_builder import apply_command_delay
from isaacsim.robot_setup.sysid.newton_sysid_bridge import (
    _commands_for_actuator_owners,
    _delayed_command_rows,
    actuator_command_delay_seconds_vector,
)
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.run_spec import SysIdRunSpec, TelemetryRunSpec
from isaacsim.robot_setup.sysid.schema_validation import validate_sysid_run_spec_payload
from isaacsim.robot_setup.sysid.telemetry_quality import (
    build_telemetry_quality_report,
    estimate_command_lag,
)
from isaacsim.robot_setup.sysid.trajectory_csv import TrajectoryDataset

_RATE_HZ = 30.0
_DT = 1.0 / _RATE_HZ
_NUM_JOINTS = 7
#: Common-mode lag (samples) matching the transport delay measured on real Franka
#: telemetry (~16 samples at 30 Hz), plus per-joint actuator offsets with median 0.
_COMMON_LAG_SAMPLES = 16.0
_JOINT_OFFSETS = (-0.6, -0.3, -0.1, 0.0, 0.2, 0.4, 0.7)


def _multisine(t: np.ndarray, joint: int) -> tuple[np.ndarray, np.ndarray]:
    """Analytic excitation position and velocity for one joint.

    Args:
        t: Value supplied for ``t``.
        joint: Value supplied for ``joint``.

    Returns:
        Result produced by the operation.
    """
    position = np.zeros_like(t)
    velocity = np.zeros_like(t)
    for k, (freq, amp) in enumerate(((0.31, 0.6), (0.83, 0.3), (1.57, 0.15))):
        w = 2.0 * np.pi * freq * (1.0 + 0.05 * joint)
        phase = 0.9 * joint + 0.7 * k
        position += amp * np.sin(w * t + phase)
        velocity += amp * w * np.cos(w * t + phase)
    return position, velocity


def _delayed_trajectory(lag_samples_by_joint: list[float], duration_seconds: float = 40.0) -> TrajectoryDataset:
    """Commands are the multisine; measurements respond ``lag`` samples later.

    Args:
        lag_samples_by_joint: Value supplied for ``lag_samples_by_joint``.
        duration_seconds: Value supplied for ``duration_seconds``.

    Returns:
        Result produced by the operation.
    """
    times = np.arange(int(round(duration_seconds * _RATE_HZ))) * _DT
    commands = np.zeros((times.shape[0], len(lag_samples_by_joint)))
    positions = np.zeros_like(commands)
    velocities = np.zeros_like(commands)
    for joint, lag in enumerate(lag_samples_by_joint):
        commands[:, joint], _ = _multisine(times, joint)
        positions[:, joint], velocities[:, joint] = _multisine(times - lag * _DT, joint)
    return TrajectoryDataset(times=times, positions=positions, velocities=velocities, commands=commands)


class CommandDelayTests(omni.kit.test.AsyncTestCase):
    """Tests for command-lag estimation, the spec key, and load-time correction."""

    async def test_lag_report_recovers_common_mode_and_residuals(self) -> None:
        """Recover the injected common-mode shift (median) and per-joint residual spread."""
        lags = [_COMMON_LAG_SAMPLES + offset for offset in _JOINT_OFFSETS]
        report = estimate_command_lag(_delayed_trajectory(lags))

        self.assertIsNotNone(report)
        self.assertIsNotNone(report.common_mode_lag_samples)
        self.assertLess(abs(report.common_mode_lag_samples - _COMMON_LAG_SAMPLES), 0.2)
        self.assertLess(abs(report.common_mode_lag_seconds - _COMMON_LAG_SAMPLES * _DT), 0.2 * _DT)
        self.assertAlmostEqual(report.suggested_command_alignment_seconds, report.common_mode_lag_seconds)
        self.assertAlmostEqual(report.suggested_command_alignment_seconds, report.common_mode_lag_seconds)
        for estimate, offset in zip(report.joints, _JOINT_OFFSETS):
            self.assertIsNotNone(estimate.lag_samples)
            self.assertGreater(estimate.correlation_peak, 0.95)
            self.assertLess(abs(estimate.residual_samples - offset), 0.25)
        self.assertGreater(report.residual_spread_samples, 0.4)
        self.assertLess(report.residual_spread_samples, 1.0)

    async def test_quality_report_warns_with_suggested_correction(self) -> None:
        """Common-mode lag above 1.5 samples surfaces a warn-first issue with the fix."""
        lags = [_COMMON_LAG_SAMPLES + offset for offset in _JOINT_OFFSETS]
        report = build_telemetry_quality_report(_delayed_trajectory(lags))

        issues = {issue.code: issue for issue in report.issues}
        self.assertIn("command_lag_common_mode", issues)
        message = issues["command_lag_common_mode"].message
        self.assertIn("command_alignment_seconds", message)
        self.assertIn(f"{report.command_lag.suggested_command_alignment_seconds:.4f}", message)
        self.assertIn("actuator-model identification", message)
        payload = report.to_dict()
        self.assertEqual(len(payload["command_lag"]["joints"]), _NUM_JOINTS)

    async def test_aligned_trajectory_reports_no_common_mode_warning(self) -> None:
        """Near-zero-lag telemetry estimates ~0 common mode and emits no lag issue."""
        # Exactly 0.0 would make commands == positions and trip the
        # command_matches_position guard, which skips lag estimation entirely.
        report = build_telemetry_quality_report(_delayed_trajectory([0.05] * _NUM_JOINTS))

        self.assertNotIn("command_lag_common_mode", {issue.code for issue in report.issues})
        self.assertIsNotNone(report.command_lag)
        self.assertLess(abs(report.command_lag.common_mode_lag_samples), 0.5)

    async def test_apply_command_delay_realigns_commands(self) -> None:
        """Shifting by the injected integer lag makes commands match the measured response."""
        lag_samples = 16.0
        trajectory = _delayed_trajectory([lag_samples] * _NUM_JOINTS)
        misalignment = float(np.sqrt(np.mean((trajectory.commands - trajectory.positions) ** 2)))

        corrected = apply_command_delay(trajectory, lag_samples * _DT)

        interior = slice(int(lag_samples), None)
        np.testing.assert_allclose(corrected.commands[interior], corrected.positions[interior], rtol=0.0, atol=1e-9)
        self.assertGreater(misalignment, 0.1)

    async def test_alignment_then_actuator_delay_compose_once_each(self) -> None:
        times = np.arange(10, dtype=np.float64)
        trajectory = TrajectoryDataset(
            times=times,
            positions=np.zeros((10, 1), dtype=np.float64),
            velocities=np.zeros((10, 1), dtype=np.float64),
            commands=times[:, None].copy(),
        )

        aligned = apply_command_delay(trajectory, 2.0)
        actuated = _delayed_command_rows(
            torch.as_tensor(aligned.commands, dtype=torch.float32),
            np.asarray([[3.0]], dtype=np.float32),
            1.0,
        )

        np.testing.assert_allclose(
            actuated[0, :, 0].numpy(),
            np.maximum(times - 5.0, 0.0),
        )

    async def test_bridge_applies_delay_once_for_each_actuator_owner(self) -> None:
        commands = torch.arange(6, dtype=torch.float32).view(-1, 1).repeat(1, 2)

        staged = _commands_for_actuator_owners(
            commands,
            np.asarray([[2.0, 2.0]], dtype=np.float32),
            actuator_dt=1.0,
            explicit_columns=(1,),
        )

        np.testing.assert_allclose(staged[0, :, 0].numpy(), np.asarray([0, 0, 0, 1, 2, 3], dtype=np.float32))
        np.testing.assert_allclose(staged[0, :, 1].numpy(), commands[:, 1].numpy())

    async def test_nonfinite_selected_actuator_delay_is_rejected(self) -> None:
        entries = [
            SysIdParameterEntry(
                param_type=SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS,
                dof_index=0,
            )
        ]

        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be finite"):
                actuator_command_delay_seconds_vector(2, entries, np.asarray([value], dtype=np.float32))

    async def test_delay_staging_rejects_invalid_timestep_and_nonfinite_rows(self) -> None:
        commands = torch.zeros((3, 1), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "actuator_dt"):
            _delayed_command_rows(commands, np.zeros((1, 1), dtype=np.float32), float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            _delayed_command_rows(commands, np.asarray([[np.nan]], dtype=np.float32), 0.1)

    async def test_load_trajectory_applies_spec_command_alignment(self) -> None:
        """Telemetry alignment shifts commands at load and is recorded separately."""
        lag_samples = 16.0
        trajectory = _delayed_trajectory([lag_samples] * _NUM_JOINTS)
        header = (
            ["time"]
            + [f"q{i + 1}" for i in range(_NUM_JOINTS)]
            + [f"dq{i + 1}" for i in range(_NUM_JOINTS)]
            + [f"cmd{i + 1}" for i in range(_NUM_JOINTS)]
        )
        rows = np.hstack([trajectory.times[:, None], trajectory.positions, trajectory.velocities, trajectory.commands])
        with tempfile.TemporaryDirectory(prefix="sysid_command_delay_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text(
                ",".join(header) + "\n" + "\n".join(",".join(f"{v:.12g}" for v in row) for row in rows) + "\n",
                encoding="utf-8",
            )
            spec = TelemetryRunSpec(
                source_type="csv",
                source_path=str(csv_path),
                command_alignment_seconds=lag_samples * _DT,
            )

            loaded = load_trajectory(spec.to_load_config())

        interior = slice(int(lag_samples), None)
        np.testing.assert_allclose(loaded.commands[interior], loaded.positions[interior], rtol=0.0, atol=1e-6)
        self.assertIsNotNone(loaded.metadata)
        self.assertAlmostEqual(loaded.metadata.extra["command_alignment_seconds"], lag_samples * _DT)

    async def test_run_spec_round_trips_canonical_command_alignment(self) -> None:
        spec = SysIdRunSpec.from_dict({"telemetry": {"command_alignment_seconds": 0.125}})

        self.assertAlmostEqual(spec.telemetry.resolved_command_alignment_seconds(), 0.125)
        self.assertAlmostEqual(spec.telemetry.to_load_config().command_alignment_seconds, 0.125)
        self.assertAlmostEqual(spec.to_dict()["telemetry"]["command_alignment_seconds"], 0.125)

    async def test_schema_validation_knows_command_alignment_key(self) -> None:
        """The new spec key is allowlisted; non-numeric values are rejected."""
        canonical = validate_sysid_run_spec_payload(
            {"schema_version": 2, "telemetry": {"command_alignment_seconds": 0.53}}
        )
        self.assertFalse([issue for issue in canonical if "command_alignment_seconds" in issue.path])

        bad = validate_sysid_run_spec_payload({"schema_version": 2, "telemetry": {"command_alignment_seconds": "half"}})
        errors = [issue for issue in bad if issue.path == "$.telemetry.command_alignment_seconds"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].severity, "error")
