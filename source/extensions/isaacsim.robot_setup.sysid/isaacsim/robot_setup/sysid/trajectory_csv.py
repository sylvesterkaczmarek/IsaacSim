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


"""CSV loader for real-world joint telemetry used in system identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .provenance import TrajectoryDatasetMetadata, build_dataset_metadata
from .trajectory_csv_core import (
    TrajectoryCsvError,
    infer_telemetry_layout_from_columns,
    load_csv_with_header,
    timestamp_comparison_tolerance,
    validate_strictly_increasing_times,
)

__all__ = [
    "TrajectoryCsvError",
    "TrajectoryDataset",
    "TrajectoryDatasetMetadata",
    "load_sysid_trajectory_csv",
]


@dataclass
class TrajectoryDataset:
    """Recorded joint trajectory for SysId playback and residual computation.

    Attributes:
        times: Shape (T,) time in seconds, strictly increasing.
        positions: Shape (T, N) measured joint positions.
        velocities: Shape (T, N) measured joint velocities.
        commands: Shape (T, N) commanded joint positions (or efforts per user convention).
        num_joints: Number of joints N.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    commands: np.ndarray
    metadata: TrajectoryDatasetMetadata | None = None
    torques: np.ndarray | None = None
    end_effector_poses: np.ndarray | None = None
    contact_forces: np.ndarray | None = None
    residual_sample_weights: np.ndarray | None = None
    resampling_diagnostics: list[dict] | None = None

    def __post_init__(self) -> None:
        """Normalize times to a one-dimensional float64 array."""
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)

    @property
    def num_joints(self) -> int:
        """Return the number of joints represented by the trajectory."""
        return int(self.positions.shape[1])

    @property
    def duration(self) -> float:
        """Return the trajectory duration in seconds."""
        return float(self.times[-1] - self.times[0])

    def slice_time_window(self, start_time: float, end_time: float) -> TrajectoryDataset:
        """Return a copy restricted to samples with start_time <= t <= end_time.

        Args:
            start_time: Inclusive window start (seconds).
            end_time: Inclusive window end (seconds).

        Returns:
            Sliced trajectory with at least two samples.

        Raises:
            TrajectoryCsvError: If the window is invalid or too short.
        """
        if end_time < start_time:
            raise TrajectoryCsvError(f"end_time ({end_time}) must be >= start_time ({start_time}).")

        # Stay within a couple of float64 representable values at the timestamp
        # magnitude. A relative tolerance would pull neighbouring high-rate
        # samples into epoch-stamped train/validation windows.
        eps = timestamp_comparison_tolerance(start_time, end_time)
        mask = (self.times >= start_time - eps) & (self.times <= end_time + eps)

        if not np.any(mask):
            raise TrajectoryCsvError(
                f"No samples in time window [{start_time}, {end_time}] (data range "
                f"[{self.times[0]}, {self.times[-1]}])."
            )

        times = self.times[mask]

        if times.shape[0] < 2:
            raise TrajectoryCsvError("Time window must contain at least two samples.")

        return TrajectoryDataset(
            times=times.copy(),
            positions=self.positions[mask].copy(),
            velocities=self.velocities[mask].copy(),
            commands=self.commands[mask].copy(),
            metadata=self.metadata,
            torques=self.torques[mask].copy() if self.torques is not None else None,
            end_effector_poses=(self.end_effector_poses[mask].copy() if self.end_effector_poses is not None else None),
            contact_forces=(self.contact_forces[mask].copy() if self.contact_forces is not None else None),
            residual_sample_weights=(
                self.residual_sample_weights[mask].copy() if self.residual_sample_weights is not None else None
            ),
            resampling_diagnostics=list(self.resampling_diagnostics or []),
        )


def load_sysid_trajectory_csv(filepath: str) -> TrajectoryDataset:
    """Load a comma-separated telemetry file for system identification.

    Supported layouts (after ``time``):

    - Full: ``q1..qN, dq1..dqN, cmd1..cmdN`` (1 + 3*N columns)
    - Positions + velocities: ``q1..qN, dq1..dqN`` (1 + 2*N; commands use positions)
    - Positions only: ``q1..qN`` (1 + N; velocities derived, commands use positions)

    Column count and optional header row (``q1``, ``dq1``, ``cmd1``, ...) determine N.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Parsed :class:`TrajectoryDataset`. ``num_joints`` is inferred from column count.

    Raises:
        TrajectoryCsvError: If the file cannot be read or column counts are wrong.
    """
    data, header_names = load_csv_with_header(filepath)
    num_joints, layout = infer_telemetry_layout_from_columns(data, header_names)

    times = data[:, 0].astype(np.float64)
    validate_strictly_increasing_times(times)

    n = num_joints
    positions = data[:, 1 : 1 + n].astype(np.float64)

    if layout == "full":
        velocities = data[:, 1 + n : 1 + 2 * n].astype(np.float64)
        commands = data[:, 1 + 2 * n : 1 + 3 * n].astype(np.float64)
    elif layout == "pos_vel":
        velocities = data[:, 1 + n : 1 + 2 * n].astype(np.float64)
        commands = positions.copy()
    else:
        commands = positions.copy()
        velocities = np.zeros_like(positions)
        if times.shape[0] >= 2:
            dt = np.diff(times)
            velocities[1:] = np.diff(positions, axis=0) / dt[:, np.newaxis]
            velocities[0] = velocities[1]

    return TrajectoryDataset(
        times=times,
        positions=positions,
        velocities=velocities,
        commands=commands,
        metadata=build_dataset_metadata(
            source_type="csv",
            source_path=filepath,
            times=times,
            positions=positions,
            velocities=velocities,
            commands=commands,
        ),
    )
