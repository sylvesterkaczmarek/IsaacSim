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


"""Shared helpers for building TrajectoryDataset instances from ingested signals."""

from __future__ import annotations

import numpy as np

from ..provenance import build_dataset_metadata
from ..trajectory_csv import TrajectoryDataset
from ..trajectory_csv_core import validate_strictly_increasing_times
from .config_types import TrajectoryIngestError


def _derive_velocities(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    velocities = np.zeros_like(positions)
    if times.shape[0] >= 2:
        dt = np.diff(times)
        dt = np.maximum(dt, 1e-12)
        velocities[1:] = np.diff(positions, axis=0) / dt[:, np.newaxis]
        velocities[0] = velocities[1]
    return velocities


def _require_finite(name: str, values: np.ndarray) -> None:
    """Reject NaN and infinity at the shared ingestion boundary.

    Args:
        name: Signal name to include in an ingestion error.
        values: Numeric signal array to validate.
    """
    if not np.all(np.isfinite(values)):
        raise TrajectoryIngestError(f"{name} contains NaN or Inf.")


def apply_command_delay(trajectory: TrajectoryDataset, delay_seconds: float) -> TrajectoryDataset:
    """Time-shift the command series against the measured signals, uniformly across joints.

    Removes a COMMON-MODE command-to-response transport/generation delay (e.g.
    trajectory-generator smoothing/lookahead): ``command'(t) = command(t - delay)``
    with edge-hold beyond the recording, so positive values delay commands to align
    them with the measured response. Measured channels and timestamps are untouched,
    so every bridge consuming the loaded trajectory benefits. Per-joint actuator
    latencies are deliberately not correctable here — they are a robot property for
    actuator-model identification.

    Args:
        trajectory: Loaded trajectory whose command channel will be shifted.
        delay_seconds: Signed time shift in seconds; positive values delay commands.

    Returns:
        The input trajectory with its command channel and provenance updated.
    """
    delay = float(delay_seconds)
    if delay == 0.0:
        return trajectory
    if not np.isfinite(delay):
        raise TrajectoryIngestError(f"command_alignment_seconds must be a finite number, got {delay_seconds!r}.")
    times = np.asarray(trajectory.times, dtype=np.float64).reshape(-1)
    commands = np.asarray(trajectory.commands, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[0] != times.shape[0]:
        raise TrajectoryIngestError("commands must have shape (T, N) matching times to apply a command delay.")
    query_times = times - delay
    shifted = np.empty_like(commands)
    for joint_index in range(commands.shape[1]):
        shifted[:, joint_index] = np.interp(
            query_times,
            times,
            commands[:, joint_index],
            left=commands[0, joint_index],
            right=commands[-1, joint_index],
        )
    trajectory.commands = shifted
    if trajectory.metadata is not None:
        trajectory.metadata.extra["command_alignment_seconds"] = delay
    return trajectory


def build_trajectory_dataset(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray | None,
    commands: np.ndarray | None,
    torques: np.ndarray | None = None,
    end_effector_poses: np.ndarray | None = None,
    contact_forces: np.ndarray | None = None,
    source_type: str,
    source_path: str,
    column_mapping_path: str | None = None,
    topic_mapping_path: str | None = None,
    velocities_required: bool = False,
    commands_required: bool = False,
    torques_required: bool = False,
    extra: dict | None = None,
) -> TrajectoryDataset:
    """Validate arrays, fill missing signals, attach provenance metadata.

    Args:
        times: One timestamp per telemetry sample.
        positions: Joint positions with shape ``(T, N)``.
        velocities: Optional joint velocities; derived from positions when omitted.
        commands: Optional joint commands; copied from positions when omitted.
        torques: Optional joint torques with shape ``(T, N)``.
        end_effector_poses: Optional pose signal with one row per sample.
        contact_forces: Optional contact-force signal with one row per sample.
        source_type: Ingestion backend identifier recorded in provenance.
        source_path: Source recording path recorded in provenance.
        column_mapping_path: Optional CSV column-map path recorded in provenance.
        topic_mapping_path: Optional topic-map path recorded in provenance.
        velocities_required: Whether an omitted velocity channel is an ingestion error.
        commands_required: Whether an omitted command channel is an ingestion error.
        torques_required: Whether an omitted torque channel is an ingestion error.
        extra: Additional source-specific provenance fields.

    Returns:
        Validated trajectory data with complete provenance metadata.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)
    if times.shape[0] < 2:
        raise TrajectoryIngestError("Trajectory must contain at least two time samples.")
    if positions.ndim != 2 or positions.shape[0] != times.shape[0]:
        raise TrajectoryIngestError("positions must have shape (T, N) matching times.")
    _require_finite("times", times)
    _require_finite("positions", positions)
    validate_strictly_increasing_times(times)

    if velocities is None:
        if velocities_required:
            raise TrajectoryIngestError(
                "The configured velocity channel could not be decoded and aligned. Differentiating positions "
                "instead would bias damping and friction estimates. Remove the velocity mapping only for a "
                "source without measured velocity."
            )
        velocities = _derive_velocities(times, positions)
    else:
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.shape != positions.shape:
            raise TrajectoryIngestError("velocities shape must match positions.")
    _require_finite("velocities", velocities)

    if commands is None:
        if commands_required:
            raise TrajectoryIngestError(
                "The configured command channel could not be decoded and aligned. "
                "Remove the command mapping only for a true positions-only source."
            )
        commands = positions.copy()
    else:
        commands = np.asarray(commands, dtype=np.float64)
        if commands.shape != positions.shape:
            raise TrajectoryIngestError("commands shape must match positions.")
    _require_finite("commands", commands)

    if torques is None:
        if torques_required:
            raise TrajectoryIngestError(
                "The configured torque channel could not be decoded and aligned. "
                "Remove the torque mapping only for a source without measured torque."
            )
    else:
        torques = np.asarray(torques, dtype=np.float64)
        if torques.shape != positions.shape:
            raise TrajectoryIngestError("torques shape must match positions.")
        _require_finite("torques", torques)

    if end_effector_poses is not None:
        end_effector_poses = np.asarray(end_effector_poses, dtype=np.float64)
        if end_effector_poses.ndim != 2 or end_effector_poses.shape[0] != times.shape[0]:
            raise TrajectoryIngestError("end_effector_poses must have shape (T, D) matching times.")
        _require_finite("end_effector_poses", end_effector_poses)
        if end_effector_poses.shape[1] == 7:
            quaternion_norms = np.linalg.norm(end_effector_poses[:, 3:7], axis=1)
            if np.any(~np.isfinite(quaternion_norms)) or np.any(quaternion_norms <= np.finfo(np.float64).eps):
                raise TrajectoryIngestError(
                    "Standard end_effector_poses use xyz+xyzw and every quaternion must have nonzero finite norm."
                )

    if contact_forces is not None:
        contact_forces = np.asarray(contact_forces, dtype=np.float64)
        if contact_forces.ndim != 2 or contact_forces.shape[0] != times.shape[0]:
            raise TrajectoryIngestError("contact_forces must have shape (T, D) matching times.")
        _require_finite("contact_forces", contact_forces)

    metadata = build_dataset_metadata(
        source_type=source_type,
        source_path=source_path,
        times=times,
        positions=positions,
        velocities=velocities,
        commands=commands,
        torques=torques,
        end_effector_poses=end_effector_poses,
        contact_forces=contact_forces,
        column_mapping_path=column_mapping_path,
        topic_mapping_path=topic_mapping_path,
        extra=extra,
    )
    return TrajectoryDataset(
        times=times,
        positions=positions,
        velocities=velocities,
        commands=commands,
        metadata=metadata,
        torques=torques,
        end_effector_poses=end_effector_poses,
        contact_forces=contact_forces,
    )
