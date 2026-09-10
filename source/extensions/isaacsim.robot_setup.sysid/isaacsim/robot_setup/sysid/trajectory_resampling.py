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

"""Trajectory resampling for physics-synchronized SysID rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .trajectory_csv import TrajectoryDataset

SAMPLING_MODE_OFF = "off"
SAMPLING_MODE_AUTO = "auto"
SAMPLING_MODE_ALWAYS = "always"
SAMPLING_MODES = (SAMPLING_MODE_OFF, SAMPLING_MODE_AUTO, SAMPLING_MODE_ALWAYS)
TARGET_DT_MODE_PHYSICS_CAPPED = "physics_dt_capped"
COMMAND_INTERPOLATION_ZERO_ORDER_HOLD = "zero_order_hold"
COMMAND_INTERPOLATION_LINEAR = "linear"
MEASURED_INTERPOLATION_LINEAR = "linear"
IRREGULAR_DT_RATIO_THRESHOLD = 1.5


@dataclass
class SamplingConfig:
    """Configuration for resampling recorded telemetry before rollout."""

    mode: str = SAMPLING_MODE_AUTO
    target_dt_mode: str = TARGET_DT_MODE_PHYSICS_CAPPED
    command_interpolation: str = COMMAND_INTERPOLATION_ZERO_ORDER_HOLD
    measured_interpolation: str = MEASURED_INTERPOLATION_LINEAR
    auto_dt_ratio_threshold: float = 2.0
    auto_jitter_threshold: float = 0.10
    max_resampled_steps: int = 0

    def __post_init__(self) -> None:
        """Reject unsupported sampling policies."""
        if self.mode not in SAMPLING_MODES:
            raise ValueError(f"Sampling mode must be one of {list(SAMPLING_MODES)}, got {self.mode!r}.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SamplingConfig:
        """Create a sampling config from a JSON/YAML payload.

        Args:
            payload: Serialized input payload.

        Returns:
            Sampling policy with defaults for missing fields.
        """
        if not isinstance(payload, dict):
            return cls()
        return cls(
            mode=str(payload.get("mode", SAMPLING_MODE_AUTO)),
            target_dt_mode=str(payload.get("target_dt_mode", TARGET_DT_MODE_PHYSICS_CAPPED)),
            command_interpolation=str(payload.get("command_interpolation", COMMAND_INTERPOLATION_ZERO_ORDER_HOLD)),
            measured_interpolation=str(payload.get("measured_interpolation", MEASURED_INTERPOLATION_LINEAR)),
            auto_dt_ratio_threshold=float(payload.get("auto_dt_ratio_threshold", 2.0)),
            auto_jitter_threshold=float(payload.get("auto_jitter_threshold", 0.10)),
            max_resampled_steps=int(payload.get("max_resampled_steps", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this sampling config.

        Returns:
            JSON-compatible sampling policy fields.
        """
        return asdict(self)


@dataclass
class ResamplingDiagnostic:
    """Diagnostics describing a trajectory resampling decision."""

    enabled: bool
    reason: str
    raw_sample_count: int
    resampled_sample_count: int
    raw_dt_median: float | None
    raw_dt_min: float | None
    raw_dt_max: float | None
    raw_jitter_ratio: float | None
    physics_dt: float
    target_dt: float | None
    capped: bool
    command_interpolation: str
    measured_interpolation: str
    time_weighting: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this resampling diagnostic.

        Returns:
            JSON-compatible resampling decision fields.
        """
        return asdict(self)


@dataclass
class ResampledTrajectory:
    """A trajectory paired with the diagnostic for the resampling decision."""

    trajectory: TrajectoryDataset
    diagnostic: ResamplingDiagnostic


def resample_trajectory_for_rollout(
    trajectory: TrajectoryDataset,
    *,
    physics_dt: float,
    config: SamplingConfig,
    max_rollout_steps: int,
) -> ResampledTrajectory:
    """Return a rollout trajectory on a physics-aligned grid when policy requires it.

    Args:
        trajectory: Recorded telemetry to align for rollout.
        physics_dt: Fixed simulation timestep in seconds.
        config: Resampling and interpolation policy.
        max_rollout_steps: Solver rollout budget used to cap the target grid.

    Returns:
        Copied or resampled trajectory with a diagnostic explaining the decision.
    """
    times = np.asarray(trajectory.times, dtype=np.float64).reshape(-1)
    stats = _time_stats(times)
    reason = _resampling_reason(stats, physics_dt=physics_dt, config=config)
    enabled = bool(reason)
    if config.mode == SAMPLING_MODE_OFF:
        enabled = False
        reason = "disabled"
    elif config.mode == SAMPLING_MODE_ALWAYS and times.shape[0] >= 2:
        enabled = True
        reason = "always"

    if not enabled or times.shape[0] < 2:
        diag = _diagnostic(
            enabled=False,
            reason=reason or "not_needed",
            raw_times=times,
            resampled_count=int(times.shape[0]),
            physics_dt=physics_dt,
            target_dt=None,
            capped=False,
            config=config,
            time_weighting=trajectory.residual_sample_weights is not None,
        )
        copied = _copy_trajectory(trajectory)
        copied.resampling_diagnostics = list(trajectory.resampling_diagnostics or []) + [diag.to_dict()]
        return ResampledTrajectory(trajectory=copied, diagnostic=diag)

    target_times, target_dt, capped = _target_time_grid(
        times,
        physics_dt=physics_dt,
        max_steps=max(2, int(config.max_resampled_steps or max_rollout_steps)),
    )
    resampled = TrajectoryDataset(
        times=target_times,
        positions=_interp_linear(times, trajectory.positions, target_times),
        velocities=_interp_linear(times, trajectory.velocities, target_times),
        commands=_interp_commands(times, trajectory.commands, target_times, config.command_interpolation),
        metadata=trajectory.metadata,
        torques=_interp_optional(times, trajectory.torques, target_times),
        end_effector_poses=_interp_end_effector_poses(times, trajectory.end_effector_poses, target_times),
        contact_forces=_interp_optional(times, trajectory.contact_forces, target_times),
        residual_sample_weights=_time_sample_weights(target_times, reference_dt=stats["median"] or target_dt),
    )
    diag = _diagnostic(
        enabled=True,
        reason=reason,
        raw_times=times,
        resampled_count=int(target_times.shape[0]),
        physics_dt=physics_dt,
        target_dt=target_dt,
        capped=capped,
        config=config,
        time_weighting=True,
    )
    resampled.resampling_diagnostics = list(trajectory.resampling_diagnostics or []) + [diag.to_dict()]
    return ResampledTrajectory(trajectory=resampled, diagnostic=diag)


def combine_sample_weights(
    user_weights: np.ndarray | None,
    trajectory_weights: np.ndarray | None,
    *,
    num_steps: int,
) -> np.ndarray | None:
    """Combine optional user residual weights with trajectory time weights.

    Args:
        user_weights: Optional configured per-sample residual multipliers.
        trajectory_weights: Optional time-integration weights from resampling.
        num_steps: Required rollout length.

    Returns:
        Product of available length-``num_steps`` weights, or ``None``.

    Raises:
        ValueError: If either supplied weight vector does not match ``num_steps``.
    """
    weights = None
    if trajectory_weights is not None:
        weights = np.asarray(trajectory_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != num_steps:
            raise ValueError(f"trajectory_weights must contain exactly {num_steps} values, got {weights.shape[0]}.")
    if user_weights is not None:
        user = np.asarray(user_weights, dtype=np.float64).reshape(-1)
        if user.shape[0] != num_steps:
            raise ValueError(f"user_weights must contain exactly {num_steps} values, got {user.shape[0]}.")
        if weights is None:
            weights = user
        else:
            weights = weights * user
    return weights


def _resampling_reason(stats: dict[str, float | None], *, physics_dt: float, config: SamplingConfig) -> str:
    if config.mode == SAMPLING_MODE_OFF:
        return ""
    median_dt = stats["median"]
    if median_dt is not None and physics_dt > 0.0 and median_dt / physics_dt > config.auto_dt_ratio_threshold:
        return "low_rate"
    jitter = stats["jitter"]
    if jitter is not None and jitter > config.auto_jitter_threshold:
        return "timestamp_jitter"
    dt_min = stats["min"]
    dt_max = stats["max"]
    if dt_min is not None and dt_max is not None and dt_min > 0.0 and dt_max / dt_min > IRREGULAR_DT_RATIO_THRESHOLD:
        return "irregular_dt"
    return ""


def _target_time_grid(times: np.ndarray, *, physics_dt: float, max_steps: int) -> tuple[np.ndarray, float, bool]:
    start = float(times[0])
    end = float(times[-1])
    duration = max(0.0, end - start)
    base_dt = float(physics_dt)
    if not np.isfinite(base_dt) or base_dt <= 0.0:
        raise ValueError(f"physics_dt must be a finite positive number, got {physics_dt!r}.")

    # Each adjacent output row corresponds to exactly one fixed physics step.
    # Do not redistribute samples with linspace: doing so changes the implied
    # timestep while the simulation continues to advance by ``physics_dt``.
    ratio = duration / base_dt
    nearest_steps = round(ratio)
    if np.isclose(ratio, nearest_steps, rtol=1e-12, atol=1e-12):
        available_steps = int(nearest_steps)
    else:
        available_steps = int(np.floor(ratio))
    if available_steps < 1:
        raise ValueError(
            f"Trajectory duration ({duration:.9g} s) must span at least one physics step ({base_dt:.9g} s)."
        )

    desired_count = available_steps + 1
    capped = desired_count > max_steps
    count = max(2, min(max_steps, desired_count))
    target = start + np.arange(count, dtype=np.float64) * base_dt
    target[0] = start
    if count == desired_count and np.isclose(target[-1], end, rtol=1e-12, atol=1e-12):
        target[-1] = end
    target_dt = base_dt
    return target, target_dt, capped


def _interp_linear(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return np.interp(target_times, source_times, arr).reshape(-1, 1)
    cols = [np.interp(target_times, source_times, arr[:, col]) for col in range(arr.shape[1])]
    return np.stack(cols, axis=1).astype(np.float64)


def _interp_optional(
    source_times: np.ndarray, values: np.ndarray | None, target_times: np.ndarray
) -> np.ndarray | None:
    if values is None:
        return None
    return _interp_linear(source_times, values, target_times)


def _interp_end_effector_poses(
    source_times: np.ndarray,
    values: np.ndarray | None,
    target_times: np.ndarray,
) -> np.ndarray | None:
    """Interpolate public ``xyz+xyzw`` poses without leaving quaternion space.

    Args:
        source_times: Original telemetry timestamps.
        values: Optional time-major pose samples.
        target_times: Physics-synchronized timestamps.

    Returns:
        Interpolated poses, or ``None`` when the source channel is absent.
    """
    if values is None:
        return None
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        return _interp_linear(source_times, poses, target_times)
    translation = _interp_linear(source_times, poses[:, :3], target_times)
    quaternion = _interp_quaternions_xyzw(source_times, poses[:, 3:7], target_times)
    return np.concatenate((translation, quaternion), axis=1)


def _interp_quaternions_xyzw(
    source_times: np.ndarray,
    quaternions: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Apply shortest-path SLERP to an ``xyzw`` quaternion sequence.

    Args:
        source_times: Original quaternion timestamps.
        quaternions: Time-major quaternion samples.
        target_times: Requested interpolation timestamps.

    Returns:
        Unit quaternions evaluated at the target timestamps.

    Raises:
        ValueError: If source timestamps are not finite and strictly increasing,
            or if any source quaternion has zero norm.
    """
    source_times = np.asarray(source_times, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(source_times)) or np.any(np.diff(source_times) <= 0.0):
        raise ValueError("Quaternion source timestamps must be finite and strictly increasing.")

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("End-effector pose quaternions must have nonzero norm before resampling.")
    normalized = quaternions / norms
    right = np.searchsorted(source_times, target_times, side="right")
    right = np.clip(right, 1, source_times.shape[0] - 1)
    left = right - 1
    interval = source_times[right] - source_times[left]
    alpha = ((target_times - source_times[left]) / interval).reshape(-1, 1)
    alpha = np.clip(alpha, 0.0, 1.0)

    start = normalized[left]
    end = normalized[right]
    dot = np.sum(start * end, axis=1, keepdims=True)
    end = np.where(dot < 0.0, -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    use_linear = sin_theta <= 1e-8
    safe_sin = np.where(use_linear, 1.0, sin_theta)
    start_weight = np.sin((1.0 - alpha) * theta) / safe_sin
    end_weight = np.sin(alpha * theta) / safe_sin
    spherical = start_weight * start + end_weight * end
    linear = (1.0 - alpha) * start + alpha * end
    result = np.where(use_linear, linear, spherical)
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _interp_commands(
    source_times: np.ndarray,
    commands: np.ndarray,
    target_times: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == COMMAND_INTERPOLATION_LINEAR:
        return _interp_linear(source_times, commands, target_times)
    arr = np.asarray(commands, dtype=np.float64)
    indices = np.searchsorted(source_times, target_times, side="right") - 1
    indices = np.clip(indices, 0, source_times.shape[0] - 1)
    return arr[indices].copy()


def _time_sample_weights(times: np.ndarray, *, reference_dt: float | None) -> np.ndarray:
    if times.shape[0] < 2:
        return np.ones(times.shape[0], dtype=np.float64)
    edges = np.empty(times.shape[0] + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    edges[0] = times[0]
    edges[-1] = times[-1]
    widths = np.maximum(np.diff(edges), 1e-12)
    # Residuals are squared after sample weights are applied, so sqrt(dt/reference_dt)
    # preserves the chunk's approximate time-integral scale when low-rate rows are densified.
    reference = max(float(reference_dt or 0.0), 1e-12)
    return np.sqrt(widths / reference)


def _time_stats(times: np.ndarray) -> dict[str, float | None]:
    if times.shape[0] < 2:
        return {"min": None, "median": None, "max": None, "jitter": None}
    dt = np.diff(times)
    median = float(np.median(dt))
    jitter = float(np.std(dt) / median) if median > 0.0 else None
    return {
        "min": float(np.min(dt)),
        "median": median,
        "max": float(np.max(dt)),
        "jitter": jitter,
    }


def _diagnostic(
    *,
    enabled: bool,
    reason: str,
    raw_times: np.ndarray,
    resampled_count: int,
    physics_dt: float,
    target_dt: float | None,
    capped: bool,
    config: SamplingConfig,
    time_weighting: bool,
) -> ResamplingDiagnostic:
    stats = _time_stats(raw_times)
    return ResamplingDiagnostic(
        enabled=enabled,
        reason=reason,
        raw_sample_count=int(raw_times.shape[0]),
        resampled_sample_count=int(resampled_count),
        raw_dt_median=stats["median"],
        raw_dt_min=stats["min"],
        raw_dt_max=stats["max"],
        raw_jitter_ratio=stats["jitter"],
        physics_dt=float(physics_dt),
        target_dt=target_dt,
        capped=bool(capped),
        command_interpolation=config.command_interpolation,
        measured_interpolation=config.measured_interpolation,
        time_weighting=bool(time_weighting),
    )


def _copy_trajectory(trajectory: TrajectoryDataset) -> TrajectoryDataset:
    return TrajectoryDataset(
        times=np.asarray(trajectory.times, dtype=np.float64).copy(),
        positions=np.asarray(trajectory.positions, dtype=np.float64).copy(),
        velocities=np.asarray(trajectory.velocities, dtype=np.float64).copy(),
        commands=np.asarray(trajectory.commands, dtype=np.float64).copy(),
        metadata=trajectory.metadata,
        torques=trajectory.torques.copy() if trajectory.torques is not None else None,
        end_effector_poses=(
            trajectory.end_effector_poses.copy() if trajectory.end_effector_poses is not None else None
        ),
        contact_forces=(trajectory.contact_forces.copy() if trajectory.contact_forces is not None else None),
        residual_sample_weights=(
            trajectory.residual_sample_weights.copy() if trajectory.residual_sample_weights is not None else None
        ),
        resampling_diagnostics=list(trajectory.resampling_diagnostics or []),
    )
