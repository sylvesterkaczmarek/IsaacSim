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

"""Chunk definitions and validation helpers for segmented SysID telemetry."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from .trajectory_csv import TrajectoryDataset
from .trajectory_csv_core import timestamp_comparison_tolerance

TELEMETRY_CHUNK_ROLE_TRAIN = "train"
TELEMETRY_CHUNK_ROLE_VALIDATION = "validation"
TELEMETRY_CHUNK_ROLES = (TELEMETRY_CHUNK_ROLE_TRAIN, TELEMETRY_CHUNK_ROLE_VALIDATION)
DEFAULT_EXCITATION_TAGS = (
    "chirp",
    "PRBS",
    "step",
    "sine",
    "gravity",
    "contact",
    "custom",
)


class TrajectorySegmentError(ValueError):
    """Raised when telemetry chunks cannot be used safely."""


@dataclass
class TelemetryChunkRunSpec:
    """Serializable chunk definition for segmented train/validation telemetry."""

    name: str = ""
    role: str = TELEMETRY_CHUNK_ROLE_TRAIN
    excitation: str = "custom"
    start: float = 0.0
    end: float = 0.0
    weight: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TelemetryChunkRunSpec:
        """Create a chunk spec from a JSON/YAML payload.

        Args:
            payload: Serialized input payload.

        Returns:
            Validated chunk definition with normalized role text.
        """
        try:
            start = float(payload.get("start", 0.0))
            end = float(payload.get("end", 0.0))
            weight = float(payload.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise TrajectorySegmentError("Chunk start, end, and weight must be numeric.") from exc
        if not all(math.isfinite(value) for value in (start, end, weight)):
            raise TrajectorySegmentError("Chunk start, end, and weight must be finite.")
        if end <= start:
            raise TrajectorySegmentError("Chunk start time must be less than end time.")
        if weight <= 0.0:
            raise TrajectorySegmentError("Chunk weight must be greater than zero.")
        return cls(
            name=str(payload.get("name", "")),
            role=str(payload.get("role", TELEMETRY_CHUNK_ROLE_TRAIN)).lower(),
            excitation=str(payload.get("excitation", "custom")),
            start=start,
            end=end,
            weight=weight,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this chunk spec.

        Returns:
            JSON-compatible chunk fields.
        """
        return asdict(self)

    def display_name(self, index: int) -> str:
        """Return a stable display name, falling back to role and index.

        Args:
            index: Zero-based position used by the fallback label.

        Returns:
            Explicit name or a stable role/index fallback.
        """
        name = self.name.strip()
        return name if name else f"{self.role.title()} {index + 1}"


@dataclass
class TrajectoryChunk:
    """A validated telemetry chunk and its sliced trajectory."""

    spec: TelemetryChunkRunSpec
    trajectory: TrajectoryDataset
    sample_count: int
    duration_seconds: float


@dataclass
class ChunkValidationMetric:
    """Validation metrics for one held-out telemetry chunk."""

    name: str
    role: str
    excitation: str
    start: float
    end: float
    sample_count: int
    cost: float
    normalized_cost: float
    position_rmse: float
    velocity_rmse: float
    torque_rmse: float | None = None
    end_effector_pose_rmse: float | None = None
    contact_force_rmse: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this validation metric.

        Returns:
            JSON-compatible validation metric fields.
        """
        return asdict(self)


def make_default_train_chunk(
    trajectory: TrajectoryDataset,
    *,
    start: float | None = None,
    end: float | None = None,
) -> TelemetryChunkRunSpec:
    """Build a single train chunk from a loaded trajectory or time window.

    Args:
        trajectory: Loaded telemetry defining the available time range.
        start: Optional train-window start; defaults to the first sample.
        end: Optional train-window end; defaults to the last sample.

    Returns:
        Single train chunk spanning the selected range.
    """
    t0 = float(trajectory.times[0]) if start is None else float(start)
    t1 = float(trajectory.times[-1]) if end is None else float(end)
    return TelemetryChunkRunSpec(
        name="Train 1",
        role=TELEMETRY_CHUNK_ROLE_TRAIN,
        excitation="custom",
        start=t0,
        end=t1,
        weight=1.0,
    )


DEFAULT_AUTO_TRAIN_FRACTION = 0.8
DEFAULT_AUTO_MIN_CHUNK_SECONDS = 1.0


def build_auto_chunks(
    trajectory: TrajectoryDataset,
    *,
    train_fraction: float = DEFAULT_AUTO_TRAIN_FRACTION,
    min_chunk_seconds: float = DEFAULT_AUTO_MIN_CHUNK_SECONDS,
    metadata_chunks: list[TelemetryChunkRunSpec] | None = None,
) -> list[TelemetryChunkRunSpec]:
    """Build an automatic train/validation split for a loaded trajectory.

    When ``metadata_chunks`` (pre-tagged excitation phases) are provided, whole
    trailing phases are reassigned to validation until the validation share
    reaches ``1 - train_fraction`` of the tagged duration; phases keep their
    excitation tags and the first phase always stays a train chunk. Without
    metadata the trajectory is split at a sample boundary into one train and
    one validation chunk (temporal holdout). Degenerate cases (too few samples
    or too-short segments) fall back to a single train chunk; the telemetry
    quality report already warns about a missing validation holdout.

    Args:
        trajectory: Loaded telemetry to split.
        train_fraction: Target fraction assigned to training.
        min_chunk_seconds: Minimum duration accepted for either split.
        metadata_chunks: Optional pre-tagged excitation phases.

    Returns:
        Train/validation chunk definitions, or one train chunk when unsplittable.
    """
    train_fraction = min(max(float(train_fraction), 0.0), 1.0)
    min_chunk_seconds = max(float(min_chunk_seconds), 0.0)

    if metadata_chunks:
        specs = sorted(metadata_chunks, key=lambda spec: float(spec.start))
        if any(spec.role == TELEMETRY_CHUNK_ROLE_VALIDATION for spec in specs):
            return list(metadata_chunks)
        total = sum(max(float(spec.end) - float(spec.start), 0.0) for spec in specs)
        target_validation = (1.0 - train_fraction) * total
        validation_duration = 0.0
        result = list(specs)
        # Reassign whole trailing phases; never the first one.
        for idx in range(len(result) - 1, 0, -1):
            if validation_duration >= target_validation:
                break
            spec = result[idx]
            result[idx] = replace(spec, role=TELEMETRY_CHUNK_ROLE_VALIDATION)
            validation_duration += max(float(spec.end) - float(spec.start), 0.0)
        return result

    times = np.asarray(trajectory.times, dtype=np.float64).reshape(-1)
    if times.shape[0] < 4:
        return [make_default_train_chunk(trajectory)]
    duration = float(times[-1] - times[0])
    split_index = int(np.searchsorted(times, times[0] + train_fraction * duration))
    split_index = min(max(split_index, 2), times.shape[0] - 2)
    train_end = float(times[split_index - 1])
    validation_start = float(times[split_index])
    if (
        train_end - float(times[0]) < min_chunk_seconds
        or float(times[-1]) - validation_start < min_chunk_seconds
        or times.shape[0] - split_index < 2
    ):
        return [make_default_train_chunk(trajectory)]
    return [
        TelemetryChunkRunSpec(
            name="Auto train",
            role=TELEMETRY_CHUNK_ROLE_TRAIN,
            excitation="custom",
            start=float(times[0]),
            end=train_end,
            weight=1.0,
        ),
        TelemetryChunkRunSpec(
            name="Auto validation",
            role=TELEMETRY_CHUNK_ROLE_VALIDATION,
            excitation="custom",
            start=validation_start,
            end=float(times[-1]),
            weight=1.0,
        ),
    ]


def resolve_chunk_specs(
    trajectory: TrajectoryDataset,
    explicit_chunks: list[TelemetryChunkRunSpec] | None,
    *,
    auto_split: bool = False,
    train_fraction: float = DEFAULT_AUTO_TRAIN_FRACTION,
    min_chunk_seconds: float = DEFAULT_AUTO_MIN_CHUNK_SECONDS,
    window: tuple[float, float] | None = None,
) -> list[TelemetryChunkRunSpec]:
    """Resolve the chunk specs a run should use: explicit > auto-split > default.

    Explicit chunks (manual or companion-imported) always win. Otherwise, with
    ``auto_split``, an automatic train/validation split of the ``window`` (or the
    full trajectory) is built. Note this window-slicing differs from the headless
    ``build_segmented_chunks`` path, which auto-splits the full loaded trajectory;
    the divergence is deliberate — the GUI materializes the resulting chunks into
    the exported run spec, so both paths solve on the same windows. The fallback
    is the single full-window train chunk.

    Args:
        trajectory: Loaded telemetry defining the available range.
        explicit_chunks: User or metadata-provided chunks, which take precedence.
        auto_split: Whether to create a temporal holdout when chunks are absent.
        train_fraction: Target fraction assigned to training.
        min_chunk_seconds: Minimum duration accepted for either split.
        window: Optional range applied before automatic/default chunking.

    Returns:
        Chunk definitions selected according to explicit, automatic, and default precedence.
    """
    if explicit_chunks:
        return list(explicit_chunks)
    t0, t1 = window if window is not None else (float(trajectory.times[0]), float(trajectory.times[-1]))
    if auto_split:
        return build_auto_chunks(
            trajectory.slice_time_window(t0, t1),
            train_fraction=train_fraction,
            min_chunk_seconds=min_chunk_seconds,
        )
    return [make_default_train_chunk(trajectory, start=t0, end=t1)]


def build_trajectory_chunks(
    trajectory: TrajectoryDataset,
    chunks: Iterable[TelemetryChunkRunSpec],
    *,
    require_train: bool = True,
    reject_overlaps: bool = True,
) -> list[TrajectoryChunk]:
    """Validate chunk specs, slice telemetry, and reject train/validation leakage.

    Args:
        trajectory: Loaded telemetry to slice.
        chunks: Chunk definitions to validate and materialize.
        require_train: Require at least one training chunk.
        reject_overlaps: Reject train/validation windows that share telemetry samples.

    Returns:
        Validated chunk definitions paired with sliced trajectories.
    """
    specs = [_coerce_chunk_spec(chunk, idx) for idx, chunk in enumerate(chunks)]
    if not specs:
        if require_train:
            raise TrajectorySegmentError("At least one train chunk is required.")
        return []

    _validate_chunk_windows(trajectory, specs, reject_overlaps=reject_overlaps)

    result: list[TrajectoryChunk] = []
    for idx, spec in enumerate(specs):
        sliced = trajectory.slice_time_window(float(spec.start), float(spec.end))
        result.append(
            TrajectoryChunk(
                spec=TelemetryChunkRunSpec(
                    name=spec.display_name(idx),
                    role=spec.role,
                    excitation=spec.excitation.strip() or "custom",
                    start=float(spec.start),
                    end=float(spec.end),
                    weight=float(spec.weight),
                ),
                trajectory=sliced,
                sample_count=int(sliced.times.shape[0]),
                duration_seconds=float(sliced.times[-1] - sliced.times[0]),
            )
        )

    if require_train and not any(chunk.spec.role == TELEMETRY_CHUNK_ROLE_TRAIN for chunk in result):
        raise TrajectorySegmentError("At least one train chunk is required.")

    return result


def split_train_validation_chunks(
    chunks: Iterable[TrajectoryChunk],
) -> tuple[list[TrajectoryChunk], list[TrajectoryChunk]]:
    """Return train and validation chunk lists in UI/run-spec order.

    Args:
        chunks: Materialized chunks in run-spec order.

    Returns:
        Training chunks followed separately by validation chunks.
    """
    train: list[TrajectoryChunk] = []
    validation: list[TrajectoryChunk] = []
    for chunk in chunks:
        if chunk.spec.role == TELEMETRY_CHUNK_ROLE_VALIDATION:
            validation.append(chunk)
        else:
            train.append(chunk)
    return train, validation


def summarize_validation_metrics(metrics: Iterable[ChunkValidationMetric]) -> str:
    """Build a compact status string grouped by excitation.

    Args:
        metrics: Held-out chunk metrics to group by excitation.

    Returns:
        Semicolon-delimited cost and RMSE summary.
    """
    by_excitation: dict[str, list[ChunkValidationMetric]] = {}
    for metric in metrics:
        by_excitation.setdefault(metric.excitation or "custom", []).append(metric)
    if not by_excitation:
        return "No validation chunks."

    parts: list[str] = []
    for excitation, rows in by_excitation.items():
        pos = sum(row.position_rmse for row in rows) / max(1, len(rows))
        vel = sum(row.velocity_rmse for row in rows) / max(1, len(rows))
        cost = sum(row.normalized_cost for row in rows) / max(1, len(rows))
        parts.append(f"{excitation}: cost={cost:.3e}, qRMSE={pos:.3e}, dqRMSE={vel:.3e}")
    return "; ".join(parts)


def _coerce_chunk_spec(chunk: TelemetryChunkRunSpec, index: int) -> TelemetryChunkRunSpec:
    if isinstance(chunk, TelemetryChunkRunSpec):
        spec = chunk
    elif isinstance(chunk, dict):
        spec = TelemetryChunkRunSpec.from_dict(chunk)
    else:
        raise TrajectorySegmentError(f"Chunk {index + 1} must be a TelemetryChunkRunSpec.")

    role = str(spec.role).lower()
    if role not in TELEMETRY_CHUNK_ROLES:
        raise TrajectorySegmentError(
            f"Chunk '{spec.display_name(index)}' role must be one of {', '.join(TELEMETRY_CHUNK_ROLES)}."
        )
    try:
        start = float(spec.start)
        end = float(spec.end)
        weight = float(spec.weight)
    except (TypeError, ValueError) as exc:
        raise TrajectorySegmentError(
            f"Chunk '{spec.display_name(index)}' start, end, and weight must be numeric."
        ) from exc
    if not all(math.isfinite(value) for value in (start, end, weight)):
        raise TrajectorySegmentError(f"Chunk '{spec.display_name(index)}' start, end, and weight must be finite.")
    if end <= start:
        raise TrajectorySegmentError(f"Chunk '{spec.display_name(index)}' start time must be less than end time.")
    if weight <= 0.0:
        raise TrajectorySegmentError(f"Chunk '{spec.display_name(index)}' weight must be greater than zero.")

    return TelemetryChunkRunSpec(
        name=spec.display_name(index),
        role=role,
        excitation=str(spec.excitation).strip() or "custom",
        start=start,
        end=end,
        weight=weight,
    )


def _validate_chunk_windows(
    trajectory: TrajectoryDataset,
    specs: list[TelemetryChunkRunSpec],
    *,
    reject_overlaps: bool,
) -> None:
    data_start = float(trajectory.times[0])
    data_end = float(trajectory.times[-1])
    # Use float64 representation error, not a relative time tolerance: epoch
    # seconds otherwise turn tiny rounding allowances into millisecond gaps.
    eps = timestamp_comparison_tolerance(data_start, data_end)

    for index, spec in enumerate(specs):
        if spec.start < data_start - eps or spec.end > data_end + eps:
            raise TrajectorySegmentError(
                f"Chunk '{spec.display_name(index)}' is outside telemetry range "
                f"[{data_start:.6g}, {data_end:.6g}] s."
            )
        sliced = trajectory.slice_time_window(spec.start, spec.end)
        if sliced.times.shape[0] < 2:
            raise TrajectorySegmentError(f"Chunk '{spec.display_name(index)}' must include at least 2 samples.")

    if not reject_overlaps:
        return

    times = np.asarray(trajectory.times, dtype=np.float64).reshape(-1)
    for left_index, left in enumerate(specs):
        for right_index in range(left_index + 1, len(specs)):
            right = specs[right_index]
            if left.role == right.role:
                continue
            intersection_start = max(float(left.start), float(right.start))
            intersection_end = min(float(left.end), float(right.end))
            if intersection_start > intersection_end + eps:
                continue
            shared_sample = np.any((times >= intersection_start - eps) & (times <= intersection_end + eps))
            if not shared_sample:
                continue
            raise TrajectorySegmentError(
                f"Train/validation chunks '{left.display_name(left_index)}' and "
                f"'{right.display_name(right_index)}' share telemetry samples."
            )
