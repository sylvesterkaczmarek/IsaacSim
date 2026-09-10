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
"""CSV parsing helpers for SysID trajectory files."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

import numpy as np

TelemetryLayout = Literal["full", "pos_vel", "pos_only"]

__all__ = [
    "TelemetryLayout",
    "TrajectoryCsvError",
    "infer_telemetry_layout_from_columns",
    "infer_telemetry_layout_from_header",
    "load_csv_with_header",
    "load_numeric_csv_table",
    "peek_csv_header_names",
    "timestamp_comparison_tolerance",
    "validate_strictly_increasing_times",
]


class TrajectoryCsvError(ValueError):
    """Raised when a trajectory CSV file is missing, malformed, or inconsistent."""


def timestamp_comparison_tolerance(*values: float) -> float:
    """Return a float64-scale tolerance without widening epoch-time windows.

    Args:
        *values: Values used to derive the comparison tolerance.

    Returns:
        Absolute tolerance scaled to float64 resolution near the inputs.
    """
    magnitude = max(
        (abs(float(value)) for value in values if math.isfinite(float(value))),
        default=1.0,
    )
    return max(1e-12, 2.0 * math.ulp(max(1.0, magnitude)))


def _scan_csv_header(path: Path) -> tuple[list[str] | None, int]:
    if not path.is_file():
        raise TrajectoryCsvError(f"Trajectory file not found: {path}")

    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                first_cell = stripped.split(",")[0].strip()
                try:
                    float(first_cell)
                    return None, 0
                except ValueError:
                    header_names = [_normalize_column_name(cell) for cell in line.split(",")]
                    for data_line in stream:
                        data_text = data_line.strip()
                        if data_text and not data_text.startswith("#"):
                            return header_names, line_number + 1
                    raise TrajectoryCsvError("Trajectory file has a header but no numeric rows.")
    except OSError as exc:
        raise TrajectoryCsvError(f"Failed to read trajectory CSV: {exc}") from exc

    raise TrajectoryCsvError("Trajectory file has no data rows.")


def _load_csv_data(path: Path, skiprows: int) -> np.ndarray:
    try:
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=skiprows)
    except OSError as exc:
        raise TrajectoryCsvError(f"Failed to read trajectory CSV: {exc}") from exc
    except Exception as exc:
        raise TrajectoryCsvError(f"Failed to parse trajectory CSV: {exc}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise TrajectoryCsvError("Trajectory file has no data rows.")
    return data


def load_numeric_csv_table(filepath: str) -> np.ndarray:
    """Load a comma-separated numeric table from a trajectory CSV file.

    Args:
        filepath: CSV path, optionally containing comments or a header.

    Returns:
        Two-dimensional numeric table.
    """
    path = Path(filepath).expanduser()
    _header_names, skiprows = _scan_csv_header(path)
    return _load_csv_data(path, skiprows)


def load_csv_with_header(filepath: str) -> tuple[np.ndarray, list[str] | None]:
    """Load a trajectory CSV and return ``(data, normalized_header_names)``.

    Args:
        filepath: CSV path, optionally containing comments or a header.

    Returns:
        Numeric table and normalized header names, when present.
    """
    path = Path(filepath).expanduser()
    header_names, skiprows = _scan_csv_header(path)
    return _load_csv_data(path, skiprows), header_names


def peek_csv_header_names(filepath: str) -> list[str] | None:
    """Return normalized header column names if the first non-comment row is non-numeric.

    Args:
        filepath: CSV path to inspect without loading its numeric table.

    Returns:
        Normalized names, or ``None`` for headerless or unreadable input.
    """
    path = Path(filepath).expanduser()
    try:
        header_names, _skiprows = _scan_csv_header(path)
    except TrajectoryCsvError:
        return None
    return header_names


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _indexed_columns(names: list[str], prefix: str) -> set[int]:
    indices: set[int] = set()
    for name in names:
        match = re.fullmatch(rf"{prefix}(\d+)", name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def infer_telemetry_layout_from_header(
    names: list[str],
) -> tuple[int, TelemetryLayout] | None:
    """Infer joint count and column layout from header names.

    Args:
        names: Raw or normalized CSV header names.

    Returns:
        Joint count and recognized layout, or ``None`` when names do not match.
    """
    if not names:
        return None
    normalized = [_normalize_column_name(name) for name in names]
    q_cols = _indexed_columns(normalized, "q")
    if not q_cols:
        return None
    num_joints = max(q_cols)
    q_names = [f"q{index}" for index in range(1, num_joints + 1)]
    dq_names = [f"dq{index}" for index in range(1, num_joints + 1)]
    cmd_names = [f"cmd{index}" for index in range(1, num_joints + 1)]
    signal_names = normalized[1:]
    if signal_names == q_names + dq_names + cmd_names:
        return num_joints, "full"
    if signal_names == q_names + dq_names:
        return num_joints, "pos_vel"
    if signal_names == q_names:
        return num_joints, "pos_only"
    return None


def infer_telemetry_layout_from_columns(
    data: np.ndarray,
    header_names: list[str] | None = None,
) -> tuple[int, TelemetryLayout]:
    """Infer joint count and CSV layout from column count and optional header.

    Args:
        data: Numeric CSV table with time in the first column.
        header_names: Optional normalized header names.

    Returns:
        Unambiguous joint count and telemetry column layout.
    """
    if header_names is not None:
        from_header = infer_telemetry_layout_from_header(header_names)
        if from_header is None:
            raise TrajectoryCsvError(
                "CSV header must use contiguous, ordered q1..qN columns followed by optional "
                "dq1..dqN and cmd1..cmdN groups."
            )
        num_joints, layout = from_header
        expected = _expected_column_count(num_joints, layout)
        if data.shape[1] != expected:
            raise TrajectoryCsvError(
                f"CSV header describes {expected} columns but numeric rows contain {data.shape[1]} columns."
            )
        return from_header

    remainder = data.shape[1] - 1
    if remainder < 1:
        raise TrajectoryCsvError("At least one joint column is required after time.")

    candidates: list[tuple[int, TelemetryLayout]] = []
    for layout, divisor in (("full", 3), ("pos_vel", 2), ("pos_only", 1)):
        if remainder % divisor != 0:
            continue
        num_joints = remainder // divisor
        if num_joints < 1:
            continue
        candidates.append((num_joints, layout))

    if not candidates:
        raise TrajectoryCsvError(
            f"Unsupported column count {data.shape[1]}. Expected 1+N, 1+2*N, or 1+3*N columns after time."
        )

    if len(candidates) > 1:
        layouts = ", ".join(f"{num_joints} joint(s) as {layout}" for num_joints, layout in candidates)
        raise TrajectoryCsvError(
            "Headerless CSV layout is ambiguous "
            f"for {data.shape[1]} columns ({layouts}). Add a recognized header or an explicit column mapping."
        )
    return candidates[0]


def _expected_column_count(num_joints: int, layout: TelemetryLayout) -> int:
    divisors = {"full": 3, "pos_vel": 2, "pos_only": 1}
    return 1 + divisors[layout] * num_joints


def validate_strictly_increasing_times(times: np.ndarray) -> None:
    """Raise if the time column is not strictly increasing.

    Args:
        times: One-dimensional timestamp vector.
    """
    if times.shape[0] < 2:
        raise TrajectoryCsvError("Trajectory must contain at least two time samples.")
    if not np.all(np.diff(times) > 0):
        raise TrajectoryCsvError("Time column must be strictly increasing.")
