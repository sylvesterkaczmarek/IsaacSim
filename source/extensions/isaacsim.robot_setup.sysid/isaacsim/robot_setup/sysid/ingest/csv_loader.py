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


"""CSV ingestion with configurable column mapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory_csv import TrajectoryDataset, load_sysid_trajectory_csv
from ..trajectory_csv_core import (
    load_csv_with_header,
)
from .config_types import (
    CsvColumnMapping,
    TrajectoryIngestError,
    TrajectoryLoadConfig,
    load_mapping_file,
)
from .trajectory_builder import build_trajectory_dataset


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _resolve_column_index(key: str | int, header_names: list[str] | None, num_columns: int) -> int:
    if isinstance(key, int):
        if key < 0 or key >= num_columns:
            raise TrajectoryIngestError(f"Column index out of range: {key}")
        return key
    if header_names is None:
        raise TrajectoryIngestError(f"Named column '{key}' requires a CSV header row.")
    normalized = _normalize_header(str(key))
    for idx, name in enumerate(header_names):
        if name == normalized:
            return idx
    raise TrajectoryIngestError(f"Column '{key}' not found in CSV header: {header_names}")


def _extract_columns(
    data: np.ndarray,
    header_names: list[str] | None,
    keys: list[str | int],
) -> np.ndarray:
    indices = [_resolve_column_index(key, header_names, data.shape[1]) for key in keys]
    return data[:, indices].astype(np.float64)


def load_csv_trajectory(config: TrajectoryLoadConfig) -> TrajectoryDataset:
    """Load CSV telemetry, optionally using a column mapping file.

    Args:
        config: CSV source and optional in-memory or file-backed column mapping.

    Returns:
        Validated trajectory loaded from the configured CSV file.
    """
    mapping = config.column_mapping
    if mapping is None and config.column_mapping_path:
        mapping = CsvColumnMapping.from_dict(load_mapping_file(config.column_mapping_path))

    if mapping is None or not mapping.position_columns:
        dataset = load_sysid_trajectory_csv(config.source_path)
        if dataset.metadata is None:
            from ..provenance import build_dataset_metadata

            dataset.metadata = build_dataset_metadata(
                source_type="csv",
                source_path=config.source_path,
                times=dataset.times,
                positions=dataset.positions,
                velocities=dataset.velocities,
                commands=dataset.commands,
                column_mapping_path=config.column_mapping_path,
            )
        return dataset

    data, header_names = load_csv_with_header(config.source_path)

    times = _extract_columns(data, header_names, [mapping.time_column]).reshape(-1)
    positions = _extract_columns(data, header_names, mapping.position_columns)
    velocities = _extract_columns(data, header_names, mapping.velocity_columns) if mapping.velocity_columns else None
    commands = _extract_columns(data, header_names, mapping.command_columns) if mapping.command_columns else None
    torques = _extract_columns(data, header_names, mapping.torque_columns) if mapping.torque_columns else None
    end_effector_poses = (
        _extract_columns(data, header_names, mapping.end_effector_pose_columns)
        if mapping.end_effector_pose_columns
        else None
    )
    contact_forces = (
        _extract_columns(data, header_names, mapping.contact_force_columns) if mapping.contact_force_columns else None
    )
    return build_trajectory_dataset(
        times=times,
        positions=positions,
        velocities=velocities,
        commands=commands,
        torques=torques,
        end_effector_poses=end_effector_poses,
        contact_forces=contact_forces,
        source_type="csv",
        source_path=config.source_path,
        column_mapping_path=config.column_mapping_path,
        extra={"column_mapping": _mapping_to_dict(mapping), "torque_semantics": mapping.torque_semantics},
    )


def _mapping_to_dict(mapping: CsvColumnMapping) -> dict[str, Any]:
    return {
        "time_column": mapping.time_column,
        "position_columns": mapping.position_columns,
        "velocity_columns": mapping.velocity_columns,
        "command_columns": mapping.command_columns,
        "torque_columns": mapping.torque_columns,
        "torque_semantics": mapping.torque_semantics,
        "end_effector_pose_columns": mapping.end_effector_pose_columns,
        "contact_force_columns": mapping.contact_force_columns,
        "joint_names": mapping.joint_names,
    }
