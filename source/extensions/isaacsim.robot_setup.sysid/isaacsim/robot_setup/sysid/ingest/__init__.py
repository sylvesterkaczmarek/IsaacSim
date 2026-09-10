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


"""Unified trajectory ingestion."""

from __future__ import annotations

from ..trajectory_csv import TrajectoryDataset
from .config_types import (
    TrajectoryIngestError,
    TrajectoryLoadConfig,
    TrajectorySourceType,
)
from .csv_loader import load_csv_trajectory
from .lerobot_loader import load_lerobot_trajectory
from .mcap_loader import load_mcap_trajectory
from .ros2_bag import load_ros2_bag_trajectory
from .trajectory_builder import apply_command_delay


def load_trajectory(config: TrajectoryLoadConfig) -> TrajectoryDataset:
    """Load a :class:`TrajectoryDataset` from CSV, ROS 2 bag, MCAP, or LeRobot sources.

    Args:
        config: Source type, path, mappings, and command-alignment settings.

    Returns:
        Validated trajectory loaded by the selected backend.
    """
    if config.source_type == TrajectorySourceType.CSV:
        trajectory = load_csv_trajectory(config)
    elif config.source_type == TrajectorySourceType.ROS2_BAG:
        trajectory = load_ros2_bag_trajectory(config)
    elif config.source_type == TrajectorySourceType.MCAP:
        trajectory = load_mcap_trajectory(config)
    elif config.source_type == TrajectorySourceType.LEROBOT:
        trajectory = load_lerobot_trajectory(config)
    else:
        raise TrajectoryIngestError(f"Unsupported source type: {config.source_type}")
    return apply_command_delay(trajectory, config.resolved_command_alignment_seconds())
