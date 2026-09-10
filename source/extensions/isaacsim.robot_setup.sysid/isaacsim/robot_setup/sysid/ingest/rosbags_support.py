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

"""Lazy rosbags integration shared by ROS 2 bag and MCAP ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_types import TrajectoryIngestError

_JOINT_TRAJECTORY_CONTROLLER_STATE_MSG = """\
std_msgs/Header header
string[] joint_names
trajectory_msgs/JointTrajectoryPoint reference
trajectory_msgs/JointTrajectoryPoint feedback
trajectory_msgs/JointTrajectoryPoint error
trajectory_msgs/JointTrajectoryPoint output
trajectory_msgs/JointTrajectoryPoint desired
trajectory_msgs/JointTrajectoryPoint actual
string[] multi_dof_joint_names
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_reference
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_feedback
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_error
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_output
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_desired
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_actual
"""


def _default_ros2_typestore() -> Any | None:
    try:
        from rosbags.typesys import (  # type: ignore
            Stores,
            TypesysError,
            get_types_from_msg,
            get_typestore,
        )
    except ImportError:
        return None

    for store_name in ("ROS2_HUMBLE", "LATEST"):
        store = getattr(Stores, store_name, None)
        if store is None:
            continue
        try:
            typestore = get_typestore(store)
            if "control_msgs/msg/JointTrajectoryControllerState" not in typestore.fielddefs:
                typestore.register(
                    get_types_from_msg(
                        _JOINT_TRAJECTORY_CONTROLLER_STATE_MSG,
                        "control_msgs/msg/JointTrajectoryControllerState",
                    )
                )
            return typestore
        except (AttributeError, KeyError, TypeError, TypesysError, ValueError):
            continue
    return None


def create_any_reader(paths: list[Path]) -> Any:
    """Create a lazy rosbags reader for SQLite or standalone MCAP storage.

    Args:
        paths: ROS 2 bag directories to open.

    Returns:
        A configured ``rosbags.highlevel.AnyReader`` instance.
    """
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ImportError as exc:
        raise TrajectoryIngestError(
            "ROS 2 bag and binary ROS 2 MCAP ingestion require the optional 'rosbags' package. "
            "Make it available in the active Python environment."
        ) from exc

    typestore = _default_ros2_typestore()
    if typestore is not None:
        try:
            return AnyReader(paths, default_typestore=typestore)
        except TypeError:
            pass
    return AnyReader(paths)
