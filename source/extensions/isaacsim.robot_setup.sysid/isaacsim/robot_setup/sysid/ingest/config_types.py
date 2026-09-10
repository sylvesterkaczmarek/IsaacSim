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


"""Configuration types for multi-source trajectory ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TrajectoryIngestError(ValueError):
    """Raised when trajectory ingestion fails."""


#: Effort/torque channel semantics accepted by the mapping configs: ``link_side``
#: (transmitted joint torque ``tau_J``, the ``sensor_msgs/JointState.effort``
#: convention) or ``external`` (estimated external torque ``tau_ext``, e.g. Franka
#: ``tau_ext_hat_filtered``). The analytical presolve picks its regression equation
#: from this value.
VALID_TORQUE_SEMANTICS = ("link_side", "external")


def _validated_torque_semantics(value: Any) -> str | None:
    if value is None:
        return None
    semantics = str(value)
    if semantics not in VALID_TORQUE_SEMANTICS:
        raise TrajectoryIngestError(
            f"torque_semantics must be one of {list(VALID_TORQUE_SEMANTICS)}, got '{semantics}'."
        )
    return semantics


class TrajectorySourceType(str, Enum):
    """Supported trajectory source backends."""

    CSV = "csv"
    ROS2_BAG = "ros2_bag"
    MCAP = "mcap"
    LEROBOT = "lerobot"


@dataclass
class CsvColumnMapping:
    """Maps CSV column names or indices to trajectory signals."""

    time_column: str | int = "time"
    position_columns: list[str | int] = field(default_factory=list)
    velocity_columns: list[str | int] | None = None
    command_columns: list[str | int] | None = None
    torque_columns: list[str | int] | None = None
    torque_semantics: str | None = None
    end_effector_pose_columns: list[str | int] | None = None
    contact_force_columns: list[str | int] | None = None
    #: Telemetry joint names in column order; used to align joint prim paths.
    joint_names: list[str] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CsvColumnMapping:
        """Create a CSV column mapping from a JSON/YAML payload.

        Args:
            payload: Serialized input payload.

        Returns:
            Column mapping with defaults for optional telemetry channels.
        """
        return cls(
            time_column=payload.get("time_column", "time"),
            position_columns=list(payload.get("position_columns", [])),
            velocity_columns=payload.get("velocity_columns"),
            command_columns=payload.get("command_columns"),
            torque_columns=payload.get("torque_columns"),
            torque_semantics=_validated_torque_semantics(payload.get("torque_semantics")),
            end_effector_pose_columns=payload.get("end_effector_pose_columns"),
            contact_force_columns=payload.get("contact_force_columns"),
            joint_names=payload.get("joint_names"),
        )


@dataclass
class TopicSignalMapping:
    """Maps ROS 2 / MCAP topics to trajectory signals (see the bundled topic-map resource)."""

    time_topic: str | None = None
    time_field: str = "timestamp"
    position_topic: str | None = None
    position_fields: list[str] = field(default_factory=lambda: ["position"])
    velocity_topic: str | None = None
    velocity_fields: list[str] = field(default_factory=lambda: ["velocity"])
    command_topic: str | None = None
    command_fields: list[str] = field(default_factory=lambda: ["position"])
    torque_topic: str | None = None
    torque_fields: list[str] = field(default_factory=lambda: ["effort", "torque"])
    torque_semantics: str | None = None
    end_effector_pose_topic: str | None = None
    end_effector_pose_fields: list[str] = field(default_factory=lambda: ["pose", "position", "orientation"])
    contact_force_topic: str | None = None
    contact_force_fields: list[str] = field(default_factory=lambda: ["force", "wrench.force", "contact_force"])
    joint_names: list[str] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TopicSignalMapping:
        """Create a topic signal mapping from a JSON/YAML payload.

        Args:
            payload: Serialized input payload.

        Returns:
            Topic mapping with default field aliases for common ROS messages.
        """
        return cls(
            time_topic=payload.get("time_topic"),
            time_field=payload.get("time_field", "timestamp"),
            position_topic=payload.get("position_topic"),
            position_fields=list(payload.get("position_fields", ["position"])),
            velocity_topic=payload.get("velocity_topic"),
            velocity_fields=list(payload.get("velocity_fields", ["velocity"])),
            command_topic=payload.get("command_topic"),
            command_fields=list(payload.get("command_fields", ["position"])),
            torque_topic=payload.get("torque_topic"),
            torque_fields=list(payload.get("torque_fields", ["effort", "torque"])),
            torque_semantics=_validated_torque_semantics(payload.get("torque_semantics")),
            end_effector_pose_topic=payload.get("end_effector_pose_topic"),
            end_effector_pose_fields=list(payload.get("end_effector_pose_fields", ["pose", "position", "orientation"])),
            contact_force_topic=payload.get("contact_force_topic"),
            contact_force_fields=list(payload.get("contact_force_fields", ["force", "wrench.force", "contact_force"])),
            joint_names=payload.get("joint_names"),
        )


@dataclass
class TrajectoryLoadConfig:
    """Unified load request for :func:`load_trajectory`."""

    source_type: TrajectorySourceType
    source_path: str
    column_mapping_path: str | None = None
    topic_mapping_path: str | None = None
    column_mapping: CsvColumnMapping | None = None
    topic_mapping: TopicSignalMapping | None = None
    lerobot_episode_index: int = 0
    #: Common-mode command-to-response alignment removed at load:
    #: ``command'(t) = command(t - delay)``, uniform across joints. Corrects a
    #: data artifact (e.g. trajectory-generator smoothing/lookahead); per-joint
    #: actuator latencies are NOT corrected here.
    command_alignment_seconds: float = 0.0
    #: Overrides the nearest-neighbor window used to align optional MCAP/ROS 2 topics to
    #: the position timebase. ``None`` derives it from the median position sample
    #: interval, capped at 20 ms.
    alignment_tolerance_seconds: float | None = None

    def resolved_command_alignment_seconds(self) -> float:
        """Return the configured command-to-response alignment.

        Returns:
            Alignment duration in seconds.
        """
        return float(self.command_alignment_seconds)

    @classmethod
    def csv(cls, path: str, *, column_mapping_path: str | None = None) -> TrajectoryLoadConfig:
        """Create a load config for a CSV telemetry file.

        Args:
            path: CSV telemetry path.
            column_mapping_path: Optional JSON or YAML column-map path.

        Returns:
            CSV trajectory load configuration.
        """
        return cls(
            source_type=TrajectorySourceType.CSV,
            source_path=path,
            column_mapping_path=column_mapping_path,
        )

    @classmethod
    def ros2_bag(cls, path: str, *, topic_mapping_path: str | None = None) -> TrajectoryLoadConfig:
        """Create a load config for a ROS 2 bag directory.

        Args:
            path: ROS 2 bag directory or recording path.
            topic_mapping_path: Optional JSON or YAML topic-map path.

        Returns:
            ROS 2 bag trajectory load configuration.
        """
        return cls(
            source_type=TrajectorySourceType.ROS2_BAG,
            source_path=path,
            topic_mapping_path=topic_mapping_path,
        )

    @classmethod
    def mcap(cls, path: str, *, topic_mapping_path: str | None = None) -> TrajectoryLoadConfig:
        """Create a load config for an MCAP recording.

        Args:
            path: MCAP recording path.
            topic_mapping_path: Optional JSON or YAML topic-map path.

        Returns:
            MCAP trajectory load configuration.
        """
        return cls(
            source_type=TrajectorySourceType.MCAP,
            source_path=path,
            topic_mapping_path=topic_mapping_path,
        )

    @classmethod
    def lerobot(cls, path: str, *, episode_index: int = 0) -> TrajectoryLoadConfig:
        """Create a load config for a local LeRobot-format dataset.

        Args:
            path: Local LeRobot-format dataset root.
            episode_index: Non-negative episode index to load.

        Returns:
            Local LeRobot trajectory load configuration.
        """
        return cls(
            source_type=TrajectorySourceType.LEROBOT,
            source_path=path,
            lerobot_episode_index=episode_index,
        )


def load_mapping_file(path: str | None) -> dict[str, Any]:
    """Load a JSON or YAML mapping file.

    Args:
        path: Optional JSON or YAML mapping path.

    Returns:
        Parsed mapping object, or an empty mapping when no path is supplied.
    """
    if not path:
        return {}
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise TrajectoryIngestError(f"Mapping file not found: {file_path}")
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrajectoryIngestError(f"Could not read mapping file '{file_path}': {exc}") from exc
    suffix = file_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise TrajectoryIngestError("PyYAML is required to read .yaml mapping files.") from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TrajectoryIngestError(f"Could not parse YAML mapping file '{file_path}': {exc}") from exc
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TrajectoryIngestError(f"Could not parse JSON mapping file '{file_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise TrajectoryIngestError(f"Mapping file must contain a JSON/YAML object: {file_path}")
    return payload


def load_topic_mapping_file(path: str | None) -> TopicSignalMapping | None:
    """Load and strictly validate a ROS 2 / MCAP topic mapping file.

    Args:
        path: JSON or YAML topic mapping path.

    Returns:
        Validated topic mapping, or ``None`` when no path was provided.
    """
    if not path:
        return None
    payload = load_mapping_file(path)

    # Import lazily to avoid a module cycle: schema_validation imports the
    # source-type enum from this module.
    from ..schema_validation import validate_topic_mapping_payload

    issues = validate_topic_mapping_payload(payload)
    if issues:
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise TrajectoryIngestError(f"Invalid topic mapping file '{Path(path).expanduser()}': {details}")
    return TopicSignalMapping.from_dict(payload)
