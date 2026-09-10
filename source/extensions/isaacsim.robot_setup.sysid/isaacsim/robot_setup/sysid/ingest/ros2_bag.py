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


"""ROS 2 bag ingestion with topic-to-signal mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..trajectory_csv import TrajectoryDataset
from .config_types import (
    TopicSignalMapping,
    TrajectoryIngestError,
    TrajectoryLoadConfig,
    load_topic_mapping_file,
)
from .mcap_loader import (
    _align_optional_series,
    _align_time_topic_series,
    _attach_time_remap_diagnostic,
    _finalize_optional_channel,
    _mark_synthesized_command_channel,
    _payload_to_vectors,
    _record_decode_failure,
    _require_full_time_topic_coverage,
    _resolved_optional_samples,
    _safe_payload_time,
    _series_payload,
    _topic_alignment_extra,
    _validate_position_times,
    load_mcap_trajectory,
    resolve_topic_alignment_tolerance,
)
from .rosbags_support import create_any_reader
from .trajectory_builder import build_trajectory_dataset

#: Recorder-written bag index whose ``relative_file_paths`` entry proves that several
#: storage files belong to one split recording.
BAG_METADATA_FILE = "metadata.yaml"


def _resolve_topic_mapping(config: TrajectoryLoadConfig) -> TopicSignalMapping:
    if config.topic_mapping is not None:
        return config.topic_mapping
    mapping = load_topic_mapping_file(config.topic_mapping_path)
    if mapping is None:
        raise TrajectoryIngestError(
            "ROS 2 bag ingestion requires a topic mapping file. "
            "Use read_config_resource('topic_map_schema.yaml') for the expected format."
        )
    return mapping


def _find_mcap_storage_dir(source_dir: Path) -> Path | None:
    """Resolve one MCAP storage directory without choosing among recordings.

    Args:
        source_dir: ROS 2 bag root or parent directory to search.

    Returns:
        The unique directory containing MCAP storage files, if one can be identified.
    """
    if any(source_dir.glob("*.mcap")):
        return source_dir

    nested_dirs = {candidate.parent for pattern in ("*/*.mcap", "*/*/*.mcap") for candidate in source_dir.glob(pattern)}
    if len(nested_dirs) == 1:
        return nested_dirs.pop()
    return None


def _find_mcap_in_bag_dir(bag_dir: Path) -> Path | None:
    storage_dir = _find_mcap_storage_dir(bag_dir)
    if storage_dir is None:
        return None
    candidates = sorted(storage_dir.glob("*.mcap"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _declared_bag_storage_paths(bag_root: Path) -> set[Path]:
    """Read the storage files declared by a ROS 2 bag index.

    Args:
        bag_root: Directory that may contain a recorder-written ``metadata.yaml``.

    Returns:
        Resolved storage-file paths declared by the index, or an empty set when the
        directory has no readable index.
    """
    metadata_path = bag_root / BAG_METADATA_FILE
    if not metadata_path.is_file():
        return set()
    try:
        import yaml  # type: ignore
    except ImportError:
        return set()
    try:
        payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return set()
    information = payload.get("rosbag2_bagfile_information") if isinstance(payload, dict) else None
    relative_paths = information.get("relative_file_paths") if isinstance(information, dict) else None
    if not isinstance(relative_paths, list):
        return set()
    declared: set[Path] = set()
    for entry in relative_paths:
        if not isinstance(entry, str):
            continue
        try:
            declared.add((bag_root / entry).resolve())
        except OSError:
            continue
    return declared


def _require_declared_split_bag(storage_dir: Path, parts: list[Path]) -> None:
    """Merge several MCAP files only when a bag index declares them as one recording.

    Reading a directory of unrelated recordings as a single bag concatenates telemetry
    from different runs into one trajectory, which is indistinguishable from a genuine
    split recording once the messages are interleaved by timestamp.

    Args:
        storage_dir: Directory holding the candidate MCAP storage files.
        parts: Candidate MCAP files discovered in that directory.

    Raises:
        TrajectoryIngestError: If no bag index declares every candidate file.
    """
    declared: set[Path] = set()
    for bag_root in {storage_dir, storage_dir.parent}:
        declared |= _declared_bag_storage_paths(bag_root)
    if declared and all(part.resolve() in declared for part in parts):
        return
    choices = ", ".join(str(part) for part in parts)
    raise TrajectoryIngestError(
        f"Multiple MCAP files were found in '{storage_dir}' but no {BAG_METADATA_FILE} declares them as one "
        "split recording, so merging them could concatenate unrelated runs. Pass an explicit MCAP file path "
        f"instead: {choices}"
    )


def _any_reader(paths: list[Path]) -> Any:
    return create_any_reader(paths)


def _delegated_mcap_config(
    mcap_path: str,
    config: TrajectoryLoadConfig,
    mapping: TopicSignalMapping,
) -> TrajectoryLoadConfig:
    """Build the MCAP load config used when a bag resolves to one recording.

    Args:
        mcap_path: Resolved MCAP recording path.
        config: Original ROS 2 bag load request.
        mapping: Topic mapping already resolved for the bag.

    Returns:
        MCAP load configuration carrying the caller's mapping and alignment settings.
    """
    mcap_config = TrajectoryLoadConfig.mcap(mcap_path, topic_mapping_path=config.topic_mapping_path)
    mcap_config.topic_mapping = mapping
    mcap_config.alignment_tolerance_seconds = config.alignment_tolerance_seconds
    return mcap_config


def resolve_ros2_bag_recording(source_path: str | Path) -> Path:
    """Resolve a ROS 2 bag path to the recording the loader will actually read.

    Preflight calls this to report an unreadable or ambiguous recording before a run
    starts, rather than letting ingestion fail once the optimizer is already building.

    Args:
        source_path: ROS 2 bag directory, run directory, or ``.mcap`` recording.

    Returns:
        The single MCAP file to load, or the directory that is read as one bag.

    Raises:
        TrajectoryIngestError: If the path is missing, is a non-MCAP file, or holds
            several MCAP recordings that no bag index declares as one split bag.
    """
    source = Path(source_path).expanduser()
    if not source.exists():
        raise TrajectoryIngestError(f"ROS 2 bag path not found: {source}")
    if source.is_file() and source.suffix.lower() == ".mcap":
        return source

    bag_dir = source if source.is_dir() else source.parent
    mcap_file = _find_mcap_in_bag_dir(bag_dir)
    if mcap_file is not None:
        return mcap_file
    mcap_storage_dir = _find_mcap_storage_dir(bag_dir)
    if mcap_storage_dir is not None:
        bag_dir = mcap_storage_dir

    if not bag_dir.is_dir():
        raise TrajectoryIngestError(f"Expected ROS 2 bag directory, got file: {bag_dir}")

    mcap_parts = sorted(bag_dir.glob("*.mcap"))
    if len(mcap_parts) > 1:
        _require_declared_split_bag(bag_dir, mcap_parts)
    return bag_dir


def load_ros2_bag_trajectory(config: TrajectoryLoadConfig) -> TrajectoryDataset:
    """Load telemetry from a ROS 2 bag directory or ``.mcap`` recording.

    Args:
        config: Bag path and topic mapping used to extract trajectory signals.

    Returns:
        Validated trajectory loaded from the ROS 2 recording.
    """
    source = Path(config.source_path).expanduser()
    if not source.exists():
        raise TrajectoryIngestError(f"ROS 2 bag path not found: {source}")

    mapping = _resolve_topic_mapping(config)

    recording = resolve_ros2_bag_recording(source)
    if recording.is_file():
        return load_mcap_trajectory(_delegated_mcap_config(str(recording), config, mapping))
    bag_dir = recording

    pos_topic = mapping.position_topic
    if not pos_topic:
        raise TrajectoryIngestError("topic mapping must define position_topic.")

    vel_topic = mapping.velocity_topic or pos_topic
    cmd_topic = mapping.command_topic or pos_topic
    torque_topic = mapping.torque_topic or pos_topic
    ee_topic = mapping.end_effector_pose_topic
    contact_topic = mapping.contact_force_topic
    channel_decode_failures: dict[str, int] = {}

    with _any_reader([bag_dir]) as reader:
        connections: dict[str, list[Any]] = {}
        for connection in reader.connections:
            connections.setdefault(connection.topic, []).append(connection)
        if pos_topic not in connections:
            raise TrajectoryIngestError(f"position_topic '{pos_topic}' not found in bag. Topics: {list(connections)}")

        pos_connections = connections[pos_topic]
        time_samples: list[tuple[float, float]] = []
        if mapping.time_topic and mapping.time_topic in connections and mapping.time_topic != pos_topic:
            time_connections = connections[mapping.time_topic]
            for message_connection, timestamp, rawdata in reader.messages(connections=time_connections):
                msg = reader.deserialize(rawdata, message_connection.msgtype)
                log_time = float(timestamp) * 1e-9
                sample_time = _safe_payload_time(msg, mapping.time_field, log_time)
                time_samples.append((log_time, sample_time))

        def read_series(
            channel: str, topic: str | None, fields: list[str], *, joint_order: bool, first_match: bool = False
        ) -> list[tuple[float, float, list[float]]]:
            """Read optional topic samples from the ROS 2 bag.

            Args:
                channel: Signal name used to record per-channel decode failures.
                topic: Optional topic containing the requested signal.
                fields: Candidate payload fields from which to extract the signal.
                joint_order: Whether to reorder values using the configured joint names.
                first_match: Whether to use only the first matching field.

            Returns:
                Timestamped signal samples, or an empty list when the topic is unavailable.
            """
            if topic is None or topic not in connections or topic == pos_topic:
                return []
            out: list[tuple[float, float, list[float]]] = []
            for message_connection, timestamp, rawdata in reader.messages(connections=connections[topic]):
                msg = reader.deserialize(rawdata, message_connection.msgtype)
                log_time = float(timestamp) * 1e-9
                sample_time = _safe_payload_time(msg, mapping.time_field, log_time)
                _record_decode_failure(
                    channel_decode_failures,
                    channel,
                    _series_payload(
                        out,
                        log_time,
                        sample_time,
                        msg,
                        fields,
                        mapping.joint_names if joint_order else None,
                        first_match=first_match,
                    ),
                )
            return out

        velocity_samples = read_series("velocity", vel_topic, mapping.velocity_fields, joint_order=True)
        command_samples = read_series("command", cmd_topic, mapping.command_fields, joint_order=True)
        torque_samples = read_series("torque", torque_topic, mapping.torque_fields, joint_order=True, first_match=True)
        ee_samples = read_series("end_effector_pose", ee_topic, mapping.end_effector_pose_fields, joint_order=False)
        contact_samples = read_series("contact_force", contact_topic, mapping.contact_force_fields, joint_order=False)

        position_log_times: list[float] = []
        position_fallback_times: list[float] = []
        positions: list[list[float]] = []
        dropped_positions = 0
        first_drop_error: str | None = None

        for message_connection, timestamp, rawdata in reader.messages(connections=pos_connections):
            msg = reader.deserialize(rawdata, message_connection.msgtype)
            try:
                pos = _payload_to_vectors(msg, mapping.position_fields, mapping.joint_names)
            except TrajectoryIngestError as exc:
                # Dropping position messages silently would load a decimated
                # trajectory that still passes the >=2-sample check. The primary
                # channel must parse cleanly, so track failures and surface them.
                dropped_positions += 1
                if first_drop_error is None:
                    first_drop_error = str(exc)
                continue
            log_time = float(timestamp) * 1e-9
            fallback_time = _safe_payload_time(msg, mapping.time_field, log_time)
            position_log_times.append(log_time)
            position_fallback_times.append(fallback_time)
            positions.append(pos)
            if vel_topic == pos_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "velocity",
                    _series_payload(
                        velocity_samples,
                        log_time,
                        fallback_time,
                        msg,
                        mapping.velocity_fields,
                        mapping.joint_names,
                    ),
                )
            if cmd_topic == pos_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "command",
                    _series_payload(
                        command_samples,
                        log_time,
                        fallback_time,
                        msg,
                        mapping.command_fields,
                        mapping.joint_names,
                    ),
                )
            if torque_topic == pos_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "torque",
                    _series_payload(
                        torque_samples,
                        log_time,
                        fallback_time,
                        msg,
                        mapping.torque_fields,
                        mapping.joint_names,
                        first_match=True,
                    ),
                )
            if ee_topic == pos_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "end_effector_pose",
                    _series_payload(ee_samples, log_time, fallback_time, msg, mapping.end_effector_pose_fields),
                )
            if contact_topic == pos_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "contact_force",
                    _series_payload(contact_samples, log_time, fallback_time, msg, mapping.contact_force_fields),
                )

    if dropped_positions:
        raise TrajectoryIngestError(
            f"ROS 2 bag: {dropped_positions} position message(s) could not be parsed with the configured "
            f"field mapping (first error: {first_drop_error}). Fix the position field mapping rather than "
            "loading a partial trajectory."
        )

    if len(position_fallback_times) < 2:
        raise TrajectoryIngestError("ROS 2 bag did not yield enough position samples.")

    times_arr = np.asarray(position_fallback_times, dtype=np.float64)
    position_time_diagnostic = None
    tolerance_override = config.alignment_tolerance_seconds
    time_topic_for_optional = mapping.time_topic if mapping.time_topic and mapping.time_topic != pos_topic else None
    if time_topic_for_optional:
        position_log_times_arr = np.asarray(position_log_times, dtype=np.float64)
        times_arr, position_time_diagnostic = _align_time_topic_series(
            position_log_times_arr,
            times_arr,
            time_samples,
            time_topic_for_optional,
            tolerance_seconds=resolve_topic_alignment_tolerance(position_log_times_arr, tolerance_override),
        )
    _require_full_time_topic_coverage(position_time_diagnostic, "ROS 2 bag")
    _validate_position_times(times_arr, "ROS 2 bag")
    pos_arr = np.asarray(positions, dtype=np.float64)
    tolerance_seconds = resolve_topic_alignment_tolerance(times_arr, tolerance_override)
    diagnostics: dict[str, dict] = {}
    resolved_velocity_samples, velocity_time_diagnostic = _resolved_optional_samples(
        velocity_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_command_samples, command_time_diagnostic = _resolved_optional_samples(
        command_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_torque_samples, torque_time_diagnostic = _resolved_optional_samples(
        torque_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_ee_samples, ee_time_diagnostic = _resolved_optional_samples(
        ee_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_contact_samples, contact_time_diagnostic = _resolved_optional_samples(
        contact_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional,
        tolerance_seconds=tolerance_seconds,
    )
    vel_arr, diagnostics["velocity"] = _align_optional_series(
        times_arr, resolved_velocity_samples, "velocity", width=pos_arr.shape[1], tolerance_seconds=tolerance_seconds
    )
    diagnostics["velocity"] = _attach_time_remap_diagnostic(diagnostics["velocity"], velocity_time_diagnostic)
    vel_arr = _finalize_optional_channel(
        vel_arr,
        diagnostics["velocity"],
        required_topic=mapping.velocity_topic,
        decode_failure_count=channel_decode_failures.get("velocity", 0),
        source="ROS 2 bag",
    )
    cmd_arr, diagnostics["command"] = _align_optional_series(
        times_arr, resolved_command_samples, "command", width=pos_arr.shape[1], tolerance_seconds=tolerance_seconds
    )
    diagnostics["command"] = _attach_time_remap_diagnostic(diagnostics["command"], command_time_diagnostic)
    cmd_arr = _finalize_optional_channel(
        cmd_arr,
        diagnostics["command"],
        required_topic=mapping.command_topic,
        decode_failure_count=channel_decode_failures.get("command", 0),
        source="ROS 2 bag",
    )
    _mark_synthesized_command_channel(diagnostics["command"], command_topic=mapping.command_topic)
    torque_arr, diagnostics["torque"] = _align_optional_series(
        times_arr, resolved_torque_samples, "torque", width=pos_arr.shape[1], tolerance_seconds=tolerance_seconds
    )
    diagnostics["torque"] = _attach_time_remap_diagnostic(diagnostics["torque"], torque_time_diagnostic)
    torque_arr = _finalize_optional_channel(
        torque_arr,
        diagnostics["torque"],
        required_topic=mapping.torque_topic,
        decode_failure_count=channel_decode_failures.get("torque", 0),
        source="ROS 2 bag",
    )
    ee_arr, diagnostics["end_effector_pose"] = _align_optional_series(
        times_arr, resolved_ee_samples, "end_effector_pose", tolerance_seconds=tolerance_seconds
    )
    diagnostics["end_effector_pose"] = _attach_time_remap_diagnostic(
        diagnostics["end_effector_pose"], ee_time_diagnostic
    )
    ee_arr = _finalize_optional_channel(
        ee_arr,
        diagnostics["end_effector_pose"],
        required_topic=None,
        decode_failure_count=channel_decode_failures.get("end_effector_pose", 0),
        source="ROS 2 bag",
    )
    contact_arr, diagnostics["contact_force"] = _align_optional_series(
        times_arr, resolved_contact_samples, "contact_force", tolerance_seconds=tolerance_seconds
    )
    diagnostics["contact_force"] = _attach_time_remap_diagnostic(diagnostics["contact_force"], contact_time_diagnostic)
    contact_arr = _finalize_optional_channel(
        contact_arr,
        diagnostics["contact_force"],
        required_topic=None,
        decode_failure_count=channel_decode_failures.get("contact_force", 0),
        source="ROS 2 bag",
    )

    extra = _topic_alignment_extra(mapping, diagnostics, tolerance_seconds=tolerance_seconds)
    if position_time_diagnostic is not None:
        extra["topic_alignment"]["time_topic"] = position_time_diagnostic

    return build_trajectory_dataset(
        times=times_arr,
        positions=pos_arr,
        velocities=vel_arr,
        commands=cmd_arr,
        torques=torque_arr,
        end_effector_poses=ee_arr,
        contact_forces=contact_arr,
        source_type="ros2_bag",
        source_path=str(bag_dir),
        topic_mapping_path=config.topic_mapping_path,
        velocities_required=mapping.velocity_topic is not None,
        commands_required=mapping.command_topic is not None,
        torques_required=mapping.torque_topic is not None,
        extra=extra,
    )
