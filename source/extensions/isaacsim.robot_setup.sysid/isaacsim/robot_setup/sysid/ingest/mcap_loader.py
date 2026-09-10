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


"""MCAP ingestion with topic-to-signal mapping."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..training_logs import find_companion_mcap_path
from ..trajectory_csv import TrajectoryDataset
from .config_types import (
    TopicSignalMapping,
    TrajectoryIngestError,
    TrajectoryLoadConfig,
    load_topic_mapping_file,
)
from .rosbags_support import create_any_reader
from .trajectory_builder import build_trajectory_dataset

#: Upper bound on the nearest-neighbor window used to align optional topics to the
#: position timebase. The effective tolerance is narrowed to half the median position
#: sample interval so a fast recording cannot borrow a sample a whole period away.
DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS = 0.02
MINIMUM_DISTINCT_SOURCE_SAMPLES = 2
#: Alignment status recorded for a command channel that was read from the position
#: topic because the mapping declares no command topic.
COMMAND_STATUS_SYNTHESIZED_FROM_POSITIONS = "synthesized_from_positions"
#: Statuses for which the channel carries usable samples in the loaded trajectory.
_PRESENT_CHANNEL_STATUSES = ("aligned", COMMAND_STATUS_SYNTHESIZED_FROM_POSITIONS)


def resolve_topic_alignment_tolerance(times: np.ndarray, override_seconds: float | None = None) -> float:
    """Scale the nearest-neighbor alignment window to the position sample rate.

    A fixed 20 ms window silently accepts a neighbor a full period away on recordings
    faster than 50 Hz, so the tolerance is capped at half the median position interval
    unless the caller supplies an explicit override.

    Args:
        times: Timestamps that define the alignment target timebase.
        override_seconds: Explicit tolerance that replaces the derived value.

    Returns:
        Positive tolerance in seconds.

    Raises:
        TrajectoryIngestError: If the override is not a positive, finite number.
    """
    if override_seconds is not None:
        tolerance = float(override_seconds)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise TrajectoryIngestError(
                f"alignment_tolerance_seconds must be a positive, finite number, got {override_seconds!r}."
            )
        return tolerance
    intervals = np.diff(np.asarray(times, dtype=np.float64).reshape(-1))
    if intervals.size == 0:
        return DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS
    median_interval = float(np.median(intervals))
    if not math.isfinite(median_interval) or median_interval <= 0.0:
        return DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS
    return min(DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS, 0.5 * median_interval)


def _resolve_topic_mapping(config: TrajectoryLoadConfig) -> TopicSignalMapping:
    if config.topic_mapping is not None:
        return config.topic_mapping
    mapping = load_topic_mapping_file(config.topic_mapping_path)
    if mapping is None:
        raise TrajectoryIngestError("MCAP ingestion requires a topic mapping file; see read_config_resource().")
    return mapping


def _read_field(payload: Any, field: str) -> Any:
    value = payload
    for part in field.split("."):
        if isinstance(value, dict):
            if part not in value:
                return None
            value = value[part]
        else:
            if not hasattr(value, part):
                return None
            value = getattr(value, part)
    return value


def _coerce_numeric_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        if "pose" in value:
            vec = _coerce_numeric_vector(value["pose"])
            if vec:
                return vec
        if "wrench" in value:
            vec = _coerce_numeric_vector(value["wrench"])
            if vec:
                return vec
        if "force" in value:
            vec = _coerce_numeric_vector(value["force"])
            if vec:
                return vec
        parts: list[float] = []
        for key in ("position", "orientation"):
            vec = _coerce_numeric_vector(value.get(key))
            if vec:
                parts.extend(vec)
        if parts:
            return parts
        xyz = [value.get(axis) for axis in ("x", "y", "z")]
        if all(v is not None for v in xyz):
            try:
                out = [float(v) for v in xyz]
                if value.get("w") is not None:
                    out.append(float(value["w"]))
                return out
            except (TypeError, ValueError):
                return None
        return None
    parts = []
    for attr in ("position", "orientation"):
        if hasattr(value, attr):
            vec = _coerce_numeric_vector(getattr(value, attr))
            if vec:
                parts.extend(vec)
    if parts:
        return parts
    xyz = [getattr(value, axis, None) for axis in ("x", "y", "z")]
    if all(v is not None for v in xyz):
        try:
            out = [float(v) for v in xyz]
            w = getattr(value, "w", None)
            if w is not None:
                out.append(float(w))
            return out
        except (TypeError, ValueError):
            return None
    return None


def _apply_joint_order(payload: Any, vector: list[float], joint_names: list[str] | None) -> list[float]:
    if not joint_names:
        return vector
    names = _read_field(payload, "name") or _read_field(payload, "names") or _read_field(payload, "joint_names")
    if names is None:
        raise TrajectoryIngestError("Topic mapping specifies joint_names, but message has no name field.")
    names = [str(name) for name in names]
    if len(names) != len(vector):
        raise TrajectoryIngestError(f"Message name count ({len(names)}) does not match vector width ({len(vector)}).")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise TrajectoryIngestError(f"Message name field contains duplicate joint names: {duplicates}")
    name_to_value = {name: vector[idx] for idx, name in enumerate(names)}
    missing = [name for name in joint_names if name not in name_to_value]
    if missing:
        raise TrajectoryIngestError(f"Message is missing requested joint_names: {missing}")
    return [float(name_to_value[name]) for name in joint_names]


def _payload_to_vectors(
    payload: Any,
    fields: list[str],
    joint_names: list[str] | None = None,
    *,
    first_match: bool = False,
) -> list[float]:
    found: list[list[float]] = []
    for field in fields:
        value = _read_field(payload, field)
        vec = _coerce_numeric_vector(value)
        if vec:
            found.append(vec)
            if (
                first_match
                or len(fields) == 1
                or field in ("pose", "wrench", "wrench.force")
                or (joint_names is not None and len(vec) == len(joint_names))
            ):
                return _apply_joint_order(payload, vec, joint_names)
    if found:
        combined = [v for vec in found for v in vec]
        return _apply_joint_order(payload, combined, joint_names)
    aliases = {
        "position": ("positions", "q"),
        "positions": ("position", "q"),
        "velocity": ("velocities", "dq"),
        "velocities": ("velocity", "dq"),
        "effort": ("efforts", "torque", "torques"),
        "torque": ("effort", "efforts", "torques"),
    }
    for field in fields:
        for key in aliases.get(field, ()):
            vec = _coerce_numeric_vector(_read_field(payload, key))
            if vec:
                return _apply_joint_order(payload, vec, joint_names)
    keys = list(payload) if isinstance(payload, dict) else [name for name in dir(payload) if not name.startswith("_")]
    raise TrajectoryIngestError(f"Could not extract numeric vector from message fields: {keys}")


def _payload_time(payload: Any, field: str, fallback: float) -> float:
    value = _read_field(payload, field)
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    sec = _read_field(value, "sec")
    nsec = _read_field(value, "nanosec")
    if sec is None:
        sec = _read_field(value, "secs")
    if nsec is None:
        nsec = _read_field(value, "nsecs")
    if sec is not None:
        return float(sec) + float(nsec or 0) * 1e-9
    return fallback


def _safe_payload_time(payload: Any, field: str, fallback: float) -> float:
    try:
        return _payload_time(payload, field, fallback)
    except (TypeError, ValueError):
        return fallback


def _try_payload_to_vectors(
    payload: Any,
    fields: list[str],
    joint_names: list[str] | None = None,
    *,
    first_match: bool = False,
) -> list[float] | None:
    try:
        return _payload_to_vectors(payload, fields, joint_names, first_match=first_match)
    except (TrajectoryIngestError, TypeError, ValueError):
        return None


def _align_optional_series(
    target_times: np.ndarray,
    samples: list[tuple[float, list[float]]],
    channel_name: str,
    *,
    width: int | None = None,
    tolerance_seconds: float = DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Align a topic series to position-sample times or omit it with diagnostics.

    Args:
        target_times: Position-sample timestamps that define the output timebase.
        samples: Source timestamps paired with numeric signal vectors.
        channel_name: Signal name included in alignment diagnostics.
        width: Required vector width, or ``None`` to infer it from the first match.
        tolerance_seconds: Maximum nearest-neighbor timestamp separation.

    Returns:
        The aligned signal and diagnostics, or ``None`` plus an omission diagnostic.
    """
    diagnostic: dict[str, Any] = {
        "channel": channel_name,
        "status": "omitted",
        "source_sample_count": len(samples),
        "target_sample_count": int(target_times.shape[0]),
        "tolerance_seconds": float(tolerance_seconds),
    }
    if not samples:
        diagnostic["reason"] = "no_samples"
        return None, diagnostic
    ordered = sorted((float(time), list(values)) for time, values in samples)
    source_times = np.asarray([row[0] for row in ordered], dtype=np.float64)
    source_values = [row[1] for row in ordered]
    signal_width = int(width) if width is not None else None
    rows: list[list[float]] = []
    matched_source_indices: set[int] = set()
    max_delta = 0.0

    for target_index, target_time in enumerate(np.asarray(target_times, dtype=np.float64).reshape(-1)):
        insert_at = int(np.searchsorted(source_times, target_time))
        candidates = [idx for idx in (insert_at - 1, insert_at) if 0 <= idx < len(source_times)]
        if not candidates:
            diagnostic.update({"reason": "no_nearest_sample", "target_index": target_index})
            return None, diagnostic
        best = min(candidates, key=lambda idx: abs(float(source_times[idx] - target_time)))
        delta = abs(float(source_times[best] - target_time))
        max_delta = max(max_delta, delta)
        if delta > tolerance_seconds:
            diagnostic.update(
                {
                    "reason": "outside_tolerance",
                    "target_index": target_index,
                    "nearest_delta_seconds": delta,
                }
            )
            return None, diagnostic
        values = source_values[best]
        if signal_width is None:
            signal_width = len(values)
        if len(values) != signal_width:
            diagnostic.update(
                {
                    "reason": "width_mismatch",
                    "target_index": target_index,
                    "expected_width": signal_width,
                    "actual_width": len(values),
                }
            )
            return None, diagnostic
        rows.append([float(value) for value in values])
        matched_source_indices.add(best)

    diagnostic.update(
        {
            "status": "aligned",
            "reason": "",
            "aligned_sample_count": len(rows),
            "distinct_source_sample_count": len(matched_source_indices),
            "max_delta_seconds": max_delta,
        }
    )
    return np.asarray(rows, dtype=np.float64), diagnostic


def _align_time_topic_series(
    target_log_times: np.ndarray,
    fallback_times: np.ndarray,
    samples: list[tuple[float, float]],
    topic_name: str,
    *,
    tolerance_seconds: float = DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve position sample times from a separate timestamp topic, falling back per sample.

    Args:
        target_log_times: Recording timestamps of the position samples.
        fallback_times: Per-position timestamps used when remapping is unavailable.
        samples: Recording timestamps paired with values from the mapped time topic.
        topic_name: Mapped time-topic name included in diagnostics.
        tolerance_seconds: Maximum nearest-neighbor timestamp separation.

    Returns:
        Resolved position timestamps and time-remapping diagnostics.
    """
    resolved = np.asarray(fallback_times, dtype=np.float64).reshape(-1).copy()
    targets = np.asarray(target_log_times, dtype=np.float64).reshape(-1)
    diagnostic: dict[str, Any] = {
        "topic": topic_name,
        "status": "fallback",
        "source_sample_count": len(samples),
        "target_sample_count": int(targets.shape[0]),
        "tolerance_seconds": float(tolerance_seconds),
        "aligned_count": 0,
        "fallback_count": 0,
        "outside_tolerance_count": 0,
        "missing_count": 0,
    }
    if targets.shape != resolved.shape:
        diagnostic.update({"reason": "target_fallback_shape_mismatch", "fallback_count": int(targets.shape[0])})
        return resolved, diagnostic
    if not samples:
        diagnostic.update(
            {"reason": "no_samples", "fallback_count": int(targets.shape[0]), "missing_count": int(targets.shape[0])}
        )
        return resolved, diagnostic

    ordered = sorted((float(log_time), float(mapped_time)) for log_time, mapped_time in samples)
    source_times = np.asarray([row[0] for row in ordered], dtype=np.float64)
    mapped_times = [row[1] for row in ordered]
    max_delta = 0.0

    for index, target_log_time in enumerate(targets):
        insert_at = int(np.searchsorted(source_times, target_log_time))
        candidates = [idx for idx in (insert_at - 1, insert_at) if 0 <= idx < len(source_times)]
        if not candidates:
            diagnostic["fallback_count"] += 1
            diagnostic["missing_count"] += 1
            continue
        best = min(candidates, key=lambda idx: abs(float(source_times[idx] - target_log_time)))
        delta = abs(float(source_times[best] - target_log_time))
        max_delta = max(max_delta, delta)
        if delta > tolerance_seconds:
            diagnostic["fallback_count"] += 1
            diagnostic["outside_tolerance_count"] += 1
            continue
        resolved[index] = mapped_times[best]
        diagnostic["aligned_count"] += 1

    fallback_count = int(diagnostic["fallback_count"])
    aligned_count = int(diagnostic["aligned_count"])
    diagnostic["max_delta_seconds"] = max_delta
    if fallback_count == 0:
        diagnostic.update({"status": "aligned", "reason": ""})
    elif aligned_count > 0:
        diagnostic.update({"status": "partial_fallback", "reason": "some_samples_outside_tolerance_or_missing"})
    else:
        reason = "outside_tolerance" if diagnostic["outside_tolerance_count"] else "no_nearest_sample"
        diagnostic.update({"status": "fallback", "reason": reason})
    return resolved, diagnostic


def _require_full_time_topic_coverage(diagnostic: dict[str, Any] | None, source: str) -> None:
    """Reject a position timebase spliced from a partially covering time topic.

    A whole-recording fallback keeps every position sample on one clock, but a partial
    remap mixes mapped time-topic stamps with fallback stamps from a different epoch.
    The result can still be strictly increasing, so it must be rejected here.

    Args:
        diagnostic: Time-remapping diagnostic for the position timebase, when a time topic is mapped.
        source: Ingestion backend name included in the error message.

    Raises:
        TrajectoryIngestError: If the mapped time topic covers only part of the recording.
    """
    if diagnostic is None or diagnostic.get("status") != "partial_fallback":
        return
    topic = diagnostic.get("topic")
    aligned_count = int(diagnostic.get("aligned_count", 0))
    target_count = int(diagnostic.get("target_sample_count", 0))
    raise TrajectoryIngestError(
        f"{source} time topic {topic!r} covers only part of the recording: {aligned_count} of "
        f"{target_count} position samples were remapped and the rest kept their original clock. "
        "Mixing both clocks in one timebase would corrupt the trajectory duration; record the time "
        "topic for the whole run or remove the time_topic mapping."
    )


def _finalize_optional_channel(
    values: np.ndarray | None,
    diagnostic: dict[str, Any],
    *,
    required_topic: str | None,
    decode_failure_count: int,
    source: str,
) -> np.ndarray | None:
    """Fail closed on an explicitly mapped channel whose aligned samples cannot be trusted.

    Nearest-neighbor alignment replays whichever source samples survive decoding, so a
    channel that lost messages, or that covers the whole trajectory from a single sample,
    would otherwise be reported as aligned while carrying fabricated data.

    Args:
        values: Aligned channel samples, or ``None`` when alignment already failed.
        diagnostic: Alignment diagnostic for the channel; updated in place with the failure count.
        required_topic: Mapped topic name when the channel is required, or ``None`` to only record counts.
        decode_failure_count: Number of source messages that could not be decoded for this channel.
        source: Ingestion backend name included in the error message.

    Returns:
        The aligned samples, or ``None`` when a degenerate alignment is demoted to omitted.

    Raises:
        TrajectoryIngestError: If a required topic had messages that could not be decoded.
    """
    diagnostic["decode_failure_count"] = int(decode_failure_count)
    if required_topic is None:
        return values
    if decode_failure_count:
        raise TrajectoryIngestError(
            f"{source} {diagnostic.get('channel', 'optional')} topic {required_topic!r} had "
            f"{int(decode_failure_count)} message(s) that could not be decoded with the configured field "
            "mapping. Alignment would replay the surviving samples across the trajectory, so fix the field "
            "mapping or remove the topic from the mapping."
        )
    if values is None:
        return None
    distinct_sources = int(diagnostic.get("distinct_source_sample_count", 0))
    target_count = int(diagnostic.get("target_sample_count", 0))
    if target_count >= MINIMUM_DISTINCT_SOURCE_SAMPLES and distinct_sources < MINIMUM_DISTINCT_SOURCE_SAMPLES:
        diagnostic.update({"status": "omitted", "reason": "degenerate_source_coverage"})
        return None
    return values


def _mark_synthesized_command_channel(diagnostic: dict[str, Any], *, command_topic: str | None) -> None:
    """Distinguish a command channel copied from positions from a genuinely aligned one.

    With no mapped command topic the loader reads the command fields off the position
    messages, so reporting the channel as ``aligned`` would suggest an independently
    recorded setpoint series that was matched to the position timebase.

    Args:
        diagnostic: Command alignment diagnostic, updated in place.
        command_topic: Mapped command topic, or ``None`` when the mapping omits one.
    """
    if command_topic is None and diagnostic.get("status") == "aligned":
        diagnostic.update({"status": COMMAND_STATUS_SYNTHESIZED_FROM_POSITIONS, "reason": "command_topic_not_mapped"})


def _record_decode_failure(failures: dict[str, int], channel: str, decoded: bool) -> None:
    """Count one optional-channel sample that could not be decoded.

    Args:
        failures: Per-channel decode-failure counts, updated in place.
        channel: Signal name whose sample was decoded or dropped.
        decoded: Whether the sample yielded a numeric vector.
    """
    if not decoded:
        failures[channel] = failures.get(channel, 0) + 1


def _series_payload(
    series: list[tuple[float, float, list[float]]],
    log_time: float,
    fallback_time: float,
    payload: Any,
    fields: list[str],
    joint_names: list[str] | None = None,
    *,
    first_match: bool = False,
) -> bool:
    """Append one decoded optional-topic sample to a channel series.

    Args:
        series: Channel samples collected so far, appended in place.
        log_time: Recording timestamp of the source message.
        fallback_time: Payload timestamp used when no time topic is mapped.
        payload: Decoded source message.
        fields: Candidate payload fields from which to extract the signal.
        joint_names: Requested joint order, or ``None`` to keep the payload order.
        first_match: Whether to use only the first matching field.

    Returns:
        Whether the payload yielded a numeric vector.
    """
    values = _try_payload_to_vectors(payload, fields, joint_names, first_match=first_match)
    if values is None:
        return False
    series.append((float(log_time), float(fallback_time), values))
    return True


def _resolved_optional_samples(
    samples: list[tuple[float, float, list[float]]],
    *,
    time_samples: list[tuple[float, float]],
    time_topic: str | None,
    tolerance_seconds: float = DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS,
) -> tuple[list[tuple[float, list[float]]], dict[str, Any] | None]:
    if not samples:
        return [], None
    log_times = np.asarray([row[0] for row in samples], dtype=np.float64)
    fallback_times = np.asarray([row[1] for row in samples], dtype=np.float64)
    diagnostic = None
    if time_topic and time_samples:
        resolved_times, diagnostic = _align_time_topic_series(
            log_times,
            fallback_times,
            time_samples,
            time_topic,
            tolerance_seconds=tolerance_seconds,
        )
    else:
        resolved_times = fallback_times
    return [
        (float(time), values) for time, (_log_time, _fallback_time, values) in zip(resolved_times, samples)
    ], diagnostic


def _attach_time_remap_diagnostic(
    channel_diagnostic: dict[str, Any],
    time_remap_diagnostic: dict[str, Any] | None,
) -> dict[str, Any]:
    if time_remap_diagnostic is not None:
        channel_diagnostic["time_topic"] = time_remap_diagnostic
    return channel_diagnostic


def _mapped_topics(mapping: TopicSignalMapping) -> set[str]:
    topics = {
        mapping.position_topic,
        mapping.time_topic,
        mapping.velocity_topic,
        mapping.command_topic,
        mapping.torque_topic,
        mapping.end_effector_pose_topic,
        mapping.contact_force_topic,
    }
    return {str(topic) for topic in topics if topic}


def _topic_filter_args(topics: set[str]) -> list[str]:
    return sorted(topics)


def _iter_reader_messages(reader: Any, topics: set[str]) -> Iterator[tuple[Any, Any, Any]]:
    topic_args = _topic_filter_args(topics)
    try:
        yield from reader.iter_messages(topics=topic_args)
        return
    except TypeError:
        pass

    for schema, channel, message in reader.iter_messages():
        if channel.topic in topics:
            yield schema, channel, message


def _validate_position_times(times: np.ndarray, source: str) -> None:
    if np.any(np.diff(times) <= 0.0):
        raise TrajectoryIngestError(f"{source} position timestamps must be strictly increasing.")


def _topic_alignment_extra(
    mapping: TopicSignalMapping,
    diagnostics: dict[str, dict[str, Any]],
    time_diagnostic: dict[str, Any] | None = None,
    *,
    tolerance_seconds: float = DEFAULT_TOPIC_ALIGNMENT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    extra = {
        "topic_mapping": mapping.__dict__,
        "torque_semantics": mapping.torque_semantics,
        "topic_alignment": {
            "tolerance_seconds": float(tolerance_seconds),
            "channels": diagnostics,
            "omitted_channels": [
                name
                for name, diagnostic in diagnostics.items()
                if diagnostic.get("status") not in _PRESENT_CHANNEL_STATUSES
            ],
        },
    }
    if time_diagnostic is not None:
        extra["topic_alignment"]["time_topic"] = time_diagnostic
    return extra


def _iter_decoded_ros2_messages(path: Path, topics: set[str]) -> Iterator[tuple[str, float, Any]]:
    with create_any_reader([path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            yield connection.topic, float(timestamp) * 1e-9, reader.deserialize(rawdata, connection.msgtype)


def _sniff_mcap_encoding(path: Path) -> str:
    """Read only the first message to detect encoding without a full file pass.

    Returns a lowercase encoding string such as ``"json"``, ``"cdr"``, or ``"ros2"``.
    Falls back to ``"json"`` when the optional reader is unavailable or the
    recording is empty. Reader and format errors propagate to the caller.

    Args:
        path: MCAP file to inspect.

    Returns:
        Detected message encoding, defaulting to ``"json"`` when no message can be inspected.
    """
    try:
        from mcap.reader import make_reader  # type: ignore

        with path.open("rb") as stream:
            reader = make_reader(stream)
            for _schema, channel, message in reader.iter_messages():
                enc = (getattr(channel, "message_encoding", "") or "").lower()
                if enc:
                    return enc
                # No explicit encoding field: probe the first message's bytes.
                try:
                    json.loads(message.data.decode("utf-8"))
                    return "json"
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return "cdr"
    except (ImportError, StopIteration):
        return "json"
    return "json"


def load_mcap_trajectory(config: TrajectoryLoadConfig) -> TrajectoryDataset:
    """Load telemetry from an MCAP file with JSON or binary ROS 2 joint telemetry messages.

    Args:
        config: MCAP path and topic mapping used to extract trajectory signals.

    Returns:
        Validated trajectory loaded from the MCAP recording.
    """
    try:
        path = Path(find_companion_mcap_path(config.source_path)).expanduser()
    except ValueError as exc:
        raise TrajectoryIngestError(str(exc)) from exc
    if not path.is_file():
        raise TrajectoryIngestError(f"MCAP file not found: {path}")

    mapping = _resolve_topic_mapping(config)
    pos_topic = mapping.position_topic
    if not pos_topic:
        raise TrajectoryIngestError("topic mapping must define position_topic.")
    topics = _mapped_topics(mapping)

    try:
        from mcap.reader import make_reader  # type: ignore
    except ImportError as exc:
        raise TrajectoryIngestError(
            "MCAP ingestion requires the optional 'mcap' package in the active Isaac Sim Python environment."
        ) from exc

    times: list[float] = []
    positions: list[list[float]] = []
    time_samples: list[tuple[float, float]] = []
    vel_topic = mapping.velocity_topic or pos_topic
    cmd_topic = mapping.command_topic or pos_topic
    torque_topic = mapping.torque_topic or pos_topic
    ee_topic = mapping.end_effector_pose_topic
    contact_topic = mapping.contact_force_topic
    json_decode_failures: dict[str, int] = {}
    channel_decode_failures: dict[str, int] = {}
    position_json_errors: list[str] = []

    def _json_messages() -> Iterator[tuple[str, float, Any]]:
        """Yield JSON messages from the MCAP file."""  # noqa: DOC402
        with path.open("rb") as stream:
            reader = make_reader(stream)
            for _schema, channel, message in _iter_reader_messages(reader, topics):
                topic = channel.topic
                timestamp = float(message.log_time) * 1e-9
                try:
                    payload = json.loads(message.data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    json_decode_failures[topic] = json_decode_failures.get(topic, 0) + 1
                    if topic == pos_topic and not position_json_errors:
                        # The encoding sniff inspects only the first message, so a
                        # malformed position payload may just mean this is a binary
                        # recording. Record the error and let the caller decide once
                        # the binary retry has had a chance to succeed.
                        position_json_errors.append(
                            f"MCAP position topic {topic!r} contains malformed JSON at "
                            f"log time {message.log_time}: {exc}"
                        )
                    continue
                if isinstance(payload, dict):
                    yield topic, timestamp, payload

    def _collect(
        use_binary: bool,
    ) -> tuple[
        list[tuple[float, float, Any, list[float]]],
        list[tuple[float, float, list[float]]],
        list[tuple[float, float, list[float]]],
        list[tuple[float, float, list[float]]],
        list[tuple[float, float, list[float]]],
        list[tuple[float, float, list[float]]],
        list[tuple[float, float]],
    ]:
        """Decode the file with the chosen codec and bucket samples per channel.

        Args:
            use_binary: Decode ROS 2 CDR messages when true; otherwise decode JSON payloads.

        Returns:
            Position records, optional-channel samples, and mapped time samples.
        """
        source = _iter_decoded_ros2_messages(path, topics) if use_binary else _json_messages()
        records: list[tuple[float, float, Any, list[float]]] = []
        vel: list[tuple[float, float, list[float]]] = []
        cmd: list[tuple[float, float, list[float]]] = []
        torque: list[tuple[float, float, list[float]]] = []
        ee: list[tuple[float, float, list[float]]] = []
        contact: list[tuple[float, float, list[float]]] = []
        time_only: list[tuple[float, float]] = []
        for topic, timestamp, payload in source:
            sample_time = _safe_payload_time(payload, mapping.time_field, timestamp)
            if mapping.time_topic and topic == mapping.time_topic and topic != pos_topic:
                time_only.append((timestamp, sample_time))
            if topic == pos_topic:
                pos = _payload_to_vectors(payload, mapping.position_fields, mapping.joint_names)
                records.append((timestamp, sample_time, payload, pos))
                continue
            if topic == vel_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "velocity",
                    _series_payload(vel, timestamp, sample_time, payload, mapping.velocity_fields, mapping.joint_names),
                )
            if topic == cmd_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "command",
                    _series_payload(cmd, timestamp, sample_time, payload, mapping.command_fields, mapping.joint_names),
                )
            if topic == torque_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "torque",
                    _series_payload(
                        torque,
                        timestamp,
                        sample_time,
                        payload,
                        mapping.torque_fields,
                        mapping.joint_names,
                        first_match=True,
                    ),
                )
            if topic == ee_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "end_effector_pose",
                    _series_payload(ee, timestamp, sample_time, payload, mapping.end_effector_pose_fields),
                )
            if topic == contact_topic:
                _record_decode_failure(
                    channel_decode_failures,
                    "contact_force",
                    _series_payload(contact, timestamp, sample_time, payload, mapping.contact_force_fields),
                )
        return records, vel, cmd, torque, ee, contact, time_only

    # Detect encoding with a one-message peek so we choose the right decoder
    # upfront and avoid an expensive wasted pass on binary files.
    use_binary = _sniff_mcap_encoding(path) not in ("json", "")
    (
        position_records,
        velocity_samples,
        command_samples,
        torque_samples,
        ee_samples,
        contact_samples,
        time_samples,
    ) = _collect(use_binary)

    # The sniff inspects only the first message; if a JSON guess yields too few
    # position samples, retry as binary before giving up so a binary file isn't
    # silently reported as empty.
    if len(position_records) < 2 and not use_binary:
        json_decode_failures.clear()
        channel_decode_failures.clear()
        (
            position_records,
            velocity_samples,
            command_samples,
            torque_samples,
            ee_samples,
            contact_samples,
            time_samples,
        ) = _collect(True)
    elif position_json_errors:
        # JSON decoding produced a usable position series, so the messages that failed
        # are real data loss rather than a wrong-codec guess. Dropping them would load
        # a silently decimated trajectory.
        raise TrajectoryIngestError(position_json_errors[0])

    if len(position_records) < 2:
        if position_json_errors:
            raise TrajectoryIngestError(position_json_errors[0])
        raise TrajectoryIngestError(
            "MCAP file did not yield enough samples on position_topic. JSON telemetry is supported "
            "directly; binary ROS 2 messages require rosbags and compatible message definitions."
        )

    time_diagnostic = None
    log_times_arr = np.asarray([row[0] for row in position_records], dtype=np.float64)
    fallback_times_arr = np.asarray([row[1] for row in position_records], dtype=np.float64)
    tolerance_override = config.alignment_tolerance_seconds
    if mapping.time_topic and mapping.time_topic != pos_topic:
        resolved_times, time_diagnostic = _align_time_topic_series(
            log_times_arr,
            fallback_times_arr,
            time_samples,
            mapping.time_topic,
            tolerance_seconds=resolve_topic_alignment_tolerance(log_times_arr, tolerance_override),
        )
    else:
        resolved_times = fallback_times_arr
    _require_full_time_topic_coverage(time_diagnostic, "MCAP")

    for position_time, (_log_time, _fallback_time, payload, pos) in zip(resolved_times, position_records):
        times.append(float(position_time))
        positions.append(pos)
        if vel_topic == pos_topic:
            _record_decode_failure(
                channel_decode_failures,
                "velocity",
                _series_payload(
                    velocity_samples,
                    float(position_time),
                    float(position_time),
                    payload,
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
                    float(position_time),
                    float(position_time),
                    payload,
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
                    float(position_time),
                    float(position_time),
                    payload,
                    mapping.torque_fields,
                    mapping.joint_names,
                    first_match=True,
                ),
            )
        if ee_topic == pos_topic:
            _record_decode_failure(
                channel_decode_failures,
                "end_effector_pose",
                _series_payload(
                    ee_samples, float(position_time), float(position_time), payload, mapping.end_effector_pose_fields
                ),
            )
        if contact_topic == pos_topic:
            _record_decode_failure(
                channel_decode_failures,
                "contact_force",
                _series_payload(
                    contact_samples, float(position_time), float(position_time), payload, mapping.contact_force_fields
                ),
            )

    times_arr = np.asarray(times, dtype=np.float64)
    _validate_position_times(times_arr, "MCAP")
    pos_arr = np.asarray(positions, dtype=np.float64)
    tolerance_seconds = resolve_topic_alignment_tolerance(times_arr, tolerance_override)
    diagnostics: dict[str, dict[str, Any]] = {}
    time_topic_for_optional = mapping.time_topic if mapping.time_topic and mapping.time_topic != pos_topic else None
    resolved_velocity_samples, velocity_time_diagnostic = _resolved_optional_samples(
        velocity_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional if vel_topic != pos_topic else None,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_command_samples, command_time_diagnostic = _resolved_optional_samples(
        command_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional if cmd_topic != pos_topic else None,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_torque_samples, torque_time_diagnostic = _resolved_optional_samples(
        torque_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional if torque_topic != pos_topic else None,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_ee_samples, ee_time_diagnostic = _resolved_optional_samples(
        ee_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional if ee_topic != pos_topic else None,
        tolerance_seconds=tolerance_seconds,
    )
    resolved_contact_samples, contact_time_diagnostic = _resolved_optional_samples(
        contact_samples,
        time_samples=time_samples,
        time_topic=time_topic_for_optional if contact_topic != pos_topic else None,
        tolerance_seconds=tolerance_seconds,
    )

    def _decode_failure_count(channel: str, topic: str | None) -> int:
        """Combine per-channel extraction failures with per-topic JSON decode failures.

        Args:
            channel: Signal name whose samples were collected.
            topic: Topic the channel was read from, when one is mapped.

        Returns:
            Number of source messages that never reached the channel.
        """
        json_failures = json_decode_failures.get(topic, 0) if topic else 0
        return channel_decode_failures.get(channel, 0) + json_failures

    vel_arr, diagnostics["velocity"] = _align_optional_series(
        times_arr, resolved_velocity_samples, "velocity", width=pos_arr.shape[1], tolerance_seconds=tolerance_seconds
    )
    diagnostics["velocity"] = _attach_time_remap_diagnostic(diagnostics["velocity"], velocity_time_diagnostic)
    vel_arr = _finalize_optional_channel(
        vel_arr,
        diagnostics["velocity"],
        required_topic=mapping.velocity_topic,
        decode_failure_count=_decode_failure_count("velocity", vel_topic),
        source="MCAP",
    )
    cmd_arr, diagnostics["command"] = _align_optional_series(
        times_arr, resolved_command_samples, "command", width=pos_arr.shape[1], tolerance_seconds=tolerance_seconds
    )
    diagnostics["command"] = _attach_time_remap_diagnostic(diagnostics["command"], command_time_diagnostic)
    cmd_arr = _finalize_optional_channel(
        cmd_arr,
        diagnostics["command"],
        required_topic=mapping.command_topic,
        decode_failure_count=_decode_failure_count("command", cmd_topic),
        source="MCAP",
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
        decode_failure_count=_decode_failure_count("torque", torque_topic),
        source="MCAP",
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
        decode_failure_count=_decode_failure_count("end_effector_pose", ee_topic),
        source="MCAP",
    )
    contact_arr, diagnostics["contact_force"] = _align_optional_series(
        times_arr, resolved_contact_samples, "contact_force", tolerance_seconds=tolerance_seconds
    )
    diagnostics["contact_force"] = _attach_time_remap_diagnostic(diagnostics["contact_force"], contact_time_diagnostic)
    contact_arr = _finalize_optional_channel(
        contact_arr,
        diagnostics["contact_force"],
        required_topic=None,
        decode_failure_count=_decode_failure_count("contact_force", contact_topic),
        source="MCAP",
    )
    extra = _topic_alignment_extra(mapping, diagnostics, time_diagnostic, tolerance_seconds=tolerance_seconds)
    if json_decode_failures:
        extra["json_decode_failures"] = dict(sorted(json_decode_failures.items()))
    return build_trajectory_dataset(
        times=times_arr,
        positions=pos_arr,
        velocities=vel_arr,
        commands=cmd_arr,
        torques=torque_arr,
        end_effector_poses=ee_arr,
        contact_forces=contact_arr,
        source_type="mcap",
        source_path=str(path),
        topic_mapping_path=config.topic_mapping_path,
        velocities_required=mapping.velocity_topic is not None,
        commands_required=mapping.command_topic is not None,
        torques_required=mapping.torque_topic is not None,
        extra=extra,
    )
