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

# ruff: noqa: D413

"""Unit tests for SysID trajectory CSV and multi-source ingest helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import isaacsim.robot_setup.sysid.ingest.lerobot_loader as lerobot_loader
import isaacsim.robot_setup.sysid.ingest.mcap_loader as mcap_loader
import isaacsim.robot_setup.sysid.ingest.ros2_bag as ros2_bag
import numpy as np
import omni.kit.test
from isaacsim.robot_setup.sysid.ingest import load_trajectory
from isaacsim.robot_setup.sysid.ingest.config_types import (
    CsvColumnMapping,
    TopicSignalMapping,
    TrajectoryIngestError,
    TrajectoryLoadConfig,
)
from isaacsim.robot_setup.sysid.ingest.csv_loader import load_csv_trajectory
from isaacsim.robot_setup.sysid.ingest.lerobot_loader import load_lerobot_trajectory
from isaacsim.robot_setup.sysid.ingest.mcap_loader import (
    _align_optional_series,
    _align_time_topic_series,
    _payload_to_vectors,
    _sniff_mcap_encoding,
    load_mcap_trajectory,
    resolve_topic_alignment_tolerance,
)
from isaacsim.robot_setup.sysid.ingest.ros2_bag import (
    _find_mcap_in_bag_dir,
    load_ros2_bag_trajectory,
)
from isaacsim.robot_setup.sysid.ingest.trajectory_builder import (
    build_trajectory_dataset,
)
from isaacsim.robot_setup.sysid.parameter_types import build_extended_parameter_specs
from isaacsim.robot_setup.sysid.trajectory_csv import (
    TrajectoryDataset,
    load_sysid_trajectory_csv,
)
from isaacsim.robot_setup.sysid.trajectory_csv_core import (
    TrajectoryCsvError,
    infer_telemetry_layout_from_header,
    peek_csv_header_names,
)


def _write_split_bag_metadata(bag_dir: Path, parts: list[str]) -> None:
    """Write a minimal ROS 2 bag index declaring storage files as one split recording.

    Args:
        bag_dir: Bag directory that receives the index.
        parts: Storage file names relative to the bag directory.
    """
    lines = ["rosbag2_bagfile_information:", "  relative_file_paths:"]
    lines.extend(f"    - {name}" for name in parts)
    (bag_dir / "metadata.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FakeRos2Connection:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.msgtype = "fake_msgs/msg/Telemetry"


class _FakeRos2Reader:
    def __init__(self, messages_by_topic: dict[str, list[tuple[int, dict]]]) -> None:
        self._messages_by_topic = messages_by_topic
        self.connections = [_FakeRos2Connection(topic) for topic in messages_by_topic]

    def __enter__(self) -> "_FakeRos2Reader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        return None

    def messages(
        self,
        *,
        connections: list[_FakeRos2Connection],
    ) -> Iterator[tuple[_FakeRos2Connection, int, dict]]:
        """Yield fake ROS 2 messages for the selected connection.

        Args:
            connections: Fake connections selected by the loader.
        """  # noqa: DOC402
        conn = connections[0]
        for timestamp, payload in self._messages_by_topic.get(conn.topic, []):
            yield conn, timestamp, payload

    def deserialize(self, rawdata: dict, msgtype: str) -> dict:
        """Return fake ROS 2 payloads without binary deserialization.

        Args:
            rawdata: Already-decoded fake payload.
            msgtype: Ignored fake message-type name.

        Returns:
            The supplied fake payload unchanged.
        """
        return rawdata


class _FakeRos2MultiConnectionReader:
    """Fake reader with multiple connection objects for the same topic.

    Args:
        rows_by_connection: Topic names paired with timestamped fake payloads.
    """

    def __init__(self, rows_by_connection: list[tuple[str, list[tuple[int, dict]]]]) -> None:
        self.connections = [_FakeRos2Connection(topic) for topic, _rows in rows_by_connection]
        self._rows_by_connection = {
            connection: rows for connection, (_topic, rows) in zip(self.connections, rows_by_connection)
        }

    def __enter__(self) -> "_FakeRos2MultiConnectionReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        return None

    def messages(
        self,
        *,
        connections: list[_FakeRos2Connection],
    ) -> Iterator[tuple[_FakeRos2Connection, int, dict]]:
        rows = [
            (connection, timestamp, payload)
            for connection in connections
            for timestamp, payload in self._rows_by_connection[connection]
        ]
        yield from sorted(rows, key=lambda row: row[1])

    def deserialize(self, rawdata: dict, msgtype: str) -> dict:
        return rawdata


class _FakeMcapChannel:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.message_encoding = "json"


class _FakeMcapMessage:
    def __init__(self, log_time_seconds: float, payload: dict | bytes) -> None:
        self.log_time = int(round(float(log_time_seconds) * 1e9))
        self.data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")


class _FakeMcapReader:
    def __init__(self, messages: list[tuple[str, float, dict | bytes]]) -> None:
        self._messages = messages

    def iter_messages(
        self,
        topics: list[str] | None = None,
    ) -> Iterator[tuple[None, _FakeMcapChannel, _FakeMcapMessage]]:
        """Yield fake MCAP messages filtered by topic.

        Args:
            topics: Optional topic allowlist.
        """  # noqa: DOC402
        requested = set(topics or [])
        for topic, log_time, payload in self._messages:
            if requested and topic not in requested:
                continue
            yield None, _FakeMcapChannel(topic), _FakeMcapMessage(log_time, payload)


class _FakeMcapModulePatch:
    def __init__(self, messages: list[tuple[str, float, dict | bytes]]) -> None:
        self._messages = messages
        self._original_mcap = None
        self._original_reader = None

    def __enter__(self) -> "_FakeMcapModulePatch":
        self._original_mcap = sys.modules.get("mcap")
        self._original_reader = sys.modules.get("mcap.reader")
        mcap_module = types.ModuleType("mcap")
        reader_module = types.ModuleType("mcap.reader")
        reader_module.make_reader = lambda stream, **kwargs: _FakeMcapReader(self._messages)
        mcap_module.reader = reader_module
        sys.modules["mcap"] = mcap_module
        sys.modules["mcap.reader"] = reader_module
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self._original_mcap is None:
            sys.modules.pop("mcap", None)
        else:
            sys.modules["mcap"] = self._original_mcap
        if self._original_reader is None:
            sys.modules.pop("mcap.reader", None)
        else:
            sys.modules["mcap.reader"] = self._original_reader


class _FakeArrowType:
    def __init__(self, kind: str, list_size: int = 0) -> None:
        self.kind = kind
        self.list_size = list_size


class _FakeArrowValues:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.null_count = 0

    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray:
        """Return fake Arrow values as a NumPy array.

        Args:
            zero_copy_only: Unused compatibility flag matching the Arrow API.

        Returns:
            Values converted to a NumPy array.
        """
        return np.asarray(self._values)


class _FakeArrowColumn:
    def __init__(self, values: list) -> None:
        self._values = values
        self.null_count = 0
        first = values[0]
        self.type = (
            _FakeArrowType("fixed_size_list", len(first))
            if isinstance(first, list)
            else _FakeArrowType("integer" if isinstance(first, int) else "floating")
        )

    def __len__(self) -> int:
        return len(self._values)

    def combine_chunks(self) -> _FakeArrowColumn:
        """Return the already contiguous fake column.

        Returns:
            This fake column.
        """
        return self

    @property
    def values(self) -> _FakeArrowValues:
        """Return flattened values for a fixed-size-list column."""
        return _FakeArrowValues([value for row in self._values for value in row])

    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray:
        """Return scalar fake column values as a NumPy array.

        Args:
            zero_copy_only: Unused compatibility flag matching the Arrow API.

        Returns:
            Column values converted to a NumPy array.
        """
        return np.asarray(self._values)

    def to_pylist(self) -> list:
        """Return fake column values as Python rows.

        Returns:
            A copy of the fake column values.
        """
        return list(self._values)


class _FakeArrowTable:
    def __init__(self, columns: dict[str, list]) -> None:
        self._columns = columns
        self.column_names = list(columns)
        self.num_rows = len(next(iter(columns.values()))) if columns else 0

    def column(self, name: str) -> _FakeArrowColumn:
        """Return one named fake Arrow column.

        Args:
            name: Column name to retrieve.

        Returns:
            Fake Arrow column for the requested name.
        """
        return _FakeArrowColumn(self._columns[name])


class _FakeArrowTypes:
    @staticmethod
    def is_fixed_size_list(column_type: _FakeArrowType) -> bool:
        """Return whether the fake type is a fixed-size list.

        Args:
            column_type: Fake Arrow type to inspect.

        Returns:
            Whether the type represents a fixed-size list.
        """
        return column_type.kind == "fixed_size_list"

    @staticmethod
    def is_list(column_type: _FakeArrowType) -> bool:
        """Return whether the fake type is a variable-size list.

        Args:
            column_type: Fake Arrow type to inspect.

        Returns:
            Whether the type represents a variable-size list.
        """
        return column_type.kind == "list"

    @staticmethod
    def is_large_list(column_type: _FakeArrowType) -> bool:
        """Return whether the fake type is a large variable-size list.

        Args:
            column_type: Fake Arrow type to inspect.

        Returns:
            Whether the type represents a large variable-size list.
        """
        return column_type.kind == "large_list"

    @staticmethod
    def is_integer(column_type: _FakeArrowType) -> bool:
        """Return whether the fake type is integral.

        Args:
            column_type: Fake Arrow type to inspect.

        Returns:
            Whether the type represents an integer.
        """
        return column_type.kind == "integer"

    @staticmethod
    def is_floating(column_type: _FakeArrowType) -> bool:
        """Return whether the fake type is floating point.

        Args:
            column_type: Fake Arrow type to inspect.

        Returns:
            Whether the type represents a floating-point number.
        """
        return column_type.kind == "floating"


class TrajectoryIngestTests(omni.kit.test.AsyncTestCase):
    """Tests for SysID module boundaries and trajectory ingestion paths."""

    async def test_mcap_encoding_sniff_propagates_reader_errors(self) -> None:
        """Preserve corrupt-container errors instead of misclassifying them as JSON."""
        with tempfile.TemporaryDirectory(prefix="sysid_bad_mcap_header_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"corrupt")
            with (
                _FakeMcapModulePatch([]),
                patch("mcap.reader.make_reader", side_effect=RuntimeError("corrupt MCAP header")),
                self.assertRaisesRegex(RuntimeError, "corrupt MCAP header"),
            ):
                _sniff_mcap_encoding(mcap_path)

    async def test_extended_parameter_specs_include_inertia_without_runtime_imports(self) -> None:
        """Build inertia parameter specs without importing optional simulation runtimes."""
        specs = build_extended_parameter_specs(
            1,
            2,
            include_basic=False,
            include_inertia_log_cholesky=True,
        )

        self.assertEqual(len(specs), 12)
        self.assertEqual({spec.entry.component_index for spec in specs}, set(range(6)))

    async def test_csv_core_normalizes_header_and_infers_layout(self) -> None:
        """Normalize CSV headers and infer the compact telemetry layout."""
        with tempfile.TemporaryDirectory(prefix="sysid_csv_core_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text(
                "# comment\nTime, q1, dq1, cmd1\n0.0, 0.0, 0.0, 0.2\n0.1, 0.1, 1.0, 0.3\n",
                encoding="utf-8",
            )

            self.assertEqual(peek_csv_header_names(str(csv_path)), ["time", "q1", "dq1", "cmd1"])
            self.assertEqual(infer_telemetry_layout_from_header(["time", "q1", "dq1", "cmd1"]), (1, "full"))

            trajectory = load_sysid_trajectory_csv(str(csv_path))
            self.assertEqual(trajectory.num_joints, 1)
            np.testing.assert_allclose(trajectory.commands[:, 0], [0.2, 0.3])

    async def test_positions_only_csv_derives_velocity_and_uses_position_commands(self) -> None:
        """Derive velocity and commands for a positions-only CSV."""
        with tempfile.TemporaryDirectory(prefix="sysid_csv_pos_only_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text("time,q1\n0.0,0.0\n0.5,1.0\n1.0,3.0\n", encoding="utf-8")

            trajectory = load_sysid_trajectory_csv(str(csv_path))

            np.testing.assert_allclose(trajectory.velocities[:, 0], [2.0, 2.0, 4.0])
            np.testing.assert_allclose(trajectory.commands, trajectory.positions)

    async def test_headerless_csv_rejects_ambiguous_joint_layout(self) -> None:
        """Reject headerless columns that could represent multiple joint layouts."""
        with tempfile.TemporaryDirectory(prefix="sysid_csv_ambiguous_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text(
                "0.0,0,1,2,3,4,5\n0.1,1,2,3,4,5,6\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TrajectoryCsvError, "ambiguous"):
                load_sysid_trajectory_csv(str(csv_path))

    async def test_csv_loader_mapping_populates_optional_signals(self) -> None:
        """Use column mapping to populate optional telemetry channels."""
        with tempfile.TemporaryDirectory(prefix="sysid_csv_mapping_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            mapping_path = Path(tmp_dir) / "columns.json"
            csv_path.write_text(
                "stamp,joint,djoint,command,effort,ee_x,ee_y,force\n"
                "0.0,0.0,0.0,0.2,1.0,0.0,1.0,5.0\n"
                "0.1,0.1,1.0,0.3,2.0,0.2,1.2,6.0\n",
                encoding="utf-8",
            )
            mapping_path.write_text(
                json.dumps(
                    {
                        "time_column": "stamp",
                        "position_columns": ["joint"],
                        "velocity_columns": ["djoint"],
                        "command_columns": ["command"],
                        "torque_columns": ["effort"],
                        "end_effector_pose_columns": ["ee_x", "ee_y"],
                        "contact_force_columns": ["force"],
                        "joint_names": ["arm_joint"],
                    }
                ),
                encoding="utf-8",
            )

            trajectory = load_csv_trajectory(
                TrajectoryLoadConfig.csv(str(csv_path), column_mapping_path=str(mapping_path))
            )

            np.testing.assert_allclose(trajectory.torques[:, 0], [1.0, 2.0])
            np.testing.assert_allclose(trajectory.end_effector_poses[:, 1], [1.0, 1.2])
            np.testing.assert_allclose(trajectory.contact_forces[:, 0], [5.0, 6.0])
            self.assertEqual(trajectory.metadata.column_mapping_path, str(mapping_path))
            self.assertEqual(
                trajectory.metadata.extra["column_mapping"]["joint_names"],
                ["arm_joint"],
            )

    async def test_unified_loader_dispatches_csv_without_optional_reader_imports(self) -> None:
        """Dispatch CSV loading without importing optional bag readers."""
        with tempfile.TemporaryDirectory(prefix="sysid_load_dispatch_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text("time,q1,dq1,cmd1\n0.0,0.0,0.0,0.2\n0.1,0.1,1.0,0.3\n", encoding="utf-8")

            trajectory = load_trajectory(TrajectoryLoadConfig.csv(str(csv_path)))

            self.assertIsInstance(trajectory, TrajectoryDataset)
            np.testing.assert_allclose(trajectory.positions[:, 0], [0.0, 0.1])

    async def test_trajectory_builder_fills_missing_velocity_and_command_channels(self) -> None:
        """Fill missing velocity and command channels in trajectory builder."""
        trajectory = build_trajectory_dataset(
            times=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
            positions=np.asarray([[0.0], [1.0], [3.0]], dtype=np.float64),
            velocities=None,
            commands=None,
            source_type="custom",
            source_path="memory",
        )

        np.testing.assert_allclose(trajectory.velocities[:, 0], [2.0, 2.0, 4.0])
        np.testing.assert_allclose(trajectory.commands, trajectory.positions)
        self.assertEqual(trajectory.metadata.source_type, "custom")

    async def test_trajectory_builder_rejects_missing_required_commands(self) -> None:
        """Do not synthesize positions when a source explicitly requested commands."""
        with self.assertRaisesRegex(TrajectoryIngestError, "configured command channel"):
            build_trajectory_dataset(
                times=np.asarray([0.0, 0.1], dtype=np.float64),
                positions=np.asarray([[0.0], [0.1]], dtype=np.float64),
                velocities=None,
                commands=None,
                source_type="custom",
                source_path="memory",
                commands_required=True,
            )

    async def test_trajectory_builder_rejects_mismatched_optional_shapes(self) -> None:
        """Reject optional telemetry channels with mismatched shapes."""
        with self.assertRaisesRegex(TrajectoryIngestError, "torques shape"):
            build_trajectory_dataset(
                times=np.asarray([0.0, 0.1], dtype=np.float64),
                positions=np.zeros((2, 2), dtype=np.float64),
                velocities=np.zeros((2, 2), dtype=np.float64),
                commands=np.zeros((2, 2), dtype=np.float64),
                torques=np.zeros((2, 1), dtype=np.float64),
                source_type="custom",
                source_path="memory",
            )

    async def test_trajectory_builder_rejects_zero_norm_pose_quaternion(self) -> None:
        """Reject invalid xyz+xyzw telemetry at the ingestion boundary."""
        with self.assertRaisesRegex(TrajectoryIngestError, "quaternion must have nonzero finite norm"):
            build_trajectory_dataset(
                times=np.asarray([0.0, 0.1], dtype=np.float64),
                positions=np.zeros((2, 1), dtype=np.float64),
                velocities=np.zeros((2, 1), dtype=np.float64),
                commands=np.zeros((2, 1), dtype=np.float64),
                end_effector_poses=np.zeros((2, 7), dtype=np.float64),
                source_type="custom",
                source_path="memory",
            )

    async def test_trajectory_builder_rejects_nonfinite_channels(self) -> None:
        """Reject NaN and infinity in every telemetry channel."""
        cases = {
            "times": {"times": np.asarray([0.0, np.inf])},
            "positions": {"positions": np.asarray([[0.0], [np.nan]])},
            "velocities": {"velocities": np.asarray([[0.0], [np.inf]])},
            "commands": {"commands": np.asarray([[0.0], [np.nan]])},
            "torques": {"torques": np.asarray([[0.0], [np.inf]])},
            "end_effector_poses": {"end_effector_poses": np.asarray([[0.0, 0.0], [np.nan, 0.0]])},
            "contact_forces": {"contact_forces": np.asarray([[0.0], [np.inf]])},
        }
        for channel, override in cases.items():
            with self.subTest(channel=channel):
                kwargs = {
                    "times": np.asarray([0.0, 0.1]),
                    "positions": np.asarray([[0.0], [0.1]]),
                    "velocities": np.asarray([[0.0], [1.0]]),
                    "commands": np.asarray([[0.0], [0.1]]),
                    "source_type": "test",
                    "source_path": "memory",
                }
                kwargs.update(override)
                with self.assertRaisesRegex(TrajectoryIngestError, channel):
                    build_trajectory_dataset(**kwargs)

    async def test_trajectory_data_slice_preserves_optional_channels(self) -> None:
        """Preserve optional channels when slicing by time window."""
        trajectory = TrajectoryDataset(
            times=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            positions=np.arange(4, dtype=np.float64).reshape(4, 1),
            velocities=np.zeros((4, 1), dtype=np.float64),
            commands=np.zeros((4, 1), dtype=np.float64),
            torques=np.ones((4, 1), dtype=np.float64),
            residual_sample_weights=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

        sliced = trajectory.slice_time_window(0.1, 0.2)

        np.testing.assert_allclose(sliced.times, [0.1, 0.2])
        np.testing.assert_allclose(sliced.torques[:, 0], [1.0, 1.0])
        np.testing.assert_allclose(sliced.residual_sample_weights, [2.0, 3.0])

    async def test_mcap_payload_helpers_apply_joint_order(self) -> None:
        """Apply requested joint order while extracting MCAP payload vectors."""
        payload = {"name": ["joint_b", "joint_a"], "position": [2.0, 1.0]}

        values = _payload_to_vectors(payload, ["position"], ["joint_a", "joint_b"])

        self.assertEqual(values, [1.0, 2.0])

    async def test_mcap_payload_helpers_reject_duplicate_joint_names(self) -> None:
        """Reject message name arrays that would silently overwrite a joint value."""
        payload = {"name": ["joint_a", "joint_a"], "position": [1.0, 2.0]}

        with self.assertRaisesRegex(TrajectoryIngestError, "duplicate joint names"):
            _payload_to_vectors(payload, ["position"], ["joint_a"])

    async def test_mcap_torque_aliases_choose_first_present_field(self) -> None:
        """Treat effort/torque mapping entries as aliases instead of concatenating them."""
        payload = {"effort": [1.0, 2.0], "torque": [10.0, 20.0]}

        values = _payload_to_vectors(payload, ["effort", "torque"], first_match=True)

        self.assertEqual(values, [1.0, 2.0])

    async def test_binary_ros2_mcap_uses_rosbags_reader(self) -> None:
        """Decode binary ROS 2 MCAP samples through the shared rosbags reader."""
        messages_by_topic = {
            "/joint_states": [(1_000_000_000, {"position": [0.1]})],
            "/ignored": [(2_000_000_000, {"position": [0.2]})],
        }
        original_create_any_reader = mcap_loader.create_any_reader
        mcap_loader.create_any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
        try:
            messages = list(mcap_loader._iter_decoded_ros2_messages(Path("telemetry.mcap"), {"/joint_states"}))
        finally:
            mcap_loader.create_any_reader = original_create_any_reader

        self.assertEqual(messages, [("/joint_states", 1.0, {"position": [0.1]})])

    async def test_topic_alignment_matches_nearest_samples_within_tolerance(self) -> None:
        """Align optional topic samples to nearest target timestamps."""
        aligned, diagnostic = _align_optional_series(
            np.asarray([0.0, 0.1], dtype=np.float64),
            [(0.005, [1.0]), (0.095, [2.0])],
            "velocity",
            width=1,
        )

        self.assertEqual(diagnostic["status"], "aligned")
        np.testing.assert_allclose(aligned[:, 0], [1.0, 2.0])

    async def test_ros2_topic_alignment_does_not_require_exact_timestamp_equality(self) -> None:
        """Allow ROS 2 optional samples to align by nearest timestamp."""
        aligned, diagnostic = _align_optional_series(
            np.asarray([10.0, 10.1, 10.2], dtype=np.float64),
            [(10.004, [0.1]), (10.096, [0.2]), (10.204, [0.3])],
            "command",
            width=1,
        )

        self.assertEqual(diagnostic["status"], "aligned")
        np.testing.assert_allclose(aligned[:, 0], [0.1, 0.2, 0.3])

    async def test_topic_alignment_omits_incomplete_channels_instead_of_zero_filling(self) -> None:
        """Omit incomplete optional channels instead of zero filling them."""
        aligned, diagnostic = _align_optional_series(
            np.asarray([0.0, 0.1], dtype=np.float64),
            [(0.0, [1.0])],
            "torque",
            width=1,
        )

        self.assertIsNone(aligned)
        self.assertEqual(diagnostic["status"], "omitted")
        self.assertEqual(diagnostic["reason"], "outside_tolerance")

    async def test_topic_alignment_allows_fewer_samples_when_nearest_covers_targets(self) -> None:
        """Allow fewer optional samples when nearest matching covers targets."""
        aligned, diagnostic = _align_optional_series(
            np.asarray([0.0, 0.01], dtype=np.float64),
            [(0.005, [1.0])],
            "velocity",
            width=1,
        )

        self.assertEqual(diagnostic["status"], "aligned")
        np.testing.assert_allclose(aligned[:, 0], [1.0, 1.0])

    async def test_topic_alignment_infers_width_from_matched_sample(self) -> None:
        """Infer optional channel width from the matched payload sample."""
        aligned, diagnostic = _align_optional_series(
            np.asarray([0.1], dtype=np.float64),
            [(0.0, [1.0, 2.0]), (0.1, [3.0, 4.0, 5.0])],
            "end_effector_pose",
        )

        self.assertEqual(diagnostic["status"], "aligned")
        np.testing.assert_allclose(aligned, [[3.0, 4.0, 5.0]])

    async def test_mcap_time_topic_alignment_uses_nearest_timestamped_samples(self) -> None:
        """Use nearest timestamped samples from the MCAP time topic."""
        aligned, diagnostic = _align_time_topic_series(
            np.asarray([10.0, 10.1], dtype=np.float64),
            np.asarray([100.0, 100.1], dtype=np.float64),
            [(10.004, 1.0), (10.096, 1.1)],
            "/clock",
        )

        self.assertEqual(diagnostic["status"], "aligned")
        self.assertEqual(diagnostic["fallback_count"], 0)
        np.testing.assert_allclose(aligned, [1.0, 1.1])

    async def test_mcap_time_topic_alignment_does_not_depend_on_message_adjacency(self) -> None:
        """Align MCAP time-topic samples without relying on message adjacency."""
        aligned, diagnostic = _align_time_topic_series(
            np.asarray([20.0, 20.1, 20.2], dtype=np.float64),
            np.asarray([200.0, 200.1, 200.2], dtype=np.float64),
            [(20.0, 2.0), (20.1, 2.1), (20.2, 2.2)],
            "/clock",
        )

        self.assertEqual(diagnostic["status"], "aligned")
        np.testing.assert_allclose(aligned, [2.0, 2.1, 2.2])

    async def test_mcap_time_topic_alignment_falls_back_with_diagnostics(self) -> None:
        """Fall back to local time and report diagnostics when alignment fails."""
        aligned, diagnostic = _align_time_topic_series(
            np.asarray([30.0, 30.1], dtype=np.float64),
            np.asarray([300.0, 300.1], dtype=np.float64),
            [(29.0, 3.0)],
            "/clock",
        )

        self.assertEqual(diagnostic["status"], "fallback")
        self.assertEqual(diagnostic["fallback_count"], 2)
        np.testing.assert_allclose(aligned, [300.0, 300.1])

    async def test_mcap_time_topic_overrides_bad_local_time_fields_and_reports_remap(self) -> None:
        """Use the MCAP time topic when local payload time fields are invalid."""
        bad_stamp = {"sec": "not-a-number", "nanosec": 0}
        mapping = TopicSignalMapping(
            time_topic="/clock",
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
        )
        messages = [
            ("/clock", 10.0, {"stamp": 1.0}),
            ("/joint_states", 10.0, {"stamp": bad_stamp, "position": [0.0]}),
            ("/joint_velocity", 10.0, {"stamp": bad_stamp, "velocity": [0.5]}),
            ("/clock", 10.1, {"stamp": 1.1}),
            ("/joint_states", 10.1, {"stamp": bad_stamp, "position": [0.1]}),
            ("/joint_velocity", 10.1, {"stamp": bad_stamp, "velocity": [0.6]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_bad_local_time_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with _FakeMcapModulePatch(messages):
                trajectory = load_mcap_trajectory(config)

        np.testing.assert_allclose(trajectory.times, [1.0, 1.1])
        np.testing.assert_allclose(trajectory.velocities[:, 0], [0.5, 0.6])
        velocity_diag = trajectory.metadata.extra["topic_alignment"]["channels"]["velocity"]
        self.assertEqual(velocity_diag["status"], "aligned")
        self.assertEqual(velocity_diag["time_topic"]["status"], "aligned")

    async def test_mcap_loader_preserves_same_topic_torque_when_topic_is_omitted(self) -> None:
        """Preserve same-topic MCAP effort as torque when torque topic is omitted."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_states",
            command_topic="/joint_states",
        )
        messages = [
            (
                "/joint_states",
                1.0,
                {"stamp": 1.0, "position": [0.0], "velocity": [0.5], "effort": [1.2], "torque": [9.2]},
            ),
            (
                "/joint_states",
                1.1,
                {"stamp": 1.1, "position": [0.1], "velocity": [0.6], "effort": [1.3], "torque": [9.3]},
            ),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_same_topic_torque_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with _FakeMcapModulePatch(messages):
                trajectory = load_mcap_trajectory(config)

        np.testing.assert_allclose(trajectory.torques[:, 0], [1.2, 1.3])
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["channels"]["torque"]["status"], "aligned")

    async def test_mcap_loader_rejects_malformed_primary_json(self) -> None:
        """Reject malformed JSON on the required position topic."""
        mapping = TopicSignalMapping(
            position_topic="/joint_states",
            position_fields=["position"],
        )
        messages = [
            ("/joint_states", 1.0, {"position": [0.0]}),
            ("/joint_states", 1.1, b"{malformed"),
            ("/joint_states", 1.2, {"position": [0.2]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_malformed_primary_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(messages),
                self.assertRaisesRegex(
                    TrajectoryIngestError,
                    "position topic.*malformed JSON",
                ),
            ):
                load_mcap_trajectory(config)

    async def test_mcap_binary_retry_clears_stale_json_decode_failures(self) -> None:
        """Do not report failed JSON-pass diagnostics after binary decoding succeeds."""
        mapping = TopicSignalMapping(
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
        )
        json_messages = [
            ("/joint_states", 1.0, {"position": [0.0]}),
            ("/joint_velocity", 1.0, b"{malformed"),
        ]
        binary_messages = [
            ("/joint_states", 1.0, {"position": [0.0]}),
            ("/joint_velocity", 1.0, {"velocity": [0.5]}),
            ("/joint_states", 1.1, {"position": [0.1]}),
            ("/joint_velocity", 1.1, {"velocity": [0.6]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_binary_retry_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(json_messages),
                patch.object(mcap_loader, "_sniff_mcap_encoding", return_value="json"),
                patch.object(
                    mcap_loader,
                    "_iter_decoded_ros2_messages",
                    side_effect=lambda *_args: iter(binary_messages),
                ),
            ):
                trajectory = load_mcap_trajectory(config)

        self.assertNotIn("json_decode_failures", trajectory.metadata.extra)
        np.testing.assert_allclose(trajectory.velocities[:, 0], [0.5, 0.6])

    async def test_mcap_malformed_position_json_still_reaches_the_binary_retry(self) -> None:
        """Let a binary recording whose first message sniffed as JSON reach the retry."""
        mapping = TopicSignalMapping(
            position_topic="/joint_states",
            position_fields=["position"],
        )
        json_messages = [
            ("/joint_states", 1.0, b"{malformed"),
            ("/joint_states", 1.1, b"{malformed"),
        ]
        binary_messages = [
            ("/joint_states", 1.0, {"position": [0.0]}),
            ("/joint_states", 1.1, {"position": [0.1]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_json_then_binary_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(json_messages),
                patch.object(mcap_loader, "_sniff_mcap_encoding", return_value="json"),
                patch.object(
                    mcap_loader,
                    "_iter_decoded_ros2_messages",
                    side_effect=lambda *_args: iter(binary_messages),
                ),
            ):
                trajectory = load_mcap_trajectory(config)

        np.testing.assert_allclose(trajectory.positions[:, 0], [0.0, 0.1])
        self.assertNotIn("json_decode_failures", trajectory.metadata.extra)

    async def test_mcap_reports_malformed_position_json_after_binary_retry_fails(self) -> None:
        """Blame malformed JSON, not an empty file, once both decoders have been tried."""
        mapping = TopicSignalMapping(
            position_topic="/joint_states",
            position_fields=["position"],
        )
        json_messages = [
            ("/joint_states", 1.0, b"{malformed"),
            ("/joint_states", 1.1, b"{malformed"),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_json_and_binary_fail_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(json_messages),
                patch.object(mcap_loader, "_sniff_mcap_encoding", return_value="json"),
                patch.object(mcap_loader, "_iter_decoded_ros2_messages", side_effect=lambda *_args: iter([])),
                self.assertRaisesRegex(TrajectoryIngestError, "position topic.*malformed JSON"),
            ):
                load_mcap_trajectory(config)

    async def test_alignment_tolerance_scales_with_position_sample_rate(self) -> None:
        """Keep the nearest-neighbor window below one sample period on fast recordings."""
        fast = resolve_topic_alignment_tolerance(np.asarray([0.0, 0.002, 0.004]))
        slow = resolve_topic_alignment_tolerance(np.asarray([0.0, 0.1, 0.2]))
        overridden = resolve_topic_alignment_tolerance(np.asarray([0.0, 0.002, 0.004]), 0.05)

        self.assertAlmostEqual(fast, 0.001)
        self.assertAlmostEqual(slow, 0.02)
        self.assertAlmostEqual(overridden, 0.05)
        with self.assertRaisesRegex(TrajectoryIngestError, "alignment_tolerance_seconds"):
            resolve_topic_alignment_tolerance(np.asarray([0.0, 0.002]), 0.0)

    async def test_mcap_rejects_optional_sample_one_period_away_on_fast_recordings(self) -> None:
        """Reject a stale optional sample that a fixed 20 ms window would have accepted."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
        )
        messages = [
            ("/joint_states", 1.000, {"stamp": 1.000, "position": [0.0]}),
            ("/joint_states", 1.005, {"stamp": 1.005, "position": [0.1]}),
            ("/joint_states", 1.010, {"stamp": 1.010, "position": [0.2]}),
            ("/joint_velocity", 1.005, {"stamp": 1.005, "velocity": [0.5]}),
            ("/joint_velocity", 1.010, {"stamp": 1.010, "velocity": [0.6]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_fast_tolerance_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(messages),
                self.assertRaisesRegex(TrajectoryIngestError, "configured velocity channel"),
            ):
                load_mcap_trajectory(config)

    async def test_mcap_rejects_omitted_explicit_command_channel(self) -> None:
        """Fail when an explicitly mapped MCAP command channel cannot be aligned."""
        mapping = TopicSignalMapping(
            position_topic="/joint_states",
            position_fields=["position"],
            command_topic="/joint_command",
            command_fields=["command"],
        )
        messages = [
            ("/joint_states", 1.0, {"position": [0.0]}),
            ("/joint_command", 1.0, {"command": [0.2]}),
            ("/joint_states", 1.1, {"position": [0.1]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_missing_command_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(messages),
                self.assertRaisesRegex(TrajectoryIngestError, "configured command channel"),
            ):
                load_mcap_trajectory(config)

    async def test_mcap_rejects_time_topic_covering_only_part_of_the_recording(self) -> None:
        """Reject a position timebase spliced from two clocks instead of stretching it."""
        mapping = TopicSignalMapping(
            time_topic="/clock",
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        messages = [
            ("/joint_states", 10.0, {"position": [0.0]}),
            ("/joint_states", 10.1, {"position": [0.1]}),
            ("/joint_states", 10.2, {"position": [0.2]}),
            ("/joint_states", 10.3, {"position": [0.3]}),
            ("/clock", 10.2, {"stamp": 1.7e9}),
            ("/clock", 10.3, {"stamp": 1.7e9 + 0.1}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_partial_clock_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(messages),
                self.assertRaisesRegex(
                    TrajectoryIngestError,
                    r"time topic '/clock' covers only part of the recording",
                ),
            ):
                load_mcap_trajectory(config)

    async def test_mcap_rejects_mapped_velocity_topic_that_cannot_be_aligned(self) -> None:
        """Do not differentiate positions when a velocity topic was explicitly mapped."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
        )
        messages = [
            ("/joint_states", 1.0, {"stamp": 1.0, "position": [0.0]}),
            ("/joint_velocity", 1.0, {"stamp": 1.0, "velocity": [0.5]}),
            ("/joint_states", 1.1, {"stamp": 1.1, "position": [0.1]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_unalignable_velocity_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(messages),
                self.assertRaisesRegex(TrajectoryIngestError, "configured velocity channel"),
            ):
                load_mcap_trajectory(config)

    async def test_mcap_rejects_mapped_command_topic_with_dropped_messages(self) -> None:
        """Refuse to replay one surviving command sample across the whole trajectory."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            command_topic="/joint_command",
            command_fields=["command"],
        )
        messages = [
            ("/joint_states", 1.000, {"stamp": 1.000, "position": [0.0]}),
            ("/joint_states", 1.005, {"stamp": 1.005, "position": [0.1]}),
            ("/joint_states", 1.010, {"stamp": 1.010, "position": [0.2]}),
            ("/joint_command", 1.000, {"stamp": 1.000, "not_command": [0.2]}),
            ("/joint_command", 1.005, {"stamp": 1.005, "command": [0.25]}),
            ("/joint_command", 1.010, {"stamp": 1.010, "not_command": [0.3]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_dropped_command_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with (
                _FakeMcapModulePatch(messages),
                self.assertRaisesRegex(
                    TrajectoryIngestError,
                    r"command topic '/joint_command' had 2 message",
                ),
            ):
                load_mcap_trajectory(config)

    async def test_mcap_loader_collects_multiple_channels_from_one_optional_topic(self) -> None:
        """Collect every mapped field when command and torque share a topic."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            command_topic="/actuator_state",
            command_fields=["command"],
            torque_topic="/actuator_state",
            torque_fields=["effort"],
        )
        messages = [
            ("/joint_states", 1.0, {"stamp": 1.0, "position": [0.0]}),
            ("/actuator_state", 1.0, {"stamp": 1.0, "command": [0.2], "effort": [1.2]}),
            ("/joint_states", 1.1, {"stamp": 1.1, "position": [0.1]}),
            ("/actuator_state", 1.1, {"stamp": 1.1, "command": [0.3], "effort": [1.3]}),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_mcap_shared_optional_topic_") as tmp_dir:
            mcap_path = Path(tmp_dir) / "telemetry.mcap"
            mcap_path.write_bytes(b"fake")
            config = TrajectoryLoadConfig.mcap(str(mcap_path))
            config.topic_mapping = mapping
            with _FakeMcapModulePatch(messages):
                trajectory = load_mcap_trajectory(config)

        np.testing.assert_allclose(trajectory.commands[:, 0], [0.2, 0.3])
        np.testing.assert_allclose(trajectory.torques[:, 0], [1.2, 1.3])

    async def test_ros2_bag_uses_companion_mcap_before_sqlite_reader(self) -> None:
        """Prefer companion MCAP files before using the SQLite bag reader."""
        with tempfile.TemporaryDirectory(prefix="sysid_ros2_bag_") as tmp_dir:
            bag_dir = Path(tmp_dir) / "bag"
            bag_dir.mkdir()
            mcap_path = bag_dir / "recording.mcap"
            mcap_path.write_bytes(b"")

            self.assertEqual(_find_mcap_in_bag_dir(bag_dir), mcap_path)

    async def test_ros2_bag_reads_split_mcap_directory_as_one_bag(self) -> None:
        """Do not silently dispatch only the first part of a split MCAP bag."""
        messages_by_topic = {
            "/joint_states": [
                (100, {"stamp": 1.0, "position": [0.0]}),
                (200, {"stamp": 1.1, "position": [0.1]}),
            ],
        }
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        with tempfile.TemporaryDirectory(prefix="sysid_ros2_split_mcap_") as tmp_dir:
            bag_dir = Path(tmp_dir)
            (bag_dir / "recording_0.mcap").write_bytes(b"")
            (bag_dir / "recording_1.mcap").write_bytes(b"")
            _write_split_bag_metadata(bag_dir, ["recording_0.mcap", "recording_1.mcap"])
            config = TrajectoryLoadConfig.ros2_bag(str(bag_dir))
            config.topic_mapping = mapping

            self.assertIsNone(_find_mcap_in_bag_dir(bag_dir))
            with (
                patch.object(ros2_bag, "_any_reader", return_value=_FakeRos2Reader(messages_by_topic)) as any_reader,
                patch.object(
                    ros2_bag,
                    "load_mcap_trajectory",
                    side_effect=AssertionError("split bag must not use the single-file loader"),
                ),
            ):
                trajectory = load_ros2_bag_trajectory(config)

            any_reader.assert_called_once_with([bag_dir])
            np.testing.assert_allclose(trajectory.positions[:, 0], [0.0, 0.1])

    async def test_ros2_run_folder_reads_nested_split_mcap_bag(self) -> None:
        """Resolve a nested split bag without reducing it to one MCAP part."""
        messages_by_topic = {
            "/joint_states": [
                (100, {"stamp": 1.0, "position": [0.0]}),
                (200, {"stamp": 1.1, "position": [0.1]}),
            ],
        }
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        with tempfile.TemporaryDirectory(prefix="sysid_ros2_nested_split_mcap_") as tmp_dir:
            root = Path(tmp_dir)
            bag_dir = root / "bag"
            bag_dir.mkdir()
            (bag_dir / "recording_0.mcap").write_bytes(b"")
            (bag_dir / "recording_1.mcap").write_bytes(b"")
            _write_split_bag_metadata(bag_dir, ["recording_0.mcap", "recording_1.mcap"])
            config = TrajectoryLoadConfig.ros2_bag(str(root))
            config.topic_mapping = mapping

            with patch.object(
                ros2_bag,
                "_any_reader",
                return_value=_FakeRos2Reader(messages_by_topic),
            ) as any_reader:
                trajectory = load_ros2_bag_trajectory(config)

            any_reader.assert_called_once_with([bag_dir])
            self.assertEqual(trajectory.metadata.source_path, str(bag_dir))

    async def test_ros2_bag_rejects_undeclared_multiple_mcap_recordings(self) -> None:
        """Refuse to concatenate unrelated recordings that no bag index ties together."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        with tempfile.TemporaryDirectory(prefix="sysid_ros2_undeclared_mcap_") as tmp_dir:
            bag_dir = Path(tmp_dir)
            (bag_dir / "run_a_0.mcap").write_bytes(b"")
            (bag_dir / "run_b_0.mcap").write_bytes(b"")
            config = TrajectoryLoadConfig.ros2_bag(str(bag_dir))
            config.topic_mapping = mapping

            with (
                patch.object(
                    ros2_bag,
                    "_any_reader",
                    side_effect=AssertionError("undeclared recordings must not be merged"),
                ),
                self.assertRaisesRegex(TrajectoryIngestError, "no metadata.yaml declares them"),
            ):
                load_ros2_bag_trajectory(config)

    async def test_ros2_bag_rejects_split_metadata_that_omits_a_recording(self) -> None:
        """Require the bag index to account for every storage file before merging."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        with tempfile.TemporaryDirectory(prefix="sysid_ros2_partial_metadata_") as tmp_dir:
            bag_dir = Path(tmp_dir)
            (bag_dir / "recording_0.mcap").write_bytes(b"")
            (bag_dir / "stray.mcap").write_bytes(b"")
            _write_split_bag_metadata(bag_dir, ["recording_0.mcap"])
            config = TrajectoryLoadConfig.ros2_bag(str(bag_dir))
            config.topic_mapping = mapping

            with self.assertRaisesRegex(TrajectoryIngestError, "no metadata.yaml declares them"):
                load_ros2_bag_trajectory(config)

    async def test_ros2_bag_reports_synthesized_command_channel(self) -> None:
        """Do not report positions reused as commands as an aligned command topic."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        messages_by_topic = {
            "/joint_states": [
                (100, {"stamp": 1.0, "position": [0.0]}),
                (200, {"stamp": 1.1, "position": [0.1]}),
            ],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_synthesized_command_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            with patch.object(ros2_bag, "_any_reader", return_value=_FakeRos2Reader(messages_by_topic)):
                trajectory = load_ros2_bag_trajectory(config)

        alignment = trajectory.metadata.extra["topic_alignment"]
        self.assertEqual(alignment["channels"]["command"]["status"], "synthesized_from_positions")
        self.assertNotIn("command", alignment["omitted_channels"])
        np.testing.assert_allclose(trajectory.commands[:, 0], [0.0, 0.1])

    async def test_topic_mapping_file_rejects_unknown_fields(self) -> None:
        """Surface mapping typos instead of silently applying defaults."""
        with tempfile.TemporaryDirectory(prefix="sysid_bad_topic_mapping_") as tmp_dir:
            root = Path(tmp_dir)
            mcap_path = root / "telemetry.mcap"
            mcap_path.write_bytes(b"")
            mapping_path = root / "topic_map.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "position_topic": "/joint_states",
                        "torque_sematic": "external",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TrajectoryIngestError, r"\$\.torque_sematic.*Unknown field"):
                load_mcap_trajectory(TrajectoryLoadConfig.mcap(str(mcap_path), topic_mapping_path=str(mapping_path)))

    async def test_ros2_bag_requires_mapping_before_reader_dependency(self) -> None:
        """Require topic mapping before constructing optional ROS 2 readers."""
        with tempfile.TemporaryDirectory(prefix="sysid_ros2_mapping_") as tmp_dir:
            with self.assertRaisesRegex(TrajectoryIngestError, "topic mapping"):
                load_ros2_bag_trajectory(TrajectoryLoadConfig.ros2_bag(tmp_dir))

    async def test_ros2_bag_uses_mapped_time_topic_for_position_time_base(self) -> None:
        """Use mapped ROS 2 time topic as the position time base."""
        mapping = TopicSignalMapping(
            time_topic="/clock",
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
            command_topic="/joint_command",
            command_fields=["command"],
        )
        messages_by_topic = {
            "/clock": [(100, {"stamp": 1.0}), (200, {"stamp": 1.1})],
            "/joint_states": [
                (101, {"stamp": 9.0, "position": [0.0]}),
                (199, {"stamp": 9.1, "position": [0.1]}),
            ],
            "/joint_velocity": [(101, {"stamp": 1.004, "velocity": [0.5]}), (199, {"stamp": 1.096, "velocity": [0.6]})],
            "/joint_command": [(102, {"stamp": 1.003, "command": [0.2]}), (198, {"stamp": 1.097, "command": [0.3]})],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_time_topic_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        np.testing.assert_allclose(trajectory.times, [1.0, 1.1])
        np.testing.assert_allclose(trajectory.velocities[:, 0], [0.5, 0.6])
        np.testing.assert_allclose(trajectory.commands[:, 0], [0.2, 0.3])
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["channels"]["velocity"]["status"], "aligned")
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["channels"]["command"]["status"], "aligned")
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["time_topic"]["status"], "aligned")

    async def test_ros2_bag_reads_all_connections_for_the_position_topic(self) -> None:
        """Combine samples from every connection associated with a mapped topic."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        rows_by_connection = [
            (
                "/joint_states",
                [
                    (100, {"stamp": 1.0, "position": [0.0]}),
                    (200, {"stamp": 1.1, "position": [0.1]}),
                ],
            ),
            (
                "/joint_states",
                [
                    (300, {"stamp": 1.2, "position": [0.2]}),
                    (400, {"stamp": 1.3, "position": [0.3]}),
                ],
            ),
        ]

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_multi_connection_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2MultiConnectionReader(rows_by_connection)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        np.testing.assert_allclose(trajectory.times, [1.0, 1.1, 1.2, 1.3])
        np.testing.assert_allclose(trajectory.positions[:, 0], [0.0, 0.1, 0.2, 0.3])

    async def test_ros2_bag_aligns_optional_topics_to_time_topic_without_payload_stamps(self) -> None:
        """Align ROS 2 optional topics to a time topic without payload stamps."""
        mapping = TopicSignalMapping(
            time_topic="/clock",
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
            command_topic="/joint_command",
            command_fields=["command"],
            torque_topic="/joint_effort",
            torque_fields=["effort"],
        )
        messages_by_topic = {
            "/clock": [(100, {"stamp": 1.0}), (200, {"stamp": 1.1})],
            "/joint_states": [(100, {"position": [0.0]}), (200, {"position": [0.1]})],
            "/joint_velocity": [(100, {"velocity": [0.5]}), (200, {"velocity": [0.6]})],
            "/joint_command": [(100, {"command": [0.2]}), (200, {"command": [0.3]})],
            "/joint_effort": [(100, {"effort": [1.2]}), (200, {"effort": [1.3]})],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_time_topic_optional_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        np.testing.assert_allclose(trajectory.times, [1.0, 1.1])
        np.testing.assert_allclose(trajectory.velocities[:, 0], [0.5, 0.6])
        np.testing.assert_allclose(trajectory.commands[:, 0], [0.2, 0.3])
        np.testing.assert_allclose(trajectory.torques[:, 0], [1.2, 1.3])
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["channels"]["velocity"]["status"], "aligned")
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["channels"]["torque"]["status"], "aligned")

    async def test_ros2_time_topic_overrides_bad_local_time_fields_and_reports_remap(self) -> None:
        """Use the ROS 2 time topic when local payload time fields are invalid."""
        bad_stamp = {"sec": "not-a-number", "nanosec": 0}
        mapping = TopicSignalMapping(
            time_topic="/clock",
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
        )
        messages_by_topic = {
            "/clock": [(100, {"stamp": 1.0}), (200, {"stamp": 1.1})],
            "/joint_states": [
                (100, {"stamp": bad_stamp, "position": [0.0]}),
                (200, {"stamp": bad_stamp, "position": [0.1]}),
            ],
            "/joint_velocity": [
                (100, {"stamp": bad_stamp, "velocity": [0.5]}),
                (200, {"stamp": bad_stamp, "velocity": [0.6]}),
            ],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_bad_local_time_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        np.testing.assert_allclose(trajectory.times, [1.0, 1.1])
        np.testing.assert_allclose(trajectory.velocities[:, 0], [0.5, 0.6])
        velocity_diag = trajectory.metadata.extra["topic_alignment"]["channels"]["velocity"]
        self.assertEqual(velocity_diag["status"], "aligned")
        self.assertEqual(velocity_diag["time_topic"]["status"], "aligned")

    async def test_ros2_bag_missing_time_topic_falls_back_to_position_message_time(self) -> None:
        """Fall back to position message time when the mapped time topic is missing."""
        mapping = TopicSignalMapping(
            time_topic="/missing_clock",
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
        )
        messages_by_topic = {
            "/joint_states": [(100, {"stamp": 2.0, "position": [0.0]}), (200, {"stamp": 2.1, "position": [0.1]})],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_missing_time_topic_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        np.testing.assert_allclose(trajectory.times, [2.0, 2.1])
        diagnostic = trajectory.metadata.extra["topic_alignment"]["time_topic"]
        self.assertEqual(diagnostic["status"], "fallback")
        self.assertEqual(diagnostic["reason"], "no_samples")

    async def test_ros2_bag_malformed_requested_command_is_fatal(self) -> None:
        """Fail when an explicitly mapped ROS 2 command channel is incomplete."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_velocity",
            velocity_fields=["velocity"],
            command_topic="/joint_command",
            command_fields=["command"],
        )
        messages_by_topic = {
            "/joint_states": [
                (100, {"stamp": 1.0, "position": [0.0]}),
                (200, {"stamp": 1.1, "position": [0.1]}),
            ],
            "/joint_velocity": [
                (100, {"stamp": 1.0, "velocity": [0.5]}),
                (200, {"stamp": 1.1, "velocity": [0.6]}),
            ],
            "/joint_command": [
                (100, {"stamp": 1.0, "command": [0.2]}),
                (200, {"stamp": 1.1, "not_command": [0.3]}),
            ],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_bad_optional_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                with self.assertRaisesRegex(
                    TrajectoryIngestError,
                    r"command topic '/joint_command' had 1 message",
                ):
                    load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

    async def test_ros2_bag_malformed_optional_pose_is_omitted_not_fatal(self) -> None:
        """Omit malformed optional ROS 2 pose channels without failing ingestion."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            end_effector_pose_topic="/ee_pose",
            end_effector_pose_fields=["pose"],
        )
        messages_by_topic = {
            "/joint_states": [
                (100, {"stamp": 1.0, "position": [0.0]}),
                (200, {"stamp": 1.1, "position": [0.1]}),
            ],
            "/ee_pose": [
                (100, {"stamp": 1.0, "pose": {"position": {"x": "bad", "y": 0.0, "z": 0.0}}}),
                (200, {"stamp": 1.1, "pose": {"position": {"x": 0.1, "y": 0.0, "z": 0.0}}}),
            ],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_bad_optional_pose_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        self.assertIsNone(trajectory.end_effector_poses)
        self.assertEqual(
            trajectory.metadata.extra["topic_alignment"]["channels"]["end_effector_pose"]["status"],
            "omitted",
        )

    async def test_ros2_bag_preserves_same_topic_torque_when_topic_is_omitted(self) -> None:
        """Preserve same-topic ROS 2 effort as torque when torque topic is omitted."""
        mapping = TopicSignalMapping(
            time_field="stamp",
            position_topic="/joint_states",
            position_fields=["position"],
            velocity_topic="/joint_states",
            command_topic="/joint_states",
        )
        messages_by_topic = {
            "/joint_states": [
                (100, {"stamp": 1.0, "position": [0.0], "velocity": [0.5], "effort": [1.2]}),
                (200, {"stamp": 1.1, "position": [0.1], "velocity": [0.6], "effort": [1.3]}),
            ],
        }

        with tempfile.TemporaryDirectory(prefix="sysid_ros2_same_topic_torque_") as tmp_dir:
            config = TrajectoryLoadConfig.ros2_bag(tmp_dir)
            config.topic_mapping = mapping
            original_any_reader = ros2_bag._any_reader
            ros2_bag._any_reader = lambda paths: _FakeRos2Reader(messages_by_topic)
            try:
                trajectory = load_ros2_bag_trajectory(config)
            finally:
                ros2_bag._any_reader = original_any_reader

        np.testing.assert_allclose(trajectory.torques[:, 0], [1.2, 1.3])
        self.assertEqual(trajectory.metadata.extra["topic_alignment"]["channels"]["torque"]["status"], "aligned")

    async def test_lerobot_local_directory_reports_missing_dataset_metadata(self) -> None:
        """Report missing LeRobot dataset metadata for local directories."""
        with tempfile.TemporaryDirectory(prefix="sysid_lerobot_") as tmp_dir:
            root = Path(tmp_dir)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(json.dumps({"fps": 30.0}), encoding="utf-8")

            with self.assertRaisesRegex(TrajectoryIngestError, "No parquet files"):
                load_lerobot_trajectory(TrajectoryLoadConfig.lerobot(tmp_dir))

    async def test_lerobot_v2_reads_selected_episode_file(self) -> None:
        """Read an episode-per-file LeRobot v2 layout without pandas or LeRobot."""
        with tempfile.TemporaryDirectory(prefix="sysid_lerobot_v2_") as tmp_dir:
            root = Path(tmp_dir)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"fps": 20.0, "codebase_version": "v2.1", "license": "Apache-2.0"}),
                encoding="utf-8",
            )
            table = _FakeArrowTable(
                {
                    "episode_index": [1, 1],
                    "frame_index": [0, 1],
                    "observation.state": [[1.0, 1.0], [1.1, 1.1]],
                    "action": [[0.2, 0.3], [0.4, 0.5]],
                }
            )
            with (
                patch.object(lerobot_loader, "_find_parquet_files", return_value=[root / "episode.parquet"]),
                patch.object(lerobot_loader, "_load_pyarrow_modules", return_value=(object(), _FakeArrowTypes())),
                patch.object(lerobot_loader, "_read_episode_table", return_value=table),
            ):
                trajectory = load_lerobot_trajectory(TrajectoryLoadConfig.lerobot(tmp_dir, episode_index=1))

        np.testing.assert_allclose(trajectory.times, [0.0, 0.05])
        np.testing.assert_allclose(trajectory.positions, [[1.0, 1.0], [1.1, 1.1]])
        np.testing.assert_allclose(trajectory.commands, [[0.2, 0.3], [0.4, 0.5]])
        self.assertEqual(trajectory.metadata.extra["loader"], "local_pyarrow")
        self.assertEqual(trajectory.metadata.extra["time_source"], "frame_index/fps")
        self.assertEqual(trajectory.metadata.extra["codebase_version"], "v2.1")

    async def test_lerobot_v3_filters_episode_from_shared_shard(self) -> None:
        """Filter one episode from a LeRobot v3 shard containing multiple episodes."""
        with tempfile.TemporaryDirectory(prefix="sysid_lerobot_v3_") as tmp_dir:
            root = Path(tmp_dir)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"fps": 30.0, "codebase_version": "v3.0"}),
                encoding="utf-8",
            )
            table = _FakeArrowTable(
                {
                    "episode_index": [7, 7],
                    "frame_index": [0, 1],
                    "timestamp": [4.0, 4.1],
                    "observation.state": [[7.0], [7.1]],
                    "observation.velocity": [[2.0], [2.0]],
                    "observation.effort": [[0.7], [0.8]],
                    "action": [[7.1], [7.2]],
                }
            )
            with (
                patch.object(lerobot_loader, "_find_parquet_files", return_value=[root / "shard.parquet"]),
                patch.object(lerobot_loader, "_load_pyarrow_modules", return_value=(object(), _FakeArrowTypes())),
                patch.object(lerobot_loader, "_read_episode_table", return_value=table),
            ):
                trajectory = load_lerobot_trajectory(TrajectoryLoadConfig.lerobot(tmp_dir, episode_index=7))

        np.testing.assert_allclose(trajectory.times, [4.0, 4.1])
        np.testing.assert_allclose(trajectory.positions[:, 0], [7.0, 7.1])
        np.testing.assert_allclose(trajectory.velocities[:, 0], [2.0, 2.0])
        np.testing.assert_allclose(trajectory.torques[:, 0], [0.7, 0.8])
        self.assertEqual(trajectory.metadata.extra["time_source"], "timestamp")

    async def test_lerobot_preflight_requires_complete_local_dataset(self) -> None:
        """Surface an incomplete local LeRobot source before ingestion."""
        from isaacsim.robot_setup.sysid.preflight import preflight_sysid_run_spec
        from isaacsim.robot_setup.sysid.run_spec import SysIdRunSpec

        spec = SysIdRunSpec()
        spec.telemetry.source_type = "lerobot"

        with tempfile.TemporaryDirectory(prefix="sysid_lerobot_preflight_") as tmp_dir:
            spec.telemetry.source_path = tmp_dir
            local_result = preflight_sysid_run_spec(spec, allow_empty_parameters=True)
        self.assertIn("lerobot_info_missing", {issue.code for issue in local_result.issues})

    async def test_lerobot_rejects_parquet_paths_outside_data_root(self) -> None:
        """Reject discovered parquet files whose resolved path escapes the dataset."""
        with tempfile.TemporaryDirectory(prefix="sysid_lerobot_escape_") as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            data_root = root / "data"
            data_root.mkdir(parents=True)
            outside = Path(tmp_dir) / "outside.parquet"
            outside.write_bytes(b"not parquet")

            with (
                patch.object(Path, "glob", return_value=[outside]),
                self.assertRaisesRegex(TrajectoryIngestError, "escapes the dataset data directory"),
            ):
                lerobot_loader._find_parquet_files(root)

    async def test_lerobot_reports_missing_pyarrow_without_relabelling_internal_failures(self) -> None:
        """Report only a genuinely absent PyArrow package as an optional dependency."""
        missing_pyarrow = ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
        with patch.object(lerobot_loader.importlib, "import_module", side_effect=missing_pyarrow):
            with self.assertRaisesRegex(TrajectoryIngestError, "requires PyArrow"):
                lerobot_loader._load_pyarrow_modules()

        internal_failure = ModuleNotFoundError("No module named 'arrow_internal'", name="arrow_internal")
        with patch.object(lerobot_loader.importlib, "import_module", side_effect=internal_failure):
            with self.assertRaises(ModuleNotFoundError) as caught:
                lerobot_loader._load_pyarrow_modules()
        self.assertEqual(caught.exception.name, "arrow_internal")

    async def test_lerobot_requires_a_recognized_position_signal(self) -> None:
        """Do not infer joint positions from arbitrary numeric bookkeeping columns."""
        with tempfile.TemporaryDirectory(prefix="sysid_lerobot_signals_") as tmp_dir:
            root = Path(tmp_dir)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(json.dumps({"fps": 30.0}), encoding="utf-8")
            table = _FakeArrowTable(
                {
                    "episode_index": [0, 0],
                    "frame_index": [0, 1],
                    "reward": [1.0, 2.0],
                }
            )

            with (
                patch.object(lerobot_loader, "_find_parquet_files", return_value=[root / "shard.parquet"]),
                patch.object(lerobot_loader, "_load_pyarrow_modules", return_value=(object(), _FakeArrowTypes())),
                patch.object(lerobot_loader, "_read_episode_table", return_value=table),
                self.assertRaisesRegex(TrajectoryIngestError, "no supported joint-position column"),
            ):
                load_lerobot_trajectory(TrajectoryLoadConfig.lerobot(tmp_dir))

    async def test_mapping_dataclasses_coerce_payloads(self) -> None:
        """Coerce mapping dataclass payloads to list-backed fields."""
        column_mapping = CsvColumnMapping.from_dict({"position_columns": ("q1",), "velocity_columns": ["dq1"]})
        topic_mapping = TopicSignalMapping.from_dict({"position_topic": "/joint", "joint_names": ["a", "b"]})

        self.assertEqual(column_mapping.position_columns, ["q1"])
        self.assertEqual(topic_mapping.position_topic, "/joint")
        self.assertEqual(topic_mapping.joint_names, ["a", "b"])

    async def test_csv_loader_reports_named_columns_without_header(self) -> None:
        """Reject named CSV column mappings when the CSV has no header."""
        with tempfile.TemporaryDirectory(prefix="sysid_csv_no_header_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text("0.0,0.0\n0.1,0.1\n", encoding="utf-8")
            config = TrajectoryLoadConfig.csv(str(csv_path))
            config.column_mapping = CsvColumnMapping(position_columns=["q1"])

            with self.assertRaisesRegex(TrajectoryIngestError, "requires a CSV header"):
                load_csv_trajectory(config)

    async def test_csv_loader_rejects_non_increasing_times(self) -> None:
        """Reject CSV files with non-increasing timestamps."""
        with tempfile.TemporaryDirectory(prefix="sysid_csv_bad_time_") as tmp_dir:
            csv_path = Path(tmp_dir) / "telemetry.csv"
            csv_path.write_text("time,q1\n0.0,0.0\n0.0,0.1\n", encoding="utf-8")

            with self.assertRaises(TrajectoryCsvError):
                load_sysid_trajectory_csv(str(csv_path))
