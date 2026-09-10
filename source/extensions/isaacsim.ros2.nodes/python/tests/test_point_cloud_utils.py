# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validate the C++ point cloud interleave binding and the PointCloud2 helpers."""

import array

import numpy as np
import omni.kit.test
from isaacsim.ros2.core.impl.ros2_test_case import ROS2TestCase
from isaacsim.ros2.nodes import interleave_point_cloud
from isaacsim.ros2.nodes.bindings import _ros2_nodes
from isaacsim.ros2.nodes.impl.point_cloud_utils import fill_point_cloud2_message


def _numpy_reference(xyz: np.ndarray, fields, point_step: int) -> np.ndarray:
    """Interleave with plain numpy as the ground truth; untouched bytes stay zero."""
    num_points = xyz.shape[0]
    out = np.zeros(num_points * point_step, dtype=np.uint8)
    view = out.reshape(num_points, point_step)
    view[:, 0:12] = np.ascontiguousarray(xyz, dtype=np.float32).reshape(num_points, 3).view(np.uint8)
    for field, offset in fields:
        field_bytes = np.ascontiguousarray(field).reshape(num_points, -1).view(np.uint8)
        view[:, offset : offset + field_bytes.shape[1]] = field_bytes
    return out


class TestFillPointCloudBuffer(omni.kit.test.AsyncTestCase):
    """Exercise the _ros2_nodes.fill_point_cloud_buffer binding and interleave_point_cloud."""

    async def test_binding_xyz_only(self) -> None:
        """Interleave xyz with the default 12-byte point_step and compare to numpy."""
        rng = np.random.default_rng(7)
        xyz = rng.standard_normal((257, 3), dtype=np.float32)

        destination = np.zeros(257 * 12, dtype=np.uint8)
        num_points = _ros2_nodes.fill_point_cloud_buffer(destination, xyz)

        self.assertEqual(num_points, 257)
        np.testing.assert_array_equal(destination, _numpy_reference(xyz, [], 12))

    async def test_binding_mixed_fields_and_padding(self) -> None:
        """Interleave mixed-dtype fields into a padded layout and compare to numpy."""
        rng = np.random.default_rng(11)
        num_points = 1000
        xyz = rng.standard_normal((num_points, 3), dtype=np.float32)
        intensity = rng.standard_normal(num_points, dtype=np.float32)
        timestamp = rng.integers(0, 2**63, num_points, dtype=np.uint64)
        channel = rng.integers(0, 128, num_points, dtype=np.uint32)
        # 4 bytes of padding between channel (ends at 28) and the end of the point
        fields = [(intensity, 12), (timestamp, 16), (channel, 24)]
        point_step = 32

        destination = np.zeros(num_points * point_step, dtype=np.uint8)
        _ros2_nodes.fill_point_cloud_buffer(destination, xyz, fields, point_step)

        np.testing.assert_array_equal(destination, _numpy_reference(xyz, fields, point_step))

    async def test_binding_zero_points(self) -> None:
        """An empty cloud is a no-op that reports zero points but still validates the layout."""
        destination = np.zeros(0, dtype=np.uint8)
        empty_xyz = np.empty((0, 3), dtype=np.float32)
        num_points = _ros2_nodes.fill_point_cloud_buffer(destination, empty_xyz)
        self.assertEqual(num_points, 0)

        # Empty fields are accepted; layout errors are still rejected
        empty_field = np.empty(0, dtype=np.float32)
        num_points = _ros2_nodes.fill_point_cloud_buffer(destination, empty_xyz, [(empty_field, 12)], 16)
        self.assertEqual(num_points, 0)
        with self.assertRaises(ValueError):  # offset overlapping xyz is invalid even with zero points
            _ros2_nodes.fill_point_cloud_buffer(destination, empty_xyz, [(empty_field, 4)], 16)
        with self.assertRaises(ValueError):  # nonempty field data with zero points is inconsistent
            _ros2_nodes.fill_point_cloud_buffer(destination, empty_xyz, [(np.zeros(3, dtype=np.float32), 12)], 16)

    async def test_binding_accepts_zero_width_field(self) -> None:
        """A field with a zero-length trailing dimension contributes no bytes but is valid."""
        xyz = np.zeros((10, 3), dtype=np.float32)
        destination = np.zeros(10 * 16, dtype=np.uint8)
        num_points = _ros2_nodes.fill_point_cloud_buffer(
            destination, xyz, [(np.zeros((10, 0), dtype=np.float32), 12)], 16
        )
        self.assertEqual(num_points, 10)

    async def test_binding_rejects_invalid_inputs(self) -> None:
        """Every documented ValueError condition is enforced."""
        xyz = np.zeros((10, 3), dtype=np.float32)
        intensity = np.zeros(10, dtype=np.float32)
        destination = np.zeros(10 * 16, dtype=np.uint8)

        with self.assertRaises(ValueError):  # xyz not float32 (wrong itemsize)
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz.astype(np.float64), [], 16)
        with self.assertRaises(ValueError):  # xyz not float32 (4-byte integer dtype)
            _ros2_nodes.fill_point_cloud_buffer(destination, np.zeros((10, 3), dtype=np.int32), [], 16)
        with self.assertRaises(ValueError):  # xyz not float32 (big-endian floats)
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz.astype(">f4"), [], 16)
        with self.assertRaises(ValueError):  # xyz not a multiple of 3 floats
            _ros2_nodes.fill_point_cloud_buffer(destination, np.zeros(10, dtype=np.float32), [], 16)
        with self.assertRaises(ValueError):  # destination too small
            _ros2_nodes.fill_point_cloud_buffer(np.zeros(10, dtype=np.uint8), xyz, [], 16)
        with self.assertRaises(ValueError):  # destination not C-contiguous
            _ros2_nodes.fill_point_cloud_buffer(np.zeros(10 * 32, dtype=np.uint8)[::2], xyz, [], 16)
        with self.assertRaises(ValueError):  # point_step smaller than xyz
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz, [], 8)
        with self.assertRaises(ValueError):  # field overlaps xyz bytes
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz, [(intensity, 8)], 16)
        with self.assertRaises(ValueError):  # field does not fit within point_step
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz, [(intensity, 14)], 16)
        with self.assertRaises(ValueError):  # field size not a multiple of the point count
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz, [(np.zeros(11, dtype=np.float32), 12)], 16)

    async def test_binding_rejects_overlapping_buffers(self) -> None:
        """Sources aliasing the destination are rejected instead of racing the parallel fill."""
        buf = np.zeros(10 * 12, dtype=np.uint8)
        aliased_xyz = buf.view(np.float32).reshape(10, 3)
        with self.assertRaises(ValueError):  # xyz overlaps destination
            _ros2_nodes.fill_point_cloud_buffer(buf, aliased_xyz)

        xyz = np.zeros((10, 3), dtype=np.float32)
        destination = np.zeros(10 * 16, dtype=np.uint8)
        aliased_field = destination[: 10 * 4].view(np.float32)
        with self.assertRaises(ValueError):  # field overlaps destination
            _ros2_nodes.fill_point_cloud_buffer(destination, xyz, [(aliased_field, 12)], 16)

    async def test_interleave_point_cloud(self) -> None:
        """The helper packs fields in order, computes descriptions, and fills a destination in place."""
        rng = np.random.default_rng(3)
        num_points = 123
        xyz = rng.standard_normal((num_points, 3), dtype=np.float32)
        intensity = rng.standard_normal(num_points, dtype=np.float32)
        normals = rng.standard_normal((num_points, 3), dtype=np.float32)
        extra_fields = [("intensity", intensity), ("normal", normals)]

        data, point_step, descriptions = interleave_point_cloud(xyz, extra_fields)

        self.assertEqual(point_step, 12 + 4 + 12)
        self.assertEqual(
            [(d.name, d.offset, d.datatype, d.count) for d in descriptions],
            [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("intensity", 12, 7, 1), ("normal", 16, 7, 3)],
        )
        reference = _numpy_reference(xyz, [(intensity, 12), (normals, 16)], point_step)
        np.testing.assert_array_equal(data, reference)

        # In-place fill of a preallocated destination returns the same buffer
        preallocated = np.zeros(num_points * point_step, dtype=np.uint8)
        result, _, _ = interleave_point_cloud(xyz, extra_fields, destination=preallocated)
        self.assertIs(result, preallocated)
        np.testing.assert_array_equal(preallocated, reference)

    async def test_interleave_point_cloud_rejects_invalid_fields(self) -> None:
        """Unsupported dtypes, wrong shapes, and mismatched point counts raise ValueError."""
        xyz = np.zeros((10, 3), dtype=np.float32)

        with self.assertRaises(ValueError):  # wrong xyz shape
            interleave_point_cloud(np.zeros((10, 4), dtype=np.float32))
        with self.assertRaises(ValueError):  # dtype with no PointField equivalent
            interleave_point_cloud(xyz, [("bad", np.zeros(10, dtype=np.float16))])
        with self.assertRaises(ValueError):  # mismatched point count
            interleave_point_cloud(xyz, [("intensity", np.zeros(9, dtype=np.float32))])
        with self.assertRaises(ValueError):  # field name duplicating a position field
            interleave_point_cloud(xyz, [("x", np.zeros(10, dtype=np.float32))])
        with self.assertRaises(ValueError):  # field name duplicating another field
            interleave_point_cloud(
                xyz, [("intensity", np.zeros(10, dtype=np.float32)), ("intensity", np.zeros(10, dtype=np.float32))]
            )


class TestFillPointCloud2Message(ROS2TestCase):
    """Fill a real sensor_msgs/PointCloud2 through the helper."""

    async def test_fill_point_cloud2_message(self) -> None:
        """The message layout, fields, and data all match the numpy reference."""
        from sensor_msgs.msg import PointCloud2, PointField

        rng = np.random.default_rng(5)
        num_points = 512
        xyz = rng.standard_normal((num_points, 3), dtype=np.float32)
        intensity = rng.standard_normal(num_points, dtype=np.float32)

        message = PointCloud2()
        fill_point_cloud2_message(message, xyz, [("intensity", intensity)])

        self.assertEqual(message.height, 1)
        self.assertEqual(message.width, num_points)
        self.assertEqual(message.point_step, 16)
        self.assertEqual(message.row_step, 16 * num_points)
        self.assertFalse(message.is_bigendian)
        self.assertEqual(
            [(f.name, f.offset, f.datatype, f.count) for f in message.fields],
            [
                ("x", 0, PointField.FLOAT32, 1),
                ("y", 4, PointField.FLOAT32, 1),
                ("z", 8, PointField.FLOAT32, 1),
                ("intensity", 12, PointField.FLOAT32, 1),
            ],
        )
        np.testing.assert_array_equal(
            np.frombuffer(message.data, dtype=np.uint8), _numpy_reference(xyz, [(intensity, 12)], 16)
        )

        # Round-trip through a structured view recovers the original arrays
        structured = np.frombuffer(
            message.data, dtype=np.dtype({"names": ["x", "y", "z", "i"], "formats": ["<f4"] * 4})
        )
        np.testing.assert_array_equal(np.column_stack((structured["x"], structured["y"], structured["z"])), xyz)
        np.testing.assert_array_equal(structured["i"], intensity)

        # Refilling reuses the message's own data storage in place (works on every rclpy
        # distribution, including those whose data setter copies on assignment)
        first_buffer = message.data
        self.assertIsInstance(first_buffer, array.array)
        fill_point_cloud2_message(message, xyz * 2.0, [("intensity", intensity)])
        self.assertIs(message.data, first_buffer)
        np.testing.assert_array_equal(
            np.frombuffer(message.data, dtype=np.uint8), _numpy_reference(xyz * 2.0, [(intensity, 12)], 16)
        )

        # A different point count resizes the same storage in place (release the numpy
        # view first: array.array refuses to resize while a buffer export is alive)
        del structured
        fill_point_cloud2_message(message, xyz[:100], [("intensity", intensity[:100])])
        self.assertIs(message.data, first_buffer)
        self.assertEqual(message.width, 100)
        self.assertEqual(len(message.data), 100 * 16)
        np.testing.assert_array_equal(
            np.frombuffer(message.data, dtype=np.uint8), _numpy_reference(xyz[:100], [(intensity[:100], 12)], 16)
        )

        # If a live view pins the storage across a size change, fresh storage is used instead
        pinned_view = np.frombuffer(message.data, dtype=np.uint8)
        fill_point_cloud2_message(message, xyz[:50], [("intensity", intensity[:50])])
        self.assertEqual(message.width, 50)
        self.assertEqual(len(message.data), 50 * 16)
        np.testing.assert_array_equal(
            np.frombuffer(message.data, dtype=np.uint8), _numpy_reference(xyz[:50], [(intensity[:50], 12)], 16)
        )
        del pinned_view
