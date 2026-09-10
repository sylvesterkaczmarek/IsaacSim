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

"""Utilities for interleaving separate per-point arrays into ``sensor_msgs/PointCloud2`` buffers.

Point cloud data typically arrives as separate per-field host arrays (x/y/z positions,
intensity, timestamps, ...), while ``PointCloud2.data`` needs them gathered into a packed
``point_step`` layout. These helpers hand the gather to the parallel C++ interleave
(``_ros2_nodes.fill_point_cloud_buffer``) instead of packing with numpy in Python.
"""

from __future__ import annotations

import array
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from isaacsim.ros2.nodes.bindings import _ros2_nodes

if TYPE_CHECKING:
    from sensor_msgs.msg import PointCloud2

__all__ = ["PointFieldDescription", "fill_point_cloud2_message", "interleave_point_cloud"]

# Number of bytes the xyz position occupies at the start of every point (3 x float32).
_XYZ_BYTES = 12


@functools.lru_cache(maxsize=1)
def _point_field_datatype_codes() -> dict[np.dtype, int]:
    """numpy dtype -> ``sensor_msgs/PointField`` datatype code.

    Resolved lazily (and cached) so importing this module never requires a ROS 2
    environment; ``sensor_msgs`` is available by the time the helpers are called in any
    session where the ROS 2 bridge has been set up.
    """
    try:
        from sensor_msgs.msg import PointField
    except ImportError as e:
        raise ImportError(
            "sensor_msgs is not importable because no ROS 2 environment is set up. Run inside an "
            "Isaac Sim session with the ROS 2 bridge enabled, or source a ROS 2 installation whose "
            "Python version matches Kit's."
        ) from e

    return {
        np.dtype(np.int8): int(PointField.INT8),
        np.dtype(np.uint8): int(PointField.UINT8),
        np.dtype(np.int16): int(PointField.INT16),
        np.dtype(np.uint16): int(PointField.UINT16),
        np.dtype(np.int32): int(PointField.INT32),
        np.dtype(np.uint32): int(PointField.UINT32),
        np.dtype(np.float32): int(PointField.FLOAT32),
        np.dtype(np.float64): int(PointField.FLOAT64),
    }


class PointFieldDescription(NamedTuple):
    """Layout of a single field within an interleaved point cloud buffer.

    Mirrors ``sensor_msgs/PointField`` so it can be converted directly into one.
    """

    #: Field name (e.g. ``"x"`` or ``"intensity"``).
    name: str
    #: Byte offset of the field within a point.
    offset: int
    #: ``sensor_msgs/PointField`` datatype code.
    datatype: int
    #: Number of elements of ``datatype`` per point.
    count: int


def _compute_layout(
    xyz: np.ndarray, extra_fields: Sequence[tuple[str, np.ndarray]]
) -> tuple[np.ndarray, list, int, list]:
    """Validate inputs and compute the packed point layout.

    Args:
        xyz: Per-point positions, shape ``(num_points, 3)``. Converted to contiguous float32.
        extra_fields: Extra per-point fields as ``(name, array)`` pairs.

    Returns:
        A tuple of (contiguous float32 xyz array, ``(array, offset)`` pairs for the C++
        interleave, point step in bytes, list of :class:`PointFieldDescription` covering
        x/y/z and every extra field).

    Raises:
        ValueError: If ``xyz`` is not ``(num_points, 3)``-shaped, or a field has an
            unsupported dtype, a first dimension that does not match ``num_points``, or a
            name that duplicates ``x``/``y``/``z`` or another field.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (num_points, 3), got {xyz.shape}")
    num_points = xyz.shape[0]

    datatype_codes = _point_field_datatype_codes()
    float32_code = datatype_codes[np.dtype(np.float32)]
    descriptions = [
        PointFieldDescription(name="x", offset=0, datatype=float32_code, count=1),
        PointFieldDescription(name="y", offset=4, datatype=float32_code, count=1),
        PointFieldDescription(name="z", offset=8, datatype=float32_code, count=1),
    ]
    raw_fields = []
    seen_names = {"x", "y", "z"}
    offset = _XYZ_BYTES
    for name, field in extra_fields:
        if name in seen_names:
            raise ValueError(f"Duplicate field name '{name}' ('x', 'y', 'z' are reserved for the positions)")
        seen_names.add(name)
        field = np.ascontiguousarray(field)
        datatype = datatype_codes.get(field.dtype)
        if datatype is None:
            raise ValueError(f"Field '{name}' has dtype {field.dtype}, which has no sensor_msgs/PointField equivalent")
        if field.ndim == 0 or field.shape[0] != num_points:
            raise ValueError(
                f"Field '{name}' must have {num_points} entries along its first dimension, got shape {field.shape}"
            )
        count = int(np.prod(field.shape[1:], dtype=np.int64))
        descriptions.append(PointFieldDescription(name=name, offset=offset, datatype=datatype, count=count))
        raw_fields.append((field, offset))
        offset += field.itemsize * count

    return xyz, raw_fields, offset, descriptions


def interleave_point_cloud(
    xyz: np.ndarray,
    extra_fields: Sequence[tuple[str, np.ndarray]] = (),
    destination: np.ndarray | array.array | None = None,
) -> tuple[np.ndarray | array.array, int, list]:
    """Interleave xyz + per-point arrays into a packed ``PointCloud2``-style byte buffer.

    Fields are packed in order with no padding: x/y/z occupy bytes ``[0, 12)`` of every
    point and each extra field follows the previous one. The gather runs in parallel in
    C++ with the GIL released.

    Args:
        xyz: Per-point positions, shape ``(num_points, 3)``. Converted to float32 if
            needed — pass a C-contiguous float32 array to avoid a per-call conversion copy.
        extra_fields: Extra per-point fields as ``(name, array)`` pairs. Each array must
            have ``num_points`` entries along its first dimension; trailing dimensions
            become the ``PointField`` count (e.g. ``(num_points, 3)`` normals). Supported
            dtypes are the ``sensor_msgs/PointField`` scalar types (int8 ... float64).
        destination: Optional preallocated writable buffer of at least
            ``num_points * point_step`` bytes to fill in place. Allocated when omitted.
            Must not overlap the source arrays.

    Returns:
        A tuple of (filled buffer, point step in bytes, list of
        :class:`PointFieldDescription` covering x/y/z and every extra field).

    Raises:
        ValueError: If inputs are inconsistent (see :func:`fill_point_cloud2_message`)
            or ``destination`` is too small.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.ros2.nodes import interleave_point_cloud
        >>>
        >>> xyz = np.random.rand(100, 3).astype(np.float32)
        >>> intensity = np.random.rand(100).astype(np.float32)
        >>> data, point_step, fields = interleave_point_cloud(xyz, [("intensity", intensity)])
        >>> point_step
        16
    """
    xyz, raw_fields, point_step, descriptions = _compute_layout(xyz, extra_fields)
    if destination is None:
        destination = np.empty(xyz.shape[0] * point_step, dtype=np.uint8)
    _ros2_nodes.fill_point_cloud_buffer(destination, xyz, raw_fields, point_step)
    return destination, point_step, descriptions


def fill_point_cloud2_message(
    message: PointCloud2, xyz: np.ndarray, extra_fields: Sequence[tuple[str, np.ndarray]] = ()
) -> None:
    """Populate a ``sensor_msgs/PointCloud2`` message in place from separate per-point arrays.

    Sets ``fields``, ``width``/``height`` (organized as a single row), ``point_step``,
    ``row_step``, ``is_bigendian``, and ``data``. The header and ``is_dense`` are left
    untouched for the caller to manage. The message's existing ``data`` storage is resized
    and filled in place, so repeated calls on the same message avoid reallocating and
    avoid the copying ``data`` setter of some rclpy distributions. When a live view
    (e.g. a ``numpy.frombuffer`` of ``message.data``) pins the storage across a size
    change, fresh storage is allocated instead.

    Args:
        message: ``sensor_msgs.msg.PointCloud2`` instance to fill.
        xyz: Per-point positions, shape ``(num_points, 3)``. Converted to float32 if
            needed — pass a C-contiguous float32 array to avoid a per-call conversion copy.
        extra_fields: Extra per-point fields as ``(name, array)`` pairs; see
            :func:`interleave_point_cloud`.

    Raises:
        ValueError: If ``xyz`` or a field array is inconsistent (wrong shape, unsupported
            dtype, mismatched point count, or duplicate field name).

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.ros2.nodes import fill_point_cloud2_message
        >>> from sensor_msgs.msg import PointCloud2
        >>>
        >>> message = PointCloud2()
        >>> xyz = np.random.rand(100, 3).astype(np.float32)
        >>> fill_point_cloud2_message(message, xyz, [("intensity", np.random.rand(100).astype(np.float32))])
        >>> message.width, message.point_step
        (100, 16)
    """
    from sensor_msgs.msg import PointField

    xyz, raw_fields, point_step, descriptions = _compute_layout(xyz, extra_fields)
    num_points = xyz.shape[0]
    total_bytes = num_points * point_step

    # rosidl stores uint8[] data as array.array('B'); resize it in place instead of
    # assigning a new buffer, since the data setter copies on some rclpy distributions
    # (Jazzy) and an assignment would defeat the storage reuse.
    data = message.data
    try:
        if len(data) > total_bytes:
            del data[total_bytes:]
        elif len(data) < total_bytes:
            data.frombytes(bytes(total_bytes - len(data)))
    except BufferError:
        # A live memoryview/numpy view pins the current storage; fall back to fresh storage
        data = array.array("B", bytes(total_bytes))
    _ros2_nodes.fill_point_cloud_buffer(data, xyz, raw_fields, point_step)

    message.height = 1
    message.width = num_points
    message.fields = [
        PointField(name=field.name, offset=field.offset, datatype=field.datatype, count=field.count)
        for field in descriptions
    ]
    message.is_bigendian = False
    message.point_step = point_step
    message.row_step = total_bytes
    if message.data is not data:
        message.data = data
