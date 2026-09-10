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

"""Test the ROS occupancy map save/load round trip.

These tests cover origin serialization, so that a map written by the product reloads with a
numeric origin and coordinate conversion keeps working.
"""

import pathlib
import tempfile

import numpy as np
import omni.kit.test
import yaml
from isaacsim.replicator.experimental.mobility_gen.impl.occupancy_map import (
    OccupancyMap,
    OccupancyMapDataValue,
    _parse_ros_origin,
)

ORIGIN = (-5.0, -3.0, 1.5)
LEGACY_YAML = """
image: map.png
resolution: 0.05
origin: (-5.0, -3.0, 1.5)
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""


def _make_map(origin: tuple[float, float, float] = ORIGIN) -> OccupancyMap:
    """Build a small occupancy map with a known origin."""
    data = np.full((4, 6), OccupancyMapDataValue.FREESPACE, dtype=np.uint8)
    return OccupancyMap(data=data, resolution=0.05, origin=origin)


class TestOccupancyMapRosRoundTrip(omni.kit.test.AsyncTestCase):
    """Test ROS occupancy map serialization."""

    async def test_ros_yaml_origin_is_numeric_sequence(self) -> None:
        """Verify the written origin parses back as three numbers, not a string."""
        parsed = yaml.safe_load(_make_map().ros_yaml())

        self.assertIsInstance(parsed["origin"], list)
        self.assertEqual([float(v) for v in parsed["origin"]], list(ORIGIN))

    async def test_save_ros_round_trip_preserves_origin(self) -> None:
        """Verify save_ros -> from_ros_yaml keeps the origin numeric and usable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_map().save_ros(tmpdir)
            reloaded = OccupancyMap.from_ros_yaml(str(pathlib.Path(tmpdir) / OccupancyMap.ROS_YAML_FILENAME))

        # Compute first: this is what raised UFuncNoLoopError when the origin reloaded as a string.
        world = reloaded.pixel_to_world_numpy(np.array([[0, 0]], dtype=np.float64))

        np.testing.assert_allclose(world, [[-5.0, -2.8]])
        self.assertEqual(reloaded.origin, ORIGIN)

    async def test_from_ros_yaml_loads_legacy_origin_file(self) -> None:
        """Verify a map file whose origin was written as a tuple repr still loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            _make_map().ros_image().save(tmp / OccupancyMap.ROS_IMAGE_FILENAME)
            (tmp / OccupancyMap.ROS_YAML_FILENAME).write_text(LEGACY_YAML)
            reloaded = OccupancyMap.from_ros_yaml(str(tmp / OccupancyMap.ROS_YAML_FILENAME))

        self.assertEqual(reloaded.origin, ORIGIN)
        np.testing.assert_allclose(reloaded.pixel_to_world_numpy(np.array([[0, 0]], dtype=np.float64)), [[-5.0, -2.8]])

    async def test_constructor_normalizes_origin(self) -> None:
        """Verify every construction path stores a numeric origin."""
        self.assertEqual(_make_map(origin="(-5.0, -3.0, 1.5)").origin, ORIGIN)
        self.assertEqual(_make_map(origin=[-5.0, -3.0, 1.5]).origin, ORIGIN)
        with self.assertRaisesRegex(ValueError, "three numeric components"):
            _make_map(origin=(-5.0, -3.0))

    async def test_parse_origin_accepts_sequence_and_string_forms(self) -> None:
        """Verify comma- and whitespace-separated scalar strings are both accepted."""
        self.assertEqual(_parse_ros_origin("(-5.0, -3.0, 1.5)"), ORIGIN)
        self.assertEqual(_parse_ros_origin("[-5.0 -3.0 1.5]"), ORIGIN)
        self.assertEqual(_parse_ros_origin([-5.0, -3.0, 1.5]), ORIGIN)
        self.assertEqual(_parse_ros_origin(np.array([-5.0, -3.0, 1.5])), ORIGIN)

    async def test_parse_origin_rejects_malformed_value(self) -> None:
        """Verify a malformed origin raises a clear error rather than a dtype failure."""
        for bad in ("not-an-origin", [1.0, 2.0], [1.0, 2.0, 3.0, 4.0], None):
            with self.assertRaisesRegex(ValueError, "three numeric components"):
                _parse_ros_origin(bad)
