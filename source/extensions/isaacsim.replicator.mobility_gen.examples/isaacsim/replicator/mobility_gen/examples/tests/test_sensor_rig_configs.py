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

"""Test that the shipped sensor-rig configs name camera prims the robots actually have.

A `sensor_prim_path` that resolves to nothing is only logged and skipped, so a wrong path
yields a robot that records no images without ever failing.
"""

import omni.kit.test
from isaacsim.replicator.experimental.mobility_gen import ROBOTS
from isaacsim.replicator.mobility_gen.examples.misc import HawkCamera
from isaacsim.replicator.mobility_gen.examples.robots import PolicyMultiSensorRobot

# Robots whose cameras are mounted by the class rather than shipped in the asset.
_MOUNTED_CAMERA_ROBOTS = ("H1MultiSensorRobot", "SpotMultiSensorRobot")


class TestSensorRigConfigs(omni.kit.test.AsyncTestCase):
    """The shipped robot YAMLs must name camera prims that exist once the robot is built."""

    async def test_mounted_paths_match_the_declared_mount(self) -> None:
        """Every policy-robot sensor path must sit under the camera its class mounts."""
        for name in _MOUNTED_CAMERA_ROBOTS:
            robot_cls = ROBOTS.get(name)
            self.assertTrue(issubclass(robot_cls, PolicyMultiSensorRobot), name)
            base = robot_cls.front_camera_base_path
            expected = {
                f"{base}/{robot_cls.front_camera_type.left_camera_path}",
                f"{base}/{robot_cls.front_camera_type.right_camera_path}",
            }

            configured = {c.sensor_prim_path for c in robot_cls.sensor_configs}

            self.assertEqual(configured, expected, f"{name} sensor paths do not match its mounted camera")

    async def test_mounted_robots_declare_every_camera_attribute(self) -> None:
        """A partially declared mount silently skips the spawn, which is the defect being guarded."""
        for name in _MOUNTED_CAMERA_ROBOTS:
            robot_cls = ROBOTS.get(name)
            for attr in (
                "front_camera_base_path",
                "front_camera_type",
                "front_camera_rotation",
                "front_camera_translation",
            ):
                self.assertIsNotNone(getattr(robot_cls, attr, None), f"{name} is missing {attr}")

    async def test_configs_keep_the_camera_aspect_ratio(self) -> None:
        """Kit renders square pixels, so a resolution off the camera's aspect silently crops FOV."""
        for name in _MOUNTED_CAMERA_ROBOTS:
            robot_cls = ROBOTS.get(name)
            camera_width, camera_height = robot_cls.front_camera_type.resolution
            for config in robot_cls.sensor_configs:
                self.assertAlmostEqual(
                    config.width_px / config.height_px,
                    camera_width / camera_height,
                    places=3,
                    msg=f"{name}.{config.name} aspect does not match {robot_cls.front_camera_type.__name__}",
                )

    async def test_stereo_pairs_share_a_resolution(self) -> None:
        """A stereo pair rendered at differing sizes would not rectify."""
        for name in _MOUNTED_CAMERA_ROBOTS:
            robot_cls = ROBOTS.get(name)
            sizes = {(c.width_px, c.height_px) for c in robot_cls.sensor_configs}
            self.assertEqual(len(sizes), 1, f"{name} sensors do not share one resolution: {sizes}")

    async def test_hawk_subpaths_are_what_the_configs_assume(self) -> None:
        """The YAML paths hard-code these suffixes, so a rename here must fail loudly."""
        self.assertEqual(HawkCamera.left_camera_path, "left/camera_left")
        self.assertEqual(HawkCamera.right_camera_path, "right/camera_right")
