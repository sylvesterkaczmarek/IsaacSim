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

"""Test that robot USD asset paths stay relative to the Isaac assets root.

Resolving the assets root while a class body executes runs at *import* time. When the
assets server is unreachable that raises, the whole `isaacsim.replicator.mobility_gen.examples`
module fails to import, and `isaacsim.replicator.mobility_gen.ui` then fails with a
confusing ``cannot import name ... (unknown location)``. Keeping `usd_url` relative and
resolving it inside `build` means an unreachable server only fails the build call.
"""

import pathlib

import omni.kit.test
import yaml
from isaacsim.replicator.mobility_gen.examples import CarterRobot, H1Robot, JetbotRobot, SpotRobot


class TestAssetUrlResolution(omni.kit.test.AsyncTestCase):
    """Robot asset URL resolution regression tests."""

    # Every robot that declares a hard-coded asset path.
    _ROBOTS = (JetbotRobot, CarterRobot, H1Robot, SpotRobot)

    async def test_usd_url_is_relative_to_assets_root(self) -> None:
        """Verify usd_url is a bare suffix, not an already-resolved absolute URL."""
        for robot in self._ROBOTS:
            with self.subTest(robot=robot.__name__):
                usd_url = robot.usd_url
                # An absolute URL here means the assets root was baked in at import time.
                self.assertNotIn("://", usd_url)
                self.assertTrue(usd_url.startswith("/Isaac/"), f"{robot.__name__}: {usd_url!r}")

    async def test_usd_url_matches_yaml_asset_path(self) -> None:
        """Verify the hard-coded suffixes agree with the YAML-driven robot configs."""
        # The multi-sensor variants read the same assets from data/robots/*.yaml, which
        # stores suffixes; the two sources must not drift apart.
        data_dir = pathlib.Path(__file__).parent.parent / "data" / "robots"
        expected = {
            "jetbot.yaml": JetbotRobot.usd_url,
            "carter.yaml": CarterRobot.usd_url,
            "h1.yaml": H1Robot.usd_url,
            "spot.yaml": SpotRobot.usd_url,
        }
        for name, usd_url in expected.items():
            with self.subTest(config=name):
                config = yaml.safe_load((data_dir / name).read_text())
                self.assertEqual(config["asset_path"], usd_url)
