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

"""Verify Robotics Examples browser registrations exposed by the extension."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import omni.kit.test

_REGISTRATIONS = (
    ("bin_filling.bin_filling_extension", "BinFillingExtension", "Manipulation", "Bin Filling"),
    (
        "follow_target_motion_generation.follow_target_extension",
        "FollowTargetExtension",
        "Manipulation",
        "Follow Target (cuMotion)",
    ),
    (
        "follow_target_ur10.follow_target_extension",
        "FollowTargetExtension",
        "Manipulation",
        "Follow Target (UR10)",
    ),
    ("palletizing.palletizing_extension", "PalletizingExtension", "Manipulation", "UR10 Palletizing"),
    ("pick_place.pick_place_example_extension", "FrankaPickPlaceExtension", "Manipulation", "Franka Pick Place"),
    (
        "replay_follow_target.replay_follow_target_extension",
        "ReplayFollowTargetExtension",
        "Manipulation",
        "Replay Follow Target",
    ),
    ("robo_factory.robo_factory_extension", "RoboFactoryExtension", "Multi-Robot", "RoboFactory"),
)


class TestBrowserRegistration(omni.kit.test.AsyncTestCase):
    """Registration contract for documented interactive manipulation examples."""

    async def test_examples_register_and_deregister_exact_browser_keys(self) -> None:
        """Check every documented example's category and name."""

        package = "isaacsim.robot_motion.examples.manipulation.interactive"
        for module_suffix, class_name, category, name in _REGISTRATIONS:
            with self.subTest(name=name):
                module = importlib.import_module(f"{package}.{module_suffix}")
                browser = MagicMock()
                with patch.object(module, "get_browser_instance", return_value=browser):
                    extension = getattr(module, class_name)()
                    extension.on_startup("isaacsim.robot_motion.examples")
                    browser.register_example.assert_called_once()
                    self.assertEqual(browser.register_example.call_args.kwargs["category"], category)
                    self.assertEqual(browser.register_example.call_args.kwargs["name"], name)
                    extension.on_shutdown()
                    browser.deregister_example.assert_called_once_with(name=name, category=category)
