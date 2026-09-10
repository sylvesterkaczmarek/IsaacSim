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

"""Tests for the deprecated manipulator-example import shims."""

import importlib
from unittest import mock

import omni.kit.test
from isaacsim.robot.experimental.manipulators.examples import _migration
from isaacsim.robot.experimental.manipulators.examples.universal_robots import ur10_palletizing as legacy
from isaacsim.robot_motion.examples.manipulation import ur10_palletizing as current


class TestImportShims(omni.kit.test.AsyncTestCase):
    async def test_equivalent_interactive_classes_preserve_identity(self) -> None:
        packages = {
            "bin_filling": (
                "bin_filling",
                "BinFilling",
                "bin_filling_extension",
                ("BinFillingExtension", "BinFillingUI"),
            ),
            "pick_place": (
                "pick_place_example",
                "FrankaPickPlaceInteractive",
                "pick_place_example_extension",
                ("FrankaPickPlaceExtension", "FrankaPickPlaceUI"),
            ),
            "replay_follow_target": (
                "replay_follow_target",
                "ReplayFollowTarget",
                "replay_follow_target_extension",
                ("ReplayFollowTargetExtension", "ReplayFollowTargetUI"),
            ),
            "robo_factory": (
                "robo_factory",
                "RoboFactory",
                "robo_factory_extension",
                ("RoboFactoryExtension", "RoboFactoryUI"),
            ),
        }
        legacy_root = "isaacsim.robot.experimental.manipulators.examples.interactive"
        current_root = "isaacsim.robot_motion.examples.manipulation.interactive"

        for package, (sample_module, sample_name, extension_module, extension_names) in packages.items():
            with self.subTest(package=package):
                old_package = importlib.import_module(f"{legacy_root}.{package}")
                new_package = importlib.import_module(f"{current_root}.{package}")
                self.assertIs(getattr(old_package, sample_name), getattr(new_package, sample_name))
                self.assertIs(getattr(old_package, extension_names[0]), getattr(new_package, extension_names[0]))

                for module, names in ((sample_module, (sample_name,)), (extension_module, extension_names)):
                    old = importlib.import_module(f"{legacy_root}.{package}.{module}")
                    new = importlib.import_module(f"{current_root}.{package}.{module}")
                    for name in names:
                        self.assertIs(getattr(old, name), getattr(new, name))

    async def test_removed_path_planning_modules_raise_actionable_errors(self) -> None:
        root = "isaacsim.robot.experimental.manipulators.examples.interactive.path_planning"
        replacement = "isaacsim.robot_motion.cumotion.examples.graph_planner"

        for module in ("", ".path_planning", ".path_planning_extension"):
            with self.subTest(module=module):
                with self.assertRaisesRegex(ImportError, replacement):
                    importlib.import_module(f"{root}{module}")

    async def test_warning_is_actionable_and_emitted_once(self) -> None:
        was_warned = _migration._warned
        self.addCleanup(setattr, _migration, "_warned", was_warned)
        _migration._warned = False

        with mock.patch.object(_migration.carb, "log_warn") as log_warn:
            _migration.warn_legacy_import()
            _migration.warn_legacy_import()

        log_warn.assert_called_once()
        warning = log_warn.call_args.args[0]
        self.assertIn("isaacsim.robot.experimental.manipulators.examples", warning)
        self.assertIn("isaacsim.robot_motion.examples", warning)

    async def test_ur10_palletizing_symbols_preserve_identity(self) -> None:
        self.assertEqual(legacy.__all__, current.__all__)
        for name in legacy.__all__:
            self.assertIs(getattr(legacy, name), getattr(current, name), name)
        self.assertEqual(legacy._PHYSICS_DT, current._PHYSICS_DT)

    async def test_removed_modules_raise_actionable_errors(self) -> None:
        modules = [
            "franka.follow_target",
            "franka.franka",
            "franka.pick_place",
            "franka.stacking",
            "interactive.follow_target_ik",
            "interactive.follow_target_ik.follow_target",
            "interactive.follow_target_motion_generation",
            "interactive.follow_target_motion_generation.follow_target",
            "interactive.follow_target_motion_generation.follow_target_extension",
            "universal_robots.follow_target",
            "universal_robots.stacking",
            "universal_robots.ur10",
        ]
        root = "isaacsim.robot.experimental.manipulators.examples"

        for module in modules:
            with self.subTest(module=module):
                with self.assertRaisesRegex(ImportError, "isaacsim.robot_motion.examples"):
                    importlib.import_module(f"{root}.{module}")

    async def test_removed_package_wildcard_imports_raise_actionable_errors(self) -> None:
        root = "isaacsim.robot.experimental.manipulators.examples"
        for package in ("franka", "universal_robots"):
            with self.subTest(package=package):
                with self.assertRaisesRegex(ImportError, "isaacsim.robot_motion.examples"):
                    exec(f"from {root}.{package} import *", {})

    async def test_unknown_package_attributes_raise_attribute_error(self) -> None:
        root = "isaacsim.robot.experimental.manipulators.examples"
        for package in ("franka", "universal_robots"):
            with self.subTest(package=package):
                module = importlib.import_module(f"{root}.{package}")
                with self.assertRaises(AttributeError):
                    getattr(module, "not_a_legacy_api")
