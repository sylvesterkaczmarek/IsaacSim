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

"""Standalone smoke tests for the SysID backend wheel."""

from __future__ import annotations

import importlib
import sys
import unittest


class TestSmoke(unittest.TestCase):
    """Verify portable imports and bundled resources."""

    def test_package_root_imports_without_kit(self) -> None:
        """Import the package root without loading Kit runtime modules."""
        importlib.import_module("isaacsim.robot_setup.sysid")

        runtime_modules = [
            name for name in sys.modules if name == "carb" or name.startswith(("carb.", "omni.", "isaacsim.core."))
        ]
        self.assertEqual(runtime_modules, [], f"Unexpected Kit runtime modules: {runtime_modules}")

    def test_portable_contracts_and_resources_import(self) -> None:
        """Import a run contract and read a packaged schema."""
        run_spec = importlib.import_module("isaacsim.robot_setup.sysid.run_spec")
        resource_loader = importlib.import_module("isaacsim.robot_setup.sysid.resource_loader")

        self.assertEqual(run_spec.SysIdRunSpec().schema_version, 2)
        self.assertIn(
            '"title": "SysID Run Spec"',
            resource_loader.read_schema_resource("sysid_run_spec.schema.json"),
        )

    def test_builtin_presets_are_packaged(self) -> None:
        """Built-in recipe presets must ship in the wheel, not only in the source tree."""
        preset_loader = importlib.import_module("isaacsim.robot_setup.sysid.preset_loader")

        names = preset_loader.list_builtin_parameter_presets()
        self.assertIn("drive_calibration_inair", names)
        self.assertEqual(
            "drive_calibration_inair",
            preset_loader.load_builtin_parameter_preset("drive_calibration_inair").name,
        )

    def test_preflight_runs_without_importing_usd(self) -> None:
        """Keep preflight usable in a wheel install, where USD is not a base dependency."""
        for name in ("isaacsim.robot_setup.sysid.preflight", "isaacsim.robot_setup.sysid.articulation_utils", "pxr"):
            sys.modules.pop(name, None)

        preflight = importlib.import_module("isaacsim.robot_setup.sysid.preflight")
        run_spec = importlib.import_module("isaacsim.robot_setup.sysid.run_spec")
        result = preflight.preflight_sysid_run_spec(
            run_spec.SysIdRunSpec(),
            source_exists_fn=lambda *_args: True,
            allow_empty_parameters=True,
        )

        self.assertNotIn("pxr", sys.modules)
        self.assertIn("stage_input_path", {issue.code for issue in result.errors})


if __name__ == "__main__":
    unittest.main()
