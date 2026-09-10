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

"""Standalone smoke tests for isaacsim-robot-schema."""

from __future__ import annotations

import subprocess
import sys
import unittest

from pxr import Plug, Usd


class TestSmoke(unittest.TestCase):
    """Import and namespace validation."""

    def test_import_robot_schema(self) -> None:
        """Verify robot_schema is importable under usd.schema.isaac namespace."""
        import usd.schema.isaac.robot_schema as rs

        self.assertTrue(hasattr(rs, "__file__"))

    def test_no_omni_modules(self) -> None:
        """Modern imports should not eagerly load legacy omni.* modules."""
        before = {m for m in sys.modules if m.startswith("omni.")}
        import usd.schema.isaac.robot_schema  # noqa: F401

        after = {m for m in sys.modules if m.startswith("omni.")}
        self.assertEqual(after, before, f"Unexpected omni modules: {sorted(after - before)}")

    def test_legacy_sensor_schema_imports(self) -> None:
        """Legacy schema imports register resources and author typed prims."""
        from omni.isaac import IsaacSensorSchema, RangeSensorSchema

        self.assertTrue(hasattr(IsaacSensorSchema, "IsaacBaseSensor"))
        self.assertTrue(hasattr(IsaacSensorSchema, "IsaacContactSensor"))
        self.assertTrue(hasattr(RangeSensorSchema, "RangeSensor"))
        self.assertTrue(hasattr(RangeSensorSchema, "Lidar"))
        self.assertTrue(Plug.Registry().GetPluginWithName("isaacSensorSchema"))
        self.assertTrue(Plug.Registry().GetPluginWithName("rangeSensorSchema"))

        stage = Usd.Stage.CreateInMemory()
        contact = IsaacSensorSchema.IsaacContactSensor.Define(stage, "/Contact")
        lidar = RangeSensorSchema.Lidar.Define(stage, "/Lidar")
        self.assertEqual(contact.GetPrim().GetTypeName(), "IsaacContactSensor")
        self.assertEqual(lidar.GetPrim().GetTypeName(), "Lidar")
        self.assertTrue(contact.GetEnabledAttr().Get())
        self.assertEqual(lidar.GetHorizontalFovAttr().Get(), 360.0)

    def test_legacy_sensor_schema_imports_in_fresh_process(self) -> None:
        """Legacy imports must register schema resources without a prior modern import."""
        script = """
from omni.isaac import IsaacSensorSchema, RangeSensorSchema
from pxr import Plug, Usd

assert Plug.Registry().GetPluginWithName("isaacSensorSchema")
assert Plug.Registry().GetPluginWithName("rangeSensorSchema")
stage = Usd.Stage.CreateInMemory()
contact = IsaacSensorSchema.IsaacContactSensor.Define(stage, "/Contact")
lidar = RangeSensorSchema.Lidar.Define(stage, "/Lidar")
assert contact.GetEnabledAttr().Get() is True
assert lidar.GetHorizontalFovAttr().Get() == 360.0
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestFunctional(unittest.TestCase):
    """Schema utility tests."""

    def test_utils_importable(self) -> None:
        """Verify utils module is importable and has UsdPhysics available."""
        from usd.schema.isaac.robot_schema import utils

        self.assertTrue(hasattr(utils, "UsdPhysics"))


if __name__ == "__main__":
    unittest.main()
