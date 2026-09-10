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

"""Verify standalone robot motion schema registration and authoring."""

import unittest

from pxr import Plug, Usd


class TestRobotMotionSchemaStandalone(unittest.TestCase):
    """Verify the robot motion schema without an Isaac Sim or Kit runtime."""

    def test_import_registers_schema_plugin(self) -> None:
        """Verify importing the package registers the codeless USD plugin."""
        import isaacsim.robot_motion.schema  # noqa: F401

        self.assertIsNotNone(Plug.Registry().GetPluginWithName("RobotMotionSchema"))

    def test_apply_motion_planning_api(self) -> None:
        """Verify the public helper applies the motion planning schema."""
        from isaacsim.robot_motion.schema import apply_motion_planning_api

        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Robot", "Xform")
        apply_motion_planning_api(prim, enabled=False)

        self.assertTrue(prim.HasAPI("IsaacMotionPlanningAPI"))
        self.assertIsNotNone(prim.GetPrimDefinition().GetPropertyDefinition("isaac:motionPlanning:collisionEnabled"))
        self.assertFalse(prim.GetAttribute("isaac:motionPlanning:collisionEnabled").Get())


if __name__ == "__main__":
    unittest.main()
