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

"""Tests for grasp-controller schema validation."""

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.replicator.teleop import GraspController
from pxr import PhysxSchema, UsdPhysics


class TestGraspValidation(omni.kit.test.AsyncTestCase):
    """Verify classification of driven, mimic, and undriven joints."""

    async def setUp(self) -> None:
        """Create a fresh procedural gripper hierarchy."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Close the procedural stage."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    def test_grasp_multi_apply_schema_query(self) -> None:
        """Angular and linear drives count, while mimic drives are not controllable."""
        stage = stage_utils.get_current_stage()
        root_path = "/World/Gripper"
        stage_utils.define_prim(root_path, "Xform")

        angular_path = f"{root_path}/AngularJoint"
        linear_path = f"{root_path}/LinearJoint"
        mimic_path = f"{root_path}/MimicJoint"
        undriven_path = f"{root_path}/UndrivenJoint"

        angular = UsdPhysics.RevoluteJoint.Define(stage, angular_path)
        UsdPhysics.DriveAPI.Apply(angular.GetPrim(), "angular")
        linear = UsdPhysics.PrismaticJoint.Define(stage, linear_path)
        UsdPhysics.DriveAPI.Apply(linear.GetPrim(), "linear")
        mimic = UsdPhysics.RevoluteJoint.Define(stage, mimic_path)
        UsdPhysics.DriveAPI.Apply(mimic.GetPrim(), "angular")
        PhysxSchema.PhysxMimicJointAPI.Apply(mimic.GetPrim(), "rotZ")
        UsdPhysics.RevoluteJoint.Define(stage, undriven_path)

        result = GraspController().validate_prim(root_path)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.total_joints, 4)
        self.assertEqual(result.drive_joints, 3)
        self.assertEqual(result.mimic_joints, 1)
        self.assertEqual(result.controllable_joints, 2)
        self.assertEqual(result.drive_joint_paths, [angular_path, linear_path])
