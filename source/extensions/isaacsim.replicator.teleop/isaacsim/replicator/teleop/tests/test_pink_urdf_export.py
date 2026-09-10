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

"""Tests for the minimal PINK URDF export graph."""

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.replicator.teleop.controllers.pink_urdf_export import _collect_minimal_urdf_graph
from pxr import Sdf, UsdPhysics


class TestPinkUrdfExport(omni.kit.test.AsyncTestCase):
    """Characterize procedural kinematic-graph collection."""

    async def setUp(self) -> None:
        """Create a fresh procedural stage."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Close the procedural stage."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    def test_collects_nested_in_scope_joint_graph(self) -> None:
        """Nested joints are collected while out-of-scope targets are ignored."""
        stage = stage_utils.get_current_stage()
        root_path = "/World/Robot"
        base_path = f"{root_path}/Base"
        arm_path = f"{root_path}/Arm"
        tool_path = f"{root_path}/Tool"
        outside_path = "/World/Outside"

        for path in (root_path, base_path, arm_path, tool_path, outside_path):
            stage_utils.define_prim(path, "Xform")

        root_joint = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/joints/root")
        root_joint.GetBody1Rel().SetTargets([Sdf.Path(base_path)])

        shoulder_path = f"{root_path}/joints/shoulder"
        shoulder = UsdPhysics.RevoluteJoint.Define(stage, shoulder_path)
        shoulder.GetBody0Rel().SetTargets([Sdf.Path(base_path)])
        shoulder.GetBody1Rel().SetTargets([Sdf.Path(arm_path)])

        tool_joint_path = f"{root_path}/assembly/wrist/joints/tool"
        tool_joint = UsdPhysics.PrismaticJoint.Define(stage, tool_joint_path)
        tool_joint.GetBody0Rel().SetTargets([Sdf.Path(arm_path)])
        tool_joint.GetBody1Rel().SetTargets([Sdf.Path(tool_path)])

        ignored = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/joints/outside")
        ignored.GetBody0Rel().SetTargets([Sdf.Path(tool_path)])
        ignored.GetBody1Rel().SetTargets([Sdf.Path(outside_path)])

        root_link_path, link_paths, joints, link_names = _collect_minimal_urdf_graph(stage, root_path)

        self.assertEqual(root_link_path, base_path)
        self.assertEqual(link_paths, [base_path, arm_path, tool_path])
        self.assertEqual([joint["prim_path"] for joint in joints], [shoulder_path, tool_joint_path])
        self.assertEqual([joint["joint_type"] for joint in joints], ["revolute", "prismatic"])
        self.assertEqual(
            link_names,
            {
                arm_path: "Robot_Arm",
                base_path: "Robot_Base",
                tool_path: "Robot_Tool",
            },
        )
