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

"""Tests for teleop prim validation helpers."""

from __future__ import annotations

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.replicator.teleop import validate_floating_end_effector, validate_marker_path


class TestValidation(omni.kit.test.AsyncTestCase):
    """Characterize the public teleop validation results."""

    async def setUp(self) -> None:
        """Create a fresh stage."""
        await stage_utils.create_new_stage_async()

    async def tearDown(self) -> None:
        """Close the test stage."""
        stage_utils.close_stage()
        await app_utils.update_app_async()

    async def test_floating_end_effector_accepts_dynamic_xform(self) -> None:
        """A rigid body with mass and controller-compatible xform ops is valid."""
        stage_utils.define_prim("/World/Gripper", "Xform")
        XformPrim("/World/Gripper", reset_xform_op_properties=True)
        RigidPrim("/World/Gripper", masses=[1.0])

        result = validate_floating_end_effector("/World/Gripper")

        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])

    async def test_floating_end_effector_reports_missing_requirements(self) -> None:
        """Missing schemas and controller-compatible xform ops retain their diagnostics."""
        stage_utils.define_prim("/World/IncompleteGripper", "Xform")

        result = validate_floating_end_effector("/World/IncompleteGripper")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors, ["Missing RigidBodyAPI"])
        self.assertEqual(
            result.warnings,
            [
                "Missing MassAPI (angular dynamics may not work)",
                "Missing translate xform op",
                "Missing orient xform op",
            ],
        )

    async def test_validation_reports_invalid_and_missing_paths(self) -> None:
        """Invalid syntax and absent prims retain distinct errors."""
        invalid = validate_floating_end_effector("")
        missing = validate_floating_end_effector("/World/Missing")

        self.assertEqual(invalid.errors, ["Invalid prim path: ''"])
        self.assertEqual(missing.errors, ["Prim not found at '/World/Missing'"])

    async def test_marker_path_reports_physics_descendants(self) -> None:
        """Rigid-body and collider descendants block marker tracking."""
        stage_utils.define_prim("/World/Marker", "Xform")
        stage_utils.define_prim("/World/Marker/Rigid", "Xform")
        RigidPrim("/World/Marker/Rigid")
        Cube("/World/Marker/Collider", sizes=1.0)
        GeomPrim("/World/Marker/Collider", apply_collision_apis=True)

        result = validate_marker_path("/World/Marker")

        self.assertTrue(result.is_valid)
        self.assertTrue(result.blocks_tracking)
        self.assertEqual(result.warnings, ["1 RigidBody descendant(s)", "1 Collider descendant(s)"])

    async def test_marker_path_includes_selected_prim(self) -> None:
        """Physics APIs on the selected marker prim itself block tracking."""
        stage_utils.define_prim("/World/Marker", "Xform")
        RigidPrim("/World/Marker")
        GeomPrim("/World/Marker", apply_collision_apis=True)

        result = validate_marker_path("/World/Marker")

        self.assertTrue(result.is_valid)
        self.assertTrue(result.blocks_tracking)
        self.assertEqual(result.warnings, ["1 RigidBody descendant(s)", "1 Collider descendant(s)"])
