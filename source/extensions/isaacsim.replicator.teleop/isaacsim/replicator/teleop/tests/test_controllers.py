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

"""Test controller instantiation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.usd
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.replicator.teleop import (
    FloatingRigidBodyController,
    GraspController,
    LocomotionController,
    LocomotionDriveMode,
    RobotIKController,
    TeleopManager,
)
from isaacsim.replicator.teleop.controllers.base import find_owning_articulation_root
from isaacsim.replicator.teleop.controllers.robot_ik import _count_chain_dofs
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics


class TestControllerInstantiation(omni.kit.test.AsyncTestCase):
    """Verify all controllers can be created without a stage or VR."""

    async def setUp(self) -> None:
        """Set up test environment."""
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Tear down test environment."""
        await app_utils.update_app_async()

    async def test_floating_controller(self) -> None:
        """Verify FloatingRigidBodyController can be created without a stage."""
        ctrl = FloatingRigidBodyController()
        self.assertIsNotNone(ctrl)

    async def test_ik_controller(self) -> None:
        """Verify RobotIKController can be created without a stage."""
        ctrl = RobotIKController()
        self.assertIsNotNone(ctrl)

    async def test_grasp_controller(self) -> None:
        """Verify GraspController can be created without a stage."""
        ctrl = GraspController()
        self.assertIsNotNone(ctrl)

    async def test_locomotion_controller(self) -> None:
        """Verify LocomotionController can be created without a stage."""
        ctrl = LocomotionController()
        self.assertIsNotNone(ctrl)
        self.assertAlmostEqual(ctrl.linear_step, LocomotionController.DEFAULT_LINEAR_STEP)
        self.assertAlmostEqual(ctrl.angular_step, LocomotionController.DEFAULT_ANGULAR_STEP)
        self.assertAlmostEqual(ctrl.linear_speed, LocomotionController.DEFAULT_LINEAR_SPEED)
        self.assertAlmostEqual(ctrl.angular_speed, LocomotionController.DEFAULT_ANGULAR_SPEED)
        self.assertEqual(ctrl.drive_mode, LocomotionDriveMode.AUTO)
        self.assertEqual(ctrl.effective_drive_mode, LocomotionDriveMode.TELEPORT)
        ctrl.set_drive_mode("velocity")
        self.assertEqual(ctrl.drive_mode, LocomotionDriveMode.VELOCITY)
        self.assertFalse(ctrl.carry_tracking_space_enabled)
        self.assertFalse(ctrl.carry_tracking_space_available)
        self.assertFalse(ctrl.set_carry_tracking_space(True))
        self.assertTrue(ctrl.set_carry_tracking_space(False))


class TestLocomotionManagerLifecycle(omni.kit.test.AsyncTestCase):
    """Verify session lifecycle boundaries halt physics-driven locomotion."""

    def setUp(self) -> None:
        """Create a manager with a mock locomotion controller."""
        self._manager = TeleopManager()
        self._controller = MagicMock()
        self._manager.set_locomotion_controller(self._controller)
        self._controller.reset_mock()

    def tearDown(self) -> None:
        """Release manager subscriptions and controller references."""
        self._manager.destroy()

    def test_disconnect_stops_locomotion(self) -> None:
        """A direct provider disconnect zeros any last commanded velocity."""
        self._manager._is_connected = True  # noqa: SLF001

        self._manager.disconnect()

        self._controller.stop_motion.assert_called_once_with()

    def test_disabling_tracking_stops_locomotion(self) -> None:
        """Suppressing locomotion updates zeros any last commanded velocity."""
        self._manager._locomotion_tracking_enabled = True  # noqa: SLF001

        self._manager.set_locomotion_tracking(False)

        self._controller.stop_motion.assert_called_once_with()


class TestLocomotionCustomAnchorSafety(omni.kit.test.AsyncTestCase):
    """Verify carry never normalizes a user-authored custom-anchor xform stack."""

    async def setUp(self) -> None:
        """Create a fresh stage for custom-anchor authoring tests."""
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World", "Xform")
        self._controller = LocomotionController()
        self._controller.set_prim_path("/World/Robot")

    async def tearDown(self) -> None:
        """Release the stage between tests."""
        await omni.usd.get_context().close_stage_async()

    def test_matrix_authored_anchor_remains_read_only(self) -> None:
        """Reject carry without replacing a custom matrix op with canonical ops."""
        stage = stage_utils.get_current_stage()
        anchor = UsdGeom.Xform.Define(stage, "/World/MatrixAnchor")
        anchor.AddTransformOp().Set(Gf.Matrix4d(1.0))
        before = anchor.GetPrim().GetPropertyNames()

        self._controller.set_tracking_space_prim_path("/World/MatrixAnchor")

        self.assertFalse(self._controller.carry_tracking_space_available)
        self.assertFalse(self._controller.set_carry_tracking_space(True))
        self.assertEqual(before, anchor.GetPrim().GetPropertyNames())
        self.assertIn("xformOp:transform", before)
        self.assertNotIn("xformOp:translate", before)

    def test_standard_xform_proxy_allows_carry(self) -> None:
        """Allow carry when the user provides a standard writable proxy Xform."""
        stage = stage_utils.get_current_stage()
        anchor = UsdGeom.Xform.Define(stage, "/World/WritableAnchor")
        anchor.AddTranslateOp()
        anchor.AddOrientOp()
        anchor.AddScaleOp()

        self._controller.set_tracking_space_prim_path("/World/WritableAnchor")

        self.assertTrue(self._controller.carry_tracking_space_available)
        self.assertTrue(self._controller.set_carry_tracking_space(True))


class TestRobotIKChainDiscovery(omni.kit.test.AsyncTestCase):
    """Characterize DOF counting across assembled articulation subtrees."""

    async def setUp(self) -> None:
        """Create a fresh procedural robot hierarchy."""
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World", "Xform")

    async def tearDown(self) -> None:
        """Release the procedural stage."""
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            stage_utils.close_stage()
            await app_utils.update_app_async()

    def test_count_chain_dofs_across_overlapping_search_roots(self) -> None:
        """Movable joints count once while fixed and out-of-set joints do not."""
        stage = stage_utils.get_current_stage()
        art_path = "/World/Robot"
        base_path = f"{art_path}/Base"
        shoulder_path = f"{art_path}/Arm/Shoulder"
        wrist_path = f"{art_path}/Arm/Wrist"
        ee_path = "/World/Tool/EndEffector"
        outside_path = "/World/Outside"
        for path in (art_path, base_path, shoulder_path, wrist_path, ee_path, outside_path):
            stage_utils.define_prim(path, "Xform")

        shoulder_joint = UsdPhysics.RevoluteJoint.Define(stage, f"{art_path}/Joints/Shoulder")
        shoulder_joint.GetBody0Rel().SetTargets([Sdf.Path(base_path)])
        shoulder_joint.GetBody1Rel().SetTargets([Sdf.Path(shoulder_path)])

        wrist_joint = UsdPhysics.FixedJoint.Define(stage, f"{art_path}/Arm/Joints/WristMount")
        wrist_joint.GetBody0Rel().SetTargets([Sdf.Path(shoulder_path)])
        wrist_joint.GetBody1Rel().SetTargets([Sdf.Path(wrist_path)])

        tool_joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Tool/Joints/ToolSlide")
        tool_joint.GetBody0Rel().SetTargets([Sdf.Path(wrist_path)])
        tool_joint.GetBody1Rel().SetTargets([Sdf.Path(ee_path)])

        outside_joint = UsdPhysics.RevoluteJoint.Define(stage, f"{art_path}/Joints/Outside")
        outside_joint.GetBody0Rel().SetTargets([Sdf.Path(base_path)])
        outside_joint.GetBody1Rel().SetTargets([Sdf.Path(outside_path)])

        robot = MagicMock()
        robot.link_names = ["base", "shoulder", "wrist", "end_effector"]
        robot.link_paths = ([Sdf.Path(base_path), Sdf.Path(shoulder_path), Sdf.Path(wrist_path), Sdf.Path(ee_path)],)
        robot.get_link_indices.return_value.numpy.return_value.item.return_value = 3

        with patch("isaacsim.replicator.teleop.controllers.robot_ik.Articulation", return_value=robot):
            self.assertEqual(_count_chain_dofs(art_path, "end_effector"), 2)

    def test_update_targets_waits_for_playing_timeline(self) -> None:
        """Stopped timelines store IK targets without applying tensor results."""
        controller = RobotIKController()
        arm = controller._arm("left")  # noqa: SLF001
        arm.running = True
        arm.ctrl = MagicMock()
        timeline = MagicMock()
        timeline.is_playing.return_value = False

        with (
            patch("omni.timeline.get_timeline_interface", return_value=timeline),
            patch.object(controller, "_apply_ik_result") as apply_result,
        ):
            controller.update_targets((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), None, None)

        arm.ctrl.set_target.assert_called_once()
        apply_result.assert_not_called()


class TestFloatingRigidBodySchema(omni.kit.test.AsyncTestCase):
    """Verify floating-controller schema repair independently from UI scenarios."""

    async def setUp(self) -> None:
        """Create one normal dynamic rigid-body handle."""
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World", "Xform")
        Cube("/World/FloatingBody", sizes=0.15, positions=(0.0, 0.0, 1.0))
        GeomPrim("/World/FloatingBody", apply_collision_apis=True)
        RigidPrim("/World/FloatingBody", masses=[1.0])
        self._controller = FloatingRigidBodyController()

    async def tearDown(self) -> None:
        """Release the controller and stage."""
        self._controller.destroy("left")
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            stage_utils.close_stage()
            await app_utils.update_app_async()

    def test_enable_restores_missing_physx_rigid_body_api(self) -> None:
        """Activation restores the optional PhysX schema when an asset omits it."""
        path = "/World/FloatingBody"
        self._controller.set_prim_path("left", path)
        self.assertTrue(self._controller.configure("left"))

        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        self.assertFalse(prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI))

        self.assertTrue(self._controller.enable("left"))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI))

    def test_floating_rigid_body_schema_query(self) -> None:
        """Validation rejects non-rigid targets and warns for articulation-root handles."""
        non_rigid_path = "/World/NotRigid"
        stage_utils.define_prim(non_rigid_path, "Xform")
        self._controller.set_prim_path("left", non_rigid_path)
        valid, message = self._controller.validate("left")
        self.assertFalse(valid)
        self.assertIn("must already have RigidBodyAPI", message)

        rigid_path = "/World/FloatingBody"
        prim = stage_utils.get_current_stage().GetPrimAtPath(rigid_path)
        prim_utils.ensure_api(prim, UsdPhysics.ArticulationRootAPI)
        self._controller.set_prim_path("left", rigid_path)
        valid, message = self._controller.validate("left")
        self.assertTrue(valid)
        self.assertIn("selected rigid body is also an articulation root", message)

    def test_articulation_root_schema_query_fallback(self) -> None:
        """The shared helper walks ancestors when the experimental lookup is unavailable."""
        root_path = "/World/Robot"
        child_path = f"{root_path}/Tool"
        root = stage_utils.define_prim(root_path, "Xform")
        stage_utils.define_prim(child_path, "Xform")
        prim_utils.ensure_api(root, UsdPhysics.ArticulationRootAPI)

        with patch.object(
            Articulation,
            "fetch_articulation_root_api_prim_paths",
            side_effect=RuntimeError("force USD fallback"),
        ):
            self.assertEqual(find_owning_articulation_root(child_path), root_path)
            self.assertIsNone(find_owning_articulation_root("/World/NotInArticulation"))
