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

"""Tests for teleop profile validation against stage state."""

from __future__ import annotations

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.replicator.teleop import (
    STAGE_STATE_NO_STAGE,
    STAGE_STATE_READY,
    BimanualControllerProfile,
    ControllerSideProfile,
    GraspControllerProfile,
    GraspSideProfile,
    LocomotionProfile,
    TeleopProfile,
    TeleopSettingsProfile,
    resolve_teleop_profile,
)
from pxr import UsdPhysics


class TestTeleopResolver(omni.kit.test.AsyncTestCase):
    """Verify teleop profile validation against current stage state."""

    async def setUp(self) -> None:
        """Set up the test fixture."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()

    async def tearDown(self) -> None:
        """Tear down the test fixture."""
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            stage_utils.close_stage()
            await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    async def test_resolver_reports_no_stage(self) -> None:
        """Verify the resolver reports no-stage when no USD stage is open."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        report = resolve_teleop_profile(TeleopProfile())

        self.assertEqual(report.stage_state, STAGE_STATE_NO_STAGE)
        self.assertFalse(report.ready)
        self.assertEqual(report.error_count, 0)

    async def test_resolver_reports_missing_stage_references(self) -> None:
        """Verify the resolver surfaces errors for prims that do not exist on stage."""
        profile = TeleopProfile(
            session=TeleopSettingsProfile(
                tracking_space_enabled=True,
                tracking_space_path="/World/MissingTrackingSpace",
            ),
            floating=BimanualControllerProfile(
                left=ControllerSideProfile(
                    enabled=True,
                    settings={"prim_path": "/World/MissingRigidBody"},
                )
            ),
            locomotion=LocomotionProfile(
                enabled=True,
                settings={"prim_path": "/World/MissingBase"},
            ),
        )

        report = resolve_teleop_profile(profile)

        self.assertEqual(report.stage_state, STAGE_STATE_READY)
        self.assertFalse(report.ready)
        self.assertGreaterEqual(report.error_count, 2)
        self.assertGreaterEqual(report.warning_count, 1)
        issue_sources = {issue.source for issue in report.issues}
        self.assertIn("Session Tracking Space", issue_sources)
        self.assertIn("Floating Left", issue_sources)
        self.assertIn("Locomotion", issue_sources)

    def test_floating_profile_schema_query(self) -> None:
        """Floating targets must be dynamic rigid bodies before the profile is ready."""
        path = "/World/FloatingHandle"
        stage_utils.define_prim(path, "Xform")
        profile = TeleopProfile(
            floating=BimanualControllerProfile(left=ControllerSideProfile(enabled=True, settings={"prim_path": path}))
        )

        report = resolve_teleop_profile(profile)
        floating_issues = [issue for issue in report.issues if issue.source == "Floating Left"]
        self.assertEqual(len(floating_issues), 1)
        self.assertIn("must already have RigidBodyAPI", floating_issues[0].message)

        RigidPrim(path)
        report = resolve_teleop_profile(profile)
        self.assertFalse([issue for issue in report.issues if issue.source == "Floating Left"])

    @staticmethod
    def _define_grasp_joint(root_path: str, joint_name: str) -> None:
        stage = stage_utils.get_current_stage()
        stage_utils.define_prim(root_path, "Xform")
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/{joint_name}")
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular").CreateTargetPositionAttr(0.0)

    def test_retargeted_grasp_requires_kind_and_aliases(self) -> None:
        """Resolver explains incomplete retargeted profile fields."""
        root_path = "/World/Hand"
        self._define_grasp_joint(root_path, "right_hand_index_0_joint")
        profile = TeleopProfile(
            grasp=GraspControllerProfile(
                right=GraspSideProfile(
                    enabled=True,
                    prim_path=root_path,
                    config_path="builtin://dex3_grasp",
                    drive_mode="retargeted",
                )
            )
        )

        report = resolve_teleop_profile(profile)
        messages = " ".join(issue.message for issue in report.issues if issue.source == "Grasp Right")

        self.assertIn("requires retargeter_kind 'trihand'", messages)
        self.assertIn("requires at least one joint alias", messages)

    def test_retargeted_grasp_aliases_resolve_config_and_stage_joints(self) -> None:
        """Resolver identifies aliases missing from the grasp config or USD hand."""
        root_path = "/World/Hand"
        self._define_grasp_joint(root_path, "right_hand_index_0_joint")
        profile = TeleopProfile(
            grasp=GraspControllerProfile(
                right=GraspSideProfile(
                    enabled=True,
                    prim_path=root_path,
                    config_path="builtin://dex3_grasp",
                    drive_mode="retargeted",
                    retargeter_kind="trihand",
                    joint_aliases={
                        "index_proximal": "missing_from_config",
                        "middle_proximal": "right_hand_middle_0_joint",
                        "ring_proximal": "right_hand_index_0_joint",
                    },
                )
            )
        )

        report = resolve_teleop_profile(profile)
        messages = " ".join(issue.message for issue in report.issues if issue.source == "Grasp Right")

        self.assertIn("Unknown TriHand semantic alias(es): ring_proximal", messages)
        self.assertIn("missing from grasp config: missing_from_config", messages)
        self.assertIn("not controllable below grasp prim", messages)
