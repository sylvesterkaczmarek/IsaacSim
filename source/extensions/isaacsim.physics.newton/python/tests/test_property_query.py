# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the Newton property query interface used for pre-physics articulation metadata.

The property query mirrors the PhysX ``QUERY_ARTICULATION`` interface so tools that inspect an
articulation before pressing play (e.g. the Gain Tuner) work identically on both backends. These
tests build minimal articulations programmatically and verify the DOF enumeration returned by the
USD-parse fallback (no live simulation view), covering ``ArticulationRootAPI`` authored on different
prims (a fixed root joint or the root rigid body) so the resolved ``add_usd`` root encloses the
whole articulation regardless of where the API is applied.
"""

from __future__ import annotations

import unittest
from typing import Any

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.kit.test
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics


async def wait_for_stage_loading() -> None:
    """Wait until USD stage loading is complete."""
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        await omni.kit.app.get_app().next_update_async()


class TestNewtonPropertyQuery(omni.kit.test.AsyncTestCase):
    """Tests for the pre-physics Newton articulation property query."""

    async def setUp(self) -> None:
        """Create a fresh stage and select the Newton physics backend."""
        await stage_utils.create_new_stage_async()
        self.stage = omni.usd.get_context().get_stage()
        UsdPhysics.Scene.Define(self.stage, "/PhysicsScene")
        success = SimulationManager.switch_physics_engine("newton")
        self.assertTrue(success, "Failed to switch to Newton physics backend")
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Close the stage."""
        await omni.usd.get_context().close_stage_async()

    def _make_body(self, path: str, z: float, mass: float = 1.0) -> Usd.Prim:
        geom = UsdGeom.Cube.Define(self.stage, path)
        geom.GetSizeAttr().Set(0.2)
        prim = geom.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(mass)
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
        return prim

    def _make_revolute(self, path: str, parent: str, child: str) -> Usd.Prim:
        joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
        joint.GetBody0Rel().SetTargets([Sdf.Path(parent)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(child)])
        joint.CreateAxisAttr("Z")
        return joint.GetPrim()

    def _query_articulation_metadata(self, root_path: str) -> tuple[int, list[str], int, list[str]]:
        """Run the articulation property query on ``root_path``.

        Args:
            root_path: USD path passed to the articulation query.

        Returns:
            Tuple of (result code, link paths, total DOF count, DOF joint paths).
        """
        from isaacsim.physics.newton.impl import get_newton_property_query_interface

        captured: dict[str, Any] = {}

        def _report(response: Any) -> None:
            captured["result"] = response.result
            captured["link_paths"] = [link.rigid_body_name for link in response.links]
            captured["dofs"] = sum(link.joint_dof for link in response.links)
            captured["joint_paths"] = [link.joint_name for link in response.links if link.joint_dof]

        get_newton_property_query_interface().query_prim(
            stage_id=stage_utils.get_stage_id(self.stage),
            query_mode=1,  # QUERY_ARTICULATION
            prim_id=PhysicsSchemaTools.sdfPathToInt(Sdf.Path(root_path)),
            articulation_fn=_report,
        )
        return (
            captured.get("result", -1),
            captured.get("link_paths", []),
            captured.get("dofs", 0),
            captured.get("joint_paths", []),
        )

    def _query_dofs(self, root_path: str) -> tuple[int, int, list[str]]:
        """Run the articulation property query and return only DOF metadata.

        Args:
            root_path: USD path passed to the articulation query.

        Returns:
            Tuple of (result code, total DOF count, DOF joint paths).
        """
        result, _link_paths, dofs, joint_paths = self._query_articulation_metadata(root_path)
        return result, dofs, joint_paths

    def _find_articulation_root_path(self, robot_path: str) -> str:
        """Find the authored articulation root under a referenced robot prim.

        Args:
            robot_path: USD path of the robot prim to search under.

        Returns:
            USD path string of the prim that has ArticulationRootAPI.
        """
        robot_prim = self.stage.GetPrimAtPath(robot_path)
        self.assertTrue(robot_prim.IsValid(), f"Missing robot prim {robot_path}")

        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return prim.GetPath().pathString

        self.fail(f"Missing ArticulationRootAPI under {robot_path}")

    async def _assert_pre_play_metadata_matches_live_newton_metadata(
        self,
        usd_path: str,
        prim_path: str,
        articulation_root_path: str | None = None,
        expected_dofs: int | None = None,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata.

        Args:
            usd_path: Path to the USD file to load as a reference.
            prim_path: Stage path at which to add the USD reference.
            articulation_root_path: Optional explicit path to the ArticulationRootAPI prim;
                auto-discovered from ``prim_path`` when not provided.
            expected_dofs: Optional expected DOF count to assert; asserts >0 when not provided.
        """
        stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)
        await wait_for_stage_loading()
        await omni.kit.app.get_app().next_update_async()

        if articulation_root_path is None:
            articulation_root_path = self._find_articulation_root_path(prim_path)
        articulation_root = self.stage.GetPrimAtPath(articulation_root_path)
        self.assertTrue(
            articulation_root.IsValid(),
            f"Missing articulation root {articulation_root_path}",
        )

        pre_result, pre_links, pre_dofs, pre_joints = self._query_articulation_metadata(articulation_root_path)
        self.assertEqual(pre_result, 0, "Pre-play query should succeed")
        if expected_dofs is not None:
            self.assertEqual(pre_dofs, expected_dofs, "Unexpected pre-play DOF count")
        else:
            self.assertGreater(pre_dofs, 0, "Pre-play query should report DOFs")
        self.assertGreater(len(pre_links), 0, "Pre-play query should report links")
        self.assertGreater(len(pre_joints), 0, "Pre-play query should report DOF joints")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        post_result, post_links, post_dofs, post_joints = self._query_articulation_metadata(articulation_root_path)
        timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        self.assertEqual(post_result, 0, "Live Newton query should succeed")
        self.assertEqual(
            pre_links,
            post_links,
            "Pre-play link paths must match live Newton link paths",
        )
        self.assertEqual(
            pre_dofs,
            post_dofs,
            "Pre-play DOF count must match live Newton DOF count",
        )
        self.assertEqual(
            pre_joints,
            post_joints,
            "Pre-play DOF joint paths must match live Newton joint paths",
        )

    async def test_articulation_root_on_fixed_root_joint(self) -> None:
        """Pre-play query resolves DOFs when ArticulationRootAPI is on a fixed root joint.

        Fixed-base robots commonly author ``ArticulationRootAPI`` on a fixed ``root_joint`` prim
        (e.g. UR robots, the Isaac robot schema). The joint prim has no rigid-body subtree, so the
        USD-parse fallback must root ``newton.ModelBuilder.add_usd`` at the enclosing prim. Before
        the fix this returned zero DOFs, leaving tools like the Gain Tuner empty until play.
        """
        self.stage.DefinePrim("/World/Robot", "Xform")
        self._make_body("/World/Robot/base_link", 0.0)
        self._make_body("/World/Robot/link1", 0.4)

        root_joint = UsdPhysics.FixedJoint.Define(self.stage, "/World/Robot/root_joint")
        root_joint.GetBody1Rel().SetTargets([Sdf.Path("/World/Robot/base_link")])
        UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())

        self._make_revolute("/World/Robot/joint1", "/World/Robot/base_link", "/World/Robot/link1")
        await omni.kit.app.get_app().next_update_async()

        result, dofs, joint_paths = self._query_dofs("/World/Robot/root_joint")
        self.assertEqual(result, 0, "Query should succeed (VALID)")
        self.assertEqual(dofs, 1, "The single revolute DOF should be reported")
        self.assertEqual(
            joint_paths,
            ["/World/Robot/joint1"],
            "Fixed root joint must be excluded from DOFs",
        )

    async def test_articulation_root_on_body(self) -> None:
        """Pre-play query resolves DOFs when ArticulationRootAPI is on a root rigid body.

        Floating-base robots commonly author ``ArticulationRootAPI`` on the root rigid body. That body does
        not have to namespace-enclose the articulation's sibling joints and links.
        """
        self.stage.DefinePrim("/World/Robot", "Xform")
        root = self._make_body("/World/Robot/base_link", 0.0)
        UsdPhysics.ArticulationRootAPI.Apply(root)
        self._make_body("/World/Robot/link1", 0.4)

        self._make_revolute("/World/Robot/joint1", "/World/Robot/base_link", "/World/Robot/link1")
        await omni.kit.app.get_app().next_update_async()

        result, dofs, joint_paths = self._query_dofs("/World/Robot/base_link")
        self.assertEqual(result, 0, "Query should succeed (VALID)")
        self.assertEqual(dofs, 1, "The single revolute DOF should be reported")
        self.assertEqual(joint_paths, ["/World/Robot/joint1"], "Revolute DOF path mismatch")

    async def test_flat_sibling_articulations_do_not_leak_dofs(self) -> None:
        """Pre-play query filters to the requested articulation when a common scope contains siblings."""
        self.stage.DefinePrim("/World/FlatRobots", "Xform")
        for name, x_offset in (("A", -0.5), ("B", 0.5)):
            base_path = f"/World/FlatRobots/{name}_base"
            link_path = f"/World/FlatRobots/{name}_link"
            joint_path = f"/World/FlatRobots/{name}_joint"
            root_joint_path = f"/World/FlatRobots/{name}_root_joint"

            self._make_body(base_path, x_offset)
            self._make_body(link_path, x_offset + 0.25)

            root_joint = UsdPhysics.FixedJoint.Define(self.stage, root_joint_path)
            root_joint.GetBody1Rel().SetTargets([Sdf.Path(base_path)])
            UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())
            self._make_revolute(joint_path, base_path, link_path)
        await omni.kit.app.get_app().next_update_async()

        result, dofs, joint_paths = self._query_dofs("/World/FlatRobots/A_root_joint")
        self.assertEqual(result, 0, "Query should succeed (VALID)")
        self.assertEqual(dofs, 1, "Only the queried articulation's DOF should be reported")
        self.assertEqual(
            joint_paths,
            ["/World/FlatRobots/A_joint"],
            "Sibling articulation DOFs must not leak",
        )

    async def test_nested_sibling_articulations_do_not_leak_dofs(self) -> None:
        """Pre-play query filters to the requested articulation when siblings live in their own subtrees.

        Two independent articulations are placed in their own container prims under a shared parent. The
        membership filter must return only the queried articulation's DOFs, never the union of both nested
        subtrees.
        """
        self.stage.DefinePrim("/World/Arena", "Xform")
        for name, x_offset in (("A", -0.5), ("B", 0.5)):
            container = f"/World/Arena/{name}"
            self.stage.DefinePrim(container, "Xform")
            base_path = f"{container}/base"
            link_path = f"{container}/link"
            joint_path = f"{container}/joint"
            root_joint_path = f"{container}/root_joint"

            self._make_body(base_path, x_offset)
            self._make_body(link_path, x_offset + 0.25)

            root_joint = UsdPhysics.FixedJoint.Define(self.stage, root_joint_path)
            root_joint.GetBody1Rel().SetTargets([Sdf.Path(base_path)])
            UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())
            self._make_revolute(joint_path, base_path, link_path)
        await omni.kit.app.get_app().next_update_async()

        result, dofs, joint_paths = self._query_dofs("/World/Arena/A/root_joint")
        self.assertEqual(result, 0, "Query should succeed (VALID)")
        self.assertEqual(dofs, 1, "Only the queried articulation's DOF should be reported")
        self.assertEqual(
            joint_paths,
            ["/World/Arena/A/joint"],
            "Nested sibling articulation DOFs must not leak",
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_ur10e(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for UR10e."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd",
            prim_path="/World/ur10e",
            articulation_root_path="/World/ur10e/root_joint",
            expected_dofs=6,
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_franka(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for Franka."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            prim_path="/World/Franka",
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_kinova_gen3(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for the Kinova Gen3.

        The Gen3 is a fixed-base manipulator shipped as an instanceable asset that authors
        ``ArticulationRootAPI`` on the referenced robot root prim. It covers the container-rooted
        resolution path for a manipulator from a different vendor than the Franka.
        """
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/Kinova/Gen3/gen3n7_instanceable.usd",
            prim_path="/World/KinovaGen3",
            expected_dofs=7,
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_anymal_c(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for ANYmal C."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd",
            prim_path="/World/Anymal",
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_spot(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for the Boston Dynamics Spot.

        Spot is a floating-base quadruped that authors ``ArticulationRootAPI`` on its root body. It
        mirrors the ANYmal C layout from a different vendor, giving a second body-rooted quadruped.
        """
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/BostonDynamics/spot/spot.usd",
            prim_path="/World/Spot",
            expected_dofs=12,
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_g1(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for the Unitree G1.

        The G1 is a floating-base humanoid that authors ``ArticulationRootAPI`` on the root rigid body
        (``pelvis``) rather than a namespace-enclosing container, and has the highest DOF count of the
        covered robots (43). It exercises the body-rooted resolution path at scale.
        """
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/Unitree/G1/g1.usd",
            prim_path="/World/G1",
            expected_dofs=43,
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_h1(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for the Unitree H1.

        H1 is a floating-base humanoid distinct from the G1, authoring ``ArticulationRootAPI`` on the
        root body. It covers a second body-rooted humanoid at a different DOF count.
        """
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/Unitree/H1/h1.usd",
            prim_path="/World/H1",
            expected_dofs=19,
        )

    async def test_pre_play_metadata_matches_live_newton_metadata_for_xarm7(
        self,
    ) -> None:
        """Compare pre-play USD fallback metadata with live Newton metadata for the UFACTORY xArm7.

        The xArm7 asset authors multiple ``ArticulationRootAPI`` prims (a gripper and the 7-DOF arm on
        a fixed root joint). The query targets the arm's root joint explicitly, verifying that a
        specific articulation resolves without leaking bodies from the sibling gripper articulation
        under the same asset.
        """
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/Ufactory/xarm7/xarm7.usd",
            prim_path="/World/xArm7",
            articulation_root_path="/World/xArm7/root_joint",
            expected_dofs=13,
        )

    # TODO(isaac): Closed-loop robots (Digit) under-report DOFs pre-play. Newton's runtime expands the
    # achilles/toe-rod loop closures into additional articulation bodies and joints (live view: 64 DOFs),
    # but neither the UsdPhysics articulation descriptor (reports 43 bodies) nor the pre-play
    # ``newton.ModelBuilder.add_usd`` membership filter captures those loop bodies (pre-play: 50 DOFs).
    # Re-enable this test once the pre-play path reproduces the runtime loop expansion.
    @unittest.skip(
        "Closed-loop DOF under-report: Newton's runtime expands Digit's leg loop closures to 64 DOFs, "
        "but the pre-play UsdPhysics articulation descriptor / add_usd path reports fewer (50); skip "
        "until the pre-play path reproduces the loop expansion."
    )
    async def test_pre_play_metadata_matches_live_newton_metadata_for_digit(
        self,
    ) -> None:
        """Document the known closed-loop pre-play DOF gap for the Agility Digit.

        Digit's parallel leg linkages form closed kinematic loops. The pre-play metadata currently
        differs from the live Newton metadata, so this test is skipped until the pre-play parse
        reproduces Newton's runtime loop expansion.
        """
        assets_root = get_assets_root_path()
        if assets_root is None:
            self.skipTest("Could not find Isaac Sim assets folder")

        await self._assert_pre_play_metadata_matches_live_newton_metadata(
            usd_path=f"{assets_root}/Isaac/Robots/Agility/Digit/digit_v4.usd",
            prim_path="/World/Digit",
            expected_dofs=64,
        )
