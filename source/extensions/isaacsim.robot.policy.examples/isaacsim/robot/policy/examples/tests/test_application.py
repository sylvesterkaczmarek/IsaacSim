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

"""Live-physics checks for the articulation and motion-generation state bridge."""

from __future__ import annotations

import asyncio
import math

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as motion_generation
import numpy as np
import omni.kit.app
import omni.kit.test
import omni.timeline
import omni.usd
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.policy.examples import application
from isaacsim.robot.policy.examples.interactive.utils import (
    restore_physics_simulation_state,
    snapshot_physics_simulation_state,
)
from pxr import Gf, UsdGeom, UsdPhysics

_JOINT_NAMES = ("joint_1", "joint_2", "joint_3")
_LINK_SPACING = 0.3


def _author_robot(prim_path: str, *, fixed_base: bool = True, yaw_degrees: float = 0.0) -> str:
    """Author a geometry-free three-DOF articulation for state-bridge tests.

    Args:
        prim_path: Stage path to author the articulation at.
        fixed_base: Author the root as a fixed base rather than a free body.
        yaw_degrees: Root yaw applied about +Z, in degrees.

    Returns:
        The articulation root prim path.
    """
    stage = stage_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.6))
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_degrees).GetQuat()
    imaginary = rotation.GetImaginary()
    root.AddOrientOp().Set(
        Gf.Quatf(
            float(rotation.GetReal()),
            Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        )
    )

    bodies = []
    for index, name in enumerate(("base", "link_1", "link_2", "link_3")):
        body = UsdGeom.Xform.Define(stage, f"{prim_path}/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(index * _LINK_SPACING, 0.0, 0.0))
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        mass = UsdPhysics.MassAPI.Apply(body_prim)
        mass.CreateMassAttr(1.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.05, 0.05, 0.05))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        bodies.append(body_prim.GetPath())

    UsdGeom.Scope.Define(stage, f"{prim_path}/joints")
    if fixed_base:
        root_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/joints/root_joint")
        root_joint.CreateBody1Rel().SetTargets([bodies[0]])

    for index, joint_name in enumerate(_JOINT_NAMES):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/joints/{joint_name}")
        joint.CreateBody0Rel().SetTargets([bodies[index]])
        joint.CreateBody1Rel().SetTargets([bodies[index + 1]])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(_LINK_SPACING / 2.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-_LINK_SPACING / 2.0, 0.0, 0.0))
        joint.CreateLowerLimitAttr(-170.0)
        joint.CreateUpperLimitAttr(170.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(200.0)
        drive.CreateDampingAttr(10.0)
        drive.CreateMaxForceAttr(1000.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)

    articulation_prim = stage.GetPrimAtPath(prim_path) if fixed_base else stage.GetPrimAtPath(bodies[0])
    UsdPhysics.ArticulationRootAPI.Apply(articulation_prim)
    return str(articulation_prim.GetPath())


class TestRobotStateBridge(omni.kit.test.AsyncTestCase):
    """Check the bridge behaviors that are not covered by complete robot rollouts."""

    async def setUp(self) -> None:
        """Create a zero-gravity CPU stage."""
        self._initial_device, self._initial_fabric = snapshot_physics_simulation_state()
        await stage_utils.create_new_stage_async()
        scene = stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
        UsdPhysics.Scene(scene).CreateGravityMagnitudeAttr().Set(0.0)
        SimulationManager.set_physics_sim_device("cpu")
        SimulationManager.set_physics_dt(1.0 / 200.0)
        self._timeline = omni.timeline.get_timeline_interface()
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Stop simulation and restore its process-wide state."""
        await omni.kit.app.get_app().next_update_async()
        self._timeline.stop()
        restore_physics_simulation_state(self._initial_device, self._initial_fabric)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    async def _initialize(self, prim_path: str) -> Articulation:
        articulation = Articulation(prim_path)
        self._timeline.play()
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()
            if articulation.is_physics_tensor_entity_valid():
                return articulation
        self.fail(f"Articulation at {prim_path!r} did not initialize.")

    async def _assert_partial_write(
        self,
        field: str,
        sentinels: tuple[float, float, float],
        commanded: tuple[float, float],
    ) -> None:
        articulation = await self._initialize(_author_robot("/World/Robot"))
        self.assertEqual(tuple(articulation.dof_names), _JOINT_NAMES)
        sentinel_row = [list(sentinels)]

        if field == "positions":
            articulation.set_dof_position_targets(sentinel_row)
        else:
            articulation.set_dof_efforts(sentinel_row)

        values = wp.array(commanded, dtype=wp.float32, device="cpu")
        joints = motion_generation.JointState.from_name(
            list(articulation.dof_names),
            **{field: (["joint_1", "joint_3"], values)},
        )
        application.apply_robot_state(articulation, motion_generation.RobotState(joints=joints))

        if field == "positions":
            actual = articulation.get_dof_position_targets().numpy()[0]
        else:
            actual = articulation.get_dof_efforts().numpy()[0]
        np.testing.assert_allclose(actual, [commanded[0], sentinels[1], commanded[1]], atol=1e-5)

    async def test_partial_position_write_preserves_unrelated_dof(self) -> None:
        """A named position command does not overwrite an attached subsystem's DOF."""
        await self._assert_partial_write("positions", (0.11, 0.22, 0.33), (-0.4, 0.5))

    async def test_partial_effort_write_preserves_unrelated_dof(self) -> None:
        """A named effort command does not overwrite an attached subsystem's DOF."""
        await self._assert_partial_write("efforts", (0.5, 1.5, 2.5), (-3.0, 4.0))

    async def test_root_orientation_is_wxyz(self) -> None:
        """Root orientation crosses the bridge in motion-generation WXYZ order."""
        yaw_degrees = 60.0
        articulation = await self._initialize(_author_robot("/World/Robot", fixed_base=False, yaw_degrees=yaw_degrees))
        quaternion = application.read_robot_state(articulation).root.orientation.numpy()
        half_yaw = math.radians(yaw_degrees) / 2.0
        expected = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
        np.testing.assert_allclose(quaternion, expected, atol=1e-4)
