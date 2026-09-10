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

"""Tests for stateless scripted teleop motion and gripper definitions."""

import math
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.replicator.teleop import (
    Pose,
    execute_pose_trajectory_async,
    get_floating_gripper_paths,
    get_floating_gripper_spec,
    get_supported_floating_grippers,
    interpolate_pose_targets,
    make_pose,
)
from isaacsim.replicator.teleop import scripted_floating_gripper as scripted_gripper
from isaacsim.replicator.teleop import (
    set_debug_grasp_async,
    wait_for_controllers_running_async,
)
from isaacsim.replicator.teleop.scripted_floating_gripper import (
    _create_floating_rigid_root,
    _disable_gravity,
)
from pxr import PhysxSchema, UsdPhysics


class TestScriptedMotion(omni.kit.test.AsyncTestCase):
    """Validate pose interpolation and tolerance-gated execution."""

    async def test_controller_wait_checks_final_allowed_update(self) -> None:
        """Readiness reached on the final update is reported without an extra frame."""
        running = {"left": False, "right": False}
        updates = 0
        controller = MagicMock()
        controller.is_running.side_effect = lambda side: running[side]

        async def update_async() -> None:
            nonlocal updates
            updates += 1
            if updates == 2:
                running.update(left=True, right=True)

        reached = await wait_for_controllers_running_async(
            {"left": controller, "right": controller},
            update_async,
            max_steps=2,
        )

        self.assertTrue(reached)
        self.assertEqual(updates, 2)

    async def test_debug_grasp_sets_each_side_and_settles(self) -> None:
        """A grasp command writes one trigger per side and advances exact steps."""
        manager = MagicMock()
        updates = 0

        async def update_async() -> None:
            nonlocal updates
            updates += 1

        await set_debug_grasp_async(
            manager,
            ("left", "right", "left"),
            True,
            update_async,
            settle_steps=3,
        )

        self.assertEqual(
            manager.set_debug_trigger.call_args_list,
            [call("left", 1.0), call("right", 1.0)],
        )
        self.assertEqual(updates, 3)

    def test_interpolation_uses_linear_position_and_slerp(self) -> None:
        """The middle sample has the expected position and half rotation."""
        samples = interpolate_pose_targets(
            {"right": make_pose((0.0, 0.0, 0.0))},
            {"right": make_pose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))},
            sample_count=3,
        )

        self.assertEqual(len(samples), 3)
        np.testing.assert_allclose(samples[1]["right"][0], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            np.abs(samples[1]["right"][1]),
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        )

    async def test_execution_waits_for_tolerance_before_advancing(self) -> None:
        """A target is held until the measured pose reaches tolerance."""
        trajectory = [
            {"right": make_pose((0.5, 0.0, 0.0))},
            {"right": make_pose((1.0, 0.0, 0.0))},
        ]
        measured = {"right": make_pose((0.0, 0.0, 0.0))}
        commanded = trajectory[0]
        applied: list[tuple[float, float, float]] = []

        def apply_targets(targets: Mapping[str, Pose]) -> None:
            nonlocal commanded
            commanded = targets
            applied.append(targets["right"][0])

        def read_poses() -> Mapping[str, Pose]:
            return measured

        async def update_async() -> None:
            current = np.asarray(measured["right"][0])
            target = np.asarray(commanded["right"][0])
            next_position = current + np.clip(target - current, -0.25, 0.25)
            measured["right"] = make_pose(next_position)

        result = await execute_pose_trajectory_async(
            trajectory,
            apply_targets,
            read_poses,
            update_async,
            position_tolerance=1e-6,
            orientation_tolerance=1e-6,
            max_steps_per_target=3,
        )

        self.assertTrue(result["reached"])
        self.assertEqual(result["completed_samples"], 2)
        self.assertEqual(result["total_steps"], 4)
        self.assertEqual(applied, [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0)])

    async def test_execution_reports_timeout(self) -> None:
        """A target that does not converge returns a diagnostic result."""
        target = {"right": make_pose((1.0, 0.0, 0.0))}

        async def update_async() -> None:
            return None

        result = await execute_pose_trajectory_async(
            [target],
            lambda targets: None,
            lambda: {"right": make_pose((0.0, 0.0, 0.0))},
            update_async,
            position_tolerance=0.01,
            max_steps_per_target=2,
        )

        self.assertFalse(result["reached"])
        self.assertEqual(result["completed_samples"], 0)
        self.assertEqual(result["total_steps"], 2)
        self.assertAlmostEqual(result["position_error"], 1.0)

    def test_floating_grippers_are_config_defined_and_trigger_driven(self) -> None:
        """Current grippers are discovered from YAML without retargeting settings."""
        supported = get_supported_floating_grippers()
        self.assertIn("xarm", supported)
        self.assertIn("dex3", supported)
        for name in ("xarm", "dex3"):
            spec = get_floating_gripper_spec(name)
            self.assertNotIn("drive_mode", spec["grasp_controller"])
            self.assertNotIn("retargeter_kind", spec["grasp_controller"])
            paths = get_floating_gripper_paths("/World/Teleop", "right", spec)
            self.assertTrue(paths["base_link"].endswith(spec["base_link_name"]))
            self.assertTrue(paths["tcp"].endswith(f"/{spec['tcp']['parent_link']}/{spec['tcp']['prim_name']}"))

        xarm = get_floating_gripper_spec("xarm")
        self.assertEqual(xarm["tcp"]["translation"], (0.0, 0.0, 0.15))

        dex3 = get_floating_gripper_spec("dex3")
        self.assertEqual(dex3["tcp"]["translation"], (0.08, 0.05, 0.0))

    def test_new_floating_gripper_is_discovered_from_yaml_only(self) -> None:
        """A complete third definition needs no Python registry change."""
        definition = """\
name: test_gripper
asset_path: /Isaac/Test/test_gripper.usd
root_xform_name: test_root
rigid_root_name: floating_handle
gripper_prim_name: gripper
base_link_name: palm
root_joint_name: root_joint
tcp:
  parent_link: palm
  prim_name: Tcp
  translation: [0.0, 0.0, 0.17]
floating_controller:
  position_kp: 25.0
  position_kd: 0.6
  orientation_kp: 21.0
  orientation_kd: 0.3
  rotation_offset_degrees: [0.0, 90.0, 0.0]
grasp_controller:
  config_path: builtin://test_gripper
"""
        with TemporaryDirectory() as temp_dir:
            definitions_dir = Path(temp_dir)
            (definitions_dir / "test_gripper.yaml").write_text(definition, encoding="utf-8")
            with patch.object(
                scripted_gripper,
                "_get_floating_grippers_dir",
                return_value=definitions_dir,
            ):
                self.assertEqual(get_supported_floating_grippers(), ("test_gripper",))
                spec = get_floating_gripper_spec("test_gripper")
                paths = get_floating_gripper_paths("/World/Teleop", "left", spec)

        self.assertEqual(spec["tcp"]["translation"], (0.0, 0.0, 0.17))
        self.assertEqual(spec["floating_controller"]["rotation_offset_degrees"], (0.0, 90.0, 0.0))
        self.assertEqual(
            paths["base_link"],
            "/World/Teleop/gripper_origin_xform/left_test_root/gripper/palm",
        )
        self.assertEqual(
            paths["tcp"],
            "/World/Teleop/gripper_origin_xform/left_test_root/gripper/palm/Tcp",
        )


class TestScriptedFloatingGripperPhysics(omni.kit.test.AsyncTestCase):
    """Characterize stopped-stage floating-root and gravity authoring."""

    async def setUp(self) -> None:
        """Create an empty stage."""
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World", "Xform")

    async def tearDown(self) -> None:
        """Release the stage."""
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            stage_utils.close_stage()
            await app_utils.update_app_async()

    def test_create_floating_rigid_root_authors_expected_physics_state(self) -> None:
        """The floating root is dynamic, non-colliding, and an articulation root."""
        path = "/World/FloatingRoot"
        _create_floating_rigid_root(path)

        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI))
        self.assertTrue(prim.HasAPI(UsdPhysics.ArticulationRootAPI))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxArticulationAPI))
        self.assertFalse(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
        self.assertFalse(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        self.assertFalse(PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get())

    def test_disable_gravity_updates_all_rigid_descendants(self) -> None:
        """Gravity is disabled on the root and nested rigid bodies without playing physics."""
        stage_utils.define_prim("/World/GripperRig", "Xform")
        stage_utils.define_prim("/World/GripperRig/Branch", "Xform")
        stage_utils.define_prim("/World/GripperRig/Branch/Body", "Xform")
        rigid_paths = ("/World/GripperRig", "/World/GripperRig/Branch/Body")
        rigid_prims = RigidPrim(rigid_paths)
        for prim in rigid_prims.prims:
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

        stage = stage_utils.get_current_stage()
        _disable_gravity(stage, "/World/GripperRig")

        for path in rigid_paths:
            api = PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(path))
            self.assertTrue(api.GetDisableGravityAttr().Get(), path)
