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

"""Unit tests for controller grasp retargeting helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.timeline
from isaacsim.replicator.teleop import (
    GraspConfig,
    GraspController,
    GraspDriveMode,
    GraspRetargeterKind,
    JointMapping,
    TeleopManager,
    compute_retargeted_joint_targets,
    compute_trihand_joint_values,
    get_builtin_teleop_profiles_dir,
    get_floating_gripper_spec,
    load_grasp_config,
    load_teleop_profile,
    map_trihand_to_joint_targets,
    parse_grasp_drive_mode,
    parse_grasp_retargeter_kind,
    parse_joint_aliases,
    scan_teleop_profiles,
    validate_trihand_joint_aliases,
)
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


@dataclass
class _FakeInputs:
    trigger_value: float = 0.0
    squeeze_value: float = 0.0


@dataclass
class _FakeController:
    inputs: _FakeInputs


class _RoutingGraspController:
    """Record manager dispatch without requiring a live OpenXR session."""

    def __init__(self) -> None:
        self.is_enabled = True
        self.config = GraspConfig(
            joints=[
                JointMapping("right_hand_index_0_joint", target_range=(0.0, 90.0)),
                JointMapping("right_hand_middle_0_joint", target_range=(0.0, 60.0)),
            ]
        )
        self.scalar_inputs: list[tuple[str, float]] = []
        self.joint_targets: list[tuple[str, dict[str, float]]] = []

    def is_side_tracking_enabled(self, _side: str) -> bool:
        return True

    def get_side_drive_settings(self, side: str) -> tuple[str, str | None, dict[str, str]]:
        if side == "right":
            return (
                "retargeted",
                "trihand",
                {
                    "index_proximal": "right_hand_index_0_joint",
                    "middle_proximal": "right_hand_middle_0_joint",
                },
            )
        return "trigger", None, {}

    def get_side_config(self, _side: str) -> GraspConfig:
        return self.config

    def set_input(self, side: str, value: float) -> None:
        self.scalar_inputs.append((side, value))

    def set_joint_targets(self, side: str, targets: dict[str, float]) -> None:
        self.joint_targets.append((side, dict(targets)))


class TestRetargetingGrasp(omni.kit.test.AsyncTestCase):
    """Verify grasp retargeting math and profile parsers."""

    async def test_parse_helpers(self) -> None:
        """Profile/UI strings normalize to grasp drive enums."""
        self.assertEqual(parse_grasp_drive_mode("retargeted"), GraspDriveMode.RETARGETED)
        self.assertEqual(parse_grasp_drive_mode(None), GraspDriveMode.TRIGGER)
        self.assertEqual(parse_grasp_retargeter_kind("trihand"), GraspRetargeterKind.TRIHAND)
        self.assertIsNone(parse_grasp_retargeter_kind("none"))
        self.assertEqual(
            parse_joint_aliases({"index_proximal": "right_hand_index_0_joint"}),
            {"index_proximal": "right_hand_index_0_joint"},
        )

    async def test_trihand_maps_trigger_and_squeeze(self) -> None:
        """TriHand fallback produces seven joint values from controller analogs."""
        open_vals = compute_trihand_joint_values(0.0, 0.0, hand_side="right")
        closed_vals = compute_trihand_joint_values(1.0, 1.0, hand_side="right")
        self.assertEqual(len(open_vals), 7)
        self.assertEqual(len(closed_vals), 7)
        self.assertAlmostEqual(open_vals[3], 0.0)
        self.assertAlmostEqual(closed_vals[3], 1.0)
        self.assertAlmostEqual(closed_vals[5], 1.0)

    async def test_trihand_targets_profile_joint_aliases(self) -> None:
        """TriHand semantic values map through profile-provided joint aliases."""
        config = GraspConfig(
            joints=[
                JointMapping("right_hand_index_0_joint", target_range=(0.0, 90.0)),
                JointMapping("right_hand_middle_0_joint", target_range=(0.0, 45.0)),
            ]
        )
        aliases = {
            "index_proximal": "right_hand_index_0_joint",
            "middle_proximal": "right_hand_middle_0_joint",
        }
        trihand_values = compute_trihand_joint_values(1.0, 0.5, hand_side="right")
        targets = map_trihand_to_joint_targets(trihand_values, config, aliases)
        self.assertAlmostEqual(targets["right_hand_index_0_joint"], 90.0)
        self.assertAlmostEqual(targets["right_hand_middle_0_joint"], 22.5)

    async def test_trihand_requires_joint_aliases(self) -> None:
        """Retargeted grasp returns no targets when aliases are missing."""
        config = GraspConfig(joints=[JointMapping("drive_joint", target_range=(0.0, 48.0))])
        ctrl = _FakeController(inputs=_FakeInputs(trigger_value=1.0, squeeze_value=1.0))
        targets = compute_retargeted_joint_targets(
            retargeter_kind=GraspRetargeterKind.TRIHAND,
            controller_snapshot=ctrl,
            grasp_config=config,
            hand_side="right",
            joint_aliases=None,
        )
        self.assertEqual(targets, {})

    async def test_compute_retargeted_joint_targets(self) -> None:
        """End-to-end retargeting uses controller snapshots and profile aliases."""
        config = GraspConfig(joints=[JointMapping("drive_joint", target_range=(0.0, 48.0))])
        ctrl = _FakeController(inputs=_FakeInputs(trigger_value=1.0, squeeze_value=0.0))
        targets = compute_retargeted_joint_targets(
            retargeter_kind=GraspRetargeterKind.TRIHAND,
            controller_snapshot=ctrl,
            grasp_config=config,
            hand_side="right",
            joint_aliases={"index_proximal": "drive_joint"},
        )
        self.assertIn("drive_joint", targets)
        self.assertAlmostEqual(targets["drive_joint"], 48.0, places=1)

    async def test_invalid_hand_side_is_rejected(self) -> None:
        """TriHand mapping rejects an unknown hand side."""
        with self.assertRaisesRegex(ValueError, "hand_side"):
            compute_trihand_joint_values(0.0, 0.0, hand_side="center")

    async def test_alias_validation_reports_unknown_and_missing_targets(self) -> None:
        """Alias validation reports semantic, config, and USD mismatches."""
        config = GraspConfig(joints=[JointMapping("configured_joint")])
        errors = validate_trihand_joint_aliases(
            config,
            {
                "index_proximal": "missing_joint",
                "ring_proximal": "configured_joint",
            },
            available_joint_names={"configured_joint"},
        )
        message = " ".join(errors)
        self.assertIn("Unknown TriHand semantic alias(es): ring_proximal", message)
        self.assertIn("missing from grasp config: missing_joint", message)
        self.assertIn("not controllable below grasp prim: missing_joint", message)

    async def test_manager_routes_synthetic_trigger_and_squeeze(self) -> None:
        """Manager dispatches trigger mode and retargeted mode independently."""
        manager = TeleopManager()
        grasp = _RoutingGraspController()
        manager.set_grasp_controller(grasp)  # type: ignore[arg-type]
        try:
            manager._update_grasp_inputs(  # noqa: SLF001
                _FakeController(inputs=_FakeInputs(trigger_value=0.25, squeeze_value=0.0)),
                _FakeController(inputs=_FakeInputs(trigger_value=1.0, squeeze_value=0.5)),
            )
        finally:
            manager.set_grasp_controller(None)
            manager.destroy()

        self.assertEqual(grasp.scalar_inputs, [("left", 0.25)])
        self.assertEqual(len(grasp.joint_targets), 1)
        side, targets = grasp.joint_targets[0]
        self.assertEqual(side, "right")
        self.assertAlmostEqual(targets["right_hand_index_0_joint"], 90.0)
        self.assertAlmostEqual(targets["right_hand_middle_0_joint"], 30.0)


class TestRetargetingGraspController(omni.kit.test.AsyncTestCase):
    """Verify explicit targets reach USD drives and live articulation physics."""

    async def setUp(self) -> None:
        """Create an empty stage for each controller test."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World", "Xform")
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Stop physics and release the procedural stage."""
        omni.timeline.get_timeline_interface().stop()
        await app_utils.update_app_async()
        stage_utils.close_stage()
        await app_utils.update_app_async()

    @staticmethod
    def _configure_retargeted(controller: GraspController, root_path: str, joint_name: str) -> None:
        config = GraspConfig(joints=[JointMapping(joint_name, target_range=(0.0, 90.0))])
        configured = controller.configure(
            root_path,
            "right",
            config,
            drive_mode="retargeted",
            retargeter_kind="trihand",
            joint_aliases={"index_proximal": joint_name},
        )
        if not configured:
            raise AssertionError("Procedural retargeted grasp controller did not configure")
        controller.set_side_tracking_enabled("right", True)

    async def test_explicit_target_writes_drive_api(self) -> None:
        """A standalone hand receives retargeted targets through USD DriveAPI."""
        stage = stage_utils.get_current_stage()
        root_path = "/World/StandaloneHand"
        joint_name = "right_hand_index_0_joint"
        stage_utils.define_prim(root_path, "Xform")
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/{joint_name}")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTargetPositionAttr(0.0)

        controller = GraspController()
        self._configure_retargeted(controller, root_path, joint_name)
        target_degrees = 30.0
        controller.set_joint_targets("right", {joint_name: target_degrees})

        self.assertAlmostEqual(float(drive.GetTargetPositionAttr().Get()), target_degrees)
        self.assertAlmostEqual(
            controller._to_articulation_position(  # noqa: SLF001
                joint_path=f"{root_path}/{joint_name}", configured_target=target_degrees
            ),
            math.radians(target_degrees),
        )

    async def test_retargeted_target_moves_procedural_articulation(self) -> None:
        """A retargeted degree target converges to the matching physical angle."""
        stage = stage_utils.get_current_stage()
        scene = UsdPhysics.Scene(stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene"))
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(0.0)

        root_path = "/World/ProceduralHand"
        joint_name = "right_hand_index_0_joint"
        root = UsdGeom.Xform.Define(stage, root_path)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(5.0)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        finger = UsdGeom.Cube.Define(stage, f"{root_path}/Finger")
        finger.CreateSizeAttr(0.2)
        finger.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(finger.GetPrim())
        UsdPhysics.MassAPI.Apply(finger.GetPrim()).CreateMassAttr(0.1)

        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/{joint_name}")
        joint.CreateBody0Rel().SetTargets([root.GetPath()])
        joint.CreateBody1Rel().SetTargets([finger.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-0.1, 0.0, 0.0))
        joint.CreateLowerLimitAttr(-90.0)
        joint.CreateUpperLimitAttr(90.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(1000.0)
        drive.CreateDampingAttr(100.0)
        drive.CreateMaxForceAttr(1000.0)
        drive.CreateTargetPositionAttr(0.0)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        await app_utils.update_app_async(steps=5)

        controller = GraspController()
        self._configure_retargeted(controller, root_path, joint_name)
        state = controller._side("right")  # noqa: SLF001
        self.assertIsNotNone(state.articulation)
        assert state.articulation is not None
        dof_index = state.articulation.dof_names.index(joint_name)

        target_degrees = 30.0
        target_radians = math.radians(target_degrees)
        controller.set_joint_targets("right", {joint_name: target_degrees})

        measured = 0.0
        for _ in range(240):
            await app_utils.update_app_async()
            values = state.articulation.get_dof_positions(dof_indices=dof_index).numpy()
            measured = float(values.reshape(-1)[0])
            if abs(measured - target_radians) < 0.03:
                break

        self.assertGreater(measured, 0.1, "Retargeted grasp should produce measurable physical joint motion")
        self.assertAlmostEqual(measured, target_radians, delta=0.05)

    async def test_trigger_target_moves_procedural_articulation(self) -> None:
        """A trigger YAML target converts degrees to articulation radians."""
        stage = stage_utils.get_current_stage()
        scene = UsdPhysics.Scene(stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene"))
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(0.0)

        root_path = "/World/TriggerHand"
        joint_name = "trigger_hand_index_joint"
        root = UsdGeom.Xform.Define(stage, root_path)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(5.0)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        finger = UsdGeom.Cube.Define(stage, f"{root_path}/Finger")
        finger.CreateSizeAttr(0.2)
        finger.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(finger.GetPrim())
        UsdPhysics.MassAPI.Apply(finger.GetPrim()).CreateMassAttr(0.1)

        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/{joint_name}")
        joint.CreateBody0Rel().SetTargets([root.GetPath()])
        joint.CreateBody1Rel().SetTargets([finger.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-0.1, 0.0, 0.0))
        joint.CreateLowerLimitAttr(-90.0)
        joint.CreateUpperLimitAttr(90.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(1000.0)
        drive.CreateDampingAttr(100.0)
        drive.CreateMaxForceAttr(1000.0)
        drive.CreateTargetPositionAttr(0.0)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        await app_utils.update_app_async(steps=5)

        controller = GraspController()
        target_degrees = 30.0
        configured = controller.configure(
            root_path,
            "right",
            GraspConfig(joints=[JointMapping(joint_name, target_range=(0.0, target_degrees))]),
        )
        self.assertTrue(configured)
        controller.set_side_tracking_enabled("right", True)
        state = controller._side("right")  # noqa: SLF001
        self.assertIsNotNone(state.articulation)
        self.assertIn(f"{root_path}/{joint_name}", state.art_joint_map)
        assert state.articulation is not None
        dof_index = state.articulation.dof_names.index(joint_name)

        controller.set_input("right", 1.0)

        target_radians = math.radians(target_degrees)
        measured = 0.0
        for _ in range(240):
            await app_utils.update_app_async()
            values = state.articulation.get_dof_positions(dof_indices=dof_index).numpy()
            measured = float(values.reshape(-1)[0])
            if abs(measured - target_radians) < 0.03:
                break

        self.assertGreater(measured, 0.1, "Trigger grasp should produce measurable physical joint motion")
        self.assertAlmostEqual(measured, target_radians, delta=0.05)

    async def test_retargeted_profile_matches_real_dex3_asset(self) -> None:
        """The built-in profile resolves and writes every aliased Dex3 drive."""
        profiles = dict(scan_teleop_profiles(get_builtin_teleop_profiles_dir()))
        profile, errors = load_teleop_profile(profiles["floating_xarm_dex3_retargeted"])
        self.assertEqual(errors, [])
        self.assertIsNotNone(profile)
        assert profile is not None
        side_profile = profile.grasp.right

        assets_root = await get_assets_root_path_async()
        self.assertTrue(assets_root, "Isaac assets root is required for the Dex3 asset smoke test")
        spec = get_floating_gripper_spec("dex3")
        parent_path = str(Sdf.Path(side_profile.prim_path).GetParentPath())
        stage_utils.define_prim(parent_path, "Xform")
        stage_utils.add_reference_to_stage(f"{assets_root}{spec['asset_path']}", side_profile.prim_path)
        for _ in range(300):
            await app_utils.update_app_async()
            if not stage_utils.is_stage_loading():
                break
        self.assertFalse(stage_utils.is_stage_loading(), "Timed out loading the Dex3 asset")

        config, config_errors = load_grasp_config(side_profile.config_path)
        self.assertEqual(config_errors, [])
        self.assertIsNotNone(config)
        assert config is not None

        controller = GraspController()
        validation = controller.validate_prim(side_profile.prim_path)
        self.assertTrue(validation.is_valid, validation.errors)
        controllable_names = {Sdf.Path(path).name for path in validation.drive_joint_paths}
        alias_targets = set(side_profile.joint_aliases.values())
        config_names = {mapping.name for mapping in config.joints}
        self.assertEqual(alias_targets - config_names, set())
        self.assertEqual(alias_targets - controllable_names, set())

        configured = controller.configure(
            side_profile.prim_path,
            "right",
            config,
            drive_mode=side_profile.drive_mode,
            retargeter_kind=side_profile.retargeter_kind,
            joint_aliases=side_profile.joint_aliases,
        )
        self.assertTrue(configured)
        controller.set_side_tracking_enabled("right", True)
        targets = compute_retargeted_joint_targets(
            retargeter_kind=GraspRetargeterKind.TRIHAND,
            controller_snapshot=_FakeController(inputs=_FakeInputs(trigger_value=1.0, squeeze_value=1.0)),
            grasp_config=config,
            hand_side="right",
            joint_aliases=side_profile.joint_aliases,
        )
        self.assertEqual(set(targets), alias_targets)
        controller.set_joint_targets("right", targets)

        stage = stage_utils.get_current_stage()
        written_targets: dict[str, float] = {}
        for joint_path in validation.drive_joint_paths:
            joint_name = Sdf.Path(joint_path).name
            if joint_name not in targets:
                continue
            prim = stage.GetPrimAtPath(joint_path)
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            written_targets[joint_name] = float(drive.GetTargetPositionAttr().Get())

        self.assertEqual(set(written_targets), alias_targets)
        for joint_name, expected in targets.items():
            self.assertAlmostEqual(written_targets[joint_name], expected, places=4)
        self.assertTrue(any(abs(value) > 1.0 for value in written_targets.values()))
