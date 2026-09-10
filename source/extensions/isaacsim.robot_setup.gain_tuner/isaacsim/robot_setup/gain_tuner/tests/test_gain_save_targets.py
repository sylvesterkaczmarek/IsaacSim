# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for save-target resolution and MuJoCo/Newton write plans."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.usd_layer_utils import get_layer_save_identifier
from pxr import Sdf, Usd, UsdPhysics

from .common import (
    IMPORTER_JOINT_PATH,
    IMPORTER_ROBOT_PATH,
    write_importer_asset_layout,
    write_scene_referencing,
)


class TestGainSaveTargets(omni.kit.test.AsyncTestCase):
    """Save-target layer resolution and mjc/Newton mirror logic (no simulation)."""

    _JOINT = "/World/Robot/joint0"
    _ACTUATOR = "/World/Robot/Actuators/act0"
    _MJC_ACTUATOR = "/World/Robot/Physics/joint0_actuator"

    @staticmethod
    def _write_physics_folder(tmp_dir: str, *, with_mjc: bool = False) -> str:
        """Create a ``payloads/Physics/`` folder with physics/physx/mujoco layers.

        The DriveAPI stiffness/damping is authored in ``physics.usda``; when
        ``with_mjc`` is set, an ``MjcActuator`` with mjc gain params targeting the
        joint is authored in ``mujoco.usda``.  Returns the root layer path.
        """
        physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
        os.makedirs(physics_dir, exist_ok=True)
        physics_path = os.path.join(physics_dir, "physics.usda")
        physx_path = os.path.join(physics_dir, "physx.usda")
        mujoco_path = os.path.join(physics_dir, "mujoco.usda")
        root_path = os.path.join(tmp_dir, "robot.usda")

        physics_stage = Usd.Stage.CreateNew(physics_path)
        UsdPhysics.RevoluteJoint.Define(physics_stage, TestGainSaveTargets._JOINT)
        drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(TestGainSaveTargets._JOINT), "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        physics_stage.Save()

        Sdf.Layer.CreateNew(physx_path).Save()

        mujoco_layer = Sdf.Layer.CreateNew(mujoco_path)
        mujoco_layer.subLayerPaths.append(physics_path)
        mujoco_stage = Usd.Stage.Open(mujoco_path)
        if with_mjc:
            actuator = mujoco_stage.DefinePrim(TestGainSaveTargets._MJC_ACTUATOR, "MjcActuator")
            actuator.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0] * 10)
            actuator.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0] * 10)
            actuator.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.String).Set("fixed")
            actuator.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")
            actuator.CreateRelationship("mjc:target").AddTarget(Sdf.Path(TestGainSaveTargets._JOINT))
        mujoco_stage.Save()

        root_layer = Sdf.Layer.CreateNew(root_path)
        root_layer.subLayerPaths.append(physics_path)
        root_layer.Save()
        return root_path

    async def test_list_save_targets_enumerates_physics_folder(self) -> None:
        """Candidate save targets enumerate the Physics folder with physics.usda default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir)
            stage = Usd.Stage.Open(root_path)
            attr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(self._JOINT), "angular").GetStiffnessAttr()
            options = gain_tuner.list_gain_save_target_layers(attr)
            self.assertTrue(options.resolved)
            self.assertEqual(set(options.display_names), {"physics.usda", "physx.usda", "mujoco.usda"})
            self.assertEqual(options.default.display_name, "physics.usda")
            # Default (neutral) layer is first in presentation order.
            self.assertEqual(options.display_names[0], "physics.usda")
            # Display-name -> identifier reverse lookup (used to route a save to the
            # layer the user picked in the combo); unknown names resolve to None.
            self.assertEqual(options.identifier_for_display_name("physics.usda"), options.default.identifier)
            self.assertIsNone(options.identifier_for_display_name("does_not_exist.usda"))

    async def test_list_save_targets_default_physics_with_overlay_opinion(self) -> None:
        """The neutral base physics.usda stays the default even with an overlay opinion.

        ``get_defining_layer`` resolves the base (weakest) authoring layer, so a
        stronger overlay opinion (e.g. a live edit on ``physx.usda``) does not move
        the resolved default away from the neutral ``physics.usda`` layer.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir)
            physx_path = os.path.join(tmp_dir, "payloads", "Physics", "physx.usda")
            physx_layer = Sdf.Layer.FindOrOpen(physx_path)
            physx_layer.subLayerPaths.append(os.path.join(tmp_dir, "payloads", "Physics", "physics.usda"))
            physx_stage = Usd.Stage.Open(physx_path)
            physx_stage.SetEditTarget(physx_stage.GetEditTargetForLocalLayer(physx_layer))
            UsdPhysics.DriveAPI(physx_stage.GetPrimAtPath(self._JOINT), "angular").GetStiffnessAttr().Set(250.0)
            attr = UsdPhysics.DriveAPI(physx_stage.GetPrimAtPath(self._JOINT), "angular").GetStiffnessAttr()
            # The base defining layer is physics.usda even though physx.usda has a stronger opinion.
            self.assertTrue(gain_tuner.get_defining_layer(attr).identifier.endswith("physics.usda"))
            options = gain_tuner.list_gain_save_target_layers(attr)
            self.assertEqual(options.default.display_name, "physics.usda")
            self.assertEqual(set(options.display_names), {"physics.usda", "physx.usda", "mujoco.usda"})

    @staticmethod
    def _write_drive_outside_physics_folder(tmp_dir: str) -> str:
        """Create a Physics folder whose DriveAPI gains are authored on the root layer.

        ``payloads/Physics/`` holds ``physics.usda`` (defines the joint, no drive
        gains), an empty ``physx.usda``, and ``mujoco.usda``; the root ``robot.usda``
        sublayers ``physics.usda`` and authors the DriveAPI stiffness/damping
        itself.  The DriveAPI's only (and therefore defining) opinion lives outside
        the Physics folder, exercising the stage-based folder discovery.  Returns
        the root layer path.
        """
        physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
        os.makedirs(physics_dir, exist_ok=True)
        physics_path = os.path.join(physics_dir, "physics.usda")
        physx_path = os.path.join(physics_dir, "physx.usda")
        mujoco_path = os.path.join(physics_dir, "mujoco.usda")
        root_path = os.path.join(tmp_dir, "robot.usda")

        physics_stage = Usd.Stage.CreateNew(physics_path)
        UsdPhysics.RevoluteJoint.Define(physics_stage, TestGainSaveTargets._JOINT)
        physics_stage.Save()

        Sdf.Layer.CreateNew(physx_path).Save()
        mujoco_layer = Sdf.Layer.CreateNew(mujoco_path)
        mujoco_layer.subLayerPaths.append(physics_path)
        mujoco_layer.Save()

        root_layer = Sdf.Layer.CreateNew(root_path)
        root_layer.subLayerPaths.append(physics_path)
        root_layer.Save()
        # Author the DriveAPI gains on the root layer (outside the Physics folder).
        root_stage = Usd.Stage.Open(root_path)
        root_stage.SetEditTarget(root_stage.GetEditTargetForLocalLayer(root_layer))
        drive = UsdPhysics.DriveAPI.Apply(root_stage.GetPrimAtPath(TestGainSaveTargets._JOINT), "angular")
        drive.CreateStiffnessAttr(321.0)
        drive.CreateDampingAttr(32.0)
        root_stage.Save()
        return root_path

    async def test_list_save_targets_offers_physx_when_drive_authored_outside_folder(self) -> None:
        """physx.usda is offered even when DriveAPI gains are authored outside the Physics folder.

        Simulates a robot whose DriveAPI stiffness/damping is authored on the main
        (root) layer while ``payloads/Physics/physics.usda`` is only sublayered in.
        The Physics-folder siblings must still be discovered from the composed
        stage, with the neutral ``physics.usda`` remaining the default.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_drive_outside_physics_folder(tmp_dir)
            stage = Usd.Stage.Open(root_path)
            attr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(self._JOINT), "angular").GetStiffnessAttr()
            # The defining (only) layer is the root layer, outside the Physics folder.
            self.assertTrue(gain_tuner.get_defining_layer(attr).identifier.endswith("robot.usda"))

            options = gain_tuner.list_gain_save_target_layers(attr)
            self.assertTrue(options.resolved)
            names = set(options.display_names)
            self.assertIn("physx.usda", names)
            self.assertTrue({"physics.usda", "physx.usda", "mujoco.usda"}.issubset(names))
            # Neutral physics.usda is still the default and listed first.
            self.assertEqual(options.default.display_name, "physics.usda")
            self.assertEqual(options.display_names[0], "physics.usda")

    async def test_save_plan_routes_drive_to_selected_physx_layer(self) -> None:
        """Selecting physx.usda routes the DriveAPI write to that layer in the save plan."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_drive_outside_physics_folder(tmp_dir)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            physx_path = os.path.join(tmp_dir, "payloads", "Physics", "physx.usda")
            stage = Usd.Stage.Open(root_path)

            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            physx_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physx_path))
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physx_id, mirror_enabled=False)
            # DriveAPI edits are routed to physx.usda, not the neutral physics.usda.
            self.assertIn(physx_id, plan)
            self.assertNotIn(physics_id, plan)

            written = gain_tuner.apply_gain_save_plan(plan, save=True)
            self.assertIn(physx_id, written)
            reopened = Sdf.Layer.OpenAsAnonymous(physx_path)
            saved = reopened.GetAttributeAtPath(f"{self._JOINT}.drive:angular:physics:stiffness").default
            self.assertAlmostEqual(saved, 321.0, places=4)

    async def test_save_plan_splits_joint_params_across_physics_and_physx(self) -> None:
        """Advanced joint params save their Newton half neutral and their PhysX half to the overlay."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            physx_path = os.path.join(tmp_dir, "payloads", "Physics", "physx.usda")
            stage = Usd.Stage.Open(root_path)
            joint = stage.GetPrimAtPath(self._JOINT)
            # Writes are per-backend now, so author each half explicitly to get a
            # joint that carries both opinions; the routing under test is unchanged.
            friction = gain_tuner.joint_param_spec("joint_friction")
            gain_tuner.author_joint_param(joint, friction, 0.35, gain_tuner.BACKEND_NEWTON)
            gain_tuner.author_joint_param(joint, friction, 0.35, gain_tuner.BACKEND_PHYSX)

            entry = SimpleNamespace(joint=joint, drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            physx_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physx_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, mirror_enabled=False)

            def _prop_names(identifier):
                return {edit.prop_path.name for edit in plan.get(identifier, [])}

            self.assertIn("newton:friction", _prop_names(physics_id))
            self.assertIn("physxJoint:jointFriction", _prop_names(physx_id))
            # Neither half leaks onto the other layer.
            self.assertNotIn("physxJoint:jointFriction", _prop_names(physics_id))
            self.assertNotIn("newton:friction", _prop_names(physx_id))

            gain_tuner.apply_gain_save_plan(plan, save=True)
            # Hold both layers: the attribute specs are invalidated with them.
            saved_physics = Sdf.Layer.OpenAsAnonymous(physics_path)
            saved_physx = Sdf.Layer.OpenAsAnonymous(physx_path)
            newton_saved = saved_physics.GetAttributeAtPath(f"{self._JOINT}.newton:friction")
            physx_saved = saved_physx.GetAttributeAtPath(f"{self._JOINT}.physxJoint:jointFriction")
            self.assertAlmostEqual(newton_saved.default, 0.35, places=4)
            self.assertAlmostEqual(physx_saved.default, 0.35, places=4)

    async def test_save_plan_omits_joint_params_for_untouched_joints(self) -> None:
        """A joint authoring neither joint schema contributes no advanced-param edits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir)
            stage = Usd.Stage.Open(root_path)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            plan = gain_tuner.build_gain_save_plan([entry], stage, None, mirror_enabled=False)

            authored = {edit.prop_path.name for edits in plan.values() for edit in edits}
            self.assertFalse({name for name in authored if name.startswith(("newton:", "physxJoint:"))})

    async def test_save_plan_persists_only_the_edited_joint_param(self) -> None:
        """Applying a joint schema must not turn its other params into saved opinions.

        Both schemas resolve all of their attributes once applied, so saving
        whatever resolves would stamp this joint's untouched velocity limit into
        the asset as an explicit `inf` that overrides every weaker opinion.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            stage = Usd.Stage.Open(root_path)
            joint = stage.GetPrimAtPath(self._JOINT)
            armature = gain_tuner.joint_param_spec("armature")
            gain_tuner.author_joint_param(joint, armature, 0.05, gain_tuner.BACKEND_NEWTON)
            gain_tuner.author_joint_param(joint, armature, 0.05, gain_tuner.BACKEND_PHYSX)

            entry = SimpleNamespace(joint=joint, drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, mirror_enabled=False)

            saved = {edit.prop_path.name for edits in plan.values() for edit in edits}
            self.assertEqual(
                {name for name in saved if name.startswith(("newton:", "physxJoint:"))},
                {"newton:armature", "physxJoint:armature"},
            )

    async def test_saved_joint_params_arrive_as_applied_schemas(self) -> None:
        """A saved layer must carry the API application, not just the value.

        `ApplyAPI` writes `apiSchemas` into the stage's edit target, which is not
        the payload layer the value is routed to.  Saving the value alone leaves a
        layer holding `newton:armature` on a prim with no `NewtonJointAPI`, which
        every consumer that iterates applied schemas ignores -- the same silent
        drop this feature exists to fix.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            physx_path = os.path.join(tmp_dir, "payloads", "Physics", "physx.usda")
            stage = Usd.Stage.Open(root_path)
            joint = stage.GetPrimAtPath(self._JOINT)
            armature = gain_tuner.joint_param_spec("armature")
            gain_tuner.author_joint_param(joint, armature, 0.05, gain_tuner.BACKEND_NEWTON)
            gain_tuner.author_joint_param(joint, armature, 0.05, gain_tuner.BACKEND_PHYSX)

            entry = SimpleNamespace(joint=joint, drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, mirror_enabled=False)
            gain_tuner.apply_gain_save_plan(plan, save=True)

            # A consumer that sublayers only the payload layers never sees the
            # edit target where ApplyAPI recorded the schema application.
            consumer = Sdf.Layer.CreateAnonymous(".usda")
            consumer.subLayerPaths.append(physics_path)
            consumer.subLayerPaths.append(physx_path)
            consumed = Usd.Stage.Open(consumer)
            consumed_joint = consumed.GetPrimAtPath(self._JOINT)

            self.assertTrue(consumed_joint.HasAPI("NewtonJointAPI"))
            self.assertTrue(consumed_joint.HasAPI("PhysxJointAPI"))
            self.assertAlmostEqual(consumed_joint.GetAttribute("newton:armature").Get(), 0.05, places=4)
            self.assertAlmostEqual(consumed_joint.GetAttribute("physxJoint:armature").Get(), 0.05, places=4)

    async def test_save_plan_keeps_physx_params_off_the_neutral_layer(self) -> None:
        """With no physx.usda overlay the PhysX mirror stays on its own layer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
            os.makedirs(physics_dir)
            physics_path = os.path.join(physics_dir, "physics.usda")
            physics_stage = Usd.Stage.CreateNew(physics_path)
            UsdPhysics.RevoluteJoint.Define(physics_stage, self._JOINT)
            physics_stage.Save()
            root_layer = Sdf.Layer.CreateNew(os.path.join(tmp_dir, "robot.usda"))
            root_layer.subLayerPaths.append(physics_path)
            root_layer.Save()

            stage = Usd.Stage.Open(root_layer.identifier)
            joint = stage.GetPrimAtPath(self._JOINT)
            armature = gain_tuner.joint_param_spec("armature")
            gain_tuner.author_joint_param(joint, armature, 0.05, gain_tuner.BACKEND_NEWTON)
            gain_tuner.author_joint_param(joint, armature, 0.05, gain_tuner.BACKEND_PHYSX)

            entry = SimpleNamespace(joint=joint, drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, mirror_enabled=False)

            neutral = {edit.prop_path.name for edit in plan.get(physics_id, [])}
            self.assertIn("newton:armature", neutral)
            self.assertNotIn("physxJoint:armature", neutral)

    def _assert_importer_folder_resolved(self, attr, asset_dir: str) -> None:
        """Assert the three Physics siblings under ``asset_dir`` are the candidates."""
        # The importer authors only maxForce, so no layer defines the gain.
        self.assertIsNone(gain_tuner.get_defining_layer(attr))
        options = gain_tuner.list_gain_save_target_layers(attr)
        self.assertTrue(options.resolved)
        self.assertEqual(set(options.display_names), {"physics.usda", "physx.usda", "mujoco.usda"})
        self.assertEqual(options.default.display_name, "physics.usda")
        self.assertEqual(options.display_names[0], "physics.usda")
        physics_dir = os.path.join(asset_dir, "payloads", "Physics")
        for candidate in options.candidates:
            self.assertEqual(os.path.dirname(candidate.identifier), physics_dir)

    async def test_list_save_targets_importer_layout_unauthored_gains(self) -> None:
        """Importer assets resolve their Physics folder even with no authored gains.

        The robot is composed through a reference plus a ``Physics`` variant
        payload, and the DriveAPI stiffness/damping are schema fallbacks with no
        defining layer.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = write_importer_asset_layout(tmp_dir, variant_selection="physics")
            stage = Usd.Stage.Open(root_path)
            joint = stage.GetPrimAtPath(IMPORTER_JOINT_PATH)
            self.assertTrue(joint.IsValid())
            attr = UsdPhysics.DriveAPI(joint, "angular").GetStiffnessAttr()
            self._assert_importer_folder_resolved(attr, tmp_dir)

    async def test_list_save_targets_importer_asset_referenced_into_scene(self) -> None:
        """A referenced importer asset resolves through its composition arcs.

        With the robot referenced into a scene the asset's ``payloads/Physics/``
        folder is not beside the stage root, so it is reachable only by walking the
        reference and variant-payload arcs.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_dir = os.path.join(tmp_dir, "robot_a")
            asset_root = write_importer_asset_layout(asset_dir, variant_selection="physics")
            scene_path = write_scene_referencing(os.path.join(tmp_dir, "scene.usda"), {"RobotA": asset_root})

            stage = Usd.Stage.Open(scene_path)
            joint = stage.GetPrimAtPath(f"/World/RobotA{IMPORTER_JOINT_PATH[len(IMPORTER_ROBOT_PATH):]}")
            self.assertTrue(joint.IsValid())
            attr = UsdPhysics.DriveAPI(joint, "angular").GetStiffnessAttr()
            self._assert_importer_folder_resolved(attr, asset_dir)

    async def test_list_save_targets_scoped_to_the_tuned_robot(self) -> None:
        """A second robot in the scene never contributes save-target candidates.

        Discovery is anchored on the joint being tuned, so tuning one robot must
        not offer (or default to) another robot's physics layers -- saving there
        would write into an unrelated asset.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            a_dir = os.path.join(tmp_dir, "robot_a")
            b_dir = os.path.join(tmp_dir, "robot_b")
            a_root = write_importer_asset_layout(a_dir, variant_selection="physics")
            b_root = write_importer_asset_layout(b_dir, variant_selection="physics")
            scene_path = write_scene_referencing(
                os.path.join(tmp_dir, "scene.usda"), {"RobotA": a_root, "RobotB": b_root}
            )

            stage = Usd.Stage.Open(scene_path)
            joint_suffix = IMPORTER_JOINT_PATH[len(IMPORTER_ROBOT_PATH) :]
            joint_a = stage.GetPrimAtPath(f"/World/RobotA{joint_suffix}")
            self.assertTrue(stage.GetPrimAtPath(f"/World/RobotB{joint_suffix}").IsValid())

            attr = UsdPhysics.DriveAPI(joint_a, "angular").GetStiffnessAttr()
            self._assert_importer_folder_resolved(attr, a_dir)
            # Display names must stay unique, so the combo box is unambiguous.
            options = gain_tuner.list_gain_save_target_layers(attr)
            self.assertEqual(len(set(options.display_names)), len(options.display_names))
            for candidate in options.candidates:
                self.assertNotIn(b_dir, candidate.identifier)

    async def test_save_plan_authors_asset_local_path_for_referenced_robot(self) -> None:
        """Gains for a referenced robot persist at the asset's own prim path.

        The joint composes at ``/World/RobotA/...`` while the asset layer authors
        ``/Robot/...``; writing the composed path would leave a spec the asset
        never reads and silently discard the tuning.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_dir = os.path.join(tmp_dir, "robot_a")
            asset_root = write_importer_asset_layout(asset_dir, variant_selection="physics")
            scene_path = write_scene_referencing(os.path.join(tmp_dir, "scene.usda"), {"RobotA": asset_root})
            physics_path = os.path.join(asset_dir, "payloads", "Physics", "physics.usda")

            stage = Usd.Stage.Open(scene_path)
            composed_joint = f"/World/RobotA{IMPORTER_JOINT_PATH[len(IMPORTER_ROBOT_PATH):]}"
            joint = stage.GetPrimAtPath(composed_joint)
            UsdPhysics.DriveAPI(joint, "angular").GetStiffnessAttr().Set(1234.0)

            entry = SimpleNamespace(joint=joint, drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, mirror_enabled=False)
            self.assertIn(physics_id, plan)
            self.assertIn(physics_id, gain_tuner.apply_gain_save_plan(plan, save=True))

            reopened = Sdf.Layer.OpenAsAnonymous(physics_path)
            native = reopened.GetAttributeAtPath(f"{IMPORTER_JOINT_PATH}.drive:angular:physics:stiffness")
            self.assertIsNotNone(native)
            self.assertAlmostEqual(native.default, 1234.0, places=4)
            # No spec is left behind at the composed scene path.
            self.assertIsNone(reopened.GetAttributeAtPath(f"{composed_joint}.drive:angular:physics:stiffness"))

    async def test_list_save_targets_in_memory_fallback(self) -> None:
        """An anonymous in-memory layer falls back to the defining layer only."""
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.RevoluteJoint.Define(stage, self._JOINT)
        drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(self._JOINT), "angular")
        drive.CreateStiffnessAttr(50.0)
        attr = drive.GetStiffnessAttr()
        options = gain_tuner.list_gain_save_target_layers(attr)
        self.assertTrue(options.resolved)
        self.assertEqual(len(options.candidates), 1)
        self.assertEqual(options.default.identifier, options.candidates[0].identifier)

    async def test_list_save_targets_empty_for_none(self) -> None:
        """Unresolvable inputs yield an empty, unresolved options object."""
        self.assertFalse(gain_tuner.list_gain_save_target_layers(None).resolved)
        self.assertFalse(gain_tuner.list_gain_save_target_layers([]).resolved)

    async def test_drive_gains_to_mjc_position_velocity_zero(self) -> None:
        """drive_gains_to_mjc encodes the position, velocity, and zero-gain cases."""
        # Position control (non-zero Kp): gainPrm[0]=kp, biasPrm=[0, -kp, -kd, ...].
        pos = gain_tuner.drive_gains_to_mjc(100.0, 10.0)
        self.assertEqual((pos.gain_type, pos.bias_type), ("fixed", "affine"))
        self.assertEqual((len(pos.gain_prm), len(pos.bias_prm)), (10, 10))
        self.assertFalse(pos.is_velocity)
        self.assertAlmostEqual(pos.gain_prm[0], 100.0)
        self.assertAlmostEqual(pos.bias_prm[1], -100.0)
        self.assertAlmostEqual(pos.bias_prm[2], -10.0)
        # Inverse of the importer mapping: stiffness = -biasPrm[1], damping = -biasPrm[2].
        self.assertAlmostEqual(pos.gain_prm[0], -pos.bias_prm[1])

        # Velocity control (zero Kp, non-zero Kd): gainPrm[0]=kd, biasPrm=[0, 0, -kd, ...].
        vel = gain_tuner.drive_gains_to_mjc(0.0, 10.0)
        self.assertTrue(vel.is_velocity)
        self.assertAlmostEqual(vel.gain_prm[0], 10.0)
        self.assertAlmostEqual(vel.bias_prm[1], 0.0)
        self.assertAlmostEqual(vel.bias_prm[2], -10.0)
        self.assertAlmostEqual(vel.gain_prm[0], -vel.bias_prm[2])

        # Both gains zero -> all-zero parameter arrays.
        zero = gain_tuner.drive_gains_to_mjc(0.0, 0.0)
        self.assertEqual(zero.gain_prm, [0.0] * 10)
        self.assertEqual(zero.bias_prm, [0.0] * 10)

    async def test_build_mjc_gain_map(self) -> None:
        """MjcActuator prims are mapped to their target DOF with gain attrs resolved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir, with_mjc=True)
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            stage = Usd.Stage.Open(mujoco_path)
            mjc_map = gain_tuner.build_mjc_gain_map(stage)
            self.assertIn(self._JOINT, mjc_map)
            source = mjc_map[self._JOINT]
            self.assertEqual(source.actuator_path, self._MJC_ACTUATOR)
            self.assertTrue(source.gain_prm_attr.IsValid())
            self.assertIsNotNone(source.defining_layer)
            self.assertTrue(source.defining_layer.identifier.endswith("mujoco.usda"))

    async def test_build_mjc_gain_map_empty_without_actuators(self) -> None:
        """Stages with no MjcActuator prims yield an empty map (defensive)."""
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.RevoluteJoint.Define(stage, self._JOINT)
        self.assertEqual(gain_tuner.build_mjc_gain_map(stage), {})
        self.assertEqual(gain_tuner.build_mjc_gain_map(None), {})

    async def test_build_mjc_gain_map_detects_gains_on_joint(self) -> None:
        """mjc:* gain params authored directly on a joint (no MjcActuator prim) are detected."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([2000.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -2000.0, -400.0] + [0.0] * 7)
        joint.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")

        mjc_map = gain_tuner.build_mjc_gain_map(stage)

        # The joint itself is both the key (target DOF) and the "actuator" path.
        self.assertIn(self._JOINT, mjc_map)
        source = mjc_map[self._JOINT]
        self.assertEqual(source.actuator_path, self._JOINT)
        self.assertTrue(source.gain_prm_attr.IsValid())
        self.assertEqual(list(source.gain_prm_attr.Get()), [2000.0] + [0.0] * 9)
        self.assertTrue(source.bias_prm_attr.IsValid())

    async def test_build_mjc_gain_map_ignores_unauthored_gain_attrs(self) -> None:
        """A joint that merely defines (but never authors) mjc gain attrs is not a source."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        # Attribute is created (defined) but no value is authored.
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray)
        self.assertEqual(gain_tuner.build_mjc_gain_map(stage), {})

    async def test_resolve_gain_write_targets_reports_mirror_sources(self) -> None:
        """Write-target resolution reports mjc mirror + Newton writeback sources."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            self._write_physics_folder(tmp_dir, with_mjc=True)
            stage = Usd.Stage.Open(mujoco_path)
            # Author a Newton actuator targeting the joint.
            stage.DefinePrim("/World/Robot/Actuators", "Scope")
            act = stage.DefinePrim(self._ACTUATOR, "Xform")
            act.AddAppliedSchema("NewtonPDControlAPI")
            act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
            act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
            act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(self._JOINT))

            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            targets = gain_tuner.resolve_gain_write_targets([entry], stage, "/World/Robot")
            self.assertTrue(targets.options.resolved)
            self.assertTrue(targets.has_mjc_mirror)
            self.assertTrue(targets.has_newton_writeback)
            self.assertIn("mujoco.usda", targets.mjc_layer_names)

    async def test_build_and_apply_save_plan_persists_to_temp(self) -> None:
        """A save plan writes DriveAPI gains + mirrored mjc gains to temp layers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir, with_mjc=True)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            # Compose both physics and mujoco so the plan can read drive + mjc.
            root_layer = Sdf.Layer.FindOrOpen(root_path)
            root_layer.subLayerPaths = [mujoco_path]
            root_layer.Save()
            stage = Usd.Stage.Open(root_path)
            # Tune the DriveAPI stiffness/damping on the composed stage.
            drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(self._JOINT), "angular")
            drive.GetStiffnessAttr().Set(200.0)
            drive.GetDampingAttr().Set(20.0)

            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, mirror_drive_to_mjc=True)
            self.assertIn(physics_id, plan)
            written = gain_tuner.apply_gain_save_plan(plan, save=True)
            self.assertIn(physics_id, written)

            # Re-open the physics layer fresh and assert the DriveAPI gains persisted.
            reopened_physics = Sdf.Layer.OpenAsAnonymous(physics_path)
            reopened_stage = Usd.Stage.Open(reopened_physics)
            saved_drive = UsdPhysics.DriveAPI(reopened_stage.GetPrimAtPath(self._JOINT), "angular")
            self.assertAlmostEqual(saved_drive.GetStiffnessAttr().Get(), 200.0, places=4)
            self.assertAlmostEqual(saved_drive.GetDampingAttr().Get(), 20.0, places=4)

            # Assert the mirrored mjc gain params persisted to mujoco.usda.
            reopened_mjc = Sdf.Layer.OpenAsAnonymous(mujoco_path)
            gain_prm = reopened_mjc.GetAttributeAtPath(f"{self._MJC_ACTUATOR}.mjc:gainPrm").default
            bias_prm = reopened_mjc.GetAttributeAtPath(f"{self._MJC_ACTUATOR}.mjc:biasPrm").default
            self.assertAlmostEqual(gain_prm[0], 200.0, places=4)
            self.assertAlmostEqual(bias_prm[1], -200.0, places=4)
            self.assertAlmostEqual(bias_prm[2], -20.0, places=4)

    async def test_default_mjc_target_resolves_to_mujoco_layer(self) -> None:
        """The default mjc mirror target resolves to the mujoco.usda candidate."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            self._write_physics_folder(tmp_dir, with_mjc=True)
            stage = Usd.Stage.Open(mujoco_path)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            targets = gain_tuner.resolve_gain_write_targets([entry], stage, None, mirror_drive_to_mjc=True)
            # Default DriveAPI target is still the neutral physics.usda.
            self.assertEqual(targets.options.default.display_name, "physics.usda")
            # Default mjc mirror target is the mujoco.usda candidate.
            self.assertIsNotNone(targets.mjc_target_identifier)
            self.assertTrue(targets.mjc_target_identifier.endswith("mujoco.usda"))
            self.assertEqual(targets.mjc_layer_names, ["mujoco.usda"])
            self.assertTrue(targets.will_mirror_mjc)

    async def test_resolve_write_targets_mjc_mirror_is_opt_in(self) -> None:
        """The mjc mirror is OFF by default; ``will_mirror_mjc`` needs the opt-in."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            self._write_physics_folder(tmp_dir, with_mjc=True)
            stage = Usd.Stage.Open(mujoco_path)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)

            # Default: source present but mirror off -> will_mirror_mjc False.
            default_targets = gain_tuner.resolve_gain_write_targets([entry], stage, None)
            self.assertTrue(default_targets.has_mjc_mirror)  # source still present
            self.assertFalse(default_targets.mirror_drive_to_mjc)
            self.assertFalse(default_targets.will_mirror_mjc)

            # Opt-in on -> will_mirror_mjc True.
            on_targets = gain_tuner.resolve_gain_write_targets([entry], stage, None, mirror_drive_to_mjc=True)
            self.assertTrue(on_targets.mirror_drive_to_mjc)
            self.assertTrue(on_targets.will_mirror_mjc)

    async def test_save_plan_mjc_mirror_off_by_default_is_drive_only(self) -> None:
        """Default (mjc mirror OFF): the plan has DriveAPI edits only, never mjc:*."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir, with_mjc=True)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            root_layer = Sdf.Layer.FindOrOpen(root_path)
            root_layer.subLayerPaths = [mujoco_path]
            root_layer.Save()
            stage = Usd.Stage.Open(root_path)
            drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(self._JOINT), "angular")
            drive.GetStiffnessAttr().Set(200.0)
            drive.GetDampingAttr().Set(20.0)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))

            # No mirror flag passed -> defaults (mjc mirror off).
            plan = gain_tuner.build_gain_save_plan([entry], stage, physics_id, articulation_root_path="/World/Robot")
            names = [edit.prop_path.name for edits in plan.values() for edit in edits]
            self.assertTrue(names)
            self.assertFalse(any(name.startswith("mjc:") for name in names))
            self.assertFalse(any(name.startswith("newton:") for name in names))
            # The mujoco.usda layer is never a target when the mjc mirror is off.
            self.assertFalse(any(identifier.endswith("mujoco.usda") for identifier in plan))

    async def test_save_plan_routes_mirror_to_selected_targets(self) -> None:
        """Toggle ON: mjc:* / newton:* edits route to the selected target layers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir, with_mjc=True)
            physics_path = os.path.join(tmp_dir, "payloads", "Physics", "physics.usda")
            physx_path = os.path.join(tmp_dir, "payloads", "Physics", "physx.usda")
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            root_layer = Sdf.Layer.FindOrOpen(root_path)
            root_layer.subLayerPaths = [mujoco_path]
            root_layer.Save()
            stage = Usd.Stage.Open(root_path)
            # Author a Newton actuator targeting the joint.
            stage.DefinePrim("/World/Robot/Actuators", "Scope")
            act = stage.DefinePrim(self._ACTUATOR, "Xform")
            act.AddAppliedSchema("NewtonPDControlAPI")
            act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
            act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
            act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(self._JOINT))
            drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(self._JOINT), "angular")
            drive.GetStiffnessAttr().Set(150.0)
            drive.GetDampingAttr().Set(15.0)

            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            physics_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physics_path))
            physx_id = get_layer_save_identifier(Sdf.Layer.FindOrOpen(physx_path))

            # Route mjc:* to physx.usda and newton:* to physics.usda (non-defaults).
            plan = gain_tuner.build_gain_save_plan(
                [entry],
                stage,
                physics_id,
                mirror_enabled=True,
                mirror_drive_to_mjc=True,
                mjc_target_identifier=physx_id,
                newton_target_identifier=physics_id,
                articulation_root_path="/World/Robot",
            )
            physx_names = [edit.prop_path.name for edit in plan.get(physx_id, [])]
            self.assertIn("mjc:gainPrm", physx_names)
            self.assertIn("mjc:biasPrm", physx_names)
            physics_names = [edit.prop_path.name for edit in plan.get(physics_id, [])]
            self.assertIn("newton:kp", physics_names)
            self.assertIn("newton:kd", physics_names)
            # Neither mjc:* nor the default mujoco.usda layer are used.
            self.assertFalse(any(identifier.endswith("mujoco.usda") for identifier in plan))
            # The mirrored mjc gainPrm value tracks the tuned stiffness (150).
            gain_prm_edit = next(edit for edit in plan[physx_id] if edit.prop_path.name == "mjc:gainPrm")
            self.assertAlmostEqual(gain_prm_edit.value[0], 150.0, places=4)

    async def test_save_plan_preserves_mjc_edits_under_mujoco_solver(self) -> None:
        """MuJoCo solver active: the mjc mirror preserves edited mjc gains (not DriveAPI)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_path = self._write_physics_folder(tmp_dir, with_mjc=True)
            mujoco_path = os.path.join(tmp_dir, "payloads", "Physics", "mujoco.usda")
            root_layer = Sdf.Layer.FindOrOpen(root_path)
            root_layer.subLayerPaths = [mujoco_path]
            root_layer.Save()
            stage = Usd.Stage.Open(root_path)
            # Edit the mjc gains directly, as the editable MuJoCo view would.
            gain_tuner.author_mjc_gains(gain_tuner.build_mjc_gain_map(stage)[self._JOINT], 3000.0, 90.0)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)

            # MuJoCo solver active: the plan authors the edited mjc gains (kp=3000),
            # not the (unedited) DriveAPI stiffness (100).
            plan = gain_tuner.build_gain_save_plan(
                [entry], stage, articulation_root_path="/World/Robot", mujoco_solver_active=True
            )
            gain_edits = [e for edits in plan.values() for e in edits if e.prop_path.name == "mjc:gainPrm"]
            self.assertTrue(gain_edits)
            self.assertAlmostEqual(gain_edits[0].value[0], 3000.0, places=4)

            # MuJoCo solver NOT active + opt-in mirror on: the mirror is derived
            # from DriveAPI (kp=100).
            plan_pd = gain_tuner.build_gain_save_plan(
                [entry],
                stage,
                articulation_root_path="/World/Robot",
                mujoco_solver_active=False,
                mirror_drive_to_mjc=True,
            )
            gain_edits_pd = [e for edits in plan_pd.values() for e in edits if e.prop_path.name == "mjc:gainPrm"]
            self.assertTrue(gain_edits_pd)
            self.assertAlmostEqual(gain_edits_pd[0].value[0], 100.0, places=4)

            # MuJoCo solver NOT active + mirror OFF (default): no mjc edits at all.
            plan_off = gain_tuner.build_gain_save_plan(
                [entry], stage, articulation_root_path="/World/Robot", mujoco_solver_active=False
            )
            gain_edits_off = [e for edits in plan_off.values() for e in edits if e.prop_path.name.startswith("mjc:")]
            self.assertFalse(gain_edits_off)


class TestMjcEditableWriteback(omni.kit.test.AsyncTestCase):
    """Part B: editable MuJoCo-native gains write back to the mjc arrays."""

    _JOINT = "/World/Robot/joint0"

    def _stage(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim("/World/Robot", "Xform")
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([100.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -100.0, -5.0] + [0.0] * 7)
        joint.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.String).Set("fixed")
        joint.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")
        return stage, joint

    async def test_author_mjc_gains_updates_arrays(self) -> None:
        """author_mjc_gains rewrites gainPrm / biasPrm from tuned (kp, kd)."""
        stage, joint = self._stage()
        source = gain_tuner.build_mjc_gain_map(stage)[self._JOINT]
        self.assertTrue(gain_tuner.author_mjc_gains(source, 2500.0, 60.0))
        gain_prm = list(joint.GetAttribute("mjc:gainPrm").Get())
        bias_prm = list(joint.GetAttribute("mjc:biasPrm").Get())
        self.assertAlmostEqual(gain_prm[0], 2500.0)
        self.assertAlmostEqual(bias_prm[1], -2500.0)
        self.assertAlmostEqual(bias_prm[2], -60.0)
        kp, kd = gain_tuner.mjc_params_to_drive_gains(gain_prm, bias_prm)
        self.assertAlmostEqual(kp, 2500.0)
        self.assertAlmostEqual(kd, 60.0)

    async def test_author_mjc_gains_none_source(self) -> None:
        """author_mjc_gains is a no-op (returns False) for a missing source."""
        self.assertFalse(gain_tuner.author_mjc_gains(None, 1.0, 1.0))
