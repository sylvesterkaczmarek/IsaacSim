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

"""Unit tests for USD layer / save-target utilities."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.usd
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.gains_tuner import get_original_spec_for_drive_api
from isaacsim.robot_setup.gain_tuner.usd_layer_utils import (
    find_layer_by_save_identifier,
    find_physics_layer,
    find_physics_layer_on_disk,
    get_layer_save_identifier,
    get_property_path_for_layer,
    is_layer_savable,
    is_physics_layer,
    is_physx_layer,
    plan_stage_layer_save,
    remap_edits_to_physics_layer,
    resolve_gain_save_target,
)
from pxr import Sdf, Usd, UsdPhysics

from .common import (
    DriveSubmodality,
    JointModality,
    TestGainTunerHarness,
)


class TestGainTunerUsdUtilities(TestGainTunerHarness):
    """USD helpers and inertia recompute stability."""

    async def test_resolve_gain_save_target_prefers_physics_layer(self) -> None:
        """Live edits on the root/session layer must not redirect save away from physics.usda."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_path = os.path.join(tmp_dir, "physics.usda")
            root_path = os.path.join(tmp_dir, "robot.usda")
            physics_layer = Sdf.Layer.CreateNew(physics_path)
            root_layer = Sdf.Layer.CreateNew(root_path)
            root_layer.subLayerPaths.append(physics_path)

            physics_stage = Usd.Stage.Open(physics_path)
            jpath = "/joint"
            UsdPhysics.RevoluteJoint.Define(physics_stage, jpath)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(jpath), "angular")
            drive.CreateStiffnessAttr(100.0)
            physics_stage.Save()

            stage = Usd.Stage.Open(root_path)
            attr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(jpath), "angular").GetStiffnessAttr()
            self.assertIsNotNone(attr)
            attr.Set(250.0)

            physics = find_physics_layer(stage)
            self.assertIsNotNone(physics)
            self.assertTrue(is_physics_layer(physics.identifier))

            target_layer, target_path = resolve_gain_save_target(attr)
            self.assertIsNotNone(target_layer)
            self.assertTrue(is_physics_layer(target_layer.identifier))
            self.assertEqual(target_path, attr.GetPath())

    async def test_resolve_gain_save_target_nested_physx_sublayer(self) -> None:
        """physics.usda nested under a physx subLayer must still be the save target."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
            os.makedirs(physics_dir, exist_ok=True)
            physics_path = os.path.join(physics_dir, "physics.usda")
            physx_path = os.path.join(physics_dir, "physx.usda")
            root_path = os.path.join(tmp_dir, "robot.usda")

            Sdf.Layer.CreateNew(physics_path)
            physx_layer = Sdf.Layer.CreateNew(physx_path)
            root_layer = Sdf.Layer.CreateNew(root_path)

            physx_layer.subLayerPaths.append("./physics.usda")
            jpath = "/robot/Physics/wheel_joint"
            physics_stage = Usd.Stage.Open(physics_path)
            UsdPhysics.RevoluteJoint.Define(physics_stage, jpath)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(jpath), "angular")
            drive.CreateStiffnessAttr(10.0)
            physics_stage.Save()
            physx_layer.Save()

            root_layer.subLayerPaths.append(os.path.join("payloads", "Physics", "physx.usda"))
            root_layer.Save()

            stage = Usd.Stage.Open(root_path)
            attr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(jpath), "angular").GetStiffnessAttr()
            self.assertIsNotNone(attr)
            attr.Set(42.0)

            joint = stage.GetPrimAtPath(jpath)
            self.assertTrue(joint.IsValid())
            physics = find_physics_layer(stage, anchor_prim=joint)
            self.assertIsNotNone(physics)
            self.assertTrue(is_physics_layer(physics.identifier))

            target_layer, target_path = resolve_gain_save_target(attr, anchor_prim=joint)
            self.assertIsNotNone(target_layer)
            self.assertTrue(is_physics_layer(target_layer.identifier))
            self.assertEqual(target_path, attr.GetPath())

    async def test_find_physics_layer_on_disk_for_robot_layout(self) -> None:
        """Standard payloads/Physics/physics.usda beside the root asset is discoverable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
            os.makedirs(physics_dir)
            physics_path = os.path.join(physics_dir, "physics.usda")
            root_path = os.path.join(tmp_dir, "robot.usda")
            physics_layer = Sdf.Layer.CreateNew(physics_path)
            root_layer = Sdf.Layer.CreateNew(root_path)
            physics_layer.Save()
            root_layer.Save()

            stage = Usd.Stage.Open(root_path)
            found = find_physics_layer_on_disk(stage)
            self.assertIsNotNone(found)
            self.assertTrue(is_physics_layer(found.identifier))
            self.assertTrue(is_layer_savable(found))

    async def test_get_original_spec_for_drive_api_root_authored(self) -> None:
        """Root-authored drive API specs are found on the root layer."""
        try:
            await stage_utils.create_new_stage_async()
            stage = omni.usd.get_context().get_stage()
            stage.DefinePrim(Sdf.Path("/World"), "Xform")
            jpath = "/World/rev"
            UsdPhysics.RevoluteJoint.Define(stage, jpath)
            drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jpath), "angular")
            drive.CreateStiffnessAttr(120.0)
            spec = get_original_spec_for_drive_api(stage, jpath, "angular")
            self.assertIsNotNone(spec)
            self.assertEqual(spec.layer, stage.GetRootLayer())
        finally:
            stage_utils.close_stage()

    async def test_joint_inertia_recompute_is_idempotent(self) -> None:
        """Second ``compute_joints_accumulated_inertia`` without scene change matches first."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        first = {str(k.GetPath()): v for k, v in self._gain_tuner._joint_accumulated_inertia.items()}
        self._gain_tuner.compute_joints_accumulated_inertia()
        second = {str(k.GetPath()): v for k, v in self._gain_tuner._joint_accumulated_inertia.items()}
        self.assertEqual(first, second)


class TestUsdLayerUtils(omni.kit.test.AsyncTestCase):
    """Pure USD layer helpers (no simulation harness)."""

    async def test_is_physics_layer_legacy_usd_suffix(self) -> None:
        """Legacy `_physics.usd` files under payloads/Physics count as physics layers."""
        self.assertTrue(is_physics_layer("/asset/payloads/Physics/_physics.usd"))

    async def test_is_physx_layer_matches_only_the_physx_overlay(self) -> None:
        """The PhysX overlay is recognized without also matching the physics base layer."""
        self.assertTrue(is_physx_layer("/asset/payloads/Physics/physx.usda"))
        self.assertTrue(is_physx_layer("/asset/payloads/Physics/_physx.usd"))
        self.assertFalse(is_physx_layer("/asset/payloads/Physics/physics.usda"))
        self.assertFalse(is_physx_layer("/asset/payloads/Physics/mujoco.usda"))

    async def test_resolve_prefers_physics_spec_in_property_stack(self) -> None:
        """Gain save target resolution prefers the physics-layer property spec."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_path = os.path.join(tmp_dir, "physics.usda")
            root_path = os.path.join(tmp_dir, "robot.usda")
            Sdf.Layer.CreateNew(physics_path)
            root_layer = Sdf.Layer.CreateNew(root_path)
            root_layer.subLayerPaths.append(physics_path)
            physics_stage = Usd.Stage.Open(physics_path)
            jpath = "/joint"
            UsdPhysics.RevoluteJoint.Define(physics_stage, jpath)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(jpath), "angular")
            drive.CreateStiffnessAttr(50.0)
            physics_stage.Save()
            stage = Usd.Stage.Open(root_path)
            attr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(jpath), "angular").GetStiffnessAttr()
            attr.Set(999.0)
            target_layer, target_path = resolve_gain_save_target(attr)
            self.assertTrue(is_physics_layer(target_layer.identifier))
            self.assertEqual(target_path, attr.GetPath())

    async def test_is_layer_savable_respects_filesystem_writable(self) -> None:
        """Layer savability follows filesystem write permissions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "physics.usda")
            Sdf.Layer.CreateNew(path).Save()
            os.chmod(path, 0o444)
            readonly_layer = Sdf.Layer.FindOrOpen(path)
            self.assertFalse(is_layer_savable(readonly_layer))
            os.chmod(path, 0o644)
            writable_layer = Sdf.Layer.FindOrOpen(path)
            self.assertTrue(is_layer_savable(writable_layer))

    async def test_remap_edits_to_physics_layer_merges_multiple_layer_keys(self) -> None:
        """Edits from multiple layers remap to one physics-layer save key."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_path = os.path.join(tmp_dir, "physics.usda")
            root_path = os.path.join(tmp_dir, "root.usda")
            physics_layer = Sdf.Layer.CreateNew(physics_path)
            root_layer = Sdf.Layer.CreateNew(root_path)
            physics_layer.Save()
            root_layer.Save()
            path = Sdf.Path("/joint.stiffness")
            edits = {
                root_layer.identifier: [(path, 1.0)],
                physics_layer.identifier: [(path, 2.0)],
            }
            merged = remap_edits_to_physics_layer(edits, physics_layer)
            self.assertEqual(len(merged), 1)
            save_id = get_layer_save_identifier(physics_layer)
            self.assertIn(save_id, merged)
            self.assertEqual(len(merged[save_id]), 1)
            # First path seen wins when the same property appears on multiple layers.
            self.assertEqual(merged[save_id][0][1], 1.0)

    async def test_get_property_path_for_layer_uses_prim_stack(self) -> None:
        """Property paths for a layer are resolved from the authored prim stack."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_path = os.path.join(tmp_dir, "physics.usda")
            physics_layer = Sdf.Layer.CreateNew(physics_path)
            physics_stage = Usd.Stage.Open(physics_path)
            jpath = "/robot/joint"
            UsdPhysics.RevoluteJoint.Define(physics_stage, jpath)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(jpath), "angular")
            drive.CreateStiffnessAttr(10.0)
            physics_stage.Save()
            stage = Usd.Stage.Open(physics_path)
            attr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(jpath), "angular").GetStiffnessAttr()
            path = get_property_path_for_layer(attr, physics_layer)
            self.assertEqual(path, attr.GetPath())

    async def test_plan_stage_layer_save_no_writable_layers(self) -> None:
        """No writable layers: no dialog, single no-permission toast."""
        layers = [
            SimpleNamespace(identifier="a", permissionToEdit=False, permissionToSave=True),
            SimpleNamespace(identifier="b", permissionToEdit=True, permissionToSave=False),
        ]
        decision = plan_stage_layer_save(layers)
        self.assertEqual(decision.writable_layer_identifiers, [])
        self.assertFalse(decision.show_dialog)
        self.assertTrue(decision.post_no_permission_toast)

    async def test_plan_stage_layer_save_some_writable_layers(self) -> None:
        """Mixed layers: dialog with writable ids and a single toast for the rest."""
        layers = [
            SimpleNamespace(identifier="rw", permissionToEdit=True, permissionToSave=True),
            SimpleNamespace(identifier="ro", permissionToEdit=False, permissionToSave=True),
        ]
        decision = plan_stage_layer_save(layers)
        self.assertEqual(decision.writable_layer_identifiers, ["rw"])
        self.assertTrue(decision.show_dialog)
        self.assertTrue(decision.post_no_permission_toast)

    async def test_plan_stage_layer_save_all_writable_layers(self) -> None:
        """All writable layers: dialog shown and no no-permission toast."""
        layers = [
            SimpleNamespace(identifier="rw1", permissionToEdit=True, permissionToSave=True),
            SimpleNamespace(identifier="rw2", permissionToEdit=True, permissionToSave=True),
        ]
        decision = plan_stage_layer_save(layers)
        self.assertEqual(decision.writable_layer_identifiers, ["rw1", "rw2"])
        self.assertTrue(decision.show_dialog)
        self.assertFalse(decision.post_no_permission_toast)

    async def test_find_layer_by_save_identifier_opens_absolute_path(self) -> None:
        """Absolute save identifiers are opened as Sdf layers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "physics.usda")
            layer = Sdf.Layer.CreateNew(path)
            layer.Save()
            save_id = get_layer_save_identifier(layer)
            found = find_layer_by_save_identifier(save_id)
            self.assertIsNotNone(found)
            self.assertEqual(os.path.normpath(found.realPath), os.path.normpath(path))


class TestGainTunerUsdUtilitiesCollectEdits(omni.kit.test.AsyncTestCase):
    """collect_gain_save_edits with a minimal composed stage."""

    async def test_collect_gain_save_edits_targets_physics_layer(self) -> None:
        """Gain save edits target the discovered physics layer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_path = os.path.join(tmp_dir, "physics.usda")
            root_path = os.path.join(tmp_dir, "robot.usda")
            Sdf.Layer.CreateNew(physics_path)
            root_layer = Sdf.Layer.CreateNew(root_path)
            root_layer.subLayerPaths.append(physics_path)
            physics_stage = Usd.Stage.Open(physics_path)
            jpath = "/joint"
            UsdPhysics.RevoluteJoint.Define(physics_stage, jpath)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(jpath), "angular")
            drive.CreateStiffnessAttr(100.0)
            drive.CreateDampingAttr(1.0)
            drive.CreateTypeAttr("force")
            physics_stage.Save()
            stage = Usd.Stage.Open(root_path)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(jpath), drive_axis=None)
            attr = gain_tuner.get_stiffness_attr(entry.joint, entry.drive_axis)
            attr.Set(200.0)
            edits, physics = gain_tuner.collect_gain_save_edits([entry], stage)
            self.assertIsNotNone(physics)
            self.assertTrue(is_physics_layer(physics.identifier))
            save_id = get_layer_save_identifier(physics)
            self.assertIn(save_id, edits)
            paths = {path for path, _ in edits[save_id]}
            values = dict(edits[save_id])
            stiffness_path = attr.GetPath()
            self.assertIn(stiffness_path, paths)
            self.assertAlmostEqual(values[stiffness_path], 200.0, places=5)
            self.assertGreaterEqual(len(edits[save_id]), 3)

    async def test_collect_gain_save_edits_mimic_joint_attrs(self) -> None:
        """Mimic joint natural-frequency and damping-ratio attrs are collected."""
        stage = Usd.Stage.CreateInMemory()
        jpath = "/mimic_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        nf_attr = joint.GetPrim().CreateAttribute("physxMimicJoint:rotX:naturalFrequency", Sdf.ValueTypeNames.Float)
        dr_attr = joint.GetPrim().CreateAttribute("physxMimicJoint:rotX:dampingRatio", Sdf.ValueTypeNames.Float)
        nf_attr.Set(12.0)
        dr_attr.Set(0.08)
        entry = SimpleNamespace(joint=joint.GetPrim(), drive_axis=None)
        widget = "isaacsim.robot_setup.gain_tuner.joint_drive_attrs"
        with (
            mock.patch(f"{widget}.is_joint_mimic", return_value=True),
            mock.patch(f"{widget}.get_mimic_natural_frequency_attr", return_value=nf_attr),
            mock.patch(f"{widget}.get_mimic_damping_ratio_attr", return_value=dr_attr),
            mock.patch(f"{widget}.get_stiffness_attr", return_value=None),
        ):
            edits, physics = gain_tuner.collect_gain_save_edits([entry], stage)
        self.assertEqual(len(edits), 1)
        collected = next(iter(edits.values()))
        collected_values = dict(collected)
        self.assertAlmostEqual(collected_values[nf_attr.GetPath()], 12.0, places=5)
        self.assertAlmostEqual(collected_values[dr_attr.GetPath()], 0.08, places=5)
