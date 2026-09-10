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

"""Unit tests for joint drive-attribute helpers."""

from __future__ import annotations

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.joint_drive_attrs import (
    get_mimic_damping_ratio_attr,
    get_mimic_natural_frequency_attr,
)
from pxr import Usd, UsdPhysics


class _FakeMimicPrim:
    """Minimal stand-in for a USD joint prim for mimic-detection unit tests.

    Only implements ``GetAppliedSchemas`` (used by mimic detection and the
    PhysX axis parser). Avoids depending on runtime schema registration of the
    PhysX/Newton mimic APIs, which may not be loaded in all test configs.
    """

    def __init__(self, applied_schemas: list[str]) -> None:
        self._applied_schemas = list(applied_schemas)

    def GetAppliedSchemas(self) -> list[str]:  # noqa: N802 - matches USD API
        return list(self._applied_schemas)


class TestJointDriveAttrs(omni.kit.test.AsyncTestCase):
    """Joint drive attribute getters from ``joint_drive_attrs`` (no UI)."""

    async def test_get_stiffness_damping_attrs_revolute(self) -> None:
        """Revolute drive stiffness and damping attrs are discovered."""
        stage = Usd.Stage.CreateInMemory()
        jpath = "/joint"
        UsdPhysics.RevoluteJoint.Define(stage, jpath)
        drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jpath), "angular")
        drive.CreateStiffnessAttr(1.0)
        drive.CreateDampingAttr(0.1)
        joint = stage.GetPrimAtPath(jpath)
        self.assertIsNotNone(gain_tuner.get_stiffness_attr(joint))
        self.assertIsNotNone(gain_tuner.get_damping_attr(joint))

    async def test_get_stiffness_attr_none_without_drive_api(self) -> None:
        """Prims without a drive API do not expose a stiffness attr."""
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/nondrive", "Xform")
        self.assertIsNone(gain_tuner.get_stiffness_attr(prim))

    async def test_is_joint_mimic_detects_physx_schema(self) -> None:
        """Joints with the legacy PhysX per-axis mimic schema are detected."""
        joint = _FakeMimicPrim(["PhysxJointAPI", "PhysxMimicJointAPI:rotZ"])
        self.assertTrue(gain_tuner.is_joint_mimic(joint))

    async def test_is_joint_mimic_detects_newton_schema(self) -> None:
        """Joints with the Newton single-apply mimic schema are detected."""
        joint = _FakeMimicPrim(["NewtonMimicAPI"])
        self.assertTrue(gain_tuner.is_joint_mimic(joint))

    async def test_is_joint_mimic_false_for_plain_joint(self) -> None:
        """Plain driven joints are not reported as mimic."""
        joint = _FakeMimicPrim(["PhysxJointAPI"])
        self.assertFalse(gain_tuner.is_joint_mimic(joint))

    async def test_newton_mimic_gain_attrs_return_none(self) -> None:
        """Newton mimic joints have no NF/DR/damping attrs, so getters return None.

        This also guards against the axis-parsing IndexError that occurs when a
        mimic joint has no PhysX per-axis schema instance.
        """
        joint = _FakeMimicPrim(["NewtonMimicAPI"])
        self.assertTrue(gain_tuner.is_joint_mimic(joint))
        self.assertIsNone(get_mimic_natural_frequency_attr(joint))
        self.assertIsNone(get_mimic_damping_ratio_attr(joint))
        self.assertIsNone(gain_tuner.get_damping_attr(joint))

    async def test_get_joint_drive_mode_newton_mimic(self) -> None:
        """Newton mimic joints report the MIMIC drive mode."""
        joint = _FakeMimicPrim(["NewtonMimicAPI"])
        self.assertEqual(gain_tuner.get_joint_drive_mode(joint), gain_tuner.JointDriveMode.MIMIC.value)
