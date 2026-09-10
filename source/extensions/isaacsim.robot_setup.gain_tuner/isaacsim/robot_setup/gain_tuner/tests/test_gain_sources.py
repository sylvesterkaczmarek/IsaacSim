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

"""Unit tests for multi-backend gain-source detection and resolution."""

from __future__ import annotations

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.gain_sources import (
    DAMPING_LABEL,
    KD_LABEL,
    KI_LABEL,
    KP_LABEL,
    MJC_DAMPING_LABEL,
    MJC_STIFFNESS_LABEL,
    STIFFNESS_LABEL,
    active_gain_source,
)
from pxr import Sdf, Usd, UsdPhysics


class TestGainSources(omni.kit.test.AsyncTestCase):
    """Gain-source detection and resolution from ``gain_sources`` (no UI)."""

    _ROOT = "/World/Robot"
    _JOINT = "/World/Robot/joint0"
    _ACTUATOR = "/World/Robot/Actuators/act0"

    def _make_stage(self, *, with_drive: bool, with_actuator: bool, is_pid: bool = False):
        """Build an in-memory stage with an optional PhysicsDrive and/or actuator.

        The actuator prim carries the schema token and ``newton:*`` attributes
        directly (via ``AddAppliedSchema`` / ``CreateAttribute``) so the test does
        not depend on the Newton schema being registered in the test config.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, self._JOINT)
        joint = stage.GetPrimAtPath(self._JOINT)
        if with_drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
            drive.CreateStiffnessAttr(100.0)
            drive.CreateDampingAttr(10.0)
        if with_actuator:
            stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
            act = stage.DefinePrim(self._ACTUATOR, "Xform")
            act.AddAppliedSchema("NewtonPIDControlAPI" if is_pid else "NewtonPDControlAPI")
            act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
            act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
            if is_pid:
                act.CreateAttribute("newton:ki", Sdf.ValueTypeNames.Float).Set(5.0)
            act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(self._JOINT))
        return stage, joint

    async def test_active_gain_source_selection(self) -> None:
        """Actuator wins (backend-independent); mjc only under the MuJoCo solver."""
        # A Newton actuator is the active source whenever authored, regardless of
        # backend (the actuator runtime zeros DriveAPI under PhysX or Newton).
        self.assertEqual(active_gain_source(True, True), gain_tuner.GainSource.ACTUATOR)
        self.assertEqual(active_gain_source(False, True), gain_tuner.GainSource.ACTUATOR)
        self.assertEqual(
            active_gain_source(True, True, True, mujoco_solver_active=True), gain_tuner.GainSource.ACTUATOR
        )
        # No actuator: DriveAPI is the active source (PhysX or non-MuJoCo Newton).
        self.assertEqual(active_gain_source(True, False), gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertEqual(active_gain_source(False, False), gain_tuner.GainSource.NONE)
        # MuJoCo-native becomes active only when the MuJoCo solver runs and no
        # actuator is present; otherwise it stays inspection-only.
        self.assertEqual(active_gain_source(False, False, True), gain_tuner.GainSource.NONE)
        self.assertEqual(
            active_gain_source(False, False, True, mujoco_solver_active=True), gain_tuner.GainSource.MUJOCO
        )
        self.assertEqual(active_gain_source(True, False, True, mujoco_solver_active=True), gain_tuner.GainSource.MUJOCO)
        # mjc present but MuJoCo solver not active: DriveAPI wins when authored.
        self.assertEqual(active_gain_source(True, False, True), gain_tuner.GainSource.PHYSICS_DRIVE)

    async def test_build_actuator_gain_map(self) -> None:
        """Actuator gains are mapped to their target DOF via ``newton:targets``."""
        stage, _ = self._make_stage(with_drive=True, with_actuator=True, is_pid=True)
        gain_map = gain_tuner.build_actuator_gain_map(stage, self._ROOT)
        self.assertIn(self._JOINT, gain_map)
        gains = gain_map[self._JOINT]
        self.assertTrue(gains.is_pid)
        self.assertAlmostEqual(gains.kp, 300.0)
        self.assertAlmostEqual(gains.kd, 30.0)
        self.assertAlmostEqual(gains.ki, 5.0)

    async def test_build_actuator_gain_map_no_scope(self) -> None:
        """Missing ``Actuators`` scope yields an empty map (defensive)."""
        stage, _ = self._make_stage(with_drive=True, with_actuator=False)
        self.assertEqual(gain_tuner.build_actuator_gain_map(stage, self._ROOT), {})
        self.assertEqual(gain_tuner.build_actuator_gain_map(None, self._ROOT), {})

    # ------------------------------------------------------------------
    # Drive/actuator gain-source *display* matrix (custom articulations).
    #
    # For each of the four authoring configs (drive-only, actuator-only, both,
    # neither) assert what `build_actuator_gain_map` maps and what
    # `resolve_joint_gains` reports (source / is_active / multiple_sources /
    # active_source).  Newton actuators are backend-independent, so the active
    # source does not depend on the active backend; the per-joint viewed-source
    # override selects a comparison view for the both-sources case.
    # ------------------------------------------------------------------

    def _resolve(self, joint, stage, *, viewed_source=None, mujoco_solver_active: bool = False):
        """Resolve gains for `joint`, optionally forcing a viewed source."""
        ctx = gain_tuner.GainReadContext(
            actuator_map=gain_tuner.build_actuator_gain_map(stage, self._ROOT),
            mjc_map=gain_tuner.build_mjc_gain_map(stage),
            mujoco_solver_active=mujoco_solver_active,
        )
        return gain_tuner.resolve_joint_gains(joint, None, ctx, viewed_source=viewed_source)

    async def test_actuator_map_membership_per_config(self) -> None:
        """`build_actuator_gain_map` maps the DOF only when an actuator is authored."""
        # Drive-only and neither: DOF is absent from the actuator map.
        for with_drive, with_actuator in ((True, False), (False, False)):
            stage, _ = self._make_stage(with_drive=with_drive, with_actuator=with_actuator)
            self.assertNotIn(self._JOINT, gain_tuner.build_actuator_gain_map(stage, self._ROOT))
        # Actuator-only and both: DOF maps to actuator gains.
        for with_drive in (False, True):
            stage, _ = self._make_stage(with_drive=with_drive, with_actuator=True)
            gain_map = gain_tuner.build_actuator_gain_map(stage, self._ROOT)
            self.assertIn(self._JOINT, gain_map)
            self.assertAlmostEqual(gain_map[self._JOINT].kp, 300.0)

    async def test_resolve_drive_only(self) -> None:
        """Drive-only: PhysicsDrive is the shown and active source."""
        stage, joint = self._make_stage(with_drive=True, with_actuator=False)
        resolved = self._resolve(joint, stage)
        self.assertEqual(resolved.source, gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertEqual(resolved.source_label, "Physics")
        self.assertTrue(resolved.is_active)
        self.assertFalse(resolved.multiple_sources)
        self.assertEqual(resolved.active_source, gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertAlmostEqual(resolved.kp, 100.0)
        self.assertAlmostEqual(resolved.kd, 10.0)
        self.assertIsNone(resolved.ki)
        # A position drive (stiffness + damping) resolves to the POSITION drive mode.
        self.assertEqual(resolved.drive_mode, gain_tuner.JointDriveMode.POSITION.value)

    async def test_resolve_actuator_only_backend_independent(self) -> None:
        """Actuator-only: the actuator is the active, editable source (backend-independent)."""
        stage, joint = self._make_stage(with_drive=False, with_actuator=True, is_pid=True)
        # An authored Newton actuator drives its joint under either backend, so it
        # is always the active source (the runtime evaluates it application-side).
        resolved = self._resolve(joint, stage)
        self.assertEqual(resolved.source, gain_tuner.GainSource.ACTUATOR)
        self.assertEqual(resolved.source_label, "Actuator")
        self.assertTrue(resolved.is_active)
        self.assertFalse(resolved.multiple_sources)
        self.assertEqual(resolved.active_source, gain_tuner.GainSource.ACTUATOR)
        self.assertAlmostEqual(resolved.kp, 300.0)
        self.assertAlmostEqual(resolved.ki, 5.0)

    async def test_resolve_both_sources_actuator_wins(self) -> None:
        """Both authored: the actuator is active/editable; the PhysicsDrive view is read-only."""
        stage, joint = self._make_stage(with_drive=True, with_actuator=True, is_pid=True)
        # Default view = the resolved active source = the actuator (backend-independent).
        act_active = self._resolve(joint, stage)
        self.assertEqual(act_active.source, gain_tuner.GainSource.ACTUATOR)
        self.assertTrue(act_active.is_active)
        self.assertEqual(act_active.active_source, gain_tuner.GainSource.ACTUATOR)
        # Forcing the PhysicsDrive comparison view is read-only (actuator drives it).
        pd_view = self._resolve(joint, stage, viewed_source=gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertEqual(pd_view.source, gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertFalse(pd_view.is_active)
        self.assertEqual(pd_view.active_source, gain_tuner.GainSource.ACTUATOR)
        # The read-only PhysicsDrive view still reads the joint's DriveAPI gains (no Ki).
        self.assertAlmostEqual(pd_view.kp, 100.0)
        self.assertIsNone(pd_view.ki)
        # multiple_sources is always True when both are authored.
        for resolved in (act_active, pd_view):
            self.assertTrue(resolved.multiple_sources)

    async def test_resolve_no_gains(self) -> None:
        """Neither source authored: source is NONE, inactive, zero gains."""
        stage, joint = self._make_stage(with_drive=False, with_actuator=False)
        resolved = self._resolve(joint, stage)
        self.assertEqual(resolved.source, gain_tuner.GainSource.NONE)
        self.assertEqual(resolved.source_label, "")
        self.assertFalse(resolved.is_active)
        self.assertFalse(resolved.multiple_sources)
        self.assertEqual(resolved.active_source, gain_tuner.GainSource.NONE)
        self.assertAlmostEqual(resolved.kp, 0.0)
        self.assertAlmostEqual(resolved.kd, 0.0)
        self.assertIsNone(resolved.ki)

    async def test_resolve_labels_physics_drive(self) -> None:
        """PhysicsDrive resolves to stiffness/damping labels with no Ki label."""
        stage, joint = self._make_stage(with_drive=True, with_actuator=False)
        resolved = self._resolve(joint, stage)
        self.assertEqual(resolved.kp_label, STIFFNESS_LABEL)
        self.assertEqual(resolved.kd_label, DAMPING_LABEL)
        self.assertIsNone(resolved.ki_label)

    async def test_resolve_labels_newton_pd(self) -> None:
        """A Newton PD actuator resolves to Kp/Kd labels with no Ki label."""
        stage, joint = self._make_stage(with_drive=False, with_actuator=True, is_pid=False)
        resolved = self._resolve(joint, stage)
        self.assertEqual(resolved.kp_label, KP_LABEL)
        self.assertEqual(resolved.kd_label, KD_LABEL)
        self.assertIsNone(resolved.ki_label)

    async def test_resolve_labels_newton_pid(self) -> None:
        """A Newton PID actuator resolves to Kp/Kd/Ki labels."""
        stage, joint = self._make_stage(with_drive=False, with_actuator=True, is_pid=True)
        resolved = self._resolve(joint, stage)
        self.assertEqual(resolved.kp_label, KP_LABEL)
        self.assertEqual(resolved.kd_label, KD_LABEL)
        self.assertEqual(resolved.ki_label, KI_LABEL)


class TestViewedSourceGating(omni.kit.test.AsyncTestCase):
    """Per-joint viewed-source selection + read-only gating (pure helpers)."""

    async def test_available_viewed_sources_order(self) -> None:
        """Available sources are ordered PhysicsDrive, MuJoCo, then Newton actuator."""
        self.assertEqual(gain_tuner.available_viewed_sources(True, False, False), [gain_tuner.GainSource.PHYSICS_DRIVE])
        self.assertEqual(gain_tuner.available_viewed_sources(False, True, False), [gain_tuner.GainSource.MUJOCO])
        self.assertEqual(gain_tuner.available_viewed_sources(False, False, True), [gain_tuner.GainSource.ACTUATOR])
        self.assertEqual(
            gain_tuner.available_viewed_sources(True, True, True),
            [gain_tuner.GainSource.PHYSICS_DRIVE, gain_tuner.GainSource.MUJOCO, gain_tuner.GainSource.ACTUATOR],
        )
        self.assertEqual(gain_tuner.available_viewed_sources(False, False, False), [])

    async def test_editable_only_when_viewed_matches_active(self) -> None:
        """A source is editable only when it is the active (consumed) source."""
        # PhysX active (active source == PhysicsDrive): only PhysicsDrive editable.
        self.assertTrue(
            gain_tuner.is_viewed_source_editable(
                gain_tuner.GainSource.PHYSICS_DRIVE, gain_tuner.GainSource.PHYSICS_DRIVE
            )
        )
        self.assertFalse(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.MUJOCO, gain_tuner.GainSource.PHYSICS_DRIVE)
        )
        self.assertFalse(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.ACTUATOR, gain_tuner.GainSource.PHYSICS_DRIVE)
        )
        # An actuator (active source == ACTUATOR, backend-independent): actuator editable.
        self.assertTrue(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.ACTUATOR, gain_tuner.GainSource.ACTUATOR)
        )
        self.assertFalse(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.PHYSICS_DRIVE, gain_tuner.GainSource.ACTUATOR)
        )
        self.assertFalse(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.MUJOCO, gain_tuner.GainSource.ACTUATOR)
        )
        # MuJoCo solver active (active source == MUJOCO): MuJoCo-native editable.
        self.assertTrue(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.MUJOCO, gain_tuner.GainSource.MUJOCO)
        )
        self.assertFalse(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.PHYSICS_DRIVE, gain_tuner.GainSource.MUJOCO)
        )
        self.assertFalse(
            gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.ACTUATOR, gain_tuner.GainSource.MUJOCO)
        )
        # MuJoCo-native is read-only whenever it is NOT the active source.
        for active in (gain_tuner.GainSource.PHYSICS_DRIVE, gain_tuner.GainSource.ACTUATOR, gain_tuner.GainSource.NONE):
            self.assertFalse(gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.MUJOCO, active))
        # NONE is never editable.
        self.assertFalse(gain_tuner.is_viewed_source_editable(gain_tuner.GainSource.NONE, gain_tuner.GainSource.NONE))

    async def test_mjc_params_to_drive_gains_round_trip(self) -> None:
        """mjc params recover the encoded gains for both position and velocity control."""
        # Position control round-trips stiffness and damping.
        pos = gain_tuner.drive_gains_to_mjc(2000.0, 400.0)
        kp, kd = gain_tuner.mjc_params_to_drive_gains(pos.gain_prm, pos.bias_prm)
        self.assertAlmostEqual(kp, 2000.0)
        self.assertAlmostEqual(kd, 400.0)
        # Velocity control recovers kp=0 and the damping term.
        vel = gain_tuner.drive_gains_to_mjc(0.0, 15.0)
        kp_v, kd_v = gain_tuner.mjc_params_to_drive_gains(vel.gain_prm, vel.bias_prm)
        self.assertAlmostEqual(kp_v, 0.0)
        self.assertAlmostEqual(kd_v, 15.0)

    async def test_mjc_params_to_drive_gains_defensive(self) -> None:
        """Missing / short param arrays recover zero gains without raising."""
        self.assertEqual(gain_tuner.mjc_params_to_drive_gains(None, None), (0.0, 0.0))
        self.assertEqual(gain_tuner.mjc_params_to_drive_gains([0.0], [0.0]), (0.0, 0.0))


class TestResolveViewedSource(omni.kit.test.AsyncTestCase):
    """`resolve_joint_gains` viewed-source override + read-only status."""

    _ROOT = "/World/Robot"
    _JOINT = "/World/Robot/joint0"
    _ACTUATOR = "/World/Robot/Actuators/act0"

    def _stage_with_all_sources(self):
        """In-memory stage whose joint carries DriveAPI + mjc + a Newton actuator."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, self._JOINT)
        joint = stage.GetPrimAtPath(self._JOINT)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        # MuJoCo-native gains authored directly on the joint (Menagerie style).
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([2000.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -2000.0, -400.0] + [0.0] * 7)
        # Newton actuator targeting the joint.
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        act = stage.DefinePrim(self._ACTUATOR, "Xform")
        act.AddAppliedSchema("NewtonPDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(self._JOINT))
        return stage, joint

    def _stage_drive_and_mjc(self):
        """In-memory stage whose joint carries DriveAPI + mjc gains (no actuator)."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, self._JOINT)
        joint = stage.GetPrimAtPath(self._JOINT)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([2000.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -2000.0, -400.0] + [0.0] * 7)
        joint.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.String).Set("fixed")
        joint.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")
        return stage, joint

    def _stage_drive_only(self):
        """In-memory stage whose joint carries only DriveAPI gains."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, self._JOINT)
        joint = stage.GetPrimAtPath(self._JOINT)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        return stage, joint

    def _ctx(self, stage, *, mujoco_solver_active: bool = False):
        """Read context for `stage`, optionally with the Newton MuJoCo solver active."""
        return gain_tuner.GainReadContext(
            actuator_map=gain_tuner.build_actuator_gain_map(stage, self._ROOT),
            mjc_map=gain_tuner.build_mjc_gain_map(stage),
            mujoco_solver_active=mujoco_solver_active,
        )

    async def test_mujoco_view_read_only_when_not_active(self) -> None:
        """Forcing the MuJoCo view when it is not the active source shows read-only mjc gains."""
        stage, joint = self._stage_with_all_sources()
        # An actuator is authored, so the actuator is the active source and the
        # forced MuJoCo comparison view is read-only.
        ctx = self._ctx(stage)
        resolved = gain_tuner.resolve_joint_gains(joint, "angular", ctx, viewed_source=gain_tuner.GainSource.MUJOCO)
        self.assertEqual(resolved.source, gain_tuner.GainSource.MUJOCO)
        self.assertFalse(resolved.is_active)
        self.assertAlmostEqual(resolved.kp, 2000.0)
        self.assertAlmostEqual(resolved.kd, 400.0)
        self.assertIsNone(resolved.ki)
        self.assertEqual(resolved.kp_label, MJC_STIFFNESS_LABEL)
        self.assertEqual(resolved.kd_label, MJC_DAMPING_LABEL)

    async def test_actuator_view_editable_backend_independent(self) -> None:
        """Newton actuator params are editable under either backend (backend-independent)."""
        stage, joint = self._stage_with_all_sources()
        for mjc_solver in (False, True):
            ctx = self._ctx(stage, mujoco_solver_active=mjc_solver)
            resolved = gain_tuner.resolve_joint_gains(
                joint, "angular", ctx, viewed_source=gain_tuner.GainSource.ACTUATOR
            )
            self.assertEqual(resolved.source, gain_tuner.GainSource.ACTUATOR)
            self.assertTrue(resolved.is_active)
            self.assertEqual(resolved.active_source, gain_tuner.GainSource.ACTUATOR)
            self.assertIsNotNone(resolved.kp_attr)

    async def test_physics_view_editable_when_drive_only(self) -> None:
        """PhysicsDrive params are editable when no actuator / active mjc overrides them."""
        stage, joint = self._stage_drive_only()
        ctx = self._ctx(stage)
        resolved = gain_tuner.resolve_joint_gains(
            joint, "angular", ctx, viewed_source=gain_tuner.GainSource.PHYSICS_DRIVE
        )
        self.assertEqual(resolved.source, gain_tuner.GainSource.PHYSICS_DRIVE)
        self.assertTrue(resolved.is_active)
        self.assertAlmostEqual(resolved.kp, 100.0)

    async def test_mujoco_editable_under_mujoco_solver(self) -> None:
        """MuJoCo-native gains become the active, editable source under the MuJoCo solver."""
        stage, joint = self._stage_drive_and_mjc()
        ctx = self._ctx(stage, mujoco_solver_active=True)
        # Default view = the resolved active source, which is now MuJoCo-native.
        resolved = gain_tuner.resolve_joint_gains(joint, "angular", ctx)
        self.assertEqual(resolved.source, gain_tuner.GainSource.MUJOCO)
        self.assertTrue(resolved.is_active)
        self.assertEqual(resolved.active_source, gain_tuner.GainSource.MUJOCO)
        self.assertAlmostEqual(resolved.kp, 2000.0)
        self.assertAlmostEqual(resolved.kd, 400.0)
        # The MjcGainSource is exposed so the editable view can write back.
        self.assertIsNotNone(resolved.mjc_source)

    async def test_mujoco_read_only_without_mujoco_solver(self) -> None:
        """MuJoCo-native gains stay read-only when the MuJoCo solver is not active."""
        stage, joint = self._stage_drive_and_mjc()
        ctx = self._ctx(stage)  # PhysicsDrive is the active source here
        resolved = gain_tuner.resolve_joint_gains(joint, "angular", ctx, viewed_source=gain_tuner.GainSource.MUJOCO)
        self.assertEqual(resolved.source, gain_tuner.GainSource.MUJOCO)
        self.assertFalse(resolved.is_active)
        self.assertEqual(resolved.active_source, gain_tuner.GainSource.PHYSICS_DRIVE)

    async def test_forced_source_absent_falls_back(self) -> None:
        """Forcing a source that is not authored falls back to the active source."""
        stage, joint = self._stage_drive_only()
        ctx = self._ctx(stage)
        # No mjc / actuator authored -> forcing MuJoCo falls back to PhysicsDrive.
        resolved = gain_tuner.resolve_joint_gains(joint, "angular", ctx, viewed_source=gain_tuner.GainSource.MUJOCO)
        self.assertEqual(resolved.source, gain_tuner.GainSource.PHYSICS_DRIVE)
