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

"""Gain table view: save targets, viewed-source toggle, and detail/table sync."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import omni.kit.test
import omni.ui as ui
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner import (
    GainReadContext,
    GainSource,
    SaveTargetCandidate,
    SaveTargetOptions,
    available_viewed_sources,
    build_actuator_gain_map,
    build_gain_save_plan,
    build_mjc_gain_map,
    damping_ratio_from_stiffness_damping_position_drive,
    is_viewed_source_editable,
    natural_frequency_hz_from_stiffness_position_drive,
    resolve_gain_write_targets,
    resolve_joint_gains,
    stiffness_and_damping_from_natural_frequency_position_drive,
)
from isaacsim.robot_setup.gain_tuner.gain_sources import MJC_DAMPING_LABEL, MJC_STIFFNESS_LABEL
from isaacsim.robot_setup.gain_tuner.ui import gain_table_view
from isaacsim.robot_setup.gain_tuner.ui.backend_context import BackendContext
from isaacsim.robot_setup.gain_tuner.ui.gain_table_model import (
    GAIN_COLUMN_KD,
    GAIN_COLUMN_KP,
    build_gain_table_rows,
    resolve_visible_columns,
    source_option_label,
)
from isaacsim.robot_setup.gain_tuner.ui.gain_table_view import GainTableView, _GainRowItem
from isaacsim.robot_setup.gain_tuner.ui.gains_tuner_backend import GainsTestMode
from isaacsim.robot_setup.gain_tuner.ui.style import MUTED_LABEL_COLOR
from isaacsim.robot_setup.gain_tuner.ui.ui_builder import TuningMode, UIBuilder
from isaacsim.robot_setup.gain_tuner.ui.ui_utils import set_wrapped_tooltip
from pxr import Sdf, Usd, UsdPhysics


class TestSaveTargetRow(omni.kit.test.AsyncTestCase):
    """Save-target row population and mjc/Newton mirror info state (F)."""

    _JOINT = "/World/Robot/joint0"
    _MJC_ACTUATOR = "/World/Robot/Physics/joint0_actuator"

    async def test_apply_save_targets_populates_candidates_and_default(self) -> None:
        """BackendContext.apply_save_targets fills the candidate list and default."""
        options = SaveTargetOptions(
            candidates=[
                SaveTargetCandidate("/a/payloads/Physics/physics.usda", "physics.usda", True),
                SaveTargetCandidate("/a/payloads/Physics/mujoco.usda", "mujoco.usda", False),
                SaveTargetCandidate("/a/payloads/Physics/physx.usda", "physx.usda", False),
            ],
            default_identifier="/a/payloads/Physics/physics.usda",
        )
        targets = SimpleNamespace(
            options=options,
            has_mjc_mirror=False,
            mjc_layer_names=[],
            has_newton_writeback=False,
            newton_layer_names=[],
        )
        ctx = BackendContext()
        ctx.apply_save_targets(targets)
        self.assertTrue(ctx.save_target_resolved)
        self.assertEqual(ctx.save_target, "physics.usda")
        self.assertEqual(ctx.save_target_identifier, "/a/payloads/Physics/physics.usda")
        self.assertEqual(
            [name for _, name in ctx.save_target_candidates], ["physics.usda", "mujoco.usda", "physx.usda"]
        )
        self.assertFalse(ctx.has_mirror_info)

    async def test_apply_save_targets_unresolved(self) -> None:
        """An empty options object clears the save-target fields."""
        targets = SimpleNamespace(
            options=SaveTargetOptions(),
            has_mjc_mirror=False,
            mjc_layer_names=[],
            has_newton_writeback=False,
            newton_layer_names=[],
        )
        ctx = BackendContext()
        ctx.apply_save_targets(targets)
        self.assertFalse(ctx.save_target_resolved)
        self.assertEqual(ctx.save_target_candidates, [])

    async def test_apply_save_targets_populates_mirror_fields_and_info(self) -> None:
        """With mjc + Newton sources opted in, the ctx exposes mirror fields + a two-target info line."""
        targets = SimpleNamespace(
            options=SaveTargetOptions(
                candidates=[SaveTargetCandidate("/a/physics.usda", "physics.usda", True)],
                default_identifier="/a/physics.usda",
            ),
            has_mjc_mirror=True,
            mjc_layer_names=["mujoco.usda"],
            has_newton_writeback=True,
            newton_layer_names=["physics.usda"],
            mirror_enabled=True,
            mirror_drive_to_mjc=True,
            will_mirror_mjc=True,
            will_write_newton=True,
            mjc_target_identifier="/a/mujoco.usda",
            newton_target_identifier="/a/physics.usda",
        )
        ctx = BackendContext()
        ctx.apply_save_targets(targets)
        # Mirror-source flags + the selected mjc target identifier populate for the pickers.
        self.assertTrue(ctx.has_mirror_sources)
        self.assertTrue(ctx.has_mjc_sources)
        self.assertTrue(ctx.has_newton_sources)
        self.assertTrue(ctx.mirror_drive_to_mjc)
        self.assertEqual(ctx.mjc_target_identifier, "/a/mujoco.usda")
        # The info line names both the MuJoCo and Newton targets.
        self.assertTrue(ctx.has_mirror_info)
        self.assertIn("MuJoCo", ctx.mirror_info)
        self.assertIn("mujoco.usda", ctx.mirror_info)
        self.assertIn("Newton", ctx.mirror_info)

    async def test_mirror_info_empty_without_sources(self) -> None:
        """No mjc / Newton sources -> no mirror info surfaced."""
        targets = SimpleNamespace(
            has_mjc_mirror=False,
            mjc_layer_names=[],
            has_newton_writeback=False,
            newton_layer_names=[],
        )
        self.assertEqual(BackendContext._format_mirror_info(targets), "")

    async def test_resolve_gain_write_targets_populates_row_from_stage(self) -> None:
        """End-to-end: a Physics-folder stage resolves candidates with physics default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
            os.makedirs(physics_dir)
            physics_path = os.path.join(physics_dir, "physics.usda")
            mujoco_path = os.path.join(physics_dir, "mujoco.usda")
            Sdf.Layer.CreateNew(os.path.join(physics_dir, "physx.usda")).Save()

            physics_stage = Usd.Stage.CreateNew(physics_path)
            UsdPhysics.RevoluteJoint.Define(physics_stage, self._JOINT)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(self._JOINT), "angular")
            drive.CreateStiffnessAttr(100.0)
            drive.CreateDampingAttr(10.0)
            physics_stage.Save()

            mujoco_layer = Sdf.Layer.CreateNew(mujoco_path)
            mujoco_layer.subLayerPaths.append(physics_path)
            mujoco_stage = Usd.Stage.Open(mujoco_path)
            actuator = mujoco_stage.DefinePrim(self._MJC_ACTUATOR, "MjcActuator")
            actuator.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0] * 10)
            actuator.CreateRelationship("mjc:target").AddTarget(Sdf.Path(self._JOINT))
            mujoco_stage.Save()

            stage = Usd.Stage.Open(mujoco_path)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)
            # Opt in to the DriveAPI->MuJoCo mirror so the target is named.
            targets = resolve_gain_write_targets([entry], stage, None, mirror_drive_to_mjc=True)

            ctx = BackendContext()
            ctx.apply_save_targets(targets)
            self.assertTrue(ctx.save_target_resolved)
            self.assertEqual(ctx.save_target, "physics.usda")
            self.assertEqual(
                {name for _, name in ctx.save_target_candidates},
                {"physics.usda", "physx.usda", "mujoco.usda"},
            )
            # The mjc actuator is discovered and the opt-in mirror names its target.
            self.assertTrue(ctx.has_mirror_info)
            self.assertIn("mujoco.usda", ctx.mirror_info)
            # Mirror sources + default mjc target are populated for the pickers.
            self.assertTrue(ctx.has_mirror_sources)
            self.assertTrue(ctx.has_mjc_sources)
            self.assertTrue(ctx.mirror_drive_to_mjc)
            self.assertTrue(ctx.mjc_target_identifier.endswith("mujoco.usda"))

    async def test_mirror_info_reflects_mjc_mirror_off_by_default(self) -> None:
        """Default (mjc mirror off): the info line states mjc:* will not be written."""
        targets = SimpleNamespace(
            options=SaveTargetOptions(
                candidates=[SaveTargetCandidate("/a/physics.usda", "physics.usda", True)],
                default_identifier="/a/physics.usda",
            ),
            has_mjc_mirror=True,
            mjc_layer_names=["mujoco.usda"],
            has_newton_writeback=False,
            newton_layer_names=[],
            mirror_enabled=True,
            mirror_drive_to_mjc=False,
            will_mirror_mjc=False,
            mjc_target_identifier="/a/mujoco.usda",
            newton_target_identifier="",
        )
        ctx = BackendContext()
        ctx.apply_save_targets(targets)
        self.assertFalse(ctx.mirror_drive_to_mjc)
        self.assertTrue(ctx.has_mirror_info)
        self.assertIn("off", ctx.mirror_info.lower())
        self.assertIn("mjc:*", ctx.mirror_info)

    async def test_mjc_mirror_checkbox_default_off_gates_plan(self) -> None:
        """The opt-in mjc mirror checkbox is OFF by default and gates mjc:* edits.

        A fresh handler flag starts False (no mjc:* edits in the save plan); once
        the checkbox handler flips it on, the plan mirrors DriveAPI Kp/Kd into the
        joint's ``mjc:*`` params.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            physics_dir = os.path.join(tmp_dir, "payloads", "Physics")
            os.makedirs(physics_dir)
            physics_path = os.path.join(physics_dir, "physics.usda")
            mujoco_path = os.path.join(physics_dir, "mujoco.usda")

            physics_stage = Usd.Stage.CreateNew(physics_path)
            UsdPhysics.RevoluteJoint.Define(physics_stage, self._JOINT)
            drive = UsdPhysics.DriveAPI.Apply(physics_stage.GetPrimAtPath(self._JOINT), "angular")
            drive.CreateStiffnessAttr(150.0)
            drive.CreateDampingAttr(15.0)
            physics_stage.Save()

            mujoco_layer = Sdf.Layer.CreateNew(mujoco_path)
            mujoco_layer.subLayerPaths.append(physics_path)
            mujoco_stage = Usd.Stage.Open(mujoco_path)
            actuator = mujoco_stage.DefinePrim(self._MJC_ACTUATOR, "MjcActuator")
            actuator.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0] * 10)
            actuator.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0] * 10)
            actuator.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")
            actuator.CreateRelationship("mjc:target").AddTarget(Sdf.Path(self._JOINT))
            mujoco_stage.Save()

            stage = Usd.Stage.Open(mujoco_path)
            entry = SimpleNamespace(joint=stage.GetPrimAtPath(self._JOINT), drive_axis=None)

            # A builder starts with the opt-in mirror OFF.
            builder = UIBuilder.__new__(UIBuilder)
            builder._mirror_drive_to_mjc = False
            builder._save_target_frame = None
            self.assertFalse(builder._mirror_drive_to_mjc)

            # Default (checkbox off): the plan authors no mjc:* edits.
            plan_off = build_gain_save_plan([entry], stage, mirror_drive_to_mjc=builder._mirror_drive_to_mjc)
            names_off = [e.prop_path.name for edits in plan_off.values() for e in edits]
            self.assertFalse(any(n.startswith("mjc:") for n in names_off))

            # Flip the checkbox on via its handler, then rebuild the plan.
            builder._on_mirror_toggle_changed(SimpleNamespace(get_value_as_bool=lambda: True))
            self.assertTrue(builder._mirror_drive_to_mjc)
            plan_on = build_gain_save_plan([entry], stage, mirror_drive_to_mjc=builder._mirror_drive_to_mjc)
            names_on = [e.prop_path.name for edits in plan_on.values() for e in edits]
            self.assertIn("mjc:gainPrm", names_on)
            self.assertIn("mjc:biasPrm", names_on)
            # The mirrored gainPrm tracks the tuned DriveAPI stiffness (150).
            gain_edit = next(e for edits in plan_on.values() for e in edits if e.prop_path.name == "mjc:gainPrm")
            self.assertAlmostEqual(gain_edit.value[0], 150.0, places=4)


class TestViewportSyncTeardown(omni.kit.test.AsyncTestCase):
    """`sync_viewport_from_scroll_frame` must stay crash-safe across teardown."""

    async def test_cleanup_clears_viewport_refs(self) -> None:
        """`cleanup()` nils the viewport-sizing widget refs so later syncs short-circuit."""
        builder = UIBuilder.__new__(UIBuilder)
        builder.wrapped_ui_elements = []
        builder._gains_tuner = SimpleNamespace(reset=lambda: None)
        # Stale references to (now destroyed) widgets left over from a built UI.
        builder._scroll_frame_ref = object()
        builder._gains_table_hstack = object()

        builder.cleanup()

        self.assertIsNone(builder._scroll_frame_ref)
        self.assertIsNone(builder._gains_table_hstack)

    async def test_sync_viewport_noop_after_refs_cleared(self) -> None:
        """With the refs cleared (post-cleanup), sync is a no-op and does not raise."""
        builder = UIBuilder.__new__(UIBuilder)
        builder._scroll_frame_ref = None
        builder._gains_table_hstack = None
        builder._test_running = False
        builder._viewport_height = 0

        # Would raise (AttributeError) if the guard let it reach `.computed_height`.
        builder.sync_viewport_from_scroll_frame()
        builder.sync_viewport_from_scroll_frame(force=True)

    async def test_sync_viewport_short_circuits_before_reading_height(self) -> None:
        """A live scroll-frame ref but no table HStack must not read `computed_height`."""
        accessed = {"computed_height": False}

        class _Frame:
            @property
            def computed_height(self) -> float:
                accessed["computed_height"] = True
                return 600.0

        builder = UIBuilder.__new__(UIBuilder)
        builder._scroll_frame_ref = _Frame()
        builder._gains_table_hstack = None
        builder._test_running = False
        builder._viewport_height = 0

        builder.sync_viewport_from_scroll_frame(force=True)

        self.assertFalse(accessed["computed_height"])


class TestViewedSourceToggle(omni.kit.test.AsyncTestCase):
    """Per-joint viewed-source toggle + read-only gating in the detail editor."""

    _ROOT = "/World/Robot"
    _JOINT = "/World/Robot/joint0"
    _ACTUATOR = "/World/Robot/Actuators/act0"

    async def test_source_full_label_matches_shared_labels(self) -> None:
        """The viewed-source toggle labels are the shared source labels (unified).

        The detail-panel toggle and the table "Source" column must read identically
        for a given source, so ``_source_full_label`` delegates to the shared
        :func:`source_option_label` mapping (Drive / MuJoCo / Newton Actuator).
        """
        self.assertEqual(UIBuilder._source_full_label(GainSource.PHYSICS_DRIVE), "Drive")
        self.assertEqual(UIBuilder._source_full_label(GainSource.ACTUATOR), "Newton Actuator")
        self.assertEqual(UIBuilder._source_full_label(GainSource.MUJOCO), "MuJoCo")
        for source in (GainSource.PHYSICS_DRIVE, GainSource.ACTUATOR, GainSource.MUJOCO, GainSource.NONE):
            self.assertEqual(UIBuilder._source_full_label(source), source_option_label(source))

    async def test_viewed_source_enum_round_trips_keys(self) -> None:
        """Each toggle key maps back to the GainSource it was registered under."""
        for source, key in UIBuilder._VIEWED_SOURCE_KEYS.items():
            self.assertEqual(UIBuilder._viewed_source_enum(key), source)

    async def test_viewed_source_enum_none_for_unset(self) -> None:
        """An unset override resolves to None (default = active source view)."""
        self.assertIsNone(UIBuilder._viewed_source_enum(None))

    def _stage_with_all_sources(self):
        """In-memory stage whose joint carries DriveAPI + mjc + a Newton actuator."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([2000.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -2000.0, -400.0] + [0.0] * 7)
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        act = stage.DefinePrim(self._ACTUATOR, "Xform")
        act.AddAppliedSchema("NewtonPDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(self._JOINT))
        return stage, joint

    async def test_toggle_lists_all_three_available_sources(self) -> None:
        """A tri-source joint exposes PhysicsDrive, MuJoCo, and actuator segments."""
        stage, joint = self._stage_with_all_sources()
        builder = UIBuilder.__new__(UIBuilder)
        builder._detail_gain_ctx = GainReadContext(
            actuator_map=build_actuator_gain_map(stage, self._ROOT),
            mjc_map=build_mjc_gain_map(stage),
        )
        entry = SimpleNamespace(joint=joint, drive_axis="angular")
        has_pd, has_act, has_mjc = builder._joint_source_availability(entry)
        self.assertEqual(
            available_viewed_sources(has_pd, has_mjc, has_act),
            [GainSource.PHYSICS_DRIVE, GainSource.MUJOCO, GainSource.ACTUATOR],
        )

    async def test_mujoco_view_read_only_when_not_active(self) -> None:
        """Selecting the MuJoCo comparison view when it is not active yields read-only fields.

        Mirrors the exact resolution + read-only decision made in
        ``_build_joint_detail_editor`` / ``_build_controller_gains_section``.  The
        tri-source joint has a Newton actuator, so the actuator is the active
        source and the forced MuJoCo view is read-only with mjc labels.
        """
        stage, joint = self._stage_with_all_sources()
        builder = UIBuilder.__new__(UIBuilder)
        ctx = GainReadContext(
            actuator_map=build_actuator_gain_map(stage, self._ROOT),
            mjc_map=build_mjc_gain_map(stage),
        )
        builder._detail_gain_ctx = ctx
        resolved = resolve_joint_gains(joint, "angular", ctx, viewed_source=builder._viewed_source_enum("mujoco"))
        read_only = not is_viewed_source_editable(resolved.source, resolved.active_source)
        self.assertEqual(resolved.source, GainSource.MUJOCO)
        self.assertTrue(read_only)
        self.assertEqual(resolved.kp_label, MJC_STIFFNESS_LABEL)
        self.assertEqual(resolved.kd_label, MJC_DAMPING_LABEL)


class TestNaturalFrequencyConversionParams(omni.kit.test.AsyncTestCase):
    """The natural-frequency panel must resolve each joint's stored-gain convention.

    Angular drives author stiffness *and* damping per degree; linear drives author
    both in stage units. Feeding a linear drive through the angular convention (or
    the reverse) rescales an authored gain by 180/pi, so the convention has to come
    from the joint rather than being assumed.
    """

    @staticmethod
    def _builder(inertia: float = 0.85) -> UIBuilder:
        """A UIBuilder stubbed just enough to resolve NF conversion inputs."""
        builder = UIBuilder.__new__(UIBuilder)
        builder._gains_tuner = SimpleNamespace(get_joint_accumulated_inertia=lambda _joint: inertia)
        return builder

    @staticmethod
    def _stage_with_joints() -> tuple[Usd.Stage, Usd.Prim, Usd.Prim]:
        """A stage holding one revolute and one prismatic force-driven joint."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(Sdf.Path("/World"), "Xform")
        revolute = UsdPhysics.RevoluteJoint.Define(stage, "/World/revolute").GetPrim()
        prismatic = UsdPhysics.PrismaticJoint.Define(stage, "/World/prismatic").GetPrim()
        for prim, axis in ((revolute, "angular"), (prismatic, "linear")):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(1.0)
            drive.CreateDampingAttr(1.0)
        return stage, revolute, prismatic

    async def test_revolute_joint_uses_angular_convention(self) -> None:
        """A revolute drive reports the per-degree convention."""
        _stage, revolute, _prismatic = self._stage_with_joints()
        params = self._builder()._nf_conversion_params(SimpleNamespace(joint=revolute, drive_axis="angular"))
        self.assertTrue(params["is_angular"])
        self.assertTrue(params["use_force"])
        self.assertAlmostEqual(params["m_eq"], 0.85, places=9)

    async def test_prismatic_joint_uses_linear_convention(self) -> None:
        """A prismatic drive must not be handed the angular per-degree convention."""
        _stage, _revolute, prismatic = self._stage_with_joints()
        params = self._builder()._nf_conversion_params(SimpleNamespace(joint=prismatic, drive_axis="linear"))
        self.assertFalse(params["is_angular"])

    async def test_d6_drive_axes_pick_convention_from_the_axis_token(self) -> None:
        """A D6 joint has no joint type to read, so the drive axis decides the convention."""
        stage = Usd.Stage.CreateInMemory()
        d6 = stage.DefinePrim(Sdf.Path("/World/d6"), "PhysicsJoint")
        builder = self._builder()
        for axis, expected_angular in (("rotX", True), ("rotZ", True), ("transX", False), ("transY", False)):
            UsdPhysics.DriveAPI.Apply(d6, axis).CreateTypeAttr("force")
            params = builder._nf_conversion_params(SimpleNamespace(joint=d6, drive_axis=axis))
            self.assertEqual(params["is_angular"], expected_angular, msg=f"drive axis {axis}")

    async def test_acceleration_drive_reports_no_force_drive(self) -> None:
        """An acceleration drive cancels inertia, so it must not be treated as a force drive."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/accel").GetPrim()
        UsdPhysics.DriveAPI.Apply(joint, "angular").CreateTypeAttr("acceleration")
        params = self._builder()._nf_conversion_params(SimpleNamespace(joint=joint, drive_axis="angular"))
        self.assertFalse(params["use_force"])
        self.assertTrue(params["is_angular"])

    async def test_prismatic_drive_supports_natural_frequency_mode(self) -> None:
        """Prismatic PhysicsDrive gains keep the NF toggle, now that the math handles them."""
        _stage, _revolute, prismatic = self._stage_with_joints()
        resolved = SimpleNamespace(source=GainSource.PHYSICS_DRIVE, is_active=True, kp_attr=object(), kd_attr=object())
        entry = SimpleNamespace(joint=prismatic, drive_axis="linear")
        self.assertTrue(UIBuilder._supports_natural_frequency_mode(None, entry, resolved))

    async def test_conversion_params_round_trip_through_drive_math(self) -> None:
        """Resolved params drive an authored-gain round trip for both joint types.

        Ties the UI's convention choice to the drive math: a joint tuned to a target
        frequency and damping ratio must read those same values back.
        """
        _stage, revolute, prismatic = self._stage_with_joints()
        for prim, axis in ((revolute, "angular"), (prismatic, "linear")):
            params = self._builder()._nf_conversion_params(SimpleNamespace(joint=prim, drive_axis=axis))
            fn, zeta = 7.0, 0.45
            stiffness, damping = stiffness_and_damping_from_natural_frequency_position_drive(
                fn, zeta, is_angular=params["is_angular"], use_force_drive=params["use_force"], m_eq=params["m_eq"]
            )
            drive = UsdPhysics.DriveAPI(prim, axis)
            drive.GetStiffnessAttr().Set(stiffness)
            drive.GetDampingAttr().Set(damping)
            fn_read = natural_frequency_hz_from_stiffness_position_drive(
                drive.GetStiffnessAttr().Get(),
                is_angular=params["is_angular"],
                use_force_drive=params["use_force"],
                m_eq=params["m_eq"],
            )
            zeta_read = damping_ratio_from_stiffness_damping_position_drive(
                drive.GetDampingAttr().Get(),
                drive.GetStiffnessAttr().Get(),
                is_angular=params["is_angular"],
                use_force_drive=params["use_force"],
                m_eq=params["m_eq"],
            )
            self.assertAlmostEqual(fn_read, fn, places=5, msg=f"{axis} natural frequency")
            self.assertAlmostEqual(zeta_read, zeta, places=5, msg=f"{axis} damping ratio")


class TestRobotSwitchReset(omni.kit.test.AsyncTestCase):
    """Switching robots must fully reset per-robot test/selection/plot state."""

    @staticmethod
    def _builder_with_stale_state() -> UIBuilder:
        """A UIBuilder pre-loaded with state left over from a previous robot."""
        builder = UIBuilder.__new__(UIBuilder)
        # Stale state from a previously loaded / tested robot.
        builder._test_running = True
        builder._test_mode = GainsTestMode.SINUSOIDAL
        builder._last_test_mode = GainsTestMode.STEP
        builder._selected_joint_index = 4
        builder._gain_column_overrides = {"kp": False, "armature": True}
        builder._name_search_query = "wrist"
        builder._joint_viewed_source = {"/World/Robot/wrist_joint": GainSource.ACTUATOR}
        builder._pending_table_selection = {1, 2}
        builder._detail_tuning_mode = TuningMode.NATURAL_FREQUENCY
        builder._detail_gain_ctx = object()
        builder._plotting_indices = [1, 2, 3]
        builder._plotting_group_colors = {1: (0xFF0000FF, 0xFF00FF00)}
        builder._test_effort_history = [object()]
        builder._test_effort_times = [0.1, 0.2]
        builder._make_plot_on_next_frame = True
        builder._test_start_time = 5.0
        builder._test_total_duration = 12.0
        builder._test_num_sequences = 6
        builder._test_seq_duration = 2.0
        builder._test_elapsed_sim = 7.0
        builder._selected_save_target_identifier = "/a/physx.usda"
        builder._selected_mjc_target_identifier = "/a/mujoco.usda"
        builder._selected_newton_target_identifier = "/a/physics.usda"
        builder._mirror_writeback_enabled = False
        builder._mirror_drive_to_mjc = True
        builder._nav_show_charts = True
        # Widgets absent in this headless unit test: the reset must guard on None.
        builder._charts_button = None
        builder._gains_settings_button = None
        builder._gain_settings_page = None
        builder._charts_page = None
        builder._test_inline_widget = None
        builder._test_controls_frame = None
        return builder

    async def test_reset_clears_stale_state(self) -> None:
        """`_reset_robot_ui_state` returns every per-robot field to its default."""
        builder = self._builder_with_stale_state()

        builder._reset_robot_ui_state()

        self.assertFalse(builder._test_running)
        self.assertEqual(builder._test_mode, GainsTestMode.SNAP_TO_LIMITS)
        self.assertIsNone(builder._last_test_mode)
        self.assertIsNone(builder._selected_joint_index)
        self.assertEqual(builder._joint_viewed_source, {})
        self.assertIsNone(builder._pending_table_selection)
        self.assertEqual(builder._detail_tuning_mode, TuningMode.STIFFNESS)
        self.assertIsNone(builder._detail_gain_ctx)
        self.assertEqual(builder._plotting_indices, [])
        self.assertEqual(builder._plotting_group_colors, {})
        self.assertEqual(builder._test_effort_history, [])
        self.assertEqual(builder._test_effort_times, [])
        self.assertFalse(builder._make_plot_on_next_frame)
        self.assertEqual(builder._test_total_duration, 0.0)
        self.assertEqual(builder._test_num_sequences, 0)
        self.assertEqual(builder._test_elapsed_sim, 0.0)
        self.assertIsNone(builder._selected_save_target_identifier)
        self.assertIsNone(builder._selected_mjc_target_identifier)
        self.assertIsNone(builder._selected_newton_target_identifier)
        self.assertTrue(builder._mirror_writeback_enabled)
        self.assertFalse(builder._mirror_drive_to_mjc)
        self.assertFalse(builder._nav_show_charts)
        # The gain-table column overrides + name filter reset so a new robot starts
        # from the schema-driven auto-selection with no search filter.
        self.assertEqual(builder._gain_column_overrides, {})
        self.assertEqual(builder._name_search_query, "")

    async def test_reset_keeps_test_gains_tab_enabled(self) -> None:
        """A robot swap must leave the Test Gains tab clickable (regression).

        The Test Gains tab hosts the test *run* controls (the Run Test button), so
        disabling it until a test completes is a deadlock: the only way to run a
        test is from that tab, so it could never be re-enabled.  ``_reset_robot_ui_state``
        must therefore keep the tab enabled through a robot swap and only reset the
        active tab back to Gain Settings.
        """
        builder = self._builder_with_stale_state()
        # A fake nav-tab button that records its enabled state (the real widget is a
        # ``ui.Button``; ``set_style`` is a no-op here).
        builder._charts_button = SimpleNamespace(enabled=False, set_style=lambda *a, **k: None)

        builder._reset_robot_ui_state()

        self.assertTrue(builder._charts_button.enabled)
        self.assertFalse(builder._nav_show_charts)


class TestDetailToTableSync(omni.kit.test.AsyncTestCase):
    """Two-way gain sync between the detail panel and the table (no rebuild).

    The detail panel and the table each snapshot gain values into their own
    ``ui.SimpleFloatModel`` cells at build time, so an edit on one side must push
    the fresh value into the other side's live models in place (targeted, no
    rebuild).  These tests check both directions end to end after a new value is
    authored to USD: detail -> table (:meth:`GainTableView.refresh_row_cells`) and
    table -> detail (:meth:`UIBuilder._refresh_detail_fields_for_selected`).
    """

    async def test_refresh_row_cells_updates_live_field_models(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.RevoluteJoint.Define(stage, "/J")
        joint = stage.GetPrimAtPath("/J")
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(10.0)
        drive.CreateDampingAttr(2.0)
        entry = SimpleNamespace(joint=joint, drive_axis=None, display_name="J")
        ctx = GainReadContext()

        rows = build_gain_table_rows([entry], ctx)
        columns = resolve_visible_columns(rows)

        # Build a view without its UI widgets and simulate the rendered Kp/Kd cells
        # (field models are populated lazily by the delegate as cells render).
        view = object.__new__(GainTableView)
        view._items = [_GainRowItem(rows[0])]
        view._columns = columns
        view._suspend_writes = False
        kp_model = ui.SimpleFloatModel(10.0)
        kd_model = ui.SimpleFloatModel(2.0)
        view._items[0].field_models = {GAIN_COLUMN_KP: kp_model, GAIN_COLUMN_KD: kd_model}

        # Author a new stiffness (as a detail-panel edit would) and re-resolve the
        # joint the same way _sync_selected_row_to_table does.
        drive.GetStiffnessAttr().Set(55.0)
        fresh = build_gain_table_rows([entry], ctx)[0]
        view.refresh_row_cells("/J", fresh.resolved, fresh.advanced_cells)

        self.assertAlmostEqual(kp_model.get_value_as_float(), 55.0)
        self.assertAlmostEqual(kd_model.get_value_as_float(), 2.0)
        # The row's cached resolved is swapped so a later selection edit reads it.
        self.assertAlmostEqual(view._items[0].row.resolved.kp, 55.0)

    async def test_refresh_detail_fields_updates_live_models(self) -> None:
        """A table edit (new USD stiffness) refreshes the detail panel's field models.

        Mirror of the detail -> table path: after a table cell writes a new
        stiffness to USD, ``_refresh_detail_fields_for_selected`` re-resolves the
        selected joint and updates the registered Controller-Gains models in place
        (Stiffness/Kp changes; Damping/Kd is untouched), without rebuilding the
        panel.  Uses a bare ``UIBuilder`` wired with just the attributes the method
        reads, so no window/widgets are built.
        """
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.RevoluteJoint.Define(stage, "/J")
        joint = stage.GetPrimAtPath("/J")
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(10.0)
        drive.CreateDampingAttr(2.0)
        entry = SimpleNamespace(joint=joint, drive_axis=None, display_name="J")

        gb = object.__new__(UIBuilder)
        gb._detail_gain_ctx = GainReadContext()
        gb._joint_viewed_source = {}
        gb._selected_joint_index = 0
        gb._suspend_detail_writes = False
        gb._detail_nf_params = None
        gb._detail_adv_models = []
        kp_model = ui.SimpleFloatModel(10.0)
        kd_model = ui.SimpleFloatModel(2.0)
        gb._detail_gain_models = {GAIN_COLUMN_KP: kp_model, GAIN_COLUMN_KD: kd_model}
        gb._get_tunable_joint_entries = lambda: [entry]

        # A table edit committed a new stiffness to USD; refresh the detail fields.
        drive.GetStiffnessAttr().Set(77.0)
        gb._refresh_detail_fields_for_selected()

        self.assertAlmostEqual(kp_model.get_value_as_float(), 77.0)
        self.assertAlmostEqual(kd_model.get_value_as_float(), 2.0)


class TestAdvancedCellRenderingFollowsTheEdit(omni.kit.test.AsyncTestCase):
    """An advanced *cell* keeps claiming its number is the engine's, or did.

    The table's counterpart to the detail panel's version of the same bug.  A cell
    for an unauthored parameter is built with ``"(default)"`` in the drag's
    ``format`` and a muted colour, both snapshotted at construction, so a write
    left the cell marking the value the user had just chosen as the engine's.

    ``refresh_row_cells`` had a second half of the same problem: it replaced the
    row's ``advanced_cells`` but not its ``param_displays``, and the rendered number
    comes from the displays -- so a cell the delegate rebuilt after an in-place
    refresh went back to the pre-edit number.
    """

    _JOINT = "/Robot/joint"

    def _newton_row_setup(self, authored: float | None = None):
        """A one-joint Newton table whose armature cell is unauthored by default."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(10.0)
        drive.CreateDampingAttr(2.0)
        joint.ApplyAPI(gain_tuner.NEWTON_JOINT_API)
        spec = gain_tuner.joint_param_spec("armature")
        if authored is not None:
            joint.GetAttribute(spec.newton_attr).Set(authored)
        entry = SimpleNamespace(joint=joint, drive_axis=None, display_name="joint")
        ctx = GainReadContext(
            viewed_backend=gain_tuner.BACKEND_NEWTON,
            active_backend=gain_tuner.BACKEND_NEWTON,
            solver=gain_tuner.SOLVER_XPBD,
        )
        self._stage = stage
        return joint, entry, ctx, spec

    def _view_with_rendered_cell(self, rows, columns, column_key: str):
        """A view with one row whose advanced cell is a real, built ``FloatDrag``.

        The delegate populates ``field_models`` / ``field_widgets`` lazily as cells
        render, so a headless test stands in for that by building the same widget
        from the same display -- which is what makes this an assertion about the
        widget rather than about the display it was built from.  The cell's tooltip
        is installed the way the delegate installs it, so a test can watch it go
        stale.
        """
        view = object.__new__(GainTableView)
        view._items = [_GainRowItem(rows[0])]
        view._columns = columns
        view._suspend_writes = False
        view._read_row_fn = None
        display = rows[0].param_display(column_key)
        window = ui.Window("gain_tuner_adv_cell_test", width=320, height=120)
        # ``format`` is omitted rather than passed as None, exactly as the delegate
        # does: ``ui.FloatDrag`` refuses a None format.
        drag_kwargs = {"format": display.field_format} if display.field_format else {}
        with window.frame:
            field = ui.FloatDrag(
                model=ui.SimpleFloatModel(display.field_value),
                style={"color": MUTED_LABEL_COLOR} if not display.authored else {},
                **drag_kwargs,
            )
        set_wrapped_tooltip(field, rows[0].param_info(column_key) or display.note)
        view._items[0].field_models = {column_key: field.model}
        view._items[0].field_widgets = {column_key: field}
        self._window = window
        return view, field

    def _recorded_tooltips(self) -> list[str]:
        """Patch in a recording ``set_wrapped_tooltip`` and return its log.

        ``omni.ui`` offers no way to read a dynamic tooltip's builder back off the
        widget, and it builds the body lazily on the first hover, so the text a cell
        would pop up cannot be read from the widget the way ``format`` and ``style``
        can.  The recorder calls straight through, so the widget really is given the
        tooltip; what is being asserted is which text it was given.
        """
        recorded: list[str] = []

        def recording(widget, text, *args, **kwargs) -> None:
            recorded.append(text)
            set_wrapped_tooltip(widget, text, *args, **kwargs)

        patcher = mock.patch.object(gain_table_view, "set_wrapped_tooltip", recording)
        patcher.start()
        self.addCleanup(patcher.stop)
        return recorded

    async def tearDown(self) -> None:
        window = getattr(self, "_window", None)
        if window is not None:
            window.destroy()
        self._window = None

    async def test_a_written_cell_stops_saying_default(self) -> None:
        """The cell the user dragged, re-derived from the stage it just wrote."""
        joint, entry, ctx, spec = self._newton_row_setup()
        rows = build_gain_table_rows([entry], ctx)
        columns = resolve_visible_columns(rows)
        view, field = self._view_with_rendered_cell(rows, columns, "armature")
        self.assertIn("default", field.format, "precondition: built as an engine default")

        # What the write path does: author the value, then re-derive the rendering
        # of every row the edit landed on.
        joint.GetAttribute(spec.newton_attr).Set(0.5)
        view._read_row_fn = lambda _row: build_gain_table_rows([entry], ctx)[0]
        view._rerender_written_cells(view._items, "armature")

        self.assertNotIn("default", field.format, f"the cell still claims a default: {field.format!r}")
        self.assertNotEqual(field.style.get("color"), MUTED_LABEL_COLOR)
        self.assertTrue(view._items[0].row.param_display("armature").authored)

    async def test_an_already_authored_cell_is_not_re_read(self) -> None:
        """The skip that keeps a drag from re-reading USD once per frame per row.

        The unauthored-to-authored transition runs one way, so a cell that already
        renders a plain number has nothing left to re-derive.
        """
        joint, entry, ctx, spec = self._newton_row_setup(authored=0.25)
        rows = build_gain_table_rows([entry], ctx)
        columns = resolve_visible_columns(rows)
        view, field = self._view_with_rendered_cell(rows, columns, "armature")
        reads = []
        view._read_row_fn = lambda row: reads.append(row) or build_gain_table_rows([entry], ctx)[0]

        view._rerender_written_cells(view._items, "armature")

        self.assertEqual(reads, [])
        self.assertNotIn("default", field.format)

    async def test_a_written_cell_stops_explaining_a_value_it_no_longer_has(self) -> None:
        """The cell's hover text is the detail panel's sentence, and lagged the same.

        The cell repainted its number and dropped ``"(default)"``, then still popped
        up "Nothing is authored here" over the value the user had just authored.
        ``omni.ui`` builds a dynamic tooltip's body once and serves it forever
        after, so the builder has to be replaced -- and, with nothing left to
        explain, removed.
        """
        joint, entry, ctx, spec = self._newton_row_setup()
        rows = build_gain_table_rows([entry], ctx)
        columns = resolve_visible_columns(rows)
        view, field = self._view_with_rendered_cell(rows, columns, "armature")
        self.assertTrue(field.has_tooltip_fn(), "precondition: the cell explains its engine default")

        joint.GetAttribute(spec.newton_attr).Set(0.5)
        view._read_row_fn = lambda _row: build_gain_table_rows([entry], ctx)[0]
        view._rerender_written_cells(view._items, "armature")

        self.assertNotIn("default", field.format, "precondition: the cell itself followed the edit")
        self.assertFalse(
            field.has_tooltip_fn(),
            "the cell still pops up the sentence it was built with, which said nothing was authored",
        )

    async def test_a_detail_edit_gives_the_cell_the_sentence_that_replaced_it(self) -> None:
        """The other direction, through the detail panel: a tooltip with new wording.

        Newton reads PhysX's armature while its own half is unauthored, so the cell
        explains the fallback.  Authoring Newton's half in the detail panel makes the
        two backends disagree, which is what the cell should offer instead -- and
        this path hands the row freshly read notes, so it is not subject to the
        re-read skip :meth:`_rerender_written_cells` applies to an already-authored
        cell.
        """
        joint, entry, ctx, spec = self._newton_row_setup()
        joint.ApplyAPI(gain_tuner.PHYSX_JOINT_API)
        joint.GetAttribute(spec.physx_attr).Set(0.25)
        rows = build_gain_table_rows([entry], ctx)
        columns = resolve_visible_columns(rows)
        view, _field = self._view_with_rendered_cell(rows, columns, "armature")
        recorded = self._recorded_tooltips()

        joint.GetAttribute(spec.newton_attr).Set(0.5)
        fresh = build_gain_table_rows([entry], ctx)[0]
        view.refresh_row_cells(
            self._JOINT,
            fresh.resolved,
            fresh.advanced_cells,
            param_infos=fresh.param_infos,
            param_displays=fresh.param_displays,
        )

        self.assertEqual(len(recorded), 1, f"the cell's tooltip was not re-derived: {recorded!r}")
        self.assertNotIn("falls back", recorded[0], f"the cell still explains a fallback: {recorded[0]!r}")
        self.assertIn(f"{spec.newton_attr} is 0.5", recorded[0])
        self.assertIn(f"{spec.physx_attr} is 0.25", recorded[0])

    async def test_refresh_row_cells_replaces_the_displays_it_renders_from(self) -> None:
        """The nit with teeth: the rendered number comes from ``param_displays``.

        Only ``advanced_cells`` was replaced, so a cell rebuilt after a detail-panel
        edit read its number out of a display describing the stage as it was before
        the edit.
        """
        joint, entry, ctx, spec = self._newton_row_setup()
        rows = build_gain_table_rows([entry], ctx)
        columns = resolve_visible_columns(rows)
        view, field = self._view_with_rendered_cell(rows, columns, "armature")

        joint.GetAttribute(spec.newton_attr).Set(0.75)
        fresh = build_gain_table_rows([entry], ctx)[0]
        view.refresh_row_cells(
            self._JOINT,
            fresh.resolved,
            fresh.advanced_cells,
            param_infos=fresh.param_infos,
            param_displays=fresh.param_displays,
        )

        display = view._items[0].row.param_display("armature")
        self.assertTrue(
            display.authored,
            "the row still holds its pre-edit display, so a cell the delegate rebuilds after this "
            "refresh renders the old number and marks it as an engine default",
        )
        self.assertAlmostEqual(display.value, 0.75, places=5)
        self.assertNotIn("default", field.format)
