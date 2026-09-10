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

"""Gain table model: columns, source cells, selection edits, and joint filters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner import GainReadContext, GainSource, build_actuator_gain_map, build_mjc_gain_map
from isaacsim.robot_setup.gain_tuner.ui.backend_context import BackendContext
from isaacsim.robot_setup.gain_tuner.ui.gain_table_model import (
    BACKEND_MUJOCO,
    BACKEND_PHYSX,
    CATEGORY_DRIVES,
    CATEGORY_MUJOCO_JOINT,
    DETAIL_PANEL_MULTI,
    DETAIL_PANEL_NONE,
    DETAIL_PANEL_SINGLE,
    GAIN_COLUMN_CATALOG,
    GAIN_COLUMN_KD,
    GAIN_COLUMN_KI,
    GAIN_COLUMN_KP,
    GAIN_COLUMN_SOURCE,
    auto_visible_column_keys,
    build_gain_table_rows,
    detail_panel_mode,
    gain_column_spec,
    grouped_columns_for_menu,
    is_angular_dof,
    mass_edit_row_indices,
    menu_backend_rows,
    menu_has_mjc,
    read_param_infos,
    resolve_visible_columns,
    selection_edit_row_indices,
    source_cell_model,
    source_option_label,
    visible_gain_columns,
)
from isaacsim.robot_setup.gain_tuner.ui.ui_builder import UIBuilder
from pxr import Sdf, Usd, UsdPhysics


class TestDetailPanelMode(omni.kit.test.AsyncTestCase):
    """The per-joint detail panel is shown only for a single selection."""

    async def test_single_selection_shows_panel(self) -> None:
        """Exactly one selected joint -> show the full per-joint detail editor."""
        self.assertEqual(detail_panel_mode(1), DETAIL_PANEL_SINGLE)

    async def test_no_selection_hides_panel(self) -> None:
        """No selected joints (and defensive negative counts) -> empty-selection prompt."""
        self.assertEqual(detail_panel_mode(0), DETAIL_PANEL_NONE)
        self.assertEqual(detail_panel_mode(-1), DETAIL_PANEL_NONE)

    async def test_multi_selection_hides_panel(self) -> None:
        """Two or more selected joints -> hide the panel (edit in the table)."""
        for count in (2, 3, 7):
            self.assertEqual(detail_panel_mode(count), DETAIL_PANEL_MULTI)


class TestSourceCellModel(omni.kit.test.AsyncTestCase):
    """The per-row "Source" column model (static label vs inline dropdown)."""

    _ROOT = "/World/Robot"

    @staticmethod
    def _entry(joint, drive_axis="angular", name="joint"):
        return SimpleNamespace(joint=joint, drive_axis=drive_axis, display_name=name)

    def _drive_joint(self, stage, path, kp=100.0, kd=10.0):
        UsdPhysics.RevoluteJoint.Define(stage, path)
        joint = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(kp)
        drive.CreateDampingAttr(kd)
        return joint

    def _add_mjc_on_joint(self, joint):
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([2000.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -2000.0, -400.0] + [0.0] * 7)
        joint.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.String).Set("fixed")
        joint.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")

    def _add_actuator(self, stage, joint_path, act_path, *, is_pid):
        act = stage.DefinePrim(act_path, "Xform")
        act.AddAppliedSchema("NewtonPIDControlAPI" if is_pid else "NewtonPDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        if is_pid:
            act.CreateAttribute("newton:ki", Sdf.ValueTypeNames.Float).Set(5.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(joint_path))

    # -- pure model helper --

    async def test_source_option_labels_are_ascii(self) -> None:
        """The compact source labels match the requested strings and are ASCII-only."""
        self.assertEqual(source_option_label(GainSource.PHYSICS_DRIVE), "Drive")
        self.assertEqual(source_option_label(GainSource.MUJOCO), "MuJoCo")
        self.assertEqual(source_option_label(GainSource.ACTUATOR), "Newton Actuator")
        self.assertEqual(source_option_label(GainSource.NONE), "-")
        for source in (GainSource.PHYSICS_DRIVE, GainSource.MUJOCO, GainSource.ACTUATOR, GainSource.NONE):
            label = source_option_label(source)
            self.assertTrue(label.isascii(), f"non-ASCII source label: {label!r}")

    async def test_single_source_is_static_label(self) -> None:
        """One available source -> static label, no dropdown."""
        model = source_cell_model([GainSource.PHYSICS_DRIVE], GainSource.PHYSICS_DRIVE)
        self.assertFalse(model.is_dropdown)
        self.assertEqual(len(model.options), 1)
        self.assertEqual(model.selected_label, "Drive")

    async def test_no_source_is_static_dash(self) -> None:
        """No available source -> static ASCII dash label, no dropdown."""
        model = source_cell_model([], GainSource.NONE)
        self.assertFalse(model.is_dropdown)
        self.assertEqual(model.options, [])
        self.assertEqual(model.selected_label, "-")

    async def test_multi_source_is_dropdown_with_options(self) -> None:
        """Two+ available sources -> dropdown listing them in order; selected index tracks."""
        model = source_cell_model([GainSource.PHYSICS_DRIVE, GainSource.MUJOCO], GainSource.MUJOCO)
        self.assertTrue(model.is_dropdown)
        self.assertEqual([o.label for o in model.options], ["Drive", "MuJoCo"])
        self.assertEqual(model.selected_index(), 1)

    # -- available sources derived from the built rows --

    async def test_drive_only_row_single_source(self) -> None:
        """A DriveAPI-only joint offers only the Drive source (static label)."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint = self._drive_joint(stage, f"{self._ROOT}/j0")
        ctx = GainReadContext()
        row = build_gain_table_rows([self._entry(joint)], ctx)[0]
        self.assertEqual(row.available_sources, [GainSource.PHYSICS_DRIVE])
        model = source_cell_model(row.available_sources, row.resolved.source)
        self.assertFalse(model.is_dropdown)
        self.assertEqual(model.selected_label, "Drive")

    async def test_drive_plus_mjc_row_dropdown(self) -> None:
        """A DriveAPI + mjc joint offers Drive + MuJoCo; under PhysX Drive is active/editable."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint = self._drive_joint(stage, f"{self._ROOT}/j0")
        self._add_mjc_on_joint(joint)
        ctx = GainReadContext(mjc_map=build_mjc_gain_map(stage))
        row = build_gain_table_rows([self._entry(joint)], ctx)[0]
        self.assertEqual(row.available_sources, [GainSource.PHYSICS_DRIVE, GainSource.MUJOCO])
        model = source_cell_model(row.available_sources, row.resolved.source)
        self.assertTrue(model.is_dropdown)
        self.assertEqual([o.label for o in model.options], ["Drive", "MuJoCo"])
        # Default (no override) under PhysX: Drive is the active, editable source.
        self.assertEqual(row.resolved.source, GainSource.PHYSICS_DRIVE)
        self.assertTrue(row.editable)

    async def test_pd_and_pid_actuator_rows_include_newton(self) -> None:
        """PD/PID actuator joints offer Drive + Newton Actuator; the actuator is active."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        j_pd = self._drive_joint(stage, f"{self._ROOT}/j_pd")
        j_pid = self._drive_joint(stage, f"{self._ROOT}/j_pid")
        self._add_actuator(stage, f"{self._ROOT}/j_pd", f"{self._ROOT}/Actuators/pd", is_pid=False)
        self._add_actuator(stage, f"{self._ROOT}/j_pid", f"{self._ROOT}/Actuators/pid", is_pid=True)
        ctx = GainReadContext(actuator_map=build_actuator_gain_map(stage, self._ROOT))
        rows = build_gain_table_rows([self._entry(j_pd, name="pd"), self._entry(j_pid, name="pid")], ctx)
        for row in rows:
            self.assertEqual(row.available_sources, [GainSource.PHYSICS_DRIVE, GainSource.ACTUATOR])
            model = source_cell_model(row.available_sources, row.resolved.source)
            self.assertTrue(model.is_dropdown)
            self.assertIn("Newton Actuator", [o.label for o in model.options])
            # The Newton actuator is the active, editable source (backend-independent).
            self.assertEqual(row.resolved.source, GainSource.ACTUATOR)
            self.assertTrue(row.editable)
        # Ki applies only to the PID row.
        self.assertIsNone(rows[0].resolved.ki)
        self.assertIsNotNone(rows[1].resolved.ki)

    async def test_switching_source_reresolves_and_regates_editability(self) -> None:
        """Forcing a row's viewed source re-resolves its gains + read-only gating.

        A DriveAPI + Newton-actuator joint's default source is the actuator
        (active, editable).  Forcing the ``PhysicsDrive`` view (as the inline
        Source dropdown does via the shared per-joint map) shows the DriveAPI
        gains read-only, since the actuator is still the active source.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        joint = self._drive_joint(stage, f"{self._ROOT}/j0", kp=100.0, kd=10.0)
        self._add_actuator(stage, f"{self._ROOT}/j0", f"{self._ROOT}/Actuators/a0", is_pid=False)
        ctx = GainReadContext(actuator_map=build_actuator_gain_map(stage, self._ROOT))
        path = f"{self._ROOT}/j0"

        # Default: actuator active + editable.
        default_row = build_gain_table_rows([self._entry(joint)], ctx)[0]
        self.assertEqual(default_row.resolved.source, GainSource.ACTUATOR)
        self.assertTrue(default_row.editable)
        self.assertAlmostEqual(default_row.resolved.kp, 300.0)

        # Force the Drive view -> Drive values, read-only (actuator still active).
        forced_row = build_gain_table_rows(
            [self._entry(joint)], ctx, viewed_source_for={path: GainSource.PHYSICS_DRIVE}
        )[0]
        self.assertEqual(forced_row.resolved.source, GainSource.PHYSICS_DRIVE)
        self.assertFalse(forced_row.editable)
        self.assertAlmostEqual(forced_row.resolved.kp, 100.0)

    async def test_source_column_always_auto_visible_first(self) -> None:
        """The Source column is auto-on and is the first data column in catalog order."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint = self._drive_joint(stage, f"{self._ROOT}/j0")
        rows = build_gain_table_rows([self._entry(joint)], GainReadContext())
        self.assertIn(GAIN_COLUMN_SOURCE, auto_visible_column_keys(rows))
        visible = resolve_visible_columns(rows)
        self.assertEqual(visible[0].key, GAIN_COLUMN_SOURCE)
        self.assertEqual(visible[0].kind, "source")


class TestGainTableModel(omni.kit.test.AsyncTestCase):
    """Row / column model for the gain table (``gain_table_model``)."""

    _ROOT = "/World/Robot"

    @staticmethod
    def _entry(joint, drive_axis="angular", name="joint"):
        """A minimal joint-list entry stand-in (``.joint`` / ``.drive_axis``)."""
        return SimpleNamespace(joint=joint, drive_axis=drive_axis, display_name=name)

    @staticmethod
    def _newton_ctx() -> GainReadContext:
        """A read context whose active backend and solver select the Newton chain."""
        return GainReadContext(
            viewed_backend=gain_tuner.BACKEND_NEWTON,
            active_backend=gain_tuner.BACKEND_NEWTON,
            solver=gain_tuner.SOLVER_XPBD,
        )

    def _define_drive_joint(self, stage, path, *, is_revolute=True, kp=100.0, kd=10.0):
        """Define a revolute/prismatic joint with DriveAPI gains at ``path``."""
        if is_revolute:
            UsdPhysics.RevoluteJoint.Define(stage, path)
            axis = "angular"
        else:
            UsdPhysics.PrismaticJoint.Define(stage, path)
            axis = "linear"
        joint = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI.Apply(joint, axis)
        drive.CreateStiffnessAttr(kp)
        drive.CreateDampingAttr(kd)
        return joint, axis

    def _define_pid_actuator(self, stage, joint_path, act_path):
        """Author a Newton PID actuator targeting ``joint_path``."""
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        act = stage.DefinePrim(act_path, "Xform")
        act.AddAppliedSchema("NewtonPIDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        act.CreateAttribute("newton:ki", Sdf.ValueTypeNames.Float).Set(5.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(joint_path))

    async def test_is_angular_dof_by_joint_type(self) -> None:
        """Revolute DOFs are angular; prismatic DOFs are linear."""
        stage = Usd.Stage.CreateInMemory()
        rev, _ = self._define_drive_joint(stage, f"{self._ROOT}/rev", is_revolute=True)
        pris, _ = self._define_drive_joint(stage, f"{self._ROOT}/pris", is_revolute=False)
        self.assertTrue(is_angular_dof(rev))
        self.assertFalse(is_angular_dof(pris))

    async def test_ki_column_shown_only_with_pid_actuator(self) -> None:
        """The Ki column appears only when a checked joint has a PID actuator."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1, _ = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        self._define_pid_actuator(stage, f"{self._ROOT}/j1", f"{self._ROOT}/Actuators/act1")
        ctx = GainReadContext(actuator_map=build_actuator_gain_map(stage, self._ROOT))

        # Only the drive-only joint checked -> no Ki column.
        rows_drive_only = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        self.assertFalse(visible_gain_columns(rows_drive_only).show_ki)

        # The PID-actuator joint checked -> Ki column shown.
        rows_with_pid = build_gain_table_rows([self._entry(j1, name="j1")], ctx)
        self.assertTrue(visible_gain_columns(rows_with_pid).show_ki)

    async def test_advanced_column_auto_appear_and_blank_cell(self) -> None:
        """An advanced column appears when one checked joint authors it; others blank."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1, _ = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        # Author armature on j1 only.
        j1.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.4)
        ctx = GainReadContext()

        rows = build_gain_table_rows([self._entry(j0, name="j0"), self._entry(j1, name="j1")], ctx)
        cols = visible_gain_columns(rows)
        adv_keys = [c.key for c in cols.advanced]
        self.assertIn("armature", adv_keys)

        # j0 (no armature) renders a blank cell; j1 reads the authored value.
        value_j0, attr_j0 = rows[0].advanced_cell("armature")
        value_j1, attr_j1 = rows[1].advanced_cell("armature")
        self.assertIsNone(value_j0)
        self.assertIsNone(attr_j0)
        self.assertAlmostEqual(value_j1, 0.4)
        self.assertIsNotNone(attr_j1)

    async def test_newton_only_joint_populates_its_advanced_cell(self) -> None:
        """A joint the URDF converter gave only `newton:*` fills its cell under Newton."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        joint.ApplyAPI("NewtonJointAPI")
        joint.GetAttribute("newton:armature").Set(0.6)
        ctx = self._newton_ctx()

        rows = build_gain_table_rows([self._entry(joint, name="j0")], ctx)
        self.assertIn("armature", [c.key for c in visible_gain_columns(rows, ctx.active_backend, ctx.solver).advanced])
        value, attr = rows[0].advanced_cell("armature")
        self.assertAlmostEqual(value, 0.6, places=5)
        self.assertEqual(attr.GetName(), "newton:armature")

    async def test_writing_an_advanced_cell_authors_only_the_active_backend(self) -> None:
        """A table edit under PhysX leaves the Newton value the importer wrote alone."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        joint.ApplyAPI("NewtonJointAPI")
        joint.GetAttribute("newton:armature").Set(0.6)
        rows = build_gain_table_rows([self._entry(joint, name="j0")], GainReadContext())

        builder = UIBuilder.__new__(UIBuilder)
        builder._backend_ctx = SimpleNamespace(backend="PhysX", solver_type="")
        with mock.patch.object(BackendContext, "active_backend_label", return_value="PhysX"):
            builder._write_gain_cell(rows[0], "armature", 0.9)

        self.assertAlmostEqual(joint.GetAttribute("physxJoint:armature").Get(), 0.9, places=5)
        self.assertAlmostEqual(joint.GetAttribute("newton:armature").Get(), 0.6, places=5)

    async def test_writing_an_advanced_cell_under_newton_leaves_physx_alone(self) -> None:
        """The same edit under Newton is the mirror image."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        joint.ApplyAPI("PhysxJointAPI")
        joint.GetAttribute("physxJoint:armature").Set(0.6)
        ctx = self._newton_ctx()
        rows = build_gain_table_rows([self._entry(joint, name="j0")], ctx)

        builder = UIBuilder.__new__(UIBuilder)
        builder._backend_ctx = SimpleNamespace(backend=ctx.active_backend, solver_type=ctx.solver)
        # The write path reads the active backend from the live engine rather than
        # from the cached context, so the engine under test has to be stated here.
        with mock.patch.object(BackendContext, "active_backend_label", return_value=ctx.active_backend):
            builder._write_gain_cell(rows[0], "armature", 0.9)

        self.assertAlmostEqual(joint.GetAttribute("newton:armature").Get(), 0.9, places=5)
        self.assertAlmostEqual(joint.GetAttribute("physxJoint:armature").Get(), 0.6, places=5)

    async def test_detail_field_refresh_resolves_through_the_active_chain(self) -> None:
        """Refreshing a detail field must not read the inactive backend's half.

        Under Newton the PhysX half reads as its schema zero while unauthored and
        may hold a deliberately different value once authored, so a refresh that
        re-read the raw attributes would show a value the backend never resolves.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        joint.ApplyAPI("NewtonJointAPI")
        joint.ApplyAPI("PhysxJointAPI")
        joint.GetAttribute("newton:armature").Set(0.25)
        spec = gain_tuner.joint_param_spec("armature")

        read_value = UIBuilder._joint_param_reader(joint, spec, gain_tuner.BACKEND_NEWTON, gain_tuner.SOLVER_XPBD)
        # PhysX is applied but unauthored, so its raw attribute resolves to 0.0.
        self.assertAlmostEqual(joint.GetAttribute("physxJoint:armature").Get(), 0.0, places=5)
        self.assertAlmostEqual(read_value(), 0.25, places=5)

        joint.GetAttribute("physxJoint:armature").Set(0.9)
        self.assertAlmostEqual(read_value(), 0.25, places=5)

    async def test_row_marks_diverging_advanced_cells(self) -> None:
        """A row carries the per-backend notes so the table can mark those cells.

        The advanced cell shows what the active backend resolves, so without this
        the other backend's value -- what it will actually simulate -- would be
        invisible until the joint was opened.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        joint.ApplyAPI("NewtonJointAPI")
        joint.ApplyAPI("PhysxJointAPI")
        joint.GetAttribute("newton:armature").Set(0.25)
        joint.GetAttribute("physxJoint:armature").Set(0.75)

        rows = build_gain_table_rows([self._entry(joint, name="j0")], GainReadContext())

        info = rows[0].param_info("armature")
        self.assertIsNotNone(info)
        self.assertIn("newton:armature is 0.25", info)
        self.assertIn("PhysX uses physxJoint:armature (0.75)", info)
        # The cell shows the value PhysX resolves, not the Newton one.
        value, _attr = rows[0].advanced_cell("armature")
        self.assertAlmostEqual(value, 0.75, places=5)

    async def test_row_marks_nothing_when_the_backends_agree(self) -> None:
        """A joint with one consistent value carries no marks: the common case."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joint, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        armature = gain_tuner.joint_param_spec("armature")
        gain_tuner.author_joint_param(joint, armature, 0.25, gain_tuner.BACKEND_PHYSX)
        gain_tuner.author_joint_param(joint, armature, 0.25, gain_tuner.BACKEND_NEWTON)

        rows = build_gain_table_rows([self._entry(joint, name="j0")], GainReadContext())

        self.assertEqual(rows[0].param_infos, {})
        self.assertIsNone(rows[0].param_info("armature"))

    async def test_row_marks_survive_a_non_usd_joint(self) -> None:
        """A joint the scan cannot read reports no marks rather than raising."""
        self.assertEqual(read_param_infos(object(), "PhysX"), {})

    async def test_advanced_column_hidden_when_no_joint_has_it(self) -> None:
        """A plain drive joint shows no advanced columns.

        The PhysX-schema columns (armature / max joint velocity / joint friction)
        are hidden until a joint applies that schema attribute, and the DriveAPI
        ``max_force`` column is authored-only, so it is hidden until a joint
        authors an explicit ``maxForce`` value (its schema fallback is ``+inf``).
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        visible_keys = [c.key for c in visible_gain_columns(rows).advanced]
        self.assertEqual(visible_keys, [])

    async def test_max_force_column_authored_only(self) -> None:
        """The Max Force column appears only when a checked joint authors maxForce.

        Hidden (blank cell) for a drive joint carrying only the ``+inf`` schema
        fallback; shown once a joint authors an explicit value, with unauthored
        joints still rendering a blank cell.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1, axis1 = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        ctx = GainReadContext()

        # No joint authors maxForce -> column hidden, both cells blank.
        rows = build_gain_table_rows([self._entry(j0, name="j0"), self._entry(j1, name="j1")], ctx)
        self.assertNotIn("max_force", [c.key for c in visible_gain_columns(rows).advanced])
        self.assertEqual(rows[0].advanced_cell("max_force"), (None, None))

        # Author an explicit maxForce on j1 only -> column shown; j0 stays blank.
        UsdPhysics.DriveAPI(j1, axis1).GetMaxForceAttr().Set(500.0)
        rows = build_gain_table_rows([self._entry(j0, name="j0"), self._entry(j1, name="j1")], ctx)
        self.assertIn("max_force", [c.key for c in visible_gain_columns(rows).advanced])
        self.assertEqual(rows[0].advanced_cell("max_force"), (None, None))
        value_j1, attr_j1 = rows[1].advanced_cell("max_force")
        self.assertAlmostEqual(value_j1, 500.0)
        self.assertIsNotNone(attr_j1)

    async def test_mass_edit_writes_all_editable_skips_read_only(self) -> None:
        """A mass edit targets every editable checked row and skips read-only ones."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        j0, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1, _ = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        # j1 has an actuator -> its active source is the actuator, so forcing the
        # PhysicsDrive comparison view makes j1's drive gains read-only.
        self._define_pid_actuator(stage, f"{self._ROOT}/j1", f"{self._ROOT}/Actuators/act1")
        ctx = GainReadContext(actuator_map=build_actuator_gain_map(stage, self._ROOT))

        j1_path = j1.GetPath().pathString
        rows = build_gain_table_rows(
            [self._entry(j0, name="j0"), self._entry(j1, name="j1")],
            ctx,
            viewed_source_for={j1_path: GainSource.PHYSICS_DRIVE},
        )
        # j0 (drive-only, active) is editable; j1 (forced non-active drive view) is not.
        self.assertTrue(rows[0].editable)
        self.assertFalse(rows[1].editable)

        targets = mass_edit_row_indices(rows, GAIN_COLUMN_KP)
        self.assertEqual(targets, [0])
        # Damping behaves the same way (only the editable row is a target).
        self.assertEqual(mass_edit_row_indices(rows, GAIN_COLUMN_KD), [0])

    async def test_mass_edit_ki_only_targets_pid_rows(self) -> None:
        """A Ki mass edit targets only PID-actuator rows (blank Ki cells are skipped)."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        j0, _ = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1, _ = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        self._define_pid_actuator(stage, f"{self._ROOT}/j1", f"{self._ROOT}/Actuators/act1")
        ctx = GainReadContext(actuator_map=build_actuator_gain_map(stage, self._ROOT))

        rows = build_gain_table_rows([self._entry(j0, name="j0"), self._entry(j1, name="j1")], ctx)
        # Only j1 (PID actuator) has a Ki cell, so it is the sole Ki mass-edit target.
        self.assertEqual(mass_edit_row_indices(rows, GAIN_COLUMN_KI), [1])


class TestSelectionEditRowIndices(omni.kit.test.AsyncTestCase):
    """Selection-driven edit: an edit applies to the selected, editable rows only."""

    _ROOT = "/World/Robot"

    @staticmethod
    def _entry(joint, drive_axis="angular", name="joint"):
        return SimpleNamespace(joint=joint, drive_axis=drive_axis, display_name=name)

    def _define_drive_joint(self, stage, path):
        UsdPhysics.RevoluteJoint.Define(stage, path)
        joint = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        return joint

    def _rows(self):
        """Three drive-only (editable) joints in one table."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        joints = [self._define_drive_joint(stage, f"{self._ROOT}/j{i}") for i in range(3)]
        ctx = GainReadContext()
        entries = [self._entry(j, name=f"j{i}") for i, j in enumerate(joints)]
        return build_gain_table_rows(entries, ctx)

    async def test_edit_applies_to_selected_editable_rows(self) -> None:
        """A multi-selection edits every selected, editable row (intersection)."""
        rows = self._rows()
        # Selecting rows 0 and 2 targets exactly those two for a Kp edit.
        self.assertEqual(selection_edit_row_indices(rows, {0, 2}, GAIN_COLUMN_KP), [0, 2])

    async def test_single_selection_is_single_row_edit(self) -> None:
        """A single selected row collapses to a plain single-row edit."""
        rows = self._rows()
        self.assertEqual(selection_edit_row_indices(rows, {1}, GAIN_COLUMN_KP), [1])

    async def test_edit_skips_unselected_rows(self) -> None:
        """Rows not in the selection are never written, even if editable."""
        rows = self._rows()
        self.assertEqual(selection_edit_row_indices(rows, {0}, GAIN_COLUMN_KD), [0])

    async def test_edit_skips_read_only_rows_in_selection(self) -> None:
        """A selected but read-only row is dropped from the edit target set."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1 = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        # j1's active source is a Newton PID actuator, so forcing the PhysicsDrive
        # comparison view makes j1's drive gains read-only.
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        act = stage.DefinePrim(f"{self._ROOT}/Actuators/act1", "Xform")
        act.AddAppliedSchema("NewtonPIDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        act.CreateAttribute("newton:ki", Sdf.ValueTypeNames.Float).Set(5.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(f"{self._ROOT}/j1"))
        ctx = GainReadContext(actuator_map=build_actuator_gain_map(stage, self._ROOT))

        rows = build_gain_table_rows(
            [self._entry(j0, name="j0"), self._entry(j1, name="j1")],
            ctx,
            viewed_source_for={j1.GetPath().pathString: GainSource.PHYSICS_DRIVE},
        )
        # Both rows selected, but only the editable j0 is written.
        self.assertEqual(selection_edit_row_indices(rows, {0, 1}, GAIN_COLUMN_KP), [0])


class TestColumnPickerVisibility(omni.kit.test.AsyncTestCase):
    """Column catalog + hamburger picker: auto-select, manual override, grouping."""

    _ROOT = "/World/Robot"

    @staticmethod
    def _entry(joint, drive_axis="angular", name="joint"):
        return SimpleNamespace(joint=joint, drive_axis=drive_axis, display_name=name)

    def _define_drive_joint(self, stage, path):
        UsdPhysics.RevoluteJoint.Define(stage, path)
        joint = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        return joint

    def _define_pid_actuator(self, stage, joint_path, act_path):
        """Author a Newton PID actuator targeting ``joint_path`` (adds a Ki source)."""
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        act = stage.DefinePrim(act_path, "NewtonActuator")
        act.AddAppliedSchema("NewtonPIDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        act.CreateAttribute("newton:ki", Sdf.ValueTypeNames.Float).Set(5.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(joint_path))

    def _author_mjc(self, joint):
        """Author a full ``mjc:*`` gain set directly on a joint prim."""
        joint.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([200.0] + [0.0] * 9)
        joint.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -200.0, -20.0] + [0.0] * 7)
        joint.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.Token).Set("fixed")
        joint.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token).Set("affine")

    async def test_catalog_kp_kd_have_unit_others_do_not(self) -> None:
        """Kp/Kd carry the backend unit flag; Ki and advanced columns do not."""
        self.assertTrue(gain_column_spec(GAIN_COLUMN_KP).has_unit)
        self.assertTrue(gain_column_spec(GAIN_COLUMN_KD).has_unit)
        self.assertFalse(gain_column_spec(GAIN_COLUMN_KI).has_unit)
        # Every advanced column is grouped and carries no unit flag.
        for spec in GAIN_COLUMN_CATALOG:
            if spec.kind == "advanced":
                self.assertFalse(spec.has_unit)

    async def test_armature_categorized_as_mujoco_joint(self) -> None:
        """The armature advanced column is grouped under MuJoCo Joint."""
        self.assertEqual(gain_column_spec("armature").category, CATEGORY_MUJOCO_JOINT)
        self.assertEqual(gain_column_spec(GAIN_COLUMN_KP).category, CATEGORY_DRIVES)

    async def test_auto_visible_kp_kd_always_ki_advanced_conditional(self) -> None:
        """Kp/Kd auto-select always; Ki/advanced only when a shown joint offers them."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        auto = auto_visible_column_keys(rows)
        self.assertIn(GAIN_COLUMN_KP, auto)
        self.assertIn(GAIN_COLUMN_KD, auto)
        self.assertNotIn(GAIN_COLUMN_KI, auto)
        self.assertNotIn("armature", auto)

    async def test_auto_visible_includes_authored_advanced(self) -> None:
        """An authored advanced attribute (armature) is auto-selected."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j0.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.4)
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        self.assertIn("armature", auto_visible_column_keys(rows))

    async def test_resolve_visible_follows_auto_without_overrides(self) -> None:
        """With no manual overrides the visible columns follow the auto-selection."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        visible = [spec.key for spec in resolve_visible_columns(rows, None)]
        self.assertEqual(visible, [GAIN_COLUMN_SOURCE, GAIN_COLUMN_KP, GAIN_COLUMN_KD])

    async def test_manual_override_ignored_for_inapplicable_column(self) -> None:
        """A manual 'on' override for an inapplicable column is ignored (scoped).

        Overrides are scoped to the columns applicable to the loaded articulation:
        forcing a column no shown joint offers (here armature on a plain drive
        joint) must not surface an all-blank column.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        visible = [spec.key for spec in resolve_visible_columns(rows, {"armature": True})]
        self.assertNotIn("armature", visible)
        self.assertEqual(visible, [GAIN_COLUMN_SOURCE, GAIN_COLUMN_KP, GAIN_COLUMN_KD])

    async def test_manual_override_hides_auto_shown_column(self) -> None:
        """A manual 'off' override hides a column the schema would auto-show."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        visible = [spec.key for spec in resolve_visible_columns(rows, {GAIN_COLUMN_KD: False})]
        self.assertNotIn(GAIN_COLUMN_KD, visible)
        self.assertIn(GAIN_COLUMN_KP, visible)

    async def test_grouped_menu_reports_checked_and_auto_state(self) -> None:
        """The menu grouping reports each applicable column's (checked, auto) state."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        # Author armature so the MuJoCo-joint category is applicable and listed.
        j0.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.4)
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0")], ctx)
        # A manual 'off' override on the (applicable) armature column hides it.
        groups = dict(grouped_columns_for_menu(rows, {"armature": False}))
        self.assertIn(CATEGORY_DRIVES, groups)
        drive_state = {spec.key: (checked, auto) for spec, checked, auto in groups[CATEGORY_DRIVES]}
        self.assertEqual(drive_state[GAIN_COLUMN_KP], (True, True))  # auto + checked
        # Armature is applicable (listed) + auto, but manually toggled off.
        mjc_state = {spec.key: (checked, auto) for spec, checked, auto in groups[CATEGORY_MUJOCO_JOINT]}
        self.assertEqual(mjc_state["armature"], (False, True))

    async def test_menu_drive_only_has_no_mujoco_or_newton_entries(self) -> None:
        """A DriveAPI-only articulation lists only Drives Kp/Kd + a PhysX backend.

        No MuJoCo-joint category, no Ki (no PID actuator), no advanced columns, and
        the read-only BACKENDS row offers PhysX only (no MuJoCo).
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        j0 = self._define_drive_joint(stage, f"{self._ROOT}/j0")
        j1 = self._define_drive_joint(stage, f"{self._ROOT}/j1")
        ctx = GainReadContext()
        rows = build_gain_table_rows([self._entry(j0, name="j0"), self._entry(j1, name="j1")], ctx)

        groups = grouped_columns_for_menu(rows, None)
        categories = [category for category, _specs in groups]
        self.assertEqual(categories, [CATEGORY_DRIVES])
        drive_keys = [spec.key for spec, _c, _a in dict(groups)[CATEGORY_DRIVES]]
        self.assertEqual(drive_keys, [GAIN_COLUMN_SOURCE, GAIN_COLUMN_KP, GAIN_COLUMN_KD])
        self.assertNotIn(GAIN_COLUMN_KI, drive_keys)

        # BACKENDS row: no mjc gains -> PhysX only, no MuJoCo entry.
        self.assertFalse(menu_has_mjc(rows, ctx.mjc_map))
        backend_labels = [label for label, _checked in menu_backend_rows(False, mujoco_active=False)]
        self.assertEqual(backend_labels, [BACKEND_PHYSX])
        self.assertNotIn(BACKEND_MUJOCO, backend_labels)

    async def test_menu_mixed_articulation_has_mujoco_and_newton_entries(self) -> None:
        """A DriveAPI + mjc + Newton PD/PID articulation lists the full applicable set.

        Ki appears (PID actuator present), the MuJoCo-joint category appears
        (armature/joint friction authored), and the BACKENDS row offers both PhysX
        and MuJoCo (mjc:* gains present).
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        # PID-actuator joint (adds Ki), mjc joint (adds MuJoCo backend), and a joint
        # authoring advanced physxJoint attrs (adds the MuJoCo-joint category).
        j_pid = self._define_drive_joint(stage, f"{self._ROOT}/j_pid")
        self._define_pid_actuator(stage, f"{self._ROOT}/j_pid", f"{self._ROOT}/Actuators/act_pid")
        j_mjc = self._define_drive_joint(stage, f"{self._ROOT}/j_mjc")
        self._author_mjc(j_mjc)
        j_mjc.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.02)
        j_pid.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.05)
        ctx = GainReadContext(
            actuator_map=build_actuator_gain_map(stage, self._ROOT),
            mjc_map=build_mjc_gain_map(stage),
            active_backend="NewtonAPI",
        )
        rows = build_gain_table_rows([self._entry(j_pid, name="j_pid"), self._entry(j_mjc, name="j_mjc")], ctx)

        groups = dict(grouped_columns_for_menu(rows, None))
        self.assertIn(CATEGORY_DRIVES, groups)
        self.assertIn(CATEGORY_MUJOCO_JOINT, groups)
        drive_keys = [spec.key for spec, _c, _a in groups[CATEGORY_DRIVES]]
        self.assertIn(GAIN_COLUMN_KI, drive_keys)  # Ki only because a PID actuator is present
        mjc_keys = [spec.key for spec, _c, _a in groups[CATEGORY_MUJOCO_JOINT]]
        self.assertIn("armature", mjc_keys)
        self.assertIn("joint_friction", mjc_keys)

        # BACKENDS row: mjc gains present -> both PhysX and MuJoCo offered.
        self.assertTrue(menu_has_mjc(rows, ctx.mjc_map))
        backend_rows = menu_backend_rows(True, mujoco_active=True)
        backend_labels = [label for label, _checked in backend_rows]
        self.assertEqual(backend_labels, [BACKEND_PHYSX, BACKEND_MUJOCO])
        checked = dict(backend_rows)
        self.assertTrue(checked[BACKEND_MUJOCO])  # MuJoCo solver active -> checked
        self.assertFalse(checked[BACKEND_PHYSX])

    async def test_menu_backend_rows_marks_active_solver(self) -> None:
        """The BACKENDS row content + checked state is dynamic (mjc presence + solver)."""
        # No mjc -> PhysX only, always checked (sole relevant backend).
        self.assertEqual(menu_backend_rows(False, mujoco_active=False), [(BACKEND_PHYSX, True)])
        self.assertEqual(menu_backend_rows(False, mujoco_active=True), [(BACKEND_PHYSX, True)])
        # mjc present under PhysX -> both offered, PhysX checked.
        self.assertEqual(menu_backend_rows(True, mujoco_active=False), [(BACKEND_PHYSX, True), (BACKEND_MUJOCO, False)])
        # mjc present under the MuJoCo solver -> both offered, MuJoCo checked.
        self.assertEqual(menu_backend_rows(True, mujoco_active=True), [(BACKEND_PHYSX, False), (BACKEND_MUJOCO, True)])


class TestJointSourceAvailability(omni.kit.test.AsyncTestCase):
    """`_joint_source_availability` surfaces the MuJoCo-native source from the context."""

    _JOINT = "/World/Robot/joint"

    async def test_mjc_source_surfaced_from_context(self) -> None:
        """A joint with DriveAPI gains and an mjc_map entry reports (pd, act, mjc)=(T, F, T)."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)

        builder = UIBuilder.__new__(UIBuilder)
        builder._detail_gain_ctx = GainReadContext(mjc_map={self._JOINT: object()}, actuator_map={})
        entry = SimpleNamespace(joint=joint, drive_axis="angular")

        has_pd, has_act, has_mjc = builder._joint_source_availability(entry)
        self.assertTrue(has_pd)
        self.assertFalse(has_act)
        self.assertTrue(has_mjc)
        # And the badge reflects both sources rather than drive-only.
        self.assertEqual(builder._source_badge(has_pd, has_act, has_mjc)[0], "Drive + MuJoCo")

    async def test_no_mjc_when_absent_from_context(self) -> None:
        """Without an mjc_map entry the joint is not reported as having mjc gains."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, self._JOINT).GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr(100.0)

        builder = UIBuilder.__new__(UIBuilder)
        builder._detail_gain_ctx = GainReadContext(mjc_map={}, actuator_map={})
        entry = SimpleNamespace(joint=joint, drive_axis="angular")

        _has_pd, _has_act, has_mjc = builder._joint_source_availability(entry)
        self.assertFalse(has_mjc)
