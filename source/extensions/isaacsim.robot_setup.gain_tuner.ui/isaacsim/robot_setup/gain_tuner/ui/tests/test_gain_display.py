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

"""Unit tests for gain-table display helpers (units and advanced columns)."""

from __future__ import annotations

import math

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.ui.gain_display import (
    ADVANCED_GAIN_COLUMNS,
    DEFAULT_FIELD_SUFFIX,
    UNLIMITED_FIELD_FORMAT,
    UNLIMITED_TEXT,
    advanced_gain_column,
    authored_values_text,
    effective_value_text,
    fallback_source_text,
    format_param_value,
    gain_angle_unit,
    gain_dof_unit,
    joint_param_info_text,
    joint_param_infos,
    max_velocity_engine_text,
    read_advanced_gain_cell,
    read_advanced_param_display,
    unread_values_text,
    visible_advanced_gain_columns,
)
from pxr import Sdf, Usd, UsdPhysics

# From the package's public surface, not the submodule: these are exported through
# ``__all__`` precisely so the schema names are not spelled out per call site.
NEWTON_JOINT_API = gain_tuner.NEWTON_JOINT_API
PHYSX_JOINT_API = gain_tuner.PHYSX_JOINT_API

NEWTON = gain_tuner.BACKEND_NEWTON
PHYSX = gain_tuner.BACKEND_PHYSX
XPBD = gain_tuner.SOLVER_XPBD
MUJOCO = gain_tuner.SOLVER_MUJOCO

# Every ``(backend, solver)`` state the panel can be asked to describe, including
# the three unsupported ones.  Those are where the wording is least exercised and
# where the panel's false sentences lived: an engine the extension has no joint
# schema for still reaches every note, format and info line.
_EVERY_BACKEND_AND_SOLVER = (
    (NEWTON, XPBD),
    (NEWTON, MUJOCO),
    (NEWTON, ""),
    (PHYSX, ""),
    ("remotesim", ""),
    ("", ""),
    ("wobble", ""),
)


class TestGainFieldUnits(omni.kit.test.AsyncTestCase):
    """Backend/DOF-dependent stiffness-damping unit selection (table + detail view)."""

    async def test_angle_unit_by_backend(self) -> None:
        """Revolute gains read in degrees under PhysX and radians under Newton."""
        self.assertEqual(gain_angle_unit("PhysX"), "deg")
        self.assertEqual(gain_angle_unit("NewtonAPI"), "rad")
        # Unknown / empty backend labels fall back to the PhysX (degrees) default.
        self.assertEqual(gain_angle_unit(""), "deg")

    async def test_dof_unit_angular_vs_linear(self) -> None:
        """Angular DOFs use the backend angle unit; prismatic DOFs use linear units."""
        # Angular (revolute): degrees under PhysX, radians under Newton.
        self.assertEqual(gain_dof_unit(True, "PhysX"), "deg")
        self.assertEqual(gain_dof_unit(True, "NewtonAPI"), "rad")
        # Prismatic (linear): the stage linear unit, regardless of backend.
        self.assertEqual(gain_dof_unit(False, "PhysX"), "m")
        self.assertEqual(gain_dof_unit(False, "NewtonAPI"), "m")
        self.assertEqual(gain_dof_unit(False, "PhysX", linear_unit="cm"), "cm")


class TestAdvancedGainColumns(omni.kit.test.AsyncTestCase):
    """Dynamic advanced-parameter column visibility / cell reads (hybrid table)."""

    _ROOT = "/World/Robot"

    def _stage_with_joint(self, path: str, *, is_revolute: bool = True):
        """Build an in-memory stage with a single drive joint at ``path``."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim(self._ROOT, "Xform")
        if is_revolute:
            UsdPhysics.RevoluteJoint.Define(stage, path)
            axis = "angular"
        else:
            UsdPhysics.PrismaticJoint.Define(stage, path)
            axis = "linear"
        joint = stage.GetPrimAtPath(path)
        drive = UsdPhysics.DriveAPI.Apply(joint, axis)
        drive.CreateStiffnessAttr(100.0)
        drive.CreateDampingAttr(10.0)
        return stage, joint

    def _column(self, key: str):
        """Return the registered :class:`AdvancedGainColumn` with ``key``."""
        return next(c for c in ADVANCED_GAIN_COLUMNS if c.key == key)

    async def test_column_hidden_when_no_joint_has_param(self) -> None:
        """An advanced column is hidden when no selected joint authors it."""
        stage, joint = self._stage_with_joint(f"{self._ROOT}/j0")
        armature = self._column("armature")
        self.assertNotIn(armature, visible_advanced_gain_columns([(joint, None)], PHYSX))

    async def test_column_shown_when_one_joint_has_param(self) -> None:
        """The column auto-appears when at least one selected joint authors it."""
        stage, joint_a = self._stage_with_joint(f"{self._ROOT}/j0")
        _, joint_b = self._stage_with_joint(f"{self._ROOT}/j1")
        # Author armature on exactly one joint.
        joint_b.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.25)
        armature = self._column("armature")
        visible = visible_advanced_gain_columns([(joint_a, None), (joint_b, None)], PHYSX)
        self.assertIn(armature, visible)

    async def test_blank_cell_for_joint_lacking_param(self) -> None:
        """A joint without the attribute reads a blank (None) advanced cell."""
        stage, joint = self._stage_with_joint(f"{self._ROOT}/j0")
        armature = self._column("armature")
        value, attr = read_advanced_gain_cell(joint, None, armature, PHYSX)
        self.assertIsNone(value)
        self.assertIsNone(attr)

    async def test_authored_cell_reads_value_and_attr(self) -> None:
        """A joint that authors the attribute reads its value and a writable attr."""
        stage, joint = self._stage_with_joint(f"{self._ROOT}/j0")
        joint.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.5)
        armature = self._column("armature")
        value, attr = read_advanced_gain_cell(joint, None, armature, PHYSX)
        self.assertIsNotNone(attr)
        self.assertAlmostEqual(value, 0.5)
        # The returned attr is the live, writable attribute.
        attr.Set(0.75)
        self.assertAlmostEqual(joint.GetAttribute("physxJoint:armature").Get(), 0.75)

    async def test_drive_max_force_hidden_until_authored(self) -> None:
        """Max Force is authored-only: blank cell and hidden column until authored.

        The DriveAPI ``maxForce`` attribute is always present with a ``+inf``
        schema fallback, so the column is non-informative until a joint authors
        an explicit value.
        """
        stage, joint = self._stage_with_joint(f"{self._ROOT}/j0")
        max_force = self._column("max_force")
        self.assertTrue(max_force.authored_only)
        # Unauthored maxForce (schema fallback +inf only) -> blank, uneditable.
        value, attr = read_advanced_gain_cell(joint, "angular", max_force, PHYSX)
        self.assertIsNone(value)
        self.assertIsNone(attr)
        # And the column is hidden while no checked joint authors maxForce.
        self.assertNotIn(max_force, visible_advanced_gain_columns([(joint, "angular")], PHYSX))

    async def test_drive_max_force_shown_when_authored(self) -> None:
        """An explicitly authored maxForce surfaces the column and an editable cell."""
        stage, joint_a = self._stage_with_joint(f"{self._ROOT}/j0")
        stage_b, joint_b = self._stage_with_joint(f"{self._ROOT}/j1")
        # Author an explicit maxForce on j_b only.
        UsdPhysics.DriveAPI(joint_b, "angular").GetMaxForceAttr().Set(500.0)
        max_force = self._column("max_force")

        # j_b reads the authored value with a writable attribute.
        value_b, attr_b = read_advanced_gain_cell(joint_b, "angular", max_force, PHYSX)
        self.assertAlmostEqual(value_b, 500.0)
        self.assertIsNotNone(attr_b)
        attr_b.Set(750.0)
        self.assertAlmostEqual(attr_b.Get(), 750.0)
        # j_a (unauthored) still renders blank.
        value_a, attr_a = read_advanced_gain_cell(joint_a, "angular", max_force, PHYSX)
        self.assertIsNone(value_a)
        self.assertIsNone(attr_a)
        # The column is shown because at least one joint authors maxForce.
        self.assertIn(max_force, visible_advanced_gain_columns([(joint_a, "angular"), (joint_b, "angular")], PHYSX))


class TestMultiSchemaAdvancedColumns(omni.kit.test.AsyncTestCase):
    """Advanced columns resolved through the active backend's own schema chain."""

    _ROOT = "/World/Robot"

    def _joint(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, f"{self._ROOT}/j0")
        joint = stage.GetPrimAtPath(f"{self._ROOT}/j0")
        UsdPhysics.DriveAPI.Apply(joint, "angular").CreateStiffnessAttr(100.0)
        # Held so the stage outlives the caller's use of the returned prim.
        self._stage = stage
        return joint

    def _column(self, key: str):
        return advanced_gain_column(key)

    async def test_multi_schema_columns_carry_a_param_spec(self) -> None:
        """The three importer-authored params resolve through a schema chain."""
        for key, newton_attr in (
            ("armature", "newton:armature"),
            ("joint_friction", "newton:friction"),
            ("max_joint_velocity", "newton:velocityLimit"),
        ):
            self.assertEqual(self._column(key).param_spec.newton_attr, newton_attr)
        # Max Force stays a DriveAPI-only column.
        self.assertIsNone(self._column("max_force").param_spec)
        self.assertIsNone(advanced_gain_column("not_a_column"))

    async def test_newton_only_joint_surfaces_the_column_under_newton(self) -> None:
        """A joint the URDF converter gave only a Newton opinion is not invisible."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.GetAttribute("newton:friction").Set(0.3)
        self.assertFalse(joint.HasAPI(PHYSX_JOINT_API))

        friction = self._column("joint_friction")
        value, attr = read_advanced_gain_cell(joint, None, friction, NEWTON, XPBD)
        self.assertAlmostEqual(value, 0.3, places=5)
        self.assertEqual(attr.GetName(), "newton:friction")
        self.assertIn(friction, visible_advanced_gain_columns([(joint, None)], NEWTON, XPBD))

    async def test_the_same_joint_reads_zero_under_physx(self) -> None:
        """PhysX has no opinion here, and the cell must not borrow Newton's."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.GetAttribute("newton:friction").Set(0.3)

        value, attr = read_advanced_gain_cell(joint, None, self._column("joint_friction"), PHYSX)
        self.assertEqual(value, 0.0)
        self.assertIsNotNone(attr)

    async def test_each_backend_reads_its_own_authored_value(self) -> None:
        """Two independently tuned values, each shown to the backend that uses it."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        joint.GetAttribute("newton:armature").Set(0.1)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        armature = self._column("armature")

        newton_value, _attr = read_advanced_gain_cell(joint, None, armature, NEWTON, XPBD)
        physx_value, _attr = read_advanced_gain_cell(joint, None, armature, PHYSX)
        self.assertAlmostEqual(newton_value, 0.1, places=5)
        self.assertAlmostEqual(physx_value, 0.9, places=5)

    async def test_mujoco_solver_changes_the_cell_value(self) -> None:
        """mjc:* sits in the MuJoCo chain, so the same stage reads differently."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        joint.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Float).Set(0.4)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        armature = self._column("armature")

        mujoco_value, _attr = read_advanced_gain_cell(joint, None, armature, NEWTON, MUJOCO)
        xpbd_value, _attr = read_advanced_gain_cell(joint, None, armature, NEWTON, XPBD)
        self.assertAlmostEqual(mujoco_value, 0.4, places=5)
        self.assertAlmostEqual(xpbd_value, 0.9, places=5)

    async def test_unknown_solver_still_reads_the_newton_value(self) -> None:
        """newton:* leads under every solver, so this answer needs no solver."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.GetAttribute("newton:armature").Set(0.1)

        value, _attr = read_advanced_gain_cell(joint, None, self._column("armature"), NEWTON, "")
        self.assertAlmostEqual(value, 0.1, places=5)

    async def test_unknown_solver_blanks_a_value_only_the_solver_decides(self) -> None:
        """With mjc:* and physxJoint:* both in play, the cell may not pick one."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        joint.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Float).Set(0.4)
        joint.GetAttribute("physxJoint:armature").Set(0.9)

        value, _attr = read_advanced_gain_cell(joint, None, self._column("armature"), NEWTON, "")
        self.assertIsNone(value)

    async def test_unlimited_velocity_column_stays_hidden(self) -> None:
        """An applied schema whose velocity limit is unauthored adds no column."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        velocity = self._column("max_joint_velocity")
        self.assertNotIn(velocity, visible_advanced_gain_columns([(joint, None)], NEWTON, XPBD))

        joint.GetAttribute("newton:velocityLimit").Set(20.0)
        self.assertIn(velocity, visible_advanced_gain_columns([(joint, None)], NEWTON, XPBD))

    async def test_unlimited_velocity_cell_is_blank_but_writable(self) -> None:
        """A velocity cell with no number keeps its attribute so one can be set.

        The column is hidden until some joint has a finite limit, so dropping the
        attribute too would leave no way to author the first one.  Zero is not an
        option: it would read as a joint held at a standstill.
        """
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        value, attr = read_advanced_gain_cell(joint, None, self._column("max_joint_velocity"), NEWTON, XPBD)
        self.assertIsNone(value)
        self.assertIsNotNone(attr)

    async def test_authored_infinite_limit_reads_blank_not_zero(self) -> None:
        """An authored "unlimited" wins the chain and is shown as no limit."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.GetAttribute("newton:velocityLimit").Set(float("inf"))
        value, attr = read_advanced_gain_cell(joint, None, self._column("max_joint_velocity"), NEWTON, XPBD)
        self.assertIsNone(value)
        self.assertIsNotNone(attr)


class TestAdvancedParamDisplay(omni.kit.test.AsyncTestCase):
    """What an advanced cell renders for a parameter nothing is authored for.

    Zero is a value no engine uses: Newton simulates its ``ModelBuilder`` armature
    default, and neither backend clamps an unauthored velocity limit at all.  These
    cover the three states a cell can be in -- authored, engine default, and nothing
    stateable -- and that the write path survives all of them.
    """

    _ROOT = "/World/Robot"

    def _joint(self, *apis):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, f"{self._ROOT}/j0")
        joint = stage.GetPrimAtPath(f"{self._ROOT}/j0")
        for api in apis:
            joint.ApplyAPI(api)
        self._stage = stage
        return joint

    def _display(self, joint, key, backend=PHYSX, solver="", engine_value=None):
        return read_advanced_param_display(joint, advanced_gain_column(key), backend, solver, engine_value)

    async def test_authored_value_is_shown_as_authored(self) -> None:
        """Nothing is marked when the stage actually says something."""
        joint = self._joint(PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.6)

        display = self._display(joint, "armature")

        self.assertTrue(display.authored)
        self.assertAlmostEqual(display.value, 0.6, places=5)
        self.assertEqual(display.note, "")
        self.assertEqual(display.field_format, "")

    async def test_unauthored_newton_armature_is_newtons_default(self) -> None:
        """0.1, marked as a default, and still editable.

        The cell says only ``"(default)"``; the *engine* is named in the note, which
        the table carries as the cell's tooltip.  See
        :data:`~isaacsim.robot_setup.gain_tuner.ui.gain_display.DEFAULT_FIELD_SUFFIX`
        for why the backend does not travel in the format string.
        """
        joint = self._joint(NEWTON_JOINT_API)

        display = self._display(joint, "armature", NEWTON, XPBD)

        self.assertFalse(display.authored)
        self.assertAlmostEqual(display.value, gain_tuner.NEWTON_DEFAULT_ARMATURE, places=6)
        self.assertAlmostEqual(display.field_value, gain_tuner.NEWTON_DEFAULT_ARMATURE, places=6)
        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), display.field_format)
        self.assertIn("Newton", display.note)
        self.assertTrue(display.editable)
        self.assertFalse(display.blank)

    async def test_unauthored_physx_armature_is_zero_but_still_marked(self) -> None:
        """PhysX's default really is zero; that does not make it an authored zero."""
        joint = self._joint(PHYSX_JOINT_API)

        display = self._display(joint, "armature")

        self.assertFalse(display.authored)
        self.assertEqual(display.value, 0.0)
        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), display.field_format)
        self.assertIn("PhysX", display.note)

    async def test_unauthored_friction_is_zero_under_both_backends(self) -> None:
        """Friction is the one parameter whose builder default genuinely is zero."""
        joint = self._joint(NEWTON_JOINT_API, PHYSX_JOINT_API)

        for backend, solver in ((NEWTON, XPBD), (PHYSX, "")):
            display = self._display(joint, "joint_friction", backend, solver)
            self.assertEqual(display.value, 0.0, msg=backend)
            self.assertFalse(display.authored, msg=backend)

    async def test_unauthored_velocity_limit_is_unlimited_not_zero(self) -> None:
        """The field must read as unlimited, never as a joint held at a standstill."""
        joint = self._joint(PHYSX_JOINT_API)

        display = self._display(joint, "max_joint_velocity")

        self.assertTrue(display.unlimited)
        self.assertIsNone(display.value)
        # Unauthored, so it also says whose opinion it is (see
        # :class:`TestDefaultIsLegibleInTheCell`).
        self.assertIn(UNLIMITED_FIELD_FORMAT, display.field_format)
        # Zero only ever reaches the widget behind that format, never as a digit.
        self.assertEqual(display.field_value, 0.0)
        self.assertTrue(display.editable)
        self.assertFalse(display.blank)

    async def test_authored_infinite_limit_is_unlimited_and_authored(self) -> None:
        """Authoring `inf` says the same thing the default does, deliberately."""
        joint = self._joint(NEWTON_JOINT_API)
        joint.GetAttribute("newton:velocityLimit").Set(math.inf)

        display = self._display(joint, "max_joint_velocity", NEWTON, XPBD)

        self.assertTrue(display.unlimited)
        self.assertTrue(display.authored)
        self.assertTrue(display.editable)

    async def test_a_measured_engine_value_beats_the_documented_default(self) -> None:
        """Where the engine can be asked, its answer is the honest placeholder."""
        joint = self._joint(NEWTON_JOINT_API)

        display = self._display(joint, "armature", NEWTON, XPBD, engine_value=0.25)

        self.assertAlmostEqual(display.value, 0.25, places=6)
        self.assertIn("simulating", display.note)

    async def test_a_measured_unlimited_engine_value_reads_as_unlimited(self) -> None:
        """`inf` from the engine is not a number to show either."""
        joint = self._joint(PHYSX_JOINT_API)

        display = self._display(joint, "max_joint_velocity", engine_value=math.inf)

        self.assertTrue(display.unlimited)
        self.assertIsNone(display.value)

    async def test_an_undetermined_chain_states_nothing(self) -> None:
        """With the solver unknown, neither the stage's value nor a default applies."""
        joint = self._joint(NEWTON_JOINT_API, PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        joint.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Float).Set(0.4)

        display = self._display(joint, "armature", NEWTON, "")

        self.assertTrue(display.blank)
        self.assertIsNone(display.value)
        self.assertFalse(display.unlimited)

    async def test_a_parameter_the_joint_does_not_have_is_blank(self) -> None:
        """No schema applied means no such parameter, so no cell and no edit."""
        joint = self._joint()

        display = self._display(joint, "armature")

        self.assertTrue(display.blank)
        self.assertFalse(display.editable)

    async def test_an_unsupported_backend_disables_the_cell_and_says_why(self) -> None:
        """Fail closed: no read, no edit, and an explanation rather than a guess.

        And *only* that explanation.  The note says the value cannot be read, so any
        sentence the panel adds beside it contradicts it: the info line used to read
        "physxJoint:armature (0.9) is not read by PhysX" under `remotesim` -- naming
        an engine that is not running, about a value it had just said it could not
        read.  Every one of those sentences describes a resolver chain, and an
        engine with no known joint schema has none.
        """
        joint = self._joint(NEWTON_JOINT_API, PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        spec = gain_tuner.joint_param_spec("armature")

        for backend in ("remotesim", "", "wobble"):
            display = self._display(joint, "armature", backend)
            resolution = gain_tuner.resolve_joint_param(joint, spec, backend, "")
            self.assertFalse(display.editable, msg=backend)
            self.assertTrue(display.blank, msg=backend)
            self.assertIn("not one this extension can author", display.note, msg=backend)
            self.assertTrue(display.note.isascii(), msg=backend)
            for text in (
                joint_param_info_text(resolution),
                joint_param_info_text(resolution, display),
                effective_value_text(resolution),
                unread_values_text(resolution),
            ):
                self.assertEqual(text, "", msg=f"{backend}: {text!r}")

    async def test_the_max_force_column_has_no_backend_display(self) -> None:
        """A plain single attribute has no per-backend resolution to describe."""
        joint = self._joint(PHYSX_JOINT_API)
        self.assertIsNone(self._display(joint, "max_force"))

    async def test_every_note_is_ascii(self) -> None:
        """omni.ui renders non-ASCII as "?", which this branch has been bitten by.

        The unsupported backends are in the matrix because that is where the wording
        is least exercised and where the false sentence lived -- an engine label the
        panel has no name for reaches the note like any other string.
        """
        joint = self._joint(NEWTON_JOINT_API, PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        spec_for = gain_tuner.joint_param_spec
        for key in ("armature", "joint_friction", "max_joint_velocity"):
            for backend, solver in _EVERY_BACKEND_AND_SOLVER:
                display = self._display(joint, key, backend, solver)
                resolution = gain_tuner.resolve_joint_param(joint, spec_for(key), backend, solver)
                where = f"{key}/{backend}/{solver}"
                self.assertTrue(display.note.isascii(), msg=f"{where}: {display.note!r}")
                self.assertTrue(display.field_format.isascii(), msg=f"{where}: {display.field_format!r}")
                info = joint_param_info_text(resolution, display)
                self.assertTrue(info.isascii(), msg=f"{where}: {info!r}")


class TestDefaultIsLegibleInTheCell(omni.kit.test.AsyncTestCase):
    """A placeholder must read as a default in the cell, not only next to one.

    Muting the text is the only other signal the table can carry, and grey
    ``#9E9E9E`` against an authored ``#D5D5D5`` is not a difference a user can read
    without the other one beside it: an unauthored ``0.0`` looked exactly like an
    authored ``0.0``.  The word travels in the field's own ``format``, which is the
    only thing the cell has room for.
    """

    _ROOT = "/World/Robot"

    def _joint(self, *apis):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, f"{self._ROOT}/j0")
        joint = stage.GetPrimAtPath(f"{self._ROOT}/j0")
        for api in apis:
            joint.ApplyAPI(api)
        self._stage = stage
        return joint

    def _display(self, joint, key, backend=PHYSX, solver="", engine_value=None):
        return read_advanced_param_display(joint, advanced_gain_column(key), backend, solver, engine_value)

    async def test_an_unauthored_number_says_default_in_its_own_field(self) -> None:
        """The case the muting alone could not carry: PhysX's default 0.0."""
        display = self._display(self._joint(PHYSX_JOINT_API), "armature")

        self.assertFalse(display.authored)
        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), display.field_format)
        # A conversion specifier is still there, so the number is still rendered.
        self.assertIn("%", display.field_format)

    async def test_a_measured_engine_value_is_still_marked_as_not_the_users(self) -> None:
        """Measured beats assumed, but it is still not something the user authored."""
        display = self._display(self._joint(NEWTON_JOINT_API), "armature", NEWTON, XPBD, engine_value=0.42)

        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), display.field_format)

    async def test_an_authored_number_is_left_alone(self) -> None:
        """The suffix is a claim about provenance; an authored value gets none."""
        joint = self._joint(PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.6)

        self.assertEqual(self._display(joint, "armature").field_format, "")

    async def test_an_unauthored_unlimited_is_marked_too(self) -> None:
        """The precedent this extends: unlimited was muted, now it also says default."""
        display = self._display(self._joint(PHYSX_JOINT_API), "max_joint_velocity")

        self.assertTrue(display.unlimited)
        self.assertFalse(display.authored)
        self.assertIn(UNLIMITED_TEXT, display.field_format)
        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), display.field_format)

    async def test_an_authored_unlimited_keeps_the_bare_word(self) -> None:
        """Authoring `inf` is a decision, so it reads as one."""
        joint = self._joint(NEWTON_JOINT_API)
        joint.GetAttribute("newton:velocityLimit").Set(math.inf)

        self.assertEqual(self._display(joint, "max_joint_velocity", NEWTON, XPBD).field_format, UNLIMITED_FIELD_FORMAT)

    async def test_every_format_is_ascii(self) -> None:
        """A format string reaches the font like any other text."""
        joint = self._joint(NEWTON_JOINT_API, PHYSX_JOINT_API)
        for key in ("armature", "joint_friction", "max_joint_velocity"):
            for backend, solver in _EVERY_BACKEND_AND_SOLVER:
                fmt = self._display(joint, key, backend, solver).field_format
                self.assertTrue(fmt.isascii(), msg=f"{key}/{backend}: {fmt!r}")


class TestPerBackendWording(omni.kit.test.AsyncTestCase):
    """The informational sentences describing per-backend joint parameter values."""

    def _resolution(self, backend=PHYSX, solver="", key="joint_friction", newton=0.25, physx=0.75, mjc=None):
        self._stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(self._stage, "/Robot/joint").GetPrim()
        spec = gain_tuner.joint_param_spec(key)
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        if newton is not None:
            joint.GetAttribute(spec.newton_attr).Set(newton)
        if physx is not None:
            joint.GetAttribute(spec.physx_attr).Set(physx)
        if mjc is not None:
            joint.CreateAttribute(spec.mjc_attr, Sdf.ValueTypeNames.Float).Set(mjc)
        return gain_tuner.resolve_joint_param(joint, spec, backend, solver)

    async def test_backend_label_detection(self) -> None:
        """Only the Newton backend label counts as Newton.

        The engine *name* is no longer decided in this module.  A local "Newton if
        Newton else PhysX" helper answered "PhysX" for `remotesim` and for a failed
        engine query, so every sentence built from it named an engine with no
        evidence; the names now come from
        :attr:`JointParamResolution.backend_label`, which is core's
        :func:`~isaacsim.robot_setup.gain_tuner.backend_display_label`.  The
        detection half went with it: it duplicated core's
        :func:`~isaacsim.robot_setup.gain_tuner.newton_backend_selected` exactly,
        and this test was its only remaining caller.
        """
        self.assertTrue(gain_tuner.newton_backend_selected(NEWTON))
        self.assertFalse(gain_tuner.newton_backend_selected(PHYSX))
        self.assertFalse(gain_tuner.newton_backend_selected(""))
        self.assertEqual(self._resolution(NEWTON, XPBD).backend_label, "Newton")
        self.assertEqual(self._resolution(PHYSX).backend_label, "PhysX")
        for backend in ("remotesim", "", "wobble"):
            self.assertNotIn("PhysX", self._resolution(backend).backend_label, msg=backend)

    async def test_values_format_compactly(self) -> None:
        """Values are trimmed, and a missing value is named rather than crashing."""
        self.assertEqual(format_param_value(0.25), "0.25")
        self.assertEqual(format_param_value(30.0), "30")
        self.assertEqual(format_param_value(None), "none")

    async def test_authored_values_lists_each_schema(self) -> None:
        """Every authored opinion is named, including the MuJoCo one."""
        text = authored_values_text(self._resolution(NEWTON, MUJOCO, mjc=0.5))
        self.assertIn("newton:friction is 0.25", text)
        self.assertIn("mjc:frictionloss is 0.5", text)
        self.assertIn("physxJoint:jointFriction is 0.75", text)

    async def test_effective_value_names_physx_under_physx(self) -> None:
        """Under PhysX the PhysX half is what the simulation obeys."""
        self.assertIn("PhysX uses physxJoint:jointFriction (0.75)", effective_value_text(self._resolution(PHYSX)))

    async def test_effective_value_names_newton_under_newton(self) -> None:
        """The winner is the inverse under Newton; the wording must not hardcode PhysX."""
        text = effective_value_text(self._resolution(NEWTON, XPBD))
        self.assertIn("Newton uses newton:friction (0.25)", text)
        self.assertNotIn("PhysX uses", text)

    async def test_effective_value_says_so_when_nothing_is_authored(self) -> None:
        """The engine default applies, and the sentence says that rather than "0"."""
        text = effective_value_text(self._resolution(NEWTON, XPBD, newton=None, physx=None))
        self.assertIn("Nothing is authored", text)
        self.assertIn("its own default applies", text)

    async def test_effective_value_refuses_to_guess_an_unknown_chain(self) -> None:
        """An unknown solver must be stated, not silently resolved as XPBD."""
        text = effective_value_text(self._resolution(NEWTON, "", key="armature", newton=None, physx=0.9, mjc=0.4))
        self.assertIn("could not be determined", text)
        self.assertIn("mjc:*", text)

    async def test_unknown_solver_does_not_block_a_newton_authored_value(self) -> None:
        """newton:* is first under every solver, so this one is not in doubt."""
        text = effective_value_text(self._resolution(NEWTON, ""))
        self.assertIn("Newton uses newton:friction (0.25)", text)

    async def test_effective_value_reports_an_unlimited_limit_as_unlimited(self) -> None:
        """A resolved "no limit" is not a number, and must not print as one."""
        resolution = self._resolution(NEWTON, XPBD, key="max_joint_velocity", newton=float("inf"), physx=107.0)
        self.assertIn("unlimited", effective_value_text(resolution))

    async def test_unread_values_names_a_physx_only_friction_under_newton(self) -> None:
        """The asymmetry surfaced in words: authored, simulated by PhysX, ignored here."""
        text = unread_values_text(self._resolution(NEWTON, XPBD, newton=None))
        self.assertIn("physxJoint:jointFriction (0.75)", text)
        self.assertIn("not read by Newton", text)

    async def test_unread_values_is_empty_when_the_chain_reads_everything(self) -> None:
        """Armature has a PhysX fallback, so nothing authored is left out."""
        self.assertEqual(unread_values_text(self._resolution(NEWTON, XPBD, key="armature")), "")

    async def test_info_text_states_both_values_and_the_one_in_effect(self) -> None:
        """The whole point: describe the divergence, never ask for a change."""
        text = joint_param_info_text(self._resolution(PHYSX))
        self.assertIn("newton:friction is 0.25", text)
        self.assertIn("physxJoint:jointFriction is 0.75", text)
        self.assertIn("PhysX uses physxJoint:jointFriction (0.75)", text)

    async def test_info_text_never_asks_the_user_to_fix_anything(self) -> None:
        """Divergence is a legitimate authoring choice, so the voice stays neutral."""
        for backend, solver in ((PHYSX, ""), (NEWTON, XPBD), (NEWTON, MUJOCO), (NEWTON, "")):
            text = joint_param_info_text(self._resolution(backend, solver)).lower()
            for word in ("sync", "mismatch", "should", "must", "fix", "warning", "conflict", "out of step"):
                self.assertNotIn(word, text, msg=f"{backend}/{solver}: {text!r}")

    async def test_info_text_is_empty_when_there_is_nothing_to_report(self) -> None:
        """Agreeing backends, a fully read chain, and a known solver say nothing."""
        self.assertEqual(joint_param_info_text(self._resolution(PHYSX, "", key="armature", newton=0.4, physx=0.4)), "")

    async def test_info_text_reports_an_unknown_chain(self) -> None:
        """Where the effective value cannot be determined, the UI must say so."""
        resolution = self._resolution(NEWTON, "", key="armature", newton=None, physx=0.4, mjc=0.4)
        self.assertIn("could not be determined", joint_param_info_text(resolution))

    async def test_info_text_stays_quiet_when_the_solver_cannot_change_the_answer(self) -> None:
        """An unknown solver is only worth reporting when it decides something."""
        resolution = self._resolution(NEWTON, "", key="armature", newton=0.4, physx=0.4)
        self.assertEqual(joint_param_info_text(resolution), "")

    async def test_an_ignored_value_that_matches_is_not_reported(self) -> None:
        """Newton not reading a PhysX friction of the same number changes nothing."""
        self.assertEqual(joint_param_info_text(self._resolution(NEWTON, XPBD, newton=0.4, physx=0.4)), "")

    async def test_info_text_is_renderable(self) -> None:
        """The app font has no glyph for non-ASCII text beyond the degree sign."""
        for backend, solver in ((PHYSX, ""), (NEWTON, XPBD), (NEWTON, MUJOCO), (NEWTON, "")):
            text = joint_param_info_text(self._resolution(backend, solver))
            self.assertTrue(text.isascii(), f"non-ASCII info text: {text!r}")

    async def test_infos_map_omits_parameters_with_nothing_to_say(self) -> None:
        """Only the parameters worth pointing out mark their cell."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint").GetPrim()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        armature = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(armature.newton_attr).Set(0.1)
        joint.GetAttribute(armature.physx_attr).Set(0.2)

        infos = joint_param_infos(gain_tuner.resolve_joint_params(joint, PHYSX))
        self.assertEqual(set(infos), {"armature"})
        self.assertEqual(joint_param_infos([]), {})

    async def test_engine_velocity_text_names_the_enforced_limit(self) -> None:
        """The panel says which limit the velocity sweep will actually use."""
        text = max_velocity_engine_text(180.0, 90.0, True)
        self.assertIn("enforces 90", text)
        self.assertIn("180", text)
        self.assertIn("\u00b0/s", text)

    async def test_engine_velocity_text_uses_linear_units_for_prismatic(self) -> None:
        """Prismatic DOFs are not measured in degrees."""
        text = max_velocity_engine_text(2.0, 1.0, False)
        self.assertIn("m/s", text)
        self.assertTrue(text.isascii(), f"non-ASCII velocity text: {text!r}")

    async def test_engine_velocity_text_only_uses_the_degree_glyph(self) -> None:
        """The degree sign is the sole non-ASCII character the font renders."""
        text = max_velocity_engine_text(180.0, 90.0, True)
        self.assertTrue(text.replace("\u00b0", "").isascii(), f"non-ASCII velocity text: {text!r}")


class TestSilentFallbackIsAnnounced(omni.kit.test.AsyncTestCase):
    """The case that used to be the quietest one in the panel.

    A joint that authors only the *other* backend's half is still simulated from
    it, because each backend resolves through a chain: Newton reads
    ``physxJoint:armature`` when ``newton:armature`` is unauthored.  The number
    looked exactly like an authored one, and the first edit moved it onto
    ``newton:armature``, which then won the chain -- so both facts are stated.
    Friction, which has no such fallback, was the one getting two full lines.
    """

    def _resolution(self, backend, solver, key, newton=None, physx=None, mjc=None):
        self._stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(self._stage, "/Robot/joint").GetPrim()
        spec = gain_tuner.joint_param_spec(key)
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        if newton is not None:
            joint.GetAttribute(spec.newton_attr).Set(newton)
        if physx is not None:
            joint.GetAttribute(spec.physx_attr).Set(physx)
        if mjc is not None:
            joint.CreateAttribute(spec.mjc_attr, Sdf.ValueTypeNames.Float).Set(mjc)
        self._joint_prim = joint
        return gain_tuner.resolve_joint_param(joint, spec, backend, solver)

    def _display(self, key, backend, solver):
        """The display for the joint the last :meth:`_resolution` call built."""
        return read_advanced_param_display(self._joint_prim, advanced_gain_column(key), backend, solver)

    async def test_newton_reading_a_physx_armature_says_so(self) -> None:
        """Names the schema that is unauthored, the one being read, and the value."""
        resolution = self._resolution(NEWTON, XPBD, "armature", physx=0.6)

        self.assertTrue(resolution.resolved_from_fallback_schema)
        text = fallback_source_text(resolution)
        self.assertIn("newton:armature", text)
        self.assertIn("physxJoint:armature", text)
        self.assertIn("0.6", text)
        self.assertTrue(text.isascii(), f"non-ASCII fallback text: {text!r}")

    async def test_it_says_where_an_edit_would_land(self) -> None:
        """The value moves on the first edit, which is the surprising half."""
        text = fallback_source_text(self._resolution(NEWTON, XPBD, "armature", physx=0.6))
        self.assertIn("An edit here authors newton:armature", text)

    async def test_a_fallback_velocity_limit_says_so_as_well(self) -> None:
        """Armature is not the only parameter with a cross-backend fallback."""
        text = fallback_source_text(self._resolution(NEWTON, XPBD, "max_joint_velocity", physx=270.0))
        self.assertIn("newton:velocityLimit", text)
        self.assertIn("physxJoint:maxJointVelocity", text)

    async def test_the_info_line_carries_it(self) -> None:
        """The annotation reaches the panel through the line under the field."""
        resolution = self._resolution(NEWTON, XPBD, "armature", physx=0.6)
        self.assertEqual(joint_param_info_text(resolution), fallback_source_text(resolution))

    async def test_a_value_read_from_the_backends_own_schema_is_not_a_fallback(self) -> None:
        """The common case, which must stay silent under both backends."""
        for backend, solver in ((NEWTON, XPBD), (PHYSX, "")):
            resolution = self._resolution(backend, solver, "armature", newton=0.6, physx=0.6)
            self.assertFalse(resolution.resolved_from_fallback_schema, msg=backend)
            self.assertEqual(fallback_source_text(resolution), "", msg=backend)

    async def test_physx_never_falls_back_to_newton(self) -> None:
        """PhysX reads its own schema only, so a Newton-only value is unread, not read."""
        resolution = self._resolution(PHYSX, "", "armature", newton=0.55)
        self.assertFalse(resolution.resolved_from_fallback_schema)
        self.assertIn("not read by PhysX", joint_param_info_text(resolution))

    async def test_friction_has_no_fallback_and_stays_short(self) -> None:
        """The line that used to be two sentences of "nothing is authored".

        Newton reads no PhysX friction at all, so there is no fallback to report --
        only the engine default the field already states, and the authored value
        Newton ignores.  Both openings said "nothing is authored", over a single
        field, and the second said nothing the first had not.

        The property asserted is that non-redundancy, not a character count: a
        budget stands in for legibility only until somebody rewords a sentence
        honestly, and then it fails for the wrong reason.
        """
        resolution = self._resolution(NEWTON, XPBD, "joint_friction", physx=1.25)
        display = self._display("joint_friction", NEWTON, XPBD)

        info = joint_param_info_text(resolution, display)
        # Exactly one of the two lines states what an unauthored friction is worth,
        # and it is the one attached to the field.
        self.assertIn("Nothing is authored", display.note)
        self.assertNotIn("Nothing is authored", info)
        # What the info line is for: the value Newton is ignoring, which the note
        # does not mention.
        self.assertIn("physxJoint:jointFriction (1.25)", info)
        self.assertNotIn("physxJoint:jointFriction", display.note)

    async def test_the_full_text_is_kept_when_no_display_is_given(self) -> None:
        """The table cell has no note beside it, so its tooltip must stand alone."""
        resolution = self._resolution(NEWTON, XPBD, "joint_friction", physx=1.25)
        text = joint_param_info_text(resolution)
        self.assertIn("Nothing is authored", text)
        self.assertIn("physxJoint:jointFriction (1.25)", text)


class TestOneWordForNoLimit(omni.kit.test.AsyncTestCase):
    """The field said "unlimited" while the sentence under it said "no limit".

    Two names for one state, six pixels apart.  Everything the panel renders now
    uses :data:`UNLIMITED_TEXT`.
    """

    def _texts(self):
        """Every user-facing string the panel can produce about an unlimited limit."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint").GetPrim()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        joint.GetAttribute(spec.newton_attr).Set(math.inf)
        joint.GetAttribute(spec.physx_attr).Set(240.0)
        column = advanced_gain_column("max_joint_velocity")
        texts = [
            format_param_value(math.inf),
            max_velocity_engine_text(math.inf, 107.0, True),
            max_velocity_engine_text(240.0, math.inf, True),
        ]
        for backend, solver in ((NEWTON, XPBD), (NEWTON, MUJOCO), (PHYSX, "")):
            resolution = gain_tuner.resolve_joint_param(joint, spec, backend, solver)
            display = read_advanced_param_display(joint, column, backend, solver)
            texts += [
                effective_value_text(resolution),
                joint_param_info_text(resolution),
                joint_param_info_text(resolution, display),
                display.note,
                display.field_format,
            ]
        # And the unauthored case, where the engine leaves the joint unlimited.
        bare = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/bare").GetPrim()
        bare.ApplyAPI(PHYSX_JOINT_API)
        texts.append(read_advanced_param_display(bare, column, PHYSX, "").note)
        return [t for t in texts if t]

    async def test_nothing_says_no_limit_any_more(self) -> None:
        """One word, everywhere the panel names the state."""
        for text in self._texts():
            self.assertNotIn("no limit", text.lower(), msg=repr(text))

    async def test_the_word_is_the_one_the_field_shows(self) -> None:
        """The sentences and the field agree, which is the whole point."""
        said = [t for t in self._texts() if UNLIMITED_TEXT in t]
        self.assertTrue(said, "no sentence names the state at all")
