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

"""The opt-in copy-to-other-backend action: targets, writes, no-op, and undo."""

from __future__ import annotations

import math
from types import SimpleNamespace

import omni.kit.test
import omni.kit.undo
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.ui.joint_param_sync import (
    COPYABLE_PARAM_KEYS,
    CopyJointParamsToOtherBackendCommand,
    copy_joint_params_to_other_backend,
    copy_summary_text,
    copy_target_schema_label,
    copyable_joint_params,
    joint_param_copy_targets,
)
from pxr import Sdf, Usd, UsdPhysics

NEWTON_JOINT_API = gain_tuner.NEWTON_JOINT_API
PHYSX_JOINT_API = gain_tuner.PHYSX_JOINT_API

NEWTON = gain_tuner.BACKEND_NEWTON
PHYSX = gain_tuner.BACKEND_PHYSX
XPBD = gain_tuner.SOLVER_XPBD


class _CopyCase(omni.kit.test.AsyncTestCase):
    """Helpers for building joints whose two backends hold different values."""

    _ROOT = "/World/Robot"

    def _stage(self) -> Usd.Stage:
        self._usd_stage = Usd.Stage.CreateInMemory()
        self._usd_stage.DefinePrim(self._ROOT, "Xform")
        return self._usd_stage

    @staticmethod
    def _entry(joint, name="joint"):
        return SimpleNamespace(joint=joint, drive_axis="angular", display_name=name)

    def _joint(self, stage: Usd.Stage, name: str) -> Usd.Prim:
        return UsdPhysics.RevoluteJoint.Define(stage, f"{self._ROOT}/{name}").GetPrim()

    def _diverging_joint(self, stage: Usd.Stage, name: str, key="joint_friction", newton=0.25, physx=0.75):
        joint = self._joint(stage, name)
        spec = gain_tuner.joint_param_spec(key)
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        if newton is not None:
            joint.GetAttribute(spec.newton_attr).Set(newton)
        if physx is not None:
            joint.GetAttribute(spec.physx_attr).Set(physx)
        return joint, spec

    def assertAuthored(self, joint, spec, newton, physx) -> None:
        """Assert both halves' authored values, allowing float storage error.

        The attributes are single-precision floats, so a value such as 0.1 comes
        back as 0.10000000149011612 and an exact comparison would fail on the
        storage rather than on the behaviour under test.
        """
        values = gain_tuner.authored_joint_param_values(joint, spec)
        for schema, want in ((gain_tuner.SCHEMA_NEWTON, newton), (gain_tuner.SCHEMA_PHYSX, physx)):
            if want is None:
                self.assertIsNone(values[schema], msg=schema)
            else:
                self.assertAlmostEqual(values[schema], want, places=5, msg=schema)


class TestCopyTargets(_CopyCase):
    """Which joints the action would actually write something for."""

    async def test_only_joints_the_other_backend_does_not_match_are_targets(self) -> None:
        """A joint the other backend already matches is not offered."""
        stage = self._stage()
        settled = self._joint(stage, "settled")
        armature = gain_tuner.joint_param_spec("armature")
        gain_tuner.author_joint_param(settled, armature, 0.1, NEWTON)
        gain_tuner.author_joint_param(settled, armature, 0.1, PHYSX)
        diverging, _spec = self._diverging_joint(stage, "diverging")

        targets = joint_param_copy_targets(
            [self._entry(settled, "settled"), self._entry(diverging, "diverging")], NEWTON, XPBD
        )

        self.assertEqual([target.display_name for target in targets], ["diverging"])
        self.assertEqual([r.spec.key for r in targets[0].resolutions], ["joint_friction"])

    async def test_the_copy_direction_follows_the_active_backend(self) -> None:
        """From PhysX the PhysX value is the source; from Newton, the Newton one."""
        stage = self._stage()
        joint, _spec = self._diverging_joint(stage, "j0")

        from_newton = copyable_joint_params(joint, NEWTON, XPBD)
        from_physx = copyable_joint_params(joint, PHYSX)

        self.assertAlmostEqual(from_newton[0].effective_value, 0.25, places=5)
        self.assertAlmostEqual(from_physx[0].effective_value, 0.75, places=5)

    async def test_a_value_with_nothing_to_copy_is_skipped(self) -> None:
        """An unresolved parameter has no number, so there is nothing to write."""
        stage = self._stage()
        joint = self._joint(stage, "j0")
        joint.ApplyAPI(NEWTON_JOINT_API)
        self.assertEqual(copyable_joint_params(joint, NEWTON, XPBD), [])

    async def test_an_unlimited_velocity_limit_is_copyable(self) -> None:
        """Unlimited is copyable: `inf` is what the other schema wants to mean it.

        Skipping it made the copy claim the other backend already held these
        values, which was untrue -- it held a real 107 while Newton was unclamped.
        """
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0", key="max_joint_velocity", newton=math.inf, physx=107.0)

        copyable = copyable_joint_params(joint, NEWTON, XPBD)

        self.assertEqual([r.spec.key for r in copyable], ["max_joint_velocity"])
        self.assertEqual(copyable[0].copy_value, math.inf)

        copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint)], NEWTON, XPBD), NEWTON, XPBD
        )

        self.assertEqual(gain_tuner.authored_joint_param_values(joint, spec)[gain_tuner.SCHEMA_PHYSX], math.inf)

    async def test_unlimited_is_not_copied_onto_an_already_unlimited_backend(self) -> None:
        """Both backends leave an unauthored limit unclamped, so nothing differs."""
        stage = self._stage()
        joint, _spec = self._diverging_joint(stage, "j0", key="max_joint_velocity", newton=math.inf, physx=None)
        self.assertEqual([r.spec.key for r in copyable_joint_params(joint, NEWTON, XPBD)], [])

    async def test_zero_armature_is_copyable_onto_newtons_nonzero_default(self) -> None:
        """An unauthored parameter is worth its engine's default, not nothing.

        PhysX simulating armature 0 and Newton defaulting to 0.1 is a real
        difference, so copying it writes; comparing against the raw authored value
        alone would have called an unauthored Newton half a match for zero.
        """
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0", key="armature", newton=None, physx=0.0)

        copyable = copyable_joint_params(joint, PHYSX)

        self.assertEqual([r.spec.key for r in copyable], ["armature"])
        self.assertAlmostEqual(copyable[0].other_effective_value, gain_tuner.NEWTON_DEFAULT_ARMATURE, places=6)

    async def test_a_copy_is_refused_when_the_backend_is_unsupported(self) -> None:
        """An engine with no known joint schema has no other side to copy to."""
        stage = self._stage()
        joint, _spec = self._diverging_joint(stage, "j0")

        self.assertEqual(copyable_joint_params(joint, "remotesim"), [])
        _success, result = CopyJointParamsToOtherBackendCommand.execute(
            stage=stage, joint_path=joint.GetPath().pathString, backend="remotesim"
        )
        self.assertTrue(result.wrote_nothing)
        self.assertEqual(result.unwritable_attrs, [])

    async def test_an_undetermined_value_is_not_copied(self) -> None:
        """With the effective value unknown there is nothing safe to propagate."""
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0", key="armature", newton=None, physx=0.9)
        joint.CreateAttribute(spec.mjc_attr, Sdf.ValueTypeNames.Float).Set(0.4)

        self.assertEqual(copyable_joint_params(joint, NEWTON, ""), [])

    async def test_no_entries_yields_no_targets(self) -> None:
        """An empty (or None) selection is not an error."""
        self.assertEqual(joint_param_copy_targets([], PHYSX), [])
        self.assertEqual(joint_param_copy_targets(None, PHYSX), [])

    async def test_every_multi_schema_param_is_copyable(self) -> None:
        """The action covers the three parameters authored per backend."""
        self.assertEqual(set(COPYABLE_PARAM_KEYS), {"armature", "joint_friction", "max_joint_velocity"})

    async def test_target_schema_label_names_the_other_backend(self) -> None:
        """The UI can say where a copy would land before it happens."""
        self.assertEqual(copy_target_schema_label(NEWTON), "physxJoint:*")
        self.assertEqual(copy_target_schema_label(PHYSX), "newton:*")

    async def test_target_schema_label_names_nothing_for_an_unknown_backend(self) -> None:
        """No schema is better than a guessed one; the caller hides the buttons."""
        for backend in ("remotesim", "", "wobble"):
            self.assertEqual(copy_target_schema_label(backend), "", msg=backend)


class TestCopyWrites(_CopyCase):
    """The write itself: the displayed value, onto the other backend only."""

    async def test_copy_from_newton_writes_the_physx_half(self) -> None:
        """The active backend's own value is untouched; the other one is overwritten."""
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0")

        result = copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )

        self.assertEqual((result.joints_written, result.params_written), (1, 1))
        self.assertEqual(result.unwritable_attrs, [])
        self.assertEqual(result.target_backend, PHYSX)
        self.assertAuthored(joint, spec, 0.25, 0.25)

    async def test_copy_from_physx_writes_the_newton_half(self) -> None:
        """The direction is the mirror image, not a hardcoded Newton target."""
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0")

        copy_joint_params_to_other_backend(stage, joint_param_copy_targets([self._entry(joint, "j0")], PHYSX), PHYSX)

        self.assertAuthored(joint, spec, 0.75, 0.75)

    async def test_copy_seeds_a_backend_with_nothing_authored(self) -> None:
        """The imported-robot case: PhysX has no opinion until this runs."""
        stage = self._stage()
        joint = self._joint(stage, "j0")
        spec = gain_tuner.joint_param_spec("joint_friction")
        gain_tuner.author_joint_param(joint, spec, 0.3, NEWTON)
        self.assertFalse(joint.HasAPI(PHYSX_JOINT_API))

        copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )

        self.assertTrue(joint.HasAPI(PHYSX_JOINT_API))
        self.assertAuthored(joint, spec, 0.3, 0.3)

    async def test_only_the_differing_params_are_written(self) -> None:
        """A parameter the other backend already matches is left alone."""
        stage = self._stage()
        joint, _spec = self._diverging_joint(stage, "j0")
        armature = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(armature.newton_attr).Set(0.5)
        joint.GetAttribute(armature.physx_attr).Set(0.5)

        result = copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )

        self.assertEqual(result.params_written, 1)
        self.assertAuthored(joint, armature, 0.5, 0.5)

    async def test_a_copy_never_authors_mjc(self) -> None:
        """MuJoCo gains stay hand-authored, even when they are in the chain."""
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0")

        copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )

        self.assertFalse(joint.GetAttribute(spec.mjc_attr).IsValid())

    async def test_param_keys_restrict_the_copy(self) -> None:
        """A caller can copy one parameter and leave the others as they are."""
        stage = self._stage()
        joint, friction = self._diverging_joint(stage, "j0")
        armature = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(armature.newton_attr).Set(0.1)
        joint.GetAttribute(armature.physx_attr).Set(0.2)

        copy_joint_params_to_other_backend(
            stage,
            joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD),
            NEWTON,
            XPBD,
            param_keys=["armature"],
        )

        self.assertAuthored(joint, armature, 0.1, 0.1)
        self.assertAuthored(joint, friction, 0.25, 0.75)

    async def test_multi_joint_copy_reports_every_joint(self) -> None:
        """Copying all joints writes each one and totals the result."""
        stage = self._stage()
        j0, _ = self._diverging_joint(stage, "j0")
        j1, _ = self._diverging_joint(stage, "j1", key="armature", newton=0.1, physx=0.2)
        entries = [self._entry(j0, "j0"), self._entry(j1, "j1")]

        result = copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets(entries, NEWTON, XPBD), NEWTON, XPBD
        )

        self.assertEqual((result.joints_written, result.params_written), (2, 2))
        self.assertEqual(joint_param_copy_targets(entries, NEWTON, XPBD), [])


class TestCopyNoOp(_CopyCase):
    """Copying a robot whose backends already match must not write."""

    async def test_no_targets_writes_nothing(self) -> None:
        """An empty target list is a safe no-op that reports as one."""
        result = copy_joint_params_to_other_backend(self._stage(), [], PHYSX)
        self.assertTrue(result.wrote_nothing)
        self.assertEqual(result.params_written, 0)

    async def test_command_on_a_matching_joint_writes_nothing(self) -> None:
        """Executing the command directly on a matching joint changes no state."""
        stage = self._stage()
        joint = self._joint(stage, "j0")
        spec = gain_tuner.joint_param_spec("armature")
        gain_tuner.author_joint_param(joint, spec, 0.4, NEWTON)
        gain_tuner.author_joint_param(joint, spec, 0.4, PHYSX)

        _success, result = CopyJointParamsToOtherBackendCommand.execute(
            stage=stage, joint_path=joint.GetPath().pathString, backend=NEWTON, solver=XPBD
        )

        self.assertTrue(result.wrote_nothing)
        self.assertAuthored(joint, spec, 0.4, 0.4)

    async def test_summary_says_nothing_was_written(self) -> None:
        """The no-op path reports itself instead of claiming a write."""
        text = copy_summary_text(copy_joint_params_to_other_backend(self._stage(), [], PHYSX))
        self.assertIn("already simulates", text)
        self.assertIn("Nothing was written", text)
        self.assertTrue(text.isascii(), f"non-ASCII summary: {text!r}")


class TestCopyUndo(_CopyCase):
    """The copy is a real, grouped undo entry, not an untracked USD write."""

    async def test_undo_restores_the_overwritten_value(self) -> None:
        """Undo puts back the value the other backend held."""
        stage = self._stage()
        joint, spec = self._diverging_joint(stage, "j0")

        copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )
        self.assertAuthored(joint, spec, 0.25, 0.25)

        omni.kit.undo.undo()

        self.assertAuthored(joint, spec, 0.25, 0.75)

    async def test_undo_removes_a_schema_the_copy_applied(self) -> None:
        """Seeding an untouched backend is undone completely, schema included."""
        stage = self._stage()
        joint = self._joint(stage, "j0")
        spec = gain_tuner.joint_param_spec("joint_friction")
        gain_tuner.author_joint_param(joint, spec, 0.3, NEWTON)

        copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )
        omni.kit.undo.undo()

        self.assertFalse(joint.HasAPI(PHYSX_JOINT_API))
        self.assertAuthored(joint, spec, 0.3, None)

    async def test_multi_joint_copy_undoes_as_one_step(self) -> None:
        """A single undo restores every joint the copy touched."""
        stage = self._stage()
        j0, spec0 = self._diverging_joint(stage, "j0")
        j1, spec1 = self._diverging_joint(stage, "j1", key="armature", newton=0.1, physx=0.2)
        entries = [self._entry(j0, "j0"), self._entry(j1, "j1")]

        copy_joint_params_to_other_backend(stage, joint_param_copy_targets(entries, NEWTON, XPBD), NEWTON, XPBD)

        omni.kit.undo.undo()

        self.assertAuthored(j0, spec0, 0.25, 0.75)
        self.assertAuthored(j1, spec1, 0.1, 0.2)

    async def test_undo_restores_every_param_on_one_joint(self) -> None:
        """A joint with two copied params is fully restored by one undo."""
        stage = self._stage()
        joint, friction = self._diverging_joint(stage, "j0")
        armature = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(armature.newton_attr).Set(0.1)
        joint.GetAttribute(armature.physx_attr).Set(0.2)

        result = copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )
        self.assertEqual(result.params_written, 2)

        omni.kit.undo.undo()

        self.assertAuthored(joint, friction, 0.25, 0.75)
        self.assertAuthored(joint, armature, 0.1, 0.2)

    async def test_undo_leaves_an_untouched_param_alone(self) -> None:
        """A parameter the copy skipped is not resurrected or cleared by undo."""
        stage = self._stage()
        joint, _friction = self._diverging_joint(stage, "j0")
        armature = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(armature.newton_attr).Set(0.5)
        joint.GetAttribute(armature.physx_attr).Set(0.5)

        copy_joint_params_to_other_backend(
            stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
        )
        omni.kit.undo.undo()

        self.assertAuthored(joint, armature, 0.5, 0.5)


class TestCopySummary(_CopyCase):
    """The toast text is ASCII, states the consequence, and reports failures."""

    async def test_summary_counts_joints_and_params(self) -> None:
        """A successful copy names what it wrote and where it landed."""
        stage = self._stage()
        j0, _ = self._diverging_joint(stage, "j0")
        j1, _ = self._diverging_joint(stage, "j1", key="armature", newton=0.1, physx=0.2)
        entries = [self._entry(j0, "j0"), self._entry(j1, "j1")]

        text = copy_summary_text(
            copy_joint_params_to_other_backend(stage, joint_param_copy_targets(entries, NEWTON, XPBD), NEWTON, XPBD)
        )

        self.assertIn("2 parameters", text)
        self.assertIn("2 joints", text)
        self.assertIn("PhysX", text)
        self.assertTrue(text.isascii(), f"non-ASCII summary: {text!r}")

    async def test_summary_states_the_consequence(self) -> None:
        """The user must be told the other backend's simulation just changed."""
        stage = self._stage()
        joint, _spec = self._diverging_joint(stage, "j0")

        text = copy_summary_text(
            copy_joint_params_to_other_backend(
                stage, joint_param_copy_targets([self._entry(joint, "j0")], NEWTON, XPBD), NEWTON, XPBD
            )
        )

        self.assertIn("now simulates the copied values", text)

    async def test_summary_surfaces_an_unreachable_schema(self) -> None:
        """A write that did not land is reported, not swallowed."""
        from isaacsim.robot_setup.gain_tuner.ui.joint_param_sync import JointParamCopyResult

        result = JointParamCopyResult(
            joints_written=1, params_written=1, target_backend=NEWTON, unwritable_attrs=["newton:friction"]
        )
        text = copy_summary_text(result)
        self.assertIn("newton:friction", text)
        self.assertIn("omni.usd.schema.newton", text)
        self.assertTrue(text.isascii(), f"non-ASCII summary: {text!r}")
