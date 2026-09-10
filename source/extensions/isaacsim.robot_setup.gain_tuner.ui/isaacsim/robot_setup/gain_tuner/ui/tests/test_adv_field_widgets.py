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

"""What the advanced joint-param *widgets* actually render.

Every other test in this extension asserts on
:class:`~isaacsim.robot_setup.gain_tuner.ui.gain_display.AdvancedParamDisplay`,
which is the right unit for deciding *what* a parameter is worth -- and blind to
the two bugs here, both of which are a correct display never reaching the widget:

* The ``"(default)"`` suffix and the muted colour are passed to ``ui.FloatDrag``
  as ``format`` and ``style`` when it is *constructed*.  A successful write turns
  an unauthored default into an authored value, and nothing re-derived them, so
  the field the user had just dragged Max Joint Velocity into still read
  ``unlimited (default)`` -- the panel asserting the joint was unclamped over a
  stage that now clamped it, and the velocity sweeps scaling their commands by a
  limit the field denied existed.  ``field_format`` was already correct at every
  point; the widget just never asked again.
* A display with nothing to state (``blank``) was handed to a ``FloatDrag``
  anyway, seeded with :attr:`AdvancedParamDisplay.field_value`'s zero and
  disabled, so it rendered ``0.00`` -- the false zero the value object exists to
  remove -- while the table blanked the identical state.
* The sentences *under* the field are the same claim in words, and they were
  built once from the same display and never re-derived.  So the field learned to
  drop ``"(default)"`` and the line beneath it did not: a Max Joint Velocity
  reading ``16.0`` directly over "Nothing is authored here, so Newton leaves this
  unlimited.  Type a value here to author a limit."  The rows are also the half
  that can have nothing left to say, or something new to say, so rewriting text in
  place is not enough on its own.

So these build the real widgets in a real window and read back what they are
rendering.  ``omni.ui`` re-emits a widget from its current properties every
frame, so ``field.format``, ``field.style``, ``label.text`` and ``row.visible``
*are* what is on screen; what these cannot see is the glyphs, which no headless
test can.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import omni.kit.app
import omni.kit.test
import omni.ui as ui
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.ui.backend_context import BackendContext
from isaacsim.robot_setup.gain_tuner.ui.gain_display import DEFAULT_FIELD_SUFFIX, UNLIMITED_TEXT
from isaacsim.robot_setup.gain_tuner.ui.style import MUTED_LABEL_COLOR
from isaacsim.robot_setup.gain_tuner.ui.ui_builder import UIBuilder
from pxr import Usd, UsdPhysics

NEWTON = gain_tuner.BACKEND_NEWTON
PHYSX = gain_tuner.BACKEND_PHYSX
XPBD = gain_tuner.SOLVER_XPBD


class _AdvFieldCase(omni.kit.test.AsyncTestCase):
    """A real window holding real advanced-param fields, built by the real code."""

    async def setUp(self) -> None:
        self._window = ui.Window("gain_tuner_adv_field_test", width=480, height=320)
        self._stage = Usd.Stage.CreateInMemory()
        self._stack = None

    async def tearDown(self) -> None:
        self._window.destroy()
        self._window = None
        self._stage = None
        self._stack = None

    def _joint(self, *api_names: str) -> Usd.Prim:
        """A revolute joint with the given joint APIs applied."""
        joint = UsdPhysics.RevoluteJoint.Define(self._stage, "/Robot/joint").GetPrim()
        for name in api_names:
            joint.ApplyAPI(name)
        return joint

    @staticmethod
    def _builder(backend: str, solver: str = "") -> UIBuilder:
        """A UIBuilder with only what the advanced-field builders touch.

        ``__new__`` rather than a constructed panel: the field builders reach into
        a handful of attributes, and standing up the whole extension to exercise
        three rows would make the test about the panel's startup instead.
        """
        builder = UIBuilder.__new__(UIBuilder)
        builder._backend_ctx = SimpleNamespace(backend=backend, solver_type=solver)
        builder._gains_tuner = None
        builder._timeline = SimpleNamespace(is_playing=lambda: False)
        builder._suspend_detail_writes = False
        builder._detail_adv_models = []
        builder._sync_selected_row_to_table = lambda: None
        builder._sync_backend_if_engine_changed = lambda: False
        return builder

    def _build_fields(self, builder: UIBuilder, joint, *param_keys: str, entry=None) -> list:
        """Build advanced fields into the window and return what they constructed.

        All of them in one pass: filling the window's frame a second time destroys
        what the first pass built, and one test needs two live fields at once.
        The enclosing stack is kept as ``self._stack``, so a test can read the
        section's height the way the user reads its spacing.

        Returns:
            One record per field row built, each carrying the row's label, the
            ``omni.ui`` widget, and the model it was given (None for a blank
            placeholder, which is half of what these tests assert).
        """
        built = []
        with self._window.frame:
            with ui.VStack(spacing=4, height=0) as stack:
                with mock.patch.object(UIBuilder, "_field_row", _recording_field_row(built)):
                    for param_key in param_keys:
                        builder._adv_joint_param_field(joint, param_key, entry=entry)
        self._stack = stack
        return built

    @staticmethod
    def _visible_lines(builder: UIBuilder) -> list[str]:
        """Every explanation the panel is showing under its advanced fields.

        Read off the built labels rather than re-derived, because "what the panel
        says" is the whole question: a row whose text is stale, and a row that is
        hidden while still holding a sentence, are the two states being ruled out.
        """
        return [
            row.label.text
            for registered in builder._detail_adv_models
            for row in registered.explain_rows
            if row.row.visible and row.label.text
        ]

    async def _settle(self, frames: int = 3) -> None:
        """Let ``omni.ui`` lay the window out again after a text change."""
        for _ in range(frames):
            await omni.kit.app.get_app().next_update_async()


def _recording_field_row(built: list):
    """Wrap ``_field_row`` so the test can see every widget it constructs.

    The row is what actually calls ``ui.FloatDrag``/``ui.FloatField``, and it
    already has an ``on_built`` hook -- but only the non-blank path passes one, and
    the blank path is half of what is being asserted.  Recording at the row catches
    both, and calls straight through so the widget really is built.
    """
    original = UIBuilder._field_row

    def wrapper(self, label, model, **kwargs):
        caller_on_built = kwargs.pop("on_built", None)

        def on_built(widget) -> None:
            built.append(SimpleNamespace(label=label, widget=widget, model=model))
            if caller_on_built is not None:
                caller_on_built(widget)

        return original(self, label, model, on_built=on_built, **kwargs)

    return wrapper


class TestTheDefaultMarkerFollowsTheEdit(_AdvFieldCase):
    """The critical one: a field that keeps claiming a value is the engine's."""

    async def test_an_unauthored_armature_field_is_marked_and_muted(self) -> None:
        """The state the marker exists for, read off the widget rather than inferred."""
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "armature")

        self.assertEqual(len(built), 1)
        field = built[0].widget
        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), field.format)
        self.assertEqual(field.style.get("color"), MUTED_LABEL_COLOR)

    async def test_authoring_a_value_clears_the_marker_and_the_muting(self) -> None:
        """The bug: a successful write left the field describing the value it replaced.

        The edit is made through the model, which is exactly what a drag or a typed
        value does -- the value-changed handler is the only path into the write, and
        it is the path that has to re-render.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        builder = self._builder(NEWTON, XPBD)
        spec = gain_tuner.joint_param_spec("armature")

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "armature")
            field, model = built[0].widget, built[0].model
            self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), field.format, "precondition: built as a default")

            model.set_value(0.5)

        self.assertAlmostEqual(joint.GetAttribute(spec.newton_attr).Get(), 0.5, places=5)
        self.assertNotIn("default", field.format, f"the field still claims a default: {field.format!r}")
        self.assertNotEqual(field.style.get("color"), MUTED_LABEL_COLOR, "the field is still muted")

    async def test_a_dragged_velocity_limit_stops_reading_unlimited(self) -> None:
        """The worst instance: the field denying a limit the stage now enforces.

        An unauthored limit renders as the word ``unlimited``, with no ``%``
        conversion at all, so the field showed no number while it was being dragged
        and read ``unlimited (default)`` once the drag landed on a real limit.  The
        velocity sweeps scale their commands by that limit.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        builder = self._builder(NEWTON, XPBD)
        spec = gain_tuner.joint_param_spec("max_joint_velocity")

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "max_joint_velocity")
            field, model = built[0].widget, built[0].model
            self.assertIn(UNLIMITED_TEXT, field.format, "precondition: built as unlimited")
            self.assertNotIn("%", field.format, "precondition: and therefore showing no number")

            model.set_value(240.0)

        self.assertAlmostEqual(joint.GetAttribute(spec.newton_attr).Get(), 240.0, places=3)
        self.assertNotIn(UNLIMITED_TEXT, field.format, f"the field still says unlimited: {field.format!r}")
        self.assertNotIn("default", field.format)
        # A format with no conversion renders no digits at all, which is what left
        # the first drag out of the unlimited state with no numeric feedback.
        self.assertTrue(
            field.format == "" or "%" in field.format,
            f"the field shows no number: {field.format!r}",
        )

    async def test_an_authored_field_is_never_marked_to_begin_with(self) -> None:
        """The control: the refresh must not mark a value the user did author."""
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.33)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "armature")
            field, model = built[0].widget, built[0].model
            self.assertNotIn("default", field.format)

            model.set_value(0.44)

        self.assertNotIn("default", field.format)
        self.assertNotEqual(field.style.get("color"), MUTED_LABEL_COLOR)

    async def test_editing_one_field_re_renders_the_others(self) -> None:
        """Authoring one parameter can change what another field is describing.

        The three advanced fields are three resolves of the same joint under the
        same backend, so the refresh covers all the registered ones rather than
        only the edited one.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            armature, velocity = self._build_fields(builder, joint, "armature", "max_joint_velocity")
            self.assertEqual(len(builder._detail_adv_models), 2)
            # Author the velocity limit; the armature field is refreshed too, and
            # must come back saying what it said before rather than being reset.
            velocity.model.set_value(120.0)

        self.assertIn(DEFAULT_FIELD_SUFFIX.strip(), armature.widget.format)
        self.assertNotIn(UNLIMITED_TEXT, velocity.widget.format)


class TestTheExplanationFollowsTheEdit(_AdvFieldCase):
    """The line under the field, which said the opposite of the field itself.

    The field's ``(default)`` marker and the sentence beneath it answer the same
    question -- is this the user's number or the engine's -- and only the marker was
    re-derived after a write.  So the panel showed ``16.0`` over "Nothing is
    authored here, so Newton leaves this unlimited", which is worse than the stale
    marker was: it states the contradiction in a full sentence.
    """

    async def test_authoring_a_limit_takes_away_the_line_denying_it(self) -> None:
        """The reported defect, in the field it was reported in.

        Newton, nothing authored, so the panel offers the engine's own unlimited
        default and says so.  A drag authors a real limit, which answers the
        sentence's question -- and there is then nothing left for it to say, so the
        row has to go rather than be reworded.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        builder = self._builder(NEWTON, XPBD)
        spec = gain_tuner.joint_param_spec("max_joint_velocity")

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "max_joint_velocity")
            field, model = built[0].widget, built[0].model
            before = self._visible_lines(builder)
            self.assertTrue(
                any(UNLIMITED_TEXT in line for line in before),
                f"precondition: the panel says the joint is unlimited: {before!r}",
            )
            self.assertTrue(field.has_tooltip_fn(), "precondition: the field says it on hover too")

            model.set_value(16.0)

        self.assertAlmostEqual(joint.GetAttribute(spec.newton_attr).Get(), 16.0, places=3)
        self.assertNotIn(UNLIMITED_TEXT, field.format, "precondition: the field itself followed the edit")
        after = self._visible_lines(builder)
        self.assertEqual(
            [line for line in after if UNLIMITED_TEXT in line or "othing is authored" in line],
            [],
            f"the panel still denies the limit it just authored, under a field reading 16.0: {after!r}",
        )
        self.assertFalse(field.has_tooltip_fn(), "the field still pops up the sentence it was built with")

    async def test_the_same_line_is_taken_away_under_physx(self) -> None:
        """The defect was reported under both engines, so it is ruled out under both.

        The sentence names whichever engine is running, and the refresh does not
        branch on it -- but the value resolves through a different schema chain
        under PhysX, so the state it is being read out of is not the same state.
        """
        joint = self._joint(gain_tuner.PHYSX_JOINT_API)
        builder = self._builder(PHYSX)
        spec = gain_tuner.joint_param_spec("max_joint_velocity")

        with mock.patch.object(BackendContext, "active_backend_label", return_value=PHYSX):
            built = self._build_fields(builder, joint, "max_joint_velocity")
            before = self._visible_lines(builder)
            self.assertTrue(
                any(UNLIMITED_TEXT in line and PHYSX in line for line in before),
                f"precondition: PhysX is leaving the joint unlimited and saying so: {before!r}",
            )

            built[0].model.set_value(16.0)

        self.assertAlmostEqual(joint.GetAttribute(spec.physx_attr).Get(), 16.0, places=3)
        after = self._visible_lines(builder)
        self.assertEqual(
            [line for line in after if UNLIMITED_TEXT in line],
            [],
            f"the panel still denies the limit it just authored: {after!r}",
        )

    async def test_a_divergence_the_edit_creates_gets_a_line_of_its_own(self) -> None:
        """The half that rewriting text in place cannot do: a line appearing.

        Both backends authored to the same number have nothing to point out, so the
        field is built with no explanation under it at all.  Editing one of them
        makes the two disagree -- which is a legitimate authoring choice, and the
        one thing the panel exists to state -- so a row has to be there to say it.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API, gain_tuner.PHYSX_JOINT_API)
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.5)
        joint.GetAttribute(spec.physx_attr).Set(0.5)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            self._build_fields(builder, joint, "armature")
            self.assertEqual(self._visible_lines(builder), [], "precondition: the backends agree, so nothing is said")

            builder._detail_adv_models[0].model.set_value(0.7)

        after = self._visible_lines(builder)
        self.assertTrue(
            any(spec.physx_attr in line for line in after),
            f"the panel does not report the divergence the edit created: {after!r}",
        )
        self.assertTrue(
            any("0.7" in line for line in after),
            f"the reported divergence is not the one on the stage: {after!r}",
        )

    async def test_the_section_reflows_rather_than_leaving_the_line_blank(self) -> None:
        """A row with nothing to say must take up no room, or the panel gaps.

        ``omni.ui`` gives an invisible child neither its own height nor the stack's
        spacing, which is what lets a sentence be taken away without leaving a
        blank line behind it -- and what keeps the rows above the dragged field
        where they were.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "max_joint_velocity")
            await self._settle()
            with_line = self._stack.computed_height
            field_top = built[0].widget.screen_position_y

            built[0].model.set_value(16.0)
            await self._settle()

        without_line = self._stack.computed_height
        self.assertLess(
            without_line,
            with_line,
            f"the section kept the removed line's height: {with_line} -> {without_line}",
        )
        self.assertGreater(without_line, 0.0, "the section collapsed entirely")
        self.assertAlmostEqual(
            built[0].widget.screen_position_y,
            field_top,
            places=3,
            msg="the dragged field moved under the cursor",
        )

    async def test_an_explanation_that_only_changes_wording_is_rewritten(self) -> None:
        """The ordinary case: the same row, a different sentence.

        Newton authors nothing, so it falls back to PhysX's half and says which one
        it is reading; the edit makes Newton's own half the answer, and the line
        becomes the divergence between the two.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API, gain_tuner.PHYSX_JOINT_API)
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.physx_attr).Set(0.5)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            self._build_fields(builder, joint, "armature")
            before = self._visible_lines(builder)
            self.assertTrue(
                any("falls back to" in line for line in before),
                f"precondition: Newton is reading PhysX's value: {before!r}",
            )

            builder._detail_adv_models[0].model.set_value(0.7)

        after = self._visible_lines(builder)
        self.assertEqual(
            [line for line in after if "falls back to" in line],
            [],
            f"the panel still says Newton falls back, over its own authored value: {after!r}",
        )
        self.assertTrue(
            any(f"{spec.newton_attr} (0.7)" in line for line in after),
            f"the panel does not name the value the edit authored: {after!r}",
        )

    async def test_a_quiet_field_is_not_given_something_to_say(self) -> None:
        """The control: the refresh must not invent an explanation.

        An authored value both backends agree on is the state with nothing to point
        out, and it has to stay that way through an edit -- otherwise every drag
        would grow a line under it.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API, gain_tuner.PHYSX_JOINT_API)
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.5)
        joint.GetAttribute(spec.physx_attr).Set(0.5)
        builder = self._builder(NEWTON, XPBD)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            self._build_fields(builder, joint, "armature")
            # Both halves move together, so the two never come to disagree.
            joint.GetAttribute(spec.physx_attr).Set(0.6)
            builder._detail_adv_models[0].model.set_value(0.6)

        self.assertEqual(self._visible_lines(builder), [])


class TestABlankFieldShowsNoNumber(_AdvFieldCase):
    """Defect 4: the detail panel rendered ``0.00`` where the table blanked."""

    async def test_an_unsupported_backend_builds_no_number_at_all(self) -> None:
        """A disabled ``0.00`` is a number no engine is using, read as authored.

        This is the state the whole value object exists to remove, and the detail
        panel was still rendering it: ``blank`` display, ``field_value`` zero,
        ``field_format`` empty, so a ``FloatDrag`` seeded with 0.0 and disabled
        showed the digit -- with the note above it saying the value could not be
        read.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API, gain_tuner.PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        builder = self._builder("remotesim")

        with mock.patch.object(BackendContext, "active_backend_label", return_value="remotesim"):
            built = self._build_fields(builder, joint, "armature")

        self.assertEqual(len(built), 1)
        self.assertIsNone(built[0].model, "a blank field has no model, so it has no number to show")
        self.assertNotIsInstance(built[0].widget, ui.FloatDrag)
        self.assertEqual(builder._detail_adv_models, [], "and nothing editable was registered")

    async def test_an_undetermined_newton_solver_builds_no_number_either(self) -> None:
        """The common case: Newton before its solver has resolved.

        Which schema chain the value resolves through is decided by the solver, so
        until one is known there is no value -- and, unlike the unsupported
        backend, no note either, so the panel showed a bare ``0.00`` with nothing
        to explain it.
        """
        joint = self._joint(gain_tuner.NEWTON_JOINT_API, gain_tuner.PHYSX_JOINT_API)
        joint.GetAttribute("physxJoint:armature").Set(0.9)
        builder = self._builder(NEWTON, solver="")

        with mock.patch.object(BackendContext, "active_backend_label", return_value=NEWTON):
            built = self._build_fields(builder, joint, "armature")

        self.assertIsNone(built[0].model)
        self.assertNotIsInstance(built[0].widget, ui.FloatDrag)

    async def test_a_supported_backend_still_builds_a_real_field(self) -> None:
        """The control: blanking must not have swallowed the normal case."""
        joint = self._joint(gain_tuner.PHYSX_JOINT_API)
        builder = self._builder(PHYSX)

        with mock.patch.object(BackendContext, "active_backend_label", return_value=PHYSX):
            built = self._build_fields(builder, joint, "armature")

        self.assertIsNotNone(built[0].model)
        self.assertIsInstance(built[0].widget, ui.FloatDrag)
