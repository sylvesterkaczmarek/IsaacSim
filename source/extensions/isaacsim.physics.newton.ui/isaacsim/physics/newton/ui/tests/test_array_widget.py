# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the numeric-array pop-up editor helpers and MuJoCo prim create command."""

from unittest.mock import patch

import omni.kit.app
import omni.kit.undo
from omni.kit.test.async_unittest import AsyncTestCase
from pxr import Sdf, Usd, Vt

from ..array_widget import (
    _OPEN_EDITORS,
    _SCROLL_OWNER,
    _append_element,
    _array_ctor,
    _ArrayEditorWindow,
    _defer_once,
    _get_array_constraint,
    _open_editor,
    _pad_constrained_values,
    _read_values,
    _remove_element,
    _summary_text,
    close_all_editors,
)
from ..newton_commands import CreateMujocoPrimCommand


class ArrayWidgetHelperTests(AsyncTestCase):
    """Validate the pure helpers behind the array editor window."""

    def setUp(self) -> None:
        """Create an in-memory stage with a keyframe-like array attribute."""
        self._stage = Usd.Stage.CreateInMemory()
        self._prim = self._stage.DefinePrim("/World/Keyframe")
        self._attr = self._prim.CreateAttribute("mjc:qpos", Sdf.ValueTypeNames.DoubleArray)
        self._attr.Set(Vt.DoubleArray([1.0, 2.0, 3.0]))

    def tearDown(self) -> None:
        """Release the in-memory stage."""
        self._stage = None

    def test_array_ctor_maps_supported_types(self) -> None:
        """Recognise the numeric array types edited by index, reject others."""
        self.assertIs(_array_ctor(Sdf.ValueTypeNames.DoubleArray), Vt.DoubleArray)
        self.assertIs(_array_ctor(Sdf.ValueTypeNames.IntArray), Vt.IntArray)
        self.assertIsNone(_array_ctor(Sdf.ValueTypeNames.Double))
        self.assertIsNone(_array_ctor(Sdf.ValueTypeNames.StringArray))

    def test_read_values_returns_list(self) -> None:
        """Read authored values, and an empty list for unauthored attributes."""
        self.assertEqual(_read_values(self._stage, self._attr.GetPath()), [1.0, 2.0, 3.0])
        empty_attr = self._prim.CreateAttribute("mjc:qvel", Sdf.ValueTypeNames.DoubleArray)
        self.assertEqual(_read_values(self._stage, empty_attr.GetPath()), [])

    def test_summary_text_previews_and_counts(self) -> None:
        """Summarise short and long arrays with a trailing count."""
        self.assertEqual(_summary_text([]), "empty (0)")
        self.assertIn("(3)", _summary_text([1.0, 2.0, 3.0]))
        long_summary = _summary_text([float(i) for i in range(12)])
        self.assertIn("...", long_summary)
        self.assertIn("(12)", long_summary)

    def test_append_element_uses_typed_default(self) -> None:
        """Append a float or integer default depending on the array type."""
        float_values: list = []
        _append_element(float_values, is_int=False)
        self.assertEqual(float_values, [0.0])
        self.assertIsInstance(float_values[0], float)
        int_values: list = []
        _append_element(int_values, is_int=True)
        self.assertEqual(int_values, [0])
        self.assertIsInstance(int_values[0], int)

    def test_remove_element_by_index(self) -> None:
        """Remove the element at a valid index and ignore out-of-range indices."""
        values = [1.0, 2.0, 3.0]
        _remove_element(values, 1)
        self.assertEqual(values, [1.0, 3.0])
        _remove_element(values, 5)
        self.assertEqual(values, [1.0, 3.0])
        _remove_element(values, -1)
        self.assertEqual(values, [1.0, 3.0])

    def test_add_then_remove_round_trip(self) -> None:
        """Appending an element and removing it again restores the original list."""
        values = [1.0, 2.0, 3.0]
        _append_element(values, is_int=False)
        _remove_element(values, len(values) - 1)
        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_fixed_size_constraints_match_mujoco_model_fields(self) -> None:
        """Use MuJoCo's compiled lengths for actuator and solver parameter vectors."""
        cases = {
            "mjc:gear": 6,
            "mjc:dynPrm": 10,
            "mjc:gainPrm": 10,
            "mjc:biasPrm": 10,
            "mjc:solref": 2,
            "mjc:solimp": 5,
            "mjc:solimpfriction": 5,
            "mjc:solimplimit": 5,
            "mjc:solreffriction": 2,
            "mjc:solreflimit": 2,
            "mjc:springdamper": 2,
            "mjc:option:o_friction": 5,
            "mjc:option:o_solimp": 5,
            "mjc:option:o_solref": 2,
            "mjc:springlength": 2,
        }
        for attr_name, expected_size in cases.items():
            constraint = _get_array_constraint(Sdf.Path(f"/World/Prim.{attr_name}"))
            self.assertIsNotNone(constraint)
            self.assertEqual(constraint.size, expected_size)
        self.assertIsNone(_get_array_constraint(Sdf.Path("/World/Prim.mjc:qpos")))
        springlength = _get_array_constraint(Sdf.Path("/World/Tendon.mjc:springlength"))
        self.assertFalse(springlength.pad_on_commit)

    def test_pad_constrained_values_uses_per_index_defaults(self) -> None:
        """Fill deleted entries from the matching MuJoCo attribute defaults."""
        constraint = _get_array_constraint(Sdf.Path("/World/Actuator.mjc:gear"))
        values = [2.0, 3.0]
        self.assertEqual(_pad_constrained_values(values, constraint), [2.0, 3.0, 0.0, 0.0, 0.0, 0.0])

    def test_springlength_does_not_pad_single_value(self) -> None:
        """Keep a single springlength entry; MuJoCo treats one value as rest length."""
        constraint = _get_array_constraint(Sdf.Path("/World/Tendon.mjc:springlength"))
        values = [-0.5]
        self.assertEqual(_pad_constrained_values(values, constraint), [-0.5])


class CreateMujocoPrimCommandTests(AsyncTestCase):
    """Validate creation of concrete MuJoCo typed prims."""

    def setUp(self) -> None:
        """Create an in-memory stage to receive new prims."""
        self._stage = Usd.Stage.CreateInMemory()

    def tearDown(self) -> None:
        """Release the in-memory stage."""
        self._stage = None

    def test_creates_typed_prim(self) -> None:
        """Create a prim with the requested MuJoCo type name."""
        command = CreateMujocoPrimCommand(self._stage, "/", "MjcActuator", "Actuator")
        prim = command.do()
        self.assertTrue(prim and prim.IsValid())
        self.assertEqual(prim.GetTypeName(), "MjcActuator")

    def test_repeated_creation_is_unique(self) -> None:
        """Avoid path collisions when creating the same type repeatedly."""
        first = CreateMujocoPrimCommand(self._stage, "/", "MjcKeyframe", "Keyframe").do()
        second = CreateMujocoPrimCommand(self._stage, "/", "MjcKeyframe", "Keyframe").do()
        self.assertNotEqual(first.GetPath(), second.GetPath())

    def test_undo_removes_created_prim(self) -> None:
        """Undo deletes the prim the command created."""
        command = CreateMujocoPrimCommand(self._stage, "/", "MjcTendon", "Tendon")
        prim = command.do()
        path = prim.GetPath()
        command.undo()
        self.assertFalse(self._stage.GetPrimAtPath(path).IsValid())


class ArrayEditorWindowTests(AsyncTestCase):
    """Validate the OK/Cancel commit semantics and teardown safety of the editor window."""

    def setUp(self) -> None:
        """Create an in-memory stage with a keyframe-like array attribute."""
        self._stage = Usd.Stage.CreateInMemory()
        self._prim = self._stage.DefinePrim("/World/Keyframe")
        self._attr = self._prim.CreateAttribute("mjc:qpos", Sdf.ValueTypeNames.DoubleArray)
        self._attr.Set(Vt.DoubleArray([1.0, 2.0, 3.0]))
        _OPEN_EDITORS.clear()
        # A commit in an earlier test may leave a pending scroll restore, which would
        # coalesce with (and hide) the one this test expects.
        _SCROLL_OWNER._restore_task = None

    def tearDown(self) -> None:
        """Release open editors and the in-memory stage."""
        close_all_editors()
        _SCROLL_OWNER._restore_task = None
        self._stage = None

    def _make_window(self) -> _ArrayEditorWindow:
        """Create an editor window bound to the in-memory keyframe attribute.

        Returns:
            A freshly built editor window instance.
        """
        return _ArrayEditorWindow(
            self._stage,
            self._attr.GetPath(),
            Sdf.ValueTypeNames.DoubleArray,
            "Qpos",
            "documentation",
        )

    def test_cancel_leaves_attribute_unchanged(self) -> None:
        """Cancel discards local edits, adds, and removes without touching USD."""
        window = self._make_window()
        try:
            window._add_entry()
            window._remove_entry(0)
            window._values[0] = 99.0
            window._on_cancel()
            self.assertEqual(list(self._attr.Get()), [1.0, 2.0, 3.0])
        finally:
            window.destroy()

    def test_ok_commits_edited_values_and_undo_restores(self) -> None:
        """OK writes the edited array in one undoable step and undo restores the original."""
        window = self._make_window()
        try:
            window._values = [9.0, 2.0, 3.0]
            window._models = []
            window._add_entry()
            window._on_ok()
            self.assertEqual(list(self._attr.Get()), [9.0, 2.0, 3.0, 0.0])
            omni.kit.undo.undo()
            self.assertEqual(list(self._attr.Get()), [1.0, 2.0, 3.0])
        finally:
            window.destroy()

    def test_ok_pads_missing_fixed_size_entries(self) -> None:
        """OK fills missing entries of a fixed-size MuJoCo array from its defaults."""
        unauthored = self._prim.CreateAttribute("mjc:gear", Sdf.ValueTypeNames.DoubleArray)
        self.assertFalse(unauthored.HasAuthoredValue())
        window = _ArrayEditorWindow(
            self._stage,
            unauthored.GetPath(),
            Sdf.ValueTypeNames.DoubleArray,
            "Gear",
            "documentation",
        )
        try:
            self.assertEqual(window._values, [])
            window._on_ok()
            self.assertTrue(unauthored.HasAuthoredValue())
            self.assertEqual(list(unauthored.Get()), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        finally:
            window.destroy()

    def test_add_stops_at_fixed_size_limit(self) -> None:
        """Do not append beyond the MuJoCo field's compiled length."""
        gear = self._prim.CreateAttribute("mjc:gear", Sdf.ValueTypeNames.DoubleArray)
        gear.Set(Vt.DoubleArray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        window = _ArrayEditorWindow(
            self._stage,
            gear.GetPath(),
            Sdf.ValueTypeNames.DoubleArray,
            "Gear",
            "documentation",
        )
        try:
            window._models = []
            window._add_entry()
            self.assertEqual(window._values, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        finally:
            window.destroy()

    def test_ok_authors_previously_unauthored_attribute(self) -> None:
        """OK writes an unauthored array and undo restores the unauthored state."""
        unauthored = self._prim.CreateAttribute("mjc:ctrl", Sdf.ValueTypeNames.DoubleArray)
        self.assertFalse(unauthored.HasAuthoredValue())
        window = _ArrayEditorWindow(
            self._stage,
            unauthored.GetPath(),
            Sdf.ValueTypeNames.DoubleArray,
            "Ctrl",
            "documentation",
        )
        try:
            window._values = [1.5, 2.5]
            window._models = []
            window._on_ok()
            self.assertTrue(unauthored.HasAuthoredValue())
            self.assertEqual(list(unauthored.Get()), [1.5, 2.5])
            omni.kit.undo.undo()
            self.assertFalse(unauthored.HasAuthoredValue())
            self.assertIsNone(unauthored.Get())
        finally:
            window.destroy()

    def test_unsigned_array_rejects_negative_values_without_opening_undo_group(self) -> None:
        """Reject invalid unsigned values before executing or grouping a USD command."""
        unsigned = self._prim.CreateAttribute("mjc:indices", Sdf.ValueTypeNames.UIntArray)
        window = _ArrayEditorWindow(
            self._stage,
            unsigned.GetPath(),
            Sdf.ValueTypeNames.UIntArray,
            "Indices",
            "documentation",
        )
        try:
            window._values = [-1]
            window._models = []
            with (
                patch("omni.kit.commands.execute") as execute_mock,
                patch("omni.kit.undo.begin_group") as begin_group_mock,
            ):
                window._commit()
            execute_mock.assert_not_called()
            begin_group_mock.assert_not_called()
            self.assertFalse(unsigned.HasAuthoredValue())
        finally:
            window.destroy()

    def test_destroy_is_idempotent(self) -> None:
        """The teardown path can run twice without error."""
        window = self._make_window()
        window.destroy()
        window.destroy()
        self.assertIsNone(window._window)

    async def test_ok_preserves_property_panel_scroll(self) -> None:
        """A committing OK saves and restores the Property panel scroll offset.

        Writing the attribute rebuilds the Property window, which resets its scrolling
        frame; without the save/restore pair the panel jumps back to the top.
        """
        calls: list[str] = []
        fake_window = type(
            "FakePropertyWindow",
            (),
            {
                "save_scroll_pos": lambda _self, reset=False: calls.append(f"save({reset})"),
                "restore_scroll_pos": lambda _self: calls.append("restore"),
            },
        )()
        window = self._make_window()
        app = omni.kit.app.get_app()
        try:
            with patch("omni.kit.window.property.get_window", return_value=fake_window):
                window._values = [4.0, 5.0, 6.0]
                window._models = []
                window._on_ok()
                self.assertEqual(calls, ["save(False)"])
                await app.next_update_async()
                await app.next_update_async()
            self.assertEqual(calls, ["save(False)", "restore"])
            self.assertEqual(list(self._attr.Get()), [4.0, 5.0, 6.0])
        finally:
            window.destroy()

    async def test_cancel_and_noop_ok_leave_scroll_untouched(self) -> None:
        """Only a commit touches scroll state; Cancel and an unedited OK must not."""
        calls: list[str] = []
        fake_window = type(
            "FakePropertyWindow",
            (),
            {
                "save_scroll_pos": lambda _self, reset=False: calls.append("save"),
                "restore_scroll_pos": lambda _self: calls.append("restore"),
            },
        )()
        app = omni.kit.app.get_app()
        with patch("omni.kit.window.property.get_window", return_value=fake_window):
            window = self._make_window()
            window._models = []
            window._on_ok()
            window.destroy()

            window = self._make_window()
            window._values = [7.0]
            window._on_cancel()
            window.destroy()

            await app.next_update_async()
            await app.next_update_async()
        self.assertEqual(calls, [])

    async def test_defer_once_deduplicates_and_skips_destroyed(self) -> None:
        """Deferred callbacks coalesce and are skipped after the owner is destroyed."""
        owner = type("Owner", (), {"_destroyed": False, "_task": None})()
        calls: list[int] = []

        def _cb() -> None:
            calls.append(1)

        _defer_once(owner, "_task", _cb)
        _defer_once(owner, "_task", _cb)
        self.assertIsNotNone(owner._task)
        app = omni.kit.app.get_app()
        # The deferred task and this test both wait on the next update; yield twice so
        # the callback is guaranteed to have run before we assert.
        await app.next_update_async()
        await app.next_update_async()
        self.assertEqual(calls, [1])
        self.assertIsNone(owner._task)

        owner._destroyed = True
        _defer_once(owner, "_task", _cb)
        await app.next_update_async()
        await app.next_update_async()
        self.assertEqual(calls, [1])

    async def test_close_all_editors_clears_registry(self) -> None:
        """Extension shutdown closes every tracked editor without leaving registry entries."""
        _open_editor(self._stage, self._attr.GetPath(), Sdf.ValueTypeNames.DoubleArray, "Qpos", "")
        self.assertEqual(len(_OPEN_EDITORS), 1)
        close_all_editors()
        self.assertEqual(len(_OPEN_EDITORS), 0)
        app = omni.kit.app.get_app()
        await app.next_update_async()
        await app.next_update_async()

    async def test_schedule_teardown_is_idempotent(self) -> None:
        """Rapid close requests schedule only one deferred destroy."""
        window = self._make_window()
        key = str(self._attr.GetPath())
        _OPEN_EDITORS[key] = window
        window._schedule_teardown()
        window._schedule_teardown()
        self.assertNotIn(key, _OPEN_EDITORS)
        self.assertIsNotNone(window._teardown_task)
        app = omni.kit.app.get_app()
        await app.next_update_async()
        await app.next_update_async()
        self.assertTrue(window._destroyed)
        self.assertIsNone(window._window)
