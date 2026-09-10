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

"""Unit tests for the Gain Tuner's shared presentation helpers."""

from __future__ import annotations

from collections.abc import Iterator

import omni.kit.app
import omni.kit.test
import omni.ui as ui
from isaacsim.robot_setup.gain_tuner.ui.ui_utils import (
    TOOLTIP_WRAP_WIDTH,
    build_wrapped_tooltip,
    set_wrapped_tooltip,
)

# The kind of sentence that drove this: the panel's per-backend explanation, which
# omni.ui laid out on one unwrapped line and the window edge cut mid-word.
LONG = (
    "newton:armature is 0.25, physxJoint:armature is 0.75. "
    "PhysX uses physxJoint:armature (0.75). Edit this field to author a value."
)


async def _settle(frames: int = 8) -> None:
    """Let omni.ui lay out and build what the last change asked for."""
    for _ in range(frames):
        await omni.kit.app.get_app().next_update_async()


def _descendants(widget: ui.Widget) -> Iterator[ui.Widget]:
    """Yield every widget under ``widget``, depth first."""
    for child in ui.Inspector.get_children(widget):
        yield child
        yield from _descendants(child)


class TestWrappedTooltips(omni.kit.test.AsyncTestCase):
    """Tooltips that fit inside the panel instead of running off its right edge.

    A plain ``tooltip=`` string is laid out on a single line, so the panel's own
    sentences -- 100-160 characters, and ~300 for the copy buttons -- were clipped
    by the window edge (``... PhysX uses physxJoint:armatu``).  A built tooltip is
    the only way to bound the width and let the text wrap.
    """

    async def test_a_built_tooltip_wraps_the_whole_sentence(self) -> None:
        """Every word survives, inside a bounded width."""
        window = ui.Window("gt_tooltip_wrap_test", width=760, height=200)
        with window.frame:
            build_wrapped_tooltip(LONG)
        await _settle()

        labels = [w for w in _descendants(window.frame) if isinstance(w, ui.Label)]
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].text, LONG)
        self.assertTrue(labels[0].word_wrap)
        # Bounded well inside the panel, and taller than one line -- which is what
        # says the text wrapped rather than being cut.
        self.assertLessEqual(labels[0].computed_width, TOOLTIP_WRAP_WIDTH)
        self.assertGreater(labels[0].computed_height, 20)
        window.destroy()

    async def test_a_narrower_width_wraps_further(self) -> None:
        """The width is the wrap point, so a smaller one produces more lines."""
        window = ui.Window("gt_tooltip_narrow_test", width=760, height=400)
        with window.frame:
            with ui.VStack():
                build_wrapped_tooltip(LONG, width=TOOLTIP_WRAP_WIDTH)
                build_wrapped_tooltip(LONG, width=TOOLTIP_WRAP_WIDTH // 2)
        await _settle()

        wide, narrow = (w for w in _descendants(window.frame) if isinstance(w, ui.Label))
        self.assertGreater(narrow.computed_height, wide.computed_height)
        window.destroy()

    async def test_a_widget_with_something_to_say_gets_a_built_tooltip(self) -> None:
        """Built, not a plain string: a string is the thing that could not wrap."""
        window = ui.Window("gt_tooltip_set_test", width=400, height=200)
        with window.frame:
            field = ui.FloatDrag()
        set_wrapped_tooltip(field, LONG)
        await _settle()

        self.assertTrue(field.has_tooltip_fn())
        self.assertEqual(field.tooltip, "")
        window.destroy()

    async def test_a_widget_with_nothing_to_say_gets_no_tooltip(self) -> None:
        """An empty sentence must not pop an empty box over the panel."""
        window = ui.Window("gt_tooltip_empty_test", width=400, height=200)
        with window.frame:
            field = ui.FloatDrag()
        set_wrapped_tooltip(field, "")
        await _settle()

        self.assertFalse(field.has_tooltip_fn())
        self.assertEqual(field.tooltip, "")
        window.destroy()

    async def test_a_tooltip_can_be_replaced_and_taken_away(self) -> None:
        """A tooltip describing a value has to follow the value being changed.

        ``omni.ui`` builds a dynamic tooltip's body the first time it is needed and
        serves that body on every later hover, so installing a fresh builder is the
        only way to change what a built tooltip says -- and a widget that has run
        out of things to say has to lose its tooltip rather than keep the last
        sentence.
        """
        window = ui.Window("gt_tooltip_replace_test", width=400, height=200)
        with window.frame:
            field = ui.FloatDrag()
        set_wrapped_tooltip(field, LONG)
        await _settle()
        self.assertTrue(field.has_tooltip_fn(), "precondition: it has something to say")

        set_wrapped_tooltip(field, "Something else entirely.")
        await _settle()
        self.assertTrue(field.has_tooltip_fn())

        set_wrapped_tooltip(field, "")
        await _settle()

        self.assertFalse(field.has_tooltip_fn(), "the widget kept the sentence it no longer means")
        window.destroy()

    async def test_an_offset_moves_the_tooltip_off_its_own_control(self) -> None:
        """The copy buttons' ~300-character tooltip used to cover the button itself.

        A non-zero offset switches omni.ui from following the mouse to anchoring on
        the widget, so the box opens below the control rather than over it.
        """
        window = ui.Window("gt_tooltip_offset_test", width=400, height=200)
        with window.frame:
            with ui.VStack():
                anchored = ui.Button("Copy To Newton", height=20)
                at_mouse = ui.Button("Something Else", height=20)
        set_wrapped_tooltip(anchored, LONG, offset_y=24)
        set_wrapped_tooltip(at_mouse, LONG)
        await _settle()

        self.assertGreater(anchored.tooltip_offset_y, 20)
        self.assertEqual(at_mouse.tooltip_offset_y, 0)
        window.destroy()
