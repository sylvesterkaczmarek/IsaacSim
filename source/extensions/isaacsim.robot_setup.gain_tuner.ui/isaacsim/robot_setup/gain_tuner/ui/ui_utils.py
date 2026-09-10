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

"""Small presentation helpers shared by the Gain Tuner's widgets."""

from __future__ import annotations

import carb
import omni.ui as ui

from .style import FONT_SIZE, TREEVIEW_BG_COLOR

TOOLTIP_BG_COLOR = TREEVIEW_BG_COLOR
"""Background a built tooltip paints behind its own text.

The panel's own dark well, so a tooltip reads as part of the panel rather than as
Kit's pale-yellow default -- which the panel's light label colour, inherited from
the window style and not overridable from inside the tooltip, cannot be read
against."""

TOOLTIP_WRAP_WIDTH = 360
"""Pixel width a built tooltip wraps its text at.

``omni.ui`` lays a plain ``tooltip=`` string out on one unwrapped line, which the
window clips: a 160-character sentence over a 112 px table cell ended mid-word at
the panel's right edge, and the copy buttons' 300-character tooltip fared worse.
Narrow enough to fit inside the panel docked at its usual widths, wide enough that
a sentence does not turn into a column of single words."""


def build_wrapped_tooltip(text: str, width: int = TOOLTIP_WRAP_WIDTH) -> None:
    """Build a tooltip body that wraps its text instead of running off the panel.

    Written for :meth:`omni.ui.Widget.set_tooltip_fn`, which calls it inside a
    ``ui.Frame`` scope.  The explicit width is what gives ``word_wrap`` something
    to wrap against; without it the label is laid out on one line again.

    The body paints its own dark background rather than sitting on Kit's pale
    yellow tooltip window: the panel's light label colour wins over anything set
    here (it comes from the window's own style), and light grey on pale yellow is
    barely readable.  Painting the background is the half of the pair this code
    can decide.

    Args:
        text: The tooltip sentence(s).
        width: Pixel width to wrap at.
    """
    with ui.ZStack(width=width, height=0):
        ui.Rectangle(style={"background_color": TOOLTIP_BG_COLOR, "border_radius": 2})
        with ui.VStack(height=0):
            ui.Spacer(height=6)
            with ui.HStack(height=0):
                ui.Spacer(width=8)
                ui.Label(text, word_wrap=True, style={"font_size": FONT_SIZE})
                ui.Spacer(width=8)
            ui.Spacer(height=6)


def set_wrapped_tooltip(widget: ui.Widget, text: str, width: int = TOOLTIP_WRAP_WIDTH, offset_y: float = 0.0) -> None:
    """Attach a wrapping tooltip to ``widget``, replacing whatever it had.

    Safe to call again on a widget that is already describing itself, which is how
    a tooltip follows a value the user changed: ``omni.ui`` builds a dynamic
    tooltip's body once, the first time it is needed, and then serves that same
    body on every later hover -- so installing a fresh builder is the only way to
    change what a built tooltip says, and an empty ``text`` has to remove it
    rather than leave the previous sentence in place.

    Args:
        widget: The widget to describe.
        text: The tooltip text; an empty string leaves the widget with no tooltip
            at all, so a widget with nothing to say does not pop up an empty box.
        width: Pixel width to wrap at.
        offset_y: When non-zero, anchor the tooltip's top-left this far below the
            widget's own top-left instead of following the mouse, so a tall
            tooltip cannot cover the control it describes.  ``omni.ui`` switches
            to widget-relative anchoring as soon as either offset is non-zero.
    """
    if not text:
        widget.set_tooltip_fn(None)
        return
    widget.set_tooltip_fn(lambda t=text, w=width: build_wrapped_tooltip(t, w))
    if offset_y:
        widget.tooltip_offset_y = float(offset_y)


def on_copy_to_clipboard(to_copy: str):
    """Copy text to system clipboard.

    Args:
        to_copy: The text to copy to the clipboard.
    """
    try:
        import pyperclip
    except ImportError:
        carb.log_warn("Could not import pyperclip.")
        return
    try:
        pyperclip.copy(to_copy)
    except pyperclip.PyperclipException:
        carb.log_warn(pyperclip.EXCEPT_MSG)
        return
