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

"""Shared visual tokens and styles for the System Identification UI."""

from __future__ import annotations

import omni.ui as ui

#: Secondary/status text (grey).
MUTED_LABEL_STYLE = {"color": 0xFF8A8A8A, "margin_width": 4}
#: Stale/attention text (blue, matches the "weak" verdict color).
STALE_LABEL_STYLE = {"color": 0xFF4A8DC2, "margin_width": 4}
#: Recommendation hints (bright blue).
HINT_LABEL_STYLE = {"color": 0xFF4AC2FF, "margin_width": 4}
#: Page and workflow-shell colors.
PANEL_BACKGROUND_COLOR = 0xFF23211F
CARD_BACKGROUND_COLOR = 0xFF2B2927
CARD_BORDER_COLOR = 0xFF403D39
ACCENT_COLOR = 0xFF8A8777
ACCENT_HOVER_COLOR = 0xFF77746A
INFO_BACKGROUND_COLOR = 0xFF302D29
SUCCESS_BACKGROUND_COLOR = 0xFF26352D
WARNING_BACKGROUND_COLOR = 0xFF3D3425
ERROR_BACKGROUND_COLOR = 0xFF3B2828

#: Shared spacing values used by the workflow shell.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16

PAGE_TITLE_STYLE = {"font_size": 18, "color": 0xFFE6E6E6}
PAGE_DETAIL_STYLE = {"font_size": 12, "color": 0xFF9E9E9E}
NAV_BUTTON_STYLE = {
    "": {"background_color": CARD_BACKGROUND_COLOR, "border_radius": 3, "padding": 5},
    ":hovered": {"background_color": ACCENT_HOVER_COLOR},
    ":selected": {"background_color": ACCENT_COLOR},
    "Button.Label": {"color": 0xFFBDBDBD, "font_size": 13},
    "Button.Label:selected": {"color": 0xFFFFFFFF},
}
SECONDARY_BUTTON_STYLE = {
    "": {"background_color": CARD_BACKGROUND_COLOR, "border_radius": 3, "padding": 5},
    ":hovered": {"background_color": ACCENT_HOVER_COLOR},
}

STATUS_BACKGROUND_COLORS = {
    "info": INFO_BACKGROUND_COLOR,
    "success": SUCCESS_BACKGROUND_COLOR,
    "warning": WARNING_BACKGROUND_COLOR,
    "error": ERROR_BACKGROUND_COLOR,
}


def status_label(text: str = "", *, visible: bool = True, style: dict | None = None) -> ui.Label:
    """Create a word-wrapped status label with the shared muted style.

    Args:
        text: Text to display.
        visible: Visible value.
        style: Style value.

    Returns:
        The resulting value.
    """
    return ui.Label(text, word_wrap=True, visible=visible, style=style or MUTED_LABEL_STYLE)


__all__ = [
    "ACCENT_COLOR",
    "CARD_BACKGROUND_COLOR",
    "HINT_LABEL_STYLE",
    "MUTED_LABEL_STYLE",
    "NAV_BUTTON_STYLE",
    "PAGE_DETAIL_STYLE",
    "PAGE_TITLE_STYLE",
    "PANEL_BACKGROUND_COLOR",
    "SECONDARY_BUTTON_STYLE",
    "SPACE_LG",
    "SPACE_MD",
    "SPACE_SM",
    "SPACE_XS",
    "STALE_LABEL_STYLE",
    "STATUS_BACKGROUND_COLORS",
    "status_label",
]
