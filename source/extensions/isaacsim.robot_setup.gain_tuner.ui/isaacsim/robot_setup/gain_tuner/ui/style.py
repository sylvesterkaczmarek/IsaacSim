# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Style configuration module for the robot gain tuner UI extension."""

import pathlib

import omni
import omni.kit.app
import omni.ui as ui
from omni.ui import color as cl

# Resolve the installed extension root so icon/font URLs work regardless of where the
# package is deployed. `get_extension_path_by_module` returns the extension root
# (which contains `icons/` and `data/fonts/`); posix paths keep omni.ui URLs valid on Windows.
EXTENSION_FOLDER_PATH = pathlib.Path(
    omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
).as_posix()

## colors
BUTTON_BG_COLOR = 0xFF24211F
NAV_BUTTON_ACTIVE_BG = 0xFF805A3D
NAV_BUTTON_INACTIVE_BG = 0xFF24211F
NAV_BUTTON_DISABLED_BG = 0xFF343433
NAV_BUTTON_DISABLED_TEXT = 0xFF6E6E6E
FRAME_BG_COLOR = 0xFF343433
FRAME_HEAD_COLOR = 0xFF8F8F8F
STRING_FIELD_LABEL_COLOR = 0xFF8F8F8F
LABEL_COLOR = 0xFFD8D8D8
LABEL_TITLE_COLOR = 0xFFCCCCCC
DISABLED_LABEL_COLOR = 0xFF6E6E6E
UNIT_COLOR = 0xFF6E6E6E
LINE_COLOR = 0xFF8F8F8F
TRIANGLE_COLOR = 0xFF8F8F8F
TREEVIEW_BG_COLOR = 0xFF23211F
TREEVIEW_SELECTED_COLOR = 0xFF4B4A42
TREEVIEW_ITEM_COLOR = 0xFF343432
TREEVIEW_HEADER_BG_COLOR = 0xFF2D2D2D
TREEVIEW_ITEM_FONT = 14
HEADER_FONT_SIZE = 16
FONT_SIZE = 14

# --- Multi-physics backend UI colours (Figma) ---
HEADER_BG_COLOR = 0xFF454545
"""Panel header background (#454545)."""

TAB_ACTIVE_BG = 0xFF805A3D
"""Top-nav tab active state (blue #3D5A80 in omni.ui ABGR byte order)."""

TAB_INACTIVE_BG = 0xFF1F2124
"""Top-nav tab inactive state (dark, #1F2124)."""

# Segmented-toggle buttons (per-joint detail editor: Kp/Kd vs Natural Frequency) —
# match the Isaac Sim Render Settings button palette: active = medium gray,
# inactive = near-black.
SUB_TAB_ACTIVE_BG = 0xFF6E6E6E
"""Segmented-toggle active button background — matches Render-Settings 'Common' gray."""

SUB_TAB_INACTIVE_BG = 0xFF2D2D2D
"""Segmented-toggle inactive button background — matches Render-Settings dark well."""

MUTED_LABEL_COLOR = 0xFF9E9E9E
"""Secondary/muted label text (#9E9E9E) — used for 'Robot:', 'Backend:', etc."""

DIVIDER_COLOR = 0xFF767676
"""Vertical separator between header fields (#767676)."""

SOURCE_NEWTON_BG = 0xFF2A3F5F
"""Background for 'Newton' source badge (dark blue)."""

SOURCE_PHYSICS_BG = 0xFF3A3A3A
"""Background for 'Physics' source badge (dark grey)."""

SAVE_ROW_BG = 0xFF2C2C2C
"""Background for the Save Target row."""

# omni.ui uses ABGR byte order, so #RRGGBB is written as 0xFFBBGGRR.  These three
# were originally written as 0xFF plus the hex colour unreversed, which renders the
# red and blue channels swapped: the amber warning came out azure and the red FAIL
# label came out blue, near enough to :data:`INFO_COLOR` that the informational and
# warning states were indistinguishable.
WARNING_COLOR = 0xFF00A0E5
"""Amber warning text / icon colour (#E5A000)."""

INFO_COLOR = 0xFFD9963A
"""Informational text / outline colour (#3A96D9).

Deliberately not :data:`WARNING_COLOR`: it marks statements of fact the user does
not have to act on, such as one advanced joint parameter holding different values
for the PhysX and Newton backends."""

PASS_COLOR = 0xFF50AF4C
"""Green 'PASS' label colour (#4CAF50)."""

FAIL_COLOR = 0xFF3539E5
"""Red 'FAIL' label colour (#E53935)."""

# --- Options-menu radio / checkmark graphics ---
# omni.ui uses ABGR byte order, so #RRGGBB is written as 0xFFBBGGRR.
MENU_ACCENT_BLUE = 0xFFD9963A
"""Selected radio bar / checkmark colour (#3A96D9)."""

MENU_RADIO_OFF_COLOR = 0xFF8F8F8F
"""Unselected radio ring colour (#8F8F8F)."""

MENU_ITEM_LABEL_COLOR = 0xFFDCDCDC
"""Options-menu item label colour (#DCDCDC)."""

MENU_FONT_SIZE = 14
"""Options-menu item font size — matches Isaac Sim's native menu font."""

# --- Chart (joint position graph) colours (Figma node 2179:6045) ---
# omni.ui uses ABGR byte order, so #RRGGBB is written as 0xFFBBGGRR.
CHART_CARD_BG = 0xFF343432
"""Chart card background (#323434)."""

CHART_BORDER = 0xFF343434
"""Chart card / viewport border (#343434)."""

CHART_VIEWPORT_BG = 0xFF171615
"""Graph viewport background (#151617)."""

CHART_GRID_COLOR = 0xFF2E2E2E
"""Subtle grid line colour inside the viewport."""

CHART_ZERO_LINE_COLOR = 0xFF6E6E6E
"""Grid line colour for y = 0 when zero falls inside the view range (#6e6e6e).

Distinctly brighter than `CHART_GRID_COLOR` so the zero crossing reads as the
chart's datum rather than as one more grid line. This matters most on the Effort
chart, where torque swings either side of zero and the sign of a reading is the
first thing looked for.
"""

CHART_VALUE_FIELD_BG = 0xFF202020
"""Value field background (#202020)."""

CHART_AXIS_LABEL_COLOR = 0xFF8F8F8F
"""In-chart axis value labels (#8f8f8f)."""

CHART_UNIT_COLOR = 0xFF767676
"""Unit suffix text in value fields (#767676)."""

CHART_CMD_COLOR = 0xFFF6823B
"""Command Joint series / legend colour (#3B82F6, blue)."""

CHART_OBS_COLOR = 0xFFC8B333
"""Observed Joint series / legend colour (#33b3c8, cyan)."""

CHART_SLIDER_TRACK = 0xFF4A4A4A
"""Range slider track colour."""

CHART_SLIDER_HANDLE = 0xFFD8D8D8
"""Range slider handle colour."""

CHART_BTN_BG = 0xFF24211F
"""'Fit Frame View' button background (#1f2124)."""

CHART_STROKE = 0xFF101010
"""1px stroke around the chart card and checkboxes (#101010)."""

CHART_READOUT_BG = 0xFF211F1E
"""Hover readout strip background (#1e1f21).

Sits between the viewport and the card in value, so the strip reads as another of
the chart's own panels rather than as part of either. Opaque: the strip is docked
in the chart's layout and has nothing to show through to.
"""

CHART_READOUT_TEXT = 0xFFEDEDED
"""Hover readout strip text (#ededed).

Near-white rather than the muted greys used for axis labels, because the readout
is the chart's only exact figure and is read while the pointer is moving. The
strip's idle prompt uses `MUTED_LABEL_COLOR` instead, so a reading is never
mistaken for the prompt.
"""


def get_style() -> dict[str, dict[str, any]]:
    """UI style configuration for the robot gain tuner extension.

    Provides comprehensive styling for various UI components including buttons, fields, labels, treeview elements,
    and other interface widgets with custom colors, fonts, and layout properties.

    Returns:
        A style dictionary containing component styling configurations with properties like colors, fonts,
        backgrounds, and layout settings.
    """
    style = {
        "Button::reset": {"background_color": 0x0, "border_radius": 1},
        "Button::reset:disabled": {"background_color": 0x0, "color": 0x0, "border_radius": 1},
        "Button::reset:hovered": {"background_color": 0x0, "border_radius": 1},
        "Button::reset:pressed": {"background_color": 0x0, "border_radius": 1},
        "Button::cell": {"background_color": 0x0, "border_radius": 1},
        "Button::cell:disabled": {"background_color": 0x0, "color": 0x0, "border_radius": 1},
        "Button::cell:hovered": {"background_color": 0x0, "border_radius": 1},
        "Button::cell:pressed": {"background_color": 0x0, "border_radius": 1},
        "CheckBox": {"border_radius": 2, "font_size": FONT_SIZE},
        "CollapsableFrame": {"background_color": FRAME_BG_COLOR, "secondary_color": FRAME_BG_COLOR},
        "CollapsableFrame:hovered": {"background_color": FRAME_BG_COLOR, "secondary_color": FRAME_BG_COLOR},
        "ComboBox::treeview_item": {
            "color": LABEL_COLOR,
            "background_selected_color": 0x00,
            "background_color": TREEVIEW_BG_COLOR,
            "secondary_color": TREEVIEW_BG_COLOR,
            "border_radius": 0,
            "padding": 0,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "ComboBox::treeview_item:selected": {
            "color": LABEL_COLOR,
            "background_selected_color": 0x0,
            "background_color": TREEVIEW_SELECTED_COLOR,
            "secondary_color": TREEVIEW_SELECTED_COLOR,
            "border_radius": 0,
            "padding": 0,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "ComboBox::treeview_item:hovered": {
            "color": LABEL_COLOR,
            "background_selected_color": 0x0,  # TREEVIEW_BG_COLOR,
            "background_color": TREEVIEW_BG_COLOR,
            "secondary_color": TREEVIEW_BG_COLOR,
            "border_radius": 0,
            "padding": 0,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "ComboBox::treeview_item:disabled": {
            "color": DISABLED_LABEL_COLOR,
            "background_selected_color": 0x0,  # TREEVIEW_BG_COLOR,
            "background_color": TREEVIEW_BG_COLOR,
            "secondary_color": TREEVIEW_BG_COLOR,
            "border_radius": 0,
            "padding": 0,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "ComboBox::articulation_menu": {
            "color": LABEL_COLOR,
            "background_color": FRAME_BG_COLOR,
            "secondary_color": FRAME_BG_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::articulation_menu:selected": {
            "color": LABEL_COLOR,
            "background_color": FRAME_BG_COLOR,
            "secondary_color": FRAME_BG_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::articulation_menu:hovered": {
            "color": LABEL_COLOR,
            "background_color": FRAME_BG_COLOR,
            "secondary_color": FRAME_BG_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::articulation_menu:disabled": {
            "color": DISABLED_LABEL_COLOR,
            "background_color": FRAME_BG_COLOR,
            "secondary_color": FRAME_BG_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "Field::StringField": {
            "background_color": BUTTON_BG_COLOR,
            "color": STRING_FIELD_LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_It.ttf",
        },
        "Field::FloatField": {
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Field::cell": {
            "background_color": 0x00,
            "color": LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "padding": 4,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Field::cell:disabled": {
            "background_color": 0xFF343433,
            "color": DISABLED_LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "padding": 4,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Field::cell:hovered": {
            "background_color": 0x00,
            "color": LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "padding": 4,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Field::cell:pressed": {
            "background_color": 0x00,
            "color": LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "padding": 4,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Image::sort": {"image_url": f"{EXTENSION_FOLDER_PATH}/icons/sort_icon.svg", "margin": 4},
        "Image::help": {"image_url": f"{EXTENSION_FOLDER_PATH}/icons/help.svg", "margin": 0},
        "Image::copy": {"image_url": f"{EXTENSION_FOLDER_PATH}/icons/copy.svg", "margin": 0},
        "Line": {"color": LINE_COLOR},
        "Label": {
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Md.ttf",
        },
        "Label::robot_header": {
            "font_size": HEADER_FONT_SIZE,
            "color": LABEL_COLOR,
        },
        "Label::dropdown_label": {
            "font_size": HEADER_FONT_SIZE,
            "color": LABEL_COLOR,
        },
        "Label::header": {
            "color": FRAME_HEAD_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Md.ttf",
        },
        "Label::collapsable_header": {
            "color": FRAME_HEAD_COLOR,
            "font_size": HEADER_FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::treeview_item": {
            "color": LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::index": {
            "color": LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::density": {
            "color": UNIT_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Lt.ttf",
        },
        "Label::exponent": {
            "color": UNIT_COLOR,
            "font_size": 8,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Lt.ttf",
        },
        "RadioButton": {"background_color": cl.transparent, "padding": 0},
        "RadioButton:checked": {"background_color": cl.transparent, "padding": 0},
        "RadioButton:hovered": {"background_color": cl.transparent, "padding": 0},
        "RadioButton.Image": {
            "image_url": f"{EXTENSION_FOLDER_PATH}/icons/radio_off.svg",
            "color": LABEL_COLOR,
        },
        "RadioButton.Image:hovered": {
            "image_url": f"{EXTENSION_FOLDER_PATH}/icons/radio_off.svg",
            "color": LABEL_COLOR,
        },
        "RadioButton.Image:checked": {"image_url": f"{EXTENSION_FOLDER_PATH}/icons/radio_on.svg", "color": LABEL_COLOR},
        "RadioButton:pressed": {"background_color": cl.transparent},
        "Triangle": {"background_color": TRIANGLE_COLOR, "color": TRIANGLE_COLOR},
        "Triangle::mask": {"color": LABEL_COLOR},
        "Rectangle::mask": {"background_color": TREEVIEW_BG_COLOR, "border_radius": 1},
        "Rectangle::reset_invalid": {"background_color": 0xFF505050, "border_radius": 1},
        "Rectangle::reset": {"background_color": 0xFFA07D4F, "border_radius": 1},
        "Rectangle::reset:disabled": {"background_color": 0x0, "border_radius": 1},
        "Rectangle::treeview_item": {
            "background_color": TREEVIEW_BG_COLOR,
            "border_width": 1,
            "border_color": 0xFF505050,
        },
        "Rectangle::treeview_item:selected": {
            "background_color": TREEVIEW_SELECTED_COLOR,
            "border_width": 1,
            "border_color": TREEVIEW_SELECTED_COLOR,
        },
        "Rectangle::treeview_first_item": {
            "background_color": 0x0,
            "border_width": 1,
            "border_color": TREEVIEW_HEADER_BG_COLOR,
            "color": TREEVIEW_HEADER_BG_COLOR,
        },
        "Rectangle::treeview_first_item:hovered": {
            "background_color": TREEVIEW_SELECTED_COLOR,
            "border_width": 1,
            "border_color": TREEVIEW_SELECTED_COLOR,
        },
        "Rectangle::treeview_first_item:pressed": {
            "background_color": TREEVIEW_SELECTED_COLOR,
            "border_width": 1,
            "border_color": TREEVIEW_SELECTED_COLOR,
        },
        "Rectangle::treeview_first_item:selected": {
            "background_color": TREEVIEW_SELECTED_COLOR,
            "border_width": 1,
            "border_color": TREEVIEW_SELECTED_COLOR,
        },
        "Rectangle::treeview_id": {"margin": 1, "background_color": TREEVIEW_ITEM_COLOR},
        "Rectangle::treeview_id:selected": {"margin": 1, "background_color": TREEVIEW_SELECTED_COLOR},
        "ScrollingFrame": {"background_color": FRAME_BG_COLOR},
        "ScrollingFrame::Treeview": {"background_color": TREEVIEW_BG_COLOR},
        "Splitter": {"background_color": 0x0, "margin_width": 0},
        "Splitter:hovered": {"background_color": 0xFFB0703B},
        "Splitter:pressed": {"background_color": 0xFFB0703B},
        "TreeView": {
            "background_selected_color": TREEVIEW_SELECTED_COLOR,
            "background_color": TREEVIEW_BG_COLOR,
            "secondary_color": TREEVIEW_SELECTED_COLOR,
        },  # the hover color of the TreeView selected item
        "TreeView::Header": {
            "background_color": TREEVIEW_HEADER_BG_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "TreeView.Header::background": {"margin": 1, "background_color": TREEVIEW_HEADER_BG_COLOR},
        "TreeView:selected": {
            "background_color": TREEVIEW_SELECTED_COLOR
        },  # selected margin color, set to scrollingFrame background color
        # --- Multi-physics backend UI ---
        "Rectangle::header_bg": {"background_color": HEADER_BG_COLOR},
        "Rectangle::divider": {"background_color": DIVIDER_COLOR},
        "Rectangle::save_row_bg": {"background_color": SAVE_ROW_BG},
        "Rectangle::source_newton": {"background_color": SOURCE_NEWTON_BG, "border_radius": 2},
        "Rectangle::source_physics": {"background_color": SOURCE_PHYSICS_BG, "border_radius": 2},
        "Button::tab_active": {
            "background_color": TAB_ACTIVE_BG,
            "border_radius": 2,
        },
        "Button::tab_active:hovered": {"background_color": TAB_ACTIVE_BG, "border_radius": 2},
        "Button::tab_inactive": {
            "background_color": TAB_INACTIVE_BG,
            "border_radius": 2,
        },
        "Button::tab_inactive:hovered": {"background_color": 0xFF2C2F33, "border_radius": 2},
        "Button::run_test": {
            "background_color": TAB_INACTIVE_BG,
            "border_radius": 2,
        },
        "Button::run_test:hovered": {"background_color": 0xFF2C2F33, "border_radius": 2},
        "Label::muted": {
            "color": MUTED_LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Md.ttf",
        },
        "Label::header_title": {
            "color": LABEL_COLOR,
            "font_size": HEADER_FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Md.ttf",
        },
        "Label::solver_label": {
            "color": MUTED_LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::source_badge": {
            "color": LABEL_COLOR,
            "font_size": TREEVIEW_ITEM_FONT,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::save_target_value": {
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::save_target_warning": {
            "color": WARNING_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::test_info_label": {
            "color": MUTED_LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::test_info_value": {
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Label::pass": {
            "color": PASS_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Md.ttf",
        },
        "Label::fail": {
            "color": FAIL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Md.ttf",
        },
        "Field::header_field": {
            "background_color": 0x00000000,
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Field::save_target_field": {
            "background_color": 0x00000000,
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        "Field::adv_param_field": {
            "background_color": TAB_INACTIVE_BG,
            "color": LABEL_COLOR,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
        },
        # Per-row gain-source cell.  The single-source static box and the
        # multi-source ComboBox share the same background/radius so the whole
        # "Source" column reads as a uniform column of boxes; the static box just
        # has no interactive dropdown arrow.
        "Rectangle::source_box": {
            "background_color": TAB_INACTIVE_BG,
            "border_radius": 2,
        },
        "ComboBox::source_combo": {
            "color": LABEL_COLOR,
            "background_color": TAB_INACTIVE_BG,
            "secondary_color": TAB_INACTIVE_BG,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::source_combo:hovered": {
            "color": LABEL_COLOR,
            "background_color": TAB_INACTIVE_BG,
            "secondary_color": TAB_INACTIVE_BG,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::header_combo": {
            "color": LABEL_COLOR,
            "background_color": TAB_INACTIVE_BG,
            "secondary_color": TAB_INACTIVE_BG,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::header_combo:hovered": {
            "color": LABEL_COLOR,
            "background_color": TAB_INACTIVE_BG,
            "secondary_color": TAB_INACTIVE_BG,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ComboBox::header_combo:disabled": {
            "color": DISABLED_LABEL_COLOR,
            "background_color": TAB_INACTIVE_BG,
            "secondary_color": TAB_INACTIVE_BG,
            "font_size": FONT_SIZE,
            "font": f"{EXTENSION_FOLDER_PATH}/data/fonts/NVIDIASans_Rg.ttf",
            "border_radius": 2,
        },
        "ScrollingFrame::joint_name": {"background_color": TREEVIEW_BG_COLOR},
    }
    return style
