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

"""Visual Cues panel — drop lines for 2D teleoperation depth perception."""

from __future__ import annotations

import carb.settings
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.ui as ui
import omni.usd
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.replicator.teleop import (
    TeleopManager,
    VisualCueSideProfile,
    VisualCuesManager,
    VisualCuesProfile,
)
from pxr import Sdf, UsdGeom

from .ui_helpers import (
    CLR_DIM,
    CLR_GREEN,
    CLR_RED,
    CLR_YELLOW,
    INDENT,
    ROW_HEIGHT,
    ROW_SPACING,
    SECTION_SPACING,
    STATUS_HEIGHT,
    build_prim_path_row,
)
from .ui_helpers import set_status as _set_status_base

_PANEL_NAME = "Visual Cues"
_LOG_NAMESPACE = "VisualCues"
_SETTINGS_PREFIX = "/persistent/exts/isaacsim.replicator.teleop/visual_cues"


def set_status(
    label: ui.Label | None,
    text: str,
    color: int = CLR_DIM,
    emit_terminal: bool = False,
    side: str | None = None,
) -> None:
    """Set the status label text and color for this panel."""
    _set_status_base(label, text, color, source=_LOG_NAMESPACE, emit_terminal=emit_terminal, side=side)


class VisualCuesPanel:
    """Drop-line visual cues with per-side controller auto-link or prim override."""

    def __init__(
        self,
        visual_cues_manager: VisualCuesManager,
        teleop_manager: TeleopManager,
        collapsed_states: dict,
    ) -> None:
        self._vcm = visual_cues_manager
        self._tm = teleop_manager
        self._collapsed = collapsed_states
        self._settings = carb.settings.get_settings()

        self._settings.set_default_float(f"{_SETTINGS_PREFIX}/reference_z", VisualCuesManager.DEFAULT_REFERENCE_Z)
        self._settings.set_default_float(f"{_SETTINGS_PREFIX}/opacity", VisualCuesManager.DEFAULT_OPACITY)
        self._settings.set_default_float(f"{_SETTINGS_PREFIX}/size", VisualCuesManager.DEFAULT_SIZE)
        for side in ("left", "right"):
            self._settings.set_default_string(f"{_SETTINGS_PREFIX}/{side}/prim_path", "")
            self._settings.set_default_bool(f"{_SETTINGS_PREFIX}/{side}/enabled", False)

        self._reference_z_field: ui.FloatDrag | None = None
        self._opacity_slider: ui.FloatSlider | None = None
        self._size_field: ui.FloatDrag | None = None
        self._widgets: dict[str, dict] = {"left": {}, "right": {}}
        self._desired_enabled: dict[str, bool] = {"left": False, "right": False}

        self._vcm.set_pose_provider(self._tm.get_input_world_position)

    def build(self) -> None:
        """Build the visual cues panel UI."""
        frame = ui.CollapsableFrame(
            _PANEL_NAME,
            height=0,
            collapsed=self._collapsed.get(_PANEL_NAME, True),
            style=get_style(),
        )
        with frame:
            frame.set_collapsed_changed_fn(lambda c, k=_PANEL_NAME: self._collapsed.__setitem__(k, c))
            with ui.VStack(spacing=0):
                with ui.VStack(spacing=SECTION_SPACING):
                    with ui.HStack(spacing=ROW_SPACING, height=ROW_HEIGHT):
                        ui.Spacer(width=INDENT)
                        ui.Label(
                            "Reference Z:",
                            width=75,
                            tooltip="World Z height where cue cylinders end.",
                        )
                        self._reference_z_field = ui.FloatDrag(
                            min=-1000.0,
                            max=1000.0,
                            step=0.01,
                            width=80,
                            tooltip="World Z reference plane for drop lines.",
                        )
                        self._reference_z_field.model.set_value(self._load_float("reference_z"))
                        self._reference_z_field.model.add_value_changed_fn(
                            lambda m: self._on_reference_z_changed(m.get_value_as_float())
                        )

                    with ui.HStack(spacing=ROW_SPACING, height=ROW_HEIGHT):
                        ui.Spacer(width=INDENT)
                        ui.Label("Opacity:", width=75, tooltip="Transparency of drop-line geometry.")
                        self._opacity_slider = ui.FloatSlider(min=0.05, max=1.0, width=ui.Fraction(1))
                        self._opacity_slider.model.set_value(self._load_float("opacity"))
                        self._opacity_slider.model.add_value_changed_fn(
                            lambda m: self._on_opacity_changed(m.get_value_as_float())
                        )

                    with ui.HStack(spacing=ROW_SPACING, height=ROW_HEIGHT):
                        ui.Spacer(width=INDENT)
                        ui.Label("Size:", width=75, tooltip="Scale used to derive the cue-cylinder thickness.")
                        self._size_field = ui.FloatDrag(
                            min=VisualCuesManager.MIN_SIZE,
                            max=1.0,
                            step=0.005,
                            width=80,
                            tooltip="Uniform visual scale for drop-line geometry.",
                        )
                        self._size_field.model.set_value(self._load_float("size"))
                        self._size_field.model.add_value_changed_fn(
                            lambda m: self._on_size_changed(m.get_value_as_float())
                        )

                self._build_side("left")
                self._build_side("right")

        self._apply_global_settings_from_ui()
        for side in ("left", "right"):
            self._vcm.set_override_prim_path(side, self._load_str(side, "prim_path"))
            if self._desired_enabled[side]:
                self._sync_side_visibility(side)

    def _build_side(self, side: str) -> None:
        w = self._widgets[side]
        side_key = f"{_PANEL_NAME}:{side}"
        with ui.CollapsableFrame(
            side.capitalize(),
            height=0,
            collapsed=self._collapsed.get(side_key, True),
            style=get_style(),
        ) as side_frame:
            side_frame.set_collapsed_changed_fn(lambda c, k=side_key: self._collapsed.__setitem__(k, c))
            with ui.VStack(spacing=SECTION_SPACING):
                path_btns: dict = {}
                w["path"] = build_prim_path_row(
                    "Tracked Prim:",
                    tooltip=(
                        "Optional prim override. Leave empty to follow VR controller input "
                        "(or debug marker input when Debug Mode is active)."
                    ),
                    on_apply_clicked=lambda s=side: self._on_override_apply(s),
                    apply_label="Set",
                    apply_tooltip="Validate and apply prim override (empty field restores auto-link).",
                    buttons_out=path_btns,
                )
                w["path"].model.set_value(self._load_str(side, "prim_path"))
                w["set_btn"] = path_btns.get("apply")
                w["plus_btn"] = path_btns.get("plus")
                w["del_btn"] = path_btns.get("delete")

                with ui.HStack(spacing=ROW_SPACING, height=ROW_HEIGHT):
                    ui.Spacer(width=INDENT)
                    w["show_btn"] = ui.Button(
                        "Show",
                        width=55,
                        clicked_fn=lambda s=side: self._on_show(s),
                        tooltip="Create the drop-line cylinder for this side.",
                    )
                    w["hide_btn"] = ui.Button(
                        "Hide",
                        width=55,
                        clicked_fn=lambda s=side: self._on_hide(s),
                        tooltip="Remove the drop line for this side.",
                        enabled=False,
                    )

                with ui.HStack(spacing=ROW_SPACING, height=STATUS_HEIGHT):
                    ui.Spacer(width=INDENT)
                    w["status"] = ui.Label("", style={"color": CLR_DIM}, word_wrap=True)

        self._desired_enabled[side] = bool(self._settings.get_as_bool(f"{_SETTINGS_PREFIX}/{side}/enabled"))
        self._refresh_side_buttons(side)

    def _save_float(self, key: str, value: float) -> None:
        self._settings.set_float(f"{_SETTINGS_PREFIX}/{key}", float(value))

    def _load_float(self, key: str) -> float:
        return float(self._settings.get_as_float(f"{_SETTINGS_PREFIX}/{key}"))

    def _save_str(self, side: str, key: str, value: str) -> None:
        self._settings.set_string(f"{_SETTINGS_PREFIX}/{side}/{key}", value)

    def _load_str(self, side: str, key: str) -> str:
        return self._settings.get_as_string(f"{_SETTINGS_PREFIX}/{side}/{key}") or ""

    def _save_bool(self, side: str, enabled: bool) -> None:
        self._settings.set_bool(f"{_SETTINGS_PREFIX}/{side}/enabled", bool(enabled))

    def _apply_global_settings_from_ui(self) -> None:
        if self._reference_z_field:
            self._vcm.set_reference_z(self._reference_z_field.model.get_value_as_float())
        if self._opacity_slider:
            self._vcm.set_opacity(self._opacity_slider.model.get_value_as_float())
        if self._size_field:
            self._vcm.set_size(self._size_field.model.get_value_as_float())

    def _on_reference_z_changed(self, value: float) -> None:
        self._save_float("reference_z", value)
        self._vcm.set_reference_z(value)

    def _on_opacity_changed(self, value: float) -> None:
        self._save_float("opacity", value)
        self._vcm.set_opacity(value)

    def _on_size_changed(self, value: float) -> None:
        self._save_float("size", value)
        self._vcm.set_size(value)

    def _on_override_apply(self, side: str) -> bool:
        """Validate and apply a side's manual override.

        Args:
            side: Controller side whose override should be applied.

        Returns:
            Whether the current field value was accepted.
        """
        w = self._widgets[side]
        path = w["path"].model.get_value_as_string().strip()
        if path and not self._validate_override_path(path):
            set_status(w["status"], "Invalid prim path", CLR_RED, side=side)
            return False
        self._save_str(side, "prim_path", path)
        self._vcm.set_override_prim_path(side, path)
        if path:
            set_status(w["status"], f"Override — {path}", CLR_GREEN, emit_terminal=True, side=side)
        else:
            set_status(w["status"], "Auto — controller input", CLR_GREEN, emit_terminal=True, side=side)
        return True

    def _on_show(self, side: str) -> None:
        if not self._on_override_apply(side):
            return
        ok, message = self._vcm.show_side(side)
        w = self._widgets[side]
        if ok:
            self._desired_enabled[side] = True
            self._save_bool(side, True)
            self._refresh_side_buttons(side)
            set_status(w["status"], message, CLR_GREEN, emit_terminal=True, side=side)
        else:
            set_status(w["status"], message, CLR_RED, emit_terminal=True, side=side)

    def _on_hide(self, side: str) -> None:
        self._vcm.hide_side(side)
        self._desired_enabled[side] = False
        self._save_bool(side, False)
        self._refresh_side_buttons(side)
        set_status(self._widgets[side]["status"], "Hidden", CLR_DIM, emit_terminal=True, side=side)

    def _refresh_side_buttons(self, side: str) -> None:
        w = self._widgets[side]
        active = self._vcm.is_side_active(side)
        if w.get("show_btn"):
            w["show_btn"].enabled = not active
        if w.get("hide_btn"):
            w["hide_btn"].enabled = active

    def _sync_side_visibility(self, side: str) -> None:
        if self._desired_enabled[side]:
            ok, message = self._vcm.show_side(side)
            color = CLR_GREEN if ok else CLR_YELLOW
            set_status(self._widgets[side]["status"], message, color, side=side)
        else:
            self._vcm.hide_side(side)
        self._refresh_side_buttons(side)

    @staticmethod
    def _validate_override_path(path: str) -> bool:
        if not path or not Sdf.Path.IsValidPathString(path):
            return False
        if not stage_utils.is_stage_set() and omni.usd.get_context().get_stage() is None:
            return False
        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(path)
        return bool(prim and prim.IsValid() and prim.IsA(UsdGeom.Xformable))

    def collect_profile(self) -> VisualCuesProfile:
        """Collect panel state into a profile section."""
        left_path = self._widgets["left"]["path"].model.get_value_as_string().strip()
        right_path = self._widgets["right"]["path"].model.get_value_as_string().strip()
        return VisualCuesProfile(
            reference_z=self._reference_z_field.model.get_value_as_float() if self._reference_z_field else 0.0,
            opacity=(
                self._opacity_slider.model.get_value_as_float()
                if self._opacity_slider
                else VisualCuesManager.DEFAULT_OPACITY
            ),
            size=(self._size_field.model.get_value_as_float() if self._size_field else VisualCuesManager.DEFAULT_SIZE),
            left=VisualCueSideProfile(enabled=self._desired_enabled["left"], prim_path=left_path),
            right=VisualCueSideProfile(enabled=self._desired_enabled["right"], prim_path=right_path),
        )

    def apply_profile(self, profile: VisualCuesProfile, *, resolve_stage: bool = True) -> None:
        """Apply a profile section to this panel."""
        del resolve_stage
        if self._reference_z_field:
            self._reference_z_field.model.set_value(profile.reference_z)
        if self._opacity_slider:
            self._opacity_slider.model.set_value(profile.opacity)
        if self._size_field:
            self._size_field.model.set_value(profile.size)
        self._apply_global_settings_from_ui()
        self._save_float("reference_z", profile.reference_z)
        self._save_float("opacity", profile.opacity)
        self._save_float("size", profile.size)

        for side, side_profile in (("left", profile.left), ("right", profile.right)):
            w = self._widgets[side]
            w["path"].model.set_value(side_profile.prim_path)
            self._save_str(side, "prim_path", side_profile.prim_path)
            self._vcm.set_override_prim_path(side, side_profile.prim_path)
            self._desired_enabled[side] = side_profile.enabled
            self._save_bool(side, side_profile.enabled)
            self._sync_side_visibility(side)

    def on_stage_closed(self) -> None:
        """Clear stage-bound cue geometry while preserving configured settings."""
        self._vcm.clear_cached_state()
        for side in ("left", "right"):
            self._refresh_side_buttons(side)
            set_status(self._widgets[side]["status"], "", CLR_DIM, side=side)

    def reset_ui(self) -> None:
        """Reset runtime visibility without clearing user settings."""
        for side in ("left", "right"):
            self._vcm.hide_side(side)
            self._refresh_side_buttons(side)
