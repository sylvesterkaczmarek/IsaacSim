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

"""Recipe bar: apply a built-in workflow preset to the whole panel stack.

Recipes are a starting point, not a lock: applying one writes into the ordinary
panel widgets, which stay fully editable. Application is an explicit button (not
apply-on-select) so browsing the dropdown never clobbers user edits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import carb
import omni.ui as ui
from isaacsim.gui.components.ui_utils import dropdown_builder

from ..preset_loader import list_builtin_parameter_presets
from .styles import MUTED_LABEL_STYLE as _STATUS_STYLE

_CUSTOM_LABEL = "Custom"


class RecipeBar:
    """Recipe dropdown + Apply button rendered above the panel stack.

    Args:
        on_apply_recipe: Callback fired with the selected recipe name on Apply.
    """

    def __init__(self, on_apply_recipe: Callable[[str], None]) -> None:
        self._on_apply_recipe = on_apply_recipe
        self._recipe_names: list[str] = []
        self._selected_label: str = _CUSTOM_LABEL
        self._applied_name: str = ""
        self._recipe_model = None
        self._apply_button: Optional[ui.Button] = None
        self._status_label: Optional[ui.Label] = None

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        """Handle build."""
        try:
            self._recipe_names = list_builtin_parameter_presets()
        except Exception as exc:
            carb.log_warn(f"SysId: could not list built-in recipes: {exc}")
            self._recipe_names = []
        labels = [_CUSTOM_LABEL, *self._recipe_names]
        with ui.VStack(spacing=4, height=50):
            with ui.HStack(spacing=6, height=24):
                self._recipe_model = dropdown_builder(
                    label="Recipe",
                    default_val=0,
                    items=labels,
                    tooltip=(
                        "Built-in workflow presets for common robot classes. Apply writes a starting "
                        "configuration into the panels below; every field stays editable."
                    ),
                    on_clicked_fn=self._on_selection_changed,
                )
                self._apply_button = ui.Button("Apply", width=60, clicked_fn=self._on_apply_clicked)
            self._status_label = ui.Label(
                "Pick a recipe to pre-fill the workflow, or configure everything manually.",
                height=18,
                word_wrap=True,
                style=_STATUS_STYLE,
            )
        if self._apply_button is not None:
            self._apply_button.enabled = False

    # ----------------------------------------------------------------- state

    def set_status(self, text: str) -> None:
        """Set status.

        Args:
            text: Text to display.
        """
        if self._status_label is not None:
            self._status_label.text = text

    def mark_dirty(self) -> None:
        """Note that panel state diverged from the last applied recipe."""
        if self._applied_name and self._status_label is not None:
            self._status_label.text = f"Modified from '{self._applied_name}'."

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled.

        Args:
            enabled: Whether the option is enabled.
        """
        if self._apply_button is not None:
            self._apply_button.enabled = bool(enabled) and self._selected_label != _CUSTOM_LABEL

    def cleanup(self) -> None:
        """Release resources."""
        self._recipe_model = None
        self._apply_button = None
        self._status_label = None

    # --------------------------------------------------------------- private

    def _on_selection_changed(self, label: str) -> None:
        self._selected_label = label
        if self._apply_button is not None:
            self._apply_button.enabled = label != _CUSTOM_LABEL

    def _on_apply_clicked(self) -> None:
        if self._selected_label == _CUSTOM_LABEL:
            return
        self._applied_name = self._selected_label
        try:
            self._on_apply_recipe(self._selected_label)
        except Exception as exc:
            carb.log_warn(f"SysId: recipe apply failed: {exc}")
            self.set_status(f"Recipe apply failed: {exc}")
