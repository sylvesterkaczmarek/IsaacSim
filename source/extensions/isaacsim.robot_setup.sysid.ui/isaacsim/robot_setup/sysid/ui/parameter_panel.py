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

"""Parameter panel: advanced flag checkboxes + parameter selection table."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CheckBox

from ..parameter_space import ParameterSpace
from .parameter_table import ParameterTableWidget


class ParameterPanel:
    """Build and manage the Parameters collapsible section.

    Args:
        on_flags_changed: Optional callback fired after an advanced parameter-family flag changes.
        on_selection_changed: Optional callback fired after the selected parameter set changes.
    """

    def __init__(
        self,
        on_flags_changed: Callable[[], None] | None = None,
        on_selection_changed: Callable[[], None] | None = None,
    ) -> None:
        self._on_flags_changed = on_flags_changed
        self._suppress_flag_callbacks = False

        self._include_com_offsets_cb: Optional[CheckBox] = None
        self._include_inertia_cb: Optional[CheckBox] = None
        self._include_joint_limits_cb: Optional[CheckBox] = None
        self._include_per_link_mass_cb: Optional[CheckBox] = None
        self._include_command_delay_cb: Optional[CheckBox] = None
        self._expert_bounds_cb: Optional[CheckBox] = None

        self._parameter_table = ParameterTableWidget(on_selection_changed=on_selection_changed)
        self.wrapped_ui_elements: list = []

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        """Handle build."""
        with ui.VStack(spacing=6):
            with ui.CollapsableFrame("Additional parameter families", collapsed=True):
                with ui.VStack(spacing=6):
                    self._include_com_offsets_cb = CheckBox(
                        "COM offsets",
                        default_value=False,
                        tooltip="Add per-link center-of-mass offset parameters from the USD articulation.",
                        on_click_fn=self._on_flag_changed,
                    )
                    self._include_inertia_cb = CheckBox(
                        "Inertia Log-Cholesky",
                        default_value=False,
                        tooltip="Add per-link positive-definite inertia tensor parameters.",
                        on_click_fn=self._on_flag_changed,
                    )
                    self._include_joint_limits_cb = CheckBox(
                        "Joint-limit scales",
                        default_value=False,
                        tooltip="Add lower/upper joint-limit scale parameters.",
                        on_click_fn=self._on_flag_changed,
                    )
                    self._include_per_link_mass_cb = CheckBox(
                        "Per-link mass",
                        default_value=False,
                        tooltip=(
                            "Add one mass-scale parameter per link (pairs with the differentiable "
                            "Newton bridge; finite differences scale poorly with many links)."
                        ),
                        on_click_fn=self._on_flag_changed,
                    )
                    self._include_command_delay_cb = CheckBox(
                        "Command delay",
                        default_value=False,
                        tooltip=(
                            "Add a shared command-delay parameter in seconds. Isaac Sim requires the "
                            "explicit Newton PD actuator compatibility mode."
                        ),
                        on_click_fn=self._on_flag_changed,
                    )
                    self._expert_bounds_cb = CheckBox(
                        "Edit bounds (Initial/Min/Max)",
                        default_value=False,
                        tooltip=(
                            "Show editable Initial/Min/Max columns per selected parameter. Edited values are "
                            "used by the solve even while the columns are hidden; the toggle only changes "
                            "what is displayed. Defaults are usually fine."
                        ),
                        on_click_fn=self._on_expert_bounds_changed,
                    )
            self.wrapped_ui_elements.extend(
                [
                    self._include_com_offsets_cb,
                    self._include_inertia_cb,
                    self._include_joint_limits_cb,
                    self._include_per_link_mass_cb,
                    self._include_command_delay_cb,
                    self._expert_bounds_cb,
                ]
            )
            self._parameter_table.build()
        self.wrapped_ui_elements.append(self._parameter_table)

    # ---------------------------------------------------------------- getters

    def get_advanced_flags(self) -> dict[str, bool]:
        """Get advanced flags.

        Returns:
            The resulting value.
        """
        return {
            "include_com_offsets": (
                bool(self._include_com_offsets_cb.get_value()) if self._include_com_offsets_cb else False
            ),
            "include_inertia_log_cholesky": (
                bool(self._include_inertia_cb.get_value()) if self._include_inertia_cb else False
            ),
            "include_joint_limit_scales": (
                bool(self._include_joint_limits_cb.get_value()) if self._include_joint_limits_cb else False
            ),
            "include_per_link_mass": (
                bool(self._include_per_link_mass_cb.get_value()) if self._include_per_link_mass_cb else False
            ),
            "include_command_delay": (
                bool(self._include_command_delay_cb.get_value()) if self._include_command_delay_cb else False
            ),
        }

    def is_advanced_enabled(self) -> bool:
        """Return whether advanced enabled.

        Returns:
            The resulting value.
        """
        return any(self.get_advanced_flags().values())

    def get_selected_parameter_run_specs(self) -> list:
        """Get selected parameter run specs.

        Returns:
            The resulting value.
        """
        return self._parameter_table.get_selected_parameter_run_specs()

    # ---------------------------------------------------------------- setters

    def set_advanced_flags(self, flags: dict[str, bool]) -> None:
        """Write registry flags into the checkboxes without per-flag table rebuilds.

        The caller is expected to trigger one parameter-space refresh afterwards.

        Args:
            flags: Flags value.
        """
        widgets = {
            "include_com_offsets": self._include_com_offsets_cb,
            "include_inertia_log_cholesky": self._include_inertia_cb,
            "include_joint_limit_scales": self._include_joint_limits_cb,
            "include_per_link_mass": self._include_per_link_mass_cb,
            "include_command_delay": self._include_command_delay_cb,
        }
        self._suppress_flag_callbacks = True
        try:
            for key, checkbox in widgets.items():
                if checkbox is not None and key in flags:
                    checkbox.set_value(bool(flags[key]))
        finally:
            self._suppress_flag_callbacks = False

    def select_parameters_for_recipe(self, param_type_values: set[str] | None = None) -> None:
        """Select table rows for a recipe: the given parameter types, or all rows when None.

        Args:
            param_type_values: Param type values value.
        """
        if param_type_values is None:
            self._parameter_table.set_selected_by_predicate(lambda spec: True)
            return
        self._parameter_table.set_selected_by_predicate(lambda spec: spec.param_type.value in param_type_values)

    def set_parameter_space(self, space: Optional[ParameterSpace]) -> None:
        """Set parameter space.

        Args:
            space: Space value.
        """
        self._parameter_table.set_parameter_space(space)

    def update_theta_values(self, theta: Any) -> None:
        """Update theta values.

        Args:
            theta: Parameter vector to display.
        """
        self._parameter_table.update_theta_values(theta)

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled.

        Args:
            enabled: Whether the option is enabled.
        """
        self._parameter_table.set_enabled(enabled)

    # ---------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Release resources."""
        for element in self.wrapped_ui_elements:
            if hasattr(element, "cleanup"):
                element.cleanup()
        self.wrapped_ui_elements.clear()
        self._parameter_table.cleanup()
        self._include_com_offsets_cb = None
        self._include_inertia_cb = None
        self._include_joint_limits_cb = None
        self._include_per_link_mass_cb = None
        self._include_command_delay_cb = None
        self._expert_bounds_cb = None

    # --------------------------------------------------------------- private

    def _on_flag_changed(self, _model: Any) -> None:
        if self._suppress_flag_callbacks:
            return
        if self._on_flags_changed:
            self._on_flags_changed()

    def _on_expert_bounds_changed(self, value: Any) -> None:
        # Column visibility only: row values always flow into the run spec;
        # toggling must not rebuild the parameter space.
        self._parameter_table.set_expert_bounds(bool(value))
