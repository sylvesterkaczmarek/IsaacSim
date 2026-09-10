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

"""Simulation panel: engine/bridge selection, robot path, and cloning options."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CheckBox, IntField, StringField
from isaacsim.gui.components.ui_utils import dropdown_builder

from ..actuator_compatibility import ACTUATOR_RUNTIME_EXPLICIT, ACTUATOR_RUNTIME_IMPLICIT
from ..run_spec import (
    NEWTON_SOLVER_FEATHERSTONE_DIFF,
    NEWTON_SOLVER_MUJOCO,
    SIMULATION_ENGINE_ISAAC_SIM,
    SIMULATION_ENGINE_NEWTON,
    NewtonSimulationRunSpec,
)
from .styles import HINT_LABEL_STYLE as _HINT_LABEL_STYLE

_ENGINE_LABELS = ("Isaac Sim (PhysX)", "Newton (MuJoCo)", "Newton differentiable (Featherstone)")
_ENGINE_SELECTION_BY_LABEL = {
    _ENGINE_LABELS[0]: (SIMULATION_ENGINE_ISAAC_SIM, NEWTON_SOLVER_MUJOCO),
    _ENGINE_LABELS[1]: (SIMULATION_ENGINE_NEWTON, NEWTON_SOLVER_MUJOCO),
    _ENGINE_LABELS[2]: (SIMULATION_ENGINE_NEWTON, NEWTON_SOLVER_FEATHERSTONE_DIFF),
}
_NEWTON_DEVICE_LABELS = ("cpu", "cuda")
_NEWTON_CONTROLLER_MODES = ("pd", "pid")
_FEEDFORWARD_MODES = ("none", "gravity", "inverse_dynamics")
_EFFORT_CLAMP_MODES = ("max_effort", "dc_motor", "none")


def engine_selection_from_label(label: str) -> tuple[str, str]:
    """Map an engine dropdown label to ``(SimulationRunSpec.engine, newton.solver)``.

    Args:
        label: Label value.

    Returns:
        The resulting value.
    """
    return _ENGINE_SELECTION_BY_LABEL.get(label, (SIMULATION_ENGINE_ISAAC_SIM, NEWTON_SOLVER_MUJOCO))


def engine_label_from_selection(engine: str, newton_solver: str) -> str:
    """Inverse of :func:`engine_selection_from_label` (unknown values map to Isaac Sim).

    Args:
        engine: Engine value.
        newton_solver: Newton solver value.

    Returns:
        The resulting value.
    """
    for label, selection in _ENGINE_SELECTION_BY_LABEL.items():
        if selection == (engine, newton_solver):
            return label
    if engine == SIMULATION_ENGINE_NEWTON:
        return _ENGINE_LABELS[1]
    return _ENGINE_LABELS[0]


class SimulationPanel:
    """Builds and manages the Simulation collapsible section.

    Args:
        on_robot_path_changed: Callback fired with the new robot prim path on edit.
        on_engine_changed: Callback fired with the new engine label after selection.
        on_newton_settings_changed: Callback fired when a check-relevant Newton
            or actuator setting changes.
    """

    def __init__(
        self,
        on_robot_path_changed: Callable[[str], None] | None = None,
        on_engine_changed: Callable[[str], None] | None = None,
        on_newton_settings_changed: Callable[[], None] | None = None,
    ) -> None:
        self._on_robot_path_changed = on_robot_path_changed
        self._on_engine_changed_callback = on_engine_changed
        self._on_newton_settings_changed_callback = on_newton_settings_changed
        self._engine_label: str = _ENGINE_LABELS[0]
        self._newton_device_label: str = "cpu"
        self._controller_label: str = _NEWTON_CONTROLLER_MODES[0]
        self._feedforward_label: str = _FEEDFORWARD_MODES[0]
        self._effort_clamp_label: str = _EFFORT_CLAMP_MODES[0]

        self._engine_model = None
        self._controller_model = None
        self._feedforward_model = None
        self._effort_clamp_model = None
        self._feedforward_hint_label: Optional[ui.Label] = None
        self._newton_compatibility_hint_label: Optional[ui.Label] = None
        self._offline_stepping_cb: Optional[CheckBox] = None
        self._cuda_graph_capture_cb: Optional[CheckBox] = None
        self._robot_path_field: Optional[StringField] = None
        self._source_env_field: Optional[StringField] = None
        self._env_root_field: Optional[StringField] = None
        self._parallel_clones_cb: Optional[CheckBox] = None
        self._co_locate_clones_cb: Optional[CheckBox] = None
        self._fabric_clones_cb: Optional[CheckBox] = None
        self._explicit_pd_cb: Optional[CheckBox] = None
        self._featherstone_substeps_field: Optional[IntField] = None
        self._clone_group: Optional[ui.Frame] = None
        self._newton_group: Optional[ui.Frame] = None
        self._diff_group: Optional[ui.Frame] = None

        self.wrapped_ui_elements: list = []

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        """Handle build."""
        with ui.VStack(spacing=6):
            self._engine_model = dropdown_builder(
                label="Physics engine",
                default_val=_ENGINE_LABELS.index(self._engine_label),
                items=list(_ENGINE_LABELS),
                tooltip=(
                    "Isaac Sim (PhysX): full-scene rollouts with parallel clones. "
                    "Newton (MuJoCo): replicated-world Newton rollouts. "
                    "Newton differentiable: contact-free Featherstone rollouts with exact gradients "
                    "(pairs with the Gradient Descent optimizer)."
                ),
                on_clicked_fn=self._handle_engine_changed,
            )
            self._robot_path_field = StringField(
                "Robot prim path",
                tooltip="USD path to the robot (e.g. /World/envs/env_0/Franka)",
                default_value="/World/envs/env_0",
                on_value_changed_fn=self._on_robot_path_edited,
            )
            with ui.CollapsableFrame("Advanced simulation settings", collapsed=True):
                with ui.VStack(spacing=6):
                    self._build_clone_group()
                    self._build_newton_group()
            self.wrapped_ui_elements.extend(
                [
                    self._robot_path_field,
                    self._source_env_field,
                    self._env_root_field,
                    self._parallel_clones_cb,
                    self._co_locate_clones_cb,
                    self._fabric_clones_cb,
                    self._explicit_pd_cb,
                    self._featherstone_substeps_field,
                    self._offline_stepping_cb,
                    self._cuda_graph_capture_cb,
                ]
            )
            self._on_parallel_clones_changed(self._parallel_clones_cb.get_value())
        self._refresh_engine_group_visibility()

    def _build_clone_group(self) -> None:
        self._clone_group = ui.Frame()
        with self._clone_group:
            with ui.VStack(spacing=6):
                self._source_env_field = StringField(
                    "Clone source env",
                    tooltip="Environment prim cloned for parallel SysId (e.g. /World/envs/env_0)",
                    default_value="/World/envs/env_0",
                )
                self._env_root_field = StringField(
                    "Clone env root",
                    tooltip="Root path for GridCloner instances (e.g. /World/envs/env)",
                    default_value="/World/envs/env",
                )
                self._parallel_clones_cb = CheckBox(
                    "Parallel env clones (M+1)",
                    default_value=True,
                    tooltip=(
                        "GridCloner creates env_0..env_M for simultaneous finite-difference rollouts. "
                        "Uncheck for sequential rollouts on env_0 only (slower, sometimes more stable)."
                    ),
                    on_click_fn=self._on_parallel_clones_changed,
                )
                self._co_locate_clones_cb = CheckBox(
                    "Co-locate parallel clones",
                    default_value=True,
                    tooltip=(
                        "Checked (default): overlapping clones with PhysX env IDs - fastest for many parameters. "
                        "Unchecked: spaced grid (~2 m) with full USD copies per env - use for debugging only. "
                        "Restart Isaac Sim after toggling."
                    ),
                )
                self._fabric_clones_cb = CheckBox(
                    "Experimental Fabric clone authoring",
                    default_value=False,
                    tooltip=(
                        "Use GridCloner clone_in_fabric=True when creating parallel envs. "
                        "Can reduce clone setup overhead for many envs; leave off if clone creation behaves "
                        "unexpectedly."
                    ),
                )
                self._explicit_pd_cb = CheckBox(
                    "Explicit Newton PD actuators (compat)",
                    default_value=False,
                    tooltip=(
                        "Drive selected joints through USD-authored NewtonPDControlAPI actuators, "
                        "with optional per-candidate command delay."
                    ),
                    on_click_fn=self._on_explicit_pd_changed,
                )
                self._offline_stepping_cb = CheckBox(
                    "Offline stepping",
                    default_value=True,
                    tooltip=(
                        "Advance Isaac Sim rollout physics directly without rendering or pumping full Kit frames. "
                        "Disable only for interactive timeline debugging."
                    ),
                    on_click_fn=self._on_offline_stepping_changed,
                )

    def _build_newton_group(self) -> None:
        self._newton_group = ui.Frame(visible=False)
        with self._newton_group:
            with ui.VStack(spacing=6):
                dropdown_builder(
                    label="Newton device",
                    default_val=_NEWTON_DEVICE_LABELS.index(self._newton_device_label),
                    items=list(_NEWTON_DEVICE_LABELS),
                    tooltip="Compute device for Newton rollouts",
                    on_clicked_fn=self._on_newton_device_changed,
                )
                self._controller_model = dropdown_builder(
                    label="Controller",
                    default_val=_NEWTON_CONTROLLER_MODES.index(self._controller_label),
                    items=list(_NEWTON_CONTROLLER_MODES),
                    tooltip=(
                        "Joint-space controller used by Newton rollouts. Differentiable Featherstone requires pd."
                    ),
                    on_clicked_fn=self._on_controller_changed,
                )
                self._feedforward_model = dropdown_builder(
                    label="Controller feedforward",
                    default_val=_FEEDFORWARD_MODES.index(self._feedforward_label),
                    items=list(_FEEDFORWARD_MODES),
                    tooltip=(
                        "Nominal-model controller feedforward applied during rollouts and included in the "
                        "reported torque channel. none: raw motor-side PD. gravity: static gravity "
                        "compensation. inverse_dynamics: full nominal inverse dynamics at the measured "
                        "trajectory (computed-torque/impedance controllers such as the Franka's). Computed "
                        "from the BASELINE inertial model; fixed base required. The Check stage recommends "
                        "a mode from the measured torques."
                    ),
                    on_clicked_fn=self._on_feedforward_changed,
                )
                self._feedforward_hint_label = ui.Label("", word_wrap=True, visible=False, style=_HINT_LABEL_STYLE)
                self._effort_clamp_model = dropdown_builder(
                    label="Effort clamp",
                    default_val=_EFFORT_CLAMP_MODES.index(self._effort_clamp_label),
                    items=list(_EFFORT_CLAMP_MODES),
                    tooltip=(
                        "Actuator effort limiting: max_effort uses authored limits, dc_motor applies the "
                        "speed-dependent motor envelope, and none disables clamping."
                    ),
                    on_clicked_fn=self._on_effort_clamp_changed,
                )
                self._newton_compatibility_hint_label = ui.Label(
                    "Differentiable Featherstone requires Controller=pd and Effort clamp=max_effort or none.",
                    word_wrap=True,
                    visible=False,
                    style=_HINT_LABEL_STYLE,
                )
                self._cuda_graph_capture_cb = CheckBox(
                    "CUDA graph capture",
                    default_value=True,
                    tooltip=(
                        "Capture fixed-shape Newton rollouts into CUDA graphs and replay them, removing "
                        "per-kernel launch overhead. Effective only on a CUDA device; capture failures log "
                        "a reason and fall back to uncaptured rollouts. Performance-only: results are unchanged."
                    ),
                    on_click_fn=self._on_newton_checkbox_changed,
                )
                self._diff_group = ui.Frame(visible=False)
                with self._diff_group:
                    with ui.VStack(spacing=6):
                        self._featherstone_substeps_field = IntField(
                            "Featherstone substeps",
                            default_value=4,
                            tooltip="Symplectic-Euler substeps per command sample (differentiable bridge)",
                        )

    # ---------------------------------------------------------------- getters

    def get_robot_path(self) -> str:
        """Get robot path.

        Returns:
            The resulting value.
        """
        return self._robot_path_field.get_value().strip() if self._robot_path_field else ""

    def get_source_env_path(self) -> str:
        """Get source env path.

        Returns:
            The resulting value.
        """
        return self._source_env_field.get_value().strip() if self._source_env_field else "/World/envs/env_0"

    def get_env_paths_root(self) -> str:
        """Get env paths root.

        Returns:
            The resulting value.
        """
        return self._env_root_field.get_value().strip() if self._env_root_field else "/World/envs/env"

    def is_parallel_clones(self) -> bool:
        """Return whether parallel clones.

        Returns:
            The resulting value.
        """
        return bool(self._parallel_clones_cb.get_value()) if self._parallel_clones_cb else True

    def is_co_locate_clones(self) -> bool:
        """Return whether co locate clones.

        Returns:
            The resulting value.
        """
        return bool(self._co_locate_clones_cb.get_value()) if self._co_locate_clones_cb else True

    def is_fabric_clones(self) -> bool:
        """Return whether fabric clones.

        Returns:
            The resulting value.
        """
        return bool(self._fabric_clones_cb.get_value()) if self._fabric_clones_cb else False

    def get_engine(self) -> str:
        """Get engine.

        Returns:
            The resulting value.
        """
        return engine_selection_from_label(self._engine_label)[0]

    def get_engine_label(self) -> str:
        """Get engine label.

        Returns:
            The resulting value.
        """
        return self._engine_label

    def get_actuator_runtime(self) -> str:
        """Get the selected actuator runtime.

        Returns:
            The resulting value.
        """
        if (
            self.get_engine() == SIMULATION_ENGINE_ISAAC_SIM
            and self._explicit_pd_cb is not None
            and bool(self._explicit_pd_cb.get_value())
        ):
            return ACTUATOR_RUNTIME_EXPLICIT
        return ACTUATOR_RUNTIME_IMPLICIT

    def get_newton_spec(self) -> NewtonSimulationRunSpec:
        """Return the Newton settings implied by the current engine selection.

        Returns:
            The resulting value.
        """
        _engine, newton_solver = engine_selection_from_label(self._engine_label)
        substeps = int(self._featherstone_substeps_field.get_value()) if self._featherstone_substeps_field else 4
        return NewtonSimulationRunSpec(
            device=self._newton_device_label,
            controller=self.get_controller_mode(),
            effort_clamp=self.get_effort_clamp_mode(),
            solver=newton_solver,
            featherstone_substeps=max(1, substeps),
            # Read the ComboBox model directly so Check sees the selection even
            # if its item-changed callback has not updated the cached label yet.
            feedforward=self.get_feedforward_mode(),
            cuda_graph_capture=(bool(self._cuda_graph_capture_cb.get_value()) if self._cuda_graph_capture_cb else True),
        )

    def get_controller_mode(self) -> str:
        """Get the selected Newton controller mode.

        Returns:
            Selected controller mode.
        """
        if self._controller_model is not None:
            try:
                index = int(self._controller_model.get_item_value_model().as_int)
                if 0 <= index < len(_NEWTON_CONTROLLER_MODES):
                    self._controller_label = _NEWTON_CONTROLLER_MODES[index]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        return self._controller_label

    def get_feedforward_mode(self) -> str:
        """Get feedforward mode.

        Returns:
            The resulting value.
        """
        if self._feedforward_model is not None:
            try:
                index = int(self._feedforward_model.get_item_value_model().as_int)
                if 0 <= index < len(_FEEDFORWARD_MODES):
                    self._feedforward_label = _FEEDFORWARD_MODES[index]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # The model can be invalidated during teardown; retain the last
                # confirmed selection in that narrow window.
                pass
        return self._feedforward_label

    def get_effort_clamp_mode(self) -> str:
        """Get the selected Newton effort-clamp mode.

        Returns:
            Selected effort-clamp mode.
        """
        if self._effort_clamp_model is not None:
            try:
                index = int(self._effort_clamp_model.get_item_value_model().as_int)
                if 0 <= index < len(_EFFORT_CLAMP_MODES):
                    self._effort_clamp_label = _EFFORT_CLAMP_MODES[index]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        return self._effort_clamp_label

    def is_offline_stepping(self) -> bool:
        """Check whether rollout physics advances without full Kit updates.

        Returns:
            True when offline stepping is enabled.
        """
        return bool(self._offline_stepping_cb.get_value()) if self._offline_stepping_cb else True

    def is_differentiable_engine(self) -> bool:
        """Return whether differentiable engine.

        Returns:
            The resulting value.
        """
        return engine_selection_from_label(self._engine_label)[1] == NEWTON_SOLVER_FEATHERSTONE_DIFF

    # ---------------------------------------------------------------- setters

    def set_engine_selection(self, engine: str, newton_solver: str = NEWTON_SOLVER_MUJOCO) -> None:
        """Select the engine dropdown entry for ``(engine, newton_solver)`` (recipe application).

        Args:
            engine: Engine value.
            newton_solver: Newton solver value.
        """
        label = engine_label_from_selection(engine, newton_solver)
        if self._engine_model is not None:
            # Fires the item-changed callback (visibility refresh + forwarded callback).
            self._engine_model.get_item_value_model().set_value(_ENGINE_LABELS.index(label))
        else:
            self._engine_label = label

    def set_parallel_clones(self, enabled: bool) -> None:
        """Set parallel clones.

        Args:
            enabled: Whether the option is enabled.
        """
        if self._parallel_clones_cb is not None:
            self._parallel_clones_cb.set_value(bool(enabled))

    def set_actuator_runtime(self, runtime: str) -> None:
        """Set the actuator runtime.

        Args:
            runtime: Runtime value.
        """
        if self._explicit_pd_cb is not None:
            self._explicit_pd_cb.set_value(runtime == ACTUATOR_RUNTIME_EXPLICIT)

    def set_feedforward_mode(self, mode: str) -> None:
        """Select the feedforward dropdown entry (Check-suggestion apply / recipes).

        Args:
            mode: Mode value.
        """
        if mode not in _FEEDFORWARD_MODES:
            return
        self._feedforward_label = mode
        if self._feedforward_model is not None:
            # Fires the item-changed callback (forwarded settings-changed callback).
            self._feedforward_model.get_item_value_model().set_value(_FEEDFORWARD_MODES.index(mode))

    def set_controller_mode(self, mode: str) -> None:
        """Set the Newton controller mode.

        Args:
            mode: Controller mode to select.
        """
        if mode not in _NEWTON_CONTROLLER_MODES:
            return
        effective_mode = "pd" if self.is_differentiable_engine() else mode
        self._controller_label = effective_mode
        if self._controller_model is not None:
            value_model = self._controller_model.get_item_value_model()
            index = _NEWTON_CONTROLLER_MODES.index(effective_mode)
            if int(value_model.as_int) != index:
                value_model.set_value(index)

    def set_effort_clamp_mode(self, mode: str) -> None:
        """Set the Newton effort-clamp mode.

        Args:
            mode: Effort-clamp mode to select.
        """
        if mode not in _EFFORT_CLAMP_MODES:
            return
        effective_mode = "max_effort" if self.is_differentiable_engine() and mode == "dc_motor" else mode
        self._effort_clamp_label = effective_mode
        if self._effort_clamp_model is not None:
            value_model = self._effort_clamp_model.get_item_value_model()
            index = _EFFORT_CLAMP_MODES.index(effective_mode)
            if int(value_model.as_int) != index:
                value_model.set_value(index)

    def set_offline_stepping(self, enabled: bool) -> None:
        """Set offline rollout stepping.

        Args:
            enabled: Whether to enable offline stepping.
        """
        if self._offline_stepping_cb is not None:
            self._offline_stepping_cb.set_value(bool(enabled))

    def set_cuda_graph_capture(self, enabled: bool) -> None:
        """Set cuda graph capture.

        Args:
            enabled: Whether the option is enabled.
        """
        if self._cuda_graph_capture_cb is not None:
            self._cuda_graph_capture_cb.set_value(bool(enabled))

    def set_feedforward_hint(self, text: str | None) -> None:
        """Show or clear the Check-stage feedforward recommendation under the dropdown.

        Args:
            text: Text to display.
        """
        if self._feedforward_hint_label is None:
            return
        self._feedforward_hint_label.text = text or ""
        self._feedforward_hint_label.visible = bool(text)

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled.

        Args:
            enabled: Whether the option is enabled.
        """
        for element in self.wrapped_ui_elements:
            if element is not None:
                element.enabled = enabled
        if enabled and self._parallel_clones_cb is not None:
            self._on_parallel_clones_changed(self._parallel_clones_cb.get_value())

    # ---------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Release resources."""
        for element in self.wrapped_ui_elements:
            if element is not None and hasattr(element, "cleanup"):
                element.cleanup()
        self.wrapped_ui_elements.clear()
        self._engine_model = None
        self._robot_path_field = None
        self._source_env_field = None
        self._env_root_field = None
        self._parallel_clones_cb = None
        self._co_locate_clones_cb = None
        self._fabric_clones_cb = None
        self._explicit_pd_cb = None
        self._featherstone_substeps_field = None
        self._offline_stepping_cb = None
        self._cuda_graph_capture_cb = None
        self._controller_model = None
        self._feedforward_model = None
        self._effort_clamp_model = None
        self._feedforward_hint_label = None
        self._newton_compatibility_hint_label = None
        self._clone_group = None
        self._newton_group = None
        self._diff_group = None

    # --------------------------------------------------------------- private

    def _on_robot_path_edited(self, model: Any) -> None:
        if self._on_robot_path_changed:
            self._on_robot_path_changed(self.get_robot_path())

    def _on_parallel_clones_changed(self, enabled: bool) -> None:
        if self._co_locate_clones_cb is not None:
            self._co_locate_clones_cb.enabled = bool(enabled)
        if self._fabric_clones_cb is not None:
            self._fabric_clones_cb.enabled = bool(enabled)

    def _handle_engine_changed(self, label: str) -> None:
        self._engine_label = label
        self.set_controller_mode(self._controller_label)
        self.set_effort_clamp_mode(self._effort_clamp_label)
        self._refresh_engine_group_visibility()
        if self._on_engine_changed_callback is not None:
            self._on_engine_changed_callback(label)

    def _on_newton_device_changed(self, label: str) -> None:
        self._newton_device_label = label

    def _on_controller_changed(self, label: str) -> None:
        effective_mode = "pd" if self.is_differentiable_engine() else label
        if effective_mode != label:
            self.set_controller_mode(effective_mode)
            return
        self._controller_label = effective_mode
        self._notify_newton_settings_changed()

    def _on_feedforward_changed(self, label: str) -> None:
        self._feedforward_label = label
        self._notify_newton_settings_changed()

    def _on_effort_clamp_changed(self, label: str) -> None:
        effective_mode = "max_effort" if self.is_differentiable_engine() and label == "dc_motor" else label
        if effective_mode != label:
            self.set_effort_clamp_mode(effective_mode)
            return
        self._effort_clamp_label = effective_mode
        self._notify_newton_settings_changed()

    def _on_newton_checkbox_changed(self, _value: Any) -> None:
        self._notify_newton_settings_changed()

    def _on_offline_stepping_changed(self, _value: Any) -> None:
        self._notify_newton_settings_changed()

    def _on_explicit_pd_changed(self, _value: Any) -> None:
        self._notify_newton_settings_changed()

    def _notify_newton_settings_changed(self) -> None:
        if self._on_newton_settings_changed_callback is not None:
            self._on_newton_settings_changed_callback()

    def _refresh_engine_group_visibility(self) -> None:
        engine, newton_solver = engine_selection_from_label(self._engine_label)
        if self._clone_group is not None:
            self._clone_group.visible = engine == SIMULATION_ENGINE_ISAAC_SIM
        if self._newton_group is not None:
            self._newton_group.visible = engine == SIMULATION_ENGINE_NEWTON
        if self._diff_group is not None:
            self._diff_group.visible = newton_solver == NEWTON_SOLVER_FEATHERSTONE_DIFF
        if self._newton_compatibility_hint_label is not None:
            self._newton_compatibility_hint_label.visible = newton_solver == NEWTON_SOLVER_FEATHERSTONE_DIFF
