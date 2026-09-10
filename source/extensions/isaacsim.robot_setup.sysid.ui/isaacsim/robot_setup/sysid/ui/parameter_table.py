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

"""Dynamic per-DOF parameter grid for SysId optimization settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import omni.ui as ui
from isaacsim.gui.components.ui_utils import get_style

from ..parameter_space import ParameterSpace
from ..parameter_types import (
    GLOBAL_DOF_INDEX,
    GLOBAL_LINK_INDEX,
    SysIdParameterEntry,
    SysIdParameterSpec,
    SysIdParameterType,
)
from ..run_spec import ParameterRunSpec

_COL_OPT = 36
_COL_NAME = ui.Fraction(1)
_COL_NUM = 72
_ROW_HEIGHT = 24
_SCROLL_MAX_LINES = 14

_HEADER_STYLE = {"font_size": 12, "color": 0xFFAAAAAA}
_LABEL_STYLE = {"font_size": 13}
_MUTED_STYLE = {"font_size": 12, "color": 0xFF888888}
_SECTION_STYLE = {"font_size": 13, "color": 0xFFCCCCCC}

_CATEGORY_LABELS = {
    "sysid": "Physical (SysID)",
    "calibration": "Solver calibration",
}


@dataclass
class ParameterRowState:
    """Runtime values for one parameter table row."""

    spec: SysIdParameterSpec
    optimize: bool
    initial: float
    min_value: float
    max_value: float


class ParameterTableWidget:
    """Grouped per-joint parameter table with scroll and bulk optimize toggles.

    Args:
        on_selection_changed: Optional callback fired after any change to which
            rows are selected for optimization (row, group, select-all, recipe).
    """

    def __init__(self, on_selection_changed: Any = None) -> None:
        self._on_selection_changed = on_selection_changed
        self._frame: Optional[ui.Frame] = None
        self._num_joints = 0
        self._expert_bounds = False
        self._rows: list[ParameterRowState] = []
        self._opt_checkboxes: list[ui.CheckBox] = []
        self._opt_models: list[ui.SimpleBoolModel] = []
        self._field_widgets: list[dict] = []
        self._group_opt_models: dict[str, ui.SimpleBoolModel] = {}
        self._joint_section_frames: dict[int, ui.CollapsableFrame] = {}
        self._link_section_frames: dict[int, ui.CollapsableFrame] = {}
        self._global_section_frame: Optional[ui.CollapsableFrame] = None
        self._section_collapsed: dict[str, bool] = {}
        self._parameter_space: Optional[ParameterSpace] = None
        # Persistent record of the user's selection (optimize flag, bounds, initial)
        # keyed by parameter entry. It survives transient rebuilds where the space is
        # briefly ``None`` -- e.g. telemetry reloading after a command-alignment shift,
        # which would otherwise clear ``_rows`` and drop the selection before the
        # reloaded space restores it.
        self._selection_memo: dict[SysIdParameterEntry, ParameterRowState] = {}
        # Structure signature of the space the memo was captured under. The memo is
        # only reused when the incoming space matches it (a same-robot reload, e.g. a
        # command-alignment shift); a structurally different space -- switching robots
        # -- clears it so the new robot does not inherit the previous selection/bounds.
        self._memo_signature: Optional[tuple] = None
        self._bulk_family: Optional[SysIdParameterType] = None
        self._bulk_initial_model = None
        self._bulk_min_model = None
        self._bulk_max_model = None
        self._bulk_bounds_status_label: Optional[ui.Label] = None

    @staticmethod
    def _group_key_for_entry(entry: SysIdParameterEntry) -> str:
        if entry.dof_index >= 0:
            return f"joint:{entry.dof_index}"
        if entry.link_index >= 0:
            return f"link:{entry.link_index}"
        return "global"

    def _rows_for_group(self, group_key: str) -> list[int]:
        return [idx for idx, row in enumerate(self._rows) if self._group_key_for_entry(row.spec.entry) == group_key]

    def set_num_joints(self, num_joints: int) -> None:
        """Rebuild row state for ``4*N + 1`` parameters (joint basics + link mass).

        Args:
            num_joints: Num joints value.
        """
        self._remember_current_selection()
        self._num_joints = max(0, int(num_joints))
        space = ParameterSpace.for_robot(self._num_joints) if self._num_joints >= 1 else None
        self._reconcile_memo_for_space(space)
        previous_rows = self._previous_rows_with_memo()
        self._parameter_space = space
        specs = space.specs if space is not None else ()
        self._rows = [self._row_from_spec(spec, previous_rows) for spec in specs]
        self._remember_current_selection()
        if self._frame is not None:
            self._capture_collapse_states()
            self._frame.rebuild()
        self._notify_selection_changed()

    def set_parameter_space(self, space: ParameterSpace | None) -> None:
        """Rebuild row state from a canonical :class:`ParameterSpace`.

        Args:
            space: Space value.
        """
        # Capture the live selection into the memo before it can be cleared: the
        # user's checkbox/bound edits live only in ``_rows``, and a None rebuild
        # empties them before the reloaded space restores from the memo.
        self._remember_current_selection()
        self._reconcile_memo_for_space(space)
        previous_rows = self._previous_rows_with_memo()
        self._parameter_space = space
        if space is None:
            self._num_joints = 0
            self._rows = []
        else:
            dof_indices = [spec.entry.dof_index for spec in space.specs if spec.entry.dof_index >= 0]
            inferred_joints = max(dof_indices) + 1 if dof_indices else 0
            self._num_joints = int(getattr(space, "_num_joints", None) or inferred_joints)
            self._rows = [self._row_from_spec(spec, previous_rows) for spec in space.specs]
        self._remember_current_selection()
        if self._frame is not None:
            self._capture_collapse_states()
            self._frame.rebuild()
        self._notify_selection_changed()

    def _row_state_by_entry(self) -> dict[SysIdParameterEntry, ParameterRowState]:
        return {row.spec.entry: row for row in self._rows}

    def _previous_rows_with_memo(self) -> dict[SysIdParameterEntry, ParameterRowState]:
        """Row state to restore from: live rows first, backfilled by the memo.

        The memo carries selection across a transient ``set_parameter_space(None)``
        (which empties ``_rows``) so a reload-driven rebuild keeps the user's choices.

        Returns:
            Row state keyed by entry, live rows overriding memoized ones.
        """
        merged = dict(self._selection_memo)
        merged.update(self._row_state_by_entry())
        return merged

    def _remember_current_selection(self) -> None:
        """Update the persistent memo from the current rows (no-op while empty)."""
        for row in self._rows:
            self._selection_memo[row.spec.entry] = row

    @staticmethod
    def _space_signature(space: ParameterSpace | None) -> Optional[tuple]:
        """Structure key of a space (its ordered entry identities), or None.

        Args:
            space: The parameter space to fingerprint, or ``None``.

        Returns:
            An ordered tuple of entry identities, or ``None`` when ``space`` is ``None``.
        """
        if space is None:
            return None
        return tuple(
            (spec.entry.param_type.value, spec.entry.dof_index, spec.entry.link_index, spec.entry.component_index)
            for spec in space.specs
        )

    def _reconcile_memo_for_space(self, space: ParameterSpace | None) -> None:
        """Drop the memo when the incoming space is a different structure.

        A ``None`` space is the transient clear during a reload and must not touch
        the memo (that is what the memo bridges). A non-None space with a signature
        different from the one the memo was captured under indicates a robot switch,
        so the memo is cleared to avoid bleeding the previous selection/bounds.

        Args:
            space: The incoming parameter space, or ``None`` for a transient clear.
        """
        if space is None:
            return
        signature = self._space_signature(space)
        if self._memo_signature is not None and signature != self._memo_signature:
            self._selection_memo.clear()
        self._memo_signature = signature

    def _row_from_spec(
        self,
        spec: SysIdParameterSpec,
        previous_rows: dict[SysIdParameterEntry, ParameterRowState],
    ) -> ParameterRowState:
        previous = previous_rows.get(spec.entry)
        if previous is None:
            initial = spec.default_initial
            # Log-Cholesky inertia entries must start from the authored baseline:
            # the registry default of 0 decodes to a ~identity inertia tensor.
            if spec.entry.param_type.value == "link_inertia_log_cholesky" and self._parameter_space is not None:
                baseline = self._parameter_space.baseline.link_inertia_lc.get(spec.entry.link_index)
                if baseline is not None:
                    initial = float(min(max(baseline[spec.entry.component_index], spec.default_min), spec.default_max))
            return ParameterRowState(
                spec=spec,
                optimize=False,
                initial=initial,
                min_value=spec.default_min,
                max_value=spec.default_max,
            )
        return ParameterRowState(
            spec=spec,
            optimize=previous.optimize,
            initial=previous.initial,
            min_value=previous.min_value,
            max_value=previous.max_value,
        )

    def build(self) -> ui.Frame:
        """Create the parameter grid inside a rebuildable frame.

        Returns:
            The resulting value.
        """
        self._frame = ui.Frame()
        self._frame.set_build_fn(self._build_table)
        return self._frame

    def _build_table(self) -> None:
        self._opt_checkboxes.clear()
        self._opt_models.clear()
        self._field_widgets.clear()
        self._group_opt_models.clear()
        self._joint_section_frames.clear()
        self._link_section_frames.clear()
        self._global_section_frame = None
        # Rebuilt only in expert-bounds mode; drop the stale reference so a Family
        # click after Edit-bounds is toggled off does not write to a destroyed widget.
        self._bulk_bounds_status_label = None
        self._field_widgets = [{} for _ in self._rows]
        self._opt_models = [ui.SimpleBoolModel() for _ in self._rows]

        if not self._rows:
            ui.Label(
                "Load a telemetry CSV to configure per-joint parameters.",
                style=_MUTED_STYLE,
                word_wrap=True,
            )
            return

        with ui.VStack(spacing=6, style=get_style()):
            self._build_toolbar()
            self._build_column_header()
            scroll_lines = min(_SCROLL_MAX_LINES, max(6, 2 + len(self._rows)))
            with ui.ScrollingFrame(
                height=ui.Length(_ROW_HEIGHT * scroll_lines + 4),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                style_type_name_override="TreeView",
            ):
                with ui.VStack(spacing=4):
                    for dof_index in range(self._num_joints):
                        row_indices = [
                            idx for idx, row in enumerate(self._rows) if row.spec.entry.dof_index == dof_index
                        ]
                        if row_indices:
                            self._build_joint_section(dof_index, row_indices)
                    for link_index in self._link_indices():
                        self._build_link_section(link_index, self._rows_for_link(link_index))
                    global_rows = self._global_row_indices()
                    if global_rows:
                        self._build_global_section(global_rows)

    def _build_toolbar(self) -> None:
        with ui.HStack(height=24):
            ui.Label(
                "Parameters to identify",
                style=_SECTION_STYLE,
                tooltip=(
                    "Expand a joint to set friction, stiffness, damping, and armature. "
                    "Check Opt to include in optimization."
                ),
            )
            ui.Spacer()
            ui.Button("Select all", width=90, clicked_fn=lambda: self._set_all_optimize(True))
            ui.Button("Clear all", width=90, clicked_fn=lambda: self._set_all_optimize(False))
        with ui.HStack(spacing=4, height=24):
            ui.Label("Families", width=58, style=_MUTED_STYLE)
            for label, parameter_type in (
                ("Friction", SysIdParameterType.JOINT_FRICTION),
                ("Stiffness", SysIdParameterType.JOINT_STIFFNESS),
                ("Damping", SysIdParameterType.JOINT_DAMPING),
                ("Armature", SysIdParameterType.JOINT_ARMATURE),
            ):
                ui.Button(
                    label,
                    width=78,
                    height=22,
                    tooltip=f"Select {label.lower()} for every joint and target it for bulk bounds",
                    clicked_fn=lambda pt=parameter_type: self._select_parameter_family(pt),
                )
            ui.Spacer()
        if self._expert_bounds:
            self._build_bulk_bounds_toolbar()

    def _build_bulk_bounds_toolbar(self) -> None:
        if self._bulk_initial_model is None:
            self._bulk_initial_model = ui.SimpleFloatModel(1.0)
            self._bulk_min_model = ui.SimpleFloatModel(0.0)
            self._bulk_max_model = ui.SimpleFloatModel(10.0)
        with ui.HStack(spacing=4, height=26):
            ui.Label("Family bounds", width=82, style=_MUTED_STYLE)
            for label, model in (
                ("Initial", self._bulk_initial_model),
                ("Min", self._bulk_min_model),
                ("Max", self._bulk_max_model),
            ):
                ui.Label(label, width=34, style=_MUTED_STYLE)
                ui.FloatField(model=model, width=68, height=22, tooltip=f"Exact {label.lower()} value")
            ui.Button(
                "Apply",
                width=58,
                height=22,
                tooltip="Apply these exact bounds to selected rows in the last chosen family",
                clicked_fn=self._apply_bulk_family_bounds,
            )
        target = self._bulk_family.value.replace("joint_", "") if self._bulk_family is not None else "none"
        self._bulk_bounds_status_label = ui.Label(
            f"Bulk target: {target}",
            height=18,
            style=_MUTED_STYLE,
            tooltip="Choose a family above, enter exact values, then click Apply.",
        )

    def _build_column_header(self) -> None:
        with ui.ZStack(height=_ROW_HEIGHT, style_type_name_override="TreeView"):
            ui.Rectangle(name="Header", style_type_name_override="TreeView")
            with ui.HStack():
                self._header_cell("Opt", _COL_OPT, ui.Alignment.CENTER)
                self._header_cell("Parameter", _COL_NAME, ui.Alignment.LEFT)
                if self._expert_bounds:
                    self._header_cell("Initial", _COL_NUM, ui.Alignment.CENTER)
                    self._header_cell("Min", _COL_NUM, ui.Alignment.CENTER)
                    self._header_cell("Max", _COL_NUM, ui.Alignment.CENTER)
                else:
                    self._header_cell("Value", _COL_NUM, ui.Alignment.CENTER)

    def _header_cell(self, text: str, width: Any, alignment: ui.Alignment) -> None:
        with ui.Frame(width=width):
            with ui.VStack():
                ui.Spacer()
                ui.Label(text, alignment=alignment, style=_HEADER_STYLE, height=0)
                ui.Spacer()

    def _joint_section_title(self, dof_index: int, row_indices: list[int]) -> str:
        if self._num_joints == 1:
            base = "Joint"
        else:
            base = f"Joint {dof_index + 1}"
        selected = sum(1 for i in row_indices if self._rows[i].optimize)
        return f"{base}  ({selected}/{len(row_indices)} selected)"

    def _build_joint_section(self, dof_index: int, row_indices: list[int]) -> None:
        section_key = f"joint:{dof_index}"
        collapsed = self._section_collapsed.get(section_key, self._num_joints > 2 and dof_index > 0)
        frame = ui.CollapsableFrame(
            self._joint_section_title(dof_index, row_indices),
            collapsed=collapsed,
            style_type_name_override="CollapsableFrame",
        )
        self._joint_section_frames[dof_index] = frame
        with frame:
            with ui.VStack(spacing=0):
                self._build_group_toolbar(section_key, row_indices, "Optimize joint")
                self._build_section_rows(row_indices)

    def _build_link_section(self, link_index: int, row_indices: list[int]) -> None:
        section_key = f"link:{link_index}"
        frame = ui.CollapsableFrame(
            self._link_section_title(link_index, row_indices),
            collapsed=self._section_collapsed.get(section_key, True),
            style_type_name_override="CollapsableFrame",
        )
        self._link_section_frames[link_index] = frame
        with frame:
            with ui.VStack(spacing=0):
                self._build_group_toolbar(section_key, row_indices, "Optimize link")
                self._build_section_rows(row_indices)

    def _build_global_section(self, row_indices: list[int]) -> None:
        frame = ui.CollapsableFrame(
            self._global_section_title(row_indices),
            collapsed=self._section_collapsed.get("global", False),
            style_type_name_override="CollapsableFrame",
        )
        self._global_section_frame = frame
        with frame:
            with ui.VStack(spacing=0):
                self._build_group_toolbar("global", row_indices, "Optimize all global")
                self._build_section_rows(row_indices)

    def _build_section_rows(self, row_indices: list[int]) -> None:
        """Build parameter rows grouped by category, with a muted sub-label per category.

        Args:
            row_indices: Row indices value.
        """
        ordered = sorted(
            row_indices,
            key=lambda idx: (self._rows[idx].spec.category.value, idx),
        )
        current_category: str | None = None
        show_labels = len({self._rows[idx].spec.category.value for idx in ordered}) > 1
        for row_index in ordered:
            row = self._rows[row_index]
            category = row.spec.category.value
            if show_labels and category != current_category:
                current_category = category
                ui.Label(_CATEGORY_LABELS.get(category, category), style=_MUTED_STYLE, height=18)
            self._build_parameter_row(
                row_index,
                row,
                row.spec.entry.short_type_label(),
                f"{row.spec.entry.display_name(self._num_joints)} [{category}]",
            )

    def _link_indices(self) -> list[int]:
        indices = {
            row.spec.entry.link_index
            for row in self._rows
            if row.spec.entry.dof_index == GLOBAL_DOF_INDEX and row.spec.entry.link_index >= 0
        }
        return sorted(indices)

    def _rows_for_link(self, link_index: int) -> list[int]:
        return [
            idx
            for idx, row in enumerate(self._rows)
            if row.spec.entry.dof_index == GLOBAL_DOF_INDEX and row.spec.entry.link_index == link_index
        ]

    def _global_row_indices(self) -> list[int]:
        return [
            idx
            for idx, row in enumerate(self._rows)
            if row.spec.entry.dof_index == GLOBAL_DOF_INDEX and row.spec.entry.link_index == GLOBAL_LINK_INDEX
        ]

    def _link_section_title(self, link_index: int, row_indices: list[int]) -> str:
        selected = sum(1 for i in row_indices if self._rows[i].optimize)
        return f"Link {link_index + 1}  ({selected}/{len(row_indices)} selected)"

    def _global_section_title(self, row_indices: list[int]) -> str:
        selected = sum(1 for i in row_indices if self._rows[i].optimize)
        return f"Global  ({selected}/{len(row_indices)} selected)"

    def _update_section_titles(self) -> None:
        for dof_index, frame in self._joint_section_frames.items():
            row_indices = [i for i, r in enumerate(self._rows) if r.spec.entry.dof_index == dof_index]
            if row_indices:
                frame.title = self._joint_section_title(dof_index, row_indices)
        for link_index, frame in self._link_section_frames.items():
            row_indices = self._rows_for_link(link_index)
            if row_indices:
                frame.title = self._link_section_title(link_index, row_indices)
        if self._global_section_frame is not None and self._rows:
            row_indices = self._global_row_indices()
            if row_indices:
                self._global_section_frame.title = self._global_section_title(row_indices)

    def _build_group_toolbar(self, group_key: str, row_indices: list[int], caption: str) -> None:
        all_on = all(self._rows[i].optimize for i in row_indices)
        with ui.HStack(height=22):
            ui.Spacer(width=4)
            group_model = ui.SimpleBoolModel()
            group_model.set_value(all_on)
            self._group_opt_models[group_key] = group_model
            ui.CheckBox(
                model=group_model,
                width=18,
                tooltip="Include all parameters in this section in optimization",
            )
            group_model.add_value_changed_fn(
                lambda model, indices=row_indices: self._set_rows_optimize(indices, model.get_value_as_bool())
            )
            ui.Label(caption, style=_MUTED_STYLE)
            ui.Spacer()

    def _build_parameter_row(
        self,
        row_index: int,
        row: ParameterRowState,
        label: str,
        tooltip: str,
    ) -> None:
        with ui.ZStack(height=_ROW_HEIGHT, style_type_name_override="TreeView"):
            ui.Rectangle(name="treeview_item", style_type_name_override="TreeView")
            with ui.HStack():
                with ui.Frame(width=_COL_OPT):
                    with ui.HStack():
                        ui.Spacer()
                        opt_model = self._opt_models[row_index]
                        opt_model.set_value(row.optimize)
                        opt_cb = ui.CheckBox(model=opt_model, tooltip=f"Include {tooltip} in optimization")
                        group_key = self._group_key_for_entry(row.spec.entry)
                        opt_model.add_value_changed_fn(
                            lambda model, r=row, idx=row_index, gk=group_key: self._on_row_opt_changed(
                                r, idx, gk, model.get_value_as_bool()
                            )
                        )
                        self._opt_checkboxes.append(opt_cb)
                        ui.Spacer()

                with ui.Frame(width=_COL_NAME):
                    with ui.VStack():
                        ui.Spacer()
                        ui.Label(label, alignment=ui.Alignment.LEFT_CENTER, style=_LABEL_STYLE, tooltip=tooltip)
                        ui.Spacer()

                fields = {}
                if self._expert_bounds:
                    for key, value, on_change in (
                        ("initial", row.initial, self._on_initial_changed),
                        ("min", row.min_value, self._on_min_changed),
                        ("max", row.max_value, self._on_max_changed),
                    ):
                        with ui.Frame(width=_COL_NUM):
                            with ui.VStack():
                                ui.Spacer()
                                drag = ui.FloatField(
                                    width=_COL_NUM - 4,
                                    tooltip=f"{tooltip} - type the exact {key} value",
                                )
                                drag.model.set_value(value)
                                drag.model.add_value_changed_fn(
                                    lambda model, r=row, fn=on_change: fn(r, model.get_value_as_float())
                                )
                                fields[key] = drag
                                ui.Spacer()
                else:
                    with ui.Frame(width=_COL_NUM):
                        with ui.VStack():
                            ui.Spacer()
                            value_label = ui.Label(
                                f"{row.initial:.5g}",
                                alignment=ui.Alignment.CENTER,
                                style=_LABEL_STYLE,
                                tooltip=(
                                    f"{tooltip} - current value (enable 'Edit bounds' to show the editable "
                                    "Initial/Min/Max columns; previously edited bounds still apply)"
                                ),
                            )
                            fields["value_label"] = value_label
                            ui.Spacer()

                self._field_widgets[row_index] = fields
                self._sync_row_enabled(row_index)

    def _notify_selection_changed(self) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()

    def _set_rows_optimize(self, row_indices: list[int], checked: bool) -> None:
        for row_index in row_indices:
            self._apply_row_optimize(row_index, checked, update_titles=False)
        self._update_section_titles()
        self._notify_selection_changed()

    def _set_all_optimize(self, checked: bool) -> None:
        for row_index in range(len(self._rows)):
            self._apply_row_optimize(row_index, checked, update_titles=False)
        for model in self._group_opt_models.values():
            if model.get_value_as_bool() != checked:
                model.set_value(checked)
        self._update_section_titles()
        self._notify_selection_changed()

    def _select_parameter_family(self, parameter_type: SysIdParameterType) -> None:
        self._bulk_family = parameter_type
        row_indices = [index for index, row in enumerate(self._rows) if row.spec.entry.param_type is parameter_type]
        self._set_rows_optimize(row_indices, True)
        for group_key in self._group_opt_models:
            self._sync_group_checkbox(group_key)
        if self._bulk_bounds_status_label is not None:
            self._bulk_bounds_status_label.text = f"Bulk target: {parameter_type.value.replace('joint_', '')}"

    def _apply_bulk_family_bounds(self) -> None:
        if self._bulk_family is None or self._bulk_bounds_status_label is None:
            return
        initial = float(self._bulk_initial_model.get_value_as_float())
        min_value = float(self._bulk_min_model.get_value_as_float())
        max_value = float(self._bulk_max_model.get_value_as_float())
        if min_value >= max_value or not min_value <= initial <= max_value:
            self._bulk_bounds_status_label.text = "Bounds not applied: require Min < Max and Min <= Initial <= Max."
            return
        row_indices = [
            index
            for index, row in enumerate(self._rows)
            if row.optimize and row.spec.entry.param_type is self._bulk_family
        ]
        for row_index in row_indices:
            row = self._rows[row_index]
            row.initial = initial
            row.min_value = min_value
            row.max_value = max_value
            fields = self._field_widgets[row_index] if row_index < len(self._field_widgets) else {}
            for key, value in (("initial", initial), ("min", min_value), ("max", max_value)):
                if key in fields:
                    fields[key].model.set_value(value)
        family_name = self._bulk_family.value.replace("joint_", "")
        self._bulk_bounds_status_label.text = f"Applied bounds to {len(row_indices)} selected {family_name} rows."
        self._notify_selection_changed()

    def set_selected_by_predicate(self, predicate: Any) -> None:
        """Select exactly the rows whose spec matches ``predicate`` (recipe application).

        Safe to call before the table frame is (re)built: row state is mutated
        directly and widget models are synced only where they already exist.

        Args:
            predicate: Predicate used to select entries.
        """
        for row_index, row in enumerate(self._rows):
            self._apply_row_optimize(row_index, bool(predicate(row.spec)), update_titles=False)
        for group_key, model in self._group_opt_models.items():
            row_indices = self._rows_for_group(group_key)
            all_checked = bool(row_indices) and all(self._rows[i].optimize for i in row_indices)
            if model.get_value_as_bool() != all_checked:
                model.set_value(all_checked)
        if self._frame is not None:
            self._update_section_titles()
        self._notify_selection_changed()

    def set_expert_bounds(self, enabled: bool) -> None:
        """Toggle the Initial/Min/Max columns (off shows a read-only Value column).

        Args:
            enabled: Whether the option is enabled.
        """
        enabled = bool(enabled)
        if enabled == self._expert_bounds:
            return
        self._expert_bounds = enabled
        self._capture_collapse_states()
        if self._frame is not None:
            self._frame.rebuild()

    def _capture_collapse_states(self) -> None:
        for dof_index, frame in self._joint_section_frames.items():
            self._section_collapsed[f"joint:{dof_index}"] = bool(frame.collapsed)
        for link_index, frame in self._link_section_frames.items():
            self._section_collapsed[f"link:{link_index}"] = bool(frame.collapsed)
        if self._global_section_frame is not None:
            self._section_collapsed["global"] = bool(self._global_section_frame.collapsed)

    def _apply_row_optimize(self, row_index: int, checked: bool, update_titles: bool = True) -> None:
        row = self._rows[row_index]
        if row.optimize == checked:
            return
        row.optimize = checked
        if row_index < len(self._opt_models):
            self._opt_models[row_index].set_value(checked)
        self._sync_row_enabled(row_index)
        if update_titles:
            self._update_section_titles()

    def _on_row_opt_changed(self, row: ParameterRowState, row_index: int, group_key: str, checked: bool) -> None:
        row.optimize = checked
        self._sync_row_enabled(row_index)
        self._sync_group_checkbox(group_key)
        self._update_section_titles()
        self._notify_selection_changed()

    def _sync_group_checkbox(self, group_key: str) -> None:
        model = self._group_opt_models.get(group_key)
        if model is None:
            return
        row_indices = self._rows_for_group(group_key)
        if not row_indices:
            return
        all_on = all(self._rows[i].optimize for i in row_indices)
        if model.get_value_as_bool() != all_on:
            model.set_value(all_on)

    def _sync_row_enabled(self, row_index: int) -> None:
        if row_index >= len(self._field_widgets):
            return
        enabled = self._rows[row_index].optimize
        for drag in self._field_widgets[row_index].values():
            drag.enabled = enabled

    def _on_initial_changed(self, row: ParameterRowState, value: float) -> None:
        row.initial = value

    def _on_min_changed(self, row: ParameterRowState, value: float) -> None:
        row.min_value = value

    def _on_max_changed(self, row: ParameterRowState, value: float) -> None:
        row.max_value = value

    def get_selected_entries(self) -> list[SysIdParameterEntry]:
        """Return optimization entries in table order for checked rows.

        Returns:
            The resulting value.
        """
        return [row.spec.entry for row in self._rows if row.optimize]

    def get_selected_parameter_run_specs(self) -> list[ParameterRunSpec]:
        """Return selected rows as serializable run spec entries.

        Returns:
            The resulting value.
        """
        return [
            ParameterRunSpec.from_entry(
                row.spec.entry,
                initial=row.initial,
                min_value=row.min_value,
                max_value=row.max_value,
            )
            for row in self._rows
            if row.optimize
        ]

    def get_optimizer_vectors(self) -> tuple[list[float], list[float], list[float]]:
        """Get optimizer vectors.

        Returns:
            The resulting value.
        """
        initials, mins, maxs = [], [], []
        for row in self._rows:
            if row.optimize:
                label = row.spec.entry.display_name(self._num_joints)
                if row.min_value >= row.max_value:
                    raise ValueError(f"{label}: min ({row.min_value}) must be less than max ({row.max_value}).")
                if not (row.min_value <= row.initial <= row.max_value):
                    raise ValueError(
                        f"{label}: initial ({row.initial}) must be within [{row.min_value}, {row.max_value}]."
                    )
                initials.append(row.initial)
                mins.append(row.min_value)
                maxs.append(row.max_value)
        return initials, mins, maxs

    def update_theta_values(self, theta: list[float]) -> None:
        """Update theta values.

        Args:
            theta: Parameter vector to display.
        """
        idx = 0
        for row_index, row in enumerate(self._rows):
            if not row.optimize:
                continue
            if idx >= len(theta):
                break
            row.initial = theta[idx]
            if row_index < len(self._field_widgets):
                fields = self._field_widgets[row_index]
                if "initial" in fields:
                    fields["initial"].model.set_value(theta[idx])
                if "value_label" in fields:
                    fields["value_label"].text = f"{theta[idx]:.5g}"
            idx += 1

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled.

        Args:
            enabled: Whether the option is enabled.
        """
        for cb in self._opt_checkboxes:
            cb.enabled = enabled
        for row_index in range(len(self._field_widgets)):
            row_enabled = enabled and self._rows[row_index].optimize
            for widget in self._field_widgets[row_index].values():
                widget.enabled = row_enabled

    def cleanup(self) -> None:
        """Release resources."""
        self._opt_checkboxes.clear()
        self._opt_models.clear()
        self._field_widgets.clear()
        self._group_opt_models.clear()
        self._joint_section_frames.clear()
        self._link_section_frames.clear()
        self._global_section_frame = None
        self._section_collapsed.clear()
        self._bulk_family = None
        self._bulk_initial_model = None
        self._bulk_min_model = None
        self._bulk_max_model = None
        self._bulk_bounds_status_label = None
        self._selection_memo = {}
        self._memo_signature = None
        self._frame = None
