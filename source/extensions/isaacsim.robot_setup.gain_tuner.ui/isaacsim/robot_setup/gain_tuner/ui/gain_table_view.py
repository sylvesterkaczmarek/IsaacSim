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

"""A single-``TreeView`` gain table with Ctrl+click multi-select and per-selection edit.

This widget renders the tunable joints as one continuous table (index + name in
the first column, then the visible gain / advanced columns).  The index/name and
data columns are fixed pixel widths and a trailing ``ui.Fraction(1)`` spacer
column absorbs slack, so the table stays compact and left-packed at both normal
and wide window widths -- data columns never balloon and the name column never
grows into a giant empty band.  The spacer column renders nothing (no header, no
divider, no cell border) so the leftover width reads as blank canvas rather than
an empty, selectable column.

Data cells (the numeric gain fields, the Source cell, and the "-" not-applicable
cells) and their headers are center-aligned; only the joint-name column stays
left-aligned.

Selection uses the TreeView's native extended-selection semantics (Ctrl+click to
toggle a row, Shift+click for a range).  Editing a gain field applies the new
value to every *selected, editable* row (:func:`selection_edit_row_indices`); a
single selected row is just a single-row edit.  Read-only and blank cells are
skipped.  Column visibility is decided by the caller (auto-selected by schema and
optionally overridden via the hamburger menu) and passed in as the ordered list
of visible :class:`GainColumnSpec` values.
"""

from __future__ import annotations

from typing import Callable

import omni.ui as ui
from isaacsim.robot_setup import gain_tuner

from .gain_table_model import (
    GainColumnSpec,
    GainTableRow,
    selection_edit_row_indices,
    source_cell_model,
)
from .style import (
    FONT_SIZE,
    INFO_COLOR,
    LABEL_COLOR,
    MUTED_LABEL_COLOR,
    TREEVIEW_BG_COLOR,
    TREEVIEW_HEADER_BG_COLOR,
)
from .ui_utils import set_wrapped_tooltip

__all__ = ["HEADER_H", "ROW_H", "NA_CELL_TEXT", "GainTableView"]

# Row and header heights (pixels) for the gain TreeView.  Public so callers can
# size a surrounding scroll region to the table's content height.
ROW_H = 26
HEADER_H = 24
_INDEX_W = 30
_SOURCE_BOX_H = 18  # height of the Source cell box (combo + matching static box)

# Column sizing.  The leading index/name column AND every gain / advanced data
# column are fixed pixel widths so they stay compact and left-packed; a trailing
# ``ui.Fraction(1)`` spacer column (rendered empty) absorbs any extra window
# width.  This keeps the table balanced at both normal and very wide widths: the
# data columns never balloon (fixed pixels) and the name column never grows into a
# giant empty band (also fixed) -- the slack goes into the trailing spacer instead
# of a stray/oversized data column or an over-stretched name column.
_NAME_W = 220  # index (#) + joint-name column (fixed; long names elide w/ tooltip)
_NAME_MIN = 150
_GAIN_W = 116  # Stiffness / Damping / Ki (values can be large, e.g. 169797.33)
_ADV_W = 112  # armature / max force / max joint velocity / joint friction
_SOURCE_W = 150  # per-row gain-source label / dropdown ("Newton Actuator" + arrow)

# Text shown in a cell that does not apply to a joint (e.g. Ki on a non-PID joint,
# or an advanced column a joint does not author).  ``ui.Label`` renders with the
# app's default font, which has **no glyph for the em/en-dash** (U+2014 / U+2013) —
# those show up as a stray "?" missing-glyph box, making an intentionally-blank
# cell look like an error.  Use a plain ASCII hyphen so the "not applicable"
# indicator always renders.  Kept as a module constant so a unit test can guard
# against reintroducing a non-ASCII (non-renderable) glyph here.
NA_CELL_TEXT = "-"


def _value_field_style(info: bool, muted: bool) -> dict:
    """Return the ``ui.FloatDrag`` style for one table cell.

    Derived rather than written inline because a cell's rendering is not fixed for
    the life of the widget: authoring an unauthored parameter clears ``muted``, and
    an edit that removes a per-backend divergence clears ``info``.  Both the build
    and the in-place refresh have to arrive at the same answer, so there is one
    place that decides it.

    Args:
        info: Whether the cell carries a per-backend note, which outlines it in
            :data:`~.style.INFO_COLOR` -- two backends holding different values is
            a legitimate authoring choice, not a warning.
        muted: Whether the value is an engine default standing in for an unauthored
            parameter, which must not read with the weight of a chosen value.

    Returns:
        The style dict.
    """
    style = {"alignment": ui.Alignment.CENTER}
    if info:
        style |= {"border_color": INFO_COLOR, "border_width": 1, "border_radius": 3}
    if muted:
        style |= {"color": MUTED_LABEL_COLOR}
    return style


class _GainRowItem(ui.AbstractItem):
    """A TreeView item wrapping one :class:`GainTableRow` and its cell field models."""

    def __init__(self, row: GainTableRow):
        super().__init__()
        self.row = row
        # key -> ui.SimpleFloatModel for cells that render a value (editable or
        # read-only).  Blank cells have no model.  Populated lazily as the delegate
        # builds each cell so a value stays authoritative across selection edits.
        self.field_models: dict[str, ui.SimpleFloatModel] = {}
        # key -> the built ui.FloatDrag.  The model carries the number; the widget
        # carries whether that number reads as authored or as an engine default, in
        # its ``format`` and ``style``.  An edit changes the second as well as the
        # first, so the refresh path needs the widget and not just the model.
        self.field_widgets: dict[str, ui.Widget] = {}


class GainTableView:
    """A multi-select gain table backed by a single ``ui.TreeView``.

    Args:
        rows: The table rows (one :class:`GainTableRow` per shown joint).
        columns: The ordered visible :class:`GainColumnSpec` values (Kp/Kd first,
            then Ki / advanced as resolved by the caller).
        backend: Active backend label, used for the stiffness/damping unit hint.
        unit_hint: The stiffness/damping unit shown in the Kp/Kd headers (``"deg"``
            / ``"rad"`` / a linear unit / ``"mixed"``).
        write_cell_fn: Callback ``(row, column_key, value)`` that persists an edit
            to USD (DriveAPI attr, actuator attr, mjc arrays, or an advanced attr).
        selection_changed_fn: Optional callback ``(focused_row_index | None,
            selected_count)`` fired when the row selection changes.  The focused
            row drives *which* joint the detail editor shows; the selected count
            drives *whether* the per-joint detail panel is shown at all (single
            selection only).
        source_changed_fn: Optional callback ``(row, new_source)`` fired when a
            row's inline "Source" dropdown selects a different gain source.  The
            caller records the per-joint viewed source and rebuilds the table
            (re-resolving that row's gains + editability) and the detail panel.
        edit_committed_fn: Optional callback ``(edited_row, column_key, value)``
            fired once after a gain-cell edit has been written to every selected,
            editable row.  The caller uses it to refresh the per-joint detail panel
            in place when the edited row is the currently selected single joint
            (table -> detail sync), mirroring the detail -> table refresh.
        read_row_fn: Optional callback ``(row) -> GainTableRow | None`` that
            re-reads one row from USD.  Used after a cell edit to re-derive how the
            written cells render: whether a number is authored or an engine default
            lives in the cell's ``format`` and ``style`` rather than in its value
            model, so a cell that has just been authored has to be repainted or it
            keeps reading ``"(default)"``.  Without it the values still update and
            only the marking goes stale.
        focused_index: The row index to preselect (the last-focused joint), or None.
        selected_indices: Row indices to preselect as a multi-selection (used to
            preserve the selection across an in-place rebuild, e.g. a source
            switch).  Takes precedence over ``focused_index`` when non-empty.
    """

    def __init__(
        self,
        rows: list[GainTableRow],
        columns: list[GainColumnSpec],
        backend: str,
        unit_hint: str,
        write_cell_fn: Callable[[GainTableRow, str, float], None],
        selection_changed_fn: Callable[[int | None, int], None] | None = None,
        source_changed_fn: Callable[[GainTableRow, gain_tuner.GainSource], None] | None = None,
        edit_committed_fn: Callable[[GainTableRow, str, float], None] | None = None,
        read_row_fn: Callable[[GainTableRow], GainTableRow | None] | None = None,
        focused_index: int | None = None,
        selected_indices: set[int] | None = None,
    ):
        self._rows = list(rows)
        self._columns = list(columns)
        self._backend = backend
        self._unit_hint = unit_hint
        self._write_cell_fn = write_cell_fn
        self._selection_changed_fn = selection_changed_fn
        self._source_changed_fn = source_changed_fn
        self._edit_committed_fn = edit_committed_fn
        self._read_row_fn = read_row_fn
        self._focused_index = focused_index
        self._preselect_indices = set(selected_indices) if selected_indices else None
        self._suspend_writes = False

        self._items = [_GainRowItem(row) for row in self._rows]
        self._model = _GainTableModel(self._items, self._columns)
        self._delegate = _GainTableDelegate(self)
        self.tree_view: ui.TreeView | None = None
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _column_widths(self) -> tuple[list, list]:
        """Return ``(column_widths, min_column_widths)``.

        The index/name column and every data column are fixed ``ui.Pixel`` widths
        so they stay compact and left-packed; a trailing ``ui.Fraction(1)`` spacer
        column absorbs any extra window width.  This keeps the table balanced at
        both normal and wide widths -- no ballooning data column and no giant empty
        name column.
        """
        widths = [ui.Pixel(_NAME_W)]
        mins = [_NAME_MIN]
        for spec in self._columns:
            if spec.kind == "source":
                w = _SOURCE_W
            elif spec.kind == "gain":
                w = _GAIN_W
            else:
                w = _ADV_W
            widths.append(ui.Pixel(w))
            mins.append(w)
        # Trailing flexible spacer column: absorbs slack at wide window widths so
        # the fixed data columns stay left-aligned instead of stretching.
        widths.append(ui.Fraction(1))
        mins.append(0)
        return widths, mins

    def _build(self) -> None:
        widths, mins = self._column_widths()
        with ui.ScrollingFrame(
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            style_type_name_override="TreeView",
            style={"ScrollingFrame": {"background_color": TREEVIEW_BG_COLOR}},
        ):
            if not self._rows:
                with ui.VStack():
                    ui.Spacer(height=16)
                    ui.Label(
                        "No tunable joints in the selected robot.",
                        alignment=ui.Alignment.CENTER,
                        style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
                    )
                    ui.Spacer()
                return
            self.tree_view = ui.TreeView(
                self._model,
                delegate=self._delegate,
                alignment=ui.Alignment.CENTER_TOP,
                column_widths=widths,
                min_column_widths=mins,
                columns_resizable=False,
                header_visible=True,
                height=ui.Fraction(1),
            )
            self.tree_view.set_selection_changed_fn(self._on_selection_changed)
            # Restore a multi-selection (preserved across an in-place rebuild such
            # as a source switch) when provided; otherwise preselect the focused
            # row so the detail editor stays in sync.
            if self._preselect_indices:
                sel = [item for item in self._items if item.row.index in self._preselect_indices]
                if sel:
                    self.tree_view.selection = sel
            elif self._focused_index is not None:
                for item in self._items:
                    if item.row.index == self._focused_index:
                        self.tree_view.selection = [item]
                        break

    # ------------------------------------------------------------------
    # Selection + editing
    # ------------------------------------------------------------------

    def _selected_indices(self) -> set[int]:
        if self.tree_view is None:
            return set()
        return {item.row.index for item in (self.tree_view.selection or []) if isinstance(item, _GainRowItem)}

    def selected_count(self) -> int:
        """Return the number of joints currently selected in the table."""
        return len(self._selected_indices())

    def selected_indices(self) -> set[int]:
        """Return the row indices currently selected (for rebuild preservation)."""
        return self._selected_indices()

    def _on_source_changed(self, item: _GainRowItem, new_source: gain_tuner.GainSource) -> None:
        """Report an inline "Source" dropdown change to the caller.

        The caller records the per-joint viewed source (shared with the detail
        panel) and rebuilds the table so the row re-resolves to the selected
        source's gains + editability.  No-op when the source is unchanged.
        """
        if new_source == item.row.resolved.source:
            return
        if self._source_changed_fn is not None:
            self._source_changed_fn(item.row, new_source)

    def _on_selection_changed(self, selection) -> None:
        """Report the focused row (last selected) + selection count to the caller.

        The focused row drives which joint the detail editor shows; the count
        drives whether the per-joint detail panel is shown at all.
        """
        focused = None
        count = 0
        for item in selection or []:
            if isinstance(item, _GainRowItem):
                focused = item.row.index
                count += 1
        self._focused_index = focused
        if self._selection_changed_fn is not None:
            self._selection_changed_fn(focused, count)

    def _item_for_joint(self, joint_path: str) -> "_GainRowItem | None":
        """Return the row item whose joint matches ``joint_path`` (or None)."""
        for item in self._items:
            try:
                if item.row.entry.joint.GetPath().pathString == joint_path:
                    return item
            except Exception:
                continue
        return None

    def refresh_row_cells(
        self,
        joint_path: str,
        resolved,
        advanced_cells: dict,
        param_infos: dict | None = None,
        param_displays: dict | None = None,
    ) -> None:
        """Update one row's cached gains + live cell value models in place.

        Called after the per-joint detail panel writes a new gain to USD so the
        matching table cell reflects the edit *without* a full table rebuild --
        selection and scroll are preserved and there is no flicker.  Only the
        visible columns that have a live value model are updated (blank cells have
        no model and are skipped); the row's ``resolved`` / ``advanced_cells`` /
        ``param_displays`` are replaced so a later selection edit, and any cell the
        delegate rebuilds after this, read the fresh values.  Writes are suspended
        while the models are updated so the value-changed callback does not
        re-trigger the selection-edit path.

        Args:
            joint_path: Prim path of the edited joint.
            resolved: Freshly resolved gains for that joint (same source/context
                the detail panel used).
            advanced_cells: Freshly read advanced-column cells
                (``key -> (value | None, attr | None)``).
            param_infos: Freshly read per-backend notes (``key -> text``), so the
                row's data stays truthful after an edit that introduced or removed a
                divergence.  Pass None to leave the previous notes in place.
            param_displays: Freshly derived
                :class:`~.gain_display.AdvancedParamDisplay` values (``key ->
                display``).  These decide the *rendering* of an advanced cell -- the
                number the delegate draws, and whether it reads as authored or as an
                engine default -- so an edit that authors a value has to replace
                them or the cell keeps saying ``"(default)"`` over it, and a cell
                the delegate rebuilds later shows the pre-edit number.  Pass None
                for a row with no advanced columns.
        """
        item = self._item_for_joint(joint_path)
        if item is None:
            return
        item.row.resolved = resolved
        item.row.advanced_cells = dict(advanced_cells or {})
        if param_infos is not None:
            item.row.param_infos = dict(param_infos)
        if param_displays is not None:
            item.row.param_displays = dict(param_displays)
        self._suspend_writes = True
        try:
            for spec in self._columns:
                model = item.field_models.get(spec.key)
                if model is None:
                    continue
                if spec.kind == "advanced":
                    value, _attr = item.row.advanced_cell(spec.key)
                    self._refresh_advanced_cell_rendering(item, spec.key)
                elif spec.kind == "gain":
                    value, _attr, _label = item.row.gain_cell(spec.key)
                else:
                    continue
                if value is not None:
                    try:
                        model.set_value(float(value))
                    except Exception:
                        pass
        finally:
            self._suspend_writes = False

    def _refresh_advanced_cell_rendering(self, item: _GainRowItem, column_key: str) -> None:
        """Re-render one advanced cell from the row's current display, in place.

        ``format``, ``style`` and the tooltip are read/write on ``ui.FloatDrag`` and
        ``omni.ui`` re-emits the widget from them on the next frame, so this is a
        repaint and not a rebuild -- which matters because the caller runs from a
        value-changed handler that fires on every frame of a drag.

        The tooltip carries the same explanation the detail panel prints under the
        field, so it goes stale the same way: a cell dragged out of ``unlimited``
        showed the limit it now enforces and, on hover, the sentence saying nothing
        was authored and the engine was leaving the joint unclamped.  Re-installing
        it is what changes it -- ``omni.ui`` builds a dynamic tooltip's body once
        and serves that body forever after (see
        :func:`~.ui_utils.set_wrapped_tooltip`).
        """
        field = item.field_widgets.get(column_key)
        display = item.row.param_display(column_key)
        if field is None or display is None:
            return
        info = item.row.param_info(column_key) or display.note
        try:
            field.format = display.field_format
            field.style = _value_field_style(bool(info), not display.authored)
            set_wrapped_tooltip(field, info)
        except Exception:
            pass

    def _on_cell_edited(self, item: _GainRowItem, column_key: str, model: ui.SimpleFloatModel) -> None:
        """Apply an edit to every selected, editable row for this column.

        The edited row is always included, so a single-row edit (nothing multi-
        selected) still works.  Read-only / blank cells are skipped by
        :func:`selection_edit_row_indices`.
        """
        if self._suspend_writes:
            return
        value = model.get_value_as_float()
        selected = self._selected_indices()
        selected.add(item.row.index)
        targets = set(selection_edit_row_indices(self._rows, selected, column_key))
        self._suspend_writes = True
        written = []
        try:
            for target in self._items:
                if target.row.index not in targets:
                    continue
                try:
                    self._write_cell_fn(target.row, column_key, value)
                    written.append(target)
                except Exception:
                    pass
                # Reflect the value on the other rows' live fields.
                other = target.field_models.get(column_key)
                if other is not None and other is not model:
                    other.set_value(value)
        finally:
            self._suspend_writes = False
        self._rerender_written_cells(written, column_key)
        # After the write(s), let the caller mirror the new value into the detail
        # panel when this edited row is the currently selected single joint (table
        # -> detail sync).  Runs outside the suspend block so the detail refresh
        # sees settled state; the caller guards its own re-entrancy.
        if self._edit_committed_fn is not None:
            try:
                self._edit_committed_fn(item.row, column_key, value)
            except Exception:
                pass

    def _rerender_written_cells(self, written: list, column_key: str) -> None:
        """Re-derive the rendering of every advanced cell an edit just authored.

        An unauthored cell renders ``"(default)"`` or ``"unlimited"`` over the
        engine's value; the write it just took makes that a claim about a number
        the user chose.  Every written row needs it, not only the dragged one: a
        multi-row edit authors the whole selection, and the rows the user is not
        pointing at are the ones nobody would think to check.

        Skips a cell that is already authored and non-blank.  The transition only
        runs one way, so an authored cell renders the plain number under every later
        edit -- and this runs on every frame of a drag, where a re-read per selected
        row per frame would be the expensive part.

        Args:
            written: Row items the edit was written to.
            column_key: The edited column.
        """
        if self._read_row_fn is None:
            return
        for target in written:
            display = target.row.param_display(column_key)
            if display is None or (display.authored and not display.blank):
                continue
            try:
                fresh = self._read_row_fn(target.row)
            except Exception:
                continue
            if fresh is None:
                continue
            target.row.advanced_cells = dict(fresh.advanced_cells or {})
            target.row.param_displays = dict(fresh.param_displays or {})
            target.row.param_infos = dict(fresh.param_infos or {})
            self._refresh_advanced_cell_rendering(target, column_key)


class _GainTableModel(ui.AbstractItemModel):
    """TreeView model over the gain rows; column 0 is the joint name, then columns."""

    def __init__(self, items: list[_GainRowItem], columns: list[GainColumnSpec]):
        super().__init__()
        self._items = items
        self._columns = columns

    def get_item_children(self, item=None):
        return [] if item is not None else self._items

    def get_item_value_model_count(self, item) -> int:
        # +1 for the leading index/name column, +1 for the trailing flexible
        # spacer column (rendered empty) that absorbs slack at wide widths.
        return len(self._columns) + 2


class _GainTableDelegate(ui.AbstractItemDelegate):
    """Renders the header and each cell (index+name, gain / advanced fields)."""

    def __init__(self, view: GainTableView):
        super().__init__()
        self._view = view

    # -- headers --

    def build_branch(self, model, item=None, column_id=0, level=0, expanded=False):
        pass

    def build_header(self, column_id: int = 0) -> None:
        if column_id > len(self._view._columns):
            # Trailing spacer column: render nothing (no header block, no divider)
            # so the leftover width reads as blank canvas, not an empty column.
            return
        with ui.ZStack(height=HEADER_H):
            ui.Rectangle(style={"background_color": TREEVIEW_HEADER_BG_COLOR})
            with ui.HStack():
                ui.Spacer(width=6)
                if column_id == 0:
                    # Index number is centered (like the data cells); the joint-name
                    # header stays left-aligned as is conventional for a name column.
                    ui.Label(
                        "#",
                        width=_INDEX_W,
                        alignment=ui.Alignment.CENTER,
                        style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
                    )
                    ui.Label(
                        "Joint",
                        alignment=ui.Alignment.LEFT_CENTER,
                        style={"color": LABEL_COLOR, "font_size": FONT_SIZE},
                    )
                else:
                    spec = self._view._columns[column_id - 1]
                    label = spec.label
                    tooltip = spec.label
                    if spec.has_unit and self._view._unit_hint:
                        label = f"{label} ({self._view._unit_hint})"
                        tooltip = (
                            "Stiffness/Damping units are backend-dependent: PhysX uses degrees, "
                            "Newton/MuJoCo use radians (prismatic DOFs use linear units)."
                        )
                    # Data-column headers are centered to match the centered values.
                    header = ui.Label(
                        label,
                        elided_text=True,
                        alignment=ui.Alignment.CENTER,
                        style={"color": LABEL_COLOR, "font_size": FONT_SIZE},
                    )
                    set_wrapped_tooltip(header, tooltip)
                    ui.Spacer(width=6)

    # -- cells --

    def build_widget(self, model, item=None, index: int = 0, level: int = 0, expanded: bool = False) -> None:
        if not isinstance(item, _GainRowItem):
            return
        if index == 0:
            self._build_name_cell(item)
            return
        if index > len(self._view._columns):
            # Trailing flexible spacer column: draw the row-highlight rectangle so
            # selection/hover extends across the full row, but no cell content.
            self._build_spacer_cell()
            return
        spec = self._view._columns[index - 1]
        if spec.kind == "source":
            self._build_source_cell(item)
        elif spec.kind == "gain":
            self._build_gain_cell(item, spec)
        else:
            self._build_advanced_cell(item, spec)

    def _build_name_cell(self, item: _GainRowItem) -> None:
        row = item.row
        with ui.ZStack(height=ROW_H):
            ui.Rectangle(name="treeview_item", style={"background_color": 0x00000000})
            with ui.HStack():
                ui.Spacer(width=6)
                ui.Label(
                    str(row.index + 1),
                    width=_INDEX_W,
                    alignment=ui.Alignment.CENTER,
                    style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
                )
                ui.Label(
                    row.display_name,
                    elided_text=True,
                    tooltip=row.display_name,
                    alignment=ui.Alignment.LEFT_CENTER,
                    style={"color": LABEL_COLOR, "font_size": FONT_SIZE},
                )

    def _build_source_cell(self, item: _GainRowItem) -> None:
        """Render the per-row gain-source cell: a static label or a source dropdown.

        A joint with a single available source shows a static label; a joint with
        more than one shows a ``ui.ComboBox`` of its available sources.  Selecting
        a different option calls back into the view (``_on_source_changed``) which
        re-resolves that row's gains + editability and keeps the detail panel's
        viewed-source toggle in sync.
        """
        model = source_cell_model(item.row.available_sources, item.row.resolved.source)
        with ui.ZStack(height=ROW_H):
            ui.Rectangle(name="treeview_item", style={"background_color": 0x00000000})
            with ui.HStack():
                ui.Spacer(width=2)
                with ui.VStack():
                    ui.Spacer()
                    if not model.is_dropdown:
                        # Single available source: render a static box styled like
                        # the combo so the Source column reads as a uniform column of
                        # boxes, just without an (interactive) dropdown arrow.
                        label = model.selected_label or NA_CELL_TEXT
                        with ui.ZStack(height=_SOURCE_BOX_H):
                            ui.Rectangle(name="source_box")
                            ui.Label(
                                label,
                                elided_text=True,
                                tooltip=label,
                                alignment=ui.Alignment.CENTER,
                                style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
                            )
                    else:
                        labels = [opt.label for opt in model.options]
                        combo = ui.ComboBox(model.selected_index(), *labels, height=_SOURCE_BOX_H, name="source_combo")
                        options = list(model.options)
                        combo.model.add_item_changed_fn(
                            lambda m, _i, it=item, opts=options: self._on_source_combo_changed(it, opts, m)
                        )
                    ui.Spacer()
                ui.Spacer(width=2)

    def _on_source_combo_changed(self, item: _GainRowItem, options: list, combo_model) -> None:
        """Translate a source ComboBox selection into a viewed-source change."""
        try:
            idx = combo_model.get_item_value_model().get_value_as_int()
        except Exception:
            return
        if 0 <= idx < len(options):
            self._view._on_source_changed(item, options[idx].source)

    def _blank_cell(self, tooltip: str = "") -> None:
        with ui.ZStack(height=ROW_H):
            ui.Rectangle(name="treeview_item", style={"background_color": 0x00000000})
            label = ui.Label(
                NA_CELL_TEXT,
                alignment=ui.Alignment.CENTER,
                style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
            )
            set_wrapped_tooltip(label, tooltip)

    def _build_spacer_cell(self) -> None:
        """Render the trailing spacer column cell: nothing at all.

        This column exists only to absorb slack at wide window widths.  It draws no
        rectangle/border and no content, so the leftover width reads as blank canvas
        (no divider, no selectable empty column).  A bare ``ui.Spacer`` keeps the
        cell height consistent with the row without painting anything.
        """
        ui.Spacer(height=ROW_H)

    def _build_value_field(
        self,
        item: _GainRowItem,
        column_key: str,
        value: float,
        editable: bool,
        info: str | None = None,
        field_format: str = "",
        muted: bool = False,
    ) -> None:
        model = ui.SimpleFloatModel(float(value))
        item.field_models[column_key] = model
        # An informational note outlines the field and carries the detail as its
        # tooltip.
        field_style = _value_field_style(bool(info), muted)
        drag_kwargs = {"format": field_format} if field_format else {}
        with ui.ZStack(height=ROW_H):
            ui.Rectangle(name="treeview_item", style={"background_color": 0x00000000})
            with ui.HStack():
                ui.Spacer(width=2)
                with ui.VStack():
                    ui.Spacer()
                    # Center the value text (via style; FloatDrag has no alignment
                    # ctor arg).  Unknown style keys are ignored, so this is safe.
                    field = ui.FloatDrag(
                        model=model,
                        step=0.1,
                        min=0.0,
                        height=18,
                        enabled=editable,
                        name="adv_param_field",
                        style=field_style,
                        **drag_kwargs,
                    )
                    # A built tooltip rather than a plain string: this is the whole
                    # per-backend sentence, and a 112 px cell cannot show it on one
                    # unwrapped line without the window clipping it mid-word.
                    set_wrapped_tooltip(field, info or "")
                    ui.Spacer()
                ui.Spacer(width=2)
        item.field_widgets[column_key] = field
        if editable:
            model.add_value_changed_fn(lambda m, it=item, k=column_key: self._view._on_cell_edited(it, k, m))

    def _build_gain_cell(self, item: _GainRowItem, spec: GainColumnSpec) -> None:
        value, attr, _ki_label = item.row.gain_cell(spec.key)
        if value is None:
            self._blank_cell()
            return
        # A gain cell is editable when the row's viewed source is the active,
        # consumed one; a MuJoCo-native editable row has no per-field attr but
        # writes through author_mjc_gains, so it is still editable.
        editable = bool(item.row.editable)
        self._build_value_field(item, spec.key, value, editable)

    def _build_advanced_cell(self, item: _GainRowItem, spec: GainColumnSpec) -> None:
        value, attr = item.row.advanced_cell(spec.key)
        display = item.row.param_display(spec.key)
        if display is None:
            # A plain single-attribute column (the DriveAPI max force): no
            # per-backend resolution, so the raw value is the whole story.
            if value is None or attr is None:
                self._blank_cell()
                return
            self._build_value_field(item, spec.key, value, editable=True, info=item.row.param_info(spec.key))
            return
        if attr is None or not display.editable or display.blank:
            self._blank_cell(tooltip=display.note)
            return
        self._build_value_field(
            item,
            spec.key,
            display.field_value,
            editable=True,
            info=item.row.param_info(spec.key) or display.note,
            field_format=display.field_format,
            muted=not display.authored,
        )
