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

"""Row / column model for the gain table (UI-independent, no Kit imports).

The Gain Settings view renders a single ``TreeView`` table of the robot's tunable
joints.  Rows are multi-selected with standard Ctrl+click / Shift+click semantics;
editing a gain field applies the value to every selected, editable row.  This
module builds the per-row data the table needs and decides which columns are
visible:

* **Always-present** gain columns — Stiffness/Kp and Damping/Kd — plus a **Ki**
  column that appears only when a joint has a PID Newton actuator.
* **Dynamic advanced columns** (armature, max force, max joint velocity, joint
  friction) that auto-appear when at least one shown joint authors the backing
  schema/attribute and auto-hide otherwise (:func:`visible_advanced_gain_columns`
  in :mod:`~isaacsim.robot_setup.gain_tuner.ui.gain_display`).  Cells that do not
  apply to a joint render blank.

A **column catalog** (:data:`GAIN_COLUMN_CATALOG`) groups every possible column
by category (Drives / Performance Envelope / MuJoCo Joint) — mirroring the Joint
Inspector column picker.  Columns *auto-select* based on the schemas the shown
joints actually have (:func:`auto_visible_column_keys`).  The hamburger menu is
then built **dynamically from the loaded articulation**, not from the full
catalog: :func:`grouped_columns_for_menu` lists only the categories/columns
applicable to the shown joints (Kp/Kd always; Ki only with a PID actuator;
advanced columns per the authored-only / schema-present rules), and
:func:`menu_backend_rows` offers only the relevant backends (PhysX always;
MuJoCo only when a joint authors ``mjc:*`` gains, per :func:`menu_has_mjc`).  The
user can still override any applicable column's visibility manually
(:func:`resolve_visible_columns`, :func:`grouped_columns_for_menu`).

It also computes which rows a **selection-driven edit** should write, honoring
the same read-only rules the detail editor uses (:func:`is_viewed_source_editable`):
only editable, applicable cells are written
(:func:`mass_edit_row_indices` / :func:`selection_edit_row_indices`).

The module deliberately imports only the UI-independent core gain logic so it can
be unit-tested against in-memory USD stages without a running app.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pxr
from isaacsim.robot_setup import gain_tuner
from pxr import UsdPhysics

from .gain_display import (
    ADVANCED_GAIN_COLUMNS,
    AdvancedParamDisplay,
    gain_dof_unit,
    joint_param_infos,
    read_advanced_gain_cell,
    read_advanced_param_display,
    visible_advanced_gain_columns,
)

# USD / attribute access can fail with these when a prim, attribute, or layer is
# missing, invalid, a duck-typed fake (unit tests substitute plain objects), or
# holds an unexpected value type.  Narrowed from a bare ``Exception`` so a real
# programming error in the resolver surfaces instead of silently emptying a row's
# per-backend notes, which would read as "this joint has nothing to report".
_USD_ACCESS_ERRORS = (AttributeError, TypeError, ValueError, pxr.Tf.ErrorException)

# Column keys for the always-present gain columns.  Advanced columns use their
# ``AdvancedGainColumn.key`` as the column key.
GAIN_COLUMN_SOURCE = "source"
GAIN_COLUMN_KP = "kp"
GAIN_COLUMN_KD = "kd"
GAIN_COLUMN_KI = "ki"

# The single, shared per-gain-source display label used by BOTH the table's
# "Source" column (and its inline dropdown) AND the detail panel's viewed-source
# toggle, so the two always read identically for the same :class:`GainSource`
# (they stay in sync on the shared per-joint selected-source state).  The wording
# is deliberately compact so it reads well in the narrow table column and in the
# toggle buttons.  ASCII only -- the app font has no em-dash / ellipsis glyph.
_SOURCE_OPTION_LABELS = {
    gain_tuner.GainSource.PHYSICS_DRIVE: "Drive",
    gain_tuner.GainSource.MUJOCO: "MuJoCo",
    gain_tuner.GainSource.ACTUATOR: "Newton Actuator",
    gain_tuner.GainSource.NONE: "-",
}


def source_option_label(source: gain_tuner.GainSource) -> str:
    """Return the shared display label for a :class:`GainSource` (ASCII only).

    Used by both the table "Source" column and the detail-panel viewed-source
    toggle so they always show the same wording for a given source.
    """
    return _SOURCE_OPTION_LABELS.get(source, "-")


# ---------------------------------------------------------------------------
# Column catalog + categories (drives the hamburger column picker)
# ---------------------------------------------------------------------------

# Category labels for the hamburger column picker, mirroring the Joint Inspector
# groupings (a subset — the gain tuner only surfaces gain + advanced-drive params).
CATEGORY_DRIVES = "Drives"
CATEGORY_PERFORMANCE_ENVELOPE = "Performance Envelope"
CATEGORY_MUJOCO_JOINT = "MuJoCo Joint"

# Backend tags shown in the picker's BACKENDS row.  PhysX ``DriveAPI`` / physxJoint
# params are PhysX; the integral (Ki) gain only exists for a Newton/MuJoCo actuator.
BACKEND_PHYSX = "PhysX"
BACKEND_MUJOCO = "MuJoCo"

# Per-joint detail/info panel visibility, keyed off the table's selection count.
# The panel (gain source, viewed-source toggle, Controller Gains, advanced params,
# validation summary) is specific to one joint, so it is shown only for a single
# selection; an empty or multi-selection shows a short placeholder instead.
DETAIL_PANEL_SINGLE = "single"
DETAIL_PANEL_NONE = "none"
DETAIL_PANEL_MULTI = "multi"


@dataclass(frozen=True)
class GainColumnSpec:
    """One selectable column in the gain table, for the hamburger column picker.

    The catalog (:data:`GAIN_COLUMN_CATALOG`) lists every possible column grouped
    by :attr:`category`; a column is *auto-selected* when a shown joint offers it
    (:func:`auto_visible_column_keys`) and can also be toggled manually.
    """

    key: str
    #: Stable column key (``"kp"`` / ``"kd"`` / ``"ki"`` or an advanced key).

    label: str
    #: User-facing header / menu label.

    category: str
    #: Picker group (Drives / Performance Envelope / MuJoCo Joint).

    kind: str
    #: ``"gain"`` for Kp/Kd/Ki, ``"advanced"`` for the dynamic schema columns.

    has_unit: bool = False
    #: True for Stiffness/Kp and Damping/Kd, whose header shows the backend unit.

    backends: tuple = (BACKEND_PHYSX, BACKEND_MUJOCO)
    #: Backends the column applies to (shown in the picker's BACKENDS row).


# Category / backend assignment for the advanced (schema-driven) columns, keyed by
# ``AdvancedGainColumn.key``.  Armature and joint friction are MuJoCo-joint params;
# max joint velocity is a performance-envelope limit; max force is a drive limit.
_ADVANCED_CATEGORY = {
    "armature": CATEGORY_MUJOCO_JOINT,
    "max_force": CATEGORY_DRIVES,
    "max_joint_velocity": CATEGORY_PERFORMANCE_ENVELOPE,
    "joint_friction": CATEGORY_MUJOCO_JOINT,
}


def _build_gain_column_catalog() -> tuple:
    """Assemble the ordered column catalog from the gain + advanced columns."""
    catalog = [
        # The per-row gain-source column (static label or an inline source-picker
        # dropdown) is the first data column, right after the joint index/name.
        GainColumnSpec(GAIN_COLUMN_SOURCE, "Source", CATEGORY_DRIVES, "source"),
        GainColumnSpec(GAIN_COLUMN_KP, "Stiffness / Kp", CATEGORY_DRIVES, "gain", has_unit=True),
        GainColumnSpec(GAIN_COLUMN_KD, "Damping / Kd", CATEGORY_DRIVES, "gain", has_unit=True),
        GainColumnSpec(GAIN_COLUMN_KI, "Ki", CATEGORY_DRIVES, "gain", backends=(BACKEND_MUJOCO,)),
    ]
    for column in ADVANCED_GAIN_COLUMNS:
        catalog.append(
            GainColumnSpec(
                key=column.key,
                label=column.label,
                category=_ADVANCED_CATEGORY.get(column.key, CATEGORY_DRIVES),
                kind="advanced",
                backends=(BACKEND_PHYSX,),
            )
        )
    return tuple(catalog)


# Every column the table can show, in display order (Kp, Kd, Ki, then advanced).
GAIN_COLUMN_CATALOG = _build_gain_column_catalog()

# Fast lookup key -> spec.
_CATALOG_BY_KEY = {spec.key: spec for spec in GAIN_COLUMN_CATALOG}


def gain_column_spec(column_key: str) -> GainColumnSpec | None:
    """Return the :class:`GainColumnSpec` for a column key, or None if unknown."""
    return _CATALOG_BY_KEY.get(column_key)


def is_angular_dof(joint: object, drive_axis: object = None) -> bool:
    """Return True when a joint's DOF is angular (revolute / D6 rotational).

    Angular DOFs use backend-dependent angle units for stiffness/damping (degrees
    under PhysX, radians under Newton); prismatic DOFs use linear units.

    Args:
        joint: The joint prim.
        drive_axis: Optional D6 drive-axis token (``rotX`` / ``transY`` / ...).

    Returns:
        True for revolute / rotational DOFs, False for prismatic / linear DOFs.
        Defaults to True (angular) when the type cannot be determined.
    """
    try:
        if joint.IsA(UsdPhysics.PrismaticJoint):
            return False
        if joint.IsA(UsdPhysics.RevoluteJoint):
            return True
    except Exception:
        pass
    token = str(drive_axis or "").lower()
    if token.startswith("trans") or token == "linear":
        return False
    if token.startswith("rot") or token == "angular":
        return True
    return True


@dataclass
class GainTableRow:
    """Per-joint data for one row of the hybrid gain table.

    A row is created for each *checked* joint.  It caches the resolved gains and
    the advanced-parameter cells so the grid can render values, gate editability,
    and route mass edits without re-reading USD per cell.
    """

    index: int
    #: Zero-based position of this row within the checked-joint set.

    entry: object
    #: The ``JointListEntry`` this row was built from.

    resolved: gain_tuner.ResolvedGains
    #: Resolved gains for the row's default/active source.

    is_angular: bool
    #: True for revolute / rotational DOFs (drives the stiffness/damping unit).

    editable: bool
    #: True when the gain cells (Kp/Kd/Ki) are editable (active, consumed source).

    advanced_cells: dict = field(default_factory=dict)
    #: Map of advanced column key -> ``(value | None, attr | None)``; a None value
    #: means the joint lacks that parameter and the cell renders blank.

    param_infos: dict = field(default_factory=dict)
    #: Map of advanced column key -> informational text for the multi-schema
    #: parameters worth pointing out: the two backends holding different values, an
    #: authored value the active backend never reads, or an undeterminable resolver
    #: chain.  Empty for a joint with nothing to report (the common case); a present
    #: entry marks the cell so it is visible without opening the joint.

    param_displays: dict = field(default_factory=dict)
    #: Map of advanced column key -> :class:`AdvancedParamDisplay` for the
    #: per-backend parameters.  Says whether the cell's number is what the stage
    #: holds or an engine default standing in for an unauthored parameter, and
    #: whether the parameter is unlimited rather than any number -- neither of
    #: which a bare float in :attr:`advanced_cells` can express.  Absent for the
    #: DriveAPI max-force column, which is a plain single attribute.

    available_sources: list = field(default_factory=list)
    #: The gain sources this joint can view/edit (:func:`available_viewed_sources`),
    #: in presentation order.  Drives the "Source" column: a single source renders
    #: a static label, more than one renders an inline source-picker dropdown.

    @property
    def display_name(self) -> str:
        """Human-readable joint name for the row/side-panel label."""
        return getattr(self.entry, "display_name", str(self.entry))

    def gain_unit(self, backend: str, linear_unit: str = "m") -> str:
        """Return the stiffness/damping unit token for this row under ``backend``."""
        return gain_dof_unit(self.is_angular, backend, linear_unit)

    def gain_cell(self, column_key: str) -> tuple[float | None, object, str | None]:
        """Return ``(value, attr, ki_label)`` for an always-present gain column.

        ``value`` is None when the column does not apply to this row (e.g. Ki on a
        non-PID joint), in which case the cell renders blank and uneditable.
        """
        resolved = self.resolved
        if column_key == GAIN_COLUMN_KP:
            return resolved.kp, resolved.kp_attr, None
        if column_key == GAIN_COLUMN_KD:
            return resolved.kd, resolved.kd_attr, None
        if column_key == GAIN_COLUMN_KI:
            if resolved.ki is None:
                return None, None, None
            return resolved.ki, resolved.ki_attr, resolved.ki_label
        return None, None, None

    def advanced_cell(self, column_key: str) -> tuple[float | None, object]:
        """Return ``(value, attr)`` for an advanced column, or ``(None, None)``."""
        return self.advanced_cells.get(column_key, (None, None))

    def param_info(self, column_key: str) -> str | None:
        """Return the column's per-backend note, or None when there is nothing to say."""
        return self.param_infos.get(column_key)

    def param_display(self, column_key: str) -> AdvancedParamDisplay | None:
        """Return how the column should render, or None when it is a plain attribute."""
        return self.param_displays.get(column_key)


@dataclass(frozen=True)
class SourceCellOption:
    """One selectable gain source in the table's "Source" column dropdown."""

    source: gain_tuner.GainSource
    #: The :class:`GainSource` this option selects (its viewed/edited source).

    label: str
    #: Compact, ASCII-only label (see :func:`source_option_label`).


@dataclass(frozen=True)
class SourceCellModel:
    """How the "Source" column cell should render for one joint row.

    A joint with a single available source renders a static label; a joint with
    more than one available source renders a dropdown so the user can switch the
    row's viewed/edited source inline (mirroring the detail panel's viewed-source
    toggle).  Selecting an option re-resolves the row's gains via
    :func:`resolve_joint_gains` (``viewed_source=option.source``) and re-gates
    editability via :func:`is_viewed_source_editable` — the same path the table
    build already runs, so no separate edit logic is needed.
    """

    selected_source: gain_tuner.GainSource
    #: The row's currently viewed/edited source (``row.resolved.source``).

    selected_label: str
    #: Compact label for :attr:`selected_source`.

    options: list  # list[SourceCellOption]
    #: Available source options in presentation order (empty when none authored).

    is_dropdown: bool
    #: True when more than one source is available (render a dropdown), else a
    #: static label.

    def selected_index(self) -> int:
        """Return the index of :attr:`selected_source` in :attr:`options` (0 if absent)."""
        for i, option in enumerate(self.options):
            if option.source == self.selected_source:
                return i
        return 0


def source_cell_model(available_sources: list, selected_source: gain_tuner.GainSource) -> SourceCellModel:
    """Build the :class:`SourceCellModel` for a row from its available sources.

    Args:
        available_sources: The joint's available :class:`GainSource` values, in
            presentation order (:func:`available_viewed_sources`).
        selected_source: The source the row currently displays (``resolved.source``).

    Returns:
        A :class:`SourceCellModel`: a static label for zero/one source, a dropdown
        (``is_dropdown=True``) with the available options for two or more.  When
        ``selected_source`` is not among the available options (e.g. a joint with
        no authored gains) the model still reports it so the cell renders a label.
    """
    options = [SourceCellOption(source=src, label=source_option_label(src)) for src in available_sources]
    return SourceCellModel(
        selected_source=selected_source,
        selected_label=source_option_label(selected_source),
        options=options,
        is_dropdown=len(options) > 1,
    )


@dataclass
class GainTableColumns:
    """The set of columns the hybrid table shows for the current checked joints."""

    show_ki: bool = False
    #: True when at least one checked joint has a PID actuator (Ki applies).

    advanced: list = field(default_factory=list)
    #: Visible :class:`AdvancedGainColumn` values (auto-appear / auto-hide).


def read_param_infos(joint: object, backend: str, solver: str = "") -> dict:
    """Return a joint's per-backend advanced-parameter notes, keyed by column key.

    An advanced cell shows the value the *active* backend resolves, which does not
    say what the other backend holds, whether an authored value is being ignored,
    or whether the resolver chain could be determined at all.  Rows carry this so
    the table can mark those cells.

    Args:
        joint: The joint prim.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        Map of advanced column key -> informational text.  Empty when the joint has
        nothing to report, which is the common case.
    """
    try:
        resolutions = gain_tuner.resolve_joint_params(joint, backend, solver)
    except _USD_ACCESS_ERRORS:
        return {}
    return joint_param_infos(resolutions)


def build_gain_table_rows(
    entries: list,
    ctx: gain_tuner.GainReadContext,
    viewed_source_for: dict | None = None,
    engine_value_fn: Callable[[object, object], float | None] | None = None,
) -> list[GainTableRow]:
    """Build the table rows for a set of checked joint entries.

    Args:
        entries: The checked ``JointListEntry`` objects (the joints shown in the
            table), in display order.
        ctx: The :class:`GainReadContext` for the active robot/backend.
        viewed_source_for: Optional map of joint prim path -> forced
            :class:`GainSource` for a per-joint comparison view.  When absent, each
            row uses the source the active backend consumes (the default view).
        engine_value_fn: Optional ``(entry, param_spec) -> float | None`` returning
            what the running engine reports for a per-backend advanced parameter.
            Used only where nothing is authored, so the cell can show the value the
            simulation is really using rather than the documented default it is
            usually, but not always, equal to.  None (the default) falls back to
            the documented default throughout.

    Returns:
        One :class:`GainTableRow` per entry, in order.
    """
    viewed_source_for = viewed_source_for or {}
    rows: list[GainTableRow] = []
    for i, entry in enumerate(entries):
        joint = entry.joint
        drive_axis = getattr(entry, "drive_axis", None)
        try:
            path = joint.GetPath().pathString
        except Exception:
            path = ""
        resolved = gain_tuner.resolve_joint_gains(joint, drive_axis, ctx, viewed_source=viewed_source_for.get(path))
        editable = gain_tuner.is_viewed_source_editable(
            getattr(resolved, "source", gain_tuner.GainSource.NONE),
            getattr(resolved, "active_source", gain_tuner.GainSource.NONE),
        )
        backend = getattr(ctx, "active_backend", "PhysX")
        solver = getattr(ctx, "solver", "")
        advanced_cells = {
            column.key: read_advanced_gain_cell(joint, drive_axis, column, backend, solver)
            for column in ADVANCED_GAIN_COLUMNS
        }
        param_displays = {}
        for column in ADVANCED_GAIN_COLUMNS:
            if column.param_spec is None:
                continue
            engine_value = None
            if engine_value_fn is not None:
                try:
                    engine_value = engine_value_fn(entry, column.param_spec)
                except _USD_ACCESS_ERRORS:
                    engine_value = None
            try:
                display = read_advanced_param_display(joint, column, backend, solver, engine_value)
            except _USD_ACCESS_ERRORS:
                continue
            if display is not None:
                param_displays[column.key] = display
        param_infos = read_param_infos(joint, backend, solver)
        # The sources this joint can view/edit (drives the "Source" column): the
        # same has_pd / has_mjc / has_act signals resolve_joint_gains reads.
        has_act = bool(getattr(ctx, "actuator_map", None)) and path in ctx.actuator_map
        has_mjc = bool(getattr(ctx, "mjc_map", None)) and path in ctx.mjc_map
        has_pd = gain_tuner.has_physics_drive(joint, drive_axis)
        available_sources = gain_tuner.available_viewed_sources(has_pd, has_mjc, has_act)
        rows.append(
            GainTableRow(
                index=i,
                entry=entry,
                resolved=resolved,
                is_angular=is_angular_dof(joint, drive_axis),
                editable=editable,
                advanced_cells=advanced_cells,
                param_infos=param_infos,
                param_displays=param_displays,
                available_sources=available_sources,
            )
        )
    return rows


def visible_gain_columns(rows: list[GainTableRow], backend: str = "PhysX", solver: str = "") -> GainTableColumns:
    """Return which optional/dynamic columns the table should show.

    The Ki column appears when any row has a PID actuator (``resolved.ki`` set);
    the advanced columns appear when any row authors the backing schema/attribute.

    Args:
        rows: The current table rows (the checked joints).
        backend: The active backend label, which selects the resolver chain the
            advanced columns resolve through.
        solver: The running Newton solver token.

    Returns:
        A :class:`GainTableColumns` describing the visible optional columns.
    """
    show_ki = any(row.resolved.ki is not None for row in rows)
    joint_specs = [(row.entry.joint, getattr(row.entry, "drive_axis", None)) for row in rows]
    advanced = visible_advanced_gain_columns(joint_specs, backend, solver)
    return GainTableColumns(show_ki=show_ki, advanced=advanced)


def auto_visible_column_keys(rows: list[GainTableRow], backend: str = "PhysX", solver: str = "") -> set[str]:
    """Return the column keys the shown joints' schemas auto-select.

    Stiffness/Kp and Damping/Kd are always auto-visible (the core gain columns);
    Ki is added when a shown joint has a PID actuator, and each advanced column is
    added when at least one shown joint authors its backing schema/attribute (the
    auto-appear rule, authored-only for Max Force).

    Args:
        rows: The current table rows (the shown joints).
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        The set of auto-selected column keys.
    """
    # Source, Stiffness/Kp and Damping/Kd are always applicable for any tunable
    # joint (every tunable joint has a gain source and drive gains).
    keys = {GAIN_COLUMN_SOURCE, GAIN_COLUMN_KP, GAIN_COLUMN_KD}
    cols = visible_gain_columns(rows, backend, solver)
    if cols.show_ki:
        keys.add(GAIN_COLUMN_KI)
    for column in cols.advanced:
        keys.add(column.key)
    return keys


def resolve_visible_columns(
    rows: list[GainTableRow],
    manual_overrides: dict | None = None,
    backend: str = "PhysX",
    solver: str = "",
) -> list[GainColumnSpec]:
    """Return the ordered visible columns given auto-selection + manual overrides.

    A column is visible when the user has manually toggled it on/off; absent a
    manual override it follows the schema-driven auto-selection
    (:func:`auto_visible_column_keys`).  Columns keep catalog order.

    Args:
        rows: The current table rows (the shown joints).
        manual_overrides: Optional map of column key -> forced visibility (True to
            show, False to hide) set from the hamburger menu.  Keys not present
            follow the auto-selection.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        The visible :class:`GainColumnSpec` values in catalog order.
    """
    manual = manual_overrides or {}
    auto = auto_visible_column_keys(rows, backend, solver)
    # A column is only selectable when it is *applicable* to the loaded articulation
    # (auto-selected by some shown joint's schema — Kp/Kd always, Ki only with a PID
    # actuator, advanced columns per the authored-only / schema-present rules).
    # Manual overrides are scoped to those applicable columns, so a stale or forced
    # override can never surface a column that no shown joint offers (which would
    # render an all-blank column and re-introduce the static-catalog bug).
    return [spec for spec in GAIN_COLUMN_CATALOG if spec.key in auto and manual.get(spec.key, True)]


def grouped_columns_for_menu(
    rows: list[GainTableRow],
    manual_overrides: dict | None = None,
    backend: str = "PhysX",
    solver: str = "",
) -> list[tuple[str, list[tuple[GainColumnSpec, bool, bool]]]]:
    """Return the column picker's grouped rows: ``[(category, [(spec, checked, auto)])]``.

    The menu is built **dynamically from the loaded articulation**, not from the
    full static catalog: a column (and therefore its category) is listed only when
    it is applicable — i.e. auto-selected by some shown joint's schema
    (:func:`auto_visible_column_keys`).  So the MuJoCo-joint category, the Ki
    column, and each advanced column appear only when a shown joint actually offers
    them; the always-present Drives basics (Kp/Kd) are always listed.

    Each entry pairs a category with its applicable columns (in catalog order) and,
    per column, whether it is currently checked (visible) and whether the
    schema-driven auto-selection would show it (always True here, since only
    applicable columns are listed).  The UI renders one section per category with a
    checkbox per column.

    Args:
        rows: The current table rows (the shown joints).
        manual_overrides: Optional map of column key -> forced visibility, scoped to
            the applicable columns.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        Ordered ``(category, [(spec, checked, auto_selected)])`` pairs, containing
        only the categories/columns applicable to the loaded articulation.
    """
    manual = manual_overrides or {}
    auto = auto_visible_column_keys(rows, backend, solver)
    order: list[str] = []
    groups: dict[str, list[tuple[GainColumnSpec, bool, bool]]] = {}
    for spec in GAIN_COLUMN_CATALOG:
        if spec.key not in auto:
            # Not applicable to the loaded articulation -> not offered in the menu.
            continue
        if spec.category not in groups:
            groups[spec.category] = []
            order.append(spec.category)
        checked = manual.get(spec.key, True)
        groups[spec.category].append((spec, checked, True))
    return [(category, groups[category]) for category in order]


def menu_has_mjc(rows: list[GainTableRow], mjc_map: dict | None) -> bool:
    """True when a shown joint authors MuJoCo-native (``mjc:*``) gains.

    Drives whether the picker offers a MuJoCo backend row.  Uses the read
    context's ``mjc_map`` (joint prim path -> MjcGainSource) intersected with the
    shown joints, so the result reflects the *loaded* articulation independently of
    the active solver: a MuJoCo-native source shown read-only under PhysX still
    makes the MuJoCo backend relevant.

    Args:
        rows: The current table rows (the shown joints).
        mjc_map: The read context's MuJoCo gain map, or None/empty when the asset
            authors no ``mjc:*`` gains.

    Returns:
        True when at least one shown joint has a ``mjc:*`` gain source.
    """
    if not mjc_map:
        return False
    paths = set(mjc_map.keys())
    for row in rows:
        try:
            path = row.entry.joint.GetPath().pathString
        except Exception:
            continue
        if path in paths:
            return True
    return False


def menu_backend_rows(has_mjc: bool, mujoco_active: bool) -> list[tuple[str, bool]]:
    """Return the picker's read-only BACKENDS rows for the loaded articulation.

    Only backends relevant to the asset's gain sources are offered:

    * **PhysX** is always relevant — every tunable joint has DriveAPI / physxJoint
      gains that are consumed under PhysX.
    * **MuJoCo** is offered only when a shown joint authors MuJoCo-native
      (``mjc:*``) gains (``has_mjc``), i.e. there is a MuJoCo-native source to
      inspect/edit.  A pure DriveAPI / physx asset therefore lists no MuJoCo row.

    Args:
        has_mjc: True when the loaded articulation has ``mjc:*`` gains (see
            :func:`menu_has_mjc`).
        mujoco_active: True when the active Newton solver is the MuJoCo solver.

    Returns:
        Ordered ``(label, checked)`` rows, where ``checked`` marks the active
        solver.  When only PhysX applies it is always marked checked (the sole
        relevant backend); when both apply the active one is checked.
    """
    if has_mjc:
        return [(BACKEND_PHYSX, not mujoco_active), (BACKEND_MUJOCO, mujoco_active)]
    return [(BACKEND_PHYSX, True)]


def mass_edit_row_indices(rows: list[GainTableRow], column_key: str) -> list[int]:
    """Return the row indices a mass edit for ``column_key`` would write.

    A mass edit applies a single value to every checked joint's cell in one
    column, but only where that cell is editable and applicable:

    * Gain columns (Kp/Kd/Ki): the row's gains must be editable (its viewed source
      is the active, consumed one) and, for Ki, the joint must have a PID actuator.
    * Advanced columns: the joint must author the column's attribute (non-blank,
      writable cell).

    Read-only cells (non-active-source gains, MuJoCo params under a non-MuJoCo
    solver) and blank cells are skipped.

    Args:
        rows: The current table rows (the checked joints).
        column_key: The column to mass-edit (``"kp"`` / ``"kd"`` / ``"ki"`` or an
            advanced column key).

    Returns:
        The indices (into ``rows``) of the rows that would be written.
    """
    indices: list[int] = []
    for row in rows:
        if column_key in (GAIN_COLUMN_KP, GAIN_COLUMN_KD, GAIN_COLUMN_KI):
            if not row.editable:
                continue
            value, attr, _label = row.gain_cell(column_key)
            # Ki only applies to PID joints; skip blank Ki cells.
            if column_key == GAIN_COLUMN_KI and value is None:
                continue
            # MuJoCo-native editable gains write through author_mjc_gains (no per-
            # field attr), so an editable MuJoCo row is still a valid mass target.
            if attr is None and not (
                row.resolved.source == gain_tuner.GainSource.MUJOCO and row.resolved.mjc_source is not None
            ):
                continue
            indices.append(row.index)
        else:
            value, attr = row.advanced_cell(column_key)
            if value is not None and attr is not None:
                indices.append(row.index)
    return indices


def selection_edit_row_indices(rows: list[GainTableRow], selected_indices, column_key: str) -> list[int]:
    """Return the rows a selection-driven edit for ``column_key`` should write.

    Editing a gain field applies the new value to every **selected** row whose
    cell in that column is editable and applicable — the intersection of the
    current multi-selection with :func:`mass_edit_row_indices`.  Read-only cells
    (non-active-source gains, MuJoCo params under a non-MuJoCo solver) and blank
    cells are skipped.  With a single selected row this collapses to a plain
    single-row edit.

    Args:
        rows: The current table rows.
        selected_indices: The row indices currently selected in the table.
        column_key: The edited column (``"kp"`` / ``"kd"`` / ``"ki"`` or advanced).

    Returns:
        The indices (into ``rows``) that should receive the edited value.
    """
    selected = {int(i) for i in selected_indices}
    return [index for index in mass_edit_row_indices(rows, column_key) if index in selected]


def detail_panel_mode(selected_count: int) -> str:
    """Return how the per-joint detail/info panel should render for a selection.

    The panel is specific to a single joint, so it is shown only when exactly one
    joint is selected.  Driving this off the selection *count* (not a focused-row
    fallback) means a multi-selection hides the panel even though one row is
    still "focused".

    Args:
        selected_count: Number of joints currently selected in the table.

    Returns:
        * :data:`DETAIL_PANEL_SINGLE` when exactly one joint is selected — show
          the full per-joint detail editor.
        * :data:`DETAIL_PANEL_MULTI` when two or more are selected — hide the
          panel and prompt the user to edit gains directly in the table.
        * :data:`DETAIL_PANEL_NONE` when none are selected — hide the panel and
          prompt the user to select a single joint.
    """
    if selected_count == 1:
        return DETAIL_PANEL_SINGLE
    if selected_count >= 2:
        return DETAIL_PANEL_MULTI
    return DETAIL_PANEL_NONE
