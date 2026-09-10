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

"""Presentation helpers for the gain table: display units and advanced columns.

These helpers exist purely to drive the Gain Settings table/detail views and are
consumed only by the UI extension (:mod:`isaacsim.robot_setup.gain_tuner.ui`):

* **Gain-field units** — the angular/linear unit label shown next to a joint's
  stiffness/damping fields, which depends on the active backend (PhysX consumes
  revolute gains in degrees; the Newton backend in radians).
* **Dynamic advanced columns** — the armature / max-force / max-joint-velocity /
  joint-friction columns that auto-appear in the hybrid gain table only when a
  shown joint authors the backing schema/attribute.  All but max-force are
  resolved per backend through
  :mod:`~isaacsim.robot_setup.gain_tuner.joint_schema_attrs`.
* **Per-backend value text** — the wording used when a joint's ``newton:*`` and
  ``physxJoint:*`` opinions differ, when the active backend's chain never reads a
  value that is authored, when the chain cannot be determined, or when the
  velocity limit the engine enforces differs from the one authored in USD.
  Divergence between the two backends is a legitimate authoring choice, so that
  wording is *informational*: it states each backend's value and which one is in
  effect, and never implies something needs fixing.

They read USD attributes directly and contain no Kit/``omni.ui`` imports, so they
can be unit-tested against in-memory stages without a running app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pxr
from isaacsim.robot_setup.gain_tuner import (
    JOINT_PARAM_SPECS,
    SCHEMA_LABELS,
    SCHEMA_NEWTON,
    SCHEMA_PHYSX,
    JointParamResolution,
    JointParamSpec,
    joint_param_attrs,
    resolve_joint_param,
)
from pxr import UsdPhysics

# USD / attribute access can fail with these when a prim, attribute, or layer is
# missing, invalid, a duck-typed fake (unit tests substitute plain objects), or
# holds an unexpected value type. We catch exactly this set at the defensive USD
# boundaries below so that any *unexpected* error type propagates.
_USD_ACCESS_ERRORS = (AttributeError, TypeError, ValueError, pxr.Tf.ErrorException)

# Newton backend label used across the gain tuner (matches BackendContext).
_NEWTON_BACKEND = "NewtonAPI"

# PhysX natural-frequency tuning mode relabels the stiffness/damping fields as
# natural frequency (Hz) and dimensionless damping ratio.  Purely presentational
# (used only by the detail editor), so they live in the UI extension.
NATURAL_FREQUENCY_LABEL = "Natural Frequency (Hz)"
DAMPING_RATIO_LABEL = "Damping Ratio"

# ---------------------------------------------------------------------------
# Gain-field units (backend-dependent) for the table / detail views
# ---------------------------------------------------------------------------

# Revolute-DOF stiffness/damping are consumed in DEGREES by PhysX and in RADIANS
# by the Newton backend (Featherstone / MuJoCo / etc.).  Prismatic (linear) DOFs
# use the stage's linear unit regardless of backend.
ANGLE_UNIT_DEGREES = "deg"
ANGLE_UNIT_RADIANS = "rad"
DEFAULT_LINEAR_UNIT = "m"


def gain_angle_unit(backend: str) -> str:
    """Return the angular unit revolute-DOF gains are expressed in for a backend.

    PhysX consumes ``UsdPhysics.DriveAPI`` stiffness/damping in degrees; the
    Newton backend (any solver, including MuJoCo) consumes them in radians.

    Args:
        backend: The active backend label (``"PhysX"`` or ``"NewtonAPI"``).

    Returns:
        ``"deg"`` for PhysX, ``"rad"`` for the Newton backend.
    """
    return ANGLE_UNIT_RADIANS if str(backend) == _NEWTON_BACKEND else ANGLE_UNIT_DEGREES


def gain_dof_unit(is_angular: bool, backend: str, linear_unit: str = DEFAULT_LINEAR_UNIT) -> str:
    """Return the display unit for a DOF's stiffness/damping fields.

    Revolute (angular) DOFs use the backend-dependent angle unit
    (:func:`gain_angle_unit`); prismatic (linear) DOFs use ``linear_unit``.

    Args:
        is_angular: True for revolute / D6-rotational DOFs, False for prismatic.
        backend: The active backend label (``"PhysX"`` or ``"NewtonAPI"``).
        linear_unit: The stage's linear length unit for prismatic DOFs.

    Returns:
        The unit token to show next to the stiffness/damping fields.
    """
    return gain_angle_unit(backend) if is_angular else linear_unit


# ---------------------------------------------------------------------------
# Dynamic advanced-parameter columns (auto-appear per selected-joint schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvancedGainColumn:
    """A dynamic advanced-parameter column shown in the hybrid gain table.

    A column is surfaced only when at least one selected joint actually authors
    the backing schema/attribute (see :func:`visible_advanced_gain_columns`); a
    joint that lacks the attribute renders a blank, uneditable cell.
    """

    key: str
    #: Stable identifier used as the column key in the table model.

    label: str
    #: User-facing column header text.

    attr_name: str | None = None
    #: USD attribute read directly from the joint prim (None for the drive
    #: max-force column and for dual-schema columns, which resolve through
    #: ``UsdPhysics.DriveAPI`` and ``param_spec`` respectively).

    is_drive_max_force: bool = False
    #: True for the special column read from the joint drive's ``maxForce``.

    param_spec: JointParamSpec | None = None
    #: Set for parameters that exist on more than one joint schema.  Reads resolve
    #: through the active backend's chain; edits author only that backend's schema
    #: (see :mod:`~isaacsim.robot_setup.gain_tuner.joint_schema_attrs`).

    authored_only: bool = False
    #: When True the column is surfaced (and its cell rendered) only for joints
    #: that have an *explicitly authored* value, not merely the schema fallback.
    #: Used for parameters whose schema default is always present and therefore
    #: non-informative (e.g. the DriveAPI ``maxForce`` default of ``+inf``).


# The advanced params the per-joint detail editor already exposes (armature /
# drive max-force / max joint velocity / joint friction).  Each auto-appears only
# when a selected joint authors it.
# The DriveAPI ``maxForce`` attribute is part of every drive joint's schema with
# a ``+inf`` fallback, so it is surfaced only when a joint authors an explicit
# value (``authored_only``); the dual-schema columns are already gated by their
# attribute existing on the prim (an applied NewtonJointAPI or PhysxJointAPI),
# which is a meaningful per-joint signal, so they keep the schema-present rule.
_PARAM_SPECS = {spec.key: spec for spec in JOINT_PARAM_SPECS}

ADVANCED_GAIN_COLUMNS: tuple[AdvancedGainColumn, ...] = (
    AdvancedGainColumn(key="armature", label="Armature", param_spec=_PARAM_SPECS["armature"]),
    AdvancedGainColumn(key="max_force", label="Max Force", is_drive_max_force=True, authored_only=True),
    AdvancedGainColumn(
        key="max_joint_velocity", label="Max Joint Velocity", param_spec=_PARAM_SPECS["max_joint_velocity"]
    ),
    AdvancedGainColumn(key="joint_friction", label="Joint Friction", param_spec=_PARAM_SPECS["joint_friction"]),
)


def advanced_gain_column(column_key: str) -> AdvancedGainColumn | None:
    """Return the advanced column registered under ``column_key``, or None.

    Args:
        column_key: The :attr:`AdvancedGainColumn.key` to look up.

    Returns:
        The matching column, or None when the key is not an advanced column.
    """
    for column in ADVANCED_GAIN_COLUMNS:
        if column.key == column_key:
            return column
    return None


def _drive_max_force_attr(joint: object, drive_axis: object) -> pxr.Usd.Attribute | None:
    """Return the joint drive's ``maxForce`` attribute, or None when unavailable."""
    try:
        axis = drive_axis
        if not axis:
            if joint.IsA(UsdPhysics.RevoluteJoint):
                axis = "angular"
            elif joint.IsA(UsdPhysics.PrismaticJoint):
                axis = "linear"
        if not axis:
            return None
        api = UsdPhysics.DriveAPI(joint, axis)
        if not api:
            return None
        return api.GetMaxForceAttr()
    except _USD_ACCESS_ERRORS:
        return None


def advanced_gain_param_attr(joint: object, drive_axis: object, column: AdvancedGainColumn) -> pxr.Usd.Attribute | None:
    """Resolve the USD attribute backing an advanced column for a joint.

    Args:
        joint: The joint prim.
        drive_axis: Optional D6 drive-axis token.
        column: The :class:`AdvancedGainColumn` to resolve.

    Returns:
        The live, valid attribute, or None when the joint does not author the
        column's schema/parameter (the cell then renders blank and uneditable).
    """
    if column.is_drive_max_force:
        attr = _drive_max_force_attr(joint, drive_axis)
    elif column.param_spec is not None:
        # Either joint schema being present means the parameter is meaningful for
        # this joint, which is what gates the column; which one an edit lands on
        # is decided per backend by ``author_joint_param``.
        newton_attr, physx_attr = joint_param_attrs(joint, column.param_spec)
        attr = newton_attr if newton_attr is not None else physx_attr
    else:
        try:
            attr = joint.GetAttribute(column.attr_name)
        except _USD_ACCESS_ERRORS:
            return None
    return attr if (attr is not None and attr.IsValid()) else None


def read_advanced_gain_cell(
    joint: object, drive_axis: object, column: AdvancedGainColumn, backend: str, solver: str = ""
) -> tuple[float | None, pxr.Usd.Attribute | None]:
    """Read an advanced-parameter cell value + editable attribute for a joint.

    Args:
        joint: The joint prim.
        drive_axis: Optional D6 drive-axis token.
        column: The :class:`AdvancedGainColumn` to read.
        backend: The active backend label, which selects the resolver chain a
            multi-schema column resolves through.  Required rather than defaulted:
            the answer differs per backend, so a default would report PhysX's
            value for whichever engine the caller did not name.
        solver: The running Newton solver token.  Ignored under PhysX.

    Returns:
        A ``(value, attr)`` tuple.  ``value`` is None (and the cell blank /
        uneditable) when the joint lacks the column's attribute, or when the
        column is ``authored_only`` and the joint has no explicitly authored
        value (only the schema fallback); otherwise it is the value the cell
        shows and ``attr`` the live attribute that made the column meaningful.
        For multi-schema columns an edit must go through
        :func:`~isaacsim.robot_setup.gain_tuner.joint_schema_attrs.author_joint_param`
        rather than writing that attribute directly, so it lands on the active
        backend's schema and leaves the other backend's value alone.
    """
    if column.param_spec is not None:
        resolution = resolve_joint_param(joint, column.param_spec, backend, solver)
        return advanced_cell_value(resolution, joint, column.param_spec), advanced_gain_param_attr(
            joint, drive_axis, column
        )

    attr = advanced_gain_param_attr(joint, drive_axis, column)
    if attr is None:
        return None, None
    if column.authored_only:
        # Surface the cell only when the joint authors an explicit value; the
        # schema fallback (e.g. DriveAPI ``maxForce`` = +inf) is non-informative,
        # so leave the cell blank/uneditable until authored elsewhere.
        try:
            if not attr.HasAuthoredValue():
                return None, None
        except _USD_ACCESS_ERRORS:
            return None, None
    value = attr.Get()
    if value is None:
        # Attribute is present (schema applied) but unauthored: still editable,
        # shown as 0.0 rather than blank so the user can author a value.
        return 0.0, attr
    try:
        return float(value), attr
    except (TypeError, ValueError):
        return None, None


UNLIMITED_TEXT = "unlimited"
"""The one word this panel uses for a parameter that imposes no limit.

Used for the field's own text, for the value inside a sentence, and for the table
cell, so the field and the line under it never name the same state two ways."""

UNLIMITED_FIELD_FORMAT = UNLIMITED_TEXT
"""``omni.ui`` drag-field ``format`` used for a parameter that enforces no limit.

A ``printf`` format with no conversion specifier, so the field renders this word
in place of a number.  No number can honestly stand in for "unlimited" -- and
zero, the number the field would otherwise show, reads as the exact opposite --
while a plain label would cost the cell its write path.  Typing over the field
still authors a real limit.
"""

DEFAULT_FIELD_SUFFIX = " (default)"
"""Suffix marking a field or cell whose value is an engine default, not authored.

Muting the text is the only other signal, and a 26% grey step is not one a user
can read in isolation: an unauthored ``0.0`` looks exactly like an authored one.
The word travels with the number into the table, where the explanatory line under
the field cannot follow.  ASCII only, since ``omni.ui`` renders non-ASCII as
``?``, which rules out an icon or a glyph.

Deliberately does not name the backend.  Only one engine is ever active, and the
header states which, so ``"(default)"`` is unambiguous where it is read; naming it
is not worth the width.  ``"0.1 (Newton default)"`` is 20 characters against the
112 px advanced cell at :data:`~isaacsim.robot_setup.gain_tuner.ui.style.FONT_SIZE`
14, so it clips -- and it clips from the right, taking the marker it was added for
and leaving a number that no longer reads as a default.  The backend is named in
:attr:`AdvancedParamDisplay.note` instead, which reaches the table as the cell's
tooltip and the detail panel as the line under the field."""

DEFAULT_FIELD_FORMAT = "%g" + DEFAULT_FIELD_SUFFIX
"""``omni.ui`` drag-field ``format`` for an unauthored parameter's engine default.

``%g`` rather than the field's usual fixed-point form: these are documented
defaults such as ``0.1``, and trailing zeros suggest a precision nobody chose."""

UNLIMITED_DEFAULT_FIELD_FORMAT = UNLIMITED_TEXT + DEFAULT_FIELD_SUFFIX
"""``omni.ui`` drag-field ``format`` for an unauthored limit the engine leaves open."""


@dataclass(frozen=True)
class AdvancedParamDisplay:
    """How one advanced joint parameter should render, authored or not.

    An unauthored parameter is not worth zero: Newton simulates its
    ``ModelBuilder`` armature default rather than nothing, and neither backend
    clamps an unauthored velocity limit at all.  Showing zero states a value no
    engine uses, and the field is live, so nudging it authors that fiction.  This
    separates *what the simulation is doing* from *what the stage says*, so an
    unauthored parameter can show the engine's own default, marked as a default,
    and stay editable.

    Built by :func:`advanced_param_display`.
    """

    value: float | None
    """The number to seed the field with, or None when there is no number.

    None means either that the parameter is unlimited (see :attr:`unlimited`) or
    that nothing can be stated about it -- an undetermined solver, an unsupported
    backend, or a parameter that does not apply to the joint.  A field seeded with
    None must still be editable whenever :attr:`editable`.
    """

    authored: bool
    """Whether :attr:`value` is what the stage says, rather than a placeholder.

    False for an engine default standing in for an unauthored parameter, which
    must be shown muted and marked so it does not read as somebody's opinion.
    """

    unlimited: bool
    """Whether the parameter in effect imposes no limit.

    Rendered as :data:`UNLIMITED_FIELD_FORMAT`, never as a number.
    """

    editable: bool
    """Whether an edit to this field can be authored on a known schema.

    False when the parameter does not apply to the joint, or when the active
    backend has no schema this extension knows how to write -- in which case the
    field must be disabled rather than silently writing somebody else's schema.
    """

    note: str
    """One sentence explaining a placeholder, for a tooltip, or ``""``.

    States what the engine does with the unauthored parameter, and names the
    engine, so the muted number is not mistaken for an authored one.  This is the
    only place the backend is named: the cell has room for
    :data:`DEFAULT_FIELD_SUFFIX` and the number, and nothing more.
    """

    @property
    def field_value(self) -> float:
        """The float to construct the field's model with.

        Zero when there is no number, which is safe only because the field then
        renders :data:`UNLIMITED_FIELD_FORMAT` or is disabled outright rather than
        showing the digit.
        """
        return 0.0 if self.value is None else float(self.value)

    @property
    def field_format(self) -> str:
        """The ``omni.ui`` drag-field ``format``, or ``""`` to leave it default.

        Carries :data:`DEFAULT_FIELD_SUFFIX` for an unauthored parameter, so the
        cell itself says the number is the engine's and not the user's.  That has
        to be in the format string rather than a neighbouring label: the table has
        room for the field and nothing else.
        """
        if self.unlimited:
            return UNLIMITED_FIELD_FORMAT if self.authored else UNLIMITED_DEFAULT_FIELD_FORMAT
        if self.value is not None and not self.authored:
            return DEFAULT_FIELD_FORMAT
        return ""

    @property
    def blank(self) -> bool:
        """Whether the cell has nothing to render and must be left blank.

        True when there is neither a number nor an "unlimited" to state: the
        parameter does not apply, the resolver chain is undetermined, or the
        backend is unsupported.
        """
        return self.value is None and not self.unlimited


def _param_applies(resolution: JointParamResolution, joint: object, spec: JointParamSpec) -> bool:
    """Return whether ``spec`` is meaningful for ``joint`` at all.

    Either joint schema being present is enough; which one an edit lands on is
    decided per backend.  A joint with neither has no such parameter to show.
    """
    newton_attr, physx_attr = joint_param_attrs(joint, spec)
    return newton_attr is not None or physx_attr is not None or resolution.any_authored


def advanced_param_display(
    resolution: JointParamResolution,
    joint: object,
    spec: JointParamSpec,
    engine_value: float | None = None,
) -> AdvancedParamDisplay:
    """Decide how an advanced joint parameter should render.

    Args:
        resolution: The parameter's resolution under the active backend.
        joint: The joint prim, used to decide whether the parameter applies at all.
        spec: The parameter being shown.
        engine_value: What the running engine reports for this parameter, when it
            can be queried (:meth:`GainTuner.get_dof_engine_armature`,
            :meth:`GainTuner.get_dof_effective_max_velocity`).  Used only for an
            *unauthored* parameter, where it is preferred over the documented
            config default because it is measured rather than assumed.
            ``math.inf`` for a limit the engine does not enforce.

    Returns:
        The :class:`AdvancedParamDisplay` for the field or cell.

    Example:

    .. code-block:: python

        >>> display = advanced_param_display(resolution, joint, spec)  # doctest: +NO_CHECK
        >>> display.authored, display.value, display.field_format  # doctest: +NO_CHECK
        (False, 0.1, '%g (default)')
    """
    applies = _param_applies(resolution, joint, spec)
    if not applies:
        return AdvancedParamDisplay(None, False, False, False, "")
    if not resolution.backend_supported:
        return AdvancedParamDisplay(
            None,
            False,
            False,
            False,
            "The active physics engine is not one this extension can author joint parameters for, "
            "so this value cannot be read or edited.",
        )
    if resolution.effective_value is not None:
        return AdvancedParamDisplay(resolution.effective_value, True, False, True, "")
    if resolution.unlimited:
        return AdvancedParamDisplay(
            None,
            True,
            True,
            True,
            f"{spec.label} is authored as {UNLIMITED_TEXT}. Type a value here to author a limit.",
        )
    if not resolution.determined:
        # Which schema the running solver reads is unknown, so neither the stage's
        # value nor a default can be attributed to it.
        return AdvancedParamDisplay(None, False, False, True, "")

    backend_label = resolution.backend_label
    default = engine_value if engine_value is not None else resolution.engine_default
    if default is None:
        return AdvancedParamDisplay(None, False, False, True, "")
    measured = engine_value is not None
    source = "is simulating with" if measured else "applies its default"
    if default == float("inf"):
        return AdvancedParamDisplay(
            None,
            False,
            True,
            True,
            f"Nothing is authored here, so {backend_label} leaves this {UNLIMITED_TEXT}. "
            "Type a value here to author a limit.",
        )
    return AdvancedParamDisplay(
        default,
        False,
        False,
        True,
        f"Nothing is authored here; {backend_label} {source} {default:g}. Edit this field to author a value.",
    )


def read_advanced_param_display(
    joint: object,
    column: AdvancedGainColumn,
    backend: str,
    solver: str = "",
    engine_value: float | None = None,
) -> AdvancedParamDisplay | None:
    """Return how a per-backend advanced column should render for one joint.

    Args:
        joint: The joint prim.
        column: The column to read.
        backend: The active backend label.  Required: which schema a value is read
            from depends on it, so a default here would answer for PhysX on behalf
            of whichever engine the caller forgot to name.
        solver: The running Newton solver token.  Ignored under PhysX.
        engine_value: What the running engine reports for this parameter, or None.

    Returns:
        The :class:`AdvancedParamDisplay`, or None for a column that is a plain
        single attribute rather than a per-backend parameter (the DriveAPI max-force
        column), which has no backend resolution to describe.
    """
    if column.param_spec is None:
        return None
    resolution = resolve_joint_param(joint, column.param_spec, backend, solver)
    return advanced_param_display(resolution, joint, column.param_spec, engine_value)


def advanced_cell_value(resolution: JointParamResolution, joint: object, spec: JointParamSpec) -> float | None:
    """Return the number a multi-schema advanced cell should show.

    Deprecated in favour of :func:`advanced_param_display`, which distinguishes an
    engine default from an authored value and "unlimited" from "unknown".  Kept
    because a bare float is all the table's column-visibility check needs.

    Args:
        resolution: The parameter's resolution under the active backend.
        joint: The joint prim (used to decide whether the parameter applies at
            all).
        spec: The parameter being shown.

    Returns:
        :attr:`AdvancedParamDisplay.value`, which is None whenever the cell has no
        number to state -- unlimited, undetermined, unsupported backend, or not
        applicable to the joint.
    """
    return advanced_param_display(resolution, joint, spec).value


def visible_advanced_gain_columns(joint_specs: list, backend: str, solver: str = "") -> list[AdvancedGainColumn]:
    """Return the advanced columns to show for a set of selected joints.

    A column is included when **any** joint in ``joint_specs`` would render a
    non-blank cell for it, and omitted when none would — the auto-appear /
    auto-hide behavior of the hybrid table.  This keeps visibility in lockstep
    with :func:`read_advanced_gain_cell`, so an ``authored_only`` column (e.g.
    the DriveAPI ``maxForce`` column, whose schema fallback ``+inf`` is always
    present) appears only when a joint authors an explicit value.

    Args:
        joint_specs: Iterable of ``(joint, drive_axis)`` tuples for the selected
            joints (the checked rows shown in the table).
        backend: The active backend label.  Required, for the reason given on
            :func:`read_advanced_param_display`: an unsupported backend renders no
            cell at all, so defaulting it would show PhysX's columns for an engine
            whose values the panel cannot read.
        solver: The running Newton solver token.

    Returns:
        The visible :class:`AdvancedGainColumn` values in registry order.
    """
    visible: list[AdvancedGainColumn] = []
    for column in ADVANCED_GAIN_COLUMNS:
        for joint, drive_axis in joint_specs:
            value, _attr = read_advanced_gain_cell(joint, drive_axis, column, backend, solver)
            if value is not None:
                visible.append(column)
                break
    return visible


# ---------------------------------------------------------------------------
# Per-backend value wording (shared by the detail panel and the table)
# ---------------------------------------------------------------------------

# Angular velocity limits are stored in degrees per second on both joint schemas
# and the engine's effective limit is converted to match, so the unit shown next
# to a velocity value only depends on the DOF type.  The degree sign is the one
# non-ASCII glyph the app's font has (see TestReadoutTextIsRenderable).
ANGULAR_VELOCITY_UNIT = "\u00b0/s"
LINEAR_VELOCITY_UNIT = "m/s"


def format_param_value(value: float | None) -> str:
    """Format an advanced-parameter value for an inline message or tooltip.

    Args:
        value: The value to format, or None.

    Returns:
        A compact ASCII representation with trailing zeros trimmed,
        :data:`UNLIMITED_TEXT` for an infinite value (``"inf"`` is a float repr,
        not a statement about the joint, and this is the same word the field
        itself shows), or ``"none"`` when the value is missing or not a number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "none"
    if math.isinf(number):
        return UNLIMITED_TEXT
    return f"{number:g}"


def authored_values_text(resolution: JointParamResolution) -> str:
    """List every schema that carries an authored value for a parameter.

    Args:
        resolution: The parameter's resolution under the active backend.

    Returns:
        An ASCII clause such as ``"newton:armature is 0.25, physxJoint:armature is
        0.75"``, or ``""`` when no schema carries an authored value.
    """
    parts = [
        f"{resolution.spec.attr_for_schema(schema)} is {format_param_value(value)}"
        for schema, value in resolution.authored.items()
        if value is not None
    ]
    return ", ".join(parts)


def effective_value_text(resolution: JointParamResolution) -> str:
    """State which value the active backend has in effect, and where it comes from.

    Args:
        resolution: The parameter's resolution under the active backend.

    Returns:
        A single ASCII sentence.  Says the value cannot be determined when the
        running solver decides it and the solver is unknown, rather than naming
        one, and ``""`` when the active engine's joint schema is unknown -- there
        is no chain to describe, and describing one anyway reported the *solver*
        as undetermined for what is really an unrecognized *engine*.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner.ui.gain_display import effective_value_text

        >>> effective_value_text(resolution)  # doctest: +NO_CHECK
        'PhysX uses physxJoint:armature (0.75).'
    """
    if not resolution.backend_supported:
        return ""
    engine = resolution.backend_label
    if not resolution.determined:
        remaining = sorted(
            {
                SCHEMA_LABELS.get(schema, schema)
                for chain in resolution.candidate_chains
                for schema in chain
                if schema not in resolution.chain
            }
        )
        reads = f" whether {engine} reads {', '.join(remaining)}" if remaining else " which schema is in effect"
        return f"The running {engine} solver could not be determined, so{reads} cannot be determined either."
    if resolution.effective_schema is None:
        chain = ", ".join(SCHEMA_LABELS.get(s, s) for s in resolution.chain)
        return f"Nothing is authored on the schemas {engine} reads ({chain}), so its own default applies."
    attr_name = resolution.spec.attr_for_schema(resolution.effective_schema)
    if resolution.unlimited:
        return f"{engine} reads {attr_name} as {UNLIMITED_TEXT}."
    return f"{engine} uses {attr_name} ({format_param_value(resolution.effective_value)})."


def reportable_unread_schemas(resolution: JointParamResolution) -> tuple[str, ...]:
    """Return the ignored authored values that change what the engine does.

    An authored value the active backend's chain never reads only matters when it
    differs from the value in effect: a schema holding the same number is ignored
    without consequence, and saying so would be noise on every joint whose
    backends happen to agree.

    Args:
        resolution: The parameter's resolution under the active backend.

    Returns:
        The unread resolver tokens worth reporting, in
        :attr:`JointParamResolution.authored` order.
    """
    return tuple(
        schema
        for schema in resolution.unread_schemas
        if resolution.effective_value is None or resolution.authored[schema] != resolution.effective_value
    )


def unread_values_text(resolution: JointParamResolution) -> str:
    """Name authored values the active backend's resolver chain never reads.

    This is how a PhysX-only joint friction surfaces under Newton: Newton's PhysX
    resolver declares no friction key, so the value is authored, simulated by
    PhysX, and ignored by Newton.

    Args:
        resolution: The parameter's resolution under the active backend.

    Returns:
        A single ASCII sentence, or ``""`` when every authored value either takes
        part in the chain or matches the value in effect, or when the active
        engine's joint schema is unknown.  In that last case every authored value
        counts as unread, because the chain is empty -- but "not read by" is a
        claim about an engine whose schema nobody here knows, so it is not made.
    """
    if not resolution.backend_supported:
        return ""
    unread = reportable_unread_schemas(resolution)
    if not unread:
        return ""
    engine = resolution.backend_label
    named = ", ".join(
        f"{resolution.spec.attr_for_schema(schema)} ({format_param_value(resolution.authored[schema])})"
        for schema in unread
    )
    verb = "is" if len(unread) == 1 else "are"
    return f"{named} {verb} not read by {engine}."


def fallback_source_text(resolution: JointParamResolution) -> str:
    """State that the value in effect came from the schema an edit would not write.

    The quietest case in the panel: the active backend authors nothing here, so
    its resolver chain reaches the other backend's half and simulates that.  The
    number looks like any other, and the first edit moves it, so both facts are
    said out loud.

    Args:
        resolution: The parameter's resolution under the active backend.

    Returns:
        A single ASCII sentence pair, or ``""`` when the value in effect is
        authored on the backend's own schema (see
        :attr:`~isaacsim.robot_setup.gain_tuner.JointParamResolution.resolved_from_fallback_schema`),
        or when either attribute has no name to quote.
    """
    if not resolution.resolved_from_fallback_schema:
        return ""
    spec = resolution.spec
    own_attr = resolution.write_attr
    read_attr = spec.attr_for_schema(resolution.effective_schema)
    if own_attr is None or read_attr is None:
        # Nothing nameable to contrast; the caller falls back to naming the
        # authored values and the winner, which needs no attribute of its own.
        return ""
    engine = resolution.backend_label
    value = UNLIMITED_TEXT if resolution.unlimited else format_param_value(resolution.effective_value)
    return (
        f"Nothing is authored on {own_attr}, so {engine} falls back to {read_attr} ({value}). "
        f"An edit here authors {own_attr}."
    )


def joint_param_info_text(resolution: JointParamResolution, display: AdvancedParamDisplay | None = None) -> str:
    """Describe a parameter whose per-backend values are worth pointing out.

    Newton and PhysX are tuned independently, so two different values are a
    legitimate authoring choice rather than a problem: this states each backend's
    value and which one is in effect, and never suggests a change.

    Args:
        resolution: The parameter's resolution under the active backend.
        display: How the field will render, when the caller also shows
            :attr:`AdvancedParamDisplay.note`.  That note already states what an
            unauthored parameter is worth, so passing it here drops the sentence
            that would say it a second time -- the two together ran to 250
            characters over one field.

    Returns:
        An ASCII sentence, or ``""`` when there is nothing left to point out --
        the backends agree, the value in effect is authored on the backend's own
        schema, every authored value is read, and the chain is known.  Also ``""``
        when the active engine's joint schema is unknown: the value cannot be read
        at all then, which :attr:`AdvancedParamDisplay.note` states, and every
        sentence this would otherwise build describes a resolver chain that does
        not exist.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner.ui.gain_display import joint_param_info_text

        >>> joint_param_info_text(resolution)  # doctest: +NO_CHECK
        'newton:friction is 0.25, physxJoint:jointFriction is 0.75. PhysX uses physxJoint:jointFriction (0.75).'
    """
    if not resolution.backend_supported:
        return ""
    if (
        resolution.determined
        and not resolution.backends_diverge
        and not resolution.resolved_from_fallback_schema
        and not reportable_unread_schemas(resolution)
    ):
        return ""
    sentences = []
    fallback = fallback_source_text(resolution)
    if fallback:
        sentences.append(fallback)
    else:
        # With nothing authored in the chain, the placeholder note the caller
        # renders already names the engine default, so restating the chain here
        # would repeat it -- and both openings were "nothing is authored".
        note_states_default = bool(display is not None and display.note and resolution.effective_schema is None)
        if not note_states_default:
            authored = authored_values_text(resolution)
            if authored:
                sentences.append(f"{authored}.")
            sentences.append(effective_value_text(resolution))
    unread = unread_values_text(resolution)
    if unread:
        sentences.append(unread)
    return " ".join(s for s in sentences if s)


def joint_param_infos(resolutions: list) -> dict:
    """Map parameter key -> informational text for the resolutions worth reporting.

    Args:
        resolutions: :class:`JointParamResolution` values for one joint.

    Returns:
        Map of :attr:`JointParamSpec.key` -> text, omitting parameters with
        nothing to point out.
    """
    infos = {}
    for resolution in resolutions or []:
        text = joint_param_info_text(resolution)
        if text:
            infos[resolution.spec.key] = text
    return infos


def max_velocity_engine_text(usd_value: float | None, engine_value: float | None, is_angular: bool) -> str:
    """Describe a velocity limit that the running engine does not agree with.

    The gain tests scale their velocity sweep by the engine's effective limit, so
    a joint whose USD opinion differs is swept at a speed the panel does not show.
    This is the sentence that says so.

    Args:
        usd_value: The limit the active backend's chain resolves, in schema units.
            ``math.inf`` when nothing limits the joint, which reads as
            "unlimited" rather than a number -- the case worth reporting, since
            the panel then shows an unlimited joint the engine is in fact
            clamping.
        engine_value: The limit the engine enforces, in the same units (from
            :meth:`~isaacsim.robot_setup.gain_tuner.GainTuner.get_dof_effective_max_velocity`).
        is_angular: True for rotational DOFs (degrees per second).

    Returns:
        A single ASCII sentence naming the enforced limit and the sweep it drives.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner.ui.gain_display import max_velocity_engine_text

        >>> max_velocity_engine_text(180.0, 90.0, True)  # doctest: +NO_CHECK
        'The running engine enforces 90 \\u00b0/s, not the 180 \\u00b0/s shown here. Velocity tests sweep to the enforced limit.'
    """
    unit = ANGULAR_VELOCITY_UNIT if is_angular else LINEAR_VELOCITY_UNIT
    enforced = _velocity_phrase(engine_value, unit)
    shown = _velocity_phrase(usd_value, unit)
    # The absence of a limit needs its own clause: "not the unlimited shown here"
    # is not a sentence, and it is the case most worth reading.
    contrast = f"while this joint reads as {shown} here" if shown == UNLIMITED_TEXT else f"not the {shown} shown here"
    return f"The running engine enforces {enforced}, {contrast}. Velocity tests sweep to the enforced limit."


def _velocity_phrase(value: float | None, unit: str) -> str:
    """Render a velocity limit as a noun phrase that reads in a sentence.

    A limit and the absence of one need different grammar: "107 deg/s" takes a
    unit, "unlimited" does not.
    """
    text = format_param_value(value)
    return text if text in (UNLIMITED_TEXT, "none") else f"{text} {unit}"
