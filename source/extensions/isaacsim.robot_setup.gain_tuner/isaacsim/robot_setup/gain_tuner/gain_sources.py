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

"""Gain-source detection for multi-backend gain tuning (core, UI-independent).

A robot joint can have its Kp/Kd(/Ki) gains authored by more than one source:

* **PhysicsDrive** — ``UsdPhysics.DriveAPI`` stiffness/damping on the joint drive.
  Consumed by the active physics backend **unless** a Newton actuator is actively
  driving the joint: Newton's application-side ``ArticulationActuators`` runtime
  zeros the joint's DriveAPI gains while it is active, so the actuator (not the
  DriveAPI) determines the motion.
* **Actuator** — a ``NewtonActuator`` prim carrying a PD/PID controller applied
  schema (``NewtonPDControlAPI`` / ``NewtonPIDControlAPI``) with ``newton:kp`` /
  ``newton:kd`` / ``newton:ki`` attributes, linked to the single DOF it drives via
  the ``newton:targets`` relationship (strictly one target DOF per actuator).
  The ``NewtonNeuralControlAPI`` variant has no Kp/Kd/Ki and is not a tunable gain
  source.  Newton actuators are **backend-independent**: the runtime evaluates
  them application-side, so an authored actuator drives its joint under either
  PhysX or Newton.  Actuator prims are discovered by traversing the whole
  articulation subtree; ``{articulation_root}/Actuators`` is only the convention
  used by the ``add_actuator`` helper, not a requirement.
* **MuJoCo-native** — ``mjc:gainPrm`` / ``mjc:biasPrm`` actuator parameters
  (authored on a ``MjcActuator`` prim or directly on the joint).  Consumed as the
  active source only when the Newton **MuJoCo** solver is running (and no Newton
  actuator is present); otherwise shown read-only for sim-to-sim inspection.

This module reads these sources directly from USD (the interim path until the
``isaacsim.core.experimental.actuators`` introspection API lands), makes a
best-effort static guess at which source the active backend/solver consumes, and
returns a resolved view the UI can render.  It contains no UI/Kit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import pxr
from pxr import Sdf, Tf, Usd, UsdPhysics

from .joint_schema_attrs import NEWTON_SOLVER_TYPES, SOLVER_MUJOCO

# USD / attribute / layer access can fail with these when a prim, attribute, or
# layer is missing, invalid, a duck-typed fake (unit tests substitute plain
# objects), or holds an unexpected value type. We catch exactly this set at the
# defensive USD boundaries below so that any *unexpected* error type propagates
# and surfaces as a real failure instead of being silently swallowed.
_USD_ACCESS_ERRORS = (AttributeError, TypeError, ValueError, Tf.ErrorException)

from .joint_drive_attrs import (
    JointDriveMode,
    get_damping_attr,
    get_stiffness_attr,
)

# Applied-schema tokens for Newton actuator controllers and the relationship that
# links an actuator prim to the single joint DOF it drives.  The ``NewtonActuator``
# prim type carries one of these PD/PID applied schemas; the ``NewtonNeuralControlAPI``
# variant has no Kp/Kd/Ki and is intentionally not treated as a gain source.
_NEWTON_ACTUATOR_TYPE = "NewtonActuator"
_NEWTON_PD_SCHEMA = "NewtonPDControlAPI"
_NEWTON_PID_SCHEMA = "NewtonPIDControlAPI"
_NEWTON_NEURAL_SCHEMA = "NewtonNeuralControlAPI"
_ACTUATOR_TARGETS_REL = "newton:targets"


class GainSource(IntEnum):
    """The authored source of a joint's editable gains."""

    NONE = 0
    """No PhysicsDrive gains and no actuator authored."""

    PHYSICS_DRIVE = 1
    """USD ``PhysicsDriveAPI`` Kp/Kd authored on the joint drive."""

    ACTUATOR = 2
    """A ``NewtonActuator`` prim with a PD/PID controller applied schema
    (``NewtonPDControlAPI`` / ``NewtonPIDControlAPI``).  Backend-independent: an
    authored actuator drives its joint under either PhysX or Newton."""

    MUJOCO = 3
    """MuJoCo-native actuator gains (``mjc:gainPrm`` / ``mjc:biasPrm``).

    Authored on the composed stage/variant independently of the active physics
    engine.  Becomes the active, editable source when the Newton **MuJoCo** solver
    is running (and no Newton actuator drives the joint); otherwise shown
    read-only for sim-to-sim inspection (PhysX and the Newton non-MuJoCo solvers
    consume ``PhysicsDrive`` or a Newton actuator, not the MuJoCo-native params)."""


_SOURCE_LABELS = {
    GainSource.NONE: "",
    GainSource.PHYSICS_DRIVE: "Physics",
    GainSource.ACTUATOR: "Actuator",
    GainSource.MUJOCO: "MuJoCo",
}

# ---------------------------------------------------------------------------
# User-facing gain field labels (dynamic per gain source)
# ---------------------------------------------------------------------------

# PhysX / ``UsdPhysics.DriveAPI`` drives expose position/velocity gains as
# stiffness and damping.
STIFFNESS_LABEL = "Stiffness"
DAMPING_LABEL = "Damping"

# Newton actuators (``NewtonPDControlAPI`` / ``NewtonPIDControlAPI``) expose
# proportional / derivative / integral gains as Kp / Kd / Ki.
KP_LABEL = "Kp"
KD_LABEL = "Kd"
KI_LABEL = "Ki"

# MuJoCo-native gains are shown for inspection as the effective stiffness/damping
# they encode (derived from ``mjc:gainPrm`` / ``mjc:biasPrm``), tagged as MuJoCo.
MJC_STIFFNESS_LABEL = "Stiffness (mjc)"
MJC_DAMPING_LABEL = "Damping (mjc)"


@dataclass
class GainLabels:
    """User-facing labels for a joint's editable gain fields.

    The labels are chosen from the resolved :class:`GainSource` so the detail
    panel names the fields the way the active backend consumes them (stiffness /
    damping for PhysX ``DriveAPI``; Kp / Kd / Ki for a Newton actuator).
    """

    kp_label: str
    """Label for the proportional / stiffness field."""

    kd_label: str
    """Label for the derivative / damping field."""

    ki_label: str | None
    """Label for the integral field, or None when the source has no Ki
    (PhysicsDrive and Newton PD control)."""


def gain_labels_for_source(source: GainSource, *, is_pid: bool = False) -> GainLabels:
    """Return the user-facing field labels for a resolved gain source.

    Args:
        source: The resolved :class:`GainSource` whose gains are shown.
        is_pid: True when the source is a Newton PID actuator (exposes Ki).

    Returns:
        A :class:`GainLabels` naming the Kp/Kd(/Ki) fields for the source —
        stiffness/damping for PhysicsDrive, Kp/Kd(/Ki) for a Newton actuator.
    """
    if source == GainSource.ACTUATOR:
        return GainLabels(kp_label=KP_LABEL, kd_label=KD_LABEL, ki_label=KI_LABEL if is_pid else None)
    if source == GainSource.MUJOCO:
        return GainLabels(kp_label=MJC_STIFFNESS_LABEL, kd_label=MJC_DAMPING_LABEL, ki_label=None)
    # PhysicsDrive and the NONE fallback both use stiffness / damping naming.
    return GainLabels(kp_label=STIFFNESS_LABEL, kd_label=DAMPING_LABEL, ki_label=None)


@dataclass
class ActuatorGains:
    """Kp/Kd(/Ki) gains read from a Newton actuator prim.

    Attributes are the live ``pxr.Usd.Attribute`` handles so callers can display
    (and, later, write) them without re-resolving the prim.
    """

    actuator_path: str
    """USD path of the actuator prim these gains were read from."""

    is_pid: bool
    """True for ``NewtonPIDControlAPI`` (exposes Ki); False for PD control."""

    kp_attr: pxr.Usd.Attribute | None = None
    kd_attr: pxr.Usd.Attribute | None = None
    ki_attr: pxr.Usd.Attribute | None = None

    @staticmethod
    def _value(attr: pxr.Usd.Attribute | None) -> float | None:
        if attr and attr.IsValid():
            value = attr.Get()
            if value is not None:
                return float(value)
        return None

    @property
    def kp(self) -> float | None:
        """Proportional gain, or None when unauthored."""
        return self._value(self.kp_attr)

    @property
    def kd(self) -> float | None:
        """Derivative gain, or None when unauthored."""
        return self._value(self.kd_attr)

    @property
    def ki(self) -> float | None:
        """Integral gain (PID only), or None when unauthored / PD control."""
        return self._value(self.ki_attr)


@dataclass
class ResolvedGains:
    """The gain view for a single joint under a given backend selection."""

    source: GainSource
    """Which source the shown values came from."""

    source_label: str
    """Short badge label for the source (``"Physics"`` / ``"Actuator"`` / ``""``)."""

    kp: float
    kd: float
    ki: float | None
    """Integral gain, or None when the source has no Ki (PhysicsDrive / PD)."""

    is_active: bool
    """True when the shown source is the one the active backend consumes.

    False means the values are informational / read-only (e.g. PhysicsDrive gains
    while Newton drives the joint via an actuator, or a comparison view of the
    non-active backend)."""

    multiple_sources: bool
    """True when both PhysicsDrive and actuator gains are authored on the joint."""

    active_source: GainSource
    """The source the active backend actually consumes (for warnings)."""

    drive_mode: int
    """Resolved :class:`JointDriveMode` value from the shown Kp/Kd."""

    kp_attr: pxr.Usd.Attribute | None = None
    kd_attr: pxr.Usd.Attribute | None = None
    ki_attr: pxr.Usd.Attribute | None = None

    mjc_source: object | None = None
    """The joint's :class:`MjcGainSource` when ``source`` is
    :attr:`GainSource.MUJOCO`, else None.  Lets editable MuJoCo views write tuned
    Kp/Kd back into the ``mjc:gainPrm`` / ``mjc:biasPrm`` arrays."""

    kp_label: str = STIFFNESS_LABEL
    """User-facing label for the Kp / stiffness field, chosen from ``source``."""

    kd_label: str = DAMPING_LABEL
    """User-facing label for the Kd / damping field, chosen from ``source``."""

    ki_label: str | None = None
    """User-facing label for the Ki field, or None when the source has no Ki."""


@dataclass
class GainReadContext:
    """Inputs that determine how a joint's gains are read and displayed."""

    viewed_backend: str = "PhysX"
    """Backend the user is currently viewing (may differ from active for comparison)."""

    active_backend: str = "PhysX"
    """Backend Isaac Sim currently has active."""

    actuator_map: dict = field(default_factory=dict)
    """Map of joint prim path -> :class:`ActuatorGains` (see build_actuator_gain_map)."""

    mjc_map: dict = field(default_factory=dict)
    """Map of joint prim path -> :class:`MjcGainSource` (see build_mjc_gain_map).

    Present when the composed stage/variant authors MuJoCo-native actuator gains
    (independent of the active physics engine)."""

    mujoco_solver_active: bool = False
    """True when the live Newton solver is the MuJoCo solver.

    When True, MuJoCo-native (``mjc:*``) gains become the active, editable source
    for joints that have them (and no Newton actuator); otherwise they are shown
    read-only for inspection."""

    solver: str = ""
    """Live Newton solver token (``"mujoco"`` / ``"xpbd"`` / ``"vbd"``).

    Empty when PhysX is active, or when the solver cannot be determined -- which
    leaves the advanced joint parameters' resolver chain unknown, because the
    MuJoCo chain differs from the XPBD / VBD one."""


def build_actuator_gain_map(stage: Usd.Stage, articulation_root_path: str) -> dict[str, ActuatorGains]:
    """Scan an articulation's subtree and map target DOF paths to actuator gains.

    Mirrors the Newton ``ArticulationActuators`` runtime, which discovers actuators
    by traversing the whole articulation subtree.  Every prim carrying a PD/PID
    controller applied schema (``NewtonPDControlAPI`` / ``NewtonPIDControlAPI``) —
    normally a ``NewtonActuator`` prim — is read regardless of where it lives; the
    ``{articulation_root}/Actuators`` scope is only the convention used by the
    ``add_actuator`` helper, so scanning that scope alone would miss actuators
    authored elsewhere.

    Args:
        stage: The USD stage to read from.
        articulation_root_path: Path of the articulation root prim (the subtree
            traversed for actuators).

    Returns:
        Dict mapping each targeted joint (DOF) prim path to its
        :class:`ActuatorGains`.  Empty when the stage/root is missing or no Newton
        actuators are authored.  Fully defensive: never raises.
    """
    result: dict[str, ActuatorGains] = {}
    if stage is None or not articulation_root_path:
        return result
    try:
        root_prim = stage.GetPrimAtPath(articulation_root_path)
        if not root_prim or not root_prim.IsValid():
            return result
        for prim in Usd.PrimRange(root_prim):
            schemas = prim.GetAppliedSchemas()
            is_pid = _NEWTON_PID_SCHEMA in schemas
            is_pd = _NEWTON_PD_SCHEMA in schemas
            if not (is_pid or is_pd):
                continue
            gains = ActuatorGains(
                actuator_path=prim.GetPath().pathString,
                is_pid=is_pid,
                kp_attr=prim.GetAttribute("newton:kp"),
                kd_attr=prim.GetAttribute("newton:kd"),
                ki_attr=prim.GetAttribute("newton:ki") if is_pid else None,
            )
            rel = prim.GetRelationship(_ACTUATOR_TARGETS_REL)
            targets = rel.GetTargets() if rel and rel.IsValid() else []
            for target in targets:
                # Each actuator targets exactly one single-DOF joint (strict 1-to-1);
                # two actuators on the same DOF is an authoring error, so keep the
                # first only defensively rather than as a consumption-order rule.
                result.setdefault(_path_string(target), gains)
    except _USD_ACCESS_ERRORS:
        return result
    return result


def _path_string(path: object) -> str:
    """Best-effort conversion of an Sdf.Path / str to a plain path string."""
    if isinstance(path, Sdf.Path):
        return path.pathString
    return str(path)


def has_physics_drive(joint: object, drive_axis: object = None) -> bool:
    """Return True when the joint has a PhysicsDrive stiffness/damping attribute."""
    stiffness = get_stiffness_attr(joint, drive_axis)
    damping = get_damping_attr(joint, drive_axis)
    return bool((stiffness and stiffness.IsValid()) or (damping and damping.IsValid()))


def active_gain_source(
    has_pd: bool,
    has_actuator: bool,
    has_mjc: bool = False,
    *,
    mujoco_solver_active: bool = False,
) -> GainSource:
    """Best-effort static guess at which source actually drives the joint.

    Newton actuators are **backend-independent**: the application-side
    ``ArticulationActuators`` runtime evaluates an authored actuator (zeroing the
    joint's DriveAPI gains) under either PhysX or Newton, so an actuator is the
    active source whenever it is authored — this is *not* gated on the active
    backend.  The precedence is:

    1. A Newton actuator is authored on the joint → :attr:`GainSource.ACTUATOR`.
    2. Else MuJoCo-native gains are authored **and** the Newton MuJoCo solver is
       running → :attr:`GainSource.MUJOCO`.
    3. Else PhysicsDrive gains are authored → :attr:`GainSource.PHYSICS_DRIVE`.
    4. Else :attr:`GainSource.NONE`.

    This is only a static guess: true actuator applicability depends on a live
    ``ArticulationActuators`` runtime that cannot be inferred from USD alone.

    Args:
        has_pd: Whether PhysicsDrive gains are authored.
        has_actuator: Whether a Newton actuator is authored on the joint.
        has_mjc: Whether MuJoCo-native (``mjc:*``) gains are authored.
        mujoco_solver_active: Whether the live Newton solver is the MuJoCo solver.

    Returns:
        The active :class:`GainSource` (``NONE`` when nothing usable is authored).
    """
    if has_actuator:
        return GainSource.ACTUATOR
    if has_mjc and mujoco_solver_active:
        return GainSource.MUJOCO
    if has_pd:
        return GainSource.PHYSICS_DRIVE
    return GainSource.NONE


def _configured_newton_solver_type() -> str:
    """Return the Newton config's solver token, or "" when it is not resolved yet.

    ``NewtonConfig.solver_cfg`` starts as the base ``NewtonSolverConfig``, whose
    ``solver_type`` is the placeholder string ``"None"``.  Newton replaces it with
    the real token only inside ``_initialize_newton_impl``, so before the first
    play this reports nothing usable.
    """
    try:
        from isaacsim.physics.newton import get_newton_config

        cfg = get_newton_config()
        solver_type = str(getattr(getattr(cfg, "solver_cfg", None), "solver_type", "") or "").lower()
    except (ImportError, AttributeError, TypeError, ValueError):
        return ""
    return solver_type if solver_type in NEWTON_SOLVER_TYPES else ""


def _staged_newton_solver_type(stage: Usd.Stage | None) -> str:
    """Return the solver token declared by the stage's Newton physics scene.

    Newton selects its solver from the solver API schema applied to a
    ``UsdPhysics.Scene`` prim (``NewtonStage._get_solver_type``), so the stage
    answers the question before the first play -- which is when the gain tuner is
    used.  A scene carrying more than one solver API is ambiguous and Newton
    rejects it, so this reports nothing rather than picking one.
    """
    if stage is None:
        return ""
    try:
        from isaacsim.physics.newton import newton_solver_to_api_schema
    except (ImportError, AttributeError):
        return ""
    try:
        scenes = list(stage.Traverse())
    except _USD_ACCESS_ERRORS:
        return ""
    for prim in scenes:
        try:
            if not prim.IsA(UsdPhysics.Scene):
                continue
            declared = [solver for solver, api in newton_solver_to_api_schema.items() if prim.HasAPI(api)]
        except _USD_ACCESS_ERRORS:
            continue
        if len(declared) == 1:
            return str(declared[0]).lower()
    return ""


def newton_solver_type(stage: Usd.Stage | None = None) -> str:
    """Return the Newton solver token that decides the joint-schema resolver order.

    The resolver order Newton reads the advanced joint parameters through depends
    on the solver -- the MuJoCo solver inserts ``mjc:*`` between ``newton:*`` and
    ``physxJoint:*`` -- so a caller reporting an effective value has to know which
    solver is running.

    The live Newton configuration is preferred, because it is what the running
    simulation actually used.  It only names a solver once Newton has
    initialized, so before the first play the answer is taken from the solver API
    schema applied to the stage's Newton physics scene, which is the same signal
    Newton itself initializes from.

    Args:
        stage: Stage to fall back to when the live configuration has no solver
            yet.  Pass None to skip the stage fallback.

    Returns:
        One of ``"mujoco"``, ``"xpbd"``, ``"vbd"``, or ``""`` when the solver
        cannot be determined -- in which case callers must report the effective
        value as undeterminable rather than assume a chain.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import newton_solver_type

        >>> newton_solver_type()  # doctest: +NO_CHECK
        'mujoco'
    """
    return _configured_newton_solver_type() or _staged_newton_solver_type(stage)


def newton_mujoco_solver_active(stage: Usd.Stage | None = None) -> bool:
    """Best-effort check for whether the live Newton solver is the MuJoCo solver.

    Args:
        stage: Stage used to resolve the solver before Newton has initialized.

    Returns:
        True when the resolved solver is the MuJoCo solver.  Fully defensive:
        returns False when the solver cannot be determined (headless, PhysX-only,
        no live app), so callers get the read-only MuJoCo behavior by default.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import newton_mujoco_solver_active

        >>> newton_mujoco_solver_active()  # doctest: +NO_CHECK
        False
    """
    return newton_solver_type(stage) == SOLVER_MUJOCO


def available_viewed_sources(has_pd: bool, has_mjc: bool, has_act: bool) -> list[GainSource]:
    """Return the gain sources a joint can display, in presentation order.

    Drives the per-joint viewed-source toggle in the detail editor: a joint may
    expose any combination of a ``PhysicsDrive``, MuJoCo-native, and Newton
    actuator gain source, and the user can switch which one is shown for
    inspection.  The order is PhysicsDrive, MuJoCo, then Newton actuator.

    Args:
        has_pd: Whether ``UsdPhysics.DriveAPI`` gains are authored.
        has_mjc: Whether MuJoCo-native (``mjc:*``) gains are authored.
        has_act: Whether a Newton actuator is authored.

    Returns:
        The ordered list of available :class:`GainSource` values (empty when none
        are authored).
    """
    sources: list[GainSource] = []
    if has_pd:
        sources.append(GainSource.PHYSICS_DRIVE)
    if has_mjc:
        sources.append(GainSource.MUJOCO)
    if has_act:
        sources.append(GainSource.ACTUATOR)
    return sources


def is_viewed_source_editable(viewed_source: GainSource, active_source: GainSource) -> bool:
    """Return True when the viewed gain source may be edited (not read-only).

    Editability is keyed to the resolved active source (per
    :func:`active_gain_source`): a viewed source is editable only when it is
    exactly that active source.  Because Newton actuators are backend-independent,
    a Newton actuator is editable whenever it is the active source (under PhysX or
    Newton).  MuJoCo-native gains are editable only while they are the active
    source — that is, when the Newton MuJoCo solver is running — and read-only
    otherwise.  ``PhysicsDrive`` gains are editable only when no actuator drives
    the joint and the active source resolves to PhysicsDrive.  Every non-active
    view is read-only for sim-to-sim inspection.

    Args:
        viewed_source: The :class:`GainSource` currently shown in the editor.
        active_source: The :class:`GainSource` the active backend/solver consumes.

    Returns:
        True when the viewed source's parameters may be edited, False for a
        read-only inspection view.
    """
    return viewed_source != GainSource.NONE and viewed_source == active_source


def _drive_mode_from_gains(kp: float, kd: float) -> int:
    """Map Kp/Kd to a :class:`JointDriveMode` value."""
    if kp > 0:
        return JointDriveMode.POSITION.value
    if kd > 0:
        return JointDriveMode.VELOCITY.value
    return JointDriveMode.NONE.value


def resolve_joint_gains(
    joint: object, drive_axis: object, ctx: GainReadContext, viewed_source: GainSource | None = None
) -> ResolvedGains:
    """Resolve the gain view for a joint under the given backend selection.

    Chooses the source to show and reads its Kp/Kd(/Ki), then flags whether that
    source is the one the active backend/solver actually consumes (and therefore
    editable) and whether multiple sources are authored.

    When ``viewed_source`` is None the default view is the **resolved active
    source** (per :func:`active_gain_source`: a Newton actuator when authored —
    backend-independent — else MuJoCo-native while the MuJoCo solver runs, else
    PhysicsDrive).  Pass ``viewed_source`` to force a specific source for the
    per-joint comparison toggle — including :attr:`GainSource.MUJOCO`, which reads
    the effective stiffness/damping encoded by the joint's MuJoCo-native
    ``mjc:gainPrm`` / ``mjc:biasPrm`` params.  A forced source that is not authored
    on the joint falls back to the resolved active source.

    Args:
        joint: The joint prim.
        drive_axis: Optional D6 drive-axis token.
        ctx: The :class:`GainReadContext` (viewed/active backend, actuator/mjc
            maps, and the MuJoCo-solver flag).
        viewed_source: Optional source to force (``PhysicsDrive`` / ``MuJoCo`` /
            actuator) for the per-joint viewed-source toggle.  None uses the
            resolved active source as the default view.

    Returns:
        A :class:`ResolvedGains` describing the values and their status.
    """
    joint_path = joint.GetPath().pathString
    actuator = ctx.actuator_map.get(joint_path)
    mjc = ctx.mjc_map.get(joint_path) if getattr(ctx, "mjc_map", None) else None
    has_pd = has_physics_drive(joint, drive_axis)
    has_act = actuator is not None
    has_mjc = mjc is not None
    multiple = (has_pd + has_act + has_mjc) > 1

    mujoco_solver_active = bool(getattr(ctx, "mujoco_solver_active", False))
    active_src = active_gain_source(has_pd, has_act, has_mjc, mujoco_solver_active=mujoco_solver_active)

    # Resolve a forced viewed source against what is actually authored; fall back
    # to the resolved active source (the default view) when the requested one is
    # absent.
    if viewed_source == GainSource.PHYSICS_DRIVE and has_pd:
        source = GainSource.PHYSICS_DRIVE
    elif viewed_source == GainSource.MUJOCO and has_mjc:
        source = GainSource.MUJOCO
    elif viewed_source == GainSource.ACTUATOR and has_act:
        source = GainSource.ACTUATOR
    elif active_src != GainSource.NONE:
        # Default view = the source the active backend/solver consumes.
        source = active_src
    elif has_pd:
        source = GainSource.PHYSICS_DRIVE
    elif has_act:
        source = GainSource.ACTUATOR
    elif has_mjc:
        source = GainSource.MUJOCO
    else:
        source = GainSource.NONE

    kp_attr = kd_attr = ki_attr = None
    mjc_source = None
    ki: float | None = None
    if source == GainSource.ACTUATOR:
        kp = actuator.kp or 0.0
        kd = actuator.kd or 0.0
        ki = actuator.ki if actuator.is_pid else None
        kp_attr, kd_attr, ki_attr = actuator.kp_attr, actuator.kd_attr, actuator.ki_attr
    elif source == GainSource.MUJOCO:
        # Derive the effective stiffness/damping from the actuator's gain/bias
        # arrays.  The MjcGainSource is carried on the result so an editable view
        # (MuJoCo solver active) can write tuned Kp/Kd back into the mjc arrays.
        mjc_source = mjc
        gain_prm = mjc.gain_prm_attr.Get() if (mjc.gain_prm_attr and mjc.gain_prm_attr.IsValid()) else None
        bias_prm = mjc.bias_prm_attr.Get() if (mjc.bias_prm_attr and mjc.bias_prm_attr.IsValid()) else None
        kp, kd = mjc_params_to_drive_gains(gain_prm, bias_prm)
    elif source == GainSource.PHYSICS_DRIVE:
        kp_attr = get_stiffness_attr(joint, drive_axis)
        kd_attr = get_damping_attr(joint, drive_axis)
        kp = float(kp_attr.Get()) if (kp_attr and kp_attr.IsValid() and kp_attr.Get() is not None) else 0.0
        kd = float(kd_attr.Get()) if (kd_attr and kd_attr.IsValid() and kd_attr.Get() is not None) else 0.0
    else:
        kp = kd = 0.0

    is_active = is_viewed_source_editable(source, active_src)
    labels = gain_labels_for_source(source, is_pid=ki is not None)

    return ResolvedGains(
        source=source,
        source_label=_SOURCE_LABELS[source],
        kp=kp,
        kd=kd,
        ki=ki,
        is_active=is_active,
        multiple_sources=multiple,
        active_source=active_src,
        drive_mode=_drive_mode_from_gains(kp, kd),
        kp_attr=kp_attr,
        kd_attr=kd_attr,
        ki_attr=ki_attr,
        mjc_source=mjc_source,
        kp_label=labels.kp_label,
        kd_label=labels.kd_label,
        ki_label=labels.ki_label,
    )


# ---------------------------------------------------------------------------
# MuJoCo-native gain read-back (used by the read-only comparison view)
# ---------------------------------------------------------------------------


def mjc_params_to_drive_gains(gain_prm: object, bias_prm: object) -> tuple[float, float]:
    """Recover the effective ``(kp, kd)`` gains encoded by MuJoCo actuator params.

    Inverse of :func:`drive_gains_to_mjc` for the supported affine/fixed mapping:
    ``kp = -biasPrm[1]`` and ``kd = -biasPrm[2]``.  Velocity control (``biasPrm[1]``
    zero) yields ``kp = 0``.  Used to show MuJoCo-native gains as the stiffness /
    damping they represent in the read-only comparison view.

    Args:
        gain_prm: The ``mjc:gainPrm`` array (unused for the inverse but accepted
            for symmetry / future mappings).  May be None.
        bias_prm: The ``mjc:biasPrm`` array.  May be None or shorter than expected.

    Returns:
        The recovered ``(kp, kd)`` gains; ``(0.0, 0.0)`` when ``bias_prm`` is
        missing or too short.
    """
    try:
        bias = list(bias_prm) if bias_prm is not None else []
    except TypeError:
        bias = []
    kp = -float(bias[1]) if len(bias) > 1 else 0.0
    kd = -float(bias[2]) if len(bias) > 2 else 0.0
    return kp, kd
