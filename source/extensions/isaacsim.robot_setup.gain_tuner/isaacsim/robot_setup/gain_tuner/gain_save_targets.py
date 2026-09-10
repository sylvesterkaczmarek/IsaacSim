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

"""Save-target layer resolution and gain write plans for multi-backend tuning.

Splits the persistence half of the gain model out of :mod:`gain_sources`: given a
robot's tunable joints and their resolved gains, decide which USD layers tuned
DriveAPI stiffness/damping may be written to, mirror those gains into MuJoCo
(``mjc:*``) and Newton actuator (``newton:*``) parameters, and build / apply the
per-layer :class:`GainSpecEdit` plan the UI save row persists.  Reads USD directly
and contains no UI / Kit imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pxr
from pxr import Sdf, Tf, Usd

from .gain_sources import ActuatorGains, build_actuator_gain_map, mjc_params_to_drive_gains
from .joint_drive_attrs import get_damping_attr, get_joint_drive_type_attr, get_stiffness_attr
from .joint_schema_attrs import (
    JOINT_PARAM_SPECS,
    NEWTON_JOINT_API,
    PHYSX_JOINT_API,
    authored_joint_param_attrs,
)
from .usd_layer_utils import (
    find_layer_by_save_identifier,
    get_layer_save_identifier,
    get_property_path_for_layer,
    is_mujoco_layer,
    is_physics_layer,
    is_physx_layer,
)

# USD / attribute / layer access can fail with these when a prim, attribute, or
# layer is missing, invalid, a duck-typed fake (unit tests substitute plain
# objects), or holds an unexpected value type.  Catch exactly this set at the
# defensive USD boundaries so any *unexpected* error type still propagates.
_USD_ACCESS_ERRORS = (AttributeError, TypeError, ValueError, Tf.ErrorException)


def _path_string(path: object) -> str:
    """Best-effort conversion of an ``Sdf.Path`` / str to a plain path string."""
    if isinstance(path, Sdf.Path):
        return path.pathString
    return str(path)


# ---------------------------------------------------------------------------
# Save-target layer resolution (which USD layer tuned DriveAPI gains persist to)
# ---------------------------------------------------------------------------

# Robot assets keep their physics gains in a ``payloads/Physics/`` folder holding
# a neutral, human-editable base layer (``physics.usda``) and optional
# backend-specific overlay layers (``physx.usda`` / ``mujoco.usda``).  Tuned
# DriveAPI gains default to the neutral layer but may be written to any sibling.
_PHYSICS_DIR_NAME = "Physics"
_USD_LAYER_SUFFIXES = (".usda", ".usd", ".usdc")


@dataclass
class SaveTargetCandidate:
    """A single layer a user may save tuned DriveAPI gains to."""

    identifier: str
    """Layer identifier (real path when available) for ``Sdf.Layer.Find`` / save dialogs."""

    display_name: str
    """Short file name shown in the UI (e.g. ``"physics.usda"``)."""

    is_default: bool = False
    """True for the neutral ``physics.usda`` layer preselected as the default target."""


@dataclass
class SaveTargetOptions:
    """The ordered set of candidate save-target layers and the default selection."""

    candidates: list[SaveTargetCandidate] = field(default_factory=list)
    """Candidate layers, neutral ``physics.usda`` first when present."""

    default_identifier: str | None = None
    """Identifier of the default (preselected) candidate, or None when unresolved."""

    @property
    def resolved(self) -> bool:
        """True when at least one candidate save-target layer was resolved."""
        return bool(self.candidates)

    @property
    def default(self) -> SaveTargetCandidate | None:
        """The default candidate, or None when nothing resolved."""
        for candidate in self.candidates:
            if candidate.identifier == self.default_identifier:
                return candidate
        return self.candidates[0] if self.candidates else None

    @property
    def display_names(self) -> list[str]:
        """Candidate display names in presentation order."""
        return [candidate.display_name for candidate in self.candidates]

    def identifier_for_display_name(self, display_name: str) -> str | None:
        """Return the identifier whose candidate has the given display name, or None."""
        for candidate in self.candidates:
            if candidate.display_name == display_name:
                return candidate.identifier
        return None


def _is_attribute(obj: object) -> bool:
    """Best-effort check for a ``pxr.Usd.Attribute`` (duck-typed for test fakes)."""
    return hasattr(obj, "GetPropertyStack") and hasattr(obj, "IsValid")


def get_defining_layer(attr: pxr.Usd.Attribute | None) -> Sdf.Layer | None:
    """Return the base layer that defines ``attr`` (its weakest authored opinion).

    Resolves the authoring layer via the attribute's property stack at the default
    time code.  USD orders ``GetPropertyStack`` strongest-to-weakest, so
    ``stack[-1]`` is the *weakest* opinion: the asset's own physics layer rather
    than a live root / session edit sitting on top of it.  That is the layer the
    "Save Gains to Physics Layer" workflow wants as its target.

    Args:
        attr: USD attribute to resolve.

    Returns:
        The defining layer, or None when ``attr`` is missing/invalid/unauthored.
    """
    if attr is None or not _is_attribute(attr) or not attr.IsValid():
        return None
    try:
        stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    except _USD_ACCESS_ERRORS:
        return None
    if not stack:
        return None
    return stack[-1].layer


def _layer_file_path(layer: Sdf.Layer | None) -> str | None:
    """Return the on-disk file path for ``layer`` when it is a real file, else None."""
    if layer is None:
        return None
    real_path = layer.realPath
    if real_path and os.path.isfile(real_path):
        return real_path
    return None


def _physics_folders_from_prim(prim: pxr.Usd.Prim | None) -> list[str]:
    """Return on-disk ``payloads/Physics/`` folders that author ``prim`` or an ancestor.

    Collects the parent folder of every layer contributing a spec to the prim's
    own composition that lives in a ``Physics`` directory.  This lets the
    save-target list offer the sibling ``physics.usda`` / ``physx.usda`` /
    ``mujoco.usda`` layers even when the DriveAPI gains themselves are authored on
    a layer outside that folder (for example the robot's main layer or a live
    root / session edit).

    Anchoring on the prim rather than scanning every layer on the stage is what
    keeps a scene holding several robots from offering one robot's physics layers
    as save targets for another.
    """
    folders: list[str] = []
    if prim is None:
        return folders
    try:
        current = prim
        while current and current.IsValid() and not current.IsPseudoRoot():
            for prim_spec in current.GetPrimStack():
                file_path = _layer_file_path(prim_spec.layer)
                if file_path is None:
                    continue
                folder = os.path.dirname(file_path)
                if os.path.basename(folder) == _PHYSICS_DIR_NAME and folder not in folders:
                    folders.append(folder)
            current = current.GetParent()
    except (*_USD_ACCESS_ERRORS, OSError):
        return folders
    return folders


def _attr_prim(attr: pxr.Usd.Attribute | None) -> pxr.Usd.Prim | None:
    """Best-effort resolution of the prim that owns ``attr`` (None for fakes)."""
    try:
        return attr.GetPrim()
    except _USD_ACCESS_ERRORS:
        return None


def _prim_path_key(prim: pxr.Usd.Prim | None) -> str | None:
    """Best-effort prim path string used to de-duplicate per-joint work."""
    if prim is None:
        return None
    try:
        return _path_string(prim.GetPath())
    except _USD_ACCESS_ERRORS:
        return None


def list_gain_save_target_layers(
    attrs: pxr.Usd.Attribute | list[pxr.Usd.Attribute] | None,
) -> SaveTargetOptions:
    """Resolve the candidate save-target layers for one or more gain attributes.

    For each attribute the defining layer is resolved; when that layer lives in a
    ``payloads/Physics/`` folder the sibling USD layers (``physics.usda``,
    ``physx.usda``, ``mujoco.usda``, and any others present) are enumerated as
    candidates.  The ``payloads/Physics/`` folder is additionally discovered from
    the layers composing the attribute's own joint prim, so those sibling layers
    are offered even when the gains are authored on a layer outside that folder
    (the robot's main layer, a live root / session edit, etc.).  The neutral
    ``physics.usda`` layer is placed first as the default; the remaining
    candidates follow sorted by name.  The defining layer is always included even
    when it is outside a ``Physics`` folder.

    A discovered folder is sufficient on its own: importer-produced assets leave
    stiffness and damping unauthored, so no attribute has a defining layer and the
    folder is the only signal available.

    Discovery is deliberately scoped to the joints being tuned rather than to
    every layer on the stage, so a scene holding several robots never offers one
    robot's physics layers as save targets for another.

    Fully defensive: never raises.  Anonymous / in-memory layers (no real file)
    fall back to the defining layer identifier only, and a missing folder yields
    just the defining layers.

    Args:
        attrs: A single attribute or a list of attributes whose save targets
            should be resolved.

    Returns:
        A :class:`SaveTargetOptions` with the ordered candidates and the default.
    """
    if attrs is None:
        return SaveTargetOptions()
    if _is_attribute(attrs):
        attrs = [attrs]

    # Resolve unique defining layers (preserve first-seen order) and the set of
    # Physics folders discovered from the attributes' own prims.
    defining_layers: list[Sdf.Layer] = []
    seen_layer_ids: set[str] = set()
    physics_folders: list[str] = []
    seen_prim_paths: set[str] = set()
    for attr in attrs:
        layer = get_defining_layer(attr)
        if layer is not None and layer.identifier not in seen_layer_ids:
            seen_layer_ids.add(layer.identifier)
            defining_layers.append(layer)
        prim = _attr_prim(attr)
        # Stiffness and damping share a prim; walk each joint only once.
        prim_key = _prim_path_key(prim)
        if prim_key is not None and prim_key not in seen_prim_paths:
            seen_prim_paths.add(prim_key)
            for folder in _physics_folders_from_prim(prim):
                if folder not in physics_folders:
                    physics_folders.append(folder)

    # Assets from the URDF / MJCF importer author only ``maxForce``, leaving
    # stiffness and damping at their schema fallbacks, so no gain has a defining
    # layer.  A Physics folder discovered from the joint is enough on its own.
    if not defining_layers and not physics_folders:
        return SaveTargetOptions()

    # Gather candidates keyed by normalized identifier to de-duplicate.
    candidates: dict[str, SaveTargetCandidate] = {}

    def _add(identifier: str, display_name: str) -> None:
        key = os.path.normpath(identifier)
        if key not in candidates:
            candidates[key] = SaveTargetCandidate(identifier=identifier, display_name=display_name)

    def _enumerate_folder(folder: str) -> None:
        try:
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith(_USD_LAYER_SUFFIXES):
                    _add(os.path.join(folder, name), name)
        except OSError:
            pass

    # Physics-folder siblings discovered from the composed stage (offered even
    # when the gains are authored outside that folder).
    for folder in physics_folders:
        _enumerate_folder(folder)

    for layer in defining_layers:
        file_path = _layer_file_path(layer)
        if file_path is None:
            # Anonymous / in-memory layer: only the layer itself is a target.
            identifier = get_layer_save_identifier(layer) or layer.identifier
            _add(identifier, os.path.basename(identifier) or identifier)
            continue
        folder = os.path.dirname(file_path)
        if os.path.basename(folder) == _PHYSICS_DIR_NAME:
            _enumerate_folder(folder)
        _add(file_path, os.path.basename(file_path))

    if not candidates:
        return SaveTargetOptions()

    # Choose the neutral physics.usda as default when present.
    default_identifier: str | None = None
    for candidate in candidates.values():
        if is_physics_layer(candidate.display_name):
            candidate.is_default = True
            default_identifier = candidate.identifier
            break
    if default_identifier is None:
        first = next(iter(candidates.values()))
        first.is_default = True
        default_identifier = first.identifier

    # Order: default first, remaining sorted by display name.
    ordered = sorted(
        candidates.values(),
        key=lambda c: (not c.is_default, c.display_name.lower()),
    )
    return SaveTargetOptions(candidates=ordered, default_identifier=default_identifier)


# ---------------------------------------------------------------------------
# MuJoCo-native gain mirroring (Option A: keep PhysX and MuJoCo/Newton in sync)
# ---------------------------------------------------------------------------

# MJCF actuator parameter arrays are length-10 ``FloatArray`` values authored on
# ``MjcActuator`` prims.  The DriveAPI <-> MuJoCo mapping is verified against the
# MJCF importer (``isaacsim.asset.importer.utils.mjc_to_physx_conversion_utils``):
# position control uses ``gainPrm = [kp, 0, 0, ...]`` and
# ``biasPrm = [0, -kp, -kd, ...]`` with ``gainType="fixed"`` / ``biasType="affine"``;
# velocity control uses ``gainPrm = [kd, 0, 0, ...]`` and ``biasPrm = [0, 0, -kd, ...]``.
_MJC_PRM_LENGTH = 10
_MJC_ACTUATOR_TYPE = "MjcActuator"
_MJC_TARGET_REL = "mjc:target"
_MJC_GAIN_TYPE_FIXED = "fixed"
_MJC_BIAS_TYPE_AFFINE = "affine"


@dataclass
class MjcGainParams:
    """MuJoCo-native actuator gain parameters mirrored from DriveAPI Kp/Kd."""

    gain_prm: list[float]
    """``mjc:gainPrm`` array (length 10)."""

    bias_prm: list[float]
    """``mjc:biasPrm`` array (length 10)."""

    gain_type: str = _MJC_GAIN_TYPE_FIXED
    """``mjc:gainType`` token (always ``"fixed"`` for the supported mapping)."""

    bias_type: str = _MJC_BIAS_TYPE_AFFINE
    """``mjc:biasType`` token (always ``"affine"`` for the supported mapping)."""

    @property
    def is_velocity(self) -> bool:
        """True when these parameters encode velocity (rather than position) control."""
        return self.gain_prm[0] > 0.0 and self.bias_prm[1] == 0.0 and self.bias_prm[2] < 0.0


def drive_gains_to_mjc(kp: float, kd: float) -> MjcGainParams:
    """Convert tuned DriveAPI ``(kp, kd)`` gains into MuJoCo actuator parameters.

    Mirrors the exact inverse of the MJCF importer's affine/fixed mapping.  A
    non-zero ``kp`` produces position-control parameters
    (``gainPrm = [kp, 0, ...]``, ``biasPrm = [0, -kp, -kd, ...]``); a zero ``kp``
    with a non-zero ``kd`` produces velocity-control parameters
    (``gainPrm = [kd, 0, ...]``, ``biasPrm = [0, 0, -kd, ...]``).  When both are
    zero all-zero arrays are returned.

    Args:
        kp: Proportional gain (DriveAPI stiffness).
        kd: Derivative gain (DriveAPI damping).

    Returns:
        The equivalent :class:`MjcGainParams`.
    """
    kp = float(kp or 0.0)
    kd = float(kd or 0.0)
    gain_prm = [0.0] * _MJC_PRM_LENGTH
    bias_prm = [0.0] * _MJC_PRM_LENGTH
    if kp > 0.0:
        # Position control: F = kp * ctrl - kp * pos - kd * vel.
        gain_prm[0] = kp
        bias_prm[1] = -kp
        bias_prm[2] = -kd
    elif kd > 0.0:
        # Velocity control: F = kd * ctrl - kd * vel.
        gain_prm[0] = kd
        bias_prm[2] = -kd
    return MjcGainParams(gain_prm=gain_prm, bias_prm=bias_prm)


def author_mjc_gains(mjc_source: object, kp: float, kd: float) -> bool:
    """Write tuned ``(kp, kd)`` gains into a joint's MuJoCo-native params in place.

    Used when the MuJoCo solver is active and the MuJoCo-native gains are the
    editable source: converts ``(kp, kd)`` to ``mjc:gainPrm`` / ``mjc:biasPrm``
    (and the ``fixed`` / ``affine`` type tokens) via :func:`drive_gains_to_mjc` and
    sets them on the live attributes, plus the scalar ``mjc:stiffness`` /
    ``mjc:damping`` attrs when the joint authors them.  Fully defensive: never
    raises.

    Args:
        mjc_source: The joint's :class:`MjcGainSource`.
        kp: Proportional gain (mjc stiffness).
        kd: Derivative gain (mjc damping).

    Returns:
        True when at least one attribute was written, else False.
    """
    if mjc_source is None:
        return False
    params = drive_gains_to_mjc(kp, kd)
    wrote = False

    def _set(attr: pxr.Usd.Attribute | None, value: object) -> None:
        nonlocal wrote
        try:
            if attr is not None and attr.IsValid():
                attr.Set(value)
                wrote = True
        except _USD_ACCESS_ERRORS:
            pass

    _set(getattr(mjc_source, "gain_prm_attr", None), params.gain_prm)
    _set(getattr(mjc_source, "bias_prm_attr", None), params.bias_prm)
    _set(getattr(mjc_source, "gain_type_attr", None), params.gain_type)
    _set(getattr(mjc_source, "bias_type_attr", None), params.bias_type)
    _set(getattr(mjc_source, "joint_stiffness_attr", None), float(kp or 0.0))
    _set(getattr(mjc_source, "joint_damping_attr", None), float(kd or 0.0))
    return wrote


@dataclass
class MjcGainSource:
    """MuJoCo-native gain attributes discovered for a joint.

    Groups the ``MjcActuator`` prim's gain-parameter attributes with the optional
    ``mjc:stiffness`` / ``mjc:damping`` attributes authored on the joint itself.
    Attributes are live ``pxr.Usd.Attribute`` handles so callers can mirror tuned
    values into them.
    """

    actuator_path: str
    """USD path of the ``MjcActuator`` prim these attributes were read from."""

    gain_prm_attr: pxr.Usd.Attribute | None = None
    bias_prm_attr: pxr.Usd.Attribute | None = None
    gain_type_attr: pxr.Usd.Attribute | None = None
    bias_type_attr: pxr.Usd.Attribute | None = None
    joint_stiffness_attr: pxr.Usd.Attribute | None = None
    joint_damping_attr: pxr.Usd.Attribute | None = None

    @property
    def has_joint_gains(self) -> bool:
        """True when the joint carries ``mjc:stiffness`` / ``mjc:damping`` attributes."""
        return (self.joint_stiffness_attr is not None and self.joint_stiffness_attr.IsValid()) or (
            self.joint_damping_attr is not None and self.joint_damping_attr.IsValid()
        )

    @property
    def defining_layer(self) -> Sdf.Layer | None:
        """The layer that authors the actuator gain params (typically ``mujoco.usda``)."""
        for attr in (self.gain_prm_attr, self.bias_prm_attr, self.gain_type_attr, self.bias_type_attr):
            layer = get_defining_layer(attr)
            if layer is not None:
                return layer
        return None


def _has_authored_mjc_gains(prim: Usd.Prim) -> bool:
    """True when a prim carries MuJoCo actuator gain params (``mjc:gainPrm`` /
    ``mjc:biasPrm``) authored directly on it (rather than on a separate
    ``MjcActuator`` prim)."""
    for name in ("mjc:gainPrm", "mjc:biasPrm"):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.HasAuthoredValue():
            return True
    return False


def build_mjc_gain_map(stage: Usd.Stage) -> dict[str, MjcGainSource]:
    """Map joint (DOF) prim paths to their MuJoCo-native actuator gains.

    Detects both authoring styles used by MuJoCo/MJCF-derived assets:

    * a separate ``MjcActuator`` prim carrying ``mjc:gainPrm`` / ``mjc:biasPrm`` and
      pointing at its target joint via the ``mjc:target`` relationship, and
    * the same ``mjc:*`` gain params authored **directly on the joint prim** (some
      Menagerie-style assets author them on the joint under the ``mujoco`` Physics
      variant instead of on a dedicated actuator prim).

    Args:
        stage: The USD stage to read from.

    Returns:
        Dict mapping each targeted joint (DOF) prim path to its
        :class:`MjcGainSource`.  Empty when the stage is missing or no MuJoCo
        actuator gains are authored.  Fully defensive: never raises.
    """
    result: dict[str, MjcGainSource] = {}
    if stage is None:
        return result
    try:
        for prim in stage.TraverseAll():
            if prim.GetTypeName() == _MJC_ACTUATOR_TYPE:
                rel = prim.GetRelationship(_MJC_TARGET_REL)
                targets = rel.GetTargets() if rel and rel.IsValid() else []
                if not targets:
                    continue
                for target in targets:
                    target_path = _path_string(target)
                    joint_prim = stage.GetPrimAtPath(target)
                    joint_stiffness = joint_damping = None
                    if joint_prim and joint_prim.IsValid():
                        joint_stiffness = joint_prim.GetAttribute("mjc:stiffness")
                        joint_damping = joint_prim.GetAttribute("mjc:damping")
                    source = MjcGainSource(
                        actuator_path=prim.GetPath().pathString,
                        gain_prm_attr=prim.GetAttribute("mjc:gainPrm"),
                        bias_prm_attr=prim.GetAttribute("mjc:biasPrm"),
                        gain_type_attr=prim.GetAttribute("mjc:gainType"),
                        bias_type_attr=prim.GetAttribute("mjc:biasType"),
                        joint_stiffness_attr=joint_stiffness,
                        joint_damping_attr=joint_damping,
                    )
                    result.setdefault(target_path, source)
            elif _has_authored_mjc_gains(prim):
                # mjc gain params authored directly on the joint prim: the joint
                # itself is the "actuator" for save/mirror purposes.
                joint_path = prim.GetPath().pathString
                source = MjcGainSource(
                    actuator_path=joint_path,
                    gain_prm_attr=prim.GetAttribute("mjc:gainPrm"),
                    bias_prm_attr=prim.GetAttribute("mjc:biasPrm"),
                    gain_type_attr=prim.GetAttribute("mjc:gainType"),
                    bias_type_attr=prim.GetAttribute("mjc:biasType"),
                    joint_stiffness_attr=prim.GetAttribute("mjc:stiffness"),
                    joint_damping_attr=prim.GetAttribute("mjc:damping"),
                )
                result.setdefault(joint_path, source)
    except _USD_ACCESS_ERRORS:
        return result
    return result


# ---------------------------------------------------------------------------
# Combined write-target resolution + save plan (used by the UI save row)
# ---------------------------------------------------------------------------


@dataclass
class GainWriteTargets:
    """Resolved DriveAPI save-target options plus mjc / Newton mirror sources.

    Produced by :func:`resolve_gain_write_targets` for a robot's joint entries so
    the UI can render the save-target dropdown, the mirror toggle + target
    pickers, and inform the user about the additional MuJoCo / Newton targets that
    will be written on save.
    """

    options: SaveTargetOptions = field(default_factory=SaveTargetOptions)
    """Candidate DriveAPI save-target layers and default."""

    mjc_sources: list[MjcGainSource] = field(default_factory=list)
    """MuJoCo-native gain sources that will be mirrored (Option A)."""

    newton_actuators: list[ActuatorGains] = field(default_factory=list)
    """Newton actuator gain sources that will receive tuned-gain writeback."""

    mirror_enabled: bool = True
    """Whether the Newton actuator writeback is enabled.  Default ON so tuned
    actuator gains are persisted back to the actuator's own layer (the active
    source for actuator-driven joints).  This does **not** govern the MuJoCo
    mirror, which is opt-in via :attr:`mirror_drive_to_mjc`."""

    mirror_drive_to_mjc: bool = False
    """Whether the opt-in DriveAPI->MuJoCo mirror is enabled.  Default OFF so a
    save never authors ``mjc:*`` actuator prims unless the user explicitly opts
    in.  When True, tuned DriveAPI Kp/Kd are also written into the joint's
    ``mjc:*`` params (MuJoCo stays a mirror; it does not become the active
    source).  Does not apply while the MuJoCo solver is active, where ``mjc:*``
    is edited directly as the active source."""

    mjc_target_identifier: str | None = None
    """Resolved default layer identifier that mjc mirror edits route to
    (the ``mujoco.usda`` candidate when present).  None when no mjc source."""

    newton_target_identifier: str | None = None
    """Resolved default layer identifier that Newton writeback routes to
    (the actuator prim's own defining layer).  None when no actuator."""

    @property
    def has_mjc_mirror(self) -> bool:
        """True when at least one MuJoCo-native gain source is present to mirror."""
        return bool(self.mjc_sources)

    @property
    def has_newton_writeback(self) -> bool:
        """True when at least one Newton actuator is present to write back."""
        return bool(self.newton_actuators)

    @property
    def will_mirror_mjc(self) -> bool:
        """True when the opt-in DriveAPI->MuJoCo mirror is on and has a source.

        Governed by :attr:`mirror_drive_to_mjc` (default OFF), *not* by
        :attr:`mirror_enabled`: MuJoCo actuator prims are never written unless the
        user explicitly opts in.
        """
        return self.mirror_drive_to_mjc and self.has_mjc_mirror

    @property
    def will_write_newton(self) -> bool:
        """True when Newton writeback is both enabled and has an actuator target."""
        return self.mirror_enabled and self.has_newton_writeback

    @property
    def mjc_layer_names(self) -> list[str]:
        """Display names of the layer(s) mjc mirroring will write to.

        Reflects the selected :attr:`mjc_target_identifier` when set, otherwise the
        mjc sources' own defining layers.
        """
        if self.mjc_target_identifier:
            return [os.path.basename(self.mjc_target_identifier)]
        names: list[str] = []
        for source in self.mjc_sources:
            layer = source.defining_layer
            identifier = get_layer_save_identifier(layer) if layer is not None else None
            name = os.path.basename(identifier) if identifier else None
            if name and name not in names:
                names.append(name)
        return names

    @property
    def newton_layer_names(self) -> list[str]:
        """Display names of the layer(s) Newton writeback will write to.

        Reflects the selected :attr:`newton_target_identifier` when set, otherwise
        the actuators' own defining layers.
        """
        if self.newton_target_identifier:
            return [os.path.basename(self.newton_target_identifier)]
        names: list[str] = []
        for actuator in self.newton_actuators:
            layer = get_defining_layer(actuator.kp_attr) or get_defining_layer(actuator.kd_attr)
            identifier = get_layer_save_identifier(layer) if layer is not None else None
            name = os.path.basename(identifier) if identifier else None
            if name and name not in names:
                names.append(name)
        return names


def _default_mjc_target_identifier(options: SaveTargetOptions, mjc_sources: list[MjcGainSource]) -> str | None:
    """Resolve the default mjc mirror target (the ``mujoco.usda`` candidate).

    Prefers a ``mujoco.usda`` layer among the Physics-folder candidates, then
    falls back to the mjc source's own defining layer.  Returns None when no mjc
    source is present.
    """
    if not mjc_sources:
        return None
    for candidate in options.candidates:
        if is_mujoco_layer(candidate.display_name):
            return candidate.identifier
    for source in mjc_sources:
        identifier = get_layer_save_identifier(source.defining_layer)
        if identifier:
            return identifier
    return None


def _physx_overlay_layer(options: SaveTargetOptions) -> Sdf.Layer | None:
    """Return the ``physx.usda`` overlay among the save-target candidates, or None.

    Args:
        options: The resolved save-target candidates for the robot.

    Returns:
        The PhysX overlay layer, or None when the robot has no such candidate
        (callers then fall back to each attribute's own defining layer).
    """
    for candidate in options.candidates:
        if is_physx_layer(candidate.display_name):
            return find_layer_by_save_identifier(candidate.identifier)
    return None


def _default_newton_target_identifier(newton_actuators: list[ActuatorGains]) -> str | None:
    """Resolve the default Newton writeback target (the actuator's defining layer)."""
    for actuator in newton_actuators:
        layer = (
            get_defining_layer(actuator.kp_attr)
            or get_defining_layer(actuator.kd_attr)
            or get_defining_layer(actuator.ki_attr)
        )
        identifier = get_layer_save_identifier(layer)
        if identifier:
            return identifier
    return None


def resolve_gain_write_targets(
    joint_entries: list,
    stage: Usd.Stage | None,
    articulation_root_path: str | None = None,
    *,
    mirror_enabled: bool = True,
    mirror_drive_to_mjc: bool = False,
    mjc_target_identifier: str | None = None,
    newton_target_identifier: str | None = None,
) -> GainWriteTargets:
    """Resolve DriveAPI save targets and mjc / Newton mirror sources for joints.

    Args:
        joint_entries: Joint table rows (objects with ``joint`` and optional
            ``drive_axis``).
        stage: USD stage hosting the robot.
        articulation_root_path: Path of the articulation root (used to locate the
            ``Actuators`` scope for Newton actuators).
        mirror_enabled: Whether the Newton actuator writeback is enabled (default
            ON; persists tuned actuator gains to their own layer).
        mirror_drive_to_mjc: Whether the opt-in DriveAPI->MuJoCo mirror is enabled
            (default OFF; ``mjc:*`` prims are only authored when the user opts in).
        mjc_target_identifier: Override layer identifier for mjc mirror edits.
            When None, resolves to the ``mujoco.usda`` candidate.
        newton_target_identifier: Override layer identifier for Newton writeback.
            When None, resolves to the actuator prim's own defining layer.

    Returns:
        A :class:`GainWriteTargets` describing the save-target options, the mirror
        toggle state, and the resolved mjc / Newton target identifiers.
    """
    if not joint_entries:
        return GainWriteTargets(mirror_enabled=mirror_enabled, mirror_drive_to_mjc=mirror_drive_to_mjc)
    mjc_map = build_mjc_gain_map(stage) if stage is not None else {}
    actuator_map = (
        build_actuator_gain_map(stage, articulation_root_path) if stage is not None and articulation_root_path else {}
    )

    drive_attrs: list[pxr.Usd.Attribute] = []
    mjc_sources: list[MjcGainSource] = []
    newton_actuators: list[ActuatorGains] = []
    for entry in joint_entries:
        joint = entry.joint
        drive_axis = getattr(entry, "drive_axis", None)
        for attr in (get_stiffness_attr(joint, drive_axis), get_damping_attr(joint, drive_axis)):
            if attr is not None and attr.IsValid():
                drive_attrs.append(attr)
        joint_path = joint.GetPath().pathString
        if joint_path in mjc_map:
            mjc_sources.append(mjc_map[joint_path])
        if joint_path in actuator_map:
            newton_actuators.append(actuator_map[joint_path])

    options = list_gain_save_target_layers(drive_attrs)
    resolved_mjc = mjc_target_identifier or _default_mjc_target_identifier(options, mjc_sources)
    resolved_newton = newton_target_identifier or _default_newton_target_identifier(newton_actuators)
    return GainWriteTargets(
        options=options,
        mjc_sources=mjc_sources,
        newton_actuators=newton_actuators,
        mirror_enabled=mirror_enabled,
        mirror_drive_to_mjc=mirror_drive_to_mjc,
        mjc_target_identifier=resolved_mjc,
        newton_target_identifier=resolved_newton,
    )


@dataclass
class GainSpecEdit:
    """A single attribute value to author onto a layer when saving tuned gains."""

    prop_path: Sdf.Path
    """Absolute property path to author (e.g. ``/Robot/joint.drive:angular:physics:stiffness``)."""

    type_name: object
    """The ``Sdf.ValueTypeName`` for the attribute spec."""

    value: object
    """The value to set as the spec default."""

    apply_api: str | None = None
    """API schema to apply on the prim spec alongside the attribute, or None.

    An attribute that belongs to an applied API schema is only *meaningful* to
    consumers that see the schema applied.  ``ApplyAPI`` authors ``apiSchemas``
    into the stage's edit target, which is generally not the payload layer this
    edit is routed to, so the schema application has to travel with the value or
    the saved layer ends up holding a value nothing recognizes.
    """


def _asset_local_prop_path(attr: pxr.Usd.Attribute, target_layer: Sdf.Layer | None) -> Sdf.Path:
    """Return the property path to author for ``attr`` on ``target_layer``.

    A robot referenced into a scene composes under the scene's namespace (for
    example ``/World/Robot/joint``) while its own asset layers author
    ``/Robot/joint``.  Authoring the composed path into an asset layer would write
    a spec that asset never reads, silently discarding the tuned gains, so the
    path is taken from the prim's spec in the target layer, or failing that from a
    spec in a sibling layer of the same asset folder (``physx.usda`` and
    ``mujoco.usda`` are overlays that need not author the joint themselves).
    """
    composed = attr.GetPath()
    if target_layer is None:
        return composed
    mapped = get_property_path_for_layer(attr, target_layer)
    if mapped and mapped != composed:
        return mapped
    target_path = _layer_file_path(target_layer)
    if target_path is None:
        return composed
    target_folder = os.path.dirname(target_path)
    prop_name = attr.GetName()
    for prim_spec in attr.GetPrim().GetPrimStack():
        spec_path = _layer_file_path(prim_spec.layer)
        if spec_path is not None and os.path.dirname(spec_path) == target_folder:
            return prim_spec.path.AppendProperty(prop_name)
    return composed


def _attr_spec_edit(
    attr: pxr.Usd.Attribute | None,
    target_layer: Sdf.Layer | None = None,
    apply_api: str | None = None,
) -> GainSpecEdit | None:
    """Build a :class:`GainSpecEdit` from a live attribute's composed value, or None.

    Args:
        attr: Attribute whose composed value should be persisted.
        target_layer: Layer the edit will be authored on, used to translate the
            property path into that layer's own namespace.  None keeps the
            composed stage path.
        apply_api: API schema to apply on the prim spec alongside the attribute.

    Returns:
        The edit to author, or None when ``attr`` is missing/invalid/valueless.
    """
    if attr is None or not attr.IsValid():
        return None
    value = attr.Get()
    if value is None:
        return None
    try:
        prop_path = _asset_local_prop_path(attr, target_layer)
    except (*_USD_ACCESS_ERRORS, OSError):
        prop_path = attr.GetPath()
    return GainSpecEdit(prop_path=prop_path, type_name=attr.GetTypeName(), value=value, apply_api=apply_api)


def build_gain_save_plan(
    joint_entries: list,
    stage: Usd.Stage | None,
    drive_target_identifier: str | None = None,
    *,
    mirror_enabled: bool = True,
    mirror_drive_to_mjc: bool = False,
    mjc_target_identifier: str | None = None,
    newton_target_identifier: str | None = None,
    articulation_root_path: str | None = None,
    mujoco_solver_active: bool = False,
) -> dict[str, list[GainSpecEdit]]:
    """Build a per-layer plan of attribute edits for persisting tuned gains.

    The plan maps each save-layer identifier to the list of :class:`GainSpecEdit`
    to author there.  DriveAPI stiffness/damping/type are always routed to
    ``drive_target_identifier`` (defaulting to the resolved neutral
    ``physics.usda`` layer).

    MuJoCo-native (``mjc:*``) writes follow a three-state model:

    1. **DriveAPI active + mirror off (default):** the plan contains **only** the
       DriveAPI edits — it never authors any ``mjc:*`` actuator prim.  This is the
       safety default so users cannot accidentally overwrite MuJoCo gains.
    2. **DriveAPI active + mirror on** (``mirror_drive_to_mjc=True``): the tuned
       DriveAPI Kp/Kd are also mirrored into the joint's ``mjc:*`` params (routed
       to ``mjc_target_identifier``, defaulting to the ``mujoco.usda`` candidate).
       MuJoCo stays a mirror — this does not change which source is active.
    3. **MuJoCo solver active** (``mujoco_solver_active=True``): for joints whose
       edited active source is ``mjc:*`` (no Newton actuator), the edited ``mjc:*``
       arrays are preserved (read back from the joint's own arrays).  This path is
       independent of ``mirror_drive_to_mjc``.

    Newton actuator gains are written back to ``newton_target_identifier``
    (defaulting to the actuator prim's own defining layer) whenever
    ``mirror_enabled`` is True (the default) — this persists the actuator's own
    active gains and writes only ``newton:*`` attrs, never ``mjc:*``.

    Args:
        joint_entries: Joint table rows (objects with ``joint`` and optional
            ``drive_axis``).
        stage: USD stage hosting the robot.
        drive_target_identifier: Identifier of the layer to write DriveAPI gains
            to.  Pass None to use the resolved default (neutral ``physics.usda``).
        mirror_enabled: Whether to write tuned Newton actuator gains back to their
            layer (default ON).  Does not govern the MuJoCo mirror.
        mirror_drive_to_mjc: Opt-in DriveAPI->MuJoCo mirror (default OFF).  When
            True, tuned DriveAPI Kp/Kd are mirrored into the joint's ``mjc:*``
            params.  Ignored under ``mujoco_solver_active`` where mjc is edited
            directly as the active source.
        mjc_target_identifier: Override layer identifier for mjc mirror edits.
            None routes to the ``mujoco.usda`` candidate / source defining layer.
        newton_target_identifier: Override layer identifier for Newton writeback.
            None routes to the actuator prim's own defining layer.
        articulation_root_path: Path of the articulation root (for Newton
            actuator discovery).
        mujoco_solver_active: When True, MuJoCo-native gains are the edited active
            source for joints that have them (and no actuator); their ``mjc:*``
            edits are preserved (read back from the joint's own mjc arrays).

    Returns:
        Dict mapping layer save identifiers to the edits to author on that layer.
    """
    plan: dict[str, list[GainSpecEdit]] = {}
    if not joint_entries or stage is None:
        return plan

    def _append(layer: Sdf.Layer | None, edit: GainSpecEdit | None) -> None:
        if layer is None or edit is None:
            return
        identifier = get_layer_save_identifier(layer)
        if identifier is None:
            return
        plan.setdefault(identifier, []).append(edit)

    targets = resolve_gain_write_targets(
        joint_entries,
        stage,
        articulation_root_path,
        mirror_enabled=mirror_enabled,
        mirror_drive_to_mjc=mirror_drive_to_mjc,
        mjc_target_identifier=mjc_target_identifier,
        newton_target_identifier=newton_target_identifier,
    )
    drive_identifier = drive_target_identifier
    if drive_identifier is None and targets.options.default is not None:
        drive_identifier = targets.options.default.identifier
    drive_layer = find_layer_by_save_identifier(drive_identifier) if drive_identifier else None

    # mjc writes happen for the opt-in mirror (Part B) or the MuJoCo-solver-active
    # direct-edit path; the actuator map is needed both for the Newton writeback
    # and to know whether a joint's active source is an actuator (actuator wins).
    want_mjc = mirror_drive_to_mjc or mujoco_solver_active
    mjc_map = build_mjc_gain_map(stage) if want_mjc else {}
    actuator_map = (
        build_actuator_gain_map(stage, articulation_root_path)
        if articulation_root_path and (mirror_enabled or want_mjc)
        else {}
    )
    # Resolve the mirror target layers once (None -> route to each source's own
    # defining layer).  targets.* fold in the caller override + default resolution.
    mjc_override_layer = find_layer_by_save_identifier(targets.mjc_target_identifier) if mjc_target_identifier else None
    newton_override_layer = (
        find_layer_by_save_identifier(targets.newton_target_identifier)
        if (mirror_enabled and newton_target_identifier)
        else None
    )
    # Advanced joint params are authored on both schemas, so each half is routed
    # to the layer that owns it: the engine-neutral ``newton:*`` opinion joins the
    # DriveAPI gains on the selected (``physics.usda``) target, while the PhysX
    # mirror belongs on the ``physx.usda`` overlay.
    physx_overlay_layer = _physx_overlay_layer(targets.options)

    for entry in joint_entries:
        joint = entry.joint
        drive_axis = getattr(entry, "drive_axis", None)
        joint_path = joint.GetPath().pathString

        # DriveAPI stiffness / damping / type -> selected target layer.
        if drive_layer is not None:
            for attr in (
                get_stiffness_attr(joint, drive_axis),
                get_damping_attr(joint, drive_axis),
                get_joint_drive_type_attr(joint, drive_axis),
            ):
                _append(drive_layer, _attr_spec_edit(attr, drive_layer))

        # MuJoCo-native writes (three-state model):
        #   (3) MuJoCo solver active + no actuator: mjc is the edited ACTIVE
        #       source -> preserve the joint's edited mjc arrays (read them back).
        #   (2) opt-in mirror on: derive mjc from the tuned DriveAPI Kp/Kd.
        #   (1) otherwise (default): mjc:* is never authored.
        direct_mjc = mujoco_solver_active and joint_path not in actuator_map
        if joint_path in mjc_map and (direct_mjc or mirror_drive_to_mjc):
            source = mjc_map[joint_path]
            if direct_mjc:
                gain_prm = (
                    source.gain_prm_attr.Get() if (source.gain_prm_attr and source.gain_prm_attr.IsValid()) else None
                )
                bias_prm = (
                    source.bias_prm_attr.Get() if (source.bias_prm_attr and source.bias_prm_attr.IsValid()) else None
                )
                kp, kd = mjc_params_to_drive_gains(gain_prm, bias_prm)
            else:
                stiffness_attr = get_stiffness_attr(joint, drive_axis)
                damping_attr = get_damping_attr(joint, drive_axis)
                kp = float(stiffness_attr.Get()) if (stiffness_attr and stiffness_attr.Get() is not None) else 0.0
                kd = float(damping_attr.Get()) if (damping_attr and damping_attr.Get() is not None) else 0.0
            params = drive_gains_to_mjc(kp, kd)
            mjc_layer = mjc_override_layer or source.defining_layer
            actuator_path = Sdf.Path(source.actuator_path)
            _append(
                mjc_layer,
                GainSpecEdit(
                    actuator_path.AppendProperty("mjc:gainPrm"), Sdf.ValueTypeNames.FloatArray, params.gain_prm
                ),
            )
            _append(
                mjc_layer,
                GainSpecEdit(
                    actuator_path.AppendProperty("mjc:biasPrm"), Sdf.ValueTypeNames.FloatArray, params.bias_prm
                ),
            )
            _append(
                mjc_layer,
                GainSpecEdit(actuator_path.AppendProperty("mjc:gainType"), Sdf.ValueTypeNames.String, params.gain_type),
            )
            _append(
                mjc_layer,
                GainSpecEdit(actuator_path.AppendProperty("mjc:biasType"), Sdf.ValueTypeNames.String, params.bias_type),
            )
            # Mirror joint-level mjc:stiffness / mjc:damping when authored.
            if source.joint_stiffness_attr is not None and source.joint_stiffness_attr.IsValid():
                _append(
                    mjc_override_layer or get_defining_layer(source.joint_stiffness_attr) or mjc_layer,
                    GainSpecEdit(source.joint_stiffness_attr.GetPath(), Sdf.ValueTypeNames.Double, kp),
                )
            if source.joint_damping_attr is not None and source.joint_damping_attr.IsValid():
                _append(
                    mjc_override_layer or get_defining_layer(source.joint_damping_attr) or mjc_layer,
                    GainSpecEdit(source.joint_damping_attr.GetPath(), Sdf.ValueTypeNames.Double, kd),
                )

        # Advanced joint params (armature / friction / velocity limit) live on
        # both the Newton and PhysX joint schemas.  Only *authored* opinions are
        # persisted: applying either schema makes all of its attributes
        # resolvable at their fallbacks, so persisting whatever resolves would
        # stamp an untouched joint's `maxJointVelocity = inf` into the asset as
        # though the user had chosen it.
        for spec in JOINT_PARAM_SPECS:
            newton_attr, physx_attr = authored_joint_param_attrs(joint, spec)
            # The PhysX mirror belongs on the physx.usda overlay; with no such
            # overlay it stays on its own layer rather than leaking a PhysX
            # opinion into the engine-neutral target.
            for attr, layer, api in (
                (newton_attr, drive_layer, NEWTON_JOINT_API),
                (physx_attr, physx_overlay_layer, PHYSX_JOINT_API),
            ):
                if attr is None:
                    continue
                layer = layer or get_defining_layer(attr)
                if layer is None:
                    continue
                _append(layer, _attr_spec_edit(attr, layer, apply_api=api))

        # Newton actuator writeback: persist tuned actuator gains to their layer.
        if mirror_enabled and joint_path in actuator_map:
            actuator = actuator_map[joint_path]
            for attr in (actuator.kp_attr, actuator.kd_attr, actuator.ki_attr):
                newton_layer = newton_override_layer or get_defining_layer(attr)
                _append(newton_layer, _attr_spec_edit(attr, newton_layer))

    return plan


def _apply_api_on_spec(prim_spec: Sdf.PrimSpec, api_name: str) -> None:
    """Prepend ``api_name`` to a prim spec's ``apiSchemas``, if not already listed.

    Authored directly on the spec rather than through ``Usd.Prim.ApplyAPI`` so the
    application lands in the same layer as the value it qualifies, which is what
    makes the saved layer self-contained.
    """
    listop = prim_spec.GetInfo("apiSchemas") if prim_spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
    if api_name in listop.explicitItems or api_name in listop.prependedItems or api_name in listop.appendedItems:
        return
    listop.prependedItems = list(listop.prependedItems) + [api_name]
    prim_spec.SetInfo("apiSchemas", listop)


def apply_gain_save_plan(plan: dict[str, list[GainSpecEdit]], *, save: bool = False) -> list[str]:
    """Author a save plan onto its target layers via the Sdf API.

    Uses direct spec authoring so it works whether or not the target layers are
    part of a composed stage (overlay layers such as ``physx.usda`` /
    ``mujoco.usda`` are sublayers that may not be in the active stage stack).

    Args:
        plan: Per-layer edits from :func:`build_gain_save_plan`.
        save: When True, persist each written layer that permits saving.

    Returns:
        Identifiers of the layers that were successfully authored.
    """
    written: list[str] = []
    for identifier, edits in plan.items():
        layer = find_layer_by_save_identifier(identifier) or Sdf.Layer.Find(identifier)
        if layer is None:
            continue
        try:
            with Sdf.ChangeBlock():
                for edit in edits:
                    prim_spec = Sdf.CreatePrimInLayer(layer, edit.prop_path.GetPrimPath())
                    attr_spec = prim_spec.attributes.get(edit.prop_path.name)
                    if attr_spec is None:
                        attr_spec = Sdf.AttributeSpec(prim_spec, edit.prop_path.name, edit.type_name)
                    attr_spec.default = edit.value
                    if edit.apply_api:
                        _apply_api_on_spec(prim_spec, edit.apply_api)
        except _USD_ACCESS_ERRORS:
            continue
        written.append(identifier)
        if save and layer.permissionToSave:
            try:
                layer.Save()
            except (Tf.ErrorException, RuntimeError):
                pass
    return written
