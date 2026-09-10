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

"""Copy a joint's advanced parameters from the active backend to the other one.

Editing an advanced joint parameter authors only the active backend's schema (see
:mod:`isaacsim.robot_setup.gain_tuner.joint_schema_attrs`), because the PhysX and
Newton formulations differ and each is tuned on its own.  So the two backends can
hold different values, and that is a legitimate authoring choice, not a fault.

Sometimes it is not the intent, though: a robot imported with only ``newton:*``
authored has nothing for PhysX to read, and someone who has finished tuning under
one backend may want the same number under the other as a starting point.  This
module is that action, and nothing more.  It is opt-in and directed: it takes the
value the active backend has in effect and writes it to the *other* backend's
schema, which changes what that backend simulates.

Editing the field cannot express this.  A ``ui.SimpleFloatModel`` fires no
callback when the user retypes the value it already has, and an edit would land
on the active backend's schema anyway -- never the other one.

Each joint is one :class:`CopyJointParamsToOtherBackendCommand`, so the edit is
undoable, and a multi-joint copy brackets the commands in ``omni.kit.undo.group()``
-- the same grouping the sibling ``isaacsim.physics.newton.ui`` schema widgets use
-- so it undoes as one step.  A parameter whose two backends already hold the same
value is skipped, so copying a settled robot writes nothing.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import omni.kit.undo
from isaacsim.robot_setup.gain_tuner import (
    JOINT_PARAM_SPECS,
    SCHEMA_JOINT_APIS,
    SCHEMA_LABELS,
    JointParamResolution,
    author_joint_param,
    backend_display_label,
    backend_write_schema,
    joint_param_spec,
    other_backend,
    resolve_joint_params,
)
from omni.kit.commands import Command, execute, register_all_commands_in_module
from pxr import Usd

# USD access can fail with these when a prim or attribute is missing, invalid, a
# duck-typed fake (unit tests substitute plain objects), or holds an unexpected
# value type.  Caught at the defensive USD boundaries below so that any
# *unexpected* error type still propagates.
_USD_ACCESS_ERRORS = (AttributeError, TypeError, ValueError)

# Advanced-parameter keys the copy action covers: every multi-schema parameter.
COPYABLE_PARAM_KEYS: tuple[str, ...] = tuple(spec.key for spec in JOINT_PARAM_SPECS)
"""The :attr:`JointParamSpec.key` values the copy action can write."""


@dataclass(frozen=True)
class JointParamCopyTarget:
    """One joint with advanced parameters that can be copied to the other backend."""

    joint_path: str
    """Prim path of the joint."""

    display_name: str
    """Joint name as shown in the gain table."""

    resolutions: tuple  # tuple[JointParamResolution, ...]
    """The joint's copyable parameters, in :data:`JOINT_PARAM_SPECS` order: those
    whose active-backend value is a number the other backend does not already
    hold."""


@dataclass
class JointParamCopyResult:
    """What a :func:`copy_joint_params_to_other_backend` call actually wrote."""

    joints_written: int = 0
    """Number of joints that had at least one parameter written."""

    params_written: int = 0
    """Total number of parameters written across those joints."""

    target_backend: str = ""
    """The backend label the values were written to."""

    unwritable_attrs: list = field(default_factory=list)
    """Attribute names a write could not reach, most often the ``newton:*`` half
    with ``omni.usd.schema.newton`` disabled.  Non-empty means the copy did not
    fully land and the caller should surface that."""

    @property
    def wrote_nothing(self) -> bool:
        """True when nothing was written (every joint already matched)."""
        return self.joints_written == 0


def copyable_joint_params(joint: object, backend: str, solver: str = "") -> list[JointParamResolution]:
    """Return the parameters of one joint that a copy would actually write.

    A parameter is copyable when the active backend has a value in effect and the
    other backend does not already simulate that same value.  An unlimited velocity
    limit counts: ``inf`` is what the other schema wants in order to mean the same
    thing, so it is copied rather than skipped.  An undetermined chain, a parameter
    nothing is authored for, and an unsupported backend have nothing to copy and are
    skipped rather than guessed at.

    Args:
        joint: The joint prim.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        The copyable :class:`JointParamResolution` values, in
        :data:`JOINT_PARAM_SPECS` order.
    """
    copyable: list[JointParamResolution] = []
    for resolution in resolve_joint_params(joint, backend, solver):
        if resolution.copy_value is None or not resolution.backend_supported:
            continue
        if resolution.other_backend_agrees:
            continue
        copyable.append(resolution)
    return copyable


def joint_param_copy_targets(entries: object, backend: str, solver: str = "") -> list[JointParamCopyTarget]:
    """Find the joints among ``entries`` that a copy would write something for.

    Args:
        entries: Iterable of ``JointListEntry``-like objects exposing ``joint``
            and ``display_name``.
        backend: The active backend label (the copy source).
        solver: The running Newton solver token.

    Returns:
        One :class:`JointParamCopyTarget` per affected joint, in the order the
        entries were given.  Empty when every joint already holds the same values
        on both backends, which is the no-op path for the copy action.
    """
    targets: list[JointParamCopyTarget] = []
    for entry in entries or []:
        joint = getattr(entry, "joint", None)
        if joint is None:
            continue
        resolutions = copyable_joint_params(joint, backend, solver)
        if not resolutions:
            continue
        try:
            joint_path = joint.GetPath().pathString
        except _USD_ACCESS_ERRORS:
            continue
        targets.append(
            JointParamCopyTarget(
                joint_path=joint_path,
                display_name=str(getattr(entry, "display_name", joint_path)),
                resolutions=tuple(resolutions),
            )
        )
    return targets


def _authored_or_none(joint: Usd.Prim, attr_name: str) -> float | None:
    """Return an attribute's explicitly authored value, or None when unauthored."""
    try:
        attr = joint.GetAttribute(attr_name)
        if not attr or not attr.IsValid() or not attr.HasAuthoredValue():
            return None
        value = attr.Get()
    except _USD_ACCESS_ERRORS:
        return None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class CopyJointParamsToOtherBackendCommand(Command):
    """Copy one joint's advanced parameters to the inactive backend, undoably.

    The value written is the one the active backend has in effect
    (:func:`~isaacsim.robot_setup.gain_tuner.resolve_joint_param`), so the action
    only ever commits a number the user has already seen.  It lands on the other
    backend's schema, which changes what that backend simulates.

    Parameters whose two backends already hold the same value are not touched, so
    executing this on a settled joint writes nothing and undoes to the same state.

    Args:
        stage: The stage the joint lives on.
        joint_path: Prim path of the joint to copy.
        backend: The active backend label (the copy source).
        solver: The running Newton solver token.
        param_keys: Restrict the copy to these :attr:`JointParamSpec.key` values.
            None (the default) copies every copyable parameter.
    """

    @classmethod
    def execute(cls, *args: Any, **kwargs: Any) -> Any:
        """Bind the constructor signature and dispatch through the command registry.

        Args:
            *args: Positional arguments forwarded to the constructor.
            **kwargs: Keyword arguments forwarded to the constructor.

        Returns:
            The result of the registered command execution.
        """
        sig = inspect.signature(cls.__init__)
        bound = sig.bind(0, *args, **kwargs)
        bound.arguments.pop("self")
        return execute(cls.__name__, **bound.kwargs)

    def __init__(
        self,
        stage: Usd.Stage,
        joint_path: str,
        backend: str,
        solver: str = "",
        param_keys: list | None = None,
    ) -> None:
        self._stage = stage
        self._joint_path = str(joint_path)
        self._backend = str(backend)
        self._solver = str(solver)
        self._param_keys = list(param_keys) if param_keys is not None else None
        # Pre-copy state, captured in do() so undo can restore it: per parameter
        # key the previously authored value of the attribute written (None =
        # unauthored), plus the API schema this command had to apply.
        self._previous: dict[str, float | None] = {}
        self._applied_apis: list[str] = []

    def _selected(self, joint: Usd.Prim) -> list[JointParamResolution]:
        """Return the joint's copyable parameters this command should write."""
        resolutions = copyable_joint_params(joint, self._backend, self._solver)
        if self._param_keys is None:
            return resolutions
        return [r for r in resolutions if r.spec.key in self._param_keys]

    def do(self) -> JointParamCopyResult:
        """Write the active backend's value onto the other backend's schema.

        Returns:
            A :class:`JointParamCopyResult` for this joint alone;
            ``wrote_nothing`` is True when its backends already matched.
        """
        target = other_backend(self._backend)
        result = JointParamCopyResult(target_backend=target or "")
        if target is None:
            # The active engine is not one whose joint schema is known, so there is
            # no "other backend" to copy to.  Refusing beats writing physxJoint:*
            # on the chance that whatever is running reads it.
            return result
        joint = self._stage.GetPrimAtPath(self._joint_path) if self._stage is not None else None
        if joint is None or not joint.IsValid():
            return result
        resolutions = self._selected(joint)
        if not resolutions:
            return result

        target_schema = backend_write_schema(target)
        api_name = SCHEMA_JOINT_APIS[target_schema]
        try:
            if not joint.HasAPI(api_name):
                self._applied_apis.append(api_name)
        except _USD_ACCESS_ERRORS:
            pass

        for resolution in resolutions:
            spec = resolution.spec
            attr_name = spec.attr_for_schema(target_schema)
            if attr_name is None:
                continue
            self._previous[spec.key] = _authored_or_none(joint, attr_name)
            written = author_joint_param(joint, spec, resolution.copy_value, target)
            if written is None:
                self._previous.pop(spec.key, None)
                result.unwritable_attrs.append(attr_name)
                continue
            result.params_written += 1
        if result.params_written:
            result.joints_written = 1
        return result

    def undo(self) -> None:
        """Restore the authored values (and applied schema) the copy replaced."""
        joint = self._stage.GetPrimAtPath(self._joint_path) if self._stage is not None else None
        if joint is None or not joint.IsValid():
            return
        other = other_backend(self._backend)
        target_schema = None if other is None else backend_write_schema(other)
        if target_schema is None:
            return
        for key, previous in self._previous.items():
            spec = joint_param_spec(key)
            if spec is None:
                continue
            attr_name = spec.attr_for_schema(target_schema)
            if attr_name is None:
                continue
            try:
                attr = joint.GetAttribute(attr_name)
                if not attr or not attr.IsValid():
                    continue
                if previous is None:
                    attr.Clear()
                else:
                    attr.Set(previous)
            except _USD_ACCESS_ERRORS:
                continue
        for api_name in self._applied_apis:
            try:
                joint.RemoveAPI(api_name)
            except _USD_ACCESS_ERRORS:
                continue
        self._previous = {}
        self._applied_apis = []


def copy_joint_params_to_other_backend(
    stage: Usd.Stage,
    targets: list,
    backend: str,
    solver: str = "",
    param_keys: list | None = None,
) -> JointParamCopyResult:
    """Copy every target joint's advanced parameters as one undo step.

    Args:
        stage: The stage the joints live on.
        targets: The :class:`JointParamCopyTarget` values to copy (from
            :func:`joint_param_copy_targets`).
        backend: The active backend label (the copy source).
        solver: The running Newton solver token.
        param_keys: Restrict the copy to these :attr:`JointParamSpec.key` values.
            None copies every copyable parameter on each joint.

    Returns:
        The combined :class:`JointParamCopyResult`.  ``wrote_nothing`` is True for
        an empty target list, so calling this when nothing differs is a safe no-op
        rather than a rewrite of unchanged values.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner.ui.joint_param_sync import (
        ...     copy_joint_params_to_other_backend,
        ...     joint_param_copy_targets,
        ... )

        >>> targets = joint_param_copy_targets(entries, "PhysX")  # doctest: +NO_CHECK
        >>> copy_joint_params_to_other_backend(stage, targets, "PhysX").params_written  # doctest: +NO_CHECK
        2
    """
    combined = JointParamCopyResult(target_backend=other_backend(backend) or "")
    if not targets:
        return combined
    # Grouped so the whole copy is a single undo step, matching how the sibling
    # Newton UI schema widgets bracket their multi-prim edits.  An empty group is
    # dropped from the undo stack so a copy that wrote nothing leaves no entry.
    with omni.kit.undo.group(remove_if_empty=True):
        for target in targets:
            _success, result = CopyJointParamsToOtherBackendCommand.execute(
                stage=stage,
                joint_path=target.joint_path,
                backend=backend,
                solver=solver,
                param_keys=param_keys,
            )
            if not isinstance(result, JointParamCopyResult):
                continue
            combined.joints_written += result.joints_written
            combined.params_written += result.params_written
            combined.unwritable_attrs.extend(result.unwritable_attrs)
    return combined


def copy_summary_text(result: JointParamCopyResult) -> str:
    """Describe what a copy wrote, for a toast in the established UI voice.

    Args:
        result: The result to describe.

    Returns:
        An ASCII sentence.  Says nothing was written when both backends already
        held the same values, and names the unreachable attributes when a write
        did not land.
    """
    target = backend_display_label(result.target_backend)
    if result.wrote_nothing:
        return f"{target} already simulates these advanced joint parameter values. Nothing was written."
    params = f"{result.params_written} parameter{'s' if result.params_written != 1 else ''}"
    joints = f"{result.joints_written} joint{'s' if result.joints_written != 1 else ''}"
    text = f"Copied {params} on {joints} to {target}. {target} now simulates the copied values."
    if result.unwritable_attrs:
        missing = ", ".join(sorted(set(result.unwritable_attrs)))
        text += f" Could not write {missing}; check that omni.usd.schema.newton is enabled."
    return text


def copy_target_schema_label(backend: str) -> str:
    """Return the attribute namespace a copy from ``backend`` writes to.

    Args:
        backend: The *active* backend label (the copy source).

    Returns:
        The other backend's schema label, e.g. ``"physxJoint:*"``, or ``""`` when
        ``backend`` is not a supported backend and has no other side.
    """
    other = other_backend(backend)
    schema = None if other is None else backend_write_schema(other)
    return SCHEMA_LABELS.get(schema, "")


register_all_commands_in_module(__name__)
