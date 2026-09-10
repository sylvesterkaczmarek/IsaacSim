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

"""Derive and bind an Isaac Lab policy interface for runtime use.

Vocabulary: the exporter writes one *record* per term into the descriptor, keyed by the
*full_path* of the Isaac Lab function or action class that produced it; derivation turns each
record into a :class:`Term` whose *semantic* is this runtime's name for that implementation.
:func:`derive_binding` maps a descriptor to ordered terms, :func:`bind_policy` resolves their
joint names against a deployed robot, and :class:`BoundPolicy` reads and decodes from them.

Observation processing is ``clip(raw, lower, upper) * scale``; action processing is
``clip(raw * scale + offset, lower, upper)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .application import RobotStateReadSpec

if TYPE_CHECKING:
    import isaacsim.robot_motion.experimental.motion_generation as mg

__all__ = [
    "BoundPolicy",
    "PolicyBinding",
    "Term",
    "bind_policy",
    "derive_binding",
]


# ---------------------------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One observation or action term of a policy interface.

    Args:
        semantic: Vocabulary key naming what the term reads or commands (e.g. ``joint_pos_rel``).
        width: Exact element count of the term's flat observation or action slice.
        joint_names: Literal articulation DOF names, one per element, for joint-bound terms.
        offsets: Per-element offsets: the defaults a ``*_rel`` observation subtracts, or the
            affine offset an action term adds after scaling.
        scale: Authored scale as exported — a number or a per-element sequence — or None.
        clip: Authored clip as exported — one (lower, upper) pair for an observation,
            per-element pairs for an action — or None.
        history_length: Number of frames retained for an observation term.
        task_refs: Explicit task-context namespace names read by task observation terms.
        binary_positions: Closed/open targets for a ``binary_joint_position`` action.

    Raises:
        ValueError: If ``history_length`` is less than one.
    """

    semantic: str
    width: int
    joint_names: tuple[str, ...] = ()
    offsets: tuple[float, ...] = ()
    scale: float | Sequence[float] | None = None
    clip: Sequence | None = None
    history_length: int = 1
    task_refs: tuple[str, ...] = ()
    binary_positions: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the history length."""
        if self.history_length < 1:
            raise ValueError(f"Term history_length must be at least 1, got {self.history_length}.")


@dataclass(frozen=True)
class PolicyBinding:
    """Complete ordered observation/action interface of one policy.

    Args:
        observation_terms: Ordered terms whose slices concatenate into the flat model input.
        action_terms: Ordered terms whose slices concatenate into the flat raw model output.
    """

    observation_terms: tuple[Term, ...]
    action_terms: tuple[Term, ...]


# ---------------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------------

#: Action semantic key to per-DOF control mode.
_ACTION_CONTROL_MODES: dict[str, str] = {
    "binary_joint_position": "position",
    "joint_position": "position",
    "joint_velocity": "velocity",
    "joint_effort": "effort",
}

# Every descriptor record carries a ``full_path``: the Python dotted path of the Isaac Lab mdp
# function (observations) or action class (actions) that produced the term at training time.
# Those are training-side isaaclab modules, not Isaac Sim ones. The tables below enumerate the
# implementations this runtime reproduces, keyed by that path; ``name`` is not used because it
# is the user-chosen term label, while ``full_path`` identifies the actual math.
_OBSERVATION_MODULE = "isaaclab.envs.mdp.observations"
_ACTION_TERM_MODULE = "isaaclab.envs.mdp.actions.joint_actions"

#: Canonical Isaac Lab observation term paths mapped to runtime semantics.
_OBSERVATION_PATHS: dict[str, str] = {
    f"{_OBSERVATION_MODULE}.base_lin_vel": "base_lin_vel",
    f"{_OBSERVATION_MODULE}.base_ang_vel": "base_ang_vel",
    f"{_OBSERVATION_MODULE}.projected_gravity": "projected_gravity",
    f"{_OBSERVATION_MODULE}.generated_commands": "generated_commands",
    f"{_OBSERVATION_MODULE}.joint_pos": "joint_pos",
    f"{_OBSERVATION_MODULE}.joint_pos_rel": "joint_pos_rel",
    f"{_OBSERVATION_MODULE}.joint_vel": "joint_vel",
    f"{_OBSERVATION_MODULE}.joint_vel_rel": "joint_vel_rel",
    f"{_OBSERVATION_MODULE}.last_action": "last_action",
}

#: Canonical Isaac Lab action class paths mapped to runtime semantics.
_ACTION_PATHS: dict[str, str] = {
    f"{_ACTION_TERM_MODULE}.JointPositionAction": "joint_position",
    f"{_ACTION_TERM_MODULE}.JointVelocityAction": "joint_velocity",
    f"{_ACTION_TERM_MODULE}.JointEffortAction": "joint_effort",
}

_GRAVITY_DIRECTION_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_JOINT_OBSERVATIONS = frozenset({"joint_pos", "joint_pos_rel", "joint_vel", "joint_vel_rel"})

# ---------------------------------------------------------------------------------------------
# Derivation from an exported IO descriptor
# ---------------------------------------------------------------------------------------------


def _observation_term_from_record(record: Mapping) -> Term:
    """Build one observation :class:`Term` from an IO descriptor record.

    A record is one entry of the export's ordered ``observations.policy`` list — the exporter's
    per-term mapping carrying ``full_path``, ``shape``, authored ``overloads``, and for joint
    terms the ``joint_names`` and default-offset arrays baked into the term.

    Args:
        record: One ordered entry of the descriptor's term list.

    Returns:
        The resulting :class:`Term`.
    """
    full_path = record.get("full_path")
    semantic = _OBSERVATION_PATHS.get(full_path)
    if semantic is None:
        raise ValueError(
            f"derive_binding: observation term {full_path!r} is not supported; "
            f"supported terms are {sorted(_OBSERVATION_PATHS)}. Policies whose interface derivation "
            f"cannot reproduce deploy through an explicit PolicySpec(binding=...) hook."
        )
    width = int(np.prod(record["shape"]))
    overloads = record.get("overloads") or {}
    history_length = int(overloads.get("history_length") or 1)
    if history_length < 1:
        raise ValueError(f"derive_binding: observation term {full_path!r} has invalid history length {history_length}.")

    joint_names: tuple[str, ...] = ()
    offsets: tuple[float, ...] = ()
    if semantic in _JOINT_OBSERVATIONS:
        joint_names = tuple(record["joint_names"])
        if semantic == "joint_pos_rel":
            offsets = tuple(record["joint_pos_offsets"])
        elif semantic == "joint_vel_rel":
            offsets = tuple(record["joint_vel_offsets"])

    return Term(
        semantic=semantic,
        width=width,
        joint_names=joint_names,
        offsets=offsets,
        scale=overloads.get("scale"),
        clip=overloads.get("clip") or None,
        history_length=history_length,
    )


def _action_term_from_record(record: Mapping) -> Term:
    """Build one action :class:`Term` from an IO descriptor record.

    A record is one entry of the export's ordered ``actions`` list — the exporter's per-term
    mapping carrying the action class ``full_path``, ``shape``, ``joint_names``, and the
    authored ``scale``/``offset``/``clip``.

    Args:
        record: One ordered entry of the descriptor's term list.

    Returns:
        The resulting :class:`Term`.
    """
    full_path = record.get("full_path")
    semantic = _ACTION_PATHS.get(full_path)
    if semantic is None:
        raise ValueError(
            f"derive_binding: action term {full_path!r} is not supported; "
            f"supported terms are {sorted(_ACTION_PATHS)}. Policies whose interface derivation "
            f"cannot reproduce deploy through an explicit PolicySpec(binding=...) hook."
        )
    width = int(np.prod(record["shape"]))
    joints = tuple(record["joint_names"])
    offset = record.get("offset")
    if isinstance(offset, (list, tuple)):
        offsets = tuple(offset)
    elif offset:
        offsets = (float(offset),) * width
    else:
        offsets = ()

    return Term(
        semantic=semantic,
        width=width,
        joint_names=joints,
        offsets=offsets,
        scale=record.get("scale"),
        clip=record.get("clip") or None,
    )


def derive_binding(io_descriptor: Mapping) -> PolicyBinding:
    """Derive ordered observation and action terms from an Isaac Lab IO descriptor.

    Args:
        io_descriptor: Parsed ``IO_descriptors.yaml`` mapping.

    Returns:
        The resulting :class:`PolicyBinding`.
    """
    observation_terms = tuple(
        _observation_term_from_record(record) for record in io_descriptor["observations"]["policy"]
    )
    action_terms = tuple(_action_term_from_record(record) for record in io_descriptor["actions"])
    return PolicyBinding(observation_terms=observation_terms, action_terms=action_terms)


# ---------------------------------------------------------------------------------------------
# Direct runtime binding
# ---------------------------------------------------------------------------------------------


def _joint_indices(joint_names: tuple[str, ...], joint_space: tuple[str, ...], owner: str) -> np.ndarray:
    """Resolve trained joint names to indices in the deployed robot.

    Args:
        joint_names: Joint names to resolve, in the order required.
        joint_space: Deployed articulation's DOF names, in order.
        owner: Owning object description used in error messages.

    Returns:
        The resulting array.
    """
    index_by_name = {name: index for index, name in enumerate(joint_space)}
    missing = [name for name in joint_names if name not in index_by_name]
    if missing:
        raise ValueError(f"{owner}: joints {missing} are not present in the robot joint space {joint_space}.")
    return np.array([index_by_name[name] for name in joint_names], dtype=np.int64)


def quaternion_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Build the float32 rotation matrix of a WXYZ quaternion.

    Args:
        quaternion: Four-element WXYZ quaternion.

    Returns:
        The resulting array.
    """
    w, x, y, z = (quaternion / np.linalg.norm(quaternion)).astype(np.float32)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _read_observation(
    term: Term,
    joint_indices: np.ndarray | None,
    estimated_state: mg.RobotState,
    setpoint_state: mg.RobotState | None,
    last_action: np.ndarray,
    context: Mapping[str, object] | None,
) -> np.ndarray:
    """Read and process one observation term.

    Args:
        term: Interface term to read or decode.
        joint_indices: Deployed DOF indices for the term's joints, or None.
        estimated_state: Current robot state read from the articulation.
        setpoint_state: Setpoint state carrying the command terms, or None.
        last_action: Raw model output of the previous control tick.
        context: Task-context values for task observation terms, or None.

    Returns:
        The resulting array.
    """
    match term.semantic:
        case "base_lin_vel":
            rotation = quaternion_wxyz_to_rotation_matrix(estimated_state.root.orientation.numpy())
            value = rotation.T @ estimated_state.root.linear_velocity.numpy()

        case "base_ang_vel":
            rotation = quaternion_wxyz_to_rotation_matrix(estimated_state.root.orientation.numpy())
            value = rotation.T @ estimated_state.root.angular_velocity.numpy()

        case "projected_gravity":
            rotation = quaternion_wxyz_to_rotation_matrix(estimated_state.root.orientation.numpy())
            value = rotation.T @ _GRAVITY_DIRECTION_W

        case "generated_commands":
            linear = setpoint_state.root.linear_velocity.numpy()
            angular = setpoint_state.root.angular_velocity.numpy()
            value = np.array([linear[0], linear[1], angular[2]], dtype=np.float32)

        case "joint_pos":
            value = estimated_state.joints.data_array.numpy()[0, joint_indices]

        case "joint_pos_rel":
            value = estimated_state.joints.data_array.numpy()[0, joint_indices]
            if term.offsets:
                value = value - np.asarray(term.offsets, dtype=np.float32)

        case "joint_vel":
            value = estimated_state.joints.data_array.numpy()[1, joint_indices]

        case "joint_vel_rel":
            value = estimated_state.joints.data_array.numpy()[1, joint_indices]
            if term.offsets:
                value = value - np.asarray(term.offsets, dtype=np.float32)

        case "task_joint_pos":
            value = context[f"{term.task_refs[0]}.joint_pos"]

        case "task_joint_vel":
            value = context[f"{term.task_refs[0]}.joint_vel"]

        case "task_frame_rel_pos":
            first_position = context[f"{term.task_refs[0]}.position"]
            second_position = context[f"{term.task_refs[1]}.position"]
            value = first_position - second_position

        case "last_action":
            value = last_action

        case _:
            raise ValueError(f"Unsupported observation semantic {term.semantic!r}.")

    result = np.asarray(value, dtype=np.float32)
    if term.clip is not None or term.scale is not None:
        result = result.copy()
    if term.clip is not None:
        np.clip(result, float(term.clip[0]), float(term.clip[1]), out=result)
    if term.scale is not None:
        result *= np.asarray(term.scale, dtype=np.float32)
    return result


def _decode_action_term(term: Term, raw_action: np.ndarray) -> np.ndarray:
    """Apply one action term's scale, offset, and clip.

    Args:
        term: Interface term to read or decode.
        raw_action: Flat raw model output for this tick.

    Returns:
        The resulting array.
    """
    values = np.asarray(raw_action, dtype=np.float32).reshape(term.width)
    if term.scale is not None:
        values = values * np.asarray(term.scale, dtype=np.float32)
    if term.offsets:
        values = values + np.asarray(term.offsets, dtype=np.float32)
    if term.clip is not None:
        bounds = np.asarray(term.clip, dtype=np.float32).reshape(term.width, 2)
        values = np.clip(values, bounds[:, 0], bounds[:, 1])
    return values


class BoundPolicy:
    """A policy interface resolved against one deployed robot's joint space.

    ``observation_sample_width`` is the size of one current-frame sample returned by
    :meth:`observe`. ``observation_width`` is the history-expanded input size expected by the
    policy model.

    Args:
        binding: Policy interface to resolve or deploy.
        robot_joint_space: Deployed articulation's DOF names, in order.
    """

    def __init__(self, binding: PolicyBinding, robot_joint_space: Sequence[str]) -> None:
        self.binding = binding
        self.robot_joint_space = tuple(str(name) for name in robot_joint_space)
        self.observation_sample_width = sum(term.width for term in binding.observation_terms)
        self.observation_width = sum(term.width * term.history_length for term in binding.observation_terms)
        self.action_width = sum(term.width for term in binding.action_terms)

        semantics = {term.semantic for term in binding.observation_terms}
        self.estimated_state_read_spec = RobotStateReadSpec(
            joint_positions=bool({"joint_pos", "joint_pos_rel"} & semantics),
            joint_velocities=bool({"joint_vel", "joint_vel_rel"} & semantics),
            root_orientation=bool({"base_lin_vel", "base_ang_vel", "projected_gravity"} & semantics),
            root_linear_velocity="base_lin_vel" in semantics,
            root_angular_velocity="base_ang_vel" in semantics,
        )

        observation_indices: list[np.ndarray | None] = []
        for index, term in enumerate(binding.observation_terms):
            owner = f"Observation term {index} ({term.semantic!r})"
            if term.semantic == "last_action" and term.width != self.action_width:
                raise ValueError(f"{owner}: last_action width {term.width} != raw action width {self.action_width}.")

            joint_indices = None
            if term.semantic in _JOINT_OBSERVATIONS:
                joint_indices = _joint_indices(term.joint_names, self.robot_joint_space, owner).reshape(term.width)
            observation_indices.append(joint_indices)
        self._observation_indices = tuple(observation_indices)

        self.joint_control_modes: dict[str, str] = {}
        for index, term in enumerate(binding.action_terms):
            owner = f"Action term {index} ({term.semantic!r})"
            control_mode = _ACTION_CONTROL_MODES.get(term.semantic)
            if control_mode is None:
                raise ValueError(f"{owner}: unsupported action semantic {term.semantic!r}.")

            joint_indices = _joint_indices(term.joint_names, self.robot_joint_space, owner)
            if term.semantic == "binary_joint_position":
                if term.width != 1 or term.binary_positions is None:
                    raise ValueError(f"{owner}: binary position actions require width 1 and closed/open positions.")
            else:
                joint_indices.reshape(term.width)
            for joint_name in term.joint_names:
                self.joint_control_modes[joint_name] = control_mode

    def observe(
        self,
        estimated_state: mg.RobotState,
        setpoint_state: mg.RobotState | None,
        last_action: np.ndarray,
        context: Mapping[str, object] | None,
    ) -> np.ndarray:
        """Build one current-frame observation sample in descriptor term order.

        History-bearing terms are expanded by :class:`IsaacLabPolicyController`; this method
        remains stateless and returns only the current value of each term.

        Args:
            estimated_state: Current robot state read from the articulation.
            setpoint_state: Setpoint state carrying the command terms, or None.
            last_action: Raw model output of the previous control tick.
            context: Task-context values for task observation terms, or None.

        Returns:
            Flat current-frame sample with ``observation_sample_width`` values. The controller
            expands it to the ``observation_width`` model input.
        """
        observations = []
        for term, joint_indices in zip(self.binding.observation_terms, self._observation_indices):
            observation = _read_observation(
                term,
                joint_indices,
                estimated_state,
                setpoint_state,
                last_action,
                context,
            )
            observations.append(observation)
        return np.concatenate(observations)

    def action(self, raw_action: np.ndarray) -> mg.RobotState:
        """Decode the flat model output into a partial named desired robot state.

        Args:
            raw_action: Flat raw model output for this tick.

        Returns:
            The resulting mg.RobotState.
        """
        import isaacsim.robot_motion.experimental.motion_generation as mg
        import warp as wp

        position_names = []
        position_values = []
        velocity_names = []
        velocity_values = []
        effort_names = []
        effort_values = []

        start = 0
        for term in self.binding.action_terms:
            stop = start + term.width
            values = _decode_action_term(term, raw_action[start:stop])
            start = stop

            match term.semantic:
                case "binary_joint_position":
                    target = term.binary_positions[0] if values[0] < 0.0 else term.binary_positions[1]
                    position_names.extend(term.joint_names)
                    position_values.append(np.full(len(term.joint_names), target, dtype=np.float32))
                case "joint_position":
                    position_names.extend(term.joint_names)
                    position_values.append(values)
                case "joint_velocity":
                    velocity_names.extend(term.joint_names)
                    velocity_values.append(values)
                case "joint_effort":
                    effort_names.extend(term.joint_names)
                    effort_values.append(values)

        positions = None
        if position_names:
            positions = (position_names, wp.array(np.concatenate(position_values), dtype=wp.float32))

        velocities = None
        if velocity_names:
            velocities = (velocity_names, wp.array(np.concatenate(velocity_values), dtype=wp.float32))

        efforts = None
        if effort_names:
            efforts = (effort_names, wp.array(np.concatenate(effort_values), dtype=wp.float32))

        joints = mg.JointState.from_name(
            robot_joint_space=list(self.robot_joint_space),
            positions=positions,
            velocities=velocities,
            efforts=efforts,
        )
        return mg.RobotState(joints=joints)


def bind_policy(binding: PolicyBinding, *, robot_joint_space: Sequence[str]) -> BoundPolicy:
    """Resolve a policy binding's trained joint names against a deployed robot.

    Args:
        binding: Policy interface to resolve or deploy.
        robot_joint_space: Deployed articulation's DOF names, in order.

    Returns:
        The resulting :class:`BoundPolicy`.
    """
    return BoundPolicy(binding, robot_joint_space)
