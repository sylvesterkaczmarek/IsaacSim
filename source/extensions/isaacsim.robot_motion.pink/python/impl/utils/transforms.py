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

"""Provides utilities for transforming between Isaac Sim (position, quaternion) and Pinocchio SE3 representations."""

from __future__ import annotations

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import pinocchio as pin
import warp as wp


def isaac_sim_position_quaternion_to_se3(
    position: np.ndarray | wp.array | list[float],
    quaternion: np.ndarray | wp.array | list[float],
) -> pin.SE3:
    """Convert Isaac Sim (position, quaternion) to a Pinocchio SE3 transform.

    Args:
        position: Translation [x, y, z].
        quaternion: Orientation as quaternion [w, x, y, z] (Isaac Sim convention).

    Returns:
        Pinocchio SE3 rigid-body transform.

    Raises:
        ValueError: If position is not size 3 or quaternion is not size 4.
    """
    position = _to_numpy(position).flatten()
    quaternion = _to_numpy(quaternion).flatten()

    if position.size != 3:
        raise ValueError(f"Position must have 3 elements, got {position.size}.")
    if quaternion.size != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {quaternion.size}.")

    # Isaac Sim: (w, x, y, z) -> Pinocchio quaternion: (x, y, z, w)
    quat_pin = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = pin.Quaternion(quat_pin).toRotationMatrix()
    return pin.SE3(rotation, position)


def se3_to_isaac_sim_position_quaternion(
    transform: pin.SE3,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Pinocchio SE3 transform to Isaac Sim (position, quaternion).

    Args:
        transform: Pinocchio SE3 rigid-body transform.

    Returns:
        Tuple of (position, quaternion) where position is shape (3,) and
        quaternion is shape (4,) in (w, x, y, z) format.
    """
    position = transform.translation.copy()
    quat_pin = pin.Quaternion(transform.rotation)
    # Pinocchio: (x, y, z, w) -> Isaac Sim: (w, x, y, z)
    quaternion = np.array([quat_pin.w, quat_pin.x, quat_pin.y, quat_pin.z])
    return position, quaternion


def map_joint_positions_to_pinocchio(
    joint_names: list[str],
    joint_positions: np.ndarray,
    model: pin.Model,
    q_current: np.ndarray | None = None,
) -> np.ndarray:
    """Map Isaac Sim joint positions to a Pinocchio configuration vector.

    Builds a full Pinocchio configuration vector by placing the provided joint values
    at their correct indices in the model. Joints not in ``joint_names`` retain values
    from ``q_current`` (or the model neutral pose if not given).

    Args:
        joint_names: Ordered joint names matching ``joint_positions``.
        joint_positions: Joint position values corresponding to ``joint_names``.
        model: Pinocchio model providing joint index mapping.
        q_current: Base configuration to fill unspecified joints. Defaults to model neutral.

    Returns:
        Full Pinocchio configuration vector of size ``model.nq``.

    Raises:
        ValueError: If an input vector has the wrong size, or a named joint has more
            than one degree of freedom.
    """
    if q_current is not None:
        q = _to_numpy(q_current).flatten().copy()
        if q.size != model.nq:
            raise ValueError(f"Expected q_current of size {model.nq}, got {q.size}.")
    else:
        q = pin.neutral(model)

    positions = _to_numpy(joint_positions).flatten()
    if positions.size != len(joint_names):
        raise ValueError(
            f"Expected one position for each of {len(joint_names)} joint names, got {positions.size} positions."
        )

    for i, name in enumerate(joint_names):
        if not model.existJointName(name):
            continue
        joint_id = model.getJointId(name)
        joint = model.joints[joint_id]
        idx_q = joint.idx_q
        if joint.nq == 1 and joint.nv == 1:
            q[idx_q] = positions[i]
        elif _is_continuous_revolute_joint(joint):
            q[idx_q : idx_q + 2] = [np.cos(positions[i]), np.sin(positions[i])]
        else:
            _raise_unsupported_joint(name, joint)

    return q


def map_pinocchio_velocity_to_joint_state(
    velocity: np.ndarray,
    model: pin.Model,
    controlled_joint_names: list[str],
    robot_joint_space: list[str],
    dt: float,
    q_current: np.ndarray,
    current_joint_positions: np.ndarray,
) -> mg.JointState:
    """Convert a Pinocchio tangent velocity to an Isaac Sim JointState with integrated positions.

    Integrates the velocity over ``dt`` to produce target positions and packages both
    positions and velocities into a ``JointState`` for the motion generation API.

    Args:
        velocity: Tangent-space velocity vector of size ``model.nv``.
        model: Pinocchio model.
        controlled_joint_names: Names of joints controlled by the IK solver.
        robot_joint_space: Full ordered joint-space of the robot in Isaac Sim.
        dt: Integration timestep in seconds.
        q_current: Current configuration vector (pre-integration).
        current_joint_positions: Current Isaac Sim positions for ``controlled_joint_names``.
            These preserve the unwrapped angle of continuous joints.

    Returns:
        JointState containing integrated target positions and velocities for controlled joints.

    Raises:
        ValueError: If an input vector has the wrong size, or a named joint has more
            than one degree of freedom.
    """
    velocity = _to_numpy(velocity).flatten()
    q_current = _to_numpy(q_current).flatten()
    current_joint_positions = _to_numpy(current_joint_positions).flatten()

    if velocity.size != model.nv:
        raise ValueError(f"Expected velocity of size {model.nv}, got {velocity.size}.")
    if q_current.size != model.nq:
        raise ValueError(f"Expected q_current of size {model.nq}, got {q_current.size}.")
    if current_joint_positions.size != len(controlled_joint_names):
        raise ValueError(
            f"Expected one current position for each of {len(controlled_joint_names)} controlled joint names, "
            f"got {current_joint_positions.size} positions."
        )

    q_new = pin.integrate(model, q_current, velocity * dt)

    target_positions = np.zeros(len(controlled_joint_names))
    target_velocities = np.zeros(len(controlled_joint_names))

    for i, name in enumerate(controlled_joint_names):
        if not model.existJointName(name):
            continue
        joint_id = model.getJointId(name)
        joint = model.joints[joint_id]
        idx_q = joint.idx_q
        idx_v = joint.idx_v
        if joint.nq == 1 and joint.nv == 1:
            target_positions[i] = q_new[idx_q]
        elif _is_continuous_revolute_joint(joint):
            wrapped_position = np.arctan2(q_new[idx_q + 1], q_new[idx_q])
            unwrapped_reference = current_joint_positions[i] + velocity[idx_v] * dt
            target_positions[i] = wrapped_position + 2.0 * np.pi * np.round(
                (unwrapped_reference - wrapped_position) / (2.0 * np.pi)
            )
        else:
            _raise_unsupported_joint(name, joint)
        target_velocities[i] = velocity[idx_v]

    return mg.JointState.from_name(
        robot_joint_space=robot_joint_space,
        positions=(controlled_joint_names, wp.from_numpy(target_positions.astype(np.float32))),
        velocities=(controlled_joint_names, wp.from_numpy(target_velocities.astype(np.float32))),
        efforts=None,
    )


def _is_continuous_revolute_joint(joint: pin.JointModel) -> bool:
    """Check whether a Pinocchio joint uses the unit-circle configuration representation.

    Args:
        joint: Pinocchio joint model to inspect.

    Returns:
        True when the joint has one tangent degree of freedom represented by cosine and sine.
    """
    return joint.nq == 2 and joint.nv == 1


def _raise_unsupported_joint(name: str, joint: pin.JointModel) -> None:
    """Raise an error for a joint that cannot be represented by one Isaac Sim position.

    Args:
        name: Joint name.
        joint: Pinocchio joint model to report.

    Raises:
        ValueError: Always raised because the joint has an unsupported number of degrees of freedom.
    """
    raise ValueError(
        f"Joint '{name}' has unsupported Pinocchio dimensions nq={joint.nq}, nv={joint.nv}. "
        "Only one-degree-of-freedom joints are supported."
    )


def _to_numpy(input_array: list[float] | wp.array | np.ndarray) -> np.ndarray:
    """Convert input to a numpy array.

    Args:
        input_array: Array-like input to convert.

    Returns:
        Numpy array view or copy of the input.
    """
    if isinstance(input_array, wp.array):
        return input_array.numpy()
    return np.asarray(input_array, dtype=np.float64)
