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

"""Articulation <-> motion-generation state bridge.

:func:`read_robot_state` snapshots selected articulation fields into a motion-generation
``RobotState`` (all root and joint fields by default). :func:`apply_robot_state` applies desired
joint fields as sparse DOF-indexed writes after checking the articulation's joint-name space;
root, link, and site commands are rejected because this bridge does not apply them.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.prims import Articulation

__all__ = ["RobotStateReadSpec", "apply_robot_state", "read_robot_state"]


@dataclass(frozen=True)
class RobotStateReadSpec:
    """Robot-state fields needed by a consumer.

    The spec selects articulation getters, not the exact shape of the returned state. Getters
    that return multiple fields contribute all of them.
    """

    joint_positions: bool = False
    joint_velocities: bool = False
    root_position: bool = False
    root_orientation: bool = False
    root_linear_velocity: bool = False
    root_angular_velocity: bool = False

    @property
    def has_root(self) -> bool:
        """Has root."""
        return self.root_position or self.root_orientation or self.root_linear_velocity or self.root_angular_velocity

    @property
    def has_joints(self) -> bool:
        """Has joints."""
        return self.joint_positions or self.joint_velocities

    @property
    def has_any(self) -> bool:
        """Has any."""
        return self.has_root or self.has_joints


_FULL_STATE_READ = RobotStateReadSpec(
    joint_positions=True,
    joint_velocities=True,
    root_position=True,
    root_orientation=True,
    root_linear_velocity=True,
    root_angular_velocity=True,
)


# -- read side -------------------------------------------------------------------------------


def read_robot_state(
    articulation: Articulation,
    read_spec: RobotStateReadSpec | None = None,
) -> mg.RobotState:
    """Read the articulation state needed by a consumer.

    Omitting ``read_spec`` preserves the full-state behavior. Each selected getter contributes
    everything it returns, so requesting either root pose field returns both position and
    orientation, and requesting either root velocity returns both linear and angular velocity.

    Args:
        articulation: Deployed robot articulation.
        read_spec: Which state fields to read; None reads the full state.

    Returns:
        The resulting mg.RobotState.
    """
    if read_spec is None:
        read_spec = _FULL_STATE_READ
    if not read_spec.has_any:
        return mg.RobotState(joints=None, root=None)

    joints = None
    if read_spec.has_joints:
        dof_names = list(articulation.dof_names)
        positions = (dof_names, articulation.get_dof_positions()) if read_spec.joint_positions else None
        velocities = (dof_names, articulation.get_dof_velocities()) if read_spec.joint_velocities else None
        joints = mg.JointState.from_name(dof_names, positions=positions, velocities=velocities)

    root_fields = {}
    if read_spec.root_position or read_spec.root_orientation:
        root_position, root_orientation = articulation.get_world_poses()
        root_fields.update(
            position=root_position.reshape([-1]),
            orientation=root_orientation.reshape([-1]),
        )

    if read_spec.root_linear_velocity or read_spec.root_angular_velocity:
        linear_velocity, angular_velocity = articulation.get_velocities()
        root_fields.update(
            linear_velocity=linear_velocity.reshape([-1]),
            angular_velocity=angular_velocity.reshape([-1]),
        )

    root = mg.RootState(**root_fields) if root_fields else None
    return mg.RobotState(joints=joints, root=root)


# -- write side ------------------------------------------------------------------------------


def apply_robot_state(articulation: Articulation, desired: mg.RobotState) -> None:
    """Apply desired joint fields as sparse, name-resolved DOF-indexed writes.

    Unrelated DOFs are untouched, and undeclared root/link/site output is rejected (no teleport).

    Args:
        articulation: Deployed robot articulation.
        desired: Partial named desired state to apply.
    """
    if desired.root is not None or desired.links is not None or desired.sites is not None:
        raise ValueError(
            "apply_robot_state only applies joint fields; desired root/link/site state is never "
            "written to the articulation. Remove non-joint components from the desired state."
        )

    joints = desired.joints
    if joints is None:
        raise ValueError("apply_robot_state requires desired joint state.")

    dof_names = list(articulation.dof_names)
    if list(joints.robot_joint_space) != dof_names:
        raise ValueError(
            f"apply_robot_state: desired joint space {list(joints.robot_joint_space)} does not match "
            f"the articulation DOF names {dof_names} at {articulation.paths[0]!r}."
        )

    if joints.positions is not None:
        articulation.set_dof_position_targets(joints.positions, dof_indices=joints.position_indices)
    if joints.velocities is not None:
        articulation.set_dof_velocity_targets(joints.velocities, dof_indices=joints.velocity_indices)
    if joints.efforts is not None:
        articulation.set_dof_efforts(joints.efforts, dof_indices=joints.effort_indices)
