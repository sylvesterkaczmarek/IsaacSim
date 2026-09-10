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

"""Small gripper leaves for composition with motion-generation controllers."""

from __future__ import annotations

from enum import Enum
from typing import Any

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp


class GripperCommand(Enum):
    """Command for opening or closing a gripper."""

    OPEN = "open"
    #: Open the gripper.
    CLOSE = "close"
    #: Close the gripper.


class JointGripperController(mg.BaseController):
    """Controller for a gripper driven by articulation joints.

    The controller reports completion when the joints reach the commanded positions. A
    closing command also completes when the fingers have moved toward the target and stop,
    allowing an object to prevent full closure.

    Args:
        robot_joint_space: Ordered names of all robot joints.
        joint_names: Names of the joints that actuate the gripper.
        open_positions: Joint positions for an open gripper.
        closed_positions: Joint positions for a closed gripper.
        command: Gripper command emitted by the controller.
        position_tolerance: Maximum joint-position error for exact completion.
        velocity_tolerance: Maximum joint speed for stopped-short completion.
        minimum_close_fraction: Minimum commanded travel required for stopped-short completion.

    Raises:
        ValueError: If the joint configuration or completion thresholds are invalid.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import GripperCommand, JointGripperController

        >>> controller = JointGripperController(
        ...     robot_joint_space=["finger_joint"],
        ...     joint_names=["finger_joint"],
        ...     open_positions=[0.04],
        ...     closed_positions=[0.0],
        ...     command=GripperCommand.CLOSE,
        ... )
    """

    def __init__(
        self,
        *,
        robot_joint_space: list[str],
        joint_names: list[str] | tuple[str, ...],
        open_positions: list[float] | tuple[float, ...],
        closed_positions: list[float] | tuple[float, ...],
        command: GripperCommand,
        position_tolerance: float = 0.002,
        velocity_tolerance: float = 0.01,
        minimum_close_fraction: float = 0.1,
    ) -> None:
        self._robot_joint_space = list(robot_joint_space)
        self._joint_names = list(joint_names)
        if not self._joint_names or not set(self._joint_names).issubset(self._robot_joint_space):
            raise ValueError("Gripper joints must be a non-empty subset of robot_joint_space.")
        if len(open_positions) != len(self._joint_names) or len(closed_positions) != len(self._joint_names):
            raise ValueError("Gripper command positions must match the number of gripper joints.")
        if position_tolerance <= 0.0 or velocity_tolerance <= 0.0:
            raise ValueError("Gripper tolerances must be positive.")
        if not 0.0 < minimum_close_fraction <= 1.0:
            raise ValueError("minimum_close_fraction must be in (0, 1].")
        self._command = command
        self._target = np.asarray(
            open_positions if command is GripperCommand.OPEN else closed_positions, dtype=np.float64
        )
        self._position_tolerance = float(position_tolerance)
        self._velocity_tolerance = float(velocity_tolerance)
        self._minimum_close_fraction = float(minimum_close_fraction)
        self._start_positions: np.ndarray | None = None
        self._output = mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._robot_joint_space,
                positions=(self._joint_names, wp.array(self._target, dtype=wp.float32, device="cpu")),
            )
        )

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        """Capture the initial gripper positions for completion tracking.

        Args:
            estimated_state: Current measured robot state.
            setpoint_state: Unused controller setpoint.
            t: Current controller time.
            **kwargs: Additional controller inputs.

        Returns:
            True if the required gripper positions are available, False otherwise.

        Example:

        .. code-block:: python

            >>> controller.reset(estimated_state, None, 0.0)  # doctest: +SKIP
            True
        """
        positions, _ = self._read_state(estimated_state)
        self._start_positions = positions
        return positions is not None

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState:
        """Return the sparse joint-position command for the gripper.

        Args:
            estimated_state: Current measured robot state.
            setpoint_state: Unused controller setpoint.
            t: Current controller time.
            **kwargs: Additional controller inputs.

        Returns:
            Robot state containing the commanded gripper joint positions.

        Example:

        .. code-block:: python

            >>> controller.forward(estimated_state, None, 0.0)  # doctest: +SKIP
        """
        return self._output

    def is_complete(self, estimated_state: mg.RobotState) -> bool:
        """Check whether the commanded gripper motion is complete.

        Args:
            estimated_state: Current measured robot state.

        Returns:
            True if the gripper reached its target or completed a stopped-short grasp.

        Example:

        .. code-block:: python

            >>> controller.is_complete(estimated_state)  # doctest: +SKIP
            False
        """
        positions, velocities = self._read_state(estimated_state)
        if positions is None:
            return False
        if np.max(np.abs(positions - self._target)) <= self._position_tolerance:
            return True
        if self._command is GripperCommand.OPEN or self._start_positions is None or velocities is None:
            return False

        commanded_displacement = self._target - self._start_positions
        measured_displacement = positions - self._start_positions

        # Exclude joints with negligible commanded travel from fractional progress calculations.
        commanded_joint_mask = np.abs(commanded_displacement) > self._position_tolerance
        if not commanded_joint_mask.any() or np.max(np.abs(velocities)) > self._velocity_tolerance:
            return False

        progress = np.ones_like(commanded_displacement)
        progress[commanded_joint_mask] = (
            measured_displacement[commanded_joint_mask] / commanded_displacement[commanded_joint_mask]
        )

        # A stopped-short grasp must move every commanded joint toward, but not beyond, its target.
        direction_valid = np.all(
            (progress[commanded_joint_mask] >= 0.0) & (progress[commanded_joint_mask] <= 1.0 + 1.0e-3)
        )
        minimum_progress_reached = np.min(progress[commanded_joint_mask]) >= self._minimum_close_fraction

        return bool(direction_valid and minimum_progress_reached)

    def _read_state(self, state: mg.RobotState) -> tuple[np.ndarray | None, np.ndarray | None]:
        joints = state.joints
        if joints is None or joints.positions is None:
            return None, None
        if not set(self._joint_names).issubset(joints.position_names):
            return None, None

        positions = joints.positions.numpy()
        selected_positions = np.asarray(
            [positions[joints.position_names.index(name)] for name in self._joint_names], dtype=np.float64
        )
        if not np.isfinite(selected_positions).all():
            return None, None

        if joints.velocities is None or not set(self._joint_names).issubset(joints.velocity_names):
            return selected_positions, None

        velocities = joints.velocities.numpy()
        selected_velocities = np.asarray(
            [velocities[joints.velocity_names.index(name)] for name in self._joint_names], dtype=np.float64
        )
        if not np.isfinite(selected_velocities).all():
            return selected_positions, None

        return selected_positions, selected_velocities


class SurfaceGripperController(mg.BaseController):
    """Controller for a USD surface gripper.

    A surface gripper is not controlled through articulation joints, so this leaf applies
    the gripper command directly and returns an empty ``RobotState``.

    Args:
        gripper_path: Absolute USD path to the surface gripper prim.
        command: Gripper command applied by the controller.
        interface: Surface-gripper interface override.
        status_type: Surface-gripper status enumeration override.

    Raises:
        ValueError: If ``gripper_path`` is not absolute.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_motion.examples.manipulation import GripperCommand, SurfaceGripperController

        >>> controller = SurfaceGripperController(
        ...     gripper_path="/World/Robot/SurfaceGripper",
        ...     command=GripperCommand.CLOSE,
        ... )
    """

    def __init__(
        self,
        *,
        gripper_path: str,
        command: GripperCommand,
        interface: Any | None = None,
        status_type: Any | None = None,
    ) -> None:
        if not gripper_path.startswith("/"):
            raise ValueError("gripper_path must be an absolute USD prim path.")
        if interface is None:
            from isaacsim.robot.surface_gripper import _surface_gripper

            interface = _surface_gripper.acquire_surface_gripper_interface()
        if status_type is None:
            from isaacsim.robot.surface_gripper.bindings._surface_gripper import GripperStatus

            status_type = GripperStatus
        self._interface = interface
        self._status_type = status_type
        self._gripper_path = gripper_path
        self._command = command

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        """Reset the stateless surface-gripper command.

        Args:
            estimated_state: Current measured robot state.
            setpoint_state: Unused controller setpoint.
            t: Current controller time.
            **kwargs: Additional controller inputs.

        Returns:
            Always True.

        Example:

        .. code-block:: python

            >>> controller.reset(estimated_state, None, 0.0)  # doctest: +SKIP
            True
        """
        return True

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState:
        """Apply the surface-gripper command when its status differs.

        Args:
            estimated_state: Current measured robot state.
            setpoint_state: Unused controller setpoint.
            t: Current controller time.
            **kwargs: Additional controller inputs.

        Returns:
            Empty robot state because the command does not target articulation joints.

        Example:

        .. code-block:: python

            >>> controller.forward(estimated_state, None, 0.0)  # doctest: +SKIP
        """
        status = self._interface.get_gripper_status(self._gripper_path)
        if self._command is GripperCommand.OPEN:
            if status != self._status_type.Open:
                self._interface.open_gripper(self._gripper_path)
        elif status not in {self._status_type.Closed, self._status_type.Closing}:
            self._interface.close_gripper(self._gripper_path)
        return mg.RobotState()

    def is_complete(self, estimated_state: mg.RobotState) -> bool:
        """Check whether the surface gripper reached the commanded status.

        Args:
            estimated_state: Unused current robot state.

        Returns:
            True if the surface-gripper status matches the command.

        Example:

        .. code-block:: python

            >>> controller.is_complete(estimated_state)  # doctest: +SKIP
            False
        """
        status = self._interface.get_gripper_status(self._gripper_path)
        expected = self._status_type.Open if self._command is GripperCommand.OPEN else self._status_type.Closed
        return status == expected


__all__ = ["GripperCommand", "JointGripperController", "SurfaceGripperController"]
