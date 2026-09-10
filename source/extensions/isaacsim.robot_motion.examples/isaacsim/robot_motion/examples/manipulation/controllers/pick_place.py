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

"""Robot-independent pick-and-place state machine."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.utils import transform as transform_utils

from .gripper import JointGripperController, SurfaceGripperController

_Pose = tuple[np.ndarray, np.ndarray]


class PickPlacePhase(Enum):
    """Phase of the pick-and-place state machine."""

    APPROACH_PICK = auto()
    #: Move above the pick pose.
    DESCEND_PICK = auto()
    #: Move to the pick pose.
    GRASP = auto()
    #: Close the gripper at the pick pose.
    LIFT = auto()
    #: Lift the grasped object.
    APPROACH_PLACE = auto()
    #: Move above the place pose.
    DESCEND_PLACE = auto()
    #: Move to the place pose.
    RELEASE = auto()
    #: Open the gripper at the place pose.
    RETREAT = auto()
    #: Move away from the released object.
    DONE = auto()
    #: Complete the pick-and-place sequence.
    FAILED = auto()
    #: Stop the sequence after an error.


_ACTIVE_PHASES = (
    PickPlacePhase.APPROACH_PICK,
    PickPlacePhase.DESCEND_PICK,
    PickPlacePhase.GRASP,
    PickPlacePhase.LIFT,
    PickPlacePhase.APPROACH_PLACE,
    PickPlacePhase.DESCEND_PLACE,
    PickPlacePhase.RELEASE,
    PickPlacePhase.RETREAT,
)

_DEFAULT_TIMEOUTS = {
    PickPlacePhase.APPROACH_PICK: 5.0,
    PickPlacePhase.DESCEND_PICK: 5.0,
    PickPlacePhase.GRASP: 2.0,
    PickPlacePhase.LIFT: 5.0,
    PickPlacePhase.APPROACH_PLACE: 7.0,
    PickPlacePhase.DESCEND_PLACE: 5.0,
    PickPlacePhase.RELEASE: 2.0,
    PickPlacePhase.RETREAT: 5.0,
}


class PickPlaceController(mg.BaseController):
    """State machine that coordinates an arm and gripper for pick-and-place.

    Call :meth:`reset` with named ``pick`` and ``place`` goal sites. The controller
    snapshots those goals, advances after the measured tool pose remains within tolerance,
    and waits for observable gripper completion during grasp and release.

    Args:
        arm_controller: Controller that generates arm commands for a tool-pose target.
        gripper_open_controller: Controller that opens the configured gripper.
        gripper_close_controller: Controller that closes the configured gripper.
        robot_site_space: Ordered names of robot spatial sites.
        tool_frame: Site name controlled by the arm controller.
        controller_to_grasp_position: Translation from the controller frame to the grasp frame.
        controller_to_grasp_orientation: Rotation from the controller frame to the grasp frame.
        grasp_orientation: Grasp orientation used when a goal omits orientation.
        phase_timeouts: Overrides for active-phase timeout durations.
        approach_height: Vertical offset used before descending to a goal.
        lift_height: Vertical offset used while carrying or retreating.
        position_tolerance: Maximum tool-position error for convergence.
        grasp_position_tolerance: Optional position tolerance used only while grasping.
        orientation_tolerance: Maximum tool-orientation error for convergence.
        stability_duration: Time the measured tool pose must remain converged.
        gripper_dwell: Minimum time spent in grasp and release phases.

    Raises:
        ValueError: If the tool frame, tolerances, transforms, or timeouts are invalid.

    Example:

    .. code-block:: python

        >>> controller = PickPlaceController(
        ...     arm_controller=arm_controller,
        ...     gripper_open_controller=open_gripper,
        ...     gripper_close_controller=close_gripper,
        ...     robot_site_space=["tool0"],
        ...     tool_frame="tool0",
        ... )  # doctest: +SKIP
    """

    PICK_SITE = "pick"
    #: Goal site name used for the pick pose.
    PLACE_SITE = "place"
    #: Goal site name used for the place pose.

    def __init__(
        self,
        *,
        arm_controller: mg.BaseController,
        gripper_open_controller: JointGripperController | SurfaceGripperController,
        gripper_close_controller: JointGripperController | SurfaceGripperController,
        robot_site_space: list[str],
        tool_frame: str,
        controller_to_grasp_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        controller_to_grasp_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        grasp_orientation: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
        phase_timeouts: Mapping[PickPlacePhase, float] | None = None,
        approach_height: float = 0.20,
        lift_height: float = 0.30,
        position_tolerance: float = 0.015,
        grasp_position_tolerance: float | None = None,
        orientation_tolerance: float = 0.12,
        stability_duration: float = 0.10,
        gripper_dwell: float = 0.20,
    ) -> None:
        if tool_frame not in robot_site_space:
            raise ValueError("tool_frame must be present in robot_site_space.")
        scalar_settings = (
            approach_height,
            lift_height,
            stability_duration,
            gripper_dwell,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in scalar_settings):
            raise ValueError("Controller heights and durations must be finite and non-negative.")
        tolerances = (position_tolerance, orientation_tolerance)
        if grasp_position_tolerance is not None:
            tolerances += (grasp_position_tolerance,)
        if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Controller tolerances must be finite and positive.")

        self._robot_site_space = list(robot_site_space)
        self._tool_frame = tool_frame
        self._controller_to_grasp_position = _validate_position(controller_to_grasp_position)
        self._controller_to_grasp_orientation = _normalize_quaternion(controller_to_grasp_orientation)
        self._default_orientation = _normalize_quaternion(grasp_orientation)
        self._approach_height = float(approach_height)
        self._lift_height = float(lift_height)
        self._position_tolerance = float(position_tolerance)
        self._grasp_position_tolerance = (
            self._position_tolerance if grasp_position_tolerance is None else float(grasp_position_tolerance)
        )
        self._orientation_tolerance = float(orientation_tolerance)
        self._stability_duration = float(stability_duration)
        self._gripper_dwell = float(gripper_dwell)

        self._timeouts = dict(_DEFAULT_TIMEOUTS)
        if phase_timeouts is not None:
            if any(phase not in _ACTIVE_PHASES for phase in phase_timeouts):
                raise ValueError("phase_timeouts may only override active phases.")
            self._timeouts.update(phase_timeouts)
        if any(not np.isfinite(self._timeouts[phase]) or self._timeouts[phase] <= 0.0 for phase in _ACTIVE_PHASES):
            raise ValueError("Every active phase timeout must be finite and positive.")

        open_phases = {
            PickPlacePhase.APPROACH_PICK,
            PickPlacePhase.DESCEND_PICK,
            PickPlacePhase.RELEASE,
            PickPlacePhase.RETREAT,
        }
        self._grippers = {
            phase: gripper_open_controller if phase in open_phases else gripper_close_controller
            for phase in _ACTIVE_PHASES
        }
        self._selector = mg.SelectableController(
            {phase: mg.CombinedController([arm_controller, self._grippers[phase]]) for phase in _ACTIVE_PHASES},
            PickPlacePhase.APPROACH_PICK,
        )

        self._phase = PickPlacePhase.APPROACH_PICK
        self._phase_started = 0.0
        self._stable_since: float | None = None
        self._last_time = 0.0
        self._pick: _Pose | None = None
        self._place: _Pose | None = None
        self._failed_reason: str | None = None

    @property
    def phase(self) -> PickPlacePhase:
        """Get the current state-machine phase.

        Returns:
            Current pick-and-place phase.

        Example:

        .. code-block:: python

            >>> controller.phase  # doctest: +SKIP
            <PickPlacePhase.APPROACH_PICK: 1>
        """
        return self._phase

    @property
    def is_done(self) -> bool:
        """Check whether the sequence completed successfully.

        Returns:
            True if the controller reached the terminal done phase.

        Example:

        .. code-block:: python

            >>> controller.is_done  # doctest: +SKIP
            False
        """
        return self._phase is PickPlacePhase.DONE

    @property
    def failed(self) -> bool:
        """Check whether the sequence stopped after an error.

        Returns:
            True if the controller reached the terminal failed phase.

        Example:

        .. code-block:: python

            >>> controller.failed  # doctest: +SKIP
            False
        """
        return self._phase is PickPlacePhase.FAILED

    @property
    def failure_reason(self) -> str | None:
        """Get the reason for the most recent controller failure.

        Returns:
            Failure description, or None if the controller has not failed.

        Example:

        .. code-block:: python

            >>> controller.failure_reason  # doctest: +SKIP
        """
        return self._failed_reason

    @property
    def phase_timeouts(self) -> Mapping[PickPlacePhase, float]:
        """Get a copy of the configured active-phase timeouts.

        Returns:
            Mapping from each active phase to its timeout duration in seconds.

        Example:

        .. code-block:: python

            >>> controller.phase_timeouts[PickPlacePhase.GRASP]  # doctest: +SKIP
            2.0
        """
        return self._timeouts.copy()

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        """Reset the sequence and snapshot the pick and place goals.

        Args:
            estimated_state: Current robot state containing the measured tool pose.
            setpoint_state: State containing named ``pick`` and ``place`` goal sites.
            t: Current controller time.
            **kwargs: Additional inputs forwarded to child controllers.

        Returns:
            True if the goals and child controllers were reset, False on failure.

        Example:

        .. code-block:: python

            >>> controller.reset(estimated_state, goals, 0.0)  # doctest: +SKIP
            True
        """
        try:
            if not np.isfinite(t):
                raise ValueError("Controller time must be finite.")
            self._controller_pose(estimated_state)
            self._pick = self._read_goal(setpoint_state, self.PICK_SITE)
            self._place = self._read_goal(setpoint_state, self.PLACE_SITE)
            self._phase = PickPlacePhase.APPROACH_PICK
            self._phase_started = float(t)
            self._stable_since = None
            self._last_time = float(t)
            self._failed_reason = None
            if not self._selector.reset(estimated_state, self._make_arm_setpoint(), t, **kwargs):
                raise RuntimeError("Initial controller reset failed.")
        except (RuntimeError, ValueError) as exc:
            self._fail(str(exc))
            return False
        return True

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState | None:
        """Advance the state machine and generate the current robot command.

        Args:
            estimated_state: Current robot state containing measured joints and tool pose.
            setpoint_state: Ignored after goals are captured by :meth:`reset`.
            t: Current controller time.
            **kwargs: Additional inputs forwarded to child controllers.

        Returns:
            Desired robot state, or None if the controller fails.

        Example:

        .. code-block:: python

            >>> controller.forward(estimated_state, None, 0.01)  # doctest: +SKIP
        """
        if self.is_done:
            return mg.RobotState()
        if self.failed:
            return None
        try:
            self._validate_time(t)
            measured = self._controller_pose(estimated_state)
            target = self._target()
            desired = self._selector.forward(estimated_state, self._make_arm_setpoint(target), t, **kwargs)
            if desired is None:
                raise RuntimeError(f"Controller composition failed in phase {self._phase.name}.")
            if self._phase_complete(estimated_state, measured, target, t):
                self._advance(t)
            elif t - self._phase_started >= self._timeouts[self._phase]:
                position_error, orientation_error = _pose_errors(measured, target)
                raise RuntimeError(
                    f"{self._phase.name} timed out after {self._timeouts[self._phase]:.3f} s "
                    f"(position error {position_error:.4f} m, orientation error {orientation_error:.4f} rad)."
                )
            self._last_time = float(t)
            return desired
        except (RuntimeError, ValueError) as exc:
            self._fail(str(exc))
            return None

    def _advance(self, t: float) -> None:
        next_index = _ACTIVE_PHASES.index(self._phase) + 1
        if next_index == len(_ACTIVE_PHASES):
            self._phase = PickPlacePhase.DONE
        else:
            self._phase = _ACTIVE_PHASES[next_index]
            self._selector.set_next_controller(self._phase)
        self._phase_started = float(t)
        self._stable_since = None

    def _phase_complete(
        self,
        estimated_state: mg.RobotState,
        measured: _Pose,
        target: _Pose,
        t: float,
    ) -> bool:
        position_error, orientation_error = _pose_errors(measured, target)
        position_tolerance = (
            self._grasp_position_tolerance if self._phase is PickPlacePhase.GRASP else self._position_tolerance
        )
        converged = position_error <= position_tolerance and orientation_error <= self._orientation_tolerance
        if not converged:
            self._stable_since = None
            return False

        if self._stable_since is None:
            self._stable_since = float(t)
        if t - self._stable_since < self._stability_duration:
            return False

        if self._phase in {PickPlacePhase.GRASP, PickPlacePhase.RELEASE}:
            return t - self._phase_started >= self._gripper_dwell and self._grippers[self._phase].is_complete(
                estimated_state
            )
        return True

    def _target(self) -> _Pose:
        if self._pick is None or self._place is None:
            raise RuntimeError("PickPlaceController has not been reset.")
        grasp = (
            self._place
            if self._phase
            in {
                PickPlacePhase.APPROACH_PLACE,
                PickPlacePhase.DESCEND_PLACE,
                PickPlacePhase.RELEASE,
                PickPlacePhase.RETREAT,
            }
            else self._pick
        )

        position = grasp[0].copy()
        if self._phase in {PickPlacePhase.APPROACH_PICK, PickPlacePhase.APPROACH_PLACE}:
            position[2] += self._approach_height
        elif self._phase in {PickPlacePhase.LIFT, PickPlacePhase.RETREAT}:
            position[2] += self._lift_height

        grasp_to_controller_orientation = transform_utils.quaternion_conjugate(
            self._controller_to_grasp_orientation, dtype=wp.float64, device="cpu"
        ).numpy()
        grasp_to_controller_position = transform_utils.rotate_vectors_by_quaternion(
            -self._controller_to_grasp_position,
            grasp_to_controller_orientation,
            dtype=wp.float64,
            device="cpu",
        ).numpy()
        controller_position = transform_utils.transform_local_to_world(
            grasp_to_controller_position,
            position,
            grasp[1],
            dtype=wp.float64,
            device="cpu",
        ).numpy()
        controller_orientation = transform_utils.quaternion_multiplication(
            grasp[1],
            grasp_to_controller_orientation,
            dtype=wp.float64,
            device="cpu",
        ).numpy()
        return controller_position, _normalize_quaternion(controller_orientation)

    def _make_arm_setpoint(self, target: _Pose | None = None) -> mg.RobotState:
        target = self._target() if target is None else target
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._robot_site_space,
                positions=(
                    [self._tool_frame],
                    wp.array([target[0]], dtype=wp.float32, device="cpu"),
                ),
                orientations=(
                    [self._tool_frame],
                    wp.array([target[1]], dtype=wp.float32, device="cpu"),
                ),
            )
        )

    def _read_goal(self, setpoint: mg.RobotState | None, name: str) -> _Pose:
        if setpoint is None or setpoint.sites is None or setpoint.sites.positions is None:
            raise ValueError("PickPlaceController requires named 'pick' and 'place' position sites.")
        sites = setpoint.sites
        if name not in sites.position_names:
            raise ValueError("PickPlaceController requires named 'pick' and 'place' position sites.")
        position = np.asarray(sites.positions.numpy()[sites.position_names.index(name)], dtype=np.float64).copy()
        orientation = self._default_orientation
        if sites.orientations is not None and name in sites.orientation_names:
            orientation = _normalize_quaternion(sites.orientations.numpy()[sites.orientation_names.index(name)])
        return _validate_position(position, label=f"Goal site {name!r}"), orientation

    def _controller_pose(self, state: mg.RobotState) -> _Pose:
        if state.sites is None or state.sites.positions is None or state.sites.orientations is None:
            raise ValueError(f"Estimated RobotState requires measured site {self._tool_frame!r}.")
        sites = state.sites
        if self._tool_frame not in sites.position_names or self._tool_frame not in sites.orientation_names:
            raise ValueError(f"Estimated RobotState requires measured site {self._tool_frame!r}.")
        position = _validate_position(sites.positions.numpy()[sites.position_names.index(self._tool_frame)])
        orientation = _normalize_quaternion(sites.orientations.numpy()[sites.orientation_names.index(self._tool_frame)])
        return position, orientation

    def _validate_time(self, t: float) -> None:
        if not np.isfinite(t):
            raise ValueError("Controller time must be finite.")
        if t < self._last_time:
            raise ValueError("Controller time moved backwards.")

    def _fail(self, reason: str) -> None:
        self._phase = PickPlacePhase.FAILED
        self._failed_reason = reason


def _normalize_quaternion(orientation: tuple[float, ...] | np.ndarray) -> np.ndarray:
    quaternion = np.asarray(orientation, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must be finite and nonzero.")
    return quaternion / norm


def _validate_position(position: tuple[float, ...] | np.ndarray, label: str = "Position") -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"{label} must contain three finite XYZ values.")
    return position.copy()


def _quaternion_distance(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    dot = abs(float(np.dot(_normalize_quaternion(lhs), _normalize_quaternion(rhs))))
    return 2.0 * float(np.arccos(np.clip(dot, 0.0, 1.0)))


def _pose_errors(measured: _Pose, target: _Pose) -> tuple[float, float]:
    position_error = float(np.linalg.norm(measured[0] - target[0]))
    orientation_error = _quaternion_distance(measured[1], target[1])
    return position_error, orientation_error


__all__ = ["PickPlaceController", "PickPlacePhase"]
