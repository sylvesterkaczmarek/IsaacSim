# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive an IMU reading from the state of the rigid body the sensor is mounted on.

Callers push one `BodyState` per step with `sample()` and pull a filtered `ImuReading` with
`read()`, so the state can come from a physics engine, a recording, or a closed-form trajectory.

Vectors are `wp.vec3d` and quaternions are `wp.quatd`, both ``float64``. Quaternions are therefore
ordered ``xyzw``, matching Warp and the physics engine tensor buffers the body state comes from, so
per-step data needs no reordering and the rotation math is Warp's rather than hand-written.

USD and the Isaac Sim public APIs order quaternions ``wxyz``, so a value that comes from a stage --
a mount offset read back from an ``Xform``, for instance -- is converted once with
`read_quaternion_from_wxyz()`.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import warp as wp

#: Identity rotation, ordered ``xyzw``.
IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
ZERO_VECTOR = (0.0, 0.0, 0.0)


def _as_vector(values: Sequence[float]) -> wp.vec3d:
    """Convert a sequence of three components to a vector.

    Args:
        values: Components, ordered ``xyz``.

    Returns:
        The vector.

    Raises:
        ValueError: If there are not exactly three components.

    """
    if isinstance(values, wp.vec3d):
        return values
    components = [float(value) for value in values]
    if len(components) != 3:
        raise ValueError(f"A vector takes 3 components, got {len(components)}.")
    return wp.vec3d(components)


def _as_quaternion(values: Sequence[float]) -> wp.quatd:
    """Convert a sequence of four components to a quaternion.

    Args:
        values: Components, ordered ``xyzw``.

    Returns:
        The quaternion.

    Raises:
        ValueError: If there are not exactly four components.

    """
    if isinstance(values, wp.quatd):
        return values
    components = [float(value) for value in values]
    if len(components) != 4:
        raise ValueError(f"A quaternion takes 4 components, got {len(components)}.")
    return wp.quatd(components)


def _scale(values: wp.vec3d, factor: float) -> wp.vec3d:
    """Scale a vector.

    Warp multiplies its vector types by a matching scalar type rather than a Python ``float``.

    Args:
        values: Value to scale.
        factor: Scalar factor.

    Returns:
        The scaled value.

    """
    return values * wp.float64(factor)


def read_quaternion_from_wxyz(values: Sequence[float]) -> wp.quatd:
    """Reorder a quaternion from USD's ``wxyz`` to ``xyzw``.

    Args:
        values: Components, ordered ``wxyz``.

    Returns:
        The same rotation, ordered ``xyzw``.

    """
    components = [float(value) for value in values]
    if len(components) != 4:
        raise ValueError(f"A quaternion takes 4 components, got {len(components)}.")
    return wp.quatd(components[1], components[2], components[3], components[0])


def average_quaternions(window: Sequence[Sequence[float]]) -> wp.quatd:
    """Average a window of quaternions.

    Warp has no sign-aligned mean, so this is the one piece of quaternion arithmetic the module
    still spells out.

    Args:
        window: Quaternions, ordered ``xyzw``.

    Returns:
        The mean rotation as a unit quaternion, or the identity for an empty window.

    """
    if len(window) == 0:
        return wp.quatd(IDENTITY_QUATERNION)
    reference = _as_quaternion(window[0])
    total = wp.quatd(0.0, 0.0, 0.0, 0.0)
    for entry in window:
        quaternion = _as_quaternion(entry)
        # A quaternion and its negation are one rotation, so align signs against the first element
        # or an alternating window averages to zero.
        total = total + (quaternion if float(wp.dot(quaternion, reference)) >= 0.0 else -quaternion)
    # Warp normalizes a zero quaternion to the identity, so a window that cancels is still safe.
    return wp.normalize(total)


@dataclass
class BodyState:
    """World-frame state of the rigid body a sensor is mounted on."""

    #: Position of the body origin.
    position: wp.vec3d = field(default_factory=lambda: wp.vec3d(ZERO_VECTOR))
    #: Orientation of the body, ordered ``xyzw``.
    orientation: wp.quatd = field(default_factory=lambda: wp.quatd(IDENTITY_QUATERNION))
    #: Linear velocity at the body origin.
    linear_velocity: wp.vec3d = field(default_factory=lambda: wp.vec3d(ZERO_VECTOR))
    #: Angular velocity of the body.
    angular_velocity: wp.vec3d = field(default_factory=lambda: wp.vec3d(ZERO_VECTOR))

    def __post_init__(self) -> None:
        """Validate the component counts and convert every field."""
        self.position = _as_vector(self.position)
        self.orientation = _as_quaternion(self.orientation)
        self.linear_velocity = _as_vector(self.linear_velocity)
        self.angular_velocity = _as_vector(self.angular_velocity)


@dataclass
class ImuReading:
    """One filtered reading, in the sensor frame unless stated otherwise.

    The fields are not all from the same instant, matching the shipped Isaac Sim IMU sensor.
    `orientation` and `angular_velocity` are means over the filter window, so they lag the newest
    sample by ``(filter_width - 1) / 2`` steps.
    """

    #: Simulation time of the newest sample, in seconds.
    time: float
    #: World position of the sensor origin.
    position: wp.vec3d
    #: World orientation of the sensor, ordered ``xyzw``.
    orientation: wp.quatd
    #: Linear velocity at the sensor origin.
    linear_velocity: wp.vec3d
    #: Angular velocity of the sensor.
    angular_velocity: wp.vec3d
    #: Specific force, ``a - g``.
    linear_acceleration: wp.vec3d


@dataclass
class _Sample:
    """One derived sample, buffered for the rolling average."""

    time: float
    position: wp.vec3d
    orientation: wp.quatd
    # World frame on purpose: see `ImuSensor.read()`.
    linear_velocity_world: wp.vec3d
    angular_velocity: wp.vec3d


class ImuSensor:
    """An IMU rigidly mounted on a body, reporting filtered readings in its own frame.

    ``2 * filter_width`` samples are buffered, so a reading can average a window of width
    ``filter_width`` and difference it against the window before it.

    Args:
        gravity: World-frame gravitational acceleration, subtracted from every reading to give
            specific force. Required, because its magnitude depends on the stage's linear unit and
            its direction on the stage's up axis. For a Z-up stage in meters it is
            ``(0.0, 0.0, -9.80665)``; pass a zero vector for coordinate acceleration.
        mount_translation: Translation of the sensor in the body frame.
        mount_orientation: Orientation of the sensor in the body frame, ordered ``xyzw``. A mount
            read back from USD is ``wxyz``, so convert it with `read_quaternion_from_wxyz()` first.
            Normalized on construction.
        filter_width: Rolling average window applied to every filtered quantity.

    Raises:
        ValueError: If `filter_width` is less than one, or a component count is wrong.

    """

    def __init__(
        self,
        gravity: Sequence[float],
        *,
        mount_translation: Sequence[float] = ZERO_VECTOR,
        mount_orientation: Sequence[float] = IDENTITY_QUATERNION,
        filter_width: int = 1,
    ) -> None:
        if filter_width < 1:
            raise ValueError(f"`filter_width` must be at least 1, got {filter_width}.")
        self._gravity = _as_vector(gravity)
        self._mount_translation = _as_vector(mount_translation)
        self._mount_orientation = wp.normalize(_as_quaternion(mount_orientation))
        self._filter_width = filter_width
        self._samples: deque[_Sample] = deque(maxlen=2 * filter_width)

    @property
    def sample_count(self) -> int:
        """Number of samples currently buffered."""
        return len(self._samples)

    def reset(self) -> None:
        """Discard every buffered sample.

        Call this when the body state becomes discontinuous, such as after a teleport or a
        simulation restart, so stale samples cannot contaminate the finite difference.
        """
        self._samples.clear()

    def sample(self, time: float, state: BodyState) -> None:
        """Derive one sample from a body state and buffer it.

        Every sample is buffered whatever its timestamp; `read()` leaves unusable differences out
        of the acceleration rather than dividing by a zero or negative interval. A time that moves
        backwards warns, since it means a discontinuity that `reset()` should have cleared.

        Args:
            time: Simulation time of the state, in seconds. Must increase between calls for the
                finite difference to be meaningful.
            state: World-frame state of the body the sensor is mounted on.

        """
        if self._samples and time < self._samples[-1].time:
            warnings.warn(
                f"Time moved backwards, from {self._samples[-1].time} to {time}. The sample is "
                "kept, but no acceleration is differenced across the rewind. Call reset() when "
                "the body state becomes discontinuous.",
                RuntimeWarning,
                stacklevel=2,
            )

        body_orientation = wp.normalize(state.orientation)
        # Rotate the mount offset into the world to get the sensor origin and the lever arm.
        lever_arm = wp.quat_rotate(body_orientation, self._mount_translation)
        # wp.mul applies the right operand first, so this is the body rotation after the mount.
        orientation = wp.normalize(wp.mul(body_orientation, self._mount_orientation))
        # A point offset from the body origin picks up the transport term w x r. Angular velocity
        # is the same everywhere on a rigid body and is only rotated.
        world_linear_velocity = state.linear_velocity + wp.cross(state.angular_velocity, lever_arm)

        self._samples.append(
            _Sample(
                time=time,
                position=state.position + lever_arm,
                orientation=orientation,
                linear_velocity_world=world_linear_velocity,
                angular_velocity=wp.quat_rotate_inv(orientation, state.angular_velocity),
            )
        )

    def read(self) -> ImuReading | None:
        """Produce a filtered reading from the buffered samples.

        Returns:
            The reading, or ``None`` until ``2 * filter_width`` samples have been buffered.

        """
        if len(self._samples) < 2 * self._filter_width:
            return None

        # Index newest first, so the window indices match the difference below.
        def raw(index: int) -> _Sample:
            return self._samples[len(self._samples) - 1 - index]

        newest = raw(0)
        window = [raw(index) for index in range(self._filter_width)]

        angular_velocity = wp.vec3d(ZERO_VECTOR)
        for entry in window:
            angular_velocity = angular_velocity + entry.angular_velocity
        angular_velocity = _scale(angular_velocity, 1.0 / len(window))

        # Difference the world-frame velocity, never the sensor-frame one. Differentiating a vector
        # already in the rotating frame gives d/dt(R^T v) = R^T a - w x (R^T v), which subtracts
        # the transport term back out: a sensor circling with its axes locked to the trajectory
        # would report zero acceleration rather than a merely inaccurate one.
        linear_acceleration = wp.vec3d(ZERO_VECTOR)
        usable = 0
        for index in range(self._filter_width):
            newer, older = raw(index), raw(index + self._filter_width)
            interval = newer.time - older.time
            if interval > 0.0:
                difference = newer.linear_velocity_world - older.linear_velocity_world
                linear_acceleration = linear_acceleration + _scale(difference, 1.0 / interval)
                usable += 1
        # Average over the usable differences, not the whole window, or a repeated or rewound
        # timestamp biases the result toward zero.
        if usable > 0:
            linear_acceleration = _scale(linear_acceleration, 1.0 / usable)

        # An accelerometer measures specific force, a_proper = a_coordinate - g, still world frame.
        linear_acceleration = linear_acceleration - self._gravity

        return ImuReading(
            time=newest.time,
            position=newest.position,
            # The window is newest first, so the newest sample sets the sign convention.
            orientation=average_quaternions([entry.orientation for entry in window]),
            # Only the reported results are rotated into the sensor frame.
            linear_velocity=wp.quat_rotate_inv(newest.orientation, newest.linear_velocity_world),
            angular_velocity=angular_velocity,
            linear_acceleration=wp.quat_rotate_inv(newest.orientation, linear_acceleration),
        )
