# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""IMU sensor runtime providing frame-based data access via the C++ interface."""

from __future__ import annotations

import numpy as np
from isaacsim.core.simulation_manager import SimulationManager

from ._sensor_base import _PhysicsSensorRuntime
from .imu import IMU

_INVALID_IMU_READING = None

# Frame layout inside the sensor-owned backing buffer. Keeping the three channels in one
# contiguous array means a copied frame costs a single allocation instead of three.
_LINEAR_ACCELERATION_SLICE = slice(0, 3)
_ANGULAR_VELOCITY_SLICE = slice(3, 6)
_ORIENTATION_SLICE = slice(6, 10)
_FRAME_BUFFER_SIZE = 10


def _get_invalid_imu_reading() -> object:
    global _INVALID_IMU_READING
    if _INVALID_IMU_READING is None:
        from .. import _physics_sensors

        _INVALID_IMU_READING = _physics_sensors.ImuSensorReading()
    return _INVALID_IMU_READING


class IMUSensor(_PhysicsSensorRuntime):
    """Runtime wrapper for an Isaac IMU sensor with frame-based data access.

    Wraps an :class:`IMU` authoring object and owns the C++ ``IImuSensor``
    Carbonite interface. Exposes :meth:`get_data` for a structured per-step
    dictionary and :meth:`get_sensor_reading` for the raw C++ struct.

    Args:
        path: Either a string USD path to an existing IsaacImuSensor prim, or a
            pre-built :class:`IMU` authoring object. To create a new prim, use
            :meth:`IMU.create`.

    Example:

    .. code-block:: python

        from isaacsim.sensors.experimental.physics import IMU, IMUSensor

        # Wrap an existing IsaacImuSensor prim
        sensor = IMUSensor("/World/Robot/body/imu")

        # Create a new sensor with custom parameters
        sensor = IMUSensor(
            IMU.create(
                "/World/Robot/body/imu",
                linear_acceleration_filter_size=5,
            )
        )

        frame = sensor.get_data()
        print(f"Linear acceleration: {frame['linear_acceleration']}")
    """

    _AUTHORING_CLASS = IMU
    _AUTHORING_ATTR = "_imu"

    @property
    def imu(self) -> IMU:
        """Authoring object encapsulated by this sensor.

        Returns:
            The :class:`IMU` instance wrapping the underlying USD prim.
        """
        return self._imu

    def _acquire_interface(self) -> object | None:
        from .extension import get_imu_sensor_interface

        return get_imu_sensor_interface()

    def _get_invalid_reading(self) -> object:
        return _get_invalid_imu_reading()

    def __init__(self, path: "str | IMU") -> None:
        super().__init__(path)
        # Held for the per-step in-place writes in get_data. Derived from the frame the base
        # class just built rather than stashed by _init_frame, so the backing buffer and the
        # views into it cannot drift apart if the frame is ever rebuilt.
        self._frame_buffer = self._current_frame["linear_acceleration"].base

    def _init_frame(self) -> dict[str, object]:
        buffer = np.zeros((_FRAME_BUFFER_SIZE,), dtype=np.float32)
        buffer[_ORIENTATION_SLICE.start] = 1.0  # Identity quaternion [w, x, y, z]
        return {
            "time": 0.0,
            "physics_step": 0.0,
            "linear_acceleration": buffer[_LINEAR_ACCELERATION_SLICE],
            "angular_velocity": buffer[_ANGULAR_VELOCITY_SLICE],
            "orientation": buffer[_ORIENTATION_SLICE],
        }

    def get_sensor_reading(self, read_gravity: bool = True) -> object:
        """Get the current IMU sensor reading as the raw C++ struct.

        See :meth:`get_data` for what ``read_gravity`` selects.

        .. deprecated:: 3.2.0

            |br| ``read_gravity`` is deprecated and still honored. ``IsaacComputeOdometry``'s
            ``globalLinearAcceleration`` is gravity-free, but reports world-frame acceleration of
            the body rather than sensor-frame acceleration at this prim.

        Args:
            read_gravity: Whether the accelerometer channel includes the gravity reaction
                term.

        Returns:
            The C++ ``ImuSensorReading`` struct, with scalar ``linear_acceleration_x`` /
            ``_y`` / ``_z``, ``angular_velocity_x`` / ``_y`` / ``_z``, and ``orientation_w`` /
            ``_x`` / ``_y`` / ``_z`` fields; use :meth:`get_data` for numpy arrays. A valid
            reading is a fresh struct, an invalid one (``is_valid`` is ``False``) a shared
            placeholder, so check ``is_valid`` before retaining it.
        """
        return self._get_reading(read_gravity)

    def get_data(self, read_gravity: bool = True) -> dict:
        """Get the current IMU sensor data as a structured frame.

        ``read_gravity=True`` reports specific force, what an accelerometer measures: ``+g`` at
        rest and ``0`` in free fall. ``read_gravity=False`` reports coordinate acceleration:
        ``0`` at rest and ``-g`` in free fall. ``g`` is in stage linear units per second squared
        (``981`` on a centimetre stage), on the sensor axis opposing gravity.

        .. deprecated:: 3.2.0

            |br| ``read_gravity`` is deprecated and still honored. ``IsaacComputeOdometry``'s
            ``globalLinearAcceleration`` is gravity-free, but reports world-frame acceleration of
            the body rather than sensor-frame acceleration at this prim.

        Args:
            read_gravity: Whether the accelerometer channel includes the gravity reaction
                term.

        Returns:
            Newly allocated frame data, independent of any other call, containing:
                - ``"linear_acceleration"``: Linear acceleration ``[x, y, z]``.
                - ``"angular_velocity"``: Angular velocity ``[x, y, z]``.
                - ``"orientation"``: Orientation as ``[w, x, y, z]`` quaternion.
                - ``"time"``: Simulation time of reading.
                - ``"physics_step"``: Physics step number.

        Note:
            The frame is refreshed only on a valid reading; otherwise the previous values are
            returned. Use :meth:`get_sensor_reading` and check ``is_valid`` to tell them apart,
            or on a hot path to read the values without allocating a frame.
        """
        reading = self.get_sensor_reading(read_gravity=read_gravity)
        frame = self._current_frame

        if reading.is_valid:
            buffer = self._frame_buffer
            buffer[0] = reading.linear_acceleration_x
            buffer[1] = reading.linear_acceleration_y
            buffer[2] = reading.linear_acceleration_z
            buffer[3] = reading.angular_velocity_x
            buffer[4] = reading.angular_velocity_y
            buffer[5] = reading.angular_velocity_z
            buffer[6] = reading.orientation_w
            buffer[7] = reading.orientation_x
            buffer[8] = reading.orientation_y
            buffer[9] = reading.orientation_z

            frame["time"] = reading.time
            frame["physics_step"] = float(SimulationManager.get_num_physics_steps())

        # One allocation for all three channels; the returned arrays are views into it, so the
        # caller still sees three independent, contiguous, writable arrays.
        buffer = self._frame_buffer.copy()
        return {
            "time": frame["time"],
            "physics_step": frame["physics_step"],
            "linear_acceleration": buffer[_LINEAR_ACCELERATION_SLICE],
            "angular_velocity": buffer[_ANGULAR_VELOCITY_SLICE],
            "orientation": buffer[_ORIENTATION_SLICE],
        }
