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

"""Effort sensor for measuring joint torques.

Reads joint effort (torque/force) values from articulated physics bodies via
the C++ ``IEffortSensor`` Carbonite interface. The sensor inherits the C++
interface lifecycle from :class:`_PhysicsSensorRuntimeBase`.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral
from typing import Any

import carb
import carb.eventdispatcher
import omni.usd
from isaacsim.core.deprecation_manager import deprecated
from isaacsim.core.simulation_manager import SimulationManager

from ._sensor_base import _PhysicsSensorRuntimeBase

_INVALID_EFFORT_READING = None


def _get_invalid_effort_reading() -> object:
    global _INVALID_EFFORT_READING
    if _INVALID_EFFORT_READING is None:
        from .. import _physics_sensors

        _INVALID_EFFORT_READING = _physics_sensors.EffortSensorReading()
    return _INVALID_EFFORT_READING


class EffortSensorReading:
    """Effort sensor reading data.

    Args:
        is_valid: Whether this reading contains valid data.
        time: Simulation time when the reading was taken.
        value: Effort (torque/force) value at the joint.
    """

    def __init__(self, is_valid: bool = False, time: float = 0, value: float = 0) -> None:
        self.is_valid = is_valid
        self.time = time
        self.value = value


class EffortSensor(_PhysicsSensorRuntimeBase):
    """Sensor for measuring joint effort (torque/force).

    Owns the C++ ``IEffortSensor`` Carbonite interface directly and maintains a
    Python-side circular buffer of readings for historical access.
    Newton does not currently expose projected joint forces, so readings are
    invalid while Newton is active.

    Args:
        path: USD path to the joint, formatted as articulation_path/joint_name.
        enabled: Whether the sensor is initially enabled.

    Example:

        .. code-block:: python

            from isaacsim.sensors.experimental.physics import EffortSensor

            # Create sensor for a robot joint
            sensor = EffortSensor("/World/Robot/joint_1")

            # Read joint effort
            reading = sensor.get_sensor_reading()
            if reading.is_valid:
                print(f"Joint torque: {reading.value}")
    """

    def __init__(self, path: str, enabled: bool = True) -> None:
        super().__init__(path)

        self.enabled = enabled

        self.body_prim_path = "/".join(path.split("/")[:-1])
        self.dof_name = path.split("/")[-1]

        self.data_buffer_size = 10
        self.sensor_reading_buffer = deque(
            (EffortSensorReading() for _ in range(self.data_buffer_size)), maxlen=self.data_buffer_size
        )

        self.physics_num_steps = 0.0
        self.current_time = 0.0

        self._stage_open_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.OPENED),
            on_event=self._stage_open_callback_fn,
            observer_name="isaacsim.sensors.experimental.physics.EffortSensor._stage_open_callback",
        )

    def _acquire_interface(self) -> object | None:
        from .extension import get_effort_sensor_interface

        return get_effort_sensor_interface()

    def _get_invalid_reading(self) -> object:
        return _get_invalid_effort_reading()

    def _stage_open_callback_fn(self, event: Any = None) -> None:
        """Handle stage open by releasing subscriptions.

        Args:
            event: Stage-open event payload.
        """
        self._stage_open_sub = None

    def on_timeline_stop(self) -> None:
        """Reset C++ interface state and clear the reading buffer."""
        super().on_timeline_stop()
        self.current_time = 0
        self.sensor_reading_buffer = deque(
            (EffortSensorReading() for _ in range(self.data_buffer_size)), maxlen=self.data_buffer_size
        )
        self.physics_num_steps = 0

    def get_sensor_reading(self) -> EffortSensorReading:
        """Get the current effort sensor reading.

        Returns:
            Reading with effort value and validity state. The reading is invalid
            when the active backend does not expose projected joint forces.
        """
        if not self.enabled:
            return EffortSensorReading()

        cpp_reading = self._get_reading()
        if not cpp_reading.is_valid:
            return EffortSensorReading()

        reading = EffortSensorReading(
            is_valid=True,
            time=cpp_reading.time,
            value=cpp_reading.value,
        )

        self.sensor_reading_buffer.appendleft(reading)

        return reading

    def get_data(self) -> dict:
        """Get the current effort sensor data as a structured frame.

        Returns:
            Frame data containing:
                - ``"value"``: Effort (torque/force) value at the joint.
                - ``"is_valid"``: Whether the reading contains valid data.
                - ``"time"``: Simulation time of the reading.
                - ``"physics_step"``: Physics step number.
        """
        reading = self.get_sensor_reading()
        return {
            "value": float(reading.value),
            "is_valid": bool(reading.is_valid),
            "time": float(reading.time),
            "physics_step": int(SimulationManager.get_num_physics_steps()),
        }

    @deprecated(replacement="a new EffortSensor constructed with the target joint path", removal_version="7.0")
    def update_dof_name(self, dof_name: str) -> None:
        """Update the DOF (degree of freedom) name being measured.

        Resets the C++ sensor and rebinds the runtime to the new joint path.

        .. deprecated:: 3.3.0
            Construct a new :class:`EffortSensor` with the target joint path instead.
            This method will be removed in Isaac Sim 7.0.

        Args:
            dof_name: Name of the joint DOF to monitor.
        """
        self.dof_name = dof_name
        new_path = self.body_prim_path + "/" + dof_name
        self._rebind(new_path)

    @deprecated(
        reason="The sensor reading history buffer is not consumed within Isaac Sim.",
        removal_version="7.0",
    )
    def change_buffer_size(self, new_buffer_size: int) -> None:
        """Change the size of the sensor reading buffer.

        .. deprecated:: 3.3.0
            The reading history buffer is not consumed within Isaac Sim.
            This method will be removed in Isaac Sim 7.0.

        Args:
            new_buffer_size: New buffer size (number of readings to store). Must be a positive
                integer. Booleans are rejected, even though ``bool`` is an integer subclass.

        Raises:
            TypeError: If ``new_buffer_size`` is a boolean or is not an integer.
            ValueError: If ``new_buffer_size`` is less than 1.
        """
        # Validate before mutating, so a rejected size leaves the sensor untouched.
        # bool subclasses int, so screen it separately; numpy integers register as Integral
        # while numpy booleans do not, so both land on the correct side of this check.
        if isinstance(new_buffer_size, bool) or not isinstance(new_buffer_size, Integral):
            raise TypeError(f"'new_buffer_size' must be an integer, got {type(new_buffer_size).__name__}")
        new_buffer_size = int(new_buffer_size)
        if new_buffer_size < 1:
            raise ValueError(f"'new_buffer_size' must be a positive integer, got {new_buffer_size}")

        old = list(self.sensor_reading_buffer)
        self.data_buffer_size = new_buffer_size
        self.sensor_reading_buffer = deque(maxlen=new_buffer_size)
        for item in old[:new_buffer_size]:
            self.sensor_reading_buffer.append(item)
        while len(self.sensor_reading_buffer) < new_buffer_size:
            self.sensor_reading_buffer.append(EffortSensorReading())
