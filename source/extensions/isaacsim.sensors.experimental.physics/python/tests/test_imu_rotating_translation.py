# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for IMU acceleration while a rigid body rotates and translates."""

from __future__ import annotations

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.timeline
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics import IMU, IMUSensor


class TestIMURotatingTranslation(omni.kit.test.AsyncTestCase):
    """Verify linear acceleration is differentiated in an inertial frame."""

    async def setUp(self) -> None:
        await stage_utils.create_new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        SimulationManager.setup_simulation(dt=1.0 / 120.0)
        self._timeline = omni.timeline.get_timeline_interface()

    async def tearDown(self) -> None:
        if self._timeline.is_playing():
            self._timeline.stop()
        SimulationManager.invalidate_physics()
        await omni.kit.app.get_app().next_update_async()

    async def test_constant_world_velocity_while_rotating_has_zero_linear_acceleration(self) -> None:
        """Rotation must not turn constant world velocity into apparent acceleration."""
        body_path = "/World/RotatingBody"
        Cube(body_path, sizes=0.2, positions=[0.0, 0.0, 2.0])
        body = RigidPrim(body_path, masses=[1.0])
        body.set_enabled_gravities([False])

        sensor = IMUSensor(
            IMU.create(
                body_path + "/imu",
                translations=[[0.0, 0.0, 0.0]],
                orientations=[[1.0, 0.0, 0.0, 0.0]],
                linear_acceleration_filter_size=1,
                angular_velocity_filter_size=1,
                orientation_filter_size=1,
            )
        )

        self._timeline.play()
        SimulationManager.initialize_physics()
        body.set_velocities(
            linear_velocities=[[1.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 2.0]],
        )

        for _ in range(30):
            SimulationManager.step(steps=1, update_fabric=False)

        reading = sensor.get_sensor_reading(read_gravity=False)
        acceleration = np.array(
            [reading.linear_acceleration_x, reading.linear_acceleration_y, reading.linear_acceleration_z]
        )

        self.assertTrue(reading.is_valid)
        self.assertLess(
            np.linalg.norm(acceleration),
            0.05,
            f"constant world velocity should have near-zero inertial acceleration, got {acceleration}",
        )
