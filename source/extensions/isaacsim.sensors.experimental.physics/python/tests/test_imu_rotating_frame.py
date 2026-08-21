# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for IMU linear acceleration on a rotating rigid body."""

from __future__ import annotations

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.timeline
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics import IMU, IMUSensor
from pxr import Gf


class TestIMURotatingFrameAcceleration(omni.kit.test.AsyncTestCase):
    """Verify IMU acceleration is differentiated in an inertial frame."""

    async def tearDown(self) -> None:
        """Stop playback and clear physics state after each test."""
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.stop()
        SimulationManager.invalidate_physics()
        await omni.kit.app.get_app().next_update_async()

    async def test_constant_world_velocity_while_rotating_has_zero_linear_acceleration(self) -> None:
        """Rotation must not turn constant inertial velocity into apparent acceleration."""
        await stage_utils.create_new_stage_async()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        SimulationManager.setup_simulation(dt=1.0 / 120.0)
        SimulationManager.get_physics_scenes()[0].set_gravity(Gf.Vec3f(0.0, 0.0, 0.0))

        cube_path = "/World/RotatingCube"
        Cube(cube_path, sizes=0.2, positions=[0.0, 0.0, 2.0])
        GeomPrim(cube_path, apply_collision_apis=True)
        body = RigidPrim(cube_path, masses=[1.0])
        sensor = IMUSensor(
            IMU.create(
                cube_path + "/imu",
                translations=[[0.0, 0.0, 0.0]],
                orientations=[[1.0, 0.0, 0.0, 0.0]],
                linear_acceleration_filter_size=1,
                angular_velocity_filter_size=1,
                orientation_filter_size=1,
            )
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        await omni.kit.app.get_app().next_update_async()

        body.set_velocities(
            linear_velocities=[[1.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 2.0]],
        )

        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()

        reading = sensor.get_sensor_reading(read_gravity=False)
        acceleration = np.array(
            [reading.linear_acceleration_x, reading.linear_acceleration_y, reading.linear_acceleration_z]
        )
        linear_velocity, _ = body.get_velocities()

        self.assertTrue(reading.is_valid)
        self.assertTrue(np.allclose(linear_velocity.numpy()[0], [1.0, 0.0, 0.0], atol=0.05))
        self.assertLess(np.linalg.norm(acceleration), 0.25)
