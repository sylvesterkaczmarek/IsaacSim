# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for AckermannController runtime configuration refresh."""

from types import SimpleNamespace

import omni.kit.test
from isaacsim.robot.wheeled_robots.nodes.ogn.python.nodes.OgnAckermannController import OgnAckermannController


class _State:
    def __init__(self) -> None:
        self.initialized = True
        self.wheel_base = 1.0
        self.track_width = 1.0
        self.front_wheel_radius = 0.3
        self.back_wheel_radius = 0.3
        self.max_wheel_velocity = 100.0
        self.invert_steering = False
        self.max_wheel_rotation_angle = 1.0
        self.max_acceleration = 2.0
        self.max_steering_angle_velocity = 3.0
        self.initialize_calls = 0

    def initialize_controller(self) -> None:
        self.initialize_calls += 1
        self.initialized = True

    def forward(self, command):
        return [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]


class TestAckermannControllerConfigRefresh(omni.kit.test.AsyncTestCase):
    """Verify configuration changes rebuild the cached Ackermann controller once."""

    @staticmethod
    def _db(state: _State):
        return SimpleNamespace(
            per_instance_state=state,
            inputs=SimpleNamespace(
                wheelBase=2.0,
                trackWidth=1.0,
                frontWheelRadius=0.3,
                backWheelRadius=0.3,
                maxWheelVelocity=100.0,
                invertSteering=False,
                maxWheelRotation=1.0,
                maxAcceleration=2.0,
                maxSteeringAngleVelocity=3.0,
                steeringAngle=0.1,
                steeringAngleVelocity=0.0,
                speed=1.0,
                acceleration=0.0,
                dt=1.0 / 60.0,
            ),
            outputs=SimpleNamespace(),
            log_warning=lambda message: None,
        )

    async def test_wheel_base_change_reinitializes_once(self) -> None:
        state = _State()
        db = self._db(state)

        self.assertTrue(OgnAckermannController.compute(db))
        self.assertEqual(state.initialize_calls, 1)
        self.assertEqual(state.wheel_base, 2.0)

        self.assertTrue(OgnAckermannController.compute(db))
        self.assertEqual(state.initialize_calls, 1)
