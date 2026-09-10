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

"""Tests for DifferentialDriveController."""

import math

import isaacsim.robot_motion.controllers as ctrl
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.kit.test
import warp as wp

# Four-joint space: two wheel joints interleaved with unrelated joints so that
# tests exercise the joint-selector logic, not just a trivially matching space.
JOINT_SPACE = ["joint_a", "left_wheel_joint", "joint_b", "right_wheel_joint"]
WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.2


# Analytic differential drive kinematics:
#   omega_L = (2v - omega * b) / (2r)
#   omega_R = (2v + omega * b) / (2r)
def _expected_wheels(v, omega, r=WHEEL_RADIUS, b=WHEEL_BASE):
    left = (2.0 * v - omega * b) / (2.0 * r)
    right = (2.0 * v + omega * b) / (2.0 * r)
    return left, right


def _assert_wheels(tc, controller, v, omega, setpoint=None):
    """Assert that controller.forward() produces the expected wheel velocities."""
    if setpoint is None:
        setpoint = _site_setpoint(v, omega)
    result = controller.forward(_EMPTY_STATE, setpoint, 0.0)
    tc.assertTrue(np.allclose(result.joints.velocities.numpy(), _expected_wheels(v, omega), atol=1e-5))


def _make_controller(**kwargs):
    defaults = dict(
        robot_joint_space=JOINT_SPACE,
        left_wheel_joint="left_wheel_joint",
        right_wheel_joint="right_wheel_joint",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
    )
    defaults.update(kwargs)
    return ctrl.DifferentialDriveController(**defaults)


def _site_setpoint(v, omega, device=None, site_name="control_point"):
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=[site_name],
            linear_velocities=([site_name], wp.array([[v, 0.0, 0.0]], dtype=wp.float32, device=device)),
            angular_velocities=([site_name], wp.array([[0.0, 0.0, omega]], dtype=wp.float32, device=device)),
        )
    )


_EMPTY_STATE = mg.RobotState()


class TestDifferentialDriveControllerConstruction(omni.kit.test.AsyncTestCase):
    """Tests for constructor validation."""

    async def test_default_construction(self):
        """Controller constructs without error using default directions."""
        controller = _make_controller()
        self.assertIsNotNone(controller)

    async def test_custom_perpendicular_directions(self):
        """Explicitly perpendicular directions (90°) are accepted."""
        controller = _make_controller(
            forward_direction=[0.0, 1.0, 0.0],
            rotation_direction=[1.0, 0.0, 0.0],
        )
        self.assertIsNotNone(controller)

    async def test_non_unit_vectors_are_normalized(self):
        """Non-unit input vectors produce identical kinematics to their unit-length equivalents."""
        controller_scaled = _make_controller(
            forward_direction=[2.0, 0.0, 0.0],
            rotation_direction=[0.0, 0.0, 5.0],
        )
        controller_unit = _make_controller(
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[0.0, 0.0, 1.0],
        )
        for v, omega in [(1.0, 0.0), (0.0, 1.0), (0.5, 2.0)]:
            setpoint = _site_setpoint(v, omega)
            scaled = controller_scaled.forward(_EMPTY_STATE, setpoint, 0.0).joints.velocities.numpy()
            unit = controller_unit.forward(_EMPTY_STATE, setpoint, 0.0).joints.velocities.numpy()
            self.assertTrue(np.allclose(scaled, unit, atol=1e-5))

    async def test_zero_forward_direction_raises(self):
        """A zero forward_direction vector raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            forward_direction=[0.0, 0.0, 0.0],
        )

    async def test_zero_rotation_direction_raises(self):
        """A zero rotation_direction vector raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            rotation_direction=[0.0, 0.0, 0.0],
        )

    async def test_wrong_shape_raises(self):
        """A non-3-element direction vector raises ValueError."""
        self.assertRaises(ValueError, _make_controller, forward_direction=[1.0, 0.0])
        self.assertRaises(ValueError, _make_controller, rotation_direction=[1.0, 0.0, 0.0, 0.0])

    async def test_parallel_directions_raise(self):
        """Parallel (0°) directions raise ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[1.0, 0.0, 0.0],
        )

    async def test_antiparallel_directions_raise(self):
        """Anti-parallel (180°) directions raise ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[-1.0, 0.0, 0.0],
        )

    async def test_directions_too_close_raises(self):
        """Directions less than 89.9° apart raise ValueError."""
        angle = math.radians(89.0)
        self.assertRaises(
            ValueError,
            _make_controller,
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[math.cos(angle), math.sin(angle), 0.0],
        )

    async def test_directions_exactly_at_limit_accepted(self):
        """Directions exactly 89.9° apart are accepted."""
        angle = math.radians(89.9)
        controller = _make_controller(
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[math.cos(angle), math.sin(angle), 0.0],
        )
        self.assertIsNotNone(controller)

    async def test_wp_array_directions(self):
        """wp.array inputs for forward_direction and rotation_direction are accepted."""
        controller = _make_controller(
            forward_direction=wp.array([1.0, 0.0, 0.0], dtype=wp.float32),
            rotation_direction=wp.array([0.0, 0.0, 1.0], dtype=wp.float32),
        )
        result = controller.forward(_EMPTY_STATE, _site_setpoint(1.0, 0.0), 0.0)
        left_exp, right_exp = _expected_wheels(1.0, 0.0)
        actual = result.joints.velocities.numpy()
        self.assertTrue(np.allclose(actual, [left_exp, right_exp], atol=1e-5))

    async def test_zero_wheel_radius_raises(self):
        """A wheel_radius of zero raises ValueError."""
        self.assertRaises(ValueError, _make_controller, wheel_radius=0.0)

    async def test_negative_wheel_radius_raises(self):
        """A negative wheel_radius raises ValueError."""
        self.assertRaises(ValueError, _make_controller, wheel_radius=-0.05)

    async def test_zero_wheel_base_raises(self):
        """A wheel_base of zero raises ValueError."""
        self.assertRaises(ValueError, _make_controller, wheel_base=0.0)

    async def test_negative_wheel_base_raises(self):
        """A negative wheel_base raises ValueError."""
        self.assertRaises(ValueError, _make_controller, wheel_base=-0.2)

    async def test_zero_max_linear_speed_raises(self):
        """A max_linear_speed of zero raises ValueError."""
        self.assertRaises(ValueError, _make_controller, max_linear_speed=0.0)

    async def test_negative_max_angular_speed_raises(self):
        """A negative max_angular_speed raises ValueError."""
        self.assertRaises(ValueError, _make_controller, max_angular_speed=-1.0)

    async def test_negative_max_wheel_speed_raises(self):
        """A negative max_wheel_speed raises ValueError."""
        self.assertRaises(ValueError, _make_controller, max_wheel_speed=-10.0)

    async def test_wheel_joint_not_in_joint_space_raises(self):
        """Wheel joint names absent from robot_joint_space raise an error."""
        self.assertRaises(
            (ValueError, KeyError),
            _make_controller,
            robot_joint_space=["joint_a", "joint_b"],
            left_wheel_joint="left_wheel_joint",
            right_wheel_joint="right_wheel_joint",
        )

    async def test_duplicate_wheel_joint_names_raise(self):
        """Identical left_wheel_joint and right_wheel_joint names raise ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            left_wheel_joint="left_wheel_joint",
            right_wheel_joint="left_wheel_joint",
        )


class TestDifferentialDriveControllerCPU(omni.kit.test.AsyncTestCase):
    """Tests that the controller works correctly on the CPU device."""

    async def test_forward_on_cpu_device(self):
        """Controller constructed on the CPU device produces correct wheel velocities."""
        controller = _make_controller(device="cpu")
        result = controller.forward(_EMPTY_STATE, _site_setpoint(1.0, 0.0, device="cpu"), 0.0)
        self.assertTrue(np.allclose(result.joints.velocities.numpy(), _expected_wheels(1.0, 0.0), atol=1e-5))


class TestDifferentialDriveControllerReset(omni.kit.test.AsyncTestCase):
    """Tests for reset()."""

    async def test_reset_returns_true(self):
        """reset() always returns True (controller is stateless)."""
        controller = _make_controller()
        self.assertTrue(controller.reset(_EMPTY_STATE, None, 0.0))
        self.assertTrue(controller.reset(_EMPTY_STATE, _site_setpoint(1.0, 0.5), 1.0))


class TestDifferentialDriveControllerForward(omni.kit.test.AsyncTestCase):
    """Tests for forward() kinematics and output structure."""

    async def setUp(self):
        self.controller = _make_controller()

    async def test_returns_none_for_none_setpoint(self):
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, None, 0.0))

    async def test_returns_none_for_missing_sites(self):
        setpoint = mg.RobotState()
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, setpoint, 0.0))

    async def test_returns_none_for_missing_linear_velocity(self):
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                angular_velocities=(["control_point"], wp.array([[0.0, 0.0, 1.0]], dtype=wp.float32)),
            )
        )
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, setpoint, 0.0))

    async def test_returns_none_for_missing_angular_velocity(self):
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32)),
            )
        )
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, setpoint, 0.0))

    async def test_output_joint_names(self):
        """Output velocities are keyed only to the two wheel joints, not the full joint space."""
        result = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0, 0.0), 0.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.joints.velocity_names, ["left_wheel_joint", "right_wheel_joint"])
        self.assertEqual(len(result.joints.velocities.numpy()), 2)

    async def test_velocity_indices_are_full_joint_space_positions(self):
        """velocity_indices must reflect the wheel joints' positions in the full robot_joint_space.

        JOINT_SPACE = ["joint_a", "left_wheel_joint", "joint_b", "right_wheel_joint"]
        so left_wheel_joint is at index 1 and right_wheel_joint is at index 3.
        Returning [0, 1] (indices within a 2-element wheel-only space) would be wrong.
        """
        result = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0, 0.0), 0.0)
        indices = result.joints.velocity_indices.numpy().tolist()
        self.assertEqual(indices, [1, 3])

    async def test_pure_forward(self):
        """v > 0, omega = 0 → equal wheel speeds."""
        _assert_wheels(self, self.controller, 1.0, 0.0)

    async def test_pure_rotation(self):
        """v = 0, omega > 0 → equal and opposite wheel speeds."""
        omega = 1.0
        _assert_wheels(self, self.controller, 0.0, omega)
        actual = self.controller.forward(_EMPTY_STATE, _site_setpoint(0.0, omega), 0.0).joints.velocities.numpy()
        self.assertTrue(np.allclose(actual[0], -actual[1], atol=1e-5))

    async def test_combined_motion(self):
        """General (v, omega) case matches analytic kinematics."""
        _assert_wheels(self, self.controller, 0.5, 2.0)

    async def test_reverse(self):
        """Negative linear velocity produces reversed wheel speeds."""
        _assert_wheels(self, self.controller, -1.0, 0.0)

    async def test_speed_clamping(self):
        """Speeds above max_linear_speed are clamped."""
        controller = _make_controller(max_linear_speed=0.5)
        # Command v=2.0 but effective v is clamped to 0.5.
        _assert_wheels(self, controller, 0.5, 0.0, setpoint=_site_setpoint(2.0, 0.0))

    async def test_custom_forward_direction(self):
        """Controller correctly projects velocity onto a non-default forward axis."""
        # Forward direction is +Y; setpoint has linear_velocity along Y.
        controller = _make_controller(
            forward_direction=[0.0, 1.0, 0.0],
            rotation_direction=[0.0, 0.0, 1.0],
        )
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[0.0, 1.0, 0.0]], dtype=wp.float32)),
                angular_velocities=(["control_point"], wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)),
            )
        )
        _assert_wheels(self, controller, 1.0, 0.0, setpoint=setpoint)

    async def test_sequential_calls_update_output(self):
        """Calling forward() twice with different setpoints produces independently correct results."""
        result1 = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0, 0.0), 0.0)
        result2 = self.controller.forward(_EMPTY_STATE, _site_setpoint(0.0, 2.0), 0.0)
        actual1 = result1.joints.velocities.numpy()
        actual2 = result2.joints.velocities.numpy()
        self.assertTrue(np.allclose(actual1, _expected_wheels(1.0, 0.0), atol=1e-5))
        self.assertTrue(np.allclose(actual2, _expected_wheels(0.0, 2.0), atol=1e-5))

    async def test_max_angular_speed_clamping(self):
        """Angular speeds above max_angular_speed are clamped."""
        controller = _make_controller(max_angular_speed=1.0)
        # Command omega=4.0 but effective omega is clamped to 1.0.
        _assert_wheels(self, controller, 0.0, 1.0, setpoint=_site_setpoint(0.0, 4.0))

    async def test_max_linear_speed_clamping_negative(self):
        """Negative speeds below -max_linear_speed are clamped."""
        controller = _make_controller(max_linear_speed=0.5)
        _assert_wheels(self, controller, -0.5, 0.0, setpoint=_site_setpoint(-2.0, 0.0))

    async def test_max_wheel_speed_clamping(self):
        """Per-wheel output is clamped to max_wheel_speed after kinematic inversion."""
        # With v=1.0, omega=10.0, r=0.05, b=0.2:
        #   left  = (2*1 - 10*0.2) / (2*0.05) = (2 - 2) / 0.1 = 0
        #   right = (2*1 + 10*0.2) / (2*0.05) = (2 + 2) / 0.1 = 40
        # A max_wheel_speed of 20 should clamp right to 20, leave left at 0.
        controller = _make_controller(max_wheel_speed=20.0)
        result = controller.forward(_EMPTY_STATE, _site_setpoint(1.0, 10.0), 0.0)
        actual = result.joints.velocities.numpy()
        self.assertTrue(np.allclose(actual[0], 0.0, atol=1e-5))
        self.assertTrue(np.allclose(actual[1], 20.0, atol=1e-5))

    async def test_custom_rotation_direction(self):
        """Controller correctly projects angular velocity onto a non-default rotation axis."""
        # Rotation axis is +Y; setpoint has angular_velocity along Y.
        controller = _make_controller(
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[0.0, 1.0, 0.0],
        )
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)),
                angular_velocities=(["control_point"], wp.array([[0.0, 1.0, 0.0]], dtype=wp.float32)),
            )
        )
        _assert_wheels(self, controller, 0.0, 1.0, setpoint=setpoint)

    async def test_orthogonal_component_ignored(self):
        """Velocity components orthogonal to forward_direction do not affect output."""
        # Add a large lateral component; only the X projection should matter.
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[1.0, 99.0, 99.0]], dtype=wp.float32)),
                angular_velocities=(["control_point"], wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)),
            )
        )
        _assert_wheels(self, self.controller, 1.0, 0.0, setpoint=setpoint)
