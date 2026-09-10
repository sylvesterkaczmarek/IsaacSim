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

"""Tests for AckermannController."""

import math

import isaacsim.robot_motion.controllers as ctrl
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.kit.test
import warp as wp

# Eight-joint space: two non-actuated joints interleaved with the actuated ones
# so that index-mapping bugs produce wrong indices rather than accidentally correct ones.
#
#   Index  Name
#   0      joint_a            (non-actuated)
#   1      left_steer_wheel   (velocity target)
#   2      joint_b            (non-actuated)
#   3      right_steer_wheel  (velocity target)
#   4      left_steering      (position target — steering angle)
#   5      left_rear_wheel    (velocity target, non-steerable)
#   6      right_steering     (position target — steering angle)
#   7      right_rear_wheel   (velocity target, non-steerable)
JOINT_SPACE = [
    "joint_a",
    "left_steer_wheel",
    "joint_b",
    "right_steer_wheel",
    "left_steering",
    "left_rear_wheel",
    "right_steering",
    "right_rear_wheel",
]

# Shared geometric constants.
STEERABLE_RADIUS = 0.3
WHEEL_BASE = 1.5
TRACK_WIDTH = 1.2

_EMPTY_STATE = mg.RobotState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _site_setpoint(v_x: float, v_y: float = 0.0, device=None, site_name: str = "control_point") -> mg.RobotState:
    """Create a control-point linear-velocity setpoint from a velocity vector.

    The command must be a real velocity vector ``[v_x, v_y, 0]``.  For a vehicle
    travelling at total speed ``v`` with body turning angle ``theta``:
    ``v_x = v * cos(theta)``, ``v_y = v * sin(theta)``.

    Args:
        v_x: Forward velocity component [m/s].
        v_y: Leftward velocity component [m/s].
        device: Warp device for the velocity array.
        site_name: Name of the control-point site.
    """
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=[site_name],
            linear_velocities=([site_name], wp.array([[v_x, v_y, 0.0]], dtype=wp.float32, device=device)),
        )
    )


def _make_controller(**kwargs):
    """Build an AckermannController over the full eight-joint JOINT_SPACE."""
    defaults = dict(
        robot_joint_space=JOINT_SPACE,
        left_steerable_wheel_joint="left_steer_wheel",
        right_steerable_wheel_joint="right_steer_wheel",
        left_steering_joint="left_steering",
        right_steering_joint="right_steering",
        steerable_wheel_radius=STEERABLE_RADIUS,
        wheel_base=WHEEL_BASE,
        track_width=TRACK_WIDTH,
        left_non_steerable_wheel_joint="left_rear_wheel",
        right_non_steerable_wheel_joint="right_rear_wheel",
    )
    defaults.update(kwargs)
    return ctrl.AckermannController(**defaults)


def _make_simple_controller(**kwargs):
    """Build an AckermannController with no non-steerable wheels."""
    simple_space = ["left_steer_wheel", "right_steer_wheel", "left_steering", "right_steering"]
    defaults = dict(
        robot_joint_space=simple_space,
        left_steerable_wheel_joint="left_steer_wheel",
        right_steerable_wheel_joint="right_steer_wheel",
        left_steering_joint="left_steering",
        right_steering_joint="right_steering",
        steerable_wheel_radius=STEERABLE_RADIUS,
        wheel_base=WHEEL_BASE,
        track_width=TRACK_WIDTH,
    )
    defaults.update(kwargs)
    return ctrl.AckermannController(**defaults)


def _run(controller, v: float, theta: float = 0.0):
    """Run one forward step and return ``(steer_angles, wheel_speeds)`` as numpy arrays.

    Args:
        v: Total vehicle speed [m/s] — negative for reversing.
        theta: Body turning angle [rad] — positive = left.
    """
    result = controller.forward(_EMPTY_STATE, _site_setpoint(v * math.cos(theta), v * math.sin(theta)), 0.0)
    return result.joints.positions.numpy(), result.joints.velocities.numpy()


# Analytic Ackermann geometry used by several kinematic tests.
def _expected_steer_angles(theta: float, wheel_base: float = WHEEL_BASE, track_width: float = TRACK_WIDTH):
    """Return (theta_left, theta_right) for a given body turning angle."""
    s, c = math.sin(theta), math.cos(theta)
    half_tw_over_wb = 0.5 * track_width / wheel_base
    theta_l = math.atan2(s, c - half_tw_over_wb * s)
    theta_r = math.atan2(s, c + half_tw_over_wb * s)
    return theta_l, theta_r


def _expected_steerable_wheel_speeds(
    theta: float,
    v: float,
    radius: float = STEERABLE_RADIUS,
    wheel_base: float = WHEEL_BASE,
    track_width: float = TRACK_WIDTH,
):
    """Return (omega_left, omega_right) for the steerable wheels."""
    omega_body = v * math.sin(theta) / wheel_base
    v_lx = v * math.cos(theta) - 0.5 * omega_body * track_width
    v_rx = v * math.cos(theta) + 0.5 * omega_body * track_width
    theta_l, theta_r = _expected_steer_angles(theta, wheel_base, track_width)
    omega_l = v_lx / (math.cos(theta_l) * radius)
    omega_r = v_rx / (math.cos(theta_r) * radius)
    return omega_l, omega_r


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestAckermannControllerConstruction(omni.kit.test.AsyncTestCase):
    """Tests for constructor validation."""

    async def test_default_construction(self):
        """Controller constructs without error with default directions."""
        self.assertIsNotNone(_make_controller())

    async def test_no_non_steerable_wheels(self):
        """Controller with zero non-steerable wheels constructs correctly."""
        self.assertIsNotNone(_make_simple_controller())

    async def test_duplicate_steerable_wheel_joints_raises(self):
        """Using the same name for left and right steerable wheels raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_simple_controller,
            left_steerable_wheel_joint="left_steer_wheel",
            right_steerable_wheel_joint="left_steer_wheel",
        )

    async def test_duplicate_steering_joints_raises(self):
        """Using the same name for left and right steering joints raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_simple_controller,
            left_steering_joint="left_steering",
            right_steering_joint="left_steering",
        )

    async def test_parallel_directions_raise(self):
        """Parallel forward and rotation directions raise ValueError."""
        self.assertRaises(
            ValueError,
            _make_simple_controller,
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[1.0, 0.0, 0.0],
        )

    async def test_directions_too_close_raises(self):
        """Directions less than 89.9° apart raise ValueError."""
        angle = math.radians(89.0)
        self.assertRaises(
            ValueError,
            _make_simple_controller,
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[math.cos(angle), math.sin(angle), 0.0],
        )

    async def test_zero_steerable_radius_raises(self):
        self.assertRaises(ValueError, _make_simple_controller, steerable_wheel_radius=0.0)

    async def test_negative_wheel_base_raises(self):
        self.assertRaises(ValueError, _make_simple_controller, wheel_base=-1.0)

    async def test_zero_track_width_raises(self):
        self.assertRaises(ValueError, _make_simple_controller, track_width=0.0)

    async def test_negative_non_steerable_radius_raises(self):
        self.assertRaises(ValueError, _make_controller, non_steerable_wheel_radius=-0.1)

    async def test_zero_non_steerable_track_width_raises(self):
        self.assertRaises(ValueError, _make_controller, non_steerable_track_width=0.0)

    async def test_zero_max_linear_speed_raises(self):
        self.assertRaises(ValueError, _make_simple_controller, max_linear_speed=0.0)

    async def test_max_turning_angle_above_pi_over_2_raises(self):
        """max_turning_angle > π/2 raises ValueError."""
        self.assertRaises(ValueError, _make_simple_controller, max_turning_angle=math.pi)

    async def test_zero_max_turning_angle_raises(self):
        self.assertRaises(ValueError, _make_simple_controller, max_turning_angle=0.0)

    async def test_single_ns_joint_without_pair_raises(self):
        """Providing only one of the non-steerable wheel joints raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            left_non_steerable_wheel_joint="left_rear_wheel",
            right_non_steerable_wheel_joint=None,
        )

    async def test_wheel_joint_not_in_joint_space_raises(self):
        """A wheel joint name absent from robot_joint_space raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            left_steerable_wheel_joint="does_not_exist",
        )

    async def test_steering_joint_not_in_joint_space_raises(self):
        """A steering joint name absent from robot_joint_space raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            left_steering_joint="does_not_exist",
        )

    async def test_ns_joint_not_in_joint_space_raises(self):
        """A non-steerable wheel joint absent from robot_joint_space raises ValueError."""
        self.assertRaises(
            ValueError,
            _make_controller,
            left_non_steerable_wheel_joint="does_not_exist",
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestAckermannControllerReset(omni.kit.test.AsyncTestCase):

    async def test_reset_returns_true(self):
        """reset() always returns True (controller is stateless)."""
        controller = _make_controller()
        self.assertTrue(controller.reset(_EMPTY_STATE, None, 0.0))
        self.assertTrue(controller.reset(_EMPTY_STATE, _site_setpoint(1.0), 1.0))


# ---------------------------------------------------------------------------
# Forward guard conditions
# ---------------------------------------------------------------------------


class TestAckermannControllerForwardGuards(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        self.controller = _make_controller()

    async def test_returns_none_for_none_setpoint(self):
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, None, 0.0))

    async def test_returns_none_for_missing_sites(self):
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, mg.RobotState(), 0.0))

    async def test_returns_none_for_missing_linear_velocity(self):
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                angular_velocities=(["control_point"], wp.array([[0.0, 0.0, 1.0]], dtype=wp.float32)),
            )
        )
        self.assertIsNone(self.controller.forward(_EMPTY_STATE, setpoint, 0.0))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestAckermannControllerOutputStructure(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        self.controller = _make_controller()

    async def test_output_velocity_joint_names(self):
        """Velocity targets are keyed to wheel joints in the declared order."""
        result = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0), 0.0)
        self.assertEqual(
            result.joints.velocity_names,
            ["left_steer_wheel", "right_steer_wheel", "left_rear_wheel", "right_rear_wheel"],
        )

    async def test_output_position_joint_names(self):
        """Position targets are keyed to steering joints in the declared order."""
        result = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0), 0.0)
        self.assertEqual(result.joints.position_names, ["left_steering", "right_steering"])

    async def test_velocity_indices_are_full_joint_space_positions(self):
        """velocity_indices must be the wheel joints' positions in the full JOINT_SPACE.

        JOINT_SPACE layout:
          0=joint_a, 1=left_steer_wheel, 2=joint_b, 3=right_steer_wheel,
          4=left_steering, 5=left_rear_wheel, 6=right_steering, 7=right_rear_wheel

        Returning [0, 1, 2, 3] (indices within the velocity-only sub-space) would be wrong.
        """
        result = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0), 0.0)
        self.assertEqual(result.joints.velocity_indices.numpy().tolist(), [1, 3, 5, 7])

    async def test_position_indices_are_full_joint_space_positions(self):
        """position_indices must be the steering joints' positions in the full JOINT_SPACE."""
        result = self.controller.forward(_EMPTY_STATE, _site_setpoint(1.0), 0.0)
        self.assertEqual(result.joints.position_indices.numpy().tolist(), [4, 6])


# ---------------------------------------------------------------------------
# Velocity projection
# ---------------------------------------------------------------------------


class TestAckermannControllerProjection(omni.kit.test.AsyncTestCase):

    async def test_lateral_component_determines_turning_angle(self):
        """A positive lateral (leftward) component produces a positive (left) steer angle."""
        controller = _make_simple_controller()
        steer_zero, _ = _run(controller, 1.0, 0.0)  # theta=0 → straight
        steer_left, _ = _run(controller, 1.0, 0.3)  # theta>0 → left turn
        steer_right, _ = _run(controller, 1.0, -0.3)  # theta<0 → right turn
        self.assertTrue(np.allclose(steer_zero, [0.0, 0.0], atol=1e-5))
        self.assertTrue(np.all(steer_left > 0))  # left turn: both steer angles positive
        self.assertTrue(np.all(steer_right < 0))  # right turn: both steer angles negative

    async def test_custom_forward_direction(self):
        """Controller correctly projects along a non-default forward axis."""
        # Forward = +Y; setpoint has linear_velocity along Y.
        controller = _make_simple_controller(
            forward_direction=[0.0, 1.0, 0.0],
            rotation_direction=[0.0, 0.0, 1.0],
        )
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[0.0, 1.0, 0.0]], dtype=wp.float32)),
            )
        )
        result = controller.forward(_EMPTY_STATE, setpoint, 0.0)
        steer = result.joints.positions.numpy()
        speeds = result.joints.velocities.numpy()
        self.assertTrue(np.allclose(steer, [0.0, 0.0], atol=1e-5))
        expected = 1.0 / STEERABLE_RADIUS
        self.assertTrue(np.allclose(speeds, [expected, expected], atol=1e-5))

    async def test_custom_rotation_direction(self):
        """Controller correctly projects turning angle with a non-default yaw axis."""
        # rotation_direction = +Y (yaw around Y instead of Z).
        # lateral = cross([0,1,0], [1,0,0]) = [0,0,-1], so leftward is -Z.
        controller = _make_simple_controller(
            forward_direction=[1.0, 0.0, 0.0],
            rotation_direction=[0.0, 1.0, 0.0],
        )
        # Straight ahead along X → zero steer, correct wheel speed.
        straight = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32)),
            )
        )
        result = controller.forward(_EMPTY_STATE, straight, 0.0)
        self.assertTrue(np.allclose(result.joints.positions.numpy(), [0.0, 0.0], atol=1e-5))
        self.assertTrue(np.allclose(result.joints.velocities.numpy(), [1.0 / STEERABLE_RADIUS] * 2, atol=1e-5))

        # Left turn: velocity along [X, Z] with a -Z component (lateral = [0,0,-1]).
        # [1, 0, -1] → vfwd=1, vlat=1 → 45° turning angle.
        left_turn = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(
                    ["control_point"],
                    wp.array([[1.0, 0.0, -1.0]], dtype=wp.float32),
                ),
            )
        )
        result_left = controller.forward(_EMPTY_STATE, left_turn, 0.0)
        steer = result_left.joints.positions.numpy()
        self.assertTrue(np.all(steer > 0.1), f"expected non-trivial positive steer angles for left turn, got {steer}")


# ---------------------------------------------------------------------------
# Kinematic correctness (adapted from newton-controllers test_ackermann.py)
# ---------------------------------------------------------------------------


class TestAckermannControllerKinematics(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        self.controller = _make_simple_controller()

    async def test_straight_ahead_zero_steer_angles(self):
        """theta=0 → both steering angles are zero."""
        steer, _ = _run(self.controller, 1.0)
        self.assertTrue(np.allclose(steer, [0.0, 0.0], atol=1e-6))

    async def test_straight_ahead_wheel_speed(self):
        """theta=0, v=1 → both steerable wheel speeds = v / steerable_wheel_radius."""
        _, speeds = _run(self.controller, 1.0)
        expected = 1.0 / STEERABLE_RADIUS
        self.assertTrue(np.allclose(speeds, [expected, expected], atol=1e-5))

    async def test_zero_speed_gives_zero_wheel_speeds(self):
        """v=0 → all wheel speeds are zero regardless of steer angle."""
        _, speeds = _run(self.controller, 0.0, 0.3)
        self.assertTrue(np.allclose(speeds, [0.0, 0.0], atol=1e-6))

    async def test_outer_wheel_steers_less_than_inner(self):
        """Left turn: the left (inner) wheel steers more sharply than the right (outer)."""
        theta = 0.3
        steer, _ = _run(self.controller, 1.0, theta)
        left_angle, right_angle = steer
        self.assertGreater(abs(left_angle), abs(right_angle))

    async def test_symmetric_steer_angles_left_right(self):
        """Left and right turns by the same angle give negated, swapped steer pairs.

        For a left turn the left wheel is inner (steeper); for a right turn it is the
        reverse — so the steer-angle vector is negated and reversed.
        """
        theta = 0.3
        steer_left, _ = _run(self.controller, 1.0, theta)
        steer_right, _ = _run(self.controller, 1.0, -theta)
        self.assertTrue(np.allclose(steer_left, -steer_right[::-1], atol=1e-6))

    async def test_max_linear_speed_clamps_input(self):
        """Speed above max_linear_speed is clamped to the limit."""
        max_speed = 2.0
        controller = _make_simple_controller(max_linear_speed=max_speed)
        _, speeds = _run(controller, 10.0, 0.0)
        expected = max_speed / STEERABLE_RADIUS
        self.assertTrue(np.allclose(speeds, [expected, expected], atol=1e-5))

    async def test_max_turning_angle_clamps_input(self):
        """Turning angle above the limit is clamped; below limit passes through."""
        max_angle = 0.3
        controller = _make_simple_controller(max_turning_angle=max_angle)
        # Request much larger than limit.
        steer_clamped, _ = _run(controller, 1.0, 1.0)
        # Request exactly at limit.
        steer_at_limit, _ = _run(controller, 1.0, max_angle)
        self.assertTrue(np.allclose(steer_clamped, steer_at_limit, atol=1e-6))

    async def test_steer_angles_match_analytic_ackermann_geometry(self):
        """Steer angles match the analytic Ackermann formula for a known turning angle."""
        theta = math.radians(20.0)
        steer, _ = _run(self.controller, 1.0, theta)
        expected_l, expected_r = _expected_steer_angles(theta)
        self.assertAlmostEqual(float(steer[0]), expected_l, places=5)
        self.assertAlmostEqual(float(steer[1]), expected_r, places=5)

    async def test_wheel_speeds_match_analytic_ackermann_geometry(self):
        """Steerable wheel speeds match the analytic Ackermann formula."""
        theta = math.radians(20.0)
        v = 2.0
        _, speeds = _run(self.controller, v, theta)
        expected_l, expected_r = _expected_steerable_wheel_speeds(theta, v)
        self.assertAlmostEqual(float(speeds[0]), expected_l, places=4)
        self.assertAlmostEqual(float(speeds[1]), expected_r, places=4)

    async def test_rear_steering_inverts_steer_angles(self):
        """steerable_wheels_at_rear=True negates the steer angles (forklift mode)."""
        front_ctrl = _make_simple_controller(steerable_wheels_at_rear=False)
        rear_ctrl = _make_simple_controller(steerable_wheels_at_rear=True)
        theta = 0.25
        steer_front, _ = _run(front_ctrl, 1.0, theta)
        steer_rear, _ = _run(rear_ctrl, 1.0, theta)
        self.assertTrue(np.allclose(steer_front, -steer_rear, atol=1e-6))

    async def test_reverse_straight_gives_zero_steer_angle(self):
        """Reversing straight (v<0, v_lat=0) must produce theta=0, not ±π.

        Without |v| in the atan2 denominator, atan2(0, -v) = ±π which clamps
        to max_turning_angle and drives the wheels to full lock while reversing.
        """
        steer_fwd, _ = _run(self.controller, 1.0, 0.0)
        steer_rev, _ = _run(self.controller, -1.0, 0.0)
        self.assertTrue(np.allclose(steer_fwd, [0.0, 0.0], atol=1e-5))
        self.assertTrue(np.allclose(steer_rev, [0.0, 0.0], atol=1e-5))

    async def test_reverse_keeps_same_steer_angles_as_forward(self):
        """Negating speed must not change the steer angles.

        The turning angle is recovered from the direction of the velocity vector,
        which is the same for ``[v·cos θ, v·sin θ]`` and ``[-v·cos θ, -v·sin θ]``.
        """
        theta = 0.3
        steer_fwd, _ = _run(self.controller, 1.0, theta)
        steer_rev, _ = _run(self.controller, -1.0, theta)
        self.assertTrue(np.allclose(steer_fwd, steer_rev, atol=1e-6))

    async def test_reverse_with_turn_reverses_wheel_speed_sign(self):
        """Reversing with a turn produces negative wheel speeds."""
        theta = 0.3
        _, speeds_fwd = _run(self.controller, 1.0, theta)
        _, speeds_rev = _run(self.controller, -1.0, theta)
        self.assertTrue(np.all(speeds_fwd > 0))
        self.assertTrue(np.all(speeds_rev < 0))

    async def test_zero_forward_speed_holds_previous_theta(self):
        """When speed drops to zero the previous theta is held, not snapped to zero.

        A stationary command ``[0, 0, 0]`` must leave steering angles unchanged
        from the preceding non-zero-speed step.
        """
        theta = 0.3
        # Prime prev_theta with a real turn.
        steer_before, _ = _run(self.controller, 1.0, theta)
        # Now command zero speed — stateful kernel should hold theta.
        zero_setpoint = _site_setpoint(0.0, 0.0)
        result = self.controller.forward(_EMPTY_STATE, zero_setpoint, 0.0)
        steer_after = result.joints.positions.numpy()
        self.assertTrue(np.allclose(steer_before, steer_after, atol=1e-5))

    async def test_vx_zero_does_not_produce_nan_or_wrong_sign(self):
        """A pure-lateral velocity (v_x=0) must not produce NaN or a negative speed.

        ``v_x = 0`` is the Ackermann singularity boundary; the stateless path
        previously fell into the reversing branch and negated ``v_y``, returning
        ``atan2(-v_y, 0) = -π/2`` and a negative speed for a forward command.
        With ``v_x >= 0.0`` guards, the output must be finite and non-negative.
        """
        setpoint = _site_setpoint(0.0, 1.0)  # v_x=0, v_y=1 → on the singularity boundary
        result = self.controller.forward(_EMPTY_STATE, setpoint, 0.0)
        steer = result.joints.positions.numpy()
        speeds = result.joints.velocities.numpy()
        self.assertFalse(np.any(np.isnan(steer)), "NaN in steering angles")
        self.assertFalse(np.any(np.isnan(speeds)), "NaN in wheel speeds")
        self.assertTrue(np.all(speeds >= 0.0), "Negative wheel speed for v_x=0, v_y>0")


# ---------------------------------------------------------------------------
# Non-steerable wheel kinematics
# ---------------------------------------------------------------------------


class TestAckermannControllerNonSteerableWheels(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        self.controller = _make_controller()  # includes left_rear and right_rear wheels

    async def test_straight_all_wheels_spin_at_v_over_r(self):
        """theta=0: all four wheels (steerable + non-steerable) spin at v/r."""
        _, speeds = _run(self.controller, 1.0, 0.0)
        expected = 1.0 / STEERABLE_RADIUS
        self.assertTrue(np.allclose(speeds, [expected, expected, expected, expected], atol=1e-5))

    async def test_ns_outer_wheel_faster_than_inner_in_turn(self):
        """Left turn: the right (outer) rear wheel spins faster than the left (inner)."""
        theta = 0.3
        _, speeds = _run(self.controller, 1.0, theta)
        left_ns_speed = speeds[2]  # left_rear_wheel (offset +TRACK_WIDTH/2 = inner in left turn)
        right_ns_speed = speeds[3]  # right_rear_wheel (offset -TRACK_WIDTH/2 = outer in left turn)
        self.assertLess(left_ns_speed, right_ns_speed)

    async def test_different_steerable_and_ns_radii(self):
        """At theta=0, steerable and non-steerable wheels spin at their respective v/r values."""
        ns_radius = 0.2  # intentionally different from STEERABLE_RADIUS
        controller = _make_controller(non_steerable_wheel_radius=ns_radius)
        _, speeds = _run(controller, 1.0, 0.0)
        expected_steerable = 1.0 / STEERABLE_RADIUS
        expected_ns = 1.0 / ns_radius
        self.assertAlmostEqual(float(speeds[0]), expected_steerable, places=5)
        self.assertAlmostEqual(float(speeds[1]), expected_steerable, places=5)
        self.assertAlmostEqual(float(speeds[2]), expected_ns, places=5)
        self.assertAlmostEqual(float(speeds[3]), expected_ns, places=5)
        self.assertNotAlmostEqual(float(speeds[0]), float(speeds[2]), places=3)

    async def test_non_steerable_track_width_changes_rear_differential(self):
        """A wider rear track produces a larger speed difference between rear wheels in a turn."""
        theta = 0.3
        v = 1.0
        controller_narrow = _make_controller(non_steerable_track_width=TRACK_WIDTH)
        controller_wide = _make_controller(non_steerable_track_width=TRACK_WIDTH * 2.0)
        _, speeds_narrow = _run(controller_narrow, v, theta)
        _, speeds_wide = _run(controller_wide, v, theta)
        diff_narrow = abs(speeds_narrow[3] - speeds_narrow[2])
        diff_wide = abs(speeds_wide[3] - speeds_wide[2])
        self.assertGreater(diff_wide, diff_narrow)


# ---------------------------------------------------------------------------
# Direct command equivalence
# ---------------------------------------------------------------------------


class TestAckermannControllerDirectCommand(omni.kit.test.AsyncTestCase):

    async def test_direct_command_equivalent_to_site_setpoint(self):
        """direct_command=True with (linear_speed, turning_angle) kwargs produces
        identical outputs to site-setpoint mode for the same (v, θ) — excluding the
        degenerate v=0 case where the site velocity vector carries no direction.
        """
        site_ctrl = _make_simple_controller()
        direct_ctrl = _make_simple_controller(direct_command=True)

        for v, theta in [(1.0, 0.0), (2.0, 0.3), (0.5, -0.25), (-1.0, 0.2), (1.5, math.radians(20))]:
            site_result = site_ctrl.forward(
                _EMPTY_STATE,
                _site_setpoint(v * math.cos(theta), v * math.sin(theta)),
                0.0,
            )
            direct_result = direct_ctrl.forward(
                _EMPTY_STATE,
                None,
                0.0,
                linear_speed=v,
                turning_angle=theta,
            )
            self.assertTrue(
                np.allclose(site_result.joints.positions.numpy(), direct_result.joints.positions.numpy(), atol=1e-5),
                msg=f"Steering angles differ for v={v}, theta={theta}",
            )
            self.assertTrue(
                np.allclose(site_result.joints.velocities.numpy(), direct_result.joints.velocities.numpy(), atol=1e-5),
                msg=f"Wheel speeds differ for v={v}, theta={theta}",
            )

    async def test_direct_command_missing_kwargs_returns_none(self):
        """direct_command=True returns None when linear_speed or turning_angle is absent."""
        ctrl = _make_simple_controller(direct_command=True)
        self.assertIsNone(ctrl.forward(_EMPTY_STATE, None, 0.0))
        self.assertIsNone(ctrl.forward(_EMPTY_STATE, None, 0.0, linear_speed=1.0))
        self.assertIsNone(ctrl.forward(_EMPTY_STATE, None, 0.0, turning_angle=0.3))
