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

"""Analytic (no-sim) tests for HolonomicController.

These tests verify that the warp-based controller correctly implements the
closed-form inverse kinematics ``phi_dot = K * (M @ [vx, vy, wz])``, that
speed clamping works, and that joint name mapping is correct.  Physics
plausibility is verified separately in ``test_holonomic_controller_sim.py``.
"""

import isaacsim.robot_motion.controllers as ctrl
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.kit.test
import warp as wp

# ── Shared fixture: 4-wheel mecanum robot ─────────────────────────────────────
#
# Wheels arranged in a rectangle (±L/2, ±W/2).  Axles are all along +x.
# Mecanum angles follow the standard X pattern: FL/RR = 45°, FR/RL = 135°.

_L, _W = 0.3, 0.2
_R = 0.05

_JOINT_SPACE = ["unrelated_a", "wheel_fl", "unrelated_b", "wheel_fr", "wheel_rl", "wheel_rr", "unrelated_c"]
_WHEEL_JOINTS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]

_WHEEL_POSITIONS = [
    [_L / 2, _W / 2, 0.0],
    [_L / 2, -_W / 2, 0.0],
    [-_L / 2, _W / 2, 0.0],
    [-_L / 2, -_W / 2, 0.0],
]
# All wheels have identity orientation (axle along local +x = world +x).
_WHEEL_ORIENTATIONS = [[1.0, 0.0, 0.0, 0.0]] * 4
_MECANUM_ANGLES = [45.0, 135.0, 135.0, 45.0]


def _make_controller(**kwargs) -> ctrl.HolonomicController:
    defaults = dict(
        robot_joint_space=_JOINT_SPACE,
        wheel_joint_names=_WHEEL_JOINTS,
        wheel_radius=_R,
        wheel_positions=_WHEEL_POSITIONS,
        wheel_orientations=_WHEEL_ORIENTATIONS,
        mecanum_angles=_MECANUM_ANGLES,
    )
    defaults.update(kwargs)
    return ctrl.HolonomicController(**defaults)


def _site_setpoint(vx: float, vy: float, wz: float, site: str = "control_point") -> mg.RobotState:
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=[site],
            linear_velocities=([site], wp.array([[vx, vy, 0.0]], dtype=wp.float32)),
            angular_velocities=([site], wp.array([[0.0, 0.0, wz]], dtype=wp.float32)),
        )
    )


_EMPTY_STATE = mg.RobotState()


def _command(controller: ctrl.HolonomicController, vx: float, vy: float, wz: float) -> dict[str, float]:
    """Return {joint_name: velocity} for the given command-site twist."""
    result = controller.forward(_EMPTY_STATE, _site_setpoint(vx, vy, wz), 0.0)
    assert result is not None, "controller.forward() returned None"
    return dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))


def _expected(controller: ctrl.HolonomicController, vx: float, vy: float, wz: float) -> dict[str, float]:
    """Analytical expected output using the precomputed M, K matrices."""
    twist = np.array([vx, vy, wz], dtype=np.float64)
    phi_dot = controller._k * (controller._m @ twist)
    return dict(zip(_WHEEL_JOINTS, phi_dot))


class TestHolonomicController(omni.kit.test.AsyncTestCase):
    """Analytic unit tests for HolonomicController."""

    async def setUp(self) -> None:
        self._ctrl = _make_controller()

    async def tearDown(self) -> None:
        pass

    # ── Kinematic matrix verification (hand-derived) ─────────────────────────
    #
    # All four wheels have identity orientation → axle = [1,0,0].
    # axle × up = [1,0,0] × [0,0,1] = [0,-1,0], so rolling_dir = [0,-1,0], cos_α = 1.
    #
    # Mecanum 45°  → γ = -45°, cos γ = s = 1/√2
    #   a = R_z(-45°) @ [0,-1,0] = [-s, -s, 0]
    # Mecanum 135° → γ = +45°, cos γ = s = 1/√2
    #   a = R_z(+45°) @ [0,-1,0] = [+s, -s, 0]
    #
    # M[i] = [a[0], a[1], a[1]*px - a[0]*py]
    # K[i] = 1 / (r * cos γ) = 1 / (_R * s) = √2 / _R
    #
    #   FL (45°,  [ L/2,  W/2]):  [-s, -s, -s*W/2 - (-s)*L/2] = [-s, -s, s(L-W)/2] ... hmm
    #
    # Third column: a[1]*px - a[0]*py
    #   FL: (-s)*(L/2) - (-s)*(W/2)  = s(-L/2 + W/2) = -s*(L-W)/2 = -0.05s
    #   FR: (-s)*(L/2) - (+s)*(-W/2) = -sL/2 + sW/2   = -s*(L-W)/2 = -0.05s
    #   RL: (-s)*(-L/2) - (+s)*(W/2) = +sL/2 - sW/2   = +s*(L-W)/2 = +0.05s
    #   RR: (-s)*(-L/2) - (-s)*(-W/2)= +sL/2 - sW/2   = +s*(L-W)/2 = +0.05s
    #   (L=0.3, W=0.2 → (L-W)/2 = 0.05)

    async def test_kinematics_matrices_match_hand_derivation(self) -> None:
        """M and K match values derived independently from the wheel geometry."""
        s = 2**-0.5  # 1/√2
        expected_m = np.array(
            [
                [-s, -s, -0.05 * s],  # FL  mecanum 45°
                [+s, -s, -0.05 * s],  # FR  mecanum 135°
                [+s, -s, +0.05 * s],  # RL  mecanum 135°
                [-s, -s, +0.05 * s],  # RR  mecanum 45°
            ]
        )
        expected_k = np.full(4, 2**0.5 / _R)  # √2 / r
        np.testing.assert_allclose(self._ctrl._m, expected_m, atol=1e-10)
        np.testing.assert_allclose(self._ctrl._k, expected_k, atol=1e-10)

    # ── Basic kinematics ──────────────────────────────────────────────────────

    async def test_zero_command_produces_zero_output(self) -> None:
        """Zero twist → all wheel speeds are zero."""
        got = _command(self._ctrl, 0.0, 0.0, 0.0)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(got[name], 0.0, places=5, msg=f"{name} not zero")

    async def test_forward_command_matches_analytical(self) -> None:
        """Forward twist output matches K * (M @ [vx, 0, 0])."""
        vx = 0.5
        got = _command(self._ctrl, vx, 0.0, 0.0)
        exp = _expected(self._ctrl, vx, 0.0, 0.0)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got[name], exp[name], places=4, msg=f"{name}: got {got[name]:.4f}, expected {exp[name]:.4f}"
            )

    async def test_lateral_command_matches_analytical(self) -> None:
        """Lateral twist output matches K * (M @ [0, vy, 0])."""
        vy = 0.4
        got = _command(self._ctrl, 0.0, vy, 0.0)
        exp = _expected(self._ctrl, 0.0, vy, 0.0)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got[name], exp[name], places=4, msg=f"{name}: got {got[name]:.4f}, expected {exp[name]:.4f}"
            )

    async def test_yaw_command_matches_analytical(self) -> None:
        """Pure yaw twist output matches K * (M @ [0, 0, wz])."""
        wz = 1.0
        got = _command(self._ctrl, 0.0, 0.0, wz)
        exp = _expected(self._ctrl, 0.0, 0.0, wz)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got[name], exp[name], places=4, msg=f"{name}: got {got[name]:.4f}, expected {exp[name]:.4f}"
            )

    async def test_combined_command_matches_analytical(self) -> None:
        """Combined forward + lateral + yaw matches analytical result."""
        got = _command(self._ctrl, 0.3, 0.2, 0.5)
        exp = _expected(self._ctrl, 0.3, 0.2, 0.5)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got[name], exp[name], places=4, msg=f"{name}: got {got[name]:.4f}, expected {exp[name]:.4f}"
            )

    # ── Speed clamping ────────────────────────────────────────────────────────

    async def test_max_linear_speed_clamps_output(self) -> None:
        """Commanding vx > max_linear_speed clamps the input uniformly."""
        limit = 0.1
        ctrl_clamped = _make_controller(max_linear_speed=limit)
        vx_requested = 1.0

        got_clamped = _command(ctrl_clamped, vx_requested, 0.0, 0.0)
        # The clamped controller should produce the same output as commanding exactly the limit.
        got_at_limit = _command(self._ctrl, limit, 0.0, 0.0)

        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got_clamped[name],
                got_at_limit[name],
                places=4,
                msg=f"{name}: clamped={got_clamped[name]:.4f}, at-limit={got_at_limit[name]:.4f}",
            )

    async def test_max_angular_speed_clamps_output(self) -> None:
        """Commanding wz > max_angular_speed clamps the yaw rate."""
        limit = 0.2
        ctrl_clamped = _make_controller(max_angular_speed=limit)
        wz_requested = 5.0

        got_clamped = _command(ctrl_clamped, 0.0, 0.0, wz_requested)
        got_at_limit = _command(self._ctrl, 0.0, 0.0, limit)

        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got_clamped[name],
                got_at_limit[name],
                places=4,
                msg=f"{name}: clamped={got_clamped[name]:.4f}, at-limit={got_at_limit[name]:.4f}",
            )

    async def test_max_wheel_speed_clamps_output(self) -> None:
        """Commanding large twist clamps per-wheel speed to max_wheel_speed."""
        limit = 1.0
        ctrl_clamped = _make_controller(max_wheel_speed=limit)
        got = _command(ctrl_clamped, 100.0, 0.0, 0.0)
        for name in _WHEEL_JOINTS:
            self.assertLessEqual(abs(got[name]), limit + 1e-5, msg=f"{name} exceeds max_wheel_speed")

    async def test_linear_speed_preserves_direction(self) -> None:
        """Uniform linear scale preserves the vx/vy ratio."""
        limit = 0.1
        ctrl_clamped = _make_controller(max_linear_speed=limit)
        vx, vy = 3.0, 4.0  # magnitude = 5, 3/4 aspect

        got_clamped = _command(ctrl_clamped, vx, vy, 0.0)
        # At the limit, scale = 0.1/5 = 0.02, so effective (vx, vy) = (0.06, 0.08).
        got_at_limit = _command(self._ctrl, vx * limit / 5.0, vy * limit / 5.0, 0.0)

        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(
                got_clamped[name],
                got_at_limit[name],
                places=4,
                msg=f"{name}: direction not preserved after linear clamp",
            )

    # ── Setpoint guard ────────────────────────────────────────────────────────

    async def test_none_setpoint_returns_none(self) -> None:
        """forward() with setpoint_state=None returns None."""
        self.assertIsNone(self._ctrl.forward(_EMPTY_STATE, None, 0.0))

    async def test_missing_sites_returns_none(self) -> None:
        """forward() with a setpoint that has no sites returns None."""
        self.assertIsNone(self._ctrl.forward(_EMPTY_STATE, _EMPTY_STATE, 0.0))

    async def test_wrong_site_name_returns_none(self) -> None:
        """forward() with a site under the wrong name returns None."""
        setpoint = _site_setpoint(0.1, 0.0, 0.0, site="wrong_name")
        self.assertIsNone(self._ctrl.forward(_EMPTY_STATE, setpoint, 0.0))

    # ── Joint name mapping ────────────────────────────────────────────────────

    async def test_velocity_names_match_wheel_joints(self) -> None:
        """The returned velocity names are exactly the wheel joint names."""
        result = self._ctrl.forward(_EMPTY_STATE, _site_setpoint(0.1, 0.0, 0.0), 0.0)
        self.assertIsNotNone(result)
        self.assertEqual(sorted(result.joints.velocity_names), sorted(_WHEEL_JOINTS))

    async def test_unrelated_joints_not_in_output(self) -> None:
        """Joints outside wheel_joint_names do not appear in the velocity output."""
        result = self._ctrl.forward(_EMPTY_STATE, _site_setpoint(0.1, 0.0, 0.0), 0.0)
        self.assertIsNotNone(result)
        for name in result.joints.velocity_names:
            self.assertIn(name, _WHEEL_JOINTS, f"unexpected joint '{name}' in output")

    # ── Reset ─────────────────────────────────────────────────────────────────

    async def test_reset_returns_true(self) -> None:
        """reset() always returns True (stateless controller)."""
        self.assertTrue(self._ctrl.reset(None, None, 0.0))

    # ── Construction error handling ───────────────────────────────────────────

    async def test_missing_wheel_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            ctrl.HolonomicController(
                robot_joint_space=_JOINT_SPACE,
                wheel_joint_names=_WHEEL_JOINTS,
                wheel_radius=None,
                wheel_positions=_WHEEL_POSITIONS,
                wheel_orientations=_WHEEL_ORIENTATIONS,
            )

    async def test_duplicate_wheel_joint_names_raises(self) -> None:
        with self.assertRaises(ValueError):
            ctrl.HolonomicController(
                robot_joint_space=_JOINT_SPACE,
                wheel_joint_names=["wheel_fl", "wheel_fl", "wheel_rl", "wheel_rr"],
                wheel_radius=_R,
                wheel_positions=_WHEEL_POSITIONS,
                wheel_orientations=_WHEEL_ORIENTATIONS,
            )

    async def test_joint_not_in_space_raises(self) -> None:
        with self.assertRaises(ValueError):
            ctrl.HolonomicController(
                robot_joint_space=_JOINT_SPACE,
                wheel_joint_names=["wheel_fl", "wheel_fr", "wheel_rl", "ghost_joint"],
                wheel_radius=_R,
                wheel_positions=_WHEEL_POSITIONS,
                wheel_orientations=_WHEEL_ORIENTATIONS,
            )

    async def test_zero_wheel_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(wheel_radius=[_R, 0.0, _R, _R])

    async def test_negative_wheel_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(wheel_radius=-_R)

    async def test_wheel_orientations_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(wheel_orientations=_WHEEL_ORIENTATIONS[:2])  # only 2 for 4 wheels

    async def test_mecanum_angles_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(mecanum_angles=[45.0, 135.0])  # only 2 for 4 wheels

    async def test_zero_rotation_direction_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(rotation_direction=[0.0, 0.0, 0.0])

    async def test_zero_wheel_axis_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(wheel_axis=[0.0, 0.0, 0.0])

    async def test_wheel_joint_count_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(wheel_joint_names=["wheel_fl", "wheel_fr"])  # 2 names, 4 wheels

    async def test_axle_parallel_to_up_raises(self) -> None:
        # 90° about Y maps wheel_axis [1,0,0] → [0,0,-1] ∥ rotation_direction [0,0,1].
        q_90y = [2**0.5 / 2, 0.0, 2**0.5 / 2, 0.0]
        with self.assertRaises(ValueError):
            _make_controller(wheel_orientations=[q_90y] + list(_WHEEL_ORIENTATIONS[1:]))

    async def test_mecanum_angle_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(mecanum_angles=[0.0, 135.0, 135.0, 45.0])

    async def test_mecanum_angle_180_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(mecanum_angles=[180.0, 135.0, 135.0, 45.0])

    async def test_negative_max_linear_speed_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(max_linear_speed=-1.0)

    async def test_zero_max_angular_speed_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_controller(max_angular_speed=0.0)

    # ── Scalar broadcast ──────────────────────────────────────────────────────

    async def test_wheel_radius_scalar_broadcast(self) -> None:
        """Scalar wheel_radius gives the same result as an explicit per-wheel array."""
        ctrl_array = _make_controller(wheel_radius=[_R, _R, _R, _R])
        got_s = _command(self._ctrl, 0.3, 0.1, 0.2)
        got_a = _command(ctrl_array, 0.3, 0.1, 0.2)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(got_s[name], got_a[name], places=5)

    async def test_mecanum_angles_scalar_broadcast(self) -> None:
        """Scalar mecanum_angles gives the same result as an explicit per-wheel array."""
        ctrl_scalar = _make_controller(mecanum_angles=90.0)
        ctrl_array = _make_controller(mecanum_angles=[90.0, 90.0, 90.0, 90.0])
        got_s = _command(ctrl_scalar, 0.3, 0.1, 0.2)
        got_a = _command(ctrl_array, 0.3, 0.1, 0.2)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(got_s[name], got_a[name], places=5)

    async def test_mecanum_angles_none_defaults_to_omni(self) -> None:
        """mecanum_angles=None defaults to 90° (omni) for all wheels."""
        ctrl_default = _make_controller(mecanum_angles=None)
        ctrl_explicit = _make_controller(mecanum_angles=[90.0, 90.0, 90.0, 90.0])
        got_d = _command(ctrl_default, 0.3, 0.1, 0.2)
        got_e = _command(ctrl_explicit, 0.3, 0.1, 0.2)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(got_d[name], got_e[name], places=5)

    # ── Custom control-point name ─────────────────────────────────────────────

    async def test_custom_control_point_name_is_read(self) -> None:
        """Controller with a custom site name reads from that site."""
        my_ctrl = _make_controller(control_point_name="my_site")
        result = my_ctrl.forward(_EMPTY_STATE, _site_setpoint(0.5, 0.0, 0.0, site="my_site"), 0.0)
        self.assertIsNotNone(result)

    async def test_custom_control_point_name_ignores_default_site(self) -> None:
        """Controller with a custom site name ignores 'control_point'."""
        my_ctrl = _make_controller(control_point_name="my_site")
        self.assertIsNone(my_ctrl.forward(_EMPTY_STATE, _site_setpoint(0.5, 0.0, 0.0), 0.0))

    # ── Negative commands ─────────────────────────────────────────────────────

    async def test_negative_commands_match_analytical(self) -> None:
        """Negative vx/vy/wz output matches the analytical formula."""
        got = _command(self._ctrl, -0.3, -0.2, -0.5)
        exp = _expected(self._ctrl, -0.3, -0.2, -0.5)
        for name in _WHEEL_JOINTS:
            self.assertAlmostEqual(got[name], exp[name], places=4)


# ── 3-wheel omni (Kaya-style) ─────────────────────────────────────────────────
#
# Three wheels equally spaced at 120° around a circle of radius 0.1 m.
# Each wheel axle is tangential (rotated 0°/120°/240° about Z).

_D3 = 0.1
_S3 = 3**0.5 / 2  # sin/cos helper

_THREE_WHEEL_SPACE = ["omni_0", "omni_1", "omni_2"]
_THREE_WHEEL_POSITIONS = [
    [_D3, 0.0, 0.0],
    [-_D3 / 2, _D3 * _S3, 0.0],
    [-_D3 / 2, -_D3 * _S3, 0.0],
]
_THREE_WHEEL_ORIENTATIONS = [
    [1.0, 0.0, 0.0, 0.0],  # 0°
    [0.5, 0.0, 0.0, _S3],  # 120°
    [-0.5, 0.0, 0.0, _S3],  # 240°
]


class TestHolonomicControllerOmni(omni.kit.test.AsyncTestCase):
    """Tests using a 3-wheel omni configuration."""

    async def setUp(self) -> None:
        self._ctrl = ctrl.HolonomicController(
            robot_joint_space=_THREE_WHEEL_SPACE,
            wheel_joint_names=_THREE_WHEEL_SPACE,
            wheel_radius=0.04,
            wheel_positions=_THREE_WHEEL_POSITIONS,
            wheel_orientations=_THREE_WHEEL_ORIENTATIONS,
        )

    async def tearDown(self) -> None:
        pass

    async def test_three_wheel_omni_forward_returns_three_velocities(self) -> None:
        """3-wheel omni controller returns velocity targets for all three wheels."""
        result = self._ctrl.forward(_EMPTY_STATE, _site_setpoint(0.4, 0.0, 0.0), 0.0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.joints.velocity_names), 3)

    async def test_three_wheel_omni_matches_analytical(self) -> None:
        """3-wheel omni output matches K * (M @ twist)."""
        result = self._ctrl.forward(_EMPTY_STATE, _site_setpoint(0.3, 0.1, 0.2), 0.0)
        got = dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))
        exp = dict(zip(_THREE_WHEEL_SPACE, self._ctrl._k * (self._ctrl._m @ np.array([0.3, 0.1, 0.2]))))
        for name in _THREE_WHEEL_SPACE:
            self.assertAlmostEqual(got[name], exp[name], places=4)


# ── Independent kinematic verification ───────────────────────────────────────
#
# Three omni wheels (r=1, mecanum_angle=90°) with hand-derivable geometry.
# Expected wheel speeds come from first principles, NOT the controller's M/K.
#
# rolling_dir = axle × up;  φ̇_i = rolling_dir_i · (v_c + ω × r_i)  (r=1)
#
# Command: vx=4, vy=5, wz=-5
#
# W1: pos=[0,1,0]   axle=rot_z(90°)@x=[0,1,0]     rolling=[1,0,0]
#   ω×r = [0,0,-5]×[0,1,0]  = [5,0,0]
#   v_c = [9,5,0]  →  φ̇ = 9
#
# W2: pos=[-1,-1,0] axle=rot_z(225°)@x=[-√½,-√½,0] rolling=[-√½,+√½,0]
#   ω×r = [0,0,-5]×[-1,-1,0] = [-5,5,0]
#   v_c = [-1,10,0] →  φ̇ = √½ + 10√½ = 11/√2
#
# W3: pos=[1,-1,0]  axle=rot_z(-45°)@x=[+√½,-√½,0] rolling=[-√½,-√½,0]
#   ω×r = [0,0,-5]×[1,-1,0]  = [-5,-5,0]
#   v_c = [-1,0,0]  →  φ̇ = √½ = 1/√2


def _rot_z(deg: float) -> list:
    """Unit quaternion [w,x,y,z] for a rotation of ``deg`` degrees about Z."""
    a = np.radians(deg) / 2.0
    return [float(np.cos(a)), 0.0, 0.0, float(np.sin(a))]


def _rot_y(deg: float) -> list:
    """Unit quaternion [w,x,y,z] for a rotation of ``deg`` degrees about Y."""
    a = np.radians(deg) / 2.0
    return [float(np.cos(a)), 0.0, float(np.sin(a)), 0.0]


class TestHolonomicControllerKinematics(omni.kit.test.AsyncTestCase):
    """Independent kinematic verification: expected values from first principles."""

    async def setUp(self) -> None:
        self._ctrl = ctrl.HolonomicController(
            robot_joint_space=["w1", "w2", "w3"],
            wheel_joint_names=["w1", "w2", "w3"],
            wheel_radius=1.0,
            wheel_positions=[[0, 1, 0], [-1, -1, 0], [1, -1, 0]],
            wheel_orientations=[_rot_z(90), _rot_z(225), _rot_z(-45)],
            mecanum_angles=90.0,
        )

    async def tearDown(self) -> None:
        pass

    async def test_independently_derived_wheel_speeds(self) -> None:
        """Wheel speeds match values derived from first-principles kinematics."""
        result = self._ctrl.forward(_EMPTY_STATE, _site_setpoint(4.0, 5.0, -5.0), 0.0)
        self.assertIsNotNone(result)
        speeds = dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))
        self.assertAlmostEqual(speeds["w1"], 9.0, places=4)
        self.assertAlmostEqual(speeds["w2"], 11.0 / 2**0.5, places=4)
        self.assertAlmostEqual(speeds["w3"], 1.0 / 2**0.5, places=4)

    async def test_rotated_wheel_frame_with_command_site_quaternion(self) -> None:
        """Geometry authored in a rotated frame plus a command-site quaternion reproduces the base scenario.

        Applies R = R_z(90°) (X→Y, Y→-X, Z→Z) to the geometry of
        test_independently_derived_wheel_speeds, then sets the command site back to the
        original frame.  The wheels are unmoved physically — only their coordinates change —
        so the same site-frame command must yield the same wheel speeds.

        Rotated positions:  W1→[-1,0,0]  W2→[1,-1,0]  W3→[1,1,0]
        Rotated axles:      W1→[-1,0,0]=rot_z(180°)@x
                            W2→[√½,-√½,0]=rot_z(-45°)@x
                            W3→[√½,+√½,0]=rot_z(45°)@x
        Command (site frame, unchanged): linear=[4,5,0], angular=[0,0,-5]
        Expected φ̇: 9, 11/√2, 1/√2  (unchanged)
        """
        rotated_frame_ctrl = ctrl.HolonomicController(
            robot_joint_space=["w1", "w2", "w3"],
            wheel_joint_names=["w1", "w2", "w3"],
            wheel_radius=1.0,
            wheel_positions=[[-1, 0, 0], [1, -1, 0], [1, 1, 0]],
            wheel_orientations=[_rot_z(180), _rot_z(-45), _rot_z(45)],
            mecanum_angles=90.0,
            command_site_quaternion=_rot_z(90),
        )
        result = rotated_frame_ctrl.forward(_EMPTY_STATE, _site_setpoint(4.0, 5.0, -5.0), 0.0)
        self.assertIsNotNone(result)
        speeds = dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))
        self.assertAlmostEqual(speeds["w1"], 9.0, places=4)
        self.assertAlmostEqual(speeds["w2"], 11.0 / 2**0.5, places=4)
        self.assertAlmostEqual(speeds["w3"], 1.0 / 2**0.5, places=4)

    async def test_rotated_command_site_reinterprets_the_same_physical_command(self) -> None:
        """Rotating the site and re-expressing the same physical command gives the same wheel speeds.

        Converse of test_rotated_wheel_frame_with_command_site_quaternion: here the wheel
        geometry is left exactly as in test_independently_derived_wheel_speeds, and only the
        command site is rotated by R_z(90°).  A command is a vector in the *site* frame, so the
        same physical motion must be written in the rotated coordinates:

        Site rotation:  q_s = rot_z(90°),  so  v_S = R_z(90°)ᵀ · v_F
        Linear:         R_z(-90°)@[4,5,0]  = [5,-4,0]
        Angular:        R_z(-90°)@[0,0,-5] = [0,0,-5]   (yaw axis unchanged by a z-rotation)
        Expected φ̇:    9, 11/√2, 1/√2  — identical to the unrotated scenario

        Passing the unrotated command [4,5,0] here would instead give ≈[0, 13.44, 7.78],
        so the assertion genuinely pins the site-frame interpretation.
        """
        rotated_site_ctrl = ctrl.HolonomicController(
            robot_joint_space=["w1", "w2", "w3"],
            wheel_joint_names=["w1", "w2", "w3"],
            wheel_radius=1.0,
            wheel_positions=[[0, 1, 0], [-1, -1, 0], [1, -1, 0]],
            wheel_orientations=[_rot_z(90), _rot_z(225), _rot_z(-45)],
            mecanum_angles=90.0,
            command_site_quaternion=_rot_z(90),
        )
        result = rotated_site_ctrl.forward(_EMPTY_STATE, _site_setpoint(5.0, -4.0, -5.0), 0.0)
        self.assertIsNotNone(result)
        speeds = dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))
        self.assertAlmostEqual(speeds["w1"], 9.0, places=4)
        self.assertAlmostEqual(speeds["w2"], 11.0 / 2**0.5, places=4)
        self.assertAlmostEqual(speeds["w3"], 1.0 / 2**0.5, places=4)

    async def test_command_site_translation_changes_wheel_speeds(self) -> None:
        """Translating the command site changes the moment arms, and so the wheel speeds.

        Moves the site to [0, 0.5, 0] with the geometry of
        test_independently_derived_wheel_speeds, leaving the wheels physically unmoved.
        Only the yaw column of M changes, via r_i → r_i − [0, 0.5, 0]:

        Yaw term:     a·(z × r) = a·(−r_y, r_x, 0) = −a_x·r_y + a_y·r_x
        Shift:        Δr_y = −0.5, Δr_x = 0  ⇒  Δ(yaw term) = 0.5·a_x
        No-slip axes: a_1=[1,0,0]  a_2=[−√½,√½,0]  a_3=[−√½,−√½,0]
        With wz=−5:   Δφ̇ = 0.5·a_x·(−5)  ⇒  −2.5, +5√2/4, +5√2/4
        Expected φ̇:  6.5, 27√2/4, 7√2/4   (from 9, 11/√2, 1/√2)

        A *lateral* offset is needed here: a_1 is exactly [1,0,0], so its yaw term
        reduces to −r_y and an x-offset alone would leave w1 untouched.
        """
        offset_ctrl = ctrl.HolonomicController(
            robot_joint_space=["w1", "w2", "w3"],
            wheel_joint_names=["w1", "w2", "w3"],
            wheel_radius=1.0,
            wheel_positions=[[0, 1, 0], [-1, -1, 0], [1, -1, 0]],
            wheel_orientations=[_rot_z(90), _rot_z(225), _rot_z(-45)],
            mecanum_angles=90.0,
            command_site_position=[0.0, 0.5, 0.0],
        )
        result = offset_ctrl.forward(_EMPTY_STATE, _site_setpoint(4.0, 5.0, -5.0), 0.0)
        self.assertIsNotNone(result)
        speeds = dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))
        self.assertAlmostEqual(speeds["w1"], 6.5, places=4)
        self.assertAlmostEqual(speeds["w2"], 27.0 * 2**0.5 / 4.0, places=4)
        self.assertAlmostEqual(speeds["w3"], 7.0 * 2**0.5 / 4.0, places=4)

    async def test_invalid_command_site_raises(self) -> None:
        """Malformed command-site position or quaternion raises ValueError."""
        for kwargs in (
            {"command_site_position": [0.0, 0.0]},
            {"command_site_quaternion": [1.0, 0.0, 0.0]},
            {"command_site_quaternion": [0.0, 0.0, 0.0, 0.0]},
        ):
            with self.assertRaises(ValueError):
                ctrl.HolonomicController(
                    robot_joint_space=["w1", "w2", "w3"],
                    wheel_joint_names=["w1", "w2", "w3"],
                    wheel_radius=1.0,
                    wheel_positions=[[0, 1, 0], [-1, -1, 0], [1, -1, 0]],
                    wheel_orientations=[_rot_z(90), _rot_z(225), _rot_z(-45)],
                    **kwargs,
                )


# ── Y-up independent kinematic verification ───────────────────────────────────
#
# The Z-up scenario from TestHolonomicControllerKinematics re-expressed in a
# Y-up frame by applying R = R_x(90°):  X→X,  Y→−Z,  Z→-Y.


class TestHolonomicControllerYUp(omni.kit.test.AsyncTestCase):
    """Same physical scenario as TestHolonomicControllerKinematics, expressed in a Y-up frame."""

    async def setUp(self) -> None:
        self._ctrl = ctrl.HolonomicController(
            robot_joint_space=["w1", "w2", "w3"],
            wheel_joint_names=["w1", "w2", "w3"],
            wheel_radius=1.0,
            wheel_positions=[[0, 0, -1], [-1, 0, 1], [1, 0, 1]],
            wheel_orientations=[_rot_y(90.0), _rot_y(225.0), _rot_y(-45.0)],
            mecanum_angles=90.0,
            rotation_direction=[0.0, 1.0, 0.0],
        )

    async def tearDown(self) -> None:
        pass

    async def test_y_up_gives_same_wheel_speeds_as_z_up(self) -> None:
        """Y-up frame with rotated geometry produces the same wheel speeds as the Z-up scenario."""
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[4.0, 0.0, -5.0]], dtype=wp.float32)),
                angular_velocities=(["control_point"], wp.array([[0.0, -5.0, 0.0]], dtype=wp.float32)),
            )
        )
        result = self._ctrl.forward(_EMPTY_STATE, setpoint, 0.0)
        self.assertIsNotNone(result)
        speeds = dict(zip(result.joints.velocity_names, result.joints.velocities.numpy()))
        self.assertAlmostEqual(speeds["w1"], 9.0, places=4)
        self.assertAlmostEqual(speeds["w2"], 11.0 / 2**0.5, places=4)
        self.assertAlmostEqual(speeds["w3"], 1.0 / 2**0.5, places=4)
